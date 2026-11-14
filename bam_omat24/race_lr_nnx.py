"""RACE_LR: RACEv2 with charge-dependent long-range electrostatics (2026-11-06; docs/RACE_LR_design.md).
RACE itself (race_nnx.py) is unchanged; this subclass adds one physical stratum:

    E = sum_i [ local RACEv2 strata + repulsion ]_i  +  E_Qeq({q_i})

    chi_i, J_i = MLP(scalar features of the last layer) + per-species offsets   (electronegativity, hardness > 0)
    q = argmin_q  chi . q + 1/2 q^T A q   s.t.  sum_i q_i = Q_graph   (charge equilibration, one linear solve)
    A_ij = Gaussian-charge Coulomb (direct sum, or Ewald for periodic graphs), A_ii = J_i + K / (sqrt(pi) sigma_i)

The total charge Q of each graph comes from data.globals['charge'] (0 if absent; train_3bpa reads it from the extxyz
info key given by cfg "total_charge_key").  Charges are latent: trained from energies and forces only (no charge
labels), as in latent-charge / 4G-HDNNP-type models.  Forces and stress follow from autodiff through the linear
solve.  Config: "model": "RACE_LR", "long_range": {"ewald_alpha": 0.3, "ewald_n_real": 2, "ewald_n_recip": 6,
"head_hidden": 32, "chi_init": "pauling" (default since 2026-11-07) | "zero", "chi_init_scale": 3.0, "hard_init": 5.0,
"sr_switch": [r_on, r_off] (switch the Coulomb energy off below r_on, smoothly on by r_off; charges unswitched)}.
Fragments: nodes['fragment'] + globals['fragment_charges'] (cfg "fragment_key" / "fragment_charges_key") constrain
the total charge of every fragment of a structure separately.  "qeq_block": B (largest structure, atoms) solves the
Qeq system per structure instead of densely over the whole batch (same result, G B^2 memory instead of N^2).  Optional charge
loss on reference charges (e.g. Hirshfeld): cfg "charge_label_key" + "charge_weight" (train_3bpa; default off).
Charge / spin embedding: "charge_spin_embedding": true adds MLP(Q, 2S+1 - 1) (zero-initialised output) to the
layer-0 scalars; the spin comes from data.globals['spin'] (cfg "total_spin_key", default 1).  "qeq": false switches
the Qeq stratum off (embedding-only variant).  Periodic graphs: Ewald per graph (dense, for small cells; see design
doc for limits).
Task embedding slot (2026-11-10): "n_tasks": N > 0 adds a zero-initialised table (N, layer-0 width); row
data.globals['task'] (integer per graph, 0 if absent, clipped to 0..N-1) is added to the layer-0 scalars, next to the
charge / spin embedding.  A task is one labelling convention (level of theory / code of a dataset).  A run that never
sets 'task' trains row 0 only and leaves the other rows at exactly zero, so a later multi-dataset stage can give every
new dataset its own row without changing the parameter tree (docs/HANDOFF_uq_omol25.md §7).
"""
from __future__ import annotations

import math

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import jraph
from flax import nnx

from bam_omat24.long_range import (coulomb_direct, ct_solve, ewald_matrix, ewald_matrix_rows, qeq_solve,
                                   short_range_part, sr_switch)
from bam_omat24.race_nnx import RACE, _dense

LR_DEFAULTS = {"ewald_alpha": 0.3, "ewald_n_real": 2, "ewald_n_recip": 6, "head_hidden": 32, "chi_scale": 1.0,
               "hard_init": 5.0, "chi_init": "pauling", "chi_init_scale": 3.0, "chi_init_missing": "mean", "qeq": True,
               "charge_spin_embedding": False, "cs_hidden": 32, "sr_switch": None, "qeq_block": None,
               "n_tasks": 0, "coulomb_energy": True,
               "cs_encoding": "scalar", "cs_layers": "first", "cs_species_gate": False,
               "cs_max_charge": 6, "cs_max_spin": 11,
               "sigma_bounds": None, "ct_mode": "global", "ct_switch": [3.0, 4.5], "ct_init": 1.15,
               "ct_label_keys": ["nbo_charges", "lowdin_charges"], "ct_bond_factor": 1.2, "spin_head": False,
               "spin_energy": False, "spin_w_init": "sdftb", "spin_mu_bias": -4.0, "spin_pair": True,
               "spin_pair_switch": None, "spin_kappa_init": 0.5, "spin_nbo_head": False, "spin_nbo_switch": None,
               "charge_nbo_head": False, "cs_local": None, "nbo_t_max": 0.5, "ct_cond_max": None, "ct_decay": None}
# "ct_decay": [beta, r0] (2026-11-15; ct_mode "conductance"): the pair conductance decays exponentially with the
# distance, s = cap(g_Zi + g_Zj) - beta (r - r0) (AFTER ct_cond_max: v2local's g sums sit far above the cap, where
# their gradient is zero, and a decay inside the cap had no effect at all; then the ct_switch), as the overlap behind a real
# charge transfer does.  The transferred charge goes as cond A / (1 + cond A), a sigmoid in ln(cond): without the decay
# ln(cond) only falls in the last ~0.15 A before ct_switch[1] (1 - C2 switch ~ 10 (1 - x)^3), so a saturated
# conductance dropped the charge transfer as a STEP there (v2local e90: F- ... H2O jumped 8 kcal/mol, force 10 eV/A,
# at F...H = 4.45 A; a lower ct_cond_max only shrank it).  With the decay the charge fades over ~4.4 / beta A.
# None: no decay.  No parameters: a checkpoint resumes with it.
# "ct_cond_max": C (2026-11-14; ct_mode "conductance"): the pair conductance exp(s), s = g_Zi + g_Zj, is capped
# smoothly at C (e^2/eV): s_eff = ln C - softplus(ln C - s) (= s well below ln C, -> ln C above).  Without the cap the
# v2local run drove g to 6 - 12 (conductances up to 3e10: in the free-flow limit cond A >> 1 the gradient on g
# vanishes and Adam's normalised steps let g drift), and I + L A lost ~11 digits (batched and single-structure
# energies differed by 1-2 meV).  C = 100 keeps cond A ~ 500 >> 1 (free flow between atoms in contact).  None: no cap.
# "nbo_t_max" (2026-11-14): the pair coefficients t_ij of spin_nbo_head / charge_nbo_head are t_max tanh(raw / t_max)
# (|t| <= t_max; None: unbounded as before).  Unbounded t gave single batches with an NBO loss 1e5 - 1e6 times the
# usual value in the v2spin run (epochs 34, 39, 42: aux loss up to 420, gradient norm 2e4 before clipping).
# "cs_local": [r_on, r_off] (2026-11-14; needs charge_spin_embedding): the charge / spin conditioning of every atom
# comes from the LOCAL charge and spin around it instead of the graph totals Q and 2S+1, so that a structure made of
# separated fragments is the sum of the fragments (size consistency; the graph totals made a separated Li+ ... I- pair
# 22-41 kcal/mol off the sum of the ions, every RACE_LR / UMA model checked, md_water/bench/size_consistency.py):
#     Qbar_i = sum_j s(r_ij) q0_j,   2Sbar_i = sum_j s(r_ij) s0_j      (j over i itself and its neighbours)
# s = 1 below r_on and a C2 switch to 0 at r_off (<= cutoff); q0 / s0 are the integer charge / 2S of every molecule
# spread evenly over its atoms (nodes 'q0' / 's0', graph_fields.formal_charges / formal_spins; without them Q / N and
# 2S / N).  A molecule splits (bond_factor x covalent radii, < r_on) while its parts are still inside r_on of each
# other, so Qbar and Sbar are continuous through the split and switch smoothly between r_on and r_off.  For a
# structure whose atoms are all closer than r_on to each other Qbar = Q and 2Sbar = 2S: the model is exactly the model
# without cs_local (fine-tuning from such a checkpoint starts unchanged).  The charge / spin rows of cs_encoding
# "table" are interpolated linearly between the integers.  The spin_energy pair features use the local S_z as well.
# "charge_nbo_head": true (2026-11-14; needs the Qeq stratum; an output, not in the energy): the charges of a second
# partition scheme (NBO) from the model charges q (trained against charge_label_key, e.g. lowdin_charges) by the same
# kind of local redistribution as spin_nbo_head, q^NBO_i = q_i + sum_j t^q_ij (q_j - q_i), on the pairs inside
# long_range ct_switch (charge moves only where the charge transfer lets it move: the integer charges of separated
# molecules are kept).  Outputs 'charges_nbo', 'charge_nbo_t'; loss (training_utils, charge_weight): the map applied
# to the charge LABELS vs the NBO labels (cfg charge_nbo_label_key, default nbo_charges).  OMol25 4M check: Lowdin -> NBO
# charges differ by RMS 0.44 e, one constant per element pair on bonds explains R^2 0.73.
# "spin_nbo_head": true (2026-11-14; needs spin_head; an output, not in the energy): the spin populations of a second
# partition scheme (NBO) from the Lowdin-like ones by a charge-conserving local redistribution on the neighbour pairs
#     m^NBO_i = m_i + sum_j t_ij (m_j - m_i),   t_ij = MLP(h_i + h_j, h_i h_j, r_ij) (1 - C2 switch over spin_nbo_switch)
# (t symmetric, zero output layer: m^NBO = m at the start; sum m^NBO = sum m exactly; linear in m, so a global spin
# flip is consistent).  spin_nbo_switch default: long_range ct_switch, so no spin moves between separated molecules.
# Outputs 'spins_nbo' (from the predicted m) and 'spin_nbo_t' (edges); the loss (training_utils, spin_weight) applies
# the map to the Lowdin LABELS and compares with the NBO labels (both from the same SCF solution: consistent signs).
# OMol25 4M check (md_water/spin_label_check.py): one constant t per element pair on bonds already gives R^2 0.76.
# "spin_energy": true (2026-11-14; needs spin_head): a spin-polarisation energy in INVARIANT spin variables, so that
# a symmetric diradical (two equivalent atoms) and the arbitrary sign of a broken-symmetry singlet are representable:
#   mu_i >= 0   local moment |m_i| of atom i (softplus head on the spin-head hidden layer, bias spin_mu_bias), raised
#               to sum_i mu_i >= 2S by w_i max(0, 2S - sum mu) (w = the softmax weights of spin_head);
#               mu = 0 on the graphs of a restricted calculation: globals 'unrestricted' = 0 (singlets only; a graph
#               without the field counts as restricted when it is a singlet and unrestricted otherwise)
#   c_ij in [-1, 1]  spin correlation of a neighbour pair (+1 parallel, -1 antiparallel): tanh of an MLP of the
#               symmetric pair features (h_i + h_j, h_i h_j, S_z), zero output layer (c = 0 at the start)
#     E_spin = sum_i [ Delta_i mu_i + 1/2 W_Z mu_i^2 ]  +  sum_{i<j} J_ij(r_ij) mu_i mu_j c_ij
#   Delta_i = delta0_Z + Linear(spin-head hidden)      promotion cost of an unpaired electron on atom i (eV, zero init)
#   W_Z                                                 on-site exchange (spin constant, eV per electron^2; "sdftb":
#                                                       atomic DFTB spin constants of the highest occupied shell for
#                                                       the elements of SDFTB_W, zero for the others; "zero")
#   J_ij = A_{Zi Zj} exp(-(kappa_Zi + kappa_Zj)(r - r_cov,i - r_cov,j)) (1 - C2 switch over spin_pair_switch, default [cutoff - 2, cutoff])
#                                                       inter-site exchange coupling (A symmetric, zero init; kappa =
#                                                       softplus, 1/A, starts at spin_kappa_init); J > 0 favours the
#                                                       antiparallel (broken-symmetry) arrangement: E_T - E_BS = 2 J mu^2
# Outputs: 'spin_moments' mu (atoms), 'spin_corr' c (edges of the neighbour list), 'spin_s2' the local-moment estimate
# <S^2> = S_z (S_z + 1) + 1/2 (sum mu - 2 S_z) per graph (OMol25 Lowdin spins: MAE 0.02 against s_squared).  A
# restricted singlet gets exactly E_spin = 0.  Delta, A and c start at zero; W starts at zero only with
# "spin_w_init": "zero" (then the model starts exactly as the model without spin_energy).  Training (spin_weight,
# training_utils): (mu - |m_ref|)^2, (c - sign(m_ref,i m_ref,j))^2 on pairs with |m_ref| >= 0.1 on both atoms, and
# the signed head only on graphs with S > 0 (the sign pattern of a singlet is arbitrary).
# Atomic spin constants W (Hartree) of the highest occupied shell, DFTB (PBE atoms): H 1s-1s, C / N / O 2p-2p.
# H and C as in the 3ob-3-1 set (H -0.0717, C 2p-2p -0.0227, checked 2026-11-14); N and O from the commonly used table,
# not checked against a primary source.  The values are only the starting point of the learned W.
SDFTB_W = {1: -0.0717, 6: -0.0227, 7: -0.026, 8: -0.028}
HARTREE_EV = 27.211386245988
# "spin_head": true (2026-11-13): atomic spin populations s_i (out 'spins') from the last-layer scalars, s_raw + w_i
# (2S - sum s_raw) with w = softmax over the graph (sums exactly to 2S = multiplicity - 1); trained with spin_weight
# against per-atom spin labels (e.g. lowdin_spins).  An auxiliary output: it does not enter the energy.
# Charge-model options of 2026-11-13 (defaults = the original model and parameter tree; docs: memory note on the
# diverging Qeq widths of the OMol25 runs):
#   "sigma_bounds": [lo, hi]   Gaussian width of the Qeq charges sigma_i = r_cov(Z_i) (lo + (hi - lo) sigmoid(u_Z)),
#                              u stored in log_sigma (initialised so that sigma = r_cov); None: sigma = exp(log_sigma)
#                              free (in the OMol25 runs it diverged to 1e3 - 1e6 A, i.e. no Coulomb coupling).
#   "ct_mode": "conductance"   distance-limited charge transfer (long_range.ct_solve): charge flows only between atoms
#                              closer than ct_switch[1] (conductance exp(g_Zi + g_Zj) times a C2 switch from 1 below
#                              ct_switch[0] to 0 at ct_switch[1]; g per element, initialised at ct_init: 1.15 gives
#                              10 e^2/eV, so that atoms in contact share charge almost freely (c A >> 1) and the split
#                              of q0 inside a contact cluster hardly matters, e.g. Na-O at 2.3 A counts as a bond of
#                              graph_fields.molecules) starting from
#                              integer charges per molecule (data nodes 'q0', graph_fields.formal_charges: the summed
#                              labels ct_label_keys, rounded); separated molecules keep integer charges.  Needs
#                              qeq_block.  "global" (default): the original Qeq over the whole structure.
# Charge / spin conditioning beyond the layer-0 embedding (2026-11-13; the defaults are the original model, with the
# original parameter tree; all three need "charge_spin_embedding": true):
#   "cs_encoding": "table"   learned rows per total charge (clipped to +-cs_max_charge) and per spin multiplicity
#                            (clipped to cs_max_spin) are added to the hidden layer of the (Q, 2S+1 - 1) MLP;
#   "cs_layers": "all"       that hidden vector is also projected onto the scalar (0e) features entering every later
#                            interaction layer (one zero-initialised projection per layer), not only onto layer 0;
#   "cs_species_gate": true  every projection is multiplied per atom by 1 + G[Z] (G zero-initialised), so that the
#                            elements can respond differently to the charge / spin state (e.g. a metal centre).
# All new parameters start at zero, so a model with these options starts as the model without them.
# "coulomb_energy": false (2026-11-11): the Qeq charges are solved as usual, but the energy keeps only the on-site
# part chi_i q_i + 1/2 (J_i + K / (sqrt(pi) sigma_i)) q_i^2; the pair Coulomb energy is left to another stratum
# (RACE_POLAR with polar.charges "qeq": the Thole block computes the electrostatics of these charges once).
# chi_init default "pauling" since 2026-11-07 (user decision): with chi = 0 a neutral structure has an exactly zero
# gradient of the Qeq energy with respect to chi, so a neutral-only dataset never develops charges.  Configs written
# before that date pin "chi_init": "zero".

# Pauling electronegativities (CRC Handbook values as in common tables; noble gases He, Ne, Ar have none).  Used only
# by "chi_init": "pauling": chi0_Z = chi_init_scale * (chi_P(Z) - mean over the species), in eV.  chi_init_scale = 3 eV
# per Pauling unit is the approximate slope of the Mulliken scale (chi_M in eV ~ 3 chi_P).
PAULING = {
    1: 2.20, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90,
    15: 2.19, 16: 2.58, 17: 3.16, 19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55, 26: 1.83,
    27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65, 31: 1.81, 32: 2.01, 33: 2.18, 34: 2.55, 35: 2.96, 36: 3.00, 37: 0.82,
    38: 0.95, 39: 1.22, 40: 1.33, 41: 1.60, 42: 2.16, 43: 1.90, 44: 2.20, 45: 2.28, 46: 2.20, 47: 1.93, 48: 1.69,
    49: 1.78, 50: 1.96, 51: 2.05, 52: 2.10, 53: 2.66, 54: 2.60, 55: 0.79, 56: 0.89, 57: 1.10, 58: 1.12, 59: 1.13,
    60: 1.14, 62: 1.17, 64: 1.20, 66: 1.22, 67: 1.23, 68: 1.24, 69: 1.25, 71: 1.27, 72: 1.30, 73: 1.50, 74: 2.36,
    75: 1.90, 76: 2.20, 77: 2.20, 78: 2.28, 79: 2.54, 80: 2.00, 81: 1.62, 82: 2.33, 83: 2.02, 84: 2.00, 85: 2.20,
}


class RACE_LR(RACE):
    def __init__(self, *args, long_range=None, **kw):
        super().__init__(*args, **kw)
        lr = dict(LR_DEFAULTS)
        lr.update(long_range or {})
        self.lr = lr
        rngs = kw['rngs']
        n_species = kw['n_species']
        self.qeq = bool(lr['qeq'])
        self.cs_embed = bool(lr['charge_spin_embedding'])
        if self.qeq:     # electronegativity / hardness head, chi0, J0, sigma: only with the Qeq stratum
            n0 = sum(mul for mul, ir in e3nn.Irreps(kw['hidden_irreps']) if ir == e3nn.Irrep('0e'))
            h = int(lr['head_hidden'])
            self.lr_head1 = nnx.Linear(n0, h, rngs=rngs)
            self.lr_head2 = nnx.Linear(h, 2, rngs=rngs, kernel_init=nnx.initializers.zeros_init())
            chi_init = str(lr['chi_init'])
            if chi_init == 'zero':
                chi0 = jnp.zeros((n_species,), jnp.float32)
            elif chi_init == 'pauling':
                zs = self.species_z
                if zs is None or len(zs) != n_species:
                    raise ValueError("chi_init 'pauling' needs species_z (one atomic number per species)")
                missing = [z for z in zs if z not in PAULING]
                if missing and lr['chi_init_missing'] != 'mean':
                    raise ValueError(f"chi_init 'pauling': no Pauling electronegativity for Z={missing} "
                                     "(set long_range chi_init_missing 'mean' to start them at the species mean)")
                known = [PAULING[z] for z in zs if z in PAULING]
                fill = sum(known) / len(known)
                v = jnp.asarray([PAULING.get(z, fill) for z in zs], jnp.float32)
                chi0 = float(lr['chi_init_scale']) * (v - jnp.mean(v))
            else:
                raise ValueError(f"long_range chi_init must be 'zero' or 'pauling', got {chi_init!r}")
            self.chi0 = nnx.Param(chi0)
            # hardness J = softplus(raw) > 0, initialised at hard_init (eV/e^2); Gaussian widths from covalent radii
            raw = jnp.log(jnp.expm1(jnp.asarray(float(lr['hard_init']))))
            self.hard0 = nnx.Param(jnp.full((n_species,), raw, jnp.float32))
            radii = kw.get('species_radii') or [1.0] * n_species
            self.cov_r = tuple(float(x) for x in radii)          # static (not a parameter)
            sb = lr['sigma_bounds']
            if sb is None:
                self.log_sigma = nnx.Param(jnp.log(jnp.asarray(radii, jnp.float32)))
            else:
                lo, hi = (float(x) for x in sb)
                if not 0.0 < lo < 1.0 < hi:
                    raise ValueError(f"long_range sigma_bounds must be [lo, hi] with 0 < lo < 1 < hi, got {sb}")
                u0 = math.log((1.0 - lo) / (hi - 1.0))           # sigmoid(u0) = (1 - lo) / (hi - lo): sigma = r_cov
                self.log_sigma = nnx.Param(jnp.full((n_species,), u0, jnp.float32))
            ct = str(lr['ct_mode'])
            if ct not in ('global', 'conductance'):
                raise ValueError(f"long_range ct_mode must be 'global' or 'conductance', got {ct!r}")
            self.ct_cond = ct == 'conductance'
            if self.ct_cond:
                if not lr['qeq_block']:
                    raise ValueError("long_range ct_mode 'conductance' needs qeq_block (largest structure, atoms)")
                r_on, r_off = (float(x) for x in lr['ct_switch'])
                if not 0.0 <= r_on < r_off <= float(kw['cutoff']):
                    raise ValueError(f"long_range ct_switch must satisfy 0 <= r_on < r_off <= cutoff, got {lr['ct_switch']}")
                if lr['ct_decay'] is not None:
                    beta, _ = (float(x) for x in lr['ct_decay'])
                    if beta < 0.0:
                        raise ValueError(f"long_range ct_decay must be [beta >= 0 (1/A), r0 (A)], got {lr['ct_decay']}")
                self.ct_log_g = nnx.Param(jnp.full((n_species,), float(lr['ct_init']), jnp.float32))
        # charge / spin embedding (optional): (Q, 2S+1 - 1) of the graph -> MLP -> added to the layer-0 scalars.
        # Output layer at zero, so the model starts as RACE (also for a later fine-tuning from a RACE checkpoint).
        if self.cs_embed:
            n_e = self.embed_irreps.dim          # layer-0 features are scalars (0e) first, then the DeNS 1o block
            self.cs_mlp1 = nnx.Linear(2, int(lr['cs_hidden']), rngs=rngs)
            self.cs_mlp2 = nnx.Linear(int(lr['cs_hidden']), n_e, rngs=rngs, kernel_init=nnx.initializers.zeros_init())
        # task embedding slot (optional): one row per labelling convention, zero-initialised (see the module docstring)
        self.n_tasks = int(lr['n_tasks'] or 0)
        if self.n_tasks < 0:
            raise ValueError(f"long_range n_tasks must be >= 0, got {self.n_tasks}")
        if self.n_tasks:
            self.task_table = nnx.Param(jnp.zeros((self.n_tasks, self.embed_irreps.dim), jnp.float32))
        # charge / spin conditioning options (created last and at zero: the other parameters keep their initial values)
        enc, lay = str(lr['cs_encoding']), str(lr['cs_layers'])
        if enc not in ('scalar', 'table') or lay not in ('first', 'all'):
            raise ValueError(f"long_range cs_encoding must be 'scalar' or 'table' and cs_layers 'first' or 'all', "
                             f"got {enc!r}, {lay!r}")
        self.cs_table, self.cs_all, self.cs_gate = enc == 'table', lay == 'all', bool(lr['cs_species_gate'])
        if (self.cs_table or self.cs_all or self.cs_gate) and not self.cs_embed:
            raise ValueError("long_range cs_encoding / cs_layers / cs_species_gate need charge_spin_embedding true")
        if self.cs_embed:
            h, n0, n_l = int(lr['cs_hidden']), self.embed_irreps.dim, len(self.layers)
            if self.cs_table:
                self.cs_qmax, self.cs_smax = int(lr['cs_max_charge']), int(lr['cs_max_spin'])
                self.cs_q_table = nnx.Param(jnp.zeros((2 * self.cs_qmax + 1, h), jnp.float32))
                self.cs_s_table = nnx.Param(jnp.zeros((self.cs_smax, h), jnp.float32))
            if self.cs_all and n_l > 1:
                # (h, (n_l - 1) n0): the projections of layers 1 .. n_l - 1 side by side (one 2-D matrix)
                self.cs_layer_w = nnx.Param(jnp.zeros((h, (n_l - 1) * n0), jnp.float32))
            if self.cs_gate:
                # (n_species, n_gate n0): leading species axis (soap_species_adam treats it as an element table)
                self.cs_n_gate = n_l if self.cs_all else 1
                self.cs_gate_table = nnx.Param(jnp.zeros((n_species, self.cs_n_gate * n0), jnp.float32))
        self.cs_local = lr['cs_local'] is not None
        if self.cs_local:
            if not self.cs_embed:
                raise ValueError("long_range cs_local needs charge_spin_embedding true")
            r_on, r_off = (float(x) for x in lr['cs_local'])
            if not 0.0 <= r_on < r_off <= float(kw['cutoff']):
                raise ValueError(f"long_range cs_local must be [r_on, r_off] with 0 <= r_on < r_off <= cutoff, got "
                                 f"{lr['cs_local']}")
        self.spin_head = bool(lr['spin_head'])
        if self.spin_head:
            n0s = sum(mul for mul, ir in e3nn.Irreps(kw['hidden_irreps']) if ir == e3nn.Irrep('0e'))
            self.sp_head1 = nnx.Linear(n0s, int(lr['head_hidden']), rngs=rngs)
            self.sp_head2 = nnx.Linear(int(lr['head_hidden']), 2, rngs=rngs, kernel_init=nnx.initializers.zeros_init())
        self.spin_energy = bool(lr['spin_energy'])
        if self.spin_energy:
            if not (self.spin_head and self.qeq):
                raise ValueError("long_range spin_energy needs spin_head true (and the Qeq stratum)")
            h = int(lr['head_hidden'])
            self.sp_mu = nnx.Linear(h, 1, rngs=rngs, kernel_init=nnx.initializers.zeros_init(),
                                    bias_init=nnx.initializers.constant(float(lr['spin_mu_bias'])))
            self.sp_c1 = nnx.Linear(2 * h + 1, h, rngs=rngs)
            self.sp_c2 = nnx.Linear(h, 1, rngs=rngs, kernel_init=nnx.initializers.zeros_init())
            self.sp_delta = nnx.Linear(h, 1, rngs=rngs, kernel_init=nnx.initializers.zeros_init())
            self.sp_delta0 = nnx.Param(jnp.zeros((n_species,), jnp.float32))
            winit = str(lr['spin_w_init'])
            if winit == 'zero':
                w = jnp.zeros((n_species,), jnp.float32)
            elif winit == 'sdftb':
                zs = self.species_z
                if zs is None or len(zs) != n_species:
                    raise ValueError("spin_w_init 'sdftb' needs species_z (one atomic number per species)")
                w = jnp.asarray([SDFTB_W.get(int(z), 0.0) * HARTREE_EV for z in zs], jnp.float32)
            else:
                raise ValueError(f"long_range spin_w_init must be 'sdftb' or 'zero', got {winit!r}")
            self.sp_w = nnx.Param(w)
            self.spin_pair = bool(lr['spin_pair'])
            if self.spin_pair:
                if lr['spin_pair_switch'] is None:          # default: switched off over the last 2 A before the cutoff
                    lr['spin_pair_switch'] = [max(0.0, float(kw['cutoff']) - 2.0), float(kw['cutoff'])]
                r_on, r_off = (float(x) for x in lr['spin_pair_switch'])
                if not 0.0 <= r_on < r_off <= float(kw['cutoff']):
                    raise ValueError(f"long_range spin_pair_switch must satisfy 0 <= r_on < r_off <= cutoff, got "
                                     f"{lr['spin_pair_switch']}")
                self.sp_j = nnx.Param(jnp.zeros((n_species, n_species), jnp.float32))
                k0 = float(lr['spin_kappa_init'])
                self.sp_kappa = nnx.Param(jnp.full((n_species,), math.log(math.expm1(k0)), jnp.float32))
        self.spin_nbo = bool(lr['spin_nbo_head'])
        if self.spin_nbo:
            if not (self.spin_head and self.qeq):
                raise ValueError("long_range spin_nbo_head needs spin_head true (and the Qeq stratum)")
            if lr['spin_nbo_switch'] is None:
                lr['spin_nbo_switch'] = [float(x) for x in lr['ct_switch']]
            r_on, r_off = (float(x) for x in lr['spin_nbo_switch'])
            if not 0.0 <= r_on < r_off <= float(kw['cutoff']):
                raise ValueError(f"long_range spin_nbo_switch must satisfy 0 <= r_on < r_off <= cutoff, got "
                                 f"{lr['spin_nbo_switch']}")
            h = int(lr['head_hidden'])
            self.sp_t1 = nnx.Linear(2 * h + 1, h, rngs=rngs)
            self.sp_t2 = nnx.Linear(h, 1, rngs=rngs, kernel_init=nnx.initializers.zeros_init())
        self.charge_nbo = bool(lr['charge_nbo_head'])
        if self.charge_nbo:
            if not self.qeq:
                raise ValueError("long_range charge_nbo_head needs the Qeq stratum (qeq true)")
            n0c = sum(mul for mul, ir in e3nn.Irreps(kw['hidden_irreps']) if ir == e3nn.Irrep('0e'))
            h = int(lr['head_hidden'])
            self.cq_h = nnx.Linear(n0c, h, rngs=rngs)
            self.cq_t1 = nnx.Linear(2 * h + 1, h, rngs=rngs)
            self.cq_t2 = nnx.Linear(h, 1, rngs=rngs, kernel_init=nnx.initializers.zeros_init())
        if not (self.qeq or self.cs_embed or self.n_tasks or getattr(self, '_allow_plain', False)):
            raise ValueError("long_range: qeq false, charge_spin_embedding false and n_tasks 0 is plain RACE; "
                             "use model RACE")

    def _graph_globals(self, data, dt):
        n_graphs = data.n_node.shape[0]
        g = data.globals if isinstance(data.globals, dict) else {}
        Q = g.get('charge')
        S = g.get('spin')
        Q = jnp.zeros((n_graphs,), dt) if Q is None else jnp.asarray(Q, dt).reshape(n_graphs)
        S = jnp.ones((n_graphs,), dt) if S is None else jnp.asarray(S, dt).reshape(n_graphs)
        return Q, S

    @staticmethod
    def _table_rows(table, x):
        """rows of a table at the (clipped, >= 0) positions x, linearly interpolated between the integer rows (at an
        integer x exactly that row, as the former lookup of the rounded value)"""
        n = table.shape[0]
        x = jnp.clip(x, 0.0, n - 1.0)
        i0 = jnp.clip(jnp.floor(x).astype(jnp.int32), 0, n - 1)
        i1 = jnp.minimum(i0 + 1, n - 1)
        f = (x - i0.astype(x.dtype))[:, None]
        return (1.0 - f) * table[i0].astype(x.dtype) + f * table[i1].astype(x.dtype)

    def _edge_r(self, positions, cell, data, node_graph, node_mask, n_graphs, dt):
        """(r (E,), good (E,)) distances of the directed edges (receiver, sender) and the mask of real pairs (r > 0)"""
        snd, rcv = data.senders, data.receivers
        ok = node_mask[snd] & node_mask[rcv]
        d = positions[snd] - positions[rcv]
        if self.periodic and cell is not None and isinstance(data.edges, dict) and 'Sij' in data.edges:
            cells = self._safe_cells(cell, n_graphs)
            d = d - jnp.einsum('ei,eij->ej', jnp.asarray(data.edges['Sij'], dt), cells[node_graph[rcv]])
        d2 = jnp.sum(d * d, axis=-1)
        good = ok & (d2 > 0)
        return jnp.sqrt(jnp.where(good, d2, 1.0)).astype(dt), good

    def _local_cs(self, positions, cell, data, node_graph, node_mask, n_graphs, dt):
        """(Qbar (N,), 2Sbar (N,)) of long_range cs_local: the reference charges q0 and spins s0 (nodes 'q0' / 's0';
        else the graph totals spread evenly) summed over the atom itself and its neighbours with the switch s(r)"""
        nodes = data.nodes if isinstance(data.nodes, dict) else {}
        Q, S = self._graph_globals(data, dt)
        cnt = jnp.maximum(jnp.asarray(data.n_node).astype(dt), 1.0)
        q0 = jnp.asarray(nodes['q0'], dt) if 'q0' in nodes else (Q / cnt)[node_graph]
        s0 = jnp.asarray(nodes['s0'], dt) if 's0' in nodes else ((S - 1.0) / cnt)[node_graph]
        q0, s0 = jnp.where(node_mask, q0, 0.0), jnp.where(node_mask, s0, 0.0)
        r_on, r_off = (float(x) for x in self.lr['cs_local'])
        r, good = self._edge_r(positions, cell, data, node_graph, node_mask, n_graphs, dt)
        w = jnp.where(good, 1.0 - sr_switch(r, r_on, r_off), 0.0).astype(dt)
        snd, rcv = data.senders, data.receivers
        qb = q0 + jnp.zeros_like(q0).at[rcv].add(w * q0[snd])
        sb = s0 + jnp.zeros_like(s0).at[rcv].add(w * s0[snd])
        return qb, sb

    def _prelude(self, positions, cell, cutoff, data, *args, **kw):
        ctx = super()._prelude(positions, cell, cutoff, data, *args, **kw)
        if self.cs_embed or self.n_tasks:
            f = ctx['features']
            dt = f.array.dtype
            n_graphs = data.n_node.shape[0]
            z, z_cs = 0.0, None
            n_nodes = f.shape[0]
            node_graph = jnp.repeat(jnp.arange(n_graphs), data.n_node, total_repeat_length=n_nodes)
            gate0 = None
            if self.cs_embed:
                if self.cs_local:          # per atom: local charge and 2S (cs_local), rows (N, .)
                    node_mask = (jraph.get_node_padding_mask(data) if n_graphs > 1 else jnp.ones((n_nodes,), bool))
                    Q, two_s = self._local_cs(positions, cell, data, node_graph, node_mask, n_graphs, dt)
                    S = two_s + 1.0
                else:                      # per graph: the totals, rows (G, .)
                    Q, S = self._graph_globals(data, dt)
                pre = self.cs_mlp1(jnp.stack([Q, S - 1.0], axis=-1))
                if self.cs_table:
                    q_row = self._table_rows(self.cs_q_table[...], jnp.clip(Q, -self.cs_qmax, self.cs_qmax) + self.cs_qmax)
                    s_row = self._table_rows(self.cs_s_table[...], jnp.clip(S - 1.0, 0.0, self.cs_smax - 1.0))
                    pre = pre + q_row.astype(pre.dtype) + s_row.astype(pre.dtype)
                hid = jax.nn.silu(pre)                                                  # (G or N, cs_hidden)
                z_cs = self.cs_mlp2(hid).astype(dt)                                     # (G or N, n_e)
                ctx['cs_hidden'], ctx['cs_node_graph'] = hid.astype(dt), node_graph
                ctx['cs_per_node'] = self.cs_local
                if self.cs_gate:
                    n0 = self.embed_irreps.dim
                    gate0 = 1.0 + self.cs_gate_table[...].astype(dt)[:, :n0][data.nodes['species']]   # (N, n0)
            if self.n_tasks:
                g = data.globals if isinstance(data.globals, dict) else {}
                t = g.get('task')
                t = (jnp.zeros((n_graphs,), jnp.int32) if t is None
                     else jnp.clip(jnp.round(jnp.asarray(t)).astype(jnp.int32).reshape(n_graphs), 0, self.n_tasks - 1))
                z = z + self.task_table[...][t].astype(dt)
            zn = z[node_graph] if self.n_tasks else None                                # (N, n_e) task rows
            if self.cs_embed:
                zc = z_cs if self.cs_local else z_cs[node_graph]                       # (N, n_e)
                zn = zc if zn is None else zn + zc
                if gate0 is not None:
                    # only the charge / spin part is gated; the task row is left as it is
                    zn = zn + zc * (gate0 - 1.0)
            add = jnp.zeros_like(f.array).at[:, :zn.shape[1]].set(zn)
            ctx['features'] = e3nn.IrrepsArray(f.irreps, f.array + add)
        return ctx

    def _condition(self, ctx, i, features):
        """long_range.cs_layers "all": the charge / spin hidden vector projected onto the scalars entering layer i >= 1
        (layer 0 gets it in _prelude)."""
        if i == 0 or not (self.cs_embed and self.cs_all) or 'cs_hidden' not in ctx:
            return features
        n0 = self.embed_irreps.dim
        sl = None
        for (mul, ir), s in zip(features.irreps, features.irreps.slices()):
            if ir.l == 0 and ir.p == 1 and mul == n0:
                sl = s
                break
        if sl is None:
            raise ValueError(f"cs_layers 'all': no {n0}x0e block in the features of layer {i} ({features.irreps})")
        dt = features.array.dtype
        hid, ng = ctx['cs_hidden'].astype(dt), ctx['cs_node_graph']
        n_rows = features.array.shape[0]
        if ctx.get('cs_per_node'):        # cs_local: one hidden vector per atom
            hid = hid[:n_rows]
            zi = (hid @ self.cs_layer_w[...].astype(dt)).reshape(hid.shape[0], -1, n0)[:, i - 1]   # (n_rows, n0)
        else:
            z = (hid @ self.cs_layer_w[...].astype(dt)).reshape(hid.shape[0], -1, n0)[:, i - 1]    # (G, n0)
            zi = z[ng[:n_rows]]
        if self.cs_gate:
            g = self.cs_gate_table[...].astype(dt).reshape(-1, self.cs_n_gate, n0)[:, i]            # (n_species, n0)
            zi = zi * (1.0 + g[ctx['data'].nodes['species'][:n_rows]])
        return e3nn.IrrepsArray(features.irreps, features.array.at[:, sl].add(zi))

    def node_energies(self, positions, cell, cutoff, data, *args, **kw):
        out = super().node_energies(positions, cell, cutoff, data, *args, **kw)
        if not self.qeq:
            return out
        species = data.nodes['species']
        n_nodes = species.shape[0]
        n_graphs = data.n_node.shape[0]
        node_mask = (jraph.get_node_padding_mask(data) if n_graphs > 1 else jnp.ones((n_nodes,), bool))
        node_graph = jnp.repeat(jnp.arange(n_graphs), data.n_node, total_repeat_length=n_nodes)
        s = _dense(out['node_features']).filter(keep='0e').array
        dt = s.dtype
        z = self.lr_head2(jax.nn.silu(self.lr_head1(s)))
        chi = (self.chi0[...][species] + float(self.lr['chi_scale']) * z[:, 0]).astype(dt)
        hard = jax.nn.softplus(self.hard0[...][species] + z[:, 1]).astype(dt)
        sigma = self._sigma(species, dt)
        Q, _ = self._graph_globals(data, dt)
        if self.lr['qeq_block']:
            q, e = self._qeq_blocked(positions, cell, chi, hard, sigma, Q, data, node_graph, node_mask, n_graphs, dt)
        else:
            same = (node_graph[:, None] == node_graph[None, :]) & node_mask[:, None] & node_mask[None, :]
            if self.periodic and cell is not None:
                gamma = self._ewald_batch(positions, cell, sigma, node_graph, same, n_graphs)
            else:
                gamma = coulomb_direct(positions, sigma, same)
            # constraint groups: one per graph, or one per (graph, fragment) when the data carry fragments
            # (nodes['fragment'] in 0..F-1, globals['fragment_charges'] (G, F)): each fragment keeps its own integer charge,
            # so no fractional charge flows between distant fragments (e.g. protein and ligand); Coulomb still couples them
            nodes = data.nodes if isinstance(data.nodes, dict) else {}
            glb = data.globals if isinstance(data.globals, dict) else {}
            if 'fragment' in nodes and 'fragment_charges' in glb:
                fq = jnp.asarray(glb['fragment_charges'], dt).reshape(n_graphs, -1)
                n_frag = fq.shape[1]
                group = node_graph * n_frag + jnp.clip(jnp.asarray(nodes['fragment'], jnp.int32), 0, n_frag - 1)
                Qg, n_groups = fq.reshape(-1), n_graphs * n_frag
            else:
                group, Qg, n_groups = node_graph, Q, n_graphs
            gamma_e = None
            if self.lr['sr_switch'] is not None:
                r_on, r_off = (float(x) for x in self.lr['sr_switch'])
                if self.periodic and cell is not None:
                    cells = self._safe_cells(cell, n_graphs)
                    sr = short_range_part(positions, sigma, same, r_on, r_off, cells[node_graph], n_img=1)
                else:
                    sr = short_range_part(positions, sigma, same, r_on, r_off)
                gamma_e = gamma - sr
            if not self.lr['coulomb_energy']:
                gamma_e = jnp.zeros_like(gamma)
            q, e = qeq_solve(chi, hard, sigma, gamma, group, node_mask, Qg, n_groups, gamma_energy=gamma_e)
        out = dict(out)
        out['node_energies'] = out['node_energies'] + e.reshape(out['node_energies'].shape).astype(out['node_energies'].dtype)
        out['charges'] = q
        if self.charge_nbo:
            hq = jax.nn.silu(self.cq_h(s))
            tq = self._pair_t(hq, self.cq_t1, self.cq_t2, self.lr['ct_switch'], positions, cell, data, node_graph,
                              node_mask, n_graphs, dt)
            out['charge_nbo_t'] = tq
            out['charges_nbo'] = self.spin_redistribute(q, tq, data)
        if self.spin_head:
            m, hid, w = self._spin_parts(s, data, node_graph, node_mask, n_graphs, dt)
            out['spins'] = m
            if self.spin_energy:
                es, extra = self._spin_energy(hid, w, positions, cell, data, node_graph, node_mask, n_graphs, dt)
                out['node_energies'] = (out['node_energies']
                                        + es.reshape(out['node_energies'].shape).astype(out['node_energies'].dtype))
                out['spin_node_energies'] = es
                out.update(extra)
            if self.spin_nbo:
                t = self._spin_nbo_t(hid, positions, cell, data, node_graph, node_mask, n_graphs, dt)
                out['spin_nbo_t'] = t
                out['spins_nbo'] = self.spin_redistribute(m, t, data)
        return out

    @staticmethod
    def spin_redistribute(m, t, data):
        """m_i + sum_j t_ij (m_j - m_i) over the directed edges (receiver i, sender j); conserves sum m per graph"""
        snd, rcv = data.senders, data.receivers
        return m + jnp.zeros_like(m).at[rcv].add(t * (m[snd] - m[rcv]))

    def _spin_nbo_t(self, hid, positions, cell, data, node_graph, node_mask, n_graphs, dt):
        """(E,) symmetric redistribution coefficients t_ij of spin_nbo_head (0 beyond spin_nbo_switch[1])"""
        return self._pair_t(hid, self.sp_t1, self.sp_t2, self.lr['spin_nbo_switch'], positions, cell, data, node_graph,
                            node_mask, n_graphs, dt)

    def _pair_t(self, hid, l1, l2, switch, positions, cell, data, node_graph, node_mask, n_graphs, dt):
        """(E,) symmetric pair coefficients MLP(h_i + h_j, h_i h_j, r_ij) (1 - C2 switch), 0 beyond switch[1]"""
        r_on, r_off = (float(v) for v in switch)
        snd, rcv = data.senders, data.receivers
        ok = node_mask[snd] & node_mask[rcv] & (snd != rcv)
        d = positions[snd] - positions[rcv]
        if self.periodic and cell is not None and isinstance(data.edges, dict) and 'Sij' in data.edges:
            cells = self._safe_cells(cell, n_graphs)
            d = d - jnp.einsum('ei,eij->ej', jnp.asarray(data.edges['Sij'], dt), cells[node_graph[rcv]])
        d2 = jnp.sum(d * d, axis=-1)
        good = ok & (d2 > 0)
        r = jnp.sqrt(jnp.where(good, d2, 1.0))
        hid = hid.astype(dt)
        x = jnp.concatenate([hid[rcv] + hid[snd], hid[rcv] * hid[snd], r[:, None]], axis=-1)
        t = l2(jax.nn.silu(l1(x)))[:, 0].astype(dt)
        tm = self.lr['nbo_t_max']
        if tm is not None:         # bounded |t| <= nbo_t_max (t_max tanh(raw / t_max) ~ raw for small raw)
            t = float(tm) * jnp.tanh(t / float(tm))
        return jnp.where(good, t * (1.0 - sr_switch(r, r_on, r_off)), 0.0)

    def _unrestricted(self, data, S, n_graphs, dt):
        """1 for the graphs whose spin density may be non-zero: every open shell (2S+1 > 1) and the singlets of an
        unrestricted calculation (globals 'unrestricted' = 1); 0 for a restricted singlet (spin density zero)."""
        g = data.globals if isinstance(data.globals, dict) else {}
        u = g.get('unrestricted')
        u = jnp.zeros((n_graphs,), dt) if u is None else jnp.asarray(u, dt).reshape(n_graphs)
        return jnp.where(S > 1.5, 1.0, jnp.clip(u, 0.0, 1.0))

    def _spin_energy(self, hid, w, positions, cell, data, node_graph, node_mask, n_graphs, dt):
        """per-atom spin-polarisation energy in the invariant variables mu (local moments) and c (pair correlations);
        formula in the comment above SDFTB_W.  Returns (energies, {'spin_moments', 'spin_corr', 'spin_s2'})."""
        sp = data.nodes['species']
        _, S = self._graph_globals(data, dt)
        two_s = S - 1.0
        hid = hid.astype(dt)
        mu = jax.nn.softplus(self.sp_mu(hid)[:, 0].astype(dt))
        mu = jnp.where(node_mask, mu, 0.0)
        tot = jax.ops.segment_sum(mu, node_graph, num_segments=n_graphs)
        mu = mu + w * jnp.maximum(two_s - tot, 0.0)[node_graph]                # sum mu >= 2S
        mu = jnp.where(node_mask, mu * self._unrestricted(data, S, n_graphs, dt)[node_graph], 0.0)
        delta = self.sp_delta0[...].astype(dt)[sp] + self.sp_delta(hid)[:, 0].astype(dt)
        e = delta * mu + 0.5 * self.sp_w[...].astype(dt)[sp] * mu * mu
        snd, rcv = data.senders, data.receivers
        ok = node_mask[snd] & node_mask[rcv]
        sz = 0.5 * two_s
        if self.cs_local:          # the local S_z of the receiver (size consistent), else that of the graph
            sz_e = 0.5 * self._local_cs(positions, cell, data, node_graph, node_mask, n_graphs, dt)[1][rcv]
        else:
            sz_e = sz[node_graph[rcv]]
        x = jnp.concatenate([hid[rcv] + hid[snd], hid[rcv] * hid[snd], sz_e[:, None]], axis=-1)
        c = jnp.tanh(self.sp_c2(jax.nn.silu(self.sp_c1(x)))[:, 0].astype(dt))
        c = jnp.where(ok, c, 0.0)
        if self.spin_pair:
            r_on, r_off = (float(v) for v in self.lr['spin_pair_switch'])
            d = positions[snd] - positions[rcv]
            if self.periodic and cell is not None and isinstance(data.edges, dict) and 'Sij' in data.edges:
                cells = self._safe_cells(cell, n_graphs)
                d = d - jnp.einsum('ei,eij->ej', jnp.asarray(data.edges['Sij'], dt), cells[node_graph[rcv]])
            d2 = jnp.sum(d * d, axis=-1)
            good = ok & (d2 > 0)
            r = jnp.sqrt(jnp.where(good, d2, 1.0))
            T = self.sp_j[...].astype(dt)
            A = (0.5 * (T + T.T))[sp[rcv], sp[snd]]
            kap = jax.nn.softplus(self.sp_kappa[...].astype(dt))
            rc = jnp.asarray(self.cov_r, dt)
            arg = jnp.minimum(-(kap[sp[rcv]] + kap[sp[snd]]) * (r - rc[sp[rcv]] - rc[sp[snd]]), 20.0)
            J = jnp.where(good, A * jnp.exp(arg) * (1.0 - sr_switch(r, r_on, r_off)), 0.0)
            # every pair is in the neighbour list twice (both directions): half of J mu_i mu_j c_ij per directed edge
            e = e + jnp.zeros_like(e).at[rcv].add(0.5 * J * mu[rcv] * mu[snd] * c)
        s2 = sz * (sz + 1.0) + 0.5 * (jax.ops.segment_sum(mu, node_graph, num_segments=n_graphs) - two_s)
        return jnp.where(node_mask, e, 0.0), {'spin_moments': mu, 'spin_corr': c, 'spin_s2': s2}

    def _spins(self, s, data, node_graph, node_mask, n_graphs, dt):
        """atomic spin populations summing to 2S = multiplicity - 1 of every graph (spin_head)"""
        return self._spin_parts(s, data, node_graph, node_mask, n_graphs, dt)[0]

    def _spin_parts(self, s, data, node_graph, node_mask, n_graphs, dt):
        """(spin populations, hidden layer of the spin head, softmax weights w of the graph)"""
        _, S = self._graph_globals(data, dt)
        hid = jax.nn.silu(self.sp_head1(s))
        z = self.sp_head2(hid).astype(dt)
        raw, logit = z[:, 0], z[:, 1]
        lg = jnp.where(node_mask, logit, -jnp.inf)
        mx = jax.lax.stop_gradient(jax.ops.segment_max(lg, node_graph, num_segments=n_graphs))
        mx = jnp.where(jnp.isfinite(mx), mx, 0.0)
        un = jnp.where(node_mask, jnp.exp(jnp.where(node_mask, logit, 0.0) - mx[node_graph]), 0.0)
        den = jax.ops.segment_sum(un, node_graph, num_segments=n_graphs)
        w = un / jnp.where(den > 0, den, 1.0)[node_graph]
        raw = jnp.where(node_mask, raw, 0.0)
        tot = jax.ops.segment_sum(raw, node_graph, num_segments=n_graphs)
        return jnp.where(node_mask, raw + w * (S - 1.0 - tot)[node_graph], 0.0), hid, w

    def _sigma(self, species, dt):
        """Gaussian widths of the Qeq charges (long_range sigma_bounds: bounded around the covalent radii)."""
        p = self.log_sigma[...]
        if self.lr['sigma_bounds'] is None:
            return jnp.exp(p[species]).astype(dt)
        lo, hi = (float(x) for x in self.lr['sigma_bounds'])
        r = jnp.asarray(self.cov_r, dt)
        return (r * (lo + (hi - lo) * jax.nn.sigmoid(p.astype(dt))))[species]

    def _ct_conductance(self, positions, cell, data, node_graph, node_mask, slot, n_graphs, B, dt):
        """(G, B, B) pair conductances of the blocks: exp(g_Zi + g_Zj) (1 - C2 switch from ct_switch[0] to [1]) on the
        edges of the graph (neighbour list), symmetric (both directions of an edge are in the list)."""
        r_on, r_off = (float(x) for x in self.lr['ct_switch'])
        snd, rcv = data.senders, data.receivers
        ok = node_mask[snd] & node_mask[rcv]
        d = positions[snd] - positions[rcv]
        if self.periodic and cell is not None and isinstance(data.edges, dict) and 'Sij' in data.edges:
            cells = self._safe_cells(cell, n_graphs)
            d = d - jnp.einsum('ei,eij->ej', jnp.asarray(data.edges['Sij'], dt), cells[node_graph[rcv]])
        d2 = jnp.sum(d * d, axis=-1)
        r = jnp.sqrt(jnp.where(ok & (d2 > 0), d2, 1.0))
        g = self.ct_log_g[...].astype(dt)
        sp = data.nodes['species']
        s = g[sp[rcv]] + g[sp[snd]]
        if self.lr['ct_cond_max'] is not None:            # smooth upper cap of the conductance (ct_cond_max)
            lc = math.log(float(self.lr['ct_cond_max']))
            s = lc - jax.nn.softplus(lc - s)
        if self.lr['ct_decay'] is not None:               # exponential decay with distance (ct_decay [beta, r0])
            beta, r0 = (float(x) for x in self.lr['ct_decay'])
            s = s - beta * (r - r0)                       # AFTER the cap: a capped (drifted) g cannot absorb it
        c = jnp.exp(s) * (1.0 - sr_switch(r, r_on, r_off))
        c = jnp.where(ok & (d2 > 0), c, 0.0)
        out = jnp.zeros((n_graphs, B + 1, B + 1), dt).at[node_graph[rcv], slot[rcv], slot[snd]].add(c)
        return out[:, :B, :B]

    def _qeq_blocked(self, positions, cell, chi, hard, sigma, Q, data, node_graph, node_mask, n_graphs, dt):
        """Qeq solved per graph (long_range.qeq_block = B, the largest number of atoms of one structure): the atoms are
        gathered into (G, B) blocks, the Coulomb / Ewald matrix, the optional short-range switch and the constrained
        solve are done per block (vmap), then scattered back.  Memory G B^2 and time G B^3 instead of N^2 / N^3 for
        the whole batch.  Same result as the dense path; a structure with more than B atoms makes the energy NaN."""
        B = int(self.lr['qeq_block'])
        N = chi.shape[0]
        n_node = jnp.asarray(data.n_node)
        local = jnp.arange(N) - (jnp.cumsum(n_node) - n_node)[node_graph]
        valid = node_mask & (local < B)
        overflow = jnp.any(node_mask & (local >= B))
        slot = jnp.where(valid, local, B)                       # padding / overflow -> dummy slot B (dropped)

        def blocks(x, fill):
            out = jnp.full((n_graphs, B + 1) + x.shape[1:], fill, x.dtype).at[node_graph, slot].set(x)
            return out[:, :B]
        P, Ch = blocks(positions, 0.0), blocks(chi, 0.0)
        Hd, Sg = blocks(hard, 1.0), blocks(sigma, 1.0)
        M = blocks(valid, False)
        nodes = data.nodes if isinstance(data.nodes, dict) else {}
        glb = data.globals if isinstance(data.globals, dict) else {}
        if 'fragment' in nodes and 'fragment_charges' in glb:
            fq = jnp.asarray(glb['fragment_charges'], dt).reshape(n_graphs, -1)
            n_frag = fq.shape[1]
            Fb = blocks(jnp.clip(jnp.asarray(nodes['fragment'], jnp.int32), 0, n_frag - 1), 0)
        else:
            fq, n_frag = Q.reshape(n_graphs, 1), 1
            Fb = jnp.zeros((n_graphs, B), jnp.int32)
        same_b = M[:, :, None] & M[:, None, :]
        lr = self.lr
        periodic = self.periodic and cell is not None
        if periodic:
            cells = self._safe_cells(cell, n_graphs)
            gamma = jax.vmap(lambda p, c, sg: ewald_matrix(p, c, sg, float(lr['ewald_alpha']), int(lr['ewald_n_real']),
                                                           int(lr['ewald_n_recip'])))(P, cells, Sg)
            gamma = jnp.where(same_b, gamma, 0.0)
        else:
            gamma = jax.vmap(coulomb_direct)(P, Sg, same_b)
        if not lr['coulomb_energy']:
            ge = jnp.zeros_like(gamma)
        elif lr['sr_switch'] is not None:
            r_on, r_off = (float(x) for x in lr['sr_switch'])
            if periodic:
                rows = jnp.broadcast_to(cells[:, None], (n_graphs, B, 3, 3))
                sr = jax.vmap(lambda p, sg, sm, cr: short_range_part(p, sg, sm, r_on, r_off, cr, n_img=1))(P, Sg, same_b,
                                                                                                         rows)
            else:
                sr = jax.vmap(lambda p, sg, sm: short_range_part(p, sg, sm, r_on, r_off))(P, Sg, same_b)
            ge = gamma - sr
        else:
            ge = gamma
        if self.ct_cond:
            if 'fragment' in nodes and 'fragment_charges' in glb:
                raise ValueError("long_range ct_mode 'conductance' takes the molecule charges from nodes 'q0', not "
                                 "from fragments")
            if 'q0' in nodes:
                q0 = jnp.asarray(nodes['q0'], dt)
            else:          # no reference charges: the total charge spread evenly (no integer split)
                q0 = (Q / jnp.maximum(n_node.astype(dt), 1.0))[node_graph]
            Cb = self._ct_conductance(positions, cell, data, node_graph, node_mask, slot, n_graphs, B, dt)
            qb, eb = jax.vmap(ct_solve)(Ch, Hd, Sg, gamma, Cb, M, blocks(jnp.where(node_mask, q0, 0.0), 0.0), ge)
        else:
            qb, eb = jax.vmap(lambda ch, hd, sg, gm, fb, m, qf, g_e: qeq_solve(ch, hd, sg, gm, fb, m, qf, n_frag,
                                                                               gamma_energy=g_e))(
                Ch, Hd, Sg, gamma, Fb, M, fq, ge)
        li = jnp.clip(local, 0, B - 1)
        q = jnp.where(valid, qb[node_graph, li], 0.0)
        e = jnp.where(valid, eb[node_graph, li], 0.0)
        e = jnp.where(overflow, jnp.nan, e)
        return q, e

    @staticmethod
    def _safe_cells(cell, n_graphs):
        """Padding graphs have a zero cell (inverse infinite): give them a large finite cell.  Their rows are masked
        anyway, but infinite values in a masked branch still turn gradients into NaN (0 * inf)."""
        cells = cell.reshape(n_graphs, 3, 3)
        ok = (jnp.abs(jnp.linalg.det(cells)) > 1e-6)[:, None, None]
        return jnp.where(ok, cells, 100.0 * jnp.eye(3, dtype=cells.dtype))

    def _ewald_batch(self, positions, cell, sigma, node_graph, same, n_graphs):
        """Ewald matrix of a batch of periodic graphs: every row uses the cell of its own graph; pairs of different
        graphs are masked (memory N^2 x images + N x k-vectors; small cells, see the design doc)."""
        cells = self._safe_cells(cell, n_graphs)
        lr = self.lr
        gamma = ewald_matrix_rows(positions, cells[node_graph], sigma, float(lr['ewald_alpha']),
                                  int(lr['ewald_n_real']), int(lr['ewald_n_recip']))
        return jnp.where(same, gamma, 0.0)
