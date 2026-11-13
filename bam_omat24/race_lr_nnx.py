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
               "ct_label_keys": ["nbo_charges", "lowdin_charges"], "ct_bond_factor": 1.2, "spin_head": False}
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
        self.spin_head = bool(lr['spin_head'])
        if self.spin_head:
            n0s = sum(mul for mul, ir in e3nn.Irreps(kw['hidden_irreps']) if ir == e3nn.Irrep('0e'))
            self.sp_head1 = nnx.Linear(n0s, int(lr['head_hidden']), rngs=rngs)
            self.sp_head2 = nnx.Linear(int(lr['head_hidden']), 2, rngs=rngs, kernel_init=nnx.initializers.zeros_init())
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
                Q, S = self._graph_globals(data, dt)
                pre = self.cs_mlp1(jnp.stack([Q, S - 1.0], axis=-1))
                if self.cs_table:
                    qi = jnp.clip(jnp.round(Q).astype(jnp.int32), -self.cs_qmax, self.cs_qmax) + self.cs_qmax
                    si = jnp.clip(jnp.round(S).astype(jnp.int32) - 1, 0, self.cs_smax - 1)
                    pre = pre + self.cs_q_table[...][qi].astype(pre.dtype) + self.cs_s_table[...][si].astype(pre.dtype)
                hid = jax.nn.silu(pre)                                                  # (G, cs_hidden)
                z = z_cs = self.cs_mlp2(hid).astype(dt)                                 # (G, n_e)
                ctx['cs_hidden'], ctx['cs_node_graph'] = hid.astype(dt), node_graph
                if self.cs_gate:
                    n0 = self.embed_irreps.dim
                    gate0 = 1.0 + self.cs_gate_table[...].astype(dt)[:, :n0][data.nodes['species']]   # (N, n0)
            if self.n_tasks:
                g = data.globals if isinstance(data.globals, dict) else {}
                t = g.get('task')
                t = (jnp.zeros((n_graphs,), jnp.int32) if t is None
                     else jnp.clip(jnp.round(jnp.asarray(t)).astype(jnp.int32).reshape(n_graphs), 0, self.n_tasks - 1))
                z = z + self.task_table[...][t].astype(dt)
            zn = z[node_graph]
            if gate0 is not None:
                # only the charge / spin part is gated; the task row (added to z above) is left as it is
                zn = zn + z_cs[node_graph] * (gate0 - 1.0)
            add = jnp.zeros_like(f.array).at[:, :z.shape[1]].set(zn)
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
        z = (hid @ self.cs_layer_w[...].astype(dt)).reshape(hid.shape[0], -1, n0)[:, i - 1]        # (G, n0)
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
        if self.spin_head:
            out['spins'] = self._spins(s, data, node_graph, node_mask, n_graphs, dt)
        return out

    def _spins(self, s, data, node_graph, node_mask, n_graphs, dt):
        """atomic spin populations summing to 2S = multiplicity - 1 of every graph (spin_head)"""
        _, S = self._graph_globals(data, dt)
        z = self.sp_head2(jax.nn.silu(self.sp_head1(s))).astype(dt)
        raw, logit = z[:, 0], z[:, 1]
        lg = jnp.where(node_mask, logit, -jnp.inf)
        mx = jax.lax.stop_gradient(jax.ops.segment_max(lg, node_graph, num_segments=n_graphs))
        mx = jnp.where(jnp.isfinite(mx), mx, 0.0)
        un = jnp.where(node_mask, jnp.exp(jnp.where(node_mask, logit, 0.0) - mx[node_graph]), 0.0)
        den = jax.ops.segment_sum(un, node_graph, num_segments=n_graphs)
        w = un / jnp.where(den > 0, den, 1.0)[node_graph]
        raw = jnp.where(node_mask, raw, 0.0)
        tot = jax.ops.segment_sum(raw, node_graph, num_segments=n_graphs)
        return jnp.where(node_mask, raw + w * (S - 1.0 - tot)[node_graph], 0.0)

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
        c = jnp.exp(g[sp[rcv]] + g[sp[snd]]) * (1.0 - sr_switch(r, r_on, r_off))
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
