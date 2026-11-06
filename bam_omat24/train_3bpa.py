"""Training script for the 3BPA benchmark with the ``bam_omat24`` RACE model.

Molecular (non-periodic) systems, single-node (one or more GPUs via JAX sharding).

Usage:
    python -m bam_omat24.train_3bpa input/input_3bpa.json

Config keys (in addition to the model hyper-parameters, see README):
    train_path, test_path | test_paths, valid_frac | valid_path (fixed validation file), atomic_numbers, atom_energies
    hidden_size (-> ``hidden_size * s2_irreps(lmax)``) or hidden_irreps
    loss_type        : "huber" | "nll" | "gauss_nll"          (default "huber")
    cov_mode         : "scalar6" | "irreps"                   (default "scalar6")
    x_mode           : "species" | "node_scalar" | "node_irreps" | "basis" (default "species");
                       "basis": x_i = Linear_t(A_i), A_i = layer-0 atomic basis, channelwise m (x) x per layer (review §11 C)
    update           : "gate" (default) | "linear"  (ungated body-ordered features, needs readout_mode "gated"; review §11 A)
    message_norm     : "layer" (default, layer norm + 1/sqrt(avg_n_neighbors)) | "avg" (no norm, 1/avg_n_neighbors; §11 B)
    x64              : bool, train/evaluate in float64 (params and data cast; MACE/BOTNet 3BPA protocol)  (default false)
    energy_loss_per  : "atom" (default) | "structure" (energy loss on the total-energy residual, MACE Eq. 15)
    readout_species  : bool, per-species stratum readout weights (default false)
    residual         : "none" | "identity" | "identity_cum" | "chebyshev" | "neumann"  (default "none")
    layer_scale      : float, LayerScale init on layer increments (default off)
    stochastic_depth : float p_max, drop stratum t with prob p_max*t/(L-1) (needs residual)
    stochastic_depth_mode : "graph" (whole molecule, default) | "node" (per atom)
    deep_supervision : float, weight of partial-sum energy losses (default 0)
    stratum_decorrelation : float, weight of sum_{s<t} Corr(E_s,E_t)^2 penalty (default 0)
    noisy_nodes      : {"sigma": A, "weight": w}, denoising auxiliary task (adds noise_head)
    sam_rho          : float, sharpness-aware minimisation radius (default 0)
    lr_schedule      : "plateau" (default) | "cosine" (+ lr_warmup_epochs, lr_cosine_min_scale)
    swa_start_epoch  : int, average raw params from this epoch on; final eval/summary = SWA
    boltzmann_tilt   : {"T_prime": K, "w_max": x}, per-frame weight exp((E-Emin)/kT') capped
    layer_lmax       : [l_0..l_{L-1}] per-layer max l of new stratum information
    layer_cutoff     : [r_0..r_{L-1}] per-layer message cutoff (A)
    radial_species_scale / radial_slater : species-pair radial transform (bool)
    embed_mode       : "onehot" (default) | "descriptor" | "both"; species embedding from a fixed
                       per-element descriptor (valence-shell <r>, eigenvalue, occupation per l from
                       atomic_shells.json + covalent radius), shared over elements
    descriptor_source: "both" (default) | "shells" | "radii"   (what goes into the descriptor)
    radial_shells    : bool, shell-pair Slater envelopes exp(-r/(rho_a(i)+rho_b(j))) as extra radial channels
    radial_shell_scale: bool, learnable per-shell-pair length scale on the envelopes (init 1)  (default false)
    in_layer_order   : 1 | 2 | 3; 2 adds the in-layer symmetric square m(x)m (BRACE-2), 3 the MACE
                       symmetric contraction (degrees 2,3, species weights; review §11.11)  (default 1)
    in_layer_order_layers : null (all layers) | n, only the first n layers use in_layer_order (default null)
    sym_contraction  : bool, use the symmetric-contraction path also for in_layer_order 2  (default false)
    radial_species_mix : null | "neighbor" | "pair"; per-layer linear mix of the radial channels by the
                       sender species (GRACE W_n(mu_j), review §12.3) or the species pair (default null)
    x_norm           : null | "rms" | "hill" | "spectral"; bound the basis factor x_i before the
                       product: per-node RMS / RMS + learned degree gate / per-block operator-norm
                       normalisation of the x projection (review §12.3, §12.5)          (default null)
    feature_norm     : null | "rms"; per-node RMS normalisation of the features passed to the next
                       layer                                                            (default null)
    tie_operator     : bool, layers 1.. share one operator and x projection (own readouts); with
                       residual "chebyshev" the strata are T_t(M) u_0 (review §12.4)    (default false)
    bond_order       : null | "power" | "log"; homogeneous bond-order scale stratum
                       -s_Z phi(b_i)(1 + g(shape)), b_i the second-moment width (review §12.7) (default null)
    bond_order_p     : null (learned exponent, init 1) | float (fixed)                   (default null)
    bond_order_shape : bool, shape factor (1 + g(q_i)) on the bond-order term          (default true)
    sym_init_scale   : float, init of the per-degree gain of the contraction terms       (default 0.1)
    sym_target       : "features" (product enters the next layer, MACE-like) | "readout" (product
                       feeds only the stratum energy; features stay nu=1)                 (default "features")
    node_weight      : {"mode": "strain"|"force", "alpha": a, "scale": s, "cap": c} per-atom force-loss
                       weight 1 + a*min(x/s, c) (x = bond strain [A] or |F_ref| [eV/A]), mean 1  (default null)
    force_encoding   : bool, DeNS force encoding for the noisy_nodes denoising task (review §11.17): the
                       denoising pass receives Linear(F_ref) as a 1o input block; energy pass gets zeros (default false)
    dens             : faithful DeNS (docs/RACE_DeNS_audit_2026-10-29.md), replaces noisy_nodes / force_encoding:
                       {"p": 0.25, "r": 0.25, "sigma": 0.05, "weight": w0, "decay": "linear"|"none",
                        "noisy_ef": true}  graphs picked with p, their atoms corrupted with r; additive l<=3
                       force encoding after layer 0 and gated-MLP noise head (RACE dens); L2MAE noise loss with
                       weight w0 (1 - epoch/n_epochs); noisy_ef true = one pass on the corrupted structure (energy
                       vs clean label, forces on uncorrupted atoms), false = clean pass + extra noisy pass for the
                       noise loss only                                                     (default null)
    global_batch_size: graphs per optimizer step, split over the devices (overrides batch_size =
                       graphs per device; multi-GPU / multi-node data parallel, review §13)   (default null)
    report_final_ema : bool, also evaluate the final-epoch EMA parameters (summary 'final_ema') (default false)
    soap_impl        : "reference" (per-leaf bam_omat24/soap_optax.py) | "grouped" (bam_omat24/soap_grouped.py:
                       same maths, same-shape leaves batched; opt_state layout differs)  (default "reference")
    soap_orth        : "householder" | "cholqr" (grouped only; regularised shifted CholeskyQR3)  (default householder)
    matmul_precision : null | "highest" | "high" | "default": jax_default_matmul_precision (float32 runs: "highest"
                       disables TF32)                                                          (default null)
    soap_species_adam: bool (grouped only): species-indexed parameter tables (leading dim = n_species) are updated
                       by Adam (same b1 / b2 / eps) instead of SOAP                            (default false)
    soap_state_dtype : null | "float64" (grouped only): SOAP moments / factors / eigenbases in float64 while the
                       model runs in float32 (turns on jax_enable_x64; params and data stay float32)  (default null)
    gauge_refit      : bool, after training refit the per-atom energy shift in closed form on the training
                       split (least squares of the energy residuals; forces are unchanged: the force-null
                       gauge direction, VarPro), report test metrics as summary 'gauge' and write
                       <ckpt>.gauge.pkl (review §13, literature sweep 2)                        (default false)
    x_readout        : "mlp" (default) | "off" | "layer0" | "linear"  per-layer x readout energy (review §11.16)
    x_readout_norm   : bool, layer norm on the x readout input                                 (default true)
    f_readout        : bool, per-layer readout of the layer input features                      (default true)
    radial_prior     : {"type": "algebraic"|"exponential"|"gaussian", "sigma": s, "p": p} regularity prior:
                       Bessel channel n rescaled by 1/gamma(n) (review §11.15)                (default null)
                       ("sigma" may be a per-layer list)
    angular_prior    : {"sigma_l": s} Y_l scaled by exp(-s^2 l^2) (angular regularity prior)  (default null)
    radial_l2        : float, L2 on the radial MLP first-layer weights (with radial_prior = Tikhonov
                       prior gamma(n)^2 c_n^2)                                                (default 0)
    taylor_consistency: {"weight": w, "eps": A, "k2": 0.5} PICPS Taylor-consistency loss between the
                       model's E(S+d) and its own Taylor estimate (no labels; 2nd forward per step) (default null)
    taylor_aug       : {"n": k, "delta": A} add k displaced copies per training frame with first-order
                       Taylor energies E - F.d (energy-only, no DFT; review §11.14)        (default null)
    init_params_from : checkpoint path; start from its EMA params with a fresh optimiser (stage-two
                       fine-tune: set learning_rate / energy_weight / n_epochs for the stage)  (default null)
    init_from_ema    : bool, use the EMA (true) or raw params of that checkpoint              (default true)
    x_basis_lmax     : int|null, x_mode "basis": keep only l <= lmax of the atomic basis A_i (0 = scalar
                       density gating only)                                                   (default null)
    x_basis_layers   : [i, ...]|null, x_mode "basis": layers that use A_i; the others use the per-species
                       x (v7_ab behaviour)                                                     (default null)
    shell_sk_gate    : bool, Slater-Koster (a,b,l) selection-rule gate on Y_l (needs radial_shells) (default false)
    strata_resum     : null | "aitken", geometric-tail resummation stratum (lambda init 0)   (default null)
    pair_repulsion   : false | true (learnable Born–Mayer stratum) | "zbl" (fixed ZBL universal repulsion) | "uff" (fixed UFF exp-6
                       repulsive part, Rappé 1992 constants); pair_repulsion_scale,
                       pair_repulsion_switch = c -> non-bonded only, switch at c (R_i + R_j) covalent radii,
                       pair_repulsion_switch_width w (A, default 0.1))
    radial_gaussians : {"n": K, "r_min": A, "r_max": A, "width": A} Gaussian radial channels near the bond lengths
    hidden_irreps    : explicit irreps string (e.g. full parity); default hidden_size*s2_irreps(lmax)
    beta_nll         : beta for beta-NLL (gauss_nll only)     (default 0.0)
    model            : "RACE" (default) | "RACE_LR" (+ charge equilibration and long-range electrostatics,
                       race_lr_nnx.py; options in "long_range": ewald_alpha, ewald_n_real, ewald_n_recip, head_hidden,
                       chi_scale, hard_init) | "Nequip"
    total_charge_key : extxyz info key of the total charge per structure (RACE_LR; default none = neutral)
    total_spin_key   : extxyz info key of the spin multiplicity 2S+1 (RACE_LR charge/spin embedding; default none)
    charge_label_key : extxyz per-atom array of reference charges, e.g. "hirshfeld_charges" (RACE_LR; default none)
    charge_weight    : weight of the charge loss mean_i (q_i - q_ref_i)^2 in e^2 (needs charge_label_key; default 0 =
                       latent charges, no charge labels used)
    graph_cache      : pickle cache of the graphs next to each xyz (default true); false recomputes the neighbour
                       lists at start and writes nothing (disk = the xyz files only)
    graph_workers    : processes for the neighbour searches (default 1; 0 = CPUs of the Slurm task, <= 64)
    pad_mode         : "bucket" (default) | "fixed" (one padded shape; pad_nodes, pad_edges, pack_max_graphs);
                       fixed also writes sizes_<split>.tsv (index, n_atoms, n_edges) next to log_file
    use_dataset_stats: init shift/scale/avg_n_neighbors from the training set
                                                              (default false)
    e0_mode          : "isolated" | "regression"              (default "isolated")
    periodic         : bool, periodic cells (graphs built with image shifts Sij, cell, volume; the loss stays
                       energy + forces unless stress_loss)                                     (default false)
    stress_loss      : bool, add the stress term (stress_weight, eV/A^3, ASE sign/Voigt order) to the loss and
                       report stress metrics; needs periodic and a 'stress' global in the data (default false)
    e0_rank_deficient: "reference" (default) | "minnorm"; with e0_mode "regression" and a rank-deficient
                       composition matrix (single stoichiometry) keep the reference E0 (per-species values
                       are not identifiable) or use the old minimum-norm split
    loss_schedule    : [{"until_epoch": N, "energy_weight": .., "force_weight": ..}, ...]
    grad_clip        : global-norm clip (null disables)       (default 1.0)
    restart          : resume from the newer of ckpt_name (best validation) and <ckpt>.last.pkl (written every
                       ckpt_last_every epochs and at the end: params, EMA, optimiser / schedule state, step,
                       best_val_loss); the data-loader RNG is advanced to the resumed epoch and the loss log
                       is appended, so a resumed run continues the uninterrupted one       (default false)
    ckpt_last_every  : epochs between <ckpt>.last.pkl saves (0 = never)          (default 25)
    eval_test_every  : epochs between test-set evaluations    (default 100)
"""

from typing import Dict, Any, List
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import time
import numpy as np
from pathlib import Path
import json
import math
import sys
import functools
import os

from bam_omat24.data_nnx import (BucketedDataLoader, MultiDeviceDataLoader, setup_mesh, default_collate_fn,
                                 fixed_collate_fn, to_host)
from bam_omat24.distributed import init_distributed
from bam_omat24.data_3bpa import (
    load_3bpa_graphs, load_frames, fit_e0, compute_dataset_stats, species_counts, e0_rank,
    split_train_valid, GraphListDataset, atomic_numbers_to_indices,
)
from bam_omat24.nequip_nnx import Nequip
from bam_omat24.logger import Logger
from bam_omat24.race_nnx import RACE
from bam_omat24.training_utils import (
    replicate,
    replicate_pytree,
    unreplicate_pytree,
    make_sharded_train_step,
    make_sharded_evaluate_step,
    evaluate_sharded,
    save_checkpoint,
    load_checkpoint,
    gauge_shift_params,
)

jax.config.update("jax_enable_x64", False)

# 3BPA defaults: H, C, N, O
DEFAULT_ATOMIC_NUMBERS = [1, 6, 7, 8]
DEFAULT_ATOM_ENERGIES = [
    -13.587222780835477,
    -1029.4889999855063,
    -1484.9814568572233,
    -2041.9816003861047,
]


def _get(cfg: Dict, key: str, default: Any = None) -> Any:
    v = cfg.get(key, default)
    return default if v is None else v


# =============================================================================
# Loss-weight schedule
# =============================================================================

# Covalent radii (Cordero et al. 2008), A; initial per-species radii for radial_species_scale
COVALENT_RADII = {1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
                  10: 0.58, 11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02,
                  18: 1.06, 19: 2.03, 20: 1.76, 21: 1.70, 22: 1.60, 23: 1.53, 24: 1.39, 25: 1.39,
                  26: 1.32, 27: 1.26, 28: 1.24, 29: 1.32, 30: 1.22, 31: 1.22, 32: 1.20, 33: 1.19,
                  34: 1.20, 35: 1.20, 36: 1.16, 53: 1.39,
                  # completed to Z = 96 (Cordero et al. 2008) for the element descriptor, 2026-10-27
                  37: 2.20, 38: 1.95, 39: 1.90, 40: 1.75, 41: 1.64, 42: 1.54, 43: 1.47, 44: 1.46, 45: 1.42,
                  46: 1.39, 47: 1.45, 48: 1.44, 49: 1.42, 50: 1.39, 51: 1.39, 52: 1.38, 54: 1.40, 55: 2.44,
                  56: 2.15, 57: 2.07, 58: 2.04, 59: 2.03, 60: 2.01, 61: 1.99, 62: 1.98, 63: 1.98, 64: 1.96,
                  65: 1.94, 66: 1.92, 67: 1.92, 68: 1.89, 69: 1.90, 70: 1.87, 71: 1.87, 72: 1.75, 73: 1.70,
                  74: 1.62, 75: 1.51, 76: 1.44, 77: 1.41, 78: 1.36, 79: 1.36, 80: 1.32, 81: 1.45, 82: 1.46,
                  83: 1.48, 84: 1.40, 85: 1.50, 86: 1.50, 87: 2.60, 88: 2.21, 89: 2.15, 90: 2.06, 91: 2.00,
                  92: 1.96, 93: 1.90, 94: 1.87, 95: 1.80, 96: 1.69}

# Valence-shell table from the spherical LDA atom solver (bam_omat24/atomic_shells.py)
SHELL_TABLE_PATH = Path(__file__).with_name('atomic_shells.json')
BOHR_A = 0.529177210903
SHELL_L = "spdf"


def _shell_table() -> Dict:
    return json.load(open(SHELL_TABLE_PATH))


def _descriptor_row(entry: Dict, z: int, source: str) -> List[float]:
    row: List[float] = []
    if source in ("shells", "both"):
        for l, name in enumerate(SHELL_L):
            sh = entry["shells"][name]
            if sh is None:
                row += [0.0, 0.0, 0.0, 0.0]
            else:
                row += [1.0, math.log(sh["r_mean"] * BOHR_A), math.log(-sh["eps"]), sh["occ"] / (2.0 * (2 * l + 1))]
        row.append(entry["n_valence"] / 18.0)
    if source in ("radii", "both"):
        row.append(COVALENT_RADII[z])
    return row


def element_descriptors(atomic_numbers, source: str = "both") -> List[List[float]]:
    """Fixed per-element descriptor rows for ``embed_mode`` != "onehot": per l (s, p, d, f) of the
    outermost occupied shell [present, log <r>/A, log(-eps/Ha), occupation fraction], the valence
    electron count, and the covalent radius; standardised with the mean / std over Z = 1..94 so the
    rows do not depend on which species a run uses."""
    if source not in ("shells", "radii", "both"):
        raise ValueError(f"descriptor_source must be 'shells', 'radii' or 'both', got {source!r}")
    table = _shell_table()
    all_rows = np.array([_descriptor_row(table[str(z)], z, source) for z in range(1, 95)], dtype=np.float64)
    mu, sd = all_rows.mean(0), all_rows.std(0) + 1e-8
    return [((np.array(_descriptor_row(table[str(int(z))], int(z), source)) - mu) / sd).tolist()
            for z in atomic_numbers]


def _radial_gaussians(spec):
    """Config {"n", "r_min", "r_max", "width"} (or explicit {"centers", "width"}) -> model dict."""
    if spec is None:
        return None
    if "centers" in spec:
        return {"centers": [float(c) for c in spec["centers"]], "width": float(spec["width"])}
    centers = np.linspace(float(spec.get("r_min", 0.9)), float(spec.get("r_max", 2.3)), int(spec.get("n", 8)))
    return {"centers": centers.tolist(), "width": float(spec.get("width", 0.15))}


def element_shell_radii(atomic_numbers) -> List[List[float]]:
    """(n_species, 4): <r> in A of the outermost occupied s, p, d, f shell (0 = shell absent)."""
    table = _shell_table()
    out = []
    for z in atomic_numbers:
        sh = table[str(int(z))]["shells"]
        out.append([0.0 if sh[name] is None else sh[name]["r_mean"] * BOHR_A for name in SHELL_L])
    return out


def loss_weights_for_epoch(cfg: Dict, epoch: int) -> Dict[str, float]:
    """Return {energy_weight, force_weight, stress_weight} for a 0-based epoch.

    ``loss_schedule`` is a list of stages, each with ``until_epoch`` (exclusive,
    may be omitted for the last stage) and optional weights; missing weights
    fall back to the global ``energy_weight``/``force_weight``/``stress_weight``.
    """
    base = {
        'energy_weight': float(_get(cfg, 'energy_weight', 1.0)),
        'force_weight': float(_get(cfg, 'force_weight', 1.0)),
        'stress_weight': float(_get(cfg, 'stress_weight', 1.0)),
    }
    sched = _get(cfg, 'loss_schedule', None)
    if not sched:
        return base
    for stage in sched:
        until = stage.get('until_epoch', None)
        if until is None or epoch < until:
            out = dict(base)
            for k in base:
                if k in stage:
                    out[k] = float(stage[k])
            return out
    # past the last stage: keep the last stage's weights
    out = dict(base)
    for k in base:
        if k in sched[-1]:
            out[k] = float(sched[-1][k])
    return out


# =============================================================================
# Data
# =============================================================================

def _graph_workers(cfg: Dict) -> int:
    """graph_workers: processes for the neighbour searches (0 = the CPUs of this Slurm task, at most 64)."""
    import os
    w = int(_get(cfg, 'graph_workers', 1))
    if w == 0:
        w = min(64, int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or len(os.sched_getaffinity(0)))
    return max(1, w)


def build_datasets(cfg: Dict, fout):
    atomic_numbers = list(_get(cfg, 'atomic_numbers', DEFAULT_ATOMIC_NUMBERS))
    atom_indices = atomic_numbers_to_indices(atomic_numbers)
    cutoff = float(_get(cfg, 'cutoff', 5.0))
    train_path = Path(_get(cfg, 'train_path', 'dataset_3BPA/train_300K.xyz'))

    train_frames = load_frames(train_path)
    ref_e0 = np.asarray(_get(cfg, 'atom_energies', DEFAULT_ATOM_ENERGIES), dtype=np.float64)
    e0_mode = _get(cfg, 'e0_mode', 'isolated')
    e0_info = {}
    if e0_mode == 'regression':
        # Per-species E0 are identifiable only if the composition matrix has full rank.  3BPA has
        # one stoichiometry (rank 1): then the reference isolated-atom E0 are kept and the
        # identifiable per-frame offset goes to the model shift (e0_rank_deficient "reference"),
        # unless "minnorm" asks for the old minimum-norm split (arbitrary per species).
        atom_energies = fit_e0(train_frames, atom_indices, reference_e0=ref_e0,
                               on_rank_deficient=_get(cfg, 'e0_rank_deficient', 'reference'), info=e0_info)
        print(f"E0 (regression): composition rank {e0_info['rank']}/{e0_info['n_species']}, "
              f"{e0_info['action']}; per-atom offset of the targets {e0_info['per_atom_offset']:.4f} eV/atom; "
              f"delta vs isolated: {np.array2string(atom_energies - ref_e0, precision=4)}", file=fout)
    elif e0_mode == 'isolated':
        atom_energies = ref_e0
        counts_tmp = species_counts(train_frames, atom_indices)
        print(f"E0 (isolated-atom reference): composition rank {e0_rank(counts_tmp)}/{counts_tmp.shape[1]} "
              f"({'per-species E0 identifiable' if e0_rank(counts_tmp) == counts_tmp.shape[1] else 'single stoichiometry: only the per-frame offset is identifiable'})",
              file=fout)
    else:
        raise ValueError(f"unknown e0_mode {e0_mode!r}")
    atom_energies = [float(x) for x in atom_energies]

    # graph_cache false: no pickle cache (neighbour lists recomputed at start, graph_workers processes)
    gl = dict(use_cache=bool(_get(cfg, 'graph_cache', True)), n_workers=_graph_workers(cfg),
              charge_key=_get(cfg, 'total_charge_key', None), charge_label_key=_get(cfg, 'charge_label_key', None),
              spin_key=_get(cfg, 'total_spin_key', None))
    all_graphs = load_3bpa_graphs(train_path, atomic_numbers, atom_energies, cutoff,
                                  frames=train_frames, **gl)
    if not bool(_get(cfg, 'periodic', False)) and any('cell' in g.globals for g in all_graphs[:1000]):
        # periodic graphs (pbc in the xyz) with a molecular model ignore the image shifts and the cell: stop
        raise ValueError(f"{train_path} holds periodic structures (pbc) but the config has periodic false; "
                         "set \"periodic\": true")
    valid_path = _get(cfg, 'valid_path', None)
    if valid_path:
        # fixed validation file (benchmark split); every frame of train_path is used for training
        train_graphs = all_graphs
        valid_graphs = load_3bpa_graphs(valid_path, atomic_numbers, atom_energies, cutoff, **gl)
    else:
        train_graphs, valid_graphs = split_train_valid(
            all_graphs, float(_get(cfg, 'valid_frac', 0.1)), seed=int(_get(cfg, 'split_seed', 42))
        )

    test_paths = _get(cfg, 'test_paths', None)
    if test_paths is None:
        tp = _get(cfg, 'test_path', None)
        test_paths = [tp] if tp else []
    test_sets = {
        Path(p).stem: GraphListDataset(load_3bpa_graphs(p, atomic_numbers, atom_energies, cutoff, **gl))
        for p in test_paths
    }

    stats = compute_dataset_stats(train_graphs, n_species=len(atomic_numbers))
    print("Training-set statistics:", file=fout)
    print(f"  n_graphs={stats['n_graphs']} n_atoms={int(stats['n_atoms'])} "
          f"avg_n_neighbors={stats['avg_n_neighbors']:.3f}", file=fout)
    print(f"  per-atom energy mean (shift_raw) = {stats['shift_raw']:.6f} eV/atom", file=fout)
    print(f"  force RMS (scale)                = {stats['scale']:.6f} eV/A", file=fout)
    print(f"  residual E0 (lstsq)              = "
          f"{np.array2string(stats['e0_residual'], precision=4)}", file=fout)

    return (atomic_numbers, atom_energies, GraphListDataset(train_graphs),
            GraphListDataset(valid_graphs), test_sets, stats)


# =============================================================================
# Model / optimizer
# =============================================================================

def build_model(cfg: Dict, n_species: int, atom_energies: List[float], stats: Dict, fout, n_devices: int = 1):
    seed = int(_get(cfg, 'seed', 42))
    model_rngs = nnx.Rngs(seed)
    lmax = int(_get(cfg, 'lmax', 3))

    import e3nn_jax as e3nn
    hidden_irreps = _get(cfg, 'hidden_irreps', None)
    if hidden_irreps is None:
        hidden_size = int(_get(cfg, 'hidden_size', 128))
        hidden_irreps = str(hidden_size * e3nn.s2_irreps(lmax))

    if _get(cfg, 'use_dataset_stats', False):
        shift = float(_get(cfg, 'shift', stats['shift_raw']))
        scale = float(_get(cfg, 'scale', stats['scale']))
        avg_n_neighbors = float(_get(cfg, 'avg_n_neighbors', stats['avg_n_neighbors']))
    else:
        shift = float(_get(cfg, 'shift', 0.0))
        scale = float(_get(cfg, 'scale', 1.0))
        avg_n_neighbors = float(_get(cfg, 'avg_n_neighbors', 25.0))
    print(f"Model init: shift={shift:.6f} scale={scale:.6f} avg_n_neighbors={avg_n_neighbors:.3f}",
          file=fout)

    model_type = _get(cfg, 'model', 'RACE')
    common = dict(
        n_species=n_species,
        lmax=lmax,
        cutoff=float(_get(cfg, 'cutoff', 5.0)),
        hidden_irreps=hidden_irreps,
        n_layers=int(_get(cfg, 'n_layers', 5)),
        radial_basis_size=int(_get(cfg, 'radial_basis_size', 8)),
        radial_mlp_size=int(_get(cfg, 'radial_mlp_size', 64)),
        radial_mlp_layers=int(_get(cfg, 'radial_mlp_layers', 3)),
        radial_polynomial_p=float(_get(cfg, 'radial_polynomial_p', 2.0)),
        mlp_init_scale=float(_get(cfg, 'mlp_init_scale', 4.0)),
        shift=shift,
        scale=scale,
        avg_n_neighbors=avg_n_neighbors,
        atom_energies=atom_energies,
        l_train=True,
        periodic=bool(_get(cfg, 'periodic', False)),
        rngs=model_rngs,
    )
    if model_type == 'Nequip':
        model = Nequip(**common)
    else:
        race_cls, lr_kw = RACE, {}
        if model_type == 'RACE_LR':      # RACEv2 + charge equilibration + long-range electrostatics (race_lr_nnx.py)
            from bam_omat24.race_lr_nnx import RACE_LR
            race_cls, lr_kw = RACE_LR, dict(long_range=_get(cfg, 'long_range', None))
        model = race_cls(**common, **lr_kw,
                     x_mode=_get(cfg, 'x_mode', 'species'),
                     x_basis_lmax=_get(cfg, 'x_basis_lmax', None),
                     x_basis_layers=_get(cfg, 'x_basis_layers', None),
                     radial_prior=_get(cfg, 'radial_prior', None),
                     angular_prior=_get(cfg, 'angular_prior', None),
                     x_readout=_get(cfg, 'x_readout', 'mlp'),
                     force_encoding=bool(_get(cfg, 'force_encoding', False)),
                     dens=bool(_get(cfg, 'dens', None)),
                     dens_seed=seed + 7919,
                     x_readout_norm=bool(_get(cfg, 'x_readout_norm', True)),
                     f_readout=bool(_get(cfg, 'f_readout', True)),
                     cov_mode=_get(cfg, 'cov_mode', 'scalar6'),
                     tp_backend=_get(cfg, 'tp_backend', 'e3nn'),
                     pair_term=bool(_get(cfg, 'pair_term', False)),
                     stratum_gain=_get(cfg, 'stratum_gain', 'uniform'),
                     readout_mode=_get(cfg, 'readout_mode', 'linear'),
                     readout_species=bool(_get(cfg, 'readout_species', False)),
                     residual=_get(cfg, 'residual', 'none'),
                     layer_scale=_get(cfg, 'layer_scale', None),
                     noise_head=bool(_get(cfg, 'noisy_nodes', None)),
                     layer_lmax=_get(cfg, 'layer_lmax', None),
                     layer_cutoff=_get(cfg, 'layer_cutoff', None),
                     radial_species_scale=bool(_get(cfg, 'radial_species_scale', False)),
                     radial_slater=bool(_get(cfg, 'radial_slater', False)),
                     species_radii=[COVALENT_RADII[int(z)] for z in cfg['atomic_numbers']],
                     embed_mode=_get(cfg, 'embed_mode', 'onehot'),
                     species_descriptors=(element_descriptors(cfg['atomic_numbers'], _get(cfg, 'descriptor_source', 'both'))
                                          if _get(cfg, 'embed_mode', 'onehot') != 'onehot' else None),
                     radial_shells=bool(_get(cfg, 'radial_shells', False)),
                     radial_shell_scale=bool(_get(cfg, 'radial_shell_scale', False)),
                     in_layer_order=int(_get(cfg, 'in_layer_order', 1)),
                     in_layer_order_layers=_get(cfg, 'in_layer_order_layers', None),
                     sym_contraction=bool(_get(cfg, 'sym_contraction', False)),
                     sym_init_scale=float(_get(cfg, 'sym_init_scale', 0.1)),
                     sym_method=_get(cfg, 'sym_method', None),
                     sym_target=_get(cfg, 'sym_target', 'features'),
                     shell_sk_gate=bool(_get(cfg, 'shell_sk_gate', False)),
                     strata_resum=_get(cfg, 'strata_resum', None),
                     species_shell_radii=(element_shell_radii(cfg['atomic_numbers'])
                                          if _get(cfg, 'radial_shells', False) else None),
                     pair_repulsion=_get(cfg, 'pair_repulsion', False),
                     pair_repulsion_scale=float(_get(cfg, 'pair_repulsion_scale', 1.0)),
                     pair_repulsion_switch=_get(cfg, 'pair_repulsion_switch', None),
                     pair_repulsion_switch_width=float(_get(cfg, 'pair_repulsion_switch_width', 0.1)),
                     species_z=[int(z) for z in cfg['atomic_numbers']],
                     radial_gaussians=_radial_gaussians(_get(cfg, 'radial_gaussians', None)),
                     update=_get(cfg, 'update', 'gate'),
                     message_norm=_get(cfg, 'message_norm', 'layer'),
                     radial_species_mix=_get(cfg, 'radial_species_mix', None),
                     x_norm=_get(cfg, 'x_norm', None),
                     feature_norm=_get(cfg, 'feature_norm', None),
                     tie_operator=bool(_get(cfg, 'tie_operator', False)),
                     bond_order=_get(cfg, 'bond_order', None),
                     bond_order_p=_get(cfg, 'bond_order_p', None),
                     bond_order_shape=bool(_get(cfg, 'bond_order_shape', True)))

    graphdef, params = nnx.split(model, nnx.Param)
    if _get(cfg, 'x64', False):
        # float64 training (MACE/BOTNet 3BPA protocol): parameters are created in float32
        params = jax.tree.map(lambda x: x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x, params)
    elif jax.config.jax_enable_x64:
        # x64 switched on only for the optimizer state (soap_state_dtype float64): float32 parameters
        params = jax.tree.map(lambda x: x.astype(jnp.float32) if jnp.issubdtype(x.dtype, jnp.floating) else x, params)
    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"Model: {model_type} hidden_irreps={hidden_irreps} "
          f"x_mode={_get(cfg, 'x_mode', 'species')} cov_mode={_get(cfg, 'cov_mode', 'scalar6')} "
          f"tp_backend={_get(cfg, 'tp_backend', 'e3nn')} readout_mode={_get(cfg, 'readout_mode', 'linear')} "
          f"readout_species={bool(_get(cfg, 'readout_species', False))} residual={_get(cfg, 'residual', 'none')} "
          f"layer_scale={_get(cfg, 'layer_scale', None)} layer_lmax={_get(cfg, 'layer_lmax', None)} "
          f"layer_cutoff={_get(cfg, 'layer_cutoff', None)} radial_species_scale={_get(cfg, 'radial_species_scale', False)} "
          f"radial_slater={_get(cfg, 'radial_slater', False)} noise_head={bool(_get(cfg, 'noisy_nodes', None))} "
          f"dens={_get(cfg, 'dens', None)} "
          f"embed_mode={_get(cfg, 'embed_mode', 'onehot')} descriptor_source={_get(cfg, 'descriptor_source', 'both')} "
          f"radial_shells={bool(_get(cfg, 'radial_shells', False))} radial_shell_scale={bool(_get(cfg, 'radial_shell_scale', False))} in_layer_order={_get(cfg, 'in_layer_order', 1)} in_layer_order_layers={_get(cfg, 'in_layer_order_layers', None)} sym_contraction={bool(_get(cfg, 'sym_contraction', False))} sym_init_scale={_get(cfg, 'sym_init_scale', 0.1)} sym_target={_get(cfg, 'sym_target', 'features')} shell_sk_gate={bool(_get(cfg, 'shell_sk_gate', False))} strata_resum={_get(cfg, 'strata_resum', None)} pair_repulsion={bool(_get(cfg, 'pair_repulsion', False))} "
          f"radial_gaussians={_get(cfg, 'radial_gaussians', None)} update={_get(cfg, 'update', 'gate')} "
          f"message_norm={_get(cfg, 'message_norm', 'layer')} x64={bool(_get(cfg, 'x64', False))} "
          f"energy_loss_per={_get(cfg, 'energy_loss_per', 'atom')}",
          file=fout)
    print(f"Model parameters: {n_params:,}", file=fout)

    grad_clip = _get(cfg, 'grad_clip', 1.0)
    chain = []
    if grad_clip is not None and grad_clip > 0:
        chain.append(optax.clip_by_global_norm(float(grad_clip)))
    opt_name = _get(cfg, 'optimizer', 'amsgrad')
    lr = float(cfg['learning_rate'])
    wd = float(_get(cfg, 'weight_decay', 0.0))
    swd = _get(cfg, 'stratum_weight_decay', None)
    if opt_name == 'amsgrad':
        chain.append(optax.scale_by_amsgrad() if swd else optax.amsgrad(lr))
    elif opt_name == 'adam':
        # adam_b1 / adam_b2 (default optax 0.9 / 0.999); b1 = b2 = 0.95 is the attribution control for
        # SOAP (same moments, no Kronecker eigenbasis rotation)
        ab = dict(b1=float(_get(cfg, 'adam_b1', 0.9)), b2=float(_get(cfg, 'adam_b2', 0.999)))
        chain.append(optax.scale_by_adam(**ab) if swd else optax.adam(lr, **ab))
    elif opt_name == 'adamw':
        chain.append(optax.chain(optax.scale_by_adam(), optax.add_decayed_weights(wd)) if swd
                     else optax.adamw(lr, weight_decay=wd))
    elif opt_name == 'soap':
        # SOAP (arXiv:2409.11321): Adam in the eigenbasis of Shampoo's L, R factors
        from bam_omat24.soap_optax import scale_by_soap, soap_leaf_summary
        soap_sb = _get(cfg, 'soap_shampoo_beta', None)
        soap_kw = dict(
            b1=float(_get(cfg, 'soap_b1', 0.95)),
            b2=float(_get(cfg, 'soap_b2', 0.95)),
            precondition_frequency=int(_get(cfg, 'soap_precondition_frequency', 10)),
            max_precond_dim=int(_get(cfg, 'soap_max_precond_dim', 10000)),
            shampoo_beta=None if soap_sb is None else float(soap_sb),
        )
        soap_impl = _get(cfg, 'soap_impl', 'reference')
        if soap_impl == 'grouped':
            # same maths, same-shape leaves batched (docs/RACE_SOAP_speedup_plan_2026-10-29.md)
            from bam_omat24.soap_grouped import scale_by_soap_grouped
            soap_fn = functools.partial(scale_by_soap_grouped, orth=_get(cfg, 'soap_orth', 'householder'),
                                        state_dtype=_get(cfg, 'soap_state_dtype', None),
                                        adam_leading_dim=(n_species if _get(cfg, 'soap_species_adam', False) else None))
        elif soap_impl == 'reference':
            if _get(cfg, 'soap_state_dtype', None):
                raise ValueError("soap_state_dtype needs soap_impl='grouped'")
            soap_fn = scale_by_soap
        else:
            raise ValueError(f"soap_impl must be 'reference' or 'grouped', got {soap_impl!r}")
        soap_tx = [soap_fn(**soap_kw)] + ([optax.add_decayed_weights(wd)] if wd > 0 else [])
        if swd:   # as adamw: SOAP -> wd -> (stratum-decay block below: decays + lr)
            chain.append(optax.chain(*soap_tx))
        else:
            chain += soap_tx + [optax.scale_by_learning_rate(lr)]
        summ = soap_leaf_summary(params, max_precond_dim=soap_kw['max_precond_dim'])
        print(f"SOAP: {soap_kw} weight_decay={wd} stratum_weight_decay={swd} "
              f"preconditioned parameters {summ['counts']} "
              f"state {summ['state_bytes'] / 2**20:.0f} MiB", file=fout)
        for path, shape, cls in summ['partial']:
            print(f"  SOAP side > max_precond_dim, preconditioned side(s) = {cls}: {shape} {path}",
                  file=fout)
    else:
        raise ValueError(f"unknown optimizer {opt_name!r}")
    # Stratum-wise (body-order) weight decay: wd_i = base * (i+1)**power on the
    # parameters of layer i (readout heads only, or the whole layer).  The explicit
    # pair term (stratum 0) is never decayed.  Mirrors the hierarchical regularisation
    # of ACE coefficients: higher body order -> stronger penalty.
    # Decoupled (AdamW-style): the decay is added *after* the Adam normalisation and
    # before the learning-rate scaling.  (Adding it to the raw gradient, before Adam,
    # made the decay scale-invariant and collapsed whole strata: s500_strata run.)
    if swd:
        base = float(swd.get('base', 1e-5)); power = float(swd.get('power', 1.0))
        scope = swd.get('scope', 'readout')
        n_layers = int(_get(cfg, 'n_layers', 5))

        def _layer_of(path):
            keys = [getattr(k, 'key', getattr(k, 'idx', getattr(k, 'name', None))) for k in path]
            keys = [str(k) for k in keys]
            for grp in ('layers', 'x_readouts', 'f_readouts', 'x_projs'):
                if grp in keys:
                    gi = keys.index(grp)
                    try:
                        li = int(keys[gi + 1])
                    except (ValueError, IndexError):
                        return None
                    if grp == 'layers' and scope == 'readout' and not any(k.startswith('readout') for k in keys[gi + 2:]):
                        return None
                    return li
            return None

        def _mask_for(layer_idx):
            return jax.tree_util.tree_map_with_path(lambda path, _: _layer_of(path) == layer_idx, params)

        if scope == 'all':
            # global decoupled decay of every parameter (embedding, x projections, radial MLPs,
            # readouts, shift / scale): one strength, no per-layer power
            n_dec = sum(int(np.size(x)) for x in jax.tree.leaves(params))
            chain = chain + [optax.add_decayed_weights(base), optax.scale_by_learning_rate(lr)]
            print(f"Global weight decay (decoupled): {base} on all {len(jax.tree.leaves(params))} leaves "
                  f"({n_dec:,} parameters)", file=fout)
        else:
            decays = []
            for li in range(n_layers):
                wd_i = base * (li + 1) ** power
                decays.append(optax.add_decayed_weights(wd_i, mask=_mask_for(li)))
            chain = chain + decays + [optax.scale_by_learning_rate(lr)]
            print(f"Stratum weight decay (decoupled): base={base} power={power} scope={scope} "
                  f"-> wd per layer {[round(base*(i+1)**power, 8) for i in range(n_layers)]}", file=fout)
    lr_mode = _get(cfg, 'lr_schedule', 'plateau')
    cosine_fn = None
    if lr_mode == 'cosine':
        steps_per_epoch = max(int(stats['n_graphs']) // (int(_get(cfg, 'batch_size', 5)) * n_devices), 1)
        total_steps = int(_get(cfg, 'n_epochs', 1000)) * steps_per_epoch
        warmup_steps = int(_get(cfg, 'lr_warmup_epochs', 10)) * steps_per_epoch
        min_scale = float(_get(cfg, 'lr_cosine_min_scale', 0.01))
        # relative factor (the lr itself is inside the Adam transform above)
        cosine_fn = optax.warmup_cosine_decay_schedule(
            init_value=min_scale, peak_value=1.0, warmup_steps=warmup_steps,
            decay_steps=total_steps, end_value=min_scale)
        chain.append(optax.scale_by_schedule(cosine_fn))
        print(f"LR schedule: cosine, warmup {warmup_steps} steps, total {total_steps} steps, "
              f"min scale {min_scale} (plateau scaling disabled)", file=fout)
    elif lr_mode != 'plateau':
        raise ValueError(f"unknown lr_schedule {lr_mode!r}")
    optimizer = optax.chain(*chain)
    print(f"Optimizer: {opt_name} lr={lr} grad_clip={grad_clip} pair_term={_get(cfg, 'pair_term', False)} "
          f"stratum_gain={_get(cfg, 'stratum_gain', 'uniform')}", file=fout)

    if lr_mode == 'cosine':
        # plateau transform kept for the state/plumbing but made inert (never fires)
        schedule = optax.contrib.reduce_on_plateau(factor=0.5, patience=10**8, min_scale=1.0)
    else:
        schedule = optax.contrib.reduce_on_plateau(
            factor=float(_get(cfg, "lr_schedule_factor", 0.8)),
            patience=int(_get(cfg, "lr_schedule_patience", 10)),
            min_scale=float(_get(cfg, "lr_schedule_min_scale", 0.001)),
        )
    build_model.cosine_fn = cosine_fn
    return graphdef, params, optimizer, schedule, grad_clip


# =============================================================================
# Entry point
# =============================================================================

def _fmt_metrics(m: Dict) -> str:
    return (f"E_MAE: {m['energy_mae_total']*1e3:7.2f} meV | "
            f"E_RMSE: {m['energy_rmse_total']*1e3:7.2f} meV | "
            f"F_MAE: {m['force_mae']*1e3:7.2f} meV/A | "
            f"F_RMSE: {m['force_rmse']*1e3:7.2f} meV/A"
            + (f" | E_MAE/atom: {m['energy_mae']*1e3:7.2f} meV | S_MAE: {m['stress_mae_comp']*160.21766:6.3f} GPa | "
               f"S_RMSE: {m['stress_rmse_comp']*160.21766:6.3f} GPa"
               if m.get('stress_rmse_comp', 0.0) > 0 else ""))


def _cast_graphs(graphs, dtype):
    """Cast every floating array of the graphs to dtype (float64 training)."""
    def cast(x):
        x = np.asarray(x)
        return x.astype(dtype) if np.issubdtype(x.dtype, np.floating) else x
    return [jax.tree.map(cast, g) for g in graphs]


def _barrier(name: str):
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices(name)


def train(cfg: Dict):
    # multi-node data parallel (review §13): one JAX process per node or per GPU, launched by srun with
    # BAM_DISTRIBUTED=1 (bam_omat24/__init__.py starts jax.distributed); no-op for a single process
    init_distributed()
    if os.environ.get("BAM_STACK_DUMP_SECS"):
        # debugging hangs (e.g. a multi-process collective): print every thread's stack periodically
        import faulthandler
        faulthandler.dump_traceback_later(float(os.environ["BAM_STACK_DUMP_SECS"]), repeat=True)
    if _get(cfg, 'x64', False) or str(_get(cfg, 'soap_state_dtype', None) or '') == 'float64':
        # soap_state_dtype float64 alone: 64-bit types for the SOAP state only; params and data stay float32
        jax.config.update("jax_enable_x64", True)
    if _get(cfg, 'matmul_precision', None):
        # e.g. "highest": no TF32 for float32 matmuls on Ampere / Hopper GPUs
        jax.config.update("jax_default_matmul_precision", str(_get(cfg, 'matmul_precision', None)))
    mesh, n_devices = setup_mesh()
    rank0 = jax.process_index() == 0
    log_path = _get(cfg, 'log_file', 'loss_3bpa.out')
    ckpt_name = _get(cfg, 'ckpt_name', 'ckpt_best.pkl')
    last_name = str(Path(ckpt_name).with_suffix('.last.pkl'))
    resuming = bool(_get(cfg, 'restart', False)) and (Path(ckpt_name).exists() or Path(last_name).exists())
    # only process 0 writes the log, checkpoints and the summary; a resumed run appends to the log
    fout = open(log_path if rank0 else os.devnull, 'a' if resuming else 'w', buffering=1)
    if resuming:
        print(f"\n===== restart {time.strftime('%Y-%m-%d %H:%M:%S')} =====", file=fout)
    print(json.dumps(cfg, indent=1), file=fout)

    loss_type = _get(cfg, 'loss_type', 'huber')
    cov_mode = _get(cfg, 'cov_mode', 'scalar6')
    beta_nll = float(_get(cfg, 'beta_nll', 0.0))
    huber_delta = float(_get(cfg, 'huber_delta', 0.1))
    ema_decay = float(_get(cfg, 'ema_decay', 0.99))
    # batch_size = graphs per device; global_batch_size (optional) = graphs per optimizer step, split
    # over the devices (the single-device run with batch_size = global_batch_size is the same
    # algorithm, training_utils.make_sharded_train_step)
    gbs = _get(cfg, 'global_batch_size', None)
    if gbs is not None:
        if int(gbs) % n_devices:
            raise ValueError(f"global_batch_size {gbs} is not divisible by the {n_devices} devices")
        cfg = {**cfg, 'batch_size': int(gbs) // n_devices}
    batch_size = int(_get(cfg, 'batch_size', 5))
    print(f"data parallel: {jax.process_count()} process(es), {n_devices} device(s), {batch_size} graphs "
          f"per device, {batch_size * n_devices} per step", file=fout)
    n_epochs = int(_get(cfg, 'n_epochs', 1000))
    eval_test_every = int(_get(cfg, 'eval_test_every', 100))
    use_schedule = bool(_get(cfg, 'loss_schedule', None))
    dens_cfg = _get(cfg, 'dens', None)
    if dens_cfg and (_get(cfg, 'noisy_nodes', None) or _get(cfg, 'force_encoding', False)):
        raise ValueError("dens replaces noisy_nodes / force_encoding")
    if dens_cfg:
        unknown = set(dens_cfg) - {'p', 'r', 'sigma', 'weight', 'decay', 'noisy_ef', 'head', 'encoding', 'loss'}
        if unknown:
            raise ValueError(f"unknown dens keys {sorted(unknown)}")
        for k, ok in (('head', 'gated_mlp'), ('encoding', 'add_l0_3'), ('loss', 'l2mae')):
            if dens_cfg.get(k, ok) != ok:
                raise NotImplementedError(f"dens {k}={dens_cfg[k]!r} is not implemented (only {ok!r})")
        dens_cfg = {'p': float(dens_cfg.get('p', 0.25)), 'r': float(dens_cfg.get('r', 0.25)),
                    'sigma': float(dens_cfg.get('sigma', 0.05)), 'weight': float(dens_cfg['weight']),
                    'decay': dens_cfg.get('decay', 'linear'), 'noisy_ef': bool(dens_cfg.get('noisy_ef', True))}
        if dens_cfg['decay'] not in ('linear', 'none'):
            raise ValueError(f"dens decay must be 'linear' or 'none', got {dens_cfg['decay']!r}")
        if dens_cfg['noisy_ef'] and loss_type != 'huber':
            # the corrupted atoms are dropped from the force loss through nodes['loss_weight'], which
            # only the 'huber' branch of training_utils._loss_terms applies
            raise ValueError("dens noisy_ef=true needs loss_type='huber'")
    # loss weights passed to the step each epoch (loss schedule, DeNS weight decay) -> no re-JIT
    dynamic_weights = use_schedule or bool(dens_cfg)

    # process 0 builds the on-disk graph caches first, the others then read them
    if not rank0:
        _barrier("datasets")
    (atomic_numbers, atom_energies, train_dataset, valid_dataset,
     test_sets, stats) = build_datasets(cfg, fout)
    if rank0:
        _barrier("datasets")
    if _get(cfg, 'x64', False):
        for ds in [train_dataset, valid_dataset] + list(test_sets.values()):
            ds.graphs = _cast_graphs(ds.graphs, np.float64)
        print("float64: parameters and data cast to float64", file=fout)
    nn_cfg = _get(cfg, 'noisy_nodes', None)

    # Boltzmann tilting: per-frame loss weight exp((E - E_mean)/kT'), capped, mean 1.
    # Reweights the 300 K training distribution towards an effective temperature
    # 1/(1/T - 1/T') (T' = 400 K, T = 300 K -> 1200 K).
    tilt = _get(cfg, 'boltzmann_tilt', None)
    if tilt:
        kB = 8.617333262e-5
        T_prime = float(tilt.get('T_prime', 1500.0)); w_max = float(tilt.get('w_max', 5.0))
        E = np.array([float(np.asarray(g.globals['energy']).reshape(-1)[0]) for g in train_dataset.graphs])
        # relative to the mean energy: typical frames keep w ~ 1, only the high-energy tail
        # is up-weighted (and capped); relative to E_min the cap would hit almost all frames
        w_raw = np.exp((E - E.mean()) / (kB * T_prime))
        n_cap = int((w_raw >= w_max).sum())
        w = np.minimum(w_raw, w_max)
        w = w / w.mean()
        new_graphs = []
        for g, wi in zip(train_dataset.graphs, w):
            e_shape = np.asarray(g.globals['energy']).shape
            new_graphs.append(g._replace(globals={**g.globals, 'loss_weight': np.full(e_shape, wi, dtype=np.float32)}))
        train_dataset.graphs = new_graphs
        print(f"Boltzmann tilt: T'={T_prime} K, w_max={w_max}: weights min {w.min():.3f} max {w.max():.3f} "
              f"(frames at cap: {n_cap}/{len(w)}), E range {E.min():.3f}..{E.max():.3f} eV",
              file=fout)

    # Per-node force-loss weights (review §11.12): w_i = 1 + alpha * min(x_i / scale, cap), mean 1,
    # x = bond strain max_j |r_ij - <r_ij>_train| over bonded j ("strain", scale in A) or the
    # reference force norm ("force", scale in eV/A).  Targets the rare strained-bond atoms that
    # carry ~half of the 1200 K error (§7.14, §11.8).  Needs a fixed molecule (3BPA).
    nw_cfg = _get(cfg, 'node_weight', None)
    if nw_cfg:
        mode = nw_cfg.get('mode', 'strain')
        alpha = float(nw_cfg.get('alpha', 2.0))
        scale_w = float(nw_cfg.get('scale', 0.1 if mode == 'strain' else 1.0))
        cap = float(nw_cfg.get('cap', 5.0))
        graphs = train_dataset.graphs
        n_at = {int(g.n_node[0]) for g in graphs}
        if len(n_at) != 1:
            raise ValueError(f"node_weight needs a fixed molecule, got atom counts {sorted(n_at)}")
        if mode == 'strain':
            pos = np.stack([np.asarray(g.nodes['positions'], dtype=np.float64) for g in graphs])   # (F, N, 3)
            D = np.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)                   # (F, N, N)
            Dm = D.mean(0)
            N = pos.shape[1]
            bond_cut = float(nw_cfg.get('bond_cut', 1.75))
            x = np.zeros((len(graphs), N))
            for i in range(N):
                nb = np.flatnonzero((Dm[i] < bond_cut) & (np.arange(N) != i))
                if len(nb):
                    x[:, i] = np.max(np.abs(D[:, i, nb] - Dm[i, nb][None]), axis=-1)
        elif mode == 'force':
            x = np.stack([np.linalg.norm(np.asarray(g.nodes['forces'], dtype=np.float64), axis=-1) for g in graphs])
        else:
            raise ValueError(f"node_weight mode must be 'strain' or 'force', got {mode!r}")
        w = 1.0 + alpha * np.minimum(x / scale_w, cap)
        w = w / w.mean()
        new_graphs = []
        for g, wf in zip(graphs, w):
            new_graphs.append(g._replace(nodes={**g.nodes, 'loss_weight': wf.astype(np.float32)}))
        train_dataset.graphs = new_graphs
        q = np.quantile(x, [0.5, 0.9, 0.99])
        print(f"node_weight: mode={mode} alpha={alpha} scale={scale_w} cap={cap}: x quantiles 50/90/99 % = "
              f"{q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}, weights min {w.min():.3f} max {w.max():.3f} "
              f"(share of atoms with w > 2: {(w > 2).mean():.3%})", file=fout)

    # First-order Taylor augmentation (review §11.14; Cooper et al. 2020): for every training frame
    # add n displaced copies r + d (each atom moved by a random vector of length delta) with the
    # energy label E(r) - sum_i F_i . d_i from the existing force labels; the copies carry no force
    # loss (node weight 0), the originals get weight 1 + n so the force normalisation is unchanged.
    # Uses only training labels (fair protocol).  Fixed copies, drawn once with the run seed.
    ta_cfg = _get(cfg, 'taylor_aug', None)
    if ta_cfg:
        n_aug = int(ta_cfg.get('n', 1))
        delta = float(ta_cfg.get('delta', 0.01))
        rng = np.random.RandomState(int(_get(cfg, 'seed', 42)) + 7)
        graphs = train_dataset.graphs
        new_graphs = []
        n_orig = len(graphs)
        for g in graphs:
            nodes = dict(g.nodes)
            n = int(g.n_node[0])
            w_orig = nodes.get('loss_weight', np.ones((n,), np.float32))
            new_graphs.append(g._replace(nodes={**nodes, 'loss_weight': (np.asarray(w_orig, np.float32) * (1 + n_aug))}))
            pos = np.asarray(nodes['positions'])
            frc = np.asarray(nodes['forces'], dtype=np.float64)
            e = np.asarray(g.globals['energy'])
            for _ in range(n_aug):
                d = rng.normal(size=(n, 3))
                d = d / np.linalg.norm(d, axis=1, keepdims=True) * delta                     # |d_i| = delta
                e_new = (e.astype(np.float64) - np.sum(frc * d)).astype(e.dtype)
                new_graphs.append(g._replace(
                    nodes={**nodes, 'positions': (pos.astype(np.float64) + d).astype(pos.dtype),
                           'loss_weight': np.zeros((n,), np.float32)},
                    globals={**g.globals, 'energy': e_new}))
        train_dataset.graphs = new_graphs
        print(f"taylor_aug: n={n_aug} delta={delta} A: {n_orig} frames -> {len(new_graphs)} graphs "
              f"(energy-only copies, force weight on originals {1 + n_aug})", file=fout)

    graphdef, params, optimizer, schedule, grad_clip = build_model(
        cfg, len(atomic_numbers), atom_energies, stats, fout, n_devices=n_devices)
    ema_params = params
    # stress in the loss and the metrics (periodic data with a 'stress' global, eV/A^3, ASE sign and Voigt order)
    use_stress = bool(_get(cfg, 'stress_loss', False))
    if use_stress and not bool(_get(cfg, 'periodic', False)):
        raise ValueError("stress_loss needs periodic=true")

    train_step = make_sharded_train_step(
        optimizer=optimizer, mesh=mesh,
        energy_weight=float(_get(cfg, 'energy_weight', 1.0)),
        force_weight=float(_get(cfg, 'force_weight', 1.0)),
        stress_weight=float(_get(cfg, 'stress_weight', 1.0)) if use_stress else 0.0,
        huber_delta=huber_delta, ema_decay=ema_decay,
        loss_type=loss_type, periodic=use_stress, cov_mode=cov_mode, beta_nll=beta_nll,
        grad_clip=float(grad_clip) if grad_clip else 1.0,
        dynamic_loss_weights=dynamic_weights,
        stochastic_depth=(float(_get(cfg, 'stochastic_depth', 0.0)), int(_get(cfg, 'n_layers', 5))),
        stochastic_depth_mode=_get(cfg, 'stochastic_depth_mode', 'graph'),
        noisy_nodes=((float(nn_cfg['sigma']), float(nn_cfg['weight'])) if nn_cfg else (0.0, 0.0)),
        deep_supervision=float(_get(cfg, 'deep_supervision', 0.0)),
        stratum_decorrelation=float(_get(cfg, 'stratum_decorrelation', 0.0)),
        sam_rho=float(_get(cfg, 'sam_rho', 0.0)),
        rng_seed=int(_get(cfg, 'seed', 42)) + 1000,
        energy_loss_per=_get(cfg, 'energy_loss_per', 'atom'),
        taylor_consistency=(lambda c: (float(c.get('weight', 1.0)), float(c.get('eps', 0.02)), float(c.get('k2', 0.5)))
                            if c else (0.0, 0.0, 0.5))(_get(cfg, 'taylor_consistency', None)),
        radial_l2=float(_get(cfg, 'radial_l2', 0.0)),
        dens=dens_cfg or None,
        charge_weight=float(_get(cfg, 'charge_weight', 0.0)),
    )
    if float(_get(cfg, 'charge_weight', 0.0)) > 0 and not _get(cfg, 'charge_label_key', None):
        raise ValueError("charge_weight needs charge_label_key")
    if float(_get(cfg, 'stochastic_depth', 0.0)) > 0 and _get(cfg, 'residual', 'none') == 'none':
        raise ValueError("stochastic_depth needs residual='identity' or 'identity_cum'")
    eval_step = make_sharded_evaluate_step(
        mesh, huber_delta=huber_delta, loss_type=loss_type, periodic=use_stress,
        cov_mode=cov_mode, beta_nll=beta_nll, energy_loss_per=_get(cfg, 'energy_loss_per', 'atom'))

    init_from = _get(cfg, 'init_params_from', None)
    if init_from and not (_get(cfg, 'restart', False) and Path(ckpt_name).exists()):
        # Stage-two fine-tuning (review §11.12, MACE "SWA stage" analogue): start from the
        # (EMA) parameters of a finished run with a fresh optimiser / schedule and this
        # config's learning rate and loss weights.  The model configuration must match.
        ck = load_checkpoint(init_from)
        src = ck['ema_params'] if _get(cfg, 'init_from_ema', True) else ck['params']
        params = jax.tree.map(jnp.asarray, src)
        ema_params = params
        print(f"Initialised parameters from {init_from} ({'EMA' if _get(cfg, 'init_from_ema', True) else 'raw'} "
              f"params, epoch {ck.get('epoch')}); fresh optimiser and schedule", file=fout)
    opt_state = optimizer.init(params)
    schedule_state = schedule.init(params)
    best_val_loss = math.inf
    start_epoch = 0
    if resuming:
        # the newer of the best-validation and the last checkpoint (same contents; best_val_loss is
        # carried by both)
        cands = [load_checkpoint(p) for p in (last_name, ckpt_name) if Path(p).exists()]
        ckpt = max(cands, key=lambda c: int(c.get('epoch', 0)))
        params = ckpt['params']
        ema_params = ckpt['ema_params']
        opt_state = ckpt['opt_state']
        schedule_state = ckpt['schedule_state']
        best_val_loss = min(c.get('best_val_loss', math.inf) for c in cands)
        start_epoch = int(ckpt.get('epoch', 0))
        del cands
        print(f"Resumed from {'last' if ckpt.get('is_last') else 'best'} checkpoint: epoch={start_epoch} "
              f"step={ckpt.get('step')} best_val_loss={best_val_loss:.6f}", file=fout)
        if _get(cfg, 'swa_start_epoch', None) is not None:
            print("warning: the SWA running sum is not checkpointed; SWA restarts at the resumed epoch", file=fout)

    params = replicate_pytree(params, mesh)
    ema_params = replicate_pytree(ema_params, mesh)
    opt_state = replicate_pytree(opt_state, mesh)
    schedule_state = replicate_pytree(schedule_state, mesh)
    step = replicate(jnp.array(int(ckpt.get('step', 0)) if start_epoch > 0 else 0), mesh)

    loader_rngs = nnx.Rngs(int(_get(cfg, 'seed', 42)) + 100)
    pad_mode = _get(cfg, 'pad_mode', 'bucket')
    if pad_mode == 'fixed':
        # ONE padded shape for every train / eval batch (one compilation each): graphs are packed until a node or
        # edge budget is full; the budgets hold on average batch_size graphs and at least the largest graph of any
        # split.  The number of graphs per step varies (up to pack_max_graphs); the loss is a per-graph / per-atom
        # mean, so this changes the noise of the step, not its scale.
        all_g = [g for ds in [train_dataset, valid_dataset] + list(test_sets.values()) for g in ds.graphs]
        nn_ = np.array([int(np.asarray(g.n_node).reshape(-1)[0]) for g in all_g])
        ne_ = np.array([int(np.asarray(g.n_edge).reshape(-1)[0]) for g in all_g])
        ntr = np.array([int(np.asarray(g.n_node).reshape(-1)[0]) for g in train_dataset.graphs])
        etr = np.array([int(np.asarray(g.n_edge).reshape(-1)[0]) for g in train_dataset.graphs])
        # readable size tables of every split next to the log (index, n_atoms, n_edges), see data_3bpa.write_sizes_tsv
        from bam_omat24.data_3bpa import write_sizes_tsv, graph_sizes
        log_dir = Path(_get(cfg, 'log_file', 'runs/x/log.out')).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        for nm, ds in [('train', train_dataset), ('valid', valid_dataset)] + list(test_sets.items()):
            write_sizes_tsv(log_dir / f'sizes_{nm}.tsv', *graph_sizes(ds.graphs), cutoff=float(_get(cfg, 'cutoff', 5.0)))
        pad_nodes = int(_get(cfg, 'pad_nodes', 0)) or int(math.ceil(max(batch_size * ntr.mean(), nn_.max() + 1) / 64) * 64)
        pad_edges = int(_get(cfg, 'pad_edges', 0)) or int(math.ceil(max(batch_size * etr.mean(), ne_.max()) / 1024) * 1024)
        pack_graphs = int(_get(cfg, 'pack_max_graphs', 4 * batch_size))
        if nn_.max() + 1 > pad_nodes or ne_.max() > pad_edges:
            raise ValueError(f"pad_nodes/pad_edges ({pad_nodes}/{pad_edges}) below the largest graph ({nn_.max()}/{ne_.max()})")
        print(f"pad_mode fixed: {pad_nodes} nodes, {pad_edges} edges, <= {pack_graphs} graphs per batch "
              f"(train mean {ntr.mean():.1f} nodes / {etr.mean():.0f} edges per graph)", file=fout)
        base_loader = BucketedDataLoader(
            train_dataset, batch_size=pack_graphs, n_buckets=8,
            shuffle=True, drop_last=False, rngs=loader_rngs,
            max_nodes_per_batch=pad_nodes, max_edges_per_batch=pad_edges,
        )
    else:
        # the node cap must never shorten a full batch (batch_size graphs per device per step)
        max_n = max(int(np.asarray(g.n_node).reshape(-1)[0]) for g in train_dataset.graphs)
        base_loader = BucketedDataLoader(
            train_dataset, batch_size=batch_size, n_buckets=8,
            shuffle=True, drop_last=True, rngs=loader_rngs,
            max_nodes_per_batch=max(256, batch_size * max_n + 1),
        )
    # one shuffle key per epoch (BucketedDataLoader._reset, plus one in __init__): continue the sequence.
    # Advance the loader's own Rngs (an nnx attribute need not be the object that was passed in).
    for _ in range(start_epoch):
        base_loader.rngs.params()
    # Padding granularity. The default collate pads nodes to multiples of 128
    # and edges to multiples of 4096, which wastes ~50 % of the compute for
    # small molecules; smaller buckets mean a few more JIT shapes but less work.
    node_bucket = int(_get(cfg, 'node_bucket_size', 128))
    edge_bucket = int(_get(cfg, 'edge_bucket_size', 4096))
    if pad_mode == 'fixed':
        collate = functools.partial(fixed_collate_fn, n_node_pad=pad_nodes, n_edge_pad=pad_edges)
        eval_bs, eval_caps = pack_graphs, dict(max_nodes_per_batch=pad_nodes, max_edges_per_batch=pad_edges)
    else:
        collate = functools.partial(default_collate_fn, node_bucket_size=node_bucket,
                                    edge_bucket_size=edge_bucket)
        eval_bs, eval_caps = batch_size, {}
    base_loader.collate_fn = collate
    train_loader = MultiDeviceDataLoader(base_loader, n_devices=n_devices, mesh=mesh,
                                         drop_incomplete=True)

    def _evaluate(dataset, params_rep, weights):
        return evaluate_sharded(
            graphdef=graphdef, params=params_rep, files_pkl=[dataset],
            batch_size=eval_bs, n_devices=n_devices, mesh=mesh,
            energy_weight=weights['energy_weight'], force_weight=weights['force_weight'],
            stress_weight=weights['stress_weight'] if use_stress else 0.0, huber_delta=huber_delta, loss_type=loss_type,
            eval_step=eval_step, params_already_replicated=True, periodic=use_stress,
            cov_mode=cov_mode, beta_nll=beta_nll, collate_fn=collate, **eval_caps,
        )

    def _log(*args, **kwargs):
        print(*args, file=fout, **kwargs)

    _log(f"\nStarting training: epochs {start_epoch+1}..{n_epochs}, "
         f"{len(base_loader)} batches/epoch, loss_type={loss_type}")
    _log("Columns: train losses are the weighted loss terms (eV units as in the loss); "
         "valid errors are total-energy meV and meV/A on the held-out split (EMA weights).")
    tlog = Logger(
        groups={"step": ["epoch", "time"],
                "train": ["loss", "E_loss", "F_loss", "aux", "grad"],
                "valid": ["loss", "E_RMSE", "F_RMSE", "E_MAE", "F_MAE"],
                "lr": ["lr", "wE", "wF"]},
        fout=fout, precision=_get(cfg, 'log_precision', 'simple'), echo=False,
        units={"E_RMSE": "meV", "F_RMSE": "meV/A", "E_MAE": "meV", "F_MAE": "meV/A"},
    )
    tlog.print_head()
    global_step = int(to_host(step))
    t_train_start = time.time()
    swa_start = _get(cfg, 'swa_start_epoch', None)
    swa_start = int(swa_start) if swa_start is not None else None
    swa_sum, swa_n = None, 0
    prev_weights = loss_weights_for_epoch(cfg, start_epoch)
    for epoch in range(start_epoch, n_epochs):
        weights = loss_weights_for_epoch(cfg, epoch)
        if weights != prev_weights:
            # New loss-schedule stage: the validation loss changes scale, so the
            # best-checkpoint tracking and the plateau counter must restart
            # (the LR scale itself is kept).
            tlog.message(f"--- loss schedule stage change at epoch {epoch+1}: {prev_weights} -> {weights}")
            best_val_loss = math.inf
            old_sched = unreplicate_pytree(schedule_state)
            fresh = schedule.init(unreplicate_pytree(params))
            schedule_state = replicate_pytree(fresh._replace(scale=old_sched.scale), mesh)
            prev_weights = weights
        lw = {k: replicate(np.asarray(v, dtype=np.float32), mesh) for k, v in weights.items()}
        dens_w = 0.0
        if dens_cfg:
            dens_w = dens_cfg['weight'] * ((1.0 - epoch / n_epochs) if dens_cfg['decay'] == 'linear' else 1.0)
            lw['dens_weight'] = replicate(np.asarray(dens_w, dtype=np.float32), mesh)

        # Metrics are accumulated on device; converting them with float() every
        # step would force a host sync and stall the asynchronous dispatch
        # (~10 % of the epoch time).
        acc = {k: replicate(np.zeros(()), mesh) for k in
               ('energy', 'force', 'graphs', 'atoms', 'grad', 'applied', 'aux', 'dens_sum', 'dens_count')}
        n_steps = 0
        epoch_start = time.time()
        for batch, _info in train_loader:
            if dynamic_weights:
                params, opt_state, ema_params, step, metrics = train_step(
                    graphdef, params, opt_state, schedule_state, ema_params, step, batch, lw)
            else:
                params, opt_state, ema_params, step, metrics = train_step(
                    graphdef, params, opt_state, schedule_state, ema_params, step, batch)
            n_g = metrics['n_graphs']
            n_a = metrics['n_atoms']
            acc['energy'] = acc['energy'] + metrics['energy_loss'] * n_g
            acc['force'] = acc['force'] + metrics['force_loss'] * 3 * n_a
            acc['graphs'] = acc['graphs'] + n_g
            acc['atoms'] = acc['atoms'] + n_a
            acc['grad'] = acc['grad'] + metrics['grad_norm']
            acc['aux'] = acc['aux'] + metrics['aux_loss']
            acc['applied'] = acc['applied'] + metrics['update_applied']
            acc['dens_sum'] = acc['dens_sum'] + metrics['dens_sum']
            acc['dens_count'] = acc['dens_count'] + metrics['dens_count']
            n_steps += 1
            global_step += 1
        acc = {k: float(to_host(v)) for k, v in acc.items()}
        acc_graphs, acc_atoms, acc_grad, acc_applied = (acc['graphs'], acc['atoms'],
                                                        acc['grad'], acc['applied'])
        epoch_time = time.time() - epoch_start

        e_loss = acc['energy'] / max(acc_graphs, 1)
        f_loss = acc['force'] / max(3 * acc_atoms, 1)
        t_loss = weights['energy_weight'] * e_loss + weights['force_weight'] * f_loss
        lr_now = cfg['learning_rate'] * float(unreplicate_pytree(schedule_state).scale)
        if getattr(build_model, 'cosine_fn', None) is not None:
            lr_now *= float(build_model.cosine_fn(global_step))

        val_metrics = _evaluate(valid_dataset, ema_params, weights)
        val_loss = val_metrics['total_loss']
        tlog.print_row(
            {"epoch": epoch + 1, "time": epoch_time},
            {"loss": t_loss, "E_loss": e_loss, "F_loss": f_loss, "aux": acc['aux'] / max(n_steps, 1),
             "grad": acc_grad / max(n_steps, 1)},
            {"loss": val_loss, "E_RMSE": val_metrics['energy_rmse_total'] * 1e3,
             "F_RMSE": val_metrics['force_rmse'] * 1e3, "E_MAE": val_metrics['energy_mae_total'] * 1e3,
             "F_MAE": val_metrics['force_mae'] * 1e3},
            {"lr": lr_now, "wE": weights['energy_weight'], "wF": weights['force_weight']},
        )
        if dens_cfg and ((epoch + 1) % 10 == 0 or epoch == start_epoch):
            tlog.message(f"  DeNS epoch {epoch+1}: weight {dens_w:.3e} noise L2MAE over the corrupted atoms "
                         f"{acc['dens_sum'] / max(acc['dens_count'], 1.0):.4f} ({acc['dens_count']:.0f} atoms) "
                         f"un-clipped steps {acc_applied / max(n_steps, 1):.3f}")

        unrep_params = unreplicate_pytree(params)
        _, new_schedule = schedule.update(
            updates=unrep_params, state=unreplicate_pytree(schedule_state), value=val_loss)
        schedule_state = replicate_pytree(new_schedule, mesh)

        if swa_start is not None and (epoch + 1) >= swa_start:
            swa_sum = unrep_params if swa_sum is None else jax.tree.map(jnp.add, swa_sum, unrep_params)
            swa_n += 1

        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            best_state = {
                'params': unrep_params,
                'ema_params': unreplicate_pytree(ema_params),
                'opt_state': unreplicate_pytree(opt_state),
                'schedule_state': new_schedule,
                'step': int(to_host(step)),
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'metrics': val_metrics,
                'config': cfg,
            }
            if rank0:
                save_checkpoint(best_state, ckpt_name)
            if _get(cfg, 'log_verbose', False):
                tlog.message(f"  *** new best val loss {best_val_loss:.6f} "
                             f"(improved by {improvement:.6f}) -> {ckpt_name}")
        elif _get(cfg, 'log_verbose', False):
            tlog.message(f"  no improvement for {int(new_schedule.plateau_count)} evaluations "
                         f"(best: {best_val_loss:.6f})")

        if eval_test_every > 0 and (epoch + 1) % eval_test_every == 0:
            for name, ds in test_sets.items():
                tm = _evaluate(ds, ema_params, weights)
                tlog.message(f"  Test[{name}] (EMA, epoch {epoch+1}) | {_fmt_metrics(tm)}")

        last_every = int(_get(cfg, 'ckpt_last_every', 25))
        if last_every > 0 and ((epoch + 1) % last_every == 0 or epoch + 1 == n_epochs):
            last_state = {
                'params': unrep_params, 'ema_params': unreplicate_pytree(ema_params),
                'opt_state': unreplicate_pytree(opt_state), 'schedule_state': new_schedule,
                'step': int(to_host(step)), 'epoch': epoch + 1, 'best_val_loss': best_val_loss,
                'config': cfg, 'is_last': True,
            }
            if rank0:
                save_checkpoint(last_state, last_name)
                if _get(cfg, 'ckpt_last_keep', False):      # debugging: keep one file per save
                    save_checkpoint(last_state, str(Path(ckpt_name).with_suffix(f'.last_e{epoch + 1}.pkl')))
            del last_state

    tlog.rule()
    _log(f"Training finished in {(time.time()-t_train_start)/3600:.2f} h")
    _barrier("final_ckpt")      # the best checkpoint is written by process 0

    # ---- final evaluation with the best checkpoint (EMA weights) ----
    if Path(ckpt_name).exists():
        ckpt = load_checkpoint(ckpt_name)
        best_ema = replicate_pytree(ckpt['ema_params'], mesh)
        weights = loss_weights_for_epoch(cfg, n_epochs - 1)
        _log(f"Final evaluation with best checkpoint (epoch {ckpt.get('epoch')}, "
             f"val loss {ckpt.get('best_val_loss'):.6f}):")
        summary = {}
        for name, ds in list(test_sets.items()) + [('valid', valid_dataset)]:
            tm = _evaluate(ds, best_ema, weights)
            summary[name] = {k: float(v) for k, v in tm.items()}
            _log(f"  {name:>12s} | {_fmt_metrics(tm)}")
        if _get(cfg, 'gauge_refit', False):
            delta = _evaluate(train_dataset, best_ema, weights)['gauge_shift']
            g_ema = gauge_shift_params(ckpt['ema_params'], delta)
            g_rep = replicate_pytree(g_ema, mesh)
            _log(f"Gauge refit on the training split: per-atom shift {delta * 1e3:+.4f} meV "
                 f"(best checkpoint, forces unchanged):")
            summary = {**summary, 'gauge': {'shift_delta_eV_per_atom': float(delta)}}
            for name, ds in list(test_sets.items()) + [('valid', valid_dataset)]:
                tm = _evaluate(ds, g_rep, weights)
                summary['gauge'][name] = {k: float(v) for k, v in tm.items()}
                _log(f"  {name:>12s} | {_fmt_metrics(tm)}")
            if rank0:
                save_checkpoint({**ckpt, 'ema_params': g_ema, 'gauge_shift_delta': float(delta)},
                                str(Path(ckpt_name).with_suffix('.gauge.pkl')))
        if _get(cfg, 'report_final_ema', False):
            _log(f"Final-epoch EMA parameters (epoch {n_epochs}):")
            summary = {**summary, 'final_ema': {}}
            for name, ds in list(test_sets.items()) + [('valid', valid_dataset)]:
                tm = _evaluate(ds, ema_params, weights)
                summary['final_ema'][name] = {k: float(v) for k, v in tm.items()}
                _log(f"  {name:>12s} | {_fmt_metrics(tm)}")
        if swa_sum is not None and swa_n > 0:
            swa_params = jax.tree.map(lambda x: x / swa_n, swa_sum)
            if rank0:
                save_checkpoint({'params': swa_params, 'ema_params': swa_params, 'config': cfg,
                                 'swa_n': swa_n, 'epoch': n_epochs}, str(Path(ckpt_name).with_suffix('.swa.pkl')))
            swa_rep = replicate_pytree(swa_params, mesh)
            _log(f"SWA evaluation (uniform average of {swa_n} epochs from epoch {swa_start}); "
                 f"summary.json = SWA, best-EMA kept under 'best_ema':")
            summary = {'best_ema': summary}
            for name, ds in list(test_sets.items()) + [('valid', valid_dataset)]:
                tm = _evaluate(ds, swa_rep, weights)
                summary[name] = {k: float(v) for k, v in tm.items()}
                _log(f"  {name:>12s} | {_fmt_metrics(tm)}")
        if rank0:
            with open(Path(log_path).with_suffix('.summary.json'), 'w') as f:
                json.dump(summary, f, indent=1)
    fout.close()


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'input/input_3bpa.json'
    with open(cfg_path) as f:
        config = json.load(f)
    train(config)
