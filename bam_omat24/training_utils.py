"""Shared training utilities for sharded training scripts.

Loss functions, device replication helpers, batch shape utilities,
sharded train/eval step factories, evaluation loop, and checkpointing.
"""

from typing import Dict, Tuple, List, Any, Optional

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map
from flax import nnx
import optax
import pickle
import jraph
import numpy as np

from bam_omat24.data_nnx import (Dataset, BucketedDataLoader, MultiDeviceDataLoader, to_host,
                                 global_from_host)
from bam_omat24.utils import node_graph_idx


# =============================================================================
# Device Replication Utilities
# =============================================================================

def replicate(x: Any, mesh: Mesh) -> Any:
    """Replicate array across all devices."""
    if jax.process_count() == 1:
        return jax.device_put(x, NamedSharding(mesh, P()))
    return global_from_host(x, NamedSharding(mesh, P()))


def replicate_pytree(pytree: Any, mesh: Mesh) -> Any:
    """Replicate entire pytree across all devices."""
    return jax.tree.map(lambda x: replicate(x, mesh), pytree)


def unreplicate(x: Any) -> Any:
    """Get value from replicated array."""
    return to_host(x)


def unreplicate_pytree(pytree: Any) -> Any:
    """Unreplicate entire pytree."""
    return jax.tree.map(unreplicate, pytree)


# =============================================================================
# Loss Functions
# =============================================================================

def huber_loss(x: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    """Huber loss."""
    abs_x = jnp.abs(x)
    return jnp.where(abs_x <= delta, 0.5 * x ** 2, delta * (abs_x - 0.5 * delta))


def _build_force_L(frc_var: jnp.ndarray) -> jnp.ndarray:
    """Build 3x3 lower triangular Cholesky L from 6 parameters per atom.

    frc_var: (n_atoms, 6) where [:, :3] are positive diagonal (via exp),
             [:, 3:] are off-diagonal.
    Returns: (n_atoms, 3, 3) lower triangular matrices.

    L = [[σ1,  0,  0 ],
         [σ6, σ2,  0 ],
         [σ5, σ4, σ3]]
    """
    n = frc_var.shape[0]
    L = jnp.zeros((n, 3, 3))
    L = L.at[:, 0, 0].set(frc_var[:, 0])
    L = L.at[:, 1, 1].set(frc_var[:, 1])
    L = L.at[:, 2, 2].set(frc_var[:, 2])
    L = L.at[:, 2, 1].set(frc_var[:, 3])
    L = L.at[:, 2, 0].set(frc_var[:, 4])
    L = L.at[:, 1, 0].set(frc_var[:, 5])
    return L


# Fraction of the isotropic part that the anisotropic (2e) part may reach.
# Guarantees a*I + B' is positive definite: eigenvalues in [a(1-rho), a(1+rho)].
COV_IRREPS_RHO = 0.9


def _traceless_symmetric_from_2e(b: jnp.ndarray) -> jnp.ndarray:
    """Map the 5 components of a 2e irrep to a traceless symmetric 3x3 tensor.

    Uses the e3nn-jax real-spherical-harmonics convention for l=2 (y is the
    polar axis), verified numerically against ``e3nn.spherical_harmonics``:

        Y_2(r) ~ sqrt(15) * [xz, xy, (2y^2 - x^2 - z^2)/(2*sqrt(3)), yz, (z^2 - x^2)/2]

    so that for  T = r r^T - |r|^2/3 * I :
        T_xz = c*b0, T_xy = c*b1, T_yz = c*b3,
        T_xx = c*(-b2/sqrt(3) - b4), T_yy = c*(2 b2/sqrt(3)), T_zz = c*(-b2/sqrt(3) + b4)
    with c = 1/sqrt(15).  The common factor c is dropped (``b`` is a learned
    linear output and is rescaled in ``_build_force_L_irreps`` anyway); the map
    is exactly equivariant: B(D(R) b) = R B(b) R^T  (see tests/test_equivariance.py).
    """
    b0, b1, b2, b3, b4 = b[..., 0], b[..., 1], b[..., 2], b[..., 3], b[..., 4]
    s3 = jnp.sqrt(3.0)
    xx = -b2 / s3 - b4
    yy = 2.0 * b2 / s3
    zz = -b2 / s3 + b4
    xz = b0
    xy = b1
    yz = b3
    return jnp.stack([
        jnp.stack([xx, xy, xz], axis=-1),
        jnp.stack([xy, yy, yz], axis=-1),
        jnp.stack([xz, yz, zz], axis=-1),
    ], axis=-2)


def _build_force_L_irreps(frc_var: jnp.ndarray) -> jnp.ndarray:
    """Build a symmetric positive-definite 3x3 factor L from [a, b_0..b_4].

    Sigma = L @ L.T = (a*I + B')^2 where
      a  = isotropic std (already positive),
      B  = traceless symmetric tensor from the 2e components,
      B' = B scaled smoothly so that ||B'||_F <= rho * a  (rho < 1).
    Hence every eigenvalue of L lies in [a(1-rho), a(1+rho)] > 0.
    Equivariant: under rotation R, L -> R L R^T.
    """
    a = frc_var[:, 0]
    B = _traceless_symmetric_from_2e(frc_var[:, 1:6])
    bnorm = jnp.sqrt(jnp.sum(B ** 2, axis=(-2, -1)) + 1e-12)
    cap = COV_IRREPS_RHO * a
    scale = cap * jnp.tanh(bnorm / cap) / bnorm
    Bp = B * scale[:, None, None]
    return a[:, None, None] * jnp.eye(3)[None] + Bp


def build_force_L(frc_var: jnp.ndarray, cov_mode: str = "scalar6") -> jnp.ndarray:
    """Dispatch on the covariance parameterisation of the model."""
    if cov_mode == "scalar6":
        return _build_force_L(frc_var)
    elif cov_mode == "irreps":
        return _build_force_L_irreps(frc_var)
    raise ValueError(f"unknown cov_mode {cov_mode!r}")


def force_whiten(frc_var: jnp.ndarray, force_diff: jnp.ndarray, cov_mode: str):
    """Return (z, log_det_L) per atom with z = L^{-1} d, Sigma = L L^T."""
    L = build_force_L(frc_var, cov_mode)
    if cov_mode == "scalar6":
        z = jax.vmap(
            lambda Li, d: jax.scipy.linalg.solve_triangular(Li, d, lower=True)
        )(L, force_diff)
        log_det_L = jnp.sum(jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1)
    else:
        z = jnp.linalg.solve(L, force_diff[..., None])[..., 0]
        log_det_L = jnp.linalg.slogdet(L)[1]
    return z, log_det_L


def _build_stress_L(stress_var: jnp.ndarray) -> jnp.ndarray:
    """Build 6x6 lower triangular Cholesky L from 21 parameters per graph.

    stress_var: (n_graphs, 21) where [:, :6] are positive diagonal (via exp),
                [:, 6:] are off-diagonal (row-major lower triangle).
    Returns: (n_graphs, 6, 6) lower triangular matrices.
    """
    n = stress_var.shape[0]
    L = jnp.zeros((n, 6, 6))
    # Diagonal (first 6 values, already positive via exp)
    for i in range(6):
        L = L.at[:, i, i].set(stress_var[:, i])
    # Off-diagonal (next 15 values, row-major lower triangle)
    idx = 6
    for i in range(1, 6):
        for j in range(i):
            L = L.at[:, i, j].set(stress_var[:, idx])
            idx += 1
    return L


def _dipole_sq(results, batch, graph_mask):
    """(sum of squared dipole errors, number of elements) of the labelled graphs (e A).  Open-boundary graphs: the
    three components of results['dipole'].  Periodic graphs (slabs; batch.globals['pbc'] and
    results['dipole_normal']): one element, the component along the slab normal (what a dipole-corrected slab
    calculation labels), as BAM-torch dipole_components "auto"."""
    dt = results['dipole'].dtype
    w = batch.globals['dipole_w'].reshape(-1).astype(dt) * graph_mask.astype(dt)
    ref = batch.globals['dipole'].reshape(-1, 3).astype(dt)
    dd = results['dipole'] - ref
    if 'dipole_normal' in results and 'pbc' in batch.globals:
        per = (batch.globals['pbc'].reshape(-1) > 0.5).astype(dt)
        dn = results['dipole_normal'] - jnp.sum(ref * results['slab_normal'], axis=-1)
        se = jnp.sum(w * (1.0 - per) * jnp.sum(dd ** 2, axis=-1)) + jnp.sum(w * per * dn ** 2)
        return se, jnp.sum(w * (3.0 - 2.0 * per))
    return jnp.sum(w[:, None] * dd ** 2), 3.0 * jnp.sum(w)


def _charge_dipole_sums(results, batch, node_mask, graph_mask):
    """Sums for the charge / dipole metrics of models that return them (RACE_LR, RACE_POLAR); zeros otherwise."""
    z = jnp.zeros(())
    out = {'charge_se': z, 'n_charge': z, 'dipole_se': z, 'n_dipole': z}
    if 'charges' in results and isinstance(batch.nodes, dict) and 'ref_charges' in batch.nodes:
        w = node_mask.astype(results['charges'].dtype)
        if 'ref_charges_w' in batch.nodes:
            w = w * batch.nodes['ref_charges_w'].astype(w.dtype)
        dq = results['charges'] - batch.nodes['ref_charges'].astype(w.dtype)
        out['charge_se'], out['n_charge'] = jnp.sum(w * dq ** 2), jnp.sum(w)
    if 'dipole' in results and isinstance(batch.globals, dict) and 'dipole' in batch.globals:
        out['dipole_se'], out['n_dipole'] = _dipole_sq(results, batch, graph_mask)       # n_dipole = elements
    return out


def _loss_terms(
    results: Dict,
    batch: jraph.GraphsTuple,
    huber_delta: float,
    loss_type: str,
    periodic: bool,
    cov_mode: str,
    beta_nll: float,
    clip_residuals: bool = True,
    energy_loss_per: str = "atom",
):
    """Per-batch loss terms shared by the train and eval steps.

    energy_loss_per: "atom" (original: the energy residual entering the loss is the per-atom
    error E/N) or "structure" (MACE Eq. 15: the total-energy error of the structure).  The
    reported metrics are always per atom / total as before.

    Returns a dict of *sums* (not averages) together with counts, so callers
    can normalise (train) or accumulate across devices/batches (eval).

    loss_type:
      'huber'     : plain Huber on raw residuals.
      'nll'       : (legacy) Huber on whitened residuals + 2*log|L|  (not a
                    proper likelihood).  For cov_mode='irreps' the Huber is
                    applied to the Mahalanobis norm |z| so the loss is
                    rotation invariant.
      'gauss_nll' : exact Gaussian NLL  0.5|z|^2 + log|L|  (+ optional
                    beta-NLL weighting, Seitzer et al. 2022).
    The reported squared/absolute errors are always computed from raw
    (unclipped) residuals.
    """
    graph_mask = jraph.get_graph_padding_mask(batch).astype(jnp.float32)
    node_mask = jraph.get_node_padding_mask(batch).astype(jnp.float32)
    n_graphs = jnp.maximum(graph_mask.sum(), 1)
    n_atoms = jnp.maximum(node_mask.sum(), 1)

    energy = results['energy']
    forces = results['forces']
    n_node = jnp.maximum(batch.n_node, 1)

    # raw residuals (for metrics)
    energy_diff_total_raw = (energy - batch.globals["energy"]) * graph_mask
    energy_diff_raw = energy_diff_total_raw / n_node          # per atom
    if energy_loss_per not in ("atom", "structure"):
        raise ValueError(f"energy_loss_per must be 'atom' or 'structure', got {energy_loss_per!r}")
    energy_diff_loss = energy_diff_total_raw if energy_loss_per == "structure" else energy_diff_raw
    force_diff_raw = (forces - batch.nodes["forces"]) * node_mask[:, None]
    # optional per-graph loss weights (training set only, e.g. Boltzmann tilting);
    # mean 1 over the training set so the n_graphs / n_atoms normalisation is unchanged
    gw = batch.globals.get("loss_weight") if isinstance(batch.globals, dict) else None
    if gw is None:
        gw = jnp.ones_like(graph_mask)
        nw = jnp.ones_like(node_mask)
    else:
        gw = gw.reshape(-1) * graph_mask
        nw = gw[node_graph_idx(batch)] * node_mask
    # optional per-node force-loss weights (training set only, e.g. bond-strain / force-magnitude
    # re-weighting, review §11.12); mean 1 over the training atoms, padded nodes are 0
    nwn = batch.nodes.get("loss_weight") if isinstance(batch.nodes, dict) else None
    if nwn is not None:
        nw = nw * nwn.reshape(-1).astype(nw.dtype) * node_mask

    # (optionally) tanh-clipped residuals for the training loss only.
    # Forces are clipped by their *norm* (direction preserved) so that the
    # clipping is rotation equivariant; a component-wise tanh is not.
    if clip_residuals:
        vmax_E, vmax_F = 5.0, 10.0
        if energy_loss_per == "structure":
            # the 5 eV clip is a per-atom scale; for the total-energy residual scale it by N
            # (otherwise an untrained offset of ~100 eV per molecule saturates the tanh and the
            # energy term gives no gradient: r15_eloss_s42, review §11.6)
            vmax_E = vmax_E * n_node
        energy_diff = vmax_E * jnp.tanh(energy_diff_loss / vmax_E)
        fnorm = jnp.sqrt(jnp.sum(force_diff_raw ** 2, axis=-1, keepdims=True) + 1e-24)
        force_diff = force_diff_raw * (vmax_F * jnp.tanh(fnorm / vmax_F) / fnorm)
    else:
        energy_diff, force_diff = energy_diff_loss, force_diff_raw

    if periodic:
        stress = results['stress']
        stress_diff_raw = (stress - batch.globals["stress"]) * graph_mask[:, None]
        # mixed corpora (stream fields 'mixed'): graphs without a stress label (molecules, OC25) have stress_w = 0
        sw = batch.globals.get("stress_w") if isinstance(batch.globals, dict) else None
        if sw is not None:
            sw = sw.reshape(-1).astype(stress_diff_raw.dtype) * graph_mask
            stress_diff_raw = stress_diff_raw * sw[:, None]
        if clip_residuals:
            vmax_S = 0.5
            stress_diff = vmax_S * jnp.tanh(stress_diff_raw / vmax_S)
        else:
            stress_diff = stress_diff_raw
    else:
        stress_diff_raw = stress_diff = None

    out = {}
    if loss_type in ('nll', 'gauss_nll'):
        frc_var = results['force_variation']          # (n_atoms, 6)
        energy_var = results.get('energy_variation')  # (n_graphs, 1) or None
        stress_var = results.get('stress_variation') if periodic else None
        MIN_VAR = 0.04
        log_min_var = jnp.log(MIN_VAR)

        # ---- Energy ----
        if energy_var is not None:
            energy_sigma = energy_var.squeeze(-1) * graph_mask + (1.0 - graph_mask)
            energy_z = energy_diff / energy_sigma
            if loss_type == 'nll':
                e_fit = jnp.sum(huber_loss(energy_z, delta=huber_delta))
                e_logdet = 2.0 * jnp.sum(jnp.log(energy_sigma) * graph_mask)
                e_const = -2.0 * log_min_var * n_graphs
            else:
                w = jax.lax.stop_gradient(energy_sigma ** 2) ** beta_nll if beta_nll > 0 else 1.0
                e_fit = jnp.sum(w * 0.5 * energy_z ** 2 * graph_mask)
                e_logdet = jnp.sum(w * jnp.log(energy_sigma) * graph_mask)
                e_const = -log_min_var * n_graphs
        else:
            e_fit = jnp.sum(huber_loss(energy_diff, delta=huber_delta))
            e_logdet = 0.0
            e_const = 0.0

        # ---- Forces ----
        frc_z, log_det_L = force_whiten(frc_var, force_diff, cov_mode)   # (n,3), (n,)
        if loss_type == 'nll':
            if cov_mode == 'scalar6':
                f_fit = jnp.sum(huber_loss(frc_z, delta=huber_delta))
            else:  # rotation-invariant: Huber on the Mahalanobis norm
                z_norm = jnp.sqrt(jnp.sum(frc_z ** 2, axis=-1) + 1e-12)
                f_fit = jnp.sum(huber_loss(z_norm, delta=huber_delta) * node_mask)
            f_logdet = 2.0 * jnp.sum(log_det_L * node_mask)
            f_const = -2.0 * log_min_var * (3 * n_atoms)
        else:
            if beta_nll > 0:
                # per-atom geometric-mean variance  (det Sigma)^(1/3) = exp(2/3 log|L|)
                w = jax.lax.stop_gradient(jnp.exp(2.0 / 3.0 * log_det_L)) ** beta_nll
            else:
                w = 1.0
            f_fit = jnp.sum(w * 0.5 * jnp.sum(frc_z ** 2, axis=-1) * node_mask)
            f_logdet = jnp.sum(w * log_det_L * node_mask)
            f_const = -log_min_var * (3 * n_atoms)

        # ---- Stress ----
        if periodic:
            if stress_var is not None:
                stress_sigma = stress_var * graph_mask[:, None] + (1.0 - graph_mask[:, None])
                stress_z = stress_diff / stress_sigma
                if loss_type == 'nll':
                    s_fit = jnp.sum(huber_loss(stress_z, delta=huber_delta))
                    s_logdet = 2.0 * jnp.sum(jnp.log(stress_sigma) * graph_mask[:, None])
                    s_const = -12.0 * log_min_var * n_graphs
                else:
                    s_fit = jnp.sum(0.5 * stress_z ** 2 * graph_mask[:, None])
                    s_logdet = jnp.sum(jnp.log(stress_sigma) * graph_mask[:, None])
                    s_const = -6.0 * log_min_var * n_graphs
            else:
                s_fit = jnp.sum(huber_loss(stress_diff, delta=huber_delta))
                s_logdet = s_const = 0.0
        else:
            s_fit = s_logdet = s_const = 0.0

        out['energy_loss'] = e_fit + e_logdet + e_const
        out['force_loss'] = f_fit + f_logdet + f_const
        out['stress_loss'] = s_fit + s_logdet + s_const
    else:  # 'huber'
        out['energy_loss'] = jnp.sum(huber_loss(energy_diff, delta=huber_delta) * gw)
        out['force_loss'] = jnp.sum(huber_loss(force_diff, delta=huber_delta) * nw[:, None])
        out['stress_loss'] = (jnp.sum(huber_loss(stress_diff, delta=huber_delta))
                              if periodic else 0.0)

    # ---- raw metrics (never clipped) ----
    out.update({
        'energy_se': jnp.sum(energy_diff_raw ** 2),            # per-atom
        'energy_ae': jnp.sum(jnp.abs(energy_diff_raw)),
        'energy_se_total': jnp.sum(energy_diff_total_raw ** 2),  # total energy
        'energy_ae_total': jnp.sum(jnp.abs(energy_diff_total_raw)),
        # least-squares refit of the per-atom energy shift (force-null gauge): sums of N_n d_n, N_n^2
        'energy_dN': jnp.sum(energy_diff_total_raw * batch.n_node * graph_mask),
        'n_node_sq': jnp.sum((batch.n_node * graph_mask) ** 2),
        'force_se': jnp.sum(force_diff_raw ** 2),
        'force_ae': jnp.sum(jnp.abs(force_diff_raw)),
        'stress_se': jnp.sum(stress_diff_raw ** 2) if periodic else 0.0,
        'stress_ae': jnp.sum(jnp.abs(stress_diff_raw)) if periodic else 0.0,
        'n_graphs': n_graphs,
        'n_atoms': n_atoms,
        **_charge_dipole_sums(results, batch, node_mask, graph_mask),
        # graphs with a stress label (= real graphs unless the batch carries globals['stress_w'])
        'n_stress': (jnp.sum(sw) if (periodic and sw is not None) else graph_mask.sum()),
    })
    return out


def compute_loss(
    graphdef: nnx.GraphDef,
    params: nnx.State,
    batch: jraph.GraphsTuple,
    energy_weight: float,
    force_weight: float,
    stress_weight: float,
    huber_delta: float,
    loss_type: str = 'huber',
    periodic: bool = True,
    cov_mode: str = 'scalar6',
    beta_nll: float = 0.0,
    deep_supervision: float = 0.0,
    stratum_decorrelation: float = 0.0,
    noisy_nodes: Tuple[float, float] = (0.0, 0.0),
    energy_loss_per: str = "atom",
    taylor_consistency: Tuple[float, float, float] = (0.0, 0.0, 0.5),
    radial_l2: float = 0.0,
    norm: Optional[Dict] = None,
    dens: Optional[Dict] = None,
    dens_weight: Any = 0.0,
    charge_weight: float = 0.0,
    dipole_weight: float = 0.0,
) -> Tuple[jnp.ndarray, Dict]:
    """Compute training loss for a single batch (no device dimension).

    norm = dict(n_graphs, n_atoms, n_rep[, n_atoms_f, n_dens]) of the *global* batch (data parallel,
    review §13): every additive term is divided by the global counts and the parameter-only /
    batch-statistic terms (radial_l2, stratum_decorrelation) by n_rep, so the psum over the devices
    of the returned loss (and of its gradient) equals the single-device loss of the union batch.  The
    decorrelation term is a batch statistic and stays the device mean.  None = counts of this batch.

    dens = dict(sigma, noisy_ef) (faithful DeNS, docs/RACE_DeNS_audit_2026-10-29.md), batch prepared by
    the train step (nodes 'dens_mask' m, 'dens_noise' m*eps, 'dens_force' or 'dens_force_in' m*F):
      noisy_ef True : one pass on the corrupted positions with force encoding; energy vs the clean
                      label for every graph, forces only on uncorrupted atoms (nodes 'loss_weight'
                      carries 1 - m, normalised by their count n_atoms_f), plus the noise loss
      noisy_ef False: clean energy / force pass unchanged + an extra forward pass on the corrupted,
                      force-encoded structure for the noise loss only
      noise loss    : dens_weight * sum_i m_i |head_i - eps_i/sigma| / sum_i m_i  (L2MAE, as the code)

    Round-20 regularisers (review §11.15):
      taylor_consistency (w, eps_max, k2): + w * mean_graphs [ (E(S+d) - E(S) + sum_i d_i.((1-k2) F_i(S)
                               + k2 F_i(S+d))) / N ]^2  with the model's own E, F (PICPS "PITC",
                               no reference labels); d = batch.nodes['taylor_disp'] (per-atom random
                               direction, |d_i| <= eps_max, 0 on padded nodes).  k2 = 1/2 = exact
                               second order.  For a conservative model this is a smoothness
                               (curvature / third-derivative) penalty along random directions.
      radial_l2 w            : + w * sum_layers ||W_1(radial MLP)||^2; with the radial basis rescaled
                               by 1/gamma(n) (RACE radial_prior) this is the Tikhonov regularity
                               prior gamma(n)^2 |c_n|^2 of Darby et al. (arXiv:2601.15072).

    Extra (all default off, RACE only):
      deep_supervision w     : + w * sum_{t<S-1} Huber(partial-sum energy of strata 0..t)  (per atom)
      stratum_decorrelation w: + w * sum_{s<t} Corr(E_s, E_t)^2 over the real atoms of the batch
                               (ACE purification / restratification as a penalty)
      noisy_nodes (sigma, w) : + w * mean |head(x + eps) - eps/sigma|^2, eps = batch.nodes['noise']
                               (Noisy Nodes denoising auxiliary task; needs model.noise_head)
      dipole_weight w        : + w * mean over labelled graphs and components of (p - p_ref)^2 (e A)^2, p =
                               results['dipole'] (RACE_POLAR), p_ref / mask = batch.globals['dipole'] / ['dipole_w']
      charge_weight w        : + w * sum_i (q_i - q_ref_i)^2 / n_atoms (e^2), q = results['charges'] (RACE_LR),
                               q_ref = batch.nodes['ref_charges'] (cfg charge_label_key); skipped if either is absent

    Args:
        loss_type: 'huber' | 'nll' (legacy Huber-whitened) | 'gauss_nll'.
        periodic: Whether system is periodic. If False, stress loss is skipped.
        cov_mode: force covariance layout of the model ('scalar6' | 'irreps').
        beta_nll: beta of beta-NLL (only for loss_type='gauss_nll'; 0 = plain).
    """
    model = nnx.merge(graphdef, params)
    results = model(batch)
    t = _loss_terms(results, batch, huber_delta, loss_type, periodic, cov_mode, beta_nll,
                    energy_loss_per=energy_loss_per)

    n_graphs, n_atoms = t['n_graphs'], t['n_atoms']
    n_rep = 1
    dens_on = dens is not None and isinstance(batch.nodes, dict) and 'dens_mask' in batch.nodes
    if dens_on:
        node_mask_d = jraph.get_node_padding_mask(batch).astype(batch.nodes['dens_mask'].dtype)
        n_atoms_f = jnp.maximum(jnp.sum(node_mask_d * (1.0 - batch.nodes['dens_mask'])), 1.0)
        n_dens = jnp.maximum(jnp.sum(batch.nodes['dens_mask']), 1.0)
    if norm is not None:
        n_graphs, n_atoms, n_rep = norm['n_graphs'], norm['n_atoms'], norm['n_rep']
        if dens_on:
            n_atoms_f, n_dens = norm['n_atoms_f'], norm['n_dens']
    # forces of the corrupted atoms are not supervised in the single DeNS pass
    n_atoms_force = n_atoms_f if (dens_on and dens['noisy_ef']) else n_atoms
    energy_loss = t['energy_loss'] / n_graphs
    force_loss = t['force_loss'] / (3 * n_atoms_force)
    stress_loss = t['stress_loss'] / n_graphs

    total_loss = (
        energy_weight * energy_loss
        + force_weight * force_loss
        + stress_weight * stress_loss
    )

    ds_loss = jnp.zeros(()); decor_loss = jnp.zeros(()); denoise_loss = jnp.zeros(())
    if deep_supervision > 0.0 or stratum_decorrelation > 0.0:
        strata = results['strata']                                  # (S, n_atoms), padded -> 0
        node_mask = jraph.get_node_padding_mask(batch).astype(strata.dtype)
        graph_mask = jraph.get_graph_padding_mask(batch).astype(strata.dtype)
        gidx = node_graph_idx(batch)
        n_graphs_pad = batch.n_node.shape[0]
    if deep_supervision > 0.0:
        n_node = jnp.maximum(batch.n_node, 1)
        e_strata = jax.vmap(lambda s: jraph.segment_sum(s, gidx, num_segments=n_graphs_pad))(strata)
        e_partial = jnp.cumsum(e_strata, axis=0) + (n_node * model.shift[...])[None, :]   # (S, G)
        target = batch.globals["energy"]
        for tt in range(e_partial.shape[0] - 1):                    # last partial sum = full energy
            d = (e_partial[tt] - target) * graph_mask / n_node
            d = 5.0 * jnp.tanh(d / 5.0)
            ds_loss = ds_loss + jnp.sum(huber_loss(d, delta=huber_delta)) / n_graphs
        total_loss = total_loss + deep_supervision * ds_loss
    if stratum_decorrelation > 0.0:
        m = node_mask
        n_real = jnp.maximum(m.sum(), 1.0)
        mean = jnp.sum(strata * m[None, :], axis=1, keepdims=True) / n_real
        c = (strata - mean) * m[None, :]
        cov = c @ c.T / n_real                                      # (S, S)
        std = jnp.sqrt(jnp.diag(cov) + 1e-12)
        corr = cov / (std[:, None] * std[None, :])
        S = corr.shape[0]
        upper = jnp.triu(jnp.ones((S, S), dtype=corr.dtype), k=1)
        decor_loss = jnp.sum((corr * upper) ** 2) / n_rep
        total_loss = total_loss + stratum_decorrelation * decor_loss
    sigma, w_dn = noisy_nodes
    if w_dn > 0.0 and isinstance(batch.nodes, dict) and 'noise' in batch.nodes:
        noise = batch.nodes['noise']
        noisy = batch._replace(nodes={**batch.nodes, 'positions': batch.nodes['positions'] + noise})
        pred = model.denoise(noisy)                                 # (n_atoms, 3)
        node_mask_dn = jraph.get_node_padding_mask(batch).astype(pred.dtype)
        err = jnp.sum((pred - noise / sigma) ** 2, axis=-1) * node_mask_dn
        n_dn = jnp.maximum(node_mask_dn.sum(), 1.0) if norm is None else n_atoms
        denoise_loss = jnp.sum(err) / (3.0 * n_dn)
        total_loss = total_loss + w_dn * denoise_loss

    dens_loss = jnp.zeros(()); dens_sum = jnp.zeros(()); dens_count = jnp.zeros(())
    if dens_on:
        m = batch.nodes['dens_mask']
        if dens['noisy_ef']:
            feats = results['node_features']
        else:
            noisy = batch._replace(nodes={**batch.nodes, 'positions': batch.nodes['positions'] + batch.nodes['dens_noise'],
                                          'dens_force': batch.nodes['dens_force_in']})
            feats = model.node_energies(noisy.nodes['positions'], None, None, noisy)['node_features']
        pred = model.dens_noise(feats)
        err = jnp.sqrt(jnp.sum((pred - batch.nodes['dens_noise'] / dens['sigma']) ** 2, axis=-1) + 1e-24) * m
        dens_sum, dens_count = jnp.sum(err), jnp.sum(m)          # raw (log: L2MAE = sum / count)
        dens_loss = dens_sum / n_dens
        total_loss = total_loss + dens_weight * dens_loss

    tc_loss = jnp.zeros(()); rl2_loss = jnp.zeros(())
    w_tc, eps_tc, k2 = taylor_consistency
    if w_tc > 0.0 and isinstance(batch.nodes, dict) and 'taylor_disp' in batch.nodes:
        disp = batch.nodes['taylor_disp']                                   # (n_atoms, 3), 0 on padding
        displaced = batch._replace(nodes={**batch.nodes, 'positions': batch.nodes['positions'] + disp})
        res2 = model(displaced)
        gidx_tc = node_graph_idx(batch)
        n_graphs_pad_tc = batch.n_node.shape[0]
        gmask_tc = jraph.get_graph_padding_mask(batch).astype(disp.dtype)
        f_mix = (1.0 - k2) * results['forces'] + k2 * res2['forces']
        work = jraph.segment_sum(jnp.sum(disp * f_mix, axis=-1), gidx_tc, num_segments=n_graphs_pad_tc)
        e_taylor = results['energy'].reshape(-1) - work                      # E(S) - sum d.F
        n_node_tc = jnp.maximum(batch.n_node, 1).astype(disp.dtype)
        d_tc = (res2['energy'].reshape(-1) - e_taylor) * gmask_tc / n_node_tc
        tc_loss = jnp.sum(d_tc ** 2) / n_graphs
        total_loss = total_loss + w_tc * tc_loss
    if radial_l2 > 0.0:
        rl2_loss = sum(jnp.sum(layer.radial_mlp.layers[0].weights[...] ** 2) for layer in model.layers) / n_rep
        total_loss = total_loss + radial_l2 * rl2_loss
    q_loss = jnp.zeros(())
    if charge_weight > 0.0 and 'charges' in results and isinstance(batch.nodes, dict) and 'ref_charges' in batch.nodes:
        qm = jraph.get_node_padding_mask(batch).astype(results['charges'].dtype)
        dq = results['charges'] - batch.nodes['ref_charges'].astype(results['charges'].dtype)
        if 'ref_charges_w' in batch.nodes:      # mixed corpora: atoms without a label have weight 0
            qm = qm * batch.nodes['ref_charges_w'].astype(qm.dtype)
            q_loss = jnp.sum(qm * dq ** 2) / jnp.maximum(jnp.sum(qm), 1.0)
        else:
            q_loss = jnp.sum(qm * dq ** 2) / n_atoms
        total_loss = total_loss + charge_weight * q_loss
    d_loss = jnp.zeros(())
    if dipole_weight > 0.0 and 'dipole' in results and isinstance(batch.globals, dict) and 'dipole' in batch.globals:
        # graph dipoles (RACE_POLAR; e A): mean squared error over the labelled elements (_dipole_sq)
        d_se, d_n = _dipole_sq(results, batch, jraph.get_graph_padding_mask(batch))
        d_loss = d_se / jnp.maximum(d_n, 1.0)
        total_loss = total_loss + dipole_weight * d_loss

    aux = {
        'energy_loss': energy_loss,
        'force_loss': force_loss,
        'stress_loss': stress_loss,
        'aux_loss': (deep_supervision * ds_loss + stratum_decorrelation * decor_loss + w_dn * denoise_loss
                     + dens_weight * dens_loss
                     + w_tc * tc_loss + radial_l2 * rl2_loss + charge_weight * q_loss
                     + dipole_weight * d_loss),
        'ds_loss': ds_loss, 'decor_loss': decor_loss, 'denoise_loss': denoise_loss, 'dens_loss': dens_loss,
        'dens_sum': dens_sum, 'dens_count': dens_count,
        'tc_loss': tc_loss, 'rl2_loss': rl2_loss, 'q_loss': q_loss, 'd_loss': d_loss,
        'energy_mse': t['energy_se'] / n_graphs,
        'force_mse': t['force_se'] / (3 * n_atoms),       # all atoms (DeNS: incl. the corrupted ones)
        'stress_mse': t['stress_se'] / n_graphs if periodic else 0.0,
        'n_graphs': n_graphs,
        'n_atoms': n_atoms_force,        # = n_atoms unless the DeNS pass drops the corrupted atoms
    }
    return total_loss, aux


# =============================================================================
# Batch Shape Utilities
# =============================================================================

def squeeze_batch(batch: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Squeeze the leading device dimension from a sharded batch.

    When using shard_map with P('dp') on the batch, each device receives
    a slice with shape (1, ...). This function removes that leading dimension.

    (1, n_nodes, ...) -> (n_nodes, ...)
    (1, n_edges, ...) -> (n_edges, ...)
    (1, n_graphs, ...) -> (n_graphs, ...)
    """
    def squeeze_array(x):
        if x is None:
            return None
        if hasattr(x, 'shape') and len(x.shape) > 0 and x.shape[0] == 1:
            return jnp.squeeze(x, axis=0)
        return x

    def squeeze_dict(d):
        if d is None:
            return None
        # For globals, ensure proper squeezing of all arrays
        return {k: jnp.squeeze(v) if hasattr(v, 'shape') and 1 in v.shape[:2] else v
                for k, v in d.items()}

    return jraph.GraphsTuple(
        n_node=squeeze_array(batch.n_node),
        n_edge=squeeze_array(batch.n_edge),
        nodes=squeeze_dict(batch.nodes) if isinstance(batch.nodes, dict) else squeeze_array(batch.nodes),
        edges=squeeze_dict(batch.edges) if isinstance(batch.edges, dict) else squeeze_array(batch.edges),
        globals=squeeze_dict(batch.globals),
        senders=squeeze_array(batch.senders),
        receivers=squeeze_array(batch.receivers),
    )


# =============================================================================
# Sharded Training Step
# =============================================================================

def make_sharded_train_step(
    optimizer: optax.GradientTransformation,
    mesh: Mesh,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    stress_weight: float = 1.0,
    huber_delta: float = 0.1,
    ema_decay: float = 0.99,
    freeze_mask: Any = None,
    loss_type: str = 'huber',
    periodic: bool = True,
    cov_mode: str = 'scalar6',
    beta_nll: float = 0.0,
    grad_clip: float = 1.0,
    dynamic_loss_weights: bool = False,
    stochastic_depth: Tuple[float, int] = (0.0, 0),
    stochastic_depth_mode: str = "graph",
    noisy_nodes: Tuple[float, float] = (0.0, 0.0),
    deep_supervision: float = 0.0,
    stratum_decorrelation: float = 0.0,
    sam_rho: float = 0.0,
    rng_seed: int = 0,
    energy_loss_per: str = "atom",
    taylor_consistency: Tuple[float, float, float] = (0.0, 0.0, 0.5),
    radial_l2: float = 0.0,
    dens: Optional[Dict] = None,
    charge_weight: float = 0.0,
    dipole_weight: float = 0.0,
):
    """Create sharded training step with gradient synchronization.

    dens = dict(p, r, sigma, noisy_ef, weight): faithful DeNS (docs/RACE_DeNS_audit_2026-10-29.md):
      per step, each graph is picked with probability p and each atom of a picked graph is
      corrupted with probability r (mask m); eps ~ N(0, sigma^2) on the corrupted atoms; reference
      forces m*F feed the model's force encoding; see compute_loss for the two modes.  The noise-loss
      weight comes from loss_weights['dens_weight'] (dynamic, e.g. linear decay) or dens['weight'].

    Regularisation options (all default off):
      stochastic_depth (p_max, n_layers): per graph and layer t, drop stratum t with
          probability p_t = p_max * t/(n_layers-1) (layer 0 never); kept strata are scaled
          by 1/(1-p_t) (inverted dropout).  Passed to the model as globals['stratum_keep'].
      noisy_nodes (sigma, w): Gaussian position noise eps ~ N(0, sigma^2) added as
          nodes['noise']; see compute_loss.
      deep_supervision / stratum_decorrelation: see compute_loss.
      sam_rho: sharpness-aware minimisation (Foret et al. 2021): the update uses the
          gradient at params + rho * g/|g| (two forward/backward passes).

    Uses shard_map to explicitly control SPMD execution:
    - Each device processes its own batch (indexed by leading dimension)
    - Each device's loss is normalised by the graph / atom counts of the global batch and the
      gradients are summed (psum) across the 'dp' axis, so the update equals the single-device
      update on the union of the device batches (compute_loss ``norm``); a mean of per-device
      means (pmean) would weight the atoms of small device batches more
    - Parameters are updated identically on all devices

    Args:
        freeze_mask: Optional pytree of bools matching params structure.
            If provided, EMA is only applied to trainable (False) leaves.
            Frozen(True) leaves keep their original values in the EMA state.
        loss_type: 'huber' | 'nll' | 'gauss_nll'.
        grad_clip: only used for the 'update_applied' diagnostic; the actual
            clipping lives in the optimizer chain built by the caller.
        dynamic_loss_weights: if True the returned step takes one extra
            trailing argument ``loss_weights`` = dict(energy_weight=..,
            force_weight=.., stress_weight=..) of scalars, so weights can be
            changed between epochs without re-JIT (used by ``loss_schedule``).
    """

    n_replicas = int(mesh.shape['dp'])

    def per_device_step(graphdef, params, opt_state, schedule_state, ema_params, step, batch,
                        loss_weights=None):
        """Training step for a single device's batch."""
        # Squeeze the leading device dimension: (1, ...) -> (...)
        batch = squeeze_batch(batch)
        if loss_weights is None:
            ew, fw, sw = energy_weight, force_weight, stress_weight
        else:
            ew, fw, sw = (loss_weights['energy_weight'],
                          loss_weights['force_weight'],
                          loss_weights['stress_weight'])
        dw = 0.0
        if dens is not None:
            dw = (loss_weights['dens_weight'] if (loss_weights is not None and 'dens_weight' in loss_weights)
                  else dens['weight'])
            # corruption: graphs with probability p, atoms of a picked graph with probability r
            kd = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(rng_seed + 7777), step),
                                    jax.lax.axis_index('dp'))
            kg, ka, ke = jax.random.split(kd, 3)
            pos = batch.nodes['positions']
            pick = (jax.random.uniform(kg, (batch.n_node.shape[0],)) < dens['p']) & jraph.get_graph_padding_mask(batch)
            m = (pick[node_graph_idx(batch)] & (jax.random.uniform(ka, (pos.shape[0],)) < dens['r'])
                 & jraph.get_node_padding_mask(batch)).astype(pos.dtype)
            eps = dens['sigma'] * jax.random.normal(ke, pos.shape, dtype=pos.dtype) * m[:, None]
            f_in = batch.nodes['forces'] * m[:, None]
            nodes = {**batch.nodes, 'dens_mask': m, 'dens_noise': eps}
            if dens['noisy_ef']:
                lw0 = batch.nodes.get('loss_weight')
                keep = 1.0 - m if lw0 is None else lw0 * (1.0 - m)
                nodes.update(positions=pos + eps, dens_force=f_in, loss_weight=keep)
            else:
                nodes['dens_force_in'] = f_in
            batch = batch._replace(nodes=nodes)

        p_max, n_layers_sd = stochastic_depth
        sigma_nn, w_nn = noisy_nodes
        w_tc, eps_tc, _k2 = taylor_consistency
        if p_max > 0.0 or w_nn > 0.0 or w_tc > 0.0:
            key = jax.random.fold_in(jax.random.PRNGKey(rng_seed), step)
            key = jax.random.fold_in(key, jax.lax.axis_index('dp'))
            k_sd, k_nn, k_tc1, k_tc2 = jax.random.split(key, 4)
        if w_tc > 0.0:
            # Taylor-consistency displacement: random direction per atom, |d| ~ U(0, eps_max],
            # re-sampled every step, zero on padded nodes
            pos = batch.nodes['positions']
            dirs = jax.random.normal(k_tc1, pos.shape, dtype=pos.dtype)
            dirs = dirs / jnp.maximum(jnp.linalg.norm(dirs, axis=-1, keepdims=True), 1e-12)
            mag = eps_tc * jax.random.uniform(k_tc2, (pos.shape[0], 1), dtype=pos.dtype)
            nmask_tc = jraph.get_node_padding_mask(batch).astype(pos.dtype)[:, None]
            batch = batch._replace(nodes={**batch.nodes, 'taylor_disp': dirs * mag * nmask_tc})
        if p_max > 0.0:
            p_t = p_max * jnp.arange(n_layers_sd) / max(n_layers_sd - 1, 1)
            if stochastic_depth_mode == "node":
                # per-atom masks: the energy noise averages over the atoms of a molecule
                n_rows = batch.nodes['positions'].shape[0]
            else:
                n_rows = batch.n_node.shape[0]
            u = jax.random.uniform(k_sd, (n_rows, n_layers_sd))
            keep = (u >= p_t[None, :]).astype(jnp.float32) / (1.0 - p_t)[None, :]
            if stochastic_depth_mode == "node":
                batch = batch._replace(nodes={**batch.nodes, 'stratum_keep': keep})
            else:
                batch = batch._replace(globals={**batch.globals, 'stratum_keep': keep})
        if w_nn > 0.0:
            noise = sigma_nn * jax.random.normal(k_nn, batch.nodes['positions'].shape,
                                                 dtype=batch.nodes['positions'].dtype)
            batch = batch._replace(nodes={**batch.nodes, 'noise': noise})

        # counts of the global batch (labels only, independent of the parameters)
        nmask = jraph.get_node_padding_mask(batch).astype(jnp.float32)
        norm = {'n_graphs': jax.lax.psum(jraph.get_graph_padding_mask(batch).astype(jnp.float32).sum(), 'dp'),
                'n_atoms': jax.lax.psum(nmask.sum(), 'dp'), 'n_rep': n_replicas}
        if dens is not None:
            md = batch.nodes['dens_mask'].astype(jnp.float32)
            norm['n_atoms_f'] = jnp.maximum(jax.lax.psum(jnp.sum(nmask * (1.0 - md)), 'dp'), 1.0)
            norm['n_dens'] = jnp.maximum(jax.lax.psum(jnp.sum(md), 'dp'), 1.0)

        def loss_fn(params):
            return compute_loss(
                graphdef, params, batch,
                ew, fw, sw, huber_delta,
                loss_type=loss_type, periodic=periodic,
                cov_mode=cov_mode, beta_nll=beta_nll,
                deep_supervision=deep_supervision,
                stratum_decorrelation=stratum_decorrelation,
                noisy_nodes=noisy_nodes,
                energy_loss_per=energy_loss_per,
                taylor_consistency=taylor_consistency,
                radial_l2=radial_l2,
                norm=norm,
                dens=None if dens is None else {'sigma': dens['sigma'], 'noisy_ef': dens['noisy_ef']},
                dens_weight=dw,
                charge_weight=charge_weight,
                dipole_weight=dipole_weight,
            )

        # Compute gradients locally
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # Sum the globally normalised per-device losses / gradients; the counts are already global
        grads = jax.lax.psum(grads, axis_name='dp')
        loss = jax.lax.psum(loss, axis_name='dp')
        aux = {k: (v if k in ('n_graphs', 'n_atoms') else jax.lax.psum(v, axis_name='dp'))
               for k, v in aux.items()}

        if sam_rho > 0.0:
            # SAM: gradient at the adversarially perturbed point params + rho * g / |g|
            g_norm = optax.global_norm(grads)
            eps = jax.tree.map(lambda g: g * (sam_rho / (g_norm + 1e-12)), grads)
            params_adv = optax.apply_updates(params, eps)
            (_, _), grads_adv = jax.value_and_grad(loss_fn, has_aux=True)(params_adv)
            grads = jax.lax.psum(grads_adv, axis_name='dp')

        grad_norm = optax.global_norm(grads)

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        updates = optax.tree_utils.tree_scale(schedule_state.scale, updates)
        new_params = optax.apply_updates(params, updates)
        if freeze_mask is not None:
            # Only apply EMA to trainable params; frozen params keep current values
            new_ema_params = jax.tree.map(
                lambda ema, p, m: jnp.where(
                    m, ema, ema_decay * ema + (1 - ema_decay) * p),
                ema_params, new_params, freeze_mask
            )
        else:
            new_ema_params = jax.tree.map(
                lambda ema, p: ema_decay * ema + (1 - ema_decay) * p,
                ema_params, new_params
            )

        metrics = {
            'loss': loss,
            'grad_norm': grad_norm,
            'update_applied': (grad_norm < grad_clip).astype(jnp.float32),  # fraction of un-clipped steps
            **aux
        }

        return new_params, new_opt_state, new_ema_params, step + 1, metrics

    in_specs = [
        P(),       # graphdef - replicated
        P(),       # params - replicated
        P(),       # opt_state - replicated
        P(),       # schedule_state - replicated
        P(),       # ema_params - replicated
        P(),       # step - replicated
        P('dp'),   # batch - sharded along device axis
    ]
    if dynamic_loss_weights:
        in_specs.append(P())   # loss_weights - replicated
        step_fn = per_device_step
    else:
        def step_fn(graphdef, params, opt_state, schedule_state, ema_params, step, batch):
            return per_device_step(graphdef, params, opt_state, schedule_state,
                                   ema_params, step, batch)

    # Create sharded version using shard_map
    sharded_step = shard_map(
        step_fn,
        mesh=mesh,
        check_vma=False,   # custom primitives (cuEquivariance) lack VMA-typed transpose rules
        in_specs=tuple(in_specs),
        out_specs=(
            P(),       # new_params
            P(),       # new_opt_state
            P(),       # new_ema_params
            P(),       # step
            P(),       # metrics
        ),
    )

    return jax.jit(sharded_step)


# =============================================================================
# Sharded Evaluation
# =============================================================================

def make_sharded_evaluate_step(mesh: Mesh, huber_delta: float, loss_type: str = 'huber',
                               periodic: bool = True, cov_mode: str = 'scalar6',
                               beta_nll: float = 0.0, energy_loss_per: str = "atom"):
    """Create sharded evaluation function that runs on all GPUs.

    Each device evaluates its own batch, then results are summed across devices.
    Loss terms use the same (tanh-clipped) residuals as training; the reported
    squared / absolute errors use raw residuals.  ``valid`` (n_devices,) weights the devices: the
    last group of an epoch is filled up with copies of its last batch (multi_device_iterator),
    which must not be counted (weight 0).  Called without ``valid`` every device counts.
    """

    def eval_single_device(graphdef, params, batch):
        """Evaluate a single batch on one device. Returns raw sums."""
        batch = squeeze_batch(batch)
        model = nnx.merge(graphdef, params)
        results = model(batch)
        return _loss_terms(results, batch, huber_delta, loss_type, periodic, cov_mode, beta_nll,
                           energy_loss_per=energy_loss_per)

    def eval_and_reduce(graphdef, params, batch, valid):
        """Evaluate on each device and sum results across devices."""
        local_metrics = eval_single_device(graphdef, params, batch)
        w = valid.reshape(())
        # Sum across all devices using psum
        return jax.tree.map(lambda x: jax.lax.psum(jnp.asarray(x) * w.astype(jnp.asarray(x).dtype),
                                                   axis_name='dp'), local_metrics)

    # Create sharded version
    sharded_eval = shard_map(
        eval_and_reduce,
        mesh=mesh,
        check_vma=False,   # custom primitives (cuEquivariance) lack VMA-typed transpose rules
        in_specs=(
            P(),       # graphdef - replicated
            P(),       # params - replicated
            P('dp'),   # batch - sharded along device axis
            P('dp'),   # valid - one weight per device
        ),
        out_specs=P(),  # metrics - reduced (same on all devices after psum)
    )
    sharded_eval = jax.jit(sharded_eval)
    n_dev = int(mesh.shape['dp'])

    def eval_step(graphdef, params, batch, valid=None):
        if valid is None:
            valid = device_weights(n_dev, n_dev, mesh)
        return sharded_eval(graphdef, params, batch, valid)

    return eval_step


def device_weights(n_valid: int, n_devices: int, mesh: Mesh) -> jax.Array:
    """(n_devices,) array sharded over 'dp': 1 for the first n_valid devices, 0 for padding copies."""
    w = (np.arange(n_devices) < n_valid).astype(np.float32)
    sharding = NamedSharding(mesh, P('dp'))
    return jax.device_put(w, sharding) if jax.process_count() == 1 else global_from_host(w, sharding)


def evaluate_sharded(
    graphdef: nnx.GraphDef,
    params: nnx.State,
    files_pkl: List[str],
    batch_size: int,
    n_devices: int,
    mesh: Mesh,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    stress_weight: float = 1.0,
    huber_delta: float = 0.02,
    loss_type: str = 'huber',
    eval_step=None,
    params_already_replicated: bool = False,
    periodic: bool = True,
    cov_mode: str = 'scalar6',
    beta_nll: float = 0.0,
    collate_fn=None,
    max_nodes_per_batch=None,
    max_edges_per_batch=None,
) -> Dict:
    """Evaluate model on validation set using all GPUs.

    Args:
        graphdef: Model graph definition
        params: Model parameters (unreplicated, or replicated if
                params_already_replicated=True)
        files_pkl: List of validation data files
        batch_size: Batch size per device
        n_devices: Number of devices
        mesh: Device mesh
        energy_weight: Weight for energy in loss
        force_weight: Weight for forces in loss
        stress_weight: Weight for stress in loss
        huber_delta: Huber loss delta
        loss_type: 'huber' or 'nll'
        eval_step: Pre-built sharded eval function (avoids re-JIT).
                   If None, one is created internally.
        params_already_replicated: If True, skip replicate_pytree on params.
        periodic: Whether system is periodic. If False, stress loss is skipped.

    Returns:
        Dictionary with evaluation metrics
    """
    if params_already_replicated:
        params_replicated = params
    else:
        params_replicated = replicate_pytree(params, mesh)

    # Initialize accumulators
    totals = {
        'energy_loss': 0.0,
        'force_loss': 0.0,
        'stress_loss': 0.0,
        'energy_se': 0.0,
        'force_se': 0.0,
        'stress_se': 0.0,
        'energy_ae': 0.0,
        'force_ae': 0.0,
        'stress_ae': 0.0,
        'energy_se_total': 0.0,
        'energy_ae_total': 0.0,
        'energy_dN': 0.0,
        'n_node_sq': 0.0,
        'n_graphs': 0.0,
        'n_atoms': 0.0,
        'n_stress': 0.0,
        'charge_se': 0.0, 'n_charge': 0.0, 'dipole_se': 0.0, 'n_dipole': 0.0,
    }

    if eval_step is None:
        eval_step = make_sharded_evaluate_step(mesh, huber_delta=huber_delta, loss_type=loss_type,
                                               periodic=periodic, cov_mode=cov_mode, beta_nll=beta_nll)

    for fname in files_pkl:
        if hasattr(fname, "iter_groups"):
            # streaming evaluation split (stream_data.StreamEvalSet): sharded batches with per-device weights
            for batch, w in fname.iter_groups(mesh):
                metrics = eval_step(graphdef, params_replicated, batch, w)
                for k in totals:
                    totals[k] += float(to_host(metrics[k]))
            continue
        # accept either a pickle path or an in-memory dataset object (with .graphs)
        dataset = fname if hasattr(fname, "graphs") else Dataset(file_path=fname)
        # same node cap rule as the training loader: a full batch of the largest graph must fit
        # (MD22 double-walled nanotube has 370 atoms > the old fixed cap of 256)
        max_n = max(int(np.asarray(g.n_node).reshape(-1)[0]) for g in dataset.graphs)
        base_loader = BucketedDataLoader (
            dataset=dataset,
            batch_size=batch_size,
            n_buckets=8,
            shuffle=False,       # No shuffle for deterministic evaluation
            drop_last=False,     # Evaluate ALL data
            rngs=nnx.Rngs(0),   # Deterministic rngs (not used when shuffle=False)
            max_nodes_per_batch=(max_nodes_per_batch if max_nodes_per_batch is not None
                                 else max(256, batch_size * max_n + 1)),
            max_edges_per_batch=max_edges_per_batch,
        )
        if collate_fn is not None:
            base_loader.collate_fn = collate_fn
        # Use multi-device loader for evaluation
        loader = MultiDeviceDataLoader(
            base_loader=base_loader,
            n_devices=n_devices,
            mesh=mesh,
            drop_incomplete=False  # Don't drop last batch during evaluation
        )

        for batch, info in loader:
            # Run sharded evaluation (padding copies of the last group get weight 0)
            metrics = eval_step(graphdef, params_replicated, batch,
                                device_weights(info.valid_device_count, n_devices, mesh))

            # Accumulate results (metrics are already summed across devices)
            for k in totals:
                totals[k] += float(to_host(metrics[k]))

    # Compute final metrics
    n_g = max(totals['n_graphs'], 1)
    n_a = max(totals['n_atoms'], 1)
    n_s = max(totals['n_stress'], 1)          # graphs with a stress label (= n_g without globals['stress_w'])
    total_loss = (energy_weight * totals['energy_loss'] / n_g +
                  force_weight * totals['force_loss'] / (3* n_a) +
                  stress_weight * totals['stress_loss'] / n_g )
    return {
        # per-atom energy errors (eV/atom) -- original convention
        'energy_mae': totals['energy_ae'] / n_g,
        'energy_rmse': np.sqrt(totals['energy_se']/n_g),
        # total energy errors (eV) -- 3BPA / MACE convention
        'energy_mae_total': totals['energy_ae_total'] / n_g,
        'energy_rmse_total': np.sqrt(totals['energy_se_total']/n_g),
        'force_mae': totals['force_ae'] / (3 * n_a),
        'stress_mae': totals['stress_ae'] / n_s,                    # sum of the 6 Voigt |errors| per graph (legacy)
        'stress_mae_comp': totals['stress_ae'] / (6 * n_s),         # per Voigt component, eV/A^3
        'stress_rmse_comp': np.sqrt(totals['stress_se'] / (6 * n_s)),
        'force_rmse': np.sqrt(totals['force_se']/(3 * n_a)),
        'energy_loss': totals['energy_loss'] / n_g,
        'force_loss': totals['force_loss'] / (3 * n_a),
        'stress_loss': totals['stress_loss'] / n_g,
        'total_loss': total_loss,
        'n_graphs': n_g,
        'n_atoms': n_a,
        # charges (e) and graph dipoles (e A, per component) of the labelled atoms / graphs; 0 without labels
        'charge_rmse': np.sqrt(totals['charge_se'] / max(totals['n_charge'], 1)),
        'dipole_rmse': np.sqrt(totals['dipole_se'] / max(totals['n_dipole'], 1)),      # per labelled element
        'n_charge': totals['n_charge'], 'n_dipole': totals['n_dipole'],
        # per-atom shift that minimises sum_n (d_n + N_n delta)^2 (VarPro / closed-form gauge refit;
        # apply with gauge_shift_params, only from a *training*-set evaluation)
        'gauge_shift': -totals['energy_dN'] / max(totals['n_node_sq'], 1.0),
    }


def gauge_shift_params(params, delta: float):
    """Add delta to the model's per-atom energy shift (top-level Param 'shift')."""
    def f(path, x):
        return x + delta if (len(path) and getattr(path[0], 'key', None) == 'shift') else x
    return jax.tree_util.tree_map_with_path(f, params)


# =============================================================================
# Checkpoint Management
# =============================================================================

def save_checkpoint(state_dict: Dict, path: str):
    """Save checkpoint."""
    with open(path, 'wb') as f:
        pickle.dump(state_dict, f)


def load_checkpoint(path: str) -> Dict:
    """Load checkpoint."""
    with open(path, 'rb') as f:
        return pickle.load(f)
