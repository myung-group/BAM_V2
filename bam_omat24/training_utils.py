"""Shared training utilities for sharded training scripts.

Loss functions, device replication helpers, batch shape utilities,
sharded train/eval step factories, evaluation loop, and checkpointing.
"""

from typing import Dict, Tuple, List, Any

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

from bam_omat24.data_nnx import Dataset, BucketedDataLoader, MultiDeviceDataLoader


# =============================================================================
# Device Replication Utilities
# =============================================================================

def replicate(x: Any, mesh: Mesh) -> Any:
    """Replicate array across all devices."""
    return jax.device_put(x, NamedSharding(mesh, P()))


def replicate_pytree(pytree: Any, mesh: Mesh) -> Any:
    """Replicate entire pytree across all devices."""
    return jax.tree.map(lambda x: replicate(x, mesh), pytree)


def unreplicate(x: Any) -> Any:
    """Get value from replicated array."""
    return np.asarray(x)


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


def _loss_terms(
    results: Dict,
    batch: jraph.GraphsTuple,
    huber_delta: float,
    loss_type: str,
    periodic: bool,
    cov_mode: str,
    beta_nll: float,
    clip_residuals: bool = True,
):
    """Per-batch loss terms shared by the train and eval steps.

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
    force_diff_raw = (forces - batch.nodes["forces"]) * node_mask[:, None]

    # (optionally) tanh-clipped residuals for the training loss only.
    # Forces are clipped by their *norm* (direction preserved) so that the
    # clipping is rotation equivariant; a component-wise tanh is not.
    if clip_residuals:
        vmax_E, vmax_F = 5.0, 10.0
        energy_diff = vmax_E * jnp.tanh(energy_diff_raw / vmax_E)
        fnorm = jnp.sqrt(jnp.sum(force_diff_raw ** 2, axis=-1, keepdims=True) + 1e-24)
        force_diff = force_diff_raw * (vmax_F * jnp.tanh(fnorm / vmax_F) / fnorm)
    else:
        energy_diff, force_diff = energy_diff_raw, force_diff_raw

    if periodic:
        stress = results['stress']
        stress_diff_raw = (stress - batch.globals["stress"]) * graph_mask[:, None]
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
        out['energy_loss'] = jnp.sum(huber_loss(energy_diff, delta=huber_delta))
        out['force_loss'] = jnp.sum(huber_loss(force_diff, delta=huber_delta))
        out['stress_loss'] = (jnp.sum(huber_loss(stress_diff, delta=huber_delta))
                              if periodic else 0.0)

    # ---- raw metrics (never clipped) ----
    out.update({
        'energy_se': jnp.sum(energy_diff_raw ** 2),            # per-atom
        'energy_ae': jnp.sum(jnp.abs(energy_diff_raw)),
        'energy_se_total': jnp.sum(energy_diff_total_raw ** 2),  # total energy
        'energy_ae_total': jnp.sum(jnp.abs(energy_diff_total_raw)),
        'force_se': jnp.sum(force_diff_raw ** 2),
        'force_ae': jnp.sum(jnp.abs(force_diff_raw)),
        'stress_se': jnp.sum(stress_diff_raw ** 2) if periodic else 0.0,
        'stress_ae': jnp.sum(jnp.abs(stress_diff_raw)) if periodic else 0.0,
        'n_graphs': n_graphs,
        'n_atoms': n_atoms,
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
) -> Tuple[jnp.ndarray, Dict]:
    """Compute training loss for a single batch (no device dimension).

    Args:
        loss_type: 'huber' | 'nll' (legacy Huber-whitened) | 'gauss_nll'.
        periodic: Whether system is periodic. If False, stress loss is skipped.
        cov_mode: force covariance layout of the model ('scalar6' | 'irreps').
        beta_nll: beta of beta-NLL (only for loss_type='gauss_nll'; 0 = plain).
    """
    model = nnx.merge(graphdef, params)
    results = model(batch)
    t = _loss_terms(results, batch, huber_delta, loss_type, periodic, cov_mode, beta_nll)

    n_graphs, n_atoms = t['n_graphs'], t['n_atoms']
    energy_loss = t['energy_loss'] / n_graphs
    force_loss = t['force_loss'] / (3 * n_atoms)
    stress_loss = t['stress_loss'] / n_graphs

    total_loss = (
        energy_weight * energy_loss
        + force_weight * force_loss
        + stress_weight * stress_loss
    )
    aux = {
        'energy_loss': energy_loss,
        'force_loss': force_loss,
        'stress_loss': stress_loss,
        'energy_mse': t['energy_se'] / n_graphs,
        'force_mse': t['force_se'] / (3 * n_atoms),
        'stress_mse': t['stress_se'] / n_graphs if periodic else 0.0,
        'n_graphs': n_graphs,
        'n_atoms': n_atoms,
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
):
    """Create sharded training step with gradient synchronization.

    Uses shard_map to explicitly control SPMD execution:
    - Each device processes its own batch (indexed by leading dimension)
    - Gradients are synchronized via pmean across the 'dp' axis
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

        def loss_fn(params):
            return compute_loss(
                graphdef, params, batch,
                ew, fw, sw, huber_delta,
                loss_type=loss_type, periodic=periodic,
                cov_mode=cov_mode, beta_nll=beta_nll,
            )

        # Compute gradients locally
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # Synchronize gradients and metrics across devices
        grads = jax.lax.pmean(grads, axis_name='dp')
        loss = jax.lax.pmean(loss, axis_name='dp')
        aux = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name='dp'), aux)

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
                               beta_nll: float = 0.0):
    """Create sharded evaluation function that runs on all GPUs.

    Each device evaluates its own batch, then results are summed across devices.
    Loss terms use the same (tanh-clipped) residuals as training; the reported
    squared / absolute errors use raw residuals.
    """

    def eval_single_device(graphdef, params, batch):
        """Evaluate a single batch on one device. Returns raw sums."""
        batch = squeeze_batch(batch)
        model = nnx.merge(graphdef, params)
        results = model(batch)
        return _loss_terms(results, batch, huber_delta, loss_type, periodic, cov_mode, beta_nll)

    def eval_and_reduce(graphdef, params, batch):
        """Evaluate on each device and sum results across devices."""
        local_metrics = eval_single_device(graphdef, params, batch)
        # Sum across all devices using psum
        return jax.tree.map(lambda x: jax.lax.psum(x, axis_name='dp'), local_metrics)

    # Create sharded version
    sharded_eval = shard_map(
        eval_and_reduce,
        mesh=mesh,
        in_specs=(
            P(),       # graphdef - replicated
            P(),       # params - replicated
            P('dp'),   # batch - sharded along device axis
        ),
        out_specs=P(),  # metrics - reduced (same on all devices after psum)
    )

    return jax.jit(sharded_eval)


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
        'n_graphs': 0.0,
        'n_atoms': 0.0
    }

    if eval_step is None:
        eval_step = make_sharded_evaluate_step(mesh, huber_delta=huber_delta, loss_type=loss_type,
                                               periodic=periodic, cov_mode=cov_mode, beta_nll=beta_nll)

    for fname in files_pkl:
        # accept either a pickle path or an in-memory dataset object (with .graphs)
        dataset = fname if hasattr(fname, "graphs") else Dataset(file_path=fname)
        base_loader = BucketedDataLoader (
            dataset=dataset,
            batch_size=batch_size,
            n_buckets=8,
            shuffle=False,       # No shuffle for deterministic evaluation
            drop_last=False,     # Evaluate ALL data
            rngs=nnx.Rngs(0)    # Deterministic rngs (not used when shuffle=False)
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
            # Run sharded evaluation
            metrics = eval_step(graphdef, params_replicated, batch)

            # Accumulate results (metrics are already summed across devices)
            for k in totals:
                totals[k] += float(metrics[k])

    # Compute final metrics
    n_g = max(totals['n_graphs'], 1)
    n_a = max(totals['n_atoms'], 1)
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
        'stress_mae': totals['stress_ae'] / n_g,
        'force_rmse': np.sqrt(totals['force_se']/(3 * n_a)),
        'energy_loss': totals['energy_loss'] / n_g,
        'force_loss': totals['force_loss'] / (3 * n_a),
        'stress_loss': totals['stress_loss'] / n_g,
        'total_loss': total_loss,
        'n_graphs': n_g,
        'n_atoms': n_a,
    }


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
