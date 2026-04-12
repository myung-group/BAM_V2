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


def compute_loss(
    graphdef: nnx.GraphDef,
    params: nnx.State,
    batch: jraph.GraphsTuple,
    energy_weight: float,
    force_weight: float,
    stress_weight: float,
    huber_delta: float,
    loss_type: str = 'huber',
) -> Tuple[jnp.ndarray, Dict]:
    """Compute training loss for a single batch (no device dimension).

    Args:
        loss_type: 'huber' for plain Huber loss, 'nll' for Huber-based NLL
                   with Cholesky-parameterized covariances.
    """
    graph_mask = jraph.get_graph_padding_mask(batch)
    node_mask = jraph.get_node_padding_mask(batch)

    n_graphs = jnp.maximum(graph_mask.sum(), 1)
    n_atoms = jnp.maximum(node_mask.sum(), 1)

    model = nnx.merge(graphdef, params)
    results = model(batch)
    energy = results['energy']
    forces = results['forces']
    stress = results['stress']

    n_node = jnp.maximum(batch.n_node, 1)
    energy_diff = (energy - batch.globals["energy"]) * graph_mask / n_node
    force_diff = (forces - batch.nodes["forces"]) * node_mask[:, None]
    stress_diff = (stress - batch.globals["stress"]) * graph_mask[:, None]

    if loss_type == 'nll':
        frc_var = results['force_variation']          # (n_atoms, 6)
        energy_var = results.get('energy_variation')  # (n_graphs, 1) or None
        stress_var = results.get('stress_variation')  # (n_graphs, 21) or None

        # ---- Energy NLL ----
        if energy_var is not None:
            energy_sigma = energy_var.squeeze(-1) * graph_mask + (1.0 - graph_mask)
            energy_z = energy_diff / energy_sigma
            energy_loss = jnp.sum(huber_loss(energy_z, delta=huber_delta)) / n_graphs
            energy_log_det = 2.0 * jnp.sum(jnp.log(energy_sigma) * graph_mask) / n_graphs
        else:
            energy_loss = jnp.sum(huber_loss(energy_diff, delta=huber_delta)) / n_graphs
            energy_log_det = 0.0

        # ---- Force NLL (3x3 Cholesky per atom) ----
        frc_L = _build_force_L(frc_var)
        frc_z = jax.vmap(
            lambda L, d: jax.scipy.linalg.solve_triangular(L, d, lower=True)
        )(frc_L, force_diff)
        force_loss = jnp.sum(huber_loss(frc_z, delta=huber_delta)) / (3 * n_atoms)
        frc_diag = jnp.diagonal(frc_L, axis1=-2, axis2=-1)
        force_log_det = 2.0 * jnp.sum(
            jnp.log(frc_diag) * node_mask[:, None]
        ) / (3 * n_atoms)

        # ---- Stress NLL (6x6 Cholesky per graph) ----
        if stress_var is not None:
            stress_L = _build_stress_L(stress_var)
            stress_z = jax.vmap(
                lambda L, d: jax.scipy.linalg.solve_triangular(L, d, lower=True)
            )(stress_L, stress_diff)
            stress_loss = jnp.sum(huber_loss(stress_z, delta=huber_delta)) / n_graphs
            stress_diag = jnp.diagonal(stress_L, axis1=-2, axis2=-1)
            stress_log_det = 2.0 * jnp.sum(
                jnp.log(stress_diag) * graph_mask[:, None]
            ) / n_graphs
        else:
            stress_loss = jnp.sum(huber_loss(stress_diff, delta=huber_delta)) / n_graphs
            stress_log_det = 0.0

        # Constant offset C = -min(log-det) to guarantee total_loss ≥ 0.
        # Per-term floors when all Cholesky diagonals = MIN_VAR:
        #   energy:  2·log(d)          (1 σ per config)
        #   force:   2·log(d)          (3 diag per atom, n_atoms cancels)
        #   stress:  12·log(d)         (6 diag per config)
        MIN_VAR = 0.04
        log_min_var = jnp.log(MIN_VAR)
        nll_const = -2.0 * log_min_var                       # force (always present)
        if energy_var is not None:
            nll_const = nll_const - 2.0 * log_min_var        # + energy offset
        if stress_var is not None:
            nll_const = nll_const - 12.0 * log_min_var       # + stress offset

        total_loss = (
            energy_weight * energy_loss
            + force_weight * force_loss
            + stress_weight * stress_loss
            + energy_log_det + force_log_det + stress_log_det
            + nll_const
        )

        aux = {
            'energy_loss': energy_loss + energy_log_det,
            'force_loss': force_loss + force_log_det,
            'stress_loss': stress_loss + stress_log_det,
        }

    else:  # 'huber' — plain Huber loss (no variance / NLL)
        energy_loss = jnp.sum(huber_loss(energy_diff, delta=huber_delta)) / n_graphs
        force_loss = jnp.sum(huber_loss(force_diff, delta=huber_delta)) / (3 * n_atoms)
        stress_loss = jnp.sum(huber_loss(stress_diff, delta=huber_delta)) / n_graphs

        total_loss = (
            energy_weight * energy_loss
            + force_weight * force_loss
            + stress_weight * stress_loss
        )

        aux = {
            'energy_loss': energy_loss,
            'force_loss': force_loss,
            'stress_loss': stress_loss,
        }

    energy_mse = jnp.sum(energy_diff ** 2) / n_graphs
    force_mse = jnp.sum(force_diff ** 2) / (3 * n_atoms)
    stress_mse = jnp.sum(stress_diff ** 2) / n_graphs

    aux.update({
        'energy_mse': energy_mse,
        'force_mse': force_mse,
        'stress_mse': stress_mse,
        'n_graphs': n_graphs,
        'n_atoms': n_atoms,
    })

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
        loss_type: 'huber' for plain Huber loss, 'nll' for Huber-based NLL.
    """

    def per_device_step(graphdef, params, opt_state, schedule_state, ema_params, step, batch):
        """Training step for a single device's batch."""
        # Squeeze the leading device dimension: (1, ...) -> (...)
        batch = squeeze_batch(batch)
        def loss_fn(params):
            return compute_loss(
                graphdef, params, batch,
                energy_weight, force_weight, stress_weight, huber_delta,
                loss_type=loss_type,
            )

        # Compute gradients locally
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # Synchronize gradients and metrics across devices
        grads = jax.lax.pmean(grads, axis_name='dp')
        loss = jax.lax.pmean(loss, axis_name='dp')
        aux = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name='dp'), aux)

        grad_norm = optax.global_norm(grads)

        # JAX-style conditional: skip update when grad_norm >= threshold
        def apply_update(_):
            """Apply gradient update and EMA when grad_norm is small."""
            updates, new_opt = optimizer.update(grads, opt_state, params)
            updates = optax.tree_utils.tree_scale(schedule_state.scale, updates)
            new_p = optax.apply_updates(params, updates)
            if freeze_mask is not None:
                # Only apply EMA to trainable params; frozen params keep current values
                new_ema = jax.tree.map(
                    lambda ema, p, m: jnp.where(
                        m, #
                        ema, # Freeze
                        ema_decay * ema + (1 - ema_decay) * p),
                        ema_params, new_p, freeze_mask
                )
            else:
                new_ema = jax.tree.map(
                    lambda ema, p: ema_decay * ema + (1 - ema_decay) * p,
                    ema_params, new_p
                )
            return new_p, new_opt, new_ema

        # Use jax.lax.cond for JAX-traceable conditional
        new_params, new_opt_state, new_ema_params = apply_update(None)
        metrics = {
            'loss': loss,
            'grad_norm': grad_norm,
            'update_applied': (grad_norm < 1.0).astype(jnp.float32),  # track skipped updates
            **aux
        }

        return new_params, new_opt_state, new_ema_params, step + 1, metrics

    # Create sharded version using shard_map
    sharded_step = shard_map(
        per_device_step,
        mesh=mesh,
        in_specs=(
            P(),       # graphdef - replicated
            P(),       # params - replicated
            P(),       # opt_state - replicated
            P(),       # schedule_state - replicated
            P(),       # ema_params - replicated
            P(),       # step - replicated
            P('dp'),   # batch - sharded along device axis
        ),
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

def make_sharded_evaluate_step(mesh: Mesh, huber_delta: float):
    """Create sharded evaluation function that runs on all GPUs.

    Each device evaluates its own batch, then results are summed across devices.
    """

    def eval_single_device(graphdef, params, batch):
        """Evaluate a single batch on one device.

        Returns raw sums (not averages) for proper aggregation across devices.
        """
        # Squeeze the leading device dimension: (1, ...) -> (...)
        batch = squeeze_batch(batch)
        graph_mask = jraph.get_graph_padding_mask(batch)
        node_mask = jraph.get_node_padding_mask(batch)

        model = nnx.merge(graphdef, params)
        energy, forces, stress = model(batch)

        vmax_E = 5.0
        n_node = jnp.maximum(batch.n_node, 1)
        tgt_energy_per_node = vmax_E*jnp.tanh(batch.globals["energy"] / n_node / vmax_E)
        prd_energy_per_node = vmax_E*jnp.tanh(energy /n_node / vmax_E)
        energy_diff = (prd_energy_per_node - tgt_energy_per_node) * graph_mask
        energy_loss = huber_loss(energy_diff, delta=huber_delta)

        vmax_F = 10.0
        tgt_forces = vmax_F*jnp.tanh (batch.nodes["forces"] / vmax_F)
        prd_forces = vmax_F*jnp.tanh (forces / vmax_F)
        forces_diff = (prd_forces - tgt_forces) * node_mask[:, None]
        force_loss = huber_loss(forces_diff, delta=huber_delta)

        vmax_S = 0.5
        tgt_stress = vmax_S * jnp.tanh (batch.globals["stress"] / vmax_S)
        prd_stress = vmax_S * jnp.tanh (stress / vmax_S)
        stress_diff = (prd_stress - tgt_stress) * graph_mask[:, None]
        stress_loss = huber_loss(stress_diff, delta=huber_delta)

        # Return sums weighted by actual counts for proper aggregation
        return {
            'energy_loss': jnp.sum(energy_loss),
            'force_loss': jnp.sum(force_loss),
            'stress_loss': jnp.sum(stress_loss),
            'energy_se': jnp.sum(energy_diff ** 2),
            'force_se': jnp.sum(forces_diff ** 2),
            'stress_se': jnp.sum(stress_diff ** 2),
            'energy_ae': jnp.sum(jnp.abs(energy_diff)),
            'force_ae': jnp.sum(jnp.abs(forces_diff)),
            'stress_ae': jnp.sum(jnp.abs(stress_diff)),
            'n_graphs': graph_mask.sum(),  # Actual count (not max with 1)
            'n_atoms': node_mask.sum(),    # Actual count
        }

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
    huber_delta: float = 0.02
) -> Dict:
    """Evaluate model on validation set using all GPUs.

    Args:
        graphdef: Model graph definition
        params: Model parameters (unreplicated)
        files_pkl: List of validation data files
        batch_size: Batch size per device
        n_devices: Number of devices
        mesh: Device mesh
        energy_weight: Weight for energy in loss
        force_weight: Weight for forces in loss
        stress_weight: Weight for stress in loss
        huber_delta: Huber loss delta

    Returns:
        Dictionary with evaluation metrics
    """
    # Replicate params for sharded computation
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
        'n_graphs': 0.0,
        'n_atoms': 0.0
    }

    # Create sharded evaluation function
    eval_step = make_sharded_evaluate_step(mesh, huber_delta=huber_delta)

    for fname in files_pkl:
        dataset = Dataset(file_path=fname)
        base_loader = BucketedDataLoader (
            dataset=dataset,
            batch_size=batch_size,
            n_buckets=8,
            shuffle=False,       # No shuffle for deterministic evaluation
            drop_last=False,     # Evaluate ALL data
            rngs=nnx.Rngs(0)    # Deterministic rngs (not used when shuffle=False)
        )
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
        'energy_mae': totals['energy_ae'] / n_g,
        'force_mae': totals['force_ae'] / (3 * n_a),
        'stress_mae': totals['stress_ae'] / n_g,
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
