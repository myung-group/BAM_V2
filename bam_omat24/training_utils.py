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


def compute_loss(
    graphdef: nnx.GraphDef,
    params: nnx.State,
    batch: jraph.GraphsTuple,
    energy_weight: float,
    force_weight: float,
    stress_weight: float,
    huber_delta: float
) -> Tuple[jnp.ndarray, Dict]:
    """Compute loss for a single batch (no device dimension)."""
    graph_mask = jraph.get_graph_padding_mask(batch)
    node_mask = jraph.get_node_padding_mask(batch)

    n_graphs = jnp.maximum(graph_mask.sum(), 1)
    n_atoms = jnp.maximum(node_mask.sum(), 1)

    model = nnx.merge(graphdef, params)
    energy, forces, stress = model(batch)

    # Energy loss
    energy_diff = (energy - batch.globals["energy"])/jnp.maximum(batch.n_node, 1)
    energy_loss = huber_loss(energy_diff, delta=huber_delta)
    energy_loss = jnp.sum(energy_loss * graph_mask) / n_graphs
    energy_mse = jnp.sum(energy_diff ** 2 * graph_mask) / n_graphs

    # Force loss
    forces_diff = forces - batch.nodes["forces"]
    force_loss = huber_loss(forces_diff, delta=huber_delta)
    force_loss = jnp.sum(force_loss * node_mask[:, None]) / (3 * n_atoms)
    force_mse = jnp.sum(forces_diff ** 2 * node_mask[:, None]) / (3 * n_atoms)

    # Stress loss
    stress_diff = (stress - batch.globals["stress"]) * graph_mask[:, None]
    stress_loss = jnp.sum(huber_loss(stress_diff, delta=huber_delta) * graph_mask[:, None]) / n_graphs
    stress_mse = jnp.sum(stress_diff ** 2 * graph_mask[:, None]) / n_graphs

    total_loss = energy_weight * energy_loss + force_weight * force_loss + stress_weight * stress_loss

    aux = {
        'energy_loss': energy_loss,
        'force_loss': force_loss,
        'stress_loss': stress_loss,
        'energy_mse': energy_mse,
        'force_mse': force_mse,
        'stress_mse': stress_mse,
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
    freeze_mask: Any = None
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
    """

    def per_device_step(graphdef, params, opt_state, schedule_state, ema_params, step, batch):
        """Training step for a single device's batch."""
        # Squeeze the leading device dimension: (1, ...) -> (...)
        batch = squeeze_batch(batch)
        def loss_fn(params):
            return compute_loss(
                graphdef, params, batch,
                energy_weight, force_weight, stress_weight, huber_delta
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

        energy_diff = (energy - batch.globals["energy"])/jnp.maximum(batch.n_node,1)
        energy_loss = huber_loss(energy_diff, delta=huber_delta)
        forces_diff = forces - batch.nodes["forces"]
        force_loss = huber_loss(forces_diff, delta=huber_delta)
        stress_diff = (stress - batch.globals["stress"]) * graph_mask[:, None]
        stress_loss = huber_loss(stress_diff, delta=huber_delta)
        # Return sums weighted by actual counts for proper aggregation
        return {
            'energy_loss': jnp.sum(energy_loss * graph_mask),
            'force_loss': jnp.sum(force_loss * node_mask[:, None]),
            'stress_loss': jnp.sum(stress_loss * graph_mask[:, None]),
            'energy_se': jnp.sum(energy_diff ** 2 * graph_mask),
            'force_se': jnp.sum(forces_diff ** 2 * node_mask[:, None]),
            'stress_se': jnp.sum(stress_diff ** 2 * graph_mask[:, None]),
            'energy_ae': jnp.sum(jnp.abs(energy_diff) * graph_mask),
            'force_ae': jnp.sum(jnp.abs(forces_diff) * node_mask[:, None]),
            'stress_ae': jnp.sum(jnp.abs(stress_diff) * graph_mask[:, None]),
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
