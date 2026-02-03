"""NequIP/RACE multi-host multi-GPU training with hybrid parallelism.

This version implements 2D parallelism:
- Data Parallel (DP) across hosts/nodes
- Tensor Parallel (TP) within each host (across GPUs in a node)

Example setup: 2 hosts × 4 GPUs = 8 total devices
- DP axis: 2 (hosts)
- TP axis: 4 (GPUs per host)

Usage with SLURM:
    srun python train_sharded_multihost.py
"""

# ============================================================================
# DETERMINISM: Must be set BEFORE importing JAX
# ============================================================================
import os
# Force XLA to use deterministic scatter/gather ops on GPU.
# os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
# Force deterministic cuDNN algorithms
# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from typing import Dict, Tuple, List, Any, Optional
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils, multihost_utils
try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map
from flax import nnx
import optax
import pickle
import jraph
import time
import numpy as np
from functools import partial
from pathlib import Path

from bam_omat24.data_nnx import Dataset, BucketedDataLoader, MultiDeviceDataLoader
from bam_omat24.race_nnx import RACE

jax.config.update("jax_enable_x64", False)


# =============================================================================
# Multi-Host Distributed Initialization
# =============================================================================

def initialize_distributed():
    """Initialize JAX distributed runtime for multi-host training.

    This function handles both SLURM-managed and manual setups.
    For SLURM: srun automatically sets the necessary environment variables.
    For manual: Set JAX_COORDINATOR_ADDRESS, JAX_NUM_PROCESSES, JAX_PROCESS_INDEX.
    """
    if 'SLURM_PROCID' in os.environ:
        # SLURM environment detected
        process_id = int(os.environ['SLURM_PROCID'])
        num_processes = int(os.environ['SLURM_NPROCS'])

        # Get coordinator address (first node in the job)
        coordinator_address = os.environ.get('JAX_COORDINATOR_ADDRESS')
        if coordinator_address is None:
            import subprocess
            result = subprocess.run(
                ['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']],
                capture_output=True, text=True
            )
            head_node = result.stdout.strip().split('\n')[0]
            coordinator_address = f"{head_node}:29500"

        print(f"[Process {process_id}/{num_processes}] Initializing with coordinator: {coordinator_address}")

        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=num_processes,
            process_id=process_id,
        )
    else:
        # Try automatic initialization (works for single-host or pre-configured multi-host)
        jax.distributed.initialize()

    # Print distributed info
    print(f"Process {jax.process_index()} of {jax.process_count()}")
    print(f"Local devices: {jax.local_device_count()}")
    print(f"Total devices: {jax.device_count()}")


# =============================================================================
# Device Mesh Setup for 2D Parallelism
# =============================================================================

def setup_mesh_2d(n_dp: Optional[int] = None, n_tp: Optional[int] = None) -> Tuple[Mesh, int, int]:
    """Setup 2D device mesh for hybrid Data Parallel + Tensor Parallel.

    Args:
        n_dp: Number of data parallel replicas (default: number of hosts)
        n_tp: Number of tensor parallel devices per replica (default: GPUs per host)

    Returns:
        mesh: JAX Mesh with axes ('dp', 'tp')
        n_dp: Actual number of DP replicas
        n_tp: Actual number of TP devices

    Example for 2 hosts × 4 GPUs:
        mesh shape: (2, 4)
        'dp' axis: across hosts (data parallel)
        'tp' axis: within host (tensor parallel)
    """
    n_hosts = jax.process_count()
    n_local_devices = jax.local_device_count()
    n_total_devices = jax.device_count()

    # Default: DP across hosts, TP within host
    if n_dp is None:
        n_dp = n_hosts
    if n_tp is None:
        n_tp = n_local_devices

    assert n_dp * n_tp == n_total_devices, \
        f"n_dp ({n_dp}) × n_tp ({n_tp}) must equal total devices ({n_total_devices})"

    print(f"Setting up 2D mesh: ({n_dp}, {n_tp}) = (DP, TP)")
    print(f"  - Data Parallel (DP): {n_dp} replicas across hosts")
    print(f"  - Tensor Parallel (TP): {n_tp} devices per replica")

    # Create device mesh
    # For multi-host: jax.devices() returns all devices across all hosts
    all_devices = np.array(jax.devices()).reshape(n_dp, n_tp)
    mesh = Mesh(all_devices, axis_names=('dp', 'tp'))

    print(f"Mesh devices:\n{mesh.devices}")

    return mesh, n_dp, n_tp


def setup_mesh_1d() -> Tuple[Mesh, int]:
    """Setup 1D device mesh for pure data parallelism (backward compatible).

    Returns:
        mesh: JAX Mesh with axis ('dp',)
        n_devices: Total number of devices
    """
    devices = jax.devices()
    n_devices = len(devices)
    print(f"Setting up 1D mesh with {n_devices} device(s)")
    for i, d in enumerate(devices):
        print(f"  Device {i}: {d}")

    device_mesh = mesh_utils.create_device_mesh((n_devices,))
    mesh = Mesh(device_mesh, axis_names=('dp',))
    return mesh, n_devices


# =============================================================================
# Sharding Utilities for 2D Parallelism
# =============================================================================

def replicate_dp(x: Any, mesh: Mesh) -> Any:
    """Replicate array across DP axis only (for TP-sharded params).

    Each DP replica has the same copy, but within each replica,
    the array can be sharded across TP devices.
    """
    return jax.device_put(x, NamedSharding(mesh, P()))


def replicate_all(x: Any, mesh: Mesh) -> Any:
    """Fully replicate array across all devices (DP and TP)."""
    return jax.device_put(x, NamedSharding(mesh, P()))


def replicate_pytree(pytree: Any, mesh: Mesh) -> Any:
    """Replicate entire pytree across all devices."""
    return jax.tree.map(lambda x: replicate_all(x, mesh), pytree)


def shard_params_tp(params: Any, mesh: Mesh) -> Any:
    """Shard model parameters across TP axis (within host).

    Strategy:
    - Linear layer kernels: shard output dimension across TP
    - Biases: shard across TP
    - LayerNorm, embeddings: replicate

    This keeps communication within the host (fast NVLink).
    """
    def get_sharding_spec(path: str, x: jnp.ndarray) -> NamedSharding:
        path_str = '/'.join(str(p) for p in path)

        # Linear layer weights: shard output dimension
        if 'kernel' in path_str and x.ndim == 2:
            # Column-parallel: (in_features, out_features) -> shard out_features
            return NamedSharding(mesh, P(None, 'tp'))
        elif 'kernel' in path_str and x.ndim == 3:
            # Some conv-like kernels: shard last dimension
            return NamedSharding(mesh, P(None, None, 'tp'))
        elif 'bias' in path_str and x.ndim == 1:
            # Bias: shard to match output sharding
            return NamedSharding(mesh, P('tp'))
        else:
            # Everything else: fully replicate
            return NamedSharding(mesh, P())

    return jax.tree_util.tree_map_with_path(
        lambda path, x: jax.device_put(x, get_sharding_spec(path, x)),
        params
    )


def shard_batch_dp(batch: jraph.GraphsTuple, mesh: Mesh, use_2d: bool = True) -> jraph.GraphsTuple:
    """Shard batch across DP axis (across hosts).

    For 2D mesh: batch dimension is sharded across DP, replicated across TP.
    For 1D mesh: batch dimension is sharded across DP.
    """
    if use_2d:
        # 2D mesh: (dp, tp) - shard batch on dp, replicate on tp
        batch_sharding = NamedSharding(mesh, P('dp', None))
    else:
        # 1D mesh: (dp,) - shard batch on dp
        batch_sharding = NamedSharding(mesh, P('dp'))

    def shard_array(x):
        if x is None:
            return None
        return jax.device_put(x, batch_sharding)

    def shard_dict(d):
        if d is None:
            return None
        return {k: shard_array(v) for k, v in d.items()}

    return jraph.GraphsTuple(
        n_node=shard_array(batch.n_node),
        n_edge=shard_array(batch.n_edge),
        nodes=shard_dict(batch.nodes),
        edges=shard_dict(batch.edges),
        globals=shard_dict(batch.globals),
        senders=shard_array(batch.senders),
        receivers=shard_array(batch.receivers),
    )


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
    energy_diff = (energy - batch.globals["energy"]) / batch.n_node
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
        return {k: squeeze_array(v) for k, v in d.items()}

    return jraph.GraphsTuple(
        n_node=squeeze_array(batch.n_node),
        n_edge=squeeze_array(batch.n_edge),
        nodes=squeeze_dict(batch.nodes),
        edges=squeeze_dict(batch.edges),
        globals=squeeze_dict(batch.globals),
        senders=squeeze_array(batch.senders),
        receivers=squeeze_array(batch.receivers),
    )


# =============================================================================
# Sharded Training Step (2D Parallelism)
# =============================================================================

def make_sharded_train_step_2d(
    optimizer: optax.GradientTransformation,
    mesh: Mesh,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    stress_weight: float = 1.0,
    huber_delta: float = 0.1,
    ema_decay: float = 0.99
):
    """Create sharded training step with 2D parallelism.

    Parallelism strategy:
    - Batch is sharded across DP axis (different batches on different hosts)
    - Parameters are replicated across DP, potentially sharded across TP
    - Gradients are:
      1. Computed locally per device
      2. Reduced (all-reduce) across TP axis within each DP replica
      3. Averaged (pmean) across DP axis
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

        # Step 1: Reduce gradients across TP axis (within-host all-reduce)
        # This combines gradients from different parts of the model
        grads = jax.lax.psum(grads, axis_name='tp')

        # Step 2: Average gradients across DP axis (cross-host all-reduce)
        # This averages gradients from different data batches
        grads = jax.lax.pmean(grads, axis_name='dp')

        # Average loss and metrics across both axes
        loss = jax.lax.pmean(loss, axis_name='dp')
        loss = jax.lax.pmean(loss, axis_name='tp')
        aux = jax.tree.map(
            lambda x: jax.lax.pmean(jax.lax.pmean(x, axis_name='dp'), axis_name='tp'),
            aux
        )

        grad_norm = optax.global_norm(grads)

        # Apply gradient update
        def apply_update(_):
            """Apply gradient update and EMA."""
            updates, new_opt = optimizer.update(grads, opt_state, params)
            updates = optax.tree_utils.tree_scale(schedule_state.scale, updates)
            new_p = optax.apply_updates(params, updates)
            new_ema = jax.tree.map(
                lambda ema, p: ema_decay * ema + (1 - ema_decay) * p,
                ema_params, new_p
            )
            return new_p, new_opt, new_ema

        new_params, new_opt_state, new_ema_params = apply_update(None)

        metrics = {
            'loss': loss,
            'grad_norm': grad_norm,
            'update_applied': (grad_norm < 1.0).astype(jnp.float32),
            **aux
        }

        return new_params, new_opt_state, new_ema_params, step + 1, metrics

    # Create sharded version using shard_map with 2D specs
    # Note: For true TP, params would have P(None, 'tp') for sharded dimensions
    # Here we use P() for simplicity (replicated params)
    sharded_step = shard_map(
        per_device_step,
        mesh=mesh,
        in_specs=(
            P(),           # graphdef - replicated
            P(),           # params - replicated (or P(None, 'tp') for TP-sharded)
            P(),           # opt_state - replicated
            P(),           # schedule_state - replicated
            P(),           # ema_params - replicated
            P(),           # step - replicated
            P('dp', None), # batch - sharded on DP, replicated on TP
        ),
        out_specs=(
            P(),           # new_params
            P(),           # new_opt_state
            P(),           # new_ema_params
            P(),           # step
            P(),           # metrics
        ),
    )

    return jax.jit(sharded_step)


def make_sharded_train_step_1d(
    optimizer: optax.GradientTransformation,
    mesh: Mesh,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    stress_weight: float = 1.0,
    huber_delta: float = 0.1,
    ema_decay: float = 0.99
):
    """Create sharded training step for 1D data parallelism (original version)."""

    def per_device_step(graphdef, params, opt_state, schedule_state, ema_params, step, batch):
        """Training step for a single device's batch."""
        batch = squeeze_batch(batch)

        def loss_fn(params):
            return compute_loss(
                graphdef, params, batch,
                energy_weight, force_weight, stress_weight, huber_delta
            )

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # Synchronize gradients and metrics across devices
        grads = jax.lax.pmean(grads, axis_name='dp')
        loss = jax.lax.pmean(loss, axis_name='dp')
        aux = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name='dp'), aux)

        grad_norm = optax.global_norm(grads)

        def apply_update(_):
            updates, new_opt = optimizer.update(grads, opt_state, params)
            updates = optax.tree_utils.tree_scale(schedule_state.scale, updates)
            new_p = optax.apply_updates(params, updates)
            new_ema = jax.tree.map(
                lambda ema, p: ema_decay * ema + (1 - ema_decay) * p,
                ema_params, new_p
            )
            return new_p, new_opt, new_ema

        new_params, new_opt_state, new_ema_params = apply_update(None)

        metrics = {
            'loss': loss,
            'grad_norm': grad_norm,
            'update_applied': (grad_norm < 1.0).astype(jnp.float32),
            **aux
        }

        return new_params, new_opt_state, new_ema_params, step + 1, metrics

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
# Sharded Evaluation (2D Parallelism)
# =============================================================================

def make_sharded_evaluate_step_2d(mesh: Mesh, huber_delta: float):
    """Create sharded evaluation function for 2D parallelism."""

    def eval_single_device(graphdef, params, batch):
        """Evaluate a single batch on one device."""
        batch = squeeze_batch(batch)
        graph_mask = jraph.get_graph_padding_mask(batch)
        node_mask = jraph.get_node_padding_mask(batch)

        model = nnx.merge(graphdef, params)
        energy, forces, stress = model(batch)

        energy_diff = (energy - batch.globals["energy"]) / batch.n_node
        energy_loss = huber_loss(energy_diff, delta=huber_delta)
        forces_diff = forces - batch.nodes["forces"]
        force_loss = huber_loss(forces_diff, delta=huber_delta)
        stress_diff = (stress - batch.globals["stress"]) * graph_mask[:, None]
        stress_loss = huber_loss(stress_diff, delta=huber_delta)

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
            'n_graphs': graph_mask.sum(),
            'n_atoms': node_mask.sum(),
        }

    def eval_and_reduce(graphdef, params, batch):
        """Evaluate on each device and sum results across all devices."""
        local_metrics = eval_single_device(graphdef, params, batch)
        # Sum across both DP and TP axes
        return jax.tree.map(
            lambda x: jax.lax.psum(jax.lax.psum(x, axis_name='tp'), axis_name='dp'),
            local_metrics
        )

    sharded_eval = shard_map(
        eval_and_reduce,
        mesh=mesh,
        in_specs=(
            P(),           # graphdef - replicated
            P(),           # params - replicated
            P('dp', None), # batch - sharded on DP
        ),
        out_specs=P(),
    )

    return jax.jit(sharded_eval)


def make_sharded_evaluate_step_1d(mesh: Mesh, huber_delta: float):
    """Create sharded evaluation function for 1D data parallelism."""

    def eval_single_device(graphdef, params, batch):
        batch = squeeze_batch(batch)
        graph_mask = jraph.get_graph_padding_mask(batch)
        node_mask = jraph.get_node_padding_mask(batch)

        model = nnx.merge(graphdef, params)
        energy, forces, stress = model(batch)

        energy_diff = (energy - batch.globals["energy"]) / batch.n_node
        energy_loss = huber_loss(energy_diff, delta=huber_delta)
        forces_diff = forces - batch.nodes["forces"]
        force_loss = huber_loss(forces_diff, delta=huber_delta)
        stress_diff = (stress - batch.globals["stress"]) * graph_mask[:, None]
        stress_loss = huber_loss(stress_diff, delta=huber_delta)

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
            'n_graphs': graph_mask.sum(),
            'n_atoms': node_mask.sum(),
        }

    def eval_and_reduce(graphdef, params, batch):
        local_metrics = eval_single_device(graphdef, params, batch)
        return jax.tree.map(lambda x: jax.lax.psum(x, axis_name='dp'), local_metrics)

    sharded_eval = shard_map(
        eval_and_reduce,
        mesh=mesh,
        in_specs=(
            P(),       # graphdef
            P(),       # params
            P('dp'),   # batch
        ),
        out_specs=P(),
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
    use_2d: bool = True
) -> Dict:
    """Evaluate model on validation set using all GPUs."""
    # Replicate params for sharded computation
    params_replicated = replicate_pytree(params, mesh)

    # Initialize accumulators
    totals = {
        'energy_loss': 0.0, 'force_loss': 0.0, 'stress_loss': 0.0,
        'energy_se': 0.0, 'force_se': 0.0, 'stress_se': 0.0,
        'energy_ae': 0.0, 'force_ae': 0.0, 'stress_ae': 0.0,
        'n_graphs': 0.0, 'n_atoms': 0.0
    }

    # Create appropriate sharded evaluation function
    if use_2d:
        eval_step = make_sharded_evaluate_step_2d(mesh, huber_delta=huber_delta)
    else:
        eval_step = make_sharded_evaluate_step_1d(mesh, huber_delta=huber_delta)

    for fname in files_pkl:
        dataset = Dataset(file_path=fname)
        base_loader = BucketedDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            n_buckets=8,
            shuffle=False,
            drop_last=False,
            rngs=nnx.Rngs(0)
        )
        loader = MultiDeviceDataLoader(
            base_loader=base_loader,
            n_devices=n_devices,
            mesh=mesh,
            drop_incomplete=False
        )

        for batch, info in loader:
            metrics = eval_step(graphdef, params_replicated, batch)
            for k in totals:
                totals[k] += float(metrics[k])

    # Compute final metrics
    n_g = max(totals['n_graphs'], 1)
    n_a = max(totals['n_atoms'], 1)
    total_loss = (energy_weight * totals['energy_loss'] / n_g +
                  force_weight * totals['force_loss'] / (3 * n_a) +
                  stress_weight * totals['stress_loss'] / n_g)

    return {
        #'energy_rmse': np.sqrt(totals['energy_se'] / n_g),
        #'force_rmse': np.sqrt(totals['force_se'] / (3 * n_a)),
        #'stress_rmse': np.sqrt(totals['stress_se'] / n_g),
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
    """Save checkpoint (only on process 0 for multi-host)."""
    if jax.process_index() == 0:
        with open(path, 'wb') as f:
            pickle.dump(state_dict, f)
        print(f"Checkpoint saved to {path}")


def load_checkpoint(path: str) -> Dict:
    """Load checkpoint."""
    with open(path, 'rb') as f:
        return pickle.load(f)


# =============================================================================
# Multi-Host Data Loading
# =============================================================================

class MultiHostDataLoader:
    """DataLoader that properly handles multi-host data sharding.

    Each host loads only its portion of the data based on process_index.
    """

    def __init__(
        self,
        base_loader: BucketedDataLoader,
        mesh: Mesh,
        n_dp: int,
        n_tp: int,
        use_2d: bool = True
    ):
        self.base_loader = base_loader
        self.mesh = mesh
        self.n_dp = n_dp
        self.n_tp = n_tp
        self.use_2d = use_2d
        self.process_id = jax.process_index()
        self.n_processes = jax.process_count()

    def __iter__(self):
        """Iterate over batches, each host gets its own slice."""
        batch_buffer = []

        for batch in self.base_loader:
            batch_buffer.append(batch)

            # When we have enough batches for all DP replicas
            if len(batch_buffer) >= self.n_dp:
                # Each process takes its corresponding batch
                local_batch = batch_buffer[self.process_id]

                # Convert to global array for multi-host
                if self.use_2d:
                    # For 2D mesh: replicate across TP within this host
                    global_batch = self._make_global_batch_2d(local_batch)
                else:
                    global_batch = self._make_global_batch_1d(local_batch)

                yield global_batch
                batch_buffer = []

    def _make_global_batch_2d(self, local_batch: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Convert local batch to global array with 2D sharding."""
        def to_global(x):
            if x is None:
                return None
            # Add leading dimension for DP axis
            x = jnp.expand_dims(x, axis=0)
            return multihost_utils.host_local_array_to_global_array(
                x, self.mesh, P('dp', None)
            )

        def to_global_dict(d):
            if d is None:
                return None
            return {k: to_global(v) for k, v in d.items()}

        return jraph.GraphsTuple(
            n_node=to_global(local_batch.n_node),
            n_edge=to_global(local_batch.n_edge),
            nodes=to_global_dict(local_batch.nodes),
            edges=to_global_dict(local_batch.edges),
            globals=to_global_dict(local_batch.globals),
            senders=to_global(local_batch.senders),
            receivers=to_global(local_batch.receivers),
        )

    def _make_global_batch_1d(self, local_batch: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Convert local batch to global array with 1D sharding."""
        def to_global(x):
            if x is None:
                return None
            x = jnp.expand_dims(x, axis=0)
            return multihost_utils.host_local_array_to_global_array(
                x, self.mesh, P('dp')
            )

        def to_global_dict(d):
            if d is None:
                return None
            return {k: to_global(v) for k, v in d.items()}

        return jraph.GraphsTuple(
            n_node=to_global(local_batch.n_node),
            n_edge=to_global(local_batch.n_edge),
            nodes=to_global_dict(local_batch.nodes),
            edges=to_global_dict(local_batch.edges),
            globals=to_global_dict(local_batch.globals),
            senders=to_global(local_batch.senders),
            receivers=to_global(local_batch.receivers),
        )


# =============================================================================
# Main Training Loop
# =============================================================================

def train_sharded(
    config: Dict,
    train_files_pkl: List[str],
    valid_files_pkl: List[str],
    use_2d_parallelism: bool = False
):
    """Main training function with hybrid parallelism support.

    Args:
        config: Training configuration
        train_files_pkl: List of training data files
        valid_files_pkl: List of validation data files
        use_2d_parallelism: If True, use 2D (DP + TP) parallelism.
                           If False, use 1D (DP only) parallelism.
    """
    # Setup mesh
    if use_2d_parallelism and jax.process_count() > 1:
        mesh, n_dp, n_tp = setup_mesh_2d()
        n_devices = n_dp * n_tp
        make_train_step = make_sharded_train_step_2d
    else:
        mesh, n_devices = setup_mesh_1d()
        n_dp, n_tp = n_devices, 1
        make_train_step = make_sharded_train_step_1d
        use_2d_parallelism = False

    # Only process 0 writes to log file
    if jax.process_index() == 0:
        fout = open(config.get('fname_log', 'loss_sharded.out'), 'w', 1)
    else:
        fout = open('/dev/null', 'w')  # Discard output from other processes

    print("=" * 70, file=fout)
    print(f"Hybrid Parallel Training", file=fout)
    print(f"  Hosts (DP): {n_dp}", file=fout)
    print(f"  GPUs per host (TP): {n_tp}", file=fout)
    print(f"  Total devices: {n_devices}", file=fout)
    print(f"  Batch size per device: {config['batch_size']}", file=fout)
    print(f"  Effective batch size: {config['batch_size'] * n_dp}", file=fout)
    print(f"  2D Parallelism: {use_2d_parallelism}", file=fout)
    print("=" * 70, file=fout)

    # Create model
    seed = config.get('seed', 42)
    model_rngs = nnx.Rngs(seed)

    model = RACE(
        n_species=len(config["atom_energies"]),
        lmax=config["lmax"],
        hidden_size=config["hidden_size"],
        n_layers=config["n_layers"],
        radial_basis_size=config["radial_basis_size"],
        radial_mlp_size=config["radial_mlp_size"],
        radial_mlp_layers=config["radial_mlp_layers"],
        radial_polynomial_p=config["radial_polynomial_p"],
        mlp_init_scale=config["mlp_init_scale"],
        shift=0.0,
        scale=1.0,
        avg_n_neighbors=25.0,
        atom_energies=config["atom_energies"],
        l_train=True,
        rngs=model_rngs
    )

    graphdef, params = nnx.split(model, nnx.Param)
    ema_params = params

    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"Model parameters: {n_params:,}", file=fout)

    # Optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.amsgrad(config["learning_rate"])
    )
    schedule = optax.contrib.reduce_on_plateau(
        factor=config["lr_schedule_factor"],
        patience=config["lr_schedule_patience"],
        min_scale=config["lr_schedule_min_scale"],
    )
    opt_state = optimizer.init(params)
    schedule_state = schedule.init(params)

    # Replicate across devices
    params = replicate_pytree(params, mesh)
    ema_params = replicate_pytree(ema_params, mesh)
    opt_state = replicate_pytree(opt_state, mesh)
    schedule_state = replicate_pytree(schedule_state, mesh)

    step = replicate_all(jnp.array(0), mesh)
    energy_weight = config.get('energy_weight', 1.0)
    force_weight = config.get('force_weight', 1.0)
    stress_weight = config.get('stress_weight', 1.0)
    huber_delta = config.get('huber_delta', 0.02)

    # Create training step
    train_step = make_train_step(
        optimizer=optimizer,
        mesh=mesh,
        energy_weight=energy_weight,
        force_weight=force_weight,
        stress_weight=stress_weight,
        huber_delta=huber_delta,
        ema_decay=config.get('ema_decay', 0.99)
    )

    # Load checkpoint if exists
    best_val_loss = float('inf')
    if config.get('restart', False) and Path('ckpt_best.pkl').exists():
        ckpt = load_checkpoint('ckpt_best.pkl')
        params = replicate_pytree(ckpt['params'], mesh)
        ema_params = replicate_pytree(ckpt['ema_params'], mesh)
        opt_state = replicate_pytree(ckpt['opt_state'], mesh)
        schedule_state = replicate_pytree(ckpt['schedule_state'], mesh)
        step = replicate_all(jnp.array(ckpt.get('step', 0)), mesh)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from step {ckpt.get('step', 0)}", file=fout)

    # Training loop
    best_state = None

    for epoch in range(config["n_epochs"]):
        epoch_start = time.time()

        for ipkl, train_pkl in enumerate(train_files_pkl):
            train_dataset = Dataset(file_path=train_pkl)
            loader_rngs = nnx.Rngs(seed + 100 * epoch + ipkl)

            base_loader = BucketedDataLoader(
                dataset=train_dataset,
                batch_size=config['batch_size'],
                n_buckets=8,
                shuffle=True,
                drop_last=True,
                rngs=loader_rngs
            )

            # Use appropriate data loader based on parallelism mode
            if use_2d_parallelism:
                loader = MultiHostDataLoader(
                    base_loader=base_loader,
                    mesh=mesh,
                    n_dp=n_dp,
                    n_tp=n_tp,
                    use_2d=True
                )
            else:
                loader = MultiDeviceDataLoader(
                    base_loader=base_loader,
                    n_devices=n_devices,
                    mesh=mesh,
                    drop_incomplete=True
                )

            # Metrics accumulators
            acc_energy_loss, acc_force_loss, acc_stress_loss = 0., 0., 0.
            acc_graphs, acc_atoms, acc_grad = 0., 0., 0.
            dataset_start = time.time()
            current_step = int(np.asarray(step))

            for batch_data in loader:
                # Handle different loader return types
                if isinstance(batch_data, tuple):
                    batch, valid_device_count = batch_data
                else:
                    batch = batch_data

                # Training step
                params, opt_state, ema_params, step, metrics = train_step(
                    graphdef, params, opt_state, schedule_state, ema_params, step, batch
                )

                # Accumulate metrics
                n_g = float(metrics['n_graphs'])
                n_a = float(metrics['n_atoms'])
                acc_energy_loss += float(metrics['energy_loss']) * n_g
                acc_force_loss += float(metrics['force_loss']) * 3 * n_a
                acc_stress_loss += float(metrics['stress_loss']) * n_g
                acc_graphs += n_g
                acc_atoms += n_a
                acc_grad += float(metrics['grad_norm']) * n_g

                current_step = int(np.asarray(step))

                if current_step % 50 == 0:
                    energy_loss = acc_energy_loss / max(acc_graphs, 1)
                    force_loss = acc_force_loss / max(3 * acc_atoms, 1)
                    stress_loss = acc_stress_loss / max(acc_graphs, 1)
                    total_loss = energy_weight * energy_loss + force_weight * force_loss + stress_weight * stress_loss
                    print(f"step {current_step} LOSS: {total_loss:.6f} E_LOSS: {energy_loss:.6f} "
                          f"F_LOSS: {force_loss:.6f}", file=fout)

            unrep_schedule = unreplicate_pytree(schedule_state)
            lr = config['learning_rate'] * float(unrep_schedule.scale)
            dataset_time = time.time() - dataset_start

            energy_loss = acc_energy_loss / max(acc_graphs, 1)
            force_loss = acc_force_loss / max(3 * acc_atoms, 1)
            stress_loss = acc_stress_loss / max(acc_graphs, 1)
            total_loss = energy_weight * energy_loss + force_weight * force_loss + stress_weight * stress_loss

            print(f"Epoch {epoch+1}/{config['n_epochs']} IPKL {ipkl+1} ({dataset_time:.1f}s) "
                  f"Step {current_step} LR: {lr:.2e}", file=fout)
            print(f"  Train | Epoch {epoch+1}/{ipkl+1} | Loss: {total_loss:.6f} | "
                  f"E_LOSS: {energy_loss:.6f} | "
                  f"F_LOSS: {force_loss:.6f} | "
                  f"S_LOSS: {stress_loss:.6f} | "
                  f"Grad: {acc_grad/max(acc_graphs,1):.4f}", file=fout)

            # Reset accumulators
            acc_energy_loss, acc_force_loss, acc_stress_loss = 0., 0., 0.
            acc_graphs, acc_atoms, acc_grad = 0., 0., 0.
            dataset_start = time.time()

            # Validation
            val_start = time.time()
            unrep_ema = unreplicate_pytree(ema_params)

            val_metrics = evaluate_sharded(
                graphdef=graphdef,
                params=unrep_ema,
                files_pkl=valid_files_pkl,
                batch_size=config['batch_size'],
                n_devices=n_devices,
                mesh=mesh,
                energy_weight=energy_weight,
                force_weight=force_weight,
                stress_weight=stress_weight,
                huber_delta=huber_delta,
                use_2d=use_2d_parallelism
            )
            val_time = time.time() - val_start

            val_loss = val_metrics['total_loss']
            print(f"  Valid ({val_time:.1f}s) | Epoch {epoch+1}/{ipkl+1} | Loss: {val_loss:.6f} | "
                  f"E_LOSS: {val_metrics['energy_loss']:.6f} | "
                  f"F_LOSS: {val_metrics['force_loss']:.6f} | "
                  f"S_LOSS: {val_metrics['stress_loss']:.6f} | "
                  f"E_MAE: {val_metrics['energy_mae']:.6f} | "
                  f"F_MAE: {val_metrics['force_mae']:.6f} | "
                  f"S_MAE: {val_metrics['stress_mae']:.6f}", file=fout)

            # Update LR schedule
            unrep_params = unreplicate_pytree(params)
            unrep_schedule = unreplicate_pytree(schedule_state)
            _, new_schedule = schedule.update(
                updates=unrep_params,
                state=unrep_schedule,
                value=val_loss
            )
            schedule_state = replicate_pytree(new_schedule, mesh)

            # Save best model
            if val_loss < best_val_loss:
                improvement = best_val_loss - val_loss
                best_val_loss = val_loss
                best_state = {
                    'params': unrep_params,
                    'ema_params': unrep_ema,
                    'opt_state': unreplicate_pytree(opt_state),
                    'schedule_state': new_schedule,
                    'step': int(np.asarray(step)),
                    'best_val_loss': best_val_loss,
                    'metrics': val_metrics,
                }
                print(f"  *** New best val loss: {best_val_loss:.6f} "
                      f"(improved by {improvement:.6f}) ***", file=fout)
                save_checkpoint(best_state, 'ckpt_best.pkl')
            else:
                print(f"  No improvement for {new_schedule.plateau_count} evaluations "
                      f"(best: {best_val_loss:.6f})", file=fout)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{config['n_epochs']} completed in {epoch_time:.1f}s", file=fout)

    # Save final checkpoint
    if best_state is not None:
        save_checkpoint(best_state, 'ckpt_best.pkl')
        print("\nFinal checkpoint saved", file=fout)

    print("\nTraining completed!", file=fout)
    fout.close()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import json
    from natsort import natsorted
    from bam_omat24.atom_energies import ATOM_ENERGIES
    # Initialize distributed runtime FIRST
    initialize_distributed()

    # Only process 0 prints startup info
    if jax.process_index() == 0:
        print("=" * 70)
        print("JAX Multi-Host Sharded Training (Hybrid Parallelism)")
        print("=" * 70)
        print(f"JAX version: {jax.__version__}")
        print(f"Process: {jax.process_index()} / {jax.process_count()}")
        print(f"Local devices: {jax.local_devices()}")
        print(f"Total devices: {jax.device_count()}")
        print("=" * 70)

    # Load config
    with open('input_omat24.json') as f:
        config = json.load(f)
    config['atom_energies'] = ATOM_ENERGIES

    train_path = Path('/home/willow/Python/BAM-nequip-main/omat24_1M/train')
    valid_path = Path('/home/willow/Python/BAM-nequip-main/omat24_1M/val')

    train_files = natsorted(list(train_path.glob('omat*.pkl')))
    valid_files = natsorted(list(valid_path.glob('omat*.pkl')))

    if jax.process_index() == 0:
        print(f"Training files: {len(train_files)}")
        print(f"Validation files: {len(valid_files)}")

    # Run training with 2D parallelism enabled
    train_sharded(
        config,
        train_files,
        valid_files,
        use_2d_parallelism=False  # Enable hybrid DP + TP
    )
