"""Improved NequIP training script with scan-based training support.

This module provides efficient training using jax.lax.scan for epoch-level
optimization, reducing Python overhead and improving training throughput.
"""

from typing import Dict, Tuple, Optional, NamedTuple
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import pickle
import jraph
import time
import numpy as np

from bam_nequip.data import DataLoader, Dataset
from bam_nequip.nequip_nnx import Nequip
from bam_nequip.race_nnx import RACE

jax.config.update("jax_enable_x64", False)

# =============================================================================
# Training State Container
# =============================================================================

class TrainState(NamedTuple):
    """Container for training state used in scan-based training."""
    graphdef: nnx.GraphDef
    params: nnx.State
    opt_state: optax.OptState
    schedule_state: optax.contrib.ReduceLROnPlateauState
    ema_params: nnx.State
    step: jnp.ndarray
    rng_key: jax.Array


class Metrics(NamedTuple):
    """Container for training metrics."""
    loss: jnp.ndarray
    energy_loss: jnp.ndarray
    force_loss: jnp.ndarray
    energy_mse: jnp.ndarray
    force_mse: jnp.ndarray
    grad_norm: jnp.ndarray


# =============================================================================
# Loss Functions
# =============================================================================

def huber_loss(x: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    """Huber loss - robust to outliers."""
    abs_x = jnp.abs(x)
    return jnp.where(
        abs_x <= delta,
        0.5 * x ** 2,
        delta * (abs_x - 0.5 * delta)
    )


# =============================================================================
# Regular Training Step (for comparison and debugging)
# =============================================================================

def make_train_step(optimizer: optax.GradientTransformation,
                    force_weight: float = 1.0,
                    energy_weight: float = 1.0,
                    huber_delta: float = 0.1,
                    ema_decay: float = 0.99):
    """Create training step function with JIT-compiled inner computation."""

    @jax.jit
    def _compute_loss_and_grads(
        graphdef: nnx.GraphDef,
        params: nnx.State,
        batch: jraph.GraphsTuple
    ) -> Tuple[jnp.ndarray, Dict, nnx.State]:
        """JIT-compiled loss and gradient computation."""
        graph_mask = jraph.get_graph_padding_mask(batch)
        node_mask = jraph.get_node_padding_mask(batch)

        n_graphs = jnp.maximum(graph_mask.sum(), 1)
        n_atoms = node_mask.sum()

        def loss_fn(params):
            model = nnx.merge(graphdef, params)
            energy, forces = model(batch)

            energy_diff = energy - batch.globals["energy"]
            forces_diff = forces - batch.nodes["forces"]

            # Energy loss (per-atom normalized)
            energy_loss = huber_loss(energy_diff / batch.n_node, delta=huber_delta)
            energy_loss = jnp.sum(energy_loss * graph_mask) / n_graphs
            energy_mse = jnp.sum(energy_diff ** 2 * graph_mask) / n_graphs

            # Force loss
            force_loss = huber_loss(forces_diff, delta=huber_delta)
            force_loss = jnp.sum(force_loss * node_mask[:, None]) / (3 * n_atoms)
            force_mse = jnp.sum(forces_diff ** 2 * node_mask[:, None]) / (3 * n_atoms)

            # Total loss
            total_loss = energy_weight * energy_loss + force_weight * force_loss

            aux = {
                'energy_loss': energy_loss,
                'force_loss': force_loss,
                'energy_mse': energy_mse,
                'force_mse': force_mse,
            }
            return total_loss, aux

        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(params)

        # Track gradient norm
        grad_norm = optax.global_norm(grads)

        metrics = {
            'loss': loss,
            'energy_loss': aux['energy_loss'],
            'force_loss': aux['force_loss'],
            'energy_mse': aux['energy_mse'],
            'force_mse': aux['force_mse'],
            'grad_norm': grad_norm,
        }

        return loss, metrics, grads

    def train_step(carry, batch: jraph.GraphsTuple):
        """Single training step."""
        graphdef, params, opt_state, schedule_state, ema_params, step = carry
        loss, metrics, grads = _compute_loss_and_grads(graphdef, params, batch)

        # Update optimizer state
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        updates = optax.tree_utils.tree_scale(schedule_state.scale, updates)
        new_params = optax.apply_updates(params, updates)

        # EMA update
        new_ema_params = jax.tree.map(
            lambda ema, p: ema_decay * ema + (1 - ema_decay) * p,
            ema_params, new_params
        )

        new_carry = (graphdef, new_params, new_opt_state, schedule_state, new_ema_params, step + 1)

        return new_carry, metrics

    return train_step


# =============================================================================
# Scan-based Training Step
# =============================================================================

def make_scan_train_step(optimizer: optax.GradientTransformation,
                         force_weight: float = 1.0,
                         energy_weight: float = 1.0,
                         huber_delta: float = 0.5,
                         ema_decay: float = 0.99,
                         grad_clip: Optional[float] = None):
    """Create a scan-compatible training step function.

    Returns a JIT-compiled function that can be used with jax.lax.scan
    for efficient epoch-level training.

    Args:
        optimizer: Optax optimizer
        force_weight: Weight for force loss
        energy_weight: Weight for energy loss
        huber_delta: Delta parameter for Huber loss
        ema_decay: Exponential moving average decay for parameters
        grad_clip: Optional gradient clipping value (max norm)

    Returns:
        A JIT-compiled scan step function
    """

    def compute_loss_and_grads(
        graphdef: nnx.GraphDef,
        params: nnx.State,
        batch: jraph.GraphsTuple
    ) -> Tuple[jnp.ndarray, Metrics, nnx.State]:
        """Loss and gradient computation."""

        graph_mask = jraph.get_graph_padding_mask(batch)
        node_mask = jraph.get_node_padding_mask(batch)

        n_graphs = jnp.maximum(graph_mask.sum(), 1)
        n_atoms = jnp.maximum(node_mask.sum(), 1)

        def loss_fn(params):
            model = nnx.merge(graphdef, params)
            energy, forces = model(batch)

            energy_diff = energy - batch.globals["energy"]
            forces_diff = forces - batch.nodes["forces"]

            # Energy loss (per-atom normalized)
            energy_loss = huber_loss(energy_diff / batch.n_node, delta=huber_delta)
            energy_loss = jnp.sum(energy_loss * graph_mask) / n_graphs
            energy_mse = jnp.sum(energy_diff ** 2 * graph_mask) / n_graphs

            # Force loss
            force_loss = huber_loss(forces_diff, delta=huber_delta)
            force_loss = jnp.sum(force_loss * node_mask[:, None]) / (3 * n_atoms)
            force_mse = jnp.sum(forces_diff ** 2 * node_mask[:, None]) / (3 * n_atoms)

            # Total loss
            total_loss = energy_weight * energy_loss + force_weight * force_loss

            return total_loss, (energy_loss, force_loss, energy_mse, force_mse)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (energy_loss, force_loss, energy_mse, force_mse)), grads = grad_fn(params)

        # Optional gradient clipping
        if grad_clip is not None:
            grads = optax.clip_by_global_norm(grad_clip).update(grads, None)[0]

        grad_norm = optax.global_norm(grads)

        metrics = Metrics(
            loss=loss,
            energy_loss=energy_loss,
            force_loss=force_loss,
            energy_mse=energy_mse,
            force_mse=force_mse,
            grad_norm=grad_norm,
        )

        return loss, metrics, grads

    @jax.jit
    def scan_train_step(carry: TrainState, graph: jraph.GraphsTuple) -> Tuple[TrainState, Metrics]:
        """Single training step compatible with jax.lax.scan.

        Args:
            carry: TrainState containing model state and optimizer state
            graph: Single batched graph

        Returns:
            new_carry: Updated TrainState
            metrics: Metrics for this step
        """
        # Compute loss and gradients
        loss, metrics, grads = compute_loss_and_grads(carry.graphdef, carry.params, graph)

        # Update optimizer state
        updates, new_opt_state = optimizer.update(grads, carry.opt_state, carry.params)
        updates = optax.tree_utils.tree_scale(carry.schedule_state.scale, updates)
        new_params = optax.apply_updates(carry.params, updates)

        # EMA update
        new_ema_params = jax.tree.map(
            lambda ema, p: ema_decay * ema + (1 - ema_decay) * p,
            carry.ema_params, new_params
        )

        new_carry = TrainState(
            graphdef=carry.graphdef,
            params=new_params,
            opt_state=new_opt_state,
            schedule_state=carry.schedule_state,
            ema_params=new_ema_params,
            step=carry.step + 1,
            rng_key=carry.rng_key,
        )

        return new_carry, metrics

    return scan_train_step


def make_scan_train_epoch(optimizer: optax.GradientTransformation,
                          force_weight: float = 1.0,
                          energy_weight: float = 1.0,
                          huber_delta: float = 0.5,
                          ema_decay: float = 0.99,
                          grad_clip: Optional[float] = None):
    """Create a function that trains for one full epoch using jax.lax.scan.

    This is more efficient than calling train_step in a Python loop because
    the entire epoch is compiled as a single XLA computation, reducing
    dispatch overhead.

    Note: For small models/batches, the regular Python loop may be faster
    due to lower compilation overhead. Use this for:
    - Large models where compilation cost is amortized
    - Long training runs where consistency matters
    - When you need deterministic training behavior

    Returns:
        A tuple of (scan_step_fn, train_one_epoch_fn) where:
        - scan_step_fn: Single step function for manual control
        - train_one_epoch_fn: Full epoch function with shuffle
    """
    scan_step = make_scan_train_step(
        optimizer=optimizer,
        force_weight=force_weight,
        energy_weight=energy_weight,
        huber_delta=huber_delta,
        ema_decay=ema_decay,
        grad_clip=grad_clip,
    )

    @jax.jit
    def _run_scan(state: TrainState, data: jraph.GraphsTuple) -> Tuple[TrainState, Metrics]:
        """Run scan over pre-shuffled data."""
        return jax.lax.scan(scan_step, state, data)

    def train_one_epoch(state: TrainState, stacked_data: jraph.GraphsTuple) -> Tuple[TrainState, Metrics]:
        """Train for one epoch using scan.

        Args:
            state: Current training state
            stacked_data: Stacked graphs with shape (n_batches, ...)

        Returns:
            new_state: Updated training state
            avg_metrics: Averaged metrics over the epoch
        """
        n_batches = stacked_data.n_node.shape[0]

        # Shuffle outside JIT for efficiency (avoid recompilation)
        new_key, shuffle_key = jax.random.split(state.rng_key)
        perm = jax.random.permutation(shuffle_key, n_batches)

        # Apply permutation
        shuffled_data = jax.tree.map(
            lambda x: x[perm] if x is not None else None,
            stacked_data
        )

        # Update state with new RNG key
        state = state._replace(rng_key=new_key)

        # Run scan (this is JIT-compiled)
        final_state, all_metrics = _run_scan(state, shuffled_data)

        # Average metrics
        avg_metrics = jax.tree.map(lambda x: jnp.mean(x), all_metrics)

        return final_state, avg_metrics

    return train_one_epoch


# =============================================================================
# Evaluation
# =============================================================================

@jax.jit
def evaluate_batch(graphdef: nnx.GraphDef,
                   params: nnx.State,
                   batch: jraph.GraphsTuple,
                   energy_weight: float = 1.0,
                   force_weight: float = 1.0,
                   huber_delta: float = 0.1) -> Dict:
    """Evaluate a single batch."""
    model = nnx.merge(graphdef, params)

    graph_mask = jraph.get_graph_padding_mask(batch)
    node_mask = jraph.get_node_padding_mask(batch)
    n_graphs = jnp.maximum(graph_mask.sum(), 1)
    n_atoms = jnp.maximum(node_mask.sum(), 1)

    energy, forces = model(batch)

    energy_diff = energy - batch.globals["energy"]
    forces_diff = forces - batch.nodes["forces"]

    # Energy metrics
    energy_loss = huber_loss(energy_diff / batch.n_node, delta=huber_delta)
    energy_loss = jnp.sum(energy_loss * graph_mask)
    energy_se = jnp.sum(energy_diff ** 2 * graph_mask)
    energy_ae = jnp.sum(jnp.abs(energy_diff) * graph_mask)

    # Force metrics
    force_loss = huber_loss(forces_diff, delta=huber_delta)
    force_loss = jnp.sum(force_loss * node_mask[:, None])
    force_se = jnp.sum(forces_diff ** 2 * node_mask[:, None])
    force_ae = jnp.sum(jnp.abs(forces_diff) * node_mask[:, None])

    # Total loss
    total_loss = energy_weight * energy_loss / n_graphs + force_weight * force_loss / (3 * n_atoms)

    return {
        'total_loss': total_loss,
        'energy_loss': energy_loss,
        'force_loss': force_loss,
        'energy_se': energy_se,
        'force_se': force_se,
        'energy_ae': energy_ae,
        'force_ae': force_ae,
        'n_graphs': n_graphs,
        'n_atoms': n_atoms,
    }


def evaluate(graphdef: nnx.GraphDef,
             params: nnx.State,
             dataloader: DataLoader,
             energy_weight: float = 1.0,
             force_weight: float = 1.0,
             huber_delta: float = 0.1) -> Dict:
    """Evaluate model on a dataset."""
    total_loss = 0.0
    total_energy_se = 0.0
    total_force_se = 0.0
    total_energy_ae = 0.0
    total_force_ae = 0.0
    total_graphs = 0.0
    total_atoms = 0.0

    for batch in dataloader:
        metrics = evaluate_batch(
            graphdef, params, batch,
            energy_weight, force_weight, huber_delta
        )

        n_graphs = float(metrics['n_graphs'])
        n_atoms = float(metrics['n_atoms'])

        total_loss += float(metrics['total_loss']) * n_graphs
        total_energy_se += float(metrics['energy_se'])
        total_force_se += float(metrics['force_se'])
        total_energy_ae += float(metrics['energy_ae'])
        total_force_ae += float(metrics['force_ae'])
        total_graphs += n_graphs
        total_atoms += n_atoms

    return {
        'total_loss': total_loss / total_graphs,
        'energy_mae': total_energy_ae / total_graphs,
        'energy_rmse': np.sqrt(total_energy_se / total_graphs),
        'force_mae': total_force_ae / (3 * total_atoms),
        'force_rmse': np.sqrt(total_force_se / (3 * total_atoms)),
    }


# =============================================================================
# Data Stacking for Scan
# =============================================================================

def stack_graphs(loader: DataLoader) -> jraph.GraphsTuple:
    """Stack all batches from a DataLoader into a single structure for scan.

    This pre-processes all batches so they can be efficiently iterated
    using jax.lax.scan instead of a Python for loop.

    Args:
        loader: DataLoader to stack

    Returns:
        Stacked GraphsTuple with shape (n_batches, ...) for each field
    """
    batches = list(loader)

    def stack_field(field_name, accessor):
        values = [accessor(g) for g in batches]
        if values[0] is None:
            return None
        if isinstance(values[0], dict):
            return {k: jnp.stack([v[k] for v in values]) for k in values[0].keys()}
        return jnp.stack(values)

    return jraph.GraphsTuple(
        nodes=stack_field('nodes', lambda g: g.nodes),
        edges=stack_field('edges', lambda g: g.edges),
        receivers=jnp.stack([g.receivers for g in batches]),
        senders=jnp.stack([g.senders for g in batches]),
        globals=stack_field('globals', lambda g: g.globals),
        n_node=jnp.stack([g.n_node for g in batches]),
        n_edge=jnp.stack([g.n_edge for g in batches]),
    )


# =============================================================================
# Main Training Function
# =============================================================================

def train(config: Dict):
    """Main training function supporting both regular and scan-based training."""

    # Setup logging
    fout = open(config.get('fname_log', 'loss.out'), 'w', 1)
    use_scan = config.get('use_scan', True)

    print(f"Training mode: {'scan-based' if use_scan else 'regular'}", file=fout)
    print("=" * 70, file=fout)

    # Load datasets
    atom_energies = config["atom_energies"]
    train_dataset = Dataset(
        file_path=config["train_path"],
        atomic_numbers=config["atomic_numbers"],
        split="train",
        cutoff=config["cutoff"],
        valid_frac=config["valid_frac"],
    )
    val_dataset = Dataset(
        file_path=config["train_path"],
        atomic_numbers=config["atomic_numbers"],
        split="val",
        cutoff=config["cutoff"],
        valid_frac=config["valid_frac"],
    )
    test_dataset = Dataset(
        file_path=config["test_path"],
        atomic_numbers=config["atomic_numbers"],
        cutoff=config["cutoff"],
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        seed=42,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        seed=42,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        seed=42,
        shuffle=False,
        drop_last=False,
    )


    # Compute training statistics
    #shift, scale, n_neighbors = train_stats(train_loader, atom_energies)
    shift = 0.0
    scale = 1.0
    n_neighbors = 25.0
    print(f"Data statistics:", file=fout)
    print(f"  shift: {shift:.6f}", file=fout)
    print(f"  scale: {scale:.6f}", file=fout)
    print(f"  avg_n_neighbors: {n_neighbors:.2f}", file=fout)

    # Create model
    rngs = nnx.Rngs(config.get('seed', 42))
    model = RACE(
        n_species=len(config["atomic_numbers"]),
        hidden_size=config["hidden_size"],
        lmax=config["lmax"],
        n_layers=config["n_layers"],
        radial_basis_size=config["radial_basis_size"],
        radial_mlp_size=config["radial_mlp_size"],
        radial_mlp_layers=config["radial_mlp_layers"],
        radial_polynomial_p=config["radial_polynomial_p"],
        mlp_init_scale=config["mlp_init_scale"],
        shift=shift,
        scale=scale,
        avg_n_neighbors=n_neighbors,
        atom_energies=atom_energies,
        rngs=rngs
    )

    graphdef, params = nnx.split(model, nnx.Param)
    ema_params = params
    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"Model parameters: {n_params:,}", file=fout)
    print(file=fout)

    # Setup optimizer
    optim = optax.amsgrad(config["learning_rate"])
    lr_schedule_patience = config["lr_schedule_patience"]
    schedule = optax.contrib.reduce_on_plateau(
        factor=config["lr_schedule_factor"],
        patience=config["lr_schedule_patience"],
        min_scale=config["lr_schedule_min_scale"],
    )

    opt_state = optim.init(params)
    schedule_state = schedule.init(params)
    # Initialize training state
    rng_key = jax.random.key(config.get('seed', 42))
    train_state = TrainState(
            graphdef=graphdef,
            params=params,
            opt_state=opt_state,
            schedule_state=schedule_state,
            ema_params=ema_params,
            step=jnp.array(0),
            rng_key=rng_key,
        )

    # Stack graphs for scan-based training
    if use_scan:
        print("Stacking graphs for scan-based training...", file=fout)
        stacked_train = stack_graphs(train_loader)
        n_batches = stacked_train.n_node.shape[0]
        print(f"  Number of batches: {n_batches}", file=fout)
        print(f"  Stacked n_node shape: {stacked_train.n_node.shape}", file=fout)

        # Create scan-based epoch trainer
        train_one_epoch = make_scan_train_epoch(
            optimizer=optim,
            force_weight=config.get('force_weight', 1.0),
            energy_weight=config.get('energy_weight', 1.0),
            huber_delta=config.get('huber_delta', 0.1),
            ema_decay=config.get('ema_decay', 0.99),
            grad_clip=config.get('grad_clip', None),
        )

    else:
        # Create regular train step
        train_step = make_train_step(
            optimizer=optim,
            force_weight=config.get('force_weight', 1.0),
            energy_weight=config.get('energy_weight', 1.0),
            huber_delta=config.get('huber_delta', 0.1),
            ema_decay=config.get('ema_decay', 0.99)
        )
        step = jnp.array(0)

    # Training loop
    best_val_loss = float("inf")
    n_epochs = config["n_epochs"]
    if config.get('restart', False) and Path('ckpt_best.pkl').exists():
        print("Loading checkpoint...", file=fout)
        best_val_loss, train_state = load_checkpoint(train_state, 'ckpt_best.pkl')
        print(f"  Resumed from step {train_state.step}", file=fout)
        params = train_state.params
        ema_params = train_state.ema_params
        step = train_state.step
        opt_state = train_state.opt_state
        schedule_state = train_state.schedule_state

    print(f"\nStarting training for {n_epochs} epochs...", file=fout)
    print("=" * 70, file=fout)

    best_state = train_state
    ema_params = best_state.ema_params
    l_best_save = False

    test_metrics = evaluate(
            graphdef, ema_params, test_loader,
            energy_weight=config["energy_weight"],
            force_weight=config["force_weight"],
            huber_delta=config.get('huber_delta', 0.1)
        )

    test_loss = test_metrics["total_loss"]
    print(f"  Test | Loss: {test_loss:.8f} | "
              f"E_MAE:  {test_metrics['energy_mae']:.6f} | "
              f"F_MAE:  {test_metrics['force_mae']:.6f} | "
              f"E_RMSE: {test_metrics['energy_rmse']:.6f} | "
              f"F_RMSE: {test_metrics['force_rmse']:.6f}", file=fout)


    for epoch in range(n_epochs):
        epoch_start = time.time()

        if use_scan:
            # Scan-based training (entire epoch in one JIT call)
            train_state, train_metrics = train_one_epoch(train_state, stacked_train)

            # Extract metrics
            train_loss = float(train_metrics.loss)
            energy_rmse = float(jnp.sqrt(train_metrics.energy_mse))
            force_rmse = float(jnp.sqrt(train_metrics.force_mse))
            grad_norm = float(train_metrics.grad_norm)

            # Update references for validation and checkpointing
            graphdef = train_state.graphdef
            params = train_state.params
            ema_params = train_state.ema_params
            schedule_state = train_state.schedule_state
            current_step = train_state.step

        else:
            # Regular training (Python loop over batches)
            train_loss, energy_mse, force_mse, total_count, grad_norm = 0, 0, 0, 0, 0
            for batch in train_loader:
                graph_mask = jraph.get_graph_padding_mask(batch)
                n_graphs = jnp.maximum(graph_mask.sum(), 1)
                carry = (graphdef, params, opt_state, schedule_state, ema_params, step)
                new_carry, train_metrics = train_step(carry, batch)
                graphdef, params, opt_state, schedule_state, ema_params, step = new_carry
                train_loss += train_metrics['loss'] * n_graphs
                energy_mse += train_metrics['energy_mse'] * n_graphs
                force_mse += train_metrics['force_mse'] * n_graphs
                total_count += n_graphs
                grad_norm += train_metrics['grad_norm'] * n_graphs

            train_loss = train_loss / total_count
            energy_rmse = jnp.sqrt(energy_mse / total_count)
            force_rmse = jnp.sqrt(force_mse / total_count)
            grad_norm = grad_norm / total_count
            current_step = step
            train_state = TrainState(
                graphdef=graphdef,
                params=params,
                opt_state=opt_state,
                schedule_state=schedule_state,
                ema_params=ema_params,
                step=current_step,
                rng_key=rng_key,
            )
        epoch_time = time.time() - epoch_start

        # Validation
        val_metrics = evaluate(
            graphdef, ema_params, val_loader,
            energy_weight=config["energy_weight"],
            force_weight=config["force_weight"],
            huber_delta=config.get('huber_delta', 0.1)
        )

        val_loss = val_metrics["total_loss"]
        current_lr = config['learning_rate'] * schedule_state.scale

        # Logging
        print(f"Epoch {epoch + 1:4d}/{n_epochs} ({epoch_time:.1f}s) | "
              f"Step: {int(current_step)} | LR: {current_lr:.2e}", file=fout)
        print(f"  Train | Loss: {train_loss:.8f} | "
              f"E_RMSE: {energy_rmse:.6f} | "
              f"F_RMSE: {force_rmse:.6f} | "
              f"Grad: {grad_norm:.4f}", file=fout)
        print(f"  Valid | Loss: {val_loss:.8f} | "
              f"E_MAE:  {val_metrics['energy_mae']:.6f} | "
              f"F_MAE:  {val_metrics['force_mae']:.6f} | "
              f"E_RMSE: {val_metrics['energy_rmse']:.6f} | "
              f"F_RMSE: {val_metrics['force_rmse']:.6f}", file=fout)

        # Update learning rate schedule
        _, new_schedule_state = schedule.update(
            updates=params,
            state=schedule_state,
            value=val_loss,
        )

        # Update schedule state in train_state if using scan
        if use_scan:
            train_state = train_state._replace(schedule_state=new_schedule_state)
        else:
            schedule_state = new_schedule_state

        # Save best model
        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            best_state = train_state
            l_best_save = True
            print(f"  *** New best val loss: {best_val_loss:.8f} "
                  f"(improved by {improvement:.8f}) ***", file=fout)
        else:
            print(f"  No improvement for {new_schedule_state.plateau_count} evaluations "
                  f"(best: {best_val_loss:.8f})", file=fout)


        if (epoch+1)%5 == 0 and l_best_save:
            save_checkpoint(best_state, best_val_loss, 'ckpt_best.pkl', val_metrics)
            print(f"  Checkpoint saved to ckpt_best.pkl", file=fout)
            l_best_save = False

        if (epoch+1)%100 == 0:
            test_metrics = evaluate(
                graphdef, ema_params, test_loader,
                energy_weight=config["energy_weight"],
                force_weight=config["force_weight"],
                huber_delta=config.get('huber_delta', 0.1)
            )

            test_loss = test_metrics["total_loss"]
            print(f"  Test | Loss: {test_loss:.8f} | "
              f"E_MAE:  {test_metrics['energy_mae']:.6f} | "
              f"F_MAE:  {test_metrics['force_mae']:.6f} | "
              f"E_RMSE: {test_metrics['energy_rmse']:.6f} | "
              f"F_RMSE: {test_metrics['force_rmse']:.6f}", file=fout)


    print("\nTraining completed!", file=fout)
    fout.close()


# =============================================================================
# Gradient Accumulation Support
# =============================================================================

def make_scan_train_step_with_accumulation(
    optimizer: optax.GradientTransformation,
    accumulation_steps: int = 1,
    force_weight: float = 1.0,
    energy_weight: float = 1.0,
    huber_delta: float = 0.5,
    ema_decay: float = 0.99,
    grad_clip: Optional[float] = None
):
    """Create a scan-compatible training step with gradient accumulation.

    This allows effective batch sizes larger than what fits in memory by
    accumulating gradients over multiple mini-batches before updating.

    Args:
        optimizer: Optax optimizer
        accumulation_steps: Number of mini-batches to accumulate before update
        force_weight: Weight for force loss
        energy_weight: Weight for energy loss
        huber_delta: Delta parameter for Huber loss
        ema_decay: EMA decay for parameters
        grad_clip: Optional gradient clipping value

    Returns:
        A function that processes accumulation_steps batches per call
    """

    def compute_loss_and_grads(graphdef, params, batch):
        """Compute loss and gradients for a single batch."""
        graph_mask = jraph.get_graph_padding_mask(batch)
        node_mask = jraph.get_node_padding_mask(batch)
        n_graphs = jnp.maximum(graph_mask.sum(), 1)
        n_atoms = jnp.maximum(node_mask.sum(), 1)

        def loss_fn(params):
            model = nnx.merge(graphdef, params)
            energy, forces = model(batch)

            energy_diff = energy - batch.globals["energy"]
            forces_diff = forces - batch.nodes["forces"]

            energy_loss = huber_loss(energy_diff / batch.n_node, delta=huber_delta)
            energy_loss = jnp.sum(energy_loss * graph_mask) / n_graphs
            energy_mse = jnp.sum(energy_diff ** 2 * graph_mask) / n_graphs

            force_loss = huber_loss(forces_diff, delta=huber_delta)
            force_loss = jnp.sum(force_loss * node_mask[:, None]) / (3 * n_atoms)
            force_mse = jnp.sum(forces_diff ** 2 * node_mask[:, None]) / (3 * n_atoms)

            total_loss = energy_weight * energy_loss + force_weight * force_loss
            return total_loss, (energy_loss, force_loss, energy_mse, force_mse)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(params)
        return loss, aux, grads

    @jax.jit
    def accumulated_train_step(state: TrainState, batches: jraph.GraphsTuple) -> Tuple[TrainState, Metrics]:
        """Process multiple batches with gradient accumulation.

        Args:
            state: Current training state
            batches: Stacked batches of shape (accumulation_steps, ...)

        Returns:
            new_state: Updated state after accumulation_steps
            metrics: Averaged metrics over the accumulated batches
        """

        def accumulate_step(carry, batch):
            """Accumulate gradients for one batch."""
            accumulated_grads, total_loss, total_aux = carry
            loss, aux, grads = compute_loss_and_grads(state.graphdef, state.params, batch)

            # Accumulate gradients
            new_accumulated = jax.tree.map(
                lambda acc, g: acc + g / accumulation_steps,
                accumulated_grads, grads
            )

            new_aux = tuple(t + a / accumulation_steps for t, a in zip(total_aux, aux))
            return (new_accumulated, total_loss + loss / accumulation_steps, new_aux), None

        # Initialize accumulator with zeros
        zero_grads = jax.tree.map(jnp.zeros_like, state.params)
        zero_aux = (0.0, 0.0, 0.0, 0.0)

        (accumulated_grads, avg_loss, avg_aux), _ = jax.lax.scan(
            accumulate_step,
            (zero_grads, 0.0, zero_aux),
            batches
        )

        # Apply gradient clipping if specified
        if grad_clip is not None:
            accumulated_grads = optax.clip_by_global_norm(grad_clip).update(
                accumulated_grads, None
            )[0]

        # Update parameters
        updates, new_opt_state = optimizer.update(
            accumulated_grads, state.opt_state, state.params
        )
        updates = optax.tree_utils.tree_scale(state.schedule_state.scale, updates)
        new_params = optax.apply_updates(state.params, updates)

        # EMA update
        new_ema_params = jax.tree.map(
            lambda ema, p: ema_decay * ema + (1 - ema_decay) * p,
            state.ema_params, new_params
        )

        new_state = TrainState(
            graphdef=state.graphdef,
            params=new_params,
            opt_state=new_opt_state,
            schedule_state=state.schedule_state,
            ema_params=new_ema_params,
            step=state.step + accumulation_steps,
            rng_key=state.rng_key,
        )

        energy_loss, force_loss, energy_mse, force_mse = avg_aux
        metrics = Metrics(
            loss=avg_loss,
            energy_loss=energy_loss,
            force_loss=force_loss,
            energy_mse=energy_mse,
            force_mse=force_mse,
            grad_norm=optax.global_norm(accumulated_grads),
        )

        return new_state, metrics

    return accumulated_train_step


# =============================================================================
# Checkpoint Management
# =============================================================================

def save_checkpoint(state: TrainState, best_val_loss: float, path: str, metrics: Dict = None):
    """Save checkpoint with metrics and graphdef for architecture reproducibility."""
    ckpt = {
        'params': state.params,
        'opt_state': state.opt_state,
        'schedule_state': state.schedule_state,
        'ema_params': state.ema_params,
        'step': state.step,
        'rng_key': state.rng_key,
        'best_val_loss': best_val_loss
    }
    if metrics is not None:
        ckpt['metrics'] = metrics

    with open(path, 'wb') as fp:
        pickle.dump(ckpt, fp)



def load_checkpoint(state: TrainState, path: str, strict: bool = True) -> TrainState:
    """Load checkpoint.

    Args:
        state: Current TrainState (used for graphdef if not in checkpoint)
        path: Path to checkpoint file
        strict: If True, warn when graphdef in checkpoint differs from current model

    Returns:
        TrainState with loaded parameters
    """
    with open(path, 'rb') as fp:
        ckpt = pickle.load(fp)

    return ckpt.get('best_val_loss', float('inf')), \
        TrainState(
        graphdef=state.graphdef,
        params=ckpt['params'],
        opt_state=ckpt['opt_state'],
        schedule_state=ckpt['schedule_state'],
        ema_params=ckpt['ema_params'],
        step=ckpt.get('step', 0),
        rng_key=ckpt['rng_key']
        )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import json
    from pathlib import Path

    config_path = Path('input.json')
    if config_path.exists():
        print(f"Loading config from {config_path}")
        with open(config_path) as f:
            config = json.load(f)
            train(config)
    else:
        print(f"Config file {config_path} not found.")
        print("Creating example config...")

        example_config = {
            "train_path": "data/train.xyz",
            "atomic_numbers": [1, 6, 7, 8],
            "atom_energies": [-13.587, -1029.489, -1484.981, -2041.982],
            "cutoff": 5.0,
            "valid_frac": 0.1,
            "batch_size": 4,
            "hidden_size": 128,
            "lmax": 2,
            "n_layers": 4,
            "radial_basis_size": 8,
            "radial_mlp_size": 64,
            "radial_mlp_layers": 3,
            "radial_polynomial_p": 6.0,
            "mlp_init_scale": 4.0,
            "learning_rate": 1e-3,
            "lr_schedule_factor": 0.8,
            "lr_schedule_patience": 20,
            "n_epochs": 100,
            "energy_weight": 1.0,
            "force_weight": 100.0,
            "huber_delta": 0.1,
            "ema_decay": 0.99,
            "seed": 42,
            "use_scan": True,  # Enable scan-based training
            "grad_clip": 1.0,  # Optional gradient clipping
        }

        with open('input_nequip_example.json', 'w') as f:
            json.dump(example_config, f, indent=2)
        print(f"Example config saved to input_nequip_example.json")
