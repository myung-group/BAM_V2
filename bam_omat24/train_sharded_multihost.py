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
from bam_omat24.train_sharded import (
    setup_mesh,
    replicate,
    replicate_pytree,
    unreplicate_pytree,
    make_sharded_train_step,
    evaluate_sharded,
    save_checkpoint,
    load_checkpoint,
)
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
# Main Training Loop
# =============================================================================

def train_sharded(
    config: Dict,
    train_files_pkl: List[str],
    valid_files_pkl: List[str],
):
    """Main training function with hybrid parallelism support.

    Args:
        config: Training configuration
        train_files_pkl: List of training data files
        valid_files_pkl: List of validation data files
    """
    # Setup mesh

    mesh, n_devices = setup_mesh()
    n_dp, n_tp = n_devices, 1

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
    print("=" * 70, file=fout)

    # Create model
    seed = config.get('seed', 42)
    model_rngs = nnx.Rngs(seed)

    model = RACE(
        n_species=len(config["atom_energies"]),
        lmax=config["lmax"],
        hidden_irreps=config["hidden_irreps"],
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

    step = replicate(jnp.array(0), mesh)
    energy_weight = config.get('energy_weight', 1.0)
    force_weight = config.get('force_weight', 1.0)
    stress_weight = config.get('stress_weight', 1.0)
    huber_delta = config.get('huber_delta', 0.02)

    # Create training step
    train_step = make_sharded_train_step(
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
        step = replicate(jnp.array(ckpt.get('step', 0)), mesh)
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
                if jax.process_index() == 0:
                    save_checkpoint(best_state, 'ckpt_best.pkl')
            else:
                print(f"  No improvement for {new_schedule.plateau_count} evaluations "
                      f"(best: {best_val_loss:.6f})", file=fout)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{config['n_epochs']} completed in {epoch_time:.1f}s", file=fout)

    # Save final checkpoint
    if best_state is not None:
        if jax.process_index() == 0:
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
        valid_files
    )
