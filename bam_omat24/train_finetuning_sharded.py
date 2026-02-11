"""NequIP/RACE multi-GPU training with true data sharding.

This version implements true data parallelism where each GPU processes
different batches simultaneously and gradients are averaged.

Both training AND evaluation are sharded across all GPUs.
"""

# ============================================================================
# DETERMINISM: Must be set BEFORE importing JAX
# ============================================================================
#import os
# Force XLA to use deterministic scatter/gather ops on GPU.
# Without this, segment_sum (used in GNN message passing) and
# other scatter-based ops are non-deterministic across runs.
#os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
# Force deterministic cuDNN algorithms
#os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from typing import Dict, Tuple, List, Any
import jax
import jax.numpy as jnp

from flax import nnx
import optax
import time
import numpy as np

from pathlib import Path

from bam_omat24.data_nnx import Dataset, BucketedDataLoader, MultiDeviceDataLoader
from bam_omat24.race_nnx import RACE
from bam_omat24.data_nnx import setup_mesh
from bam_omat24.training_utils import (
    replicate,
    replicate_pytree,
    unreplicate_pytree,
    make_sharded_train_step,
    evaluate_sharded,
    save_checkpoint,
    load_checkpoint,
)
from bam_omat24.finetuning_nnx import (
    FineTuningPotential,
    filter_delta_params,
)

jax.config.update("jax_enable_x64", False)
# Disable XLA autotuning cache (autotuning can pick different algorithms between runs)
#jax.config.update("jax_xla_profile_enabled", False)


# =============================================================================
# Main Training Loop
# =============================================================================

def train_finetuning_sharded(
    config: Dict,
    train_files_pkl: List[str],
    valid_files_pkl: List[str]
):
    """Main training fine-tuning function with true data sharding on all GPUs."""

    # Setup
    mesh, n_devices = setup_mesh()
    fout = open(config.get('fname_log', 'loss_finetuning_sharded.out'), 'w', 1)

    print("=" * 70, file=fout)
    print(f"Sharded Training on {n_devices} device(s)", file=fout)
    print(f"Effective batch size: {config['batch_size']} x {n_devices} = {config['batch_size'] * n_devices}", file=fout)
    print(f"Training AND evaluation are parallelized across all GPUs", file=fout)
    print("=" * 70, file=fout)

    # Create model - use SEPARATE rngs for model init and data loading
    seed = config.get('seed', 42)
    model_rngs = nnx.Rngs(seed)      # For model parameter initialization only
    rngs = nnx.Rngs(seed + 1)        # For data loading (shuffling) only

    foundation = RACE(
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
        rngs=model_rngs  # Dedicated rngs for model init
    )

    graphdef, params = nnx.split(foundation, nnx.Param)
    ckpt = load_checkpoint('ckpt_best.pkl')
    ema_params = ckpt['ema_params']
    foundation = nnx.merge (graphdef, ema_params)

    ft_model = FineTuningPotential(
        foundation_model=foundation,
        radial_mlp_size=config["radial_mlp_size"],
        radial_mlp_layers=config["radial_mlp_layers"],
        mlp_init_scale=config["mlp_init_scale"],
        delta_scale=1.0,
        delta_shift=0.0,
        strategy='delta_1', # [delta_0, delta_1, delta_2]
        freeze_foundation=True,
        rngs=rngs,
    )

    graphdef, params = nnx.split(ft_model, nnx.Param)
    ema_params = params
    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"   Total parameters: {n_params:,}")

    # Count trainable vs frozen
    trainable, frozen = filter_delta_params(params)
    n_trainable = sum(x.size for x in jax.tree.leaves(trainable))
    n_frozen = sum(x.size for x in jax.tree.leaves(frozen))
    print(f"   Trainable: {n_trainable:,}, Frozen: {n_frozen:,}")
    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"Model parameters: {n_params:,}", file=fout)

    # Optimizer
    #optimizer = create_optimizer_with_frozen_params (
    #    ft_model,
    #    learning_rate = config["learning_rate"]
    #)
    # Build trainable mask for selective EMA (skip frozen params)
    def is_freeze(path):
        path_str = str(path).lower()
        return not ('delta_' in path_str or 'feature_adapter' in path_str)

    freeze_mask = jax.tree_util.tree_map_with_path(
        lambda path, _: is_freeze(path),
        params
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.amsgrad(config["learning_rate"]),
        optax.transforms.freeze(freeze_mask),
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
    energy_weight=config.get('energy_weight', 1.0)
    force_weight=config.get('force_weight', 1.0)
    stress_weight=config.get('stress_weight', 1.0)
    huber_delta=config.get('huber_delta', 0.02)
    # Training step
    train_step = make_sharded_train_step(
        optimizer=optimizer,
        mesh=mesh,
        energy_weight=energy_weight,
        force_weight=force_weight,
        stress_weight=stress_weight,
        huber_delta=huber_delta,
        ema_decay=config.get('ema_decay', 0.99),
        freeze_mask=freeze_mask
    )

    # Load checkpoint if exists
    best_val_loss = float('inf')
    if config.get('restart', False) and Path('ckpt_ft_best.pkl').exists():
        ckpt = load_checkpoint('ckpt_ft_best.pkl')
        params = replicate_pytree(ckpt['params'], mesh)
        ema_params = replicate_pytree(ckpt['ema_params'], mesh)
        opt_state = replicate_pytree(ckpt['opt_state'], mesh)
        schedule_state = replicate_pytree(ckpt['schedule_state'], mesh)
        step = replicate(jnp.array(ckpt.get('step', 0)), mesh)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from step {ckpt.get('step', 0)}", file=fout)

    # Training loop
    best_state = None
    #n_val = 0

    for epoch in range(config["n_epochs"]):
        epoch_start = time.time()

        for ipkl, train_pkl in enumerate(train_files_pkl):
            train_dataset = Dataset(file_path=train_pkl)
            # Create multi-device dataloader for training
            # Use deterministic seed based on epoch + file index
            loader_rngs = nnx.Rngs(seed + 100 * epoch + ipkl)
            base_loader = BucketedDataLoader (
                dataset=train_dataset,
                batch_size=config['batch_size'],
                n_buckets=8,
                shuffle=True,
                drop_last=True,
                rngs=loader_rngs  # Deterministic per epoch + file
            )
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


            for batch, valid_device_count in loader:
                # Training step (sharded across all GPUs)
                params, opt_state, ema_params, step, metrics = train_step(
                    graphdef, params, opt_state, schedule_state, ema_params, step, batch
                )

                # Accumulate (metrics are already averaged across devices via pmean)
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
                    energy_loss = acc_energy_loss/max(acc_graphs,1)
                    force_loss = acc_force_loss/max(3*acc_atoms,1)
                    stress_loss = acc_stress_loss/max(acc_graphs,1)
                    total_loss = energy_weight*energy_loss + force_weight*force_loss + stress_weight*stress_loss
                    print(f"step {current_step} LOSS: {total_loss:.6f} E_LOSS: {energy_loss:.6f} "
                          f"F_LOSS: {force_loss:.6f}", file=fout)


            unrep_schedule = unreplicate_pytree(schedule_state)
            lr = config['learning_rate'] * float(unrep_schedule.scale)
            dataset_time = time.time() - dataset_start

            energy_loss = acc_energy_loss/max(acc_graphs,1)
            force_loss = acc_force_loss/max(3*acc_atoms,1)
            stress_loss = acc_stress_loss/max(acc_graphs,1)
            total_loss = energy_weight*energy_loss + force_weight*force_loss + stress_weight*stress_loss

            print(f"Epoch {epoch+1}/{config['n_epochs']} IPKL {ipkl+1} ({dataset_time:.1f}s) "
                  f"Step {current_step} LR: {lr:.2e}", file=fout)
            print(f"  Train | Epoch {epoch+1}/{ipkl+1} | Loss: {total_loss:.6f} | "
                  f"E_LOSS: {energy_loss:.6f} | "
                  f"F_LOSS: {force_loss:.6f} | "
                  f"S_LOSS: {stress_loss:.6f} | "
                  f"Grad: {acc_grad/max(acc_graphs,1):.4f}", file=fout)

            # Reset accumulators
            #acc_energy_loss, acc_force_loss, acc_stress_loss = 0., 0., 0.
            #acc_graphs, acc_atoms, acc_grad = 0., 0., 0.
            #dataset_start = time.time()


            # Validation (sharded across all GPUs)
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
                huber_delta=huber_delta
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
                save_checkpoint(best_state, 'ckpt_ft_best.pkl')
                print("  Checkpoint saved to ckpt_ft_best.pkl", file=fout)
            else:
                print(f"  No improvement for {new_schedule.plateau_count} evaluations "
                      f"(best: {best_val_loss:.6f})", file=fout)


        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{config['n_epochs']} completed in {epoch_time:.1f}s", file=fout)

    # Save final checkpoint
    if best_state is not None:
        save_checkpoint(best_state, 'ckpt_ft_best.pkl')
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

    print("=" * 70)
    print("JAX Sharded Training (Multi-GPU)")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Device count: {jax.local_device_count()}")
    print("=" * 70)

    # Load config
    with open('input_omat24.json') as f:
        config = json.load(f)
    config['atom_energies'] = ATOM_ENERGIES

    train_path = Path('/home/willow/Python/BAM-nequip-main/omat24_1M/train')
    valid_path = Path('/home/willow/Python/BAM-nequip-main/omat24_1M/val')

    train_files = natsorted(list(train_path.glob('omat*.pkl')))
    valid_files = natsorted(list(valid_path.glob('omat*.pkl')))

    #train_path = Path('/home/willow/Python/BAM-nequip-main/finetuning/train')
    #valid_path = Path('/home/willow/Python/BAM-nequip-main/finetuning/val')

    #train_files = natsorted(list(train_path.glob('ft*.pkl')))
    #valid_files = natsorted(list(valid_path.glob('ft*.pkl')))

    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(valid_files)}")

    train_finetuning_sharded(config, train_files, valid_files)
