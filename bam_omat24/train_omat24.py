"""Unified OMAT24 training script.

Two independent axes of configuration:

  config['mode']      - What to train:
    'pretrain'  (default) : RACE or Nequip pretraining from scratch
    'finetune'            : Delta fine-tuning with a frozen foundation model

  config['multihost'] - Hardware environment (default: false):
    false : Single-node (one or more GPUs on one machine)
    true  : Multi-node SLURM / distributed (jax.distributed)

All four combinations are supported:
  mode=pretrain,  multihost=false  →  standard single-node pretraining
  mode=pretrain,  multihost=true   →  distributed multi-node pretraining
  mode=finetune,  multihost=false  →  standard single-node fine-tuning
  mode=finetune,  multihost=true   →  distributed multi-node fine-tuning

Usage:
    python train_omat24.py          (reads input_omat24.json)
    srun python train_omat24.py     (multihost=true, SLURM)
"""

# ============================================================================
# DETERMINISM: Set BEFORE importing JAX if needed
# ============================================================================
import os
# os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
# os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from typing import Dict, List
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import time
import numpy as np
from pathlib import Path

from bam_omat24.data_nnx import Dataset, BucketedDataLoader, MultiDeviceDataLoader, setup_mesh
from bam_omat24.nequip_nnx import Nequip
from bam_omat24.race_nnx import RACE
from bam_omat24.training_utils import (
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
# Multi-Host Distributed Initialization  (multihost=true only)
# =============================================================================

def initialize_distributed():
    """Initialize JAX distributed runtime for multi-host training.

    Handles both SLURM-managed and manual setups:
    - SLURM : srun sets environment variables automatically.
    - Manual : Set JAX_COORDINATOR_ADDRESS, JAX_NUM_PROCESSES, JAX_PROCESS_INDEX.
    """
    if 'SLURM_PROCID' in os.environ:
        process_id    = int(os.environ['SLURM_PROCID'])
        num_processes = int(os.environ['SLURM_NPROCS'])

        coordinator_address = os.environ.get('JAX_COORDINATOR_ADDRESS')
        if coordinator_address is None:
            import subprocess
            result = subprocess.run(
                ['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']],
                capture_output=True, text=True,
            )
            head_node           = result.stdout.strip().split('\n')[0]
            coordinator_address = f"{head_node}:29500"

        print(f"[Process {process_id}/{num_processes}] Coordinator: {coordinator_address}")
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=num_processes,
            process_id=process_id,
        )
    else:
        # Works for pre-configured environments (JAX_* env vars already set)
        jax.distributed.initialize()

    print(f"Process {jax.process_index()} of {jax.process_count()}")
    print(f"Local devices : {jax.local_device_count()}")
    print(f"Total devices : {jax.device_count()}")


# =============================================================================
# Model / Optimizer Setup Helpers
# =============================================================================

def _build_pretrain_components(config: Dict, mesh, n_devices: int, fout):
    """Build model, optimizer, and train_step for pretrain mode."""
    seed       = config.get('seed', 42)
    model_rngs = nnx.Rngs(seed)

    model_name   = config.get('model', 'RACE').upper()
    model_kwargs = dict(
        n_species           = len(config["atom_energies"]),
        lmax                = config["lmax"],
        hidden_irreps       = config["hidden_irreps"],
        n_layers            = config["n_layers"],
        radial_basis_size   = config["radial_basis_size"],
        radial_mlp_size     = config["radial_mlp_size"],
        radial_mlp_layers   = config["radial_mlp_layers"],
        radial_polynomial_p = config["radial_polynomial_p"],
        mlp_init_scale      = config["mlp_init_scale"],
        shift               = 0.0,
        scale               = 1.0,
        avg_n_neighbors     = 25.0,
        atom_energies       = config["atom_energies"],
        l_train             = True,
        rngs                = model_rngs,
    )

    model = RACE(**model_kwargs) if model_name == 'RACE' else Nequip(**model_kwargs)

    graphdef, params = nnx.split(model, nnx.Param)
    ema_params       = params

    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"Model parameters: {n_params:,}", file=fout)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.amsgrad(config["learning_rate"]),
    )
    schedule = optax.contrib.reduce_on_plateau(
        factor    = config["lr_schedule_factor"],
        patience  = config["lr_schedule_patience"],
        min_scale = config["lr_schedule_min_scale"],
    )

    energy_weight = config.get('energy_weight', 1.0)
    force_weight  = config.get('force_weight',  1.0)
    stress_weight = config.get('stress_weight', 1.0)
    huber_delta   = config.get('huber_delta',   0.02)
    loss_type     = config.get('loss_type',     'huber')

    train_step = make_sharded_train_step(
        optimizer     = optimizer,
        mesh          = mesh,
        energy_weight = energy_weight,
        force_weight  = force_weight,
        stress_weight = stress_weight,
        huber_delta   = huber_delta,
        ema_decay     = config.get('ema_decay', 0.99),
        loss_type     = loss_type,
    )

    return (graphdef, params, ema_params, optimizer, schedule,
            train_step, energy_weight, force_weight, stress_weight, huber_delta)


def _build_finetune_components(config: Dict, mesh, n_devices: int, fout):
    """Build model, optimizer, and train_step for finetune mode."""
    from bam_omat24.finetuning_nnx import FineTuningPotential, filter_ft_params

    seed       = config.get('seed', 42)
    model_rngs = nnx.Rngs(seed)
    ft_rngs    = nnx.Rngs(seed + 1)

    # ── Load pretrained foundation ────────────────────────────────────────────
    model_name   = config.get('model', 'RACE').upper()
    model_kwargs = dict(
        n_species           = len(config["atom_energies"]),
        lmax                = config["lmax"],
        hidden_irreps       = config["hidden_irreps"],
        n_layers            = config["n_layers"],
        radial_basis_size   = config["radial_basis_size"],
        radial_mlp_size     = config["radial_mlp_size"],
        radial_mlp_layers   = config["radial_mlp_layers"],
        radial_polynomial_p = config["radial_polynomial_p"],
        mlp_init_scale      = config["mlp_init_scale"],
        shift               = 0.0,
        scale               = 1.0,
        avg_n_neighbors     = 25.0,
        atom_energies       = config["atom_energies"],
        l_train             = True,
        rngs                = model_rngs,
    )

    foundation = RACE(**model_kwargs) if model_name == 'RACE' else Nequip(**model_kwargs)

    graphdef_base, _ = nnx.split(foundation, nnx.Param)
    ckpt             = load_checkpoint('ckpt_best.pkl')
    foundation       = nnx.merge(graphdef_base, ckpt['ema_params'])

    # ── Wrap with fine-tuning potential ───────────────────────────────────────
    ft_model = FineTuningPotential(
        foundation_model = foundation,
        radial_mlp_size  = config["radial_mlp_size"],
        radial_mlp_layers= config["radial_mlp_layers"],
        mlp_init_scale   = config["mlp_init_scale"],
        delta_scale      = 1.0,
        delta_shift      = 0.0,
        strategy         = config.get('ft_strategy', 'delta_1'),
        rngs             = ft_rngs,
    )

    graphdef, params = nnx.split(ft_model, nnx.Param)
    ema_params       = params

    n_params         = sum(x.size for x in jax.tree.leaves(params))
    trainable, frozen = filter_ft_params(params)
    n_trainable      = sum(x.size for x in jax.tree.leaves(trainable))
    n_frozen         = sum(x.size for x in jax.tree.leaves(frozen))
    print(f"   Total parameters : {n_params:,}")
    print(f"   Trainable: {n_trainable:,}  |  Frozen: {n_frozen:,}")
    print(f"Model parameters: {n_params:,}", file=fout)

    # ── Freeze mask (only ft_ layers are trainable) ────────────────────────────
    def _is_frozen(path):
        return not ('ft_' in str(path).lower())

    freeze_mask = jax.tree_util.tree_map_with_path(
        lambda path, _: _is_frozen(path), params
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.amsgrad(config["learning_rate"]),
        optax.transforms.freeze(freeze_mask),
    )
    schedule = optax.contrib.reduce_on_plateau(
        factor    = config["lr_schedule_factor"],
        patience  = config["lr_schedule_patience"],
        min_scale = config["lr_schedule_min_scale"],
    )

    energy_weight = config.get('energy_weight', 1.0)
    force_weight  = config.get('force_weight',  1.0)
    stress_weight = config.get('stress_weight', 1.0)
    huber_delta   = config.get('huber_delta',   0.02)
    loss_type     = config.get('loss_type',     'huber')

    train_step = make_sharded_train_step(
        optimizer     = optimizer,
        mesh          = mesh,
        energy_weight = energy_weight,
        force_weight  = force_weight,
        stress_weight = stress_weight,
        huber_delta   = huber_delta,
        ema_decay     = config.get('ema_decay', 0.99),
        freeze_mask   = freeze_mask,
        loss_type     = loss_type,
    )

    return (graphdef, params, ema_params, optimizer, schedule,
            train_step, energy_weight, force_weight, stress_weight, huber_delta)


# =============================================================================
# Main Training Function
# =============================================================================

def train(config: Dict, train_files_pkl: List[str], valid_files_pkl: List[str]):
    """Unified training entry point.

    Reads two independent flags from config:
      mode      : 'pretrain' | 'finetune'   (what to train)
      multihost : True | False              (hardware environment)
    """
    mode         = config.get('mode', 'pretrain').lower()
    is_multihost = bool(config.get('multihost', False))

    assert mode in ('pretrain', 'finetune'), \
        f"Unknown mode '{mode}'. Choose from: pretrain, finetune."

    # ── Logging ───────────────────────────────────────────────────────────────
    # multihost: only process 0 writes to the real log; others discard output.
    if is_multihost and jax.process_index() != 0:
        fout = open(os.devnull, 'w')
    else:
        default_log = 'loss_finetune.out' if mode == 'finetune' else 'loss_pretrain.out'
        fout = open(config.get('fname_log', default_log), 'w', 1)

    # ── Mesh ──────────────────────────────────────────────────────────────────
    mesh, n_devices = setup_mesh()

    # ── Header ────────────────────────────────────────────────────────────────
    task_label = 'Fine-Tuning' if mode == 'finetune' else 'Pretraining'
    hw_label   = 'Multi-Host' if is_multihost else 'Single-Host'
    print("=" * 70, file=fout)
    print(f"{task_label}  [{hw_label}]", file=fout)
    if is_multihost:
        print(f"  Hosts (DP)      : {jax.process_count()}", file=fout)
        print(f"  GPUs per host   : {jax.local_device_count()}", file=fout)
    print(f"  Total devices   : {n_devices}", file=fout)
    print(f"  Batch/device    : {config['batch_size']}", file=fout)
    print(f"  Effective batch : {config['batch_size'] * n_devices}", file=fout)
    print("=" * 70, file=fout)

    # ── Model / Optimizer / Train Step ────────────────────────────────────────
    if mode == 'finetune':
        (graphdef, params, ema_params, optimizer, schedule,
         train_step, energy_weight, force_weight, stress_weight, huber_delta
        ) = _build_finetune_components(config, mesh, n_devices, fout)
        ckpt_name = config.get('ckpt_name', 'ckpt_ft_best.pkl')
    else:  # pretrain
        (graphdef, params, ema_params, optimizer, schedule,
         train_step, energy_weight, force_weight, stress_weight, huber_delta
        ) = _build_pretrain_components(config, mesh, n_devices, fout)
        ckpt_name = config.get('ckpt_name', 'ckpt_best.pkl')

    opt_state      = optimizer.init(params)
    schedule_state = schedule.init(params)

    # ── Replicate across devices ───────────────────────────────────────────────
    params         = replicate_pytree(params,         mesh)
    ema_params     = replicate_pytree(ema_params,     mesh)
    opt_state      = replicate_pytree(opt_state,      mesh)
    schedule_state = replicate_pytree(schedule_state, mesh)
    step           = replicate(jnp.array(0), mesh)

    # ── Load checkpoint (restart) ──────────────────────────────────────────────
    best_val_loss = float('inf')
    if config.get('restart', False) and Path(ckpt_name).exists():
        ckpt = load_checkpoint(ckpt_name)
        params     = replicate_pytree(ckpt['ema_params'], mesh)
        ema_params = replicate_pytree(ckpt['ema_params'], mesh)
        opt_state      = replicate_pytree(ckpt['opt_state'],      mesh)
        schedule_state = replicate_pytree(ckpt['schedule_state'], mesh)
        step           = replicate(jnp.array(ckpt.get('step', 0)), mesh)
        best_val_loss  = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from step {ckpt.get('step', 0)}", file=fout)

    seed = config.get('seed', 42)

    # ── Training loop ──────────────────────────────────────────────────────────
    best_state           = None
    no_improvement_count = 0

    for epoch in range(config["n_epochs"]):
        epoch_start = time.time()

        for ipkl, train_pkl in enumerate(train_files_pkl):
            train_dataset = Dataset(file_path=train_pkl)
            loader_rngs   = nnx.Rngs(seed + 100 * epoch + ipkl)

            base_loader = BucketedDataLoader(
                dataset     = train_dataset,
                batch_size  = config['batch_size'],
                n_buckets   = 8,
                shuffle     = True,
                drop_last   = True,
                rngs        = loader_rngs,
            )
            loader = MultiDeviceDataLoader(
                base_loader     = base_loader,
                n_devices       = n_devices,
                mesh            = mesh,
                drop_incomplete = True,
            )

            # Metrics accumulators
            acc_energy_loss = acc_force_loss = acc_stress_loss = 0.
            acc_graphs = acc_atoms = acc_grad = 0.
            dataset_start = time.time()
            current_step  = int(np.asarray(step))

            for batch_data in loader:
                batch = batch_data[0] if isinstance(batch_data, tuple) else batch_data

                params, opt_state, ema_params, step, metrics = train_step(
                    graphdef, params, opt_state, schedule_state, ema_params, step, batch
                )

                n_g = float(metrics['n_graphs'])
                n_a = float(metrics['n_atoms'])
                acc_energy_loss += float(metrics['energy_loss']) * n_g
                acc_force_loss  += float(metrics['force_loss'])  * 3 * n_a
                acc_stress_loss += float(metrics['stress_loss']) * n_g
                acc_graphs      += n_g
                acc_atoms       += n_a
                acc_grad        += float(metrics['grad_norm'])   * n_g

                current_step = int(np.asarray(step))

                if current_step % 50 == 0:
                    e = acc_energy_loss / max(acc_graphs, 1)
                    f = acc_force_loss  / max(3 * acc_atoms, 1)
                    s = acc_stress_loss / max(acc_graphs, 1)
                    t = energy_weight * e + force_weight * f + stress_weight * s
                    print(f"step {current_step} LOSS: {t:.6f} "
                          f"E_LOSS: {e:.6f} F_LOSS: {f:.6f}", file=fout)

            # ── Per-pkl train summary ─────────────────────────────────────────
            unrep_schedule = unreplicate_pytree(schedule_state)
            lr             = config['learning_rate'] * float(unrep_schedule.scale)
            dataset_time   = time.time() - dataset_start

            energy_loss = acc_energy_loss / max(acc_graphs, 1)
            force_loss  = acc_force_loss  / max(3 * acc_atoms, 1)
            stress_loss = acc_stress_loss / max(acc_graphs, 1)
            total_loss  = (energy_weight * energy_loss
                           + force_weight * force_loss
                           + stress_weight * stress_loss)

            print(f"Epoch {epoch+1}/{config['n_epochs']} IPKL {ipkl+1} "
                  f"({dataset_time:.1f}s) Step {current_step} LR: {lr:.2e}", file=fout)
            print(f"  Train | Epoch {epoch+1}/{ipkl+1} | Loss: {total_loss:.6f} | "
                  f"E_LOSS: {energy_loss:.6f} | F_LOSS: {force_loss:.6f} | "
                  f"S_LOSS: {stress_loss:.6f} | "
                  f"Grad: {acc_grad/max(acc_graphs,1):.4f}", file=fout)

            # ── Validation ────────────────────────────────────────────────────
            val_start = time.time()
            unrep_ema = unreplicate_pytree(ema_params)

            val_metrics = evaluate_sharded(
                graphdef      = graphdef,
                params        = unrep_ema,
                files_pkl     = valid_files_pkl,
                batch_size    = config['batch_size'],
                n_devices     = n_devices,
                mesh          = mesh,
                energy_weight = energy_weight,
                force_weight  = force_weight,
                stress_weight = stress_weight,
                huber_delta   = huber_delta,
            )
            val_time = time.time() - val_start
            val_loss = val_metrics['total_loss']

            print(f"  Valid ({val_time:.1f}s) | Epoch {epoch+1}/{ipkl+1} | "
                  f"Loss: {val_loss:.6f} | "
                  f"E_LOSS: {val_metrics['energy_loss']:.6f} | "
                  f"F_LOSS: {val_metrics['force_loss']:.6f} | "
                  f"S_LOSS: {val_metrics['stress_loss']:.6f} | "
                  f"E_MAE: {val_metrics['energy_mae']:.6f} | "
                  f"F_MAE: {val_metrics['force_mae']:.6f} | "
                  f"S_MAE: {val_metrics['stress_mae']:.6f}", file=fout)

            # ── LR schedule update ────────────────────────────────────────────
            unrep_params   = unreplicate_pytree(params)
            unrep_schedule = unreplicate_pytree(schedule_state)
            _, new_schedule = schedule.update(
                updates = unrep_params,
                state   = unrep_schedule,
                value   = val_loss,
            )
            schedule_state = replicate_pytree(new_schedule, mesh)

            # ── Best model / checkpoint ───────────────────────────────────────
            if val_loss < best_val_loss:
                improvement   = best_val_loss - val_loss
                best_val_loss = val_loss
                no_improvement_count = 0

                best_state = {
                    'params':         unrep_params,
                    'ema_params':     unrep_ema,
                    'opt_state':      unreplicate_pytree(opt_state),
                    'schedule_state': new_schedule,
                    'step':           int(np.asarray(step)),
                    'best_val_loss':  best_val_loss,
                    'metrics':        val_metrics,
                }
                print(f"  *** New best val loss: {best_val_loss:.6f} "
                      f"(improved by {improvement:.6f}) ***", file=fout)

                # multihost: only process 0 writes the checkpoint file
                if not is_multihost or jax.process_index() == 0:
                    save_checkpoint(best_state, ckpt_name)
                    print(f"  Checkpoint saved to {ckpt_name}", file=fout)
            else:
                no_improvement_count += 1
                print(f"  No improvement for {no_improvement_count} evaluations "
                      f"(best: {best_val_loss:.6f})", file=fout)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{config['n_epochs']} completed in "
              f"{epoch_time:.1f}s", file=fout)

    # ── Final checkpoint ───────────────────────────────────────────────────────
    if best_state is not None:
        if not is_multihost or jax.process_index() == 0:
            save_checkpoint(best_state, ckpt_name)
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

    # ── Config ────────────────────────────────────────────────────────────────
    with open('input_omat24.json') as f:
        config = json.load(f)
    config['atom_energies'] = ATOM_ENERGIES

    mode         = config.get('mode', 'pretrain').lower()
    is_multihost = bool(config.get('multihost', False))

    # ── Distributed init (must happen before any JAX call) ───────────────────
    if is_multihost:
        initialize_distributed()

    # ── Startup info ──────────────────────────────────────────────────────────
    if not is_multihost or jax.process_index() == 0:
        print("=" * 70)
        print(f"JAX OMAT24 Training  [mode={mode}  multihost={is_multihost}]")
        print("=" * 70)
        print(f"JAX version   : {jax.__version__}")
        if is_multihost:
            print(f"Process       : {jax.process_index()} / {jax.process_count()}")
            print(f"Local devices : {jax.local_devices()}")
        else:
            print(f"Devices       : {jax.devices()}")
        print(f"Device count  : {jax.device_count()}")
        print("=" * 70)

    # ── Data paths ────────────────────────────────────────────────────────────
    train_path    = Path(config.get('train_path',    '/home/willow/Python/BAM-nequip-main/omat24_1M/train'))
    valid_path    = Path(config.get('valid_path',    '/home/willow/Python/BAM-nequip-main/omat24_1M/val'))
    train_pattern = config.get('train_pattern', 'omat*.pkl')
    valid_pattern = config.get('valid_pattern', 'omat*.pkl')

    train_files = natsorted(list(train_path.glob(train_pattern)))
    valid_files = natsorted(list(valid_path.glob(valid_pattern)))

    if not is_multihost or jax.process_index() == 0:
        print(f"Training files   : {len(train_files)}")
        print(f"Validation files : {len(valid_files)}")

    train(config, train_files, valid_files)
