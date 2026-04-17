"""Training script for 3BPA dataset — JAX/Flax port of bam_omat24/train_omat24.py.

Supports pretrain (RACE or Nequip).
Single-GPU and single-node multi-GPU via JAX sharding.

Usage:
    python -m bam_omat24.train_3bpa input.json
"""

from typing import Dict, List, Any
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

# 3BPA atom energies: H, C, N, O
atom_energies = np.array(
        [-13.587222780835477,
        -1029.4889999855063,
        -1484.9814568572233,
        -2041.9816003861047
        ])
# =============================================================================
# Config helpers
# =============================================================================

def _get(cfg: Dict, key: str, default: Any = None) -> Any:
    return cfg.get(key, default)


def build_model(cfg: Dict, mesh, n_devices: int, fout):
    seed = _get(cfg, 'seed', 42)
    model_rngs = nnx.Rngs (seed)

    model_type = _get(cfg, 'model', 'RACE')
    common = dict(
        n_species      = len(atom_energies),
        lmax           = _get(cfg, 'lmax', 3),
        cutoff         = _get(cfg, 'cutoff', 6.0),
        hidden_irreps  = _get(cfg, 'hidden_irreps', '64x0e + 64x1o + 64x2e'),
        n_layers       = _get(cfg, 'n_layers', 5),
        radial_basis_size  = _get(cfg, 'radial_basis_size', 8),
        radial_mlp_size    = _get(cfg, 'radial_mlp_size', 64),
        radial_mlp_layers  = _get(cfg, 'radial_mlp_layers', 3),
        radial_polynomial_p= _get(cfg, 'radial_polynomial_p', 2.0),
        mlp_init_scale     = _get(cfg, 'mlp_init_scale', 4.0),
        shift              = _get(cfg, 'shift', 0.0),
        scale              = _get(cfg, 'scale', 1.0),
        avg_n_neighbors    = _get(cfg, 'avg_n_neighbors', 25.0),
        atom_energies      = atom_energies.tolist(),
        l_train            = True,
        periodic           = _get(cfg, 'periodic', False),
        rngs               = model_rngs,
    )
    if model_type == 'Nequip':
        model = Nequip(**common)
    else:
        model = RACE(**common)
    
    graphdef, params = nnx.split(model, nnx.Param)
    ema_params       = params

    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"Model parameters: {n_params:,}", file=fout)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.amsgrad(cfg["learning_rate"]),
    )
    schedule = optax.contrib.reduce_on_plateau(
        factor    = _get(cfg,"lr_schedule_factor", 0.8),
        patience  = _get(cfg, "lr_schedule_patience", 10),
        min_scale = _get(cfg,"lr_schedule_min_scale", 0.001)
    )

    energy_weight = _get(cfg, 'energy_weight', 1.0)
    force_weight  = _get(cfg, 'force_weight',  1.0)
    stress_weight = _get(cfg, 'stress_weight', 1.0)
    huber_delta   = _get(cfg, 'huber_delta',   0.5)
    loss_type     = _get(cfg, 'loss_type',    'huber')
    print ("loss_type", loss_type)
    
    periodic = _get(cfg, 'periodic', False)
    train_step = make_sharded_train_step(
        optimizer     = optimizer,
        mesh          = mesh,
        energy_weight = energy_weight,
        force_weight  = force_weight,
        stress_weight = stress_weight,
        huber_delta   = huber_delta,
        ema_decay     = cfg.get('ema_decay', 0.99),
        loss_type     = loss_type,
        periodic      = periodic,
    )
    return (graphdef, params, ema_params, optimizer, schedule,
            train_step, energy_weight, force_weight, stress_weight, huber_delta)


# =============================================================================
# Entry point
# =============================================================================

def train(cfg: Dict):
    
    #rank, local_rank, world_size = setup_distributed()
    #device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    mesh, n_devices = setup_mesh()
    loss_type = _get (cfg, 'loss_type', 'huber')

    log_path = _get(cfg, 'log_file', 'loss_3bpa.out')
    fout = open(log_path, 'w', buffering=1)

    (graphdef, params, ema_params, optimizer, schedule,
    train_step, energy_weight, force_weight, stress_weight, huber_delta
    ) = build_model (cfg, mesh, n_devices, fout)
    ckpt_name = cfg.get('ckpt_name', 'ckpt_best.pkl')

    opt_state      = optimizer.init(params)
    schedule_state = schedule.init(params)
    # ── Replicate across devices ───────────────────────────────────────────────
    params         = replicate_pytree(params,         mesh)
    ema_params     = replicate_pytree(ema_params,     mesh)
    opt_state      = replicate_pytree(opt_state,      mesh)
    schedule_state = replicate_pytree(schedule_state, mesh)
    step           = replicate(jnp.array(0), mesh)

    train_fname = 'dataset_3BPA/train_300K_cutoff_6.0.pkl'
    val_files = ['dataset_3BPA/valid_300K_cutoff_6.0.pkl']
    test_files = ['dataset_3BPA/test_300K_cutoff_6.0.pkl']
    train_dataset = Dataset(train_fname)
    loader_rngs = nnx.Rngs (100)
    batch_size = _get (cfg, 'batch_size', 5)
    train_loader = BucketedDataLoader(
        train_dataset, batch_size=batch_size, n_buckets=8,
        shuffle=True, drop_last=True,
        rngs = loader_rngs
    )

    # ── Build eval_step once (avoids re-JIT on every validation call) ─────────
    from bam_omat24.training_utils import make_sharded_evaluate_step
    periodic = _get(cfg, 'periodic', False)
    eval_step = make_sharded_evaluate_step(mesh, huber_delta=huber_delta, loss_type=loss_type, periodic=periodic)

    def _log(*args, **kwargs):
        print(*args, file=fout, **kwargs)

    best_val_loss        = math.inf
    best_state           = None
    no_improvement_count = 0
    global_step          = 0
    n_epochs             = _get(cfg, 'n_epochs', 1000)
    for epoch in range(n_epochs):
        
        # Per-epoch accumulators
        acc_energy = acc_force = 0.0
        acc_graphs = acc_atoms = acc_grad = 0.0
        dataset_start = time.time()

        for batch_data in train_loader:
            batch = batch_data

            params, opt_state, ema_params, step, metrics = train_step(
                graphdef, params, opt_state, schedule_state, ema_params, step, batch
            )
            
            n_g = metrics['n_graphs']
            n_a = metrics['n_atoms']
            acc_energy += metrics['energy_loss'] * n_g
            acc_force  += metrics['force_loss']  * 3 * n_a

            acc_graphs += n_g
            acc_atoms  += n_a
            acc_grad   += metrics['grad_norm']   * n_g
            global_step += 1

        dataset_time = time.time() - dataset_start
        e_loss = acc_energy / max(acc_graphs, 1)
        f_loss = acc_force  / max(3 * acc_atoms, 1)

        t_loss = energy_weight * e_loss + force_weight * f_loss
        unrep_schedule = unreplicate_pytree(schedule_state)
        lr_now = cfg['learning_rate'] * float(unrep_schedule.scale)
        

        _log(f"Epoch {epoch+1}/{n_epochs} "
             f"({dataset_time:.1f}s) Step {global_step} LR: {lr_now:.2e}")
        _log(f"  Train | Loss: {t_loss:.6f} | "
             f"E_LOSS: {e_loss:.6f} | F_LOSS: {f_loss:.6f} |"
             f"Grad: {acc_grad / max(acc_graphs, 1):.4f}")

        val_start   = time.time()
        unrep_ema = unreplicate_pytree(ema_params)

        val_metrics = evaluate_sharded(
                graphdef      = graphdef,
                params        = ema_params,
                files_pkl     = val_files,
                batch_size    = batch_size,
                n_devices     = n_devices,
                mesh          = mesh,
                energy_weight = energy_weight,
                force_weight  = force_weight,
                stress_weight = stress_weight,
                huber_delta   = huber_delta,
                loss_type     = loss_type,
                eval_step     = eval_step,
                params_already_replicated = True,
            )

        val_time = time.time() - val_start
        val_loss = val_metrics['total_loss']
        
        _log(f"  Valid ({val_time:.1f}s) | Loss: {val_loss:.7f} | "
             f"E_MAE: {val_metrics['energy_mae']:.6f} | "
             f"F_MAE: {val_metrics['force_mae']:.6f} | "
             f"E_RMSE: {val_metrics['energy_rmse']:.6f} | "
             f"F_RMSE: {val_metrics['force_rmse']:.6f} | "
            )

        # ── LR schedule update ────────────────────────────────────────────
        unrep_params   = unreplicate_pytree(params)
        unrep_schedule = unreplicate_pytree(schedule_state)
        _, new_schedule = schedule.update(
            updates = unrep_params,
            state   = unrep_schedule,
            value   = val_loss,
        )
        schedule_state = replicate_pytree(new_schedule, mesh)

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
            
            save_checkpoint(best_state, ckpt_name)
            _log(f"  *** New best val loss: {best_val_loss:.6f} "
                 f"(improved by {improvement:.6f}) *** -> saved to {ckpt_name}")
        else:
            no_improvement_count += 1
            _log(f"  No improvement for {no_improvement_count} evaluations "
                     f"(best: {best_val_loss:.6f})")

        if (epoch+1)%100 == 0:
            tst_start   = time.time()
            
            tst_metrics = evaluate_sharded(
                graphdef      = graphdef,
                params        = ema_params,
                files_pkl     = test_files,
                batch_size    = batch_size,
                n_devices     = n_devices,
                mesh          = mesh,
                energy_weight = energy_weight,
                force_weight  = force_weight,
                stress_weight = stress_weight,
                huber_delta   = huber_delta,
                loss_type     = loss_type,
                eval_step     = eval_step,
                params_already_replicated = True,
            )
            
            tst_time = time.time() - tst_start
            _log(f"  Test ({tst_time:.1f}s) | "
             f"E_MAE: {tst_metrics['energy_mae']:.6f} | "
             f"F_MAE: {tst_metrics['force_mae']:.6f} | "
             f"E_RMSE: {tst_metrics['energy_rmse']:.6f} | "
             f"F_RMSE: {tst_metrics['force_rmse']:.6f} | "
            )
    
    fout.close()

if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'input.json'
    with open(cfg_path) as f:
        config = json.load(f)
    train(config)
