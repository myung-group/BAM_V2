"""Training script for 3BPA dataset — PyTorch port of bam_omat24/train_omat24.py.

Supports pretrain (RACE or Nequip).
Single-GPU, single-node multi-GPU, and multi-node multi-GPU via DDP.

Usage (single GPU):
    python -m bam_torch_cue.train_3bpa input.json

Usage (single-node, 4 GPUs):
    torchrun --nproc_per_node=4 -m bam_torch_cue.train_3bpa input.json

Usage (multi-node, 2 nodes x 4 GPUs):
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
             --master_addr=<IP> --master_port=29500 \
             -m bam_torch_cue.train_3bpa input.json
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import torch.nn as nn
import torch.optim as optim

from bam_torch_cue.data import BucketedDataLoader, GraphDataset
from bam_torch_cue.distributed import (
    barrier,
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    setup_distributed,
    unwrap_model,
    wrap_model_ddp,
)
from bam_torch_cue.nequip_torch import NEQUIPModel
from bam_torch_cue.nequip import Nequip
from bam_torch_cue.race import RACE
from bam_torch_cue.training_utils import (
    evaluate,
    init_ema,
    load_checkpoint,
    make_train_step,
    save_checkpoint,
)
import numpy as np

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


def build_model(cfg: Dict, device: torch.device) -> nn.Module:
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
    )
    if model_type == 'Nequip':
        model = NEQUIPModel(**common)
    elif model_type == 'Nequip0':
        print ("HI NEQUIP0")
        model = Nequip(**common)
    else:
        model = RACE(**common)
    return model.to(device)


# =============================================================================
# Entry point
# =============================================================================

def train(cfg: Dict):
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    train_fname = 'dataset_3BPA/train_300K_cutoff_6.0.pkl'
    val_files = ['dataset_3BPA/valid_300K_cutoff_6.0.pkl']
    test_files = ['dataset_3BPA/test_300K_cutoff_6.0.pkl']
    train_dataset = GraphDataset(train_fname)
    train_loader = BucketedDataLoader(
        train_dataset, batch_size=5, shuffle=False, periodic=False,
        rank=rank, world_size=world_size,
    )

    log_path = _get(cfg, 'log_file', 'loss_3bpa_cue.out')
    fout = open(log_path, 'w', buffering=1) if is_main_process() else None

    def _log(*args, **kwargs):
        if is_main_process():
            print(*args, file=fout, **kwargs)

    _log(f"Device: {device} | Rank: {rank} | World size: {world_size}")

    torch.set_float32_matmul_precision('high')
    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    _log(f"Parameters: {n_params:,}")

    # Wrap model with DDP
    ddp_model = wrap_model_ddp(model, local_rank)
    raw_model = unwrap_model(ddp_model)

    lr          = _get(cfg, 'learning_rate', 1e-3)
    optimizer   = optim.Adam(raw_model.parameters(), lr=lr, amsgrad=True)
    lr_patience = _get(cfg, 'lr_schedule_patience', 10)
    lr_factor   = _get(cfg, 'lr_schedule_factor', 0.8)
    scheduler   = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor,
        patience=_get(cfg, 'lr_patience', lr_patience),
    )

    ema_state        = init_ema(raw_model)
    batch_size       = _get(cfg, 'batch_size', 20)
    loss_type        = _get(cfg, 'loss_type', 'huber')
    huber_delta      = _get(cfg, 'huber_delta', 0.1)
    e_weight         = _get(cfg, 'energy_weight', 1.0)
    f_weight         = _get(cfg, 'force_weight', 1.0)
    s_weight         = _get(cfg, 'stress_weight', 0.1)
    ema_decay        = _get(cfg, 'ema_decay', 0.99)
    n_epochs         = _get(cfg, 'n_epochs', 10)
    ckpt_path        = _get(cfg, 'checkpoint_path', 'ckpt_best.pkl')

    train_step = make_train_step(
        raw_model, optimizer,
        energy_weight=e_weight, force_weight=f_weight, stress_weight=s_weight,
        huber_delta=huber_delta, ema_decay=ema_decay, loss_type=loss_type,
        forward_model=ddp_model,
    )

    best_val_loss        = math.inf
    no_improvement_count = 0
    global_step          = 0

    for epoch in range(n_epochs):
        ddp_model.train()
        train_loader.set_epoch(epoch)

        # Per-epoch accumulators
        acc_energy = acc_force = 0.0
        acc_graphs = acc_atoms = acc_grad = 0.0
        dataset_start = time.time()

        for batch_np in train_loader:
            batch = batch_np.to(device)
            ema_state, metrics = train_step(batch, ema_state)

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

        t_loss = e_weight * e_loss + f_weight * f_loss
        lr_now = optimizer.param_groups[0]['lr']

        _log(f"Epoch {epoch+1}/{n_epochs} "
             f"({dataset_time:.1f}s) Step {global_step} LR: {lr_now:.2e}")
        _log(f"  Train | Loss: {t_loss:.6f} | "
             f"E_LOSS: {e_loss:.6f} | F_LOSS: {f_loss:.6f} |"
             f"Grad: {acc_grad / max(acc_graphs, 1):.4f}")

        val_start   = time.time()
        original_params = {n: p.data.clone() for n, p in raw_model.named_parameters()}
        with torch.no_grad():
            for name, param in raw_model.named_parameters():
                if name in ema_state:
                    param.data.copy_(ema_state[name])

        val_metrics = evaluate(
                raw_model, val_files, batch_size, device,
                energy_weight=e_weight, force_weight=f_weight, stress_weight=s_weight,
                huber_delta=huber_delta,
                loss_type=loss_type,
                periodic=False,
            )

        with torch.no_grad():
            for name, param in raw_model.named_parameters():
                param.data.copy_(original_params[name])
        val_time = time.time() - val_start
        val_loss = val_metrics['total_loss']
        scheduler.step(val_loss)

        _log(f"  Valid ({val_time:.1f}s) | Loss: {val_loss:.7f} | "
             f"E_MAE: {val_metrics['energy_mae']:.6f} | "
             f"F_MAE: {val_metrics['force_mae']:.6f} | "
             f"E_RMSE: {val_metrics['energy_rmse']:.6f} | "
             f"F_RMSE: {val_metrics['force_rmse']:.6f} | "
            )

        if val_loss < best_val_loss:
            improvement   = best_val_loss - val_loss
            best_val_loss = val_loss
            no_improvement_count = 0
            if is_main_process():
                save_checkpoint({
                    'model_state': raw_model.state_dict(),
                    'ema_state':   ema_state,
                    'optimizer':   optimizer.state_dict(),
                    'epoch':       epoch,
                    'step':        global_step,
                    'val_loss':    val_loss,
                    'cfg':         cfg,
                }, ckpt_path)
            _log(f"  *** New best val loss: {best_val_loss:.6f} "
                 f"(improved by {improvement:.6f}) *** -> saved to {ckpt_path}")
        else:
            no_improvement_count += 1
            _log(f"  No improvement for {no_improvement_count} evaluations "
                     f"(best: {best_val_loss:.6f})")

        if (epoch+1)%100 == 0:
            tst_start   = time.time()
            original_params = {n: p.data.clone() for n, p in raw_model.named_parameters()}
            with torch.no_grad():
                for name, param in raw_model.named_parameters():
                    if name in ema_state:
                        param.data.copy_(ema_state[name])

            tst_metrics = evaluate(
                raw_model, test_files, batch_size, device,
                energy_weight=e_weight, force_weight=f_weight, stress_weight=s_weight,
                huber_delta=huber_delta,
                loss_type=loss_type,
                periodic=False,
            )

            with torch.no_grad():
                for name, param in raw_model.named_parameters():
                    param.data.copy_(original_params[name])
            tst_time = time.time() - tst_start
            _log(f"  Test ({tst_time:.1f}s) | "
             f"E_MAE: {tst_metrics['energy_mae']:.6f} | "
             f"F_MAE: {tst_metrics['force_mae']:.6f} | "
             f"E_RMSE: {tst_metrics['energy_rmse']:.6f} | "
             f"F_RMSE: {tst_metrics['force_rmse']:.6f} | "
            )

    if fout is not None:
        fout.close()
    cleanup_distributed()

if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else 'input.json'
    with open(cfg_path) as f:
        config = json.load(f)
    train(config)
