"""Training utilities — PyTorch port of bam_omat24/training_utils.py.

Loss functions (Huber / NLL with Cholesky covariance), training step,
evaluation loop, and checkpointing.
Multi-GPU / multi-node via torch.nn.parallel.DistributedDataParallel (DDP).
"""

import pickle
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from bam_torch.data import BucketedDataLoader, GraphDataset, TorchGraphData
from bam_torch.distributed import get_rank, get_world_size, is_distributed, reduce_dict
import numpy as np

# =============================================================================
# Loss functions
# =============================================================================

def huber_loss(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Element-wise Huber loss."""
    abs_x = x.abs()
    return torch.where(abs_x <= delta, 0.5 * x ** 2, delta * (abs_x - 0.5 * delta))


def _build_force_L(frc_var: torch.Tensor) -> torch.Tensor:
    """Build (N,3,3) lower-triangular Cholesky L from (N,6) frc_var.

    Column layout: [d0, d1, d2, off_21, off_20, off_10]
    L = [[d0,      0,  0],
         [off_10, d1,  0],
         [off_20, off_21, d2]]
    """
    N = frc_var.shape[0]
    L = torch.zeros(N, 3, 3, dtype=frc_var.dtype, device=frc_var.device)
    L[:, 0, 0] = frc_var[:, 0]
    L[:, 1, 1] = frc_var[:, 1]
    L[:, 2, 2] = frc_var[:, 2]
    L[:, 2, 1] = frc_var[:, 3]
    L[:, 2, 0] = frc_var[:, 4]
    L[:, 1, 0] = frc_var[:, 5]
    return L


def compute_loss(
    model: nn.Module,
    batch: TorchGraphData,
    energy_weight: float,
    force_weight: float,
    stress_weight: float,
    huber_delta: float,
    loss_type: str = 'huber',
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Compute training loss for a single batch.

    Args:
        loss_type: ``'huber'`` for plain Huber loss,
                   ``'nll'``   for Huber-NLL with Cholesky covariance.

    Returns:
        (total_loss, aux_metrics_dict, model_results)
    """
    n_graphs = batch.n_node.shape[0]
    n_atoms  = batch.nodes['positions'].shape[0]

    results = model(batch)
    energy  = results['energy']   # (G,)
    forces  = results['forces']   # (N,3)
    has_stress = 'stress' in batch.globals

    vmax_E, vmax_F, vmax_S = 5.0, 10.0, 0.5

    n_node_safe = batch.n_node.float().clamp(min=1)
    energy_diff = (energy - batch.globals['energy']) / n_node_safe
    force_diff  = forces - batch.nodes['forces']

    energy_diff = vmax_E * torch.tanh(energy_diff / vmax_E)
    force_diff  = vmax_F * torch.tanh(force_diff  / vmax_F)

    if has_stress:
        stress      = results['stress']   # (G,6)
        stress_diff = stress - batch.globals['stress']
        stress_diff = vmax_S * torch.tanh(stress_diff / vmax_S)

    if loss_type == 'nll':
        frc_var    = results['force_variation']          # (N,6)
        energy_var = results.get('energy_variation')     # (G,1) or None
        stress_var = results.get('stress_variation')     # (G,6) or None

        # --- Energy NLL ---
        if energy_var is not None:
            e_sigma = energy_var.squeeze(-1)
            energy_z    = energy_diff / e_sigma
            energy_loss = huber_loss(energy_z, delta=huber_delta).sum() / n_graphs
            energy_logdet = 2.0 * e_sigma.log().sum() / n_graphs
        else:
            energy_loss   = huber_loss(energy_diff, delta=huber_delta).sum() / n_graphs
            energy_logdet = 0.0

        # --- Force NLL (3×3 Cholesky per atom) ---
        frc_L   = _build_force_L(frc_var)
        frc_z   = torch.linalg.solve_triangular(frc_L, force_diff.unsqueeze(-1), upper=False).squeeze(-1)
        force_loss    = huber_loss(frc_z, delta=huber_delta).sum() / (3 * n_atoms)
        frc_diag      = frc_L.diagonal(dim1=-2, dim2=-1)            # (N,3)
        force_logdet  = 2.0 * frc_diag.log().sum() / (3 * n_atoms)

        # --- Stress NLL (diagonal covariance per graph) ---
        if has_stress:
            if stress_var is not None:
                s_sigma    = stress_var
                stress_z   = stress_diff / s_sigma
                stress_loss  = huber_loss(stress_z, delta=huber_delta).sum() / n_graphs
                stress_logdet= 2.0 * s_sigma.log().sum() / n_graphs
            else:
                stress_loss   = huber_loss(stress_diff, delta=huber_delta).sum() / n_graphs
                stress_logdet = 0.0
        else:
            stress_loss   = torch.tensor(0.0, device=energy.device)
            stress_logdet = 0.0

        MIN_VAR      = 0.04
        log_min_var  = torch.log(torch.tensor(MIN_VAR))
        energy_const = -2.0 * log_min_var if energy_var is not None else 0.0
        force_const  = -2.0 * log_min_var
        stress_const = -12.0 * log_min_var if (has_stress and stress_var is not None) else 0.0

        energy_nll = energy_loss + energy_logdet + energy_const
        force_nll  = force_loss  + force_logdet  + force_const
        stress_nll = stress_loss + stress_logdet + stress_const

        total_loss = (
            energy_weight * energy_nll
            + force_weight  * force_nll
            + stress_weight * stress_nll
        )
        aux = {'energy_loss': energy_nll, 'force_loss': force_nll, 'stress_loss': stress_nll}

    else:  # plain Huber
        energy_loss = huber_loss(energy_diff, delta=huber_delta).sum() / n_graphs
        force_loss  = huber_loss(force_diff,  delta=huber_delta).sum() / (3 * n_atoms)
        if has_stress:
            stress_loss = huber_loss(stress_diff, delta=huber_delta).sum() / n_graphs
        else:
            stress_loss = torch.tensor(0.0, device=energy.device)
        total_loss  = (energy_weight * energy_loss
                       + force_weight  * force_loss
                       + stress_weight * stress_loss)
        aux = {'energy_loss': energy_loss, 'force_loss': force_loss, 'stress_loss': stress_loss}

    aux.update({
        'energy_mse': (energy_diff ** 2).sum() / n_graphs,
        'force_mse':  (force_diff  ** 2).sum() / (3 * n_atoms),
        'stress_mse': (stress_diff ** 2).sum() / n_graphs if has_stress else 0.0,
        'n_graphs':   n_graphs,
        'n_atoms':    n_atoms,
    })
    return total_loss, aux, results


# =============================================================================
# Training step
# =============================================================================

def make_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    stress_weight: float = 1.0,
    huber_delta: float = 0.1,
    ema_decay: float = 0.99,
    loss_type: str = 'huber',
    max_grad_norm: float = 10.0,
    forward_model: Optional[nn.Module] = None,
):
    """Return a callable train_step(batch, ema_params) -> (ema_params, metrics).

    Args:
        model: The raw (unwrapped) model — used for EMA updates and gradient
            clipping.  Parameter names in the EMA dict match this model.
        forward_model: Optional DDP-wrapped model used for the forward and
            backward pass so that gradient synchronisation occurs.  When
            ``None`` (single-GPU), ``model`` is used for the forward pass.

    EMA params are plain tensors (not nn.Parameters) for simplicity.
    """
    fwd_model = forward_model if forward_model is not None else model

    def train_step(
        batch: TorchGraphData,
        ema_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Dict[str, float]]:
        fwd_model.train()
        optimizer.zero_grad()

        loss, aux, _ = compute_loss(
            fwd_model, batch,
            energy_weight, force_weight, stress_weight,
            huber_delta, loss_type,
        )
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # EMA update (uses raw model so key names are stable)
        if ema_state is not None:
            with torch.no_grad():
                new_ema = {}
                for name, param in model.named_parameters():
                    if name in ema_state:
                        new_ema[name] = (
                            ema_decay * ema_state[name]
                            + (1.0 - ema_decay) * param.data
                        )
                    else:
                        new_ema[name] = param.data.clone()
        else:
            new_ema = None

        metrics = {k: float(v.detach()) if hasattr(v, 'detach') else float(v)
                   for k, v in aux.items()}
        metrics['loss']      = float(loss.detach())
        metrics['grad_norm'] = float(grad_norm)
        return new_ema, metrics

    return train_step


def init_ema(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Initialise EMA state from model parameters."""
    return {name: param.data.clone() for name, param in model.named_parameters()}


# =============================================================================
# Evaluation loop
# =============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    files_pkl: List[str],
    batch_size: int,
    device: torch.device,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    stress_weight: float = 1.0,
    huber_delta: float = 0.02,
    loss_type: str = 'huber',
    periodic: bool = True,
) -> Dict[str, float]:
    """Evaluate model on a list of pickle files and return aggregated metrics.

    In distributed mode each rank evaluates a shard of the data (via
    ``BucketedDataLoader(rank=..., world_size=...)``), then the raw
    accumulators are all-reduced before computing final averages.

    Note: force gradients are still needed even in eval mode, so we use
    torch.enable_grad() inside the model's energy_forces_stress call.
    Forces require grad w.r.t. positions even during inference.
    """
    model.eval()

    rank = get_rank()
    world_size = get_world_size()

    totals: Dict[str, float] = {
        'energy_loss': 0.0, 'force_loss': 0.0, 'stress_loss': 0.0,
        'energy_se': 0.0, 'force_se': 0.0, 'stress_se': 0.0,
        'energy_ae': 0.0, 'force_ae': 0.0, 'stress_ae': 0.0,
        'n_graphs': 0.0, 'n_atoms': 0.0,
    }

    for fname in files_pkl:
        dataset = GraphDataset(file_path=fname)
        loader  = BucketedDataLoader(
            dataset, batch_size=batch_size, shuffle=False, periodic=periodic,
            rank=rank, world_size=world_size,
        )

        for batch_np in loader:
            batch = batch_np.to(device)

            # Forces require grad even during eval
            with torch.enable_grad():
                _, aux, results = compute_loss(
                    model, batch,
                    energy_weight, force_weight, stress_weight,
                    huber_delta, loss_type,
                )

            n_graphs   = float(batch.n_node.shape[0])
            n_atoms    = float(batch.nodes['positions'].shape[0])

            vmax_E, vmax_F, vmax_S = 5.0, 10.0, 0.5
            n_node_safe = batch.n_node.float().clamp(min=1)
            ed = (results['energy'].detach() - batch.globals['energy']) / n_node_safe
            fd = results['forces'].detach() - batch.nodes['forces']
            ed = vmax_E * torch.tanh(ed / vmax_E)
            fd = vmax_F * torch.tanh(fd / vmax_F)

            totals['energy_loss'] += float(aux['energy_loss']) * n_graphs
            totals['force_loss']  += float(aux['force_loss'])  * 3 * n_atoms
            totals['stress_loss'] += float(aux['stress_loss']) * n_graphs
            totals['energy_se']   += float((ed ** 2).sum())
            totals['force_se']    += float((fd ** 2).sum())
            totals['energy_ae']   += float(ed.abs().sum())
            totals['force_ae']    += float(fd.abs().sum())

            if 'stress' in batch.globals:
                sd = results['stress'].detach() - batch.globals['stress']
                sd = vmax_S * torch.tanh(sd / vmax_S)
                totals['stress_se'] += float((sd ** 2).sum())
                totals['stress_ae'] += float(sd.abs().sum())

            totals['n_graphs']    += n_graphs
            totals['n_atoms']     += n_atoms

    # All-reduce accumulators across ranks before computing averages
    totals = reduce_dict(totals, device)

    n_g = max(totals['n_graphs'], 1)
    n_a = max(totals['n_atoms'],  1)

    total_loss = (
        energy_weight * totals['energy_loss'] / n_g
        + force_weight  * totals['force_loss']  / (3 * n_a)
        + stress_weight * totals['stress_loss'] / n_g
    )

    return {
        'energy_mae':  totals['energy_ae'] / n_g,
        'force_mae':   totals['force_ae']  / (3 * n_a),
        'stress_mae':  totals['stress_ae'] / n_g,
        'energy_rmse':  np.sqrt(totals['energy_se'] / n_g),
        'force_rmse':   np.sqrt(totals['force_se']  / (3 * n_a)),
        'stress_rmse':  np.sqrt(totals['stress_se'] / n_g),
        'energy_loss': totals['energy_loss'] / n_g,
        'force_loss':  totals['force_loss']  / (3 * n_a),
        'stress_loss': totals['stress_loss'] / n_g,
        'total_loss':  total_loss,
        'n_graphs':    n_g,
        'n_atoms':     n_a,
    }


# =============================================================================
# Checkpoint management
# =============================================================================

def save_checkpoint(state: Dict[str, Any], path: str):
    with open(path, 'wb') as f:
        pickle.dump(state, f)


def load_checkpoint(path: str) -> Dict[str, Any]:
    with open(path, 'rb') as f:
        return pickle.load(f)
