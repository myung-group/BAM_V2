"""Distributed training utilities for multi-GPU and multi-node training.

Supports:
- Single-node multi-GPU via DistributedDataParallel (DDP)
- Multi-node multi-GPU via torchrun / torch.distributed.launch

Usage (single-node, 4 GPUs):
    torchrun --nproc_per_node=4 -m bam_torch_cue.train input.json

Usage (multi-node, 2 nodes x 4 GPUs):
    # Node 0:
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \\
             --master_addr=<IP> --master_port=29500 -m bam_torch_cue.train input.json
    # Node 1:
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 \\
             --master_addr=<IP> --master_port=29500 -m bam_torch_cue.train input.json
"""

import os
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed(backend: str = 'nccl') -> tuple:
    """Initialize distributed process group.

    Returns:
        (rank, local_rank, world_size)

    Expects environment variables set by ``torchrun``:
        RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT.
    Falls back to single-GPU when WORLD_SIZE <= 1 or is unset.
    """
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if world_size > 1:
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed():
    """Destroy distributed process group if active."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not is_distributed() or dist.get_rank() == 0


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def wrap_model_ddp(
    model: nn.Module,
    local_rank: int,
    find_unused_parameters: bool = True,
) -> nn.Module:
    """Wrap model with DistributedDataParallel.

    ``find_unused_parameters=True`` is needed because variance heads
    may be unused depending on loss_type (huber vs nll) or periodic mode.
    """
    if is_distributed():
        return DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=find_unused_parameters,
        )
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    """Get underlying model from a DDP wrapper (no-op if not wrapped)."""
    if isinstance(model, DDP):
        return model.module
    return model


def reduce_dict(
    d: Dict[str, float],
    device: torch.device,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
) -> Dict[str, float]:
    """All-reduce a dict of float metrics across ranks."""
    if not is_distributed():
        return d
    keys = sorted(d.keys())
    vals = torch.tensor([d[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(vals, op=op)
    return {k: float(v) for k, v in zip(keys, vals)}


def barrier():
    """Synchronize all ranks."""
    if is_distributed():
        dist.barrier()
