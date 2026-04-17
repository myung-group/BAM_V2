"""Equivariant linear layers for PyTorch.

Uses cuequivariance_torch instead of e3nn.
Supports:
  - vanilla:  standard cuet.Linear
  - indexed:  per-species weights (weight_classes)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import cuequivariance as cue
import cuequivariance_torch as cuet

from bam_torch_cue.utils import LAYOUT, cue_irreps, is_scalar_ir


class Linear(nn.Module):
    """Equivariant linear layer, optionally with per-species indexed weights.

    Args:
        irreps_in:          Input irreps (cue.Irreps or string).
        irreps_out:         Output irreps (cue.Irreps or string).
        num_indexed_weights: If given, create per-index weight tables via
                             cuet.Linear's weight_classes parameter.
        biases:             Add learnable biases to scalar outputs.
        force_irreps_out:   If False (default), output irreps are filtered to
                            only those reachable from irreps_in.
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        num_indexed_weights: Optional[int] = None,
        biases: bool = False,
        force_irreps_out: bool = False,
    ):
        super().__init__()
        if isinstance(irreps_in, str):
            irreps_in = cue_irreps(irreps_in)
        if isinstance(irreps_out, str):
            irreps_out = cue_irreps(irreps_out)

        if not force_irreps_out:
            # Keep only irreps reachable from irreps_in (same as e3nn_jax default)
            keep = set(mi.ir for mi in irreps_in)
            irreps_out = cue.Irreps("O3", [(mi.mul, mi.ir) for mi in irreps_out if mi.ir in keep])
            irreps_out = irreps_out.simplify()

        self.irreps_in  = irreps_in
        self.irreps_out = irreps_out
        self.num_indexed_weights = num_indexed_weights

        if num_indexed_weights is not None:
            self._lin = cuet.Linear(
                irreps_in, irreps_out,
                weight_classes=num_indexed_weights,
                layout=LAYOUT,
            )
            # Custom init: smaller std for indexed weights
            with torch.no_grad():
                std = 1.0 / math.sqrt(max(irreps_in.dim, 1))
                self._lin.weight.normal_(0, std)

            # Optional bias: one per output-irreps scalar dim
            if biases:
                n_scalars = sum(mi.mul for mi in irreps_out if is_scalar_ir(mi.ir))
                self.bias = nn.Parameter(torch.zeros(n_scalars)) if n_scalars > 0 else None
            else:
                self.bias = None
        else:
            self._lin = cuet.Linear(
                irreps_in, irreps_out,
                layout=LAYOUT,
            )
            # cuet.Linear doesn't support biases; add manually if requested
            if biases:
                n_scalars = sum(mi.mul for mi in irreps_out if is_scalar_ir(mi.ir))
                self.bias = nn.Parameter(torch.zeros(n_scalars)) if n_scalars > 0 else None
            else:
                self.bias = None

    def forward(
        self,
        x: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       (N, irreps_in.dim) float tensor.
            indices: (N,) long tensor of species indices (only for indexed mode).

        Returns:
            (N, irreps_out.dim) float tensor.
        """
        if self.num_indexed_weights is not None:
            assert indices is not None, "indexed Linear requires species indices"
            out = self._lin(x, weight_indices=indices)
        else:
            out = self._lin(x)

        if self.bias is not None:
            # Add bias to the scalar portion of the output
            n_scalars = self.bias.shape[0]
            out = out.clone()
            out[:, :n_scalars] = out[:, :n_scalars] + self.bias

        return out
