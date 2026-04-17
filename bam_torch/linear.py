"""Equivariant linear layers for PyTorch.

Replaces bam_omat24/linear_nnx.py.  Uses e3nn (PyTorch) instead of e3nn_jax.
Supports:
  - vanilla:  standard e3nn.o3.Linear
  - indexed:  per-species weights (species → weight table lookup)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from e3nn import o3


class Linear(nn.Module):
    """Equivariant linear layer, optionally with per-species indexed weights.

    Args:
        irreps_in:          Input irreps string.
        irreps_out:         Output irreps string.
        num_indexed_weights: If given, create a (num_indexed_weights, weight_numel)
                             table; the forward takes an integer `indices` arg.
        biases:             Add learnable biases to scalar outputs.
        force_irreps_out:   If False (default), output irreps are filtered to
                            only those reachable from irreps_in.
    """

    def __init__(
        self,
        irreps_in: str,
        irreps_out: str,
        num_indexed_weights: Optional[int] = None,
        biases: bool = False,
        force_irreps_out: bool = False,
    ):
        super().__init__()
        irreps_in  = o3.Irreps(irreps_in)
        irreps_out = o3.Irreps(irreps_out)

        if not force_irreps_out:
            # Keep only irreps reachable from irreps_in (same as e3nn_jax default)
            keep = set(ir for _, ir in irreps_in)
            irreps_out = o3.Irreps([(mul, ir) for mul, ir in irreps_out if ir in keep])
            irreps_out = irreps_out.simplify()

        self.irreps_in  = irreps_in
        self.irreps_out = irreps_out
        self.num_indexed_weights = num_indexed_weights

        if num_indexed_weights is not None:
            # Weights are NOT stored inside the Linear; we pass them per-call.
            # Biases are handled as a separate parameter (e3nn biases + shared_weights=False
            # can conflict in some versions).
            self._lin = o3.Linear(
                irreps_in, irreps_out,
                biases=False,           # bias handled separately below
                internal_weights=False,
                shared_weights=False,
            )
            # Weight table: (n_species, weight_numel)
            n_w = self._lin.weight_numel
            std = 1.0 / math.sqrt(max(irreps_in.dim, 1))
            self.weight_table = nn.Parameter(
                torch.randn(num_indexed_weights, n_w) * std
            )
            # Optional bias: one per output-irreps scalar dim
            if biases:
                # Bias only applies to scalar (0e) outputs in e3nn convention
                n_scalars = sum(mul for mul, ir in irreps_out if ir.is_scalar())
                self.bias = nn.Parameter(torch.zeros(n_scalars)) if n_scalars > 0 else None
            else:
                self.bias = None
        else:
            self._lin = o3.Linear(irreps_in, irreps_out, biases=biases)
            self.bias = None   # bias stored internally in _lin

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
            w = self.weight_table[indices]   # (N, weight_numel)
            out = self._lin(x, weight=w)
            if self.bias is not None:
                # Add bias to the scalar portion of the output
                n_scalars = self.bias.shape[0]
                out = out.clone()
                out[:, :n_scalars] = out[:, :n_scalars] + self.bias
            return out
        return self._lin(x)
