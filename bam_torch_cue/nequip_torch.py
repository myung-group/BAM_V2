# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NequIP model implementation using PyTorch and cuequivariance_torch.

PyTorch port of the Flax NNX / cuequivariance_jax implementation. Uses
dict[Irrep, Tensor] representation with tensors of shape (batch, mul, ir.dim)
throughout, and flattens only at SegmentedPolynomial boundaries.

Reference: Batzner et al. (2022) https://arxiv.org/abs/2101.03164
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance import Irrep

from bam_torch.data import TorchGraphData
from bam_torch_cue.utils import (
    compute_node_edge_features,
    energy_forces_stress,
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _poly_method(prefer: str = "uniform_1d") -> str:
    """Pick `uniform_1d` when CUDA is available, otherwise fall back to `naive`.

    The CUDA-backed kernel is ~10x faster for the channelwise tensor product
    used in message passing, but it is only compiled for CUDA devices.
    """
    return prefer if torch.cuda.is_available() else "naive"


def normalize_function(f: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    """Rescale an activation so that E[f(x)^2] = 1 for x ~ N(0, 1).

    Mirrors `cuex.normalize_function`. The normalization constant is computed
    once via Monte Carlo sampling in float64 with a fixed seed, so repeated
    calls on the same activation produce the exact same scale (critical for
    equivariance -- otherwise two forward passes would silently use slightly
    different activation scales).
    """
    if f is None:
        return None
    with torch.no_grad():
        gen = torch.Generator(device="cpu").manual_seed(0)
        x = torch.randn(1_000_000, dtype=torch.float64, generator=gen)
        scale = 1.0 / f(x).pow(2).mean().sqrt().item()

    def normalized(x: Tensor) -> Tensor:
        return f(x) * scale

    return normalized


def radial_basis(r: Tensor, r_max: float, num_basis: int) -> Tensor:
    """Bessel radial basis with polynomial envelope cutoff."""
    p = 5
    u = r / r_max
    up = torch.pow(u, p)
    envelope = (
        1.0
        - 0.5 * (p + 1) * (p + 2) * up
        + p * (p + 2) * up * u
        - 0.5 * p * (p + 1) * up * u * u
    )
    envelope = torch.where(r < r_max, envelope, torch.zeros_like(envelope))
    ns = torch.arange(1, num_basis + 1, dtype=r.dtype, device=r.device)
    bessel = (
        (2.0 / r_max) ** 0.5
        * torch.pi
        * ns
        / r_max
        * torch.sinc(ns * r[..., None] / r_max)
    )
    return bessel * envelope[..., None]


# ---------------------------------------------------------------------------
# dict[Irrep, Tensor] helpers -- shape convention: (batch, mul, ir.dim)
# ---------------------------------------------------------------------------

def irreps_add(
    a: dict[Irrep, Tensor], b: dict[Irrep, Tensor]
) -> dict[Irrep, Tensor]:
    """Element-wise sum of two irrep-dicts, taking union of keys."""
    out = dict(a)
    for ir, v in b.items():
        out[ir] = out[ir] + v if ir in out else v
    return out


def irreps_dict_to_flat(d: dict[Irrep, Tensor], irreps: cue.Irreps) -> Tensor:
    """Concatenate per-irrep tensors into a flat (batch, irreps.dim) tensor.

    Assumes `irreps` is regrouped: each irrep type appears at most once.
    Input tensors have shape (batch, mul, ir.dim); the flattened layout is
    **ir_mul** (for each irrep, ir.dim is the outer axis and mul is the
    inner axis), which is the convention expected by cuet.SegmentedPolynomial.
    """
    parts = []
    for mul, ir in irreps:
        v = d[ir]  # (..., mul, ir.dim)
        # (..., mul, ir.dim) -> (..., ir.dim, mul) -> (..., ir.dim * mul)
        parts.append(v.transpose(-1, -2).reshape(*v.shape[:-2], mul * ir.dim))
    return torch.cat(parts, dim=-1)


def flat_to_irreps_dict(flat: Tensor, irreps: cue.Irreps) -> dict[Irrep, Tensor]:
    """Split a flat (batch, irreps.dim) tensor into a dict keyed by irrep.

    Inverse of `irreps_dict_to_flat`: expects **ir_mul** layout in `flat`
    and produces per-irrep tensors of shape (batch, mul, ir.dim). If the
    same irrep appears multiple times in `irreps`, chunks are concatenated
    along the mul axis.
    """
    out: dict[Irrep, Tensor] = {}
    offset = 0
    lead = flat.shape[:-1]
    for mul, ir in irreps:
        size = mul * ir.dim
        # (..., ir.dim * mul) -> (..., ir.dim, mul) -> (..., mul, ir.dim)
        chunk = (
            flat[..., offset : offset + size]
            .reshape(*lead, ir.dim, mul)
            .transpose(-1, -2)
            .contiguous()
        )
        out[ir] = torch.cat([out[ir], chunk], dim=-2) if ir in out else chunk
        offset += size
    return out


def scalar_activation(
    x: dict[Irrep, Tensor],
    even_act: Callable[[Tensor], Tensor],
    odd_act: Callable[[Tensor], Tensor] | None = None,
) -> dict[Irrep, Tensor]:
    """Apply activation to scalar irreps only."""
    even_act_n = normalize_function(even_act)
    odd_act_n = normalize_function(odd_act) if odd_act is not None else None

    def apply(ir: Irrep, v: Tensor) -> Tensor:
        if ir.is_scalar():
            if ir.p == 1:
                return even_act_n(v)
            return odd_act_n(v) if odd_act_n is not None else v
        return v

    return {ir: apply(ir, v) for ir, v in x.items()}


# ---------------------------------------------------------------------------
# Basic building blocks (dict[Irrep, Tensor] -> dict[Irrep, Tensor])
# ---------------------------------------------------------------------------

class SeparatedLayerNorm(nn.Module):
    """Layer normalisation for ``dict[Irrep, Tensor]`` features.

    Mirrors ``bam_torch_cue.utils.separated_layer_norm`` but operates on the
    per-irrep dict representation used throughout this module.

    * **Scalars** (``0e``): mean-centred + RMS normalisation (each scalar
      group independently).
    * **Non-scalars**: shared RMS normalisation across *all* non-scalar
      components (no mean centring — it would break equivariance).
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: dict[Irrep, Tensor]) -> dict[Irrep, Tensor]:
        out: dict[Irrep, Tensor] = {}

        # ---- non-scalar: shared RMS ----
        ns_sq_parts: list[Tensor] = []
        ns_count = 0
        for ir, v in x.items():
            if not ir.is_scalar():
                # v: (batch, mul, ir.dim)
                ns_sq_parts.append(v.pow(2).sum(dim=(-2, -1)))  # (batch,)
                ns_count += v.shape[-2] * v.shape[-1]

        if ns_count > 0:
            ns_sq_sum = sum(ns_sq_parts)  # (batch,)
            ns_rms_inv = torch.rsqrt(ns_sq_sum / ns_count + self.eps)  # (batch,)
            ns_rms_inv = ns_rms_inv[:, None, None]  # broadcast over (mul, ir.dim)

        for ir, v in x.items():
            if ir.is_scalar() and ir.p == 1:
                # 0e: mean-centred + RMS (over the mul axis)
                mean = v.mean(dim=-2, keepdim=True)       # (batch, 1, 1)
                centered = v - mean
                rms_inv = torch.rsqrt(
                    centered.pow(2).mean(dim=-2, keepdim=True) + self.eps
                )
                out[ir] = centered * rms_inv
            elif ir.is_scalar():
                # 0o (rare): pass through unchanged
                out[ir] = v
            else:
                # non-scalar: shared RMS (no mean centring)
                out[ir] = v * ns_rms_inv

        return out


class MLP(nn.Module):
    """Simple MLP with 1/sqrt(fan_in) normalization at forward time.

    Mirrors the convention used throughout the model: unit-variance Gaussian
    init plus forward-time 1/sqrt(fan_in) scaling, giving NTK-style
    normalization that is robust to width changes.
    """

    def __init__(
        self,
        layers: list[int],
        activation: Callable[[Tensor], Tensor],
        use_bias: bool = False,
        *,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.use_bias = use_bias
        self.activation = normalize_function(activation)

        weights, biases = [], []
        for d_in, d_out in zip(layers[:-1], layers[1:]):
            weights.append(nn.Parameter(torch.randn(d_in, d_out, dtype=dtype)))
            if use_bias:
                biases.append(nn.Parameter(torch.zeros(d_out, dtype=dtype)))
        self.weights = nn.ParameterList(weights)
        self.biases = nn.ParameterList(biases) if use_bias else None

    def forward(self, x: Tensor) -> Tensor:
        n_layers = len(self.weights)
        for i, w in enumerate(self.weights):
            x = (x @ w) / (w.shape[0] ** 0.5)
            if self.use_bias:
                x = x + self.biases[i]
            if i < n_layers - 1:
                x = self.activation(x)
        return x


class IrrepsLinear(nn.Module):
    """Equivariant linear layer on dict[Irrep, Tensor].

    Mixes channels within each irrep type (no cross-irrep coupling); the
    equivariance is manifest from the per-irrep block structure.
    """

    def __init__(
        self,
        irreps_in: cue.Irreps,
        irreps_out: cue.Irreps,
        *,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.irreps_out = irreps_out
        weights: dict[str, nn.Parameter] = {}
        for mul_out, ir in irreps_out:
            in_mul = sum(m for m, i in irreps_in if i == ir)
            if in_mul > 0:
                weights[str(ir)] = nn.Parameter(
                    torch.randn(in_mul, mul_out, dtype=dtype)
                )
        self.weights = nn.ParameterDict(weights)

    def forward(self, x: dict[Irrep, Tensor]) -> dict[Irrep, Tensor]:
        out: dict[Irrep, Tensor] = {}
        for mul_out, ir in self.irreps_out:
            key = str(ir)
            if key in self.weights and ir in x:
                x_ir = x[ir]  # (batch, in_mul, ir.dim)
                in_mul = x_ir.shape[1]
                w = self.weights[key]
                out[ir] = torch.einsum("nmi,mo->noi", x_ir, w) / (in_mul ** 0.5)
        return out


class SphericalHarmonics(nn.Module):
    """Spherical harmonics up to max_ell, returning dict[Irrep, Tensor].

    Thin wrapper around `cuet.SegmentedPolynomial`; each output irrep is
    cue.O3(L, (-1)**L), shape (batch, 1, 2L+1).
    """

    def __init__(self, max_ell: int):
        super().__init__()
        ls = list(range(max_ell + 1))
        self.irreps_out = cue.Irreps(
            cue.O3, [(1, cue.O3(L, (-1) ** L)) for L in ls]
        )
        e = cue.descriptors.spherical_harmonics(cue.O3(1, -1), ls)
        self.poly = cuet.SegmentedPolynomial(e.polynomial, method=_poly_method())

    def forward(self, vecs: Tensor) -> dict[Irrep, Tensor]:
        # vecs: (batch, 3) -> flat (batch, irreps_out.dim)
        [flat] = self.poly([vecs])
        return flat_to_irreps_dict(flat, self.irreps_out)


# ---------------------------------------------------------------------------
# Gated linear layers (outputs scalars + gates + nonscalars)
# ---------------------------------------------------------------------------

class GatedLinear(nn.Module):
    """Linear layer outputting to gated irreps structure (scalars + nonscalars + gates)."""

    def __init__(
        self,
        irreps_in: cue.Irreps,
        num_scalars: int,
        num_gates: int,
        irreps_nonscalar: cue.Irreps,
        *,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.irreps_nonscalar = irreps_nonscalar

        in_0e = sum(m for m, ir in irreps_in if ir == cue.O3(0, 1))
        self.w_0e_scalar = nn.Parameter(torch.randn(in_0e, num_scalars, dtype=dtype))
        self.w_0e_gates = nn.Parameter(torch.randn(in_0e, num_gates, dtype=dtype))

        w_nonscalar: dict[str, nn.Parameter] = {}
        for mul, ir in irreps_nonscalar:
            in_mul = sum(m for m, i in irreps_in if i == ir)
            if in_mul > 0:
                w_nonscalar[str(ir)] = nn.Parameter(
                    torch.randn(in_mul, mul, dtype=dtype)
                )
        self.w_nonscalar = nn.ParameterDict(w_nonscalar)

    def forward(self, x: dict[Irrep, Tensor]) -> dict[Irrep, Tensor]:
        result: dict[Irrep, Tensor] = {}
        ir_0e = cue.O3(0, 1)
        if ir_0e in x:
            x_0e = x[ir_0e][..., 0]  # (batch, in_mul)
            norm = x_0e.shape[-1] ** 0.5
            scalars = (x_0e @ self.w_0e_scalar) / norm
            gates = (x_0e @ self.w_0e_gates) / norm
            result[ir_0e] = torch.cat([scalars, gates], dim=-1)[..., None]

        w_keys = set(self.w_nonscalar.keys())
        for mul, ir in self.irreps_nonscalar:
            if str(ir) in w_keys and ir in x:
                x_ir = x[ir]
                w = self.w_nonscalar[str(ir)]
                result[ir] = torch.einsum("bni,no->boi", x_ir, w) / (x_ir.shape[1] ** 0.5)
        return result


class GatedIndexedLinear(nn.Module):
    """Indexed linear layer outputting to gated irreps structure.

    Different weights per species; used for the species-indexed skip
    connection. Weights are expanded with `repeat_interleave` based on the
    species count for each atom (species are assumed pre-sorted).
    """

    def __init__(
        self,
        irreps_in: cue.Irreps,
        num_scalars: int,
        num_gates: int,
        num_indices: int,
        irreps_nonscalar: cue.Irreps,
        *,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.num_scalars = num_scalars
        self.num_gates = num_gates
        self.num_indices = num_indices
        self.irreps_nonscalar = irreps_nonscalar
        self.dtype = dtype

        in_0e = sum(m for m, ir in irreps_in if ir == cue.O3(0, 1))
        self.in_0e = in_0e
        self.w_0e_scalar = nn.Parameter(
            torch.randn(num_indices, in_0e * num_scalars, dtype=dtype)
        )
        self.w_0e_gates = nn.Parameter(
            torch.randn(num_indices, in_0e * num_gates, dtype=dtype)
        )

        w_nonscalar: dict[str, nn.Parameter] = {}
        self._in_mul_nonscalar: dict[str, int] = {}
        self._mul_out_nonscalar: dict[str, int] = {}
        for mul_out, ir in irreps_nonscalar:
            in_mul = sum(m for m, i in irreps_in if i == ir)
            if in_mul > 0:
                w_nonscalar[str(ir)] = nn.Parameter(
                    torch.randn(num_indices, in_mul * mul_out, dtype=dtype)
                )
                self._in_mul_nonscalar[str(ir)] = in_mul
                self._mul_out_nonscalar[str(ir)] = mul_out
        self.w_nonscalar = nn.ParameterDict(w_nonscalar)

    def forward(
        self, x: dict[Irrep, Tensor], species_counts: Tensor
    ) -> dict[Irrep, Tensor]:
        ir_0e = cue.O3(0, 1)
        if ir_0e not in x:
            return {}

        x_0e = x[ir_0e]
        batch_size, in_0e = x_0e.shape[0], x_0e.shape[1]
        x_0e_flat = x_0e[..., 0]  # (batch, in_mul)

        def expand(w: Tensor, in_mul: int, out_mul: int) -> Tensor:
            # (num_indices, in_mul * out_mul) -> (batch, in_mul, out_mul)
            w = w.reshape(self.num_indices, in_mul, out_mul)
            return torch.repeat_interleave(
                w, species_counts, dim=0, output_size=batch_size
            )

        norm_0e = in_0e ** 0.5
        w_s = expand(self.w_0e_scalar, in_0e, self.num_scalars)
        w_g = expand(self.w_0e_gates, in_0e, self.num_gates)
        scalars = torch.einsum("bi,bio->bo", x_0e_flat, w_s) / norm_0e
        gates = torch.einsum("bi,bio->bo", x_0e_flat, w_g) / norm_0e
        result: dict[Irrep, Tensor] = {
            ir_0e: torch.cat([scalars, gates], dim=-1)[..., None]
        }

        w_keys = set(self.w_nonscalar.keys())
        for mul_out, ir in self.irreps_nonscalar:
            ir_key = str(ir)
            if ir_key in w_keys and ir in x:
                x_ir = x[ir]
                in_mul = x_ir.shape[1]
                w_ir = expand(self.w_nonscalar[ir_key], in_mul, mul_out)
                result[ir] = torch.einsum("bid,bio->bod", x_ir, w_ir) / (in_mul ** 0.5)
            else:
                result[ir] = torch.zeros(
                    batch_size,
                    mul_out,
                    ir.dim,
                    dtype=self.dtype,
                    device=x_0e.device,
                )
        return result


# ---------------------------------------------------------------------------
# Channelwise tensor product with gather/scatter fused in
# ---------------------------------------------------------------------------

class MessagePassing(nn.Module):
    """Channelwise tensor product with gather on senders and scatter on receivers.

    The `cuet.SegmentedPolynomial` call fuses
        msg_{ij} = TP(node_feats[senders_{ij}], sph_{ij}; weights_{ij})
        out[receivers_{ij}] += msg_{ij}
    into a single kernel call via `input_indices` and `output_indices`.
    """

    def __init__(
        self,
        irreps_in: cue.Irreps,
        irreps_sh: cue.Irreps,
        irreps_out: cue.Irreps,
        epsilon: float,
        *,
        name: str = "",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.name = name
        e = (
            cue.descriptors.channelwise_tensor_product(
                irreps_in, irreps_sh, irreps_out, True
            )
            * epsilon
        )
        # Cache dims / irreps before the descriptor gets wrapped.
        self.weight_numel = e.inputs[0].dim
        self.irreps_in = irreps_in
        self.irreps_sh = irreps_sh
        self.irreps_out = e.outputs[0].irreps
        self.poly = cuet.SegmentedPolynomial(e.polynomial, method=_poly_method())

    def forward(
        self,
        weights: Tensor,
        node_feats: dict[Irrep, Tensor],
        sph: dict[Irrep, Tensor],
        senders: Tensor,
        receivers: Tensor,
        num_nodes: int,
    ) -> dict[Irrep, Tensor]:
        # Flatten the dict operands to mul-ir layout for the polynomial.
        x1_flat = irreps_dict_to_flat(node_feats, self.irreps_in)  # (N, irreps_in.dim)
        x2_flat = irreps_dict_to_flat(sph, self.irreps_sh)  # (E, irreps_sh.dim)

        # Shape-hint tensor tells the kernel the batch size of the indexed output.
        # Only shape[0] is read, but the API asserts ndim == 2.
        shape_hint = weights.new_empty(num_nodes, 1)

        [out_flat] = self.poly(
            [weights, x1_flat, x2_flat],
            input_indices={1: senders},
            output_indices={0: receivers},
            output_shapes={0: shape_hint},
        )
        # out_flat: (num_nodes, self.irreps_out.dim)
        return flat_to_irreps_dict(out_flat, self.irreps_out)


# ---------------------------------------------------------------------------
# Full NequIP interaction layer
# ---------------------------------------------------------------------------

class NEQUIPLayer(nn.Module):
    """Single NequIP interaction layer with gated nonlinearity.

    Accepts precomputed spherical harmonics (dict[Irrep, Tensor]) and
    radial basis (E, radial_dim) so that those quantities are computed
    once in the model and shared across layers.
    """

    def __init__(
        self,
        input_irreps: cue.Irreps,
        output_irreps: cue.Irreps,
        num_species: int,
        max_ell: int,
        radial_dim: int,
        epsilon: float,
        mlp_hidden: int,
        mlp_layers: int,
        even_activation: Callable[[Tensor], Tensor],
        odd_activation: Callable[[Tensor], Tensor],
        gate_activation: Callable[[Tensor], Tensor],
        mlp_activation: Callable[[Tensor], Tensor],
        name: str = "",
        *,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.even_activation = even_activation
        self.odd_activation = odd_activation
        self.gate_activation_fn = normalize_function(gate_activation)

        sph_irreps = cue.Irreps(
            cue.O3, [(1, cue.O3(L, (-1) ** L)) for L in range(max_ell + 1)]
        )
        filtered = output_irreps.filter(keep="0e+0o+1o+1e+2e+2o+3o+3e").regroup()
        irreps_nonscalar = filtered.filter(drop="0e+0o")
        self.num_scalars = sum(m for m, ir in filtered if ir == cue.O3(0, 1))
        self.num_nonscalar = irreps_nonscalar.num_irreps
        conv_out = (output_irreps + cue.Irreps(cue.O3, "0e")).regroup()

        self.layer_norm = SeparatedLayerNorm()
        self.linear_up = IrrepsLinear(input_irreps, input_irreps, dtype=dtype)
        self.message = MessagePassing(
            input_irreps,
            sph_irreps,
            conv_out,
            epsilon,
            name=f"{name}_tensor_product",
            dtype=dtype,
        )
        self.radial_mlp = MLP(
            [radial_dim] + [mlp_hidden] * mlp_layers + [self.message.weight_numel],
            mlp_activation,
            use_bias=False,
            dtype=dtype,
        )
        self.linear_down = GatedLinear(
            self.message.irreps_out,
            self.num_scalars,
            self.num_nonscalar,
            irreps_nonscalar,
            dtype=dtype,
        )
        self.linear_skip = GatedIndexedLinear(
            input_irreps,
            self.num_scalars,
            self.num_nonscalar,
            num_species,
            irreps_nonscalar,
            dtype=dtype,
        )

    def forward(
        self,
        sph: dict[Irrep, Tensor],
        radial_basis_vals: Tensor,
        node_feats: dict[Irrep, Tensor],
        species_counts: Tensor,
        senders: Tensor,
        receivers: Tensor,
        num_nodes: int,
    ) -> dict[Irrep, Tensor]:
        node_feats_norm = self.layer_norm(node_feats)
        h = self.linear_up(node_feats_norm)

        radial_weights = self.radial_mlp(radial_basis_vals)

        h = self.message(radial_weights, h, sph, senders, receivers, num_nodes)
        h = self.linear_down(h)
        skip = self.linear_skip(node_feats, species_counts)
        h = irreps_add(h, skip)

        # Gated activation: split 0e into scalars and gates, multiply nonscalars.
        if self.num_nonscalar > 0:
            ir_0e = cue.O3(0, 1)
            h_0e = h[ir_0e]
            scalars = h_0e[:, : self.num_scalars, :]
            gates = h_0e[:, self.num_scalars :, :]
            gates = self.gate_activation_fn(gates)

            gate_idx = 0
            gated: dict[Irrep, Tensor] = {ir_0e: scalars} if self.num_scalars > 0 else {}
            for ir, v in h.items():
                if not ir.is_scalar():
                    mul = v.shape[1]
                    gated[ir] = v * gates[:, gate_idx : gate_idx + mul, :]
                    gate_idx += mul
            h = gated

        return scalar_activation(h, self.even_activation, self.odd_activation)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class NEQUIPModel(nn.Module):
    """NequIP model for energies, forces, and stress.

    Args match bam_torch_cue/nequip.py:Nequip — uses the shared
    ``compute_node_edge_features`` and ``energy_forces_stress`` utilities
    so that force / stress computation is handled uniformly.
    """

    def __init__(
        self,
        n_species: int,
        lmax: int = 3,
        cutoff: float = 5.0,
        hidden_irreps: str = "64x0e + 64x1o + 64x2e",
        n_layers: int = 5,
        radial_basis_size: int = 8,
        radial_mlp_size: int = 64,
        radial_mlp_layers: int = 2,
        radial_polynomial_p: float = 2.0,
        mlp_init_scale: float = 4.0,
        shift: float = 0.0,
        scale: float = 1.0,
        avg_n_neighbors: float = 25.0,
        atom_energies: Optional[Sequence[float]] = None,
        l_train: bool = True,
        periodic: bool = True,
        *,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.lmax = lmax
        self.cutoff = cutoff
        self.periodic = periodic
        self.n_species = n_species
        self.radial_basis_size = radial_basis_size
        self.radial_polynomial_p = radial_polynomial_p
        self.l_train = l_train
        self.dtype = dtype

        self.shift = nn.Parameter(torch.tensor(shift))
        self.scale = nn.Parameter(torch.tensor(scale))
        self.avg_n_neighbors = avg_n_neighbors

        if atom_energies is not None:
            self.register_buffer(
                "atom_energies",
                torch.tensor(atom_energies, dtype=torch.float32),
            )
        else:
            self.register_buffer(
                "atom_energies", torch.zeros(n_species)
            )

        # Spherical harmonics module (created once, reused every forward pass)
        self.sh_module = cuet.SphericalHarmonics(
            list(range(lmax + 1)), normalize=True,
        )

        output_irreps = cue.Irreps(
            cue.O3,
            [(int(s.split("x")[0]), cue.O3.from_string(s.split("x")[1]))
             for s in hidden_irreps.replace(" ", "").split("+")],
        )
        self.output_irreps = output_irreps

        # Determine num_features from the 0e multiplicity for embedding
        num_features = sum(m for m, ir in output_irreps if ir == cue.O3(0, 1))
        self.num_features = num_features
        self.embedding = nn.Parameter(
            torch.randn(n_species, num_features, dtype=dtype)
        )

        layers: list[NEQUIPLayer] = []
        for i in range(n_layers):
            layers.append(
                NEQUIPLayer(
                    input_irreps=output_irreps.filter(keep="0e")
                    if i == 0
                    else output_irreps,
                    output_irreps=output_irreps,
                    num_species=n_species,
                    max_ell=lmax,
                    radial_dim=radial_basis_size,
                    epsilon=1.0 / avg_n_neighbors,
                    mlp_hidden=radial_mlp_size,
                    mlp_layers=radial_mlp_layers,
                    even_activation=F.silu,
                    odd_activation=torch.sigmoid, #torch.tanh,
                    gate_activation=F.silu,
                    mlp_activation=F.silu,
                    name=f"layer_{i}",
                    dtype=dtype,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.readout = IrrepsLinear(
            output_irreps, cue.Irreps(cue.O3, "1x0e"), dtype=dtype
        )

        # SH irreps for converting flat SH tensor → dict[Irrep, Tensor]
        self.sh_irreps = cue.Irreps(
            cue.O3, [(1, cue.O3(L, (-1) ** L)) for L in range(lmax + 1)]
        )

    # -----------------------------------------------------------------
    # Core forward path: positions → per-atom energies (for autograd)
    # -----------------------------------------------------------------

    def node_energies(
        self,
        positions: Tensor,
        cell: Optional[Tensor],
        cutoff: Optional[Tensor],
        data: TorchGraphData,
    ) -> Dict[str, Tensor]:
        """Compute per-atom energies from positions (autograd-friendly).

        Signature matches ``bam_torch_cue.nequip.Nequip.node_energies`` so
        it can be passed directly to ``energy_forces_stress``.
        """
        node_features, radial_basis_vals, sh_flat = compute_node_edge_features(
            positions=positions,
            cell=cell,
            cutoff=cutoff,
            data=data,
            n_species=self.n_species,
            model_cutoff=self.cutoff,
            radial_basis_size=self.radial_basis_size,
            radial_polynomial_p=self.radial_polynomial_p,
            lmax=self.lmax,
            periodic=self.periodic,
            sh_module=self.sh_module,
        )

        # Convert flat SH (E, sh_dim) → dict[Irrep, Tensor] for the layers
        sph = flat_to_irreps_dict(sh_flat, self.sh_irreps)

        species = data.nodes["species"]
        senders = data.senders
        receivers = data.receivers

        # Sort atoms by species (required by indexed-weight kernels).
        perm = torch.argsort(species, stable=True)
        species_sorted = species[perm]
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(
            perm.shape[0], device=perm.device, dtype=perm.dtype
        )
        senders_sorted = inv_perm[senders]
        receivers_sorted = inv_perm[receivers]

        num_nodes = species_sorted.shape[0]
        species_counts = torch.bincount(
            species_sorted, minlength=self.n_species
        ).to(torch.int64)

        # Species embedding → initial node features (scalar 0e only)
        node_feats_flat = torch.repeat_interleave(
            self.embedding, species_counts, dim=0, output_size=num_nodes
        )
        node_feats: dict[Irrep, Tensor] = {
            cue.O3(0, 1): (node_feats_flat / (self.n_species ** 0.5))[
                :, :, None
            ]
        }

        for layer in self.layers:
            node_feats = layer(
                sph, radial_basis_vals, node_feats, species_counts,
                senders_sorted, receivers_sorted, num_nodes,
            )

        out = self.readout(node_feats)
        atom_energies_sorted = out[next(iter(out))].squeeze(-1).squeeze(-1)

        # Un-permute back to original atom order, apply scale / shift
        ne = atom_energies_sorted[inv_perm] * self.scale + self.shift

        return {
            "node_energies": ne,
        }

    def forward(self, data: TorchGraphData) -> Dict[str, Tensor]:
        """Compute energy, forces, and stress via autograd."""
        return energy_forces_stress(
            self.node_energies,
            data,
            self.periodic,
            self.l_train,
            self.atom_energies,
            create_graph=self.training,
        )


if __name__ == "__main__":
    import pickle
    import numpy as np
    from bam_torch.atom_energies import ATOM_ENERGIES
    from bam_torch.data import GraphData, batch_graphs

    def jraph_to_graphdata(g) -> GraphData:
        """Convert a jraph.GraphsTuple (from existing pkl files) to GraphData."""
        return GraphData(
            n_node=np.asarray(g.n_node),
            n_edge=np.asarray(g.n_edge),
            nodes={k: np.asarray(v) for k, v in g.nodes.items()},
            edges={k: np.asarray(v) for k, v in g.edges.items()},
            senders=np.asarray(g.senders),
            receivers=np.asarray(g.receivers),
            globals={k: np.asarray(v) for k, v in g.globals.items()},
        )

    print("=" * 70)
    print("NequIP (PyTorch + cuequivariance, dict[Irrep, Tensor] impl)")
    print("=" * 70)

    with open("omat24_val_0.pkl", "rb") as f:
        graphset = pickle.load(f)

    data_np = jraph_to_graphdata(graphset[0])
    data = batch_graphs([data_np]).to(torch.device("cpu"))

    model = NEQUIPModel(
        n_species=len(ATOM_ENERGIES),
        lmax=3,
        hidden_irreps="64x0e + 64x1o + 64x2e",
        n_layers=3,
        radial_basis_size=8,
        radial_mlp_size=64,
        radial_mlp_layers=2,
        mlp_init_scale=4.0,
        shift=0.0,
        scale=1.0,
        avg_n_neighbors=25.0,
        atom_energies=ATOM_ENERGIES.tolist(),
        periodic=True,
        l_train=True,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nNumber of parameters: {n_params:,}")

    results = model(data)

    energies = results["energy"]
    forces = results["forces"]
    stress = results["stress"]

    print(f"Energies shape : {energies.shape}")
    print(f"Forces shape   : {forces.shape}")
    print(f"Total energy   : {energies[0].item():.6f} eV")
    print(f"Force magnitude: {forces.norm(dim=-1).mean().item():.6f} eV/Å")
    print(f"Stress (Voigt) : {stress[0].detach().numpy()}")
    print(f"\nTarget energy  : {data_np.globals['energy'][0]:.6f} eV")
    print(f"Target stress  : {data_np.globals['stress'][0]} eV/Å³")
