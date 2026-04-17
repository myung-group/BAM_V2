"""Shared utilities for equivariant ML potential models (PyTorch).

Replaces bam_omat24/utils.py.
Uses cuequivariance / cuequivariance_torch instead of e3nn.
"""

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import cuequivariance as cue
import cuequivariance_torch as cuet

from bam_torch.data import TorchGraphData

# Layout used throughout — ir_mul is cuequivariance's native layout and
# works on both CPU and CUDA (mul_ir requires CUDA-only transpose kernels).
LAYOUT = cue.ir_mul


# =============================================================================
# Irreps helpers (replacing e3nn o3.Irreps convenience methods)
# =============================================================================

def cue_irreps(s: str) -> cue.Irreps:
    """Create O3 irreps from a string like '64x0e + 64x1o + 64x2e'."""
    return cue.Irreps("O3", s)


def spherical_harmonics_irreps(lmax: int) -> cue.Irreps:
    """Equivalent to e3nn's ``o3.Irreps.spherical_harmonics(lmax)``."""
    return cue.Irreps("O3", [(1, cue.O3(l, (-1) ** l)) for l in range(lmax + 1)])


def filter_scalar_irreps(irreps: cue.Irreps) -> cue.Irreps:
    """Keep only scalar (0e) irreps. Equivalent to ``irreps.filter(keep=['0e'])``."""
    return cue.Irreps("O3", [(mi.mul, mi.ir) for mi in irreps if is_scalar_ir(mi.ir)])


def is_scalar_ir(ir) -> bool:
    """Check if an irrep is scalar (l=0, even parity)."""
    return ir.l == 0 and ir.p == 1


# =============================================================================
# Channelwise CG tensor product (replaces unweighted e3nn TensorProduct)
# =============================================================================

def build_cg_tp(
    irreps_in: cue.Irreps,
    sh_irreps: cue.Irreps,
    filter_irreps: cue.Irreps,
) -> Tuple[cuet.ChannelWiseTensorProduct, cue.Irreps]:
    """Build a channelwise CG ``ChannelWiseTensorProduct`` with external weights.

    The returned TP absorbs both the CG coupling **and** per-irrep-instance
    radial scaling into a single operation::

        tp, tp_irreps = build_cg_tp(input_irreps, sh_irreps, output_irreps)
        radial_weights = radial_mlp(radial_basis)   # (E, tp.weight_numel)
        out = tp(features[senders], sh, weight=radial_weights)

    This replaces the old pattern of ``e3nn.TensorProduct`` + ``scale_irreps_by_scalar``.
    """
    filter_irs = list(set(mi.ir for mi in filter_irreps))
    tp = cuet.ChannelWiseTensorProduct(
        irreps_in,
        sh_irreps,
        filter_irreps_out=filter_irs,
        shared_weights=False,
        internal_weights=False,
        layout=LAYOUT,
    )
    return tp, tp.irreps_out


# =============================================================================
# Radial basis and cutoff
# =============================================================================

def bessel_basis(
    x: torch.Tensor,
    num_basis: int,
    r_max: torch.Tensor,
) -> torch.Tensor:
    """Bessel radial basis functions.

    Args:
        x:         (E,) distances.
        num_basis: Number of basis functions.
        r_max:     (E,) per-edge cutoff radii.

    Returns:
        (E, num_basis) basis values.
    """
    r_max = r_max.unsqueeze(-1)                                  # (E,1)
    x     = x.unsqueeze(-1)                                      # (E,1)
    n     = torch.arange(1, num_basis + 1, dtype=x.dtype, device=x.device)  # (B,)
    prefactor = 2.0 / r_max
    arg = n * math.pi * x / r_max                                # (E,B)
    # Safe: avoid 0/0 at x=0
    return prefactor * torch.where(
        x == 0.0,
        n * math.pi / r_max,
        torch.sin(arg) / x,
    )


def polynomial_cutoff(
    x: torch.Tensor,
    r_max: torch.Tensor,
    p: float,
) -> torch.Tensor:
    """Smooth polynomial envelope (goes to 0 at r_max)."""
    xn = x / r_max
    out = (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * xn ** p
        + p * (p + 2.0) * xn ** (p + 1.0)
        - (p * (p + 1.0) / 2.0) * xn ** (p + 2.0)
    )
    return out * (xn < 1.0).float()


# =============================================================================
# Separated layer normalisation
# =============================================================================

def build_norm_info(irreps: cue.Irreps) -> Dict:
    """Precompute layout info for :func:`separated_layer_norm`.

    Returns a dict with:
        scalar_ranges:   list of (start, end) for each scalar (0e) group.
        nonscalar_range: (start, end) covering all non-scalar components,
                         or None if there are no non-scalars.
        ns_count:        total number of non-scalar components.
    """
    scalar_ranges: List[Tuple[int, int]] = []
    ns_start: Optional[int] = None
    ns_end = 0
    ns_count = 0
    pos = 0
    for mi in irreps:
        size = mi.mul * mi.ir.dim
        if is_scalar_ir(mi.ir):
            scalar_ranges.append((pos, pos + size))
        else:
            if ns_start is None:
                ns_start = pos
            ns_end = pos + size
            ns_count += size
        pos += size
    return {
        'scalar_ranges': scalar_ranges,
        'nonscalar_range': (ns_start, ns_end) if ns_start is not None else None,
        'ns_count': ns_count,
    }


def separated_layer_norm(
    features: torch.Tensor,
    norm_info: Dict,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Layer-norm that handles scalars and non-scalars separately.

    Scalars (0e): mean-centred + RMS normalisation.
    Non-scalars: shared RMS normalisation (no mean centring — not invariant).

    Args:
        features:  (N, irreps.dim)
        norm_info: Precomputed layout from :func:`build_norm_info`.

    Returns:
        (N, irreps.dim) normalised features.
    """
    out = features.clone()
    ns_range = norm_info['nonscalar_range']
    ns_count = norm_info['ns_count']

    # Non-scalar: shared RMS normalisation
    if ns_range is not None and ns_count > 0:
        ns_start, ns_end = ns_range
        ns_part = features[:, ns_start:ns_end]
        ns_rms_inv = torch.rsqrt(
            (ns_part ** 2).sum(dim=-1, keepdim=True) / ns_count + eps
        )
        out[:, ns_start:ns_end] = ns_part * ns_rms_inv

    # Scalar groups: independent mean-centring + RMS
    for start, end in norm_info['scalar_ranges']:
        chunk = features[:, start:end]
        mean = chunk.mean(dim=-1, keepdim=True)
        centered = chunk - mean
        rms_inv = torch.rsqrt((centered ** 2).mean(dim=-1, keepdim=True) + eps)
        out[:, start:end] = centered * rms_inv

    return out


# =============================================================================
# MLP
# =============================================================================

class MLP(nn.Module):
    """Multi-layer perceptron with SiLU activations (no activation on last layer)."""

    def __init__(
        self,
        sizes: Sequence[int],
        activation: Callable = F.silu,
        use_bias: bool = False,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.activation = activation
        layers = []
        for i in range(len(sizes) - 1):
            is_last = (i == len(sizes) - 2)
            scale = 1.0 if is_last else init_scale
            lin = nn.Linear(sizes[i], sizes[i + 1], bias=use_bias)
            # He-style init with optional scale
            nn.init.normal_(lin.weight, std=math.sqrt(scale / sizes[i]))
            if use_bias and lin.bias is not None:
                nn.init.zeros_(lin.bias)
            layers.append(lin)
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


# =============================================================================
# Scatter utilities
# =============================================================================

def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    """Sum src values into output of shape (dim_size, *src.shape[1:]) indexed by index."""
    shape = (dim_size,) + src.shape[1:]
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out.scatter_add_(0, idx, src)
    return out


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    sums = scatter_sum(src, index, dim_size)
    counts = scatter_sum(torch.ones(src.shape[0], device=src.device), index, dim_size)
    counts = counts.clamp(min=1).view(-1, *([1] * (src.dim() - 1)))
    return sums / counts


def node_graph_idx(n_node: torch.Tensor) -> torch.Tensor:
    """Map each node to its graph index.

    Args:
        n_node: (n_graphs,) nodes per graph.

    Returns:
        (total_nodes,) graph index for each node.
    """
    n_graphs = n_node.shape[0]
    graph_idx = torch.arange(n_graphs, device=n_node.device)
    return graph_idx.repeat_interleave(n_node)


# =============================================================================
# Compute node and edge features
# =============================================================================

def compute_node_edge_features(
    positions: torch.Tensor,
    cell: Optional[torch.Tensor],
    cutoff: Optional[torch.Tensor],
    data: TorchGraphData,
    n_species: int,
    model_cutoff: float,
    radial_basis_size: int,
    radial_polynomial_p: float,
    lmax: int,
    periodic: bool,
    sh_module: cuet.SphericalHarmonics,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute species one-hot, radial basis, and spherical harmonics.

    Args:
        positions: (N, 3) — may have requires_grad=True for force computation.
        cell:      (G, 3, 3) or None.
        cutoff:    (G,) or None.
        data:      TorchGraphData (topology, species, shifts).
        sh_module: ``cuet.SphericalHarmonics`` instance.

    Returns:
        node_features: (N, n_species) one-hot species embedding.
        radial_basis:  (E, radial_basis_size) radial basis values.
        sh:            (E, sh_dim) spherical harmonics.
    """
    # One-hot species features
    node_features = F.one_hot(data.nodes['species'], n_species).float()

    if periodic:
        # Cartesian displacement = pos[src] - (pos[dst] + S @ cell)
        # Each edge belongs to the graph of its source atom; expand cell to edges
        Sij = data.edges['Sij']                                      # (E, 3)
        num_edges = Sij.shape[0]
        # Expand cell per edge: use n_edge to replicate cell rows
        cell_per_edge = cell.repeat_interleave(data.n_edge, dim=0)   # (E, 3, 3)
        shift_cart = (Sij.unsqueeze(-2) @ cell_per_edge).squeeze(-2) # (E, 3)
        cutoff_per_edge = cutoff.repeat_interleave(data.n_edge)       # (E,)
        r = positions[data.senders] - (positions[data.receivers] + shift_cart)
    else:
        r = positions[data.senders] - positions[data.receivers]
        num_edges = data.senders.shape[0]
        cutoff_per_edge = torch.full(
            (num_edges,), model_cutoff, dtype=positions.dtype, device=positions.device
        )

    cutoff_per_edge = cutoff_per_edge.clamp(min=1.0)

    # Safe norm
    r_sq = (r ** 2).sum(dim=-1)
    r_norm = torch.where(r_sq == 0.0, torch.zeros_like(r_sq), r_sq.sqrt())

    radial_basis = (
        bessel_basis(r_norm, radial_basis_size, cutoff_per_edge)
        * polynomial_cutoff(r_norm, cutoff_per_edge, radial_polynomial_p).unsqueeze(-1)
    )  # (E, radial_basis_size)

    # Spherical harmonics (cuequivariance normalises the direction vector internally)
    sh = sh_module(r)  # (E, sh_dim)

    return node_features, radial_basis, sh


# =============================================================================
# Energy / forces / stress
# =============================================================================

def energy_forces_stress(
    node_energy_fn: Callable,
    data: TorchGraphData,
    periodic: bool,
    l_train: bool,
    atom_energies: torch.Tensor,
    create_graph: bool = True,
) -> Dict[str, torch.Tensor]:
    """Compute graph energies, forces, and stress via autograd.

    Analogous to bam_omat24/utils.py:energy_forces_stress.

    Args:
        node_energy_fn: callable(positions, cell, cutoff, data) → dict with
                        at least 'node_energies' (N,) and optional variation keys.
        data:           TorchGraphData.
        periodic:       Whether to compute stress.
        l_train:        Training mode — if True, skip atom energy shift.
        atom_energies:  (n_species,) reference energies.
        create_graph:   If True (default), build graph of the backward pass for
                        second-order gradients (needed during training).  Set to
                        False during evaluation to save ~50% GPU memory.

    Returns:
        Dict with 'energy', 'forces', 'stress', 'force_variation',
        'energy_variation', 'stress_variation'.
    """
    n_graphs = data.n_node.shape[0]
    node_gr_idx = node_graph_idx(data.n_node)            # (N,)

    # Clone positions / cell as fresh leaf tensors with grad
    positions = data.nodes['positions'].detach().requires_grad_(True)

    if periodic:
        cell   = data.globals['cell'].detach().requires_grad_(True)
        cutoff = data.globals['cutoff']
    else:
        cell = cutoff = None

    # Forward pass
    results = node_energy_fn(positions, cell, cutoff, data)
    ne = results['node_energies'].reshape(-1)  # (N,)

    if not l_train:
        ne = ne + atom_energies[data.nodes['species']].detach()

    total_energy = ne.sum()

    # Graph-level energies via scatter
    graph_energies = scatter_sum(ne, node_gr_idx, n_graphs)  # (G,)

    # Forces: -dE/d positions
    (grad_pos,) = torch.autograd.grad(
        total_energy, positions,
        create_graph=create_graph,
        retain_graph=True,
    )
    grad_pos = torch.nan_to_num(grad_pos, nan=0.0)
    forces = -grad_pos   # (N, 3)

    # Stress
    if periodic:
        (grad_cell,) = torch.autograd.grad(
            total_energy, cell,
            create_graph=create_graph,
        )
        grad_cell = torch.nan_to_num(grad_cell, nan=0.0)

        volume = data.globals['volume'].clamp(min=1e-6)  # (G,)

        # Virial = (cell_grad^T @ cell) + sum_atoms(grad_r ⊗ r)
        stress_cell = grad_cell.transpose(-2, -1) @ data.globals['cell']  # (G,3,3)

        # Outer product contribution
        grad_outer = grad_pos.unsqueeze(-1) * positions.unsqueeze(-2)  # (N,3,3)
        grad_outer = grad_outer.reshape(grad_outer.shape[0], 9)
        stress_grad_full = scatter_sum(grad_outer, node_gr_idx, n_graphs)   # (G,9)
        stress_grad = stress_grad_full.view(n_graphs, 3, 3)

        virial = (stress_cell + stress_grad).reshape(n_graphs, 9)
        # Voigt: xx, yy, zz, yz, xz, xy  →  indices 0,4,8,5,2,1
        virial_voigt = virial[:, [0, 4, 8, 5, 2, 1]]
        stress = virial_voigt / volume.unsqueeze(-1)   # (G,6)
    else:
        stress = torch.zeros(n_graphs, 6, dtype=positions.dtype, device=positions.device)

    # Aggregate variation outputs
    frc_variation    = results.get('frc_variation')    # (N,6) or None
    energy_variation = results.get('energy_variation') # (N,1) or None
    stress_variation = results.get('stress_variation') # (N,6) or None

    if energy_variation is not None:
        energy_variation = scatter_mean(energy_variation, node_gr_idx, n_graphs)  # (G,1)

    if stress_variation is not None:
        stress_variation = scatter_mean(stress_variation, node_gr_idx, n_graphs)  # (G,6)

    return {
        'energy':           graph_energies,
        'forces':           forces,
        'stress':           stress,
        'force_variation':  frc_variation,
        'energy_variation': energy_variation,
        'stress_variation': stress_variation,
    }
