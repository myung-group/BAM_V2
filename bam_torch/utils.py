"""Shared utilities for equivariant ML potential models (PyTorch).

Replaces bam_omat24/utils.py.
"""

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3

from bam_torch.data import TorchGraphData


# =============================================================================
# Unweighted CG tensor product (matches JAX e3nn.tensor_product)
# =============================================================================

def compute_tp_out_irreps(
    irreps_in: o3.Irreps,
    sh_irreps: o3.Irreps,
    filter_irreps: o3.Irreps,
) -> Tuple[o3.Irreps, list]:
    """Compute output irreps and TensorProduct instructions for an
    **unweighted** CG tensor product, replicating JAX's
    ``e3nn.tensor_product(x, sh, filter_ir_out=filter_irreps)``.

    Each valid CG path ``(ir_in ⊗ ir_sh → ir_out)`` becomes one output
    group with ``mul = mul_in`` (since SH always has mul=1) and one
    instruction ``(i_in, i_sh, i_out, 'uvu', False)``.

    Returns:
        tp_irreps:    Output ``Irreps`` of the tensor product.
        instructions: List of TensorProduct instructions.
    """
    filter_set = {ir for _, ir in filter_irreps}
    instructions = []
    tp_out_list: List[Tuple[int, o3.Irrep]] = []
    i_out = 0
    for i1, (mul1, ir1) in enumerate(irreps_in):
        for i2, (_, ir2) in enumerate(sh_irreps):
            for ir_out in ir1 * ir2:
                if ir_out in filter_set:
                    instructions.append((i1, i2, i_out, "uvu", False))
                    tp_out_list.append((mul1, ir_out))
                    i_out += 1
    return o3.Irreps(tp_out_list), instructions


def build_cg_tp(
    irreps_in: o3.Irreps,
    sh_irreps: o3.Irreps,
    filter_irreps: o3.Irreps,
) -> Tuple[o3.TensorProduct, o3.Irreps]:
    """Build an unweighted CG ``TensorProduct`` + its output ``Irreps``.

    Usage::

        tp, tp_irreps = build_cg_tp(input_irreps, sh_irreps, output_irreps)
        out = tp(features[senders], sh)          # (E, tp_irreps.dim)
        out = scale_irreps_by_scalar(out, radial, tp_irreps)
    """
    tp_irreps, instructions = compute_tp_out_irreps(
        irreps_in, sh_irreps, filter_irreps
    )
    tp = o3.TensorProduct(
        irreps_in,
        sh_irreps,
        tp_irreps,
        instructions=instructions,
        shared_weights=True,
        internal_weights=False,
    )
    return tp, tp_irreps


def build_weight_expand_index(irreps: o3.Irreps) -> torch.Tensor:
    """Precompute index tensor for :func:`scale_irreps_by_scalar`.

    Maps each feature component to its corresponding weight index so that
    the per-irrep-instance scalar multiplication becomes a single gather +
    element-wise multiply (no Python loop, no ``torch.cat``).

    Returns:
        Long tensor of shape ``(irreps.dim,)``.
    """
    indices: List[int] = []
    w_pos = 0
    for mul, ir in irreps:
        for m in range(mul):
            indices.extend([w_pos + m] * ir.dim)
        w_pos += mul
    return torch.tensor(indices, dtype=torch.long)


def scale_irreps_by_scalar(
    features: torch.Tensor,
    weights: torch.Tensor,
    expand_idx: torch.Tensor,
) -> torch.Tensor:
    """Multiply each irrep instance by a per-instance scalar weight.

    Mirrors JAX ``IrrepsArray * scalar_array`` broadcasting.

    Args:
        features:   ``(B, irreps.dim)``
        weights:    ``(B, irreps.num_irreps)`` — one scalar per irrep instance.
        expand_idx: Precomputed index from :func:`build_weight_expand_index`.

    Returns:
        ``(B, irreps.dim)`` scaled features.
    """
    return features * weights[:, expand_idx]


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

def build_norm_info(irreps: o3.Irreps) -> Dict:
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
    for mul, ir in irreps:
        size = mul * ir.dim
        if ir.is_scalar():
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
    sh_irreps: o3.Irreps,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute species one-hot, radial basis, and spherical harmonics.

    Args:
        positions: (N, 3) — may have requires_grad=True for force computation.
        cell:      (G, 3, 3) or None.
        cutoff:    (G,) or None.
        data:      TorchGraphData (topology, species, shifts).
        sh_irreps: Spherical harmonics output irreps (e3nn).

    Returns:
        node_features: (N, n_species) one-hot species embedding.
        radial_basis:  (E, radial_basis_size) radial basis values.
        sh:            (E, sh_irreps.dim) spherical harmonics.
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

    # Spherical harmonics (e3nn normalises the direction vector internally)
    sh = o3.spherical_harmonics(
        sh_irreps,
        r,
        normalize=True,
        normalization="component",
    )  # (E, sh_irreps.dim)

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
