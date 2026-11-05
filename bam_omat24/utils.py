"""Shared utilities for equivariant ML potential models.

This module contains model-agnostic building blocks used across
nequip_nnx, race_nnx, and finetuning_nnx.
"""

import math
from typing import Callable, Optional, Sequence, Tuple, Dict

import e3nn_jax as e3nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jraph


def bessel_basis(x: jax.Array, num_basis: int, r_max: jax.Array) -> jax.Array:
    """Compute Bessel radial basis functions.

    Args:
        x: Distance array of shape (n_edges,)
        num_basis: Number of basis functions
        r_max: Cutoff radius per edge of shape (n_edges,)

    Returns:
        Bessel basis values of shape (n_edges, num_basis)
    """
    r_max = r_max[:, None]
    x = x[:, None]
    prefactor = 2.0 / r_max
    bessel_weights = jnp.linspace(1.0, num_basis, num_basis) * jnp.pi

    return prefactor * jnp.where(
        x == 0.0,
        bessel_weights / r_max,
        jnp.sin(bessel_weights * x / r_max) / x,
    )


def polynomial_cutoff(x: jax.Array, r_max: jax.Array, p: float) -> jax.Array:
    """Compute polynomial cutoff function.

    Args:
        x: Distance array
        r_max: Cutoff radius
        p: Polynomial order

    Returns:
        Cutoff values (smoothly goes to zero at r_max)
    """
    factor = 1.0 / r_max
    x = x * factor
    out = 1.0
    out = out - (((p + 1.0) * (p + 2.0) / 2.0) * jnp.power(x, p))
    out = out + (p * (p + 2.0) * jnp.power(x, p + 1.0))
    out = out - ((p * (p + 1.0) / 2) * jnp.power(x, p + 2.0))
    return out * jnp.where(x < 1.0, 1.0, 0.0)


def separated_layer_norm(input: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
    """Apply separated layer normalization to IrrepsArray.

    Scalars are normalized with mean centering and RMS.
    Non-scalars share a common RMS normalization.

    Args:
        input: Input IrrepsArray

    Returns:
        Normalized IrrepsArray
    """
    eps = 1e-6

    # Compute RMS for non-scalar features
    xi_s = jnp.zeros((input.shape[0], 1))
    cnt = 0
    for (mul, ir), chunk in zip(input.irreps, input.chunks):
        if not ir.is_scalar():
            xi = chunk.reshape(-1, mul * ir.dim)
            xi_s += jnp.sum(xi**2, axis=-1, keepdims=True)
            cnt += mul * ir.dim

    # Use rsqrt for numerical stability
    xi_rms_inv_nonscalar = jax.lax.rsqrt(xi_s / jnp.maximum(cnt, 1) + eps)

    new_chunks = []
    for (mul, ir), chunk in zip(input.irreps, input.chunks):
        if ir.is_scalar():
            xi = chunk.reshape(-1, mul)
            xi_centered = xi - jnp.mean(xi, axis=-1, keepdims=True)
            xi_rms_inv_scalar = jax.lax.rsqrt(
                jnp.mean(xi_centered**2, axis=-1, keepdims=True) + eps
            )
            chunk = (xi_centered * xi_rms_inv_scalar).reshape(-1, mul, ir.dim)
        else:
            xi = chunk.reshape(-1, mul * ir.dim)
            chunk = (xi * xi_rms_inv_nonscalar).reshape(-1, mul, ir.dim)

        new_chunks.append(chunk)

    return e3nn.from_chunks(
        input.irreps,
        new_chunks,
        (input.shape[0],)
    )


class DenseLinear(nnx.Module):
    """Simple dense linear layer with optional bias."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        use_bias: bool = True,
        init_scale: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ):
        scale = math.sqrt(init_scale / in_size)
        self.weights = nnx.Param(
            jax.random.normal(rngs.params(), (in_size, out_size)) * scale
        )
        self.use_bias = use_bias
        self.bias: nnx.Param[jax.Array] | None
        if use_bias:
            self.bias = nnx.Param(jnp.zeros(out_size))
        else:
            self.bias = None

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.dot(x, self.weights[...])
        if self.use_bias:
            x = x + self.bias[...]
        return x


class MLP(nnx.Module):
    """Multi-layer perceptron with configurable activation."""

    def __init__(
        self,
        sizes: Sequence[int],
        activation: Callable = jax.nn.silu,
        init_scale: float = 1.0,
        use_bias: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.activation = activation
        layers = []

        for i in range(len(sizes) - 1):
            layer = DenseLinear(
                sizes[i],
                sizes[i + 1],
                use_bias=use_bias,
                # don't scale last layer since no activation
                init_scale=init_scale if i < len(sizes) - 2 else 1.0,
                rngs=rngs,
            )
            layers.append(layer)

        self.layers = nnx.List(layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


def compute_node_edge_features(
    positions: jax.Array,
    cell: Optional[jax.Array],
    cutoff: Optional[jax.Array],
    data: jraph.GraphsTuple,
    n_species: int,
    model_cutoff: float,
    radial_basis_size: int,
    radial_polynomial_p: float,
    lmax: int,
    periodic: bool,
) -> Tuple[e3nn.IrrepsArray, jax.Array, e3nn.IrrepsArray]:
    """Compute edge features: displacement, distance, radial basis, and spherical harmonics.

    This is a shared utility to avoid code duplication across models.

    Args:
        positions: Atomic positions (n_atoms, 3)
        cell: Unit cell (n_graphs, 3, 3) or None
        cutoff: Per-graph cutoff (n_graphs,) or None
        data: Graph data structure
        model_cutoff: Default cutoff for non-periodic systems
        radial_basis_size: Number of radial basis functions
        radial_polynomial_p: Polynomial cutoff order
        lmax: Maximum angular momentum
        periodic: Whether system is periodic

    Returns:
        Tuple of (node_features, radial_basis, sh)
    """
    node_features = e3nn.IrrepsArray(
        e3nn.Irreps(f"{n_species}x0e"),
        jax.nn.one_hot(data.nodes["species"], n_species)
    )

    if periodic:
        # Transform cell-shift vectors Sij into Cartesian displacements
        Sij = data.edges['Sij']
        num_edges = Sij.shape[0]
        Sij = jnp.einsum(
            'ei,eij->ej',
            Sij,
            jnp.repeat(
                cell,
                data.n_edge,
                axis=0,
                total_repeat_length=num_edges,
            )
        )
        cutoff_per_edge = jnp.repeat(
            cutoff,
            data.n_edge,
            axis=0,
            total_repeat_length=num_edges
        )
        r = positions[data.senders] - (positions[data.receivers] + Sij)
    else:
        # Direct displacement, no cell shifts
        r = positions[data.senders] - positions[data.receivers]
        num_edges = data.senders.shape[0]
        cutoff_per_edge = jnp.full((num_edges,), model_cutoff)

    cutoff_per_edge = jnp.where(cutoff_per_edge > 1.0, cutoff_per_edge, 1.0)
    cutoff_per_edge = cutoff_per_edge.squeeze()

    # Safe norm (avoids nan for r = 0)
    square_r_norm = jnp.sum(r**2, axis=-1)
    r_norm = jnp.where(square_r_norm == 0.0, 0.0, jnp.sqrt(square_r_norm))

    radial_basis = (
        bessel_basis(r_norm, radial_basis_size, cutoff_per_edge)
        * polynomial_cutoff(r_norm, cutoff_per_edge, radial_polynomial_p)[:, None]
    )

    # Compute spherical harmonics of edge displacements
    sh = e3nn.spherical_harmonics(
        e3nn.s2_irreps(lmax),
        r,
        normalize=True,
        normalization="component",
    )

    return node_features, radial_basis, sh


def node_graph_idx(data: jraph.GraphsTuple) -> jnp.ndarray:
    """Returns the index of the graph for each node."""
    n_graph = data.n_node.shape[0]
    sum_n_node = jax.tree_util.tree_leaves(data.nodes)[0].shape[0]
    graph_idx = jnp.arange(n_graph)
    node_gr_idx = jnp.repeat(
        graph_idx, data.n_node, axis=0, total_repeat_length=sum_n_node
    )
    return node_gr_idx


def energy_forces_stress(
    node_energy_fn: Callable,
    data: jraph.GraphsTuple,
    periodic: bool,
    l_train: bool,
    atom_energies: jax.Array,
    frc_variation_fill: Optional[jax.Array] = None,
) -> Dict:
    """Compute graph energies, forces, and stress from a node energy function.

    Shared implementation used by Nequip, RACE, and FineTuningPotential.

    Args:
        node_energy_fn: (positions, cell, cutoff, data) -> dict(node_energies, node_features)
        data: Graph data structure
        periodic: Whether system is periodic
        l_train: Whether in training mode (affects atom energy handling)
        atom_energies: Isolated atom energies array
        frc_variation_fill: value written into ``frc_variation`` of padded nodes
            (shape (6,) or scalar). Default 1.0 (original behaviour). Models with
            a non-Cholesky covariance layout (RACE ``cov_mode="irreps"``) pass a
            layout-specific neutral value.

    Returns:
        Tuple of (graph_energies, forces, stress) where:
            - graph_energies: Total energy per graph, shape (n_graphs,)
            - forces: Forces on each atom, shape (n_atoms, 3)
            - stress: Stress tensor per graph, shape (n_graphs, 6)
    """
    assert data.nodes is not None, "Graph must have nodes"
    assert "positions" in data.nodes, "Nodes must have 'positions'"
    assert "species" in data.nodes, "Nodes must have 'species'"

    n_graphs = data.n_node.shape[0]
    if n_graphs > 1:
        node_mask = jraph.get_node_padding_mask(data)
    else:
        node_mask = jnp.ones(data.nodes['positions'].shape[0])

    if periodic:
        assert "cell" in data.globals, "Periodic systems require 'cell' in globals"
        assert "cutoff" in data.globals, "Periodic systems require 'cutoff' in globals"

        def total_energy_fn(
            positions: jax.Array,
            cell: jax.Array,
            cutoff: jax.Array
        ) -> Tuple[jax.Array, dict]:
            results = node_energy_fn(positions, cell, cutoff, data)
            ne = results["node_energies"].reshape(-1)
            ne = ne*node_mask #jnp.where(node_mask, ne, 0.0)
            aux = {
                'node_energies': ne,
                'frc_variation': results.get('frc_variation'),
                'energy_variation': results.get('energy_variation'),
                'stress_variation': results.get('stress_variation'),
                'strata': results.get('strata'),
                'node_features': results.get('node_features'),
            }
            return jnp.sum(ne), aux

        grad_fn = jax.grad(total_energy_fn, argnums=(0, 1), has_aux=True)
        (grad, cellgrad), aux = grad_fn(
            data.nodes["positions"],
            data.globals["cell"],
            data.globals["cutoff"],
        )
        node_energies = aux['node_energies']
        frc_variation = aux['frc_variation']
        energy_variation = aux['energy_variation']
        stress_variation = aux['stress_variation']
    else:
        def total_energy_fn(
            positions: jax.Array
        ) -> Tuple[jax.Array, dict]:
            results = node_energy_fn(positions, None, None, data)
            ne = results["node_energies"].reshape(-1)
            ne = ne*node_mask #jnp.where(node_mask, ne, 0.0)
            aux = {
                'node_energies': ne,
                'frc_variation': results.get('frc_variation'),
                'energy_variation': results.get('energy_variation'),
                'stress_variation': results.get('stress_variation'),
                'strata': results.get('strata'),
                'node_features': results.get('node_features'),
            }
            return jnp.sum(ne), aux

        grad_fn = jax.grad(total_energy_fn, has_aux=True)
        grad, aux = grad_fn(data.nodes["positions"])
        node_energies = aux['node_energies']
        frc_variation = aux['frc_variation']
        energy_variation = aux['energy_variation']
        stress_variation = aux['stress_variation']

    if not l_train:
        node_energies = node_energies + jax.lax.stop_gradient(
            atom_energies[data.nodes["species"]]
        )

    if n_graphs > 1:
        grad = jnp.where(node_mask[:, None], grad, 0.0)
        if frc_variation_fill is None:
            frc_variation_fill = 1.0
        frc_variation = jnp.where(node_mask[:, None], frc_variation, frc_variation_fill)
        if energy_variation is not None:
            energy_variation = jnp.where(node_mask[:, None], energy_variation, 0.0)
        if stress_variation is not None:
            # Neutral: sigma=1.0 for all 6 Voigt components
            stress_variation = jnp.where(
                node_mask[:, None], stress_variation, 1.0
            )

    grad = jnp.where(jnp.isnan(grad), 0.0, grad)

    graph_energies = jraph.segment_sum(
        node_energies,
        node_graph_idx(data),
        num_segments=n_graphs,
        indices_are_sorted=True,
    )

    if periodic:
        cellgrad = jnp.where(jnp.isnan(cellgrad), 0.0, cellgrad)

        volume = data.globals["volume"]
        volume = jnp.where(volume > 0.0, volume, 1.0)
        stress_cell = (
            jnp.transpose(cellgrad, (0, 2, 1)) @ data.globals["cell"]
        )

        stress_grad = jnp.einsum("iu,iv->iuv", grad, data.nodes["positions"])
        stress_grad = jraph.segment_sum(
            stress_grad,
            node_graph_idx(data),
            num_segments=n_graphs,
            indices_are_sorted=True,
        )
        virial = (stress_cell + stress_grad).reshape(-1, 9)[:, [0, 4, 8, 5, 2, 1]]
        stress = virial / volume[:, None]
    else:
        stress = jnp.zeros((n_graphs, 6))


    # Aggregate per-atom variations to per-graph
    _node_idx = node_graph_idx(data)
    if energy_variation is not None:
        # Per-atom log_sigma (n_atoms, 1) → per-graph mean (n_graphs, 1)
        energy_variation = jraph.segment_mean(
            energy_variation, _node_idx, num_segments=n_graphs,
            indices_are_sorted=True,
        )
    if stress_variation is not None:
        # Per-atom log_sigma (n_atoms, 6) → per-graph mean (n_graphs, 6)
        stress_variation = jraph.segment_mean(
            stress_variation, _node_idx, num_segments=n_graphs,
            indices_are_sorted=True,
        )

    strata = aux.get('strata')
    if strata is not None:
        strata = strata * node_mask[None, :]       # (n_strata, n_atoms), padded nodes -> 0

    return {
        'energy': graph_energies,
        'forces': -jnp.array(grad),
        'stress': stress,
        'force_variation': frc_variation,
        'energy_variation': energy_variation,
        'stress_variation': stress_variation,
        'strata': strata,
        'node_features': aux.get('node_features'),     # final node features (DeNS noise head)
    }
