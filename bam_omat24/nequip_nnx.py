"""NequIP model implementation using Flax NNX.

This module implements the NequIP (Neural Equivariant Interatomic Potential) model
using Flax NNX framework with e3nn_jax for equivariant operations.

"""


import math
from typing import Callable, Optional, Sequence, Tuple

import e3nn_jax as e3nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jraph

from bam_omat24.linear_nnx import Linear as e3nn_nnx_Linear


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


class Linear(nnx.Module):
    """Simple linear layer with optional bias."""

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
            layer = Linear(
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


class NequipConvolution(nnx.Module):
    """NequIP convolution layer with equivariant message passing."""

    def __init__(
        self,
        input_irreps: e3nn.Irreps,
        output_irreps: e3nn.Irreps,
        sh_irreps: e3nn.Irreps,
        n_species: int,
        radial_basis_size: int,
        radial_mlp_size: int,
        radial_mlp_layers: int,
        mlp_init_scale: float,
        avg_n_neighbors: float,
        *,
        rngs: nnx.Rngs,
    ):
        self.output_irreps = output_irreps
        self.avg_n_neighbors = avg_n_neighbors

        tp_irreps = e3nn.tensor_product(
            input_irreps, sh_irreps, filter_ir_out=output_irreps
        )

        self.linear_1 = e3nn_nnx_Linear(
            irreps_in=input_irreps,
            irreps_out=input_irreps,
            rngs=rngs,
        )

        self.radial_mlp = MLP(
            sizes=[radial_basis_size]
            + [radial_mlp_size] * radial_mlp_layers
            + [tp_irreps.num_irreps],
            activation=jax.nn.silu,
            use_bias=False,
            init_scale=mlp_init_scale,
            rngs=rngs,
        )

        # Add extra irreps to output to account for gate
        gate_irreps = e3nn.Irreps(
            f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e"
        )
        gated_output_irreps = (output_irreps + gate_irreps).regroup()

        self.linear_2 = e3nn_nnx_Linear(
            irreps_in=tp_irreps,
            irreps_out=gated_output_irreps,
            rngs=rngs,
        )

        # Skip connection has per-species weights
        self.skip = e3nn_nnx_Linear(
            irreps_in=input_irreps,
            irreps_out=gated_output_irreps,
            linear_type="indexed",
            num_indexed_weights=n_species,
            force_irreps_out=True,
            rngs=rngs,
        )

    def __call__(
        self,
        features: e3nn.IrrepsArray,
        species: jax.Array,
        sh: e3nn.IrrepsArray,
        radial_basis: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ) -> e3nn.IrrepsArray:
        # Follow equiformer_v2: layer norm before message passing
        features_norm = separated_layer_norm(features)
        messages = self.linear_1(features_norm)[senders]
        messages = e3nn.tensor_product(messages, sh, filter_ir_out=self.output_irreps)
        radial_message = jax.vmap(self.radial_mlp)(radial_basis)
        messages = messages * radial_message

        messages_agg = e3nn.scatter_sum(
            messages, dst=receivers, output_size=features.shape[0]
        ) / jnp.sqrt(self.avg_n_neighbors)

        features = self.linear_2(messages_agg) + self.skip(species, features)

        # Follow equiformer: gated activation
        return e3nn.gate(
            features,
            even_act=jax.nn.silu,
            odd_act=jax.nn.sigmoid,
            even_gate_act=jax.nn.silu,
        )


class Nequip(nnx.Module):
    """NequIP model for predicting energies and forces.

    Neural Equivariant Interatomic Potential using E(3)-equivariant
    graph neural networks.

    Args:
        n_species: Number of atom species
        lmax: Maximum angular momentum for spherical harmonics
        cutoff: Radial cutoff distance
        hidden_irreps: Hidden feature dimension
        n_layers: Number of convolution layers
        radial_basis_size: Number of radial basis functions
        radial_mlp_size: Hidden size of radial MLP
        radial_mlp_layers: Number of radial MLP layers
        radial_polynomial_p: Polynomial cutoff order
        mlp_init_scale: Initialization scale for MLPs
        shift: Energy shift
        scale: Energy scale
        avg_n_neighbors: Average number of neighbors for normalization
        atom_energies: Isolated atom energies for each species
        l_train: Whether in training mode (affects atom energy handling)
        periodic: Whether system is periodic
        rngs: Flax NNX random number generator
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
        radial_mlp_layers: int = 3,
        radial_polynomial_p: float = 2.0,
        mlp_init_scale: float = 4.0,
        shift: float = 0.0,
        scale: float = 1.0,
        avg_n_neighbors: float = 25.0,
        atom_energies: Optional[Sequence[float]] = None,
        l_train: bool = True,
        periodic: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        # Store static configuration
        self.lmax = lmax
        self.cutoff = cutoff
        self.periodic = periodic
        self.n_species = n_species
        self.radial_basis_size = radial_basis_size
        self.radial_polynomial_p = radial_polynomial_p
        self.l_train = l_train
        
        # Store trainable scale parameters
        self.shift = nnx.Param(jnp.array(shift))
        self.scale = nnx.Param(jnp.array(scale))
        self.avg_n_neighbors = nnx.Param(jnp.array(avg_n_neighbors))

        if atom_energies is not None:
            self.atom_energies = nnx.Param(jnp.array(atom_energies))
        else:
            self.atom_energies = nnx.Param(jnp.zeros(n_species, dtype=jnp.float32))

        # Parse and store irreps
        self.hidden_irreps = e3nn.Irreps(hidden_irreps)
        input_irreps = self.hidden_irreps.filter(keep="0e")
        sh_irreps = e3nn.s2_irreps(lmax)

        self.species_embedding = e3nn_nnx_Linear(
            irreps_in=e3nn.Irreps(f"{n_species+1}x0e"),
            irreps_out=input_irreps,
            rngs=rngs,
        )

        layers = []
        for i in range(n_layers):
            layer = NequipConvolution(
                input_irreps=input_irreps if i == 0 else self.hidden_irreps,
                output_irreps=self.hidden_irreps,
                sh_irreps=sh_irreps,
                n_species=n_species,
                radial_basis_size=radial_basis_size,
                radial_mlp_size=radial_mlp_size,
                radial_mlp_layers=radial_mlp_layers,
                mlp_init_scale=mlp_init_scale,
                avg_n_neighbors=avg_n_neighbors,
                rngs=rngs,
            )
            layers.append(layer)

        self.layers = nnx.List(layers)

        # Extract scalar dimension for readout
        hidden_scalar_dim = self.hidden_irreps.filter(keep="0e").dim
        hidden_irreps_2 = e3nn.Irreps(f"{hidden_scalar_dim // 2}x0e")
        
        self.readout1 = e3nn_nnx_Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=hidden_irreps_2,
            biases=True,
            rngs=rngs,
        )
        self.readout = e3nn_nnx_Linear(
            irreps_in=hidden_irreps_2,
            irreps_out=e3nn.Irreps("0e"),
            biases=True,
            rngs=rngs,
        )

    def node_energies(
        self,
        positions: jax.Array,
        cell: jax.Array,
        cutoff: jax.Array,
        data: jraph.GraphsTuple
    ) -> jax.Array:
        """Compute per-node energies.

        Args:
            positions: Atomic positions of shape (n_atoms, 3)
            cell: (n_graph, 3, 3) for periodic, None for molecular
            cutoff: (n_graph,) for periodic, None for molecular
            data: Graph data structure

        Returns:
            Per-node energies of shape (n_atoms, 1)
        """
        # Compute edge features using shared utility
        features, radial_basis, sh, bonds_per_node = compute_node_edge_features(
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
        )

        features = self.species_embedding(features)

        for layer in self.layers:
            features = layer(
                features,
                data.nodes["species"],
                sh,
                radial_basis,
                data.senders,
                data.receivers,
            )

        features = self.readout1(features)
        node_energies = self.readout(features)

        # Scale and shift energies
        node_energies = node_energies * self.scale[...] + self.shift[...]

        return node_energies.array

    def __call__(
        self, data: jraph.GraphsTuple
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Compute energies and forces for a batch of graphs.

        Args:
            data: Graph data structure with nodes containing 'positions' and 'species'.
                  For periodic systems, globals must contain 'cell', 'cutoff', 'volume'.
                  For molecular systems, these are not required.

        Returns:
            Tuple of (graph_energies, forces, stress) where:
                - graph_energies: Total energy per graph, shape (n_graphs,)
                - forces: Forces on each atom, shape (n_atoms, 3)
                - stress: Stress tensor per graph, shape (n_graphs, 6)
                          (Voigt: xx, yy, zz, yz, xz, xy)
                          Zero for molecular systems.
        """
        # Input validation
        assert data.nodes is not None, "Graph must have nodes"
        assert "positions" in data.nodes, "Nodes must have 'positions'"
        assert "species" in data.nodes, "Nodes must have 'species'"
        
        n_graphs = data.n_node.shape[0]
        node_mask = jraph.get_node_padding_mask(data)

        if self.periodic:
            assert "cell" in data.globals, "Periodic systems require 'cell' in globals"
            assert "cutoff" in data.globals, "Periodic systems require 'cutoff' in globals"
            
            # Periodic: differentiate w.r.t. positions AND cell
            def total_energy_fn(
                positions: jax.Array,
                cell: jax.Array,
                cutoff: jax.Array
            ) -> Tuple[jax.Array, jax.Array]:
                node_energies = self.node_energies(positions, cell, cutoff, data)
                node_energies = jnp.where(node_mask[:, None], node_energies, 0.0)
                return jnp.sum(node_energies), node_energies

            grad_fn = jax.grad(total_energy_fn, argnums=(0, 1), has_aux=True)
            (grad, cellgrad), node_energies = grad_fn(
                data.nodes["positions"],
                data.globals["cell"],
                data.globals['cutoff']
            )
        else:
            # Molecular: differentiate w.r.t. positions only
            def total_energy_fn(positions: jax.Array) -> Tuple[jax.Array, jax.Array]:
                node_energies = self.node_energies(positions, None, None, data)
                node_energies = jnp.where(node_mask[:, None], node_energies, 0.0)
                return jnp.sum(node_energies), node_energies

            grad_fn = jax.grad(total_energy_fn, has_aux=True)
            grad, node_energies = grad_fn(data.nodes["positions"])

        if not self.l_train:
            # Add isolated atom energies to each node as prior
            # Note: This is added AFTER gradient computation, so it doesn't affect forces
            # (which is physically correct - isolated atom energies are reference energies)
            node_energies = node_energies + jax.lax.stop_gradient(
                self.atom_energies[...][data.nodes["species"], None]
            )

        # Apply mask to gradients for padded graphs
        if n_graphs > 1:
            grad = jnp.where(node_mask[:, None], grad, 0.0)

        # Handle any NaN forces from numerical issues
        grad = jnp.where(jnp.isnan(grad), 0.0, grad)

        # Compute total energies across each subgraph
        graph_energies = jraph.segment_sum(
            node_energies,
            node_graph_idx(data),
            num_segments=n_graphs,
            indices_are_sorted=True,
        )

        if self.periodic:
            # Stress for periodic systems
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
            # No stress for molecular systems
            stress = jnp.zeros((n_graphs, 6))

        return graph_energies[:, 0], -grad, stress


def node_graph_idx(data: jraph.GraphsTuple) -> jnp.ndarray:
    """Returns the index of the graph for each node."""
    n_graph = data.n_node.shape[0]
    sum_n_node = jax.tree_util.tree_leaves(data.nodes)[0].shape[0]
    graph_idx = jnp.arange(n_graph)
    node_gr_idx = jnp.repeat(
        graph_idx, data.n_node, axis=0, total_repeat_length=sum_n_node
    )
    return node_gr_idx


if __name__ == "__main__":
    # Example usage
    import numpy as np
    import pickle
    from bam_omat24.atom_energies import ATOM_ENERGIES

    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("Nequip-Style Equivariant Potential")
    print("=" * 70)

    with open('omat24_val_0.pkl', 'rb') as f:
        graphset = pickle.load(f)
    data = graphset[0]

    # Create model
    rngs = nnx.Rngs(42)
    model = Nequip(
        n_species=len(ATOM_ENERGIES),
        lmax=3,
        hidden_irreps="64x0e + 64x1o + 64x2e",
        n_layers=3,
        radial_basis_size=8,
        radial_mlp_size=64,
        radial_mlp_layers=3,
        mlp_init_scale=4.0,
        shift=0.0,
        scale=1.0,
        avg_n_neighbors=25,
        atom_energies=ATOM_ENERGIES,
        periodic=True,
        rngs=rngs,
    )

    # Count parameters
    graphdef, state = nnx.split(model)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(state))
    print(f"\nNumber of parameters: {n_params:,}")

    # Run forward pass
    energies, forces, stress = model(data)
    print(f"Energies shape: {energies.shape}")
    print(f"Forces shape: {forces.shape}")
    print(f"Total energy: {energies[0]:.6f}")
    print(f"Force magnitude: {np.linalg.norm(forces, axis=-1).mean():.6f}")
    print(f"Stress: {stress}")
    print(f"Target energies: {data.globals['energy']}")
    print(f"Target stress: {data.globals['stress']} eV/A^3")
