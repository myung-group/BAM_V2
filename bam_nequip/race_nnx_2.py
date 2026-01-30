"""RACE model implementation using Flax NNX.

This module implements the RACE model
using Flax NNX framework with e3nn_jax for equivariant operations.
"""

import json
import math
from typing import Callable, Optional, Sequence

import e3nn_jax as e3nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jraph
from .nequip_nnx import bessel_basis, polynomial_cutoff, separated_layer_norm, MLP
from .linear_nnx import Linear as e3nn_nnx_Linear


class RACEXReadout(nnx.Module):
    """RACE x_node_feature's readout layer with equivariant message passing."""

    def __init__(
        self,
        hidden_irreps: e3nn.Irreps,
        output_irreps: e3nn.Irreps,
        x_irreps: e3nn.Irreps,
        rngs: nnx.Rngs,
    ):
        self.x_readout1 = e3nn_nnx_Linear(
            irreps_in=x_irreps,
            irreps_out=hidden_irreps,
            biases=True, # willow
            rngs=rngs,
        )
        self.x_readout2 = e3nn_nnx_Linear(
            irreps_in=hidden_irreps,
            irreps_out=hidden_irreps//2,
            biases=True, # willow
            rngs=rngs,
            force_irreps_out=True,
        )
        self.x_readout3 = e3nn_nnx_Linear(
            irreps_in=hidden_irreps//2,
            irreps_out=output_irreps,
            biases=True, # willow
            rngs=rngs,
            force_irreps_out=True,
        )

    def __call__(
        self,
        x_feats: e3nn.IrrepsArray,
    ) -> e3nn.IrrepsArray:

        x_feats_norm = separated_layer_norm (x_feats)
        #x_energy = self.x_readout(data.nodes["species"], x_feats_norm)
        x_features = self.x_readout1(x_feats_norm)
        x_features = self.x_readout2(x_features)
        x_features = e3nn.gate(
            x_features,
            even_act=jax.nn.silu,
            odd_act=jax.nn.sigmoid,
            #odd_act=jax.nn.tanh,
            even_gate_act=jax.nn.silu,
        )
        #x_features = e3nn.scalar_activation(x_features, [jax.nn.silu])
        x_energy = self.x_readout3(x_features)

        return x_energy

class RACEConvolution(nnx.Module):
    """RACE convolution layer with equivariant message passing."""

    def __init__(
        self,
        input_irreps: e3nn.Irreps,
        output_irreps: e3nn.Irreps,
        x_irreps: e3nn.Irreps,
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

        # add extra irreps to output to account for gate
        gate_irreps = e3nn.Irreps(
            f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e"
        )
        gated_output_irreps = (output_irreps + gate_irreps).regroup()
        '''
        self.full_tp = FullyConnectedTensorProduct (
                irreps_in1=tp_irreps,
                irreps_in2=x_irreps,
                irreps_out=output_irreps,
                rngs=rngs)
        '''

        self.linear_2 = e3nn_nnx_Linear(
            irreps_in=tp_irreps,
            #irreps_in=output_irreps,
            irreps_out=gated_output_irreps,
            rngs=rngs,
        )

        # skip connection has per-species weights
        self.skip = e3nn_nnx_Linear(
            irreps_in=input_irreps,
            irreps_out=gated_output_irreps,
            linear_type="indexed",
            num_indexed_weights=n_species,
            force_irreps_out=True,
            rngs=rngs,
        )

        output_irreps_2 = output_irreps//2
        self.readout1 = e3nn_nnx_Linear(
            irreps_in=output_irreps,
            irreps_out=output_irreps_2,
            biases=True, # willow
            rngs=rngs,
        )
        self.readout = e3nn_nnx_Linear(
            irreps_in=output_irreps_2,
            irreps_out=e3nn.Irreps("0e"),
            biases=True, # willow
            rngs=rngs,
        )

    def __call__(
        self,
        features: e3nn.IrrepsArray,
        x_feats: e3nn.IrrepsArray,
        species: jax.Array,
        sh: e3nn.IrrepsArray,
        radial_basis: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ) -> e3nn.IrrepsArray:

        features_norm = separated_layer_norm (features)
        messages = self.linear_1(features_norm)[senders]
        messages = e3nn.tensor_product(messages, sh, filter_ir_out=self.output_irreps)
        radial_message = jax.vmap(self.radial_mlp)(radial_basis)
        messages = messages * radial_message

        messages_agg = e3nn.scatter_sum(
            messages, dst=receivers, output_size=features.shape[0]
        ) / jnp.sqrt(self.avg_n_neighbors)

        #messages_agg = self.full_tp (messages_agg, x_feats)
        messages_agg = e3nn.tensor_product(messages_agg, x_feats, filter_ir_out=self.output_irreps)

        features = self.linear_2(messages_agg) + self.skip(species, features)

        features = e3nn.gate(
            features,
            even_act=jax.nn.silu,
            odd_act=jax.nn.sigmoid,
            #odd_act=jax.nn.tanh,
            even_gate_act=jax.nn.silu,
        )

        node_energies = self.readout1(features)
        node_energies = self.readout(node_energies)

        return node_energies, features


class RACE(nnx.Module):
    """RACE model for predicting energies and forces.

    Neural Equivariant Interatomic Potential using E(3)-equivariant
    graph neural networks.

    Args:
        n_species: Number of atom species
        lmax: Maximum angular momentum for spherical harmonics
        cutoff: Radial cutoff distance
        hidden_size: Hidden feature dimension
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
        rngs: Flax NNX random number generator
    """

    def __init__(
        self,
        n_species: int,
        lmax: int = 3,
        cutoff: float = 5.0,
        hidden_size: int = 128,
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
        *,
        rngs: nnx.Rngs,
    ):
        # Store static configuration
        self.lmax = lmax
        self.cutoff = cutoff
        self.n_species = n_species
        self.radial_basis_size = radial_basis_size
        self.radial_polynomial_p = radial_polynomial_p
        self.n_layers = n_layers

        # Store trainable scale parameters
        self.shift = nnx.Param(jnp.array(shift))
        self.scale = nnx.Param(jnp.array(scale))
        self.avg_n_neighbors = nnx.Param(jnp.array(avg_n_neighbors))

        if atom_energies is not None:
            self.atom_energies = nnx.Param(jnp.array(atom_energies))
        else:
            self.atom_energies = nnx.Param(jnp.zeros(n_species, dtype=jnp.float32))

        input_irreps = e3nn.Irreps(f"{hidden_size}x0e")
        x_irreps = e3nn.Irreps("1x0e")
        self.x_linear = e3nn_nnx_Linear(
            irreps_in=input_irreps,
            irreps_out=x_irreps,
            linear_type="indexed",
            num_indexed_weights=n_species,
            force_irreps_out=True,
            rngs=rngs,
        )
        """        
        self.x_readout = e3nn_nnx_Linear(
            irreps_in=x_irreps,
            irreps_out=e3nn.Irreps("1x0e"),
            linear_type="indexed",
            num_indexed_weights=n_species,
            force_irreps_out=True,
            biases=True, # willow
            rngs=rngs,
        )
        """
    
        sh_irreps = e3nn.s2_irreps(lmax)
        hidden_irreps = hidden_size * e3nn.s2_irreps(lmax)

        self.x_readout = RACEXReadout(
            hidden_irreps=e3nn.Irreps("64x0e"),
            output_irreps=e3nn.Irreps("0e"),
            x_irreps=x_irreps,
            rngs=rngs,
        )
        self.species_embedding = e3nn_nnx_Linear(
            irreps_in=e3nn.Irreps(f"{n_species}x0e"),
            irreps_out=input_irreps,
            rngs=rngs,
        )

        layers = []
        for i in range(n_layers):
            layer = RACEConvolution(
                input_irreps=input_irreps if i == 0 else hidden_irreps,
                output_irreps=hidden_irreps,
                x_irreps=x_irreps,
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
        #self.layers = layers


    def node_energies(self, positions: jax.Array, data: jraph.GraphsTuple) -> jax.Array:
        """Compute per-node energies.

        Args:
            positions: Atomic positions of shape (n_atoms, 3)
            data: Graph data structure

        Returns:
            Per-node energies of shape (n_atoms, 1)
        """
        # input features are one-hot encoded species
        features = e3nn.IrrepsArray(
            e3nn.Irreps(f"{self.n_species}x0e"),
            jax.nn.one_hot(data.nodes["species"], self.n_species),
        )
        features = self.species_embedding (features)
        x_feats = self.x_linear(data.nodes["species"], features)

        x_energy = self.x_readout(x_feats)

        r = positions[data.senders] - positions[data.receivers]

        # safe norm (avoids nan for r = 0)
        square_r_norm = jnp.sum(r**2, axis=-1)
        r_norm = jnp.where(square_r_norm == 0.0, 0.0, jnp.sqrt(square_r_norm))

        radial_basis = (
            bessel_basis(r_norm, self.radial_basis_size, self.cutoff)
            * polynomial_cutoff(
                r_norm,
                self.cutoff,
                self.radial_polynomial_p,
            )[:, None]
        )

        # compute spherical harmonics of edge displacements
        sh = e3nn.spherical_harmonics(
            e3nn.s2_irreps(self.lmax),
            r,
            normalize=True,
            normalization="component",
        )

        outputs = []
        for layer in self.layers:
            node_energies, features = layer(
                features,
                x_feats,
                data.nodes["species"],
                sh,
                radial_basis,
                data.senders,
                data.receivers,
            )
            outputs += [node_energies.array[:,0] + x_energy.array.squeeze()]

        node_energies = jnp.sum(jnp.stack(outputs, axis=1), axis=-1, keepdims=True)
        node_energies = e3nn.IrrepsArray(e3nn.Irreps("0e"), node_energies)
        # scale and shift energies
        node_energies = node_energies * (
            self.scale[...]
        ) + (self.shift[...])

        # add isolated atom energies to each node as prior
        node_energies = node_energies + jax.lax.stop_gradient(
            self.atom_energies[...][data.nodes["species"], None]
        )

        return node_energies.array

    def __call__(self, data: jraph.GraphsTuple) -> tuple[jax.Array, jax.Array]:
        """Compute energies and forces for a batch of graphs.

        Args:
            data: Graph data structure with nodes containing 'positions' and 'species'

        Returns:
            Tuple of (graph_energies, forces) where:
                - graph_energies: Total energy per graph, shape (n_graphs,)
                - forces: Forces on each atom, shape (n_atoms, 3)
        """
        # compute forces as gradient of total energy
        def total_energy_fn(positions: jax.Array) -> tuple[jax.Array, jax.Array]:
            node_energies = self.node_energies(positions, data)
            return jnp.sum(node_energies), node_energies

        grad_fn = jax.grad(total_energy_fn, has_aux=True)
        minus_forces, node_energies = grad_fn(data.nodes["positions"])

        # Get node mask for padded graphs
        # jraph.get_node_padding_mask treats the last graph as padding,
        # so for single graphs or non-padded batches, we need special handling
        n_graphs = data.n_node.shape[0]
        if n_graphs > 1:
            # For padded batches, use jraph's mask (last graph is padding)
            node_mask = jraph.get_node_padding_mask(data)[:, None]
            minus_forces = jnp.where(node_mask, minus_forces, 0.0)
        # For single graphs, no masking needed (all nodes are valid)

        # Handle any NaN forces from numerical issues
        minus_forces = jnp.where(jnp.isnan(minus_forces), 0.0, minus_forces)

        # compute total energies across each subgraph
        graph_energies = jraph.segment_sum(
            node_energies,
            node_graph_idx(data),
            num_segments=data.n_node.shape[0],
            indices_are_sorted=True,
        )

        return graph_energies[:, 0], -minus_forces


def node_graph_idx(data: jraph.GraphsTuple) -> jnp.ndarray:
    """Returns the index of the graph for each node."""
    n_graph = data.n_node.shape[0]
    sum_n_node = jax.tree_util.tree_leaves(data.nodes)[0].shape[0]
    graph_idx = jnp.arange(n_graph)
    node_gr_idx = jnp.repeat(
        graph_idx, data.n_node, axis=0, total_repeat_length=sum_n_node
    )
    return node_gr_idx



# =============================================================================
# Data Loading
# =============================================================================

def get_graphset(
    data,
    cutoff: float,
    nbatch: int,
    uniq_element: dict,
    enr_avg_per_element: dict
) -> list:
    """
    Convert ASE atoms to padded graph batches.

    Args:
        data: List of ASE Atoms objects
        cutoff: Interaction cutoff radius
        nbatch: Batch size
        uniq_element: Dict mapping atomic numbers to indices
        enr_avg_per_element: Dict of average energy per element type

    Returns:
        List of batched and padded GraphsTuples
    """
    from matscipy.neighbours import neighbour_list

    graph_list = []

    for atoms in data:
        crds = atoms.get_positions()
        species = jnp.array([uniq_element[z] for z in atoms.numbers])
        node_enr_avg = jnp.array([
            enr_avg_per_element[z] for z in atoms.numbers
        ])
        enr = atoms.get_potential_energy() - node_enr_avg.sum()
        frc = atoms.get_forces()
        cell = atoms.get_cell()
        volume = atoms.get_volume()

        stress = jnp.zeros(6)
        if 'stress' in atoms.calc.results:
            stress = atoms.get_stress()

        iatoms, jatoms, Sij = neighbour_list(
            quantities='ijS', atoms=atoms, cutoff=cutoff
        )

        graph = jraph.GraphsTuple(
            nodes={
                "positions": jnp.array(crds),
                "species": species,
                "forces": jnp.array(frc)
            },
            edges={'Sij': jnp.array(Sij)},
            globals={
                'energy': jnp.array([enr]),
                'cell': jnp.array([cell.array]),
                'volume': jnp.array([volume]),
                'stress': jnp.array([stress])
            },
            senders=jnp.array(iatoms),
            receivers=jnp.array(jatoms),
            n_node=jnp.array([crds.shape[0]]),
            n_edge=jnp.array([iatoms.shape[0]])
        )
        graph_list.append(graph)

    # Determine padding sizes
    n_data = len(data)
    pad_nodes_to = 0
    pad_edges_to = 0

    for ist0 in range(0, n_data, nbatch):
        ied0 = min(ist0 + nbatch, n_data)
        batch = jraph.batch(graph_list[ist0:ied0])
        pad_nodes_to = max(pad_nodes_to, int(batch.n_node.sum()))
        pad_edges_to = max(pad_edges_to, int(batch.n_edge.sum()))

    pad_nodes_to += 1
    pad_graphs_to = nbatch + 1

    dataset_list = []
    for ist0 in range(0, n_data, nbatch):
        ied0 = min(ist0 + nbatch, n_data)
        batch = jraph.batch(graph_list[ist0:ied0])
        batch = jraph.pad_with_graphs(batch, pad_nodes_to, pad_edges_to, pad_graphs_to)
        dataset_list.append(batch)

    return dataset_list


if __name__ == "__main__":
    # Example usage
    import numpy as np
    from ase.io import read

    print("=" * 70)
    print("RACE-Style Equivariant Potential")
    print("=" * 70)

    uniq_element = {z: i for i, z in enumerate([1, 6, 7, 8])}
    #enr_avg_per_element = {i: -654.09 for i in range(4)}
    enr_avg_per_element = {
        1: -13.587222780835477,
        6: -1029.4889999855063,
        7: -1484.9814568572233,
        8: -2041.9816003861047,
    }

    # Load data
    atoms = read('dataset_3BPA/train_300K.xyz', index=0)
    graphset = get_graphset(
        [atoms],
        cutoff=5.0,
        nbatch=1,
        uniq_element=uniq_element,
        enr_avg_per_element=enr_avg_per_element
    )

    data = graphset[0]

    # Create model
    rngs = nnx.Rngs(42)
    model = RACE(
        n_species=4,
        lmax=3,
        cutoff=5.0,
        hidden_size=128,
        n_layers=3,
        radial_basis_size=8,
        radial_mlp_size=64,
        radial_mlp_layers=3,
        rngs=rngs,
    )


    # Count parameters
    graphdef, state = nnx.split(model)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(state))
    print(f"\nNumber of parameters: {n_params:,}")


    # Run forward pass
    energies, forces = model(data)
    print(f"Energies shape: {energies.shape}")
    print(f"Forces shape: {forces.shape}")
    print(f"Total energy: {energies[0]:.6f}")
    print(f"Force magnitude: {np.linalg.norm(forces, axis=-1).mean():.6f}")

    # Test equivariance
    print("\nTesting equivariance...")
    angle_z = jnp.pi / 3.0
    rotation_z = jnp.array([
        [jnp.cos(angle_z), -jnp.sin(angle_z), 0],
        [jnp.sin(angle_z), jnp.cos(angle_z), 0],
        [0, 0, 1]
    ])
    angle_y = jnp.pi / 3.5
    rotation_y = jnp.array([
        [jnp.cos(angle_y), 0, -jnp.sin(angle_y)],
        [0, 1, 0],
        [jnp.sin(angle_y), 0, jnp.cos(angle_y)]
    ])
    rotation = rotation_y @ rotation_z

    positions = data.nodes['positions']
    rotated_positions = positions @ rotation.T
    data_rotated = data._replace(
        nodes={**data.nodes, 'positions': rotated_positions}
    )
    energies_rotated, forces_rotated = model(data_rotated)

    energy_error = jnp.abs(energies[0] - energies_rotated[0])
    forces_should_be = forces @ rotation.T
    forces_error = jnp.max(jnp.abs(forces_rotated - forces_should_be))

    print(f"  Energy error: {energy_error:.2e}")
    print(f"  Forces error: {forces_error:.2e}")

