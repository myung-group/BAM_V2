"""RACE model implementation using Flax NNX.

This module implements the RACE model
using Flax NNX framework with e3nn_jax for equivariant operations.

Changes from original:
- Fixed hardcoded irreps in RACEXReadout (now derived from hidden_irreps)
- Fixed avg_n_neighbors Param usage (now consistent)

"""

from typing import Optional, Sequence, Tuple

import e3nn_jax as e3nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jraph

from bam_omat24.utils import (
    separated_layer_norm,
    MLP,
    node_graph_idx,
    compute_node_edge_features,
    energy_forces_stress,
)
from bam_omat24.linear_nnx import Linear as e3nn_nnx_Linear



class RACEXReadout(nnx.Module):
    """RACE x_node_feature's readout layer with equivariant message passing."""

    def __init__(
        self,
        hidden_irreps: e3nn.Irreps,
        output_irreps: e3nn.Irreps,
        x_irreps: e3nn.Irreps,
        n_species: int,
        avg_n_neighbors: float,
        rngs: nnx.Rngs,
    ):
        self.x_readout1 = e3nn_nnx_Linear(
            irreps_in=x_irreps,
            irreps_out=hidden_irreps,
            linear_type="indexed",
            num_indexed_weights=n_species,
            force_irreps_out=True,
            rngs=rngs,
        )
        self.x_readout2 = e3nn_nnx_Linear(
            irreps_in=hidden_irreps,
            irreps_out=hidden_irreps // 2,
            biases=True,
            rngs=rngs,
            force_irreps_out=True,
        )
        self.x_readout3 = e3nn_nnx_Linear(
            irreps_in=hidden_irreps // 2,
            irreps_out=output_irreps,
            biases=True,
            rngs=rngs,
            force_irreps_out=True,
        )
        self.avg_n_neighbors = avg_n_neighbors

    def __call__(
        self,
        species: jax.Array,
        x_feats: e3nn.IrrepsArray,
    ) -> e3nn.IrrepsArray:
        x_feats_norm = separated_layer_norm(x_feats) / jnp.sqrt(self.avg_n_neighbors)
        x_features = self.x_readout1(species, x_feats_norm)
        x_features = self.x_readout2(x_features)
        x_features = e3nn.gate(
            x_features,
            even_act=jax.nn.silu,
            odd_act=jax.nn.sigmoid,
            even_gate_act=jax.nn.silu,
        )
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

        output_irreps_2 = output_irreps // 2
        self.readout1 = e3nn_nnx_Linear(
            irreps_in=output_irreps,
            irreps_out=output_irreps_2,
            biases=True,
            rngs=rngs,
        )
        self.readout = e3nn_nnx_Linear(
            irreps_in=output_irreps_2,
            irreps_out=e3nn.Irreps("0e"),
            biases=True,
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
    ) -> Tuple[e3nn.IrrepsArray, e3nn.IrrepsArray]:
        # Follow equiformer_v2: layer norm before message passing
        features_norm = separated_layer_norm(features)
        messages = self.linear_1(features_norm)[senders]
        messages = e3nn.tensor_product(messages, sh, filter_ir_out=self.output_irreps)
        radial_message = jax.vmap(self.radial_mlp)(radial_basis)
        messages = messages * radial_message

        messages_agg = e3nn.scatter_sum(
            messages, dst=receivers, output_size=features.shape[0]
        ) / jnp.sqrt(self.avg_n_neighbors)

        messages_agg = e3nn.tensor_product(
            messages_agg,
            x_feats,
            filter_ir_out=self.output_irreps
        )
        features = self.linear_2(messages_agg) + self.skip(species, features)

        # Follow equiformer: gated activation
        features = e3nn.gate(
            features,
            even_act=jax.nn.silu,
            odd_act=jax.nn.sigmoid,
            even_gate_act=jax.nn.silu,
        )

        node_energies = self.readout1(features)
        node_energies = self.readout(node_energies)

        return node_energies, features


class RACE(nnx.Module):
    """RACE model for predicting energies and forces.

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
        l_train: Whether in training mode
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
        self._avg_n_neighbors_value = avg_n_neighbors  # Keep float for sub-module init

        if atom_energies is not None:
            self.atom_energies = nnx.Param(jnp.array(atom_energies))
        else:
            self.atom_energies = nnx.Param(jnp.zeros(n_species, dtype=jnp.float32))

        self.hidden_irreps = e3nn.Irreps(hidden_irreps)
        input_irreps = self.hidden_irreps.filter(keep="0e")
        x_irreps = e3nn.Irreps("1x0e")
        
        self.x_linear = e3nn_nnx_Linear(
            irreps_in=input_irreps,
            irreps_out=x_irreps,
            linear_type="indexed",
            num_indexed_weights=n_species,
            force_irreps_out=True,
            rngs=rngs,
        )

        self.sh_irreps = e3nn.s2_irreps(lmax)

        self.species_embedding = e3nn_nnx_Linear(
            irreps_in=e3nn.Irreps(f"{n_species}x0e"),
            irreps_out=input_irreps,
            rngs=rngs,
        )

        layers = []
        x_readouts = []
        f_readouts = []
        for i in range(n_layers):
            layer = RACEConvolution(
                input_irreps=input_irreps if i == 0 else self.hidden_irreps,
                output_irreps=self.hidden_irreps,
                x_irreps=x_irreps,
                sh_irreps=self.sh_irreps,
                n_species=n_species,
                radial_basis_size=radial_basis_size,
                radial_mlp_size=radial_mlp_size,
                radial_mlp_layers=radial_mlp_layers,
                mlp_init_scale=mlp_init_scale,
                avg_n_neighbors=avg_n_neighbors,
                rngs=rngs,
            )
            layers.append(layer)

            # FIX: Use derived readout_hidden_irreps instead of hardcoded "64x0e"
            readout_hidden_irreps = self.hidden_irreps.filter(keep="0e")
            x_readout = RACEXReadout(
                hidden_irreps=readout_hidden_irreps, #e3nn.Irreps("64x0e"), 
                output_irreps=e3nn.Irreps("0e"),
                x_irreps=x_irreps,
                n_species=n_species,
                avg_n_neighbors=avg_n_neighbors,
                rngs=rngs,
            )
            x_readouts.append(x_readout)

            f_readout = RACEXReadout(
                hidden_irreps=readout_hidden_irreps, #e3nn.Irreps("64x0e"), 
                output_irreps=e3nn.Irreps("0e"),
                x_irreps=input_irreps if i == 0 else self.hidden_irreps,
                n_species=n_species,
                avg_n_neighbors=avg_n_neighbors,
                rngs=rngs,
            )
            f_readouts.append(f_readout)

        self.layers = nnx.List(layers)
        self.x_readouts = nnx.List(x_readouts)
        self.f_readouts = nnx.List(f_readouts)

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
        features, radial_basis, sh = compute_node_edge_features(
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
        x_feats = self.x_linear(data.nodes['species'], features)

        outputs = []
        for (layer, x_readout, f_readout) in zip(
            self.layers,
            self.x_readouts,
            self.f_readouts
        ):
            f_energy = f_readout(
                data.nodes["species"],
                features
            )
            #node_energies, features = jax.checkpoint(layer)(
            node_energies, features = layer(
                features,
                x_feats,
                data.nodes["species"],
                sh,
                radial_basis,
                data.senders,
                data.receivers,
            )
            x_energy = x_readout(
                data.nodes["species"],
                x_feats
            )
            outputs += [
                node_energies.array[:, 0]
                + x_energy.array.squeeze()
                + f_energy.array.squeeze()
            ]

        node_energies = jnp.sum(
            jnp.stack(outputs, axis=1),
            axis=-1,
            keepdims=True
        )
        node_energies = e3nn.IrrepsArray(e3nn.Irreps("0e"), node_energies)

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
        return energy_forces_stress(
            self.node_energies, data, self.periodic,
            self.l_train, self.atom_energies[...],
        )


if __name__ == "__main__":
    # Example usage
    import numpy as np
    import pickle
    from bam_omat24.atom_energies import ATOM_ENERGIES

    print("=" * 70)
    print("RACE-Style Equivariant Potential")
    print("=" * 70)

    with open('omat24_val_0.pkl', 'rb') as f:
        graphset = pickle.load(f)
    data = graphset[0]

    # Create model
    rngs = nnx.Rngs(42)
    model = RACE(
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
        l_train=True,
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
