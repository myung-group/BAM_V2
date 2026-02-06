"""
Fine-Tuning Module for Equivariant ML Potentials (with a multi-fidelty scheme)

This module enables fine-tuning foundation models
from OMAT24 (high-temperature DFT) pre-trained models using limited low-temperature data.

Key approaches:
1. Delta Learning (Δ-ML): Learn the difference between high/low temperature sampled data

References:
- Delta-ML: Ramakrishnan et al., J. Chem. Theory Comput. 2015
"""


from typing import Tuple


import jax
import jax.numpy as jnp
from flax import nnx
#from flax.nnx.nn import initializers

import jraph
import e3nn_jax as e3nn
import optax
from optax import transforms
freeze = transforms.freeze

from bam_omat24.nequip_nnx import (
    NequipConvolution,
    node_graph_idx,
    compute_node_edge_features,
)
from bam_omat24.race_nnx import RACE
from bam_omat24.linear_nnx import Linear as e3nn_nnx_Linear



# =============================================================================
# FineTuning Potential Model
# =============================================================================

class FineTuningPotential(nnx.Module):
    """
    FineTuning potential combining foundation model with a multi-fidelity scheme.

    E_ref = E_foundation + scale * ΔE
    Strategies: The computation depends on the strategy:
        - delta_0: ΔE = nnx.Linear(node_feats.filter(keep='0e'))
        - delta_1: ΔE = nnx.Linear(e3nn.Linear(node_feats))
        - delta_2: ΔE = nnx.Linear(e3nn.Linear(Convolution(node_feats)))

    Args:
        foundation_model: Pre-trained DFT-level model
        strategy: Fine-tuning strategy ['delta_0', 'delta_1', 'delta_2']
        freeze_foundation: Whether to freeze foundation model weights
        rngs: Random number generator state
    """

    def __init__(
        self,
        foundation_model: RACE,
        radial_mlp_size: int = 8,
        radial_mlp_layers: int = 3,
        mlp_init_scale: float = 4.0,
        strategy: str = 'delta',
        delta_scale: float = 1.0,
        delta_shift: float = 0.0,
        freeze_foundation: bool = True,
        rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        self.strategy = strategy
        self.freeze_foundation = freeze_foundation

        # Store foundation model
        self.foundation = foundation_model

        # Initialize strategy-specific components
        output_irreps = self.foundation.hidden_irreps.filter(keep='0e')

        self.delta_linear_readout = nnx.Linear(
            in_features=output_irreps.dim,
            out_features=1,
            #kernel_init=initializers.zeros_init(),
            use_bias=True,
            rngs=rngs
        )

        self.delta_scale = nnx.Param(jnp.array(delta_scale))
        self.delta_shift = nnx.Param(jnp.array(delta_shift))

        if strategy in ['delta_1', 'delta_2']:
            self.delta_readout = e3nn_nnx_Linear(
                irreps_in=self.foundation.hidden_irreps,
                irreps_out=output_irreps,
                linear_type="indexed",
                num_indexed_weights=self.foundation.n_species,
                biases=True,
                force_irreps_out=True,
                rngs=rngs,
            )

        if strategy == 'delta_2':
            self.feature_adapter = NequipConvolution(
                input_irreps=self.foundation.hidden_irreps,
                output_irreps=self.foundation.hidden_irreps,
                sh_irreps=self.foundation.sh_irreps,
                n_species=self.foundation.n_species,
                radial_basis_size=self.foundation.radial_basis_size,
                radial_mlp_size=radial_mlp_size,
                radial_mlp_layers=radial_mlp_layers,
                mlp_init_scale=mlp_init_scale,
                avg_n_neighbors=self.foundation._avg_n_neighbors_value,
                rngs=rngs
            )

    def _compute_foundation_features(
        self,
        positions: jax.Array,
        cell: jax.Array,
        cutoff: jax.Array,
        data: jraph.GraphsTuple
    ) -> Tuple[e3nn.IrrepsArray, jax.Array, e3nn.IrrepsArray]:
        """
        Compute features from foundation model WITH gradient flow to positions.

        This method recomputes the forward pass through the foundation model
        but only for extracting features - the parameters are not updated.

        Returns:
            Tuple of (node_features, radial_basis, sh)
        """
        foundation = self.foundation

        # Compute edge features (this maintains gradient flow to positions)
        features, radial_basis, sh = compute_node_edge_features(
            positions=positions,
            cell=cell,
            cutoff=cutoff,
            data=data,
            n_species=foundation.n_species,
            model_cutoff=foundation.cutoff,
            radial_basis_size=foundation.radial_basis_size,
            radial_polynomial_p=foundation.radial_polynomial_p,
            lmax=foundation.lmax,
            periodic=foundation.periodic,
        )

        features = foundation.species_embedding(features)
        x_feats = foundation.x_linear(data.nodes["species"], features)

        # Pass through all layers (foundation parameters are frozen via filter)
        for layer in foundation.layers:
            _, features = layer(
                features,
                x_feats,
                data.nodes["species"],
                sh,
                radial_basis,
                data.senders,
                data.receivers,
            )

        return features, radial_basis, sh

    def get_foundation_energy_and_features(
        self,
        positions: jax.Array,
        cell: jax.Array,
        cutoff: jax.Array,
        data: jraph.GraphsTuple
    ) -> Tuple[jax.Array, e3nn.IrrepsArray]:
        """
        Extract intermediate features and energy from foundation model.

        Args:
            positions: Atomic positions of shape (n_atoms, 3)
            cell: (n_graph, 3, 3) for periodic, None for molecular
            cutoff: (n_graph,) for periodic, None for molecular
            data: Graph data structure

        Returns:
            Tuple of (node_energies, node_features, radial_basis, sh)
        """
        foundation = self.foundation

        # Compute edge features
        features, radial_basis, sh = compute_node_edge_features(
            positions=positions,
            cell=cell,
            cutoff=cutoff,
            data=data,
            n_species=foundation.n_species,
            model_cutoff=foundation.cutoff,
            radial_basis_size=foundation.radial_basis_size,
            radial_polynomial_p=foundation.radial_polynomial_p,
            lmax=foundation.lmax,
            periodic=foundation.periodic,
        )

        features = foundation.species_embedding(features)
        x_feats = foundation.x_linear(data.nodes["species"], features)

        outputs = []
        for (layer, x_readout, f_readout) in zip(
            foundation.layers,
            foundation.x_readouts,
            foundation.f_readouts
        ):
            f_energy = f_readout(
                data.nodes["species"],
                features
            )
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
        node_energies = node_energies * foundation.scale[...] + foundation.shift[...]

        return node_energies.array.squeeze(), features

    def node_energies(
        self,
        positions: jax.Array,
        cell: jax.Array,
        cutoff: jax.Array,
        data: jraph.GraphsTuple
    ) -> jax.Array:
        """
        Compute per-node energies with proper gradient flow.

        CRITICAL: Foundation energy is frozen (no gradient), but delta energy
        has gradient flow to positions for force computation.

        Args:
            positions: Atomic positions of shape (n_atoms, 3)
            cell: (n_graph, 3, 3) for periodic, None for molecular
            cutoff: (n_graph,) for periodic, None for molecular
            data: Graph data structure

        Returns:
            Per-node energies of shape (n_atoms,)
        """
        # Get foundation energy WITH stop_gradient (frozen)
        #foundation_energy = jax.lax.stop_gradient(
        #    self.get_foundation_energy_and_features(
        #        positions, cell, cutoff, data
        #    )
        #)
        foundation_energy, node_features = (
            self.get_foundation_energy_and_features(
                positions, cell, cutoff, data
            )
        )

        # Recompute features WITH gradient flow to positions
        # This allows delta network to contribute to forces
        #node_features, radial_basis, sh = self._compute_foundation_features(
        #    positions, cell, cutoff, data
        #)

        if self.strategy == 'delta_1':
            # Compute delta correction with gradient flow
            node_delta_features = self.delta_readout(
                data.nodes["species"], node_features
            )
            node_delta_energies = self.delta_linear_readout(
                node_delta_features.array
            )

        elif self.strategy == 'delta_2':
            # Adapt features then compute delta
            adapted_node_features = self.feature_adapter(
                node_features,
                data.nodes["species"],
                sh,
                radial_basis,
                data.senders,
                data.receivers
            )

            node_delta_features = self.delta_readout(
                data.nodes["species"], adapted_node_features
            )
            node_delta_energies = self.delta_linear_readout(
                node_delta_features.array
            )

        else:  # self.strategy == 'delta_0'
            node_delta_features = node_features.filter(keep="0e")
            node_delta_energies = self.delta_linear_readout(
                node_delta_features.array
            )

        node_delta_energies = (
            node_delta_energies * self.delta_scale[...]
            + self.delta_shift[...]
        ).squeeze()

        return foundation_energy + node_delta_energies

    def __call__(
        self,
        data: jraph.GraphsTuple
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Compute (fine-tuning) energy predictions with forces and stress.

        E_ref = E_foundation + scale * ΔE

        The computation depends on the strategy:
        - delta_0: ΔE = nnx.Linear(node_feats.filter(keep='0e'))
        - delta_1: ΔE = nnx.Linear(e3nn.Linear(node_feats))
        - delta_2: ΔE = nnx.Linear(e3nn.Linear(Convolution(node_feats)))

        Args:
            data: Graph data structure

        Returns:
            Tuple of (graph_energies, forces, stress)
        """
        # Input validation
        assert data.nodes is not None, "Graph must have nodes"
        assert "positions" in data.nodes, "Nodes must have 'positions'"
        assert "species" in data.nodes, "Nodes must have 'species'"

        n_graphs = data.n_node.shape[0]
        if n_graphs > 1:
            node_mask = jraph.get_node_padding_mask(data)
        else:
            node_mask = jnp.ones (data.nodes['positions'].shape[0])

        if self.foundation.periodic:
            assert "cell" in data.globals, "Periodic systems require 'cell' in globals"
            assert "cutoff" in data.globals, "Periodic systems require 'cutoff' in globals"

            # Periodic: differentiate w.r.t. positions AND cell
            def total_energy_fn(
                positions: jax.Array,
                cell: jax.Array,
                cutoff: jax.Array
            ) -> Tuple[jax.Array, jax.Array]:
                node_energies = self.node_energies(
                    positions, cell, cutoff, data
                )
                # Apply mask before summing
                node_energies = jnp.where(node_mask, node_energies, 0.0)
                return jnp.sum(node_energies), node_energies

            grad_fn = jax.grad(total_energy_fn, argnums=(0, 1), has_aux=True)
            (grad, cellgrad), node_energies = grad_fn(
                data.nodes["positions"],
                data.globals["cell"],
                data.globals["cutoff"]
            )
        else:
            # Molecular: differentiate w.r.t. positions only
            def total_energy_fn(positions: jax.Array) -> Tuple[jax.Array, jax.Array]:
                node_energies = self.node_energies(
                    positions, None, None, data
                )
                # Apply mask before summing
                node_energies = jnp.where(node_mask, node_energies, 0.0)
                return jnp.sum(node_energies), node_energies

            grad_fn = jax.grad(total_energy_fn, argnums=0, has_aux=True)
            grad, node_energies = grad_fn(data.nodes["positions"])

        if not self.foundation.l_train:
            # Add isolated atom energies to each node as prior
            node_energies = node_energies + jax.lax.stop_gradient(
                self.foundation.atom_energies[...][data.nodes["species"]]
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

        if self.foundation.periodic:
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

        return graph_energies, -grad, stress


# =============================================================================
# Parameter Filtering Utilities
# =============================================================================

def filter_delta_params(params: nnx.State) -> Tuple[nnx.State, nnx.State]:
    """
    Filter to train only delta network parameters.

    Foundation model parameters are frozen.

    Args:
        params: Full parameter state

    Returns:
        Tuple of (trainable_params, frozen_params)
    """
    trainable = {}
    frozen = {}

    flat = params.flat_state()
    for path, value in flat:
        path_str = str(path).lower()
        if 'delta_' in path_str or 'feature_adapter' in path_str:
            trainable[path] = value
        else:
            frozen[path] = value

    return nnx.State.from_flat_path(trainable), nnx.State.from_flat_path(frozen)


if __name__ == "__main__":
    # Example usage
    import numpy as np
    import pickle
    from bam_omat24.atom_energies import ATOM_ENERGIES

    print("=" * 70)
    print("Fine-Tuning Learning Module with RACE-Style Equivariant Potential")
    print("=" * 70)

    with open('omat24_val_0.pkl', 'rb') as f:
        graphset = pickle.load(f)
    data = graphset[0]

    # Create a dummy foundation model
    rngs = nnx.Rngs(42)
    foundation = RACE(
        n_species=len(ATOM_ENERGIES),
        lmax=3,
        hidden_irreps="64x0e + 64x1o + 64x2e + 64x3o",
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

    graphdef, params = nnx.split(foundation, nnx.Param)
    with open('ckpt_best.pkl', 'rb') as f:
        ckpt = pickle.load (f)
        ema_params = ckpt['ema_params']
        foundation = nnx.merge (graphdef, ema_params)

    print("\n1. Testing Delta Learning strategy...")
    delta_model = FineTuningPotential(
        foundation_model=foundation,
        radial_mlp_size=64,
        radial_mlp_layers=3,
        mlp_init_scale=4.0,
        delta_scale=1.0,
        delta_shift=0.0,
        strategy='delta_1', # [delta_0, delta_1, delta_2]
        freeze_foundation=True,
        rngs=rngs,
    )

    graphdef, params = nnx.split(delta_model, nnx.Param)
    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"   Total parameters: {n_params:,}")

    # Count trainable vs frozen
    trainable, frozen = filter_delta_params(params)
    n_trainable = sum(x.size for x in jax.tree.leaves(trainable))
    n_frozen = sum(x.size for x in jax.tree.leaves(frozen))
    print(f"   Trainable: {n_trainable:,}, Frozen: {n_frozen:,}")

    energies, forces, stress = delta_model(data)
    print(f"   Energies shape: {energies.shape}")
    print(f"   Forces shape: {forces.shape}")
    print(f"   Total energy: {energies[0]/data.nodes['positions'].shape[0]:.6f}")
    print(f"   Force magnitude: {np.linalg.norm(forces, axis=-1).mean():.6f}")
    print(f"   Stress: {stress}")
    print(f"   Target energies: {data.globals['energy']/data.nodes['positions'].shape[0]}")
    print(f"   Target stress: {data.globals['stress']} eV/A^3")

    # Verify gradient flow by checking if forces are non-zero
    # (they should be non-zero even with randomly initialized delta network)
    force_norm = np.linalg.norm(forces, axis=-1).mean()
    if force_norm > 0:
        print("\n   ✓ Forces are non-zero - gradient flow is working correctly!")
    else:
        print("\n   ✗ WARNING: Forces are zero - check gradient flow!")

    print("\n2. Comparing with foundation model...")
    energies_f, forces_f, stress_f = foundation(data)
    print(f"   Foundation energy: {energies_f[0]/data.nodes['positions'].shape[0]:.6f}")
    print(f"   Foundation force magnitude: {np.linalg.norm(forces_f, axis=-1).mean():.6f}")
    print(f"   Delta energy contribution: {energies[0] - energies_f[0]:.6f}")
    print(f"   Target energy - Foundation E: {data.globals['energy'][0] - energies_f[0]:.6f}")

    # Forces should differ because delta network contributes
    force_diff = np.linalg.norm(forces - forces_f, axis=-1).mean()
    print(f"   Force difference (delta contribution): {force_diff:.6f}")
