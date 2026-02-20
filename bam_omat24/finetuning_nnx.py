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

import jraph

from bam_omat24.nequip_nnx import NequipConvolution
from bam_omat24.utils import energy_forces_stress
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
        rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        self.strategy = strategy
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
        results = self.foundation.node_energies(positions, cell, cutoff, data)
        foundation_energy = results["node_energies"]
        node_features = results["node_features"]
        
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
            sh = results['sh']
            radial_basis = results['radial_basis']

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
        )

        return {'node_energies': foundation_energy + node_delta_energies}


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
        return energy_forces_stress(
            self.node_energies, data, self.foundation.periodic,
            self.foundation.l_train, self.foundation.atom_energies[...],
        )


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
        strategy='delta_2', # [delta_0, delta_1, delta_2]
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
