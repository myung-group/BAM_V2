"""
Fine-Tuning Module for Equivariant ML Potentials (with a multi-fidelty scheme)

This module enables fine-tuning foundation models
from OMAT24 (high-temperature DFT) pre-trained models using limited low-temperature data.

Key approaches:
1. Delta Learning (Δ-ML): Learn the difference between high/low temperature sampled data

References:
- Delta-ML: Ramakrishnan et al., J. Chem. Theory Comput. 2015
"""


from typing import Tuple, Dict


import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from flax import nnx

import jraph

from bam_omat24.utils import energy_forces_stress, MLP
from bam_omat24.race_nnx import RACE
from bam_omat24.linear_nnx import Linear as e3nn_nnx_Linear



# =============================================================================
# FineTuning Potential Model
# =============================================================================

class FineTuningPotential(nnx.Module):
    """
    FineTuning potential combining foundation model with an affine transformation.

    E_tgt = MLP_scale(node_feats)*E_foundation + MLP_shift(node_feats)

    Args:
        foundation_model: Pre-trained DFT-level model
        rngs: Random number generator state
    """

    def __init__(
        self,
        foundation_model: RACE,
        mlp_init_scale: float = 4.0,
        mlp_hidden_size: int = 64,
        mlp_hidden_layers: int = 2,
        rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        # Store foundation model
        self.foundation = foundation_model

        # Initialize strategy-specific components
        output_irreps = self.foundation.hidden_irreps.filter(keep='0e')

        self.ft_readout = e3nn_nnx_Linear(
                irreps_in=self.foundation.hidden_irreps,
                irreps_out=output_irreps,
                linear_type="indexed",
                num_indexed_weights=self.foundation.n_species,
                biases=True,
                force_irreps_out=True,
                rngs=rngs,
            )


        # feature-dependent affine transformation
        # new_node_energy = scale(features) * foundation_energy + shift(features)
        scalar_dim = output_irreps.dim
        scale_sizes = [scalar_dim] + [mlp_hidden_size] * mlp_hidden_layers + [1]
        shift_sizes = [scalar_dim] + [mlp_hidden_size] * mlp_hidden_layers + [1]

        self.ft_scale_mlp = MLP(
            sizes=scale_sizes,
            init_scale=mlp_init_scale,
            use_bias=True,
            rngs=rngs,
        )
        self.ft_shift_mlp = MLP(
            sizes=shift_sizes,
            init_scale=mlp_init_scale,
            use_bias=True,
            rngs=rngs,
        )

        # Zero-init last layer kernels so output = bias at initialization
        # Scale MLP: bias=1 → initial scale is 1
        # Shift MLP: bias=0 → initial shift is 0
        last_scale_layer = self.ft_scale_mlp.layers[-1]
        last_scale_layer.weights = nnx.Param(jnp.zeros_like(last_scale_layer.weights[...]))
        last_scale_layer.bias = nnx.Param(jnp.ones(1))

        last_shift_layer = self.ft_shift_mlp.layers[-1]
        last_shift_layer.weights = nnx.Param(jnp.zeros_like(last_shift_layer.weights[...]))
        last_shift_layer.bias = nnx.Param(jnp.zeros(1))

        # Variation heads (ft_ prefix → trainable via freeze_mask)
        hidden_irreps = self.foundation.hidden_irreps
        self.ft_frc_logvar = e3nn_nnx_Linear(
            irreps_in=hidden_irreps,
            irreps_out=e3nn.Irreps("6x0e"),
            biases=False,
            rngs=rngs,
        )
        self.ft_energy_logvar = e3nn_nnx_Linear(
            irreps_in=hidden_irreps,
            irreps_out=e3nn.Irreps("1x0e"),
            biases=False,
            rngs=rngs,
        )
        # 6x6 lower triangular: 6 diagonal + 15 off-diagonal = 21 params
        self.ft_stress_logvar = e3nn_nnx_Linear(
            irreps_in=hidden_irreps,
            irreps_out=e3nn.Irreps("21x0e"),
            biases=False,
            rngs=rngs,
        )

    def node_energies(
        self,
        positions: jax.Array,
        cell: jax.Array,
        cutoff: jax.Array,
        data: jraph.GraphsTuple
    ) -> Dict:
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


        # Feature-dependent affine transformation of foundation energy
        scalar_features = self.ft_readout(
                data.nodes["species"],
                node_features
            )
        #scalar_features = node_features.filter(keep="0e").array
        scale = self.ft_scale_mlp(scalar_features.array)  # (n_atoms, 1)
        shift = self.ft_shift_mlp(scalar_features.array)  # (n_atoms, 1)
        node_energies = scale * foundation_energy + shift


        # Variation heads (own ft_ parameters, trainable)
        # Force variation: 3x3 Cholesky L
        frc_logvar = self.ft_frc_logvar(node_features)
        frc_var = jnp.hstack([
            jax.nn.softplus(frc_logvar.array[:, :3]) + 0.04,
            frc_logvar.array[:, 3:]
        ])
        # Energy variation: per-atom sigma
        energy_var = jax.nn.softplus(
            self.ft_energy_logvar(node_features).array
        ) + 0.04
        # Stress variation: 6x6 Cholesky L_s
        stress_raw = self.ft_stress_logvar(node_features).array
        stress_var = jnp.concatenate([
            jax.nn.softplus(stress_raw[:, :6]) + 0.04,
            stress_raw[:, 6:]
        ], axis=-1)

        return {'node_energies': node_energies,
                'frc_variation': frc_var,
                'energy_variation': energy_var,
                'stress_variation': stress_var}


    def __call__(
        self,
        data: jraph.GraphsTuple
    ) -> Dict:
        """
        Compute (fine-tuning) energy predictions with forces and stress.

        E_ref = MLP_scale * E_foundation + MLP_shift

        Args:
            data: Graph data structure

        Returns:
            Dict (graph_energies, forces, stress)
        """
        return energy_forces_stress(
            self.node_energies, data, self.foundation.periodic,
            self.foundation.l_train, self.foundation.atom_energies[...],
        )


# =============================================================================
# Parameter Filtering Utilities
# =============================================================================

def filter_ft_params(params: nnx.State) -> Tuple[nnx.State, nnx.State]:
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
        if 'ft_' in path_str: # or 'delta_3' in path_str:
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
        mlp_init_scale=4.0,
        mlp_hidden_size=64,
        mlp_hidden_layers=2,
        rngs=rngs,
    )

    graphdef, params = nnx.split(delta_model, nnx.Param)
    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"   Total parameters: {n_params:,}")

    # Count trainable vs frozen
    trainable, frozen = filter_ft_params(params)
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
