"""RACE model implementation using Flax NNX.

This module implements the RACE model
using Flax NNX framework with e3nn_jax for equivariant operations.

Changes from original:
- Fixed hardcoded irreps in RACEXReadout (now derived from hidden_irreps)
- Fixed avg_n_neighbors Param usage (now consistent)
- avg_n_neighbors / atom_energies are now non-trainable constants
  (static Python float / tuple) instead of nnx.Param.
- New option ``x_mode``: "species" (default, original behaviour: x_feats is a
  per-species scalar) or "node_scalar" (x_feats is a per-layer linear projection
  of the receiving node's current scalar (0e) features).
- New option ``cov_mode``: "scalar6" (default, original behaviour: 6 invariant
  scalars fill a 3x3 Cholesky factor, which is NOT rotation-equivariant) or
  "irreps" (predict ``1x0e + 1x2e`` and build Sigma = (a*I + B)^2 with B a
  traceless symmetric tensor, which is rotation-equivariant).

"""

from typing import Optional, Sequence, Tuple, Dict

import e3nn_jax as e3nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jraph

from bam_omat24.utils import (
    separated_layer_norm,
    MLP,
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
        if x_feats.irreps.dim > 1:
            x_feats_norm = separated_layer_norm(x_feats) / jnp.sqrt(self.avg_n_neighbors)
        else:
            x_feats_norm = x_feats / jnp.sqrt(self.avg_n_neighbors)
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
        x_mode: str = "species",
        cov_mode: str = "scalar6",
        *,
        rngs: nnx.Rngs,
    ):
        if x_mode not in ("species", "node_scalar"):
            raise ValueError(f"x_mode must be 'species' or 'node_scalar', got {x_mode!r}")
        if cov_mode not in ("scalar6", "irreps"):
            raise ValueError(f"cov_mode must be 'scalar6' or 'irreps', got {cov_mode!r}")

        # Store static configuration
        self.lmax = lmax
        self.cutoff = cutoff
        self.periodic = periodic
        self.n_species = n_species
        self.radial_basis_size = radial_basis_size
        self.radial_polynomial_p = radial_polynomial_p
        self.l_train = l_train
        self.x_mode = x_mode
        self.cov_mode = cov_mode

        # Store trainable scale parameters
        self.shift = nnx.Param(jnp.array(shift))
        self.scale = nnx.Param(jnp.array(scale))

        # Non-trainable constants.  Stored as plain Python objects so they are
        # part of the (hashable) graphdef and never receive gradients/updates.
        self.avg_n_neighbors = float(avg_n_neighbors)
        if atom_energies is not None:
            atom_energies = tuple(float(e) for e in atom_energies)
            if len(atom_energies) != n_species:
                raise ValueError(
                    f"len(atom_energies)={len(atom_energies)} != n_species={n_species}"
                )
        else:
            atom_energies = tuple(0.0 for _ in range(n_species))
        self.atom_energies = atom_energies

        self.hidden_irreps = e3nn.Irreps(hidden_irreps)
        input_irreps = self.hidden_irreps.filter(keep="0e")
        x_irreps = e3nn.Irreps("1x0e")

        # x_mode == "species": a single per-species scalar (original behaviour)
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
        x_projs = []
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

            # x_mode == "node_scalar": project the receiving node's current
            # scalar features (after layer norm) onto x_irreps, per layer.
            if x_mode == "node_scalar":
                layer_input_irreps = input_irreps if i == 0 else self.hidden_irreps
                x_projs.append(
                    e3nn_nnx_Linear(
                        irreps_in=layer_input_irreps.filter(keep="0e"),
                        irreps_out=x_irreps,
                        biases=True,
                        force_irreps_out=True,
                        rngs=rngs,
                    )
                )

        self.layers = nnx.List(layers)
        self.x_readouts = nnx.List(x_readouts)
        self.f_readouts = nnx.List(f_readouts)
        self.x_projs = nnx.List(x_projs)

        if cov_mode == "scalar6":
            # 6 invariant scalars -> 3x3 lower-triangular Cholesky factor.
            # NOTE: not rotation-equivariant (kept for backward compatibility).
            self.frc_logvar = e3nn_nnx_Linear(
                irreps_in=hidden_irreps,
                irreps_out=e3nn.Irreps("6x0e"),
                biases=False,
                rngs=rngs,
            )
        else:
            # 1x0e (isotropic) + 1x2e (traceless symmetric) -> equivariant Sigma
            self.frc_logvar = e3nn_nnx_Linear(
                irreps_in=hidden_irreps,
                irreps_out=e3nn.Irreps("1x0e + 1x2e"),
                biases=False,
                force_irreps_out=True,
                rngs=rngs,
            )

        self.energy_logvar = e3nn_nnx_Linear(
            irreps_in=hidden_irreps,
            irreps_out=e3nn.Irreps("1x0e"),
            biases=False,
            rngs=rngs,
        )

        # Diagonal-only stress variance: 6 params (one per Voigt component)
        self.stress_logvar = e3nn_nnx_Linear(
            irreps_in=hidden_irreps,
            irreps_out=e3nn.Irreps("6x0e"),
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
        # x_mode == "species": per-species scalar, identical for all layers.
        x_feats = self.x_linear(data.nodes['species'], features)

        outputs = []
        for i, (layer, x_readout, f_readout) in enumerate(zip(
            self.layers,
            self.x_readouts,
            self.f_readouts
        )):
            if self.x_mode == "node_scalar":
                # Receiving node's current scalar (0e) features -> 1x0e
                x_feats = self.x_projs[i](
                    separated_layer_norm(features).filter(keep="0e")
                )
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

        frc_logvar = self.frc_logvar(features)
        if self.cov_mode == "scalar6":
            # Force variation: 6 params for 3x3 lower triangular Cholesky L_f
            # [diag0, diag1, diag2, off0, off1, off2] -> softplus on diagonal, min 0.04
            frc_var = jnp.hstack([
                jax.nn.softplus(frc_logvar.array[:, :3]) + 0.04,
                frc_logvar.array[:, 3:]
            ])
        else:
            # Force variation (equivariant): [a, b_0..b_4]
            #   a : isotropic std (0e), softplus + min 0.04
            #   b : 5 components of a 2e irrep (traceless symmetric B)
            # Sigma = (a*I + B')^2 is assembled in training_utils.build_force_L
            frc_var = jnp.hstack([
                jax.nn.softplus(frc_logvar.array[:, :1]) + 0.04,
                frc_logvar.array[:, 1:]
            ])

        # Energy variation: per-atom sigma, min 0.04
        energy_var = jax.nn.softplus(self.energy_logvar(features).array) + 0.04  # (n_atoms, 1)

        # Stress variation: 6 diagonal sigma values (one per Voigt component)
        stress_raw = self.stress_logvar(features).array  # (n_atoms, 6)
        stress_var = jax.nn.softplus(stress_raw) + 0.04

        return {
            'node_energies': node_energies.array,
            'node_features': features,
            'frc_variation': frc_var,
            'energy_variation': energy_var,
            'stress_variation': stress_var,
        }

    @property
    def frc_variation_fill(self) -> jax.Array:
        """Neutral (identity covariance) fill value for padded nodes."""
        if self.cov_mode == "scalar6":
            return jnp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        return jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def __call__(
        self, data: jraph.GraphsTuple
    ) -> Dict:
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
            self.l_train, jnp.asarray(self.atom_energies, dtype=jnp.float32),
            frc_variation_fill=self.frc_variation_fill,
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
    #energies, forces, stress = model(data)
    result = model(data)
    print(f"Energies shape: {result['energy'].shape}")
    print(f"Forces shape: {result['forces'].shape}")
    print(f"Total energy: {result['energy'][0]:.6f}")
    print(f"Force magnitude: {np.linalg.norm(result['forces'], axis=-1).mean():.6f}")
    print(f"Stress: {result['stress']}")
    print(f"Target energies: {data.globals['energy']}")
    print(f"Target stress: {data.globals['stress']} eV/A^3")
