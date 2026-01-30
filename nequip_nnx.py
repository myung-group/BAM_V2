"""NequIP model implementation using Flax NNX.

This module implements the NequIP (Neural Equivariant Interatomic Potential) model
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

from bam_nequip.linear_nnx import Linear as e3nn_nnx_Linear


def bessel_basis(x: jax.Array, num_basis: int, r_max: float) -> jax.Array:
    """Compute Bessel radial basis functions.

    Args:
        x: Distance array of shape (n_edges,)
        num_basis: Number of basis functions
        r_max: Cutoff radius

    Returns:
        Bessel basis values of shape (n_edges, num_basis)
    """
    prefactor = 2.0 / r_max
    bessel_weights = jnp.linspace(1.0, num_basis, num_basis) * jnp.pi
    x = x[:, None]
    return prefactor * jnp.where(
        x == 0.0,
        bessel_weights / r_max,
        jnp.sin(bessel_weights * x / r_max) / x,
    )


def polynomial_cutoff(x: jax.Array, r_max: float, p: float) -> jax.Array:
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


def separated_layer_norm (
    input: e3nn.IrrepsArray
)->e3nn.IrrepsArray:

    xi_s = jnp.zeros ( (input.shape[0], 1) )
    cnt  = 0
    for (mul, ir), chunk in zip (input.irreps, input.chunks):
        if not ir.is_scalar():
            xi = chunk.reshape(-1,mul*ir.dim)
            xi_s += jnp.sum(xi**2, axis=-1, keepdims=True)
            cnt  += mul*ir.dim
    xi_rms = jnp.sqrt ( xi_s/cnt )

    new_chunks = []
    for (mul, ir), chunk in zip (input.irreps, input.chunks):
        if ir.is_scalar():
            xi = chunk.reshape(-1,mul)
            xi_centered = xi - jnp.mean(xi, axis=-1, keepdims=True)
            xi_rms = jnp.sqrt (jnp.mean(xi_centered**2, axis=-1, keepdims=True)+1e-6)
            chunk = (xi_centered/xi_rms).reshape(-1, mul, ir.dim)
        else:
            xi = chunk.reshape(-1,mul*ir.dim)
            chunk = (xi/xi_rms).reshape(-1, mul, ir.dim)

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
        #self.layers = layers

    def __call__(self, x: jax.Array) -> jax.Array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x


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

        # add extra irreps to output to account for gate
        gate_irreps = e3nn.Irreps(
            f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e"
        )
        gated_output_irreps = (output_irreps + gate_irreps).regroup()

        self.linear_2 = e3nn_nnx_Linear(
            irreps_in=tp_irreps,
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

    def __call__(
        self,
        features: e3nn.IrrepsArray,
        species: jax.Array,
        sh: e3nn.IrrepsArray,
        radial_basis: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
    ) -> e3nn.IrrepsArray:
        # follow equiformer_v2
        features_norm = separated_layer_norm (features)
        messages = self.linear_1(features_norm)[senders]
        messages = e3nn.tensor_product(messages, sh, filter_ir_out=self.output_irreps)
        radial_message = jax.vmap(self.radial_mlp)(radial_basis)
        messages = messages * radial_message

        messages_agg = e3nn.scatter_sum(
            messages, dst=receivers, output_size=features.shape[0]
        ) / jnp.sqrt(self.avg_n_neighbors)

        features = self.linear_2(messages_agg) + self.skip(species, features)

        # follow equiformer
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

        # Store trainable scale parameters
        self.shift = nnx.Param(jnp.array(shift))
        self.scale = nnx.Param(jnp.array(scale))
        self.avg_n_neighbors = nnx.Param(jnp.array(avg_n_neighbors))

        if atom_energies is not None:
            self.atom_energies = nnx.Param(jnp.array(atom_energies))
        else:
            self.atom_energies = nnx.Param(jnp.zeros(n_species, dtype=jnp.float32))

        input_irreps = e3nn.Irreps(f"{hidden_size}x0e")
        sh_irreps = e3nn.s2_irreps(lmax)
        hidden_irreps = hidden_size * e3nn.s2_irreps(lmax)

        self.species_embedding = e3nn_nnx_Linear(
            irreps_in=e3nn.Irreps(f"{n_species}x0e"),
            irreps_out=input_irreps,
            rngs=rngs,
        )

        layers = []
        for i in range(n_layers):
            layer = NequipConvolution(
                input_irreps=input_irreps if i == 0 else hidden_irreps,
                output_irreps=hidden_irreps,
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

        hidden_irreps_2 = (hidden_size//2) * e3nn.s2_irreps(lmax)
        self.readout1 = e3nn_nnx_Linear(
            irreps_in=hidden_irreps,
            irreps_out=hidden_irreps_2,
            biases=True, # willow
            rngs=rngs,
        )
        self.readout = e3nn_nnx_Linear(
            irreps_in=hidden_irreps_2,
            irreps_out=e3nn.Irreps("0e"),
            biases=True, # willow
            rngs=rngs,
        )

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

        # scale and shift energies
        node_energies = node_energies * (
            self.scale[...]
        ) + (self.shift[...])

        # add isolated atom energies to each node as prior
        node_energies = node_energies + jax.lax.stop_gradient(
            self.atom_energies[...][data.nodes["species"], None]
        )

        return node_energies

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
            node_energies = self.node_energies(positions, data).array
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


def save_model(path: str, model: Nequip, config: dict):
    """Save a model and its config to a file.

    Args:
        path: File path to save the model
        model: NequIP model instance
        config: Model configuration dictionary
    """
    import pickle

    # Get the model state (parameters only)
    graphdef, state = nnx.split(model)

    # Convert state to a serializable format
    state_dict = {}
    for path_tuple, value in state.flat_state().items():
        # Convert path tuple to string key
        key = '/'.join(str(p) for p in path_tuple)
        if hasattr(value, 'value'):
            state_dict[key] = value.value
        else:
            state_dict[key] = value

    with open(path, "wb") as f:
        config_str = json.dumps(config)
        f.write((config_str + "\n").encode())
        # Use pickle for reliable array serialization
        pickle.dump(state_dict, f)


def load_model(path: str) -> tuple[Nequip, dict]:
    """Load a model and its config from a file.

    Args:
        path: File path to load the model from

    Returns:
        Tuple of (model, config)
    """
    import pickle

    with open(path, "rb") as f:
        config = json.loads(f.readline().decode())
        state_dict = pickle.load(f)

    # Create model with dummy rngs
    model = Nequip(
        n_species=len(config["atomic_numbers"]),
        hidden_size=config["hidden_size"],
        lmax=config["lmax"],
        n_layers=config["n_layers"],
        radial_basis_size=config["radial_basis_size"],
        radial_mlp_size=config["radial_mlp_size"],
        radial_mlp_layers=config["radial_mlp_layers"],
        radial_polynomial_p=config["radial_polynomial_p"],
        mlp_init_scale=config["mlp_init_scale"],
        rngs=nnx.Rngs(0),
    )

    # Get the model's state structure
    graphdef, state = nnx.split(model)
    flat_state = state.flat_state()

    # Update with loaded values
    for path_tuple, var in flat_state.items():
        key = '/'.join(str(p) for p in path_tuple)
        if key in state_dict:
            if hasattr(var, 'value'):
                var.value = state_dict[key]

    # Reconstruct model
    state = nnx.State.from_flat_path(flat_state)
    model = nnx.merge(graphdef, state)

    return model, config


# Alternative serialization using orbax (recommended for production)
def save_model_orbax(path: str, model: Nequip, config: dict):
    """Save model using orbax checkpointing (more robust).

    Requires: pip install orbax-checkpoint

    Args:
        path: Directory path to save the checkpoint
        model: NequIP model instance
        config: Model configuration dictionary
    """
    import orbax.checkpoint as ocp

    _, state = nnx.split(model)

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(
        path,
        {
            "state": state,
            "config": config,
        }
    )


def load_model_orbax(path: str) -> tuple[Nequip, dict]:
    """Load model using orbax checkpointing.

    Args:
        path: Directory path to load the checkpoint from

    Returns:
        Tuple of (model, config)
    """
    import orbax.checkpoint as ocp

    checkpointer = ocp.StandardCheckpointer()
    restored = checkpointer.restore(path)
    config = restored["config"]

    # Create model structure
    model = Nequip(
        n_species=len(config["atomic_numbers"]),
        hidden_size=config["hidden_size"],
        lmax=config["lmax"],
        n_layers=config["n_layers"],
        radial_basis_size=config["radial_basis_size"],
        radial_mlp_size=config["radial_mlp_size"],
        radial_mlp_layers=config["radial_mlp_layers"],
        radial_polynomial_p=config["radial_polynomial_p"],
        mlp_init_scale=config["mlp_init_scale"],
        rngs=nnx.Rngs(0),
    )

    # Merge with restored state
    graphdef, _ = nnx.split(model)
    model = nnx.merge(graphdef, restored["state"])

    return model, config


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
    print("Nequip-Style Equivariant Potential")
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
    model = Nequip(
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
