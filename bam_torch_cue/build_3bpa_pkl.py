import pickle
from pathlib import Path

import ase
import matscipy.neighbours
import numpy as np
from tqdm import tqdm

from bam_torch.data import GraphData


def preprocess_graph(
    atoms: ase.Atoms,
    atom_indices: dict[int, int],
    default_cutoff: float,
    targets: bool = True,
) -> GraphData | None:
    """Preprocess ase.Atoms object into a jraph.GraphsTuple, with optional targets"""
    cutoff = default_cutoff
    src, dst, Sij = matscipy.neighbours.neighbour_list("ijS", atoms, cutoff)

    atomic_number_to_index = {
        1: 0,
        6: 1,
        7: 2,
        8: 3
    }
    atom_energies = np.array(
        [
            -13.587222780835477,
            -1029.4889999855063,
            -1484.9814568572233,
            -2041.9816003861047
        ]
    )

    species = np.array(
        [ atomic_number_to_index[n] for n in atoms.get_atomic_numbers()],
        dtype=np.int32
    )
    node_e0 = atom_energies[species]
    graph_e0 = node_e0.sum()

    forces = atoms.get_forces().astype(np.float32) if targets else None
    energy = np.array([atoms.get_potential_energy()-graph_e0], dtype=np.float32) \
            if targets else np.array([0.0], dtype=np.float32)

    nodes: dict = {
        'species': species,
        'positions' : atoms.positions.astype(np.float32)
    }
    if forces is not None:
        nodes['forces'] = forces

    return GraphData(
        n_node=np.array([len(atoms)]).astype(np.int32),
        n_edge=np.array([len(src)]).astype(np.int32),
        nodes=nodes,
        edges = {'Sij':Sij},
        senders=dst.astype(np.int32),
        receivers=src.astype(np.int32),
        globals={
            'energy': energy,
            'cutoff': cutoff,
        },

    )