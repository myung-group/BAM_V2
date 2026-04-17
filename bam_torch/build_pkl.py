"""Data preprocessing — PyTorch port of bam_omat24/build_pkl.py.

Converts ASE Atoms → GraphData (numpy-based), saves as pickle.
Compatible with bam_torch.data.GraphDataset.
"""

import pickle
from pathlib import Path

import ase
import matscipy.neighbours
import numpy as np
from tqdm import tqdm

from bam_torch.atom_energies import ATOM_ENERGIES, ATOMIC_NUMBER_TO_INDEX
from bam_torch.data import GraphData


def preprocess_graph(
    atoms: ase.Atoms,
    default_cutoff: float,
    targets: bool = True,
) -> GraphData | None:
    """Convert ase.Atoms → GraphData.

    Args:
        atoms:          ASE Atoms object.
        default_cutoff: Neighbour list cutoff (Å).
        targets:        Whether to include forces / energy / stress.

    Returns:
        GraphData or None if no neighbours found.
    """
    cutoff = default_cutoff
    src, dst, Sij = matscipy.neighbours.neighbour_list("ijS", atoms, cutoff)

    if len(src) < 20:
        cutoff = default_cutoff + 3.0
        src, dst, Sij = matscipy.neighbours.neighbour_list("ijS", atoms, cutoff)
        if len(src) == 0:
            return None

    species = np.array(
        [ATOMIC_NUMBER_TO_INDEX[n] for n in atoms.get_atomic_numbers()],
        dtype=np.int32,
    )

    node_e0  = ATOM_ENERGIES[species]
    graph_e0 = node_e0.sum()

    forces   = atoms.get_forces().astype(np.float32)   if targets else None
    energy   = np.array([atoms.get_potential_energy() - graph_e0], dtype=np.float32) \
               if targets else np.array([0.0], dtype=np.float32)
    stress   = np.array([atoms.get_stress()], dtype=np.float32) \
               if targets else np.zeros((1, 6), dtype=np.float32)

    cell     = np.array([atoms.get_cell()[:]],   dtype=np.float32)
    volume   = np.array([atoms.get_volume()],    dtype=np.float32)
    g_cutoff = np.array([cutoff],                dtype=np.float32)

    nodes: dict = {
        'species':   species,
        'positions': atoms.positions.astype(np.float32),
    }
    if forces is not None:
        nodes['forces'] = forces

    return GraphData(
        n_node=np.array([len(atoms)], dtype=np.int32),
        n_edge=np.array([len(src)],   dtype=np.int32),
        nodes=nodes,
        edges={'Sij': Sij.astype(np.float32)},
        senders=dst.astype(np.int32),
        receivers=src.astype(np.int32),
        globals={
            'energy':  energy,
            'cell':    cell,
            'volume':  volume,
            'stress':  stress,
            'cutoff':  g_cutoff,
        },
    )


if __name__ == "__main__":
    from fairchem.core.datasets import AseDBDataset

    dir_train   = "/home/willow/Python/BAM-nequip-main/omat24_1M/train/"
    sample_list = [
        "aimd-from-PBE-1000-npt", "aimd-from-PBE-1000-nvt",
        "aimd-from-PBE-3000-npt", "aimd-from-PBE-3000-nvt",
        "rattled-1000", "rattled-1000-subsampled",
        "rattled-300",  "rattled-300-subsampled",
        "rattled-500",  "rattled-500-subsampled",
        "rattled-relax",
    ]
    trainset_paths = [dir_train + s for s in sample_list]
    trainset = AseDBDataset(config=dict(src=trainset_paths))

    ndata  = len(trainset)
    rng    = np.random.RandomState(seed=42)
    perm   = rng.permutation(ndata)
    nbatch = 55000
    cutoff = 6.0
    ipkl   = 0

    for ist in range(0, ndata, nbatch):
        ied      = min(ist + nbatch, ndata)
        pkl_path = f"omat24_train_{ipkl}.pkl"
        print(f"Preprocessing {ist}~{ied} → {pkl_path}")
        graphs = [
            preprocess_graph(trainset.get_atoms(perm[i]), cutoff, targets=True)
            for i in tqdm(range(ist, ied))
        ]
        safe = [g for g in graphs if g is not None]
        print(f"  Valid graphs: {len(safe)}")
        with open(pkl_path, "wb") as f:
            pickle.dump(safe, f)
        ipkl += 1
