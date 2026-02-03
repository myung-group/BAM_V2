import pickle
from pathlib import Path
import ase
import jraph
import matscipy.neighbours
import numpy as np
from tqdm import tqdm
from fairchem.core.datasets import AseDBDataset
from bam_omat24.atom_energies import (
    ATOM_ENERGIES,
    ATOMIC_NUMBER_TO_INDEX
)

def preprocess_graph(
    atoms: ase.Atoms,
    default_cutoff: float,
    targets: bool,
) -> jraph.GraphsTuple:
    """Preprocess ase.Atoms object into a jraph.GraphsTuple, with optional targets"""

    cutoff = default_cutoff
    src, dst, Sij = matscipy.neighbours.neighbour_list("ijS", atoms, cutoff)
    if len(src) < 20:
        cutoff = default_cutoff + 3.0
        src, dst, Sij = matscipy.neighbours.neighbour_list("ijS", atoms, cutoff)
        print (len(src), atoms.get_atomic_numbers())
        if len(src) == 0:
            return None

    forces = atoms.get_forces().astype(np.float32) if targets else None
    species = np.array(
                [ATOMIC_NUMBER_TO_INDEX[n] for n in atoms.get_atomic_numbers()],
            ).astype(np.int32)

    node_e0 = ATOM_ENERGIES[species]
    graph_e0 = node_e0.sum()

    energy = (
        np.array([atoms.get_potential_energy()-graph_e0]) if targets else None
    )
    cell = (
        np.array([atoms.get_cell()])
    )
    volume = (
        np.array([atoms.get_volume()])
    )
    stress = (
        np.array([atoms.get_stress()])
    )

    globals_cutoff = (
        np.array([cutoff])
    )

    return jraph.GraphsTuple(
        n_node=np.array([len(atoms)]).astype(np.int32),
        n_edge=np.array([len(src)]).astype(np.int32),
        nodes={
            "species": species,
            "positions": atoms.positions,
            "forces": forces,
        },
        edges={
            'Sij':Sij
        },
        senders=dst.astype(np.int32),
        receivers=src.astype(np.int32),
        globals={
            "energy": energy,
            "cell": cell,
            "volume": volume,
            "stress": stress,
            "cutoff": globals_cutoff
        },
    )


if __name__ == "__main__":
    from ase.io import write

    config_kwargs = {}
    dir_train = "/home/willow/Python/BAM-nequip-main/omat24_1M/train/"
    sample_list = [
        "aimd-from-PBE-1000-npt",
        "aimd-from-PBE-1000-nvt",
        "aimd-from-PBE-3000-npt",
        "aimd-from-PBE-3000-nvt",
        "rattled-1000",
        "rattled-1000-subsampled",
        "rattled-300",
        "rattled-300-subsampled",
        "rattled-500",
        "rattled-500-subsampled",
        "rattled-relax"
    ]
    trainset_paths = [
        dir_train + sample for sample in sample_list
    ]
    trainset = AseDBDataset(config=dict(src=trainset_paths,**config_kwargs))

    ndata = len(trainset)
    rng = np.random.RandomState(seed=42)
    perm = rng.permutation(ndata)
    nbatch = 55000
    #nbatch = 80000
    ipkl = 0
    cutoff = 6.0

    for ist in range (0, ndata, nbatch):
        ied = min (ist+nbatch, ndata)
        pkl_path = f'omat24_train_{ipkl}.pkl'
        print (f"preprocessing graphs...{ist}~{ied}")
        graphs = [
            preprocess_graph(
                trainset.get_atoms(perm[id]),
                cutoff,
                True)
            for id in tqdm(range(ist, ied))
        ]

        safe_graphs = [g for g in graphs if g is not None]
        print ("safe_graphs", len(safe_graphs))
        with open(pkl_path, "wb") as f:
            pickle.dump(safe_graphs, f)


        ipkl += 1
