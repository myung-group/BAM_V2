"""3BPA (molecular, non-periodic) data pipeline for the ``bam_omat24`` RACE model.

* extended-XYZ -> ``jraph.GraphsTuple`` (no cell, no stress, ``edges=None``)
* optional per-species E0 linear regression (``fit_e0``)
* dataset statistics for model initialisation (``compute_dataset_stats``)
* on-disk pickle cache keyed by cutoff and the E0 used

Energy convention (same as ``bam_omat24.build_pkl``): the stored graph energy is
``E_total - sum_i E0[species_i]`` so the model is trained with ``l_train=True``
and *does not* add isolated atom energies itself.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import ase
import ase.io
import jraph
import matscipy.neighbours
import numpy as np
from tqdm import tqdm


# =============================================================================
# Graph construction
# =============================================================================

def atomic_numbers_to_indices(atomic_numbers: Sequence[int]) -> Dict[int, int]:
    """Map atomic number -> species index (sorted order)."""
    return {int(n): i for i, n in enumerate(sorted(int(z) for z in atomic_numbers))}


def preprocess_graph(
    atoms: ase.Atoms,
    atom_indices: Dict[int, int],
    atom_energies: Sequence[float],
    cutoff: float,
    targets: bool = True,
) -> jraph.GraphsTuple:
    """ase.Atoms -> GraphsTuple for a non-periodic molecule.

    ``atom_energies`` is indexed by species index and subtracted from the
    total energy.
    """
    src, dst = matscipy.neighbours.neighbour_list("ij", atoms, cutoff)
    species = np.array(
        [atom_indices[int(n)] for n in atoms.get_atomic_numbers()], dtype=np.int32
    )
    e0 = float(np.sum(np.asarray(atom_energies, dtype=np.float64)[species]))

    nodes = {
        "species": species,
        "positions": atoms.get_positions().astype(np.float32),
    }
    if targets:
        nodes["forces"] = atoms.get_forces().astype(np.float32)
        energy = np.array([atoms.get_potential_energy() - e0], dtype=np.float32)
    else:
        energy = np.array([0.0], dtype=np.float32)

    return jraph.GraphsTuple(
        n_node=np.array([len(atoms)], dtype=np.int32),
        n_edge=np.array([len(src)], dtype=np.int32),
        nodes=nodes,
        edges=None,
        senders=dst.astype(np.int32),
        receivers=src.astype(np.int32),
        globals={"energy": energy},
    )


# =============================================================================
# E0 regression and dataset statistics
# =============================================================================

def species_counts(frames: Sequence[ase.Atoms], atom_indices: Dict[int, int]) -> np.ndarray:
    """(n_frames, n_species) composition matrix."""
    n_species = len(atom_indices)
    counts = np.zeros((len(frames), n_species), dtype=np.float64)
    for g, atoms in enumerate(frames):
        for z in atoms.get_atomic_numbers():
            counts[g, atom_indices[int(z)]] += 1.0
    return counts


def fit_e0(
    frames: Sequence[ase.Atoms],
    atom_indices: Dict[int, int],
    reference_e0: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Per-species E0 by least squares  E_g ~ sum_s N_{g,s} E0_s.

    If ``reference_e0`` is given the regression is done on the residual
    ``E_g - sum_s N_{g,s} ref_s`` and the result is ``ref + delta`` (this is the
    minimum-norm solution when the composition matrix is rank deficient, e.g. a
    dataset with a single stoichiometry such as 3BPA).
    """
    counts = species_counts(frames, atom_indices)
    energies = np.array([a.get_potential_energy() for a in frames], dtype=np.float64)
    ref = (np.zeros(counts.shape[1]) if reference_e0 is None
           else np.asarray(reference_e0, dtype=np.float64))
    residual = energies - counts @ ref
    delta, *_ = np.linalg.lstsq(counts, residual, rcond=None)
    return ref + delta


def compute_dataset_stats(
    graphs: Sequence[jraph.GraphsTuple],
    n_species: int,
) -> Dict[str, float]:
    """Statistics of a list of (single-molecule) graphs whose ``globals['energy']``
    already has the reference E0 subtracted.

    Returns
    -------
    e0_residual : (n_species,) per-species least-squares residual E0 (eV) that
                  remains in the stored energies (0 if E0 regression was used).
    shift       : mean per-atom energy after removing ``e0_residual`` (eV/atom).
                  (This is 0 up to numerics when e0_residual is non-trivial; it is
                  reported so the model init can use either convention.)
    shift_raw   : mean per-atom energy of the stored targets (eV/atom)  -- the
                  value used for the model ``shift`` by default.
    scale       : RMS of force components (eV/A)  -- used for the model ``scale``.
    avg_n_neighbors : mean number of neighbours per atom.
    n_graphs, n_atoms, n_edges
    """
    n_graphs = len(graphs)
    counts = np.zeros((n_graphs, n_species), dtype=np.float64)
    energies = np.zeros(n_graphs, dtype=np.float64)
    n_atoms_g = np.zeros(n_graphs, dtype=np.float64)
    n_edges = 0.0
    force_sq_sum = 0.0
    force_count = 0
    for g, graph in enumerate(graphs):
        sp = np.asarray(graph.nodes["species"])
        counts[g] = np.bincount(sp, minlength=n_species)[:n_species]
        energies[g] = float(np.asarray(graph.globals["energy"]).reshape(-1)[0])
        n_atoms_g[g] = len(sp)
        n_edges += float(np.asarray(graph.n_edge).sum())
        f = np.asarray(graph.nodes["forces"], dtype=np.float64)
        force_sq_sum += float(np.sum(f ** 2))
        force_count += f.size

    e0_residual, *_ = np.linalg.lstsq(counts, energies, rcond=None)
    resid = energies - counts @ e0_residual
    n_atoms = float(n_atoms_g.sum())
    return {
        "e0_residual": e0_residual,
        "shift": float(np.sum(resid) / n_atoms),
        "shift_raw": float(np.sum(energies) / n_atoms),
        "scale": float(np.sqrt(force_sq_sum / max(force_count, 1))),
        "avg_n_neighbors": float(n_edges / n_atoms),
        "n_graphs": n_graphs,
        "n_atoms": n_atoms,
        "n_edges": n_edges,
    }


# =============================================================================
# Loading with cache
# =============================================================================

def _cache_path(xyz_path: Path, cutoff: float, atom_energies: Sequence[float], cache_dir: Optional[Path]) -> Path:
    tag = hashlib.md5(np.asarray(atom_energies, dtype=np.float64).tobytes()).hexdigest()[:8]
    cache_dir = xyz_path.parent if cache_dir is None else Path(cache_dir)
    return cache_dir / f"{xyz_path.stem}_omat24_cutoff_{cutoff}_e0_{tag}.pkl"


def load_frames(xyz_path: str | Path) -> List[ase.Atoms]:
    return ase.io.read(str(xyz_path), index=":", format="extxyz")


def load_3bpa_graphs(
    xyz_path: str | Path,
    atomic_numbers: Sequence[int],
    atom_energies: Sequence[float],
    cutoff: float,
    cache_dir: Optional[str | Path] = None,
    use_cache: bool = True,
    frames: Optional[List[ase.Atoms]] = None,
) -> List[jraph.GraphsTuple]:
    """Read an extxyz file into graphs (with pickle cache)."""
    xyz_path = Path(xyz_path)
    assert xyz_path.exists(), f"{xyz_path} does not exist"
    cache = _cache_path(xyz_path, cutoff, atom_energies, cache_dir)
    if use_cache and cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    atom_indices = atomic_numbers_to_indices(atomic_numbers)
    if frames is None:
        frames = load_frames(xyz_path)
    graphs = [
        preprocess_graph(atoms, atom_indices, atom_energies, cutoff, targets=True)
        for atoms in tqdm(frames, desc=f"graphs {xyz_path.name}", leave=False)
    ]
    if use_cache:
        with open(cache, "wb") as f:
            pickle.dump(graphs, f)
    return graphs


def split_train_valid(
    graphs: Sequence[jraph.GraphsTuple], valid_frac: float, seed: int = 42
) -> Tuple[List[jraph.GraphsTuple], List[jraph.GraphsTuple]]:
    """Same deterministic split as ``bam_3bpa.data.Dataset`` (RandomState(seed))."""
    rng = np.random.RandomState(seed=seed)
    perm = rng.permutation(len(graphs))
    n_train = int(len(perm) * (1 - valid_frac))
    train_idx, valid_idx = perm[:n_train], perm[n_train:]
    return [graphs[i] for i in train_idx], [graphs[i] for i in valid_idx]


class GraphListDataset:
    """Minimal in-memory dataset compatible with ``BucketedDataLoader``."""

    def __init__(self, graphs: Sequence[jraph.GraphsTuple]):
        self.graphs = list(graphs)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> jraph.GraphsTuple:
        return self.graphs[idx]
