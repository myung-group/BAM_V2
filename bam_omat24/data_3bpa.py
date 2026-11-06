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

from bam_omat24.neighbors import neighbor_pairs


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
    charge_key: Optional[str] = None,
) -> jraph.GraphsTuple:
    """ase.Atoms -> GraphsTuple for a non-periodic molecule.

    ``atom_energies`` is indexed by species index and subtracted from the
    total energy.
    """
    periodic = bool(np.any(atoms.pbc))
    # periodic: i (receiver), j (sender), S (image of j).  D_ij = x_j - x_i + S.cell (ASE / matscipy convention); the
    # model computes r = x_sender - (x_receiver + Sij.cell), so Sij = -S gives r = D_ij (neighbors.neighbor_pairs).
    src, dst, S = neighbor_pairs(atoms, cutoff)
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

    edges = None
    glob = {"energy": energy}
    if charge_key:     # total charge of the structure (RACE_LR); only when asked, so other graphs keep their keys
        glob["charge"] = np.array([float(atoms.info.get(charge_key, 0.0))], dtype=np.float32)
    if periodic:
        edges = {"Sij": (-S).astype(np.float32)}
        glob.update(cell=np.array(atoms.get_cell().array, dtype=np.float32)[None],   # .array: ASE Cell.__array__ breaks under NumPy 2
                    cutoff=np.array([cutoff], dtype=np.float32),
                    volume=np.array([atoms.get_volume()], dtype=np.float32))
        # stress target (eV/A^3, ASE sign: positive = tensile, Voigt xx yy zz yz xz xy); zeros if the frame has none
        # (only read by the loss with stress_loss=true)
        stress = np.zeros(6, dtype=np.float32)
        if targets:
            try:
                stress = np.asarray(atoms.get_stress(voigt=True), dtype=np.float32).reshape(6)
            except Exception:
                pass
        glob.update(stress=stress[None])
    return jraph.GraphsTuple(
        n_node=np.array([len(atoms)], dtype=np.int32),
        n_edge=np.array([len(src)], dtype=np.int32),
        nodes=nodes,
        edges=edges,
        senders=dst.astype(np.int32),
        receivers=src.astype(np.int32),
        globals=glob,
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


def e0_rank(counts: np.ndarray) -> int:
    """Rank of the (n_frames, n_species) composition matrix.  Per-species E0 are identifiable
    only if the rank equals the number of species; a dataset with a single stoichiometry (3BPA)
    has rank 1: the data determine the per-frame offset sum_s N_s E0_s and nothing else."""
    return int(np.linalg.matrix_rank(counts))


def fit_e0(
    frames: Sequence[ase.Atoms],
    atom_indices: Dict[int, int],
    reference_e0: Optional[Sequence[float]] = None,
    on_rank_deficient: str = "reference",
    info: Optional[Dict] = None,
) -> np.ndarray:
    """Per-species E0 by least squares  E_g ~ sum_s N_{g,s} E0_s.

    If ``reference_e0`` is given the regression is done on the residual
    ``E_g - sum_s N_{g,s} ref_s`` and the result is ``ref + delta``.

    When the composition matrix is rank deficient (fewer distinct stoichiometries than species,
    e.g. 3BPA with one molecule) the per-species values are **not identifiable**: any split of the
    per-frame offset over the species gives identical training targets, and the minimum-norm
    lstsq split (proportional to the composition) is physically meaningless and not comparable
    to isolated-atom / foundation E0.  ``on_rank_deficient`` controls what is returned then:

    * ``"reference"`` (default): keep ``reference_e0`` exactly (the identifiable per-frame offset
      is left in the targets and belongs to the model's per-atom ``shift``); without a reference
      fall back to ``"minnorm"`` with a warning.
    * ``"minnorm"``: the old behaviour (minimum-norm correction).

    ``info`` (optional dict) receives rank, n_species, the mean per-atom offset of the residual and
    the action taken.
    """
    import warnings
    counts = species_counts(frames, atom_indices)
    energies = np.array([a.get_potential_energy() for a in frames], dtype=np.float64)
    n_species = counts.shape[1]
    ref = (np.zeros(n_species) if reference_e0 is None
           else np.asarray(reference_e0, dtype=np.float64))
    residual = energies - counts @ ref
    rank = e0_rank(counts)
    per_atom_offset = float(np.sum(residual) / max(np.sum(counts), 1.0))
    if on_rank_deficient not in ("reference", "minnorm"):
        raise ValueError(f"on_rank_deficient must be 'reference' or 'minnorm', got {on_rank_deficient!r}")
    action = "lstsq"
    if rank < n_species:
        if reference_e0 is not None and on_rank_deficient == "reference":
            delta = np.zeros(n_species)
            action = "reference kept (rank deficient)"
        else:
            delta, *_ = np.linalg.lstsq(counts, residual, rcond=None)
            action = "minimum-norm split (rank deficient, arbitrary per species)"
        warnings.warn(
            f"fit_e0: composition matrix rank {rank} < n_species {n_species}: per-species E0 are not "
            f"identifiable (only the per-frame offset, {per_atom_offset:.4f} eV/atom, is); {action}.")
    else:
        delta, *_ = np.linalg.lstsq(counts, residual, rcond=None)
    if info is not None:
        info.update(rank=rank, n_species=n_species, per_atom_offset=per_atom_offset, action=action,
                    delta=np.asarray(delta))
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

    rank = e0_rank(counts)
    if rank < n_species:
        # single (or too few) stoichiometries: a per-species residual E0 is not identifiable, do not
        # invent one; the identifiable offset is the per-atom shift (shift == shift_raw)
        e0_residual = np.zeros(n_species)
    else:
        e0_residual, *_ = np.linalg.lstsq(counts, energies, rcond=None)
    resid = energies - counts @ e0_residual
    n_atoms = float(n_atoms_g.sum())
    return {
        "e0_residual": e0_residual,
        "e0_rank": rank,
        "e0_identifiable": bool(rank == n_species),
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

def _cache_path(xyz_path: Path, cutoff: float, atom_energies: Sequence[float], cache_dir: Optional[Path],
                charge_key: Optional[str] = None) -> Path:
    tag = hashlib.md5(np.asarray(atom_energies, dtype=np.float64).tobytes()).hexdigest()[:8]
    cache_dir = xyz_path.parent if cache_dir is None else Path(cache_dir)
    q = f"_q-{charge_key}" if charge_key else ""
    return cache_dir / f"{xyz_path.stem}_omat24_cutoff_{cutoff}_e0_{tag}{q}.pkl"


def load_frames(xyz_path: str | Path) -> List[ase.Atoms]:
    return ase.io.read(str(xyz_path), index=":", format="extxyz")


def _graphs_chunk(args):
    """Worker of build_graphs: one chunk of frames -> graphs (module level: picklable for process pools)."""
    frames, atomic_numbers, atom_energies, cutoff, charge_key = args
    atom_indices = atomic_numbers_to_indices(atomic_numbers)
    return [preprocess_graph(a, atom_indices, atom_energies, cutoff, targets=True, charge_key=charge_key)
            for a in frames]


def build_graphs(frames, atomic_numbers, atom_energies, cutoff, n_workers: int = 1, chunk: int = 256,
                 desc: str = "graphs", charge_key: Optional[str] = None) -> List[jraph.GraphsTuple]:
    """Graphs of a list of frames; n_workers > 1 runs the neighbour searches in a process pool (forkserver: no fork
    of a process that already initialised JAX / CUDA).  Order is preserved."""
    if n_workers <= 1 or len(frames) <= chunk:
        return _graphs_chunk((frames, atomic_numbers, atom_energies, cutoff, charge_key))
    import concurrent.futures as cf
    import multiprocessing as mp
    jobs = [(frames[k:k + chunk], list(atomic_numbers), list(atom_energies), cutoff, charge_key)
            for k in range(0, len(frames), chunk)]
    out: List[jraph.GraphsTuple] = []
    with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("forkserver")) as ex:
        for part in tqdm(ex.map(_graphs_chunk, jobs), total=len(jobs), desc=f"{desc} ({n_workers} workers)",
                         leave=False):
            out.extend(part)
    return out


def graph_sizes(graphs) -> Tuple[np.ndarray, np.ndarray]:
    """(n_atoms, n_edges) per graph."""
    n = np.array([int(np.asarray(g.n_node).reshape(-1)[0]) for g in graphs], dtype=np.int64)
    e = np.array([int(np.asarray(g.n_edge).reshape(-1)[0]) for g in graphs], dtype=np.int64)
    return n, e


def write_sizes_tsv(path, n_atoms, n_edges, cutoff: float, source: str = "") -> None:
    """Human-readable size table (one line per structure): index, n_atoms, n_edges."""
    with open(path, "w") as fh:
        fh.write(f"# source {source}\n# cutoff {cutoff}\nindex\tn_atoms\tn_edges\n")
        for k, (a, b) in enumerate(zip(n_atoms, n_edges)):
            fh.write(f"{k}\t{int(a)}\t{int(b)}\n")


def read_sizes_tsv(path) -> Tuple[np.ndarray, np.ndarray]:
    d = np.loadtxt(path, comments="#", skiprows=3, dtype=np.int64, ndmin=2)
    return d[:, 1], d[:, 2]


def load_3bpa_graphs(
    xyz_path: str | Path,
    atomic_numbers: Sequence[int],
    atom_energies: Sequence[float],
    cutoff: float,
    cache_dir: Optional[str | Path] = None,
    use_cache: bool = True,
    frames: Optional[List[ase.Atoms]] = None,
    n_workers: int = 1,
    charge_key: Optional[str] = None,
) -> List[jraph.GraphsTuple]:
    """Read an extxyz file into graphs.  use_cache: pickle cache next to the file (read if present, else written);
    use_cache=False recomputes the neighbour lists (n_workers processes) and writes nothing."""
    xyz_path = Path(xyz_path)
    assert xyz_path.exists(), f"{xyz_path} does not exist"
    cache = _cache_path(xyz_path, cutoff, atom_energies, cache_dir, charge_key)
    if use_cache and cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)

    if frames is None:
        frames = load_frames(xyz_path)
    graphs = build_graphs(frames, atomic_numbers, atom_energies, cutoff, n_workers=n_workers,
                          desc=f"graphs {xyz_path.name}", charge_key=charge_key)
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
