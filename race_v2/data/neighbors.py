"""Neighbour search of the training graphs, without JAX (importable by stand-alone tools: ``import neighbors`` from
this directory, or ``race_v2.data.neighbors``).

    src, dst, S = neighbor_pairs(atoms, cutoff)

src = receiver i, dst = sender j, S = integer image shift of j (periodic) or None.  Same convention as the graphs of
``data.xyz_graphs.preprocess_graph``: the model edge vector is x_j - (x_i + Sij . cell) with Sij = -S.

Backend: environment variable BAM_NEIGHBORS = "ase" (default: ASE for periodic, matscipy for molecules) or "vesin"
(C++ cell list, ``pip install vesin``; 2026-11-09 local test: the same pair set as ASE on 200 periodic Au2/MgO
structures, 57x faster for periodic 110-atom cells, 2x for molecules).  vesin pairs are sorted by (i, j, S), so the
edge order differs from the ASE backend (same graph; sums in another order).  Use one backend for the size pass and
training of a run.
"""
from __future__ import annotations

import os

import numpy as np


def _vesin_pairs(atoms, cutoff: float, periodic: bool):
    from vesin import NeighborList
    nl = NeighborList(cutoff=float(cutoff), full_list=True)
    box = atoms.cell.array if periodic else np.zeros((3, 3))
    for attempt in range(6):
        try:
            i, j, S = nl.compute(points=np.asarray(atoms.positions, np.float64), box=np.asarray(box, np.float64),
                                 periodic=periodic, quantities="ijS")
            break
        except RuntimeError as e:          # vesin starts threads per call: EAGAIN when many workers do so at once
            if 'temporarily unavailable' not in str(e):
                raise
            if attempt == 5:
                return None                # caller falls back to ASE (same pairs, tests/test_neighbors_vesin.py)
            import time
            time.sleep(0.05 * 2 ** attempt)
    S = np.asarray(S, np.int64).reshape(-1, 3)
    k = np.lexsort((S[:, 2], S[:, 1], S[:, 0], j, i))
    i, j, S = np.asarray(i, np.int64)[k], np.asarray(j, np.int64)[k], S[k]
    return i, j, (S if periodic else None)


def neighbor_pairs(atoms, cutoff: float):
    periodic = bool(np.any(atoms.pbc))
    backend = os.environ.get("BAM_NEIGHBORS", "ase").lower()
    if backend == "vesin":
        if periodic and not bool(np.all(atoms.pbc)):
            raise ValueError("vesin backend: mixed pbc is not supported (use BAM_NEIGHBORS=ase)")
        res = _vesin_pairs(atoms, cutoff, periodic)
        if res is not None:
            return res
        backend = "ase"                  # vesin could not start (thread limit): same pairs from ASE, other order
    if backend != "ase":
        raise ValueError(f"BAM_NEIGHBORS={backend!r}: expected 'ase' or 'vesin'")
    if periodic:
        # ASE neighbour list (matscipy's periodic path fails under NumPy 2)
        import ase.neighborlist
        src, dst, S = ase.neighborlist.neighbor_list("ijS", atoms, cutoff)
        return src, dst, S
    import matscipy.neighbours
    if atoms.cell.rank < 3:
        # molecule without a cell (e.g. converted rMD17): any box works for the open-boundary search
        boxed = atoms.copy()
        span = np.ptp(boxed.positions, axis=0) if len(boxed) else np.zeros(3)
        boxed.set_cell(np.diag(span + 2.0 * cutoff + 10.0)); boxed.pbc = False
        src, dst = matscipy.neighbours.neighbour_list("ij", boxed, cutoff)
    else:
        src, dst = matscipy.neighbours.neighbour_list("ij", atoms, cutoff)
    return src, dst, None


def count_edges(atoms, cutoff: float) -> int:
    return int(len(neighbor_pairs(atoms, cutoff)[0]))
