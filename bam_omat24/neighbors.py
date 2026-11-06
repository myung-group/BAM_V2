"""Neighbour search of the training graphs, without JAX (importable by stand-alone tools: ``import neighbors`` from
this directory, or ``bam_omat24.neighbors``).

    src, dst, S = neighbor_pairs(atoms, cutoff)

src = receiver i, dst = sender j, S = integer image shift of j (periodic) or None.  Same convention as the graphs of
``data_3bpa.preprocess_graph``: the model edge vector is x_j - (x_i + Sij . cell) with Sij = -S.
"""
from __future__ import annotations

import numpy as np


def neighbor_pairs(atoms, cutoff: float):
    periodic = bool(np.any(atoms.pbc))
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
