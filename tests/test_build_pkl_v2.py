"""build_pkl_v2.preprocess_graph: every periodic edge has the minimum-image length the model sees.

The model computes r = x_sender - (x_receiver + Sij . cell) (bam_omat24/utils.py,
compute_node_edge_features); the edge length must equal matscipy's distance for the same pair.
"""
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
import matscipy.neighbours

from bam_omat24.build_pkl_v2 import preprocess_graph


def _with_calc(atoms):
    n = len(atoms)
    atoms.calc = SinglePointCalculator(atoms, energy=-1.0 * n, forces=np.zeros((n, 3)), stress=np.zeros(6))
    return atoms


def _crystals():
    rng = np.random.default_rng(0)
    si = bulk("Si", "diamond", a=5.43) * (2, 2, 2)           # 16 atoms, 5.43*sqrt(2)... cell < 2 cutoffs
    si.positions += rng.normal(scale=0.05, size=si.positions.shape)
    nacl = bulk("NaCl", "rocksalt", a=5.64)                  # 2-atom primitive cell: almost every edge is an image
    tri = bulk("Cu", "fcc", a=3.6, orthorhombic=False)       # 1-atom triclinic cell
    tri.positions += 0.3                                     # atom outside the reference position
    return [si, nacl, tri]


@pytest.mark.parametrize("k", range(3))
def test_periodic_edge_lengths(k):
    atoms = _with_calc(_crystals()[k])
    cutoff = 5.0
    g = preprocess_graph(atoms, cutoff, targets=True)
    pos = np.asarray(g.nodes["positions"], np.float64)
    cell = np.asarray(g.globals["cell"][0], np.float64)
    S = np.asarray(g.edges["Sij"], np.float64)
    r = pos[g.senders] - (pos[g.receivers] + S @ cell)        # the model's displacement
    src, dst, d = matscipy.neighbours.neighbour_list("ijd", atoms, float(g.globals["cutoff"][0]))
    assert np.array_equal(g.receivers, src) and np.array_equal(g.senders, dst)
    assert np.abs(np.linalg.norm(r, axis=1) - d).max() < 1e-10
    assert np.any(S != 0)                                    # the test really exercises image edges
    # every edge has its reverse with the opposite vector (j -> i through -S)
    key = {(int(a), int(b), *map(int, s)): v for a, b, s, v in zip(g.senders, g.receivers, g.edges["Sij"], r)}
    for (a, b, *s), v in key.items():
        w = key[(b, a, *[-x for x in s])]
        assert np.abs(v + w).max() < 1e-10
