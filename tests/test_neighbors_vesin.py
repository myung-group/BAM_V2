"""BAM_NEIGHBORS=vesin gives the same neighbour pairs (i, j, S) as the ASE / matscipy backend (2026-11-09)."""
import numpy as np
import pytest

pytest.importorskip('vesin')
import ase  # noqa: E402
from bam_omat24.neighbors import neighbor_pairs  # noqa: E402


def _canon(i, j, S):
    S = np.zeros((len(i), 3), np.int64) if S is None else np.asarray(S, np.int64)
    k = np.lexsort((S[:, 2], S[:, 1], S[:, 0], j, i))
    return np.asarray(i)[k], np.asarray(j)[k], S[k]


def _cases():
    rng = np.random.default_rng(0)
    out = []
    for n, L, skew in [(8, 3.1, 0.0), (40, 9.0, 0.3), (110, 12.0, 0.15)]:   # small cells: many images
        cell = np.diag([L, L * 1.1, L * 0.9]) + skew * L * np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        out.append(ase.Atoms('Si%d' % n, scaled_positions=rng.random((n, 3)), cell=cell, pbc=True))
    out.append(ase.Atoms('C20', positions=rng.random((20, 3)) * 6.0))                     # molecule, no cell
    return out


@pytest.mark.parametrize('k', range(4))
def test_vesin_same_pairs(monkeypatch, k):
    a = _cases()[k]
    monkeypatch.setenv('BAM_NEIGHBORS', 'ase')
    ref = _canon(*neighbor_pairs(a, 5.0))
    monkeypatch.setenv('BAM_NEIGHBORS', 'vesin')
    i, j, S = neighbor_pairs(a, 5.0)
    assert (S is None) == (not a.pbc.any())
    got = _canon(i, j, S)
    assert len(ref[0]) > 0
    for x, y in zip(ref, got):
        assert np.array_equal(x, y)
