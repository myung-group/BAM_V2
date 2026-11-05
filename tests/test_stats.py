"""compute_dataset_stats / fit_e0 recover known E0, force RMS and neighbour count."""
import ase
import jraph
import numpy as np

from bam_omat24.data_3bpa import compute_dataset_stats, fit_e0, atomic_numbers_to_indices


def _synthetic_graphs(rng, n_graphs=60, n_species=4, e0=(-1.0, -10.0, -20.0, -30.0),
                      force_std=0.7, shift=0.05):
    graphs = []
    for _ in range(n_graphs):
        n = rng.randint(5, 16)
        sp = rng.randint(0, n_species, size=n).astype(np.int32)
        e = float(np.sum(np.asarray(e0)[sp]) + shift * n)
        f = rng.normal(scale=force_std, size=(n, 3)).astype(np.float32)
        n_edge = rng.randint(3 * n, 8 * n)
        graphs.append(jraph.GraphsTuple(
            n_node=np.array([n], np.int32), n_edge=np.array([n_edge], np.int32),
            nodes={"species": sp, "positions": np.zeros((n, 3), np.float32), "forces": f},
            edges=None, senders=np.zeros(n_edge, np.int32), receivers=np.zeros(n_edge, np.int32),
            globals={"energy": np.array([e], np.float32)},
        ))
    return graphs


def test_compute_dataset_stats_recovers_known_values():
    rng = np.random.RandomState(0)
    e0 = (-1.0, -10.0, -20.0, -30.0)
    shift = 0.05
    graphs = _synthetic_graphs(rng, e0=e0, shift=shift, force_std=0.7)
    stats = compute_dataset_stats(graphs, n_species=4)
    # per-species E0 regression absorbs the per-atom shift: E0_s + shift
    assert np.allclose(stats["e0_residual"], np.asarray(e0) + shift, atol=1e-3)
    assert abs(stats["shift"]) < 1e-3
    # force RMS within 3 % of the generating std (finite sample)
    assert abs(stats["scale"] - 0.7) / 0.7 < 0.03
    n_atoms = sum(int(g.n_node[0]) for g in graphs)
    n_edges = sum(int(g.n_edge[0]) for g in graphs)
    assert np.isclose(stats["avg_n_neighbors"], n_edges / n_atoms)
    assert stats["n_graphs"] == len(graphs)


def test_fit_e0_from_frames():
    rng = np.random.RandomState(1)
    zs = [1, 6, 7, 8]
    idx = atomic_numbers_to_indices(zs)
    e0 = np.array([-13.5, -1029.5, -1485.0, -2042.0])
    frames = []
    for _ in range(40):
        numbers = rng.choice(zs, size=rng.randint(4, 12))
        atoms = ase.Atoms(numbers=numbers, positions=rng.normal(size=(len(numbers), 3)))
        e = float(sum(e0[idx[int(z)]] for z in numbers)) + rng.normal(scale=1e-4)
        atoms.calc = ase.calculators.singlepoint.SinglePointCalculator(atoms, energy=e)
        frames.append(atoms)
    fitted = fit_e0(frames, idx)
    assert np.allclose(fitted, e0, atol=1e-3)
    # regression on residual with a reference returns reference + correction
    fitted2 = fit_e0(frames, idx, reference_e0=e0 + 1.0)
    assert np.allclose(fitted2, e0, atol=1e-3)


def _single_stoichiometry_frames(rng, e0, offset_per_atom=0.3, n=30):
    """3BPA-like: every frame has the same composition (H12 C12 N2 O1) -> rank-1 composition matrix."""
    import ase.calculators.singlepoint
    zs = [1, 6, 7, 8]
    idx = atomic_numbers_to_indices(zs)
    numbers = np.array([1] * 12 + [6] * 12 + [7] * 2 + [8])
    frames = []
    for _ in range(n):
        atoms = ase.Atoms(numbers=numbers, positions=rng.normal(size=(len(numbers), 3)))
        e = float(sum(e0[idx[int(z)]] for z in numbers)) + offset_per_atom * len(numbers) + rng.normal(scale=1e-3)
        atoms.calc = ase.calculators.singlepoint.SinglePointCalculator(atoms, energy=e)
        frames.append(atoms)
    return frames, idx


def test_fit_e0_single_stoichiometry_keeps_reference():
    import warnings
    rng = np.random.RandomState(2)
    e0 = np.array([-13.5, -1029.5, -1485.0, -2042.0])
    frames, idx = _single_stoichiometry_frames(rng, e0)
    info = {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fitted = fit_e0(frames, idx, reference_e0=e0, info=info)
        assert any("not identifiable" in str(x.message) for x in w)
    assert info["rank"] == 1 and info["n_species"] == 4
    assert np.array_equal(fitted, e0)                                   # reference kept exactly
    assert abs(info["per_atom_offset"] - 0.3) < 1e-3                    # identifiable offset reported
    # the old split spreads the offset in proportion to the composition (12:12:2:1)
    fitted_mn = fit_e0(frames, idx, reference_e0=e0, on_rank_deficient="minnorm")
    d = fitted_mn - e0
    assert np.allclose(d / d[0], np.array([1, 1, 2 / 12, 1 / 12]), atol=1e-6)
    assert abs(np.dot(d, [12, 12, 2, 1]) - 0.3 * 27) < 1e-2            # same per-frame total
    # no reference -> min-norm with warning (nothing better is possible)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        f0 = fit_e0(frames, idx)
        assert any("not identifiable" in str(x.message) for x in w)
    assert np.allclose(f0 / f0[0], np.array([1, 1, 2 / 12, 1 / 12]), atol=1e-6)


def test_dataset_stats_single_stoichiometry_no_fake_e0():
    rng = np.random.RandomState(3)
    graphs = []
    sp = np.array([0] * 12 + [1] * 12 + [2] * 2 + [3], np.int32)
    for _ in range(20):
        e = -131.5 + rng.normal(scale=0.1)
        graphs.append(jraph.GraphsTuple(
            n_node=np.array([27], np.int32), n_edge=np.array([200], np.int32),
            nodes={"species": sp, "positions": np.zeros((27, 3), np.float32),
                   "forces": rng.normal(size=(27, 3)).astype(np.float32)},
            edges=None, senders=np.zeros(200, np.int32), receivers=np.zeros(200, np.int32),
            globals={"energy": np.array([e], np.float32)}))
    stats = compute_dataset_stats(graphs, n_species=4)
    assert stats["e0_rank"] == 1 and not stats["e0_identifiable"]
    assert np.all(stats["e0_residual"] == 0.0)
    assert abs(stats["shift"] - stats["shift_raw"]) < 1e-12 and abs(stats["shift_raw"] + 131.5 / 27) < 0.01
