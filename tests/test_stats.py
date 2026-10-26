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
