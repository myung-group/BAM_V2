"""Periodic support of the 3BPA pipeline (review §12.4, liquid water): graph building with image shifts,
edge vectors = minimum-image neighbour vectors, invariance under lattice translations / rotations of
positions + cell, forces = -dE/dx (finite differences), finite stress path."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
import ase
from ase.calculators.singlepoint import SinglePointCalculator
import matscipy.neighbours

from conftest import make_model, as_float64, random_rotation
from bam_omat24.data_3bpa import preprocess_graph, atomic_numbers_to_indices
from bam_omat24.utils import compute_node_edge_features
from test_round11_shells import SHELLS

ZS = [1, 6, 7, 8]
V9 = dict(update="linear", readout_mode="gated", message_norm="avg", avg_n_neighbors=16.7, x_mode="basis",
          radial_prior={"type": "gaussian", "sigma": 0.224}, radial_shells=True, species_shell_radii=SHELLS)


def _water_box(seed=0, n=8, L=6.5):
    """n water molecules at random positions/orientations in a cubic box of side L (A)."""
    rng = np.random.RandomState(seed)
    pos, num = [], []
    for _ in range(n):
        o = rng.uniform(0, L, 3)
        R = random_rotation(rng.randint(1000)).astype(float)
        h1 = o + R @ np.array([0.96, 0.0, 0.0]); h2 = o + R @ np.array([-0.24, 0.93, 0.0])
        pos += [o, h1, h2]; num += [8, 1, 1]
    a = ase.Atoms(numbers=num, positions=np.array(pos), cell=np.eye(3) * L, pbc=True)
    a.wrap()
    a.calc = SinglePointCalculator(a, energy=-10.0, forces=np.zeros((len(a), 3)))
    return a


def _graph(atoms, cutoff=5.0):
    return preprocess_graph(atoms, atomic_numbers_to_indices(ZS), [0.0] * 4, cutoff)


def _model(x64=True, **kw):
    return make_model(n_layers=2, periodic=True, **{**V9, **kw})


def _e(model, g):
    return float(np.asarray(model(g)["energy"])[0])


def _f64(g):
    return jax.tree.map(lambda x: np.asarray(x, np.float64) if np.issubdtype(np.asarray(x).dtype, np.floating) else x, g)


def test_periodic_graph_edges_are_minimum_image_vectors(x64):
    a = _water_box()
    g = _graph(a)
    assert g.edges is not None and "Sij" in g.edges and g.globals["cell"].shape == (1, 3, 3)
    import ase.neighborlist
    src, dst, D = ase.neighborlist.neighbor_list("ijD", a, 5.0)      # same routine (and order) as the builder
    # model convention: r = x_sender - (x_receiver + Sij . cell)
    pos = np.asarray(g.nodes["positions"], np.float64); cell = np.asarray(g.globals["cell"][0], np.float64)
    r = pos[g.senders] - (pos[g.receivers] + np.asarray(g.edges["Sij"], np.float64) @ cell)
    assert np.allclose(r, D, atol=1e-5)
    assert np.all(np.linalg.norm(r, axis=1) < 5.0 + 1e-6)
    # the box is smaller than 2 cutoffs: images are needed
    assert np.any(np.abs(np.asarray(g.edges["Sij"])) > 0)


def test_periodic_lattice_translation_and_rotation_invariance(x64):
    model, _ = as_float64(_model(), _graph(_water_box()) if False else _graph(_water_box()))
    a = _water_box(seed=1)
    g = _f64(_graph(a))
    e0 = _e(model, g)
    # translate one atom by a lattice vector (re-built graph): identical structure
    b = a.copy(); b.positions[0] += b.cell[1]
    b.calc = SinglePointCalculator(b, energy=-10.0, forces=np.zeros((len(b), 3)))
    tol = 1e-6 * max(1.0, abs(e0))      # graphs store positions in float32 (as in training)
    assert abs(_e(model, _f64(_graph(b))) - e0) < tol
    # rigid translation of all atoms + re-wrap
    c = a.copy(); c.positions += np.array([0.31, -0.77, 1.9]); c.wrap()
    c.calc = SinglePointCalculator(c, energy=-10.0, forces=np.zeros((len(c), 3)))
    assert abs(_e(model, _f64(_graph(c))) - e0) < tol
    # rotation of positions and cell together
    R = random_rotation(5).astype(float)
    d = ase.Atoms(numbers=a.numbers, positions=a.positions @ R.T, cell=np.asarray(a.cell) @ R.T, pbc=True)
    d.calc = SinglePointCalculator(d, energy=-10.0, forces=np.zeros((len(d), 3)))
    gd = _f64(_graph(d))
    assert abs(_e(model, gd) - e0) < tol
    f = np.asarray(model(g)["forces"]); fd = np.asarray(model(gd)["forces"])
    assert np.allclose(f @ R.T, fd, atol=1e-5)


def test_periodic_forces_are_minus_gradient(x64):
    model, _ = as_float64(_model(), _graph(_water_box()))
    a = _water_box(seed=2)
    g = _f64(_graph(a))
    forces = np.asarray(model(g)["forces"])
    i, k = divmod(int(np.argmax(np.abs(forces))), 3)
    h = 1e-4
    def e_at(dx):
        # move atom i at fixed neighbour list (no re-build: valid for |dx| << skin)
        gg = g._replace(nodes={**g.nodes, "positions": np.asarray(g.nodes["positions"]) + dx})
        return _e(model, gg)
    dp = np.zeros_like(forces); dp[i, k] = h
    fd = -(e_at(dp) - e_at(-dp)) / (2 * h)
    assert abs(fd - forces[i, k]) / abs(forces[i, k]) < 1e-4, (fd, forces[i, k])
    out = model(g)
    assert np.all(np.isfinite(np.asarray(out["energy"]))) and np.all(np.isfinite(forces))


def test_molecule_without_cell_builds_graph():
    """rMD17-style frames carry no cell: the open-boundary neighbour search must still work."""
    a = ase.Atoms(numbers=[6, 1, 1, 1, 1], positions=[[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63],
                                                     [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]])
    a.calc = SinglePointCalculator(a, energy=-1.0, forces=np.zeros((5, 3)))
    g = _graph(a)
    assert int(g.n_edge[0]) == 20 and g.edges is None and "cell" not in g.globals
