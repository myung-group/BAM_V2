"""Round-12 options: fixed-form Born–Mayer repulsion stratum (pair_repulsion) and bond-length
Gaussian radial channels (radial_gaussians)."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from conftest import make_model, as_float64, pad_single
from test_equivariance import _check_e3, TOL
from bam_omat24.training_utils import compute_loss

RADII = [0.31, 0.76, 0.71, 0.66]                      # H, C, N, O covalent radii (A)
GAUSS = {"centers": np.linspace(0.9, 2.3, 8).tolist(), "width": 0.15}


def _energy(model, graph):
    return float(np.asarray(model(graph)["energy"])[0])


@pytest.mark.parametrize("kw", [
    dict(pair_repulsion=True, species_radii=RADII),
    dict(radial_gaussians=GAUSS),
    dict(pair_repulsion=True, species_radii=RADII, radial_gaussians=GAUSS, pair_term=True, residual="identity"),
])
def test_e3_round12_options(x64, graph, kw):
    model, g = as_float64(make_model(n_layers=3, **kw), graph)
    _check_e3(model, g, TOL)


def test_forces_fd_round12(x64, graph):
    model, graph = as_float64(make_model(n_layers=3, pair_repulsion=True, species_radii=RADII,
                                         radial_gaussians=GAUSS), graph)
    forces = np.asarray(model(graph)["forces"])
    pos = np.asarray(graph.nodes["positions"], dtype=np.float64)

    def energy(p):
        g = graph._replace(nodes={**graph.nodes, "positions": p.astype(np.float64)})
        return float(np.asarray(model(g)["energy"])[0])

    cands = [(i, k) for i in range(pos.shape[0]) for k in range(3) if abs(forces[i, k]) > 0.05]
    rng = np.random.RandomState(4)
    picks = [cands[j] for j in rng.choice(len(cands), size=2, replace=False)]
    h = 1e-4
    for i, k in picks:
        pp, pm = pos.copy(), pos.copy(); pp[i, k] += h; pm[i, k] -= h
        fd = -(energy(pp) - energy(pm)) / (2 * h)
        assert abs(fd - forces[i, k]) / max(abs(forces[i, k]), 1e-6) < 1e-3, (fd, forces[i, k])


def test_repulsion_stratum_is_positive_and_decays(graph):
    """The repulsion stratum is the first stratum: positive on every real atom and it shrinks
    when the molecule is uniformly expanded."""
    m = make_model(n_layers=2, pair_repulsion=True, species_radii=RADII, seed=3)
    out = m(graph)
    strata = np.asarray(out["strata"])                               # (S, n_atoms)
    assert strata.shape[0] == 3                                      # repulsion + 2 layers
    assert np.all(strata[0] > 0)
    g2 = graph._replace(nodes={**graph.nodes, "positions": np.asarray(graph.nodes["positions"]) * 1.1})
    strata2 = np.asarray(m(g2)["strata"])
    assert strata2[0].sum() < strata[0].sum()
    # initial magnitude is small (A = 1 eV, rho ~ 0.3 A): below ~0.1 eV per atom
    assert strata[0].sum() / strata.shape[1] < 0.1


def test_options_change_energy_default_unchanged(graph):
    base = make_model(n_layers=2, seed=9)
    assert _energy(base, graph) == _energy(make_model(n_layers=2, seed=9, radial_gaussians=None), graph)
    assert abs(_energy(base, graph) - _energy(make_model(n_layers=2, seed=9, radial_gaussians=GAUSS), graph)) > 1e-6
    assert abs(_energy(base, graph) - _energy(make_model(n_layers=2, seed=9, pair_repulsion=True, species_radii=RADII), graph)) > 1e-6


def test_round12_grads_finite_with_padding(graph):
    m = make_model(n_layers=2, pair_repulsion=True, species_radii=RADII, radial_gaussians=GAUSS)
    graphdef, params = nnx.split(m, nnx.Param)
    g = pad_single(graph)

    def loss(p):
        l, _ = compute_loss(graphdef, p, g, 1.0, 10.0, 0.0, 10.0, periodic=False)
        return l

    val, grads = jax.value_and_grad(loss)(params)
    assert np.isfinite(float(val))
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(grads))
    ga = grads["rep_loga"]; ga = ga.value if hasattr(ga, "value") else ga
    gr = grads["rep_rho_raw"]; gr = gr.value if hasattr(gr, "value") else gr
    assert bool(jnp.any(jnp.abs(jnp.asarray(ga)) > 0)) and bool(jnp.any(jnp.abs(jnp.asarray(gr)) > 0))
