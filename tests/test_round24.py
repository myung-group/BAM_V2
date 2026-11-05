"""Round 24 (review §12.3, GRACE-inspired options): neighbour-species radial mixing
(radial_species_mix "neighbor" | "pair": per-layer linear mix of the radial channels by the sender
species or the species pair, identity at init) and in-layer symmetric products of degree 4
(in_layer_order=4 on a two-layer model)."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from conftest import make_model, as_float64
from test_equivariance import _check_e3, TOL

V9 = dict(update="linear", readout_mode="gated", message_norm="avg", avg_n_neighbors=16.7, x_mode="basis",
          radial_prior={"type": "gaussian", "sigma": 0.224})
MIX_N = dict(**V9, radial_species_mix="neighbor")
MIX_P = dict(**V9, radial_species_mix="pair")
NU4 = dict(**V9, in_layer_order=4)


def _e(model, g):
    return float(np.asarray(model(g)["energy"])[0])


def _fd_forces(model, graph):
    forces = np.asarray(model(graph)["forces"])
    pos = np.asarray(graph.nodes["positions"], dtype=np.float64)

    def energy(p):
        g = graph._replace(nodes={**graph.nodes, "positions": p.astype(np.float64)})
        return _e(model, g)

    order = np.argsort(-np.abs(forces).ravel())[:2]
    picks = [(int(o) // 3, int(o) % 3) for o in order]
    assert abs(forces[picks[1]]) > 1e-4, forces[picks[1]]
    h = 1e-4
    for i, k in picks:
        pp, pm = pos.copy(), pos.copy(); pp[i, k] += h; pm[i, k] -= h
        fd = -(energy(pp) - energy(pm)) / (2 * h)
        assert abs(fd - forces[i, k]) / max(abs(forces[i, k]), 1e-6) < 1e-3, (fd, forces[i, k])


@pytest.mark.parametrize("kw", [MIX_N, MIX_P, NU4])
def test_e3_round24(x64, graph, kw):
    model, g = as_float64(make_model(n_layers=2, **kw), graph)
    _check_e3(model, g, TOL)


@pytest.mark.parametrize("kw", [MIX_N, NU4])
def test_forces_fd_round24(x64, graph, kw):
    model, g = as_float64(make_model(n_layers=2, **kw), graph)
    _fd_forces(model, g)


def test_mix_identity_init_equals_baseline(x64, graph):
    """radial_mix is the identity at init and consumes no random numbers: same energy as without it."""
    base, g = as_float64(make_model(n_layers=2, **V9), graph)
    for kw in (MIX_N, MIX_P):
        mixed, _ = as_float64(make_model(n_layers=2, **kw), graph)
        assert abs(_e(mixed, g) - _e(base, g)) < 1e-9 * max(1.0, abs(_e(base, g)))


def test_mix_species_dependence(x64, graph):
    """Scaling the mix of one sender species changes the energy; the mix of the pair variant is
    indexed (receiver, sender)."""
    model, g = as_float64(make_model(n_layers=2, **MIX_N), graph)
    e0 = _e(model, g)
    n_r = model.layers[0].radial_mix[...].shape[-1]
    sp = np.asarray(g.nodes["species"])
    present = sorted(set(int(s) for s in sp))
    assert len(present) >= 2
    w = np.asarray(model.layers[0].radial_mix[...])
    w2 = w.copy(); w2[present[0]] = 1.5 * np.eye(n_r)
    model.layers[0].radial_mix[...] = jnp.asarray(w2)
    assert abs(_e(model, g) - e0) > 1e-6
    # pair variant: shape (n_species, n_species, n_r, n_r), identity init
    mp, _ = as_float64(make_model(n_layers=2, **MIX_P), graph)
    wp = np.asarray(mp.layers[0].radial_mix[...])
    assert wp.shape == (model.n_species, model.n_species, n_r, n_r)
    assert np.allclose(wp, np.eye(n_r)[None, None])


def test_order4_degrees_and_finite(x64, graph):
    model, g = as_float64(make_model(n_layers=2, **NU4), graph)
    assert [l.in_layer_order for l in model.layers] == [4, 4]
    assert model.layers[0].sym_degrees == (2, 3, 4)
    out = model(g)
    assert np.all(np.isfinite(np.asarray(out["energy"]))) and np.all(np.isfinite(np.asarray(out["forces"])))
    with pytest.raises(ValueError):
        make_model(n_layers=2, **dict(V9, in_layer_order=5))


# feature-explosion guards (review §12.3): per-node RMS normalisation of x_i or of the features
XN = dict(**V9, x_norm="rms")
FN = dict(**V9, feature_norm="rms")


@pytest.mark.parametrize("kw", [XN, FN])
def test_e3_norm_guards(x64, graph, kw):
    model, g = as_float64(make_model(n_layers=3, **kw), graph)
    _check_e3(model, g, TOL)


@pytest.mark.parametrize("kw", [XN, FN])
def test_forces_fd_norm_guards(x64, graph, kw):
    model, g = as_float64(make_model(n_layers=3, **kw), graph)
    _fd_forces(model, g)


def test_norm_guards_change_energy_and_validate(x64, graph):
    base, g = as_float64(make_model(n_layers=3, **V9), graph)
    for kw in (XN, FN):
        m, _ = as_float64(make_model(n_layers=3, **kw), graph)
        assert np.isfinite(_e(m, g)) and abs(_e(m, g) - _e(base, g)) > 1e-8
    with pytest.raises(ValueError):
        make_model(n_layers=2, **dict(V9, x_mode="species"), x_norm="rms")
    with pytest.raises(ValueError):
        make_model(n_layers=2, **V9, feature_norm="layer")
