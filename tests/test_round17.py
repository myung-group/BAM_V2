"""Round-17 option (review §11.11): in-layer symmetric contraction of the aggregated message
(in_layer_order=3: degrees 2 and 3, species weights, cuEquivariance descriptor; in_layer_order=2 with
sym_contraction=True: degree 2 on the same path), optionally on the leading layers only
(in_layer_order_layers)."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from conftest import make_model, as_float64
from test_equivariance import _check_e3, TOL
from race_v2.models.blocks import compute_node_edge_features

V8 = dict(update="linear", readout_mode="gated", message_norm="avg", avg_n_neighbors=16.7, x_mode="basis")
NU3 = dict(**V8, in_layer_order=3)
NU3_L2 = dict(**V8, in_layer_order=3, in_layer_order_layers=2)
NU2 = dict(**V8, in_layer_order=2, sym_contraction=True)


def _layer0_message(model, g):
    feats, rb, sh = compute_node_edge_features(
        g.nodes["positions"], None, None, g, model.n_species, model.cutoff,
        model.radial_basis_size, model.radial_polynomial_p, model.lmax, periodic=False)
    h0 = model.species_embedding(feats)
    layer = model.layers[0]
    return layer, layer.aggregate(h0, sh, rb, g.senders, g.receivers)


@pytest.mark.parametrize("kw", [NU3, NU3_L2, NU2])
def test_e3_round17(x64, graph, kw):
    model, g = as_float64(make_model(n_layers=3, **kw), graph)
    _check_e3(model, g, TOL)


@pytest.mark.parametrize("kw", [NU3, NU2])
def test_forces_fd_round17(x64, graph, kw):
    model, graph = as_float64(make_model(n_layers=3, **kw), graph)
    forces = np.asarray(model(graph)["forces"])
    pos = np.asarray(graph.nodes["positions"], dtype=np.float64)

    def energy(p):
        g = graph._replace(nodes={**graph.nodes, "positions": p.astype(np.float64)})
        return float(np.asarray(model(g)["energy"])[0])

    order = np.argsort(-np.abs(forces).ravel())[:2]
    picks = [(int(o) // 3, int(o) % 3) for o in order]
    assert abs(forces[picks[1]]) > 1e-4, forces[picks[1]]
    h = 1e-4
    for i, k in picks:
        pp, pm = pos.copy(), pos.copy(); pp[i, k] += h; pm[i, k] -= h
        fd = -(energy(pp) - energy(pm)) / (2 * h)
        assert abs(fd - forces[i, k]) / max(abs(forces[i, k]), 1e-6) < 1e-3, (fd, forces[i, k])


def test_layer_orders_per_layer():
    m = make_model(n_layers=3, **NU3_L2)
    assert [l.in_layer_order for l in m.layers] == [3, 3, 1]
    assert [l.sym_contraction for l in m.layers] == [True, True, False]
    assert m.layers[0].sym_degrees == (2, 3)
    m2 = make_model(n_layers=2, **NU2)
    assert m2.layers[0].sym_degrees == (2,) and m2.layers[0].sym_contraction
    # legacy r13 path untouched
    m3 = make_model(n_layers=2, in_layer_order=2)
    assert not m3.layers[0].sym_contraction and hasattr(m3.layers[0], "linear_sq")
    with pytest.raises(ValueError):
        make_model(n_layers=2, in_layer_order=5)      # 4 is allowed since round 24


def test_sym_term_is_exact_polynomial(x64, graph):
    """sym_term(lambda * A) = lambda^2 f2 + lambda^3 f3 exactly (degrees 2, 3): fit f2, f3 from
    lambda = 1, 2 and predict lambda = 3; degree-2 model is exactly quadratic."""
    model, g = as_float64(make_model(n_layers=2, seed=5, **NU3), graph)
    layer, m = _layer0_message(model, g)
    species = g.nodes["species"]
    f = lambda lam: np.asarray(layer.sym_term(m * lam, species).array, dtype=np.float64)
    f0, f1, f2_, f3_ = f(0.0), f(1.0), f(2.0), f(3.0)
    assert np.max(np.abs(f0)) < 1e-12
    # f1 = a + b, f2_ = 4a + 8b  ->  b = (f2_ - 4 f1) / 4, a = f1 - b
    b = (f2_ - 4 * f1) / 4
    a = f1 - b
    pred3 = 9 * a + 27 * b
    assert np.allclose(pred3, f3_, rtol=1e-9, atol=1e-9 * np.max(np.abs(f3_)))
    assert np.max(np.abs(b)) > 1e-8  # the cubic term is present
    model2, g2 = as_float64(make_model(n_layers=2, seed=5, **NU2), graph)
    layer2, m2 = _layer0_message(model2, g2)
    q1 = np.asarray(layer2.sym_term(m2, species).array, dtype=np.float64)
    q2 = np.asarray(layer2.sym_term(m2 * 2.0, species).array, dtype=np.float64)
    assert np.allclose(q2, 4 * q1, rtol=1e-9, atol=1e-9 * np.max(np.abs(q1)))


def test_sym_weights_are_species_dependent(x64, graph):
    model, g = as_float64(make_model(n_layers=2, seed=7, **NU3), graph)
    layer, m = _layer0_message(model, g)
    species = np.asarray(g.nodes["species"])
    other = (species + 1) % model.n_species
    t1 = np.asarray(layer.sym_term(m, species).array)
    t2 = np.asarray(layer.sym_term(m, jnp.asarray(other)).array)
    assert layer.sym_weights[...].shape == (model.n_species, layer.sym_n_weights)
    assert not np.allclose(t1, t2)
    # per-degree gain: zero gain -> zero term; gain is the stacked-weights order (2, 3)
    assert layer.sym_gain[...].shape == (2,) and sum(layer.sym_degree_sizes) == layer.sym_n_weights
    layer.sym_gain.value = jnp.zeros_like(layer.sym_gain.value)
    assert np.max(np.abs(np.asarray(layer.sym_term(m, species).array))) == 0.0


NU3_RO = dict(**V8, in_layer_order=3, sym_target="readout")


@pytest.mark.parametrize("kw", [NU3_RO])
def test_e3_and_fd_readout_target(x64, graph, kw):
    model, g = as_float64(make_model(n_layers=3, **kw), graph)
    _check_e3(model, g, TOL)
    forces = np.asarray(model(g)["forces"])
    pos = np.asarray(g.nodes["positions"], dtype=np.float64)
    i, k = divmod(int(np.argmax(np.abs(forces))), 3)
    h = 1e-4
    pp, pm = pos.copy(), pos.copy(); pp[i, k] += h; pm[i, k] -= h
    e = lambda p: float(np.asarray(model(g._replace(nodes={**g.nodes, "positions": p}))["energy"])[0])
    fd = -(e(pp) - e(pm)) / (2 * h)
    assert abs(fd - forces[i, k]) / abs(forces[i, k]) < 1e-3


def test_readout_target_keeps_features_nu1(x64, graph):
    """sym_target='readout': the propagated features do not depend on the product weights
    (zeroing the gain leaves h unchanged) while the energy does."""
    model, g = as_float64(make_model(n_layers=2, seed=3, **NU3_RO), graph)
    layer, m = _layer0_message(model, g)
    species = g.nodes["species"]
    feats, rb, sh = compute_node_edge_features(
        g.nodes["positions"], None, None, g, model.n_species, model.cutoff,
        model.radial_basis_size, model.radial_polynomial_p, model.lmax, periodic=False)
    h0 = model.species_embedding(feats)
    x = model.x_projs[0](model.layers[0].aggregate(h0, sh, rb, g.senders, g.receivers))
    e1, h1 = layer(h0, x, species, sh, rb, g.senders, g.receivers)
    layer.sym_gain.value = jnp.zeros_like(layer.sym_gain.value)
    e2, h2 = layer(h0, x, species, sh, rb, g.senders, g.receivers)
    # GPU scatter sums are not bit-reproducible between calls -> tight tolerance, not equality
    assert np.allclose(np.asarray(h1.array), np.asarray(h2.array), rtol=1e-9, atol=1e-12)
    assert not np.allclose(np.asarray(e1.array), np.asarray(e2.array), rtol=1e-6, atol=1e-9)
    assert hasattr(layer, "readout_sym") and layer.sym_target == "readout"


def test_sym_term_grads_finite(x64, graph):
    from flax import nnx
    model, g = as_float64(make_model(n_layers=2, **NU3), graph)
    graphdef, params = nnx.split(model, nnx.Param)

    def loss(p):
        out = nnx.merge(graphdef, p)(g)
        return jnp.sum(out["energy"] ** 2) + jnp.sum(out["forces"] ** 2)

    grads = jax.grad(loss)(params)
    leaves = jax.tree.leaves(grads)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in leaves)
    gw = jax.tree.leaves(grads["layers"][0]["sym_weights"])[0]
    assert float(jnp.max(jnp.abs(gw))) > 0.0
