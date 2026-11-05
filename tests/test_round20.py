"""Round-20 (review §11.15): radial regularity prior (radial_prior + radial_l2) and the PICPS
Taylor-consistency loss (taylor_consistency) on the v8 recipe."""
import math
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from conftest import make_model, as_float64, pad_single
from test_equivariance import _check_e3, TOL
from test_round11_shells import SHELLS
from bam_omat24.training_utils import compute_loss

V8 = dict(update="linear", readout_mode="gated", message_norm="avg", avg_n_neighbors=16.7, x_mode="basis")
SHELL = dict(radial_shells=True, species_shell_radii=SHELLS)


def _energy(model, graph):
    return float(np.asarray(model(graph)["energy"])[0])


@pytest.mark.parametrize("rp", [dict(type="algebraic", sigma=0.5, p=2.0), dict(type="gaussian", sigma=0.3),
                                dict(type="exponential", sigma=0.4)])
def test_radial_prior_scale_and_e3(x64, graph, rp):
    model, g = as_float64(make_model(n_layers=2, **V8, **SHELL, radial_prior=rp), graph)
    nb = model.radial_basis_size
    sc = np.asarray(model.radial_scale)
    assert sc.shape == (nb + model.n_shells ** 2,)
    n = np.arange(1, nb + 1)
    if rp["type"] == "algebraic":
        gam = (1 + rp["sigma"] * n) ** rp["p"]
    elif rp["type"] == "gaussian":
        gam = np.exp(rp["sigma"] ** 2 * n ** 2)
    else:
        gam = np.exp(rp["sigma"] * n)
    assert np.allclose(sc[:nb], 1.0 / gam) and np.all(sc[nb:] == 1.0)
    assert sc[0] > sc[nb - 1] > 0.0
    for layer in model.layers:
        assert np.allclose(np.asarray(layer.radial_scale), sc)
    _check_e3(model, g, TOL)
    # the prior changes the energy of the same-seed model
    plain, _ = as_float64(make_model(n_layers=2, seed=0, **V8, **SHELL), graph)
    assert abs(_energy(plain, g) - _energy(model, g)) > 1e-9


def test_radial_prior_rejects_unknown_type():
    with pytest.raises(ValueError):
        make_model(n_layers=2, radial_prior=dict(type="cosine"))


def test_radial_l2_penalty_value(graph):
    m = make_model(n_layers=2, **V8)
    graphdef, params = nnx.split(m, nnx.Param)
    g = pad_single(graph)
    _, a0 = compute_loss(graphdef, params, g, 1.0, 1.0, 0.0, 1e6, periodic=False)
    l1, a1 = compute_loss(graphdef, params, g, 1.0, 1.0, 0.0, 1e6, periodic=False, radial_l2=1e-3)
    l0, _ = compute_loss(graphdef, params, g, 1.0, 1.0, 0.0, 1e6, periodic=False)
    w2 = sum(float(jnp.sum(layer.radial_mlp.layers[0].weights[...] ** 2)) for layer in m.layers)
    assert w2 > 0 and abs(float(a1["rl2_loss"]) - w2) / w2 < 1e-6
    assert abs(float(l1) - float(l0) - 1e-3 * w2) / (1e-3 * w2) < 1e-5
    assert float(a0["rl2_loss"]) == 0.0


def _with_disp(graph, eps, seed=0):
    rng = np.random.RandomState(seed)
    n = int(graph.n_node[0])
    d = rng.normal(size=(n, 3)); d = d / np.linalg.norm(d, axis=1, keepdims=True) * eps
    return graph._replace(nodes={**graph.nodes, "taylor_disp": d.astype(np.float32)})


def test_taylor_consistency_matches_formula_and_order(x64, graph):
    """tc_loss == mean_graphs [(E(S+d) - E(S) + sum d.((1-k2)F(S) + k2 F(S+d))) / N]^2 computed
    independently from the model; the k2 = 1/2 (trapezoid, exact 2nd order) residual is never larger
    than the k2 = 0 one and shrinks faster with eps; zero displacement -> exactly 0."""
    model, g = as_float64(make_model(n_layers=2, seed=2, **V8), graph)
    graphdef, params = nnx.split(model, nnx.Param)
    n = int(g.n_node[0])

    def tc(eps, k2, seed=0):
        gg = pad_single(_with_disp(g, eps, seed))
        d64 = np.asarray(gg.nodes["taylor_disp"], np.float64)
        gg = gg._replace(nodes={**gg.nodes, "taylor_disp": d64})
        _, aux = compute_loss(graphdef, params, gg, 1.0, 1.0, 0.0, 1e6, periodic=False,
                              taylor_consistency=(1.0, eps, k2))
        # independent evaluation on the unpadded graph
        d = d64[:n]
        o1 = model(g); o2 = model(g._replace(nodes={**g.nodes, "positions": np.asarray(g.nodes["positions"]) + d}))
        e1, e2 = float(np.asarray(o1["energy"])[0]), float(np.asarray(o2["energy"])[0])
        f_mix = (1 - k2) * np.asarray(o1["forces"]) + k2 * np.asarray(o2["forces"])
        r = e2 - (e1 - float(np.sum(d * f_mix)))
        return float(aux["tc_loss"]), r

    assert tc(0.0, 0.5)[0] < 1e-30                 # zero displacement: residual at float rounding
    for eps in (0.02, 0.04):
        for k2 in (0.0, 0.5):
            loss, r = tc(eps, k2)
            ref = (r / n) ** 2                       # one real graph in the padded batch
            assert abs(loss - ref) <= 1e-9 * max(ref, 1e-30) + 1e-40, (loss, ref)
    # second-order estimate is at least as consistent as first order, for several directions
    for seed in range(4):
        r05 = [abs(tc(eps, 0.5, seed)[1]) for eps in (0.02, 0.04)]
        r00 = [abs(tc(eps, 0.0, seed)[1]) for eps in (0.02, 0.04)]
        assert r05[1] <= r00[1] * 1.001, (r05, r00)
        assert r05[1] / max(r05[0], 1e-300) >= 2.0        # grows at least as fast as eps^1 (expected eps^3)


def test_taylor_consistency_adds_to_loss_and_grads(graph):
    m = make_model(n_layers=2, **V8)
    graphdef, params = nnx.split(m, nnx.Param)
    g = pad_single(_with_disp(graph, 0.02))
    l0, a0 = compute_loss(graphdef, params, g, 1.0, 1.0, 0.0, 1e6, periodic=False)
    l1, a1 = compute_loss(graphdef, params, g, 1.0, 1.0, 0.0, 1e6, periodic=False, taylor_consistency=(3.0, 0.02, 0.5))
    tcv = float(a1["tc_loss"])
    assert tcv > 0
    # float32 total loss (~8) cannot resolve a 1e-9 term: check the aux bookkeeping and that the
    # weighted term is added (difference within float32 rounding of the total)
    assert abs(float(a1["aux_loss"]) - 3.0 * tcv) <= 1e-6 * 3.0 * tcv
    assert abs(float(l1) - float(l0) - 3.0 * tcv) <= 2e-6 * abs(float(l0))
    # without the displacement field the term is silently zero
    assert float(a0["tc_loss"]) == 0.0

    def loss(p):
        l, _ = compute_loss(graphdef, p, g, 1.0, 1.0, 0.0, 1e6, periodic=False, taylor_consistency=(1.0, 0.02, 0.5))
        return l
    grads = jax.grad(loss)(params)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(grads))
