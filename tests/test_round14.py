"""Round-14 options (review §10): Chebyshev three-term recurrence over layers (residual="chebyshev"),
Slater–Koster shell-pair × l selection-rule gate on Y_l (shell_sk_gate) and the Aitken geometric-tail
resummation stratum (strata_resum="aitken")."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from conftest import make_model, as_float64, pad_single
from test_equivariance import _check_e3, TOL
from test_round11_shells import SHELLS
from bam_omat24.training_utils import compute_loss

SK = dict(radial_shells=True, species_shell_radii=SHELLS, shell_sk_gate=True)


def _energy(model, graph):
    return float(np.asarray(model(graph)["energy"])[0])


@pytest.mark.parametrize("kw", [
    dict(residual="chebyshev"),
    dict(**SK),
    dict(strata_resum="aitken"),
    dict(residual="chebyshev", strata_resum="aitken", **SK),
])
def test_e3_round14_options(x64, graph, kw):
    model, g = as_float64(make_model(n_layers=4, **kw), graph)
    _check_e3(model, g, TOL)


def _perturb(model):
    """Move the init-zero parameters off zero so the options are active in the FD / grad tests."""
    if hasattr(model, "sk_coeff"):
        model.sk_coeff.value = model.sk_coeff.value + 0.3
    if hasattr(model, "resum_lambda"):
        model.resum_lambda.value = model.resum_lambda.value + 0.7
    return model


@pytest.mark.parametrize("kw", [dict(residual="chebyshev"), dict(**SK), dict(strata_resum="aitken")])
def test_forces_fd_round14(x64, graph, kw):
    model = _perturb(make_model(n_layers=4, **kw))
    model, graph = as_float64(model, graph)
    forces = np.asarray(model(graph)["forces"])
    pos = np.asarray(graph.nodes["positions"], dtype=np.float64)

    def energy(p):
        g = graph._replace(nodes={**graph.nodes, "positions": p.astype(np.float64)})
        return float(np.asarray(model(g)["energy"])[0])

    cands = [(i, k) for i in range(pos.shape[0]) for k in range(3) if abs(forces[i, k]) > 0.05]
    rng = np.random.RandomState(14)
    picks = [cands[j] for j in rng.choice(len(cands), size=2, replace=False)]
    h = 1e-4
    for i, k in picks:
        pp, pm = pos.copy(), pos.copy(); pp[i, k] += h; pm[i, k] -= h
        fd = -(energy(pp) - energy(pm)) / (2 * h)
        assert abs(fd - forces[i, k]) / max(abs(forces[i, k]), 1e-6) < 1e-3, (fd, forces[i, k])


def test_sk_gate_and_resum_are_identity_at_init(graph):
    base = make_model(n_layers=3, seed=5, radial_shells=True, species_shell_radii=SHELLS)
    sk = make_model(n_layers=3, seed=5, **SK)
    assert abs(_energy(base, graph) - _energy(sk, graph)) < 1e-5
    rs = make_model(n_layers=3, seed=5, radial_shells=True, species_shell_radii=SHELLS, strata_resum="aitken")
    assert abs(_energy(base, graph) - _energy(rs, graph)) < 1e-5
    assert np.asarray(rs(graph)["strata"]).shape[0] == 4           # 3 layers + resummation stratum
    # off-zero parameters change the energy
    assert abs(_energy(base, graph) - _energy(_perturb(sk), graph)) > 1e-6
    assert abs(_energy(base, graph) - _energy(_perturb(rs), graph)) > 1e-6


def test_sk_mask_selection_rule():
    m = make_model(n_layers=2, **SK)                                  # lmax 2 in make_model
    mask = np.asarray(m.sk_mask)                                      # (16, 3): rows (a, b) with a, b in s p d f
    n = 4
    def row(a, b): return mask[a * n + b].tolist()
    assert row(0, 0) == [1, 0, 0]                                     # s-s: l = 0
    assert row(0, 1) == [0, 1, 0]                                     # s-p: l = 1
    assert row(1, 1) == [1, 0, 1]                                     # p-p: l = 0, 2
    assert row(1, 2) == [0, 1, 0]                                     # p-d: l = 1 (3 cut by lmax)
    assert row(2, 2) == [1, 0, 1]                                     # d-d: l = 0, 2 (4 cut by lmax)
    assert np.array_equal(mask, mask.reshape(n, n, -1).transpose(1, 0, 2).reshape(n * n, -1))   # symmetric in (a, b)
    assert m.sk_coeff.value.shape == (2, 16, 3)


def test_chebyshev_changes_energy_and_needs_three_layers(graph):
    base = make_model(n_layers=3, seed=9)
    cheb = make_model(n_layers=3, seed=9, residual="chebyshev")
    assert abs(_energy(base, graph) - _energy(cheb, graph)) > 1e-6
    # with 2 layers no layer has a hidden-irreps h_{t-1}: identical to residual "none"
    assert abs(_energy(make_model(n_layers=2, seed=9), graph)
               - _energy(make_model(n_layers=2, seed=9, residual="chebyshev"), graph)) < 1e-6


def test_resum_tail_is_bounded(graph):
    m = _perturb(make_model(n_layers=3, seed=4, strata_resum="aitken"))
    strata = np.asarray(m(graph)["strata"])                          # (4, n_atoms)
    tail, e_last = strata[3], strata[2]
    # |rho| < 0.5 -> |tail| <= lambda * |e_last| * 0.5 / 0.5 = 0.7 |e_last|
    assert np.all(np.abs(tail) <= 0.7 * np.abs(e_last) + 1e-9)


def test_round14_grads_finite_with_padding(graph):
    m = make_model(n_layers=3, residual="chebyshev", strata_resum="aitken", **SK)
    graphdef, params = nnx.split(m, nnx.Param)
    g = pad_single(graph)

    def loss(p):
        l, _ = compute_loss(graphdef, p, g, 1.0, 10.0, 0.0, 10.0, periodic=False)
        return l

    val, grads = jax.value_and_grad(loss)(params)
    assert np.isfinite(float(val))
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(grads))
    for name in ("sk_coeff", "resum_lambda"):
        gv = grads[name]; gv = gv.value if hasattr(gv, "value") else gv
        assert bool(jnp.any(jnp.abs(jnp.asarray(gv)) > 0)), name
