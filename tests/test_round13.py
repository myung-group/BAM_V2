"""Round-13 options: learnable shell-pair length scale (radial_shell_scale), in-layer symmetric
square (in_layer_order=2, "BRACE-2") and tapered per-l multiplicity."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from conftest import make_model, as_float64, pad_single
from test_equivariance import _check_e3, TOL
from test_round11_shells import SHELLS
from bam_omat24.training_utils import compute_loss

TAPER = "16x0e + 8x1o + 4x2e"


def _energy(model, graph):
    return float(np.asarray(model(graph)["energy"])[0])


@pytest.mark.parametrize("kw", [
    dict(radial_shells=True, species_shell_radii=SHELLS, radial_shell_scale=True),
    dict(in_layer_order=2),
    dict(in_layer_order=2, radial_shells=True, species_shell_radii=SHELLS, radial_shell_scale=True),
    dict(hidden_irreps=TAPER),
])
def test_e3_round13_options(x64, graph, kw):
    model, g = as_float64(make_model(n_layers=3, **kw), graph)
    _check_e3(model, g, TOL)


@pytest.mark.parametrize("kw", [dict(in_layer_order=2), dict(hidden_irreps=TAPER)])
def test_forces_fd_round13(x64, graph, kw):
    model, graph = as_float64(make_model(n_layers=3, radial_shells=True, species_shell_radii=SHELLS,
                                         radial_shell_scale=True, **kw), graph)
    forces = np.asarray(model(graph)["forces"])
    pos = np.asarray(graph.nodes["positions"], dtype=np.float64)

    def energy(p):
        g = graph._replace(nodes={**graph.nodes, "positions": p.astype(np.float64)})
        return float(np.asarray(model(g)["energy"])[0])

    cands = [(i, k) for i in range(pos.shape[0]) for k in range(3) if abs(forces[i, k]) > 0.05]
    rng = np.random.RandomState(13)
    picks = [cands[j] for j in rng.choice(len(cands), size=2, replace=False)]
    h = 1e-4
    for i, k in picks:
        pp, pm = pos.copy(), pos.copy(); pp[i, k] += h; pm[i, k] -= h
        fd = -(energy(pp) - energy(pm)) / (2 * h)
        assert abs(fd - forces[i, k]) / max(abs(forces[i, k]), 1e-6) < 1e-3, (fd, forces[i, k])


def test_shell_scale_init_matches_fixed_envelopes(graph):
    """softplus(raw) = 1 at init: the model with the learnable scale equals the fixed one."""
    fixed = make_model(n_layers=2, seed=7, radial_shells=True, species_shell_radii=SHELLS)
    scaled = make_model(n_layers=2, seed=7, radial_shells=True, species_shell_radii=SHELLS,
                        radial_shell_scale=True)
    assert abs(_energy(fixed, graph) - _energy(scaled, graph)) < 1e-5
    assert np.allclose(jax.nn.softplus(scaled.shell_scale_raw[...]), 1.0, atol=1e-6)
    # a changed scale changes the energy
    scaled.shell_scale_raw.value = scaled.shell_scale_raw.value - 1.0
    assert abs(_energy(fixed, graph) - _energy(scaled, graph)) > 1e-6


def test_shell_scale_requires_shells():
    with pytest.raises(ValueError):
        make_model(n_layers=2, radial_shell_scale=True)


def test_in_layer_order_changes_energy_default_unchanged(graph):
    base = make_model(n_layers=2, seed=9)
    assert _energy(base, graph) == _energy(make_model(n_layers=2, seed=9, in_layer_order=1), graph)
    assert abs(_energy(base, graph) - _energy(make_model(n_layers=2, seed=9, in_layer_order=2), graph)) > 1e-6
    with pytest.raises(ValueError):
        make_model(n_layers=2, in_layer_order=4)      # 3 is the round-17 symmetric contraction
    with pytest.raises(ValueError):
        make_model(n_layers=2, in_layer_order=2, hidden_irreps=TAPER)


def test_in_layer_square_is_quadratic_in_message(graph):
    """Scaling the aggregated message by c scales the square term by c^2 (and is non-zero)."""
    m = make_model(n_layers=2, seed=3, in_layer_order=2)
    layer = m.layers[1]
    import e3nn_jax as e3nn
    rng = np.random.RandomState(0)
    irreps = e3nn.Irreps(layer.linear_sq_in.irreps_in)          # tp_irreps (input of the square)
    msg = e3nn.IrrepsArray(irreps, jnp.asarray(rng.randn(5, irreps.dim), jnp.float32))

    def sq(ma):
        m1 = layer.linear_sq_in(ma).mul_to_axis(layer.sq_mul)
        return layer.linear_sq(e3nn.tensor_product(m1, m1, filter_ir_out=layer.sq_irreps_1).axis_to_mul()).array

    a, b = np.asarray(sq(msg)), np.asarray(sq(msg * 2.0))
    assert np.abs(a).max() > 1e-6
    assert np.allclose(b, 4.0 * a, rtol=1e-4, atol=1e-5)


def test_round13_grads_finite_with_padding(graph):
    m = make_model(n_layers=2, in_layer_order=2, radial_shells=True, species_shell_radii=SHELLS,
                   radial_shell_scale=True)
    graphdef, params = nnx.split(m, nnx.Param)
    g = pad_single(graph)

    def loss(p):
        l, _ = compute_loss(graphdef, p, g, 1.0, 10.0, 0.0, 10.0, periodic=False)
        return l

    val, grads = jax.value_and_grad(loss)(params)
    assert np.isfinite(float(val))
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(grads))
    gs = grads["shell_scale_raw"]; gs = gs.value if hasattr(gs, "value") else gs
    assert bool(jnp.any(jnp.abs(jnp.asarray(gs)) > 0))
