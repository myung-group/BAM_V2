"""Round 26 (review §12.7, literature sweep 1B): homogeneous bond-order scale stratum
E_bo = -sum_i s_Z phi(b_i) (1 + g(q_i)), b_i = (eps + sum_j beta_ZZ'(r_ij)^2)^(1/2)."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from conftest import make_model, as_float64
from test_equivariance import _check_e3, TOL
from test_round24 import V9, _e, _fd_forces
from bam_omat24.utils import polynomial_cutoff

HILL = dict(**V9, residual="chebyshev", layer_scale=0.5, x_norm="hill")
BO_POW = dict(**HILL, bond_order="power")
BO_P1 = dict(**HILL, bond_order="power", bond_order_p=1.0)
BO_LOG = dict(**HILL, bond_order="log")


def _set(model, name, value):
    getattr(model, name)[...] = jnp.asarray(value, getattr(model, name)[...].dtype)


def _switch_on(model, seed=0):
    """Non-trivial bond-order parameters: s_Z != 0, asymmetric c, non-zero shape MLP output layer."""
    rng = np.random.default_rng(seed)
    _set(model, "bo_s", rng.normal(size=model.bo_s[...].shape) * 0.3)
    _set(model, "bo_c", rng.normal(size=model.bo_c[...].shape))
    if model.bond_order_shape:
        last = model.bo_shape_mlp.layers[-1]
        last.weights[...] = jnp.asarray(rng.normal(size=last.weights[...].shape) * 0.1, last.weights[...].dtype)
    return model


@pytest.mark.parametrize("kw", [BO_POW, BO_LOG])
def test_e3_and_forces_bond_order(x64, graph, kw):
    model, g = as_float64(_switch_on(make_model(n_layers=4, **kw)), graph)
    _check_e3(model, g, TOL)
    _fd_forces(model, g)


def test_bond_order_starts_as_base(x64, graph):
    base, g = as_float64(make_model(n_layers=4, **HILL), graph)
    bo, _ = as_float64(make_model(n_layers=4, **BO_POW), graph)
    assert abs(_e(base, g) - _e(bo, g)) < 1e-10 * max(1.0, abs(_e(base, g)))


def test_bond_order_width_closed_form(x64, graph):
    """No shape factor, p = 1: E_bo = -sum_i s_Z b_i with b_i from a direct NumPy evaluation."""
    kw = dict(BO_P1, bond_order_shape=False)
    on, g = as_float64(_switch_on(make_model(n_layers=3, **kw)), graph)
    off = make_model(n_layers=3, **kw)
    off, _ = as_float64(off, graph)
    # same parameters except s_Z = 0
    gd, st = nnx.split(on, nnx.Param)
    st0 = jax.tree_util.tree_map_with_path(
        lambda p, x: jnp.zeros_like(x) if any(getattr(k, "key", None) == "bo_s" for k in p) else x, st)
    off = nnx.merge(gd, st0)
    dE = _e(on, g) - _e(off, g)
    pos = np.asarray(g.nodes["positions"], np.float64)
    sp = np.asarray(g.nodes["species"]); snd, rcv = np.asarray(g.senders), np.asarray(g.receivers)
    r = np.linalg.norm(pos[snd] - pos[rcv], axis=-1)
    c = np.log1p(np.exp(np.asarray(on.bo_c[...], np.float64))); c = 0.5 * (c + c.transpose(1, 0, 2))
    env = np.asarray(polynomial_cutoff(jnp.asarray(r), jnp.full_like(jnp.asarray(r), on.cutoff), on.radial_polynomial_p))
    beta = (c[sp[rcv], sp[snd]] * np.exp(-r[:, None] * np.array(on.bo_alpha)[None])).sum(-1) * env
    b = np.sqrt(1e-6 + np.bincount(rcv, beta ** 2, minlength=len(sp)))
    ref = -(np.asarray(on.bo_s[...], np.float64)[sp] * b).sum() * float(np.asarray(on.scale[...]))
    assert abs(dE - ref) < 1e-9 * max(1.0, abs(ref)), (dE, ref)


def test_bond_order_validation():
    with pytest.raises(ValueError):
        make_model(n_layers=3, **dict(HILL, bond_order="cubic"))
