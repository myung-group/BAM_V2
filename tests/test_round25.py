"""Round 25 (review §12.4, Chebyshev-RACE): with update="linear" a layer is a linear map h -> M(R) h
for fixed geometry; tie_operator shares that operator across layers 1.. (own readouts), residual
"chebyshev" turns the strata into T_t(M) u_0, x_norm bounds the A_i factor, and stratum_gain
"jackson" damps the high orders with the KPM Jackson kernel."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from conftest import make_model, as_float64
from test_equivariance import _check_e3, TOL
from test_round24 import V9, _e, _fd_forces
from bam_omat24.race_nnx import _jackson_gain

CHEB = dict(**V9, residual="chebyshev", layer_scale=0.5, x_norm="rms")
TIED = dict(**CHEB, tie_operator=True)
TIED_J = dict(**TIED, stratum_gain="jackson")


def _n_params(model):
    return sum(int(np.size(x)) for x in jax.tree.leaves(nnx.state(model, nnx.Param)))


@pytest.mark.parametrize("kw", [CHEB, TIED, TIED_J])
def test_e3_cheb(x64, graph, kw):
    model, g = as_float64(make_model(n_layers=4, **kw), graph)
    _check_e3(model, g, TOL)


@pytest.mark.parametrize("kw", [CHEB, TIED_J])
def test_forces_fd_cheb(x64, graph, kw):
    model, g = as_float64(make_model(n_layers=4, **kw), graph)
    _fd_forces(model, g)


def test_tied_shares_operator_and_survives_split_merge(x64, graph):
    untied = make_model(n_layers=4, **CHEB)
    tied = make_model(n_layers=4, **TIED)
    assert _n_params(tied) < _n_params(untied)
    l1, l3 = tied.layers[1], tied.layers[3]
    assert l1.linear_2 is l3.linear_2 and l1.radial_mlp is l3.radial_mlp and l1.skip is l3.skip
    assert l1.readout1 is not l3.readout1 and tied.x_projs[1] is tied.x_projs[3]
    assert tied.x_projs[0] is not tied.x_projs[1]
    gd, st = nnx.split(tied)
    back = nnx.merge(gd, st)
    assert back.layers[1].linear_2 is back.layers[3].linear_2
    model, g = as_float64(tied, graph)
    model2, _ = as_float64(back, graph)
    assert abs(_e(model, g) - _e(model2, g)) < 1e-10 * max(1.0, abs(_e(model, g)))


def test_tied_validation():
    with pytest.raises(ValueError):
        make_model(n_layers=2, **TIED)
    with pytest.raises(ValueError):
        make_model(n_layers=3, **dict(V9, stratum_gain="kpm"))


def test_jackson_gain():
    g = [_jackson_gain(n, 5) for n in range(5)]
    assert abs(g[0] - 1.0) < 1e-12 and abs(g[4] - 1.0 / 12.0) < 1e-12
    assert all(a > b > 0 for a, b in zip(g, g[1:]))


NEUM = dict(**V9, residual="neumann", layer_scale=0.5, x_norm="rms", tie_operator=True)


def test_e3_and_forces_neumann(x64, graph):
    model, g = as_float64(make_model(n_layers=4, **NEUM), graph)
    _check_e3(model, g, TOL)
    _fd_forces(model, g)


def test_neumann_is_partial_resolvent(x64, graph):
    """With a shared operator, h_{t+1} - h_t = M^t u_0 = M (h_t - h_{t-1}) (geometric partial sums)."""
    import bam_omat24.race_nnx as R
    import e3nn_jax as e3nn
    model, g = as_float64(make_model(n_layers=4, **NEUM), graph)
    cap = []
    orig = R.RACEConvolution.__call__
    def rec(self, *a, **k):
        out = orig(self, *a, **k); cap.append((self, a, k, out[1])); return out
    R.RACEConvolution.__call__ = rec
    try:
        model.node_energies(g.nodes["positions"], None, None, g)     # forward only (model() takes grads)
    finally:
        R.RACEConvolution.__call__ = orig
    h = [c[3].array for c in cap]                     # h_1 .. h_4 (layer outputs)
    lay, a, k, _ = cap[2]
    m_d = orig(lay, e3nn.IrrepsArray(a[0].irreps, h[1] - h[0]), *a[1:], **dict(k, prev_features=None))[1].array
    np.testing.assert_allclose(h[2] - h[1], m_d, rtol=1e-8, atol=1e-10)


# review §12.5: x_norm "hill" (rms + bounded degree gate) and "spectral" (global operator-norm bound)
HILL = dict(**V9, residual="neumann", layer_scale=0.5, x_norm="hill", tie_operator=True)
SPEC = dict(**V9, residual="chebyshev", layer_scale=0.5, x_norm="spectral")


@pytest.mark.parametrize("kw", [HILL, SPEC])
def test_e3_and_forces_hill_spectral(x64, graph, kw):
    model, g = as_float64(make_model(n_layers=4, **kw), graph)
    _check_e3(model, g, TOL)
    _fd_forces(model, g)


def test_hill_init_is_rms_and_gate_values(x64, graph):
    import e3nn_jax as e3nn
    from bam_omat24.race_nnx import _hill_normalize, _rms_normalize
    rms = make_model(n_layers=4, **dict(HILL, x_norm="rms"))
    hill = make_model(n_layers=4, **HILL)
    hill_st = nnx.state(hill, nnx.Param)
    rms_st = nnx.state(rms, nnx.Param)
    assert "x_hill" in hill_st and "x_hill" not in rms_st
    m1, g = as_float64(rms, graph)
    m2, _ = as_float64(hill, graph)
    assert abs(_e(m1, g) - _e(m2, g)) < 1e-10 * max(1.0, abs(_e(m1, g)))
    x = e3nn.IrrepsArray("2x0e+1x1o", jnp.asarray(np.random.default_rng(0).normal(size=(3, 5)) * 3.0))
    sp = jnp.array([0, 1, 0])
    ab = jnp.array([[1.0, 1.0], [0.3, -0.2]])          # a = 1 (both species), b per species
    r = jnp.sqrt(jnp.mean(x.array ** 2, -1, keepdims=True) + 1e-6)
    ref = 2.0 * x.array / (r + jnp.exp(-ab[1][sp])[:, None])
    assert np.allclose(_hill_normalize(x, ab, sp).array, ref, rtol=1e-12)
    assert np.allclose(_hill_normalize(x, jnp.zeros((2, 2)), sp).array, _rms_normalize(x).array, rtol=1e-12)


def test_spectral_block_operator_norm_is_one(x64):
    """The normalised x projection has operator norm 1 on every irrep block (power iteration on the
    map itself, through lin.__call__, so the weight bookkeeping in _spectral_normalize is checked)."""
    import e3nn_jax as e3nn
    from bam_omat24.race_nnx import _spectral_normalize
    from bam_omat24.linear_nnx import Linear
    irr_in, irr_out = e3nn.Irreps("6x0e+4x0e+5x1o"), e3nn.Irreps("7x0e+3x1o+2x2e")
    lin = Linear(irreps_in=irr_in, irreps_out=irr_out, biases=False, force_irreps_out=True, rngs=nnx.Rngs(3))
    nnx.update(lin, jax.tree.map(lambda v: v.astype(jnp.float64), nnx.state(lin, nnx.Param)))
    f = lambda a: _spectral_normalize(lin, lin(e3nn.IrrepsArray(irr_in, a[None]))).array[0]
    rng = np.random.default_rng(1)
    for blk, (mul, ir) in zip(e3nn.Irreps(irr_out).slices(), irr_out):
        if ir.l == 2:          # unreachable from the input: structurally zero, scale 1
            assert np.allclose(f(jnp.asarray(rng.normal(size=(irr_in.dim,))))[blk], 0.0)
            continue
        J = jax.jacfwd(lambda a: f(a)[blk])(jnp.zeros(irr_in.dim))
        assert abs(np.linalg.norm(np.asarray(J), 2) - 1.0) < 1e-10
