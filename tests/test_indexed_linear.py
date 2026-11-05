"""BAM_INDEXED_LINEAR=ragged (jax.lax.ragged_dot over species groups) must reproduce the per-node weight gather of
the species-indexed Linear: outputs, first and second derivatives, and full RACEv2 energies / forces / stress."""
import os

import numpy as np
import jax
import jax.numpy as jnp
import e3nn_jax as e3nn
from flax import nnx
import pytest

from bam_omat24.linear_nnx import Linear
# imported here, not inside the tests: importing train_3bpa (via test_oam_stress) switches jax_enable_x64 off,
# which inside a test would undo the x64 fixture and compare float32 (TF32) results
from conftest import as_float64                                     # noqa: E402
from test_oam_stress import _model, _cell, _graph, _f64            # noqa: E402


def _set(impl):
    os.environ["BAM_INDEXED_LINEAR"] = impl


@pytest.fixture(autouse=True)
def _restore():
    old = os.environ.get("BAM_INDEXED_LINEAR")
    yield
    if old is None:
        os.environ.pop("BAM_INDEXED_LINEAR", None)
    else:
        os.environ["BAM_INDEXED_LINEAR"] = old


@pytest.mark.parametrize("irreps_in,irreps_out,biases", [("8x0e + 4x1o + 2x2e", "6x0e + 3x1o + 2x2e", False),
                                                         ("16x0e", "1x0e", True),
                                                         ("4x0e + 4x1o", "5x0e + 2x1o + 3x2e", True)])
def test_ragged_matches_gather(x64, irreps_in, irreps_out, biases):
    lin = Linear(irreps_out=irreps_out, irreps_in=irreps_in, biases=biases, linear_type="indexed",
                 num_indexed_weights=7, input_dtype=jnp.float64, rngs=nnx.Rngs(0))
    graphdef, params = nnx.split(lin, nnx.Param)
    params = jax.tree.map(lambda p: p.astype(jnp.float64), params)
    key = jax.random.PRNGKey(1)
    x = e3nn.normal(irreps_in, key, (23,), dtype=jnp.float64)
    sp = jax.random.randint(jax.random.PRNGKey(2), (23,), 0, 7)
    sp = sp.at[:3].set(5)                       # unsorted, repeated, some species absent

    def f(p, xa):
        return nnx.merge(graphdef, p)(sp, e3nn.IrrepsArray(irreps_in, xa)).array

    outs = {}
    for impl in ("gather", "ragged"):
        _set(impl)
        y = f(params, x.array)
        g = jax.grad(lambda p: jnp.sum(jnp.sin(f(p, x.array))))(params)
        gx = jax.grad(lambda xa: jnp.sum(jnp.sin(f(params, xa))))(x.array)
        h = jax.grad(lambda xa: jnp.sum(jax.grad(lambda z: jnp.sum(f(params, z) ** 3))(xa)))(x.array)
        outs[impl] = (y, g, gx, h)
    a, b = outs["gather"], outs["ragged"]
    assert np.allclose(a[0], b[0], rtol=1e-12, atol=1e-12)
    for la, lb in zip(jax.tree.leaves(a[1]), jax.tree.leaves(b[1])):
        assert np.allclose(la, lb, rtol=1e-11, atol=1e-12)
    assert np.allclose(a[2], b[2], rtol=1e-11, atol=1e-12)
    assert np.allclose(a[3], b[3], rtol=1e-10, atol=1e-11)


def test_race_model_identical_with_ragged(x64):
    a = _cell(2)
    model, _ = as_float64(_model(), _graph(a))
    g = _f64(_graph(a))
    _set("gather"); r0 = model(g)
    _set("ragged"); r1 = model(g)
    for k in ("energy", "forces", "stress"):
        assert np.allclose(np.asarray(r0[k]), np.asarray(r1[k]), rtol=1e-11, atol=1e-11), k


def test_remat_layers_identical(x64):
    """BAM_REMAT_LAYERS=1 (per-layer rematerialisation) gives the same energies, forces and stress."""
    a = _cell(3)
    model, _ = as_float64(_model(), _graph(a))
    g = _f64(_graph(a))
    old = os.environ.get("BAM_REMAT_LAYERS")
    try:
        os.environ["BAM_REMAT_LAYERS"] = "0"; r0 = model(g)
        os.environ["BAM_REMAT_LAYERS"] = "1"; r1 = model(g)
    finally:
        if old is None:
            os.environ.pop("BAM_REMAT_LAYERS", None)
        else:
            os.environ["BAM_REMAT_LAYERS"] = old
    for k in ("energy", "forces", "stress"):
        assert np.allclose(np.asarray(r0[k]), np.asarray(r1[k]), rtol=1e-12, atol=1e-12), k
