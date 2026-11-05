"""tp_backend="cue" (cuEquivariance fused edge tensor product) must reproduce the
e3nn backend with identical parameters.  Skipped when cuequivariance-jax is absent."""
import numpy as np
import pytest
import jax
from flax import nnx

from conftest import make_model, pad_single, random_rotation

cuex = pytest.importorskip("cuequivariance_jax")


@pytest.mark.parametrize("kw", [
    dict(hidden_irreps="16x0e + 16x1o + 16x2e", lmax=2, n_layers=2),
    dict(hidden_irreps="16x0e+16x1o+16x2e+16x3o", lmax=3, n_layers=3),
])
def test_cue_matches_e3nn_same_params(graph, kw):
    m_e = make_model(**kw)
    graphdef_e, params = nnx.split(m_e, nnx.Param)
    m_c = make_model(tp_backend="cue", **kw)
    graphdef_c, params_c = nnx.split(m_c, nnx.Param)
    assert jax.tree.structure(params) == jax.tree.structure(params_c)
    m_c = nnx.merge(graphdef_c, params)          # cue model with the e3nn parameters
    g = pad_single(graph)
    oe, oc = m_e(g), m_c(g)
    e_scale = max(1.0, float(np.abs(np.asarray(oe["energy"])).max()))
    f_scale = max(1.0, float(np.abs(np.asarray(oe["forces"])).max()))
    assert np.abs(np.asarray(oe["energy"]) - np.asarray(oc["energy"])).max() < 1e-4 * e_scale
    assert np.abs(np.asarray(oe["forces"]) - np.asarray(oc["forces"])).max() < 1e-3 * f_scale


def test_cue_equivariance_float32(graph):
    model = make_model(tp_backend="cue")
    R = random_rotation(4)
    pos = np.asarray(graph.nodes["positions"])
    g2 = graph._replace(nodes={**graph.nodes, "positions": (pos @ R.T).astype(np.float32)})
    o1, o2 = model(graph), model(g2)
    assert np.abs(np.asarray(o1["energy"]) - np.asarray(o2["energy"])).max() < 2e-3 * max(1.0, float(np.abs(np.asarray(o1["energy"])).max()))
    assert np.abs(np.asarray(o1["forces"]) @ R.T - np.asarray(o2["forces"])).max() < 2e-3 * max(1.0, float(np.abs(np.asarray(o1["forces"])).max()))
