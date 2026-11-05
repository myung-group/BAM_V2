"""Round-18 options (review §11.12): per-node force-loss weights (nodes['loss_weight']), restricted
basis-x (x_basis_lmax, x_basis_layers); init_params_from is covered by the mini run."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from conftest import make_model, as_float64, pad_single
from test_equivariance import _check_e3, TOL
from bam_omat24.training_utils import compute_loss

V8 = dict(update="linear", readout_mode="gated", message_norm="avg", avg_n_neighbors=16.7, x_mode="basis")


def _energy(model, graph):
    return float(np.asarray(model(graph)["energy"])[0])


def test_node_loss_weight_scales_force_loss(graph):
    m = make_model(n_layers=2, **V8)
    graphdef, params = nnx.split(m, nnx.Param)
    g0 = pad_single(graph)
    n = int(graph.n_node[0])
    w = np.full((n,), 2.0, np.float32)
    g2 = pad_single(graph._replace(nodes={**graph.nodes, "loss_weight": w}))
    _, a0 = compute_loss(graphdef, params, g0, 1.0, 1.0, 0.0, 1e6, periodic=False)
    _, a2 = compute_loss(graphdef, params, g2, 1.0, 1.0, 0.0, 1e6, periodic=False)
    f0, f2 = float(a0["force_loss"]), float(a2["force_loss"])
    assert f0 > 0 and abs(f2 - 2 * f0) / f2 < 1e-5, (f0, f2)
    # energy term and metrics untouched
    assert abs(float(a0["energy_loss"]) - float(a2["energy_loss"])) < 1e-9
    # non-uniform weights: weighted sum of per-node terms
    w = np.linspace(0.5, 1.5, n).astype(np.float32)
    g3 = pad_single(graph._replace(nodes={**graph.nodes, "loss_weight": w}))
    _, a3 = compute_loss(graphdef, params, g3, 1.0, 1.0, 0.0, 1e6, periodic=False)
    assert float(a3["force_loss"]) != f0
    # padded nodes carry weight 0: a padded graph with weights 1 equals the unweighted loss
    g1 = pad_single(graph._replace(nodes={**graph.nodes, "loss_weight": np.ones((n,), np.float32)}))
    _, a1 = compute_loss(graphdef, params, g1, 1.0, 1.0, 0.0, 1e6, periodic=False)
    assert abs(float(a1["force_loss"]) - f0) / f0 < 1e-6


@pytest.mark.parametrize("kw", [dict(x_basis_lmax=0), dict(x_basis_lmax=1), dict(x_basis_layers=[0, 1]),
                                dict(x_basis_layers=[2])])
def test_e3_round18(x64, graph, kw):
    model, g = as_float64(make_model(n_layers=3, **V8, **kw), graph)
    _check_e3(model, g, TOL)


def test_x_basis_lmax_mask(graph):
    m = make_model(n_layers=2, **V8, x_basis_lmax=0)
    mask = np.asarray(m.x_basis_mask)
    irreps = m.hidden_irreps
    assert mask.shape == (irreps.dim,)
    i = 0
    for mul, ir in irreps:
        blk = mask[i:i + mul * ir.dim]
        assert np.all(blk == (1.0 if ir.l == 0 else 0.0))
        i += mul * ir.dim
    # the restriction changes the energy relative to the full basis-x model (same seed)
    full = make_model(n_layers=2, seed=0, **V8)
    assert abs(_energy(full, graph) - _energy(m, graph)) > 1e-7
    # lmax >= hidden lmax is the identity
    same = make_model(n_layers=2, seed=0, **V8, x_basis_lmax=2)
    assert abs(_energy(full, graph) - _energy(same, graph)) < 1e-6


def test_x_basis_layers_fallback(graph):
    """Layers outside x_basis_layers use the per-species x: with x_basis_layers=[] the model equals
    the same model whose basis projections are irrelevant (energy independent of x_projs)."""
    m = make_model(n_layers=2, seed=1, **V8, x_basis_layers=[])
    e1 = _energy(m, graph)
    for p in m.x_projs:
        graphdef, state = nnx.split(p, nnx.Param)
        state = jax.tree.map(lambda a: a * 0.0, state)
        nnx.update(p, state)
    e2 = _energy(m, graph)
    assert abs(e1 - e2) < 1e-9
    m2 = make_model(n_layers=2, seed=1, **V8, x_basis_layers=[0])
    assert abs(_energy(m2, graph) - e1) > 1e-7
