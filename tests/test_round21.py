"""Round-21 (review §11.16): readout variants — x_readout off|layer0|linear, x_readout_norm False,
f_readout False — on the v8 recipe."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from conftest import make_model, as_float64
from test_equivariance import _check_e3, TOL
from bam_omat24.utils import compute_node_edge_features

V8 = dict(update="linear", readout_mode="gated", message_norm="avg", avg_n_neighbors=16.7, x_mode="basis")


def _energy(model, graph):
    return float(np.asarray(model(graph)["energy"])[0])


def _n_params(m):
    return sum(int(np.prod(x.shape)) for x in jax.tree.leaves(nnx.split(m, nnx.Param)[1]))


@pytest.mark.parametrize("kw", [dict(x_readout="off"), dict(x_readout="layer0"), dict(x_readout="linear"),
                                dict(x_readout_norm=False), dict(f_readout=False),
                                dict(x_readout="linear", f_readout=False, x_readout_norm=False)])
def test_e3_round21(x64, graph, kw):
    model, g = as_float64(make_model(n_layers=3, **V8, **kw), graph)
    _check_e3(model, g, TOL)


def test_x_readout_off_and_layer0(graph):
    base = make_model(n_layers=3, seed=4, **V8)
    off = make_model(n_layers=3, seed=4, **V8, x_readout="off")
    l0 = make_model(n_layers=3, seed=4, **V8, x_readout="layer0")
    assert len(off.x_readouts) == 0 and len(l0.x_readouts) == 1 and len(base.x_readouts) == 3
    assert _n_params(off) < _n_params(l0) < _n_params(base)
    # the energy of 'off' does not depend on anything x-readout related: zeroing layer-0 x readout
    # of 'layer0' changes its energy, 'off' has none to zero
    e_l0 = _energy(l0, graph)
    gd, st = nnx.split(l0.x_readouts[0], nnx.Param)
    nnx.update(l0.x_readouts[0], jax.tree.map(lambda a: a * 0.0, st))
    assert abs(_energy(l0, graph) - e_l0) > 1e-7
    with pytest.raises(ValueError):
        make_model(n_layers=2, x_readout="quadratic")


def test_x_readout_linear_is_linear_in_x(x64, graph):
    model, g = as_float64(make_model(n_layers=2, seed=5, **V8, x_readout="linear"), graph)
    ro = model.x_readouts[0]
    feats, rb, sh = compute_node_edge_features(
        g.nodes["positions"], None, None, g, model.n_species, model.cutoff,
        model.radial_basis_size, model.radial_polynomial_p, model.lmax, periodic=False)
    h0 = model.species_embedding(feats)
    A = model.layers[0].aggregate(h0, sh, rb, g.senders, g.receivers)
    x = model.x_projs[0](A)
    species = g.nodes["species"]
    f1 = np.asarray(ro(species, x).array); f2 = np.asarray(ro(species, x * 2.0).array)
    f0 = np.asarray(ro(species, x * 0.0).array)                       # bias only
    assert np.allclose(f2 - f0, 2.0 * (f1 - f0), rtol=1e-10, atol=1e-12)
    assert ro.linear and not hasattr(ro, "x_readout2")


def test_x_readout_norm_flag(x64, graph):
    """With the layer norm the x readout is degree-0 (scale-invariant) in x; without it, it is not."""
    on, g = as_float64(make_model(n_layers=2, seed=6, **V8), graph)
    off, _ = as_float64(make_model(n_layers=2, seed=6, **V8, x_readout_norm=False), graph)
    feats, rb, sh = compute_node_edge_features(
        g.nodes["positions"], None, None, g, on.n_species, on.cutoff,
        on.radial_basis_size, on.radial_polynomial_p, on.lmax, periodic=False)
    h0 = on.species_embedding(feats)
    x = on.x_projs[0](on.layers[0].aggregate(h0, sh, rb, g.senders, g.receivers))
    species = g.nodes["species"]
    a1 = np.asarray(on.x_readouts[0](species, x).array); a2 = np.asarray(on.x_readouts[0](species, x * 3.0).array)
    rel_on = np.max(np.abs(a1 - a2)) / np.max(np.abs(a1))              # LN (with its eps) ~ removes the scale
    b1 = np.asarray(off.x_readouts[0](species, x).array); b2 = np.asarray(off.x_readouts[0](species, x * 3.0).array)
    rel_off = np.max(np.abs(b1 - b2)) / np.max(np.abs(b1))
    assert rel_on < 0.02, rel_on
    assert rel_off > 5 * rel_on and rel_off > 0.1, (rel_on, rel_off)


def test_f_readout_off(graph):
    base = make_model(n_layers=3, seed=7, **V8)
    noff = make_model(n_layers=3, seed=7, **V8, f_readout=False)
    assert len(noff.f_readouts) == 0 and len(base.f_readouts) == 3 and not noff.use_f_readout
    assert _n_params(noff) < _n_params(base)
    assert np.isfinite(_energy(noff, graph))
