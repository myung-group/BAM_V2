"""Round-23 (review §11.18): per-layer radial prior sigma, angular prior on Y_l, linear radial MLP."""
import math
import numpy as np
import pytest

from conftest import make_model, as_float64
from test_equivariance import _check_e3, TOL

V9 = dict(update="linear", readout_mode="gated", message_norm="avg", avg_n_neighbors=16.7, x_mode="basis",
          radial_prior={"type": "gaussian", "sigma": 0.224})


def _energy(model, graph):
    return float(np.asarray(model(graph)["energy"])[0])


def test_per_layer_sigma(x64, graph):
    sig = [0.1, 0.2, 0.3]
    model, g = as_float64(make_model(n_layers=3, **{**V9, "radial_prior": {"type": "gaussian", "sigma": sig}}), graph)
    nb = model.radial_basis_size
    n = np.arange(1, nb + 1)
    for t, layer in enumerate(model.layers):
        assert np.allclose(np.asarray(layer.radial_scale)[:nb], np.exp(-(sig[t] ** 2) * n ** 2))
    assert np.asarray(model.layers[2].radial_scale)[nb - 1] < np.asarray(model.layers[0].radial_scale)[nb - 1]
    assert model.radial_scale == model.layers[0].radial_scale
    _check_e3(model, g, TOL)
    with pytest.raises(ValueError):
        make_model(n_layers=3, **{**V9, "radial_prior": {"type": "gaussian", "sigma": [0.1, 0.2]}})


def test_angular_prior(x64, graph):
    model, g = as_float64(make_model(n_layers=2, **V9, angular_prior={"sigma_l": 0.3}), graph)
    sc = np.asarray(model.angular_scale)
    assert sc.shape == (model.sh_irreps.dim,)
    i = 0
    for mul, ir in model.sh_irreps:
        blk = sc[i:i + mul * ir.dim]
        assert np.allclose(blk, math.exp(-0.09 * ir.l ** 2)); i += mul * ir.dim
    lmax_t = max(ir.l for _, ir in model.sh_irreps)
    assert sc[0] == 1.0 and abs(sc[-1] - math.exp(-0.09 * lmax_t ** 2)) < 1e-12 and sc[-1] < 1.0
    _check_e3(model, g, TOL)
    plain, _ = as_float64(make_model(n_layers=2, seed=0, **V9), graph)
    assert abs(_energy(plain, g) - _energy(model, g)) > 1e-9


def test_linear_radial_mlp(x64, graph):
    model, g = as_float64(make_model(n_layers=2, **V9, radial_mlp_layers=0), graph)
    for layer in model.layers:
        assert len(layer.radial_mlp.layers) == 1           # a single Linear: radial weights linear in the basis
    _check_e3(model, g, TOL)
    assert np.isfinite(_energy(model, g))
