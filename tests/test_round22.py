"""Round-22 (review §11.17): DeNS force encoding for the Noisy-Nodes denoising task."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from conftest import make_model, as_float64, random_rotation
from test_equivariance import _check_e3, TOL

V8 = dict(update="linear", readout_mode="gated", message_norm="avg", avg_n_neighbors=16.7, x_mode="basis")
DENS = dict(**V8, noise_head=True, force_encoding=True)


def _energy(model, graph):
    return float(np.asarray(model(graph)["energy"])[0])


def test_force_encoding_structure_and_e3(x64, graph):
    model, g = as_float64(make_model(n_layers=3, **DENS), graph)
    assert model.force_encoding and str(model.layers[0].input_irreps).endswith("x1o")
    assert model.layers[0].input_irreps.count("1o") == model.embed_irreps.num_irreps
    _check_e3(model, g, TOL)                       # energy pass: zero force block, equivariant as before
    plain, _ = as_float64(make_model(n_layers=3, **V8, noise_head=True), graph)
    assert not plain.force_encoding and plain.layers[0].input_irreps.count("1o") == 0


def test_energy_pass_never_sees_reference_forces(x64, graph):
    model, g = as_float64(make_model(n_layers=2, seed=3, **DENS), graph)
    e1 = _energy(model, g)
    rng = np.random.RandomState(0)
    g2 = g._replace(nodes={**g.nodes, "forces": rng.normal(size=np.asarray(g.nodes["forces"]).shape)})
    assert _energy(model, g2) == e1
    # the denoising head does see them
    d1 = np.asarray(model.denoise(g)); d2 = np.asarray(model.denoise(g2))
    assert d1.shape == (int(g.n_node[0]), 3) and not np.allclose(d1, d2)


def test_denoise_is_equivariant_with_forces(x64, graph):
    model, g = as_float64(make_model(n_layers=2, seed=5, **DENS), graph)
    R = random_rotation(3).astype(np.float64)
    pos = np.asarray(g.nodes["positions"], np.float64); frc = np.asarray(g.nodes["forces"], np.float64)
    g_rot = g._replace(nodes={**g.nodes, "positions": pos @ R.T, "forces": frc @ R.T})
    d = np.asarray(model.denoise(g)); d_rot = np.asarray(model.denoise(g_rot))
    assert np.allclose(d @ R.T, d_rot, rtol=1e-8, atol=1e-10)
    # zero forces -> same as the model without encoding information (block is zero)
    g0 = g._replace(nodes={**g.nodes, "forces": np.zeros_like(frc)})
    d0 = np.asarray(model.denoise(g0))
    assert not np.allclose(d0, d)


def test_force_enc_grads_finite(graph):
    m = make_model(n_layers=2, **DENS)
    graphdef, params = nnx.split(m, nnx.Param)

    def loss(p):
        mm = nnx.merge(graphdef, p)
        return jnp.sum(mm.denoise(graph) ** 2) + jnp.sum(mm(graph)["energy"] ** 2)

    grads = jax.grad(loss)(params)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(grads))
    assert float(jnp.max(jnp.abs(jax.tree.leaves(grads["force_enc"])[0]))) > 0.0
