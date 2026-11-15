"""Round-15 options (review §11): linear body-ordered update (update="linear"), message
normalisation without layer norm (message_norm="avg"), x_mode="basis" (channelwise product with the
layer-0 atomic basis) and the per-structure energy loss (energy_loss_per="structure")."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from conftest import make_model, as_float64, pad_single
from test_equivariance import _check_e3, TOL
from test_round11_shells import SHELLS
from race_v2.training.utils import compute_loss

A = dict(update="linear", readout_mode="gated")
AB = dict(update="linear", readout_mode="gated", message_norm="avg", avg_n_neighbors=16.7)
ABC = dict(update="linear", readout_mode="gated", message_norm="avg", avg_n_neighbors=16.7, x_mode="basis")
SHELL = dict(radial_shells=True, species_shell_radii=SHELLS)


def _energy(model, graph):
    return float(np.asarray(model(graph)["energy"])[0])


@pytest.mark.parametrize("kw", [A, AB, ABC, dict(**ABC, **SHELL), dict(x_mode="basis")])
def test_e3_round15_options(x64, graph, kw):
    model, g = as_float64(make_model(n_layers=3, **kw), graph)
    _check_e3(model, g, TOL)


@pytest.mark.parametrize("kw", [A, AB, ABC])
def test_forces_fd_round15(x64, graph, kw):
    model, graph = as_float64(make_model(n_layers=3, **kw), graph)
    forces = np.asarray(model(graph)["forces"])
    pos = np.asarray(graph.nodes["positions"], dtype=np.float64)

    def energy(p):
        g = graph._replace(nodes={**graph.nodes, "positions": p.astype(np.float64)})
        return float(np.asarray(model(g)["energy"])[0])

    # the two largest force components (an untrained model without layer norm has small forces)
    order = np.argsort(-np.abs(forces).ravel())[:2]
    picks = [(int(o) // 3, int(o) % 3) for o in order]
    assert abs(forces[picks[1]]) > 1e-4, forces[picks[1]]
    h = 1e-4
    for i, k in picks:
        pp, pm = pos.copy(), pos.copy(); pp[i, k] += h; pm[i, k] -= h
        fd = -(energy(pp) - energy(pm)) / (2 * h)
        assert abs(fd - forces[i, k]) / max(abs(forces[i, k]), 1e-6) < 1e-3, (fd, forces[i, k])


def test_linear_update_requires_gated_readout():
    with pytest.raises(ValueError):
        make_model(n_layers=2, update="linear")


def test_linear_update_features_are_linear_in_message(x64, graph):
    """With update='linear' and message_norm='avg' the layer output is linear in the radial-MLP
    output: scaling the radial weights of layer 0 by c scales (h1 - skip) by c exactly."""
    model, g = as_float64(make_model(n_layers=2, seed=3, **AB), graph)
    layer = model.layers[0]
    # species embedding (layer-0 input) and the plain edge features
    from race_v2.models.blocks import compute_node_edge_features
    feats, rb, sh = compute_node_edge_features(
        g.nodes["positions"], None, None, g, model.n_species, model.cutoff,
        model.radial_basis_size, model.radial_polynomial_p, model.lmax, periodic=False)
    h0 = model.species_embedding(feats)
    x = model.x_linear(g.nodes["species"], h0)
    _, h1 = layer(h0, x, g.nodes["species"], sh, rb, g.senders, g.receivers)
    skip = layer.skip(g.nodes["species"], h0)
    skip = np.asarray(jnp.pad(skip.array, ((0, 0), (0, h1.array.shape[1] - skip.array.shape[1])))) \
        if skip.array.shape[1] != h1.array.shape[1] else np.asarray(skip.array)
    # scale the last radial-MLP layer by 2 -> message doubles
    w = layer.radial_mlp.layers[-1].weights
    w.value = 2.0 * w.value
    _, h1b = layer(h0, x, g.nodes["species"], sh, rb, g.senders, g.receivers)
    d1 = np.asarray(h1.array) - skip
    d2 = np.asarray(h1b.array) - skip
    assert np.allclose(d2, 2.0 * d1, rtol=1e-8, atol=1e-10)


def test_gate_update_is_not_linear(x64, graph):
    model, g = as_float64(make_model(n_layers=2, seed=3, message_norm="avg", avg_n_neighbors=16.7), graph)
    layer = model.layers[0]
    from race_v2.models.blocks import compute_node_edge_features
    feats, rb, sh = compute_node_edge_features(
        g.nodes["positions"], None, None, g, model.n_species, model.cutoff,
        model.radial_basis_size, model.radial_polynomial_p, model.lmax, periodic=False)
    h0 = model.species_embedding(feats)
    x = model.x_linear(g.nodes["species"], h0)
    _, h1 = layer(h0, x, g.nodes["species"], sh, rb, g.senders, g.receivers)
    w = layer.radial_mlp.layers[-1].weights
    w.value = 2.0 * w.value
    _, h1b = layer(h0, x, g.nodes["species"], sh, rb, g.senders, g.receivers)
    assert not np.allclose(np.asarray(h1b.array), 2.0 * np.asarray(h1.array), rtol=1e-3)


def test_basis_mode_uses_layer0_basis(graph):
    """x_mode='basis': the x projection of layer 0 reads the aggregated layer-0 message; zeroing the
    l>0 part of the x projection changes the energy (the equivariant product is used)."""
    base = make_model(n_layers=2, seed=7, x_mode="species")
    m = make_model(n_layers=2, seed=7, x_mode="basis")
    assert m.x_projs[0].irreps_in == m.hidden_irreps
    e = _energy(m, graph)
    assert np.isfinite(e) and abs(e - _energy(base, graph)) > 1e-6
    n_zeroed = 0
    for orig, attr in m.x_projs[1]._weight_name_mapping.items():
        if not orig.split(",")[-1].strip().endswith("0e"):          # output irrep of the path is l>0
            p = getattr(m.x_projs[1], attr)
            p.value = jnp.zeros_like(p.value)
            n_zeroed += 1
    assert n_zeroed == 2                                             # 1o and 2e paths (lmax 2 in make_model)
    assert abs(_energy(m, graph) - e) > 1e-7


def test_energy_loss_per_structure(graph):
    m = make_model(n_layers=2, seed=1)
    graphdef, params = nnx.split(m, nnx.Param)
    # target = prediction - 0.05 eV: a residual inside the tanh clip (5 eV) of the loss
    pred = float(np.asarray(m(graph)["energy"])[0])
    graph = graph._replace(globals={**graph.globals, "energy": np.array([pred - 0.05], np.float32)})
    g = pad_single(graph)
    n = int(graph.n_node[0])
    _, aux_a = compute_loss(graphdef, params, g, 1.0, 0.0, 0.0, 1e6, periodic=False, energy_loss_per="atom")
    _, aux_s = compute_loss(graphdef, params, g, 1.0, 0.0, 0.0, 1e6, periodic=False, energy_loss_per="structure")
    ea, es = float(aux_a["energy_loss"]), float(aux_s["energy_loss"])
    assert ea > 0 and abs(es - n ** 2 * ea) / es < 1e-4, (ea, es, n)
    # metrics unchanged
    assert abs(float(aux_a["energy_mse"]) - float(aux_s["energy_mse"])) < 1e-12


def test_round15_grads_finite_with_padding(graph):
    m = make_model(n_layers=3, **ABC, **SHELL)
    graphdef, params = nnx.split(m, nnx.Param)
    g = pad_single(graph)

    def loss(p):
        l, _ = compute_loss(graphdef, p, g, 1.0, 10.0, 0.0, 10.0, periodic=False)
        return l

    val, grads = jax.value_and_grad(loss)(params)
    assert np.isfinite(float(val))
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(grads))
