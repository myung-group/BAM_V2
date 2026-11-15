"""Round-10 options: LayerScale, stochastic depth, noise head, hyperbolic-cross truncation,
per-layer cutoff, species-pair radial transform, and the new loss terms."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from conftest import make_model, as_float64, random_rotation, pad_single
from test_equivariance import _check_e3, TOL
from race_v2.training.utils import compute_loss, _loss_terms


def _energy(model, graph):
    return float(np.asarray(model(graph)["energy"])[0])


@pytest.mark.parametrize("kw", [
    dict(residual="identity", layer_scale=0.1),
    dict(layer_lmax=[2, 1, 1], layer_cutoff=[5.0, 4.0, 3.0]),
    dict(radial_species_scale=True, radial_slater=True, species_radii=[0.76, 0.31, 0.71, 0.66]),
    dict(residual="identity", layer_scale=0.1, layer_lmax=[2, 2, 1], layer_cutoff=[5.0, 4.5, 4.0],
         radial_species_scale=True, species_radii=[0.76, 0.31, 0.71, 0.66]),
])
def test_e3_round10_options(x64, graph, kw):
    model, g = as_float64(make_model(n_layers=3, **kw), graph)
    _check_e3(model, g, TOL)


def test_forces_fd_round10(x64, graph):
    model, graph = as_float64(make_model(
        n_layers=3, residual="identity", layer_scale=0.1, layer_lmax=[2, 2, 1],
        layer_cutoff=[5.0, 4.5, 4.0], radial_species_scale=True, radial_slater=True,
        species_radii=[0.76, 0.31, 0.71, 0.66]), graph)
    forces = np.asarray(model(graph)["forces"])
    pos = np.asarray(graph.nodes["positions"], dtype=np.float64)

    def energy(p):
        g = graph._replace(nodes={**graph.nodes, "positions": p.astype(np.float64)})
        return float(np.asarray(model(g)["energy"])[0])

    cands = [(i, k) for i in range(pos.shape[0]) for k in range(3) if abs(forces[i, k]) > 0.05]
    rng = np.random.RandomState(2)
    picks = [cands[j] for j in rng.choice(len(cands), size=2, replace=False)]
    h = 1e-4
    for i, k in picks:
        pp, pm = pos.copy(), pos.copy(); pp[i, k] += h; pm[i, k] -= h
        fd = -(energy(pp) - energy(pm)) / (2 * h)
        assert abs(fd - forces[i, k]) / max(abs(forces[i, k]), 1e-6) < 1e-3, (fd, forces[i, k])


def test_layer_cutoff_shorter_than_model_changes_energy(graph):
    m_full = make_model(n_layers=3, seed=5)
    m_cut = make_model(n_layers=3, seed=5, layer_cutoff=[5.0, 3.0, 3.0])
    assert abs(_energy(m_full, graph) - _energy(m_cut, graph)) > 1e-6


def test_layer_lmax_truncates_new_information(graph):
    """With msg_lmax=0 for all layers the message TP keeps only 0e: l>0 features can only
    come from the skip term; with 0e-only input (layer 0) they are exactly zero."""
    import e3nn_jax as e3nn
    m = make_model(n_layers=1, layer_lmax=[0])
    layer = m.layers[0]
    assert layer.msg_irreps == e3nn.Irreps("16x0e")
    _ = m(graph)  # runs


def test_stochastic_depth_mask(graph):
    """stratum_keep: ones == no mask; zero column t -> stratum t contributes nothing and the
    residual stream skips the layer."""
    m = make_model(n_layers=3, residual="identity", seed=7)
    e_ref = _energy(m, graph)
    n_g = graph.n_node.shape[0]
    g1 = graph._replace(globals={**graph.globals, "stratum_keep": np.ones((n_g, 3), np.float32)})
    assert abs(_energy(m, g1) - e_ref) < 1e-6
    keep = np.ones((n_g, 3), np.float32); keep[:, 2] = 0.0
    g2 = graph._replace(globals={**graph.globals, "stratum_keep": keep})
    out = m.node_energies(g2.nodes["positions"], None, None, g2)
    strata = np.asarray(out["strata"])
    assert np.allclose(strata[2], 0.0)
    assert not np.allclose(strata[1], 0.0)
    # dropping layer 2 of 3 with identity residual == 2-layer model energy? no (readouts differ),
    # but the energy must change versus the unmasked model
    assert abs(_energy(m, g2) - e_ref) > 1e-8


def test_noise_head_is_equivariant(graph):
    m = make_model(n_layers=2, noise_head=True)
    v0 = np.asarray(m.denoise(graph))
    R = random_rotation(3)
    g2 = graph._replace(nodes={**graph.nodes, "positions": np.asarray(graph.nodes["positions"]) @ R.T})
    v2 = np.asarray(m.denoise(g2))
    assert v0.shape == (graph.nodes["positions"].shape[0], 3)
    assert np.abs(v0 @ R.T - v2).max() < 1e-3 * max(1.0, np.abs(v0).max())


def test_extra_loss_terms_run_and_are_finite(graph):
    m = make_model(n_layers=3, noise_head=True)
    graphdef, params = nnx.split(m, nnx.Param)
    rng = np.random.RandomState(0)
    graph = pad_single(graph)
    g = graph._replace(nodes={**graph.nodes,
                              "noise": (0.05 * rng.normal(size=graph.nodes["positions"].shape)).astype(np.float32)})
    loss, aux = compute_loss(graphdef, params, g, 1.0, 10.0, 0.0, 10.0, periodic=False,
                             deep_supervision=0.1, stratum_decorrelation=1e-4, noisy_nodes=(0.05, 1e-4))
    assert np.isfinite(float(loss))
    for k in ("ds_loss", "decor_loss", "denoise_loss"):
        assert np.isfinite(float(aux[k])) and float(aux[k]) > 0.0, k
    # decorrelation is a sum of squared correlations: bounded by S(S-1)/2 = 3 for S = 3
    assert float(aux["decor_loss"]) <= 3.0 + 1e-6
    loss0, aux0 = compute_loss(graphdef, params, graph, 1.0, 10.0, 0.0, 10.0, periodic=False)
    assert abs(float(loss0) - (float(aux0["energy_loss"]) + 10.0 * float(aux0["force_loss"]))) < 1e-6 * max(1.0, abs(float(loss0)))


def test_loss_weight_ones_equals_unweighted(graph):
    m = make_model(n_layers=2)
    graph = pad_single(graph)
    res = m(graph)
    t0 = _loss_terms(res, graph, 10.0, "huber", False, "scalar6", 0.0)
    n_g = graph.n_node.shape[0]
    g1 = graph._replace(globals={**graph.globals, "loss_weight": np.ones((n_g, 1), np.float32)})
    t1 = _loss_terms(res, g1, 10.0, "huber", False, "scalar6", 0.0)
    assert abs(float(t0["energy_loss"]) - float(t1["energy_loss"])) < 1e-9
    assert abs(float(t0["force_loss"]) - float(t1["force_loss"])) < 1e-9
    g2 = graph._replace(globals={**graph.globals, "loss_weight": 2.0 * np.ones((n_g, 1), np.float32)})
    t2 = _loss_terms(res, g2, 10.0, "huber", False, "scalar6", 0.0)
    assert abs(float(t2["force_loss"]) - 2.0 * float(t0["force_loss"])) < 1e-7


def test_species_radial_grads_finite_with_padding(graph):
    """Padded edges have r = 0; the parameter gradients of rho / zeta must stay finite
    (the r10_rsp run went NaN at step 1 because of inf * 0 in the Bessel 1/x cotangent)."""
    m = make_model(n_layers=2, radial_species_scale=True, radial_slater=True,
                   species_radii=[0.76, 0.31, 0.71, 0.66])
    graphdef, params = nnx.split(m, nnx.Param)
    g = pad_single(graph)

    def loss(p):
        l, _ = compute_loss(graphdef, p, g, 1.0, 10.0, 0.0, 10.0, periodic=False)
        return l

    val, grads = jax.value_and_grad(loss)(params)
    assert np.isfinite(float(val))
    leaves = jax.tree.leaves(grads)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in leaves)
    assert bool(jnp.any(jnp.abs(jnp.asarray(grads["species_rho"].value if hasattr(grads["species_rho"], "value") else grads["species_rho"])) > 0))
