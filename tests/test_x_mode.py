"""x_mode option: 'node_scalar' runs with identical output shapes; 'species' x_feats
depend on the element only."""
import jax
import jax.numpy as jnp
import numpy as np
import e3nn_jax as e3nn
from flax import nnx

from conftest import make_model, pad_single


def test_node_scalar_forward_shapes(graph):
    g = pad_single(graph)
    out_s = make_model(x_mode="species")(g)
    out_n = make_model(x_mode="node_scalar")(g)
    for k in ("energy", "forces", "force_variation", "energy_variation"):
        assert out_s[k].shape == out_n[k].shape, k
        assert np.all(np.isfinite(np.asarray(out_n[k]))), k
    # node_scalar mode has extra per-layer projections -> more parameters
    n_s = sum(x.size for x in jax.tree.leaves(nnx.split(make_model(x_mode="species"), nnx.Param)[1]))
    n_n = sum(x.size for x in jax.tree.leaves(nnx.split(make_model(x_mode="node_scalar"), nnx.Param)[1]))
    assert n_n > n_s


def test_species_x_feats_depend_only_on_element(graph):
    model = make_model(x_mode="species")
    species = jnp.asarray(graph.nodes["species"])
    onehot = e3nn.IrrepsArray(e3nn.Irreps(f"{model.n_species}x0e"),
                              jax.nn.one_hot(species, model.n_species))
    x = np.asarray(model.x_linear(species, model.species_embedding(onehot)).array)[:, 0]
    sp = np.asarray(species)
    for s in np.unique(sp):
        vals = x[sp == s]
        assert np.allclose(vals, vals[0], atol=1e-6), (s, vals)
    # ... and the values differ between (at least some) elements
    assert len(np.unique(np.round(x, 6))) == len(np.unique(sp))


def test_node_scalar_x_feats_vary_within_element(graph):
    """In node_scalar mode x_feats are geometry dependent: atoms of the same element
    get different values (checked via the per-layer projection of layer 1)."""
    model = make_model(x_mode="node_scalar")
    g = pad_single(graph)
    # run node_energies manually up to layer-1 input features
    from bam_omat24.utils import compute_node_edge_features, separated_layer_norm
    feats, rb, sh = compute_node_edge_features(
        g.nodes["positions"], None, None, g, model.n_species, model.cutoff,
        model.radial_basis_size, model.radial_polynomial_p, model.lmax, False)
    feats = model.species_embedding(feats)
    x0 = model.x_projs[0](separated_layer_norm(feats).filter(keep="0e"))
    _, feats1 = model.layers[0](feats, x0, g.nodes["species"], sh, rb, g.senders, g.receivers)
    x1 = np.asarray(model.x_projs[1](separated_layer_norm(feats1).filter(keep="0e")).array)[:, 0]
    sp = np.asarray(g.nodes["species"])
    mask = np.asarray(jax.numpy.arange(len(sp)) < int(graph.n_node[0]))
    hydrogens = x1[mask & (sp == 0)]
    assert hydrogens.std() > 1e-4
