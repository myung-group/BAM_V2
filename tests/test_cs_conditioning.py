"""RACE_LR charge / spin conditioning options of 2026-11-13 (long_range cs_encoding "table", cs_layers "all",
cs_species_gate): the model starts as the model without them, the new parameters make the energy depend on the
spin multiplicity, and energy / forces stay consistent and rotation invariant.  Run in a Slurm job (JAX)."""
import numpy as np

from conftest import pad_single, random_rotation, as_float64
from test_race_lr import _lr_model  # noqa: E402

NEW = {'charge_spin_embedding': True, 'cs_encoding': 'table', 'cs_layers': 'all', 'cs_species_gate': True}


def _run(model, g, Q=0.0, S=1.0):
    import jax
    import jax.numpy as jnp
    g = g._replace(globals={**g.globals, 'charge': np.array([Q, 0.0]), 'spin': np.array([S, 1.0])})
    return model(jax.tree.map(jnp.asarray, g))


def test_cs_options_start_as_the_original_model(x64, graph):
    old, g = as_float64(_lr_model(lr={'charge_spin_embedding': True}), pad_single(graph))
    new, _ = as_float64(_lr_model(lr=NEW), pad_single(graph))
    for Q, S in ((0.0, 1.0), (1.0, 2.0), (-2.0, 3.0)):
        a, b = _run(old, g, Q, S), _run(new, g, Q, S)
        assert abs(float(a['energy'][0]) - float(b['energy'][0])) < 1e-10
        assert np.allclose(np.asarray(a['forces']), np.asarray(b['forces']), atol=1e-10)


def test_cs_options_condition_on_the_spin(x64, graph):
    import jax.numpy as jnp
    model, g = as_float64(_lr_model(lr=NEW), pad_single(graph))
    rng = np.random.default_rng(0)
    for name, scale in (('cs_s_table', 1.0), ('cs_q_table', 1.0), ('cs_layer_w', 0.3), ('cs_gate_table', 0.3)):
        p = getattr(model, name)
        p.value = jnp.asarray(scale * rng.normal(size=p.value.shape), p.value.dtype)
    # layer 0 projection off: what remains is the per-layer path and the tables
    r1, r3, r1q = _run(model, g, 0.0, 1.0), _run(model, g, 0.0, 3.0), _run(model, g, 1.0, 1.0)
    assert abs(float(r3['energy'][0]) - float(r1['energy'][0])) > 1e-4          # spin dependence
    assert abs(float(r1q['energy'][0]) - float(r1['energy'][0])) > 1e-4         # charge dependence
    # multiplicities above cs_max_spin share the last row, the scalar input still tells them apart
    assert np.isfinite(float(_run(model, g, 9.0, 15.0)['energy'][0]))
    # forces = -dE/dx
    h = 1e-5
    p0 = np.asarray(g.nodes['positions']).copy()
    for i, c in [(2, 1), (9, 0)]:
        pp, pm = p0.copy(), p0.copy(); pp[i, c] += h; pm[i, c] -= h
        ep = float(_run(model, g._replace(nodes={**g.nodes, 'positions': pp}), 0.0, 3.0)['energy'][0])
        em = float(_run(model, g._replace(nodes={**g.nodes, 'positions': pm}), 0.0, 3.0)['energy'][0])
        assert abs(-(ep - em) / (2 * h) - float(r3['forces'][i, c])) < 1e-6
    # rotation invariance
    u, _, vt = np.linalg.svd(random_rotation(3).astype(np.float64))
    rot = _run(model, g._replace(nodes={**g.nodes, 'positions': p0 @ (u @ vt).T}), 0.0, 3.0)
    assert abs(float(rot['energy'][0]) - float(r3['energy'][0])) < 1e-9


def test_cs_options_need_the_embedding():
    import pytest
    with pytest.raises(ValueError):
        _lr_model(lr={'cs_layers': 'all'})
