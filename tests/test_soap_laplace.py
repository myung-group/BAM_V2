"""SOAP-curvature Laplace sampler (2026-11-09): the per-leaf state of a short SOAP run gives samples that collapse
onto the centre as kappa, lam -> infinity, have the leaves' shapes and dtypes, and whose spread follows the SOAP
basis: a direction with a large second moment nu moves less than one with a small nu."""
import numpy as np
import jax
import jax.numpy as jnp

from bam_omat24.soap_optax import scale_by_soap
from bam_omat24.soap_laplace import sample_params, leaf_specs


def _run_soap(n_steps=12, seed=0):
    rng = np.random.default_rng(seed)
    params = {'w': jnp.asarray(rng.normal(size=(6, 4))), 'v': jnp.asarray(rng.normal(size=(3,))),
              'i': jnp.asarray(np.arange(5))}
    tx = scale_by_soap(b1=0.9, b2=0.9, precondition_frequency=2)
    st = tx.init(params)
    for _ in range(n_steps):
        g = {'w': jnp.asarray(rng.normal(size=(6, 4)) * np.array([[10.0, 1, 1, 1]] * 6)),
             'v': jnp.asarray(rng.normal(size=(3,))), 'i': jnp.zeros(5, params['i'].dtype)}
        _, st = tx.update(g, st, params)
    return params, st


def test_sampler_limits_and_shapes():
    params, st = _run_soap()
    leaves, treedef = jax.tree.flatten(params)
    cols = [list(treedef.flatten_up_to(x)) for x in st[1:]]
    cfg = {'soap_max_precond_dim': 10000, 'atomic_numbers': [1, 6]}
    specs = leaf_specs(cfg, leaves)
    assert [sp.kind for sp in specs] == ['skip', 'adam', 'soap']        # i, v, w (flatten order)
    rng = np.random.default_rng(0)
    same = sample_params(leaves, specs, cols, 1e300, 1e300, rng)
    for a, b in zip(same, leaves):
        assert a.shape == np.shape(b) and a.dtype == np.asarray(b).dtype
        assert np.array_equal(a, np.asarray(b))
    draws = np.stack([sample_params(leaves, specs, cols, 1.0, 1e-6, np.random.default_rng(s))[2] for s in range(400)])
    assert np.isfinite(draws).all()
    # spread in the SOAP basis: rotate the deviations back with Q_L, Q_R and compare with 1 / sqrt(kappa nu + lam)
    ql, qr, nu = np.asarray(cols[4][2]), np.asarray(cols[5][2]), np.asarray(cols[1][2])
    dev = draws - np.asarray(leaves[2])
    rot = np.einsum('ab,sbc,cd->sad', ql.T, dev, qr)
    emp = rot.std(0)
    assert np.allclose(emp, 1 / np.sqrt(nu + 1e-6), rtol=0.25), (emp, 1 / np.sqrt(nu + 1e-6))
    assert nu.max() / nu.min() > 10          # the gradient column scale made the spectrum anisotropic
