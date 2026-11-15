"""SOAP-curvature Laplace sampler (2026-11-09): the per-leaf state of a short SOAP run gives samples that collapse
onto the centre as kappa, lam -> infinity, have the leaves' shapes and dtypes, and whose spread follows the SOAP
basis: a direction with a large second moment nu moves less than one with a small nu."""
import numpy as np
import jax
import jax.numpy as jnp

from race_v2.training.soap_optax import scale_by_soap
from race_v2.eval.soap_laplace import sample_params, leaf_specs


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


def test_ggn_diag_exact_and_hutchinson():
    """GGN diagonal in a rotated basis: the exact mode equals the dense Jacobian result, Hutchinson converges to it."""
    from race_v2.eval.soap_laplace import ggn_diag
    rng = np.random.default_rng(1)
    params = {'w': jnp.asarray(rng.normal(size=(3, 4))), 'v': jnp.asarray(rng.normal(size=(4,)))}
    leaves, treedef = jax.tree.flatten(params)             # v (adam), w (soap)
    specs = leaf_specs({'soap_max_precond_dim': 10000, 'atomic_numbers': [1]}, leaves)
    assert [sp.kind for sp in specs] == ['adam', 'soap']
    ql, _ = np.linalg.qr(rng.normal(size=(3, 3))); qr_, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    cols = [None, [np.zeros(4), np.zeros((3, 4))], None, None, [None, jnp.asarray(ql)], [None, jnp.asarray(qr_)]]
    n, n_pad = 5, 7

    def out_fn(p, x):
        mask = jnp.arange(n_pad) < n
        def e(pos):
            h = jnp.tanh(pos @ p['w'])
            return jnp.sum(jnp.where(mask, h @ p['v'], 0.0))
        en, g = jax.value_and_grad(e)(x)
        return en, -g, mask
    batches = [jnp.asarray(rng.normal(size=(n_pad, 3))) for _ in range(3)]
    w_e, w_f = 1 / 0.3 ** 2, 1 / 0.05 ** 2
    # dense reference: Jacobian of (E, F[:n]) wrt the leaves, rotated into the SOAP basis, squared, weighted
    ref_w = np.zeros((3, 4)); ref_v = np.zeros(4)
    for x in batches:
        J = jax.jacrev(lambda p: jnp.concatenate([out_fn(p, x)[0][None], out_fn(p, x)[1][:n].reshape(-1)]))(params)
        wts = np.concatenate([[w_e], np.full(3 * n, w_f)])
        Jw = np.einsum('ab,kbc,cd->kad', ql.T, np.asarray(J['w']), qr_)
        ref_w += np.einsum('k,kad->ad', wts, Jw ** 2); ref_v += wts @ np.asarray(J['v']) ** 2
    ref_w /= len(batches); ref_v /= len(batches)
    nu_s, nu_a = ggn_diag(out_fn, params, batches, specs, cols, treedef, w_e, w_f, probes=0, chunk=4)
    assert np.allclose(nu_s[0], ref_w, rtol=1e-4) and np.allclose(nu_a[0], ref_v, rtol=1e-4)      # float32 model
    nu_s, nu_a = ggn_diag(out_fn, params, batches, specs, cols, treedef, w_e, w_f, probes=3000, seed=3)
    assert np.allclose(nu_s[0], ref_w, rtol=0.15) and np.allclose(nu_a[0], ref_v, rtol=0.15), (nu_s[0] / ref_w)
