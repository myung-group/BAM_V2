"""Equivalence of bam_omat24.soap_grouped (shape-grouped SOAP) vs bam_omat24.soap_optax.scale_by_soap.

Protocol (identical to tests/test_soap.py's seeded comparison): step 0 is run with the current
implementation, its state is converted to the grouped layout (exact reshapes), then both run
from the same state.  Every factor is full rank before the first refresh (freq 10), so the
Householder QR is unique up to column signs and the updates must agree to round-off.
"""
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bam_omat24.soap_optax import ScaleBySoapState, scale_by_soap          # noqa: E402
from bam_omat24.soap_grouped import (ScaleBySoapGroupedState, convert_opt_state,  # noqa: E402
                                soap_grouped_build)

SHAPES = {**{f"sq{i}": (12, 12) for i in range(9)},            # many same-shape leaves
          **{f"sp{i}": (3, 12, 12) for i in range(4)},         # species-indexed -> merges into the 12x12 group
          "tall": (30, 12), "wide": (12, 30), "col": (8, 1), "row": (1, 8),
          "hill": (5, 2, 4), "vec": (5,), "s": (), "big": (40, 300)}
KW = dict(b1=0.95, b2=0.95, precondition_frequency=10, max_precond_dim=256)   # 'big' -> left-only


def _grads(rng, step):
    g = {}
    for k, s in SHAPES.items():
        x = rng.normal(size=s)
        if len(s) >= 2:
            x = x * np.linspace(0.2, 2.0, s[-2])[:, None] * np.linspace(1.5, 0.3, s[-1])[None, :]
        g[k] = jnp.asarray(x)
    g["sp3"] = g["sp3"].at[2].set(0.0)                         # absent species slice
    return g


def _sign_free(st):
    """Rotation-invariant parts of a per-leaf state: factors, the unrotated first moment, |Q^T Q|."""
    out = {}
    for k in SHAPES:
        ql, qr, mu = st.q_left[k], st.q_right[k], st.exp_avg[k]
        if mu.ndim >= 2 and mu.size:
            x = mu
            if ql.size:
                x = jnp.matmul(ql, x)
            if qr.size:
                x = jnp.matmul(x, jnp.swapaxes(qr, -1, -2))
            out[k + "/m"] = x
        for nm, f in (("gl", st.gg_left[k]), ("gr", st.gg_right[k])):
            if f.size:
                out[k + "/" + nm] = f
    return out


@pytest.mark.parametrize("layout", ["grouped", "per_leaf"])
def test_grouped_equals_per_leaf(x64, layout):
    rng = np.random.RandomState(0)
    params = {k: jnp.zeros(s) for k, s in SHAPES.items()}
    ref = scale_by_soap(**KW)
    tx, to_g, to_p = soap_grouped_build(**KW, layout=layout)
    leaves, td = jax.tree.flatten(params)
    s_ref = ref.init(params)
    g0 = _grads(rng, 0)
    u0, s_ref = jax.jit(ref.update)(g0, s_ref)
    s_new = to_g(s_ref, leaves, td) if layout == "grouped" else s_ref
    up_ref, up_new = jax.jit(ref.update), jax.jit(tx.update)
    for step in range(1, 36):                                  # refreshes at 10, 20, 30
        g = _grads(rng, step)
        ur, s_ref = up_ref(g, s_ref)
        un, s_new = up_new(g, s_new)
        for k in SHAPES:
            np.testing.assert_allclose(np.asarray(un[k]), np.asarray(ur[k]), rtol=1e-9, atol=1e-10,
                                       err_msg=f"{layout} step {step} leaf {k}")
        if step == 18:   # checkpoint round trip
            s_new = pickle.loads(pickle.dumps(jax.device_get(s_new)))
    s_cmp = to_p(s_new, leaves, td, like=s_ref) if layout == "grouped" else s_new
    a, b = _sign_free(s_cmp), _sign_free(s_ref)
    for k in a:
        np.testing.assert_allclose(np.asarray(a[k]), np.asarray(b[k]), rtol=1e-9, atol=1e-12, err_msg=k)
    if layout == "per_leaf":
        assert jax.tree.structure(s_new) == jax.tree.structure(s_ref)


def test_first_step_and_convert_chain(x64):
    import optax
    params = {k: jnp.zeros(s) for k, s in SHAPES.items()}
    tx, _, _ = soap_grouped_build(**KW)
    st = tx.init(params)
    assert isinstance(st, ScaleBySoapGroupedState)
    u, st = jax.jit(tx.update)(_grads(np.random.RandomState(1), 0), st)
    for k, s in SHAPES.items():
        if len(s) >= 2:
            assert np.all(np.asarray(u[k]) == 0.0)
    chain_old = optax.chain(optax.clip_by_global_norm(1.0), scale_by_soap(**KW)).init(params)
    chain_new = convert_opt_state(chain_old, params, to="grouped", **KW)
    assert isinstance(chain_new[1], ScaleBySoapGroupedState)
    back = convert_opt_state(chain_new, params, to="per_leaf", **KW)
    assert jax.tree.structure(back) == jax.tree.structure(chain_old)


def test_cholqr_close_on_well_conditioned(x64):
    """orth='cholqr' (regularised sCholQR3) vs Householder: same updates while every factor is
    well conditioned (here: the first refresh only); later steps may drift at the 1e-10 level."""
    rng = np.random.RandomState(2)
    params = {k: jnp.zeros(s) for k, s in SHAPES.items()}
    ta, _, _ = soap_grouped_build(**KW)
    tb, _, _ = soap_grouped_build(**KW, orth="cholqr", cholqr_reg=1e-12)
    sa, sb = ta.init(params), tb.init(params)
    ua_f, ub_f = jax.jit(ta.update), jax.jit(tb.update)
    for step in range(12):
        g = _grads(rng, step)
        ua, sa = ua_f(g, sa)
        ub, sb = ub_f(g, sb)
        if step == 0:   # step-0 eigh is shared code: copy the bases so the null-space choice agrees
            sb = sb._replace(ql=sa.ql, qr=sa.qr)
        for k in SHAPES:
            np.testing.assert_allclose(np.asarray(ub[k]), np.asarray(ua[k]), rtol=1e-6, atol=1e-8,
                                       err_msg=f"step {step} leaf {k}")


def test_species_tables_use_adam(x64):
    """adam_leading_dim: leaves with leading dim == n_species are plain Adam; the others are unchanged SOAP."""
    import optax
    from bam_omat24.soap_grouped import scale_by_soap_grouped
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    params = {'table': jax.random.normal(k1, (5, 3, 4), jnp.float64), 'shared': jax.random.normal(k2, (6, 7), jnp.float64)}
    tx_s = scale_by_soap_grouped(b1=0.95, b2=0.95, eps=1e-8, precondition_frequency=2, adam_leading_dim=5)
    tx_ref = scale_by_soap_grouped(b1=0.95, b2=0.95, eps=1e-8, precondition_frequency=2)
    tx_adam = optax.scale_by_adam(b1=0.95, b2=0.95, eps=1e-8)
    s_s, s_r, s_a = tx_s.init(params), tx_ref.init(params), tx_adam.init(params['table'])
    for t in range(5):
        g = jax.tree.map(lambda x: jax.random.normal(jax.random.fold_in(k3, t), x.shape, x.dtype), params)
        u_s, s_s = tx_s.update(g, s_s, params)
        u_r, s_r = tx_ref.update(g, s_r, params)
        u_a, s_a = tx_adam.update(g['table'], s_a, params['table'])
        assert np.allclose(np.asarray(u_s['table']), np.asarray(u_a), rtol=1e-10, atol=1e-12)
        assert np.allclose(np.asarray(u_s['shared']), np.asarray(u_r['shared']), rtol=1e-10, atol=1e-12)


def test_species_adam_any_axis_keeps_absent_species(x64):
    """adam_any_axis: a table with the species on a trailing axis, shape (3, 2, n_species), goes to Adam too, and the
    entries of species that never get a gradient stay exactly at their initial value; with the leading-dim rule the
    same leaf is rotated by SOAP and they drift (the reason for the option)."""
    from bam_omat24.soap_grouped import scale_by_soap_grouped
    n_sp, present = 9, np.array([0, 3, 4, 7])
    key = jax.random.PRNGKey(1)
    params = {'hill': jnp.zeros((3, 2, n_sp), jnp.float64), 'shared': jnp.zeros((6, 7), jnp.float64)}
    mask = np.zeros(n_sp); mask[present] = 1.0
    drift = {}
    for name, any_axis in (('leading', False), ('any', True)):
        tx = scale_by_soap_grouped(b1=0.95, b2=0.95, eps=1e-8, precondition_frequency=2, adam_leading_dim=n_sp,
                                   adam_any_axis=any_axis)
        st, tot = tx.init(params), np.zeros((3, 2, n_sp))
        for t in range(12):
            g = jax.tree.map(lambda x: jax.random.normal(jax.random.fold_in(key, t), x.shape, x.dtype), params)
            g = {**g, 'hill': g['hill'] * mask}
            u, st = tx.update(g, st, params)
            tot += np.asarray(u['hill'])
        drift[name] = np.abs(tot[..., mask == 0]).max()
        assert np.abs(tot[..., present]).max() > 0.1
    assert drift['any'] == 0.0 and drift['leading'] > 1e-3
