"""Shape-grouped SOAP (review §13, docs/RACE_SOAP_speedup_plan_2026-10-29.md).

The algorithm of ``bam_omat24/soap_optax.py`` (same maths, same Householder QR / eigh unless
``orth='cholqr'``) with every same-shape matrix (and every species slice of a 3-D leaf) stacked
into one batched array, so one optimizer step issues O(#groups) instead of O(#leaves) kernels.
The per-leaf ``scale_by_soap`` stays the reference implementation (tests/test_soap_grouped.py).

Differences from the per-leaf implementation (all exact in exact arithmetic):
  * stacking: batched gemm / QR instead of per-leaf calls (round-off level only; QR column
    signs may differ, which cancels in Q_L [.] Q_R^T);
  * a side of length 1 is not rotated (``drop_unit``): its "basis" is +-1, the sign cancels
    (its 1x1 factor is not stored);
  * one ``lax.switch`` per step (regular / first-eigh / QR refresh) instead of two
    ``lax.cond``; the Adam step is computed every step and masked to 0 at t == 0;
  * the refresh computes F @ Q once and permutes its columns (the per-leaf code computes
    F @ Q for diag(Q^T F Q) and again F @ Q[:, perm]);  F (Q P) = (F Q) P;
  * QR dispatch: batched ``jnp.linalg.qr`` for k <= ``batched_qr_max_k`` (batched path, 11x
    faster at k = 128, B = 361), unrolled per-matrix calls above it (the batched path is
    13-48x SLOWER at k = 512..896 in jax 0.12 on GH200).
Optional, NOT exact:
  * ``orth='cholqr'``: regularised shifted CholeskyQR3 of (F/tr F + c I) Q_perm (same
    eigenbasis fixed point, differs only in directions with eigenvalue < c tr F);
  * ``axis_name``: inside shard_map, each device orthonormalises 1/D of every factor group
    and the bases are all-gathered (each QR is computed once, on one device).

State layouts: ``layout='grouped'`` (default; ``ScaleBySoapGroupedState``, keyed by shape
group) or ``'per_leaf'`` (the ``ScaleBySoapState`` of ``scale_by_soap``, stacked / split every
step); ``convert_opt_state`` converts optax chain states between the two exactly.
"""
from __future__ import annotations

import math
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax
from jax import lax

from bam_omat24.soap_optax import ScaleBySoapState, _Spec, _spec, _mm, _t


class ScaleBySoapGroupedState(NamedTuple):
    count: jax.Array
    adam_mu: Any        # params-structured; (0,) for non-Adam leaves
    adam_nu: Any
    mu: dict            # group key -> (B, m, n)
    nu: dict
    gl: dict            # group key -> (B, m, m) or (0,)
    gr: dict
    ql: dict
    qr: dict


class _Mem(NamedTuple):
    leaf: int
    shape: tuple        # leaf shape
    start: int          # offset in the group batch
    count: int          # number of slices (prod of leading dims)


class _Grp(NamedTuple):
    key: str
    m: int
    n: int
    left: bool
    right: bool
    dtype: Any
    members: tuple
    B: int


def make_plan(shapes, dtypes, max_dim=10000, pre1d=False, drop_unit=True, adam_leading_dim=None):
    """Static grouping from leaf shapes / dtypes (deterministic: first-occurrence order).
    adam_leading_dim: leaves with ndim >= 2 whose LEADING dimension equals this value (the species-indexed tables
    of RACE, shape (n_species, ...)) are updated by plain Adam instead of SOAP (2026-11-06: 89 % of the 89-element
    model; their eigenbasis refresh dominated the SOAP step time and rare-element slices get almost no gradient)."""
    specs = [_spec(s, d, max_dim, pre1d) for s, d in zip(shapes, dtypes)]
    if adam_leading_dim is not None:
        specs = [_Spec('adam') if (sp.kind == 'soap' and len(s) >= 2 and int(s[0]) == int(adam_leading_dim)) else sp
                 for sp, s in zip(specs, shapes)]
    acc = {}
    for i, sp in enumerate(specs):
        if sp.kind != 'soap':
            continue
        lead, (m, n) = sp.mat_shape[:-2], sp.mat_shape[-2:]
        left = sp.left and not (drop_unit and m == 1)
        right = sp.right and not (drop_unit and n == 1)
        dt = jnp.dtype(sp.dtype)
        key = f"{m}x{n}_{'L' if left else '-'}{'R' if right else '-'}_{dt.name}"
        acc.setdefault(key, [m, n, left, right, dt, []])[5].append((i, tuple(int(x) for x in shapes[i]),
                                                                    math.prod(lead)))
    groups = []
    for key, (m, n, left, right, dt, mem) in acc.items():
        start, mems = 0, []
        for i, shape, c in mem:
            mems.append(_Mem(i, shape, start, c))
            start += c
        groups.append(_Grp(key, m, n, left, right, dt, tuple(mems), start))
    return specs, tuple(groups)


def _stack(xs, g, dt):
    parts = [jnp.asarray(xs[mb.leaf]).astype(dt).reshape((mb.count, g.m, g.n)) for mb in g.members]
    return parts[0] if len(parts) == 1 else jnp.concatenate(parts, 0)


def _stack_fac(xs, g, k):
    parts = [xs[mb.leaf].reshape((mb.count, k, k)) for mb in g.members]
    return parts[0] if len(parts) == 1 else jnp.concatenate(parts, 0)


def _proj(x, ql, qr, g):
    if g.left:
        x = _mm(_t(ql), x)
    if g.right:
        x = _mm(x, qr)
    return x


def _proj_back(x, ql, qr, g):
    if g.left:
        x = _mm(ql, x)
    if g.right:
        x = _mm(x, _t(qr))
    return x


# ----------------------------------------------------------------------------- orthonormalisation
def _householder(Y, batched_max_k):
    B, k = Y.shape[0], Y.shape[-1]
    if B == 1:
        return jnp.linalg.qr(Y[0])[0][None]
    if k <= batched_max_k:
        return jnp.linalg.qr(Y)[0]
    return jnp.stack([jnp.linalg.qr(Y[b])[0] for b in range(B)])     # unrolled, NOT lax.map (while = sync/iter)


def _cholqr(Y, shift=False):
    G = _mm(_t(Y), Y)
    if shift:   # Fukaya et al. 2020 shifted CholeskyQR
        k = Y.shape[-1]
        s = 11.0 * (k * k + k * (k + 1)) * jnp.finfo(Y.dtype).eps * jnp.sum(Y * Y, axis=(-2, -1), keepdims=True)
        G = G + s * jnp.eye(k, dtype=Y.dtype)
    L = jnp.linalg.cholesky(G)
    return _t(lax.linalg.triangular_solve(L, _t(Y), left_side=True, lower=True))    # Y L^{-T}


def _orth(Y, Qp, F, mode, batched_max_k, reg):
    if mode == 'householder':
        return _householder(Y, batched_max_k)
    # regularised shifted CholeskyQR3 of (F / tr F + reg I) Q_perm  (== Y / tr F + reg Q_perm)
    tr = jnp.trace(F, axis1=-2, axis2=-1)[:, None, None]
    Yr = jnp.where(tr > 0, Y / jnp.where(tr > 0, tr, 1.0), 0.0) + reg * Qp
    Q = _cholqr(_cholqr(_cholqr(Yr, shift=True)))
    ok = jnp.all(jnp.isfinite(Q), axis=(-2, -1), keepdims=True)
    return jnp.where(ok, Q, Qp)                          # failure -> keep the (permuted) old basis


def _orth_partitioned(Y, Qp, F, axis_name, D, **kw):
    """Each device orthonormalises a contiguous 1/D chunk; identity padding is QR/Cholesky safe."""
    B, k = Y.shape[0], Y.shape[-1]
    per = -(-B // D)
    pad = per * D - B
    if pad:
        eye = jnp.broadcast_to(jnp.eye(k, dtype=Y.dtype), (pad, k, k))
        Y, Qp = jnp.concatenate([Y, eye], 0), jnp.concatenate([Qp, eye], 0)
        F = jnp.concatenate([F, eye], 0)
    d = lax.axis_index(axis_name)
    sl = lambda a: lax.dynamic_slice_in_dim(a, d * per, per, axis=0)
    Qm = _orth(sl(Y), sl(Qp), sl(F), **kw)
    return lax.all_gather(Qm, axis_name, axis=0, tiled=True)[:B]


def _eigh_desc(F, batched_max_k):
    k = F.shape[-1]
    Fe = F + 1e-30 * jnp.eye(k, dtype=F.dtype)
    if F.shape[0] > 1 and k > batched_max_k:
        q = jnp.stack([jnp.linalg.eigh(Fe[b])[1] for b in range(F.shape[0])])
    else:
        q = jnp.linalg.eigh(Fe)[1]
    return q[..., ::-1]


# ----------------------------------------------------------------------------- transform
def soap_grouped_build(b1=0.95, b2=0.95, eps=1e-8, precondition_frequency=10, max_precond_dim=10000,
                          shampoo_beta=None, precondition_1d=False, *, drop_unit=True, layout='grouped',
                          orth='householder', batched_qr_max_k=128, cholqr_reg=1e-12,
                          axis_name: Optional[str] = None, axis_size: int = 1, state_dtype=None,
                          adam_leading_dim: Optional[int] = None):
    """layout='grouped': ScaleBySoapGroupedState (no per-step state copies);
    layout='per_leaf': the current ScaleBySoapState (checkpoint compatible; stacks / unstacks
    the state every step, ~2-3 GB of HBM traffic, ~1 ms).

    state_dtype: None (= parameter dtype) or e.g. 'float64' (needs jax_enable_x64): the rotated
    moments, Kronecker factors and eigenbases of the SOAP leaves are kept and computed in this dtype
    while the parameters and the returned updates stay in their own dtype (mixed precision; grouped
    layout only).  Rationale (2026-11-06): in float32 the rotation Q_L^T G Q_R leaves round-off of
    order 1e-7 |G| in directions whose true gradient is zero; with eps = 1e-8 the Adam division turns
    it into O(1) normalised steps, and the OAM-subset runs diverged within ~30 steps."""
    b1, b2, eps, freq = float(b1), float(b2), float(eps), int(precondition_frequency)
    beta_s = b2 if shampoo_beta is None or shampoo_beta < 0 else float(shampoo_beta)
    plan_kw = dict(max_dim=int(max_precond_dim), pre1d=bool(precondition_1d), drop_unit=bool(drop_unit),
                   adam_leading_dim=adam_leading_dim)
    if adam_leading_dim is not None and layout != 'grouped':
        raise ValueError("adam_leading_dim needs layout='grouped'")
    okw = dict(mode=orth, batched_max_k=int(batched_qr_max_k), reg=float(cholqr_reg))

    if state_dtype is not None and layout != 'grouped':
        raise ValueError("state_dtype needs layout='grouped'")
    sdt = None if state_dtype is None else jnp.dtype(state_dtype)
    if sdt is not None and sdt.itemsize == 8 and not jax.config.jax_enable_x64:
        raise ValueError("state_dtype float64 needs jax_enable_x64 (the parameters may stay float32)")

    def plan(leaves):
        dts = [jnp.result_type(x) for x in leaves]
        if sdt is not None:
            dts = [sdt if jnp.issubdtype(d, jnp.floating) else d for d in dts]
        return make_plan([jnp.shape(x) for x in leaves], dts, **plan_kw)

    # ---- grouped state <-> per-leaf state (exact reshapes; also the checkpoint converters)
    def to_grouped(st: ScaleBySoapState, leaves, treedef):
        specs, groups = plan(leaves)
        f = [list(treedef.flatten_up_to(x)) for x in st[1:]]
        mu_l, nu_l, gl_l, gr_l, ql_l, qr_l = f
        e = lambda dt: jnp.zeros((0,), dt)
        amu = [mu_l[i] if sp.kind == 'adam' else e(jnp.float32) for i, sp in enumerate(specs)]
        anu = [nu_l[i] if sp.kind == 'adam' else e(jnp.float32) for i, sp in enumerate(specs)]
        mu, nu, gl, gr, ql, qr = {}, {}, {}, {}, {}, {}
        for g in groups:
            parts = []
            for mb in g.members:                     # un-rotate dropped unit sides: mu_true = q * mu_rot, q = +-1
                sp, x = specs[mb.leaf], mu_l[mb.leaf].reshape((mb.count, g.m, g.n))
                if sp.left and not g.left:
                    x = x * ql_l[mb.leaf].reshape((mb.count, 1, 1))
                if sp.right and not g.right:
                    x = x * qr_l[mb.leaf].reshape((mb.count, 1, 1))
                parts.append(x)
            mu[g.key] = (parts[0] if len(parts) == 1 else jnp.concatenate(parts, 0)).astype(g.dtype)
            nu[g.key] = _stack(nu_l, g, g.dtype)
            gl[g.key] = _stack_fac(gl_l, g, g.m).astype(g.dtype) if g.left else e(g.dtype)
            gr[g.key] = _stack_fac(gr_l, g, g.n).astype(g.dtype) if g.right else e(g.dtype)
            ql[g.key] = _stack_fac(ql_l, g, g.m).astype(g.dtype) if g.left else e(g.dtype)
            qr[g.key] = _stack_fac(qr_l, g, g.n).astype(g.dtype) if g.right else e(g.dtype)
        return ScaleBySoapGroupedState(st.count, treedef.unflatten(amu), treedef.unflatten(anu),
                                       mu, nu, gl, gr, ql, qr)

    def to_per_leaf(gs: ScaleBySoapGroupedState, leaves, treedef, like: Optional[ScaleBySoapState] = None):
        specs, groups = plan(leaves)
        n = len(leaves)
        if like is not None:
            cols = [list(treedef.flatten_up_to(x)) for x in like[1:]]
        else:   # fresh layout (unit-side factors: F = 0, Q = 1)
            cols = [list(treedef.flatten_up_to(x)) for x in init_per_leaf(treedef.unflatten(leaves))[1:]]
        amu, anu = treedef.flatten_up_to(gs.adam_mu), treedef.flatten_up_to(gs.adam_nu)
        for i, sp in enumerate(specs):
            if sp.kind == 'adam':
                cols[0][i], cols[1][i] = amu[i], anu[i]
        for g in groups:
            for mb in g.members:
                sl = slice(mb.start, mb.start + mb.count)
                lead = mb.shape[:-2] if len(mb.shape) >= 2 else ()
                x = gs.mu[g.key][sl]
                sp = specs[mb.leaf]
                if sp.left and not g.left:
                    x = x * cols[4][mb.leaf].reshape((mb.count, 1, 1))
                if sp.right and not g.right:
                    x = x * cols[5][mb.leaf].reshape((mb.count, 1, 1))
                cols[0][mb.leaf] = x.reshape(mb.shape)
                cols[1][mb.leaf] = gs.nu[g.key][sl].reshape(mb.shape)
                if g.left:
                    cols[2][mb.leaf] = gs.gl[g.key][sl].reshape(lead + (g.m, g.m))
                    cols[4][mb.leaf] = gs.ql[g.key][sl].reshape(lead + (g.m, g.m))
                if g.right:
                    cols[3][mb.leaf] = gs.gr[g.key][sl].reshape(lead + (g.n, g.n))
                    cols[5][mb.leaf] = gs.qr[g.key][sl].reshape(lead + (g.n, g.n))
        return ScaleBySoapState(gs.count, *[treedef.unflatten(c) for c in cols])

    from bam_omat24.soap_optax import scale_by_soap as _per_leaf_soap
    init_per_leaf = _per_leaf_soap(b1=b1, b2=b2, eps=eps, precondition_frequency=freq,
                                   max_precond_dim=max_precond_dim, shampoo_beta=shampoo_beta,
                                   precondition_1d=precondition_1d).init

    def init_fn(params):
        st = init_per_leaf(params)
        if layout == 'per_leaf':
            return st
        leaves, treedef = jax.tree.flatten(params)
        return to_grouped(st, leaves, treedef)

    def grouped_update(leaves, specs, groups, gs: ScaleBySoapGroupedState, treedef):
        t = gs.count
        t1 = t + jnp.asarray(1, t.dtype)
        out = [None] * len(leaves)
        amu, anu = list(treedef.flatten_up_to(gs.adam_mu)), list(treedef.flatten_up_to(gs.adam_nu))
        for i, (g, sp) in enumerate(zip(leaves, specs)):
            if sp.kind == 'skip':
                out[i] = jnp.zeros_like(g)
            elif sp.kind == 'adam':                           # == optax.scale_by_adam
                m = (1.0 - b1) * g + b1 * amu[i]
                v = (1.0 - b2) * (g * g) + b2 * anu[i]
                tf = t1.astype(g.dtype)
                out[i] = (m / (1.0 - b1 ** tf)) / (jnp.sqrt(v / (1.0 - b2 ** tf)) + eps)
                amu[i], anu[i] = m, v
        G = {g.key: _stack(leaves, g, g.dtype) for g in groups}
        gl, gr = dict(gs.gl), dict(gs.gr)
        for g in groups:
            x = G[g.key]
            if g.left:
                gl[g.key] = beta_s * gl[g.key] + (1.0 - beta_s) * _mm(x, _t(x))
            if g.right:
                gr[g.key] = beta_s * gr[g.key] + (1.0 - beta_s) * _mm(_t(x), x)
        first = t == 0
        mu, nu, U = {}, {}, {}
        for g in groups:
            k = g.key
            gp = _proj(G[k], gs.ql[k], gs.qr[k], g)
            m1 = b1 * gs.mu[k] + (1.0 - b1) * gp
            v1 = b2 * gs.nu[k] + (1.0 - b2) * (gp * gp)
            tf = jnp.maximum(t, 1).astype(g.dtype)
            step = jnp.sqrt(1.0 - b2 ** tf) / (1.0 - b1 ** tf)
            u = step * _proj_back(m1 / (jnp.sqrt(v1) + eps), gs.ql[k], gs.qr[k], g)
            mu[k] = jnp.where(first, gs.mu[k], m1)
            nu[k] = jnp.where(first, gs.nu[k], v1)
            U[k] = jnp.where(first, jnp.zeros_like(u), u)

        def factor_sides():
            for g in groups:
                if g.left:
                    yield g, 'L', g.m
                if g.right:
                    yield g, 'R', g.n

        def keep(ops):
            mu, nu, ql, qr = ops
            return mu, nu, ql, qr

        def first_eigh(ops):
            mu, nu, ql, qr = ops
            ql, qr = dict(ql), dict(qr)
            by_k = {}
            for g, side, kd in factor_sides():
                by_k.setdefault((kd, g.dtype.name), []).append((g, side))
            for _, items in by_k.items():
                F = jnp.concatenate([(gl if s == 'L' else gr)[g.key] for g, s in items], 0)
                Q = _eigh_desc(F, okw['batched_max_k'])
                o = 0
                for g, s in items:
                    (ql if s == 'L' else qr)[g.key] = Q[o:o + g.B]
                    o += g.B
            return mu, nu, ql, qr

        def refresh(ops):
            mu, nu, ql, qr = ops
            ql, qr, nu = dict(ql), dict(qr), dict(nu)
            morig = {g.key: _proj_back(mu[g.key], ql[g.key], qr[g.key], g) for g in groups}
            by_k = {}
            for g, side, kd in factor_sides():
                F = (gl if side == 'L' else gr)[g.key]
                Q = (ql if side == 'L' else qr)[g.key]
                Y = _mm(F, Q)                                       # once: est and power iteration
                est = jnp.sum(Q * Y, axis=-2)                       # diag(Q^T F Q), (B, k)
                perm = jnp.argsort(-est, axis=-1, stable=True)
                idx = jnp.broadcast_to(perm[:, None, :], Y.shape)
                Ypm = jnp.take_along_axis(Y, idx, axis=-1)          # == F @ Q[:, perm]
                Qpm = jnp.take_along_axis(Q, idx, axis=-1)
                v = nu[g.key]
                if side == 'L':
                    v = jnp.take_along_axis(v, jnp.broadcast_to(perm[:, :, None], v.shape), axis=-2)
                else:
                    v = jnp.take_along_axis(v, jnp.broadcast_to(perm[:, None, :], v.shape), axis=-1)
                nu[g.key] = v
                by_k.setdefault((kd, g.dtype.name), []).append((g, side, Ypm, Qpm, F))
            for _, items in by_k.items():
                cat = lambda j: items[0][j] if len(items) == 1 else jnp.concatenate([it[j] for it in items], 0)
                Y, Qp, F = cat(2), cat(3), cat(4)
                if axis_name is None or int(axis_size) == 1:
                    Qn = _orth(Y, Qp, F, **okw)
                else:
                    Qn = _orth_partitioned(Y, Qp, F, axis_name, int(axis_size), **okw)
                o = 0
                for g, s, *_ in items:
                    (ql if s == 'L' else qr)[g.key] = Qn[o:o + g.B]
                    o += g.B
            mu = {g.key: _proj(morig[g.key], ql[g.key], qr[g.key], g) for g in groups}
            return mu, nu, ql, qr

        branch = jnp.where(first, 1, jnp.where(t % freq == 0, 2, 0)).astype(jnp.int32)
        mu, nu, ql, qr = lax.switch(branch, (keep, first_eigh, refresh), (mu, nu, dict(gs.ql), dict(gs.qr)))
        for g in groups:
            for mb in g.members:
                out[mb.leaf] = U[g.key][mb.start:mb.start + mb.count].reshape(mb.shape).astype(leaves[mb.leaf].dtype)
        new = ScaleBySoapGroupedState(t1, treedef.unflatten(amu), treedef.unflatten(anu), mu, nu, gl, gr, ql, qr)
        return out, new

    def update_fn(updates, state, params=None):
        del params
        leaves, treedef = jax.tree.flatten(updates)
        specs, groups = plan(leaves)
        if layout == 'per_leaf':
            gs = to_grouped(state, leaves, treedef)
            out, gs = grouped_update(leaves, specs, groups, gs, treedef)
            new = to_per_leaf(gs, leaves, treedef, like=state)
            # unit (length-1) sides are not rotated in the grouped maths, so their 1x1 factors are
            # not carried in the grouped state; keep them exact here (the reference accumulates them)
            gl_l, gr_l = list(treedef.flatten_up_to(new.gg_left)), list(treedef.flatten_up_to(new.gg_right))
            for i, sp in enumerate(specs):
                if sp.kind != 'soap':
                    continue
                m, n = sp.mat_shape[-2:]
                x = leaves[i].astype(sp.dtype).reshape(sp.mat_shape)
                if sp.left and m == 1 and plan_kw['drop_unit']:
                    gl_l[i] = beta_s * gl_l[i] + (1.0 - beta_s) * _mm(x, _t(x))
                if sp.right and n == 1 and plan_kw['drop_unit']:
                    gr_l[i] = beta_s * gr_l[i] + (1.0 - beta_s) * _mm(_t(x), x)
            new = new._replace(gg_left=treedef.unflatten(gl_l), gg_right=treedef.unflatten(gr_l))
            return treedef.unflatten(out), new
        out, gs = grouped_update(leaves, specs, groups, state, treedef)
        return treedef.unflatten(out), gs

    return optax.GradientTransformation(init_fn, update_fn), to_grouped, to_per_leaf


def scale_by_soap_grouped(*args, **kw) -> optax.GradientTransformation:
    """Shape-grouped SOAP transform (see ``soap_grouped_build`` for the options)."""
    return soap_grouped_build(*args, **kw)[0]


def convert_opt_state(opt_state, params, to='grouped', **kw):
    """Convert every SOAP state inside an optax chain state (checkpoint restore / save).
    ``kw`` must be the same SOAP options as the transform (the plan is shape-only)."""
    _, to_g, to_p = soap_grouped_build(**kw)
    leaves, treedef = jax.tree.flatten(params)

    def conv(s):
        if to == 'grouped' and isinstance(s, ScaleBySoapState):
            return to_g(s, leaves, treedef)
        if to == 'per_leaf' and isinstance(s, ScaleBySoapGroupedState):
            return to_p(s, leaves, treedef)
        return s
    return jax.tree.map(conv, opt_state, is_leaf=lambda s: isinstance(s, (ScaleBySoapState, ScaleBySoapGroupedState)))
