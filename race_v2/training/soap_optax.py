"""SOAP optimizer as an optax ``GradientTransformation``.

SOAP = "ShampoO with Adam in the Preconditioner's eigenbasis"
(Vyas, Morwani, Zhao, Kwun, Shapira, Brandfonbrener, Janson, Kakade,
"SOAP: Improving and Stabilizing Shampoo using Adam", arXiv:2409.11321).
This is a port of the reference PyTorch implementation
(github.com/nikhilvyas/SOAP, ``soap.py``) to a pure-functional optax transform.

Algorithm (per matrix leaf ``G`` of shape ``(m, n)``; ``t`` = Adam step)
-----------------------------------------------------------------------
Shampoo factors, an exponential moving average exactly as the reference
(``GG.lerp_(outer, 1 - shampoo_beta)``)::

    L <- beta_s L + (1 - beta_s) G G^T          (m x m)
    R <- beta_s R + (1 - beta_s) G^T G          (n x n)

``beta_s = shampoo_beta`` or ``b2``.  (The paper writes ``L <- beta L + G G^T``;
the two differ by the constant factor ``1 - beta_s``, which changes neither the
eigenvectors nor the eigenvalue order, so the algorithm is the same.)

* call 0 (first gradient): ``L, R`` are accumulated, ``Q_L, Q_R`` are the
  eigenvectors of ``L + 1e-30 I`` / ``R + 1e-30 I`` in descending eigenvalue
  order (``eigh`` + flip), the Adam moments stay zero and the returned update
  is ZERO ("first step is skipped so that we never use the current gradients in
  the projection").
* call t >= 1 (Adam step ``t``)::

      G'  = Q_L^T G Q_R                                  (rotate)
      m   = b1 m + (1 - b1) G'                           (m lives in the rotated basis)
      v   = b2 v + (1 - b2) G'^2
      u   = sqrt(1 - b2^t) / (1 - b1^t) * Q_L [ m / (sqrt(v) + eps) ] Q_R^T

  then the factors are updated with ``G`` (after the step, as in the
  reference) and, if ``t % precondition_frequency == 0``, the eigenbases are
  refreshed by one power iteration + QR, per side::

      est  = diag(Q^T F Q)           (F = L or R, Q = current basis)
      perm = argsort(est, descending)
      v    = v[perm] along that side's axis           (reference's re-ordering of exp_avg_sq)
      Q    = qr(F @ Q[:, perm]).Q

  and the first moment is carried into the new basis,
  ``m <- Q_L'^T (Q_L m Q_R^T) Q_R'`` (``v`` is only permuted, not rotated --
  this is the reference's behaviour).

The returned update is the direction ``+u`` (no learning rate, no sign flip):
chain ``optax.scale_by_learning_rate(lr)`` after it.

Leaf handling
-------------
* ``ndim == 2``: as above.  A side whose dimension exceeds ``max_precond_dim``
  is left unpreconditioned (``Q = I``, no factor stored), as in the reference.
* ``ndim >= 3``: every index of the leading axes ``shape[:-2]`` is an
  independent matrix; the last two axes are preconditioned with their own
  ``L, R, Q_L, Q_R`` per leading index (batched linear algebra), so the state is
  ``(*lead, m, m)`` / ``(*lead, n, n)``.  The reference would instead build one
  Kronecker factor per axis, i.e. also a factor that rotates (mixes) the leading
  index and bases shared across it.  The rule is purely shape-based and applies
  to every ``ndim >= 3`` leaf, whatever its axes mean.  The motivating case is the
  species-indexed e3nn weight ``(n_species, mul_in, mul_out)``: each species slice
  is a different linear map driven by different environments (an absent species
  has zero gradient), so per-slice preconditioning keeps species separate.  It
  does NOT keep every species axis out of the rotation: a species axis that is
  one of the last two axes is a preconditioned matrix side exactly as in the
  reference -- 2D species leaves (the ``(n_species, mul)`` species embedding,
  ``sym_weights (n_species, k)``, ``pair_coeff (n_species, n_species)``) get the
  reference two-sided rotation, and e.g. ``x_hill (n_layers, 2, n_species)`` /
  ``bo_c (n_species, n_species, n_alpha)`` get per-leading-slice factors whose
  ``n_species`` side is rotated.  (To exclude such leaves, wrap the transform in
  ``optax.partition`` / ``optax.masked`` by parameter path.)
* ``ndim <= 1`` (biases, gains, scalars): plain bias-corrected Adam with the same
  ``b1, b2, eps`` -- numerically identical to ``optax.scale_by_adam(b1, b2, eps)``,
  i.e. ``m_hat / (sqrt(v_hat) + eps)`` from the first call on.  With
  ``precondition_1d=True`` a vector leaf of length ``n <= max_precond_dim``
  is instead treated as an ``(n, 1)`` matrix with a left factor only, which is
  the reference's ``precondition_1d`` path (including the skipped first step).
* non-floating leaves get a zero update.

Deviations from the reference (all deliberate)
----------------------------------------------
1. 0D/1D leaves (``precondition_1d=False``): optax Adam.  The reference runs them
   through the same code path as matrices with an identity rotation, i.e. it
   discards the first gradient and uses ``eps`` inside the bias-correction
   factor (``m / (sqrt(v) + eps) * sqrt(bc2) / bc1``).  The differences are
   one step of delay and an ``eps`` of order 1e-8; optax-Adam makes these leaves
   behave exactly like the existing ``adam`` option.
2. ``ndim >= 3``: batched per-leading-index preconditioning (see above) instead
   of one factor per axis; ``merge_dims`` / ``data_format`` (conv layers) are
   not implemented.
3. Linear algebra precision: the reference casts every factor to float32 for
   ``eigh`` / ``qr`` (``m.data.float()``) even for float64 parameters.  Here
   ``eigh``/``qr`` run in the parameter dtype (float64 for RACE); only
   half-precision leaves are promoted to float32.  All matmuls use
   ``Precision.HIGHEST`` (no TF32 for float32 leaves).
4. ``exp_avg`` is re-expressed in the new basis only on refresh steps.  The
   reference does ``project(project_back(exp_avg))`` after every step, which is
   the identity (``Q^T Q = I``) whenever ``Q`` did not change; skipping it saves
   four matmuls per leaf per step and changes results only at round-off level.
5. No learning rate / weight decay inside (``optax`` composition), always
   bias-corrected (``correct_bias=True``), no ``normalize_grads``.  The reference
   also skips its decoupled weight decay on the first step; with an
   ``add_decayed_weights`` after this transform the decay is applied from step 1.

``max_precond_dim`` defaults to the reference's 10000.  In the 3BPA v9a RACE
(``input_3bpa_omat24_v9a_e3k_s42_cue.json``) every matrix side is <= 1024 except
the radial-MLP output weights ``layers[1..4].radial_mlp.layers[3].weights`` of
shape ``(64, 2944)`` (753,664 of 6.86 M parameters); with the default both of
their sides are preconditioned (``R``, ``Q_R``: 2944 x 2944 float64 = 69 MB each,
0.55 GB for the four leaves; cost measured in the Integration notes below).  With
``max_precond_dim < 2944`` their 2944 side falls back to per-coordinate Adam
(Shampoo rotation on the 64 side only).  :func:`soap_leaf_summary` lists which
leaves are one-sided for a given ``max_precond_dim``; the integration branch prints
it.  Note that a wide side's factor gains rank <= (other side) per step, so it is
rank-deficient for the first ~n/m steps; inside its null space the eigenbasis is
whatever ``eigh``/``qr`` return (as in the reference).

The ``eigh``/``qr`` sign and (degenerate-subspace) conventions do not matter:
flipping the sign of a basis vector flips the corresponding rotated coordinate of
``G'`` and ``m`` and cancels in ``Q_L [.] Q_R^T``; ``v`` is sign-free.

JAX notes
---------
Pure functional, no Python branching on traced values: the first-step
(``eigh``) / regular-step choice and the refresh (``qr``) are two top-level
``lax.cond`` on the shared step count (one each per optimizer step, not per
leaf -- ``lax.cond`` on GPU costs a host sync).  Works under ``jax.jit`` and inside
``jax.shard_map`` with replicated params / state (every device computes the same
rotation from the ``pmean``-ed gradient).  The state mirrors the parameter
pytree (any pytree, e.g. an ``nnx.State`` of ``nnx.Param``; no assumption on leaf
names); unused factors are stored as empty ``(0,)`` arrays, so every state field
has exactly the parameter tree structure and pickles like the Adam state.
"""
from __future__ import annotations

import math
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax
from jax import lax

_HI = lax.Precision.HIGHEST


def _mm(a, b):
    return jnp.matmul(a, b, precision=_HI)


def _t(a):
    return jnp.swapaxes(a, -1, -2)


class ScaleBySoapState(NamedTuple):
    """State of :func:`scale_by_soap`.

    ``count`` is the number of ``update`` calls so far (int32 scalar).  All other
    fields have the parameter tree structure; per leaf:

    * ``exp_avg`` / ``exp_avg_sq``: Adam moments, parameter shape (for a
      preconditioned leaf they live in the rotated basis, as in the reference);
    * ``gg_left`` / ``gg_right``: Shampoo factors ``L (*lead, m, m)`` /
      ``R (*lead, n, n)``, or an empty ``(0,)`` array when that side is not
      preconditioned (dimension > ``max_precond_dim``, vector or scalar leaf);
    * ``q_left`` / ``q_right``: the corresponding eigenbases (same shapes).
    """
    count: jax.Array
    exp_avg: Any
    exp_avg_sq: Any
    gg_left: Any
    gg_right: Any
    q_left: Any
    q_right: Any


class _Spec(NamedTuple):
    kind: str                 # 'soap' | 'adam' | 'skip'
    mat_shape: tuple = ()     # matrix view (*lead, m, n) of the leaf ('soap' only)
    left: bool = False        # left side (m) preconditioned
    right: bool = False       # right side (n) preconditioned
    dtype: Any = None         # dtype of the SOAP linear algebra / state


def _spec(shape, dtype, max_precond_dim: int, precondition_1d: bool) -> _Spec:
    shape = tuple(int(s) for s in shape)
    if not jnp.issubdtype(dtype, jnp.floating):
        return _Spec('skip')
    nd = len(shape)
    if nd == 0 or (nd == 1 and not precondition_1d):
        return _Spec('adam')
    lin = jnp.promote_types(dtype, jnp.float32)
    if nd == 1:   # reference precondition_1d: a single n x n factor
        return _Spec('soap', (shape[0], 1), shape[0] <= max_precond_dim, False, lin)
    return _Spec('soap', shape, shape[-2] <= max_precond_dim, shape[-1] <= max_precond_dim, lin)


def _project(x, ql, qr, sp: _Spec):
    """Q_L^T x Q_R (identity on unpreconditioned sides)."""
    if sp.left:
        x = _mm(_t(ql), x)
    if sp.right:
        x = _mm(x, qr)
    return x


def _project_back(x, ql, qr, sp: _Spec):
    """Q_L x Q_R^T (identity on unpreconditioned sides)."""
    if sp.left:
        x = _mm(ql, x)
    if sp.right:
        x = _mm(x, _t(qr))
    return x


def _eigh_desc(f):
    """Eigenvectors of a (batched) symmetric factor, descending eigenvalue order."""
    k = f.shape[-1]
    _, q = jnp.linalg.eigh(f + 1e-30 * jnp.eye(k, dtype=f.dtype))
    return q[..., ::-1]


def _qr_refresh(f, q, v, axis: int):
    """Reference ``get_orthogonal_matrix_QR`` for one side.

    Sort the current basis by the estimated eigenvalues ``diag(Q^T F Q)``
    (descending), permute the second moment ``v`` along ``axis`` accordingly, then
    one power iteration ``F Q`` followed by QR.
    """
    est = jnp.sum(q * _mm(f, q), axis=-2)                       # diag(Q^T F Q), (*lead, k)
    perm = jnp.argsort(-est, axis=-1, stable=True)
    q = jnp.take_along_axis(q, jnp.broadcast_to(perm[..., None, :], q.shape), axis=-1)
    if axis == -2:
        v = jnp.take_along_axis(v, jnp.broadcast_to(perm[..., :, None], v.shape), axis=-2)
    else:
        v = jnp.take_along_axis(v, jnp.broadcast_to(perm[..., None, :], v.shape), axis=-1)
    q_new, _ = jnp.linalg.qr(_mm(f, q))
    return q_new, v


def scale_by_soap(
    b1: float = 0.95,
    b2: float = 0.95,
    eps: float = 1e-8,
    precondition_frequency: int = 10,
    max_precond_dim: int = 10000,
    shampoo_beta: Optional[float] = None,
    precondition_1d: bool = False,
) -> optax.GradientTransformation:
    """SOAP preconditioning (Adam in the eigenbasis of Shampoo's factors).

    Args:
      b1, b2: Adam moment decays (reference ``betas``).
      eps: Adam epsilon (added to ``sqrt(v)``; reference convention for
        preconditioned leaves, optax convention for plain-Adam leaves).
      precondition_frequency: eigenbasis refresh (power iteration + QR) every
        this many Adam steps.
      max_precond_dim: a side longer than this is not preconditioned (default
        10000, as the reference; see :func:`soap_leaf_summary`).
      shampoo_beta: decay of the Shampoo factors; ``None`` -> ``b2``.
      precondition_1d: also precondition vector leaves (one ``n x n`` factor).

    Returns:
      An ``optax.GradientTransformation`` returning the (positive) update
      direction; chain ``optax.scale_by_learning_rate(lr)`` after it.
    """
    b1 = float(b1)
    b2 = float(b2)
    eps = float(eps)
    freq = int(precondition_frequency)
    if freq < 1:
        raise ValueError(f"precondition_frequency must be >= 1, got {precondition_frequency}")
    max_dim = int(max_precond_dim)
    beta_s = b2 if shampoo_beta is None or shampoo_beta < 0 else float(shampoo_beta)
    pre1d = bool(precondition_1d)

    def leaf_spec(x):
        return _spec(jnp.shape(x), jnp.result_type(x), max_dim, pre1d)

    def init_leaf(p):
        p = jnp.asarray(p)
        sp = leaf_spec(p)
        if sp.kind == 'skip':
            e = jnp.zeros((0,), jnp.float32)
            return e, e, e, e, e, e
        if sp.kind == 'adam':
            e = jnp.zeros((0,), p.dtype)
            return jnp.zeros_like(p), jnp.zeros_like(p), e, e, e, e
        dt = sp.dtype
        lead, (m, n) = sp.mat_shape[:-2], sp.mat_shape[-2:]
        e = jnp.zeros((0,), dt)
        mu = jnp.zeros(p.shape, dt)
        gl = jnp.zeros(lead + (m, m), dt) if sp.left else e
        gr = jnp.zeros(lead + (n, n), dt) if sp.right else e
        ql = jnp.broadcast_to(jnp.eye(m, dtype=dt), lead + (m, m)) if sp.left else e
        qr = jnp.broadcast_to(jnp.eye(n, dtype=dt), lead + (n, n)) if sp.right else e
        return mu, jnp.zeros_like(mu), gl, gr, ql, qr

    def init_fn(params):
        leaves, treedef = jax.tree.flatten(params)
        cols = list(zip(*[init_leaf(p) for p in leaves])) if leaves else [()] * 6
        fields = [treedef.unflatten(list(c)) for c in cols]
        return ScaleBySoapState(jnp.zeros([], jnp.int32), *fields)

    def update_fn(updates, state, params=None):
        del params
        leaves, treedef = jax.tree.flatten(updates)
        mu_l, nu_l, gl_l, gr_l, ql_l, qr_l = [list(treedef.flatten_up_to(f)) for f in state[1:]]
        t = state.count                                  # Adam step of the SOAP leaves (0 = skipped)
        t_next = t + jnp.asarray(1, t.dtype)             # Adam step of the plain-Adam leaves
        specs = [leaf_spec(g) for g in leaves]
        out = [None] * len(leaves)
        soap_idx = []

        for i, (g, sp) in enumerate(zip(leaves, specs)):
            if sp.kind == 'skip':
                out[i] = jnp.zeros_like(g)
            elif sp.kind == 'adam':
                # == optax.scale_by_adam(b1, b2, eps)
                mu = (1.0 - b1) * g + b1 * mu_l[i]
                nu = (1.0 - b2) * (g * g) + b2 * nu_l[i]
                tf = t_next.astype(g.dtype)
                mu_hat = mu / (1.0 - b1 ** tf)
                nu_hat = nu / (1.0 - b2 ** tf)
                out[i] = mu_hat / (jnp.sqrt(nu_hat) + eps)
                mu_l[i], nu_l[i] = mu, nu
            else:
                soap_idx.append(i)

        if soap_idx:
            sps = [specs[i] for i in soap_idx]
            G = [leaves[i].astype(sp.dtype).reshape(sp.mat_shape) for i, sp in zip(soap_idx, sps)]
            MU = [mu_l[i].reshape(sp.mat_shape) for i, sp in zip(soap_idx, sps)]
            NU = [nu_l[i].reshape(sp.mat_shape) for i, sp in zip(soap_idx, sps)]
            QL = [ql_l[i] for i in soap_idx]
            QR = [qr_l[i] for i in soap_idx]
            # Shampoo factors with the current gradient.  The same formula holds on the
            # first call (factors start at zero: lerp from 0 = (1 - beta_s) * outer).
            GL, GR = [], []
            for i, sp, g in zip(soap_idx, sps, G):
                GL.append(beta_s * gl_l[i] + (1.0 - beta_s) * _mm(g, _t(g)) if sp.left else gl_l[i])
                GR.append(beta_s * gr_l[i] + (1.0 - beta_s) * _mm(_t(g), g) if sp.right else gr_l[i])

            def first_step(ops):
                mus, nus, qls, qrs = ops
                us = [jnp.zeros_like(g) for g in G]
                qls = [_eigh_desc(f) if sp.left else q for f, q, sp in zip(GL, qls, sps)]
                qrs = [_eigh_desc(f) if sp.right else q for f, q, sp in zip(GR, qrs, sps)]
                return us, mus, nus, qls, qrs

            def adam_step(ops):
                mus, nus, qls, qrs = ops
                us, mus1, nus1 = [], [], []
                for g, mu, nu, ql, qr, sp in zip(G, mus, nus, qls, qrs, sps):
                    gp = _project(g, ql, qr, sp)
                    mu = b1 * mu + (1.0 - b1) * gp
                    nu = b2 * nu + (1.0 - b2) * (gp * gp)
                    tf = t.astype(sp.dtype)
                    step_size = jnp.sqrt(1.0 - b2 ** tf) / (1.0 - b1 ** tf)
                    us.append(step_size * _project_back(mu / (jnp.sqrt(nu) + eps), ql, qr, sp))
                    mus1.append(mu)
                    nus1.append(nu)
                return us, mus1, nus1, qls, qrs

            us, MU, NU, QL, QR = lax.cond(t == 0, first_step, adam_step, (MU, NU, QL, QR))

            def refresh(ops):
                mus, nus, qls, qrs = ops
                mus1, nus1, qls1, qrs1 = [], [], [], []
                for mu, nu, ql, qr, fl, fr, sp in zip(mus, nus, qls, qrs, GL, GR, sps):
                    m_orig = _project_back(mu, ql, qr, sp)
                    if sp.left:
                        ql, nu = _qr_refresh(fl, ql, nu, axis=-2)
                    if sp.right:
                        qr, nu = _qr_refresh(fr, qr, nu, axis=-1)
                    mus1.append(_project(m_orig, ql, qr, sp))
                    nus1.append(nu)
                    qls1.append(ql)
                    qrs1.append(qr)
                return mus1, nus1, qls1, qrs1

            do_refresh = jnp.logical_and(t > 0, t % freq == 0)
            MU, NU, QL, QR = lax.cond(do_refresh, refresh, lambda ops: ops, (MU, NU, QL, QR))

            for k, (i, sp) in enumerate(zip(soap_idx, sps)):
                g = leaves[i]
                out[i] = us[k].reshape(g.shape).astype(g.dtype)
                mu_l[i] = MU[k].reshape(g.shape)
                nu_l[i] = NU[k].reshape(g.shape)
                gl_l[i], gr_l[i], ql_l[i], qr_l[i] = GL[k], GR[k], QL[k], QR[k]

        new_state = ScaleBySoapState(
            t_next,
            *[treedef.unflatten(c) for c in (mu_l, nu_l, gl_l, gr_l, ql_l, qr_l)])
        return treedef.unflatten(out), new_state

    return optax.GradientTransformation(init_fn, update_fn)


def soap(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.95,
    b2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    precondition_frequency: int = 10,
    max_precond_dim: int = 10000,
    shampoo_beta: Optional[float] = None,
    precondition_1d: bool = False,
) -> optax.GradientTransformation:
    """SOAP with learning rate and optional decoupled (AdamW-style) weight decay.

    ``scale_by_soap -> add_decayed_weights(weight_decay) -> scale_by_learning_rate``,
    i.e. ``p <- p - lr * (u + wd * p)`` as in the reference.
    """
    chain = [scale_by_soap(b1=b1, b2=b2, eps=eps, precondition_frequency=precondition_frequency,
                           max_precond_dim=max_precond_dim, shampoo_beta=shampoo_beta,
                           precondition_1d=precondition_1d)]
    if weight_decay:
        chain.append(optax.add_decayed_weights(weight_decay))
    chain.append(optax.scale_by_learning_rate(learning_rate))
    return optax.chain(*chain)


def soap_leaf_summary(params, max_precond_dim: int = 10000, precondition_1d: bool = False) -> dict:
    """How :func:`scale_by_soap` will treat each leaf of ``params`` (shapes only, no compute).

    Returns a dict with

    * ``'counts'``: number of parameters per class -- ``'both'`` (two-sided SOAP),
      ``'left'`` / ``'right'`` (only that side preconditioned, the other side longer
      than ``max_precond_dim``), ``'none'`` (matrix path with identity rotation on both
      sides), ``'adam'`` (0D/1D leaves, optax Adam), ``'skip'`` (non-floating);
    * ``'partial'``: ``[(path, shape, cls), ...]`` for every leaf of class ``left`` /
      ``right`` / ``none`` (the leaves that silently lose preconditioning on a side);
    * ``'state_bytes'``: size of the :class:`ScaleBySoapState` (moments + factors + bases).
    """
    counts = dict(both=0, left=0, right=0, none=0, adam=0, skip=0)
    partial, nbytes = [], 0
    for path, x in jax.tree_util.tree_leaves_with_path(params):
        shape = tuple(int(s) for s in jnp.shape(x))
        sp = _spec(shape, jnp.result_type(x), int(max_precond_dim), bool(precondition_1d))
        size = math.prod(shape)
        if sp.kind == 'skip':
            counts['skip'] += size
            continue
        if sp.kind == 'adam':
            counts['adam'] += size
            nbytes += 2 * size * jnp.dtype(jnp.result_type(x)).itemsize
            continue
        cls = 'both' if sp.left and sp.right else 'left' if sp.left else 'right' if sp.right else 'none'
        counts[cls] += size
        if cls != 'both':
            partial.append((jax.tree_util.keystr(path), shape, cls))
        lead, (m, n) = sp.mat_shape[:-2], sp.mat_shape[-2:]
        nlead = math.prod(lead)
        item = jnp.dtype(sp.dtype).itemsize
        nbytes += item * (2 * size + 2 * nlead * ((m * m if sp.left else 0) + (n * n if sp.right else 0)))
    return dict(counts=counts, partial=partial, state_bytes=nbytes)


__doc__ = (__doc__ or '') + '''
Integration
-----------
In ``race_v2/training/train.py`` ``build_model``, add this branch to the
``if opt_name == 'amsgrad': ... elif ...`` chain (before the final
``else: raise ValueError``).  ``chain`` already starts with
``optax.clip_by_global_norm`` (clip first, on the raw gradient); the cosine
``scale_by_schedule`` is appended after it by the existing code, and the
``reduce_on_plateau`` scale is applied after ``optimizer.update`` in
``training.utils``.  The decay follows the ``adamw`` branch exactly: the global
decoupled decay ``wd`` is chained right after the SOAP scaling in both cases, and
with ``stratum_weight_decay`` set the existing block then appends the per-layer
(or ``scope='all'``) decays and ``scale_by_learning_rate(lr)`` itself.  ``params``
(already float64 with ``x64``) is in scope at that point::

    elif opt_name == 'soap':
        # SOAP (arXiv:2409.11321): Adam in the eigenbasis of Shampoo's L, R factors
        from race_v2.training.soap_optax import scale_by_soap, soap_leaf_summary
        soap_sb = _get(cfg, 'soap_shampoo_beta', None)
        soap_kw = dict(
            b1=float(_get(cfg, 'soap_b1', 0.95)),
            b2=float(_get(cfg, 'soap_b2', 0.95)),
            precondition_frequency=int(_get(cfg, 'soap_precondition_frequency', 10)),
            max_precond_dim=int(_get(cfg, 'soap_max_precond_dim', 10000)),
            shampoo_beta=None if soap_sb is None else float(soap_sb),
        )
        soap_tx = [scale_by_soap(**soap_kw)] + ([optax.add_decayed_weights(wd)] if wd > 0 else [])
        if swd:   # as adamw: SOAP -> wd -> (stratum-decay block below: decays + lr)
            chain.append(optax.chain(*soap_tx))
        else:
            chain += soap_tx + [optax.scale_by_learning_rate(lr)]
        summ = soap_leaf_summary(params, max_precond_dim=soap_kw['max_precond_dim'])
        print(f"SOAP: {soap_kw} weight_decay={wd} stratum_weight_decay={swd} "
              f"preconditioned parameters {summ['counts']} "
              f"state {summ['state_bytes'] / 2**20:.0f} MiB", file=fout)
        for path, shape, cls in summ['partial']:
            print(f"  SOAP side > max_precond_dim, preconditioned side(s) = {cls}: {shape} {path}",
                  file=fout)

Config example: ``"optimizer": "soap", "learning_rate": 3e-3, "soap_b1": 0.95,
"soap_b2": 0.95, "soap_precondition_frequency": 10``.  Notes: the first optimizer
step returns a zero update for every matrix leaf (factor accumulation only);
``opt_state`` of an ``amsgrad``/``adam`` checkpoint is not compatible (restart the
optimizer state, e.g. load only params / ema_params); per matrix the state holds
``m, v`` plus the ``m x m`` / ``n x n`` factors and bases (per species slice for
species-indexed weights), i.e. 3x the Adam state for square 128x128 weights.
Measured cost on the 3BPA v9a model (float64, one GH200, ``precondition_frequency=10``,
optimizer update alone, ``runs/diag/soap_fix/probe64.txt``): optax Adam 5.4 ms per
step; SOAP 31-41 ms per regular step, ~0.81 s / ~0.89 s per refresh step (every
10th), 0.59 s / 0.80 s for the first (eigh) step, state 511 / 1040 MiB, for
``max_precond_dim`` 1024 / 10000.  The default therefore costs ~0.53 GB and
~80 ms per refresh (~8 ms per step amortised) to precondition both sides of the
four (64, 2944) radial weights.  The SOAP overhead as a whole (~+30 ms per step
plus ~80 ms amortised refresh, against <= 45 ms for a whole v9a training step:
~4 s per 90-step epoch incl. validation) comes from the ~190 separate per-leaf
matmul / eigh / QR calls, not from the wide leaves; a larger
``soap_precondition_frequency`` amortises the refresh.
Tested in ``tests/test_soap.py`` (vs ``optax.scale_by_adam``, vs a NumPy port of
the reference incl. seeded non-square / ``precondition_1d`` cases, Kronecker least
squares, 3D slices, jit + shard_map, leaf summary, small RACE).
'''
