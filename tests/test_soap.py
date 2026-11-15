"""SOAP optax port (race_v2/training/soap_optax.py).

(a) vector / scalar leaves == optax.scale_by_adam
(b) matrix leaves == an independent NumPy port of the reference soap.py
    (full-rank cases directly; non-square two-sided and precondition_1d cases with
    the reference's first-step basis seeded from the implementation)
(c) ill-conditioned Kronecker least squares: SOAP clearly faster than Adam
(d) a 3D leaf == 2D SOAP run on each leading slice
(e) a side longer than max_precond_dim stays unpreconditioned; jit, finite
(f) jit + shard_map (1-device mesh, replicated params) on an nnx.State, pickling
(g) small RACE smoke test
(h) soap_leaf_summary == the state tx.init builds; the build_model integration
    snippet (from the module docstring) == the explicit optax chain
"""
import io
import pickle
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from jax.sharding import Mesh, PartitionSpec as P

try:
    from jax import shard_map
except ImportError:  # older jax
    from jax.experimental.shard_map import shard_map

import race_v2.training.soap_optax as soap_optax
from race_v2.training.soap_optax import ScaleBySoapState, scale_by_soap, soap, soap_leaf_summary

pytestmark = pytest.mark.usefixtures("x64")


# ---------------------------------------------------------------------------
# Independent NumPy port of the reference implementation (nikhilvyas/SOAP soap.py,
# merge_dims=False, correct_bias=True, normalize_grads=False, no weight decay).
# Written against the PyTorch code (tensordot projections, lerp factors, exp_avg
# projected back / forward after every step), not against soap_optax.py.
# ---------------------------------------------------------------------------
class RefSOAP:
    def __init__(self, shape, b1=0.95, b2=0.95, eps=1e-8, precondition_frequency=10,
                 max_precond_dim=10000, shampoo_beta=None, precondition_1d=False):
        self.b1, self.b2, self.eps, self.freq = b1, b2, eps, precondition_frequency
        self.sb = b2 if shampoo_beta is None or shampoo_beta < 0 else shampoo_beta
        self.step = 0
        self.exp_avg = np.zeros(shape)
        self.exp_avg_sq = np.zeros(shape)
        if len(shape) == 1:
            self.GG = [np.zeros((shape[0],) * 2) if precondition_1d and shape[0] <= max_precond_dim else None]
        else:
            self.GG = [np.zeros((s, s)) if s <= max_precond_dim else None for s in shape]
        self.Q = None

    def project(self, x):
        for q in self.Q:
            x = np.tensordot(x, q, axes=([0], [0])) if q is not None else np.transpose(x, list(range(1, x.ndim)) + [0])
        return x

    def project_back(self, x):
        for q in self.Q:
            x = np.tensordot(x, q, axes=([0], [1])) if q is not None else np.transpose(x, list(range(1, x.ndim)) + [0])
        return x

    def get_orthogonal_matrix(self):
        out = []
        for m in self.GG:
            if m is None:
                out.append(None)
                continue
            _, q = np.linalg.eigh(m + 1e-30 * np.eye(m.shape[0]))
            out.append(np.flip(q, 1))
        return out

    def get_orthogonal_matrix_QR(self):
        exp_avg_sq = self.exp_avg_sq
        out = []
        for ind, (m, o) in enumerate(zip(self.GG, self.Q)):
            if m is None:
                out.append(None)
                continue
            est_eig = np.diag(o.T @ m @ o)
            sort_idx = np.argsort(-est_eig, kind="stable")
            exp_avg_sq = np.take(exp_avg_sq, sort_idx, axis=ind)
            o = o[:, sort_idx]
            q, _ = np.linalg.qr(m @ o)
            out.append(q)
        self.exp_avg_sq = exp_avg_sq
        return out

    def update_preconditioner(self, g):
        if self.Q is not None:
            self.exp_avg = self.project_back(self.exp_avg)
        for idx, m in enumerate(self.GG):
            if m is None:
                continue
            if g.ndim == 1:
                outer = np.outer(g, g)
            else:
                axes = [k for k in range(g.ndim) if k != idx]
                outer = np.tensordot(g, g, axes=(axes, axes))
            self.GG[idx] = m + (1.0 - self.sb) * (outer - m)          # torch lerp_
        if self.Q is None:
            self.Q = self.get_orthogonal_matrix()
        if self.step > 0 and self.step % self.freq == 0:
            self.Q = self.get_orthogonal_matrix_QR()
        if self.step > 0:
            self.exp_avg = self.project(self.exp_avg)

    def __call__(self, g):
        """Returns +step_size * norm_grad (the reference does p -= lr * that)."""
        g = np.asarray(g, np.float64)
        if self.Q is None:
            self.update_preconditioner(g)
            return np.zeros_like(g)
        gp = self.project(g)
        self.step += 1
        self.exp_avg = self.b1 * self.exp_avg + (1 - self.b1) * gp
        self.exp_avg_sq = self.b2 * self.exp_avg_sq + (1 - self.b2) * gp ** 2
        denom = np.sqrt(self.exp_avg_sq) + self.eps
        step_size = np.sqrt(1 - self.b2 ** self.step) / (1 - self.b1 ** self.step)
        u = step_size * self.project_back(self.exp_avg / denom)
        self.update_preconditioner(g)
        return u


def _structured_grad(rng, shape):
    """Random gradient with unequal row / column scales (anisotropic factors)."""
    g = rng.normal(size=shape)
    if len(shape) >= 2:
        g = g * np.linspace(0.2, 2.0, shape[-2])[:, None] * np.linspace(1.5, 0.3, shape[-1])[None, :]
    return g


# (a) ----------------------------------------------------------------------------
def test_vector_and_scalar_leaves_match_optax_adam():
    rng = np.random.RandomState(0)
    params = {"b": jnp.zeros(7), "s": jnp.zeros(()), "w": jnp.zeros((4, 5))}
    tx = scale_by_soap(b1=0.95, b2=0.95, eps=1e-8, precondition_frequency=3)
    ref = optax.scale_by_adam(b1=0.95, b2=0.95, eps=1e-8)
    st = tx.init(params)
    rst = ref.init({"b": params["b"], "s": params["s"]})
    upd, rupd = jax.jit(tx.update), jax.jit(ref.update)
    for _ in range(20):
        scale = 10.0 ** rng.uniform(-3, 1)
        g = {"b": jnp.asarray(scale * rng.normal(size=7)), "s": jnp.asarray(scale * rng.normal()),
             "w": jnp.asarray(rng.normal(size=(4, 5)))}
        u, st = upd(g, st)
        ru, rst = rupd({"b": g["b"], "s": g["s"]}, rst)
        for k in ("b", "s"):
            assert u[k].dtype == jnp.float64 and u[k].shape == params[k].shape
            np.testing.assert_allclose(np.asarray(u[k]), np.asarray(ru[k]), rtol=1e-12, atol=1e-12)
    assert st.exp_avg["b"].dtype == jnp.float64 and st.gg_left["b"].shape == (0,)

    # soap(lr) = scale_by_soap -> scale_by_learning_rate
    full = soap(0.1, precondition_frequency=3)
    fst = full.init(params)
    g = {"b": jnp.ones(7), "s": jnp.asarray(2.0), "w": jnp.ones((4, 5))}
    fu, _ = full.update(g, fst)
    u0, _ = tx.update(g, tx.init(params))
    for k in g:
        np.testing.assert_allclose(np.asarray(fu[k]), -0.1 * np.asarray(u0[k]), rtol=1e-14, atol=1e-14)


# (b) ----------------------------------------------------------------------------
@pytest.mark.parametrize("shape,kw", [
    ((7, 7), {}),
    ((7, 7), dict(b1=0.9, b2=0.99, shampoo_beta=0.8)),
    ((6, 12), dict(max_precond_dim=8)),       # right side unpreconditioned
    ((12, 6), dict(max_precond_dim=8)),       # left side unpreconditioned
])
def test_matches_numpy_reference(shape, kw):
    # Square / one-sided cases only: the first factor L_0 = G_0 G_0^T (resp. R_0) must be
    # full rank, otherwise the eigenbasis inside its null space is whatever the eigh
    # routine returns (LAPACK here, cuSOLVER there) and the Adam step legitimately differs.
    rng = np.random.RandomState(1)
    kw = dict(kw, precondition_frequency=5)
    tx = scale_by_soap(**kw)
    ref = RefSOAP(shape, **kw)
    st = tx.init({"w": jnp.zeros(shape)})
    upd = jax.jit(tx.update)
    for step in range(25):
        g = _structured_grad(rng, shape)
        u, st = upd({"w": jnp.asarray(g)}, st)
        u_ref = ref(g)
        if step == 0:
            assert np.all(np.asarray(u["w"]) == 0.0)
        np.testing.assert_allclose(np.asarray(u["w"]), u_ref, rtol=1e-8, atol=1e-8,
                                   err_msg=f"step {step}")
    assert int(st.count) == 25 and ref.step == 24
    # second moment (including the reference's re-ordering on refresh steps)
    np.testing.assert_allclose(np.asarray(st.exp_avg_sq["w"]), ref.exp_avg_sq, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("shape,kw", [
    ((6, 9), dict(precondition_frequency=5)),                        # R_0 rank 6 < 9
    ((9, 6), dict(precondition_frequency=5)),                        # L_0 rank 6 < 9
    ((5, 16), dict(b1=0.9, b2=0.99, precondition_frequency=3)),      # R_0 rank 5 < 16
    ((9,), dict(precondition_1d=True, precondition_frequency=10)),   # reference 1D path, L_0 rank 1
    ((9, 1), dict(precondition_frequency=10)),                       # same, as an (n, 1) matrix
])
def test_matches_numpy_reference_seeded(shape, kw):
    # A rank-deficient first factor has an eigh-implementation-defined basis inside its
    # null space (LAPACK vs cuSOLVER), so the reference port's step-0 basis is replaced by
    # the implementation's; everything after that (Adam in the rotated basis, factor EMA,
    # sort + permute + QR refresh, exp_avg re-projection) is compared.  The frequency is
    # chosen so that every refresh sees a full-rank factor (a QR of a rank-deficient
    # F Q is again round-off-defined in the null space): rank grows by <= min(m, n)
    # per step, e.g. 1D n = 9 needs freq >= 8.
    rng = np.random.RandomState(7)
    tx = scale_by_soap(**kw)
    ref = RefSOAP(shape, **kw)
    st = tx.init({"w": jnp.zeros(shape)})
    upd = jax.jit(tx.update)
    n_steps = 40
    for step in range(n_steps):
        g = _structured_grad(rng, shape) if len(shape) == 2 else rng.normal(size=shape)
        u, st = upd({"w": jnp.asarray(g)}, st)
        u_ref = ref(g)
        if step == 0:
            assert np.all(np.asarray(u["w"]) == 0.0)
            facs = [np.asarray(f) for f in (st.gg_left["w"], st.gg_right["w"]) if f.size]
            assert any(np.linalg.matrix_rank(f) < f.shape[0] for f in facs)   # seeding is needed
            qs = [np.asarray(q) if q.size else None for q in (st.q_left["w"], st.q_right["w"])]
            ref.Q = qs[:1] if len(shape) == 1 else qs
        np.testing.assert_allclose(np.asarray(u["w"]), u_ref, rtol=1e-8, atol=1e-8,
                                   err_msg=f"step {step}")
    assert ref.step == n_steps - 1 and np.abs(u_ref).max() > 1e-2
    np.testing.assert_allclose(np.asarray(st.exp_avg_sq["w"]), ref.exp_avg_sq, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(np.asarray(st.gg_left["w"]), ref.GG[0], rtol=1e-12, atol=1e-14)


def test_precondition_1d_state_and_reshape():
    # Shape / reshape check of the precondition_1d path (its numerics are compared with
    # the reference port in test_matches_numpy_reference_seeded): a vector leaf has one
    # n x n factor and gives the same updates as the equivalent (n, 1) matrix leaf.
    rng = np.random.RandomState(5)
    kw = dict(precondition_frequency=5)
    t1, t2 = scale_by_soap(precondition_1d=True, **kw), scale_by_soap(**kw)
    s1, s2 = t1.init({"v": jnp.zeros(9)}), t2.init({"v": jnp.zeros((9, 1))})
    assert s1.gg_left["v"].shape == (9, 9) and s1.gg_right["v"].shape == (0,)
    u1f, u2f = jax.jit(t1.update), jax.jit(t2.update)
    for step in range(16):
        g = rng.normal(size=9)
        u1, s1 = u1f({"v": jnp.asarray(g)}, s1)
        u2, s2 = u2f({"v": jnp.asarray(g[:, None])}, s2)
        assert u1["v"].shape == (9,)
        np.testing.assert_allclose(np.asarray(u1["v"]), np.asarray(u2["v"])[:, 0], rtol=1e-9, atol=1e-10,
                                   err_msg=f"step {step}")
    assert np.abs(np.asarray(u1["v"])).max() > 0


# (c) ----------------------------------------------------------------------------
def _kron_lsq(n=8, cond=100.0, seed=0):
    rng = np.random.RandomState(seed)

    def mat():
        u, _ = np.linalg.qr(rng.normal(size=(n, n)))
        v, _ = np.linalg.qr(rng.normal(size=(n, n)))
        return u @ np.diag(np.linspace(1.0, 1.0 / cond, n)) @ v.T

    a, b = mat(), mat()
    c = rng.normal(size=(n, n))     # residual energy spread over all output directions
    return a, b, c


def _steps_to_target(tx, lr, a, b, c, n_steps, target=1e-3):
    a, b, c = jnp.asarray(a), jnp.asarray(b), jnp.asarray(c)
    l0 = jnp.sum(c ** 2)
    tx = optax.chain(tx, optax.scale_by_learning_rate(lr))
    x0 = jnp.zeros((a.shape[1], b.shape[0]))

    def mm(p, q):
        return jnp.matmul(p, q, precision=jax.lax.Precision.HIGHEST)

    @jax.jit
    def run(x0):
        def body(carry, _):
            x, st = carry
            r = mm(mm(a, x), b) - c
            g = 2.0 * mm(mm(a.T, r), b.T)      # grad of ||A X B - C||^2
            u, st = tx.update(g, st, x)
            return (x + u, st), jnp.sum(r ** 2) / l0
        _, losses = jax.lax.scan(body, (x0, tx.init(x0)), None, length=n_steps)
        return losses

    losses = np.asarray(run(x0))
    hit = np.nonzero(losses < target)[0]
    return (int(hit[0]) if hit.size else None), float(losses[-1])


def test_kronecker_least_squares_faster_than_adam():
    a, b, c = _kron_lsq()
    assert 50 < np.linalg.cond(a) < 200 and 50 < np.linalg.cond(b) < 200
    n_steps = 3000
    lrs = (0.3, 1.0, 3.0)        # brackets Adam's best constant lr on this problem
    adam = {lr: _steps_to_target(optax.scale_by_adam(b1=0.95, b2=0.95, eps=1e-8), lr, a, b, c, n_steps)
            for lr in lrs}
    so = {lr: _steps_to_target(scale_by_soap(b1=0.95, b2=0.95, eps=1e-8, precondition_frequency=10),
                               lr, a, b, c, n_steps) for lr in lrs}
    print(f"\n[soap lsq] steps to 1e-3 (final rel loss) Adam: {adam}  SOAP: {so}")
    adam_best = min((s if s is not None else n_steps + 1) for s, _ in adam.values())
    soap_best = min((s if s is not None else n_steps + 1) for s, _ in so.values())
    assert soap_best <= n_steps, so
    assert 3 * soap_best < adam_best, (adam, so)


# (d) ----------------------------------------------------------------------------
def test_3d_leaf_is_per_slice_2d_soap():
    rng = np.random.RandomState(2)
    shape = (4, 6, 6)                      # slice 3: an absent species, zero gradient
    kw = dict(precondition_frequency=5)
    tx = scale_by_soap(**kw)
    st3 = tx.init({"w": jnp.zeros(shape)})
    st2 = [tx.init({"w": jnp.zeros(shape[1:])}) for _ in range(shape[0])]
    upd = jax.jit(tx.update)
    assert st3.gg_left["w"].shape == (4, 6, 6) and st3.q_right["w"].shape == (4, 6, 6)
    for step in range(16):
        g = np.stack([_structured_grad(rng, shape[1:]) for _ in range(shape[0])])
        g[3] = 0.0
        u3, st3 = upd({"w": jnp.asarray(g)}, st3)
        u3 = np.asarray(u3["w"])
        assert np.all(np.isfinite(u3))
        for s in range(shape[0]):
            u2, st2[s] = upd({"w": jnp.asarray(g[s])}, st2[s])
            np.testing.assert_allclose(u3[s], np.asarray(u2["w"]), rtol=1e-9, atol=1e-10,
                                       err_msg=f"step {step} slice {s}")
        assert np.all(u3[3] == 0.0)


# (e) ----------------------------------------------------------------------------
def test_side_larger_than_max_precond_dim():
    rng = np.random.RandomState(3)
    shapes = {"wide": (4, 2000), "tall": (1500, 3)}
    kw = dict(precondition_frequency=5, max_precond_dim=1024)
    tx = scale_by_soap(**kw)
    st = tx.init({k: jnp.zeros(s) for k, s in shapes.items()})
    assert st.gg_left["wide"].shape == (4, 4) and st.gg_right["wide"].shape == (0,)
    assert st.gg_left["tall"].shape == (0,) and st.gg_right["tall"].shape == (3, 3)
    refs = {k: RefSOAP(s, **kw) for k, s in shapes.items()}
    upd = jax.jit(tx.update)
    for step in range(12):
        g = {k: _structured_grad(rng, s) for k, s in shapes.items()}
        u, st = upd({k: jnp.asarray(v) for k, v in g.items()}, st)
        for k in shapes:
            uk = np.asarray(u[k])
            assert uk.shape == shapes[k] and np.all(np.isfinite(uk))
            if step > 0:
                assert np.abs(uk).max() > 0
            np.testing.assert_allclose(uk, refs[k](g[k]), rtol=1e-8, atol=1e-8)
    assert st.q_right["wide"].shape == (0,) and st.q_left["tall"].shape == (0,)


# (f) ----------------------------------------------------------------------------
class _Tiny(nnx.Module):
    """0D, 1D, 2D and species-indexed 3D parameters."""

    def __init__(self, rngs):
        self.lin = nnx.Linear(6, 6, rngs=rngs, param_dtype=jnp.float64)       # (6, 6) + (6,)
        self.w_sp = nnx.Param(jax.random.normal(rngs.params(), (3, 6, 6), jnp.float64) / 3.0)
        self.gain = nnx.Param(jnp.asarray(0.7, jnp.float64))

    def __call__(self, x, species):
        h = jnp.tanh(self.lin(x))
        y = jnp.einsum("bi,bij->bj", h, self.w_sp[species])
        return self.gain * jnp.sum(jnp.sin(y) ** 2)


def test_jit_shard_map_nnx_state():
    model = _Tiny(nnx.Rngs(0))
    graphdef, params = nnx.split(model, nnx.Param)
    rng = np.random.RandomState(4)
    # 24 samples, 8 per species: every Shampoo factor (6x6) is full rank from the first step
    x = jnp.asarray(rng.normal(size=(1, 24, 6)))
    species = jnp.asarray((np.arange(24) % 3).reshape(1, 24).astype(np.int32))

    tx = optax.chain(optax.clip_by_global_norm(1.0),
                     scale_by_soap(precondition_frequency=2),
                     optax.add_decayed_weights(1e-4),
                     optax.scale_by_learning_rate(1e-2))

    def loss_fn(p, xb, sb):
        return nnx.merge(graphdef, p)(xb, sb)

    def step(p, s, xb, sb, axis=None):
        grads = jax.grad(loss_fn)(p, xb[0], sb[0])
        if axis is not None:
            grads = jax.lax.pmean(grads, axis_name=axis)
        updates, s = tx.update(grads, s, p)
        return optax.apply_updates(p, updates), s

    mesh = Mesh(jax.devices()[:1], ("dp",))
    sharded = jax.jit(shard_map(lambda p, s, xb, sb: step(p, s, xb, sb, axis="dp"),
                                mesh=mesh, check_vma=False,
                                in_specs=(P(), P(), P("dp"), P("dp")), out_specs=(P(), P())))
    plain = jax.jit(step)

    opt_state = tx.init(params)
    soap_state = opt_state[1]
    assert isinstance(soap_state, ScaleBySoapState)
    assert jax.tree.structure(soap_state.exp_avg) == jax.tree.structure(params)
    assert all(l.dtype == jnp.float64 for l in jax.tree.leaves(soap_state.gg_left) if l.size)

    p_a, s_a = params, opt_state
    p_b, s_b = params, opt_state
    for i in range(7):
        p_a, s_a = sharded(p_a, s_a, x, species)
        p_b, s_b = plain(p_b, s_b, x, species)
        if i == 3:   # checkpoint round trip mid-run (training.utils pickles opt_state)
            s_a = pickle.loads(pickle.dumps(jax.device_get(s_a)))
    for la, lb, l0 in zip(jax.tree.leaves(p_a), jax.tree.leaves(p_b), jax.tree.leaves(params)):
        assert la.dtype == jnp.float64 and np.all(np.isfinite(np.asarray(la)))
        # The two programs fuse differently (1e-16 level).  With rank-deficient factors the
        # eigenbasis inside the null space would be round-off-defined and the per-coordinate
        # Adam step would amplify that (seen with (6, 4) slices / 3 samples per species:
        # 1e-9); with full-rank factors the programs agree to round-off.
        np.testing.assert_allclose(np.asarray(la), np.asarray(lb), rtol=1e-10, atol=1e-11)
        assert np.abs(np.asarray(la) - np.asarray(l0)).max() > 0     # every leaf moved
    w_state = jax.tree.leaves(s_a[1].gg_left)
    assert sorted(l.shape for l in w_state) == sorted([(0,), (0,), (6, 6), (3, 6, 6)])
    assert int(s_a[1].count) == 7


# (g) ----------------------------------------------------------------------------
def test_race_smoke(graph):
    """A few SOAP steps on a small float64 RACE (species-indexed readout) force loss."""
    from conftest import as_float64, make_model
    model, g64 = as_float64(make_model(readout_species=True, n_layers=2), graph)
    graphdef, params = nnx.split(model, nnx.Param)
    f_ref = jnp.asarray(g64.nodes["forces"])

    def loss_fn(p):
        out = nnx.merge(graphdef, p)(g64)
        return jnp.mean((out["forces"] - f_ref) ** 2)

    tx = optax.chain(optax.clip_by_global_norm(10.0), scale_by_soap(precondition_frequency=5),
                     optax.scale_by_learning_rate(3e-3))

    @jax.jit
    def step(p, s):
        loss, grads = jax.value_and_grad(loss_fn)(p)
        u, s = tx.update(grads, s, p)
        return optax.apply_updates(p, u), s, loss

    s = tx.init(params)
    ndims = sorted({l.ndim for l in jax.tree.leaves(params)})
    losses = []
    for _ in range(21):
        params, s, loss = step(params, s)
        losses.append(float(loss))
    print(f"\n[soap race] leaf ndims {ndims}, {len(jax.tree.leaves(params))} leaves, "
          f"loss {losses[0]:.4g} -> {losses[-1]:.4g}")
    assert all(np.isfinite(losses)) and all(np.all(np.isfinite(np.asarray(l))) for l in jax.tree.leaves(params))
    assert all(l.dtype == jnp.float64 for l in jax.tree.leaves(s[1].exp_avg))
    assert losses[-1] < losses[0]


# (h) ----------------------------------------------------------------------------
def test_soap_leaf_summary():
    params = {"rad": jnp.zeros((64, 2944)), "sq": jnp.zeros((8, 8)), "sp": jnp.zeros((3, 6, 6)),
              "b": jnp.zeros(5), "s": jnp.zeros(()), "i": jnp.zeros(3, jnp.int32)}
    for mpd in (1024, 10000):
        summ = soap_leaf_summary(params, max_precond_dim=mpd)
        c = summ["counts"]
        assert c["adam"] == 6 and c["skip"] == 3 and c["right"] == 0 and c["none"] == 0
        if mpd == 1024:
            assert c["left"] == 64 * 2944 and c["both"] == 64 + 108
            assert summ["partial"] == [("['rad']", (64, 2944), "left")]
        else:
            assert c["left"] == 0 and c["both"] == 64 * 2944 + 64 + 108 and summ["partial"] == []
        # == the bytes of the state tx.init builds (shapes only, nothing allocated)
        st = jax.eval_shape(scale_by_soap(max_precond_dim=mpd).init, params)
        nbytes = sum(int(np.prod(l.shape)) * l.dtype.itemsize for f in st[1:] for l in jax.tree.leaves(f))
        assert summ["state_bytes"] == nbytes
    assert soap_leaf_summary(params)["counts"]["both"] == 64 * 2944 + 64 + 108    # default 10000


def _integration_snippet():
    doc = soap_optax.__doc__
    a = doc.index("    elif opt_name == 'soap':")
    b = doc.index("\nConfig example:", a)
    return textwrap.dedent(doc[a:b]).replace("elif opt_name == 'soap':", "if opt_name == 'soap':", 1)


@pytest.mark.parametrize("swd", [None, {"base": 1e-3, "scope": "all"}])
def test_integration_snippet(swd):
    rng = np.random.RandomState(8)
    params = {"w": jnp.asarray(rng.normal(size=(6, 9))), "b": jnp.asarray(rng.normal(size=6))}
    wd, lr = 0.01, 0.1
    cfg = {"optimizer": "soap", "learning_rate": lr, "weight_decay": wd,
           "soap_precondition_frequency": 2, "soap_max_precond_dim": 8, "stratum_weight_decay": swd}

    def _get(c, k, default=None):            # == training.train._get
        v = c.get(k, default)
        return default if v is None else v

    fout = io.StringIO()
    ns = dict(cfg=cfg, chain=[optax.clip_by_global_norm(1.0)], params=params, swd=swd, wd=wd, lr=lr,
              fout=fout, _get=_get, optax=optax, opt_name="soap")
    exec(_integration_snippet(), ns)
    chain = ns["chain"]
    if swd:   # the existing scope='all' stratum block of build_model
        chain = chain + [optax.add_decayed_weights(swd["base"]), optax.scale_by_learning_rate(lr)]
    tx = optax.chain(*chain)
    ref = optax.chain(optax.clip_by_global_norm(1.0),
                      scale_by_soap(precondition_frequency=2, max_precond_dim=8),
                      optax.add_decayed_weights(wd),                       # kept with swd (as adamw)
                      *([optax.add_decayed_weights(swd["base"])] if swd else []),
                      optax.scale_by_learning_rate(lr))
    st, rst = tx.init(params), ref.init(params)
    for step in range(5):
        g = {"w": jnp.asarray(rng.normal(size=(6, 9))), "b": jnp.asarray(rng.normal(size=6))}
        u, st = tx.update(g, st, params)
        ru, rst = ref.update(g, rst, params)
        for k in params:
            np.testing.assert_allclose(np.asarray(u[k]), np.asarray(ru[k]), rtol=1e-12, atol=1e-14)
        if step == 0:   # SOAP's matrix update is zero: what is left is -lr * (wd [+ base]) * p
            dec = wd + (swd["base"] if swd else 0.0)
            np.testing.assert_allclose(np.asarray(u["w"]), -lr * dec * np.asarray(params["w"]), rtol=1e-12)
        params = optax.apply_updates(params, u)
    out = fout.getvalue()
    assert "SOAP:" in out and "'left': 54" in out
    assert "preconditioned side(s) = left: (6, 9) ['w']" in out
