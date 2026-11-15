"""Feasibility probe for 1C, the monotone variational resolvent (docs/RACE_1C_monotone_resolvent_design.md).

Model r25p_ntied_hill_s42 (residual "neumann", tie_operator, x_norm "hill", layer_scale 0.5).  For fixed
geometry the tied layer map h -> layer(h)[1] with prev_features=None is the linear operator M(R) (padding
nodes masked on input and output), and the model computes h_{t+1} = u_0 + M h_t, u_0 = output of layer 0
(= prev_features of layers 1..4), so its last features are the 5-term partial sum h_5 = sum_{k<5} M^k u_0.

(1) adjoint via jax.vjp: <M u, v> = <u, M^T v>; M(0) = 0, linearity, block structure (no leak between frames
    or into padding), tied layers 2..4 give the same M as layer 1.
(2) jax.linear_transpose of the same map (cue backend): works or not, and agreement with the vjp adjoint.
(3) (I - M) h* = u_0: jax.scipy.sparse.linalg.gmres on the 5-frame batch, and a per-frame Arnoldi-GMRES
    (numpy CGS2 + Givens, GPU matvec) for the iteration count; ||h_T - h*|| / ||h*|| for the Neumann partial
    sums T = 5 (= the model), 10, 20, 40; energy change of the last stratum readout r_4(h*) - r_4(h_T)
    (gate D4); the model energy error of the same frames for context.
(4) spectrum of M per frame (ARPACK on the frame block, dim 27 x 2048): eigenvalues of largest real part and
    of largest modulus; lambda_max / lambda_min of the symmetric part (M + M^T)/2 (the monotone class
    I - M >= m I needs lambda_max(sym M) <= 1 - m); sigma_max(M); skew fraction ||M - M^T||_F / ||M + M^T||_F.
(5) CG iterations on the SPD stand-in A = m I + M^T M (m = 0.1, 0.5), tolerance 1e-8 (cost model of 1C).
(6) gate D4 on every frame of train / test_300K / test_600K / test_1200K.
Sets: 5 training frames, 5 test_1200K frames.
Any n_layers L of a tied neumann model works: the last features are then S_L = sum_{k<L} M^k u_0, layers 2..L-1
are checked against layer 1, the readout of D4 is the last stratum r_{L-1}, and the partial sums T include L, 2L
and 4L.  The last line per set is the converged-series check of design §8 Stage 1a: rho(M) < 1 and
||S_L - h*|| / ||h*|| <= 0.05 on every probed frame (only then does a T = L-trained model approximate a resolvent).
usage: python runs/diag/probe_1c.py [--run r25p_ntied_hill_s42] [--ckpt PATH] [--items 1,2,3,4,5,6] [--nf 5]
       (needs --xla_gpu_enable_triton_gemm=false; runs/sbatch_py.sh sets it).  Keep each run's output under its own
       name, e.g. > runs/diag/probe_1c_<tag>_${SLURM_JOB_ID}.txt
"""
import argparse
import io
import json
import pickle
import time

import numpy as np
import jax
import jax.numpy as jnp
import jraph
import e3nn_jax as e3nn
import scipy.sparse.linalg as spla
from scipy.linalg import solve_triangular
from flax import nnx

import bam_omat24.race_nnx as R
from bam_omat24 import train_3bpa as T
from bam_omat24.data_nnx import default_collate_fn
from bam_omat24.training_utils import load_checkpoint

jax.config.update("jax_enable_x64", True)

CAP = []
_orig = R.RACEConvolution.__call__


def _cap(self, *a, **k):
    CAP.append((self, a, k))
    return _orig(self, *a, **k)


R.RACEConvolution.__call__ = _cap

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="r25p_ntied_hill_s42", help="input/input_3bpa_omat24_<run>_cue.json")
ap.add_argument("--ckpt", default=None, help="checkpoint path (default: ckpt_name of the config)")
ap.add_argument("--items", default="1,2,3,4,5,6", help="probe items to run (3 is required by 4's summary)")
ap.add_argument("--nf", type=int, default=5, help="frames per probed set")
ARGS = ap.parse_args()
RUN = ARGS.run
NF = ARGS.nf
ITEMS = {int(i) for i in ARGS.items.split(",") if i.strip()}
T_START = time.time()


def P(*a):
    print(*a, flush=True)


def el():
    return f"[{time.time() - T_START:6.0f} s]"


# ----------------------------------------------------------------------------- numpy Krylov helpers
def gmres_np(Aop, b, tol=1e-10, kmax=800):
    """Unrestarted GMRES (CGS2 Arnoldi, Givens), returns x, iterations, estimated residual, {tol: its},
    Ritz values of A from the Arnoldi Hessenberg matrix."""
    beta = np.linalg.norm(b)
    n = b.size
    V = np.zeros((kmax + 1, n))
    V[0] = b / beta
    H = np.zeros((kmax + 1, kmax))
    Rm = np.zeros((kmax, kmax))
    cs, sn = np.zeros(kmax), np.zeros(kmax)
    g = np.zeros(kmax + 1)
    g[0] = beta
    its = {}
    res = 1.0
    k = 0
    for k in range(kmax):
        w = Aop(V[k])
        h = V[:k + 1] @ w
        w = w - V[:k + 1].T @ h
        h2 = V[:k + 1] @ w
        w = w - V[:k + 1].T @ h2
        h = h + h2
        hn = np.linalg.norm(w)
        H[:k + 1, k] = h
        H[k + 1, k] = hn
        col = np.concatenate([h, [hn]])
        for j in range(k):
            t = cs[j] * col[j] + sn[j] * col[j + 1]
            col[j + 1] = -sn[j] * col[j] + cs[j] * col[j + 1]
            col[j] = t
        r = np.hypot(col[k], col[k + 1])
        cs[k], sn[k] = col[k] / r, col[k + 1] / r
        col[k] = r
        g[k + 1] = -sn[k] * g[k]
        g[k] = cs[k] * g[k]
        Rm[:k + 1, k] = col[:k + 1]
        res = abs(g[k + 1]) / beta
        for t in (1e-4, 1e-6, 1e-8, 1e-10):
            if res < t and t not in its:
                its[t] = k + 1
        if res < tol or hn < 1e-300:
            break
        V[k + 1] = w / hn
    m = k + 1
    y = solve_triangular(Rm[:m, :m], g[:m])
    x = V[:m].T @ y
    ritz = np.linalg.eigvals(H[:m, :m])
    return x, m, res, its, ritz


def cg_np(Aop, b, tol=1e-8, kmax=3000):
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rr = r @ r
    bn = np.sqrt(b @ b)
    for k in range(kmax):
        Ap = Aop(p)
        alpha = rr / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rr_new = r @ r
        if np.sqrt(rr_new) / bn < tol:
            return x, k + 1, np.sqrt(rr_new) / bn
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x, kmax, np.sqrt(rr) / bn


def arpack(op, n, k, which, sym=False, **kw):
    L = spla.LinearOperator((n, n), matvec=op, dtype=np.float64)
    try:
        if sym:
            ev = spla.eigsh(L, k=k, which=which, return_eigenvectors=False, **kw)
        else:
            ev = spla.eigs(L, k=k, which=which, return_eigenvectors=False, **kw)
        return ev, "conv"
    except spla.ArpackNoConvergence as ex:
        return np.asarray(ex.eigenvalues), f"noconv({len(ex.eigenvalues)})"


def fmt_c(z):
    return f"{z.real:+.4f}{z.imag:+.4f}i"


# ----------------------------------------------------------------------------- model
cfg = json.load(open(f"input/input_3bpa_omat24_{RUN}_cue.json"))
fo = io.StringIO()
an, ae, tr, va, te, st = T.build_datasets(cfg, fo)
gd, p0, *_ = T.build_model(cfg, len(an), ae, st, fo)
CKPT = ARGS.ckpt or cfg["ckpt_name"]
for _try in range(20):  # save_checkpoint is a plain pickle.dump: a running job may be mid-write
    try:
        ck = load_checkpoint(CKPT)
        break
    except (EOFError, pickle.UnpicklingError, ValueError) as ex:
        print(f"checkpoint not readable yet ({type(ex).__name__}), retry in 15 s", flush=True)
        time.sleep(15)
else:
    raise RuntimeError(f"could not read {CKPT}")
params = jax.tree.map(jnp.asarray, ck["ema_params"])
model = nnx.merge(gd, params)
energy_fn = jax.jit(lambda p, b: nnx.merge(gd, p)(b)["energy"])
NL = int(cfg["n_layers"])
TAILS = sorted({5, 10, 20, 40, NL, 2 * NL, 4 * NL})
P(f"== probe_1c  run {RUN}  ckpt {CKPT} epoch {ck.get('epoch')}  n_layers {NL}  tp_backend {cfg['tp_backend']}  "
  f"residual {cfg['residual']}  tie_operator {cfg['tie_operator']}  x_norm {cfg['x_norm']}  "
  f"layer_scale {cfg['layer_scale']}  stratum_gain {model.stratum_gain}  items {sorted(ITEMS)}  jax {jax.__version__}  "
  f"device {jax.devices()[0].device_kind}")
assert cfg["residual"] == "neumann" and cfg["tie_operator"], "the probe needs a tied neumann model"


def run_set(name, graphs):
    P(f"\n######## set {name}: {NF} frames  {el()}")
    gs = T._cast_graphs(graphs[:NF], np.float64)
    b = default_collate_fn(gs, NF, 16, 256)
    nmask = jnp.asarray(jraph.get_node_padding_mask(b), dtype=jnp.float64)
    n_node = np.asarray(b.n_node)[:NF].astype(int)
    starts = np.concatenate([[0], np.cumsum(n_node)[:-1]]).astype(int)
    e_pred = np.asarray(energy_fn(params, b))[:NF]
    e_ref = np.array([float(np.asarray(g.globals["energy"]).ravel()[0]) for g in gs])
    dE_model = (e_pred - e_ref) * 1e3

    CAP.clear()
    out = model.node_energies(b.nodes["positions"], None, None, b)
    layers = list(CAP)
    assert len(layers) == NL, (len(layers), NL)
    lay1, a1, k1 = layers[1]
    irr = a1[0].irreps
    shape = a1[0].array.shape
    D = shape[1]
    u0 = k1["prev_features"].array * nmask[:, None]
    P(f"  padded nodes {shape[0]}, real {int(nmask.sum())}, feature dim {D} ({irr}), n_edges {b.senders.shape[0]}; "
      f"||u0 - h_1(layer-1 input)|| = {float(jnp.linalg.norm(u0 - a1[0].array * nmask[:, None])):.2e}")

    def make_M(lay, a, k):
        kw = dict(k, prev_features=None)

        def M(h):
            return _orig(lay, e3nn.IrrepsArray(irr, h * nmask[:, None]), *a[1:], **kw)[1].array * nmask[:, None]
        return M

    M = make_M(lay1, a1, k1)
    Mj = jax.jit(M)
    MTj = jax.jit(lambda v: jax.vjp(M, jnp.zeros_like(v))[1](v)[0])
    rnd = lambda i: jax.random.normal(jax.random.PRNGKey(i), shape, jnp.float64) * nmask[:, None]

    reps = 20
    if 1 in ITEMS:
        probe_adjoint(M, Mj, MTj, rnd, u0, shape, nmask, n_node, starts, layers, make_M, reps)
    if 2 in ITEMS:
        probe_linear_transpose(M, MTj, rnd, u0, shape, reps)
    summ = {}
    if 3 in ITEMS:
        summ.update(probe_solve(M, Mj, MTj, u0, out, nmask, layers, irr, shape, D, b, n_node, starts, dE_model))
    if 4 in ITEMS:
        summ.update(probe_spectrum(Mj, MTj, shape, D, n_node, starts))
    if 5 in ITEMS:
        probe_cg(Mj, MTj, u0, shape, D, n_node, starts)
    if "tailL" in summ and "rho" in summ:
        ok = bool(np.all(summ["rho"] < 1.0) and np.all(summ["tailL"] <= 0.05))
        P(f"  converged-series check (rho(M) < 1 and ||S_{NL} - h*||/||h*|| <= 0.05 on every frame): "
          f"{'PASS' if ok else 'FAIL'} | rho(M) {summ['rho'].min():.4f}-{summ['rho'].max():.4f} | "
          f"max Re lam {summ['lr'].min():+.4f}-{summ['lr'].max():+.4f} | tail at T={NL}: "
          f"{summ['tailL'].min():.4f}-{summ['tailL'].max():.4f} | ||h*||/||u0|| {summ['hs'].min():.3f}-{summ['hs'].max():.3f}")
    P(f"  set {name} done {el()}")


def frame_ops_factory(Mj, MTj, shape, D, n_node, starts):
    def frame_ops(g):
        s, n = int(starts[g]), int(n_node[g])

        def emb(x):
            full = np.zeros(shape)
            full[s:s + n] = x.reshape(n, D)
            return jnp.asarray(full)

        mv = lambda x: np.asarray(Mj(emb(x)))[s:s + n].ravel()
        mtv = lambda x: np.asarray(MTj(emb(x)))[s:s + n].ravel()
        return s, n, emb, mv, mtv
    return frame_ops


def probe_adjoint(M, Mj, MTj, rnd, u0, shape, nmask, n_node, starts, layers, make_M, reps):
    # ---------------- (1) adjoint
    t = time.time(); Mj(rnd(0)).block_until_ready(); t_c1 = time.time() - t
    t = time.time(); MTj(rnd(0)).block_until_ready(); t_c2 = time.time() - t
    t = time.time()
    for _ in range(reps):
        y = Mj(u0)
    y.block_until_ready(); t_mv = (time.time() - t) / reps
    t = time.time()
    for _ in range(reps):
        y = MTj(u0)
    y.block_until_ready(); t_mtv = (time.time() - t) / reps
    P(f"  (1) compile M {t_c1:.1f} s, M^T(vjp) {t_c2:.1f} s; matvec M {t_mv * 1e3:.2f} ms, M^T {t_mtv * 1e3:.2f} ms "
      f"(batch of {NF})")
    u, v = rnd(1), rnd(2)
    Mu = Mj(u)
    f0 = float(jnp.linalg.norm(Mj(jnp.zeros(shape))))
    lin = float(jnp.linalg.norm(Mj(2.0 * u - 3.0 * v) - 2.0 * Mu + 3.0 * Mj(v)) / jnp.linalg.norm(Mu))
    P(f"      ||M 0|| = {f0:.2e}; linearity rel err {lin:.2e}")
    for i in range(3):
        u, v = rnd(10 + i), rnd(20 + i)
        Mu, MTv = Mj(u), MTj(v)
        l, r = float(jnp.vdot(Mu, v)), float(jnp.vdot(u, MTv))
        rel = abs(l - r) / (float(jnp.linalg.norm(Mu)) * float(jnp.linalg.norm(v)))
        P(f"      adjoint test {i}: <Mu,v> = {l:+.15e}  <u,M^T v> = {r:+.15e}  |diff|/(|Mu||v|) = {rel:.2e}  "
          f"|diff|/|<Mu,v>| = {abs(l - r) / abs(l):.2e}")
    leak_pad = float(jnp.linalg.norm(MTj(rnd(30)) * (1 - nmask)[:, None]))
    fm0 = jnp.zeros(shape[0]).at[starts[0]:starts[0] + n_node[0]].set(1.0)
    w0 = rnd(31) * fm0[:, None]
    leak_fr = float(jnp.linalg.norm(Mj(w0) * (1 - fm0)[:, None]) / jnp.linalg.norm(Mj(w0)))
    leak_frT = float(jnp.linalg.norm(MTj(w0) * (1 - fm0)[:, None]) / jnp.linalg.norm(MTj(w0)))
    P(f"      leak into padding of M^T: {leak_pad:.2e}; leak of frame 0 into other frames: M {leak_fr:.2e}, M^T {leak_frT:.2e}")
    u = rnd(40)
    Mu = Mj(u)
    for tl in range(2, len(layers)):
        Mt = jax.jit(make_M(*layers[tl]))
        P(f"      tied: ||M_{tl} u - M_1 u|| / ||M_1 u|| = {float(jnp.linalg.norm(Mt(u) - Mu) / jnp.linalg.norm(Mu)):.2e}")


def probe_linear_transpose(M, MTj, rnd, u0, shape, reps):
    # ---------------- (2) linear_transpose
    try:
        t = time.time()
        lt = jax.linear_transpose(M, jax.ShapeDtypeStruct(shape, jnp.float64))
        ltj = jax.jit(lambda c: lt(c)[0])
        v = rnd(50)
        w = ltj(v).block_until_ready()
        ref = MTj(v)
        P(f"  (2) jax.linear_transpose OK ({time.time() - t:.1f} s incl. compile): "
          f"||lt(v) - vjp(v)|| / ||vjp(v)|| = {float(jnp.linalg.norm(w - ref) / jnp.linalg.norm(ref)):.2e}")
        t = time.time()
        for _ in range(reps):
            y = ltj(u0)
        y.block_until_ready()
        P(f"      linear_transpose matvec {(time.time() - t) / reps * 1e3:.2f} ms")
    except Exception as ex:  # noqa: BLE001
        P(f"  (2) jax.linear_transpose FAILED: {type(ex).__name__}: {str(ex)[:600]}")


def probe_solve(M, Mj, MTj, u0, out, nmask, layers, irr, shape, D, b, n_node, starts, dE_model):
    # ---------------- (3) solve (I - M) h = u0
    frame_ops = frame_ops_factory(Mj, MTj, shape, D, n_node, starts)
    Aop = lambda h: h - M(h)
    from jax.scipy.sparse.linalg import gmres
    t = time.time()
    hj = jax.jit(lambda bb: gmres(Aop, bb, tol=1e-12, atol=0.0, restart=200, maxiter=20,
                                  solve_method="batched")[0])(u0).block_until_ready()
    t_gm = time.time() - t
    rj = u0 - jax.jit(Aop)(hj)

    # Neumann partial sums S_T = sum_{k<T} M^k u0  (S_NL = model features)
    S = {}
    h = u0
    for T_ in range(1, max(TAILS) + 1):
        if T_ > 1:
            h = u0 + Mj(h)
        if T_ in TAILS:
            S[T_] = h
    hfin = out["node_features"].array * nmask[:, None]
    P(f"  (3) model last features vs S_{NL} = sum_(k<{NL}) M^k u0: rel diff "
      f"{float(jnp.linalg.norm(hfin - S[NL]) / jnp.linalg.norm(hfin)):.2e}")
    P(f"      jax.scipy gmres (batch, restart 200, tol 1e-12): {t_gm:.1f} s incl. compile")

    lay4 = layers[-1][0]  # last stratum readout r_{NL-1}
    species = b.nodes["species"]
    scale = jnp.asarray(model.scale[...], jnp.float64)

    @jax.jit
    def r_last(hh):
        ne = lay4.readout1(e3nn.IrrepsArray(irr, hh))
        if lay4.readout_mode == "gated":
            ne = e3nn.gate(ne, even_act=jax.nn.silu, odd_act=jnp.tanh, even_gate_act=jax.nn.silu)
        ne = lay4.readout(species, ne) if lay4.readout_species else lay4.readout(ne)
        return ne.array[:, 0] * nmask * scale

    hstar = np.zeros(shape)
    P(f"      per frame: GMRES its to rel. res 1e-4/1e-6/1e-8/1e-10 | true res | jax-gmres res, ||h_jax-h*||/||h*|| | "
      f"||h*||/||u0||, ||S{NL}||/||u0|| | ||S_T - h*||/||h*|| T={','.join(map(str, TAILS))} | max Re Ritz(M) from u0-Krylov")
    rows = []
    for g in range(NF):
        s, n, emb, mv, mtv = frame_ops(g)
        bvec = np.asarray(u0)[s:s + n].ravel()
        t = time.time()
        x, m_it, res, its, ritz = gmres_np(lambda z: z - mv(z), bvec, tol=1e-10, kmax=800)
        t_g = time.time() - t
        true_res = np.linalg.norm(bvec - (x - mv(x))) / np.linalg.norm(bvec)
        hstar[s:s + n] = x.reshape(n, D)
        hjf = np.asarray(hj)[s:s + n].ravel()
        rjf = np.linalg.norm(np.asarray(rj)[s:s + n]) / np.linalg.norm(bvec)
        xs = np.linalg.norm(x)
        tails = [np.linalg.norm(np.asarray(S[T_])[s:s + n].ravel() - x) / xs for T_ in TAILS]
        lam_ritz = 1.0 - ritz
        rows.append(dict(tailL=tails[TAILS.index(NL)], hs=xs / np.linalg.norm(bvec)))
        P(f"      frame {g}: its {[its.get(t_, '>' + str(m_it)) for t_ in (1e-4, 1e-6, 1e-8, 1e-10)]} ({t_g:.1f} s) | "
          f"{true_res:.1e} | {rjf:.1e}, {np.linalg.norm(hjf - x) / xs:.1e} | "
          f"{xs / np.linalg.norm(bvec):.3f}, {np.linalg.norm(np.asarray(S[NL])[s:s + n]) / np.linalg.norm(bvec):.3f} | "
          f"{' '.join(f'{v_:.3e}' for v_ in tails)} | {lam_ritz.real.max():+.4f} "
          f"(ritz |lam|max {np.abs(lam_ritz).max():.4f})")
    # D4: energy change of the last stratum readout
    hstar_j = jnp.asarray(hstar)
    gid = np.repeat(np.arange(NF), n_node)
    e_star = np.asarray(r_last(hstar_j))[:len(gid)]
    dEr = {}
    for T_ in TAILS:
        eT = np.asarray(r_last(S[T_]))[:len(gid)]
        dEr[T_] = np.bincount(gid, e_star - eT, minlength=NF) * 1e3
    P(f"      D4  r_{NL - 1}(h*) - r_{NL - 1}(S_T) per frame [meV]  (T = {NL} is the model):")
    for T_ in TAILS:
        P(f"          T={T_:2d}: {' '.join(f'{v_:+9.3f}' for v_ in dEr[T_])}")
    P(f"      model energy error E_pred - E_ref [meV]: {' '.join(f'{v_:+9.3f}' for v_ in dE_model)}")
    e_strata = np.asarray(out["strata"])[:, :len(gid)]
    P(f"      last-stratum energy r_{NL - 1}(S_{NL}) per frame [meV, total incl. mean]: "
      f"{' '.join(f'{v_:+.1f}' for v_ in np.bincount(gid, e_strata[-1], minlength=NF) * 1e3)}")
    return dict(tailL=np.array([r_["tailL"] for r_ in rows]), hs=np.array([r_["hs"] for r_ in rows]))


def probe_spectrum(Mj, MTj, shape, D, n_node, starts):
    # ---------------- (4) spectrum per frame
    frame_ops = frame_ops_factory(Mj, MTj, shape, D, n_node, starts)
    P("  (4) per frame: lam(M) max Re (ARPACK LR) | top |lam| (LM) | lam_max, lam_min of sym(M) | sigma_max(M) | "
      "skew ||M-M^T||_F/||M+M^T||_F")
    rng = np.random.default_rng(0)
    RHO, LRM = [], []
    for g in range(NF):
        s, n, emb, mv, mtv = frame_ops(g)
        N = n * D
        t = time.time()
        evLR, cLR = arpack(mv, N, 4, "LR", ncv=40, tol=1e-6, maxiter=150)
        evLM, cLM = arpack(mv, N, 4, "LM", ncv=40, tol=1e-6, maxiter=150)
        sym = lambda z: 0.5 * (mv(z) + mtv(z))
        evLA, cLA = arpack(sym, N, 2, "LA", sym=True, ncv=30, tol=1e-6, maxiter=300)
        evSA, cSA = arpack(sym, N, 2, "SA", sym=True, ncv=30, tol=1e-6, maxiter=300)
        evMM, cMM = arpack(lambda z: mtv(mv(z)), N, 1, "LA", sym=True, ncv=20, tol=1e-6, maxiter=300)
        num = den = 0.0
        for _ in range(8):
            z = rng.choice([-1.0, 1.0], N)
            a_, b_ = mv(z), mtv(z)
            num += np.sum((a_ - b_) ** 2)
            den += np.sum((a_ + b_) ** 2)
        lr = evLR[np.argsort(-evLR.real)] if len(evLR) else evLR
        lm = evLM[np.argsort(-np.abs(evLM))] if len(evLM) else evLM
        RHO.append(abs(lm[0]) if len(lm) else np.nan)
        LRM.append(lr[0].real if len(lr) else np.nan)
        P(f"      frame {g}: LR[{cLR}] {' '.join(fmt_c(z_) for z_ in lr[:3])} | LM[{cLM}] "
          f"{' '.join(fmt_c(z_) + f' (|{abs(z_):.4f}|)' for z_ in lm[:2])} | "
          f"sym LA[{cLA}] {np.max(evLA) if len(evLA) else float('nan'):+.4f}, SA[{cSA}] "
          f"{np.min(evSA) if len(evSA) else float('nan'):+.4f} | sigma_max[{cMM}] "
          f"{np.sqrt(np.max(evMM)) if len(evMM) else float('nan'):.4f} | skew {np.sqrt(num / den):.3f}  ({time.time() - t:.0f} s)")
    return dict(rho=np.array(RHO), lr=np.array(LRM))


def probe_cg(Mj, MTj, u0, shape, D, n_node, starts):
    # ---------------- (5) CG cost on the SPD stand-in A = m I + M^T M
    frame_ops = frame_ops_factory(Mj, MTj, shape, D, n_node, starts)
    P("  (5) CG on A = m I + M^T M (b = u0 of the frame), its to rel. res 1e-8  [1 iteration = 1 M + 1 M^T]")
    for g in range(min(NF, 3)):
        s, n, emb, mv, mtv = frame_ops(g)
        bvec = np.asarray(u0)[s:s + n].ravel()
        line = f"      frame {g}:"
        for m_ in (0.5, 0.1):
            x, it, res = cg_np(lambda z: m_ * z + mtv(mv(z)), bvec, tol=1e-8, kmax=3000)
            line += f"  m={m_}: {it} its (res {res:.1e}, ||h*||/||u0|| {np.linalg.norm(x) / np.linalg.norm(bvec):.3f} <= 1/m)"
        P(line)


def d4_all(sets, bs=10):
    """(6) gate D4 on every frame: r_n = ||S_5 - h*|| / ||h*|| per frame, dEt_n = r_4(h*) - r_4(S_5) (last
    stratum readout of the trained model applied to the exact resolvent instead of the 5-term partial sum)
    and the model energy error dE_n = E_pred - E_ref; correlations per set and pooled."""
    P(f"\n######## (6) D4 on all frames (batch {bs})  {el()}")
    lay1, lay4 = model.layers[1], model.layers[NL - 1]
    IRR = {}

    @jax.jit
    def fwd(p, b):
        m = nnx.merge(gd, p)
        CAP.clear()
        out = m.node_energies(b.nodes["positions"], None, None, b)
        _, a, k = CAP[1]
        assert k.get("edge_scale") is None and k.get("drop_scale") is None
        IRR.update(h=a[0].irreps, x=a[1].irreps, sh=a[3].irreps)
        e = m(b)["energy"]
        return dict(u0=k["prev_features"].array, xf=a[1].array, sh=a[3].array, rb=a[4],
                    hfin=out["node_features"].array, e=e)

    def M_apply(h, xf, sp, sh, rb, snd, rcv, nm):
        return _orig(lay1, e3nn.IrrepsArray(IRR["h"], h * nm[:, None]), e3nn.IrrepsArray(IRR["x"], xf), sp,
                     e3nn.IrrepsArray(IRR["sh"], sh), rb, snd, rcv, prev_features=None)[1].array * nm[:, None]

    from jax.scipy.sparse.linalg import gmres

    @jax.jit
    def solve(u0, *c):
        return gmres(lambda h: h - M_apply(h, *c), u0, tol=1e-11, atol=0.0, restart=200, maxiter=10,
                     solve_method="batched")[0]

    @jax.jit
    def resid(h, u0, *c):
        return u0 - (h - M_apply(h, *c))

    scale = jnp.asarray(model.scale[...], jnp.float64)

    @jax.jit
    def r_last(hh, sp, nm):
        ne = lay4.readout1(e3nn.IrrepsArray(IRR["h"], hh))
        if lay4.readout_mode == "gated":
            ne = e3nn.gate(ne, even_act=jax.nn.silu, odd_act=jnp.tanh, even_gate_act=jax.nn.silu)
        ne = lay4.readout(sp, ne) if lay4.readout_species else lay4.readout(ne)
        return ne.array[:, 0] * nm * scale

    pooled = []
    for name, graphs in sets:
        gs_all = T._cast_graphs(graphs, np.float64)
        R_, DT, DE, RES, CHK = [], [], [], [], []
        t0 = time.time()
        for s0 in range(0, len(gs_all), bs):
            gs = gs_all[s0:s0 + bs]
            b = default_collate_fn(gs, bs, 16, 256)
            ng = len(gs)
            nm_np = np.asarray(jraph.get_node_padding_mask(b), np.float64)
            nm = jnp.asarray(nm_np)
            o = fwd(params, b)
            c = (o["xf"], b.nodes["species"], o["sh"], o["rb"], b.senders, b.receivers, nm)
            u0 = o["u0"] * nm[:, None]
            hs = solve(u0, *c)
            rr = np.asarray(resid(hs, u0, *c))
            s5 = np.asarray(o["hfin"]) * nm_np[:, None]
            hs_np = np.asarray(hs)
            gid = np.repeat(np.arange(b.n_node.shape[0]), np.asarray(b.n_node))
            sq = lambda x: np.bincount(gid, np.sum(x * x, 1), minlength=b.n_node.shape[0])[:ng]
            R_.append(np.sqrt(sq(s5 - hs_np) / sq(hs_np)))
            RES.append(np.sqrt(sq(rr) / sq(np.asarray(u0))))
            de = np.asarray(r_last(hs, b.nodes["species"], nm)) - np.asarray(r_last(o["hfin"], b.nodes["species"], nm))
            DT.append(np.bincount(gid, de, minlength=b.n_node.shape[0])[:ng] * 1e3)
            eref = np.array([float(np.asarray(g.globals["energy"]).ravel()[0]) for g in gs])
            DE.append((np.asarray(o["e"])[:ng] - eref) * 1e3)
        r, dt, dE, res = map(np.concatenate, (R_, DT, DE, RES))
        pooled.append((name, r, dt, dE))
        dEc = dE - dE.mean()
        slope = np.polyfit(dt - dt.mean(), dEc, 1)[0]
        r2 = np.corrcoef(dt, dE)[0, 1] ** 2
        P(f"  {name:11s} n={len(r):4d} ({time.time() - t0:.0f} s) | max rel. residual {res.max():.1e} | "
          f"r_n = ||S5-h*||/||h*|| mean {r.mean():.3f} sd {r.std():.3f} [min {r.min():.3f} max {r.max():.3f}] | "
          f"dEt = r4(h*)-r4(S5) mean {dt.mean():+.1f} sd {dt.std():.1f} meV | model dE mean {dE.mean():+.2f} "
          f"RMSE {np.sqrt(np.mean(dE ** 2)):.2f} meV | corr(r_n,|dE|) {np.corrcoef(r, np.abs(dE))[0, 1]:+.3f} "
          f"corr(r_n,|dE-mean|) {np.corrcoef(r, np.abs(dEc))[0, 1]:+.3f} corr(dEt,dE) {np.corrcoef(dt, dE)[0, 1]:+.3f} "
          f"(slope {slope:+.4f}, R2 {r2:.3f})")
    # pooled over the test sets: does the truncation term explain the temperature-dependent error?
    names = [p[0] for p in pooled if p[0].startswith("test")]
    if names:
        r = np.concatenate([p[1] for p in pooled if p[0] in names])
        dt = np.concatenate([p[2] for p in pooled if p[0] in names])
        dE = np.concatenate([p[3] for p in pooled if p[0] in names])
        X = np.column_stack([np.ones_like(dt), dt])
        coef, *_ = np.linalg.lstsq(X, dE, rcond=None)
        rmse_after = np.sqrt(np.mean((dE - X @ coef) ** 2))
        P(f"  pooled test sets {names}: corr(r_n,|dE|) {np.corrcoef(r, np.abs(dE))[0, 1]:+.3f}, corr(dEt,dE) "
          f"{np.corrcoef(dt, dE)[0, 1]:+.3f}; dE ~ a + b dEt: b = {coef[1]:+.4f}, RMSE {np.sqrt(np.mean(dE ** 2)):.2f} -> "
          f"{rmse_after:.2f} meV (in-sample upper bound of what a re-weighted truncation tail explains)")
        for name, r_, dt_, dE_ in pooled:
            if name in names:
                P(f"     {name}: RMSE after the pooled fit {np.sqrt(np.mean((dE_ - coef[0] - coef[1] * dt_) ** 2)):.2f} "
                  f"(before {np.sqrt(np.mean(dE_ ** 2)):.2f})")


if ITEMS & {1, 2, 3, 4, 5}:
    run_set("train", tr.graphs)
    run_set("test_1200K", te["test_1200K"].graphs)
if 6 in ITEMS:
    d4_all([("train", tr.graphs), ("test_300K", te["test_300K"].graphs), ("test_600K", te["test_600K"].graphs),
            ("test_1200K", te["test_1200K"].graphs)])
P(f"\n== done {el()}")
