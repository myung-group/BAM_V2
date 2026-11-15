"""Training-free gates for the literature-sweep ideas (docs/RACE_literature_sweep_2026-10-29.md):

D3 (go/no-go for 1B, shape/scale bond-order term): per frame S_t = sum_i log rms(x_i^(t)) of the x factor
   *before* the per-node normalisation (x_norm rms / hill), per layer t; partial R^2 of the signed
   1200 K energy error on S_t beyond the bond strain (sum over bonded pairs |r_ij - <r_ij>_train|).
D6 (go/no-go for 3a, rational readouts): fraction of the per-layer stratum-readout pre-activations
   (input of the gate) at 300 / 1200 K test frames that fall outside the per-channel [min, max] of the
   training frames, and the largest overshoot in units of the training range.

usage: python research/diag/gates_1b.py run [run ...]       (needs the XLA Triton-GEMM flag; sbatch_py.sh sets it)
"""
import json, sys, io
import numpy as np, jax, jax.numpy as jnp, jraph
import e3nn_jax as e3nn
from flax import nnx
import race_v2.models.race as R
from race_v2.training import train as T
from race_v2.data.loaders import default_collate_fn
from race_v2.training.utils import load_checkpoint
jax.config.update("jax_enable_x64", True)

XCAP, GCAP = [], []
_rms, _hill, _gate = R._rms_normalize, R._hill_normalize, e3nn.gate


def rms_hook(x, eps=1e-6):
    XCAP.append(jnp.sqrt(jnp.mean(x.array ** 2, -1) + eps)); return _rms(x, eps)


def hill_hook(x, ab, species, eps=1e-6):
    XCAP.append(jnp.sqrt(jnp.mean(x.array ** 2, -1) + eps)); return _hill(x, ab, species, eps)


def gate_hook(x, *a, **k):
    if k.get("odd_act") is jnp.tanh:          # stratum readout of RACEConvolution (x readout uses sigmoid)
        GCAP.append(x.array)
    return _gate(x, *a, **k)


R._rms_normalize, R._hill_normalize, R.e3nn.gate = rms_hook, hill_hook, gate_hook
BS = 10


def evaluate(gd, params, graphs):
    @jax.jit
    def step(p, b):
        m = nnx.merge(gd, p)
        e = m(b)["energy"]
        XCAP.clear(); GCAP.clear()
        m.node_energies(b.nodes["positions"], None, None, b)
        return e, list(XCAP), list(GCAP)
    E, Xs, Gs = [], [], []
    for k in range(0, len(graphs), BS):
        gs = graphs[k:k + BS]
        b = default_collate_fn(gs, BS, 16, 256)
        e, xs, gl = step(params, b)
        nm = np.asarray(jraph.get_node_padding_mask(b)); gm = np.asarray(jraph.get_graph_padding_mask(b))
        gid = np.repeat(np.arange(b.n_node.shape[0]), np.asarray(b.n_node))[nm]
        E.append(np.asarray(e)[gm])
        # per-frame sum over atoms of log rms(x) per layer
        Xs.append(np.stack([np.bincount(gid, np.log(np.asarray(x)[nm]), minlength=b.n_node.shape[0])[gm]
                            for x in xs], 1) if xs else np.zeros((gm.sum(), 0)))
        Gs.append([np.asarray(g)[nm] for g in gl])
    return (np.concatenate(E), np.concatenate(Xs),
            [np.concatenate([g[t] for g in Gs]) for t in range(len(Gs[0]))])


def r2(cols, y):
    X = np.column_stack([np.ones(len(y))] + list(cols)); c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1.0 - (y - X @ c).var() / y.var()


for run in sys.argv[1:]:
    cfg = json.load(open(f"research/input/input_3bpa_omat24_{run}_cue.json"))
    f = io.StringIO()
    an, ae, tr, va, te, st = T.build_datasets(cfg, f)
    gd, *_ = T.build_model(cfg, len(an), ae, st, f)
    ck = load_checkpoint(cfg["ckpt_name"]); params = jax.tree.map(jnp.asarray, ck["ema_params"])
    sets = {"train": tr.graphs, "test_300K": te["test_300K"].graphs, "test_1200K": te["test_1200K"].graphs}
    res = {}
    for s, gs in sets.items():
        gs = T._cast_graphs(gs, np.float64)
        E, S, G = evaluate(gd, params, gs)
        Eref = np.array([float(np.asarray(g.globals["energy"]).ravel()[0]) for g in gs])
        pos = np.stack([np.asarray(g.nodes["positions"], np.float64) for g in gs])
        res[s] = dict(dE=(E - Eref) * 1e3, S=S, G=G, pos=pos, Eref=Eref)
    D = lambda P: np.linalg.norm(P[:, :, None] - P[:, None], axis=-1)
    Dtr = D(res["train"]["pos"]).mean(0)
    bonded = np.triu((Dtr < 1.7) & (Dtr > 0), 1)
    out = [f"== {run} (epoch {ck.get('epoch')})"]
    for s in ("test_300K", "test_1200K"):
        r = res[s]; y = r["dE"] - res["train"]["dE"].mean()
        strain = np.abs(D(r["pos"]) - Dtr)[:, bonded].sum(1)
        line = f"  {s}: dE mean {y.mean():+.2f} sd {y.std():.2f} | R2 strain {r2([strain], y):.3f}"
        if r["S"].shape[1]:
            Sc = r["S"] - res["train"]["S"].mean(0)
            line += (f" | +S_t(all layers) {r2([strain, *Sc.T], y):.3f} (partial {r2([strain, *Sc.T], y) - r2([strain], y):+.3f})"
                     f" | per-layer corr(dE,S_t) {np.round([np.corrcoef(y, c)[0, 1] for c in Sc.T], 2).tolist()}"
                     f" | S_t shift vs train {np.round(Sc.mean(0), 2).tolist()}")
        out.append(line)
        g_tr = res["train"]["G"]
        fr, ov = [], []
        for t in range(len(r["G"])):
            lo, hi = g_tr[t].min(0), g_tr[t].max(0); rng = np.maximum(hi - lo, 1e-12)
            x = r["G"][t]; outside = (x < lo) | (x > hi)
            fr.append(outside.mean() * 100)
            ov.append(float(np.max(np.maximum(lo - x, x - hi) / rng)))
        out.append(f"     D6 readout pre-activations outside train range (% per layer) {np.round(fr, 3).tolist()} "
                   f"| max overshoot (x train range) {np.round(ov, 3).tolist()}")
    print("\n".join(out), flush=True)
