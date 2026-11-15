"""Azobenzene with the norm guards: per-frame E error on train and valid frames in three programs
(cue energy+forces, cue forward-only, e3nn energy+forces), and per-layer feature norms.
usage: python research/diag/diag_azo_guard.py run [run ...]"""
import json, sys, io
import numpy as np, jax, jax.numpy as jnp
from flax import nnx
import race_v2.models.race as R
from race_v2.training import train as T
from race_v2.data.loaders import default_collate_fn
from race_v2.training.utils import load_checkpoint
jax.config.update("jax_enable_x64", True)
LOG = []
orig = R.RACEConvolution.__call__
def cap(self, *a, **k):
    out = orig(self, *a, **k)
    jax.debug.callback(lambda h, d: LOG.append((float(h), float(d))), jnp.mean(a[0].array ** 2), jnp.mean(out[1].array ** 2))
    return out
R.RACEConvolution.__call__ = cap
for run in sys.argv[1:]:
    cfg = json.load(open(f"research/input/input_3bpa_omat24_{run}_cue.json"))
    f = io.StringIO()
    an, ae, tr, va, te, st = T.build_datasets(cfg, f)
    gd, *_ = T.build_model(cfg, len(an), ae, st, f)
    gd3, *_ = T.build_model(dict(cfg, tp_backend="e3nn"), len(an), ae, st, f)
    ck = load_checkpoint(f"runs/{run}_cue/ckpt_{run}_cue.pkl")
    p = jax.tree.map(jnp.asarray, ck["ema_params"])
    print(f"== {run}: checkpoint epoch {ck.get('epoch')}")
    full = jax.jit(lambda p, b: nnx.merge(gd, p)(b)["energy"])
    full3 = jax.jit(lambda p, b: nnx.merge(gd3, p)(b)["energy"])
    fwd = jax.jit(lambda p, b: jnp.sum(nnx.merge(gd, p).node_energies(b.nodes["positions"], None, None, b)["node_energies"]
                                       .reshape(-1)[: b.nodes["positions"].shape[0]]))
    for name, ds in (("train", tr), ("valid", va)):
        gs = T._cast_graphs(ds.graphs[:10], np.float64)
        b = default_collate_fn(gs, 10, 16, 256)
        tgt = np.array([float(np.asarray(g.globals["energy"]).ravel()[0]) for g in gs])
        LOG.clear(); e = np.asarray(full(p, b))[:10]; jax.effects_barrier(); norms = list(LOG)
        e3 = np.asarray(full3(p, b))[:10]
        print(f"  {name}: cue E+F   dE {np.round(e - tgt, 3)}")
        print(f"  {name}: e3nn E+F  dE {np.round(e3 - tgt, 3)}")
        if name == "train":
            print("  per-layer mean |h_in|^2, |h_out|^2:", [(f"{a:.2e}", f"{c:.2e}") for a, c in norms])
