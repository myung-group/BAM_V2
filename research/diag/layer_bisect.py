"""Per-layer stratum energy sum and output feature checksum in the energy-only vs energy+forces
programs (cue and e3nn backends).  usage: layer_bisect.py run"""
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
    ne, h = orig(self, *a, **k)
    kk = jax.random.normal(jax.random.PRNGKey(3), h.array.shape[1:], dtype=h.array.dtype)
    jax.debug.callback(lambda *v: LOG.append(tuple(float(x) for x in v)),
                       jnp.sum(ne.array), jnp.sum(h.array * kk), jnp.sum(a[0].array ** 2), jnp.sum(a[1].array ** 2))
    return ne, h
R.RACEConvolution.__call__ = cap
run = sys.argv[1]
cfg = json.load(open(f"research/input/input_3bpa_omat24_{run}_cue.json"))
f = io.StringIO()
an, ae, tr, va, te, st = T.build_datasets(cfg, f)
p = jax.tree.map(jnp.asarray, load_checkpoint(f"runs/{run}_cue/ckpt_{run}_cue.pkl")["ema_params"])
gs = T._cast_graphs(tr.graphs[:5], np.float64)
b = default_collate_fn(gs, 5, 16, 256)
for backend in ("cue", "e3nn"):
    gd, *_ = T.build_model(dict(cfg, tp_backend=backend), len(an), ae, st, f)
    eo = jax.jit(lambda p, b: nnx.merge(gd, p)(b)["energy"])
    ef = jax.jit(lambda p, b: (lambda o: (o["energy"], o["forces"]))(nnx.merge(gd, p)(b)))
    for name, fn in (("E-only", eo), ("E+F", ef)):
        LOG.clear(); out = fn(p, b); jax.block_until_ready(out); jax.effects_barrier()
        e = np.asarray(out[0] if isinstance(out, tuple) else out)[:5]
        print(f"{backend:4s} {name:6s} E sum {e.sum():.6f}")
        for i, v in enumerate(LOG):
            print(f"     layer {i}: sum node E {v[0]:+.9e}  proj(h_out) {v[1]:+.9e}  |h_in|2 {v[2]:.9e}  |x|2 {v[3]:.9e}")
