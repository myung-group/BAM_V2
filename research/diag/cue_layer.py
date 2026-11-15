"""First layer where the cue model's strata differ between a forward-only program and the
forward+grad(positions) program (training path); e3nn reference."""
import json, sys, io, functools
import numpy as np, jax, jax.numpy as jnp
from flax import nnx
from race_v2.training import train as T
from race_v2.data.loaders import default_collate_fn
from race_v2.training.utils import load_checkpoint
jax.config.update("jax_enable_x64", True)
mol = sys.argv[1]
cfg = json.load(open(f"research/input/input_3bpa_omat24_rmd17_{mol}_v9a_s42_cue.json"))
f = io.StringIO()
an, ae, tr, va, te, st = T.build_datasets(cfg, f)
gd, *_ = T.build_model(cfg, len(an), ae, st, f)
gd3, *_ = T.build_model(dict(cfg, tp_backend="e3nn"), len(an), ae, st, f)
params = jax.tree.map(jnp.asarray, load_checkpoint(f"runs/rmd17_{mol}_v9a_s42_cue/ckpt_rmd17_{mol}_v9a_s42_cue.pkl")["ema_params"])
b = default_collate_fn(T._cast_graphs(tr.graphs[:5], np.float64), 5, 16, 256)
def strata_fwd(gdef, p, b):
    m = nnx.merge(gdef, p); return m.node_energies(b.nodes["positions"], None, None, b)["strata"]
def strata_grad(gdef, p, b):
    m = nnx.merge(gdef, p)
    def tot(pos):
        out = m.node_energies(pos, None, None, b); return jnp.sum(out["node_energies"]), out["strata"]
    (_, s), g = jax.value_and_grad(tot, has_aux=True)(b.nodes["positions"])
    return s, g
n = int(np.sum(np.asarray(b.n_node)[:5]))
s_f = np.asarray(jax.jit(functools.partial(strata_fwd, gd))(params, b))[:, :n]
s_g, g_g = jax.jit(functools.partial(strata_grad, gd))(params, b); s_g = np.asarray(s_g)[:, :n]
s_3 = np.asarray(jax.jit(functools.partial(strata_fwd, gd3))(params, b))[:, :n]
s_3g, g_3 = jax.jit(functools.partial(strata_grad, gd3))(params, b)
print(f"{mol}: per-layer max|strata diff| (atoms), cue fwd vs cue fwd+grad / cue fwd vs e3nn")
for t in range(s_f.shape[0]):
    print(f"  layer {t}: {np.max(np.abs(s_f[t] - s_g[t])):.3e} / {np.max(np.abs(s_f[t] - s_3[t])):.3e}   (sum cue+grad {s_g[t].sum():9.3f}, e3nn {s_3[t].sum():9.3f})")
print("  max|grad cue - grad e3nn|", float(np.max(np.abs(np.asarray(g_g)[:n] - np.asarray(g_3)[:n]))))
