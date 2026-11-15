"""Which cue edge-TP call computes a different primal in the forward+grad program?  Records, for every
cue call, checksums of (weights, x, sh, out) in the energy-only and in the energy+forces program, and
repeats the energy+forces program with method="naive" (pure-JAX cue).  usage: cue_bisect.py run"""
import json, sys, io
import numpy as np, jax, jax.numpy as jnp
from flax import nnx
import race_v2.models.race as R
from race_v2.training import train as T
from race_v2.data.loaders import default_collate_fn
from race_v2.training.utils import load_checkpoint
jax.config.update("jax_enable_x64", True)
LOG = []; METHOD = ["uniform_1d"]
orig = R.cue_edge_tensor_product
def traced(irreps_in, sh_irreps, irreps_out, node_feats, sh, weights, senders, receivers, method="uniform_1d"):
    out = orig(irreps_in, sh_irreps, irreps_out, node_feats, sh, weights, senders, receivers, METHOD[0])
    k = jax.random.normal(jax.random.PRNGKey(7), out.array.shape[1:], dtype=out.array.dtype)
    jax.debug.callback(lambda *v: LOG.append(tuple(float(x) for x in v)),
                       jnp.sum(weights ** 2), jnp.sum(node_feats ** 2), jnp.sum(sh ** 2), jnp.sum(out.array ** 2),
                       jnp.sum(out.array * k), senders.shape[0], node_feats.shape[0])
    return out
R.cue_edge_tensor_product = traced
run = sys.argv[1]
cfg = json.load(open(f"research/input/input_3bpa_omat24_{run}_cue.json"))
f = io.StringIO()
an, ae, tr, va, te, st = T.build_datasets(cfg, f)
gd, *_ = T.build_model(cfg, len(an), ae, st, f)
p = jax.tree.map(jnp.asarray, load_checkpoint(f"runs/{run}_cue/ckpt_{run}_cue.pkl")["ema_params"])
gs = T._cast_graphs(tr.graphs[:5], np.float64)
b = default_collate_fn(gs, 5, 16, 256)
tgt = np.array([float(np.asarray(g.globals["energy"]).ravel()[0]) for g in gs])
e_only = jax.jit(lambda p, b: nnx.merge(gd, p)(b)["energy"])                       # model() takes grads inside
def ef(p, b):
    o = nnx.merge(gd, p)(b); return o["energy"], o["forces"]
def fwd_only(p, b):
    m = nnx.merge(gd, p)
    return m.node_energies(b.nodes["positions"], None, None, b)["node_energies"]
progs = [("fwd-only (node_energies)", jax.jit(fwd_only), "uniform_1d"),
         ("model() energy output", e_only, "uniform_1d"),
         ("model() energy+forces", jax.jit(ef), "uniform_1d"),
         ("model() energy+forces naive", jax.jit(ef), "naive")]
res = {}
for name, fn, meth in progs:
    METHOD[0] = meth; LOG.clear()
    out = fn(p, b); jax.block_until_ready(out); jax.effects_barrier()
    res[name] = list(LOG)
    if name.startswith("fwd-only"):
        e = None
    else:
        e = np.asarray(out[0] if isinstance(out, tuple) else out)[:5]
    print(f"{name:28s} calls {len(LOG)}" + ("" if e is None else f" | E - target {np.round(e - tgt, 3)}"), flush=True)
ref = res["fwd-only (node_energies)"]
for name in res:
    for i, (a, c) in enumerate(zip(ref, res[name])):
        rel = [abs(x - y) / max(abs(x), 1e-300) for x, y in zip(a[:5], c[:5])]
        flag = "  <-- differs" if max(rel) > 1e-9 else ""
        print(f"  {name:28s} call {i}: |w|2 {c[0]:.6e} |x|2 {c[1]:.6e} |sh|2 {c[2]:.6e} |out|2 {c[3]:.6e} proj {c[4]:+.6e} "
              f"edges {int(c[5])} nodes {int(c[6])} max rel diff vs fwd-only {max(rel):.1e}{flag}")
