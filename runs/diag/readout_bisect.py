"""Inside RACEXReadout: checksums of input, layer-norm output, indexed-linear output and final output
in the forward-only vs value_and_grad programs (cue).  usage: readout_bisect.py run"""
import json, sys, io
import numpy as np, jax, jax.numpy as jnp, jraph
import e3nn_jax as e3nn
from flax import nnx
import bam_omat24.race_nnx as R
from bam_omat24 import train_3bpa as T
from bam_omat24.data_nnx import default_collate_fn
from bam_omat24.training_utils import load_checkpoint
jax.config.update("jax_enable_x64", True)
LOG = []
def chk(tag, a):
    a = a.array if isinstance(a, e3nn.IrrepsArray) else a
    k = jax.random.normal(jax.random.PRNGKey(5), a.shape, dtype=a.dtype)
    jax.debug.callback(lambda s, q: LOG.append((tag, float(s), float(q), str(a.dtype), a.shape)), jnp.sum(a * k), jnp.sum(a * a))
orig = R.RACEXReadout.__call__
def rec(self, species, x_feats):
    chk("in", x_feats)
    xn = R.separated_layer_norm(R._dense(x_feats)) / jnp.sqrt(self.avg_n_neighbors) if (x_feats.irreps.dim > 1 and self.norm and not self.linear) else R._dense(x_feats) / jnp.sqrt(self.avg_n_neighbors)
    chk("ln", xn)
    l1 = self.x_readout1(species, xn); chk("lin1", l1)
    l2 = self.x_readout2(l1); chk("lin2", l2)
    g = e3nn.gate(l2, even_act=jax.nn.silu, odd_act=jax.nn.sigmoid, even_gate_act=jax.nn.silu); chk("gate", g)
    o3 = self.x_readout3(g); chk("lin3", o3)
    out = orig(self, species, x_feats)
    chk("out", out)
    return out
R.RACEXReadout.__call__ = rec
run = sys.argv[1]
cfg = json.load(open(f"input/input_3bpa_omat24_{run}_cue.json"))
f = io.StringIO()
an, ae, tr, va, te, st = T.build_datasets(cfg, f)
p = jax.tree.map(jnp.asarray, load_checkpoint(f"runs/{run}_cue/ckpt_{run}_cue.pkl")["ema_params"])
b = default_collate_fn(T._cast_graphs(tr.graphs[:5], np.float64), 5, 16, 256)
mask = np.asarray(jraph.get_node_padding_mask(b))
gd, *_ = T.build_model(cfg, len(an), ae, st, f)
def fwd(p, b):
    return nnx.merge(gd, p).node_energies(b.nodes["positions"], None, None, b)["node_energies"]
def vg(p, b):
    m = nnx.merge(gd, p)
    def tot(pos):
        ne = m.node_energies(pos, None, None, b)["node_energies"].reshape(-1)
        return jnp.sum(ne * jnp.asarray(mask, ne.dtype))
    return jax.value_and_grad(tot)(b.nodes["positions"])
res = {}
for name, fn in (("fwd", fwd), ("vg", vg)):
    LOG.clear(); jax.block_until_ready(jax.jit(fn)(p, b)); jax.effects_barrier(); res[name] = list(LOG)
print("calls", len(res["fwd"]), len(res["vg"]))
# group by tag, in call order
for name in res:
    print(name, [t for t, *_ in res[name]][:12])
from collections import defaultdict
seq = {nm: defaultdict(list) for nm in res}
for nm in res:
    for t, s, q, dt, sh in res[nm]:
        seq[nm][t].append((s, q, dt, sh))
for t in seq["fwd"]:
    for i, (a, c) in enumerate(zip(seq["fwd"][t], seq["vg"][t])):
        rel = abs(a[0] - c[0]) / max(abs(a[0]), 1e-300)
        print(f"  {t:7s} #{i}: fwd proj {a[0]:+.9e} |.|2 {a[1]:.6e} {a[2]} {a[3]} | vg proj {c[0]:+.9e} |.|2 {c[1]:.6e} {c[2]} | rel {rel:.1e}{'  <--' if rel > 1e-9 else ''}")
