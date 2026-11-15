"""Per-node final energies in the forward-only program and in the value_and_grad program (cue), and
the graph energies assembled from them.  usage: node_bisect.py run"""
import json, sys, io
import numpy as np, jax, jax.numpy as jnp, jraph
from flax import nnx
from race_v2.training import train as T
from race_v2.data.loaders import default_collate_fn
from race_v2.training.utils import load_checkpoint
jax.config.update("jax_enable_x64", True)
run = sys.argv[1]
cfg = json.load(open(f"research/input/input_3bpa_omat24_{run}_cue.json"))
f = io.StringIO()
an, ae, tr, va, te, st = T.build_datasets(cfg, f)
p = jax.tree.map(jnp.asarray, load_checkpoint(f"runs/{run}_cue/ckpt_{run}_cue.pkl")["ema_params"])
gs = T._cast_graphs(tr.graphs[:5], np.float64)
b = default_collate_fn(gs, 5, 16, 256)
mask = np.asarray(jraph.get_node_padding_mask(b)); n = int(mask.sum())
sp = np.asarray(b.nodes["species"])
for backend in ("cue", "e3nn"):
    gd, *_ = T.build_model(dict(cfg, tp_backend=backend), len(an), ae, st, f)
    def fwd(p, b):
        o = nnx.merge(gd, p).node_energies(b.nodes["positions"], None, None, b)
        return o["node_energies"].reshape(-1), o["strata"], o["strata_parts"]
    def vg(p, b):
        m = nnx.merge(gd, p)
        def tot(pos):
            o = m.node_energies(pos, None, None, b)
            ne = o["node_energies"].reshape(-1)
            return jnp.sum(ne * jnp.asarray(mask, ne.dtype)), (ne, o["strata"], o["strata_parts"])
        (E, (ne, s, sp_)), g = jax.value_and_grad(tot, has_aux=True)(b.nodes["positions"])
        return ne, s, sp_, g
    ne1, s1, P1 = jax.jit(fwd)(p, b); ne1, s1 = np.asarray(ne1), np.asarray(s1)
    ne2, s2, P2, g = jax.jit(vg)(p, b); ne2, s2 = np.asarray(ne2), np.asarray(s2)
    for k in P1:
        a1, a2 = np.asarray(P1[k]), np.asarray(P2[k])
        print(f"   part {k:4s}: max |d| per layer (real nodes) {[f'{np.abs(a2[i, :n] - a1[i, :n]).max():.2e}' for i in range(a1.shape[0])]}"
              f"  | real-node sums fwd {[f'{a1[i, :n].sum():.3e}' for i in range(a1.shape[0])]}")
    d = ne2 - ne1
    print(f"{backend}: real nodes {n}; sum ne fwd {ne1[:n].sum():.6f}  vg {ne2[:n].sum():.6f}")
    print(f"   max |d ne| real {np.abs(d[:n]).max():.3e}  padded {np.abs(d[n:]).max():.3e}; per species mean d "
          f"{[round(float(d[:n][sp[:n] == k].mean()), 4) for k in range(3)]}")
    ds = s2 - s1
    print(f"   strata max |d| per stratum (real nodes): {[f'{np.abs(ds[i, :n]).max():.2e}' for i in range(ds.shape[0])]}")
    print(f"   strata real-node sums fwd: {[f'{s1[i, :n].sum():.4e}' for i in range(s1.shape[0])]}")
