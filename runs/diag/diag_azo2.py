"""Per-species node energies and per-stratum sums of a checkpoint (flat-direction check)."""
import json, sys, io
import numpy as np, jax, jax.numpy as jnp
from flax import nnx
from bam_omat24 import train_3bpa as T
from bam_omat24.data_nnx import default_collate_fn
from bam_omat24.training_utils import load_checkpoint
cfg = json.load(open(sys.argv[1])); jax.config.update("jax_enable_x64", True)
f = io.StringIO()
an, ae, tr, va, te, st = T.build_datasets(cfg, f)
tr.graphs = T._cast_graphs(tr.graphs, np.float64)
gd, *_ = T.build_model(cfg, len(an), ae, st, f)
for ck in sys.argv[2:]:
    c = load_checkpoint(ck)
    model = nnx.merge(gd, jax.tree.map(jnp.asarray, c["ema_params"]))
    g = default_collate_fn(tr.graphs[:1], 1, 16, 256)
    out = model.node_energies(jnp.asarray(g.nodes["positions"]), None, None, g)
    n = int(g.n_node[0]); sp = np.asarray(g.nodes["species"])[:n]
    ne = np.asarray(out["node_energies"]).ravel()[:n]; S = np.asarray(out["strata"])[:, :n]
    print(ck.split("/")[-1], "epoch", c["epoch"], "shift", np.asarray(model.shift[...]), "scale", np.asarray(model.scale[...]))
    for s in sorted(set(sp)):
        m = sp == s
        print(f"  species {s}: n={m.sum()} node E mean {ne[m].mean():+.4f} sd {ne[m].std():.4f} | per-stratum mean", np.round(S[:, m].mean(1), 3))
    print("  total", ne.sum(), "target", float(np.asarray(g.globals["energy"]).ravel()[0]))
