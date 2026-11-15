"""Azobenzene energy offset diagnostic: per-graph predictions of the saved checkpoint (EMA and raw)
on train / valid graphs built exactly as in training, batched as in training and one graph at a time."""
import json, sys, io
import numpy as np, jax, jax.numpy as jnp
from flax import nnx
from bam_omat24 import train_3bpa as T
from bam_omat24.data_nnx import default_collate_fn
from bam_omat24.training_utils import load_checkpoint
cfg = json.load(open(sys.argv[1])); ck = sys.argv[2]
jax.config.update("jax_enable_x64", True)
f = io.StringIO()
an, ae, tr, va, te, st = T.build_datasets(cfg, f)
print("atom_energies", ae)
for ds in (tr, va): ds.graphs = T._cast_graphs(ds.graphs, np.float64)
gd, p0, *_ = T.build_model(cfg, len(an), ae, st, f)
c = load_checkpoint(ck); print("ckpt keys", list(c), "epoch", c.get("epoch"))
for key in [k for k in ("ema_params", "params") if k in c]:
    model = nnx.merge(gd, jax.tree.map(jnp.asarray, c[key]))
    for name, ds in (("train", tr), ("valid", va)):
        gs = ds.graphs[:5]
        b = default_collate_fn(gs, 5, int(cfg.get("node_bucket_size", 16)), int(cfg.get("edge_bucket_size", 256)))
        eb = np.asarray(model(b)["energy"])[:5]
        e1 = [float(np.asarray(model(default_collate_fn([g], 1, 16, 256))["energy"])[0]) for g in gs]
        tgt = [float(np.asarray(g.globals["energy"]).ravel()[0]) for g in gs]
        print(key, name, "target", np.round(tgt, 4), "batched", np.round(eb, 4), "single", np.round(e1, 4))
