"""Train 1 epoch, then in the SAME process rebuild the model and evaluate the saved checkpoint."""
import json, sys, io, runpy
import numpy as np, jax, jax.numpy as jnp
from flax import nnx
from race_v2.training import train as T
from race_v2.data.loaders import default_collate_fn
from race_v2.training.utils import load_checkpoint
cfg = json.load(open(sys.argv[1])); cfg.update(n_epochs=1, ckpt_name="research/diag/ckpt_same.pkl", log_file="research/diag/loss_same.out")
T.train(cfg)
c = load_checkpoint("research/diag/ckpt_same.pkl")
print("logged valid E_RMSE_total", c["metrics"]["energy_rmse_total"])
f = io.StringIO()
an, ae, tr, va, te, st = T.build_datasets(cfg, f)
va.graphs = T._cast_graphs(va.graphs, np.float64)
gd, *_ = T.build_model(cfg, len(an), ae, st, f)
model = nnx.merge(gd, jax.tree.map(jnp.asarray, c["ema_params"]))
gs = va.graphs[:5]
print("same-process rebuilt: target", [round(float(np.asarray(g.globals["energy"]).ravel()[0]), 3) for g in gs],
      "pred", np.round(np.asarray(model(default_collate_fn(gs, 5, 16, 256))["energy"])[:5], 3))
