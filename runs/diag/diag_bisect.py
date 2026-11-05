"""Bisect the azobenzene offset: capture the training-process graphdef and the pre-pickle best_state,
then compare energies of (train graphdef, in-memory ema) / (new graphdef, in-memory ema) /
(new graphdef, pickle-loaded ema) on the same valid graphs."""
import json, sys, io
import numpy as np, jax, jax.numpy as jnp
from flax import nnx
import bam_omat24.train_3bpa as T
from bam_omat24.data_nnx import default_collate_fn
from bam_omat24.training_utils import load_checkpoint
cfg = json.load(open(sys.argv[1])); cfg.update(n_epochs=1, ckpt_name="runs/diag/ckpt_bisect.pkl", log_file="runs/diag/loss_bisect.out")
cap = {}
orig_bm, orig_sc = T.build_model, T.save_checkpoint
def bm(*a, **k):
    out = orig_bm(*a, **k); cap["gd"] = out[0]; cap["p0"] = out[1]; return out
def sc(state, path):
    cap["state"] = state; orig_sc(state, path)
T.build_model, T.save_checkpoint = bm, sc
T.train(cfg)
T.build_model = orig_bm
print("logged valid E_RMSE_total", cap["state"]["metrics"]["energy_rmse_total"])
f = io.StringIO()
an, ae, tr, va, te, st = T.build_datasets(cfg, f)
va.graphs = T._cast_graphs(va.graphs, np.float64)
gd_new, p_new, *_ = orig_bm(cfg, len(an), ae, st, f)
ema_mem = cap["state"]["ema_params"]; ema_load = load_checkpoint("runs/diag/ckpt_bisect.pkl")["ema_params"]
lm, ll = jax.tree.leaves(ema_mem), jax.tree.leaves(ema_load)
print("graphdef equal:", cap["gd"] == gd_new, "| n leaves", len(lm), len(ll),
      "| max |mem-load|", max(float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(lm, ll)),
      "| dtypes mem", {str(np.asarray(a).dtype) for a in lm}, "load", {str(np.asarray(a).dtype) for a in ll})
# leaves of the initial params vs ema: which changed most (relative)
pi = jax.tree.leaves(cap["p0"])
print("init dtypes", {str(np.asarray(a).dtype) for a in pi})
gs = va.graphs[:5]; b = default_collate_fn(gs, 5, 16, 256)
tgt = np.round([float(np.asarray(g.globals["energy"]).ravel()[0]) for g in gs], 3)
def pred(gd, p):
    return np.round(np.asarray(nnx.merge(gd, jax.tree.map(jnp.asarray, p))(b)["energy"])[:5], 3)
print("target            ", tgt)
print("train gd, mem ema ", pred(cap["gd"], ema_mem))
print("new gd,   mem ema ", pred(gd_new, ema_mem))
print("new gd,   load ema", pred(gd_new, ema_load))
print("train gd, init p  ", pred(cap["gd"], cap["p0"]))
print("new gd,   init p  ", pred(gd_new, p_new))
# the exact training eval path on the loaded params
from bam_omat24.training_utils import make_sharded_evaluate_step, evaluate_sharded, replicate_pytree
from bam_omat24.data_nnx import setup_mesh
mesh, nd = setup_mesh()
es = make_sharded_evaluate_step(mesh, huber_delta=10.0, loss_type="huber", periodic=False, cov_mode="scalar6", beta_nll=0.0)
import functools
col = functools.partial(default_collate_fn, node_bucket_size=16, edge_bucket_size=256)
for name, gd, p in (("train gd mem", cap["gd"], ema_mem), ("new gd load", gd_new, ema_load)):
    m = evaluate_sharded(graphdef=gd, params=replicate_pytree(jax.tree.map(jnp.asarray, p), mesh), files_pkl=[va], batch_size=5,
                         n_devices=nd, mesh=mesh, energy_weight=1.0, force_weight=10.0, stress_weight=0.0, huber_delta=10.0,
                         loss_type="huber", eval_step=es, params_already_replicated=True, periodic=False, cov_mode="scalar6",
                         beta_nll=0.0, collate_fn=col)
    print("evaluate_sharded", name, "E_RMSE_total", m["energy_rmse_total"])
