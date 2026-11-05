"""Eager vs jit vs shard_map forward of one checkpoint on one batch: per-graph energies and
per-layer strata, to locate the azobenzene offset."""
import json, sys, io, functools
import numpy as np, jax, jax.numpy as jnp
from flax import nnx
from bam_omat24 import train_3bpa as T
from bam_omat24.data_nnx import default_collate_fn, setup_mesh
from bam_omat24.data_3bpa import GraphListDataset
from bam_omat24.training_utils import (load_checkpoint, make_sharded_evaluate_step, evaluate_sharded, replicate_pytree)
cfg = json.load(open(sys.argv[1])); ck = sys.argv[2]
jax.config.update("jax_enable_x64", True)
f = io.StringIO()
an, ae, tr, va, te, st = T.build_datasets(cfg, f)
tr.graphs = T._cast_graphs(tr.graphs, np.float64)
gd, *_ = T.build_model(cfg, len(an), ae, st, f)
params = jax.tree.map(jnp.asarray, load_checkpoint(ck)["ema_params"])
gs = tr.graphs[:5]
b = default_collate_fn(gs, 5, 16, 256)
tgt = np.array([float(np.asarray(g.globals["energy"]).ravel()[0]) for g in gs])
print("targets", np.round(tgt, 3))

def fwd(p, batch):
    return nnx.merge(gd, p)(batch)["energy"][:5]
def strata(p, batch):
    m = nnx.merge(gd, p)
    out = m.node_energies(batch.nodes["positions"], None, None, batch)
    return out["strata"], out["node_energies"]

e_eager = np.asarray(fwd(params, b)); e_jit = np.asarray(jax.jit(fwd)(params, b))
print("eager energy", np.round(e_eager, 3)); print("jit   energy", np.round(e_jit, 3))
s_e, ne_e = [np.asarray(x) for x in strata(params, b)]
s_j, ne_j = [np.asarray(x) for x in jax.jit(strata)(params, b)]
n = int(b.n_node[0]); sp = np.asarray(b.nodes["species"])[:n]
print("per-layer stratum sum over graph 0: eager", np.round(s_e[:, :n].sum(1), 3), "jit", np.round(s_j[:, :n].sum(1), 3))
for s in range(3):
    m = sp == s
    print(f"species {s}: node E mean eager {ne_e[:n][m].mean():+.4f} jit {ne_j[:n][m].mean():+.4f}")
print("max |eager-jit| node energies (all padded nodes):", float(np.max(np.abs(ne_e - ne_j))))
# shard_map path on the same 5 graphs
mesh, nd = setup_mesh()
es = make_sharded_evaluate_step(mesh, huber_delta=10.0, loss_type="huber", periodic=False, cov_mode="scalar6", beta_nll=0.0)
col = functools.partial(default_collate_fn, node_bucket_size=16, edge_bucket_size=256)
m = evaluate_sharded(graphdef=gd, params=replicate_pytree(params, mesh), files_pkl=[GraphListDataset(gs)], batch_size=5,
                     n_devices=nd, mesh=mesh, energy_weight=1.0, force_weight=10.0, stress_weight=0.0, huber_delta=10.0,
                     loss_type="huber", eval_step=es, params_already_replicated=True, periodic=False, cov_mode="scalar6",
                     beta_nll=0.0, collate_fn=col)
print("shard_map evaluate_sharded E_RMSE_total on these 5:", m["energy_rmse_total"],
      "| eager RMSE", float(np.sqrt(np.mean((e_eager - tgt) ** 2))), "| jit RMSE", float(np.sqrt(np.mean((e_jit - tgt) ** 2))))
# jit on CPU for the same batch
cpu = jax.devices("cpu")[0]
pc = jax.tree.map(lambda x: jax.device_put(x, cpu), params); bc = jax.tree.map(lambda x: jax.device_put(x, cpu), b)
print("cpu jit energy", np.round(np.asarray(jax.jit(fwd)(pc, bc)), 3), "cpu eager", np.round(np.asarray(fwd(pc, bc)), 3))
