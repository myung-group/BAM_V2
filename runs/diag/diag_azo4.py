"""Compare the batch produced by the training/eval data loaders with a direct collate of the same
graphs, and the eager forward on the loader batch vs the sharded eval step."""
import json, sys, io, functools
import numpy as np, jax, jax.numpy as jnp
from flax import nnx
from bam_omat24 import train_3bpa as T
from bam_omat24.data_nnx import default_collate_fn, setup_mesh, BucketedDataLoader, MultiDeviceDataLoader
from bam_omat24.data_3bpa import GraphListDataset
from bam_omat24.training_utils import (load_checkpoint, make_sharded_evaluate_step, replicate_pytree, squeeze_batch,
                                       _loss_terms)
cfg = json.load(open(sys.argv[1])); ck = sys.argv[2]
jax.config.update("jax_enable_x64", True)
f = io.StringIO()
an, ae, tr, va, te, st = T.build_datasets(cfg, f)
tr.graphs = T._cast_graphs(tr.graphs, np.float64)
gd, *_ = T.build_model(cfg, len(an), ae, st, f)
params = jax.tree.map(jnp.asarray, load_checkpoint(ck)["ema_params"])
gs = tr.graphs[:5]
mine = default_collate_fn(gs, 5, 16, 256)
mesh, nd = setup_mesh()
col = functools.partial(default_collate_fn, node_bucket_size=16, edge_bucket_size=256)
base = BucketedDataLoader(dataset=GraphListDataset(gs), batch_size=5, n_buckets=8, shuffle=False, drop_last=False, rngs=nnx.Rngs(0))
base.collate_fn = col
loader = MultiDeviceDataLoader(base_loader=base, n_devices=nd, mesh=mesh, drop_incomplete=False)
batch, info = next(iter(loader))
print("loader batch leading shapes:", {k: tuple(np.asarray(v).shape) for k, v in batch.nodes.items()},
      "globals", {k: tuple(np.asarray(v).shape) for k, v in batch.globals.items()}, "n_node", tuple(np.asarray(batch.n_node).shape))
sq = squeeze_batch(batch)
def cmp(name, a, b):
    a, b = np.asarray(a), np.asarray(b)
    print(f"  {name:12s} shapes {a.shape} vs {b.shape}", "equal" if a.shape == b.shape and np.array_equal(a, b) else f"DIFFER max|d|={np.max(np.abs(a.astype(float)-b.astype(float))) if a.shape==b.shape else 'shape'}")
print("loader (squeezed) vs direct collate:")
for k in mine.nodes: cmp(k, sq.nodes[k], mine.nodes[k])
for k in mine.globals: cmp(k, sq.globals[k], mine.globals[k])
cmp("senders", sq.senders, mine.senders); cmp("receivers", sq.receivers, mine.receivers)
cmp("n_node", sq.n_node, mine.n_node); cmp("n_edge", sq.n_edge, mine.n_edge)
model = nnx.merge(gd, params)
print("eager energy on loader batch", np.round(np.asarray(model(sq)["energy"])[:5], 3),
      "targets", np.round(np.asarray(sq.globals["energy"]).ravel()[:5], 3))
es = make_sharded_evaluate_step(mesh, huber_delta=10.0, loss_type="huber", periodic=False, cov_mode="scalar6", beta_nll=0.0)
m = es(gd, replicate_pytree(params, mesh), batch)
print("eval_step on loader batch: energy_se_total", float(m["energy_se_total"]), "n_graphs", float(m["n_graphs"]),
      "-> RMSE", float(np.sqrt(m["energy_se_total"] / m["n_graphs"])))
t = _loss_terms(model(sq), sq, 10.0, "huber", False, "scalar6", 0.0)
print("eager _loss_terms on squeezed batch: RMSE", float(np.sqrt(t["energy_se_total"] / t["n_graphs"])))
# the same eval step but through plain jit (no shard_map)
def single(gdef, p, b):
    return _loss_terms(nnx.merge(gdef, p)(b), b, 10.0, "huber", False, "scalar6", 0.0)["energy_se_total"]
print("jit(single) on squeezed batch RMSE", float(np.sqrt(jax.jit(single, static_argnums=0)(gd, params, sq) / 5)))
