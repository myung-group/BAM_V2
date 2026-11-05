"""Energies seen by the sharded eval step itself (graphdef passed as a shard_map argument, as in
training) vs a closure over graphdef, for batch sizes 1, 2, 5, 10, on the first 10 train graphs."""
import json, sys, io
import numpy as np, jax, jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax import shard_map
from flax import nnx
from bam_omat24 import train_3bpa as T
from bam_omat24.data_nnx import default_collate_fn, setup_mesh
from bam_omat24.training_utils import load_checkpoint, replicate_pytree, squeeze_batch
cfg = json.load(open(sys.argv[1])); ck = sys.argv[2]
jax.config.update("jax_enable_x64", True)
f = io.StringIO()
an, ae, tr, va, te, st = T.build_datasets(cfg, f)
tr.graphs = T._cast_graphs(tr.graphs, np.float64)
gd, *_ = T.build_model(cfg, len(an), ae, st, f)
params = jax.tree.map(jnp.asarray, load_checkpoint(ck)["ema_params"])
mesh, nd = setup_mesh()
prep = replicate_pytree(params, mesh)
print("graphdef pytree leaves:", len(jax.tree.leaves(gd)))
def body_arg(gdef, p, batch):
    b = squeeze_batch(batch)
    return nnx.merge(gdef, p)(b)["energy"]
def body_closure(p, batch):
    b = squeeze_batch(batch)
    return nnx.merge(gd, p)(b)["energy"]
f_arg = jax.jit(shard_map(body_arg, mesh=mesh, check_vma=False, in_specs=(P(), P(), P('dp')), out_specs=P()))
f_clo = jax.jit(shard_map(body_closure, mesh=mesh, check_vma=False, in_specs=(P(), P('dp')), out_specs=P()))
gs = tr.graphs[:10]
tgt = np.array([float(np.asarray(g.globals["energy"]).ravel()[0]) for g in gs])
print("targets", np.round(tgt, 3))
for bs in (1, 2, 5, 10):
    ea, ec, ee = [], [], []
    for i in range(0, 10, bs):
        b = default_collate_fn(gs[i:i + bs], bs, 16, 256)
        bb = jax.tree.map(lambda x: np.asarray(x)[None], b)
        ea += list(np.asarray(f_arg(gd, prep, bb))[:bs]); ec += list(np.asarray(f_clo(prep, bb))[:bs])
        ee += list(np.asarray(nnx.merge(gd, params)(b)["energy"])[:bs])
    print(f"bs {bs:2d}: arg {np.round(ea, 2)}\n       closure {np.round(ec, 2)}\n       eager {np.round(ee, 2)}")
