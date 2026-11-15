"""Which forward is right?  Same parameters through (a) cue backend eager, (b) e3nn backend eager,
(c) cue backend inside a shard_map wrapper (as the train/eval steps), per-layer strata and per-species
node energies, on the first 5 training graphs of the given config."""
import json, sys, io, functools
import numpy as np, jax, jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax import shard_map
from flax import nnx
from race_v2.training import train as T
from race_v2.data.loaders import default_collate_fn, setup_mesh
from race_v2.training.utils import load_checkpoint, replicate_pytree, squeeze_batch
cfg = json.load(open(sys.argv[1])); ck = sys.argv[2]
jax.config.update("jax_enable_x64", True)
f = io.StringIO()
an, ae, tr, va, te, st = T.build_datasets(cfg, f)
tr.graphs = T._cast_graphs(tr.graphs, np.float64)
gd_cue, *_ = T.build_model(cfg, len(an), ae, st, f)
gd_e3, *_ = T.build_model(dict(cfg, tp_backend="e3nn"), len(an), ae, st, f)
params = jax.tree.map(jnp.asarray, load_checkpoint(ck)["ema_params"])
gs = tr.graphs[:5]; b = default_collate_fn(gs, 5, 16, 256)
n = int(b.n_node[0]); sp = np.asarray(b.nodes["species"])[:n]
tgt = np.array([float(np.asarray(g.globals["energy"]).ravel()[0]) for g in gs])

def strata(gd, p, batch):
    m = nnx.merge(gd, p)
    out = m.node_energies(batch.nodes["positions"], None, None, batch)
    return out["strata"], out["node_energies"], m(batch)["energy"][:5]

def show(name, s, ne, e):
    s, ne, e = np.asarray(s), np.asarray(ne).ravel(), np.asarray(e)
    print(f"{name:22s} energies {np.round(e, 3)} | RMSE {np.sqrt(np.mean((e - tgt) ** 2)):.3f}")
    print(f"{'':22s} strata(graph0) {np.round(s[:, :n].sum(1), 3)} | species mean "
          f"{[round(float(ne[:n][sp == k].mean()), 3) for k in range(3)]}")

print("targets", np.round(tgt, 3))
show("cue eager", *strata(gd_cue, params, b))
try:
    show("e3nn eager", *strata(gd_e3, params, b))
except Exception as ex:
    print("e3nn eager failed:", type(ex).__name__, str(ex)[:300])
mesh, nd = setup_mesh()
def body(p, batch):
    batch = squeeze_batch(batch)
    s, ne, e = strata(gd_cue, p, batch)
    return s, ne, e
sm = jax.jit(shard_map(body, mesh=mesh, check_vma=False, in_specs=(P(), P('dp')), out_specs=P()))
bb = jax.tree.map(lambda x: x[None], b)
show("cue shard_map", *sm(replicate_pytree(params, mesh), bb))
try:
    sm2 = jax.jit(shard_map(body, mesh=mesh, check_vma=True, in_specs=(P(), P('dp')), out_specs=P()))
    show("cue shard_map vma", *sm2(replicate_pytree(params, mesh), bb))
except Exception as ex:
    print("shard_map check_vma=True failed:", type(ex).__name__, str(ex)[:200])
def body_e3(p, batch):
    return strata(gd_e3, p, squeeze_batch(batch))
try:
    sm3 = jax.jit(shard_map(body_e3, mesh=mesh, check_vma=False, in_specs=(P(), P('dp')), out_specs=P()))
    show("e3nn shard_map", *sm3(replicate_pytree(params, mesh), bb))
except Exception as ex:
    print("e3nn shard_map failed:", type(ex).__name__, str(ex)[:200])
