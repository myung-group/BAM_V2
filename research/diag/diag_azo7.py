"""Inside one shard_map body: energies from model(batch) and the energy_se_total of _loss_terms
on the same results, vs the library eval step."""
import json, sys, io
import numpy as np, jax, jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax import shard_map
from flax import nnx
from race_v2.training import train as T
from race_v2.data.loaders import default_collate_fn, setup_mesh
from race_v2.training.utils import (load_checkpoint, replicate_pytree, squeeze_batch, _loss_terms,
                                       make_sharded_evaluate_step)
cfg = json.load(open(sys.argv[1])); ck = sys.argv[2]
jax.config.update("jax_enable_x64", True)
f = io.StringIO()
an, ae, tr, va, te, st = T.build_datasets(cfg, f)
tr.graphs = T._cast_graphs(tr.graphs, np.float64)
gd, *_ = T.build_model(cfg, len(an), ae, st, f)
params = jax.tree.map(jnp.asarray, load_checkpoint(ck)["ema_params"])
mesh, nd = setup_mesh(); prep = replicate_pytree(params, mesh)
b = default_collate_fn(tr.graphs[:5], 5, 16, 256); bb = jax.tree.map(lambda x: np.asarray(x)[None], b)
def body(gdef, p, batch):
    batch = squeeze_batch(batch)
    res = nnx.merge(gdef, p)(batch)
    t = _loss_terms(res, batch, 10.0, "huber", False, "scalar6", 0.0)
    return res["energy"], batch.globals["energy"], t["energy_se_total"], t["n_graphs"]
fn = jax.jit(shard_map(body, mesh=mesh, check_vma=False, in_specs=(P(), P(), P('dp')), out_specs=P()))
e, tg, se, ng = fn(gd, prep, bb)
print("in-body energy", np.round(np.asarray(e), 3), "| target", np.round(np.asarray(tg), 3), "| se_total", float(se), "n_graphs", float(ng))
es = make_sharded_evaluate_step(mesh, huber_delta=10.0, loss_type="huber", periodic=False, cov_mode="scalar6", beta_nll=0.0)
m = es(gd, prep, bb)
print("library eval_step se_total", float(m["energy_se_total"]), "n_graphs", float(m["n_graphs"]), "energy_ae_total", float(m["energy_ae_total"]))
res = nnx.merge(gd, params)(b)
t = _loss_terms(res, b, 10.0, "huber", False, "scalar6", 0.0)
print("eager _loss_terms se_total", float(t["energy_se_total"]), "| energy shape", res["energy"].shape, "target shape", b.globals["energy"].shape)
import race_v2.training.utils as TU
orig = TU._loss_terms
def patched(results, batch, *a, **k):
    out = orig(results, batch, *a, **k)
    out["dbg_energy"] = results["energy"]; out["dbg_target"] = batch.globals["energy"]
    out["dbg_pos0"] = batch.nodes["positions"][:3]; out["dbg_species"] = batch.nodes["species"][:30].astype(jnp.float64)
    out["dbg_nnode"] = batch.n_node.astype(jnp.float64)
    return out
TU._loss_terms = patched
es2 = TU.make_sharded_evaluate_step(mesh, huber_delta=10.0, loss_type="huber", periodic=False, cov_mode="scalar6", beta_nll=0.0)
m2 = es2(gd, prep, bb)
print("library eval_step energies", np.round(np.asarray(m2["dbg_energy"]), 3), "targets", np.round(np.asarray(m2["dbg_target"]), 3))
print("library batch n_node", np.asarray(m2["dbg_nnode"]), "species[:30]", np.asarray(m2["dbg_species"]).astype(int))
print("my batch     n_node", np.asarray(b.n_node), "species[:30]", np.asarray(b.nodes["species"][:30]))
print("pos0 lib", np.asarray(m2["dbg_pos0"]).round(4).tolist(), "mine", np.asarray(b.nodes["positions"][:3]).round(4).tolist())
