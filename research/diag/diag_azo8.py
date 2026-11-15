"""Does the result depend on what the compiled program returns / on the TP backend?"""
import json, sys, io
import numpy as np, jax, jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax import shard_map
from flax import nnx
from race_v2.training import train as T
from race_v2.data.loaders import default_collate_fn, setup_mesh
from race_v2.training.utils import (load_checkpoint, replicate_pytree, squeeze_batch, _loss_terms, make_sharded_evaluate_step)
cfg = json.load(open(sys.argv[1])); ck = sys.argv[2]
jax.config.update("jax_enable_x64", True)
f = io.StringIO()
an, ae, tr, va, te, st = T.build_datasets(cfg, f)
tr.graphs = T._cast_graphs(tr.graphs, np.float64)
gd, *_ = T.build_model(cfg, len(an), ae, st, f)
gd3, *_ = T.build_model(dict(cfg, tp_backend="e3nn"), len(an), ae, st, f)
params = jax.tree.map(jnp.asarray, load_checkpoint(ck)["ema_params"])
mesh, nd = setup_mesh(); prep = replicate_pytree(params, mesh)
b = default_collate_fn(tr.graphs[:5], 5, 16, 256); bb = jax.tree.map(lambda x: np.asarray(x)[None], b)
def rms(se): return float(np.sqrt(se / 5))
for name, g in (("cue", gd), ("e3nn", gd3)):
    es = make_sharded_evaluate_step(mesh, huber_delta=10.0, loss_type="huber", periodic=False, cov_mode="scalar6", beta_nll=0.0)
    print(f"{name:5s} library eval_step E RMSE {rms(float(es(g, prep, bb)['energy_se_total'])):.3f}")
    def only_se(gdef, p, batch):
        batch = squeeze_batch(batch); res = nnx.merge(gdef, p)(batch)
        return _loss_terms(res, batch, 10.0, "huber", False, "scalar6", 0.0)["energy_se_total"]
    def se_and_e(gdef, p, batch):
        batch = squeeze_batch(batch); res = nnx.merge(gdef, p)(batch)
        return _loss_terms(res, batch, 10.0, "huber", False, "scalar6", 0.0)["energy_se_total"], res["energy"]
    def se_and_f(gdef, p, batch):
        batch = squeeze_batch(batch); res = nnx.merge(gdef, p)(batch)
        return _loss_terms(res, batch, 10.0, "huber", False, "scalar6", 0.0)["energy_se_total"], res["forces"]
    for fn_name, fn in (("only_se", only_se), ("se+energy", se_and_e), ("se+forces", se_and_f)):
        out = jax.jit(shard_map(fn, mesh=mesh, check_vma=False, in_specs=(P(), P(), P('dp')), out_specs=P()))(g, prep, bb)
        se = out if not isinstance(out, tuple) else out[0]
        print(f"{name:5s} shard_map {fn_name:10s} E RMSE {rms(float(se)):.3f}")
        out = jax.jit(fn)(g, params, b)
        se = out if not isinstance(out, tuple) else out[0]
        print(f"{name:5s} plain jit {fn_name:10s} E RMSE {rms(float(se)):.3f}")
