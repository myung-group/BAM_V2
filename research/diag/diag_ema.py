"""Raw vs EMA parameters: E / F errors on train and valid frames (energy+forces program).
usage: python research/diag/diag_ema.py run [run ...]"""
import json, sys, io
import numpy as np, jax, jax.numpy as jnp
from flax import nnx
from race_v2.training import train as T
from race_v2.data.loaders import default_collate_fn
from race_v2.training.utils import load_checkpoint
jax.config.update("jax_enable_x64", True)
for run in sys.argv[1:]:
    cfg = json.load(open(f"research/input/input_3bpa_omat24_{run}_cue.json"))
    f = io.StringIO()
    an, ae, tr, va, te, st = T.build_datasets(cfg, f)
    gd, *_ = T.build_model(cfg, len(an), ae, st, f)
    ck = load_checkpoint(f"runs/{run}_cue/ckpt_{run}_cue.pkl")
    fn = jax.jit(lambda p, b: (lambda o: (o["energy"], o["forces"]))(nnx.merge(gd, p)(b)))
    print(f"== {run}: epoch {ck.get('epoch')}")
    for which in ("params", "ema_params"):
        p = jax.tree.map(jnp.asarray, ck[which])
        for name, ds in (("train", tr), ("valid", va)):
            gs = T._cast_graphs(ds.graphs[:20], np.float64)
            b = default_collate_fn(gs, 20, 16, 256)
            n = int(sum(int(g.n_node[0]) for g in gs))
            tgt = np.array([float(np.asarray(g.globals["energy"]).ravel()[0]) for g in gs])
            e, fr = fn(p, b); e = np.asarray(e)[:20]; fr = np.asarray(fr)[:n]
            fref = np.concatenate([np.asarray(g.nodes["forces"]) for g in gs])
            de = e - tgt
            print(f"  {which:10s} {name}: E RMSE {np.sqrt(np.mean(de**2))*1e3:10.2f} meV  mean dE {np.mean(de)*1e3:10.2f}  "
                  f"sd dE {np.std(de)*1e3:8.2f} | F RMSE {np.sqrt(np.mean((fr-fref)**2))*1e3:7.2f} meV/A", flush=True)
