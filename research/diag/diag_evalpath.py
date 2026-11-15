"""Same checkpoint through (a) the training-time evaluate_sharded on the valid split, (b) direct
collate + model at batch sizes 5 and 20.  usage: python runs/diag/diag_evalpath.py run"""
import json, sys, io, functools
import numpy as np, jax, jax.numpy as jnp
from flax import nnx
from bam_omat24 import train_3bpa as T
from bam_omat24.data_nnx import default_collate_fn, setup_mesh
from bam_omat24.training_utils import load_checkpoint, make_sharded_evaluate_step, evaluate_sharded, replicate_pytree
jax.config.update("jax_enable_x64", True)
run = sys.argv[1]
cfg = json.load(open(f"input/input_3bpa_omat24_{run}_cue.json"))
f = io.StringIO()
an, ae, tr, va, te, st = T.build_datasets(cfg, f)
for ds in [tr, va]:
    ds.graphs = T._cast_graphs(ds.graphs, np.float64)
gd, *_ = T.build_model(cfg, len(an), ae, st, f)
ck = load_checkpoint(f"runs/{run}_cue/ckpt_{run}_cue.pkl")
p = jax.tree.map(jnp.asarray, ck["ema_params"])
print("epoch", ck.get("epoch"), "atom_energies", np.round(ae, 4), "shift param", float(np.asarray(nnx.merge(gd, p).shift[...])))
mesh, nd = setup_mesh()
es = make_sharded_evaluate_step(mesh, huber_delta=10.0, loss_type="huber", periodic=False, cov_mode="scalar6", beta_nll=0.0)
col = functools.partial(default_collate_fn, node_bucket_size=16, edge_bucket_size=256)
m = evaluate_sharded(graphdef=gd, params=replicate_pytree(p, mesh), files_pkl=[va], batch_size=5, n_devices=nd, mesh=mesh,
                     energy_weight=1.0, force_weight=10.0, stress_weight=0.0, huber_delta=10.0, loss_type="huber",
                     eval_step=es, params_already_replicated=True, periodic=False, cov_mode="scalar6", beta_nll=0.0,
                     collate_fn=col)
print(f"evaluate_sharded valid: E RMSE {m['energy_rmse_total']*1e3:.2f} meV, F RMSE {m['force_rmse']*1e3:.2f}")
fn = jax.jit(lambda p, b: nnx.merge(gd, p)(b)["energy"])
for bs in (1, 5, 20):
    gs = va.graphs[:20]; dE = []
    for i in range(0, 20, bs):
        b = default_collate_fn(gs[i:i + bs], bs, 16, 256)
        e = np.asarray(fn(p, b))[:bs]
        dE += list(e - np.array([float(np.asarray(g.globals["energy"]).ravel()[0]) for g in gs[i:i + bs]]))
    print(f"direct bs {bs:2d}: mean dE {np.mean(dE)*1e3:12.2f} meV, sd {np.std(dE)*1e3:8.2f}")
