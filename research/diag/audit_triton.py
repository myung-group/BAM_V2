"""Triton-GEMM audit (review §12.4): for each run, the valid / test metrics through the training-time
evaluate_sharded and the train-step loss + parameter gradient (norm and a fixed random projection) on
the first 40 training batches (batch 5).  Prints one JSON line per run; run once with and once
without XLA_FLAGS=--xla_gpu_enable_triton_gemm=false and diff.  usage: audit_triton.py run [run ...]"""
import json, sys, io, functools
import numpy as np, jax, jax.numpy as jnp
from flax import nnx
from bam_omat24 import train_3bpa as T
from bam_omat24.data_nnx import default_collate_fn, setup_mesh
from bam_omat24.training_utils import (load_checkpoint, make_sharded_evaluate_step, evaluate_sharded,
                                       replicate_pytree, compute_loss)
jax.config.update("jax_enable_x64", True)
mesh, nd = setup_mesh()
for run in sys.argv[1:]:
    cfg = json.load(open(f"input/input_3bpa_omat24_{run}_cue.json"))
    f = io.StringIO()
    an, ae, tr, va, te, st = T.build_datasets(cfg, f)
    for ds in [tr, va] + list(te.values()):
        ds.graphs = T._cast_graphs(ds.graphs, np.float64)
    gd, *_ = T.build_model(cfg, len(an), ae, st, f)
    ck = load_checkpoint(f"runs/{run}_cue/ckpt_{run}_cue.pkl")
    p = jax.tree.map(jnp.asarray, ck["ema_params"])
    hd = float(cfg.get("huber_delta", 10.0)); per = cfg.get("energy_loss_per", "atom")
    nb, eb = int(cfg.get("node_bucket_size", 16)), int(cfg.get("edge_bucket_size", 256))
    es = make_sharded_evaluate_step(mesh, huber_delta=hd, loss_type="huber", periodic=False, cov_mode="scalar6",
                                    beta_nll=0.0, energy_loss_per=per)
    col = functools.partial(default_collate_fn, node_bucket_size=nb, edge_bucket_size=eb)
    out = {"run": run, "epoch": ck.get("epoch")}
    for name, ds in [("valid", va)] + list(te.items()):
        m = evaluate_sharded(graphdef=gd, params=replicate_pytree(p, mesh), files_pkl=[ds], batch_size=5, n_devices=nd,
                             mesh=mesh, energy_weight=1.0, force_weight=10.0, stress_weight=0.0, huber_delta=hd,
                             loss_type="huber", eval_step=es, params_already_replicated=True, periodic=False,
                             cov_mode="scalar6", beta_nll=0.0, collate_fn=col)
        out[name] = [round(m["energy_rmse_total"] * 1e3, 6), round(m["force_rmse"] * 1e3, 6)]
    vg = jax.jit(jax.value_and_grad(lambda q, b: compute_loss(gd, q, b, float(cfg.get("energy_weight", 1.0)),
                                                              float(cfg.get("force_weight", 10.0)), 0.0, hd,
                                                              periodic=False, energy_loss_per=per)[0]))
    leaves = jax.tree.leaves(p)
    keys = [jax.random.normal(jax.random.PRNGKey(i), x.shape, dtype=x.dtype) for i, x in enumerate(leaves)]
    L, G, PR = [], [], []
    for i in range(0, min(200, len(tr.graphs)), 5):
        b = col(tr.graphs[i:i + 5], 5)
        loss, g = vg(p, b)
        gl = jax.tree.leaves(g)
        L.append(float(loss)); G.append(float(jnp.sqrt(sum(jnp.sum(x * x) for x in gl))))
        PR.append(float(sum(jnp.sum(x * k) for x, k in zip(gl, keys))))
    out["train_loss"] = [float(np.sum(L)), float(np.max(L))]
    out["grad_norm"] = [float(np.mean(G)), float(np.max(G))]
    out["grad_proj"] = PR
    print("AUDIT " + json.dumps(out), flush=True)
