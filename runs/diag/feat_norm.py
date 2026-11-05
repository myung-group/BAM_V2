"""Per-layer feature norms (|x|^2 into every cue edge TP, forward only) for rMD17 molecules, at the
initial parameters and at a trained checkpoint: is the azobenzene explosion present at init?
usage: python runs/diag/feat_norm.py mol[:run] ...   (run defaults to rmd17_<mol>_v9a_s42)"""
import json, sys, io
import numpy as np, jax, jax.numpy as jnp
from flax import nnx
import bam_omat24.race_nnx as R
from bam_omat24 import train_3bpa as T
from bam_omat24.data_nnx import default_collate_fn
from bam_omat24.training_utils import load_checkpoint
jax.config.update("jax_enable_x64", True)
LOG = []
orig = R.cue_edge_tensor_product
def traced(irreps_in, sh_irreps, irreps_out, node_feats, sh, weights, senders, receivers, method="uniform_1d"):
    out = orig(irreps_in, sh_irreps, irreps_out, node_feats, sh, weights, senders, receivers, method)
    jax.debug.callback(lambda a, c: LOG.append((float(a), float(c))), jnp.sum(node_feats ** 2) / node_feats.shape[0],
                       jnp.sum(out.array ** 2) / out.array.shape[0])
    return out
R.cue_edge_tensor_product = traced
for arg in sys.argv[1:]:
    mol, run = (arg.split(":") + [None])[:2]
    run = run or f"rmd17_{mol}_v9a_s42"
    cfg = json.load(open(f"input/input_3bpa_omat24_{run}_cue.json"))
    f = io.StringIO()
    an, ae, tr, va, te, st = T.build_datasets(cfg, f)
    gd, p0, *_ = T.build_model(cfg, len(an), ae, st, f)
    gs = T._cast_graphs(tr.graphs[:5], np.float64)
    b = default_collate_fn(gs, 5, 16, 256)
    n = int(sum(int(g.n_node[0]) for g in gs))
    tgt = np.array([float(np.asarray(g.globals["energy"]).ravel()[0]) for g in gs])
    try:
        ck = load_checkpoint(f"runs/{run}_cue/ckpt_{run}_cue.pkl")
        trained = jax.tree.map(jnp.asarray, ck["ema_params"]); ep = ck.get("epoch")
    except Exception as ex:
        trained, ep = None, str(ex)[:80]
    for tag, p in (("init", p0), (f"trained ep {ep}", trained)):
        if p is None:
            print(f"{run} {tag}: no checkpoint"); continue
        LOG.clear()
        e = np.asarray(jax.jit(lambda p, b: nnx.merge(gd, p)(b)["energy"])(p, b))[:5]
        jax.effects_barrier()
        xs = " ".join(f"{a:.2e}" for a, _ in LOG); os_ = " ".join(f"{c:.2e}" for _, c in LOG)
        print(f"{run:34s} {tag:16s} atoms/graph {n // 5:3d} | per-node |x|^2 by call: {xs}\n{'':51s} per-edge |out|^2: {os_}"
              f"\n{'':51s} E - target (eV): {np.round(e - tgt, 3)}", flush=True)
