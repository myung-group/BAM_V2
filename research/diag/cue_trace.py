"""Record every cue edge-TP call (inputs and output norms) in the forward-only and in the
forward+grad program of the full model, via jax.debug.callback."""
import json, sys, io, functools
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
    jax.debug.callback(lambda a, b, c, d: LOG.append((float(a), float(b), float(c), float(d))),
                       jnp.sum(node_feats ** 2), jnp.sum(sh ** 2), jnp.sum(weights ** 2), jnp.sum(out.array ** 2))
    return out
R.cue_edge_tensor_product = traced
mol = sys.argv[1]
cfg = json.load(open(f"input/input_3bpa_omat24_rmd17_{mol}_v9a_s42_cue.json"))
f = io.StringIO()
an, ae, tr, va, te, st = T.build_datasets(cfg, f)
gd, *_ = T.build_model(cfg, len(an), ae, st, f)
params = jax.tree.map(jnp.asarray, load_checkpoint(f"runs/rmd17_{mol}_v9a_s42_cue/ckpt_rmd17_{mol}_v9a_s42_cue.pkl")["ema_params"])
b = default_collate_fn(T._cast_graphs(tr.graphs[:5], np.float64), 5, 16, 256)
def fwd(p, b):
    return nnx.merge(gd, p).node_energies(b.nodes["positions"], None, None, b)["node_energies"]
def withgrad(p, b):
    m = nnx.merge(gd, p)
    def tot(pos):
        ne = m.node_energies(pos, None, None, b)["node_energies"]; return jnp.sum(ne), ne
    (_, ne), g = jax.value_and_grad(tot, has_aux=True)(b.nodes["positions"]); return ne, g
LOG.clear(); jax.block_until_ready(jax.jit(fwd)(params, b)); L1 = list(LOG)
LOG.clear(); jax.block_until_ready(jax.jit(withgrad)(params, b)); L2 = list(LOG)
print(f"{mol}: {len(L1)} cue calls in fwd, {len(L2)} in fwd+grad (|x|^2, |sh|^2, |w|^2, |out|^2)")
for i in range(max(len(L1), len(L2))):
    a = L1[i] if i < len(L1) else None; c = L2[i] if i < len(L2) else None
    print(f"  call {i}: fwd {None if a is None else tuple(f'{v:.6e}' for v in a)}\n          grad {None if c is None else tuple(f'{v:.6e}' for v in c)}")
