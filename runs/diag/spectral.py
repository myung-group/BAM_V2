"""Spectral radius rho and operator norm sigma_max of each layer's linear map h -> M_t(R) h
(update "linear": the layer output for fixed geometry, prev_features=None, padding nodes masked),
by power iteration on M (rho: mean log growth of the last 10 of 40 steps) and on M^T M via vjp.
Init and trained parameters, 5 training frames.  usage: python runs/diag/spectral.py run [run ...]"""
import json, sys, io
import numpy as np, jax, jax.numpy as jnp, jraph
import e3nn_jax as e3nn
from flax import nnx
import bam_omat24.race_nnx as R
from bam_omat24 import train_3bpa as T
from bam_omat24.data_nnx import default_collate_fn
from bam_omat24.training_utils import load_checkpoint
jax.config.update("jax_enable_x64", True)
CAP = []
orig = R.RACEConvolution.__call__
def cap(self, *a, **k):
    CAP.append((self, a, k)); return orig(self, *a, **k)
R.RACEConvolution.__call__ = cap

def spectra(model, b, nmask):
    CAP.clear(); model.node_energies(b.nodes["positions"], None, None, b)   # forward only (no grad tracers)
    out = []
    for i, (lay, a, k) in enumerate(CAP[1:], start=1):
        irr = a[0].irreps
        kw = dict(k, prev_features=None)
        f = jax.jit(lambda h: orig(lay, e3nn.IrrepsArray(irr, h), *a[1:], **kw)[1].array * nmask[:, None])
        v = jax.random.normal(jax.random.PRNGKey(i), a[0].array.shape, dtype=a[0].array.dtype) * nmask[:, None]
        logs = []
        for it in range(40):
            w = f(v); n = jnp.linalg.norm(w) / jnp.linalg.norm(v)
            logs.append(float(jnp.log(n))); v = w / jnp.linalg.norm(w)
        rho = float(np.exp(np.mean(logs[-10:])))
        u = v
        for it in range(30):
            y, vjp = jax.vjp(f, u); z = vjp(y)[0]
            s2 = float(jnp.vdot(z, u) / jnp.vdot(u, u)); u = z / jnp.linalg.norm(z)
        out.append((rho, float(np.sqrt(max(s2, 0.0)))))
    return out

for run in sys.argv[1:]:
    cfg = json.load(open(f"input/input_3bpa_omat24_{run}_cue.json"))
    f = io.StringIO()
    an, ae, tr, va, te, st = T.build_datasets(cfg, f)
    gd, p0, *_ = T.build_model(cfg, len(an), ae, st, f)
    b = default_collate_fn(T._cast_graphs(tr.graphs[:5], np.float64), 5, 16, 256)
    nmask = jnp.asarray(jraph.get_node_padding_mask(b), dtype=jnp.float64)
    sets = [("init", p0)]
    try:
        ck = load_checkpoint(f"runs/{run}_cue/ckpt_{run}_cue.pkl")
        sets.append((f"trained ep {ck.get('epoch')}", jax.tree.map(jnp.asarray, ck["ema_params"])))
    except Exception as ex:
        print(f"{run}: no checkpoint ({str(ex)[:60]})")
    for tag, p in sets:
        sp = spectra(nnx.merge(gd, p), b, nmask)
        print(f"{run:28s} {tag:16s} rho(M_t) {' '.join(f'{r:9.3g}' for r, _ in sp)} | sigma_max {' '.join(f'{s:9.3g}' for _, s in sp)}", flush=True)
