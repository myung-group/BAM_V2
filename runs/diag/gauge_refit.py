"""Closed-form refit (single-stoichiometry datasets: the per-frame constant) of the force-null energy constant (single stoichiometry: per-frame offset) on the
TRAINING split only, then test RMSE before/after.  usage: python runs/diag/gauge_refit.py run [run ...]"""
import json, sys, io
import numpy as np, jax, jax.numpy as jnp
from flax import nnx
from bam_omat24 import train_3bpa as T
from bam_omat24.data_nnx import default_collate_fn
from bam_omat24.training_utils import load_checkpoint
import jraph

BS = 10
for run in sys.argv[1:]:
    cfg = json.load(open(f"input/input_3bpa_omat24_{run}_cue.json"))
    if cfg.get("x64"):
        jax.config.update("jax_enable_x64", True)
    f = io.StringIO()
    an, ae, tr, va, te, st = T.build_datasets(cfg, f)
    gd, *_ = T.build_model(cfg, len(an), ae, st, f)
    ck = load_checkpoint(cfg["ckpt_name"]); params = jax.tree.map(jnp.asarray, ck["ema_params"])
    step = jax.jit(lambda p, b: nnx.merge(gd, p)(b))

    def pred(graphs):
        graphs = T._cast_graphs(graphs, np.float64 if cfg.get("x64") else np.float32)
        E, Er, F, Fr = [], [], [], []
        for k in range(0, len(graphs), BS):
            gs = graphs[k:k + BS]
            b = default_collate_fn(gs, BS, 16, 256)
            out = step(params, b)
            gm = np.asarray(jraph.get_graph_padding_mask(b)); nm = np.asarray(jraph.get_node_padding_mask(b))
            E.append(np.asarray(out["energy"])[gm]); Er.append(np.asarray(b.globals["energy"]).reshape(-1)[gm])
            F.append(np.asarray(out["forces"])[nm]); Fr.append(np.asarray(b.nodes["forces"])[nm])
        return (np.concatenate(E) - np.concatenate(Er)) * 1e3, (np.concatenate(F) - np.concatenate(Fr)) * 1e3

    dtr, _ = pred(tr.graphs)
    c = -dtr.mean()                                   # meV per frame, from the training split only
    rows = [f"== {run} (epoch {ck.get('epoch')}): train offset {dtr.mean():+.2f} meV (sd {dtr.std():.2f}), correction c = {c:+.2f} meV"]
    for name, ds in list(te.items()) + [("valid", va)]:
        dE, dF = pred(ds.graphs)
        rows.append(f"  {name:>12s}: E RMSE {np.sqrt((dE**2).mean()):6.2f} -> {np.sqrt(((dE + c)**2).mean()):6.2f}  "
                    f"(mean {dE.mean():+.2f}, centred sd {dE.std():.2f}) | F RMSE {np.sqrt((dF**2).mean()):6.2f}")
    print("\n".join(rows), flush=True)
    out = {"shift_delta_meV_per_frame": float(c)}
    for name, ds in list(te.items()) + [("valid", va)]:
        dE, dF = pred(ds.graphs)
        out[name] = {"energy_rmse_total": float(np.sqrt(((dE + c) ** 2).mean())) * 1e-3,
                     "force_rmse": float(np.sqrt((dF ** 2).mean())) * 1e-3}
    json.dump(out, open(f"runs/{run}_cue/gauge_refit.json", "w"), indent=1)
