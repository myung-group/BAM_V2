"""cuEquivariance vs e3nn backend on the program the training uses (energy + forces under jit):
max |dE| per graph and max |dF| per component on the first batches of every test set of a run."""
import json, sys, io, functools
import numpy as np, jax, jax.numpy as jnp
from flax import nnx
from bam_omat24 import train_3bpa as T
from bam_omat24.data_nnx import default_collate_fn
from bam_omat24.training_utils import load_checkpoint
jax.config.update("jax_enable_x64", True)
for name in sys.argv[1:]:
    cfg = json.load(open(f"input/input_3bpa_omat24_{name}_cue.json"))
    f = io.StringIO()
    an, ae, tr, va, te, st = T.build_datasets(cfg, f)
    gd, *_ = T.build_model(cfg, len(an), ae, st, f)
    gd3, *_ = T.build_model(dict(cfg, tp_backend="e3nn"), len(an), ae, st, f)
    params = jax.tree.map(jnp.asarray, load_checkpoint(f"runs/{name}_cue/ckpt_{name}_cue.pkl")["ema_params"])
    bs = int(cfg.get("batch_size", 5)); nb, eb = int(cfg.get("node_bucket_size", 128)), int(cfg.get("edge_bucket_size", 4096))
    def ef(gdef, p, b):
        o = nnx.merge(gdef, p)(b); return o["energy"], o["forces"]
    f_cue, f_e3 = jax.jit(functools.partial(ef, gd)), jax.jit(functools.partial(ef, gd3))
    sets = {"train": tr, **te}
    for sname, ds in sets.items():
        graphs = T._cast_graphs(ds.graphs[:4 * bs], np.float64)
        dE, dF, Emag = 0.0, 0.0, 0.0
        for i in range(0, len(graphs), bs):
            b = default_collate_fn(graphs[i:i + bs], bs, nb, eb)
            ng = len(graphs[i:i + bs]); nn = int(np.sum(np.asarray(b.n_node)[:ng]))
            e1, F1 = f_cue(params, b); e2, F2 = f_e3(params, b)
            dE = max(dE, float(np.max(np.abs(np.asarray(e1)[:ng] - np.asarray(e2)[:ng]))))
            dF = max(dF, float(np.max(np.abs(np.asarray(F1)[:nn] - np.asarray(F2)[:nn]))))
            Emag = max(Emag, float(np.max(np.abs(np.asarray(e2)[:ng]))))
        flag = "MISMATCH" if (dE > 1e-6 * max(1.0, Emag) or dF > 1e-6) else "ok"
        print(f"{name:32s} {sname:14s} max|dE| {dE:.3e} eV  max|dF| {dF:.3e} eV/A  ({flag})", flush=True)
