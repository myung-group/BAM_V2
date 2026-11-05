"""Per-stratum (per-layer = per-body-order) energy / force contributions of a trained RACE.

For each dataset (train, valid, test_300K/600K/1200K) reports, per stratum s:
  * RMS over graphs of the stratum total energy  E_s                      [meV]
  * RMS over atoms  of the stratum force          |F_s|                   [meV/A]
  * the error change when stratum s is removed:  RMSE(F - F_s vs F_ref)   [meV/A]
plus the full-model errors, so one can see which body order carries the
error growth from 300 K to 1200 K.

Usage:
    python -m bam_omat24.analyze_strata <train config json> [--ckpt path] [--max-frames N] [--out json]
"""
import argparse
import functools
import json
import sys
from pathlib import Path

import jax
jax.devices()
import jax.numpy as jnp
import numpy as np
from flax import nnx

from bam_omat24.data_3bpa import load_3bpa_graphs, split_train_valid, GraphListDataset
from bam_omat24.data_nnx import BucketedDataLoader, default_collate_fn
from bam_omat24.training_utils import load_checkpoint
from bam_omat24 import train_3bpa as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ema", action="store_true", default=True)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    if cfg.get("x64"):
        jax.config.update("jax_enable_x64", True)      # train_3bpa switches x64 off on import
    ckpt_path = args.ckpt or cfg.get("ckpt_name", "ckpt_best.pkl")
    ckpt = load_checkpoint(ckpt_path)
    params = ckpt["ema_params"] if args.ema else ckpt["params"]
    params = jax.tree.map(jnp.asarray, params)   # checkpoints hold numpy arrays

    fout = sys.stdout
    (atomic_numbers, atom_energies, train_ds, valid_ds, test_sets, stats) = T.build_datasets(cfg, fout)
    if cfg.get("x64"):
        for ds in [train_ds, valid_ds] + list(test_sets.values()):
            ds.graphs = T._cast_graphs(ds.graphs, np.float64)
    graphdef, _params0, _opt, _sched, _gc = T.build_model(cfg, len(atomic_numbers), atom_energies, stats, fout)
    model = nnx.merge(graphdef, params)
    print(f"checkpoint: {ckpt_path} (epoch {ckpt.get('epoch')}, val loss {ckpt.get('best_val_loss'):.3e})")

    collate = functools.partial(default_collate_fn, node_bucket_size=16, edge_bucket_size=256)

    @jax.jit
    def strata_step(batch):
        E_s, F_s = model.strata_energies_forces(batch)         # (S, G), (S, N, 3)
        return E_s, F_s

    datasets = {"train": train_ds, "valid": valid_ds, **test_sets}
    report = {}
    for name, ds in datasets.items():
        graphs = ds.graphs[: args.max_frames]
        loader = BucketedDataLoader(GraphListDataset(graphs), batch_size=args.batch_size, n_buckets=1,
                                    shuffle=False, drop_last=False, rngs=nnx.Rngs(0))
        loader.collate_fn = collate
        E_list, F_list, Eref, Fref, natoms = [], [], [], [], []
        for batch in loader:
            import jraph
            gmask = np.asarray(jraph.get_graph_padding_mask(batch))
            nmask = np.asarray(jraph.get_node_padding_mask(batch))
            E_s, F_s = strata_step(batch)
            E_s = np.asarray(E_s)[:, gmask]; F_s = np.asarray(F_s)[:, nmask]
            E_list.append(E_s); F_list.append(F_s)
            Eref.append(np.asarray(batch.globals["energy"])[gmask]); Fref.append(np.asarray(batch.nodes["forces"])[nmask])
            natoms.append(np.asarray(batch.n_node)[gmask])
        E_s = np.concatenate(E_list, axis=1); F_s = np.concatenate(F_list, axis=1)
        Eref = np.concatenate(Eref); Fref = np.concatenate(Fref); natoms = np.concatenate(natoms)
        S = E_s.shape[0]
        E_tot = E_s.sum(0) + float(np.asarray(model.shift[...])) * natoms
        F_tot = F_s.sum(0)
        e_rmse = np.sqrt(np.mean((E_tot - Eref) ** 2)) * 1e3
        f_rmse = np.sqrt(np.mean((F_tot - Fref) ** 2)) * 1e3
        rows = []
        for s in range(S):
            e_rms = np.sqrt(np.mean((E_s[s] - E_s[s].mean()) ** 2)) * 1e3        # fluctuation of stratum energy
            f_rms = np.sqrt(np.mean(F_s[s] ** 2)) * 1e3
            f_rmse_wo = np.sqrt(np.mean((F_tot - F_s[s] - Fref) ** 2)) * 1e3
            rows.append(dict(stratum=s, E_rms_meV=float(e_rms), F_rms_meVA=float(f_rms), F_rmse_without_meVA=float(f_rmse_wo)))
        report[name] = dict(n_frames=int(len(Eref)), E_rmse_meV=float(e_rmse), F_rmse_meVA=float(f_rmse), strata=rows)
        print(f"\n[{name}] n={len(Eref)}  full model: E_RMSE {e_rmse:7.2f} meV   F_RMSE {f_rmse:7.2f} meV/A")
        print(f"  {'stratum':>7s} {'E_fluct[meV]':>13s} {'|F_s| rms[meV/A]':>17s} {'F_RMSE w/o s':>13s}")
        for r in rows:
            print(f"  {r['stratum']:>7d} {r['E_rms_meV']:13.2f} {r['F_rms_meVA']:17.2f} {r['F_rmse_without_meVA']:13.2f}")
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
