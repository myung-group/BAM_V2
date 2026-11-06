"""Per-frame energy / force errors of a checkpoint on extxyz files -> .npz for offline analysis.

    python -m bam_omat24.dump_frame_errors <train config json> --ckpt <pkl> --xyz <file.xyz> [--xyz ...]
           --out <prefix> [--batch-size 32] [--index-key qm9_index]

Writes <prefix>_<file stem>.npz with, per frame (file order): e_pred, e_ref (eV, both after the config's
atom_energies were subtracted, EMA parameters), natoms, counts (n_frames, n_species; species in the order
of the config's atomic_numbers), index (atoms.info[index_key] or -1), f_rmse (per-frame force RMSE, eV/A;
NaN when the file has no force labels).  With --per-atom (fixed-size frames only): f_pred, f_ref (n_frames, N, 3).
"""
import argparse, functools, json, sys
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp, jraph
from flax import nnx

from bam_omat24 import train_3bpa as T
from bam_omat24.data_3bpa import load_frames, GraphListDataset, atomic_numbers_to_indices
from bam_omat24.data_nnx import BucketedDataLoader, default_collate_fn
from bam_omat24.training_utils import load_checkpoint
from bam_omat24.eval_sets import _frames_to_graphs
from bam_omat24.analyze_errors import _tag_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--xyz", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--index-key", default="qm9_index")
    ap.add_argument("--per-atom", action="store_true")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    if cfg.get("x64"):
        jax.config.update("jax_enable_x64", True)
    (atomic_numbers, atom_energies, _tr, _va, _te, stats) = T.build_datasets(cfg, sys.stdout)
    graphdef, _p0, _opt, _sched, _gc = T.build_model(cfg, len(atomic_numbers), atom_energies, stats, sys.stdout)
    ckpt = load_checkpoint(args.ckpt)
    model = nnx.merge(graphdef, jax.tree.map(jnp.asarray, ckpt["ema_params"]))
    collate = functools.partial(default_collate_fn, node_bucket_size=int(T._get(cfg, "node_bucket_size", 128)),
                                edge_bucket_size=int(T._get(cfg, "edge_bucket_size", 4096)))
    atom_indices = atomic_numbers_to_indices(atomic_numbers)
    cutoff = float(T._get(cfg, "cutoff", 5.0))
    n_species = len(atomic_numbers)

    @jax.jit
    def step(batch):
        r = model(batch)
        return r["energy"].reshape(-1), r["forces"]

    for xyz in args.xyz:
        frames = load_frames(xyz)
        graphs, has_forces = _frames_to_graphs(frames, atom_indices, atom_energies, cutoff,
                                               charge_key=T._get(cfg, "total_charge_key", None),
                                               spin_key=T._get(cfg, "total_spin_key", None))
        if cfg.get("x64"):
            graphs = T._cast_graphs(graphs, np.float64)
        graphs = _tag_frames(graphs)
        loader = BucketedDataLoader(GraphListDataset(graphs), batch_size=args.batch_size, n_buckets=1,
                                    shuffle=False, drop_last=False, rngs=nnx.Rngs(0))
        loader.collate_fn = collate
        n = len(frames)
        e_pred = np.full(n, np.nan); e_ref = np.full(n, np.nan); f_rmse = np.full(n, np.nan)
        f_pred = np.full((n, len(frames[0]), 3), np.nan) if args.per_atom else None
        f_ref = np.full_like(f_pred, np.nan) if args.per_atom else None
        for batch in loader:
            gmask = np.asarray(jraph.get_graph_padding_mask(batch))
            nmask = np.asarray(jraph.get_node_padding_mask(batch))
            E, F = step(batch)
            fid = np.asarray(batch.globals["frame_id"]).reshape(-1)
            e_pred[fid[gmask]] = np.asarray(E)[gmask]
            e_ref[fid[gmask]] = np.asarray(batch.globals["energy"]).reshape(-1)[gmask]
            if has_forces:
                gidx = np.repeat(np.arange(batch.n_node.shape[0]), np.asarray(batch.n_node))
                se = np.sum((np.asarray(F) - np.asarray(batch.nodes["forces"])) ** 2, -1) * nmask
                per_g = np.bincount(gidx, weights=se, minlength=batch.n_node.shape[0])
                cnt = np.bincount(gidx, weights=nmask.astype(float), minlength=batch.n_node.shape[0])
                f_rmse[fid[gmask]] = np.sqrt(per_g[gmask] / (3.0 * np.maximum(cnt[gmask], 1)))
            if args.per_atom:
                N = f_pred.shape[1]
                keep = np.flatnonzero(nmask)
                f_pred[fid[gmask]] = np.asarray(F)[keep].reshape(-1, N, 3)
                f_ref[fid[gmask]] = np.asarray(batch.nodes["forces"])[keep].reshape(-1, N, 3)
        natoms = np.array([len(a) for a in frames])
        counts = np.array([[int(np.sum(a.numbers == z)) for z in atomic_numbers] for a in frames])
        index = np.array([int(a.info.get(args.index_key, -1)) for a in frames])
        out = f"{args.out}_{Path(xyz).stem}.npz"
        extra = dict(f_pred=f_pred, f_ref=f_ref) if args.per_atom else {}
        np.savez(out, e_pred=e_pred, e_ref=e_ref, natoms=natoms, counts=counts, index=index, f_rmse=f_rmse,
                 atomic_numbers=np.array(atomic_numbers), **extra)
        err = (e_pred - e_ref) * 1e3
        print(f"{Path(xyz).name}: n={n} E MAE {np.mean(np.abs(err)):.2f} meV RMSE {np.sqrt(np.mean(err**2)):.2f} meV"
              + (f" | F RMSE {np.sqrt(np.nanmean(f_rmse**2))*1e3:.1f} meV/A" if has_forces else "") + f" -> {out}")


if __name__ == "__main__":
    main()
