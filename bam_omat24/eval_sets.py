"""Evaluate a trained 3BPA-pipeline checkpoint on additional extxyz test sets (review §12).

    python -m bam_omat24.eval_sets <train config json> --ckpt <pkl> --xyz <file.xyz> [--xyz ...]
           [--group-key dihedrals --group-index 0] [--tag dih] [--batch-size 10]

* Energies are compared after subtracting the config's atom_energies (as in training).
* Files without force labels (e.g. dihedral / H-transfer scans) are evaluated on energies only;
  the force columns are then reported as NaN.
* --group-key splits a file by an ``atoms.info`` entry (optionally one index of a list value, e.g.
  the first dihedral angle beta of the 3BPA ``dihedrals`` field) and reports every group.
* Writes runs/<run>/eval_<tag>.json next to the checkpoint and prints E RMSE (meV) / F RMSE (meV/A).
"""
import argparse, functools, json, sys
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from flax import nnx
import ase.calculators.singlepoint

from bam_omat24 import train_3bpa as T
from bam_omat24.data_3bpa import load_frames, preprocess_graph, GraphListDataset, atomic_numbers_to_indices
from bam_omat24.data_nnx import setup_mesh, default_collate_fn
from bam_omat24.training_utils import (load_checkpoint, replicate_pytree, make_sharded_evaluate_step,
                                       evaluate_sharded)


def _frames_to_graphs(frames, atom_indices, atom_energies, cutoff):
    has_forces = all(("forces" in a.calc.results) if a.calc is not None else False for a in frames)
    graphs = []
    for a in frames:
        if a.calc is None:
            raise ValueError("frame without energy")
        if "forces" not in a.calc.results:
            a.calc.results["forces"] = np.zeros((len(a), 3))
        graphs.append(preprocess_graph(a, atom_indices, atom_energies, cutoff))
    return graphs, has_forces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--xyz", action="append", required=True)
    ap.add_argument("--group-key", default=None)
    ap.add_argument("--group-index", type=int, default=None)
    ap.add_argument("--tag", default="extra")
    ap.add_argument("--batch-size", type=int, default=10)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    if cfg.get("x64"):
        jax.config.update("jax_enable_x64", True)
    fout = sys.stdout
    (atomic_numbers, atom_energies, train_ds, valid_ds, test_sets, stats) = T.build_datasets(cfg, fout)
    graphdef, _p0, _opt, _sched, _gc = T.build_model(cfg, len(atomic_numbers), atom_energies, stats, fout)
    ckpt = load_checkpoint(args.ckpt)
    params = jax.tree.map(jnp.asarray, ckpt["ema_params"])
    mesh, n_devices = setup_mesh()
    params_rep = replicate_pytree(params, mesh)
    huber_delta = float(T._get(cfg, "huber_delta", 0.1)); loss_type = T._get(cfg, "loss_type", "huber")
    cov_mode = T._get(cfg, "cov_mode", "scalar6"); beta_nll = float(T._get(cfg, "beta_nll", 0.0))
    eval_step = make_sharded_evaluate_step(mesh, huber_delta=huber_delta, loss_type=loss_type, periodic=False,
                                           cov_mode=cov_mode, beta_nll=beta_nll,
                                           energy_loss_per=T._get(cfg, "energy_loss_per", "atom"))
    collate = functools.partial(default_collate_fn, node_bucket_size=int(T._get(cfg, "node_bucket_size", 128)),
                                edge_bucket_size=int(T._get(cfg, "edge_bucket_size", 4096)))
    atom_indices = atomic_numbers_to_indices(atomic_numbers)
    cutoff = float(T._get(cfg, "cutoff", 5.0))

    def evaluate(graphs):
        if cfg.get("x64"):
            graphs = T._cast_graphs(graphs, np.float64)
        return evaluate_sharded(
            graphdef=graphdef, params=params_rep, files_pkl=[GraphListDataset(graphs)],
            batch_size=args.batch_size, n_devices=n_devices, mesh=mesh,
            energy_weight=1.0, force_weight=1.0, stress_weight=0.0, huber_delta=huber_delta,
            loss_type=loss_type, eval_step=eval_step, params_already_replicated=True, periodic=False,
            cov_mode=cov_mode, beta_nll=beta_nll, collate_fn=collate)

    report = {"checkpoint": args.ckpt, "epoch": ckpt.get("epoch"), "sets": {}}
    print(f"checkpoint {args.ckpt} (epoch {ckpt.get('epoch')})")
    print(f"{'set':34s} {'n':>5s} | {'E_RMSE meV':>10s} {'E_MAE':>8s} | {'F_RMSE meV/A':>12s} {'F_MAE':>8s}")
    for xyz in args.xyz:
        frames = load_frames(xyz)
        # decide once per file whether force labels exist (the graph builder adds zero forces to
        # energy-only frames, which must not flip this flag for later groups)
        file_has_forces = all((a.calc is not None) and ("forces" in a.calc.results) for a in frames)
        groups = {"all": frames}
        if args.group_key:
            groups = {}
            for a in frames:
                v = a.info.get(args.group_key)
                if args.group_index is not None and v is not None:
                    v = v[args.group_index]
                groups.setdefault(str(v), []).append(a)
        for gname, fr in ([("all", frames)] if args.group_key else []) + sorted(groups.items()):
            if args.group_key and gname == "all" and len(groups) == 1:
                continue
            graphs, _ = _frames_to_graphs(fr, atom_indices, atom_energies, cutoff)
            has_forces = file_has_forces
            m = evaluate(graphs)
            e_rmse, e_mae = m["energy_rmse_total"] * 1e3, m["energy_mae_total"] * 1e3
            f_rmse = m["force_rmse"] * 1e3 if has_forces else float("nan")
            f_mae = m["force_mae"] * 1e3 if has_forces else float("nan")
            label = f"{Path(xyz).name}" + (f"[{args.group_key}={gname}]" if args.group_key else "")
            print(f"{label:34s} {len(fr):5d} | {e_rmse:10.2f} {e_mae:8.2f} | {f_rmse:12.2f} {f_mae:8.2f}")
            report["sets"][label] = dict(n=len(fr), energy_rmse_meV=e_rmse, energy_mae_meV=e_mae,
                                         force_rmse_meVA=f_rmse, force_mae_meVA=f_mae, has_forces=has_forces)
    out = Path(args.ckpt).parent / f"eval_{args.tag}.json"
    json.dump(report, open(out, "w"), indent=1)
    print("written", out)


if __name__ == "__main__":
    main()
