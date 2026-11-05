"""Stage-two full-batch L-BFGS refinement of a trained 3BPA-pipeline checkpoint (review §12.3; the
GRACE / linear-ACE fitting recipe: squared-error loss, large force weight, L-BFGS-B, float64).

    python -m bam_omat24.lbfgs_refine <train config json> --ckpt <pkl> [--iters 200] [--batch-size 10]
           [--energy-weight 1] [--force-weight 500] [--l2 0] [--memory 20] [--eval-every 10] [--out <pkl>]

* Loss = wE * mean_graphs Huber_E + wF * mean_components Huber_F (+ l2 * ||theta||^2), the same
  ``_loss_terms`` as training (Huber delta from the config: delta = 10 is quadratic for every
  realistic residual), accumulated over the whole training split and minimised with
  ``optax.lbfgs`` (zoom line search) on one device.  The batches are padded to one shape and stacked,
  the full-batch loss is a ``lax.scan`` over them (each batch rematerialised), and one L-BFGS step
  (value/grad + line search) is a single jitted program: a Python sum over batches would be re-traced
  inside the line-search ``while_loop`` as one program per batch.
* Starts from the EMA parameters of the checkpoint; valid / test metrics are printed every
  ``--eval-every`` iterations; the parameters with the lowest validation loss are written to
  ``--out`` (default: ckpt_lbfgs_<run>.pkl next to the input checkpoint) in the training
  checkpoint format (``params`` = ``ema_params``).
"""
import argparse, functools, json, sys, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import jraph
import optax
from flax import nnx

from bam_omat24 import train_3bpa as T
from bam_omat24.data_nnx import setup_mesh, default_collate_fn, nearest_bucket
from bam_omat24.training_utils import (_loss_terms, load_checkpoint, save_checkpoint, replicate_pytree,
                                       make_sharded_evaluate_step, evaluate_sharded)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--energy-weight", type=float, default=1.0)
    ap.add_argument("--force-weight", type=float, default=500.0)
    ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--memory", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=10)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    if cfg.get("x64"):
        jax.config.update("jax_enable_x64", True)
    fout = sys.stdout
    (atomic_numbers, atom_energies, train_ds, valid_ds, test_sets, stats) = T.build_datasets(cfg, fout)
    if cfg.get("x64"):
        for ds in [train_ds, valid_ds] + list(test_sets.values()):
            ds.graphs = T._cast_graphs(ds.graphs, np.float64)
    graphdef, _p0, _opt, _sched, _gc = T.build_model(cfg, len(atomic_numbers), atom_energies, stats, fout)
    ckpt = load_checkpoint(args.ckpt)
    params = jax.tree.map(jnp.asarray, ckpt["ema_params"])
    huber_delta = float(T._get(cfg, "huber_delta", 10.0)); loss_type = T._get(cfg, "loss_type", "huber")
    cov_mode = T._get(cfg, "cov_mode", "scalar6"); beta_nll = float(T._get(cfg, "beta_nll", 0.0))
    energy_loss_per = T._get(cfg, "energy_loss_per", "atom")
    collate = functools.partial(default_collate_fn, node_bucket_size=int(T._get(cfg, "node_bucket_size", 16)),
                                edge_bucket_size=int(T._get(cfg, "edge_bucket_size", 256)))
    gs = train_ds.graphs
    bs = args.batch_size
    raw = [jraph.batch_np(gs[i:i + bs]) for i in range(0, len(gs), bs)]
    nb, eb = int(T._get(cfg, "node_bucket_size", 16)), int(T._get(cfg, "edge_bucket_size", 256))
    pad_n = nearest_bucket(max(int(np.sum(b.n_node)) for b in raw) + 1, nb)
    pad_e = nearest_bucket(max(int(np.sum(b.n_edge)) for b in raw) + 1, eb)
    batches = [jraph.pad_with_graphs(b, pad_n, pad_e, bs + 1) for b in raw]
    stacked = jax.tree.map(lambda *x: jnp.asarray(np.stack(x)), *batches)
    n_graphs = float(len(gs)); n_atoms = float(sum(int(g.n_node[0]) for g in gs))
    wE, wF = args.energy_weight, args.force_weight

    @jax.checkpoint
    def batch_loss(p, batch):
        model = nnx.merge(graphdef, p)
        t = _loss_terms(model(batch), batch, huber_delta, loss_type, False, cov_mode, beta_nll,
                        energy_loss_per=energy_loss_per)
        return wE * t["energy_loss"] / n_graphs + wF * t["force_loss"] / (3.0 * n_atoms)

    def loss_fn(p, data):
        dtype = jax.tree.leaves(p)[0].dtype
        total, _ = jax.lax.scan(lambda c, b: (c + batch_loss(p, b).astype(dtype), None), jnp.zeros((), dtype), data)
        if args.l2 > 0:
            total = total + args.l2 * sum(jnp.sum(x * x) for x in jax.tree.leaves(p))
        return total

    mesh, n_devices = setup_mesh()
    eval_step = make_sharded_evaluate_step(mesh, huber_delta=huber_delta, loss_type=loss_type, periodic=False,
                                           cov_mode=cov_mode, beta_nll=beta_nll, energy_loss_per=energy_loss_per)

    def evaluate(ds, p):
        return evaluate_sharded(graphdef=graphdef, params=replicate_pytree(p, mesh), files_pkl=[ds], batch_size=bs,
                                n_devices=n_devices, mesh=mesh, energy_weight=wE, force_weight=wF, stress_weight=0.0,
                                huber_delta=huber_delta, loss_type=loss_type, eval_step=eval_step,
                                params_already_replicated=True, periodic=False, cov_mode=cov_mode, beta_nll=beta_nll,
                                collate_fn=collate)

    def report(it, p, train_loss):
        v = evaluate(valid_ds, p)
        line = (f"iter {it:4d} | train loss {train_loss:.6e} | valid loss {v['total_loss']:.6e} "
                f"E {v['energy_rmse_total'] * 1e3:7.2f} F {v['force_rmse'] * 1e3:7.2f}")
        for name, ds in test_sets.items():
            m = evaluate(ds, p)
            line += (f" | {name} E {m['energy_rmse_total'] * 1e3:7.2f}/{m['energy_mae_total'] * 1e3:6.2f}"
                     f" F {m['force_rmse'] * 1e3:7.2f}/{m['force_mae'] * 1e3:6.2f}")
        print(line, flush=True)
        return v

    out = Path(args.out) if args.out else Path(args.ckpt).parent / ("ckpt_lbfgs_" + Path(args.ckpt).parent.name + ".pkl")
    print(f"L-BFGS refinement of {args.ckpt} (epoch {ckpt.get('epoch')}): {len(batches)} batches of {bs}, "
          f"wE={wE} wF={wF} l2={args.l2} memory={args.memory}; metrics E meV (RMSE/MAE), F meV/A", flush=True)
    opt = optax.lbfgs(memory_size=args.memory,
                      linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=20, verbose=False))
    state = opt.init(params)

    @jax.jit
    def lbfgs_step(p, st, data):
        lf = lambda q: loss_fn(q, data)
        value, grad = optax.value_and_grad_from_state(lf)(p, state=st)
        updates, st = opt.update(grad, st, p, value=value, grad=grad, value_fn=lf)
        return optax.apply_updates(p, updates), st, value

    print(f"padded batches: {len(batches)} x (nodes {pad_n}, edges {pad_e}, graphs {bs + 1})", flush=True)
    v0 = report(0, params, float(jax.jit(loss_fn)(params, stacked)))
    best = (v0["total_loss"], 0); best_params = params
    t0 = time.time()
    for it in range(1, args.iters + 1):
        params, state, value = lbfgs_step(params, state, stacked)
        if it % args.eval_every == 0 or it == args.iters:
            v = report(it, params, float(value))
            if v["total_loss"] < best[0]:
                best = (v["total_loss"], it); best_params = params
                save_checkpoint({"params": jax.tree.map(np.asarray, params), "ema_params": jax.tree.map(np.asarray, params),
                                 "config": cfg, "epoch": ckpt.get("epoch"), "lbfgs_iter": it, "best_val_loss": float(v["total_loss"]),
                                 "metrics": v, "step": ckpt.get("step")}, str(out))
            print(f"          {(time.time() - t0) / it:.1f} s/iter, best valid loss {best[0]:.6e} at iter {best[1]}", flush=True)
    print(f"written {out} (iter {best[1]})")


if __name__ == "__main__":
    main()
