"""Inference-speed benchmark of a 3BPA-pipeline checkpoint (review §12.3, cost comparison with
GRACE 2.3 ms / molecule on an A100 and MACE).

    python -m bam_omat24.bench_speed <train config json> --ckpt <pkl> [--n 200] [--batches 1 10]
           [--node-bucket 16] [--edge-bucket 256]

Times energy + forces (jit, block_until_ready) of the first frames of the first test set, batched
``--batches`` molecules at a time, in the config precision and in float32, and reports ms per call,
ms per molecule, the padding used and the parameter count.
"""
import argparse, functools, json, sys, time
import numpy as np
import jax, jax.numpy as jnp
from flax import nnx

from bam_omat24 import train_3bpa as T
from bam_omat24.data_nnx import default_collate_fn
from bam_omat24.training_utils import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 10])
    ap.add_argument("--node-bucket", type=int, default=16); ap.add_argument("--edge-bucket", type=int, default=256)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    if cfg.get("x64"):
        jax.config.update("jax_enable_x64", True)
    fout = sys.stdout
    (atomic_numbers, atom_energies, train_ds, valid_ds, test_sets, stats) = T.build_datasets(cfg, fout)
    graphdef, _p0, *_ = T.build_model(cfg, len(atomic_numbers), atom_energies, stats, fout)
    params64 = jax.tree.map(jnp.asarray, load_checkpoint(args.ckpt)["ema_params"])
    n_params = sum(int(x.size) for x in jax.tree.leaves(params64))
    name, ds = next(iter(test_sets.items()))
    print(f"device {jax.devices()[0].device_kind}; {n_params:,} parameters; frames from {name}; "
          f"{int(ds.graphs[0].n_node[0])} atoms, {int(ds.graphs[0].n_edge[0])} edges per molecule", flush=True)

    @jax.jit
    def fwd(p, batch):
        out = nnx.merge(graphdef, p)(batch)
        return out["energy"], out["forces"]

    for dtype in ([np.float64, np.float32] if cfg.get("x64") else [np.float32]):
        p = jax.tree.map(lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x, params64)
        for b in args.batches:
            graphs = T._cast_graphs(ds.graphs[:b], dtype)
            batch = default_collate_fn(graphs, b, args.node_bucket, args.edge_bucket)
            e, f = fwd(p, batch); jax.block_until_ready((e, f))
            for _ in range(5):
                jax.block_until_ready(fwd(p, batch))
            t0 = time.perf_counter()
            for _ in range(args.n):
                jax.block_until_ready(fwd(p, batch))
            dt = (time.perf_counter() - t0) / args.n * 1e3
            print(f"{np.dtype(dtype).name:8s} batch {b:3d}: {dt:8.3f} ms/call = {dt / b:8.3f} ms/molecule "
                  f"(padded to {int(batch.nodes['positions'].shape[0])} nodes, {int(batch.senders.shape[0])} edges)", flush=True)


if __name__ == "__main__":
    main()
