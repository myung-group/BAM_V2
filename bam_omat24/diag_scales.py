"""Per-layer feature / symmetric-contraction magnitudes of a RACE at initialisation (review §11.11).

For every layer prints the RMS (and max |.|, finiteness) of the layer input h, the aggregated
message A, the symmetric-contraction increment B (when in_layer_order >= 2 with sym_contraction)
and the layer output, for one training batch, eagerly (no jit) by hooking RACEConvolution.

    python -m bam_omat24.diag_scales <train config json> [--method naive|uniform_1d] [--x64]
"""
import argparse, functools, json, sys
import numpy as np
import jax, jax.numpy as jnp
from flax import nnx

from bam_omat24 import train_3bpa as T
from bam_omat24 import race_nnx as R
from bam_omat24.data_3bpa import GraphListDataset
from bam_omat24.data_nnx import BucketedDataLoader, default_collate_fn

STATS = []


def _st(name, arr):
    a = np.asarray(arr, dtype=np.float64)
    fin = np.isfinite(a)
    return dict(name=name, rms=float(np.sqrt(np.mean(a[fin] ** 2))) if fin.any() else float("nan"),
                max=float(np.max(np.abs(a[fin]))) if fin.any() else float("nan"),
                nonfinite=int((~fin).sum()), size=int(a.size))


def install_hooks():
    orig_call, orig_agg, orig_sym = R.RACEConvolution.__call__, R.RACEConvolution.aggregate, R.RACEConvolution.sym_term

    def call(self, features, x_feats, species, *a, **k):
        STATS.append(_st("h_in", features.array))
        STATS.append(_st("x_feats", x_feats.array))
        out = orig_call(self, features, x_feats, species, *a, **k)
        STATS.append(_st("stratum_E", out[0].array))
        STATS.append(_st("h_out", out[1].array))
        return out

    def agg(self, *a, **k):
        m = orig_agg(self, *a, **k)
        STATS.append(_st("A_msg", m.array))
        return m

    def sym(self, messages_agg, species):
        a = R._dense(self.linear_sym_in(messages_agg))
        STATS.append(_st("A_lin", a.array))
        b = orig_sym(self, messages_agg, species)
        STATS.append(_st(f"B_sym(deg {self.sym_degrees}, {self.sym_method})", b.array))
        return b

    R.RACEConvolution.__call__, R.RACEConvolution.aggregate, R.RACEConvolution.sym_term = call, agg, sym


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--method", default=None, help="override sym_method (naive | uniform_1d)")
    ap.add_argument("--x64", action="store_true")
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--set", action="append", default=[], help="config override key=json_value")
    ap.add_argument("--scan", action="store_true", help="scan all train/valid batches for non-finite outputs")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
        print(f"override {k} = {cfg[k]}")
    if args.x64:
        cfg["x64"] = True
    if cfg.get("x64"):
        jax.config.update("jax_enable_x64", True)
    fout = sys.stdout
    (atomic_numbers, atom_energies, train_ds, valid_ds, test_sets, stats) = T.build_datasets(cfg, fout)
    graphdef, params, _opt, _sched, _gc = T.build_model(cfg, len(atomic_numbers), atom_energies, stats, fout)
    model = nnx.merge(graphdef, params)
    if args.method:
        for layer in model.layers:
            if getattr(layer, "sym_contraction", False):
                layer.sym_method = args.method
    collate = functools.partial(default_collate_fn, node_bucket_size=16, edge_bucket_size=256)
    graphs = train_ds.graphs[: args.batch_size]
    if cfg.get("x64"):
        graphs = T._cast_graphs(graphs, np.float64)
    loader = BucketedDataLoader(GraphListDataset(graphs), batch_size=args.batch_size, n_buckets=1,
                                shuffle=False, drop_last=False, rngs=nnx.Rngs(0))
    loader.collate_fn = collate
    batch = next(iter(loader))
    if args.scan:
        # scan every train / valid batch for non-finite energies or forces (jitted forward);
        # the hooked pass below then runs on the first offending batch (or the first batch)
        import jraph
        fwd = jax.jit(lambda b: model(b))
        for name, ds in (("train", train_ds), ("valid", valid_ds)):
            gs = ds.graphs
            if cfg.get("x64"):
                gs = T._cast_graphs(gs, np.float64)
            ld = BucketedDataLoader(GraphListDataset(gs), batch_size=args.batch_size, n_buckets=8,
                                    shuffle=False, drop_last=False, rngs=nnx.Rngs(0))
            ld.collate_fn = collate
            bad = None
            for k, b in enumerate(ld):
                o = fwd(b)
                gmask = np.asarray(jraph.get_graph_padding_mask(b)); nmask = np.asarray(jraph.get_node_padding_mask(b))
                e = np.asarray(o["energy"]); f = np.asarray(o["forces"])
                e_real_bad = int((~np.isfinite(e[gmask])).sum()); e_pad_bad = int((~np.isfinite(e[~gmask])).sum())
                f_real_bad = int((~np.isfinite(f[nmask])).sum()); f_pad_bad = int((~np.isfinite(f[~nmask])).sum())
                if e_real_bad or e_pad_bad or f_real_bad or f_pad_bad:
                    print(f"[{name} batch {k}] n_graphs={len(e)} nodes={len(f)} non-finite: E real {e_real_bad} pad {e_pad_bad} | "
                          f"F real {f_real_bad} pad {f_pad_bad} | E real max {np.max(np.abs(e[gmask])):.3e}")
                    bad = bad or b
            print(f"[{name}] scanned {k + 1} batches; first bad batch: {'none' if bad is None else 'found'}")
            if bad is not None:
                batch = bad
    out = model(batch)                      # full forward + forces (under jax.grad, no hooks)
    install_hooks()                         # eager node-energy pass with the hooks (concrete arrays)
    model.node_energies(batch.nodes["positions"], None, None, batch)
    dtype = batch.nodes["positions"].dtype
    print(f"\n== {args.config}  dtype={dtype}  layers={len(model.layers)} "
          f"orders={[l.in_layer_order for l in model.layers]}")
    layer = -1
    for s in STATS:
        if s["name"] == "h_in":
            layer += 1
            print(f"-- layer {layer}")
        print(f"   {s['name']:34s} rms {s['rms']:12.4e}  max {s['max']:12.4e}  nonfinite {s['nonfinite']}/{s['size']}")
    print("energy:", _st("energy", out["energy"]))
    print("forces:", _st("forces", out["forces"]))


if __name__ == "__main__":
    main()
