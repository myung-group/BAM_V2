"""Pickle (in-memory graphs) vs streaming (StreamLoader workers) data path: speed on one small dataset (2026-11-09).

Same model, same fixed batch shape (stream_plan.derive_buckets, one bucket), same optimiser step; only the data path
differs.  Measures
  A  one-time preprocessing: read extxyz + neighbour lists + pickle dump / load (pickle path) vs the size pass (stream);
  B  loader only: structures/s of one epoch without a model (pickle loader; StreamLoader with W workers);
  C  training: structures/s of n_steps optimiser steps fed by each path, against the GPU-only rate (the same steps on
     batches already in memory = the ideal, no data waiting), and the main-thread time spent getting the next batch.

usage (repo root):  python research/diag/bench_stream/bench_stream_vs_pkl.py <config.json> <out.json> [--workers 1,2,4,8]
                    [--steps 300] [--override '{"x64": false}']
"""
import argparse
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'race_v2' / 'data'))     # stream_io / stream_plan are top-level modules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config'); ap.add_argument('out')
    ap.add_argument('--workers', default='1,2,4,8'); ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--graphs-per-pack', type=int, default=0, help='default: config batch_size')
    ap.add_argument('--override', default='{}')
    ap.add_argument('--size-workers', type=int, default=8)
    a = ap.parse_args()
    cfg = {**json.load(open(a.config)), **json.loads(a.override)}
    import jax
    from race_v2.training import train as T
    if cfg.get('x64'):
        jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from flax import nnx
    from race_v2.data.xyz_graphs import load_frames, build_graphs, GraphListDataset
    from race_v2.data.loaders import BucketedDataLoader, fixed_collate_fn, setup_mesh, stack_batches_for_devices
    from race_v2.training.utils import make_sharded_train_step, replicate, replicate_pytree
    import stream_io as S
    import stream_plan as P
    from race_v2.data.stream_data import StreamLoader

    os.chdir(ROOT)
    xyz = cfg['train_path']; cutoff = float(cfg.get('cutoff', 5.0))
    Z = [int(z) for z in cfg['atomic_numbers']]; e0 = list(cfg['atom_energies'])
    assert Z == sorted(Z), 'stream mode needs sorted atomic_numbers'
    dtype = np.float64 if cfg.get('x64') else np.float32
    gpp = a.graphs_per_pack or int(cfg.get('batch_size', 16))
    res = {'config': a.config, 'xyz': xyz, 'dtype': str(np.dtype(dtype)), 'graphs_per_pack': gpp,
           'cpu': os.cpu_count(), 'gpu': str(jax.devices()[0]),
           'sm_pct': os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')}
    tmp = Path(tempfile.mkdtemp(prefix='bench_stream_', dir=os.environ.get('BENCH_TMP')))

    # ---- A: one-time preprocessing
    t = time.time(); frames = load_frames(xyz); res['A_read_s'] = time.time() - t
    n_str = len(frames); res['n_structures'] = n_str
    res['n_atoms_mean'] = float(np.mean([len(f) for f in frames]))
    t = time.time(); graphs = build_graphs(frames, Z, e0, cutoff, n_workers=1); res['A_build_w1_s'] = time.time() - t
    W8 = a.size_workers
    t = time.time(); build_graphs(frames, Z, e0, cutoff, n_workers=W8); res[f'A_build_w{W8}_s'] = time.time() - t
    pk = tmp / 'graphs.pkl'
    t = time.time(); pickle.dump(graphs, open(pk, 'wb')); res['A_pickle_dump_s'] = time.time() - t
    res['A_pickle_MB'] = pk.stat().st_size / 1e6
    t = time.time(); graphs = pickle.load(open(pk, 'rb')); res['A_pickle_load_s'] = time.time() - t
    if dtype == np.float64:
        graphs = T._cast_graphs(graphs, np.float64)
    t = time.time(); S.size_pass([f'extxyz:{xyz}'], cutoff, tmp / 'sizes', workers=W8, block=256)
    res[f'A_size_pass_w{W8}_s'] = time.time() - t
    table = P.load_size_table(tmp / 'sizes', cutoff)
    bk = P.derive_buckets(table['n_atoms'], table['n_edges'], n_buckets=1, graphs_per_pack=gpp,
                          max_graphs=4 * gpp)
    b0 = bk[0]
    res['bucket'] = dict(pad_nodes=b0.pad_nodes, pad_edges=b0.pad_edges, max_graphs=b0.max_graphs)
    print(json.dumps(res, indent=1), flush=True)

    def pkl_batches(seed):
        ld = BucketedDataLoader(GraphListDataset(graphs), batch_size=b0.max_graphs, n_buckets=8, shuffle=True,
                                drop_last=False, rngs=nnx.Rngs(seed), max_nodes_per_batch=b0.pad_nodes,
                                max_edges_per_batch=b0.pad_edges)
        ld.collate_fn = lambda gs, bs: fixed_collate_fn(gs, bs, b0.pad_nodes, b0.pad_edges)
        while True:
            for b in ld:
                yield b

    import jraph

    def n_real(b):
        return int(np.sum(jraph.get_graph_padding_mask(b)))

    # ---- B: loader only, one epoch
    res['B'] = {}
    t = time.time(); n = 0
    it = pkl_batches(0)
    while n < n_str:
        n += n_real(next(it))
    res['B']['pickle'] = n / (time.time() - t)
    workers = [int(w) for w in a.workers.split(',')]
    for w in workers:
        t0 = time.time()
        ld = StreamLoader(table, bk, Z, e0, cutoff, n_local=1, workers=w, seed=0, block_rows=64, dtype=dtype)
        list(zip(range(1), ld.iter_epoch(99)))          # start the worker pool (not timed)
        n = 0; t = time.time()
        for _, bs in ld.iter_epoch(0):
            n += n_real(bs[0])
        res['B'][f'stream_w{w}'] = n / (time.time() - t)
        ld.close()
        print(f'B stream W={w}: {res["B"][f"stream_w{w}"]:.0f} structures/s (pool start {t - t0:.1f} s)', flush=True)
    print('B', res['B'], flush=True)

    # ---- C: training steps
    stats = {'shift_raw': 0.0, 'scale': 1.0, 'avg_n_neighbors': float(table['n_edges'].sum() / table['n_atoms'].sum()),
             'n_graphs': n_str, 'n_atoms': int(table['n_atoms'].sum())}
    import io
    graphdef, params, optimizer, schedule, grad_clip = T.build_model(cfg, len(Z), e0, stats, io.StringIO())
    mesh, nd = setup_mesh()
    step_fn = make_sharded_train_step(optimizer=optimizer, mesh=mesh, energy_weight=1.0,
                                      force_weight=float(cfg.get('force_weight', 10.0)),
                                      huber_delta=float(cfg.get('huber_delta', 0.1)), ema_decay=0.99,
                                      loss_type=cfg.get('loss_type', 'huber'), periodic=False,
                                      grad_clip=float(grad_clip) if grad_clip else 1.0)
    state0 = (replicate_pytree(params, mesh), replicate_pytree(optimizer.init(params), mesh),
              replicate_pytree(schedule.init(params), mesh), replicate_pytree(params, mesh))

    def run(batches, label, n_steps=None):
        """Optimiser steps over an iterable of collated batches (one epoch unless n_steps); times from the first
        batch request to the last update on the device."""
        p, o, s, e = state0
        stp = replicate(jnp.array(0), mesh)
        jax.block_until_ready(p)
        t_next = 0.0; n = 0; k = 0; t = time.time()
        it = iter(batches)
        while n_steps is None or k < n_steps:
            t1 = time.time()
            try:
                b = next(it)
            except StopIteration:
                break
            sb, _ = stack_batches_for_devices([b], mesh); t_next += time.time() - t1
            n += n_real(b); k += 1
            p, o, e, stp, m = step_fn(graphdef, p, o, s, e, stp, sb)
        jax.block_until_ready(p)
        dt = time.time() - t
        out = {'structures_per_s': n / dt, 'steps': k, 'structures': n, 's_per_step': dt / k,
               'next_batch_frac': t_next / dt, 'graphs_per_step': n / k}
        print(f'C {label}: ' + ', '.join(f'{q} {v:.4g}' for q, v in out.items()), flush=True)
        return out

    def pkl_epoch(seed):
        ld = BucketedDataLoader(GraphListDataset(graphs), batch_size=b0.max_graphs, n_buckets=8, shuffle=True,
                                drop_last=False, rngs=nnx.Rngs(seed), max_nodes_per_batch=b0.pad_nodes,
                                max_edges_per_batch=b0.pad_edges)
        ld.collate_fn = lambda gs, bs: fixed_collate_fn(gs, bs, b0.pad_nodes, b0.pad_edges)
        return ld

    def stream_epoch(ld, ep):
        for _, bs in ld.iter_epoch(ep):
            yield bs[0]

    reps = int(os.environ.get('BENCH_REPS', 2))
    res['C'] = {}
    mem_pk = list(pkl_epoch(1))
    ld = StreamLoader(table, bk, Z, e0, cutoff, n_local=1, workers=max(workers), seed=0, block_rows=64, dtype=dtype)
    mem_st = list(stream_epoch(ld, 0)); ld.close()
    run(mem_pk, 'warm-up', n_steps=40)          # compilation and kernel autotuning, not timed
    for r in range(reps):
        res['C'].setdefault('gpu_only_pickle_batches', []).append(run(mem_pk, f'gpu_only_pickle_batches #{r}'))
        res['C'].setdefault('gpu_only_stream_batches', []).append(run(mem_st, f'gpu_only_stream_batches #{r}'))
        res['C'].setdefault('pickle', []).append(run(pkl_epoch(10 + r), f'pickle #{r}'))
        for w in workers:
            ld = StreamLoader(table, bk, Z, e0, cutoff, n_local=1, workers=w, seed=0, block_rows=64, dtype=dtype)
            try:
                list(zip(range(1), stream_epoch(ld, 99)))       # start the worker pool (not timed)
                res['C'].setdefault(f'stream_w{w}', []).append(run(stream_epoch(ld, r), f'stream_w{w} #{r}'))
            finally:
                ld.close()
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
