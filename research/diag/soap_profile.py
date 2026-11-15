"""Time breakdown of one training step on a real fixed-shape OAM batch: inference (energy + forces), loss + gradient,
SOAP update (regular step / eigenbasis refresh step) and AMSGrad update, for float64 or float32.

    python research/diag/soap_profile.py --dtype float64 research/input/input_3bpa_omat24_oam64_salex_v2_s42_cue.json [...]

float32 is timed even though float32 SOAP diverges in training (timing only).  Each number is the median of repeated
jitted calls after compilation (jax.block_until_ready).
"""
import argparse
import functools
import json
import math
import time

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from race_v2.training import train as T
from race_v2.data.loaders import BucketedDataLoader, fixed_collate_fn
from race_v2.training.utils import compute_loss


def bench(f, *args, n=15):
    jax.block_until_ready(f(*args))                    # compile + warm-up
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); jax.block_until_ready(f(*args)); ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts))


def first_batch(cfg, tr):
    bs = int(cfg['batch_size'])
    ntr = np.array([int(g.n_node[0]) for g in tr.graphs]); etr = np.array([int(g.n_edge[0]) for g in tr.graphs])
    pad_n = int(math.ceil(max(bs * ntr.mean(), ntr.max() + 1) / 64) * 64)
    pad_e = int(math.ceil(max(bs * etr.mean(), etr.max()) / 1024) * 1024)
    loader = BucketedDataLoader(tr, batch_size=int(cfg.get('pack_max_graphs', 64)), n_buckets=8, shuffle=True,
                                drop_last=False, rngs=nnx.Rngs(0), max_nodes_per_batch=pad_n, max_edges_per_batch=pad_e)
    loader.collate_fn = functools.partial(fixed_collate_fn, n_node_pad=pad_n, n_edge_pad=pad_e)
    return jax.tree.map(jnp.asarray, next(iter(loader))), pad_n, pad_e


def profile(path, dtype):
    cfg = json.load(open(path))
    cfg['x64'] = dtype == 'float64'
    jax.config.update('jax_enable_x64', cfg['x64'])
    (zs, e0, tr, va, tests, stats) = T.build_datasets(cfg, open('/dev/null', 'w'))
    if cfg['x64']:
        tr.graphs = T._cast_graphs(tr.graphs, np.float64)
    batch, pad_n, pad_e = first_batch(cfg, tr)
    out = {'run': path.split('omat24_')[-1].replace('_cue.json', ''), 'dtype': dtype, 'pad': f'{pad_n}/{pad_e}'}
    for opt in ('soap', 'amsgrad'):
        c = dict(cfg); c['optimizer'] = opt
        graphdef, params, optimizer, schedule, gc = T.build_model(c, len(zs), e0, stats, open('/dev/null', 'w'))
        if opt == 'soap':
            out['params_M'] = sum(x.size for x in jax.tree.leaves(params)) / 1e6
            infer = jax.jit(lambda p, b: (lambda r: (r['energy'], r['forces']))(nnx.merge(graphdef, p)(b)))
            out['infer_ms'] = bench(infer, params, batch)

            def lf(p, b):
                return compute_loss(graphdef, p, b, 1.0, 10.0, 10.0, 10.0, 'huber', True, 'scalar6', 0.0)[0]
            vg = jax.jit(jax.value_and_grad(lf))
            out['loss_grad_ms'] = bench(vg, params, batch)
        _, g = vg(params, batch)
        state = optimizer.init(params)
        upd = jax.jit(optimizer.update)
        jax.block_until_ready(upd(g, state, params))   # compile (all lax.switch branches)
        ts = []
        for k in range(41):
            t0 = time.perf_counter()
            u, state = upd(g, state, params)
            jax.block_until_ready((u, state)); ts.append(1e3 * (time.perf_counter() - t0))
        if opt == 'soap':
            freq = int(cfg.get('soap_precondition_frequency', 10))
            out['soap_regular_ms'] = float(np.median([t for k, t in enumerate(ts) if k > 0 and k % freq]))
            out['soap_refresh_ms'] = float(np.median([t for k, t in enumerate(ts) if k > 0 and k % freq == 0]))
        else:
            out['amsgrad_ms'] = float(np.median(ts[1:]))
    out['soap_step_avg_ms'] = out['soap_regular_ms'] * 0.9 + out['soap_refresh_ms'] * 0.1
    out['share_soap'] = out['soap_step_avg_ms'] / (out['soap_step_avg_ms'] + out['loss_grad_ms'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dtype', choices=['float64', 'float32'], default='float64')
    ap.add_argument('configs', nargs='+')
    a = ap.parse_args()
    for p in a.configs:
        o = profile(p, a.dtype)
        print(json.dumps(o), flush=True)
        print(f"[{o['run']} {o['dtype']}] params {o['params_M']:.1f}M pad {o['pad']} | inference {o['infer_ms']:.1f} ms | "
              f"loss+grad {o['loss_grad_ms']:.1f} ms | SOAP regular {o['soap_regular_ms']:.1f} ms, refresh "
              f"{o['soap_refresh_ms']:.1f} ms (avg {o['soap_step_avg_ms']:.1f}, {100 * o['share_soap']:.0f}% of a step) | "
              f"AMSGrad {o['amsgrad_ms']:.1f} ms", flush=True)


if __name__ == '__main__':
    main()
