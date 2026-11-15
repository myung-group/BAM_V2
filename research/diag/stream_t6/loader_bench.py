"""T6 loader-only throughput (2026-11-09, docs/HANDOFF_stream_loader.md section 6 item 3): structures/s of the
training StreamLoader (no model, no GPU) for several worker counts, read from the shared file system.

The size table is made when missing (or reused; the trainer reads the same directory via stream.train.sizes).
Per worker count: pool start + the first block (worker init, extxyz offset scan) timed apart, then `--steps` steps
of data epoch 0.  BAM_NEIGHBORS (ase | vesin) is taken from the environment.

usage (repo root): python -I research/diag/stream_t6/loader_bench.py --source extxyz:<train.xyz> --cutoff 5.0 \
    --sizes <dir> --atomic-numbers-from <config.json> --workers 2,4,8,16 --steps 500 --out <out.json>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")      # loader only: no GPU (jraph imports jax)
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'race_v2' / 'data'))     # stream_io / stream_plan are top-level modules


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', action='append', required=True)
    ap.add_argument('--cutoff', type=float, required=True)
    ap.add_argument('--sizes', required=True)
    ap.add_argument('--atomic-numbers-from', required=True, help='config with atomic_numbers and atom_energies')
    ap.add_argument('--workers', default='2,4,8,16')
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--graphs-per-pack', type=int, default=16)
    ap.add_argument('--n-buckets', type=int, default=1)
    ap.add_argument('--mix-blocks', type=int, default=16)
    ap.add_argument('--size-workers', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import stream_io as S
    import stream_plan as P
    from race_v2.data.stream_data import StreamLoader
    import jraph
    cfg = json.load(open(a.atomic_numbers_from))
    Z = [int(z) for z in cfg['atomic_numbers']]; e0 = [float(x) for x in cfg['atom_energies']]
    res = {'sources': a.source, 'neighbors': os.environ.get('BAM_NEIGHBORS', 'ase'), 'cpu': os.cpu_count(),
           'affinity': len(os.sched_getaffinity(0)), 'host': os.uname().nodename, 'steps': a.steps}
    d = Path(a.sizes)
    if (d / 'sizes_meta.json').exists():
        res['size_pass_s'] = None
    else:
        t = time.time(); S.size_pass(a.source, a.cutoff, d, workers=a.size_workers, block=256)
        res['size_pass_s'] = time.time() - t
    table = P.load_size_table(d, a.cutoff)
    if [x['spec'] for x in table['meta']['sources']] != a.source:
        raise SystemExit(f'{d}: size table of other sources')
    gpp = a.graphs_per_pack
    bk = P.derive_buckets(table['n_atoms'], table['n_edges'], n_buckets=a.n_buckets, graphs_per_pack=gpp,
                          max_graphs=4 * gpp)
    res.update(n_structures=int(len(table['n_atoms'])), mean_atoms=float(table['n_atoms'].mean()),
               mean_edges=float(table['n_edges'].mean()),
               buckets=[dict(pad_nodes=b.pad_nodes, pad_edges=b.pad_edges, max_graphs=b.max_graphs) for b in bk],
               runs={})
    print(json.dumps({k: v for k, v in res.items() if k != 'runs'}, indent=1), flush=True)
    for w in [int(x) for x in a.workers.split(',')]:
        t0 = time.time()
        ld = StreamLoader(table, bk, Z, e0, a.cutoff, n_local=1, workers=w, seed=0, block_rows=128,
                          dtype=np.float32, mix_blocks=a.mix_blocks)
        it = ld.iter_epoch(0)
        next(it)                                        # pool start, worker init, first window (timed apart)
        t_first = time.time() - t0
        n = steps = 0; t = time.time()
        for _, bs in it:
            n += int(np.sum(jraph.get_graph_padding_mask(bs[0])))
            steps += 1
            if steps >= a.steps:
                break
        dt = time.time() - t
        ld.close()
        r = dict(first_step_s=t_first, steps=steps, structures=n, s=dt, structures_per_s=n / dt, steps_per_s=steps / dt)
        res['runs'][f'W{w}'] = r
        print(f"W={w:3d}: first step {t_first:6.1f} s, then {r['structures_per_s']:8.0f} structures/s "
              f"({r['steps_per_s']:.1f} steps/s over {steps} steps)", flush=True)
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
