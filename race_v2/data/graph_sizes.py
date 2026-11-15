"""Size table of a dataset before training: atoms and edges (at the model cutoff) of every structure, written as
readable text, plus a summary and the padded shapes a run would use.  No JAX; run inside a Slurm job (not on a login
node) for large files.

    python -I race_v2/data/graph_sizes.py <file.xyz> [<file.xyz> ...] --cutoff 5.0 [--workers 32] [--batch-size 16]
        [--out-dir DIR] [--buckets 4]

For every input <stem>.xyz it writes DIR/<stem>.sizes.tsv (columns index, n_atoms, n_edges; same format as the
sizes_<split>.tsv that a pad_mode "fixed" training run writes next to its log) and DIR/<stem>.sizes_summary.txt:
  * quantiles of atoms and edges per structure,
  * the pad_mode "fixed" budgets of training/train.py (nodes: max(batch * mean, largest + 1) rounded up to 64; edges:
    max(batch * mean, largest) rounded up to 1024) and the fraction of the padded edge slots that hold real edges
    when the structures are packed greedily in random order,
  * the same for --buckets size classes (equal-count split by edges per structure; one padded shape per class).
Only the neighbour counts are computed: the cost is one neighbour search per structure, done in --workers processes.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neighbors import count_edges  # noqa: E402  (no package import: race_v2/__init__ pulls in JAX)


def _count_chunk(args):
    frames, cutoff = args
    return [(len(a), count_edges(a, cutoff)) for a in frames]


def sizes_of(path, cutoff, workers, chunk=512):
    import ase.io
    import concurrent.futures as cf
    import multiprocessing as mp
    frames = ase.io.read(str(path), index=':')
    jobs = [(frames[k:k + chunk], cutoff) for k in range(0, len(frames), chunk)]
    out = []
    if workers <= 1:
        for j in jobs:
            out += _count_chunk(j)
    else:
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('forkserver')) as ex:
            for part in ex.map(_count_chunk, jobs):
                out += part
    a = np.array(out, dtype=np.int64).reshape(-1, 2)
    return a[:, 0], a[:, 1]


def budgets(n, e, batch):
    pn = int(math.ceil(max(batch * n.mean(), n.max() + 1) / 64) * 64)
    pe = int(math.ceil(max(batch * e.mean(), e.max()) / 1024) * 1024)
    return pn, pe


def pack_fill(n, e, pn, pe, max_graphs, seed=0):
    """Greedy packing in random order (as the fixed-shape loader): fraction of padded edge / node slots used."""
    order = np.random.default_rng(seed).permutation(len(n))
    batches, cn, ce, cg = 0, 0, 0, 0
    for k in order:
        if cg and (cn + n[k] + 1 > pn or ce + e[k] > pe or cg >= max_graphs):
            batches += 1; cn = ce = cg = 0
        cn += n[k]; ce += e[k]; cg += 1
    batches += 1 if cg else 0
    return batches, float(e.sum()) / (batches * pe), float(n.sum()) / (batches * pn)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('xyz', nargs='+'); ap.add_argument('--cutoff', type=float, required=True)
    ap.add_argument('--workers', type=int, default=0, help='0 = CPUs of this task (at most 64)')
    ap.add_argument('--batch-size', type=int, default=16); ap.add_argument('--max-graphs', type=int, default=64)
    ap.add_argument('--buckets', type=int, default=4); ap.add_argument('--out-dir', default=None)
    a = ap.parse_args()
    workers = a.workers or min(64, int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or len(os.sched_getaffinity(0)))
    for p in map(Path, a.xyz):
        t0 = time.time()
        n, e = sizes_of(p, a.cutoff, workers)
        out = Path(a.out_dir) if a.out_dir else p.parent
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f'{p.stem}.sizes.tsv', 'w') as fh:
            fh.write(f'# source {p}\n# cutoff {a.cutoff}\nindex\tn_atoms\tn_edges\n')
            for k, (x, y) in enumerate(zip(n, e)):
                fh.write(f'{k}\t{x}\t{y}\n')
        q = [0, 50, 90, 99, 100]
        lines = [f'{p}  cutoff {a.cutoff} A  {len(n)} structures  ({time.time() - t0:.0f} s, {workers} workers)',
                 'quantile        ' + '  '.join(f'p{v:>3}' for v in q) + '     mean',
                 'atoms/structure ' + '  '.join(f'{int(np.percentile(n, v)):>4}' for v in q) + f'  {n.mean():7.1f}',
                 'edges/structure ' + '  '.join(f'{int(np.percentile(e, v)):>4}' for v in q) + f'  {e.mean():7.1f}',
                 f'edges per atom: mean {e.sum() / n.sum():.2f}']
        pn, pe = budgets(n, e, a.batch_size)
        nb, fe, fn = pack_fill(n, e, pn, pe, a.max_graphs)
        lines.append(f'pad_mode fixed (batch {a.batch_size}): {pn} nodes / {pe} edges per step -> {nb} steps per epoch, '
                     f'{100 * fe:.0f} % of the edge slots and {100 * fn:.0f} % of the node slots real')
        if a.buckets > 1:
            cuts = np.quantile(e, np.linspace(0, 1, a.buckets + 1)[1:-1])
            cls = np.searchsorted(cuts, e, side='right')
            tot_slots, tot_steps = 0, 0
            for c in range(a.buckets):
                m = cls == c
                if not m.any():
                    continue
                pn_c, pe_c = budgets(n[m], e[m], a.batch_size)
                nb_c, fe_c, _ = pack_fill(n[m], e[m], pn_c, pe_c, a.max_graphs)
                tot_slots += nb_c * pe_c; tot_steps += nb_c
                lines.append(f'  bucket {c}: {int(m.sum())} structures, edges {int(e[m].min())}-{int(e[m].max())}: '
                             f'{pn_c} nodes / {pe_c} edges, {nb_c} steps, {100 * fe_c:.0f} % edge slots real')
            lines.append(f'{a.buckets} buckets: {tot_steps} steps per epoch, {100 * e.sum() / tot_slots:.0f} % of the '
                         f'edge slots real ({a.buckets} compiled shapes)')
        txt = '\n'.join(lines)
        print(txt, flush=True)
        (out / f'{p.stem}.sizes_summary.txt').write_text(txt + '\n')


if __name__ == '__main__':
    main()
