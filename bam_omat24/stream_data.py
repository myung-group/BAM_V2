"""Streaming loader of the pickle-free pipeline, main-process side (docs/stream_loader_design.md section 4.6).
Single host for now; multi-host stacking (make_array_from_process_local_data) and the train_3bpa integration
(data_format "stream") are the next steps (docs/HANDOFF.md).

    loader = StreamLoader(sources, table, buckets, atomic_numbers, atom_energies, cutoff, n_local=1, workers=8)
    for bucket, batches in loader.iter_epoch(data_epoch):      # batches: list of n_local padded jraph.GraphsTuple
        ...
    loader.close()

Workers (forkserver, JAX-free: stream_io imported as a top-level module) read blocks of structures and build the
graph arrays (stream_io.graph_arrays = data_3bpa.preprocess_graph); the main thread packs them in the planned order
(stream_plan.plan_data_epoch) with data_nnx.fixed_collate_fn.  At most 2 * workers blocks are in flight.  Every
structure's edge count is checked against the size table (a mismatch would overflow the fixed shape).
"""
from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import os
import sys
from collections import deque

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import stream_plan as P  # noqa: E402

_W = {}


def _worker_init(kinds, files, atomic_numbers, atom_energies, cutoff):
    import stream_io as S
    _W['S'] = S
    _W['src'] = {}
    _W['kinds'], _W['files'] = kinds, files
    _W['idx'] = {int(z): i for i, z in enumerate(sorted(atomic_numbers))}
    _W['e0'], _W['cutoff'] = list(atom_energies), float(cutoff)


def _worker_block(task):
    """task = (seq, file, keys) -> (seq, list of graph-array dicts)."""
    S = _W['S']
    seq, fi, keys = task
    if fi not in _W['src']:
        kind, path = _W['kinds'][fi], _W['files'][fi]
        _W['src'][fi] = S.ExtxyzSource([path]) if kind == 'extxyz' else S.AselmdbSource([path])
    sts = _W['src'][fi].read_block(0, keys)
    return seq, [S.graph_arrays(st, _W['idx'], _W['e0'], _W['cutoff']) for st in sts]


def _to_graph(d):
    import jraph
    nodes = {'species': d['species'], 'positions': d['positions'], 'forces': d['forces']}
    glob = {'energy': d['energy']}
    edges = None
    if 'Sij' in d:
        edges = {'Sij': d['Sij']}
        glob.update(cell=d['cell'], cutoff=d['cutoff'], volume=d['volume'], stress=d['stress'])
    return jraph.GraphsTuple(n_node=d['n_node'], n_edge=d['n_edge'], nodes=nodes, edges=edges,
                             senders=d['senders'], receivers=d['receivers'], globals=glob)


class StreamLoader:
    def __init__(self, table, buckets, atomic_numbers, atom_energies, cutoff, n_local=1, workers=8, seed=0,
                 block_rows=128, shuffle=True, dtype=np.float64):
        meta = table['meta']
        self.table, self.buckets, self.n_local, self.seed = table, buckets, n_local, seed
        self.block_rows, self.shuffle, self.dtype = block_rows, shuffle, dtype
        if list(atomic_numbers) != sorted(atomic_numbers):
            raise ValueError("stream mode needs a sorted atomic_numbers list (species index = sorted rank)")
        self.pool = cf.ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context('forkserver'), initializer=_worker_init,
            initargs=(meta['kinds'], meta['files'], list(atomic_numbers), list(atom_energies), float(cutoff)))
        self.workers = workers

    def close(self):
        self.pool.shutdown(cancel_futures=True)

    def plan(self, data_epoch, host=0, n_hosts=1, counts_allgather=None):
        return P.plan_data_epoch(self.table, self.buckets, data_epoch, host, n_hosts, self.n_local, seed=self.seed,
                                 block_rows=self.block_rows, shuffle=self.shuffle, counts_allgather=counts_allgather)

    def _graphs(self, rows_needed):
        """Yield (row, graph dict) for the rows in the order of rows_needed, reading them block-wise."""
        t = self.table
        order = list(rows_needed)
        # contiguous runs of the same file -> blocks of <= block_rows keys (stream order is block-wise already)
        tasks, cur = [], []
        for r in order:
            if cur and (t['file'][r] != t['file'][cur[-1]] or len(cur) >= self.block_rows):
                tasks.append(cur); cur = []
            cur.append(r)
        if cur:
            tasks.append(cur)
        inflight, nxt = deque(), 0
        while nxt < len(tasks) or inflight:
            while nxt < len(tasks) and len(inflight) < 2 * self.workers:
                rows = tasks[nxt]
                inflight.append((rows, self.pool.submit(
                    _worker_block, (nxt, int(t['file'][rows[0]]), t['key'][rows]))))
                nxt += 1
            rows, fut = inflight.popleft()
            _, gs = fut.result()
            for r, g in zip(rows, gs):
                if int(g['n_edge'][0]) != int(t['n_edges'][r]):
                    raise RuntimeError(f"row {r}: {int(g['n_edge'][0])} edges, size table says {int(t['n_edges'][r])}"
                                       " (neighbour search differs from the size pass)")
                yield r, g

    def iter_epoch(self, data_epoch, host=0, n_hosts=1, counts_allgather=None):
        """Yield (bucket, [n_local padded GraphsTuple]) per planned step of this host."""
        from bam_omat24.data_nnx import fixed_collate_fn
        pl = self.plan(data_epoch, host, n_hosts, counts_allgather)
        need = []
        for b, ids in pl.steps:
            for i in ids:
                need.extend(int(r) for r in pl.packs[b][i])
        # read in stream order (the order of the packs' members) and assemble packs as they complete
        cache = {}
        it = self._graphs(need)
        for b, ids in pl.steps:
            bk = self.buckets[b]
            out = []
            for i in ids:
                members = [int(r) for r in pl.packs[b][i]]
                while not all(m in cache for m in members):
                    r, g = next(it)
                    cache[r] = g
                gs = [_to_graph(_cast(cache.pop(m), self.dtype)) for m in members]
                out.append(fixed_collate_fn(gs, bk.max_graphs, bk.pad_nodes, bk.pad_edges))
            yield b, out


def _cast(d, dtype):
    return {k: (v.astype(dtype) if np.issubdtype(v.dtype, np.floating) else v) for k, v in d.items()}
