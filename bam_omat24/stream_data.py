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
import math
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


def _worker_init(kinds, files, atomic_numbers, atom_energies, cutoff, fields=None):
    import stream_io as S
    from graph_fields import field_keys
    _W['S'] = S
    _W['src'] = {}
    _W['fields'] = dict(fields or {})
    _W['keys'] = field_keys(**_W['fields'])
    _W['kinds'], _W['files'] = kinds, files
    _W['idx'] = {int(z): i for i, z in enumerate(sorted(atomic_numbers))}
    _W['e0'], _W['cutoff'] = list(atom_energies), float(cutoff)


def _worker_block(task):
    """task = (seq, file, keys) -> (seq, list of graph-array dicts)."""
    S = _W['S']
    seq, fi, keys = task
    if fi not in _W['src']:
        kind, path = _W['kinds'][fi], _W['files'][fi]
        _W['src'][fi] = (S.ExtxyzSource([path], info_keys=_W['keys'][0], array_keys=_W['keys'][1])
                         if kind == 'extxyz' else S.AselmdbSource([path]))
    sts = _W['src'][fi].read_block(0, keys)
    return seq, [S.graph_arrays(st, _W['idx'], _W['e0'], _W['cutoff'], _W['fields']) for st in sts]


def _to_graph(d):
    import jraph
    nodes = {'species': d['species'], 'positions': d['positions'], 'forces': d['forces']}
    glob = {'energy': d['energy']}
    edges = None
    if 'Sij' in d:
        edges = {'Sij': d['Sij']}
        glob.update(cell=d['cell'], cutoff=d['cutoff'], volume=d['volume'], stress=d['stress'])
    for k, v in d.items():           # optional fields of graph_arrays(fields=...): 'n:<name>' nodes, 'g:<name>' globals
        if k.startswith('n:'):
            nodes[k[2:]] = v
        elif k.startswith('g:'):
            glob[k[2:]] = v
    return jraph.GraphsTuple(n_node=d['n_node'], n_edge=d['n_edge'], nodes=nodes, edges=edges,
                             senders=d['senders'], receivers=d['receivers'], globals=glob)


class StreamLoader:
    def __init__(self, table, buckets, atomic_numbers, atom_energies, cutoff, n_local=1, workers=8, seed=0,
                 block_rows=128, shuffle=True, dtype=np.float64, fields=None, mix_blocks=1):
        meta = table['meta']
        self.table, self.buckets, self.n_local, self.seed = table, buckets, n_local, seed
        self.block_rows, self.shuffle, self.dtype, self.mix_blocks = block_rows, shuffle, dtype, int(mix_blocks)
        if list(atomic_numbers) != sorted(atomic_numbers):
            raise ValueError("stream mode needs a sorted atomic_numbers list (species index = sorted rank)")
        self.pool = cf.ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context('forkserver'), initializer=_worker_init,
            initargs=(meta['kinds'], meta['files'], list(atomic_numbers), list(atom_energies), float(cutoff),
                      dict(fields or {})))
        self.workers = workers

    def close(self):
        self.pool.shutdown(cancel_futures=True)

    def plan(self, data_epoch, host=0, n_hosts=1, counts_allgather=None):
        return P.plan_data_epoch(self.table, self.buckets, data_epoch, host, n_hosts, self.n_local, seed=self.seed,
                                 block_rows=self.block_rows, shuffle=self.shuffle, counts_allgather=counts_allgather,
                                 mix_blocks=self.mix_blocks)

    def _graphs(self, rows_needed):
        """Yield (row, graph dict) for the rows of rows_needed, read block-wise: in that order, or with mix_blocks > 1
        in (file, key) order inside each window of block_rows * mix_blocks rows."""
        t = self.table
        order = list(rows_needed)
        if self.mix_blocks > 1:
            # mixed blocks: read each window of block_rows * mix_blocks rows sorted by (file, key), so the tasks stay
            # whole blocks; the callers cache the graphs until a pack is complete (at most one window)
            w = self.block_rows * self.mix_blocks
            order = [r for i in range(0, len(order), w)
                     for r in sorted(order[i:i + w], key=lambda q: (int(t['file'][q]), int(t['key'][q])))]
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

    def iter_epoch(self, data_epoch, host=0, n_hosts=1, counts_allgather=None, start_step=0, plan=None):
        """Yield (bucket, [n_local padded GraphsTuple]) per planned step of this host, from step start_step on (resume:
        the earlier steps are skipped without reading their structures)."""
        from bam_omat24.data_nnx import fixed_collate_fn
        pl = plan if plan is not None else self.plan(data_epoch, host, n_hosts, counts_allgather)
        steps = pl.steps[int(start_step):]
        need = []
        for b, ids in steps:
            for i in ids:
                need.extend(int(r) for r in pl.packs[b][i])
        # read in stream order (the order of the packs' members) and assemble packs as they complete
        cache = {}
        it = self._graphs(need)
        for b, ids in steps:
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


def stack_local(batches, mesh):
    """The n_local padded batches of this process (one per local device, same shape) -> one global array sharded over
    the 'dp' mesh axis.  One process: data_nnx.stack_batches_for_devices.  Several processes: every process gives
    only its own devices' batches (make_array_from_process_local_data); untested on several hosts (design T5)."""
    import jax
    from bam_omat24.data_nnx import stack_batches_for_devices
    if jax.process_count() == 1:
        return stack_batches_for_devices(batches, mesh)[0]
    from jax.sharding import NamedSharding, PartitionSpec
    sh = NamedSharding(mesh, PartitionSpec('dp'))
    local = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], 0), *batches)
    return jax.tree.map(lambda x: jax.make_array_from_process_local_data(sh, x), local)


def counts_allgather_jax(local):
    """Per-host pack counts (n_hosts, n_buckets) for the lock-step plan; identity for one process."""
    import jax
    if jax.process_count() == 1:
        return np.asarray(local)[None]
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(np.asarray(local, np.int64)))


class StreamSteps:
    """Training steps of the streaming loader across data epochs, with a resumable cursor (data_epoch, step).

        steps = StreamSteps(loader, mesh, cursor=(0, 0))
        for batch in steps.take(n):        # sharded batches, ready for the train step
            ...
        steps.cursor                       # (data epoch, next step inside it): store in the checkpoint
    """

    def __init__(self, loader: StreamLoader, mesh, cursor=(0, 0)):
        import jax
        self.loader, self.mesh = loader, mesh
        self.host, self.n_hosts = jax.process_index(), jax.process_count()
        self.cursor = (int(cursor[0]), int(cursor[1]))
        self._it, self._plan = None, None
        self.dropped = {}

    def _plan_of(self, ep):
        return self.loader.plan(ep, self.host, self.n_hosts, counts_allgather_jax)

    def steps_in_data_epoch(self, ep=None):
        ep = self.cursor[0] if ep is None else ep
        return len(self._plan_of(ep).steps)

    def take(self, n):
        k = 0
        while k < n:
            ep, st = self.cursor
            if self._it is None:
                self._plan = self._plan_of(ep)
                self.dropped[ep] = self._plan.dropped
                if st >= len(self._plan.steps):
                    self.cursor = (ep + 1, 0)
                    continue
                self._it = self.loader.iter_epoch(ep, self.host, self.n_hosts, counts_allgather_jax, start_step=st,
                                                  plan=self._plan)
            try:
                _, batches = next(self._it)
            except StopIteration:
                self._it = None
                self.cursor = (ep + 1, 0)
                continue
            self.cursor = (ep, st + 1)
            k += 1
            yield stack_local(batches, self.mesh), batches

    def close(self):
        self._it = None
        self.loader.close()



class StreamEvalSet:
    """Evaluation split read by the stream workers: every structure (or a fixed seeded subset of max_structures)
    exactly once per pass, in file order, packed into the training buckets (plus one extra bucket for structures that
    fit none of them).  Every host computes the same packs (deterministic) and takes packs host, host + n_hosts, ...
    of each bucket; a device without a pack in a group gets a copy of the bucket's first pack with weight 0 (as
    training_utils.evaluate_sharded does for the in-memory sets).

        ev = StreamEvalSet(table, buckets, atomic_numbers, atom_energies, cutoff, n_local, workers, dtype, fields)
        for batch, weights in ev.iter_groups(mesh): ...      # sharded batch, (n_devices,) weights over 'dp'
    """

    def __init__(self, table, buckets, atomic_numbers, atom_energies, cutoff, n_local=1, workers=4,
                 dtype=np.float64, fields=None, max_structures=None, seed=0, block_rows=128, name='eval'):
        import jax
        self.name, self.n_local, self.dtype = name, int(n_local), dtype
        n = len(table['n_atoms'])
        rows = np.arange(n)
        if max_structures and n > int(max_structures):
            rows = np.sort(np.random.default_rng(seed).choice(n, int(max_structures), replace=False))
        rows = rows[np.lexsort((table['key'][rows], table['file'][rows]))]
        na, ne = table['n_atoms'], table['n_edges']
        self.buckets = list(buckets)
        bidx = P.bucket_of(ne[rows], self.buckets) if len(rows) else np.zeros(0, np.int64)
        fits = np.array([na[r] + 1 <= self.buckets[b].pad_nodes and ne[r] <= self.buckets[b].pad_edges
                         for r, b in zip(rows, bidx)], bool)
        if len(rows) and not fits.all():
            # one extra shape (one more compilation) for the structures larger than every training bucket
            big = rows[~fits]
            self.buckets.append(P.Bucket(int(math.ceil((na[big].max() + 1) / 64) * 64),
                                         int(math.ceil(ne[big].max() / 1024) * 1024), self.buckets[-1].max_graphs))
            bidx = np.where(fits, bidx, len(self.buckets) - 1)
        host, n_hosts = jax.process_index(), jax.process_count()
        self.all_packs = [P._pack(rows[bidx == b], na, ne, bk) for b, bk in enumerate(self.buckets)]
        self.mine = [ps[host::n_hosts] for ps in self.all_packs]
        # the same group count on every host: ceil(max over hosts of its pack count / n_local)
        self.groups = [-(-len(ps[0::n_hosts]) // self.n_local) if ps else 0 for ps in self.all_packs]
        self.n_structures = int(sum(len(p) for ps in self.mine for p in ps))
        self.loader = StreamLoader(table, self.buckets, atomic_numbers, atom_energies, cutoff, n_local=self.n_local,
                                   workers=workers, dtype=dtype, fields=fields, block_rows=block_rows, shuffle=False)

    def iter_groups(self, mesh):
        import jax
        from jax.sharding import NamedSharding, PartitionSpec
        from bam_omat24.data_nnx import fixed_collate_fn
        sh = NamedSharding(mesh, PartitionSpec('dp'))
        order = []          # (bucket, pack) in reading order: per bucket, groups of n_local own packs + the filler
        for k, ps in enumerate(self.mine):
            for gi in range(self.groups[k]):
                own = ps[gi * self.n_local:(gi + 1) * self.n_local]
                order.append((k, own, len(own) < self.n_local))
        need = []
        for k, own, short in order:
            for p in own + ([self.all_packs[k][0]] if short else []):
                need += [int(r) for r in p]
        it = self.loader._graphs(need)       # a structure read twice (filler) is yielded twice
        cache = {}

        def build(k, p):
            got = []
            for m in p:
                m = int(m)
                while not cache.get(m):
                    r, g = next(it)
                    cache.setdefault(r, []).append(g)
                got.append(cache[m].pop(0))
            bk = self.buckets[k]
            return fixed_collate_fn([_to_graph(_cast(g, self.dtype)) for g in got], bk.max_graphs, bk.pad_nodes,
                                    bk.pad_edges)

        for k, own, short in order:
            batches = [build(k, p) for p in own]
            w = np.zeros(self.n_local, np.float32); w[:len(batches)] = 1.0
            if short:
                filler = build(k, self.all_packs[k][0])
                batches += [filler] * (self.n_local - len(batches))
            wg = (jax.device_put(w, sh) if jax.process_count() == 1
                  else jax.make_array_from_process_local_data(sh, w))
            yield stack_local(batches, mesh), wg

    def close(self):
        self.loader.close()
