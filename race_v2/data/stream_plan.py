"""Per-epoch data plan of the streaming pipeline from the size table only (pure numpy, JAX-free, deterministic;
docs/stream_loader_design.md sections 4.4-4.5).  Importable as a top-level module (like stream_io).

    table   = load_size_table(sizes_dir, cutoff)                      # sizes.npz + sizes_meta.json
    buckets = derive_buckets(table['n_atoms'], table['n_edges'], n_buckets=1, graphs_per_pack=16, max_graphs=64)
    plan    = plan_data_epoch(table, buckets, data_epoch=0, host=0, n_hosts=1, n_local=4, seed=0)
    plan.packs[b][p]  -> row indices of the table (structures) of pack p of bucket b, in stream order
    plan.steps        -> list of (bucket, [pack ids of the n_local devices of this host]) for every step

Packing rule = the fixed-shape loader (data.loaders.fixed_collate_fn): nodes < pad_nodes, edges <= pad_edges,
graphs <= max_graphs; greedy next-fit in stream order.  Lock-step across hosts: steps per bucket = min over hosts of
floor(packs / n_local); counts_allgather(local_counts) -> per-host counts (identity for one host).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Bucket:
    pad_nodes: int
    pad_edges: int
    max_graphs: int
    edge_lo: int = 0           # structures with edge_lo <= n_edges < edge_hi belong to this bucket
    edge_hi: int = 2 ** 62


@dataclass
class HostPlan:
    packs: list                 # per bucket: list of int arrays (table rows)
    steps: list                 # (bucket, [pack ids]) per step
    dropped: int                # structures in packs dropped for lock-step
    signature: str
    n_structures: int
    fill: dict = field(default_factory=dict)


def load_size_table(sizes_dir, cutoff=None) -> dict:
    d = Path(sizes_dir)
    meta = json.load(open(d / 'sizes_meta.json'))
    if cutoff is not None and abs(float(meta['cutoff']) - float(cutoff)) > 1e-9:
        raise ValueError(f"{d}: size table made for cutoff {meta['cutoff']}, model cutoff {cutoff}")
    z = np.load(d / 'sizes.npz')
    return {'n_atoms': z['n_atoms'].astype(np.int64), 'n_edges': z['n_edges'].astype(np.int64),
            'file': z['file_index'].astype(np.int64), 'key': z['key'].astype(np.int64), 'meta': meta,
            'cell': z['cell'] if 'cell' in z.files else None}


def _budget(n, e, graphs_per_pack):
    pn = int(math.ceil(max(graphs_per_pack * n.mean(), n.max() + 1) / 64) * 64)
    pe = int(math.ceil(max(graphs_per_pack * e.mean(), e.max()) / 1024) * 1024)
    return pn, pe


def derive_buckets(n_atoms, n_edges, n_buckets=1, graphs_per_pack=16, max_graphs=64):
    """Equal-count split by edges per structure; budget per bucket as training.train pad_mode 'fixed'."""
    n_atoms, n_edges = np.asarray(n_atoms), np.asarray(n_edges)
    if n_buckets <= 1:
        pn, pe = _budget(n_atoms, n_edges, graphs_per_pack)
        return [Bucket(pn, pe, max_graphs)]
    cuts = np.unique(np.quantile(n_edges, np.linspace(0, 1, n_buckets + 1)[1:-1]).astype(np.int64))
    edges = [0] + [int(c) for c in cuts] + [2 ** 62]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (n_edges >= lo) & (n_edges < hi)
        if m.any():
            pn, pe = _budget(n_atoms[m], n_edges[m], graphs_per_pack)
            out.append(Bucket(pn, pe, max_graphs, lo, hi))
    return out


def bucket_of(n_edges, buckets):
    b = np.full(len(n_edges), -1, np.int64)
    for i, bk in enumerate(buckets):
        b[(n_edges >= bk.edge_lo) & (n_edges < bk.edge_hi)] = i
    if (b < 0).any():
        raise ValueError("structure outside every bucket")
    return b


def _blocks(table, block_rows):
    """Blocks of consecutive rows of one file (rows sorted by (file, key) in the table)."""
    f = table['file']
    order = np.lexsort((table['key'], f))
    fs = f[order]                                        # sorted by file: one slice per file (O(N), not O(N * files))
    cut = np.flatnonzero(np.diff(fs)) + 1
    out = []
    for s, e in zip(np.concatenate([[0], cut]).tolist(), np.concatenate([cut, [len(fs)]]).tolist()):
        if e > s:
            fi, rows = int(fs[s]), order[s:e]
            out += [(fi, rows[k:k + block_rows]) for k in range(0, len(rows), block_rows)]
    return out


def _pack(rows, n_atoms, n_edges, bk):
    packs, cur, cn, ce = [], [], 0, 0
    rows = np.asarray(rows)
    for r, a, e in zip(rows.tolist(), np.asarray(n_atoms)[rows].tolist(), np.asarray(n_edges)[rows].tolist()):
        if a + 1 > bk.pad_nodes or e > bk.pad_edges:
            raise ValueError(f"structure {r} ({a} atoms, {e} edges) exceeds its bucket {bk}")
        if cur and (cn + a + 1 > bk.pad_nodes or ce + e > bk.pad_edges or len(cur) >= bk.max_graphs):
            packs.append(np.asarray(cur, np.int64)); cur, cn, ce = [], 0, 0
        cur.append(r); cn += a; ce += e
    if cur:
        packs.append(np.asarray(cur, np.int64))
    return packs


def plan_data_epoch(table, buckets, data_epoch, host, n_hosts, n_local, seed=0, block_rows=128, shuffle=True,
                    counts_allgather=None, mix_blocks=1) -> HostPlan:
    """mix_blocks > 1: the rows of every mix_blocks consecutive blocks of the stream are shuffled together, so a pack
    draws from several blocks (consecutive rows of a file are often the same molecule; a pack from one block is then
    low in diversity).  Reads stay within a window of block_rows * mix_blocks rows."""
    n_atoms, n_edges = table['n_atoms'], table['n_edges']
    rng = np.random.default_rng([int(seed), int(data_epoch)])
    blocks = _blocks(table, block_rows)
    # files -> hosts: random order, then greedy to the least-loaded host (edge totals)
    files = sorted({fi for fi, _ in blocks})
    load_f = {fi: 0 for fi in files}
    for fi, rows in blocks:
        load_f[fi] += int(n_edges[rows].sum())
    forder = list(rng.permutation(len(files))) if shuffle else list(range(len(files)))
    host_load, owner = np.zeros(n_hosts, np.int64), {}
    for k in forder:
        fi = files[k]
        h = int(np.argmin(host_load)); owner[fi] = h; host_load[h] += load_f[fi]
    border = rng.permutation(len(blocks)) if shuffle else np.arange(len(blocks))
    stream = []
    for bi in border:
        fi, rows = blocks[bi]
        if owner[fi] != host:
            continue
        stream.append(rng.permutation(rows) if shuffle else rows)
    if shuffle and mix_blocks > 1:
        stream = [rng.permutation(np.concatenate(stream[i:i + mix_blocks])) for i in range(0, len(stream), mix_blocks)]
    rows = np.concatenate(stream) if stream else np.zeros(0, np.int64)
    bidx = bucket_of(n_edges[rows], buckets) if len(rows) else np.zeros(0, np.int64)
    packs = [_pack(rows[bidx == b], n_atoms, n_edges, bk) for b, bk in enumerate(buckets)]
    local = np.array([len(p) // n_local for p in packs], np.int64)
    allc = np.asarray(counts_allgather(local)) if counts_allgather is not None else local[None]
    steps_b = allc.min(axis=0)
    dropped = sum(int(sum(len(p) for p in packs[b][steps_b[b] * n_local:])) for b in range(len(buckets)))
    # stratified interleaving of the buckets (identical on every host)
    keys = [((t + 0.5) / max(steps_b[b], 1), b, t) for b in range(len(buckets)) for t in range(int(steps_b[b]))]
    keys.sort()
    steps = [(b, list(range(t * n_local, (t + 1) * n_local))) for _, b, t in keys]
    fill = {}
    for b, bk in enumerate(buckets):
        used = packs[b][:steps_b[b] * n_local]
        if used:
            fill[b] = (float(sum(n_edges[p].sum() for p in used)) / (len(used) * bk.pad_edges),
                       float(sum(n_atoms[p].sum() for p in used)) / (len(used) * bk.pad_nodes))
    h = hashlib.sha256()
    h.update(json.dumps([seed, data_epoch, n_hosts, n_local, block_rows, [b.__dict__ for b in buckets]]
                        + ([mix_blocks] if mix_blocks > 1 else [])).encode())
    h.update(np.ascontiguousarray(n_edges).tobytes()); h.update(np.ascontiguousarray(table['key']).tobytes())
    return HostPlan(packs=packs, steps=steps, dropped=dropped, signature=h.hexdigest()[:16],
                    n_structures=int(len(rows)), fill=fill)
