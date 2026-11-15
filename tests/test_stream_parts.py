"""stream.train.parts with repeat (train_3bpa._stream_parts): every copy of a repeated part has its own file indices,
so the copies of a structure are spread over the data epoch instead of landing in the same read block, and the
loader workers cache the open sources by path (an lmdb environment may be opened only once per process)."""
import io
import json

import numpy as np

from race_v2.data import stream_plan as SP
from race_v2.data import stream_worker as SW
from race_v2.training import train as T


def _fake_table(d, name, n_files, rows_per_file, seed):
    """A size table as stream_io.size_pass writes it (sizes.npz + sizes_meta.json), without any data files."""
    rng = np.random.default_rng(seed)
    n = n_files * rows_per_file
    na = rng.integers(2, 30, n); ne = na * rng.integers(20, 60, n)
    out = d / name; out.mkdir()
    files = [f'/nonexistent/{name}_{i:03d}.aselmdb' for i in range(n_files)]
    np.savez(out / 'sizes.npz', n_atoms=na.astype(np.uint16), n_edges=ne.astype(np.uint32),
             file_index=np.repeat(np.arange(n_files), rows_per_file).astype(np.uint32),
             key=np.tile(np.arange(1, rows_per_file + 1), n_files))
    src = f'aselmdb:/nonexistent/{name}'
    json.dump({'cutoff': 7.0, 'sources': [{'spec': src, 'kind': 'aselmdb', 'files': files}], 'files': files,
               'kinds': ['aselmdb'] * n_files, 'n': int(n), 'sum_atoms': int(na.sum())}, open(out / 'sizes_meta.json', 'w'))
    return {'sources': [src], 'sizes': str(out)}


def _merged(tmp_path, rep=8):
    a = _fake_table(tmp_path, 'small', 3, 400, 0); b = _fake_table(tmp_path, 'large', 5, 2000, 1)
    table, _, tr = T._stream_parts([dict(a, repeat=rep), dict(b, repeat=1)], 7.0, 1, tmp_path, True, io.StringIO())
    return table, tr


def test_parts_rows_and_file_indices(tmp_path):
    table, tr = _merged(tmp_path)
    assert len(table['n_atoms']) == 8 * 1200 + 10000
    assert len(tr['sources']) == 2
    m = table['meta']
    assert len(m['files']) == len(m['kinds']) == 8 * 3 + 5          # the 3 paths of the repeated part, 8 times
    assert len(set(m['files'])) == 3 + 5
    # the 8 copies of one structure (same path, same key) carry 8 different file indices
    paths = np.array(m['files'])[table['file']]
    sel = (paths == m['files'][0]) & (table['key'] == 7)
    assert sel.sum() == 8 and len(set(table['file'][sel].tolist())) == 8


def test_parts_copies_are_spread_over_the_epoch(tmp_path):
    table, _ = _merged(tmp_path)
    m = table['meta']
    ident = np.array([hash((m['files'][f], int(k))) for f, k in zip(table['file'], table['key'])])
    # no read block holds two copies of the same structure
    for _, rows in SP._blocks(table, 128):
        assert len(set(ident[rows].tolist())) == len(rows)
    bk = SP.derive_buckets(table['n_atoms'], table['n_edges'], graphs_per_pack=32, max_graphs=128)
    pl = SP.plan_data_epoch(table, bk, 0, 0, 1, 1, seed=42, block_rows=128, shuffle=True, mix_blocks=16)
    rows = np.concatenate([np.concatenate([pl.packs[b][i] for i in ids]) for b, ids in pl.steps])
    pos = {}
    for p, r in enumerate(rows.tolist()):
        pos.setdefault(ident[r], []).append(p)
    spans = [max(v) - min(v) for v in pos.values() if len(v) == 8]
    assert len(spans) > 1000
    # 8 uniform draws span 7/9 of the epoch on average; the old layout (shared file index) gave ~0.1 %
    assert np.median(spans) > 0.5 * len(rows)


def test_worker_source_cache_is_keyed_by_path(monkeypatch):
    opened = []

    class _S:
        @staticmethod
        def AselmdbSource(paths):
            opened.append(paths[0]); return object()
    monkeypatch.setitem(SW._W, 'S', _S); monkeypatch.setitem(SW._W, 'src', {})
    monkeypatch.setitem(SW._W, 'kinds', ['aselmdb'] * 4); monkeypatch.setitem(SW._W, 'files', ['a', 'b', 'a', 'b'])
    monkeypatch.setitem(SW._W, 'keys', ((), ()))
    assert SW._source(0) is SW._source(2) and SW._source(1) is SW._source(3)
    assert opened == ['a', 'b']
