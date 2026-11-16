"""Streaming data pipeline, JAX-free parts (stream_io, stream_plan) and the equivalence with the in-memory graphs
(docs/stream_loader_design.md tests T2, T3).  The last test imports data.xyz_graphs (jraph -> JAX): run it on a compute node."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'race_v2' / 'data'))
import stream_io as S  # noqa: E402
import stream_plan as P  # noqa: E402

MOL = ROOT / 'dataset_3BPA' / 'train_300K.xyz'
PER = Path(os.environ.get('RACE_OAM_SUBSETS', '/scratch/t589a02/datasets_ext/oam_subsets')) / 'omat24_valid.xyz'   # periodic, with stress


def _small(tmp_path, src, n):
    import ase.io
    p = tmp_path / f'{src.stem}_{n}.xyz'
    ase.io.write(str(p), ase.io.read(str(src), index=f':{n}'), format='extxyz')
    return p


def test_extxyz_offsets_and_blocks(tmp_path):
    import ase.io
    for src in [MOL] + ([PER] if PER.exists() else []):
        p = _small(tmp_path, src, 40)
        frames = ase.io.read(str(p), index=':')
        off = S.extxyz_offsets(p)
        assert len(off) == len(frames)
        sts = S.ExtxyzSource([p], [off]).read_block(0, [7, 3, 39])
        for st, k in zip(sts, [7, 3, 39]):
            a = frames[k]
            assert np.array_equal(st['numbers'], a.numbers)
            assert np.allclose(st['positions'], a.positions, atol=0, rtol=0)
            assert np.allclose(st['forces'], a.get_forces(), atol=0, rtol=0)
            assert st['energy'] == a.get_potential_energy()


def test_size_pass_matches_graph_sizes_tool(tmp_path):
    import subprocess
    p = _small(tmp_path, MOL, 60)
    meta = S.size_pass([f'extxyz:{p}'], 5.0, tmp_path / 'sz', workers=2, block=16)
    t = P.load_size_table(tmp_path / 'sz', 5.0)
    r = subprocess.run([sys.executable, '-I', str(ROOT / 'race_v2' / 'data' / 'graph_sizes.py'), str(p), '--cutoff', '5.0',
                        '--workers', '1', '--out-dir', str(tmp_path / 'gs')], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = np.loadtxt(tmp_path / 'gs' / f'{p.stem}.sizes.tsv', comments='#', skiprows=3, dtype=np.int64, ndmin=2)
    assert np.array_equal(t['n_atoms'], d[:, 1]) and np.array_equal(t['n_edges'], d[:, 2])
    assert meta['n'] == 60 and abs(meta['avg_n_neighbors'] - d[:, 2].sum() / d[:, 1].sum()) < 1e-12
    with pytest.raises(ValueError):
        P.load_size_table(tmp_path / 'sz', 6.0)


def _synthetic(n=5000, n_files=13, seed=1):
    rng = np.random.default_rng(seed)
    na = rng.integers(2, 120, n)
    return {'n_atoms': na, 'n_edges': na * rng.integers(10, 40, n), 'file': rng.integers(0, n_files, n),
            'key': rng.permutation(n), 'meta': {}}


@pytest.mark.parametrize('n_buckets', [1, 3])
def test_plan_covers_every_structure_once(n_buckets):
    t = _synthetic()
    bks = P.derive_buckets(t['n_atoms'], t['n_edges'], n_buckets=n_buckets, graphs_per_pack=16, max_graphs=64)
    n_hosts, n_local = 4, 2
    # lock-step counts from all hosts, as the allgather would give them
    local = [P.plan_data_epoch(t, bks, 3, h, n_hosts, n_local, seed=7) for h in range(n_hosts)]
    counts = np.stack([[len(p) // n_local for p in pl.packs] for pl in local])
    plans = [P.plan_data_epoch(t, bks, 3, h, n_hosts, n_local, seed=7, counts_allgather=lambda c: counts)
             for h in range(n_hosts)]
    seen = []
    for pl in plans:
        assert len({pl.signature for pl in plans}) == 1
        assert len(pl.steps) == len(plans[0].steps)                         # lock-step
        for b, ids in pl.steps:
            for i in ids:
                rows = pl.packs[b][i]
                bk = bks[b]
                assert t['n_atoms'][rows].sum() < bk.pad_nodes and t['n_edges'][rows].sum() <= bk.pad_edges
                assert len(rows) <= bk.max_graphs
                seen.append(rows)
    seen = np.concatenate(seen)
    assert len(np.unique(seen)) == len(seen)                                  # no structure twice
    assert len(seen) + sum(pl.dropped for pl in plans) == len(t['n_atoms'])    # all but the reported leftovers
    # deterministic, and different for another epoch
    again = P.plan_data_epoch(t, bks, 3, 1, n_hosts, n_local, seed=7, counts_allgather=lambda c: counts)
    assert all(np.array_equal(a, b) for a, b in zip(again.packs[0], plans[1].packs[0]))
    other = P.plan_data_epoch(t, bks, 4, 1, n_hosts, n_local, seed=7)
    assert other.signature != plans[1].signature


def test_graph_arrays_equal_preprocess_graph():
    """T2: needs data.xyz_graphs (imports jraph / JAX) -> compute node."""
    pytest.importorskip('jraph')
    if os.environ.get('SLURM_JOB_ID') is None:
        pytest.skip('imports JAX: run inside a Slurm job')
    import ase.io
    from race_v2.data.xyz_graphs import preprocess_graph, atomic_numbers_to_indices
    for src in [MOL] + ([PER] if PER.exists() else []):
        frames = ase.io.read(str(src), index=':50')
        zs = sorted({int(z) for a in frames for z in a.numbers})
        idx = atomic_numbers_to_indices(zs)
        e0 = list(np.linspace(-3.0, -1.0, len(zs)))
        for a in frames:
            if src == PER:
                a.pbc = True
            g = preprocess_graph(a, idx, e0, 5.0, targets=True)
            d = S.graph_arrays(S.structure_from_atoms(a), idx, e0, 5.0)
            assert np.array_equal(g.senders, d['senders']) and np.array_equal(g.receivers, d['receivers'])
            for k in ('species', 'positions', 'forces'):
                assert np.array_equal(g.nodes[k], d[k]), k
            assert np.array_equal(g.globals['energy'], d['energy'])
            if g.edges is not None:
                assert np.array_equal(g.edges['Sij'], d['Sij'])
                for k in ('cell', 'cutoff', 'volume', 'stress'):
                    assert np.allclose(g.globals[k], d[k], rtol=1e-7, atol=0), k


def test_stream_loader_batches_equal_in_memory(tmp_path):
    """StreamLoader packs (workers + graph_arrays) == fixed_collate_fn of preprocess_graph graphs (compute node)."""
    pytest.importorskip('jraph')
    if os.environ.get('SLURM_JOB_ID') is None:
        pytest.skip('imports JAX: run inside a Slurm job')
    import ase.io
    from race_v2.data.xyz_graphs import preprocess_graph, atomic_numbers_to_indices
    from race_v2.data.loaders import fixed_collate_fn
    from race_v2.data.stream_data import StreamLoader
    src = PER if PER.exists() else MOL
    p = _small(tmp_path, src, 120)
    S.size_pass([f'extxyz:{p}'], 5.0, tmp_path / 'sz', workers=2, block=32)
    t = P.load_size_table(tmp_path / 'sz', 5.0)
    frames = ase.io.read(str(p), index=':')
    zs = sorted({int(z) for a in frames for z in a.numbers})
    e0 = list(np.linspace(-3.0, -1.0, len(zs)))
    bks = P.derive_buckets(t['n_atoms'], t['n_edges'], n_buckets=2, graphs_per_pack=8, max_graphs=32)
    ld = StreamLoader(t, bks, zs, e0, 5.0, n_local=2, workers=2, seed=3, block_rows=16, dtype=np.float32)
    try:
        pl = ld.plan(0)
        idx = atomic_numbers_to_indices(zs)
        k = 0
        for (b, batches), (b2, ids) in zip(ld.iter_epoch(0), pl.steps):
            assert b == b2
            for bt, i in zip(batches, ids):
                rows = pl.packs[b][i]
                gs = [preprocess_graph(frames[int(t['key'][r])], idx, e0, 5.0) for r in rows]
                rb = fixed_collate_fn(gs, bks[b].max_graphs, bks[b].pad_nodes, bks[b].pad_edges)
                assert np.array_equal(np.asarray(bt.senders), np.asarray(rb.senders))
                assert np.array_equal(np.asarray(bt.nodes['positions']), np.asarray(rb.nodes['positions']))
                assert np.array_equal(np.asarray(bt.globals['energy']), np.asarray(rb.globals['energy']))
                k += 1
        assert k == sum(len(ids) for _, ids in pl.steps)
    finally:
        ld.close()
