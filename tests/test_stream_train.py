"""Streaming training path (data_format "stream", 2026-11-09): optional graph fields equal to the in-memory path,
evaluation sets cover every structure once, the resumable cursor.  CPU (JAX on CPU is fine)."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bam_omat24'))
import stream_io as S  # noqa: E402
import stream_plan as P  # noqa: E402

pytest.importorskip('jraph')


def _frames(n=40, periodic=False, seed=0):
    import ase
    from ase.calculators.singlepoint import SinglePointCalculator
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        m = int(rng.integers(3, 9))
        z = rng.choice([1, 6, 8], m)
        pos = rng.random((m, 3)) * 4.0
        a = ase.Atoms(numbers=z, positions=pos, cell=np.eye(3) * 6.0 if periodic else None, pbc=periodic)
        a.info['charge'] = int(k % 3) - 1
        a.info['spin'] = 1 + (k % 2)
        a.arrays['ref_charges'] = rng.normal(size=m) * 0.3
        frag = (np.arange(m) >= m // 2).astype(int)
        a.arrays['is_ligand'] = frag
        a.info['fragment_charges'] = [a.info['charge'], 0]
        a.calc = SinglePointCalculator(a, energy=float(rng.normal()), forces=rng.normal(size=(m, 3)),
                                       **({'stress': rng.normal(size=6)} if periodic else {}))
        out.append(a)
    return out


FIELDS = dict(charge_key='charge', spin_key='spin', charge_label_key='ref_charges',
              fragments=('is_ligand', 'fragment_charges', 2))


@pytest.mark.parametrize('periodic', [False, True])
def test_graph_fields_equal_in_memory(tmp_path, periodic):
    import ase.io
    from bam_omat24.data_3bpa import preprocess_graph, atomic_numbers_to_indices
    from bam_omat24.stream_data import _to_graph
    from graph_fields import field_keys
    p = tmp_path / 'f.xyz'
    ase.io.write(str(p), _frames(periodic=periodic), format='extxyz')
    frames = ase.io.read(str(p), ':')
    zs = [1, 6, 8]; idx = atomic_numbers_to_indices(zs); e0 = [-1.0, -2.0, -3.0]
    ik, ak = field_keys(**FIELDS)
    src = S.ExtxyzSource([str(p)], info_keys=ik, array_keys=ak)
    sts = src.read_block(0, np.arange(len(frames)))
    for a, st in zip(frames, sts):
        g = preprocess_graph(a, idx, e0, 3.0, **FIELDS)
        h = _to_graph(S.graph_arrays(st, idx, e0, 3.0, FIELDS))
        assert set(g.nodes) == set(h.nodes) and set(g.globals) == set(h.globals)
        for k in g.nodes:
            assert np.array_equal(g.nodes[k], h.nodes[k]), k
        for k in g.globals:
            assert np.array_equal(g.globals[k], h.globals[k]), k
        assert np.array_equal(g.senders, h.senders) and np.array_equal(g.receivers, h.receivers)


def test_aselmdb_row_fields():
    """decode_row puts key_value_pairs / data into info, so the fields of an ASE db row are found."""
    d = {'numbers': [1, 8], 'positions': [[0, 0, 0], [0, 0, 1.0]], 'cell': [[0] * 3] * 3, 'pbc': [False] * 3,
         'energy': -1.0, 'forces': [[0, 0, 0]] * 2, 'key_value_pairs': {'charge': -1, 'spin': 2},
         'data': {'ref_charges': {'__ndarray__': [[2], 'float64', [-0.4, -0.6]]}}}
    st = S.decode_row(d=d)
    from graph_fields import extra_fields
    nx, gx = extra_fields(st['info'], st['arrays'], 2, charge_key='charge', spin_key='spin',
                          charge_label_key='ref_charges')
    assert gx['charge'][0] == -1 and gx['spin'][0] == 2 and np.allclose(nx['ref_charges'], [-0.4, -0.6])


def _table(tmp_path, n=60):
    import ase.io
    p = tmp_path / 't.xyz'
    ase.io.write(str(p), _frames(n), format='extxyz')
    S.size_pass([f'extxyz:{p}'], 3.0, tmp_path / 'sz', workers=1, block=16)
    return P.load_size_table(tmp_path / 'sz', 3.0)


def test_eval_set_every_structure_once(tmp_path):
    import jax
    from bam_omat24.data_nnx import setup_mesh
    from bam_omat24.stream_data import StreamEvalSet
    import jraph
    t = _table(tmp_path)
    bk = P.derive_buckets(t['n_atoms'], t['n_edges'], n_buckets=2, graphs_per_pack=4, max_graphs=8)
    mesh, nd = setup_mesh()
    ev = StreamEvalSet(t, bk, [1, 6, 8], [-1.0, -2.0, -3.0], 3.0, n_local=nd, workers=1, dtype=np.float32,
                       fields=FIELDS)
    try:
        seen = sorted(int(r) for ps in ev.mine for p in ps for r in p)
        assert seen == list(range(len(t['n_atoms'])))
        n_real = 0
        for batch, w in ev.iter_groups(mesh):
            m = np.asarray(jax.vmap(jraph.get_graph_padding_mask)(batch)).sum(axis=1)
            n_real += int(np.sum(m * np.asarray(w)))
        assert n_real == len(t['n_atoms'])
        sub = StreamEvalSet(t, bk, [1, 6, 8], [-1.0, -2.0, -3.0], 3.0, n_local=nd, workers=1, max_structures=17)
        assert sub.n_structures == 17
        sub.close()
    finally:
        ev.close()


@pytest.mark.parametrize('mix', [1, 4])
def test_stream_steps_resume_cursor(tmp_path, mix):
    """take(5) + take(7) from a fresh StreamSteps at the saved cursor = take(12) in one go (same packs)."""
    from bam_omat24.data_nnx import setup_mesh
    from bam_omat24.stream_data import StreamLoader, StreamSteps
    t = _table(tmp_path, 80)
    bk = P.derive_buckets(t['n_atoms'], t['n_edges'], n_buckets=1, graphs_per_pack=4, max_graphs=8)
    mesh, nd = setup_mesh()

    def run(cursor, n):
        ld = StreamLoader(t, bk, [1, 6, 8], [-1.0, -2.0, -3.0], 3.0, n_local=nd, workers=1, seed=5, block_rows=8,
                          dtype=np.float32, mix_blocks=mix)
        ss = StreamSteps(ld, mesh, cursor)
        out = [np.asarray(b[0].nodes['positions']).copy() for _, b in ss.take(n)]
        cur = ss.cursor
        ss.close()
        return out, cur

    pl = P.plan_data_epoch(t, bk, 0, 0, 1, nd, seed=5, block_rows=8, mix_blocks=mix)
    per_epoch = len(pl.steps)
    assert sorted(int(r) for p in pl.packs[0] for r in p) == list(range(len(t['n_atoms'])))
    n_all = per_epoch + 3          # crosses a data-epoch boundary
    full, _ = run((0, 0), n_all)
    a, cur = run((0, 0), 5)
    b, _ = run(cur, n_all - 5)
    assert len(full) == n_all
    for x, y in zip(full, a + b):
        assert np.array_equal(x, y)


def test_mixed_blocks_spread_packs(tmp_path):
    """With mix_blocks, a pack draws from several blocks of the file (diversity), each structure still once."""
    t = _table(tmp_path, 80)
    bk = P.derive_buckets(t['n_atoms'], t['n_edges'], n_buckets=1, graphs_per_pack=8, max_graphs=8)
    span = {}
    for mix in (1, 4):
        pl = P.plan_data_epoch(t, bk, 0, 0, 1, 1, seed=1, block_rows=8, mix_blocks=mix)
        assert sorted(int(r) for p in pl.packs[0] for r in p) == list(range(80))
        span[mix] = np.mean([len(np.unique(t['key'][p] // 8)) for p in pl.packs[0]])
    assert span[4] > span[1] + 1
