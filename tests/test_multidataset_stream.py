"""Multi-dataset stream sources (bam_omat24/stream_io.py, 2026-11-11): seeded subsets per shard, meta-data
exclusions, task rules, E0 per task, energy key, mixed periodic / open-boundary graphs, optional charge and dipole
labels, and the cell columns of the size table.  JAX-free (synthetic aselmdb shards)."""
import json
import os
import sys
import zlib

import numpy as np
import pytest

lmdb = pytest.importorskip("lmdb")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bam_omat24"))
import stream_io as S  # noqa: E402

Z = [1, 8, 26]
IDX = {z: i for i, z in enumerate(Z)}


def _row(rng, periodic, numbers, **data):
    n = len(numbers)
    cell = np.diag([6.0, 6.5, 7.0]) if periodic else np.zeros((3, 3))
    d = {'numbers': list(numbers), 'positions': (rng.random((n, 3)) * 5.0).tolist(), 'cell': cell.tolist(),
         'pbc': [periodic] * 3, 'energy': float(-3.0 * n), 'forces': rng.normal(size=(n, 3)).tolist(), 'data': data}
    if periodic:
        d['stress'] = rng.normal(size=6).tolist()
    return d


def _shard(path, rows):
    env = lmdb.open(str(path), subdir=False, map_size=1 << 24)
    with env.begin(write=True) as txn:
        for i, r in enumerate(rows, 1):
            txn.put(str(i).encode(), zlib.compress(json.dumps(r).encode()))
        txn.put(b'nextid', zlib.compress(json.dumps(len(rows) + 1).encode()))
    env.close()


@pytest.fixture()
def corpus(tmp_path):
    rng = np.random.default_rng(0)
    bulk = tmp_path / 'bulk'; bulk.mkdir()
    for k in range(2):
        _shard(bulk / f'data{k:04d}.aselmdb',
               [_row(rng, True, [26, 8, 8] if i % 2 else [1, 1, 8], alt_energy=-1.0 * i) for i in range(20)])
    mol = tmp_path / 'mol'; mol.mkdir()
    _shard(mol / 'data0000.aselmdb',
           [_row(rng, False, [8, 1, 1], charge=i % 2, spin=1 + i % 2,
                 multipoles=[[0, 0, 0], [0.1 * i, 0.0, -0.2], [0] * 6],
                 **({'nbo_charges': [-0.8 + 0.5 * (i % 2), 0.4, 0.4 + 0.5 * (i % 2)]} if i < 6 else {}))
            for i in range(12)])
    meta = tmp_path / 'meta'; meta.mkdir()
    ex = np.zeros(20, bool); ex[[0, 3]] = True
    np.savez(meta / 'data0000.npz', row_id=np.arange(1, 21), exclude=ex)
    return tmp_path


def test_subset_exclusion_and_size_table(corpus):
    src = {'name': 'bulk', 'path': f'aselmdb:{corpus / "bulk"}', 'max_frames_per_shard': 7, 'meta_dir': str(corpus / 'meta'),
           'task': 2, 'task_rules': {'mp2020_hubbard': 5}}
    mol = {'name': 'mol', 'path': f'aselmdb:{corpus / "mol"}', 'task': 3, 'charge_label_key': 'nbo_charges',
           'charge_label_scale': 0.5}
    kind, files = S._source_files(src)
    o = S.parse_source(src)['opts']
    k0, k0b = S.source_keys(kind, files[0], o, 0), S.source_keys(kind, files[0], o, 0)
    assert len(k0) == 7 and np.array_equal(k0, k0b) and np.all(np.diff(k0) > 0)
    assert not set(k0) & {1, 4}                                    # rows 1 and 4 are flagged exclude (ids are 1-based)
    k1 = S.source_keys(kind, files[1], o, 1)
    assert not np.array_equal(k0, k1)                              # another draw per shard
    assert not np.array_equal(k0, S.source_keys(kind, files[0], dict(o, subset_seed=1), 0))
    meta = S.size_pass([src, mol], 5.0, corpus / 'sizes', workers=1)
    assert meta['n'] == 7 + 7 + 12 and meta['n_periodic'] == 14 and meta['file_source'] == [0, 0, 1]
    assert abs(meta['max_cell_volume'] - 6.0 * 6.5 * 7.0) < 1e-3 and np.allclose(meta['max_cell_length'], [6.0, 6.5, 7.0])
    assert [m['n'] for m in meta['sources']] == [14, 12]
    z = np.load(corpus / 'sizes' / 'sizes.npz')
    assert z['cell'].shape == (26, 4) and np.all(z['cell'][14:] == 0)
    fo = S.file_opts(meta)
    assert fo[0]['task'] == 2 and fo[2]['charge_label_key'] == 'nbo_charges'
    with pytest.raises(ValueError):
        S.parse_source({'path': 'aselmdb:x', 'max_frames': 3})


def test_graph_arrays_options(corpus):
    e0 = {2: np.array([-1.0, -2.0, -5.0]), 5: np.array([-1.5, -2.5, -9.0]), 3: np.array([-10.0, -20.0, 0.0])}
    fields = dict(charge_key='charge', spin_key='spin', mixed=True, opt_charge_labels=True, dipole=True, task=0)
    bulk = S.AselmdbSource([str(corpus / 'bulk' / 'data0000.aselmdb')]).read_block(0, [1, 2])
    o = {'task': 2, 'task_rules': {'mp2020_hubbard': 5}}
    g = [S.graph_arrays(st, IDX, np.zeros(3), 5.0, fields, o, e0) for st in bulk]
    # row 1: H2O (plain task 2), row 2: FeO2 (GGA+U rule -> task 5); E0 of the structure's own task
    assert int(g[0]['g:task'][0]) == 2 and int(g[1]['g:task'][0]) == 5
    assert abs(float(g[0]['energy'][0]) - (-9.0 - (-1.0 - 1.0 - 2.0))) < 1e-6
    assert abs(float(g[1]['energy'][0]) - (-9.0 - (-9.0 - 2.5 - 2.5))) < 1e-6
    assert g[0]['g:pbc'][0] == 1 and g[0]['g:stress_w'][0] == 1 and g[0]['g:dipole_w'][0] == 0
    assert np.all(g[0]['n:ref_charges_w'] == 0)
    ge = S.graph_arrays(bulk[1], IDX, np.zeros(3), 5.0, fields, dict(o, energy_key='alt_energy'), e0)
    assert abs(float(ge['energy'][0]) - (-1.0 + 14.0)) < 1e-6
    mol = S.AselmdbSource([str(corpus / 'mol' / 'data0000.aselmdb')]).read_block(0, [2, 8])
    om = {'task': 3, 'charge_label_key': 'nbo_charges', 'charge_label_scale': 0.5}
    m = [S.graph_arrays(st, IDX, np.zeros(3), 5.0, fields, om, e0) for st in mol]
    assert sorted(m[0]) == sorted(g[0])                             # one key set for the whole mixed batch
    assert m[0]['g:pbc'][0] == 0 and m[0]['g:stress_w'][0] == 0 and np.all(m[0]['Sij'] == 0)
    assert m[0]['g:charge'][0] == 1 and m[0]['g:spin'][0] == 2 and int(m[0]['g:task'][0]) == 3
    assert np.allclose(m[0]['g:dipole'][0], np.array([0.1, 0.0, -0.2]) * S.BOHR) and m[0]['g:dipole_w'][0] == 1
    # labels (-0.3, 0.4, 0.9) scaled about the mean by 0.5: the total charge 1 is kept
    assert np.allclose(m[0]['n:ref_charges'], 0.5 * np.array([-0.3, 0.4, 0.9]) + 0.5 / 3) and np.all(m[0]['n:ref_charges_w'] == 1)
    assert abs(float(m[0]['n:ref_charges'].sum()) - 1.0) < 1e-6
    assert np.all(m[1]['n:ref_charges_w'] == 0)                     # row 8 has no NBO charges
    nd = S.graph_arrays(mol[0], IDX, np.zeros(3), 5.0, fields, dict(om, dipole=False), e0)
    assert nd['g:dipole_w'][0] == 0


def test_e0_table_formats(tmp_path):
    f = tmp_path / 'e0.txt'
    f.write_text("({0: -1.0, 1: -2.0, 2: -5.0},\n {1: 0, 8: 1, 26: 2})")
    assert np.allclose(S.load_e0_table(str(f), Z), [-1.0, -2.0, -5.0])
    assert np.allclose(S.load_e0_table({'8': -2.0, '1': -1.0, '26': -5.0}, Z), [-1.0, -2.0, -5.0])
    assert np.allclose(S.load_e0_table([-1.0, -2.0, -5.0], Z), [-1.0, -2.0, -5.0])
    with pytest.raises(ValueError):
        S.load_e0_table({1: -1.0}, Z)


def test_shard_meta_groups_and_slab_fields(tmp_path):
    rng = np.random.default_rng(1)
    rows = [_row(rng, True, [26, 26, 8, 1, 1, 8, 1]), _row(rng, False, [8, 1, 1])]
    rows[0]['dipole'] = [0.1, 0.2, 0.5]
    _shard(tmp_path / 'data0000.aselmdb', rows)
    meta = tmp_path / 'meta'; meta.mkdir()
    np.savez(meta / 'data0000.npz', row_id=np.array([1, 2]), atom_ptr=np.array([0, 7, 10]),
             electrode_mask=np.array([1, 1, 0, 0, 0, 0, 0, 1, 1, 1], bool),
             charge_group=np.array([-1, -1, 4, 4, 4, 9, 9, -1, -1, -1], np.int16),
             fixed_charge=np.array([0, 0, 0, 0, 0, -0.5, -0.5, 0, 0, 0], np.float32))
    sm = S.ShardMeta(S.meta_path(str(meta), tmp_path / 'data0000.aselmdb'))
    assert isinstance(sm.a['charge_group'], np.memmap)                 # uncompressed npz members are mapped
    em, cg, fc = sm.get(1)
    assert em.tolist() == [1, 1, 0, 0, 0, 0, 0] and cg.tolist() == [-1, -1, 4, 4, 4, 9, 9]
    with pytest.raises(KeyError):
        sm.get(3)
    sts = S.AselmdbSource([str(tmp_path / 'data0000.aselmdb')]).read_block(0, [1, 2])
    for st, k in zip(sts, (1, 2)):
        st['groups'] = sm.get(k)
    fields = dict(mixed=True, groups=4, slab=True, dipole=True)
    g = S.graph_arrays(sts[0], IDX, np.zeros(3), 5.0, fields, {'slab': True, 'dipole_sign': -1.0})
    assert g['n:charge_group'].tolist() == [0, 0, 1, 1, 1, 2, 2] and np.allclose(g['n:fixed_charge'][5:], -0.5)
    assert g['g:slab'][0] == 1 and g['n:unwrap'].shape == (7, 3)
    assert np.allclose(g['g:dipole'][0], [-0.1, -0.2, -0.5]) and g['g:dipole_w'][0] == 1
    m = S.graph_arrays(sts[1], IDX, np.zeros(3), 5.0, fields, {})
    assert sorted(m) == sorted(g) and m['g:slab'][0] == 0 and np.all(m['n:charge_group'] == 0)
    nd = S.graph_arrays(sts[0], IDX, np.zeros(3), 5.0, fields, {})     # periodic, not a slab source: no dipole label
    assert nd['g:dipole_w'][0] == 0 and nd['g:slab'][0] == 0
    with pytest.raises(ValueError):
        S.graph_arrays(sts[0], IDX, np.zeros(3), 5.0, dict(fields, groups=2), {'slab': True})
    two = tmp_path / 'm2'; (two / 'a').mkdir(parents=True); (two / 'b').mkdir()
    for d in ('a', 'b'):
        np.savez(two / d / 'data0000.npz', row_id=np.array([1]))
    with pytest.raises(ValueError):
        S.meta_path(str(two), tmp_path / 'data0000.aselmdb')
