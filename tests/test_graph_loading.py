"""Graph loading without the pickle cache (2026-11-06): parallel neighbour searches give the same graphs as the
serial path, the size tables round-trip, use_cache=False writes nothing, and graph_sizes.py counts the same edges."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
XYZ = ROOT / 'dataset_3BPA' / 'train_300K.xyz'


def _frames(n=300):
    import ase.io
    return ase.io.read(str(XYZ), index=f':{n}')


def test_parallel_graphs_identical():
    from bam_omat24.data_3bpa import build_graphs
    from bam_omat24.train_3bpa import DEFAULT_ATOMIC_NUMBERS, DEFAULT_ATOM_ENERGIES
    fr = _frames()
    a = build_graphs(fr, DEFAULT_ATOMIC_NUMBERS, DEFAULT_ATOM_ENERGIES, 5.0, n_workers=1)
    b = build_graphs(fr, DEFAULT_ATOMIC_NUMBERS, DEFAULT_ATOM_ENERGIES, 5.0, n_workers=3, chunk=64)
    assert len(a) == len(b) == len(fr)
    for x, y in zip(a, b):
        assert np.array_equal(x.senders, y.senders) and np.array_equal(x.receivers, y.receivers)
        assert np.array_equal(x.nodes['positions'], y.nodes['positions'])
        assert np.array_equal(x.globals['energy'], y.globals['energy'])


def test_no_cache_writes_nothing(tmp_path):
    from bam_omat24.data_3bpa import load_3bpa_graphs, _cache_path
    from bam_omat24.train_3bpa import DEFAULT_ATOMIC_NUMBERS, DEFAULT_ATOM_ENERGIES
    import shutil
    xyz = tmp_path / 'small.xyz'
    import ase.io
    ase.io.write(str(xyz), _frames(40), format='extxyz')
    g = load_3bpa_graphs(xyz, DEFAULT_ATOMIC_NUMBERS, DEFAULT_ATOM_ENERGIES, 5.0, use_cache=False, n_workers=2)
    assert len(g) == 40
    assert not _cache_path(xyz, 5.0, DEFAULT_ATOM_ENERGIES, None).exists()
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_sizes_tsv_roundtrip_and_tool(tmp_path):
    from bam_omat24.data_3bpa import build_graphs, graph_sizes, write_sizes_tsv, read_sizes_tsv
    from bam_omat24.train_3bpa import DEFAULT_ATOMIC_NUMBERS, DEFAULT_ATOM_ENERGIES
    import ase.io
    fr = _frames(60)
    xyz = tmp_path / 'part.xyz'
    ase.io.write(str(xyz), fr, format='extxyz')
    n, e = graph_sizes(build_graphs(fr, DEFAULT_ATOMIC_NUMBERS, DEFAULT_ATOM_ENERGIES, 5.0))
    write_sizes_tsv(tmp_path / 'a.tsv', n, e, 5.0, source=str(xyz))
    n2, e2 = read_sizes_tsv(tmp_path / 'a.tsv')
    assert np.array_equal(n, n2) and np.array_equal(e, e2)
    # stand-alone tool (no JAX import) counts the same edges
    r = subprocess.run([sys.executable, '-I', str(ROOT / 'bam_omat24' / 'graph_sizes.py'), str(xyz), '--cutoff', '5.0',
                        '--workers', '2', '--out-dir', str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    n3, e3 = read_sizes_tsv(tmp_path / 'part.sizes.tsv')
    assert np.array_equal(n, n3) and np.array_equal(e, e3)
    assert (tmp_path / 'part.sizes_summary.txt').exists()
