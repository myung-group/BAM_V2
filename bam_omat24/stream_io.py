"""JAX-free readers and size pass for the streaming (pickle-free) data pipeline (docs/stream_loader_design.md).

Run as a stand-alone script or import as a top-level module (``sys.path.insert(0, <bam_omat24 dir>)``; the package
``bam_omat24/__init__`` imports JAX and the model, which worker processes must not do):

    python -I bam_omat24/stream_io.py sizes --source extxyz:<file.xyz> [--source aselmdb:<dir> ...] \
        --cutoff 5.0 --out <sizes_dir> [--workers 0]

Size pass: one neighbour search per structure (the same ``neighbors.neighbor_pairs`` as the training graphs) and the
result written as
  <sizes_dir>/<tag>.sizes.tsv   readable, one line per structure: index, n_atoms, n_edges, file, key
                                (read_sizes_tsv of data_3bpa reads columns 1-2, so it stays compatible)
  <sizes_dir>/sizes.npz         the same columns for fast loading (n_atoms, n_edges, file_index, key)
  <sizes_dir>/sizes_meta.json   cutoff, sources (file list per source), N, sums, avg_n_neighbors of the full split,
                                ase / numpy versions, sha256 of each TSV
``file`` indexes the global file list of the meta file; ``key`` is the position inside the file (extxyz: frame
index, aselmdb: integer row id).

Sources (``read_block(file, keys) -> list of structures``; a structure is a dict of numbers, positions, cell, pbc,
energy, forces, stress):
  * ExtxyzSource: frames located by byte offsets (index built once per file), so a block is read in O(block).
  * AselmdbSource: ASE lmdb databases (OMat24, sAlex), random access by row id; needs the ``lmdb`` package.  Decoding
    copied from BAM-GAMP-jax scripts/convert_aselmdb_to_shards.py (open_lmdb, decode_value, _arr, _scalar,
    decode_row, list_row_ids) -- keep in sync.

``graph_arrays`` turns a structure into the per-graph numpy arrays of ``data_3bpa.preprocess_graph`` (same edge
convention, dtypes and energy reference), so the streaming and the in-memory paths build identical graphs
(tests/test_stream_io.py checks this).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neighbors import neighbor_pairs  # noqa: E402

_VOIGT_I = np.array([0, 1, 2, 1, 0, 0])
_VOIGT_J = np.array([0, 1, 2, 2, 2, 1])
RESERVED_KEYS = (b"nextid", b"deleted_ids", b"metadata")


# ----------------------------------------------------------------------------------------------- structures -> graphs
def structure_from_atoms(a) -> dict:
    """ase.Atoms (with a single-point calculator, or none) -> structure dict."""
    res = getattr(a.calc, 'results', {}) if a.calc is not None else {}
    st = res.get('stress')
    if st is not None:
        st = np.asarray(st, np.float64)
        st = st.reshape(3, 3)[_VOIGT_I, _VOIGT_J] if st.size == 9 else st.reshape(6)
    return {'numbers': np.asarray(a.numbers, np.int64), 'positions': np.asarray(a.positions, np.float64),
            'cell': np.asarray(a.cell.array, np.float64), 'pbc': np.asarray(a.pbc, bool),
            'energy': None if res.get('energy') is None else float(res['energy']),
            'forces': None if res.get('forces') is None else np.asarray(res['forces'], np.float64),
            'stress': st}


def _atoms(st):
    import ase
    return ase.Atoms(numbers=st['numbers'], positions=st['positions'], cell=st['cell'], pbc=st['pbc'])


def graph_arrays(st: dict, atom_indices: dict, atom_energies, cutoff: float) -> dict:
    """Arrays of one training graph, identical to data_3bpa.preprocess_graph(targets=True):
    nodes species i32 / positions f32 / forces f32, receivers = i, senders = j, edges Sij = -S (f32, periodic),
    globals energy = E - sum E0 (f32 of a float64 difference), cell / cutoff / volume / stress (periodic)."""
    a = _atoms(st)
    src, dst, S = neighbor_pairs(a, cutoff)
    species = np.array([atom_indices[int(z)] for z in st['numbers']], dtype=np.int32)
    e0 = float(np.sum(np.asarray(atom_energies, dtype=np.float64)[species]))
    out = {'n_node': np.array([len(species)], np.int32), 'n_edge': np.array([len(src)], np.int32),
           'species': species, 'positions': st['positions'].astype(np.float32),
           'forces': (np.zeros((len(species), 3)) if st['forces'] is None else st['forces']).astype(np.float32),
           'energy': np.array([(st['energy'] if st['energy'] is not None else e0) - e0], dtype=np.float32),
           'senders': dst.astype(np.int32), 'receivers': src.astype(np.int32)}
    if S is not None:
        out['Sij'] = (-S).astype(np.float32)
        out['cell'] = np.array(st['cell'], dtype=np.float32)[None]
        out['cutoff'] = np.array([cutoff], dtype=np.float32)
        out['volume'] = np.array([abs(np.linalg.det(st['cell']))], dtype=np.float32)
        s6 = np.zeros(6, np.float32) if st['stress'] is None else np.asarray(st['stress'], np.float32).reshape(6)
        out['stress'] = s6[None]
    return out


# ----------------------------------------------------------------------------------------------- extxyz
def extxyz_offsets(path) -> np.ndarray:
    """Byte offset of every frame of an extxyz file (one sequential pass)."""
    offs = []
    with open(path, 'rb') as fh:
        pos = 0
        while True:
            line = fh.readline()
            if not line:
                break
            if not line.strip():
                pos += len(line)
                continue
            n = int(line.split()[0])
            offs.append(pos)
            pos += len(line)
            for _ in range(n + 1):
                pos += len(fh.readline())
    return np.asarray(offs, np.int64)


class ExtxyzSource:
    def __init__(self, paths, offsets=None):
        self.paths = [str(p) for p in paths]
        self.offsets = offsets if offsets is not None else [extxyz_offsets(p) for p in self.paths]

    def read_block(self, file: int, keys) -> list:
        import ase.io
        out = []
        with open(self.paths[file], 'rb') as fh:
            for k in keys:
                fh.seek(int(self.offsets[file][int(k)]))
                first = fh.readline()
                n = int(first.split()[0])
                buf = first + b''.join(fh.readline() for _ in range(n + 1))
                a = ase.io.read(io.StringIO(buf.decode()), format='extxyz')
                out.append(structure_from_atoms(a))
        return out


# ----------------------------------------------------------------------------------------------- aselmdb
def open_lmdb(path):
    import lmdb
    return lmdb.open(str(path), subdir=False, readonly=True, lock=False, readahead=False, meminit=False,
                     max_readers=4)


def _json_loads(buf):
    try:
        import orjson
        return orjson.loads(buf)
    except ImportError:
        return json.loads(buf)


def decode_value(raw: bytes):
    try:
        buf = zlib.decompress(raw)
    except zlib.error:
        buf = raw
    return _json_loads(buf)


def _arr(v, dtype):
    if v is None:
        return None
    if isinstance(v, dict):
        if "__ndarray__" in v:
            shape, _dt, flat = v["__ndarray__"]
            return np.asarray(flat, dtype=dtype).reshape(shape)
        if "array" in v:
            return _arr(v["array"], dtype)
        raise ValueError(f"unsupported encoded field with keys {sorted(v)[:4]}")
    return np.asarray(v, dtype=dtype)


def _scalar(v):
    if v is None:
        return None
    if isinstance(v, dict):
        a = _arr(v, np.float64)
        return float(a.reshape(-1)[0]) if a.size == 1 else None
    return float(v)


def decode_row(raw: bytes = None, d: dict = None) -> dict:
    """raw: the LMDB value (zlib JSON); or d: the already decoded row dict."""
    d = decode_value(raw) if d is None else d
    if not isinstance(d, dict):
        raise ValueError(f"row is a {type(d).__name__}, not a dict")
    st = _arr(d.get("stress"), np.float64)
    if st is not None:
        st = st.reshape(3, 3)[_VOIGT_I, _VOIGT_J] if st.size == 9 else st.reshape(6)
    cell = _arr(d.get("cell"), np.float64)
    return {"numbers": _arr(d.get("numbers"), np.int64), "positions": _arr(d.get("positions"), np.float64),
            "cell": None if cell is None else cell.reshape(3, 3), "pbc": _arr(d.get("pbc"), bool),
            "energy": _scalar(d.get("energy")), "forces": _arr(d.get("forces"), np.float64), "stress": st}


def list_row_ids(env) -> list:
    entries = env.stat()["entries"]
    with env.begin() as txn:
        n_reserved = sum(txn.get(k) is not None for k in RESERVED_KEYS)
        raw = txn.get(b"nextid")
        if raw is not None:
            nextid = int(decode_value(raw))
            deleted = set()
            raw_del = txn.get(b"deleted_ids")
            if raw_del is not None:
                deleted = {int(i) for i in (decode_value(raw_del) or [])}
            ids = [i for i in range(1, nextid) if i not in deleted]
            if len(ids) == entries - n_reserved and (
                    not ids or (txn.get(str(ids[0]).encode()) is not None
                                and txn.get(str(ids[-1]).encode()) is not None)):
                return ids
        return sorted(int(k) for k in txn.cursor().iternext(keys=True, values=False) if k.isdigit())


class AselmdbSource:
    def __init__(self, paths, lru: int = 16):
        self.paths = [str(p) for p in paths]
        self._envs, self.lru = {}, lru

    def _env(self, file):
        if file not in self._envs:
            if len(self._envs) >= self.lru:
                k = next(iter(self._envs))
                self._envs.pop(k).close()
            self._envs[file] = open_lmdb(self.paths[file])
        return self._envs[file]

    def row_ids(self, file):
        return list_row_ids(self._env(file))

    def read_block(self, file: int, keys) -> list:
        out = []
        with self._env(file).begin() as txn:
            for k in keys:
                raw = txn.get(str(int(k)).encode('ascii'))
                if raw is None:
                    raise KeyError(f"{self.paths[file]}: row {int(k)} not found")
                out.append(decode_row(raw))
        return out


# ----------------------------------------------------------------------------------------------- size pass
def _source_files(spec: str):
    kind, path = spec.split(':', 1)
    p = Path(path)
    if kind == 'extxyz':
        return kind, [str(p)]
    if kind == 'aselmdb':
        # the user's own directory tree only (bounded listing)
        return kind, ([str(p)] if p.is_file() else sorted(str(f) for f in p.rglob('*.aselmdb')))
    raise ValueError(f"unknown source kind {kind!r} (extxyz | aselmdb)")


def _count_task(args):
    kind, path, keys, cutoff = args
    src = ExtxyzSource([path]) if kind == 'extxyz' else AselmdbSource([path])
    sts = src.read_block(0, keys)
    return [(len(s['numbers']), int(len(neighbor_pairs(_atoms(s), cutoff)[0]))) for s in sts]


def size_pass(sources, cutoff, out_dir, workers=0, block=512):
    import concurrent.futures as cf
    import multiprocessing as mp
    import ase
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    workers = workers or min(64, int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or len(os.sched_getaffinity(0)))
    files, kinds, cols = [], [], {'n_atoms': [], 'n_edges': [], 'file': [], 'key': []}
    meta = {'cutoff': cutoff, 'sources': [], 'ase': ase.__version__, 'numpy': np.__version__, 'tsv': {}}
    t0 = time.time()
    for spec in sources:
        kind, fl = _source_files(spec)
        meta['sources'].append({'spec': spec, 'kind': kind, 'files': fl})
        for path in fl:
            fi = len(files); files.append(path); kinds.append(kind)
            keys = (np.arange(len(extxyz_offsets(path))) if kind == 'extxyz'
                    else np.asarray(AselmdbSource([path]).row_ids(0), np.int64))
            tasks = [(kind, path, keys[k:k + block], cutoff) for k in range(0, len(keys), block)]
            res = []
            if workers <= 1:
                for t in tasks:
                    res += _count_task(t)
            else:
                with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('forkserver')) as ex:
                    for part in ex.map(_count_task, tasks):
                        res += part
            r = np.asarray(res, np.int64).reshape(-1, 2)
            tag = f"{fi:04d}_{Path(path).stem}"
            tsv = out / f'{tag}.sizes.tsv'
            with open(tsv, 'w') as fh:
                fh.write(f'# source {path}\n# cutoff {cutoff}\nindex\tn_atoms\tn_edges\tfile\tkey\n')
                for j, ((na, ne), k) in enumerate(zip(r, keys)):
                    fh.write(f'{j}\t{na}\t{ne}\t{fi}\t{int(k)}\n')
            meta['tsv'][tag] = hashlib.sha256(tsv.read_bytes()).hexdigest()
            cols['n_atoms'].append(r[:, 0]); cols['n_edges'].append(r[:, 1])
            cols['file'].append(np.full(len(r), fi, np.int64)); cols['key'].append(keys)
            print(f'{path}: {len(r)} structures, {time.time() - t0:.0f} s', flush=True)
    arr = {k: np.concatenate(v) if v else np.zeros(0, np.int64) for k, v in cols.items()}
    np.savez(out / 'sizes.npz', n_atoms=arr['n_atoms'].astype(np.uint16 if arr['n_atoms'].max(initial=0) < 65535
                                                            else np.uint32),
             n_edges=arr['n_edges'].astype(np.uint32), file_index=arr['file'].astype(np.uint32), key=arr['key'])
    meta.update(files=files, kinds=kinds, n=int(len(arr['n_atoms'])), sum_atoms=int(arr['n_atoms'].sum()),
                sum_edges=int(arr['n_edges'].sum()),
                avg_n_neighbors=float(arr['n_edges'].sum() / max(arr['n_atoms'].sum(), 1)),
                max_atoms=int(arr['n_atoms'].max(initial=0)), max_edges=int(arr['n_edges'].max(initial=0)))
    json.dump(meta, open(out / 'sizes_meta.json', 'w'), indent=1)
    print(f"size pass: {meta['n']} structures, avg_n_neighbors {meta['avg_n_neighbors']:.3f}, "
          f"max {meta['max_atoms']} atoms / {meta['max_edges']} edges, {time.time() - t0:.0f} s ({workers} workers)")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sizes')
    s.add_argument('--source', action='append', required=True, help='extxyz:<file> or aselmdb:<file|dir>')
    s.add_argument('--cutoff', type=float, required=True); s.add_argument('--out', required=True)
    s.add_argument('--workers', type=int, default=0); s.add_argument('--block', type=int, default=512)
    a = ap.parse_args(argv)
    if a.cmd == 'sizes':
        size_pass(a.source, a.cutoff, a.out, a.workers, a.block)
    return 0


if __name__ == '__main__':
    sys.exit(main())
