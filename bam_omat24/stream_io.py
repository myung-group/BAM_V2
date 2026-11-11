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
from graph_fields import extra_fields  # noqa: E402

_VOIGT_I = np.array([0, 1, 2, 1, 0, 0])
_VOIGT_J = np.array([0, 1, 2, 2, 2, 1])
RESERVED_KEYS = (b"nextid", b"deleted_ids", b"metadata")


# ----------------------------------------------------------------------------------------------- structures -> graphs
def structure_from_atoms(a, info_keys=(), array_keys=()) -> dict:
    """ase.Atoms (with a single-point calculator, or none) -> structure dict.  info_keys / array_keys: the optional
    fields to carry (graph_fields.field_keys), under 'info' / 'arrays'."""
    res = getattr(a.calc, 'results', {}) if a.calc is not None else {}
    st = res.get('stress')
    if st is not None:
        st = np.asarray(st, np.float64)
        st = st.reshape(3, 3)[_VOIGT_I, _VOIGT_J] if st.size == 9 else st.reshape(6)
    return {'numbers': np.asarray(a.numbers, np.int64), 'positions': np.asarray(a.positions, np.float64),
            'cell': np.asarray(a.cell.array, np.float64), 'pbc': np.asarray(a.pbc, bool),
            'energy': None if res.get('energy') is None else float(res['energy']),
            'forces': None if res.get('forces') is None else np.asarray(res['forces'], np.float64),
            'stress': st,
            'info': {k: a.info[k] for k in info_keys if k in a.info},
            'arrays': {k: np.asarray(a.arrays[k]) for k in array_keys if k in a.arrays}}


def _atoms(st):
    import ase
    return ase.Atoms(numbers=st['numbers'], positions=st['positions'], cell=st['cell'], pbc=st['pbc'])


def graph_arrays(st: dict, atom_indices: dict, atom_energies, cutoff: float, fields: dict = None) -> dict:
    """Arrays of one training graph, identical to data_3bpa.preprocess_graph(targets=True, **fields):
    nodes species i32 / positions f32 / forces f32, receivers = i, senders = j, edges Sij = -S (f32, periodic),
    globals energy = E - sum E0 (f32 of a float64 difference), cell / cutoff / volume / stress (periodic).
    fields: charge_key / spin_key / charge_label_key / fragments (graph_fields.extra_fields); their arrays are
    returned under 'n:<name>' (nodes) and 'g:<name>' (globals)."""
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
    if fields:
        nx, gx = extra_fields(st.get('info'), st.get('arrays'), len(species), **fields)
        out.update({f'n:{k}': v for k, v in nx.items()})
        out.update({f'g:{k}': v for k, v in gx.items()})
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
    def __init__(self, paths, offsets=None, info_keys=(), array_keys=()):
        self.paths = [str(p) for p in paths]
        self.offsets = offsets if offsets is not None else [extxyz_offsets(p) for p in self.paths]
        self.info_keys, self.array_keys = tuple(info_keys), tuple(array_keys)

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
                out.append(structure_from_atoms(a, self.info_keys, self.array_keys))
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


def _maybe_arr(x):
    if isinstance(x, dict):
        try:
            return _arr(x, np.float64)
        except ValueError:
            return x
    return x


def decode_row(raw: bytes = None, d: dict = None) -> dict:
    """raw: the LMDB value (zlib JSON); or d: the already decoded row dict."""
    d = decode_value(raw) if d is None else d
    if not isinstance(d, dict):
        raise ValueError(f"row is a {type(d).__name__}, not a dict")
    st = _arr(d.get("stress"), np.float64)
    if st is not None:
        st = st.reshape(3, 3)[_VOIGT_I, _VOIGT_J] if st.size == 9 else st.reshape(6)
    cell = _arr(d.get("cell"), np.float64)
    # optional fields (total charge, spin, per-atom labels): ASE db key_value_pairs and data, as ase.db row.toatoms
    # puts them in atoms.info (the OMol25 layout is to be checked with data_prep/convert_omol25.py inspect)
    info = {}
    for part in ("key_value_pairs", "data"):
        v = d.get(part)
        if isinstance(v, dict):
            info.update({k: _maybe_arr(x) for k, x in v.items()})
    return {"numbers": _arr(d.get("numbers"), np.int64), "positions": _arr(d.get("positions"), np.float64),
            "cell": None if cell is None else cell.reshape(3, 3), "pbc": _arr(d.get("pbc"), bool),
            "energy": _scalar(d.get("energy")), "forces": _arr(d.get("forces"), np.float64), "stress": st,
            "info": info, "arrays": {}}


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


def load_filters(spec) -> dict:
    """Row filters of the size pass: a JSON file (or a BAM-GAMP convert_config.json, whose 'filters' entry is used).

    Keys (all optional): max_atoms, max_force (eV/A, per-atom norm), max_stress (eV/A^3, any Voigt component),
    max_energy_per_atom (raw E/N above it is dropped), min_energy_residual / max_energy_residual (eV/atom after
    subtracting the reference energies 'e0': {"Z": E0}; a species without E0 drops the row), require_pbc (default
    true when any filter is given), min_volume (|det cell|, default 1e-3).  Unknown keys are ignored, so the
    converter's energy_floor_ref (a table of another repository) is NOT applied: the E0 residual bound replaces it.
    """
    if spec is None or isinstance(spec, dict):
        return spec
    d = json.load(open(spec))
    d = dict(d.get('filters', d))
    if isinstance(d.get('e0'), dict):              # JSON keys are strings: {"26": E0} -> {26: E0}
        d['e0'] = {int(k): float(v) for k, v in d['e0'].items()}
    return d


def drop_reason(st: dict, flt: dict):
    """None when the row is kept, else the name of the first filter that drops it."""
    z, f, e = st['numbers'], st['forces'], st['energy']
    n = len(z)
    if n == 0 or e is None or f is None or not np.isfinite(e) or not np.all(np.isfinite(f)):
        return 'missing'
    if flt.get('max_atoms') and n > flt['max_atoms']:
        return 'max_atoms'
    if flt.get('require_pbc', True):
        if st['cell'] is None or st['pbc'] is None or not np.all(st['pbc']):
            return 'pbc'
        if abs(np.linalg.det(st['cell'])) < flt.get('min_volume', 1e-3):
            return 'volume'
    if flt.get('max_force') is not None and float(np.sqrt((f * f).sum(-1).max())) > flt['max_force']:
        return 'max_force'
    if flt.get('max_stress') is not None and st['stress'] is not None:
        if not np.all(np.isfinite(st['stress'])) or float(np.abs(st['stress']).max()) > flt['max_stress']:
            return 'max_stress'
    if flt.get('max_energy_per_atom') is not None and e / n > flt['max_energy_per_atom']:
        return 'max_energy_per_atom'
    e0 = flt.get('e0')
    if e0 is not None:
        try:
            r = (e - sum(e0[int(k)] for k in z)) / n
        except KeyError:
            return 'species'
        if flt.get('min_energy_residual') is not None and r < flt['min_energy_residual']:
            return 'min_energy_residual'
        if flt.get('max_energy_residual') is not None and r > flt['max_energy_residual']:
            return 'max_energy_residual'
    return None


def _count_task(args):
    """(kind, path, keys | None, cutoff, filters) -> (keys kept, n_atoms, n_edges, {drop reason: count}).

    keys None (aselmdb): every row of the file in one sequential cursor pass."""
    kind, path, keys, cutoff, flt = args
    if kind == 'extxyz':
        sts = ExtxyzSource([path]).read_block(0, keys)
    elif keys is not None:
        sts = AselmdbSource([path]).read_block(0, keys)
    else:
        env = open_lmdb(path)
        keys, sts = [], None
        with env.begin() as txn:
            items = [(int(k), bytes(v)) for k, v in txn.cursor() if k.isdigit()]
        env.close()
        items.sort()
        keys = [k for k, _ in items]
        sts = (decode_row(v) for _, v in items)
    kk, na, ne, drops = [], [], [], {}
    for k, s in zip(keys, sts):
        why = drop_reason(s, flt) if flt else None
        if why is not None:
            drops[why] = drops.get(why, 0) + 1
            continue
        kk.append(int(k)); na.append(len(s['numbers'])); ne.append(int(len(neighbor_pairs(_atoms(s), cutoff)[0])))
    return np.asarray(kk, np.int64), np.asarray(na, np.int64), np.asarray(ne, np.int64), drops


def size_pass(sources, cutoff, out_dir, workers=0, block=512, filters=None):
    """One pool for all files.  extxyz: tasks of `block` frames; aselmdb: one task per file.  With `filters`
    (load_filters) only the kept rows enter the table, so training never reads the dropped ones."""
    import concurrent.futures as cf
    import multiprocessing as mp
    import ase
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    workers = workers or min(64, int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or len(os.sched_getaffinity(0)))
    filters = load_filters(filters)
    files, kinds, tasks, owner = [], [], [], []
    meta = {'cutoff': cutoff, 'sources': [], 'ase': ase.__version__, 'numpy': np.__version__, 'tsv': {},
            'neighbors': os.environ.get('BAM_NEIGHBORS', 'ase')}       # edge counts depend on nothing else
    if filters:
        meta['filters'] = {k: (v if k != 'e0' else f'{len(v)} species') for k, v in filters.items()}
    t0 = time.time()
    for spec in sources:
        kind, fl = _source_files(spec)
        meta['sources'].append({'spec': spec, 'kind': kind, 'files': fl})
        for path in fl:
            fi = len(files); files.append(path); kinds.append(kind)
            if kind == 'extxyz':
                keys = np.arange(len(extxyz_offsets(path)))
                for k in range(0, len(keys), block):
                    tasks.append((kind, path, keys[k:k + block], cutoff, filters)); owner.append(fi)
            else:
                tasks.append((kind, path, None, cutoff, filters)); owner.append(fi)
    per = [[] for _ in files]
    drops, done = {}, 0

    def take(i, res):
        nonlocal done
        per[owner[i]].append(res[:3])
        for k, v in res[3].items():
            drops[k] = drops.get(k, 0) + v
        done += 1
        if done % max(1, len(tasks) // 50) == 0 or done == len(tasks):
            print(f'size pass: {done}/{len(tasks)} tasks, {time.time() - t0:.0f} s', flush=True)

    if workers <= 1:
        for i, t in enumerate(tasks):
            take(i, _count_task(t))
    else:
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('forkserver')) as ex:
            for i, res in enumerate(ex.map(_count_task, tasks, chunksize=1)):
                take(i, res)
    cols = {'n_atoms': [], 'n_edges': [], 'file': [], 'key': []}
    for fi, path in enumerate(files):
        keys = np.concatenate([r[0] for r in per[fi]]) if per[fi] else np.zeros(0, np.int64)
        na = np.concatenate([r[1] for r in per[fi]]) if per[fi] else np.zeros(0, np.int64)
        ne = np.concatenate([r[2] for r in per[fi]]) if per[fi] else np.zeros(0, np.int64)
        tag = f"{fi:04d}_{Path(path).stem}"
        tsv = out / f'{tag}.sizes.tsv'
        with open(tsv, 'w') as fh:
            fh.write(f'# source {path}\n# cutoff {cutoff}\nindex\tn_atoms\tn_edges\tfile\tkey\n')
            fh.write(''.join(f'{j}\t{a}\t{e}\t{fi}\t{k}\n' for j, (a, e, k) in
                             enumerate(zip(na.tolist(), ne.tolist(), keys.tolist()))))
        meta['tsv'][tag] = hashlib.sha256(tsv.read_bytes()).hexdigest()
        cols['n_atoms'].append(na); cols['n_edges'].append(ne)
        cols['file'].append(np.full(len(na), fi, np.int64)); cols['key'].append(keys)
        if len(files) <= 20:
            print(f'{path}: {len(na)} structures, {time.time() - t0:.0f} s', flush=True)
    arr = {k: np.concatenate(v) if v else np.zeros(0, np.int64) for k, v in cols.items()}
    np.savez(out / 'sizes.npz', n_atoms=arr['n_atoms'].astype(np.uint16 if arr['n_atoms'].max(initial=0) < 65535
                                                            else np.uint32),
             n_edges=arr['n_edges'].astype(np.uint32), file_index=arr['file'].astype(np.uint32), key=arr['key'])
    meta.update(files=files, kinds=kinds, n=int(len(arr['n_atoms'])), sum_atoms=int(arr['n_atoms'].sum()),
                sum_edges=int(arr['n_edges'].sum()),
                avg_n_neighbors=float(arr['n_edges'].sum() / max(arr['n_atoms'].sum(), 1)),
                max_atoms=int(arr['n_atoms'].max(initial=0)), max_edges=int(arr['n_edges'].max(initial=0)),
                dropped=drops)
    json.dump(meta, open(out / 'sizes_meta.json', 'w'), indent=1)
    print(f"size pass: {meta['n']} structures kept, dropped {drops or 0}, avg_n_neighbors "
          f"{meta['avg_n_neighbors']:.3f}, max {meta['max_atoms']} atoms / {meta['max_edges']} edges, "
          f"{time.time() - t0:.0f} s ({workers} workers)")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sizes')
    s.add_argument('--source', action='append', required=True, help='extxyz:<file> or aselmdb:<file|dir>')
    s.add_argument('--cutoff', type=float, required=True); s.add_argument('--out', required=True)
    s.add_argument('--workers', type=int, default=0); s.add_argument('--block', type=int, default=512)
    s.add_argument('--filters', help='JSON of row filters, or a convert_config.json (see load_filters)')
    s.add_argument('--e0-npz', help='reference energies for the residual filters: npz with Z_to_idx / atom_ref')
    s.add_argument('--max-residual', type=float, help='also drop rows with E0 residual above this (eV/atom)')
    a = ap.parse_args(argv)
    if a.cmd == 'sizes':
        flt = load_filters(a.filters)
        if a.e0_npz:
            z = np.load(a.e0_npz)
            enc, ref = z['species_encoding'], z['atom_ref']
            flt = dict(flt or {})
            flt['e0'] = {int(Z): float(ref[i]) for Z, i in enumerate(enc) if i >= 0}
            if a.max_residual is not None:
                flt['max_energy_residual'] = a.max_residual
        size_pass(a.source, a.cutoff, a.out, a.workers, a.block, filters=flt)
    return 0


if __name__ == '__main__':
    sys.exit(main())
