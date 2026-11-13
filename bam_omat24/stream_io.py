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

Multi-dataset sources (2026-11-11; fairchem-style aselmdb corpora: OC25, MPtrj, sAlex, OMC25, OMol25, OPoly26).  A
source is a string as above or a dict ``{"path": "aselmdb:<dir>", "name": ..., <options>}``:
  max_frames_per_shard  seeded subset of every shard (size pass and training use only these rows); ``subset_seed``
                        (default 0) picks another subset (a new size table) for a continued run
  meta_dir              BAM-torch meta-data directory (<shard stem>.npz); rows flagged ``exclude`` there are dropped
  task                  task index (labelling convention) of the source; ``task_rules`` {"mp2020_hubbard": 5} routes
                        the matching structures to another task (first match wins)
  energy_key            row key (key_value_pairs / data) used as the energy instead of ``energy``
  charge_label_key      per-atom reference charges of this source (e.g. "nbo_charges"), with
                        fields['opt_charge_labels']: nodes 'ref_charges' / 'ref_charges_w' (weight 0 for rows and
                        sources without labels); ``charge_label_scale`` s: q' = s q + (1 - s) Q / N
  dipole                false: no dipole label from this source.  With fields['dipole'] every graph gets globals
                        'dipole' (e A: ORCA ``multipoles`` converted from a.u., about the centre of mass; or the
                        row's ``dipole`` of a slab source) times ``dipole_sign`` (VASP / OC25: -1), and 'dipole_w'
                        (0 without a label; periodic rows only from ``slab`` sources with fields['slab'])
  slab                  true: the structures are slabs (RACE_POLAR slab dipole correction; fields['slab'] gives
                        globals 'slab' and nodes 'unwrap', the image offsets of unwrap_offsets)
  charge_groups         false: ignore the charge-group arrays of meta_dir.  With fields['groups'] = F every graph
                        gets nodes 'charge_group' (0 = reservoir / electrode atoms, 1.. other groups, < F) and
                        'fixed_charge' from <meta_dir>/<shard stem>.npz (electrode_mask, charge_group,
                        fixed_charge); sources without meta-data: one group per structure
``graph_arrays(..., opts=<source options>, e0_tables={task: E0 per species})`` applies them; with
fields['mixed'] (periodic and open-boundary structures in one batch) every graph gets edges 'Sij', a cell, the
flag globals 'pbc' (1 = periodic) and 'stress_w' (1 = the row has a stress label); an open-boundary structure gets
zero shifts and a placeholder box.

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


MP2020_U_ELEMENTS = frozenset({27, 24, 26, 25, 42, 28, 23, 74})      # Co Cr Fe Mn Mo Ni V W
MP2020_U_ANIONS = frozenset({8, 9})


def mp2020_hubbard(numbers) -> bool:
    """True when the Materials Project runs this composition with GGA+U (BAM-torch utils/dataset_rules.py)."""
    z = set(int(n) for n in np.asarray(numbers).reshape(-1))
    return bool(z & MP2020_U_ANIONS) and bool(z & MP2020_U_ELEMENTS)


TASK_RULES = {'mp2020_hubbard': mp2020_hubbard}
BOHR, DEBYE_AU = 0.529177210903, 0.393430269519          # Angstrom per bohr; atomic units (e bohr) per debye


def structure_task(st: dict, opts: dict, default=None):
    """Task index of a structure: the first matching rule of opts['task_rules'], else opts['task'], else default."""
    for name, target in (opts.get('task_rules') or {}).items():
        if TASK_RULES[name](st['numbers']):
            return int(target)
    t = opts.get('task', default)
    return t


def dipole_label(st: dict):
    """Dipole label (e Angstrom) of a row, or None: the top-level ``dipole`` (VASP, e A; OC25) or the ORCA
    ``multipoles`` [monopole, dipole (a.u., e bohr), quadrupole] of OMol25 / OPoly26 rows."""
    info = st.get('info') or {}
    d = info.get('dipole')
    if d is not None:
        return np.asarray(d, np.float64).reshape(3)
    m = info.get('multipoles')
    if m is not None and len(m) >= 2 and m[1] is not None:
        return np.asarray(m[1], np.float64).reshape(3) * BOHR
    return None


def unwrap_offsets(recv, send, shifts, n_nodes: int):
    """Integer image offsets that make every bonded fragment of a periodic structure contiguous (BAM-torch
    thole.unwrap_fragments, on the neighbour graph within the cutoff).  Edge (i = recv, j = send) with ASE shift S:
    the image of j next to i is r_j + S . cell, so the offsets propagate as n_j = n_i + S from the lowest-index
    atom of each connected component.  A component whose edges cannot be satisfied (it spans the cell: a slab, a
    bulk solid) percolates and keeps zero offsets.  Returns (offsets (n, 3) int, percolating (n,) bool)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order, connected_components
    offsets = np.zeros((n_nodes, 3), np.int64)
    perc = np.zeros(n_nodes, bool)
    if n_nodes == 0 or len(recv) == 0:
        return offsets, perc
    recv, send = np.asarray(recv, np.int64), np.asarray(send, np.int64)
    S = np.rint(np.asarray(shifts)).astype(np.int64)
    adj = coo_matrix((np.ones(len(recv), np.int8), (recv, send)), shape=(n_nodes, n_nodes)).tocsr()
    n_comp, comp = connected_components(adj, directed=False)
    # one shift per ordered pair that the tree may use (any edge of the pair: a consistent component gives the
    # same offsets whichever it is, an inconsistent one is flagged below)
    order = np.lexsort((send, recv))
    key = recv[order] * n_nodes + send[order]
    first = np.concatenate([[True], key[1:] != key[:-1]])
    pair_shift = dict(zip(key[first].tolist(), S[order][first]))
    for c in range(n_comp):
        root = int(np.argmax(comp == c))
        nodes, pred = breadth_first_order(adj, root, directed=False)
        for j in nodes[1:]:
            i = int(pred[j])
            sh = pair_shift.get(i * n_nodes + int(j))
            offsets[j] = offsets[i] + (sh if sh is not None else -pair_shift[int(j) * n_nodes + i])
    bad = np.any(offsets[send] - offsets[recv] != S, axis=1)
    bad_comp = np.zeros(n_comp, bool)
    bad_comp[comp[recv[bad]]] = True
    perc = bad_comp[comp]
    offsets[perc] = 0
    return offsets, perc


class ShardMeta:
    """Per-atom charge-constraint arrays of one shard from the BAM-torch meta-data (<meta_dir>/<stem>.npz:
    row_id, atom_ptr, electrode_mask, charge_group, fixed_charge).  The arrays of an uncompressed npz are
    memory-mapped, so a worker holds no copy of the (hundreds of MB) per-atom arrays."""
    KEYS = ('row_id', 'atom_ptr', 'electrode_mask', 'charge_group', 'fixed_charge')

    def __init__(self, path):
        import zipfile
        self.path = str(path)
        self.a = {}
        with zipfile.ZipFile(self.path) as zf, open(self.path, 'rb') as fh:
            for k in self.KEYS:
                info = zf.getinfo(k + '.npy')
                if info.compress_type != zipfile.ZIP_STORED:
                    with zf.open(info) as f:
                        self.a[k] = np.lib.format.read_array(f)
                    continue
                fh.seek(info.header_offset)
                hdr = fh.read(30)
                n_name, n_extra = int.from_bytes(hdr[26:28], 'little'), int.from_bytes(hdr[28:30], 'little')
                fh.seek(info.header_offset + 30 + n_name + n_extra)
                version = np.lib.format.read_magic(fh)
                shape, fortran, dtype = (np.lib.format.read_array_header_1_0(fh) if version == (1, 0)
                                         else np.lib.format.read_array_header_2_0(fh))
                self.a[k] = np.memmap(self.path, dtype=dtype, mode='r', offset=fh.tell(), shape=shape,
                                      order='F' if fortran else 'C')
        rid = self.a['row_id']
        self.sorted = bool(len(rid) < 2 or np.all(rid[1:] > rid[:-1]))
        self.index = None if self.sorted else {int(r): i for i, r in enumerate(rid)}

    def get(self, row_id: int):
        """(electrode_mask, charge_group, fixed_charge) of one row."""
        rid = self.a['row_id']
        if self.sorted:
            i = int(np.searchsorted(rid, row_id))
            if i >= len(rid) or int(rid[i]) != int(row_id):
                raise KeyError(f"{self.path}: row {row_id} not in the meta-data")
        else:
            i = self.index[int(row_id)]
        a, b = int(self.a['atom_ptr'][i]), int(self.a['atom_ptr'][i + 1])
        return (np.asarray(self.a['electrode_mask'][a:b], bool), np.asarray(self.a['charge_group'][a:b], np.int64),
                np.asarray(self.a['fixed_charge'][a:b], np.float32))


def meta_path(meta_dir, shard_path):
    """<meta_dir>/**/<shard stem>.npz, or None."""
    if not meta_dir:
        return None
    f = Path(meta_dir) / (Path(shard_path).stem + '.npz')
    if f.exists():
        return f
    hits = sorted(Path(meta_dir).rglob(Path(shard_path).stem + '.npz'))
    if len(hits) > 1:      # e.g. <meta>/train and <meta>/val both hold data0000.npz: give the split's own directory
        raise ValueError(f"meta_dir {meta_dir}: {len(hits)} files named {Path(shard_path).stem}.npz; "
                         "point meta_dir to the directory of this split")
    return hits[0] if hits else None


def local_groups(electrode_mask, charge_group, fixed_charge, max_groups: int):
    """Meta-data arrays of one frame -> (local group (n,) int32: 0 = reservoir (electrode atoms), 1.. = the other
    charge groups in order of their label; fixed charge (n,) float32, zero on the reservoir)."""
    em = np.asarray(electrode_mask, bool)
    lg = np.zeros(len(em), np.int32)
    if (~em).any():
        _, inv = np.unique(np.asarray(charge_group)[~em], return_inverse=True)
        lg[~em] = 1 + inv
    if lg.max(initial=0) >= max_groups:
        raise ValueError(f"{int(lg.max()) + 1} charge groups in one structure > polar.max_groups {max_groups}")
    return lg, np.where(em, 0.0, np.asarray(fixed_charge, np.float32)).astype(np.float32)


def graph_arrays(st: dict, atom_indices: dict, atom_energies, cutoff: float, fields: dict = None,
                 opts: dict = None, e0_tables: dict = None) -> dict:
    """Arrays of one training graph, identical to data_3bpa.preprocess_graph(targets=True, **fields):
    nodes species i32 / positions f32 / forces f32, receivers = i, senders = j, edges Sij = -S (f32, periodic),
    globals energy = E - sum E0 (f32 of a float64 difference), cell / cutoff / volume / stress (periodic).
    fields: charge_key / spin_key / charge_label_key / fragments (graph_fields.extra_fields); their arrays are
    returned under 'n:<name>' (nodes) and 'g:<name>' (globals).  fields['mixed']: see the module docstring.
    opts: options of the structure's source (module docstring); e0_tables: {task: E0 per species index}."""
    opts = opts or {}
    fields = dict(fields or {})
    mixed = bool(fields.pop('mixed', False))
    want_dipole = bool(fields.pop('dipole', False))
    n_groups = int(fields.pop('groups', 0) or 0)
    want_slab = bool(fields.pop('slab', False))
    opt_labels = bool(fields.pop('opt_charge_labels', False))
    fc_opts = fields.pop('formal_charges', None)
    want_quad = bool(fields.pop('quadrupole', False))
    spin_opts = fields.pop('spin_labels', None)
    a = _atoms(st)
    src, dst, S = neighbor_pairs(a, cutoff)
    species = np.array([atom_indices[int(z)] for z in st['numbers']], dtype=np.int32)
    task = structure_task(st, opts)
    if task is not None:
        fields['task'] = int(task)
    elif isinstance(fields.get('task'), str):
        task = (st.get('info') or {}).get(fields['task'], 0)
    else:
        task = fields.get('task')
    if e0_tables and task is not None and int(task) in e0_tables:
        atom_energies = e0_tables[int(task)]
    e0 = float(np.sum(np.asarray(atom_energies, dtype=np.float64)[species]))
    energy = st['energy']
    if opts.get('energy_key'):
        energy = (st.get('info') or {}).get(opts['energy_key'])
        if energy is None:
            raise KeyError(f"row without the energy key {opts['energy_key']!r}")
        energy = float(energy)
    out = {'n_node': np.array([len(species)], np.int32), 'n_edge': np.array([len(src)], np.int32),
           'species': species, 'positions': st['positions'].astype(np.float32),
           'forces': (np.zeros((len(species), 3)) if st['forces'] is None else st['forces']).astype(np.float32),
           'energy': np.array([(energy if energy is not None else e0) - e0], dtype=np.float32),
           'senders': dst.astype(np.int32), 'receivers': src.astype(np.int32)}
    if S is not None or mixed:
        periodic = S is not None
        if periodic:
            cell = np.asarray(st['cell'], np.float64)
        else:        # placeholder box of an open-boundary structure: never enters the energy (zero shifts)
            span = np.ptp(st['positions'], axis=0) if len(species) else np.zeros(3)
            cell = np.diag(span + 2.0 * cutoff + 10.0)
        out['Sij'] = (-S).astype(np.float32) if periodic else np.zeros((len(src), 3), np.float32)
        out['cell'] = cell.astype(np.float32)[None]
        out['cutoff'] = np.array([cutoff], dtype=np.float32)
        out['volume'] = np.array([abs(np.linalg.det(cell))], dtype=np.float32)
        has_stress = periodic and st['stress'] is not None
        s6 = np.asarray(st['stress'], np.float32).reshape(6) if has_stress else np.zeros(6, np.float32)
        out['stress'] = s6[None]
        if mixed:
            out['g:pbc'] = np.array([1.0 if periodic else 0.0], np.float32)
            out['g:stress_w'] = np.array([1.0 if has_stress else 0.0], np.float32)
    label_key = opts.get('charge_label_key')
    if want_quad:
        # traceless quadrupole label (e A^2, about the centre of mass, xx yy zz xy xz yz) from the ORCA multipoles
        # [origin, dipole, quadrupole (a.u., Cartesian second moment, xx yy zz xy xz yz)] of open-boundary rows
        m = (st.get('info') or {}).get('multipoles')
        qd, w = np.zeros(6, np.float64), 0.0
        if S is None and m is not None and len(m) >= 3 and m[2] is not None and len(m[2]) == 6:
            v = np.asarray(m[2], np.float64) * BOHR ** 2
            tr = (v[0] + v[1] + v[2]) / 3.0
            qd, w = np.array([v[0] - tr, v[1] - tr, v[2] - tr, v[3], v[4], v[5]]), 1.0
        out['g:quad'] = qd.astype(np.float32)[None]
        out['g:quad_w'] = np.array([w], np.float32)
    if spin_opts:
        # atomic spin populations: the row's label (spin_opts['key'], e.g. lowdin_spins); a restricted closed-shell row
        # has zero spin density (label 0, weight 1); an open-shell row without labels gets weight 0
        info = st.get('info') or {}
        v = info.get(spin_opts.get('key', 'lowdin_spins'))
        n_at = len(species)
        mult = int(round(float(info.get('spin') or 1)))
        if v is not None and len(np.asarray(v).reshape(-1)) == n_at and np.all(np.isfinite(np.asarray(v, float))):
            sv, sw = np.asarray(v, np.float64).reshape(-1), np.ones(n_at)
        elif mult == 1 and not bool(info.get('unrestricted', False)):
            sv, sw = np.zeros(n_at), np.ones(n_at)
        else:
            sv, sw = np.zeros(n_at), np.zeros(n_at)
        out['n:ref_spins'], out['n:ref_spins_w'] = sv.astype(np.float32), sw.astype(np.float32)
    if fc_opts:
        # per-atom reference charges q0 of the distance-limited charge transfer (RACE_LR long_range ct_mode
        # "conductance"): integer charge per molecule, from the summed charge labels (graph_fields.formal_charges)
        from graph_fields import formal_charges as _formal_charges
        info = st.get('info') or {}
        q_tot = float(info.get(fields.get('charge_key') or 'charge', 0.0) or 0.0)
        shifts = None if S is None else np.asarray(S, np.float64) @ np.asarray(st['cell'], np.float64)
        q0, _ = _formal_charges(st['numbers'], np.asarray(st['positions'], np.float64), src, dst, q_tot, info,
                                st.get('arrays'), shifts, label_keys=tuple(fc_opts.get('label_keys',
                                                                                       ('nbo_charges', 'lowdin_charges'))),
                                bond_factor=float(fc_opts.get('bond_factor', 1.2)))
        out['n:q0'] = q0.astype(np.float32)
    if fields:
        nx, gx = extra_fields(st.get('info'), st.get('arrays'), len(species), **fields)
        out.update({f'n:{k}': v for k, v in nx.items()})
        out.update({f'g:{k}': v for k, v in gx.items()})
    if opt_labels:
        # per-source charge labels: a structure without them keeps zeros with weight 0
        info = st.get('info') or {}
        v = None if not label_key else (st.get('arrays') or {}).get(label_key, info.get(label_key))
        q = np.zeros(len(species), np.float32)
        w = np.zeros(len(species), np.float32)
        if v is not None:
            v = np.asarray(v, np.float64).reshape(-1)
            if len(v) != len(species):
                raise ValueError(f"{label_key!r}: {len(v)} values for {len(species)} atoms")
            sc = float(opts.get('charge_label_scale', 1.0))
            if sc != 1.0:       # scaled about the mean: the total charge is kept
                v = sc * v + (1.0 - sc) * v.sum() / max(len(v), 1)
            q, w = v.astype(np.float32), np.ones(len(species), np.float32)
        out['n:ref_charges'], out['n:ref_charges_w'] = q, w
    is_slab = bool(S is not None and opts.get('slab'))
    if n_groups:
        # charge-constraint groups (RACE_POLAR charge_constraint "electrode"): from the meta-data of the row
        # (st['groups'], attached by the stream worker); without them the whole structure is the reservoir group
        ann = st.get('groups') if opts.get('charge_groups', True) else None
        if ann is None:
            lg, fx = np.zeros(len(species), np.int32), np.zeros(len(species), np.float32)
        else:
            if len(ann[0]) != len(species):
                raise ValueError(f"meta-data of {len(ann[0])} atoms for a row of {len(species)} atoms")
            lg, fx = local_groups(ann[0], ann[1], ann[2], n_groups)
        out['n:charge_group'], out['n:fixed_charge'] = lg, fx
    if want_slab:
        # slab dipole correction: flag per graph and the image offsets that make bonded fragments contiguous
        out['g:slab'] = np.array([1.0 if is_slab else 0.0], np.float32)
        off = unwrap_offsets(src, dst, S, len(species))[0] if is_slab else np.zeros((len(species), 3))
        out['n:unwrap'] = off.astype(np.float32)
    if want_dipole:
        # open-boundary structures (ORCA, about the centre of mass) and, with the slab correction, slabs (the
        # component along the slab normal is the label; VASP counts electrons positive: dipole_sign -1)
        ok = opts.get('dipole', True) and (S is None or (is_slab and want_slab))
        d = dipole_label(st) if ok else None
        if d is not None:
            d = d * float(opts.get('dipole_sign', 1.0))
        out['g:dipole'] = (np.zeros(3, np.float32) if d is None else d.astype(np.float32))[None]
        out['g:dipole_w'] = np.array([0.0 if d is None else 1.0], np.float32)
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
    for k in ("dipole", "free_energy"):          # top-level results of some corpora (OC25: VASP dipole, e A)
        if d.get(k) is not None:
            info[k] = _maybe_arr(d[k])
    arrays = {}
    if d.get("tags") is not None:                # OC25: 0 bulk-like slab, 1 surface, 2 adsorbate / solvent
        arrays["tags"] = _arr(d["tags"], np.int64)
    return {"numbers": _arr(d.get("numbers"), np.int64), "positions": _arr(d.get("positions"), np.float64),
            "cell": None if cell is None else cell.reshape(3, 3), "pbc": _arr(d.get("pbc"), bool),
            "energy": _scalar(d.get("energy")), "forces": _arr(d.get("forces"), np.float64), "stress": st,
            "info": info, "arrays": arrays}


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
def worker_context():
    """forkserver context for the reader / graph workers, with the fork server started in an environment without
    GPUs.  A worker re-imports the main module (``python -m bam_omat24.train_3bpa`` -> the bam_omat24 package -> JAX):
    with the parent's environment every worker would create a CUDA context (GPU memory per worker, and a crash when
    the GPU is full; seen 2026-11-11) or join jax.distributed.  Workers only need numpy."""
    import multiprocessing as mp
    from multiprocessing import forkserver
    ctx = mp.get_context('forkserver')
    force = {'JAX_PLATFORMS': 'cpu', 'CUDA_VISIBLE_DEVICES': '', 'BAM_DISTRIBUTED': None}
    old = {k: os.environ.get(k) for k in force}
    try:
        for k, v in force.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        forkserver.ensure_running()          # the server keeps this environment; its children inherit it
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return ctx


SOURCE_OPTS = ('name', 'max_frames_per_shard', 'subset_seed', 'meta_dir', 'task', 'task_rules', 'energy_key',
               'charge_label_key', 'charge_label_scale', 'dipole', 'dipole_sign', 'slab', 'charge_groups')


def parse_source(spec) -> dict:
    """'kind:path' or {"path": "kind:path", <options>} -> {'kind', 'path', 'opts'}."""
    if isinstance(spec, str):
        spec = {'path': spec}
    spec = dict(spec)
    kind, path = spec.pop('path').split(':', 1)
    if kind not in ('extxyz', 'aselmdb'):
        raise ValueError(f"unknown source kind {kind!r} (extxyz | aselmdb)")
    unknown = [k for k in spec if k not in SOURCE_OPTS and not k.startswith('_')]
    if unknown:
        raise ValueError(f"source {path}: unknown options {unknown}; known {list(SOURCE_OPTS)}")
    for name in (spec.get('task_rules') or {}):
        if name not in TASK_RULES:
            raise ValueError(f"source {path}: unknown task rule {name!r}; known {sorted(TASK_RULES)}")
    return {'kind': kind, 'path': path, 'opts': {k: v for k, v in spec.items() if not k.startswith('_')}}


def _source_files(spec):
    src = parse_source(spec)
    kind, p = src['kind'], Path(src['path'])
    if kind == 'extxyz':
        return kind, [str(p)]
    # the user's own directory tree only (bounded listing)
    return kind, ([str(p)] if p.is_file() else sorted(str(f) for f in p.rglob('*.aselmdb')))


def _excluded_rows(meta_dir, path):
    """Row ids flagged ``exclude`` in the BAM-torch meta-data of a shard (<meta_dir>/<stem>.npz), or None."""
    f = meta_path(meta_dir, path)
    if f is None:
        return None
    z = np.load(f)
    if 'exclude' not in z.files:
        return None
    return np.asarray(z['row_id'])[np.asarray(z['exclude'], bool)]


def source_keys(kind, path, opts, shard_index):
    """Keys (frame indices / row ids) of one file after the source options: meta-data exclusions, then the seeded
    subset of max_frames_per_shard rows (sorted; seed = [subset_seed, shard index])."""
    keys = (np.arange(len(extxyz_offsets(path))) if kind == 'extxyz'
            else np.asarray(AselmdbSource([path]).row_ids(0), np.int64))
    bad = _excluded_rows(opts.get('meta_dir'), path)
    if bad is not None and len(bad):
        keys = keys[~np.isin(keys, bad)]
    m = opts.get('max_frames_per_shard')
    if m and len(keys) > int(m):
        rng = np.random.default_rng([int(opts.get('subset_seed', 0)), int(shard_index)])
        keys = np.sort(rng.choice(keys, int(m), replace=False))
    return keys


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
    """(kind, path, keys | None, cutoff, filters) -> (keys kept, n_atoms, n_edges, cell (n, 4), {drop reason: count}).

    keys None (aselmdb): every row of the file in one sequential cursor pass.
    cell columns: volume, |a|, |b|, |c| (0 = open boundary); RACE_POLAR sizes its reciprocal sums from them."""
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
    kk, na, ne, cells, drops = [], [], [], [], {}
    for k, s in zip(keys, sts):
        why = drop_reason(s, flt) if flt else None
        if why is not None:
            drops[why] = drops.get(why, 0) + 1
            continue
        periodic = s['pbc'] is not None and bool(np.any(s['pbc'])) and s['cell'] is not None
        cl = np.linalg.norm(s['cell'], axis=-1) if periodic else np.zeros(3)
        vol = abs(float(np.linalg.det(s['cell']))) if periodic else 0.0
        kk.append(int(k)); na.append(len(s['numbers'])); ne.append(int(len(neighbor_pairs(_atoms(s), cutoff)[0])))
        cells.append((vol, *cl))
    return (np.asarray(kk, np.int64), np.asarray(na, np.int64), np.asarray(ne, np.int64),
            np.asarray(cells, np.float32).reshape(-1, 4), drops)


def size_pass(sources, cutoff, out_dir, workers=0, block=512, filters=None):
    """One pool for all files.  extxyz: tasks of `block` frames.  aselmdb without a row selection: one task per
    file (sequential cursor pass); with a row selection of the source (max_frames_per_shard, meta_dir): tasks of
    `block` selected rows.  With `filters` (load_filters) only the kept rows enter the table, so training never
    reads the dropped ones.  Sources are 'kind:path' strings or dicts with options (parse_source)."""
    import concurrent.futures as cf
    import ase
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    workers = workers or min(64, int(os.environ.get('SLURM_CPUS_PER_TASK', 0)) or len(os.sched_getaffinity(0)))
    filters = load_filters(filters)
    files, kinds, file_source, tasks, owner = [], [], [], [], []
    meta = {'cutoff': cutoff, 'sources': [], 'ase': ase.__version__, 'numpy': np.__version__, 'tsv': {},
            'neighbors': os.environ.get('BAM_NEIGHBORS', 'ase')}       # edge counts depend on nothing else
    if filters:
        meta['filters'] = {k: (v if k != 'e0' else f'{len(v)} species') for k, v in filters.items()}
    t0 = time.time()
    for si, spec in enumerate(sources):
        kind, fl = _source_files(spec)
        opts = parse_source(spec)['opts']
        meta['sources'].append({'spec': spec, 'kind': kind, 'files': fl, 'opts': opts})
        select = bool(opts.get('max_frames_per_shard') or opts.get('meta_dir'))
        for shard, path in enumerate(fl):
            fi = len(files); files.append(path); kinds.append(kind); file_source.append(si)
            if kind == 'aselmdb' and not select:
                tasks.append((kind, path, None, cutoff, filters)); owner.append(fi)
            else:
                keys = source_keys(kind, path, opts, shard)
                for k in range(0, len(keys), block):
                    tasks.append((kind, path, keys[k:k + block], cutoff, filters)); owner.append(fi)
    per = [[] for _ in files]
    drops, done = {}, 0

    def take(i, res):
        nonlocal done
        per[owner[i]].append(res[:4])
        for k, v in res[4].items():
            drops[k] = drops.get(k, 0) + v
        done += 1
        if done % max(1, len(tasks) // 50) == 0 or done == len(tasks):
            print(f'size pass: {done}/{len(tasks)} tasks, {time.time() - t0:.0f} s', flush=True)

    if workers <= 1:
        for i, t in enumerate(tasks):
            take(i, _count_task(t))
    else:
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=worker_context()) as ex:
            for i, res in enumerate(ex.map(_count_task, tasks, chunksize=1)):
                take(i, res)
    cols = {'n_atoms': [], 'n_edges': [], 'file': [], 'key': [], 'cell': []}
    n_src = [0] * len(meta['sources'])
    for fi, path in enumerate(files):
        keys = np.concatenate([r[0] for r in per[fi]]) if per[fi] else np.zeros(0, np.int64)
        na = np.concatenate([r[1] for r in per[fi]]) if per[fi] else np.zeros(0, np.int64)
        ne = np.concatenate([r[2] for r in per[fi]]) if per[fi] else np.zeros(0, np.int64)
        cell = np.concatenate([r[3] for r in per[fi]]) if per[fi] else np.zeros((0, 4), np.float32)
        tag = f"{fi:04d}_{Path(path).stem}"
        tsv = out / f'{tag}.sizes.tsv'
        with open(tsv, 'w') as fh:
            fh.write(f'# source {path}\n# cutoff {cutoff}\nindex\tn_atoms\tn_edges\tfile\tkey\n')
            fh.write(''.join(f'{j}\t{a}\t{e}\t{fi}\t{k}\n' for j, (a, e, k) in
                             enumerate(zip(na.tolist(), ne.tolist(), keys.tolist()))))
        meta['tsv'][tag] = hashlib.sha256(tsv.read_bytes()).hexdigest()
        cols['n_atoms'].append(na); cols['n_edges'].append(ne)
        cols['file'].append(np.full(len(na), fi, np.int64)); cols['key'].append(keys)
        cols['cell'].append(cell)                 # volume, |a|, |b|, |c| (0 = open boundary)
        n_src[file_source[fi]] += len(na)
        if len(files) <= 20:
            print(f'{path}: {len(na)} structures, {time.time() - t0:.0f} s', flush=True)
    for si, n in enumerate(n_src):
        meta['sources'][si]['n'] = int(n)
    cellinfo = np.concatenate(cols.pop('cell')) if cols['cell'] else np.zeros((0, 4), np.float32)
    arr = {k: np.concatenate(v) if v else np.zeros(0, np.int64) for k, v in cols.items()}
    np.savez(out / 'sizes.npz', n_atoms=arr['n_atoms'].astype(np.uint16 if arr['n_atoms'].max(initial=0) < 65535
                                                            else np.uint32),
             n_edges=arr['n_edges'].astype(np.uint32), file_index=arr['file'].astype(np.uint32), key=arr['key'],
             cell=cellinfo)
    meta.update(files=files, kinds=kinds, file_source=file_source,
                n=int(len(arr['n_atoms'])), sum_atoms=int(arr['n_atoms'].sum()),
                sum_edges=int(arr['n_edges'].sum()),
                avg_n_neighbors=float(arr['n_edges'].sum() / max(arr['n_atoms'].sum(), 1)),
                max_atoms=int(arr['n_atoms'].max(initial=0)), max_edges=int(arr['n_edges'].max(initial=0)),
                max_cell_volume=float(cellinfo[:, 0].max(initial=0.0)),
                max_cell_length=[float(x) for x in cellinfo[:, 1:].max(axis=0, initial=0.0)],
                n_periodic=int((cellinfo[:, 0] > 0).sum()), dropped=drops)
    json.dump(meta, open(out / 'sizes_meta.json', 'w'), indent=1)
    print(f"size pass: {meta['n']} structures kept, dropped {drops or 0}, avg_n_neighbors "
          f"{meta['avg_n_neighbors']:.3f}, max {meta['max_atoms']} atoms / {meta['max_edges']} edges, "
          f"{time.time() - t0:.0f} s ({workers} workers)")
    return meta


def file_opts(meta: dict) -> list:
    """Source options of every file of a size table (empty dicts for tables made before 2026-11-11)."""
    fs = meta.get('file_source')
    if fs is None:
        return [{} for _ in meta['files']]
    return [dict(meta['sources'][si].get('opts') or {}) for si in fs]


def load_e0_table(spec, atomic_numbers) -> np.ndarray:
    """E0 per species index (order of atomic_numbers).  spec: a list (already per species), a dict {Z: E0}, or a
    file: JSON of either, or the BAM-torch text format ``({index: E0}, {Z: index})`` (get_enr_avg_per_element_*.txt)."""
    import ast
    if isinstance(spec, (str, os.PathLike)):
        text = Path(spec).read_text()
        try:
            spec = json.loads(text)
        except json.JSONDecodeError:
            spec = ast.literal_eval(text)
    if isinstance(spec, (tuple, list)) and len(spec) == 2 and all(isinstance(x, dict) for x in spec):
        e_by_index, index_of_z = spec
        spec = {int(z): float(e_by_index[i]) for z, i in index_of_z.items()}
    if isinstance(spec, dict):
        spec = {int(k): float(v) for k, v in spec.items()}
        missing = [z for z in atomic_numbers if int(z) not in spec]
        if missing:
            raise ValueError(f"E0 table without Z = {missing}")
        return np.asarray([spec[int(z)] for z in atomic_numbers], np.float64)
    out = np.asarray(spec, np.float64).reshape(-1)
    if len(out) != len(atomic_numbers):
        raise ValueError(f"E0 list of {len(out)} values for {len(atomic_numbers)} species")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sizes')
    s.add_argument('--source', action='append', required=True,
                   help='extxyz:<file>, aselmdb:<file|dir>, or a JSON object {"path": ..., options}')
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
        size_pass([json.loads(x) if x.lstrip().startswith('{') else x for x in a.source], a.cutoff, a.out,
                  a.workers, a.block, filters=flt)
    return 0


if __name__ == '__main__':
    sys.exit(main())
