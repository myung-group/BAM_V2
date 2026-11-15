"""OMol25 (Meta FAIR, ωB97M-V/def2-TZVPD; Hugging Face facebook/OMol25, data CC BY 4.0) aselmdb -> extxyz subsets
with total charge and spin, for the RACEv2 / RACE_LR pretraining pilot.

usage:
  python -I data_prep/convert_omol25.py inspect <aselmdb file or dir> [--rows 3]
      prints the files, row counts and the key layout of the first rows (where charge / spin / partial charges are)
  python -I data_prep/convert_omol25.py subset --train <train aselmdb dir> --eval <val aselmdb dir> --out <dir>
      [--n-train 100000] [--n-valid 2000] [--n-test 2000] [--seed 0] [--max-atoms 350]

Protocol (fair: the official splits only):
  train.xyz + valid.xyz : disjoint uniform samples of the official TRAIN split (numpy RandomState(seed) over the
                          global row list of all files, in sorted file order); valid = model selection only
  test.xyz              : uniform sample of the official VAL split (the OMol25 test labels are not public)
  e0.json / meta.json   : per-element reference energies from a least-squares fit on train.xyz only, element list,
                          avg_n_neighbors (5 A, trainer graph code), counts per (charge, spin), md5 of the outputs
The aselmdb files are read as data (LMDB + zlib JSON rows, race_v2.data.stream_io readers); nothing is executed.
Units: OMol25 stores eV and eV/A (fairchem); the converter does not rescale.  Total charge -> info["charge"] (int),
spin multiplicity 2S+1 -> info["spin"] (int).  The location of these two fields in the rows is searched in this
order: row top level, row["data"], row["key_value_pairs"]; a row without them stops the conversion (no default is
guessed) unless --neutral-if-missing is given (charge 0, spin 1 for such rows; counted in meta.json).
Per-atom partial charges are copied to arrays["ref_charges"] when a row has one of PARTIAL_KEYS.
--train and --eval may be the same files (a piece with one public split): test is then drawn from the unused rows.
Periodic rows (other aselmdb datasets, e.g. OMat24-type): only with --allow-periodic; the cell, pbc and the stress
(stream_io.decode_row: eV/A^3, ASE sign and Voigt order, as stored) are kept.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from race_v2.data.stream_io import open_lmdb, list_row_ids, decode_value, decode_row, _arr, _scalar   # noqa: E402

OPTS = {'allow_periodic': False, 'neutral_if_missing': False}
STATS = {'neutral_default': 0}

PARTIAL_KEYS = ('lowdin_charges', 'mulliken_charges', 'hirshfeld_charges', 'charges')


def files_of(p):
    p = Path(p)
    return [p] if p.is_file() else sorted(p.rglob('*.aselmdb'))


def _find(d, key):
    for where in (d, d.get('data') or {}, d.get('key_value_pairs') or {}):
        if isinstance(where, dict) and key in where and where[key] is not None:
            return where[key]
    return None


def to_atoms(d, where):
    import ase
    from ase.calculators.singlepoint import SinglePointCalculator
    r = decode_row(raw=None, d=d)
    q, s = _find(d, 'charge'), _find(d, 'spin')
    if q is None or s is None:
        if not OPTS['neutral_if_missing']:
            raise ValueError(f"{where}: no total charge / spin in the row (keys {sorted(d)[:12]}); run 'inspect', "
                             "or pass --neutral-if-missing for a dataset that is neutral by construction")
        q = 0.0 if q is None else q
        s = 1.0 if s is None else s
        STATS['neutral_default'] += 1
    q, s = _scalar(q), _scalar(s)
    assert abs(q - round(q)) < 1e-6 and abs(s - round(s)) < 1e-6 and s >= 1, (where, q, s)
    pbc = r['pbc']
    periodic = pbc is not None and bool(np.any(pbc))
    if periodic and not OPTS['allow_periodic']:
        raise ValueError(f"{where}: periodic row; pass --allow-periodic to keep cell, pbc and stress")
    if periodic:
        a = ase.Atoms(numbers=r['numbers'], positions=r['positions'].reshape(-1, 3), cell=r['cell'], pbc=pbc)
    else:
        a = ase.Atoms(numbers=r['numbers'], positions=r['positions'].reshape(-1, 3), pbc=False)
    for k in PARTIAL_KEYS:
        v = _find(d, k)
        if v is not None:
            v = _arr(v, np.float64).reshape(-1)
            if v.shape[0] == len(a):
                a.arrays['ref_charges'] = v
                a.info['ref_charges_key'] = k
                break
    e, f = r['energy'], r['forces']
    if e is None or f is None:
        raise ValueError(f"{where}: no energy / forces")
    kw = dict(stress=r['stress']) if (periodic and r['stress'] is not None) else {}
    a.calc = SinglePointCalculator(a, energy=e, forces=f.reshape(-1, 3), **kw)
    a.info.update(charge=int(round(q)), spin=int(round(s)), source=where)
    return a


def inspect(args):
    fs = files_of(args.path)
    print(f'{len(fs)} aselmdb files under {args.path}')
    total = 0
    for i, f in enumerate(fs):
        env = open_lmdb(f)
        ids = list_row_ids(env)
        total += len(ids)
        if i < 3 or i == len(fs) - 1:
            print(f'  {f.name}: {len(ids)} rows')
        if i == 0:
            with env.begin() as txn:
                for k in ids[:args.rows]:
                    d = decode_value(txn.get(str(k).encode()))
                    print(f'  row {k}: top-level keys {sorted(d)}')
                    for sub in ('data', 'key_value_pairs'):
                        if isinstance(d.get(sub), dict):
                            print(f'    {sub}: { {kk: type(vv).__name__ for kk, vv in d[sub].items()} }')
                    print(f'    charge={_find(d, "charge")!r} spin={_find(d, "spin")!r} n_atoms={len(_arr(d.get("numbers"), np.int64))}')
        env.close()
    print(f'total rows: {total}')


def sample(files, n, rng, exclude=None, max_atoms=None):
    """Uniform rows without replacement over all files: list of (file index, row id)."""
    rows = []
    for i, f in enumerate(files):
        env = open_lmdb(f)
        rows.extend((i, k) for k in list_row_ids(env))
        env.close()
    if exclude:
        rows = [r for r in rows if r not in exclude]
    order = rng.permutation(len(rows))
    out = []
    for j in order:
        if len(out) == n:
            break
        out.append(rows[j])
    return out, len(rows)


def read_rows(files, picks, max_atoms):
    by_file = {}
    for i, k in picks:
        by_file.setdefault(i, []).append(k)
    atoms, skipped = {}, 0
    for i, ks in by_file.items():
        env = open_lmdb(files[i])
        with env.begin() as txn:
            for k in ks:
                d = decode_value(txn.get(str(k).encode()))
                a = to_atoms(d, f'{files[i].name}:{k}')
                if max_atoms and len(a) > max_atoms:
                    skipped += 1
                    continue
                atoms[(i, k)] = a
        env.close()
    return [atoms[p] for p in picks if p in atoms], skipped


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def subset(args):
    import ase.io
    from race_v2.data.xyz_graphs import preprocess_graph, atomic_numbers_to_indices
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tr_files, ev_files = files_of(args.train), files_of(args.eval)
    rng = np.random.RandomState(args.seed)
    picks, n_rows_tr = sample(tr_files, args.n_train + args.n_valid, rng)
    frames, sk1 = read_rows(tr_files, picks, args.max_atoms)
    train, valid = frames[:args.n_train], frames[args.n_train:]
    # same files given as --train and --eval (a dataset piece with one public split only): test rows are drawn from
    # the rows not used for train / valid, so the three sets stay disjoint
    same = [f.resolve() for f in tr_files] == [f.resolve() for f in ev_files]
    tpicks, n_rows_ev = sample(ev_files, args.n_test, np.random.RandomState(args.seed + 1),
                               exclude=set(picks) if same else None)
    test, sk2 = read_rows(ev_files, tpicks, args.max_atoms)
    Z = sorted({int(z) for a in train for z in a.numbers})
    missing = sorted({int(z) for a in valid + test for z in a.numbers} - set(Z))
    if missing:
        print(f'WARNING: elements {missing} occur in valid/test but not in train; those frames are dropped')
        valid = [a for a in valid if set(a.numbers) <= set(Z)]
        test = [a for a in test if set(a.numbers) <= set(Z)]
    X = np.array([[np.sum(a.numbers == z) for z in Z] for a in train], float)
    y = np.array([a.get_potential_energy() for a in train])
    e0, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ e0
    paths = {}
    for name, fr in (('train', train), ('valid', valid), ('test', test)):
        paths[name] = out / f'{name}.xyz'
        ase.io.write(paths[name], fr, format='extxyz')
    ai = atomic_numbers_to_indices(Z)
    probe = train[:2000]
    nb = sum(int(preprocess_graph(a, ai, [0.0] * len(Z), 5.0).n_edge.sum()) for a in probe) / sum(len(a) for a in probe)
    cs = lambda fr: {f'{q:+d}/{s}': n for (q, s), n in sorted(Counter((a.info['charge'], a.info['spin']) for a in fr).items())}
    meta = dict(source=dict(train=str(args.train), eval=str(args.eval), rows_train_split=n_rows_tr,
                            rows_val_split=n_rows_ev),
                seed=args.seed, max_atoms=args.max_atoms, skipped_over_max_atoms=sk1 + sk2,
                rule='train+valid: RandomState(seed).permutation over all rows of the TRAIN split, first n_train = '
                     'train, next n_valid = valid; test: RandomState(seed+1) over the VAL split'
                     + (' (same files as TRAIN: over the rows not drawn for train / valid)' if same else ''),
                periodic=bool(any(a.pbc.any() for a in train)),
                stress=bool(any('stress' in a.calc.results for a in train)),
                rows_with_neutral_default=STATS['neutral_default'],
                atomic_numbers=Z, atom_energies=[float(v) for v in e0], e0_rank=int(rank),
                e0_residual_std_eV=float(res.std()), avg_n_neighbors=nb,
                ref_charges_key=sorted({a.info.get('ref_charges_key') for a in train if 'ref_charges_key' in a.info}),
                **{name: dict(path=str(paths[name]), n=len(fr), n_atoms_mean=float(np.mean([len(a) for a in fr])),
                              n_atoms_max=int(max(len(a) for a in fr)), charge_spin=cs(fr), md5=md5(paths[name]))
                   for name, fr in (('train', train), ('valid', valid), ('test', test))})
    json.dump(meta, open(out / 'meta.json', 'w'), indent=1)
    json.dump(dict(atomic_numbers=Z, atom_energies=meta['atom_energies']), open(out / 'e0.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k not in ('atom_energies',)}, indent=1)[:3000])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('inspect')
    a.add_argument('path')
    a.add_argument('--rows', type=int, default=3)
    b = sub.add_parser('subset')
    b.add_argument('--train', required=True)
    b.add_argument('--eval', required=True)
    b.add_argument('--out', required=True)
    b.add_argument('--n-train', type=int, default=100000)
    b.add_argument('--n-valid', type=int, default=2000)
    b.add_argument('--n-test', type=int, default=2000)
    b.add_argument('--seed', type=int, default=0)
    b.add_argument('--max-atoms', type=int, default=350)
    b.add_argument('--allow-periodic', action='store_true', help='keep periodic rows (cell, pbc, stress)')
    b.add_argument('--neutral-if-missing', action='store_true',
                   help='rows without charge / spin get charge 0 and spin 1 (counted in meta.json)')
    args = ap.parse_args()
    if args.cmd == 'subset':
        OPTS.update(allow_periodic=args.allow_periodic, neutral_if_missing=args.neutral_if_missing)
    inspect(args) if args.cmd == 'inspect' else subset(args)


if __name__ == '__main__':
    main()
