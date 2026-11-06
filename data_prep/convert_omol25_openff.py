"""OMol25 ANI-2x part, as redistributed by OpenFF (Hugging Face openforcefield/descent-format-ani2x, CC BY 4.0,
revision f1a802f2ed77d3b424ff245d9e455cf0a48ea19e) -> extxyz subsets for the local RACEv2 / RACE_LR pilot.

usage: python -I data_prep/convert_omol25_openff.py <raw_dir> <out_dir> [--n-train 10000] [--n-valid 1000]
           [--n-test 2932] [--seed 0] [--force-sign 1] [--max-rows N]
  raw_dir: descent_data/train-00000-of-00010.parquet and metadata/train-00000-of-00006.parquet (only shard 0 of each
           is used; both tables start at global row 0, and the SMILES of every used row are checked to be equal)

What the files hold (checked 2026-11-06 on shard 0): every row is neutral (charge 0) and closed shell (spin 1), elements
H C N O F S Cl, one conformer per row; OMol25_split train / val (all 2932 val rows are in shard 0).  So this set cannot
test the total-charge / spin input; it tests the pipeline, the optimiser / precision choice and the Qeq stratum on
neutral molecules, and has Löwdin charges (-> arrays["ref_charges"]) for the charge-loss variant.
Atom order: the coordinates follow the atom-map numbers 1..N of the mapped SMILES; the element of map number k is read
from the bracket atom [El...:k].
Units (descent / smee convention): coords Å, energy kcal/mol, forces kcal/mol/Å; converted with 1 kcal/mol =
0.0433641043 eV.  The force sign is not documented in the card: --force-sign (+1 = forces as stored) is checked
empirically (runs/diag/check_force_sign.py: only the right sign lets energies and forces be fitted together).
Energies are stored as float32 (resolution about 0.06 kcal/mol = 2.7 meV at 1e6 kcal/mol): a floor for energy errors.
Split (fair): train / valid = disjoint RandomState(seed) samples of the official TRAIN rows of shard 0, test = the
official VAL rows (all of them by default).  E0 = per-element least squares on train only.
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

KCAL = 0.0433641043          # eV per kcal/mol
ATOM_RE = re.compile(r'\[([A-Z][a-z]?|[cnosp])[^\]:]*:(\d+)\]')     # lower case = aromatic (c n o s p)
SYMBOLS = {'H', 'C', 'N', 'O', 'F', 'S', 'Cl'}


def symbols_of(smiles, n):
    found = {int(k): (el.upper() if el.islower() else el) for el, k in ATOM_RE.findall(smiles)}
    if sorted(found) != list(range(1, n + 1)):
        raise ValueError(f'atom maps {sorted(found)[:5]}... do not cover 1..{n}: {smiles}')
    syms = [found[k] for k in range(1, n + 1)]
    assert set(syms) <= SYMBOLS, (set(syms), smiles)
    return syms


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw_dir', type=Path)
    ap.add_argument('out_dir', type=Path)
    ap.add_argument('--n-train', type=int, default=10000)
    ap.add_argument('--n-valid', type=int, default=1000)
    ap.add_argument('--n-test', type=int, default=2932)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--force-sign', type=int, default=1, choices=[1, -1])
    ap.add_argument('--max-rows', type=int, default=None, help='use only the first N rows (quick tests)')
    args = ap.parse_args()
    import pyarrow.parquet as pq
    import ase
    import ase.io
    from ase.calculators.singlepoint import SinglePointCalculator
    data = pq.read_table(args.raw_dir / 'descent_data' / 'train-00000-of-00010.parquet')
    n_rows = data.num_rows if args.max_rows is None else min(args.max_rows, data.num_rows)
    meta = pq.read_table(args.raw_dir / 'metadata' / 'train-00000-of-00006.parquet',
                         columns=['OMol25_id', 'OMol25_split', 'charge', 'OpenFF_spin=1', 'smiles', 'lowdin_charges']
                         ).slice(0, n_rows).to_pydict()
    split = np.array(meta['OMol25_split'])
    rng = np.random.RandomState(args.seed)
    tr_rows = np.flatnonzero(split == 'train')
    va_rows = np.flatnonzero(split == 'val')
    pick = rng.permutation(tr_rows)[:args.n_train + args.n_valid]
    sets = {'train': np.sort(pick[:args.n_train]), 'valid': np.sort(pick[args.n_train:]),
            'test': np.sort(rng.permutation(va_rows)[:args.n_test])}
    need = np.unique(np.concatenate(list(sets.values())))
    d = data.take(need).to_pydict()
    row_of = {int(r): i for i, r in enumerate(need)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    for name, rows in sets.items():
        out = []
        for r in rows:
            i = row_of[int(r)]
            smi = d['smiles'][i]
            assert smi == meta['smiles'][r], f'row {r}: SMILES of the two tables differ'
            assert meta['charge'][r] == 0 and meta['OpenFF_spin=1'][r], (r, meta['charge'][r])
            xyz = np.asarray(d['coords'][i], np.float64).reshape(-1, 3)
            n = len(xyz)
            a = ase.Atoms(symbols_of(smi, n), positions=xyz, pbc=False)
            e = float(d['energy'][i][0]) * KCAL
            f = args.force_sign * np.asarray(d['forces'][i], np.float64).reshape(-1, 3) * KCAL
            q = meta['lowdin_charges'][r]
            if q is not None and len(q) == n:
                a.arrays['ref_charges'] = np.asarray(q, np.float64)
            a.calc = SinglePointCalculator(a, energy=e, forces=f)
            a.info.update(charge=0, spin=1, omol25_id=int(meta['OMol25_id'][r]), openff_row=int(r))
            out.append(a)
        frames[name] = out
    Z = sorted({int(z) for a in frames['train'] for z in a.numbers})
    X = np.array([[np.sum(a.numbers == z) for z in Z] for a in frames['train']], float)
    y = np.array([a.get_potential_energy() for a in frames['train']])
    e0, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    for name in ('valid', 'test'):
        frames[name] = [a for a in frames[name] if set(a.numbers) <= set(Z)]
    info = dict(source='openforcefield/descent-format-ani2x@f1a802f2ed77d3b424ff245d9e455cf0a48ea19e (shard 0 of '
                       'descent_data and metadata), OMol25 ANI-2x part, CC BY 4.0',
                rows_used=n_rows, seed=args.seed, force_sign=args.force_sign, units_in='kcal/mol, kcal/mol/A, A',
                atomic_numbers=Z, atom_energies=[float(v) for v in e0], e0_rank=int(rank),
                e0_residual_std_eV=float((y - X @ e0).std()), periodic=False, stress=False,
                ref_charges_key=['lowdin_charges'])
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bam_omat24.data_3bpa import preprocess_graph, atomic_numbers_to_indices
    ai = atomic_numbers_to_indices(Z)
    probe = frames['train'][:2000]
    info['avg_n_neighbors'] = sum(int(preprocess_graph(a, ai, [0.0] * len(Z), 5.0).n_edge.sum()) for a in probe) / \
        sum(len(a) for a in probe)
    for name, fr in frames.items():
        p = args.out_dir / f'{name}.xyz'
        ase.io.write(p, fr, format='extxyz')
        info[name] = dict(path=str(p), n=len(fr), n_atoms_mean=float(np.mean([len(a) for a in fr])),
                          n_atoms_max=int(max(len(a) for a in fr)), charge_spin={'+0/1': len(fr)}, md5=md5(p))
    info['elements_train'] = dict(Counter(int(z) for a in frames['train'] for z in a.numbers))
    json.dump(info, open(args.out_dir / 'meta.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != 'atom_energies'}, indent=1))


if __name__ == '__main__':
    main()
