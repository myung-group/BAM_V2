"""MPtrj (MPtrj_2022.9_full.json, figshare 23713842 v2, md5 50ead5f27f9a4f6beb7564c4188f1e9f) -> lmdb shards that
race_v2/stream_io.AselmdbSource reads (zlib JSON rows with integer keys, as ASE's lmdb backend stores them).

    python -I scripts/data/convert_mptrj.py <MPtrj_2022.9_full.json> <out_dir> [--val-frac 0.05] [--seed 0] [--shards 32]

Layout:  <out_dir>/train/mptrj_train_XXX.aselmdb, <out_dir>/val/mptrj_val_000.aselmdb, <out_dir>/convert_mptrj.json
(counts, split ids, per-element least-squares reference energies of the training split).

Conventions (stated because they decide what the model predicts):
  * energy  = uncorrected_total_energy (raw VASP; the label of MACE-MP-0 and most MPtrj-only models).  The MP2020-
    corrected total energy and the other MPtrj fields are kept in key_value_pairs (corrected_total_energy,
    energy_per_atom, ef_per_atom, e_per_atom_relaxed, mp_id, frame_id), so another label can be chosen later.
  * stress  = -0.1 GPa/kBar * (VASP stress in kBar), in eV/A^3: the ASE sign convention (CHGNet uses the same
    conversion to GPa).
  * split   = by material (mp_id), so no trajectory is shared between training and validation; seeded.
No frame is dropped here; the row filters of the size pass (stream_io.load_filters) apply at training time.
"""
import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np

KBAR_TO_EV_A3 = 0.1 / 160.21766208          # kBar -> GPa -> eV/A^3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('out')
    ap.add_argument('--val-frac', type=float, default=0.05); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--shards', type=int, default=32); ap.add_argument('--limit-materials', type=int, default=0)
    a = ap.parse_args()
    import lmdb
    from ase.data import atomic_numbers
    out = Path(a.out)
    (out / 'train').mkdir(parents=True, exist_ok=True); (out / 'val').mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = json.load(open(a.src))
    print(f'loaded {len(data)} materials, {time.time() - t0:.0f} s', flush=True)
    mids = sorted(data)
    if a.limit_materials:
        mids = mids[:a.limit_materials]
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(mids))
    n_val = int(round(a.val_frac * len(mids)))
    val_ids = {mids[i] for i in perm[:n_val]}
    shard_of = {mids[i]: k % a.shards for k, i in enumerate(perm[n_val:])}      # materials dealt round-robin

    def open_env(p):
        return lmdb.open(str(p), subdir=False, map_size=1 << 40, meminit=False, map_async=True, lock=False)
    envs = {('train', k): open_env(out / 'train' / f'mptrj_train_{k:03d}.aselmdb') for k in range(a.shards)}
    envs[('val', 0)] = open_env(out / 'val' / 'mptrj_val_000.aselmdb')
    txns = {k: e.begin(write=True) for k, e in envs.items()}
    nextid = {k: 1 for k in envs}
    ata, atb = np.zeros((119, 119)), np.zeros(119)                    # normal equations of the E0 fit (training split)
    stats = {'train': 0, 'val': 0, 'atoms_train': 0, 'atoms_val': 0, 'max_force': 0.0, 'max_abs_stress_gpa': 0.0,
             'skipped': 0}
    for m, mid in enumerate(mids):
        tgt = ('val', 0) if mid in val_ids else ('train', shard_of[mid])
        for fid, fr in data[mid].items():
            st = fr['structure']
            try:
                z = [atomic_numbers[s['species'][0]['element']] for s in st['sites']]
                pos = [s['xyz'] for s in st['sites']]
                f = fr['force']; e = fr['uncorrected_total_energy']
                if e is None or f is None or len(f) != len(z) or any(len(s['species']) != 1 for s in st['sites']):
                    raise ValueError('incomplete frame')
            except (KeyError, ValueError, TypeError):
                stats['skipped'] += 1
                continue
            sk = fr.get('stress')
            row = {'numbers': z, 'positions': pos, 'cell': st['lattice']['matrix'], 'pbc': [True, True, True],
                   'energy': float(e), 'forces': f,
                   'stress': None if sk is None else (-KBAR_TO_EV_A3 * np.asarray(sk, np.float64)).tolist(),
                   'key_value_pairs': {k: fr.get(k) for k in ('corrected_total_energy', 'energy_per_atom',
                                                             'ef_per_atom', 'e_per_atom_relaxed')},
                   'data': {'mp_id': mid, 'frame_id': fid, 'magmom': fr.get('magmom')}}
            k = nextid[tgt]
            txns[tgt].put(str(k).encode('ascii'), zlib.compress(json.dumps(row).encode('utf-8'), 1))
            nextid[tgt] = k + 1
            sp = tgt[0]
            stats[sp] += 1; stats[f'atoms_{sp}'] += len(z)
            fa = np.asarray(f, np.float64)
            stats['max_force'] = max(stats['max_force'], float(np.sqrt((fa * fa).sum(-1).max())))
            if sk is not None:
                stats['max_abs_stress_gpa'] = max(stats['max_abs_stress_gpa'], 0.1 * float(np.abs(sk).max()))
            if sp == 'train':
                c = np.bincount(z, minlength=119).astype(np.float64)
                ata += np.outer(c, c); atb += c * float(e)
        if (m + 1) % 20000 == 0:
            for kk in txns:
                txns[kk].commit(); txns[kk] = envs[kk].begin(write=True)
            print(f'{m + 1}/{len(mids)} materials, train {stats["train"]} val {stats["val"]}, '
                  f'{time.time() - t0:.0f} s', flush=True)
    for kk in txns:
        txns[kk].put(b'nextid', zlib.compress(json.dumps(nextid[kk]).encode()))
        txns[kk].commit(); envs[kk].sync(); envs[kk].close()
    present = np.flatnonzero(np.diag(ata) > 0)
    e0 = np.linalg.lstsq(ata[np.ix_(present, present)], atb[present], rcond=None)[0]
    meta = {'source': str(a.src), 'energy': 'uncorrected_total_energy', 'stress': 'eV/A^3, ASE sign (-0.1*kBar GPa)',
            'val_frac': a.val_frac, 'seed': a.seed, 'shards': a.shards, 'n_materials': len(mids),
            'n_val_materials': n_val, **stats,
            'e0_ls_train': {int(zz): float(v) for zz, v in zip(present, e0)},
            'val_materials': sorted(val_ids)}
    json.dump(meta, open(out / 'convert_mptrj.json', 'w'), indent=1)
    print({k: v for k, v in meta.items() if k not in ('val_materials', 'e0_ls_train')})
    print('elements', len(present), 'E0 range', float(e0.min()), float(e0.max()), f'{time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
