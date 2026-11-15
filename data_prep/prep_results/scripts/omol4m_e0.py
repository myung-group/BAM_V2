"""Per-element reference energies (least squares, training rows only) of OMol25 aselmdb shards, for data_format
"stream" (the trainer needs atom_energies with e0_mode "isolated").

JAX-free: imports bam_omat24/stream_io.py as a plain module (the package __init__ would import JAX).
    BAM_REPO=<repo> python -I omol4m_e0.py --sizes <train size-table dir> --out <dir> [--workers 64] [--max-rows N]

The training sources are taken from <sizes>/sizes_meta.json, so E0 is fitted on exactly the files the trainer reads.
Writes <out>/e0.json (atomic_numbers, atom_energies) and <out>/meta.json with the fields that
runs/diag/make_omol_configs.py reads (atomic_numbers, atom_energies, avg_n_neighbors, periodic, train / valid / test).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(os.environ.get('BAM_REPO', '/scratch/t510a13/BAM_V2')) / 'bam_omat24'))
import stream_io as S   # noqa: E402

Z_MAX = 118
LABEL = 'lowdin_charges'      # per-atom reference charges of the optional charge loss


def _task(args):
    path, keys = args
    src = S.AselmdbSource([path])
    counts = np.zeros((len(keys), Z_MAX + 1), np.uint16)
    energy = np.full(len(keys), np.nan)
    charge = np.zeros(len(keys), np.int16)
    spin = np.zeros(len(keys), np.int16)
    periodic, labelled = 0, 0
    for i, st in enumerate(src.read_block(0, keys)):
        lab = st['info'].get(LABEL)
        labelled += bool(lab is not None and np.size(lab) == len(st['numbers'])
                         and np.all(np.isfinite(np.asarray(lab, np.float64))))
        counts[i] = np.bincount(st['numbers'], minlength=Z_MAX + 1)[:Z_MAX + 1]
        if st['energy'] is not None:
            energy[i] = st['energy']
        charge[i] = int(round(float(st['info'].get('charge') or 0)))
        spin[i] = int(round(float(st['info'].get('spin') or 1)))
        periodic += bool(st['pbc'] is not None and np.any(st['pbc']))
    return counts, energy, charge, spin, periodic, labelled


def main():
    import concurrent.futures as cf
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', required=True, help='size-table directory of the training split')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--block', type=int, default=1024)
    ap.add_argument('--max-rows', type=int, default=0, help='rows per file (0 = all; for a quick check)')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    smeta = json.load(open(Path(a.sizes) / 'sizes_meta.json'))
    files = smeta['files']
    t0 = time.time()
    tasks = []
    for path in files:
        keys = np.asarray(S.AselmdbSource([path]).row_ids(0), np.int64)
        if a.max_rows:
            keys = keys[:a.max_rows]
        tasks += [(path, keys[k:k + a.block]) for k in range(0, len(keys), a.block)]
    print(f'{len(files)} files, {sum(len(t[1]) for t in tasks)} rows, {len(tasks)} blocks', flush=True)
    if a.workers <= 1:
        parts = [_task(t) for t in tasks]
    else:
        with cf.ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context('forkserver')) as ex:
            parts = list(ex.map(_task, tasks, chunksize=4))
    counts = np.concatenate([p[0] for p in parts])
    energy = np.concatenate([p[1] for p in parts])
    charge = np.concatenate([p[2] for p in parts])
    spin = np.concatenate([p[3] for p in parts])
    n_periodic = int(sum(p[4] for p in parts))
    n_labelled = int(sum(p[5] for p in parts))
    print(f'read {len(energy)} rows in {time.time() - t0:.0f} s; rows with finite {LABEL} for every atom: '
          f'{n_labelled} ({len(energy) - n_labelled} without)', flush=True)
    if not a.max_rows and len(energy) != smeta['n']:
        raise SystemExit(f"row count {len(energy)} != size table {smeta['n']}")
    ok = np.isfinite(energy)
    n_skip = int((~ok).sum())
    counts, energy = counts[ok], energy[ok]
    Z = [int(z) for z in np.nonzero(counts.sum(0))[0]]
    assert Z and Z[0] >= 1, Z
    X = counts[:, Z].astype(np.float64)
    e0, _, rank, _ = np.linalg.lstsq(X, energy, rcond=None)
    res = energy - X @ e0
    n_at = X.sum(1)
    print(f'elements {len(Z)} (Z max {max(Z)}), rank {rank}; rows without energy {n_skip}; periodic rows {n_periodic}')
    print(f'residual: std {res.std():.4f} eV, per atom std {(res / n_at).std():.5f} eV/atom, '
          f'abs max {np.abs(res).max():.2f} eV; percentiles 1/50/99 {np.percentile(res, [1, 50, 99]).round(3).tolist()}')
    print('  Z  structures      atoms            E0 (eV)')
    for j, z in enumerate(Z):
        print(f'{z:3d} {int((X[:, j] > 0).sum()):11d} {int(X[:, j].sum()):10d} {e0[j]:18.6f}')
    cs = {}
    for q, s in zip(charge[ok].tolist(), spin[ok].tolist()):
        cs[(q, s)] = cs.get((q, s), 0) + 1
    top = sorted(cs.items(), key=lambda kv: -kv[1])[:12]
    print('most frequent (charge, spin):', {f'{q:+d}/{s}': n for (q, s), n in top})
    if rank < len(Z):
        raise SystemExit(f'rank {rank} < {len(Z)} elements: E0 is not identifiable, nothing written')
    meta = dict(source=dict(sizes=str(a.sizes), files=files), rule='E0: least squares per element over all rows of the '
                'training sources (official train split, 4M subset); no validation or test rows',
                n_rows=int(len(energy)), rows_without_energy=n_skip, max_rows_per_file=a.max_rows,
                atomic_numbers=Z, atom_energies=[float(v) for v in e0], e0_rank=int(rank),
                e0_residual_std_eV=float(res.std()), e0_residual_std_eV_per_atom=float((res / n_at).std()),
                avg_n_neighbors=float(smeta['avg_n_neighbors']), cutoff=float(smeta['cutoff']),
                periodic=bool(n_periodic), stress=False, ref_charges_key=[LABEL], rows_with_charge_labels=n_labelled,
                train=dict(path='stream', n=int(len(energy))), valid=dict(path='stream'), test=dict(path='stream'))
    json.dump(meta, open(out / 'meta.json', 'w'), indent=1)
    json.dump(dict(atomic_numbers=Z, atom_energies=meta['atom_energies']), open(out / 'e0.json', 'w'), indent=1)
    print(f'wrote {out}/e0.json and meta.json; {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
