"""Random subset of the MPtrj training rows as one extxyz file, for the per-element energy refit (fair protocol:
training split only).  JAX-free (lmdb + ase); rows that the training filters drop are skipped, as in training.

    python -I refit_subset.py --lmdb /scratch/t510a13/data/mptrj_lmdb/train --filters .../filters.json \
        --n 20000 --seed 0 --out runs/mbd/refit/mptrj_train_20k.extxyz
info['material_id'] = tr<file>_<row>, info['ref_energy'] = the training label (uncorrected total energy, eV)."""
import argparse, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--lmdb', required=True); ap.add_argument('--filters'); ap.add_argument('--n', type=int, default=20000)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', required=True)
ap.add_argument('--repo', default='/scratch/t510a13/BAM_V2_merged/race_v2')
a = ap.parse_args()
sys.path.insert(0, a.repo)
import stream_io as S  # noqa: E402
import ase, ase.io  # noqa: E402
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402

files = sorted(str(p) for p in Path(a.lmdb).glob('*.aselmdb'))
src = S.AselmdbSource(files, lru=len(files))
flt = S.load_filters(a.filters) if a.filters else None
ids = [(f, r) for f in range(len(files)) for r in src.row_ids(f)]
rng = np.random.RandomState(a.seed)
order = rng.permutation(len(ids))
frames, n_drop = [], 0
for k in order:
    if len(frames) >= a.n:
        break
    f, r = ids[k]
    st = src.read_block(f, [r])[0]
    if flt is not None and S.drop_reason(st, flt) is not None:
        n_drop += 1
        continue
    at = ase.Atoms(numbers=st['numbers'], positions=st['positions'], cell=st['cell'], pbc=True)
    at.info['material_id'] = f'tr{f:02d}_{r}'; at.info['ref_energy'] = float(st['energy'])
    frames.append(at)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
ase.io.write(a.out, frames, format='extxyz')
n_at = np.array([len(x) for x in frames])
print(f'{len(ids)} training rows in {len(files)} files; wrote {len(frames)} frames ({n_drop} dropped by the filters), '
      f'atoms per frame mean {n_at.mean():.1f} max {n_at.max()}, elements {len(set(int(z) for x in frames for z in x.numbers))} -> {a.out}')
