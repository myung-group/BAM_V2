"""Force sign and unit check of an energy / force dataset without a model: for pairs of structures of the same molecule
with the same atom order (same mapped SMILES), the work along the straight path, trapezoid rule
    dE_pred = -1/2 (F_1 + F_2) . (x_2 - x_1)
must equal E_2 - E_1 to second order.  The slope of dE vs dE_pred is +1 for forces = -dE/dx in the energy units per
length unit, -1 for the opposite sign; a slope far from +-1 means inconsistent units.

    python -I scripts/data/check_force_sign.py <descent_data shard.parquet> [--rows 200000] [--max-rms 0.05]
(--max-rms: only pairs whose atomic displacement RMS is below this many Å, where the trapezoid rule is accurate)
"""
import argparse
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('parquet')
    ap.add_argument('--rows', type=int, default=200000)
    ap.add_argument('--max-rms', type=float, default=0.05)
    args = ap.parse_args()
    import pyarrow.parquet as pq
    t = pq.read_table(args.parquet).slice(0, args.rows).to_pydict()
    groups = defaultdict(list)
    for i, s in enumerate(t['smiles']):
        groups[s].append(i)
    de, dp, rms_l = [], [], []
    for idx in groups.values():
        if len(idx) < 2:
            continue
        X = [np.asarray(t['coords'][i], np.float64).reshape(-1, 3) for i in idx]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                dx = X[b] - X[a]
                rms = np.sqrt((dx ** 2).sum(1).mean())
                if rms > args.max_rms:
                    continue
                fa = np.asarray(t['forces'][idx[a]], np.float64).reshape(-1, 3)
                fb = np.asarray(t['forces'][idx[b]], np.float64).reshape(-1, 3)
                de.append(t['energy'][idx[b]][0] - t['energy'][idx[a]][0])
                dp.append(-0.5 * np.sum((fa + fb) * dx))
                rms_l.append(rms)
    de, dp = np.array(de), np.array(dp)
    n_multi = sum(len(v) > 1 for v in groups.values())
    print(f'{len(groups)} molecules in {args.rows} rows, {n_multi} with >1 conformer; {len(de)} close pairs '
          f'(rms < {args.max_rms} A)')
    if len(de) < 10:
        print('too few close pairs; raise --max-rms or --rows')
        return
    slope = float(np.sum(de * dp) / np.sum(dp * dp))
    r = float(np.corrcoef(de, dp)[0, 1])
    print(f'slope dE / dE_pred = {slope:+.4f}, correlation {r:+.4f}, |dE| median {np.median(np.abs(de)):.3f} (energy '
          f'units), residual median {np.median(np.abs(de - slope * dp)):.4f}')
    print('=> forces = -dE/dx as stored' if slope > 0.9 and r > 0.99 else
          '=> forces = +dE/dx as stored (use --force-sign -1)' if slope < -0.9 and r < -0.99 else
          '=> inconclusive or inconsistent units')


if __name__ == '__main__':
    main()
