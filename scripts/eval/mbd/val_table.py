"""val_table.py <run suffix> [first epoch]: per-epoch validation E MAE / F MAE / F RMSE / S RMSE (physical units)."""
import re, sys
import os
W = os.environ.get('RACE_WORK', '/scratch/t589a02/BAM_V2_more/work')
def read(n):
    L = open(f'{W}/runs/{n}_cue/loss_{n}_cue.out', errors='replace').read().splitlines(); d = {}
    for i, l in enumerate(L):
        m = re.match(r'^\d+/\d+/\d+ +[\d:]+ +(\d+) ', l)
        if m:
            v = l.split('|')[2].split(); lr = l.split('|')[3].split()[0]; t = L[i + 1] if i + 1 < len(L) else ''
            em = re.search(r'E_MAE ([\d.]+)', t); sr = re.search(r'S_RMSE ([\d.]+)', t)
            d[int(m.group(1))] = (float(em.group(1)) if em else float('nan'), float(v[4]), float(v[2]), float(sr.group(1)) if sr else float('nan'), lr)
    return d
for n in sys.argv[1].split(','):
    d = read(n); e0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    print(n); [print(f'  e{e}: {v[0]:.1f} {v[1]:.1f} {v[2]:.1f} {v[3]:.3f}  lr {v[4]}') for e, v in sorted(d.items()) if e >= e0]
