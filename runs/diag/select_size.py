"""Apply the pre-registered size selection rule (docs/size_screening_protocol_2026-11-03.md) to finished runs:
final-epoch EMA validation metrics only (summary['final_ema']['valid']); candidates = energy RMSE within +10 % of the
best; choose the lowest force RMSE; ties (< 2 %) -> the smaller model (fewer parameters, read from stdout.txt).

    python -I runs/diag/select_size.py sz100k_L5x64_omat24_v2_s42 sz100k_L5x128_omat24_v2_s42 ...
"""
import json
import re
import sys
from pathlib import Path


def params(name):
    t = Path(f'runs/{name}_cue/stdout.txt').read_text(errors='ignore').replace('\r', '\n')
    m = re.search(r'Model parameters: ([\d,]+)', t)
    return int(m.group(1).replace(',', '')) if m else None


rows = []
for n in sys.argv[1:]:
    p = Path(f'runs/{n}_cue/loss_{n}_cue.summary.json')
    if not p.exists():
        print(f'{n}: no summary yet'); continue
    v = json.load(open(p))['final_ema']['valid']
    rows.append((n, 1e3 * v['energy_rmse'], 1e3 * v['force_rmse'], 1e3 * v['energy_mae'], params(n)))
if not rows:
    sys.exit()
best_e = min(r[1] for r in rows)
cand = [r for r in rows if r[1] <= 1.10 * best_e]
best_f = min(r[2] for r in cand)
tied = [r for r in cand if r[2] <= 1.02 * best_f]
pick = min(tied, key=lambda r: (r[4] or 0))
print(f"{'run':45s} {'E RMSE':>8s} {'F RMSE':>8s} {'E MAE':>8s} {'params':>11s}  (final-epoch EMA, validation; meV/atom, meV/A)")
for r in sorted(rows, key=lambda r: r[2]):
    tag = ' <- selected' if r is pick else (' candidate' if r in cand else '')
    print(f"{r[0]:45s} {r[1]:8.2f} {r[2]:8.2f} {r[3]:8.2f} {r[4] or 0:11,d}{tag}")
