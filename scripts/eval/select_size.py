"""Apply the pre-registered size selection rule (docs/size_screening_protocol_2026-11-03.md) to finished runs:
final-epoch EMA validation metrics only (summary['final_ema']['valid']); candidates = energy RMSE within +10 % of the
best; choose the lowest force RMSE; ties (< 2 %) -> the smaller model (fewer parameters).  Runs after --controls are
listed for comparison but take no part in the rule.  Secondary (not part of the rule): mean of the last 5 epochs of
the per-epoch validation table (energy RMSE per structure, force RMSE and MAE).

    python -I scripts/eval/select_size.py <run> <run> ... [--controls <run> ...]
"""
import json
import re
import sys
from pathlib import Path


def params(name):
    t = ''
    for f in (f'runs/{name}_cue/stdout.txt', f'runs/{name}_cue/loss_{name}_cue.out'):
        if Path(f).exists():
            t += Path(f).read_text(errors='ignore')
    m = re.search(r'Model parameters: ([\d,]+)', t)
    return int(m.group(1).replace(',', '')) if m else 0


def last5(name):
    rows = {}
    for l in open(f'runs/{name}_cue/loss_{name}_cue.out', errors='ignore'):
        if re.match(r'^\d\d/\d\d/\d{4} ', l):
            p = [x.split() for x in l.split('|')]
            rows[int(p[0][2])] = (float(p[2][1]), float(p[2][2]), float(p[2][4]))
    eps = sorted(rows)[-5:]
    return [sum(rows[e][k] for e in eps) / len(eps) for k in range(3)]


args = sys.argv[1:]
controls = args[args.index('--controls') + 1:] if '--controls' in args else []
names = args[:args.index('--controls')] if '--controls' in args else args
rows = []
for n in names + controls:
    p = Path(f'runs/{n}_cue/loss_{n}_cue.summary.json')
    if not p.exists():
        print(f'{n}: no summary yet'); continue
    v = json.load(open(p))['final_ema']['valid']
    rows.append(dict(n=n, E=1e3 * v['energy_rmse'], F=1e3 * v['force_rmse'], Emae=1e3 * v['energy_mae'],
                     Fmae=1e3 * v['force_mae'], P=params(n), ctrl=n in controls, l5=last5(n)))
cand_all = [r for r in rows if not r['ctrl']]
best_e = min(r['E'] for r in cand_all)
cand = [r for r in cand_all if r['E'] <= 1.10 * best_e]
best_f = min(r['F'] for r in cand)
tied = [r for r in cand if r['F'] <= 1.02 * best_f]
pick = min(tied, key=lambda r: r['P'])
print('final-epoch EMA, validation: E per atom (meV), F (meV/Å) | last-5-epoch means of the per-epoch table: '
      'E RMSE per structure (meV), F RMSE, F MAE')
print(f"{'run':40s} {'E RMSE':>7s} {'E MAE':>6s} {'F RMSE':>7s} {'F MAE':>6s} {'params':>11s} | {'E5':>6s} {'F5':>6s} {'Fmae5':>6s}")
for r in sorted(rows, key=lambda r: (r['ctrl'], r['F'])):
    tag = ' <- selected' if r is pick else (' (control)' if r['ctrl'] else (' candidate' if r in cand else ''))
    print(f"{r['n']:40s} {r['E']:7.2f} {r['Emae']:6.2f} {r['F']:7.2f} {r['Fmae']:6.2f} {r['P']:11,d} | "
          f"{r['l5'][0]:6.0f} {r['l5'][1]:6.1f} {r['l5'][2]:6.1f}{tag}")
