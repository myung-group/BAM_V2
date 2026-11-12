"""Rank of a RACEv2 result among the Matbench Discovery leaderboard snapshot (unique prototypes).
usage: python rank_mbd.py F1=0.624 MAE=0.057 R2=0.765 [DAF=3.54 Prec=0.541 kSRME=1.07 RMSD=0.09] [--check]
CPS = 0.5 F1 + 0.4 (1 - kSRME/2) + 0.1 (1 - RMSD/0.15), computed when F1, kSRME and RMSD are all given."""
import sys, glob, os
import pandas as pd
tsv = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mbd_leaderboard_*.tsv')))[-1]
d = pd.read_csv(tsv, sep='\t', comment='#')
cps = lambda f1, k, r: 0.5 * f1 + 0.4 * (1 - min(k, 2) / 2) + 0.1 * (1 - min(r, 0.15) / 0.15)
if '--check' in sys.argv:                       # the snapshot was typed from the web page: is its arithmetic consistent?
    c = d.dropna(subset=['CPS']); dev = (c.apply(lambda r: cps(r.F1, r.kSRME, r.RMSD), axis=1) - c.CPS).abs()
    print(os.path.basename(tsv), len(d), 'models,', int(d.mptrj_only.sum()), 'MPtrj-only; max |CPS recomputed - CPS| =', round(dev.max(), 4),
          'at', c.model[dev.idxmax()])
me = {k: float(v) for k, v in (a.split('=') for a in sys.argv[1:] if '=' in a)}
if all(k in me for k in ('F1', 'kSRME', 'RMSD')):
    me['CPS'] = round(cps(me['F1'], me['kSRME'], me['RMSD']), 3)
lower_better = {'MAE', 'kSRME', 'RMSD'}
for k, v in me.items():
    for name, t in (('all', d), ('MPtrj-only', d[d.mptrj_only == 1])):
        x = t[k].dropna(); better = (x < v).sum() if k in lower_better else (x > v).sum()
        print(f'{k} {v}: {name} rank {better + 1} of {len(x) + 1}', end='   ')
    print()
