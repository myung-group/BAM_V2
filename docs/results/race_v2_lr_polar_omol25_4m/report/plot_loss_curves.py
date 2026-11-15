"""v3 per-epoch curves: train/val loss, train loss terms, val E MAE, val F MAE (+ neutral_val tests)."""
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG, OUT = sys.argv[1], sys.argv[2]
rows, tests = [], []
for ln in open(LOG):
    if re.match(r'^\d\d/\d\d/\d{4}', ln):
        t = ln.split()
        rows.append([float(t[2])] + [float(t[i]) for i in (5, 6, 7, 8, 11, 12, 13, 14, 15, 17)])
    m = re.search(r'Test\[neutral_val\] \(EMA, epoch (\d+)\) \| E_MAE:\s+([\d.]+) meV.*F_MAE:\s+([\d.]+)', ln)
    if m:
        tests.append([float(x) for x in m.groups()])
a = np.array(rows); _, i = np.unique(a[:, 0], return_index=True); a = a[np.sort(i)]
t = np.array(tests).reshape(-1, 3)
ep = a[:, 0]
BLUE, ORANGE, GREEN, INK, MUT = '#2a78d6', '#eb6834', '#1f9d73', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': INK, 'xtick.color': MUT,
                     'ytick.color': MUT, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(2, 2, figsize=(13, 8.4))
b = ax[0, 0]
b.semilogy(ep, a[:, 1], color=BLUE, lw=2, label='train'); b.semilogy(ep, a[:, 5], color=ORANGE, lw=2, label='val (EMA)')
b.set_ylim(0.0015, 0.008); b.set_title('total loss', loc='left', color=INK); b.legend(frameon=False)
b.annotate(f'e{int(ep[-1])}: train {a[-1, 1]:.5f} / val {a[-1, 5]:.5f}', (ep[-1], a[-1, 5]), xytext=(-150, 90),
           textcoords='offset points', color=INK, fontsize=9)
b = ax[0, 1]
b.semilogy(ep, a[:, 3], color=BLUE, lw=2, label='force term'); b.semilogy(ep, a[:, 4], color=GREEN, lw=2, label='aux term (charges, spins, dipole)')
b.semilogy(ep, 10 * a[:, 2], color=ORANGE, lw=2, label='energy term (x weight 10)')
b.set_ylim(1.2e-4, 8e-3); b.set_title('train loss terms', loc='left', color=INK); b.legend(frameon=False, fontsize=9)
for b, k, tk, ttl, lim in [(ax[1, 0], 8, 1, 'val energy MAE (meV)', (50, 220)), (ax[1, 1], 9, 2, 'val force MAE (meV/A)', (19, 36))]:
    b.plot(ep, a[:, k], color=BLUE, lw=2, label='stream val (all domains)')
    b.plot(t[:, 0], t[:, tk], ls='none', marker='D', ms=8, color=ORANGE, mec='white', mew=1.5, label='neutral_val test')
    for e, v in zip(t[:, 0], t[:, tk]):
        b.annotate(f'{v:.1f}', (e, v), xytext=(0, -15), textcoords='offset points', ha='center', fontsize=8, color=MUT)
    b.annotate(f'{a[-1, k]:.1f}', (ep[-1], a[-1, k]), xytext=(4, 6), textcoords='offset points', fontsize=9, color=INK)
    b.set_ylim(*lim); b.set_title(ttl, loc='left', color=INK); b.legend(frameon=False, fontsize=9)
for b in ax.flat:
    b.set_xlabel('epoch'); b.grid(alpha=0.25, which='both'); b.set_xlim(0, 170)
fig.suptitle(f'RACE v2 LR-POLAR (OMol25 4M, 3 nodes): epochs 1-{int(ep[-1])} of 170', x=0.01, ha='left', color=INK, fontsize=12)
fig.tight_layout(); fig.savefig(OUT, dpi=140)
print(OUT)
