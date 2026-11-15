"""Validation metrics in physical units of the 5 A runs and the 7 A V1-setting runs (login node, matplotlib)."""
import re, sys, datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

W = '/scratch/t510a13/race_v2/work'
RUNS = [('5 Å baseline, 45.9M, SOAP 0.01', 'L5x128_hub001_s42', '#2a78d6', '-'),
        ('5 Å reduced K=32, 17.8M, SOAP 0.01', 'L5x128_hub001_sr32_s42', '#eb6834', '-'),
        ('5 Å 64 ch + K=32, 4.7M, SOAP 0.01', 'L5x64_hub001_sr32_s42', '#1baf7a', '-'),
        ('7 Å V1 settings, K=32, SOAP 0.01 cosine', 'v1set_sr32_soap_s42', '#eda100', '-'),
        ('7 Å V1 settings, K=32, Adam 1e-3 plateau (stopped)', 'v1set_sr32_adam_s42', '#e87ba4', '-'),
        ('7 Å V1 settings, K=32, SOAP 0.01 cosine, E:F:S 4:1:1', 'v1set_sr32_soap_ew4_s42', '#7b5cd6', '-'),
        ('7 Å SOAP 1:1:1, linear LR cool-down from epoch 20', 'v1set_sr32_soap_cd20_s42', '#8a5a00', '--')]
INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'

def read(n):
    L = open(f'{W}/runs/mptrj_{n}_cue/loss_mptrj_{n}_cue.out', errors='replace').read().splitlines()
    d = {}
    for i, l in enumerate(L):
        m = re.match(r'^\d+/\d+/\d+ +[\d:]+ +(\d+) ', l)
        if m:
            v = l.split('|')[2].split(); t = L[i + 1] if i + 1 < len(L) else ''
            em = re.search(r'E_MAE ([\d.]+)', t); sr = re.search(r'S_RMSE ([\d.]+)', t)
            d[int(m.group(1))] = dict(emae=float(em.group(1)) if em else None, fmae=float(v[4]), frmse=float(v[2]),
                                      srmse=float(sr.group(1)) if sr else None)
    return d

data = [(lab, read(n), c, ls) for lab, n, c, ls in RUNS]
XMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 21
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.titlecolor': INK, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
                     'figure.facecolor': SURF, 'axes.facecolor': SURF})
fig, axs = plt.subplots(2, 2, figsize=(13, 8.6))
for ax, key, title, top in ((axs[0, 0], 'emae', 'Energy MAE (meV/atom)', 95), (axs[0, 1], 'fmae', 'Force MAE (meV/Å)', 120),
                            (axs[1, 0], 'frmse', 'Force RMSE (meV/Å)', 300), (axs[1, 1], 'srmse', 'Stress RMSE (GPa)', 2.4)):
    for lab, d, c, ls in data:
        ep = [e for e in sorted(d) if e <= XMAX and d[e][key] is not None]
        if not ep:
            continue
        y = [d[e][key] for e in ep]
        ax.plot(ep, y, color=c, lw=2, ls=ls, marker='o', ms=4.5 if len(ep) > 1 else 7, mec=SURF, mew=1.2, label=lab, zorder=3)
        if lab.startswith('7'):
            ax.annotate(f'{y[-1]:.1f}' if key != 'srmse' else f'{y[-1]:.2f}', (ep[-1], y[-1]), xytext=(6, 0),
                        textcoords='offset points', va='center', fontsize=9, color=INK2)
    ax.set_title(title, loc='left'); ax.set_xlabel('epoch (one pass over the training split)')
    ax.set_xlim(0.5, XMAX + 0.5); ax.set_ylim(top=top)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.995))
stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
fig.text(0.01, 0.005, 'Same MPtrj validation split (20k structures, EMA weights). Not like-for-like: the 7 Å runs differ in cutoff, radial basis,\n'
         f'loss (Huber 0.05/0.25/0.05, weights 1/1/1 vs 0.01, 8/20/2), EMA (0.999 vs 0.99) and batch (~215 vs ~122 structures per step).  {stamp}',
         color=MUTED, fontsize=8.5)
fig.tight_layout(rect=(0, 0.035, 1, 0.93))
fig.savefig(sys.argv[1], dpi=130, facecolor=SURF)
print(sys.argv[1], {lab: sorted(d)[-3:] for lab, d, _, _ in data})
