"""OMat24 pretraining: validation metrics per virtual epoch (5000 steps) with the LR phases (login node, matplotlib)."""
import re, sys, datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W = '/scratch/t510a13/race_v2/work'
R = sys.argv[2] if len(sys.argv) > 2 else 'omat24_v1set_sr32_soap_pre'
INK, INK2, MUTED, GRID, AXIS, SURF, C = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb', '#2a78d6'
L = open(f'{W}/runs/{R}_cue/loss_{R}_cue.out', errors='replace').read().splitlines()
ep, d = [], {k: [] for k in ('emae', 'fmae', 'frmse', 'srmse', 'lr')}
for i, l in enumerate(L):
    m = re.match(r'^\d+/\d+/\d+ +[\d:]+ +(\d+) ', l)
    if m:
        v = l.split('|')[2].split(); t = L[i + 1] if i + 1 < len(L) else ''
        em = re.search(r'E_MAE ([\d.]+)', t); sr = re.search(r'S_RMSE ([\d.]+)', t)
        ep.append(int(m.group(1))); d['emae'].append(float(em.group(1))); d['fmae'].append(float(v[4]))
        d['frmse'].append(float(v[2])); d['srmse'].append(float(sr.group(1))); d['lr'].append(float(l.split('|')[3].split()[0]))
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.titlecolor': INK, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
                     'figure.facecolor': SURF, 'axes.facecolor': SURF})
fig, axs = plt.subplots(2, 2, figsize=(12, 7.6))
P = [('emae', 'Energy MAE (meV/atom)', '{:.1f}'), ('fmae', 'Force MAE (meV/Å)', '{:.1f}'),
     ('frmse', 'Force RMSE (meV/Å)', '{:.1f}'), ('srmse', 'Stress RMSE (GPa)', '{:.3f}')]
dec = next((e for e, lr in zip(ep, d['lr']) if lr < 0.0099 and e > 5), None)
for ax, (k, title, fmt) in zip(axs.flat, P):
    y = d[k]; s = 2 if len(ep) > 3 else 0          # skip the warm-up epoch(s) so the scale shows the late changes
    if dec:
        ax.axvspan(dec - 0.5, max(ep) + 0.5, color='#eef3fb', zorder=0)
    ax.plot(ep[s:], y[s:], color=C, lw=2, marker='o', ms=4, mec=SURF, mew=1.0, zorder=3)
    ax.annotate(fmt.format(y[-1]), (ep[-1], y[-1]), xytext=(6, 0), textcoords='offset points', va='center', fontsize=9, color=INK2)
    for x, lab in ((30, '1 OMat epoch'),):
        if max(ep) >= x:
            ax.axvline(x, color=AXIS, lw=1, ls=':', zorder=1)
    ax.set_title(title, loc='left'); ax.set_xlabel('virtual epoch (5000 steps)')
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
fig.suptitle('RACEv2 OMat24 pretraining (7 Å, 17.8M), OMat24 validation (20k)', x=0.01, ha='left', fontsize=12, fontweight='bold', color=INK)
fig.text(0.01, 0.005, f'Shaded: linear LR decay 0.01 → 0.0001 (from epoch {dec}). Dotted: about one pass over omat24_train (100M). '
         f'Epochs 1-2 (warm-up) left out.  {datetime.datetime.now():%Y-%m-%d %H:%M}', color=MUTED, fontsize=8.5)
fig.tight_layout(rect=(0, 0.03, 1, 0.96))
fig.savefig(sys.argv[1], dpi=130, facecolor=SURF)
print(sys.argv[1], 'epochs', ep[-1])
