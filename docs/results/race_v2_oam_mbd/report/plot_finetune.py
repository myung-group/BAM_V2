"""OAM fine-tuning: validation metrics per virtual epoch and the WBM 3k-sample trajectory (matplotlib, login node).
usage: python -I plot_finetune.py <out.png> [wbm_points.json]"""
import json, re, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W = '/scratch/t510a13/race_v2/work'
R = 'oam_v1set_sr32_soap_ft'
INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
C1, C2, CR = '#2a78d6', '#eb6834', '#898781'
L = open(f'{W}/runs/{R}_cue/loss_{R}_cue.out', errors='replace').read().splitlines()
ep, emae, fmae, lr = [], [], [], []
for i, l in enumerate(L):
    m = re.match(r'^\d+/\d+/\d+ +[\d:]+ +(\d+) ', l)
    if m:
        v = l.split('|')[2].split(); t = L[i + 1] if i + 1 < len(L) else ''
        em = re.search(r'E_MAE ([\d.]+)', t)
        e = int(m.group(1))
        if ep and e <= ep[-1]:                       # a restart repeats no epoch line, but keep the last value if it did
            k = ep.index(e); del ep[k:], emae[k:], fmae[k:], lr[k:]
        ep.append(e); emae.append(float(em.group(1))); fmae.append(float(v[4])); lr.append(float(l.split('|')[3].split()[0]))
# 3k WBM tracking sample (unique prototypes, float32 inference): virtual epoch -> (MAE eV/atom, F1)
P = {4: (0.0245, 0.839), 10: (0.0239, 0.840), 12: (0.0207, 0.879), 13: (0.0203, 0.876), 16: (0.0198, 0.887),
     18: (0.0199, 0.893), 24: (0.0197, 0.886), 30: (0.0197, 0.889)}
if len(sys.argv) > 2:
    P.update({int(k): tuple(v) for k, v in json.load(open(sys.argv[2])).items()})
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.titlecolor': INK, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
                     'figure.facecolor': SURF, 'axes.facecolor': SURF})
fig, axs = plt.subplots(2, 2, figsize=(12, 7.2))
dec = next((e for e, x in zip(ep, lr) if x < 0.99 * max(lr) and e > 2), None)
pe = sorted(P)
series = [(axs[0, 0], ep, emae, 'Validation energy MAE (meV/atom), MPtrj val', C1, 13.7, '{:.1f}'),
          (axs[0, 1], ep, fmae, 'Validation force MAE (meV/Å), MPtrj val', C1, 42.6, '{:.1f}'),
          (axs[1, 0], pe, [P[e][0] * 1000 for e in pe], 'WBM 3k sample: MAE (meV/atom)', C2, 38.9, '{:.1f}'),
          (axs[1, 1], pe, [P[e][1] for e in pe], 'WBM 3k sample: F1', C2, 0.775, '{:.3f}')]
for ax, x, y, title, c, ref, fmt in series:
    ax.axvspan(0.5, 10.5, color='#f3efe6', zorder=0)
    if dec:
        ax.axvspan(dec - 0.5, max(ep) + 0.5, color='#eef3fb', zorder=0)
    ax.plot(x, y, color=c, lw=2, marker='o', ms=4.5, mec=SURF, mew=1.0, zorder=3)
    ax.axhline(ref, color=CR, lw=1.3, ls='--', zorder=2)
    ax.annotate(f'MPtrj-only final {fmt.format(ref)}', (max(ep), ref), xytext=(0, 4), textcoords='offset points',
                ha='right', fontsize=8.5, color=INK2)
    ax.annotate(fmt.format(y[-1]), (x[-1], y[-1]), xytext=(6, 0), textcoords='offset points', va='center', fontsize=9, color=INK2)
    ax.set_title(title, loc='left'); ax.set_xlabel('virtual epoch (5000 steps)'); ax.set_xlim(0.5, max(ep) + 2.5)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
fig.text(0.01, 0.005, f'Beige: epochs 1-10 with the clustered MPtrj copies (before the data-layout fix).  Blue: linear LR decay '
         f'(from epoch {dec}).  Dashed: MPtrj-only model.', color=MUTED, fontsize=8.5)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(sys.argv[1], dpi=130, facecolor=SURF)
print(sys.argv[1], 'epochs', ep[-1], 'wbm points', pe)
