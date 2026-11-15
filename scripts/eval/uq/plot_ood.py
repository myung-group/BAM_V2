"""plot_ood.py <ood result dir (lam_xxx)> <out prefix> [label]: figures of the Laplace uncertainty study (matplotlib, login node).
  <prefix>_detection.png   score distributions, ROC curves, sigma versus predicted energy (cf. Fig. S3 of the RACE UQ paper)
  <prefix>_error.png       predicted sigma versus actual error with Gaussian quantile bands (cf. Fig. 2), calibration
                           curves (cf. Fig. 3), WBM error by uncertainty decile"""
import sys
from math import erf, sqrt
from statistics import NormalDist
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D, OUT = sys.argv[1], sys.argv[2]
LABEL = sys.argv[3] if len(sys.argv) > 3 else 'RACE v2 OAM'
INK, INK2, MUTED, GRID, AXIS, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
SETS = [('mptrj_val', 'MPtrj val (ID)', '#2a78d6', '-', 'o'), ('omol25', 'OMol25 (OOD)', '#eb6834', '--', '^'),
        ('salex_val', 'sAlex val (ID)', '#1baf7a', '-', 's'), ('opoly26', 'OPoly26 (OOD)', '#eda100', '--', 'v'),
        ('oc25', 'OC25 (OOD)', '#e87ba4', '--', 'D'), ('wbm', 'WBM (test)', '#008300', ':', 'P')]
Z = {k: dict(np.load(f'{D}/{k}.npz', allow_pickle=True)) for k, *_ in SETS}
ID, OOD = ('mptrj_val', 'salex_val'), ('omol25', 'opoly26', 'oc25')
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.titlecolor': INK, 'axes.titlesize': 11, 'axes.titleweight': 'bold',
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'legend.frameon': False, 'legend.fontsize': 8.5})


def fin(x):
    x = np.asarray(x, float); return x[np.isfinite(x)]


def auroc(a, b):                         # P(score of b > score of a), ties count half
    a, b = fin(a), fin(b); s = np.concatenate([a, b]); o = np.argsort(s, kind='stable'); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    for v in np.unique(s[np.r_[False, np.diff(np.sort(s)) == 0]]):
        m = s == v; r[m] = r[m].mean()
    return (r[len(a):].sum() - len(b) * (len(b) + 1) / 2) / (len(a) * len(b))


def roc(a, b):
    a, b = fin(a), fin(b); t = np.sort(np.unique(np.concatenate([a, b])))[::-1]
    return np.r_[0, [(a >= x).mean() for x in t], 1], np.r_[0, [(b >= x).mean() for x in t], 1]


def style(ax, title, xl, yl):
    ax.set_title(title, loc='left'); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)


pool = lambda key, names: np.concatenate([Z[n][key] for n in names])
# ---------------------------------------------------------------- figure 1: detection
fig, axs = plt.subplots(1, 3, figsize=(16, 5.0))
ax = axs[0]; FLOOR = 3e-4
bins = np.logspace(np.log10(FLOOR), np.log10(max(fin(Z[k]['sigFmax']).max() for k in Z) * 1.1), 60)
for k, lab, c, ls, mk in SETS:
    v = np.clip(fin(Z[k]['sigFmax']), FLOOR, None)
    ax.hist(v, bins=bins, weights=np.full(len(v), 1 / len(v)), histtype='step', lw=2, color=c, ls=ls, label=lab)
ax.set_xscale('log'); ax.legend(loc='upper right')
style(ax, 'a  Distribution of the OOD score', r'max-atom force uncertainty $\sigma_{F,\max}$ (eV/Å)', 'fraction of structures')
ax = axs[1]; ax.plot([0, 1], [0, 1], color=AXIS, lw=1.2, ls='--')
for k, lab, c, ls, mk in SETS:
    if k in ID:
        continue
    x, y = roc(pool('sigFmax', ID), Z[k]['sigFmax'])
    ax.plot(x, y, color=c, lw=2, ls='-' if k in OOD else ':', label=f"{lab.split(' (')[0]} vs ID   AUROC {auroc(pool('sigFmax', ID), Z[k]['sigFmax']):.3f}")
ax.legend(loc='lower right'); ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
style(ax, 'b  ROC curves (ID = MPtrj val + sAlex val)', 'false positive rate (ID flagged as OOD)', 'true positive rate (OOD flagged)')
ax = axs[2]
for k, lab, c, ls, mk in SETS:
    if k == 'wbm':
        continue
    ax.scatter(Z[k]['e_mean'], np.clip(Z[k]['sigFmax'], FLOOR, None), s=9, color=c, alpha=0.55, lw=0, marker=mk, label=lab, rasterized=True)
ax.set_yscale('log'); ax.legend(loc='upper right', markerscale=2)
style(ax, r'c  $\sigma_{F,\max}$ versus predicted energy', 'predicted energy relative to E0 (eV/atom)', r'$\sigma_{F,\max}$ (eV/Å)')
fig.text(0.01, 0.01, f'{LABEL}, SOAP-eigenbasis Laplace (GGN, 16 weight samples).  1,000 structures per set, WBM: 3k tracking sample.  '
         f'Values below {FLOOR:g} are drawn at {FLOOR:g}.  Solid: in-distribution, dashed: out-of-distribution, dotted: WBM.', color=MUTED, fontsize=8.5)
fig.tight_layout(rect=(0, 0.035, 1, 1)); fig.savefig(f'{OUT}_detection.png', dpi=130, facecolor=SURF); plt.close(fig)

# ---------------------------------------------------------------- figure 2: uncertainty versus error
nd = NormalDist(); hq = lambda p: nd.inv_cdf((1 + p) / 2)            # quantile of |N(0,1)|
BANDS = [((0.15, 0.85), '#6250d6', '[15-85 %]'), ((0.05, 0.95), '#e34948', '[5-95 %]'), ((0.005, 0.995), '#eda100', '[0.5-99.5 %]')]
fig, axs = plt.subplots(2, 2, figsize=(13, 10.6))
ax = axs[0, 0]; lim = (2e-4, 2.0); xx = np.array(lim)
for k, lab, c, ls, mk in SETS:
    if k == 'wbm':
        continue
    m = np.isfinite(Z[k]['f_mae']) & (Z[k]['sigF'] > 0) & (Z[k]['f_mae'] > 0)
    ax.scatter(Z[k]['sigF'][m], Z[k]['f_mae'][m], s=9, color=c, alpha=0.5, lw=0, marker=mk, label=lab, rasterized=True)
ax.plot(xx, xx, color=INK, lw=1.2, ls='--'); ax.plot(xx, 10 * xx, color=MUTED, lw=1, ls=':'); ax.plot(xx, 100 * xx, color=MUTED, lw=1, ls=':')
for f, t in ((1, 'error = σ'), (10, '10 σ'), (100, '100 σ')):
    ax.annotate(t, (lim[1] / f * 0.55, lim[1] * 0.55), fontsize=8, color=INK2, ha='right', va='bottom', rotation=38)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(*lim); ax.set_ylim(2e-4, 2.0); ax.legend(loc='lower right', markerscale=2)
style(ax, 'a  Force: predicted uncertainty versus error', r'$\sigma_F$, mean over atoms (eV/Å)', 'force MAE of the structure (eV/Å)')
ax = axs[0, 1]; lim = (1e-4, 1.0); xx = np.array(lim)
E = {'mptrj_val': Z['mptrj_val']['e_abs_err'], 'salex_val': Z['salex_val']['e_abs_err'], 'wbm': np.abs(Z['wbm']['e_form_err'])}
for k, lab, c, ls, mk in SETS:
    if k in E:
        m = np.isfinite(E[k]) & (E[k] > 0) & (Z[k]['sigE'] > 0)
        ax.scatter(Z[k]['sigE'][m], E[k][m], s=9, color=c, alpha=0.45, lw=0, marker=mk, label=lab, rasterized=True)
ax.plot(xx, xx, color=INK, lw=1.2, ls='--', label='error = σ')
for (lo, hi), c, lab in BANDS:
    ax.plot(xx, hq(lo) * xx, color=c, lw=1.1); ax.plot(xx, hq(hi) * xx, color=c, lw=1.1, label=lab)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(*lim); ax.set_ylim(1e-6, 2.0); ax.legend(loc='lower right', markerscale=2, ncol=2)
style(ax, 'b  Energy: predicted uncertainty versus error', r'$\sigma_E$ (eV/atom)', '|energy error| (eV/atom)')
ax = axs[1, 0]; ax.plot([0, 1], [0, 1], color=INK, lw=1.2, ls='--', label='perfect calibration'); p = np.linspace(0, 1, 41)
m0 = np.isfinite(E['mptrj_val']) & (Z['mptrj_val']['sigE'] > 0)
scale = float(np.median(E['mptrj_val'][m0] / Z['mptrj_val']['sigE'][m0])) / hq(0.5)        # one factor from MPtrj val: half of the errors inside the 50 % interval
for k, lab, c, ls, mk in SETS:
    if k in E:
        m = np.isfinite(E[k]) & (Z[k]['sigE'] > 0); z = E[k][m] / Z[k]['sigE'][m]
        for s, st, sfx in ((1.0, '-', 'as predicted'), (scale, '--', f'σ × {scale:.0f}')):
            need = np.array([erf(v / s / sqrt(2)) for v in z])
            ax.plot(p, [(need <= q).mean() for q in p], color=c, lw=2, ls=st, marker=mk if s == 1.0 else None, ms=4, mec=SURF, mew=0.6,
                    label=f"{lab.split(' (')[0]}, {sfx}")
ax.legend(loc='upper left', ncol=1); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
style(ax, 'c  Energy calibration', 'expected confidence level', 'observed confidence level')
ax = axs[1, 1]; w = Z['wbm']; m = np.isfinite(w['e_form_err']) & np.isfinite(w['sigE']); err = np.abs(w['e_form_err'][m]); sg = w['sigE'][m]
edges = np.quantile(sg, np.linspace(0, 1, 11)); idx = np.clip(np.searchsorted(edges, sg, side='right') - 1, 0, 9)
mean = [err[idx == i].mean() * 1000 for i in range(10)]; med = [np.median(err[idx == i]) * 1000 for i in range(10)]
frac = [(err[idx == i] > 0.1).mean() * 100 for i in range(10)]
ax.plot(range(1, 11), mean, color='#2a78d6', lw=2, marker='o', ms=6, mec=SURF, mew=1, label='mean |error|')
ax.plot(range(1, 11), med, color='#eb6834', lw=2, marker='s', ms=6, mec=SURF, mew=1, label='median |error|')
ax.axhline(err.mean() * 1000, color=MUTED, lw=1.2, ls='--'); ax.annotate(f'all structures: {err.mean() * 1000:.1f}', (10, err.mean() * 1000),
                                                                      xytext=(0, 4), textcoords='offset points', ha='right', fontsize=8.5, color=INK2)
for i in (0, 9):
    ax.annotate(f'{mean[i]:.1f}', (i + 1, mean[i]), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9, color=INK2)
ax.set_xticks(range(1, 11)); ax.set_ylim(0, None); ax.legend(loc='upper left')
style(ax, r'd  WBM: error by $\sigma_E$ decile', r'decile of $\sigma_E$ (1 = most certain, 10 = least certain)', '|formation energy error| (meV/atom)')
fig.text(0.01, 0.008, f'{LABEL}.  Lines in b: quantiles of |N(0, σ²)|; a calibrated model has 70 % / 90 % / 99 % of the points inside the bands.\n'
         'c: dashed = σ multiplied by one factor fitted on MPtrj val (median).  OMol25 / OPoly26 energies use another DFT level and energy zero, so they are left out of b and c.', color=MUTED, fontsize=8.5)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f'{OUT}_error.png', dpi=130, facecolor=SURF); plt.close(fig)

# ---------------------------------------------------------------- numbers
print('scale fitted on MPtrj val: %.1f' % scale)
for sc, fn in (('sigFmax', lambda z: z['sigFmax']), ('sigF', lambda z: z['sigF']), ('sigE', lambda z: z['sigE']),
               ('sigE*sqrt(N)', lambda z: z['sigE'] * np.sqrt(z['n_atoms'])), ('sigE*N', lambda z: z['sigE'] * z['n_atoms'])):
    a = np.concatenate([fn(Z[n]) for n in ID])
    print(f'{sc:14s} ' + '  '.join(f"{k} {auroc(a, fn(Z[k])):.3f}" for k in OOD + ('wbm',)))
for k in E:
    m = np.isfinite(E[k]) & (Z[k]['sigE'] > 0); z = E[k][m] / Z[k]['sigE'][m]
    print(f'{k}: fraction with |err| < 1 sigma {np.mean(z < 1):.3f} (ideal 0.683), after scaling {np.mean(z < scale):.3f}; < 2 sigma scaled {np.mean(z < 2 * scale):.3f} (ideal 0.954)')
print('wbm decile mean |err| meV:', np.round(mean, 1), ' frac |err|>0.1 eV (%):', np.round(frac, 1))
print(f'{OUT}_detection.png', f'{OUT}_error.png')
