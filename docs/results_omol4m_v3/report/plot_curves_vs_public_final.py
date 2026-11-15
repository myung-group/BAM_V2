"""Curves of the physics benchmarks and the ion ... H2O scans: DFT, RACE v3 / e249 and the public fairchem models.
    python plot_curves_vs_public.py <out_bench.png> <out_iw.png>"""
import json
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MD = '/scratch/t510a13/md_water'
KC = 23.0605
MODELS = [  # label, bench_report.txt, ion_water.json, colour, style
    ('RACE v2 LR-POLAR', f'{MD}/eval/v3_final/bench_report.txt', f'{MD}/eval/v3_final/iw/ion_water.json', '#2a78d6', '-'),
    ('earlier run (v2local e90)', f'{MD}/eval/v2local_e90/bench_report.txt', f'{MD}/eval/v2local_e90/iw/ion_water.json',
     '#eb6834', '-'),
    ('eSEN-md', f'{MD}/fc/esen_md/bench_report.txt', f'{MD}/fc/esen_md/ion_water.json', '#1baf7a', '-'),
    ('UMA-s-1.1', f'{MD}/fc/uma_s_1p1/bench_report.txt', f'{MD}/fc/uma_s_1p1/ion_water.json', '#e87ba4', '--'),
    ('UMA-s-1.2.1', f'{MD}/fc/uma_s_1p2p1/bench_report.txt', f'{MD}/fc/uma_s_1p2p1/ion_water.json', '#6250d6', '-'),
]
INK, MUT, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': INK, 'xtick.color': MUT,
                     'ytick.color': MUT, 'axes.spines.top': False, 'axes.spines.right': False})


def tables(path):
    """{title: (x, dft, model)} from a bench_report.txt (first three numbers of every row)"""
    out, title, rows = {}, None, []
    try:
        lines = open(path).read().splitlines()
    except FileNotFoundError:
        return out
    for ln in lines + ['== end']:
        if ln.startswith('== '):
            if title and rows:
                a = np.array(rows)
                out[title] = (a[:, 0], a[:, 1], a[:, 2])
            title, rows = ln[3:].split('(')[0].strip(), []
            continue
        tok = ln.split()
        if title and len(tok) >= 3 and re.match(r'^\s+[-+]?\d', ln):
            try:                                   # x, DFT, model ('nan' when the model failed)
                rows.append([float(x) for x in tok[:3]])
            except ValueError:
                pass
    return out


T = {lab: tables(b) for lab, b, *_ in MODELS}
PANELS = [('salt_bridge: acetate...methylammonium', 'salt bridge  acetate...methylammonium', 'COO...N distance (A)'),
          ('ionic_diss: NaCl', 'NaCl dissociation', 'Na-Cl (A)'),
          ('ionic_diss: LiF', 'LiF dissociation', 'Li-F (A)'),
          ('polarization: Li+...I-', 'Li+ ... I-', 'distance (A)'),
          ('polarization: Li+...Xe', 'Li+ ... Xe', 'distance (A)'),
          ('dispersion: Ar2', 'Ar2', 'Ar-Ar (A)'),
          ('dispersion: CH4 dimer', 'CH4 dimer', 'C-C (A)'),
          ('dispersion: benzene sandwich', 'benzene sandwich', 'ring-ring (A)'),
          ('halogen_bond: CF3Br...NH3 distance', 'halogen bond  CF3Br...NH3', 'Br-N (A)')]
fig, axs = plt.subplots(3, 3, figsize=(16, 13))
for ax, (key, ttl, xl) in zip(axs.ravel(), PANELS):
    dft = None
    lo, hi = np.inf, -np.inf
    for lab, _, _, c, ls in MODELS:
        if key not in T[lab]:
            continue
        x, d, m = T[lab][key]
        dft = (x, d)
        ax.plot(x, m, ls, color=c, lw=2, label=lab)
    if dft is not None:
        ax.plot(dft[0], dft[1], 'o', color=INK, ms=5, label='DFT', zorder=5)
        lo, hi = float(np.min(dft[1])), float(np.max(dft[1]))
        span = max(hi - lo, 1.0)
        ax.set_ylim(lo - 0.45 * span, hi + 0.45 * span)
    ax.set_title(ttl, loc='left', color=INK)
    ax.set_xlabel(xl)
    ax.set_ylabel('E_int (kcal/mol)')
    ax.grid(color=GRID)
    ax.set_axisbelow(True)
axs[0, 0].legend(frameon=False, fontsize=9)
fig.suptitle('Physics benchmarks: DFT vs RACE and public OMol25 models (y range = DFT range +- 45 %)', x=0.01,
             ha='left', fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(sys.argv[1], dpi=110, facecolor='#fcfcfb')

# ---- ion ... H2O
ref = json.load(open(f'{MD}/ion_water_ref/ion_water_ref.json'))['scans']
IW = {lab: json.load(open(p))['scans'] for lab, _, p, *_ in MODELS if p}
ions = ['Li+_c2v', 'Na+_c2v', 'K+_c2v', 'F-_c2v', 'F-_hb', 'Cl-_c2v', 'Cl-_hb']
fig, axs = plt.subplots(2, 4, figsize=(18, 9))
for ax, ion in zip(axs.ravel(), ions):
    rd = np.array([r['R'] for r in ref[ion]['rows']])
    ed = np.array([r['E_int_eV'] for r in ref[ion]['rows']]) * KC
    for lab, _, p, c, ls in MODELS:
        if lab not in IW or ion not in IW[lab]:
            continue
        rows = IW[lab][ion]['rows']
        ax.plot([r['R'] for r in rows], [r['E_int_eV'] * KC for r in rows], ls, color=c, lw=2, label=lab)
    ax.plot(rd, ed, 'o', color=INK, ms=4.5, label='DFT (wB97M-V/def2-TZVPD)', zorder=5)
    emin = float(ed.min())
    ax.set_ylim(emin * 1.35, max(4.0, -0.25 * emin))
    ax.set_xlim(1.5, 8.0)
    ax.axhline(0, color='#9a9893', lw=0.7)
    ax.set_title(ion.replace('_c2v', ' (c2v)').replace('_hb', ' (H-bond)') + ' ... H2O', loc='left', color=INK)
    ax.set_xlabel('ion-O distance (A)')
    ax.set_ylabel('E_int (kcal/mol)')
    ax.grid(color=GRID)
    ax.set_axisbelow(True)
# last panel: MAE 1.9-10 A per model and ion
ax = axs.ravel()[-1]
labs = [lab for lab, *_ in MODELS if lab in IW]
w = 0.8 / len(labs)
for k, lab in enumerate(labs):
    v = []
    for ion in ions:
        rd = np.array([r['R'] for r in ref[ion]['rows']])
        ed = np.array([r['E_int_eV'] for r in ref[ion]['rows']]) * KC
        rows = IW[lab][ion]['rows']
        rm = np.array([r['R'] for r in rows])
        em = np.array([r['E_int_eV'] for r in rows]) * KC
        sel = (rd >= 1.9) & (rd <= 10.0)
        v.append(np.mean(np.abs(np.interp(rd[sel], rm, em) - ed[sel])))
    c = [m[3] for m in MODELS if m[0] == lab][0]
    ax.bar(np.arange(len(ions)) + (k - (len(labs) - 1) / 2) * w, v, w * 0.92, color=c, label=lab,
           edgecolor='white', linewidth=0.6)
ax.set_xticks(np.arange(len(ions)))
ax.set_xticklabels([i.replace('_c2v', '').replace('_hb', ' hb') for i in ions], rotation=30)
ax.set_ylabel('MAE 1.9-10 A (kcal/mol)')
ax.set_title('MAE vs DFT', loc='left', color=INK)
ax.grid(axis='y', color=GRID)
ax.set_axisbelow(True)
axs[0, 0].legend(frameon=False, fontsize=9, loc='lower right')
fig.suptitle('Ion ... H2O interaction curves: DFT vs RACE v2 LR-POLAR and public OMol25 models', x=0.01, ha='left',
             fontsize=13, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(sys.argv[2], dpi=110, facecolor='#fcfcfb')
print('wrote', sys.argv[1:])
