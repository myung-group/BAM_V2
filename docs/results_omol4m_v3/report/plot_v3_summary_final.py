"""v3 final-test summary: internal group errors, physics benchmarks, ion-water 0.05 A scans (v3 vs earlier versions)."""
import json
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EV = '/scratch/t510a13/md_water/eval'
OUT = sys.argv[1]
VERS = [('e249', 'e249 (old, 249 ep)', '#eda100'), ('v2local_e90', 'earlier run v2local e90', '#eb6834'),
        ('v3_e70', 'LR-POLAR e70', '#1baf7a'), ('v3_final', 'LR-POLAR final (e170)', '#2a78d6')]
INK, MUT, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#c9c8c2', 'axes.labelcolor': INK, 'xtick.color': MUT,
                     'ytick.color': MUT, 'axes.spines.top': False, 'axes.spines.right': False})


def groups(d):
    return json.load(open(f'{EV}/{d}/groups.json'))['groups']


def bench(d):
    out, h = {}, None
    for ln in open(f'{EV}/{d}/bench_report.txt'):
        if ln.startswith('== '):
            h = ln[3:].split('(')[0].strip()
        m = re.search(r'MAE ([\d.]+) kcal/mol over', ln)
        if m and h:
            out[h] = float(m.group(1))
    return out


fig = plt.figure(figsize=(17, 11))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.42, wspace=0.28)

# ---- internal groups: energy and force MAE per domain
doms = ['all', 'dom:biomolecules', 'dom:elytes', 'dom:metal_complexes', 'dom:reactivity', 's:open', 'q:q|2+|']
names = ['all', 'biomol.', 'elytes', 'metal cx', 'reactivity', 'open shell', '|q|>=2']
G = {d: groups(d) for d, _, _ in VERS}
for col, (key, ttl, unit) in enumerate([('E_MAE_meV', 'Internal groups: energy MAE', 'meV'),
                                         ('F_MAE_meV_A', 'Internal groups: force MAE', 'meV/A')]):
    ax = fig.add_subplot(gs[0, col])
    w = 0.8 / len(VERS)
    x = np.arange(len(doms))
    for k, (d, lab, c) in enumerate(VERS):
        v = [G[d][g][key] for g in doms]
        ax.bar(x + (k - 1.5) * w, v, w * 0.92, color=c, label=lab, edgecolor='white', linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel(unit)
    ax.set_title(ttl, loc='left', color=INK)
    ax.grid(axis='y', color=GRID)
    ax.set_axisbelow(True)
    if col == 0:
        ax.legend(frameon=False, fontsize=9)

# ---- physics benchmarks (MAE, log scale), without e249 (other bench set version)
ax = fig.add_subplot(gs[0, 2])
B = {d: bench(d) for d, _, _ in VERS[1:]}
items = [('salt_bridge: acetate...methylammonium', 'salt bridge'), ('ionic_diss: NaCl', 'NaCl diss.'),
         ('ionic_diss: LiF', 'LiF diss.'), ('polarization: Li+...I-', 'Li+...I-'),
         ('dispersion: benzene sandwich', 'benzene'), ('halogen_bond: CF3Br...NH3 distance', 'halogen bond'),
         ('dispersion: Ar2', 'Ar2'), ('polarization: Li+...Xe', 'Li+...Xe')]
y = np.arange(len(items))
w = 0.8 / 3
for k, (d, lab, c) in enumerate(VERS[1:]):
    v = [B[d].get(it, np.nan) for it, _ in items]
    ax.barh(y + (k - 1) * w, v, w * 0.92, color=c, label=lab, edgecolor='white', linewidth=0.6)
ax.set_yticks(y)
ax.set_yticklabels([n for _, n in items])
ax.invert_yaxis()
ax.set_xscale('log')
ax.set_xlabel('MAE vs DFT (kcal/mol, log)')
ax.set_title('Physics benchmarks', loc='left', color=INK)
ax.grid(axis='x', color=GRID, which='both')
ax.set_axisbelow(True)
ax.axhspan(5.5, 7.5, color='#fbe9e7', zorder=0)
ax.text(0.98, 0.03, 'shaded: noble gases (worse)', transform=ax.transAxes, ha='right', fontsize=8, color=MUT)

# ---- ion-water 0.05 A scans: the 4.5 A step
SC = [('debug_iw_step_v2local_decay', 'v2local e99 (no decay)', '#eb6834', '-'),
      ('debug_iw_step_v2localdec_now', 'v2localdec e110 (beta 3)', '#9a9893', '--'),
      ('debug_iw_step_v3_final', 'LR-POLAR final (learned decay)', '#2a78d6', '-')]
scans = {d: json.load(open(f'{EV}/{d}/iw_step.json'))['scans'] for d, *_ in SC}
ref = json.load(open('/scratch/t510a13/md_water/ion_water_ref/ion_water_ref.json'))['scans']
ions = ['F-_c2v', 'Cl-_c2v', 'Na+_c2v']
sub = gs[1, :].subgridspec(1, 3, wspace=0.22)
for j, ion in enumerate(ions):
    ax = fig.add_subplot(sub[0, j])
    if ion in ref:
        rd = np.array([r['R'] for r in ref[ion]['rows']])
        ed = np.array([r['E_int_eV'] for r in ref[ion]['rows']]) * 23.0605
        m = (rd >= 3.5) & (rd <= 7.0)
        ax.plot(rd[m], ed[m], 'o', color=INK, ms=5, label='DFT (wB97M-V)', zorder=5)
    for d, lab, c, ls in SC:
        p = scans[d][ion]['on']
        ax.plot([r['R'] for r in p], [r['E_int_kcal'] for r in p], ls, color=c, lw=2, label=lab)
    ax.axvspan(4.4, 4.6, color='#f3f2ee', zorder=0)
    ax.set_xlim(3.5, 7.0)
    ax.set_xlabel('ion-O distance (A)')
    if j == 0:
        ax.set_ylabel('E_int (kcal/mol)')
        ax.legend(frameon=False, fontsize=9, loc='lower right')
    ax.set_title(f'{ion.replace("_c2v", "")} ... H2O, 0.05 A scan', loc='left', color=INK)
    ax.grid(color=GRID)
    ax.set_axisbelow(True)
fig.suptitle('RACE v2 LR-POLAR final test (epoch 170) vs earlier runs', x=0.01, ha='left', fontsize=14, color=INK)
fig.savefig(OUT, dpi=120, facecolor='#fcfcfb', bbox_inches='tight')
print('wrote', OUT)
