"""table_sota_all.py <out prefix>: Matbench Discovery comparison tables with every leaderboard column (discovery,
RMSD, kappa, CPS, and the MD and diatomics scores CMDS and CDS) as images (PNG + PDF), paper-table style.

Writes <out prefix>.png (all metrics and the ranks of RACE v2), <out prefix>_md_diatomics.png (components of CMDS
and CDS) and <out prefix>.pdf (both tables, one page each).  Inputs next to this file:
  leaderboard_6d587c1.json   every model of the benchmark repository (scripts/eval/mbd/leaderboard_from_yaml.py)
  final_results_all.json     RACE v2 numbers
and the list of the models shown on the leaderboard page, scripts/eval/mbd/mbd_leaderboard_2026-11-12.tsv.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
HL = '#f6d5d0'                                       # highlight of the lab rows
LB = json.load(open(HERE / 'leaderboard_6d587c1.json'))
OURS = json.load(open(HERE / 'final_results_all.json'))
SHOWN = [l.split('\t')[0] for l in open(HERE.parents[3] / 'scripts/eval/mbd/mbd_leaderboard_2026-11-12.tsv')
         if l.strip() and not l.startswith('#') and not l.startswith('model\t')]
BY = {m['name']: m for m in LB['models']}
shown = [BY[n] for n in SHOWN if n in BY]
EPOCH = {'Prophet-OAME-MBD': '3 + 10', 'TECE-OAM-RRA-1.0': '3 + 2', 'EquiformerV3+DeNS-OAM': '6 + 2', 'PET-OAM-XL': '15 + 1',
         'TACE-OAM-L': '3 + 3', 'eSEN-30M-OAM': '4 + 1', 'Nequip-OAM-XL': '30 + -', 'SevenNet-Omni-i12': '2',
         'GRACE-2L-OAM-L': '8 + 4', 'GRACE-2L-OAM': '8 + 4', 'GRACE-1L-OAM': '12 + 8'}
race = {**OURS['race_v2_oam'], 'name': '*RACE v2 OAM', 'params': 17757551, 'epoch': '2 + 6.2'}
race.update({'md_' + k: v for k, v in race['md'].items()}); race.update({'di_' + k: v for k, v in race['diatomics'].items()})
race.update({'p_' + k: v for k, v in race['pillars'].items()})
mptrj = {**OURS['race_v2_mptrj'], 'name': '*RACE v2 MPtrj-only', 'params': 17757551, 'epoch': '40 (MPtrj)'}
for m in LB['models']:
    m.update({'md_' + k: v for k, v in m['md'].items()}); m.update({'di_' + k: v for k, v in m['diatomics'].items()})
    m.update({'p_' + k: v for k, v in m['pillars'].items()}); m['epoch'] = EPOCH.get(m['name'], '-')
LOWER = {'MAE', 'RMSD', 'kSRME', 'md_vdos_error', 'md_adf_error', 'md_pressure_error', 'md_run_time_sec', 'di_pbe_energy_mae',
         'di_pbe_force_mae'}


def fmt(v, nd=3):
    if v is None:
        return '-'
    if isinstance(v, str):
        return v
    if abs(v) >= 1e7:
        return f'{v:.0e}'.replace('e+0', 'e').replace('e+', 'e')
    if abs(v) >= 100:
        return f'{v:,.0f}'
    return f'{v:.{nd}f}'


def par(v):
    return '-' if v is None else (f'{v / 1e6:.3g}M' if v < 9.95e8 else f'{v / 1e9:.3g}B')


def rank(key):
    """rank of RACE v2 OAM among the shown leaderboard models that have the value (ties do not count against it)."""
    vals = [m[key] for m in shown if m.get(key) is not None]
    mine = race[key]
    better = sum((v < mine) if key in LOWER else (v > mine) for v in vals)
    return f'{better + 1} / {len(vals) + 1}'


def draw(out, blocks, note, width):
    """blocks: (title, cols, rows, xs, groups); rows: tuples of strings, the name first ('*' = lab row)."""
    RH, FS = 0.30, 13
    H = sum(RH * (len(b[2]) + 2) + 0.95 + (0.3 if b[4] else 0) for b in blocks) + 0.45 * (len(blocks) - 1) + 0.25 * note.count('\n') + 1.0
    fig = plt.figure(figsize=(width, H)); ax = fig.add_axes([0.02, 0, 0.96, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, H); ax.axis('off')
    y = H - 0.3
    for title, cols, rows, xs, groups in blocks:
        ax.text(0.0, y, title, fontsize=FS, va='center', ha='left'); y -= 0.42
        ax.plot([0, 1], [y, y], color='k', lw=1.6)
        if groups:
            y -= RH * 0.62
            for x0, x1, g in groups:
                ax.text((x0 + x1) / 2, y, g, fontsize=FS, style='italic', va='center', ha='center')
                ax.plot([x0 - 0.02, x1 + 0.02], [y - RH * 0.5, y - RH * 0.5], color='k', lw=0.6)
            y -= RH * 0.38
        y -= RH * 0.62
        for x, c in zip(xs, cols):
            ax.text(x, y, c, fontsize=FS, weight='bold', va='center', ha='center')
        y -= RH * 0.62; ax.plot([0, 1], [y, y], color='k', lw=0.9); y -= RH * 0.12
        for r in rows:
            lab = r[0].startswith('*')
            if lab:
                ax.add_patch(Rectangle((0, y - RH), 1, RH, color=HL, lw=0, zorder=0))
            ax.text(xs[0], y - RH / 2, r[0].lstrip('*'), fontsize=FS, va='center', ha='right', weight='bold' if lab else 'normal')
            for x, v in zip(xs[1:], r[1:]):
                ax.text(x, y - RH / 2, v, fontsize=FS, va='center', ha='center', weight='bold' if lab else 'normal')
            y -= RH
        y -= RH * 0.12; ax.plot([0, 1], [y, y], color='k', lw=1.6); y -= 0.5
    ax.text(0.0, y + 0.2, note, fontsize=10.5, va='top', ha='left', color='#333333', linespacing=1.5)
    bottom = y + 0.2 - 0.24 * (note.count('\n') + 1) - 0.1                 # crop the page below the note
    fig.set_size_inches(width, H - bottom); ax.set_ylim(bottom, H)
    fig.savefig(f'{out}.png', dpi=200, facecolor='white', bbox_inches='tight', pad_inches=0.15)
    print(out + '.png')
    return fig


# the rows of the earlier table: every shown model down to CPS 0.837, GRACE-1L-OAM and the models of our lab
top = sorted((m for m in shown if m['CPS'] is not None and m['CPS'] >= 0.8365), key=lambda m: -m['CPS'])
models = sorted(top + [race], key=lambda m: -m['CPS']) + [mptrj, BY['GRACE-1L-OAM'], {**BY['BAM-MP-DeNS'], 'name': '*BAM-MP-DeNS'},
                                                           {'name': '*BAM-OAM', 'params': 10.5e6, 'epoch': '4 + 64', 'MAE': 0.027}]
K1 = ['F1', 'DAF', 'Prec', 'Acc', 'MAE', 'R2', 'RMSD', 'kSRME', 'CPS', 'CMDS', 'CDS']
C1 = ['', 'Params', 'Epoch', 'F1', 'DAF', 'Prec', 'Acc', 'MAE', r'R$^2$', 'RMSD', r'$\kappa$ SRME', 'CPS', 'CMDS', 'CDS']
X1 = [0.205, 0.255, 0.33] + [0.395 + 0.058 * i for i in range(11)]
rows1 = [(m['name'], par(m.get('params')), m.get('epoch', '-')) + tuple(fmt(m.get(k)) for k in K1) for m in models]
XR = [0.205, 0.29] + X1[3:]
rrows = [('*RACE v2 OAM', 'shown models + RACE v2') + tuple(rank(k) for k in K1),
         ('*RACE v2 MPtrj-only', 'MPtrj-only models (19)', '7', '-', '-', '-', '6', '5', '6', '5', '6', '-', '-')]
note1 = ('Epoch: pretraining + fine-tuning.  RACE v2 OAM: OMat24 2 epochs, then MPtrj$\\times$8 + sAlex$\\times$1 for 6.2 epochs; float64 inference.  '
         'MAE in eV/atom, RMSD in Å.  Shaded rows: models of our lab.  "-": not collected or not measured.\n'
         f'Other models: model YAML files of janosh/matbench-discovery (commit {LB["commit"]}); CPS, CMDS and CDS recomputed with the site formulas.  '
         'CMDS (molecular dynamics) and CDS (diatomic curves) are not part of CPS.\n'
         'RACE v2 OAM, CMDS: 17 DynaMat systems, summed rollout time 22,174 s on one NVIDIA GH200 (float64).  CDS: 87 homonuclear curves; the mean energy error is dominated by\n'
         'Ne, Ar and Kr, whose energies diverge below about 1 Å (the accuracy pillar is 0).  Without Ne, Ar and Kr the same formula gives 0.768 (for reference only).\n'
         'Ranks: 1 + number of shown leaderboard models (49, page read on 2026-11-12) with a better value, out of the models that report the metric plus RACE v2.')
figs = [draw(sys.argv[1], [('Matbench Discovery (unique prototypes), sorted by CPS: every leaderboard model down to CPS 0.837, and GRACE-1L-OAM.', C1, rows1, X1, None),
                   ('Rank of RACE v2 for each metric (1 = best) / number of models compared.', ['', 'Compared with'] + C1[3:], rrows, XR, None)], note1, 17.5)]

# components of CMDS and CDS
K2 = ['CMDS', 'md_vdos_error', 'md_adf_error', 'md_pressure_error', 'md_run_time_sec', 'md_speed',
      'CDS', 'p_accuracy', 'p_geometry', 'p_speed', 'p_physicality', 'di_pbe_energy_mae', 'di_pbe_force_mae']
N2 = [3, 2, 2, 2, 0, 3, 3, 3, 3, 3, 3, 2, 2]
C2 = ['', 'CMDS', 'vDOS', 'ADF', 'Pressure', 'Time (s)', 'Speed', 'CDS', 'Accuracy', 'Geometry', 'Speed', 'Physical.', 'E MAE', 'F MAE']
X2 = [0.215] + [0.27 + 0.0585 * i for i in range(6)] + [0.635 + 0.0585 * i for i in range(7)]
have = [m for m in models if m.get('CMDS') is not None or m.get('CDS') is not None]
rows2 = [(m['name'],) + tuple(fmt(m.get(k), n) for k, n in zip(K2, N2)) for m in have]
rr2 = [('*RACE v2 OAM',) + tuple(rank(k) for k in K2)]
note2 = ('CMDS = 0.3 (1 - vDOS error / 100) + 0.3 (1 - pressure error / 100) + 0.2 (1 - ADF error / 100) + 0.2 speed;  speed = 1 at 9,000 s and 0 at 300,000 s of summed rollout time (log scale).\n'
         'CDS = (4 accuracy + 2 geometry + 1 speed + 2 physicality) / 9 times the fraction of the 87 scored elements;  E MAE (eV) and F MAE (eV/Å) are means over the elements and enter the\n'
         'accuracy pillar (0 at 4 eV and 5 eV/Å).  Run times are on different hardware (RACE v2: NVIDIA GH200, float64).  '
         f'Other models: YAML files of janosh/matbench-discovery (commit {LB["commit"]}).\n'
         'RACE v2 OAM diatomics: median E MAE 0.66 eV, 82 of 87 elements below 4 eV; mean without Ne, Ar, Kr 0.85 eV and 1.69 eV/Å (CDS 0.768), without the five noble gases 0.73 eV and 0.91 eV/Å (CDS 0.831).')
figs.append(draw(sys.argv[1] + '_md_diatomics',
     [('Molecular dynamics (CMDS) and diatomic curves (CDS): components, same models as the main table.', C2, rows2, X2,
       [(X2[1], X2[6], 'Molecular dynamics, 17 systems (errors in %)'), (X2[7], X2[13], 'Diatomics, 87 homonuclear curves')]),
      ('Rank of RACE v2 OAM for each column (1 = best) / number of models compared.', C2, rr2, X2, None)], note2, 17.5))
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
with PdfPages(sys.argv[1] + '.pdf') as pdf:                                 # both tables, one page each
    for f in figs:
        pdf.savefig(f, facecolor='white', bbox_inches='tight', pad_inches=0.15)
print(sys.argv[1] + '.pdf')
