"""Plot of the NVE MD runs (md_nve.py --out) against the physically expected behaviour: total energy constant
(Etot - Etot(0) = 0) and the total charge equal to Q; the training range of the minimum interatomic distance."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK, INK2, MUTED, GRID, BASE, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
C_RACE, C_LR = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': BASE, 'axes.labelcolor': INK2, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False})
TRAIN_DMIN = {'carbon_chain': 0.993, 'nacl': 2.209}          # minimum over the training subset (Å)
TITLE = {'carbon_chain': 'C₁₀H₃⁺ (Q = +1), 300 K start', 'nacl': 'Na₉Cl₈⁺ (Q = +1), 300 K start'}


def ref_label(ax, y, text, x=0.99, below=False):
    ax.annotate(text, (x, y), xycoords=('axes fraction', 'data'), textcoords='offset points',
                xytext=(0, -4 if below else 4), ha='right', va='top' if below else 'bottom', color=INK2, fontsize=9)


fig, ax = plt.subplots(3, 2, figsize=(11, 8.4), sharex='col', gridspec_kw={'height_ratios': [1.4, 1, 0.7]})
for col, s in enumerate(['carbon_chain', 'nacl']):
    a, b, c = ax[0, col], ax[1, col], ax[2, col]
    a.axhline(0, ls='--', color=INK2, lw=1.5); ref_label(a, 0, 'ideal: Etot constant', below=True)
    b.axhline(TRAIN_DMIN[s], ls='--', color=INK2, lw=1.5)
    ref_label(b, TRAIN_DMIN[s], 'smallest distance in the training data', below=(s == 'nacl'))
    c.axhline(0, ls='--', color=INK2, lw=1.5); ref_label(c, 0, 'ideal: Σq = Q (RACE_LR)')
    for m, col_ in (('race', C_RACE), ('racelr_pauling_q', C_LR)):
        d = np.load(f'runs/local_4g/md/{s}_{m}.npz')
        de = (d['epot'] + d['ekin'] - d['epot'][0] - d['ekin'][0]) * 1e3
        a.plot(d['t_ps'], de, color=col_, lw=1.5 if s == 'carbon_chain' else 2)
        b.plot(d['t_ps'], d['dmin'], color=col_, lw=1.5 if s == 'carbon_chain' else 2)
        if str(d['stopped']) != 'None':
            a.plot(d['t_ps'][-1], de[-1], marker='X', ms=10, color=col_, mec=SURF, mew=1.5, ls='none')
            a.annotate(f'runaway at {d["t_ps"][-1]:.2f} ps', (d['t_ps'][-1], de[-1]), textcoords='offset points',
                       xytext=(-10, 0), ha='right', va='center', color=INK2, fontsize=9)
        if m != 'race':
            c.plot(d['t_ps'], (d['sum_q'] - float(d['charge'])) * 1e6, color=col_, lw=2)
    if s == 'nacl':
        a.set_yscale('symlog', linthresh=1.0)
        a.set_ylabel('Etot(t) − Etot(0)  (meV, symlog)')
    else:
        a.set_ylim(-4, 4)
        a.set_ylabel('Etot(t) − Etot(0)  (meV)')
    a.set_title(TITLE[s], color=INK, fontsize=11, loc='left')
    b.set_ylabel('min interatomic distance (Å)')
    c.set_ylabel('Σq − Q  (10⁻⁶ e)')
    c.set_ylim(-1, 1)
    c.set_xlabel('time (ps), NVE, dt 0.5 fs')
handles = [Line2D([], [], ls='--', color=INK2, lw=1.5, label='ideal / reference'),
           Line2D([], [], color=C_RACE, lw=2, label='RACE (local)'),
           Line2D([], [], color=C_LR, lw=2, label='RACE_LR (Pauling init + Hirshfeld charge loss)')]
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.01, 0.935), ncol=3, frameon=False, labelcolor=INK2)
fig.suptitle('NVE MD with the local 4G-HDNNP checkpoints (reduced model, 1000 training structures)\n'
             'Energy is conserved to about 2 meV until the NaCl runs leave the training range and fall into a hole',
             color=INK, fontsize=11, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig('runs/local_4g/plots/md_nve.png', dpi=150)
print('saved')
