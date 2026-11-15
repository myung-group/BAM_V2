"""Plot of the NaCl dissociation scans (dissociation.py outputs) against the physically expected curves."""
import json
import numpy as np
import ase.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

K = 14.399645
INK, INK2, MUTED, GRID, BASE, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
C_RACE, C_GLOB, C_FRAG = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': BASE, 'axes.labelcolor': INK2, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False})


def centroid_distance(a0, i, ds, rmax=30.0):
    u = a0.positions[i] - a0.positions.mean(0); u /= np.linalg.norm(u)
    rest = np.delete(a0.positions, i, 0); com = rest.mean(0); R = []
    for d in ds:
        lo, hi = 0.0, rmax + 10
        for _ in range(60):
            t = 0.5 * (lo + hi)
            if np.linalg.norm(a0.positions[i] + t * u - rest, axis=1).min() < d:
                lo = t
            else:
                hi = t
        R.append(np.linalg.norm(a0.positions[i] + hi * u - com))
    return np.array(R)


a0 = ase.io.read('data_local/4ghdnnp/nacl/test.xyz', 0)
cases = [('Cl1', 1, -1.0, 'Pull Cl⁻ out of Na₉Cl₈⁺  (fragments +2 / −1)'),
         ('Na11', 11, +1.0, 'Pull Na⁺ out of Na₉Cl₈⁺  (fragments 0 / +1)')]
fig, ax = plt.subplots(2, 2, figsize=(11, 7.2), sharex='col', gridspec_kw={'height_ratios': [1.6, 1]})
for col, (tag, i, q_at, title) in enumerate(cases):
    o = json.load(open(f'runs/local_4g/nacl_racelr_pauling_q/dissociation_{tag}.json'))
    d = np.array(o['d']); R = centroid_distance(a0, i, d)
    ideal = K * q_at * (1.0 - q_at) / R; ideal -= ideal[-1]
    a = ax[0, col]
    a.axhline(0, color=BASE, lw=1)
    a.plot(d, ideal, ls='--', color=INK2, lw=1.5, label='ideal: monopole K q₁q₂/R' if q_at < 0 else 'ideal: no monopole (neutral rest)')
    for key, c, lab in (('race', C_RACE, 'RACE (local)'), ('lr_global', C_GLOB, 'RACE_LR, total-charge constraint'),
                        ('lr_frag', C_FRAG, 'RACE_LR, fragment constraints')):
        e = np.array(o[key]); a.plot(d, e - e[-1], color=c, lw=2, label=lab)
    a.axvline(5.0, color=MUTED, lw=1, ls=':'); a.text(5.15, a.get_ylim()[0] * 0.95 if q_at < 0 else a.get_ylim()[1] * 0.9,
                                                       'cutoff 5 Å', color=MUTED, fontsize=9, va='bottom' if q_at < 0 else 'top')
    a.set_title(title, color=INK, fontsize=11, loc='left')
    a.set_ylabel('E(d) − E(30 Å)  (eV)')
    a.legend(frameon=False, fontsize=9, loc='lower right' if q_at < 0 else 'upper right', labelcolor=INK2)
    b = ax[1, col]
    b.axhline(q_at, ls='--', color=INK2, lw=1.5, label=f'ideal: integer charge {q_at:+.0f} e')
    b.plot(d, o['q_global'], color=C_GLOB, lw=2, label='RACE_LR, total-charge constraint')
    b.plot(d, o['q_frag'], color=C_FRAG, lw=2, label='RACE_LR, fragment constraints')
    b.text(d[-1], o['q_global'][-1], f" {o['q_global'][-1]:+.2f} e", color=INK2, fontsize=9, va='center')
    b.set_ylabel('charge of the pulled atom (e)')
    b.set_xlabel('distance to the nearest atom of the rest, d (Å)')
    b.legend(frameon=False, fontsize=9, loc='upper right' if q_at < 0 else 'lower right', labelcolor=INK2)
    b.set_ylim(-1.15, 1.15)
    b.set_xlim(None, 34)
fig.suptitle('NaCl cluster dissociation, local RACE_LR checkpoint (Pauling init + Hirshfeld charge loss)\n'
             'Checks the form of the long-range tail; there is no DFT reference beyond the training geometries',
             color=INK, fontsize=11, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig('runs/local_4g/plots/nacl_dissociation.png', dpi=150)
print('saved')
