"""plot_parity.py <out.png> <tag>:<label> [<tag>:<label> ...]: WBM formation-energy parity plots (model vs DFT, MP2020
corrected), hexbin with log10 point density, unique prototypes (the leaderboard subset), one panel per model."""
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W = '/scratch/t510a13/race_v2/work'
S = pd.read_csv('/scratch/t510a13/probabilistic_symmetrization/bundle/scratch/t589a02/mp_data/wbm/2023-12-13-wbm-summary.csv.gz',
                usecols=['material_id', 'e_form_per_atom_mp2020_corrected', 'unique_prototype'])
S = S[S.unique_prototype.astype(bool)]
runs = [a.split(':', 1) for a in sys.argv[2:]]
fig, axs = plt.subplots(1, len(runs), figsize=(5.6 * len(runs) + 0.8, 5.2), squeeze=False)
lim = (-5, 5)
for ax, (tag, label) in zip(axs[0], runs):
    d = pd.read_csv(f'{W}/runs/mbd/{tag}/discovery.csv.gz').merge(S, on='material_id')
    x, y = d['e_form_per_atom_mp2020_corrected'].to_numpy(), d['e_form_per_atom_racev2'].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    hb = ax.hexbin(x, y, gridsize=180, extent=(*lim, *lim), bins='log', cmap='viridis', mincnt=1, linewidths=0)
    ax.plot(lim, lim, ls='--', color='0.45', lw=0.8)
    mae = np.mean(np.abs(y - x)); r2 = 1 - np.sum((y - x) ** 2) / np.sum((x - x.mean()) ** 2)
    ax.text(0.04, 0.96, f'{label}\nMAE {mae * 1000:.1f} meV/atom   R² {r2:.3f}\nn = {len(x):,} (unique prototypes)',
            transform=ax.transAxes, va='top', fontsize=9.5)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    ax.set_xlabel('DFT formation energy (eV/atom)'); ax.set_ylabel(f'{label} formation energy (eV/atom)')
cb = fig.colorbar(hb, ax=axs[0].tolist(), shrink=0.9, pad=0.02)
cb.set_label('point density (log scale)')
fig.savefig(sys.argv[1], dpi=150, bbox_inches='tight', facecolor='white')
print(sys.argv[1])
