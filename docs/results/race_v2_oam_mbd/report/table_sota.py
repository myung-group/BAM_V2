"""table_sota.py <out prefix>: Matbench Discovery comparison table and per-metric ranks as an image (PNG + PDF), paper-table style."""
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
HL = '#f6d5d0'                                       # highlight of the lab rows
COLS = ['', 'Params', 'Epoch', 'F1', 'MAE', r'R$^2$', 'RMSD', r'$\kappa$ SRME', 'CPS']
ROWS = [('DPA-4.1-Max-OAM-PFT', '-', '-', '0.935', '0.017', '0.872', '0.058', '0.081', '0.913'),
        ('Prophet-OAME-MBD', '62.3M', '3 + 10', '0.928', '0.018', '0.869', '0.059', '0.065', '0.912'),
        ('DPA-4.1-Max-OAM', '-', '-', '0.933', '0.017', '0.872', '0.057', '0.090', '0.910'),
        ('TECE-OAM-RRA-1.0', '222M', '3 + 2', '0.929', '0.018', '0.871', '0.058', '0.093', '0.908'),
        ('EquFlashV2', '44.9M', '-', '0.929', '0.018', '0.873', '0.058', '0.094', '0.907'),
        ('EquiformerV3+DeNS-OAM', '30.3M', '6 + 2', '0.931', '0.018', '0.868', '0.059', '0.118', '0.902'),
        ('GRACE-3L-OAM-L', '42.1M', '-', '0.925', '0.018', '0.875', '0.058', '0.121', '0.900'),
        ('PET-OAM-XL', '730M', '15 + 1', '0.924', '0.019', '0.864', '0.059', '0.119', '0.898'),
        ('DPA-4.1-Air-OAM', '-', '-', '0.921', '0.019', '0.868', '0.060', '0.136', '0.893'),
        ('TACE-OAM-L', '82.9M', '3 + 3', '0.910', '0.020', '0.868', '0.061', '0.126', '0.889'),
        ('eSEN-30M-OAM', '30.2M', '4 + 1', '0.925', '0.018', '0.866', '0.061', '0.170', '0.888'),
        ('EquFlash', '28.7M', '-', '0.919', '0.019', '0.871', '0.060', '0.158', '0.888'),
        ('Nequip-OAM-XL', '32.1M', '30 + -', '0.906', '0.020', '0.872', '0.063', '0.125', '0.886'),
        ('MatRIS-10M-OAM', '-', '-', '0.921', '0.019', '0.871', '0.060', '0.218', '0.877'),
        ('MACE-SOYO-OAM', '-', '-', '0.903', '0.020', '0.868', '0.062', '0.184', '0.873'),
        ('SevenNet-Omni-i12', '54.9M', '2', '0.906', '0.021', '0.868', '0.062', '0.192', '0.873'),
        ('Nequip-OAM-L', '-', '-', '0.893', '0.022', '0.865', '0.065', '0.166', '0.870'),
        ('GRACE-2L-OAM-L', '26.4M', '8 + 4', '0.883', '0.022', '0.862', '0.064', '0.169', '0.865'),
        ('ORB v3', '-', '-', '0.905', '0.024', '0.821', '0.075', '0.210', '0.861'),
        ('*RACE v2 OAM', '17.8M', '2 + 6.2', '0.893', '0.021', '0.872', '0.064', '0.229', '0.858'),
        ('DPA-4.0.1-Pro-MPtrj', '-', '-', '0.857', '0.029', '0.836', '0.069', '0.211', '0.840'),
        ('Allegro-OAM-L', '-', '-', '0.895', '0.022', '0.868', '0.065', '0.319', '0.840'),
        ('GRACE-2L-OAM', '12.6M', '8 + 4', '0.880', '0.023', '0.862', '0.067', '0.294', '0.837'),
        ('*RACE v2 MPtrj-only', '17.8M', '40 (MPtrj)', '0.777', '0.040', '0.800', '0.079', '0.375', '0.761'),
        ('GRACE-1L-OAM', '3.45M', '12 + 8', '0.824', '0.031', '0.842', '0.072', '0.517', '0.761'),
        ('*BAM-MP-DeNS', '-', '-', '0.761', '0.043', '0.797', '0.083', '0.766', '0.672'),
        ('*BAM-OAM', '10.5M', '4 + 64', '-', '0.027', '-', '-', '-', '-')]
RCOLS = ['', 'Compared with', 'F1', 'MAE', r'R$^2$', 'RMSD', r'$\kappa$ SRME', 'CPS']
RROWS = [('*RACE v2 OAM', 'all leaderboard models (49-50)', '21', '18', '3', '18', '21', '20'),
         ('*RACE v2 MPtrj-only', 'MPtrj-only models (19)', '7', '6', '5', '6', '5', '6')]
X1 = [0.265, 0.355, 0.47, 0.56, 0.64, 0.72, 0.80, 0.895, 0.975]          # column centres (first = right edge of the name)
X2 = [0.265, 0.435, 0.59, 0.66, 0.73, 0.80, 0.885, 0.965]
RH, FS = 0.30, 13
n1, n2 = len(ROWS), len(RROWS)
H = RH * (n1 + n2 + 2) + 2.1
fig = plt.figure(figsize=(11.4, H)); ax = fig.add_axes([0.02, 0, 0.96, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, H); ax.axis('off')


def table(y, cols, rows, xs, title):
    ax.text(0.0, y, title, fontsize=FS, va='center', ha='left'); y -= 0.42
    ax.plot([0, 1], [y, y], color='k', lw=1.6); y -= RH * 0.62
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
    y -= RH * 0.12; ax.plot([0, 1], [y, y], color='k', lw=1.6)
    return y


y = table(H - 0.3, COLS, ROWS, X1, 'Matbench Discovery (unique prototypes), sorted by CPS: every leaderboard model down to CPS 0.837 (read on 2026-11-15), and GRACE-1L-OAM.')
y = table(y - 0.5, RCOLS, RROWS, X2, 'Rank of RACE v2 for each metric (1 = best).')
ax.text(0.0, y - 0.3, r'Epoch: pretraining + fine-tuning.  RACE v2 OAM: OMat24 2 epochs, then MPtrj$\times$8 + sAlex$\times$1 for 6.2 epochs; float64 inference.  '
        'MAE in eV/atom, RMSD in Å.\nBAM-OAM: only the MAE is known.  Shaded rows: models of our lab.  "-" in Params / Epoch: not collected.', fontsize=10.5, va='top', ha='left', color='#333333')
for ext in ('png', 'pdf'):
    fig.savefig(f'{sys.argv[1]}.{ext}', dpi=200, facecolor='white', bbox_inches='tight', pad_inches=0.15)
print(sys.argv[1] + '.png', sys.argv[1] + '.pdf')
