"""3BPA dihedral test set figure: the molecule with its three rotatable dihedrals, and the DFT energy landscape.

    ~/.conda/envs/bam_cue/bin/python docs/make_3bpa_dihedral_fig.py  -> docs/fig_3bpa_dihedral.{png,pdf}

test_dih.xyz (Kovacs et al. JCTC 2021): 7047 DFT-optimised structures with the three dihedral angles held fixed,
alpha = C13-O5-C1-C4 (ring-ether), beta = C1-O5-C13-C14 (ether-CH2, fixed to 120/150/180 deg),
gamma = O5-C13-C14-C18 (CH2-phenyl); alpha and gamma on a ~4/6 deg grid.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ase.neighborlist import natural_cutoffs, NeighborList

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XYZ = "/scratch/t589a02/datasets_ext/BOTNet-datasets/dataset_3BPA/test_dih.xyz"
COL = {"C": "#5b6068", "H": "#d9dce1", "N": "#2a78d6", "O": "#e34948"}
RAD = {"C": 0.34, "H": 0.22, "N": 0.33, "O": 0.32}
DIH = {"α": (13, 5, 1, 4), "β": (1, 5, 13, 14), "γ": (5, 13, 14, 18)}
DCOL = {"α": "#eb6834", "β": "#6250d6", "γ": "#1baf7a"}


def main():
    plt.rcParams.update({"font.family": ["NanumGothic", "DejaVu Sans"], "font.size": 10})
    from matplotlib import font_manager
    for f in ("NanumGothic-Regular.ttf", "NanumGothic-Bold.ttf"):
        font_manager.fontManager.addfont(os.path.join(ROOT, "docs", "fonts", f))
    a = read(XYZ, index=0)
    E, D = [], []
    with open(XYZ) as fh:
        for line in fh:
            if "dihedrals=" in line:
                D.append([float(x) for x in re.search(r'dihedrals="_JSON \[([^\]]*)\]"', line).group(1).split(",")])
                E.append(float(re.search(r"energy=(-?[\d.]+)", line).group(1)))
    D, E = np.array(D), np.array(E)
    E = (E - E.min()) * 1e3                                       # meV relative to the global minimum

    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.5, 1, 1, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0])
    # molecule: project on its two principal axes
    p = a.positions - a.positions.mean(0)
    _, _, vt = np.linalg.svd(p, full_matrices=False)
    xy, depth = p @ vt[:2].T, p @ vt[2]
    nl = NeighborList(natural_cutoffs(a), self_interaction=False, bothways=False); nl.update(a)
    axis_bonds = {frozenset(q[1:3]): k for k, q in DIH.items()}
    for i in range(len(a)):
        for j in nl.get_neighbors(i)[0]:
            k = axis_bonds.get(frozenset((i, j)))
            ax.plot(*xy[[i, j]].T, color=DCOL[k] if k else "#9aa0a6", lw=5 if k else 2, zorder=1,
                    solid_capstyle="round")
            if k:
                m = xy[[i, j]].mean(0)
                ax.annotate(k, m, xytext=(0, 9), textcoords="offset points", ha="center", fontsize=15,
                            fontweight="bold", color=DCOL[k])
    for i in np.argsort(depth):
        s = a[i].symbol
        ax.add_patch(plt.Circle(xy[i], RAD[s], color=COL[s], ec="#1f2328", lw=0.6, zorder=2))
    ax.set_aspect("equal"); ax.axis("off"); ax.margins(0.04)
    ax.set_title("3BPA (C₁₇H₁₈N₂O₂)과 회전하는 세 결합", fontsize=11, loc="left")
    ax.text(0.0, -0.06, "굵은 선 = 이면각의 회전축 (빨강 O, 파랑 N, 회색 C, 흰색 H)", transform=ax.transAxes, fontsize=8.5, color="#5b6068")

    vmax = np.percentile(E, 99)
    for k, beta in enumerate((120.0, 150.0, 180.0)):
        axh = fig.add_subplot(gs[k + 1])
        sel = D[:, 1] == beta
        al, ga = np.unique(D[:, 0]), np.unique(D[:, 2])
        grid = np.full((len(ga), len(al)), np.nan)
        grid[np.searchsorted(ga, D[sel, 2]), np.searchsorted(al, D[sel, 0])] = E[sel]
        sc = axh.pcolormesh(al, ga, grid, shading="nearest", cmap="viridis", vmin=0, vmax=vmax)
        axh.set_title(f"β = {beta:.0f}° 단면 ({sel.sum()}개 구조)", fontsize=11, loc="left")
        axh.set_xlabel("α (°)")
        if k == 0:
            axh.set_ylabel("γ (°)")
        axh.set_xlim(D[:, 0].min() - 3, D[:, 0].max() + 3); axh.set_ylim(D[:, 2].min() - 3, D[:, 2].max() + 3)
    cb = fig.colorbar(sc, ax=fig.axes[1:], fraction=0.025, pad=0.02)
    cb.set_label("DFT 에너지, 최저점 기준 (meV)")
    fig.suptitle("3BPA 이면각 테스트: 세 이면각을 고정하고 나머지를 DFT로 최적화한 7,047개 구조", fontsize=12, x=0.01, ha="left", y=1.02)
    for ext in ("png", "pdf"):
        out = os.path.join(ROOT, "docs", f"fig_3bpa_dihedral.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print("written", out)
    print("alpha", D[:, 0].min(), D[:, 0].max(), "gamma", D[:, 2].min(), D[:, 2].max(), "E max meV", E.max())


if __name__ == "__main__":
    main()
