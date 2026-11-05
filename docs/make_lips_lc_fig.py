"""LiPS learning curve figure (force and energy MAE vs number of training frames), RACEv2 vs published models.

    ~/.conda/envs/bam_cue/bin/python docs/make_lips_lc_fig.py     -> docs/fig_lips_learning_curve.{png,pdf}

RACEv2: final recipe (L5h128 + UFF-nb), 3 seeds per size, best-validation checkpoint (summary.json; energies after
the gauge refit, meV/atom).  A size whose runs have not finished is drawn from the latest logged test evaluation
with an open marker and marked "provisional" in the legend.  Reference values are the published test MAEs
(same Fu et al. 5001-frame test set): FreeCG / GemNet-T / NequIP at 19k (Fu et al. TMLR 2023 and the FreeCG paper),
NequIP 2500 (Batzner et al. Nat. Commun. 2022), CarNet-3L 1000, CAMP 2500/100/10.
Reference markers that share a training size are shifted slightly along x (JITTER) so none hides another.
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZES = [("lc10", 10), ("lc100", 100), ("lc1000", 1000), ("lc2500", 2500), ("fu", 19000)]
TEST = "fu_test"

# published references: model -> list of (n_train, E meV/atom or None, F meV/A)
REFS = {
    "FreeCG": [(19000, None, 1.2)],
    "GemNet-T": [(19000, None, 1.3)],
    "NequIP": [(19000, None, 3.7), (2500, 0.08, 4.7)],
    "CarNet-3L": [(1000, 0.09, 5.6)],
    "CAMP": [(2500, 0.08, 4.9), (100, 0.44, 21.6), (10, 1.63, 76.4)],
}
# reference categorical palette, fixed order (dataviz skill references/palette.md); marker shape = second encoding
STYLE = {"RACEv2": ("#2a78d6", "o"), "NequIP": ("#eb6834", "s"), "CAMP": ("#1baf7a", "^"),
         "CarNet-3L": ("#eda100", "D"), "FreeCG": ("#e87ba4", "v"), "GemNet-T": ("#008300", "P")}
INK, MUTED, GRID = "#1f2328", "#5b6068", "#e3e5e8"
# horizontal offsets (x factor) for coinciding training sizes; RACEv2 stays at the true x
JITTER = {("FreeCG", 19000): 0.80, ("GemNet-T", 19000): 1.25, ("NequIP", 19000): 1.0,
          ("NequIP", 2500): 0.85, ("CAMP", 2500): 1.18}


def race_point(tag):
    E, F, prov = [], [], False
    for s in (42, 43, 44):
        n = f"lipsF_{tag}_s{s}"
        summ = os.path.join(ROOT, "runs", f"{n}_cue", f"loss_{n}_cue.summary.json")
        log = os.path.join(ROOT, "runs", f"{n}_cue", f"loss_{n}_cue.out")
        if os.path.exists(summ):
            d = json.load(open(summ))
            E.append(d["gauge"][TEST]["energy_mae"] * 1e3)
            F.append(d[TEST]["force_mae"] * 1e3)
        elif os.path.exists(log):
            t = [l for l in open(log) if f"Test[{TEST}]" in l]
            if t:
                m = re.search(r"E_MAE:\s+([\d.]+).*F_MAE:\s+([\d.]+)", t[-1])
                n_atoms = 83.0
                E.append(float(m.group(1)) / n_atoms); F.append(float(m.group(2))); prov = True
    if not F:
        return None
    return np.mean(E), np.std(E), np.mean(F), np.std(F), prov, len(F)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    pts = {n: race_point(tag) for tag, n in SIZES}
    pts = {n: v for n, v in pts.items() if v is not None}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, which, ylab in [(axes[0], "F", "Force MAE (meV/Å)"), (axes[1], "E", "Energy MAE (meV/atom)")]:
        xs = np.array(sorted(pts))
        k = 2 if which == "F" else 0
        y = np.array([pts[x][k] for x in xs]); e = np.array([pts[x][k + 1] for x in xs])
        prov = np.array([pts[x][4] for x in xs])
        c, mk = STYLE["RACEv2"]
        ax.plot(xs, y, color=c, lw=2, zorder=3)
        ax.errorbar(xs[~prov], y[~prov], yerr=e[~prov], fmt=mk, ms=8, color=c, mec="white", mew=2, capsize=3,
                    zorder=4, label="RACEv2 (this work, 3 seeds)")
        if prov.any():
            ax.plot(xs[prov], y[prov], mk, ms=8, mfc="white", mec=c, mew=2, zorder=4, label="RACEv2 (provisional)")
        for name, rows in REFS.items():
            rr = [(n, (ee if which == "E" else ff)) for n, ee, ff in rows if (ee if which == "E" else ff) is not None]
            if not rr:
                continue
            cc, mm = STYLE[name]
            # same training size as another marker: shift horizontally on the log axis so no marker hides another
            xj = [r[0] * JITTER.get((name, r[0]), 1.0) for r in rr]
            ax.plot(xj, [r[1] for r in rr], mm, ms=9, color=cc, mec="white", mew=1.5, ls="none",
                    zorder=5, label=name)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Training frames (83-atom Li$_{6.75}$P$_3$S$_{11}$ cells)")
        ax.set_ylabel(ylab)
        ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xticks([10, 100, 1000, 2500, 19000]); ax.set_xticklabels(["10", "100", "1000", "2500", "19k"])
        # direct label for the RACEv2 curve (single series of interest)
        ax.annotate("RACEv2", (xs[0], y[0]), xytext=(8, -2), textcoords="offset points", color=INK, fontsize=10,
                    fontweight="bold", va="top")
    axes[0].set_title("LiPS learning curve: forces", color=INK, fontsize=11, loc="left")
    axes[1].set_title("LiPS learning curve: energies", color=INK, fontsize=11, loc="left")
    h, l = axes[0].get_legend_handles_labels()
    order = sorted(range(len(l)), key=lambda i: (not l[i].startswith("RACEv2"), i))
    h, l = [h[i] for i in order], [l[i] for i in order]
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ("png", "pdf"):
        out = os.path.join(ROOT, "docs", f"fig_lips_learning_curve.{ext}")
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        print("written", out)
    for n in sorted(pts):
        print(n, "E %.3f±%.3f  F %.2f±%.2f  provisional=%s seeds=%d" % pts[n])


if __name__ == "__main__":
    main()
