"""OAM subset screening: per-epoch training losses and validation errors of RACEv2 vs V1 on the OMat24, MPtrj and
sAlex subsets (10k train each, float64, runs oam64_<dataset>_<v2|v1>_s42).

    ~/.conda/envs/bam_cue/bin/python docs/make_oam_subset_fig.py  -> docs/fig_oam_subset_curves.{png,pdf}

Reads the per-epoch table of runs/<run>_cue/loss_<run>_cue.out (columns: EPOCH TIME | LOSS E_LOSS F_LOSS AUX GRAD |
valid LOSS E_RMSE[meV] F_RMSE[meV/A] E_MAE F_MAE | LR WE WF); runs still in progress are plotted up to their last
epoch.  Linear y axes (user request); the first epoch is drawn but the y range follows the later epochs.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = [("omat24", "OMat24"), ("mptrj", "MPtrj"), ("salex", "sAlex")]
STYLE = {"v2": ("RACEv2", "#4169E1", "o"), "v1": ("V1", "#DC143C", "s")}      # royal blue / crimson (user choice)
INK, MUTED, GRID = "#1f2328", "#5b6068", "#e3e5e8"
ROWS = [("train_E", "train energy loss (Huber, per atom)"), ("train_F", "train force loss (Huber)"),
        ("val_E", "valid energy RMSE (meV/structure)"), ("val_F", "valid force RMSE (meV/Å)")]


def read_log(run):
    path = os.path.join(ROOT, "runs", f"{run}_cue", f"loss_{run}_cue.out")
    rows = {}
    if not os.path.exists(path):
        return None
    for line in open(path):
        if not re.match(r"^\d\d/\d\d/\d{4} ", line):
            continue
        parts = [p.split() for p in line.split("|")]
        ep = int(parts[0][2]); t = float(parts[0][3])
        tr = [float(x) for x in parts[1]]; va = [float(x) for x in parts[2]]
        rows[ep] = dict(time=t, train_E=tr[1], train_F=tr[2], val_E=va[1], val_F=va[2])   # resumed runs: last wins
    if not rows:
        return None
    eps = sorted(rows)
    return {"epoch": np.array(eps), **{k: np.array([rows[e][k] for e in eps]) for k in rows[eps[0]]}}


def main():
    from matplotlib import font_manager
    for f in ("NanumGothic-Regular.ttf", "NanumGothic-Bold.ttf"):
        p = os.path.join(ROOT, "docs", "fonts", f)
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
    plt.rcParams.update({"font.family": ["NanumGothic", "DejaVu Sans"], "font.size": 9.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.unicode_minus": False})
    fig, axes = plt.subplots(len(ROWS), len(DATASETS), figsize=(12, 11), sharex="col")
    summary = []
    for j, (ds, dname) in enumerate(DATASETS):
        logs = {v: read_log(f"oam64_{ds}_{v}_s42") for v in ("v2", "v1")}
        for i, (key, label) in enumerate(ROWS):
            ax = axes[i, j]
            later = []
            for v, lg in logs.items():
                if lg is None:
                    continue
                name, c, mk = STYLE[v]
                ax.plot(lg["epoch"], lg[key], color=c, lw=2, marker=mk, ms=3.5, mec="white", mew=0.6, label=name)
                later.append(lg[key][lg["epoch"] > 1])
            vals = np.concatenate([x for x in later if x.size]) if later else np.array([1.0])
            if vals.size:
                ax.set_ylim(0, 1.08 * np.percentile(vals, 98))
            ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if j == 0:
                ax.set_ylabel(label, color=INK)
            if i == 0:
                ax.set_title(dname, color=INK, fontsize=12, loc="left", fontweight="bold")
            if i == len(ROWS) - 1:
                ax.set_xlabel("epoch")
        for v, lg in logs.items():
            if lg is not None:
                summary.append((dname, STYLE[v][0], int(lg["epoch"][-1]), float(np.median(lg["time"][1:])) if len(lg["time"]) > 1 else float(lg["time"][0]),
                                float(lg["val_E"][-1]), float(lg["val_F"][-1])))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("OAM subsets (10k training structures, float64): RACEv2 vs V1, per-epoch losses and validation errors", x=0.01, ha="left",
                 fontsize=12, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    for ext in ("png", "pdf"):
        out = os.path.join(ROOT, "docs", f"fig_oam_subset_curves.{ext}")
        fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
        print("written", out)
    for r in summary:
        print("%-7s %-7s epoch %3d  %5.0f s/epoch  val E_RMSE %8.1f meV  F_RMSE %6.1f meV/A" % r)


if __name__ == "__main__":
    main()
