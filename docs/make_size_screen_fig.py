"""100k OMat24 size screening: per-epoch training losses and validation errors of the size candidates (SOAP with
species tables by Adam, float64) and the optimizer control (L5x128 with AMSGrad), runs sz100k_<L>x<H>[amsgrad]_omat24_v2_s42.

    ~/.conda/envs/bam_cue/bin/python docs/make_size_screen_fig.py  -> docs/fig_size_screen_100k.{png,pdf}

Runs in progress are plotted up to their last epoch.  Colour = model size (categorical, fixed order); the AMSGrad
control shares the colour of L5x128 and is dashed; every line also has its own marker and an end label.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = [("L5x64", "#2a78d6", "o", "-"), ("L5x128", "#eb6834", "s", "-"), ("L5x192", "#1baf7a", "^", "-"),
        ("L4x128", "#eda100", "D", "-"), ("L6x128", "#e87ba4", "v", "-"),
        ("L5x128amsgrad", "#eb6834", "x", "--")]
LABEL = {"L5x128amsgrad": "L5x128 AMSGrad (control)"}
INK, MUTED, GRID = "#1f2328", "#5b6068", "#e3e5e8"
PANELS = [("train_E", "train energy loss (Huber, per atom)"), ("train_F", "train force loss (Huber)"),
          ("val_E", "valid energy RMSE (meV/structure)"), ("val_F", "valid force RMSE (meV/Å)")]


def read_log(run):
    path = os.path.join(ROOT, "runs", f"{run}_cue", f"loss_{run}_cue.out")
    if not os.path.exists(path):
        return None
    rows = {}
    for line in open(path, errors="ignore"):
        if not re.match(r"^\d\d/\d\d/\d{4} ", line):
            continue
        p = [x.split() for x in line.split("|")]
        tr = [float(x) for x in p[1]]; va = [float(x) for x in p[2]]
        rows[int(p[0][2])] = dict(time=float(p[0][3]), train_E=tr[1], train_F=tr[2], val_E=va[1], val_F=va[2])
    if not rows:
        return None
    eps = sorted(rows)
    return {"epoch": np.array(eps), **{k: np.array([rows[e][k] for e in eps]) for k in rows[eps[0]]}}


def main():
    plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 9.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    logs = {name: read_log(f"sz100k_{name}_omat24_v2_s42") for name, *_ in RUNS}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharex=True)
    rows = []
    for ax, (key, title) in zip(axes.flat, PANELS):
        later = []
        for name, col, mk, ls in RUNS:
            lg = logs[name]
            if lg is None:
                continue
            ax.plot(lg["epoch"], lg[key], color=col, ls=ls, lw=2, marker=mk, ms=4.5, markevery=3,
                    label=LABEL.get(name, name), zorder=3 if ls == "-" else 2)
            later += list(lg[key][lg["epoch"] >= 3])
        if later:
            # robust range: single-epoch spikes (> 3x the median) stay off scale and are marked with an arrow
            later = np.asarray(later)
            med = np.median(later)
            keep = later[later < 3 * med]
            lo, hi = keep.min(), keep.max()
            top = hi + 0.12 * (hi - lo)
            ax.set_ylim(lo - 0.08 * (hi - lo), top)
            for name, col, mk, ls in RUNS:
                lg = logs[name]
                if lg is None:
                    continue
                for ep, v in zip(lg["epoch"], lg[key]):
                    if ep >= 3 and v >= 3 * med:
                        ax.annotate(f"{LABEL.get(name, name)} epoch {ep}: {v:,.0f} (off scale)", xy=(ep, top),
                                    xytext=(ep - 9, top - 0.08 * (top - lo)), fontsize=8, color=INK,
                                    arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("epoch"); ax.set_xlim(0.5, 33.5)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Model size on 100k OMat24 structures (float64, SOAP + species-table Adam, 30 epochs; "
                 "dashed = same L5x128 model trained with AMSGrad)", fontsize=10.5, color=INK, y=0.99)
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "docs", f"fig_size_screen_100k.{ext}"), dpi=170)
    print(f"{'run':26s} {'epoch':>5s} {'s/epoch':>8s} {'valid E RMSE':>13s} {'valid F RMSE':>13s}")
    for name, *_ in RUNS:
        lg = logs[name]
        if lg is not None:
            print(f"{LABEL.get(name, name):26s} {lg['epoch'][-1]:5d} {np.median(lg['time'][1:]):8.0f} "
                  f"{lg['val_E'][-1]:13.1f} {lg['val_F'][-1]:13.2f}")


if __name__ == "__main__":
    main()
