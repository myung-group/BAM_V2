"""RACEv2 architecture figure in the layout of RACE v1 paper Figure 1 (b: total pipeline, c: interaction layer).

    ~/.conda/envs/bam_cue/bin/python docs/make_architecture_fig.py      -> docs/RACEv2_architecture.{png,pdf}

Everything drawn here follows the final recipe (input_3bpa_omat24_r37_uffw03_s42_cue.json): x_mode basis,
x_norm hill, update linear, readout gated, message_norm avg, residual chebyshev, layer_scale 0.5,
radial_shells, radial_prior gaussian, pair_repulsion uff (non-bonded switch c=1.25, width 0.3 A).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

DOCS = os.path.dirname(os.path.abspath(__file__))
for f in ("NanumGothic-Regular.ttf", "NanumGothic-Bold.ttf"):
    font_manager.fontManager.addfont(os.path.join(DOCS, "fonts", f))
plt.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

GREY, GREEN, BLUE, RED, ORANGE, PURPLE = "#6b6b6b", "#2ca02c", "#1f4e79", "#c00000", "#d2691e", "#7b3fa0"
BOXFC, BOXEC = "#ededed", "#9a9a9a"
NEW = "#fff2cc"       # fill of boxes that are new in v2 (vs v1)
NEWEC = "#c98f00"



# English labels for `--lang en` (docs/RACEv2_architecture_en.*, used by the English README)
LANG = "ko"
EN = {
 "국소 화학 환경": "Local chemical environment",
 "(컷오프 5 Å 이내)": "(within cutoff 5 Å)",
 "원소 표현\n(원소 종류 → 벡터)": "Element embedding\n(species → vector)",
 "방향 표현\n$Y_l(\\hat{\\mathbf{r}}_{ij}),\\ l\\leq 3$": "Angular basis\n$Y_l(\\hat{\\mathbf{r}}_{ij}),\\ l\\leq 3$",
 "거리 표현\n(Bessel + 고차 진동 억제, 원소 껍질 채널)": "Radial basis\n(Bessel + oscillation prior,\nelement shell channels)",
 "원자 주변 요약 $A_i$\n(이웃 정보를 한 번 모은 벡터)": "Atomic basis $A_i$\n(neighbourhood summed once)",
 "$x_i^{(t)}$ = $A_i$의 변형\n(크기 조절)": "$x_i^{(t)}$ = projection of $A_i$\n(hill scaling)",
 "두 층 전\n특징 빼기": "subtract\nlayer t-2",
 "$h_i^{(5)}$ (미사용)": "$h_i^{(5)}$ (unused)",
 "UFF 반발력 $E_i^{\\rm rep}$\n(형태 고정, 결합 안 한 쌍만;\n전역 배율 s 공유)": "UFF repulsion $E_i^{\\rm rep}$\n(fixed form, non-bonded\npairs only; one global scale s)",
 "총 에너지  $E=\\Sigma_i\\,(E_0+E_i^{\\rm rep}+\\Sigma_t\\,E_i^{(t)})$\n힘  $\\mathbf{F}=-\\nabla E$ (자동미분)": "Total energy  $E=\\Sigma_i\\,(E_0+E_i^{\\rm rep}+\\Sigma_t\\,E_i^{(t)})$\nForces  $\\mathbf{F}=-\\nabla E$ (autodiff)",
 "학습: SOAP 옵티마이저 · Huber 손실 · EMA · 학습 후 에너지 상수 보정": "Training: SOAP optimizer · Huber loss · EMA · energy-constant refit after training",
 "회색: 원자 특징 h   초록: 원자 주변 요약 x   파랑: 방향 $Y$   빨강: 거리 $\\tilde e$   보라: 두 층 전 특징   노랑 상자: v2에서 추가/변경": "grey: node features h   green: atomic basis x   blue: angular $Y$   red: radial $\\tilde e$   purple: features of layer t-2   yellow boxes: new/changed in v2",
 "상호작용 층 t  (최종 모델)": "Interaction layer t  (final model)",
 "자기 자신 항\n(원소별 Linear)": "Self term\n(species-wise Linear)",
 "방향 정보 결합": "angular coupling",
 "거리 가중치 $R_t(r_{ij})$": "radial weights $R_t(r_{ij})$",
 "이웃 합 ÷ 평균 이웃 수": "sum over neighbours ÷ mean neighbour count",
 "원자 주변 요약 $x_i$ 곱": "product with atomic basis $x_i$",
 "더하기 (비선형 없음)": "add (no nonlinearity)",
 "증분 배율 γ (LayerScale)": "increment scale γ (LayerScale)",
 "2×증분 − 두 층 전 특징 (Chebyshev)": "2 × increment - features of layer t-2 (Chebyshev)",
 "$E_i^{(t)}$ (이 층의 에너지 조각)": "$E_i^{(t)}$ (energy stratum of this layer)",
 "비선형은 여기에만": "nonlinearity only here",
 "방향 정보 결합(텐서곱)": "angular coupling (tensor product)",
 "곱": "product",
 "이웃 합": "sum over neighbours",
 "합": "sum",
 "두 층 전 빼기": "subtract layer t-2",
 "RACEv2 최종 모델 구조 (v1 논문 Figure 1 b·c 구도)": "RACEv2 final architecture (layout of Figure 1 b, c of the RACE paper)"
}


def T(text):
    return EN.get(text, text) if LANG == "en" else text


def box(ax, x, y, w, h, text, fc=BOXFC, ec=BOXEC, fs=8.5, bold=False, lw=1.0, rounding=0.18, color="black"):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}",
                       fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(p)
    ax.text(x, y, T(text), ha="center", va="center", fontsize=fs, zorder=4, color=color,
            fontweight="bold" if bold else "normal", linespacing=1.15)
    return p


def arrow(ax, p0, p1, color=GREY, lw=1.3, style="-|>", ls="-", shrink=0, zorder=2, ms=9, connection=None):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, color=color, lw=lw, linestyle=ls, mutation_scale=ms,
                        shrinkA=shrink, shrinkB=shrink, zorder=zorder,
                        connectionstyle=connection or "arc3,rad=0")
    ax.add_patch(a)
    return a


def line(ax, pts, color=GREY, lw=1.3, ls="-", zorder=2):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=zorder, solid_capstyle="round")


def opnode(ax, x, y, sym, r=0.19, fs=9, color="black"):
    ax.add_patch(Circle((x, y), r, fc="white", ec=color, lw=1.1, zorder=4))
    ax.text(x, y, sym, ha="center", va="center", fontsize=fs, zorder=5, color=color)


def label(ax, x, y, text, fs=8, color="black", ha="center", va="center", bold=False, bg=None):
    kw = {}
    if bg:
        kw["bbox"] = dict(boxstyle="round,pad=0.12", fc=bg, ec="none")
    ax.text(x, y, T(text), ha=ha, va=va, fontsize=fs, color=color, zorder=5,
            fontweight="bold" if bold else "normal", **kw)


# ----------------------------------------------------------------------------------------------- panel b
def panel_pipeline(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 15.2); ax.set_aspect("equal"); ax.axis("off")
    label(ax, 0.25, 14.95, "b", fs=13, bold=True, ha="left")
    label(ax, 5.0, 14.9, "국소 화학 환경", fs=10, bold=True)
    label(ax, 4.2, 14.5, r"$\{|\mathbf{r}_{ij}|\},\ \{Z_i, Z_j\},\ \{\hat{\mathbf{r}}_{ij}\}$", fs=8.5)
    label(ax, 6.55, 14.5, "(컷오프 5 Å 이내)", fs=8)

    # atom cluster
    cx, cy = 5.0, 13.1
    import math, random
    random.seed(3)
    ax.add_patch(Circle((cx, cy), 1.05, fc="none", ec="#bbbbbb", lw=0.8, ls="--", zorder=1))
    for k in range(13):
        ang = 2 * math.pi * k / 13 + random.uniform(-0.15, 0.15)
        rr = random.uniform(0.45, 0.98)
        px, py = cx + rr * math.cos(ang), cy + rr * math.sin(ang)
        line(ax, [(cx, cy), (px, py)], color="#b39ddb", lw=0.8, zorder=1)
        ax.add_patch(Circle((px, py), 0.09, fc="#2a6fdb", ec="#1a4b9a", lw=0.5, zorder=2))
    ax.add_patch(Circle((cx, cy), 0.13, fc="#e0157a", ec="#9e0f55", lw=0.6, zorder=3))
    label(ax, cx + 1.25, cy + 0.15, r"$r_c$", fs=8, color="#888888")

    # embeddings
    ye = 11.1
    arrow(ax, (2.0, 12.25), (2.0, ye + 0.42)); label(ax, 2.35, 11.9, r"$\{Z_i\}$", fs=8, ha="left")
    arrow(ax, (5.0, 12.0), (5.0, ye + 0.42)); label(ax, 5.3, 11.75, r"$\{\hat{\mathbf{r}}_{ij}\}$", fs=8, ha="left")
    arrow(ax, (8.0, 12.25), (8.0, ye + 0.42)); label(ax, 8.3, 11.9, r"$\{|\mathbf{r}_{ij}|\}$", fs=8, ha="left")
    box(ax, 2.0, ye, 2.6, 0.8, "원소 표현\n(원소 종류 → 벡터)")
    box(ax, 5.0, ye, 2.4, 0.8, "방향 표현\n$Y_l(\\hat{\\mathbf{r}}_{ij}),\\ l\\leq 3$")
    box(ax, 8.0, ye, 3.2, 0.8, "거리 표현\n(Bessel + 고차 진동 억제, 원소 껍질 채널)", fc=NEW, ec=NEWEC, fs=6.8)
    label(ax, 2.0, 10.45, r"$h_i^{(0)}$", fs=8.5)
    label(ax, 5.35, 10.45, r"$Y(\hat{\mathbf{r}}_{ij})$", fs=8.5, color=BLUE)
    label(ax, 8.45, 10.45, r"$\tilde{e}_{ij}$", fs=8.5, color=RED)

    # atomic basis box (new): A_i
    ya = 9.45
    arrow(ax, (2.0, ye - 0.4), (2.0, ya + 0.55))
    line(ax, [(5.0, ye - 0.4), (5.0, ya + 0.9)], color=BLUE)      # Y line continues down
    arrow(ax, (5.0, ya + 0.9), (3.65, ya + 0.5), color=BLUE)
    line(ax, [(8.0, ye - 0.4), (8.0, ya + 0.9)], color=RED)
    arrow(ax, (8.0, ya + 0.9), (3.65, ya + 0.3), color=RED)
    box(ax, 2.0, ya, 3.3, 1.0, "원자 주변 요약 $A_i$\n(이웃 정보를 한 번 모은 벡터)", fc=NEW, ec=NEWEC, fs=8.5)
    # x projections: per layer Lin_t + hill
    xg = 0.75
    line(ax, [(2.0 - 1.65, ya), (0.25, ya)], color=GREEN)
    line(ax, [(0.25, ya), (0.25, 8.36)], color=GREEN)
    box(ax, 0.95, 8.0, 1.75, 0.72, "$x_i^{(t)}$ = $A_i$의 변형\n(크기 조절)", fc=NEW, ec=NEWEC, fs=7.0)
    line(ax, [(0.25, 8.36), (0.25, 1.95)], color=GREEN)

    # layers
    n_layers = 5
    y_top = 7.3
    dy = 1.25
    ys = [y_top - k * dy for k in range(4)]     # layer1, layer2, dots, layer eta
    names = ["Layer 1", "Layer 2", None, "Layer η = 5"]
    label(ax, 2.0, ya - 0.95, r"$h_i^{(0)}$", fs=8.5)
    arrow(ax, (2.0, ya - 0.55), (2.0, ys[0] + 0.33))
    # vertical Y / R lines on the right
    line(ax, [(5.0, ya + 0.9), (5.0, ys[3] + 0.12)], color=BLUE)
    line(ax, [(8.0, ya + 0.9), (8.0, ys[3] - 0.12)], color=RED)
    for k, (y, nm) in enumerate(zip(ys, names)):
        if nm is None:
            label(ax, 2.0, y + 0.2, "⋮", fs=11); label(ax, 3.55, y + 0.2, "⋮", fs=11)
            continue
        box(ax, 2.0, y, 2.4, 0.66, nm, bold=True, fs=8.5)
        # inputs from Y and R
        arrow(ax, (5.0, y + 0.12), (3.2, y + 0.12), color=BLUE)
        arrow(ax, (8.0, y - 0.12), (3.2, y - 0.12), color=RED)
        # x input from the green rail
        arrow(ax, (0.25, y), (0.8, y), color=GREEN)
        # outputs
        if k < 3:
            arrow(ax, (2.0, y - 0.33), (2.0, y - dy + 0.33 if k < 2 else y - dy + 0.55))
        if k == 3:
            arrow(ax, (2.0, y - 0.33), (2.0, y - 0.9))
        arrow(ax, (2.55, y - 0.33), (2.55, y - 0.75), connection="angle,angleA=90,angleB=0,rad=0")
        line(ax, [(2.55, y - 0.75), (3.3, y - 0.75)])
        arrow(ax, (3.3, y - 0.75), (3.55, y - 0.75))
        t = {0: "1", 1: "2", 3: "5"}[k]
        label(ax, 3.65, y - 0.75, f"$E_i^{{({t})}}$", fs=8.5, ha="left")
        if k < 2:
            label(ax, 2.3, y - 0.72, f"$h_i^{{({t})}}$", fs=8, ha="right")
    # Chebyshev skip: h^(t-1) feeds layer t+1 (drawn as dashed loops on the left)
    for k in (0, 1):
        y0 = ys[k] - 0.33; y1 = ys[k + 2] + 0.33 if k == 0 else ys[3] + 0.33
        arrow(ax, (1.25, y0 - 0.1), (1.25, y1 + 0.05), color=PURPLE, ls="--", lw=1.0,
              connection="arc3,rad=0.0")
    label(ax, 1.15, ys[1] - 1.0, "두 층 전\n특징 빼기", fs=6.8, color=PURPLE, ha="right")
    label(ax, 2.0, ys[3] - 0.95, r"$h_i^{(5)}$ (미사용)", fs=7.5, color="#888888")

    # red rail ends at the last layer; a branch continues to the repulsion prior
    # pair repulsion prior (new), bottom right
    yr = 1.5
    line(ax, [(8.0, ys[3] - 0.12), (8.0, yr + 0.8)], color=RED)
    label(ax, 8.15, yr + 1.0, r"$\{|\mathbf{r}_{ij}|, Z_i, Z_j\}$", fs=7, ha="left", color=RED)
    box(ax, 8.2, yr, 3.2, 1.3, "UFF 반발력 $E_i^{\\rm rep}$\n(형태 고정, 결합 안 한 쌍만;\n전역 배율 s 공유)", fc=NEW, ec=NEWEC, fs=8.0)

    # energy sum
    ysum = 1.5
    box(ax, 3.3, ysum, 5.0, 1.3, "총 에너지  $E=\\Sigma_i\\,(E_0+E_i^{\\rm rep}+\\Sigma_t\\,E_i^{(t)})$\n힘  $\\mathbf{F}=-\\nabla E$ (자동미분)", fc="#e8f1fb", ec="#5b8fc9", fs=8.8)
    # strata arrows into sum
    for k in (0, 1, 3):
        y = ys[k]
        line(ax, [(3.95, y - 0.75), (4.3, y - 0.75)], color="#444444")
    line(ax, [(4.3, ys[0] - 0.75), (4.3, ysum + 0.9)], color="#444444")
    arrow(ax, (4.3, ysum + 1.1), (4.3, ysum + 0.65), color="#444444")
    arrow(ax, (6.6, yr), (5.8, yr), color=NEWEC)
    label(ax, 6.2, yr + 0.25, r"$E_i^{\rm rep}$", fs=8, color=NEWEC)
    label(ax, 4.5, 2.75, r"$\sum_t E_i^{(t)}$", fs=8, ha="left", color="#444444")

    # training note
    label(ax, 5.0, 0.45, "학습: SOAP 옵티마이저 · Huber 손실 · EMA · 학습 후 에너지 상수 보정", fs=7.2, color="#555555")
    # legend of colours
    label(ax, 0.3, 0.05, "회색: 원자 특징 h   초록: 원자 주변 요약 x   파랑: 방향 $Y$   빨강: 거리 $\\tilde e$   보라: 두 층 전 특징   노랑 상자: v2에서 추가/변경",
          fs=6.4, color="#444444", ha="left")


# ----------------------------------------------------------------------------------------------- panel c
def panel_layer(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 15.2); ax.set_aspect("equal"); ax.axis("off")
    label(ax, 0.25, 14.95, "c", fs=13, bold=True, ha="left")
    label(ax, 5.0, 14.9, "상호작용 층 t  (최종 모델)", fs=10, bold=True)

    # rails
    line(ax, [(0.6, 14.3), (0.6, 0.9)], color=GREEN)
    line(ax, [(8.6, 14.3), (8.6, 0.9)], color=BLUE)
    line(ax, [(9.3, 14.3), (9.3, 0.9)], color=RED)
    label(ax, 0.6, 14.5, r"$x_i^{(t)}$", fs=8, color=GREEN)
    label(ax, 8.6, 14.5, r"$Y_{ij}$", fs=8, color=BLUE)
    label(ax, 9.3, 14.5, r"$\tilde e_{ij}$", fs=8, color=RED)

    # layer frame
    ax.add_patch(FancyBboxPatch((1.3, 1.05), 6.7, 12.45, boxstyle="round,pad=0,rounding_size=0.25",
                                fc="none", ec="#888888", lw=1.0, ls=(0, (4, 3)), zorder=1))
    label(ax, 1.45, 13.3, "Layer t", fs=8.5, bold=True, ha="left")

    xm = 5.0      # main column
    xs = 2.6      # skip column
    # input
    label(ax, xm, 14.25, r"$h_i^{(t-1)}$", fs=9)
    arrow(ax, (xm, 14.0), (xm, 12.95))
    line(ax, [(xm, 13.6), (xs, 13.6)]); arrow(ax, (xs, 13.6), (xs, 12.95))
    # skip (species-wise linear)
    box(ax, xs, 12.5, 1.9, 0.85, "자기 자신 항\n(원소별 Linear)", fs=7.3)
    # main: Linear
    box(ax, xm, 12.5, 1.5, 0.75, "Linear", fs=8.5)
    arrow(ax, (xm, 12.12), (xm, 11.55))
    # ⊗ Y
    opnode(ax, xm, 11.3, "⊗"); arrow(ax, (8.6, 11.3), (xm + 0.22, 11.3), color=BLUE)
    label(ax, 6.8, 11.55, "방향 정보 결합", fs=7.5, color=BLUE)
    arrow(ax, (xm, 11.08), (xm, 10.55))
    # ⊙ R(r)
    opnode(ax, xm, 10.3, "⊙"); arrow(ax, (9.3, 10.3), (xm + 0.22, 10.3), color=RED)
    label(ax, 6.9, 10.55, "거리 가중치 $R_t(r_{ij})$", fs=7.5, color=RED)
    arrow(ax, (xm, 10.08), (xm, 9.55))
    # aggregation
    opnode(ax, xm, 9.3, "Σ"); label(ax, xm + 0.35, 9.3, "이웃 합 ÷ 평균 이웃 수", fs=7.5, ha="left", bg=NEW)
    arrow(ax, (xm, 9.08), (xm, 8.55))
    label(ax, xm - 0.25, 8.8, r"$m_i^{(t)}$", fs=7.5, ha="right")
    # ⊗ x (channelwise product with the atomic-basis factor)
    opnode(ax, xm, 8.3, "⊗"); arrow(ax, (0.6, 8.3), (xm - 0.22, 8.3), color=GREEN)
    label(ax, 2.9, 8.55, "원자 주변 요약 $x_i$ 곱", fs=7.5, color=GREEN, bg=NEW)
    arrow(ax, (xm, 8.08), (xm, 7.55))
    # Linear
    box(ax, xm, 7.2, 1.5, 0.7, "Linear", fs=8.5)
    arrow(ax, (xm, 6.85), (xm, 6.35))
    # ⊕ skip
    opnode(ax, xm, 6.1, "⊕")
    line(ax, [(xs, 12.07), (xs, 6.1)]); arrow(ax, (xs, 6.1), (xm - 0.22, 6.1))
    label(ax, xm + 0.3, 6.1, "더하기 (비선형 없음)", fs=7.5, ha="left", bg=NEW)
    arrow(ax, (xm, 5.88), (xm, 5.4))
    label(ax, xm - 0.25, 5.65, r"$\delta_i^{(t)}$", fs=7.5, ha="right")
    # LayerScale
    opnode(ax, xm, 5.15, "⊙"); label(ax, xm + 0.3, 5.15, "증분 배율 γ (LayerScale)", fs=7.5, ha="left", bg=NEW)
    arrow(ax, (xm, 4.93), (xm, 4.45))
    # Chebyshev
    opnode(ax, xm, 4.2, "⊖", color=PURPLE)
    arrow(ax, (1.8, 4.2), (xm - 0.22, 4.2), color=PURPLE, ls="--", lw=1.0)
    label(ax, 1.75, 4.45, r"$h_i^{(t-2)}$", fs=7.5, color=PURPLE, ha="left")
    label(ax, xm + 0.3, 4.2, "2×증분 − 두 층 전 특징 (Chebyshev)", fs=7.2, ha="left", color=PURPLE, bg=NEW)
    arrow(ax, (xm, 3.98), (xm, 0.85))
    label(ax, xm, 0.6, r"$h_i^{(t)}$", fs=9)
    # readout branch
    line(ax, [(xm, 3.4), (6.9, 3.4)]); arrow(ax, (6.9, 3.4), (6.9, 3.15))
    box(ax, 6.9, 2.85, 1.4, 0.5, "Linear", fs=7.8)
    arrow(ax, (6.9, 2.6), (6.9, 2.45)); box(ax, 6.9, 2.2, 1.4, 0.5, "Gate", fs=7.8)
    arrow(ax, (6.9, 1.95), (6.9, 1.8)); box(ax, 6.9, 1.55, 1.4, 0.5, "Linear", fs=7.8)
    arrow(ax, (6.9, 1.3), (6.9, 0.85))
    label(ax, 6.9, 0.6, r"$E_i^{(t)}$ (이 층의 에너지 조각)", fs=8)
    label(ax, 7.65, 3.25, "비선형은 여기에만", fs=6.3, ha="left", color="#555555")

    # legend
    lx, ly = 1.5, 2.95
    ax.add_patch(FancyBboxPatch((lx - 0.15, ly - 1.25), 3.3, 1.55, boxstyle="round,pad=0,rounding_size=0.12",
                                fc="white", ec="#aaaaaa", lw=0.8, zorder=3))
    for k, (s, t) in enumerate([("⊗", "방향 정보 결합(텐서곱)"), ("⊙", "곱"), ("Σ", "이웃 합"), ("⊕", "합"), ("⊖", "두 층 전 빼기")]):
        opnode(ax, lx + 0.1, ly + 0.05 - 0.28 * k, s, r=0.11, fs=6.5)
        label(ax, lx + 0.35, ly + 0.05 - 0.28 * k, t, fs=6.5, ha="left")


def main():
    global LANG
    import sys
    if "--lang" in sys.argv:
        LANG = sys.argv[sys.argv.index("--lang") + 1]
    fig, (axb, axc) = plt.subplots(1, 2, figsize=(14.5, 11.0), gridspec_kw=dict(wspace=0.02))
    panel_pipeline(axb)
    panel_layer(axc)
    fig.suptitle(T("RACEv2 최종 모델 구조 (v1 논문 Figure 1 b·c 구도)"), fontsize=12, fontweight="bold", y=0.985)
    suffix = "_en" if LANG == "en" else ""
    for ext in ("png", "pdf"):
        out = os.path.join(DOCS, f"RACEv2_architecture{suffix}.{ext}")
        fig.savefig(out, dpi=220 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        print("written", out)


if __name__ == "__main__":
    main()
