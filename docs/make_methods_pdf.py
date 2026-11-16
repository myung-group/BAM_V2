"""RACEv2 methods & results summary -> docs/RACEv2_methods_summary.pdf

Only the FINAL model: its methods (equations + function plots) and its results against the corrected
state-of-the-art references (benchmark audit 2026-11-03).  No intermediate recipes.
All numbers come from docs/methods_pdf_data.json — edit that file and rerun:

    ~/.conda/envs/bam_cue/bin/python docs/make_methods_pdf.py
"""
import os, re, json, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph as _Paragraph, Spacer, Image, Table, TableStyle)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
FONT = os.path.join(DOCS, "fonts", "NanumGothic-Regular.ttf")
FONTB = os.path.join(DOCS, "fonts", "NanumGothic-Bold.ttf")
DEJAVU = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
OUT_DIR = os.path.join(DOCS, "methods_pdf_assets")
os.makedirs(OUT_DIR, exist_ok=True)
D = json.load(open(os.path.join(DOCS, "methods_pdf_data.json")))
DATE = datetime.date.today().isoformat()
PDF = os.path.join(DOCS, "RACEv2_methods_summary.pdf")

pdfmetrics.registerFont(TTFont("Nanum", FONT))
pdfmetrics.registerFont(TTFont("NanumB", FONTB))
pdfmetrics.registerFont(TTFont("DejaVu", DEJAVU))
font_manager.fontManager.addfont(FONT); font_manager.fontManager.addfont(FONTB)
plt.rcParams["font.family"] = ["NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

SYM_RE = re.compile("[\u00c0-\u00ff\u1e00-\u1eff\u02b0-\u02ff\u1d00-\u1dbf\u2c7c\u2010-\u2027Ͱ-Ͽ←-⇿∀-⋿⌀-⏿①-⑳⟨⟩·×±²³⁰-₟≤≥≈√−]+")


def sym(text):
    return SYM_RE.sub(lambda m: '<font name="DejaVu">' + m.group(0) + "</font>", str(text))


def Paragraph(text, style):
    return _Paragraph(sym(text), style)


H1 = ParagraphStyle("H1", fontName="NanumB", fontSize=16, leading=22, spaceBefore=8, spaceAfter=6)
H2 = ParagraphStyle("H2", fontName="NanumB", fontSize=12.5, leading=17, spaceBefore=12, spaceAfter=4, textColor=colors.HexColor("#1a3a6b"))
P = ParagraphStyle("P", fontName="Nanum", fontSize=9.6, leading=14.5, spaceAfter=4)
PS = ParagraphStyle("PS", fontName="Nanum", fontSize=8.3, leading=12, textColor=colors.HexColor("#444444"))
CAP = ParagraphStyle("CAP", fontName="Nanum", fontSize=8.3, leading=11.5, textColor=colors.HexColor("#333333"), spaceBefore=2, spaceAfter=8)
TH = ParagraphStyle("TH", fontName="NanumB", fontSize=8.0, leading=10.5, alignment=1)
TD = ParagraphStyle("TD", fontName="Nanum", fontSize=8.0, leading=10.5, alignment=1)
TDL = ParagraphStyle("TDL", fontName="Nanum", fontSize=8.0, leading=10.5, alignment=0)
BOX = ParagraphStyle("BOX", fontName="Nanum", fontSize=9.2, leading=13.5, backColor=colors.HexColor("#f3f6fb"),
                     borderColor=colors.HexColor("#9fb3d1"), borderWidth=0.6, borderPadding=6, spaceBefore=6, spaceAfter=10)
W = A4[0] - 36 * mm


def eq(tex, name, fs=13):
    fig = plt.figure(figsize=(7.2, 0.6))
    fig.text(0.5, 0.5, tex, ha="center", va="center", fontsize=fs)
    path = os.path.join(OUT_DIR, f"eq_{name}.png")
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.04, transparent=True); plt.close(fig)
    from PIL import Image as PIL
    w, h = PIL.open(path).size
    scale = min(W * 0.9 / w, 0.9 * 72 / 220 * 1.2)
    return Image(path, width=w * scale, height=h * scale)


def png_image(path, width=W):
    from PIL import Image as PIL
    w, h = PIL.open(path).size
    return Image(path, width=width, height=width * h / w)


def fig_image(fig, name, width=W, dpi=200):
    path = os.path.join(OUT_DIR, f"fig_{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    from PIL import Image as PIL
    w, h = PIL.open(path).size
    return Image(path, width=width, height=width * h / w)


RED = '<font color="#c00000">'


def table(rows, col_widths=None, header_rows=1, hi_rows=(), red_rows=()):
    data = [[Paragraph((RED + str(c) + "</font>") if i in red_rows else str(c), TH if i < header_rows else (TDL if j == 0 else TD)) for j, c in enumerate(r)] for i, r in enumerate(rows)]
    t = Table(data, colWidths=col_widths, repeatRows=header_rows)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#aaaaaa")),
          ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#dfe7f3")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    for i in range(header_rows, len(rows)):
        if (i - header_rows) % 2 == 1:
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f7f7f7")))
    for i in hi_rows:
        st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#e6f4e6")))
    t.setStyle(TableStyle(st))
    return t


def ef(v):
    if v is None:
        return "–"
    e = "–" if v[0] is None else f"{v[0]:.2f}"
    f = "–" if v[1] is None else f"{v[1]:.2f}"
    return f"{e} / {f}"


# ----------------------------------------------------------------------------- physics
def polynomial_cutoff(r, rc, p=2.0):
    x = r / rc
    return (1 - (p + 1) * (p + 2) / 2 * x ** p + p * (p + 2) * x ** (p + 1) - p * (p + 1) / 2 * x ** (p + 2)) * (x < 1)


def zbl(z1, z2, r):
    a = 0.46850 / (z1 ** 0.23 + z2 ** 0.23); x = r / a
    phi = 0.18175 * np.exp(-3.19980 * x) + 0.50986 * np.exp(-0.94229 * x) + 0.28022 * np.exp(-0.40290 * x) + 0.02817 * np.exp(-0.20162 * x)
    return 14.399645 * z1 * z2 / r * phi


K = 0.0433641
UFF = {"H": (2.886, 0.044 * K, 12.0), "C": (3.851, 0.105 * K, 12.73), "N": (3.660, 0.069 * K, 13.407), "O": (3.500, 0.060 * K, 14.085)}
RCOV = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66}
Z = {"H": 1, "C": 6, "N": 7, "O": 8}


def uff(a, b, r):
    x = np.sqrt(UFF[a][0] * UFF[b][0]); d = np.sqrt(UFF[a][1] * UFF[b][1]); z = np.sqrt(UFF[a][2] * UFF[b][2])
    return d * 6 / (z - 6) * np.exp(z * (1 - r / x))


def switch(a, b, r, c, w):
    return 1 / (1 + np.exp(-(r - c * (RCOV[a] + RCOV[b])) / w))


# ----------------------------------------------------------------------------- figures
def fig_v1v2():
    b = D["bpa"]; keys = ["300K", "600K", "1200K"]; labels = ["300 K", "600 K", "1200 K"]
    refs = [("GRACE (PRX 2024, 5 fits)", "#2e8b57", "s"), ("PACE (TMLR 2024, 3 seeds)", "#8e44ad", "D"),
            ("TACE spe-cart ν=3 (1 seed)", "#e08e0b", "^"), ("MACE L=2", "#d9534f", "o")]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8)); xs = np.arange(3); wd = 0.3
    for i, (ax, ttl, unit) in enumerate(zip(axs, ["Energy RMSE", "Force RMSE"], ["meV/molecule", "meV/Å"])):
        ax.bar(xs - wd / 2, [b["V1"][k][i] for k in keys], wd, label="RACE V1 (paper)", color="#9e9e9e")
        ax.bar(xs + wd / 2, [b["V2"][k][i] for k in keys], wd, label="RACEv2 (final)", color="#0b3d91")
        for (name, col, mk) in refs:
            ax.scatter(xs + wd / 2 + 0.22, [b["refs"][name][k][i] for k in keys], marker=mk, color=col, zorder=5, s=30, label=name.split(" (")[0])
        ax.set_xticks(xs); ax.set_xticklabels(labels); ax.set_ylabel(f"{ttl} ({unit})"); ax.set_title(f"3BPA {ttl}: V1 → V2 and the best literature values", fontsize=10)
        ax.set_ylim(0, None); ax.grid(axis="y", alpha=0.3)
        if i == 0: ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    fig.tight_layout(); return fig_image(fig, "v1v2")


def fig_bessel():
    rc = 5.0; r = np.linspace(0.05, rc, 500); sig = 0.224; fc = polynomial_cutoff(r, rc)
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for n in range(1, 9):
        bnr = 2 / rc * np.sin(n * np.pi * r / rc) / r * fc
        axs[0].plot(r, bnr, lw=1, label=f"n={n}" if n in (1, 4, 8) else None); axs[1].plot(r, bnr * np.exp(-sig ** 2 * n ** 2), lw=1)
    axs[0].set_title("Bessel basis × f(cut) (n=1..8, r_c=5 Å)"); axs[0].legend(fontsize=8)
    axs[1].set_title("After × exp(−σₙ² n²) (σₙ=0.224)")
    n = np.arange(1, 9); axs[2].bar(n, np.exp(-sig ** 2 * n ** 2), color="#2b6cb0"); axs[2].set_title("Reduction factor exp(−σₙ² n²)"); axs[2].set_ylim(0, 1.05)
    for a in axs: a.grid(alpha=0.3); a.set_xlabel("r (Å)" if a is not axs[2] else "Bessel order n")
    fig.tight_layout(); return fig_image(fig, "bessel")


def fig_shells():
    B = 0.529177
    rm = {"H 1s": 1.6733 * B, "C 2s": 1.5838 * B, "C 2p": 1.7942 * B, "N 2p": 1.4814 * B, "O 2p": 1.2645 * B}
    r = np.linspace(0.3, 5.0, 400); fc = polynomial_cutoff(r, 5.0)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for a, b in [("H 1s", "H 1s"), ("H 1s", "C 2p"), ("C 2p", "C 2p"), ("H 1s", "O 2p"), ("N 2p", "C 2p")]:
        rho = rm[a] + rm[b]; ax.plot(r, np.exp(-r / rho) * fc, label=f"{a}–{b} (ρ(a)+ρ(b) = {rho:.2f} Å)")
    ax.set_xlabel("r (Å)"); ax.set_ylabel("exp(−r/(ρ(a)+ρ(b)))·f(cut)"); ax.set_title("Shell radial function S(a,b)(r) per element pair"); ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
    return fig_image(fig, "shells", width=W * 0.72)


def fig_hill():
    rr = np.logspace(-2, 1.5, 300)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for a, b, lab in [(0, 0, "a=0, b=0 (pure RMS normalisation)"), (0.5, 0, "a=0.5, b=0"), (1, 0, "a=1, b=0"), (1, -1, "a=1, b=−1"), (1, 1, "a=1, b=+1")]:
        ax.semilogx(rr, 2 / (1 + np.exp(-(a * np.log(rr) + b))), label=lab)
    ax.set_xlabel("rᵢ = mean magnitude of xᵢ (RMS)"); ax.set_ylabel("Scale 2σ(a·log rᵢ + b)"); ax.set_title("Magnitude-scaling factor (output magnitude ≤ 2)")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
    return fig_image(fig, "hill", width=W * 0.72)


def fig_cheb():
    x = np.linspace(-1, 1, 400)
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.0))
    for n in range(5):
        axs[0].plot(x, x ** n, label=f"x^{n}"); axs[1].plot(x, np.cos(n * np.arccos(x)), label=f"T_{n}(x)")
    axs[0].set_title("Ordinary layer stacking xⁿ: more alike with depth"); axs[1].set_title("Chebyshev Tₙ: a different shape per layer")
    for a in axs: a.legend(fontsize=7.5, ncol=2); a.grid(alpha=0.3)
    fig.tight_layout(); return fig_image(fig, "cheb")


def fig_repulsion():
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.0))
    pairs = [("H", "H", 1.5, 2.13, "H–H: collision 1.5 Å, training minimum 2.13 Å"), ("C", "H", 2.13, 2.60, "C–H: collision 2.13 Å, training minimum 2.60 Å"), ("C", "N", 2.59, 3.05, "C–N: collision 2.59 Å, training minimum 3.05 Å")]
    for k, (a, b, rcol, rmin, title) in enumerate(pairs):
        r = np.linspace(0.8, 4.0, 600); fc = polynomial_cutoff(r, 5.0)
        ez = zbl(Z[a], Z[b], r) * fc; eu = uff(a, b, r) * fc; s1 = switch(a, b, r, 1.25, 0.1); s3 = switch(a, b, r, 1.25, 0.3)
        ax = axs[0, k]
        ax.plot(r, ez * 1e3, "k:", lw=1, label="ZBL original"); ax.plot(r, ez * s1 * 1e3, color="#2b6cb0", label="ZBL-nb (c=1.25, w=0.1 Å)")
        ax.plot(r, eu * 1e3, color="#e08e0b", ls=":", lw=1, label="UFF original"); ax.plot(r, eu * s3 * 1e3, color="#d9534f", lw=2, label="UFF-nb w0.3 (final)")
        ax.axvline(rcol, color="gray", ls=":", lw=1); ax.axvline(rmin, color="gray", ls="-.", lw=1)
        ax.set_ylim(0, 1500); ax.set_title(title, fontsize=9.5); ax.set_ylabel("Repulsion energy (meV)"); ax.grid(alpha=0.3)
        if k == 0: ax.legend(fontsize=7)
        ax = axs[1, k]
        for e, lab, col, ls, lw in [(ez, "ZBL", "k", ":", 1), (ez * s1, "ZBL-nb", "#2b6cb0", "-", 1.2), (eu, "UFF", "#e08e0b", ":", 1), (eu * s3, "UFF-nb w0.3", "#d9534f", "-", 2)]:
            ax.plot(r, -np.gradient(e * 1e3, r), color=col, ls=ls, lw=lw, label=lab)
        ax.axvline(rcol, color="gray", ls=":", lw=1); ax.axvline(rmin, color="gray", ls="-.", lw=1)
        ax.set_ylim(0, 5000); ax.set_xlabel("r (Å)"); ax.set_ylabel("Repulsive force −dE/dr (meV/Å)"); ax.grid(alpha=0.3)
    fig.suptitle("Short-range repulsion prior: dotted line = dihedral collision distance, dash-dotted line = minimum distance of the training set", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); return fig_image(fig, "repulsion")


def fig_switch():
    r = np.linspace(0.5, 3.0, 400)
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for (a, b), col in [(("C", "H"), "#2b6cb0"), (("H", "H"), "#d9534f"), (("C", "N"), "#2e8b57")]:
        ax.plot(r, switch(a, b, r, 1.25, 0.3), color=col, label=f"{a}–{b}, w=0.3 Å (final)"); ax.plot(r, switch(a, b, r, 1.25, 0.1), color=col, ls=":", label=f"{a}–{b}, w=0.1 Å")
    ax.axvspan(1.0, 1.4, color="orange", alpha=0.15, label="C–H bonds stretched at 1200 K")
    ax.set_title("Non-bonded switch σ((r − 1.25(Rᵢ+Rⱼ))/w)"); ax.set_xlabel("r (Å)"); ax.legend(fontsize=6.8, ncol=2); ax.grid(alpha=0.3)
    return fig_image(fig, "switch", width=W * 0.72)


def fig_rmd17():
    m = D["rmd17"]; mols = m["mols"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4)); xs = np.arange(len(mols)); w = 0.27
    for ax, i, t in [(axs[0], 0, "energy MAE (meV)"), (axs[1], 1, "force MAE (meV/Å)")]:
        v1 = [m["V1"].get(k, [np.nan, np.nan])[i] for k in mols]
        ax.bar(xs - w, v1, w, label="RACE V1", color="#9e9e9e"); ax.bar(xs, [m["V2"][k][i] for k in mols], w, label="RACEv2", color="#0b3d91")
        ax.bar(xs + w, [m["MACE"][k][i] for k in mols], w, label="MACE", color="#d9534f")
        ax.set_xticks(xs); ax.set_xticklabels([k[:7] for k in mols], fontsize=8); ax.set_title("rMD17 " + t); ax.grid(axis="y", alpha=0.3)
    axs[0].legend(fontsize=8); fig.tight_layout(); return fig_image(fig, "rmd17")


# ----------------------------------------------------------------------------- document
def build():
    b = D["bpa"]; s = []
    s.append(Paragraph("RACEv2: methods and results of the final model", H1))
    s.append(Paragraph(f"Updated {DATE} (numbers as of {D['date_note']}). The numbers are read from docs/methods_pdf_data.json. "
                       "RMSE by default; MAE for rMD17, LiPS and QM9. Energies are after the post-training constant refit (δ* in Section 4), and where the effect is large the raw value is also given in parentheses. "
                       "Reference values are only from models with a fair protocol (no pretraining, no extra data), verified against the original papers in the benchmark audit of 2026-11-03.", PS))

    s.append(Paragraph("0. At a glance", H2))
    s.append(fig_v1v2())
    V1, V2 = b["V1"], b["V2"]
    s.append(Paragraph(f"Figure 1. 3BPA (trained on 500 frames at 300 K). V1 → V2: 1200 K force {V1['1200K'][1]} → {V2['1200K'][1]} meV/Å (−{100*(1-V2['1200K'][1]/V1['1200K'][1]):.0f} %), "
                       f"300 K energy {V1['300K'][0]} → {V2['300K'][0]} meV (−{100*(1-V2['300K'][0]/V1['300K'][0]):.0f} %). All 6 temperature metrics are lower than GRACE, PACE and MACE.", CAP))
    s.append(Paragraph("<b>The final model in one line.</b> A rotation-equivariant neural network that exchanges neighbour information 5 times (128 channels, angular information l ≤ 3, neighbours within 5 Å) + an energy stratum from every layer, summed "
                       "+ the message of every layer multiplied by the 'atomic-environment summary', which raises the number of atoms that take part simultaneously (with magnitude scaling) + a Chebyshev-type update that subtracts the features of two layers before + an increment scale (LayerScale) "
                       "+ suppression of the high-order oscillations of the radial functions + radial channels that reflect the element shell sizes + UFF repulsion for closely approaching non-bonded atom pairs (fixed form, shared global scale), SOAP optimiser (lr 1e-2) + post-training energy-constant refit, 64-bit arithmetic. "
                       f"{D['final_recipe']['selection_note']}", BOX))
    s.append(Paragraph("<b>Glossary.</b> "
                       "<b>Energy stratum</b>: the energy contribution of each layer. The total energy is the sum of these strata. "
                       "<b>Atomic-environment summary Aᵢ</b>: a vector that collects the species, distances and directions of the neighbours around the central atom in one pass (layer-0 aggregation). "
                       "<b>Product with the atomic-environment summary (xᵢ product)</b>: multiplying the message of each layer by xᵢ, a transform of Aᵢ. Each product raises by one step 'up to how many simultaneously interacting atoms the terms can represent'. "
                       "<b>Magnitude scaling (hill)</b>: brings the magnitude of xᵢ to a uniform level, but keeps, within an upper bound, the information on whether the neighbours are dense or sparse. "
                       "<b>Chebyshev-type update</b>: new layer features = 2×(increment of this layer) − (features of two layers before). The layers then carry less overlapping information. "
                       "<b>Increment scale (LayerScale)</b>: a learnable scale that multiplies the layer increment (initial 0.5). "
                       "<b>Fixed term (prior)</b>: a term that is not learned from data but set in advance by a physical formula. "
                       "<b>Energy-constant refit (gauge refit)</b>: a per-atom constant energy does not affect the forces and is therefore not determined by training, so it is fitted in one step on the training set after training.", BOX))
    s.append(png_image(os.path.join(DOCS, "RACEv2_architecture_en.png"), width=W))
    s.append(Paragraph("Figure 2. Architecture of the final model (layout of Figure 1 b, c of the v1 paper; docs/make_architecture_fig.py). b: the whole pipeline. Element, angular and radial representations → the atomic-environment summary Aᵢ, which collects the neighbour information once → "
                       "xᵢ, a per-layer transform of Aᵢ (green) → each of the 5 layers gives an energy stratum Eᵢ⁽ᵗ⁾, and these are summed together with the UFF repulsion Eᵢʳᵉᵖ. Forces are obtained by differentiating the energy with respect to the coordinates. "
                       "c: inside one layer. Yellow boxes are the parts added or changed relative to v1 (neighbour sum divided by the mean neighbour count, xᵢ product, addition without nonlinearity, increment scale, subtraction of the features of two layers before). "
                       "The nonlinearity is only in the small network that reads out the energy stratum (Linear–Gate–Linear).", CAP))

    s.append(Paragraph("1. Previous (V1) → current (V2): comparison of the changed equations", H2))
    s.append(Paragraph("The two models are written side by side with <b>the same symbols and the same steps</b>. The computation of one layer (the t-th) is divided into five steps, and at each step the left-hand side (what is computed) of the previous and the current equation is the same. "
                       "The previous equations are the V1 paper (Methods Eqs. 13–16) and the default settings of the V1 code (x_mode species, message_norm layer, update gate, readout linear, residual none) rewritten in the symbols of this document.", P))
    s.append(Paragraph("Symbols: hᵢ⁽ᵗ⁾ features of atom i after layer t (Aᵢ⁽η⁾ of the V1 paper), mᵢ⁽ᵗ⁾ message collected from the neighbours (Aᵢ⁽η⁻¹⁾ of V1 Eq. 15), xᵢ factor that multiplies the message (a⁽¹⁾ of V1), "
                       "δᵢ⁽ᵗ⁾ layer increment, Yᵢⱼ neighbour direction (spherical harmonics, eᵢⱼ of V1), Rₜ(rᵢⱼ) radial weight (M(ẽᵢⱼ) of V1), ⟨N⟩ mean neighbour count, Lin rotation-equivariant linear map, ⊗ tensor product, ⊙ channel-wise product.", P))
    rows = [["Step (what is computed)", "Previous (V1)", "Current (V2)"],
            ["① Message mᵢ⁽ᵗ⁾", "Neighbour sum after LayerNorm, ÷√⟨N⟩", "Neighbour sum without LayerNorm, ÷⟨N⟩"],
            ["② Multiplying factor xᵢ", "One constant per element a(Z) (shared by all layers)", "Per-layer linear map of the atomic-environment summary Aᵢ"],
            ["③ Magnitude of the factor x̂ᵢ", "No scaling", "Magnitude scaling (hill): scale 0–2"],
            ["④ Layer increment δᵢ⁽ᵗ⁾", "Lin(m⊗x) + self term", "Same (x̂ instead of x)"],
            ["⑤ Features of the next layer hᵢ⁽ᵗ⁾", "Gate(δ) (nonlinear)", "2γδ − h⁽ᵗ⁻²⁾ (linear, Chebyshev)"],
            ["⑥ Layer energy stratum Eᵢ⁽ᵗ⁾", "Lin(Lin(h)) (nonlinearity in ⑤)", "Lin(Gate(Lin(h))) (nonlinearity only here)"],
            ["⑦ Total energy E", "Σᵢ [E₀ + Σₜ Eᵢ⁽ᵗ⁾]", "Σᵢ [E₀ + Eᵢʳᵉᵖ + Σₜ Eᵢ⁽ᵗ⁾] + Nδ*"]]
    s.append(table(rows, col_widths=[W * 0.28, W * 0.36, W * 0.36]))
    s.append(Paragraph("Table 0. Step-by-step comparison of the computation of one layer. Sections 1-1 to 1-6 below give the equation of each step, previous first and then current.", CAP))

    s.append(Paragraph("<b>1-1. ① Message: collecting the neighbour information.</b>", P))
    s.append(eq(r"Previous:  $m_i^{(t)}=\frac{1}{\sqrt{\langle N\rangle}}\sum_{j\in N(i)} R_t(r_{ij})\,\left(W_t\,\mathrm{LN}(h_j^{(t-1)})\otimes Y_{ij}\right)$", "u_msg1", fs=12))
    s.append(eq(r"Current:  $m_i^{(t)}=\frac{1}{\langle N\rangle}\sum_{j\in N(i)} R_t(r_{ij})\,\left(W_t\,h_j^{(t-1)}\otimes Y_{ij}\right)$", "u_msg2", fs=12))
    s.append(Paragraph("Plain explanation: for each neighbour atom j, ① the features hⱼ are transformed by the matrix Wₜ, ② coupled with the neighbour direction Yᵢⱼ, ③ multiplied by the radial weight Rₜ(rᵢⱼ) and ④ all summed. Rₜ is a value that a small neural network produces from the radial representation ẽᵢⱼ. "
                       "<b>What changed:</b> LayerNorm (LN) is removed, which keeps the magnitude information of the features, and the sum is scaled by 1/⟨N⟩ instead of 1/√⟨N⟩, so that the message magnitude stays constant when the number of neighbours grows (less fluctuation when extrapolating to structures of a different density).", P))

    s.append(Paragraph("<b>1-2. ② Factor that multiplies the message.</b>", P))
    s.append(eq(r"Previous:  $x_i^{(t)}=a_{Z_i}$   (learned constant per element, shared by all layers)", "u_x1", fs=12))
    s.append(eq(r"Current:  $x_i^{(t)}=\mathrm{Lin}_t(A_i),\qquad A_i=\frac{1}{\langle N\rangle}\sum_{j\in N(i)} R_0(r_{ij})\,\left(W_0\,h_j^{(0)}\otimes Y_{ij}\right)$", "u_x2", fs=12))
    s.append(Paragraph("Plain explanation: the previous factor a(Z) is a constant determined by the atom species alone, so multiplying by it did not add information about the surroundings. The current Aᵢ is a vector that collects the neighbour information once at layer 0 (the atomic basis of ACE, the atomic-environment summary), and "
                       "it becomes xᵢ⁽ᵗ⁾ through a linear map Linₜ that differs per layer. Each time it multiplies the message, the number of neighbour sums in the expression grows by one, so that layer 1 represents 2-body terms, layer 2 3-body terms, … and layer 5 up to 6-body terms (the recursive product of ACE).", P))

    s.append(Paragraph("<b>1-3. ③ Magnitude scaling of the factor (hill).</b>", P))
    s.append(eq(r"Previous:  $\hat x_i^{(t)}=x_i^{(t)}$   (no scaling)", "u_hill1", fs=12))
    s.append(eq(r"Current:  $\hat x_i^{(t)}=\frac{x_i^{(t)}}{r_i}\cdot 2\,\sigma\left(a\log r_i+b\right),\qquad r_i=\mathrm{RMS}(x_i^{(t)}),\quad \sigma(u)=\frac{1}{1+e^{-u}}$", "u_hill2", fs=12))
    s.append(Paragraph("Plain explanation: the magnitude rᵢ of xᵢ grows when the neighbours are many and close. Multiplying by xᵢ as it is makes the values explode from layer to layer, and only dividing by the magnitude loses the information 'are the neighbours dense'. "
                       "Therefore xᵢ is divided by its magnitude and then multiplied by a scale between 0 and 2. a and b are two numbers learned per element and per layer. a=0 is the same as pure normalisation, and for a&gt;0 the scale grows with the density but does not exceed 2 (Figure 3).", P))
    s.append(fig_hill())
    s.append(Paragraph("Figure 3. Magnitude-scaling factor 2σ(a log rᵢ + b). For a=b=0 the magnitude information is lost completely; for a=1 the output follows the original magnitude at small rᵢ and stops at 2 at large rᵢ.", CAP))

    s.append(Paragraph("<b>1-4. ④ Layer increment.</b>", P))
    s.append(eq(r"Previous:  $\delta_i^{(t)}=\mathrm{Lin}\left(m_i^{(t)}\otimes x_i^{(t)}\right)+\mathrm{Lin}_{Z_i}\left(h_i^{(t-1)}\right)$", "u_d1", fs=12))
    s.append(eq(r"Current:  $\delta_i^{(t)}=\mathrm{Lin}\left(m_i^{(t)}\otimes \hat x_i^{(t)}\right)+\mathrm{Lin}_{Z_i}\left(h_i^{(t-1)}\right)$", "u_d2", fs=12))
    s.append(Paragraph("Plain explanation: the product of the message and the factor (channel-wise tensor product) is transformed linearly, and a self term with element-wise weights is added. The form of the equation is the same; only the multiplying factor is replaced by the one changed in ② and ③.", P))

    s.append(Paragraph("<b>1-5. ⑤ Features passed to the next layer.</b>", P))
    s.append(eq(r"Previous:  $h_i^{(t)}=\mathrm{Gate}\left(\delta_i^{(t)}\right)$", "u_h1", fs=12))
    s.append(eq(r"Current:  $h_i^{(t)}=2\,\gamma_t\odot\delta_i^{(t)}-h_i^{(t-2)}\ \ (t\geq 3),\qquad h_i^{(t)}=\gamma_t\odot\delta_i^{(t)}\ \ (t=1,2)$", "u_h2", fs=12))
    s.append(Paragraph("Plain explanation: previously the nonlinear Gate was applied to the increment and the result was passed to the next layer as it is. Now there is no nonlinearity, and the features of two layers before are subtracted from twice the increment. γₜ is a channel-wise scale (initial 0.5, learned). "
                       "When layers are simply stacked, shapes that resemble each other, like x, x², x³ …, overlap, and the per-layer energy strata repeat the same information. Following Tₙ₊₁ = 2x·Tₙ − Tₙ₋₁ (the Chebyshev recurrence) gives a different shape for every layer, so that each layer carries new information (Figure 4).", P))
    s.append(fig_cheb())
    s.append(Paragraph("Figure 4. Ordinary layer stacking (left) piles up similar shapes, like powers, whereas the Chebyshev-type update (right) gives a different shape for every layer.", CAP))

    s.append(Paragraph("<b>1-6. ⑥ Layer energy stratum and ⑦ total energy.</b>", P))
    s.append(eq(r"Previous:  $E_i^{(t)}=\mathrm{Lin}\left(\mathrm{Lin}(h_i^{(t)})\right),\qquad E=\sum_i\left[E_{0,Z_i}+\sum_{t=1}^{T}E_i^{(t)}\right]$", "u_E1", fs=12))
    s.append(eq(r"Current:  $E_i^{(t)}=\mathrm{Lin}\left(\mathrm{Gate}(\mathrm{Lin}(h_i^{(t)}))\right),\qquad E=\sum_i\left[E_{0,Z_i}+E_i^{\rm rep}+\sum_{t=1}^{T}E_i^{(t)}\right]+N\,\delta^{*}$", "u_E2", fs=12))
    s.append(Paragraph("Plain explanation: the rotation-invariant component (l=0) of the features h⁽ᵗ⁾ of each layer is read by a small network to give the energy stratum of that layer. Previously the nonlinearity was in ⑤ (between layers); now the path between layers is kept linear and the nonlinearity is only in this readout step. "
                       "Eᵢʳᵉᵖ (the UFF repulsion of Section 3) and δ* (the post-training constant refit of Section 4, multiplied by the number of atoms N) are added to the total. E₀ is the reference energy per element and T the number of layers (5). Forces are obtained by automatic differentiation of E with respect to the coordinates. In the implementation, the whole Eᵢʳᵉᵖ + Σₜ Eᵢ⁽ᵗ⁾ inside the brackets is multiplied by a learned global scale s and a per-atom constant is added (see Section 3).", P))

    s.append(Paragraph("2. Radial representation: Bessel functions + suppression of high-order oscillations + element shell-size channels", H2))
    s.append(eq(r"Previous:  $\tilde e_{ij}=[\,B_n(r_{ij})\,f_{\rm cut}(r_{ij})\,],\qquad n=1,\dots,8$", "v1_rad", fs=12))
    s.append(eq(r"Current:  $\tilde e_{ij}=[\,B_n(r_{ij})\,e^{-\sigma_n^2 n^2}f_{\rm cut}(r_{ij})\ \|\ S_{ab}(r_{ij})],\quad B_n(r)=\frac{2}{r_c}\frac{\sin(n\pi r/r_c)}{r},\quad S_{ab}(r)=e^{-r/(\rho_a(Z_i)+\rho_b(Z_j))}f_{\rm cut}(r)$", "v2_rad", fs=10.5))
    s.append(Paragraph("Plain explanation: before the distance rᵢⱼ enters the neural network, it is converted into a set of function values. Previously the 8 Bessel functions (the larger n, the faster the oscillation) were only multiplied by the cutoff function. "
                       "Now (i) the n-th function is multiplied by exp(−σₙ²n²), which reduces the fast-oscillating high-order functions (σₙ = 0.224, suppresses fluctuations shorter than about 0.5 Å). This keeps the high-order functions from oscillating freely at distances outside the training range and "
                       "reduces the force error of stretched bonds (1200 K) (a transfer of the method that Darby et al. 2026 used for linear ACE). (ii) An exponential function S(a,b), whose decay length is the sum of the outermost-shell radii ρ of the two atoms (values from atomic calculations, fixed in a table), is "
                       "appended as channels (‖ denotes concatenation). It gives each element pair its own 'sense of distance' on physical grounds. f(cut) = 1 − 6x² + 8x³ − 3x⁴ (x = r/r_c) is a function that goes smoothly to 0 at the cutoff.", P))
    s.append(fig_bessel())
    s.append(Paragraph("Figure 5. Left: the 8 Bessel functions. Centre: after suppression, the fast-oscillating high-order functions become small. Right: reduction factor per order.", CAP))
    s.append(fig_shells())
    s.append(Paragraph("Figure 6. Shell functions of the 3BPA element pairs. The decay length differs from pair to pair (H–H 1.8, C–C 1.9, N–C 1.7 Å, etc.).", CAP))

    s.append(Paragraph("3. Repulsion of closely approaching atom pairs: UFF formula, only for non-bonded pairs", H2))
    s.append(Paragraph("The error of the dihedral scan (constrained-optimisation structures) is concentrated in the 'collision' frames, where the two rings touch, and the model saw the force with which atoms repel each other 0.5 Å inside the minimum distance of the training set as too weak"
                       f" (repulsive-force slope {D['collision']['no_prior']['slope']}, energy bias {D['collision']['no_prior']['bias']} meV; the 3 seeds point in the same direction with 0.98 = systematic error). "
                       "The training data do not contain this information, so a repulsion term is added whose functional form and constants are taken unchanged from UFF. However, this term too is multiplied by the global energy scale s of the model (a single learned scalar, initial value 1). "
                       "After training s is 0.20–0.33 (3BPA 0.25, rMD17 0.20–0.23, AcAc 0.23, Si 0.30–0.33, LiPS 0.26), so the effective repulsion strength is about 1/4 of the original UFF. The accurate statement is therefore not 'no learned parameter at all' but 'fixed functional form, and the overall strength shares one global scale'.", P))
    s.append(eq(r"Previous:  $E_i^{\rm rep}=0$   (repulsion learned entirely from data)", "v1_rep", fs=12))
    s.append(eq(r"Current:  $E_i^{\rm rep}=\frac{1}{2}\sum_{j}D_{ij}\frac{6}{\zeta_{ij}-6}\exp\left[\zeta_{ij}\left(1-\frac{r_{ij}}{x_{ij}}\right)\right]f_{\rm cut}(r_{ij})\cdot\sigma\left(\frac{r_{ij}-1.25\,(R_i+R_j)}{0.3\,\mathrm{\AA}}\right)$", "v2_rep", fs=11.5))
    s.append(eq(r"$x_{ij}=\sqrt{x_i x_j},\quad D_{ij}=\sqrt{D_i D_j},\quad \zeta_{ij}=\sqrt{\zeta_i\zeta_j}$   (Rappé 1992 UFF table values, fixed)", "v2_rep2", fs=11.5))
    s.append(Paragraph("Plain explanation: the exponential part is the repulsion term of the UFF force field (Rappé 1992), which grows steeply as two atoms come closer than the equilibrium distance xᵢⱼ. Dᵢⱼ is the energy scale (tens of meV) and ζᵢⱼ the steepness (about 12–14), and "
                       "the per-element table values are turned into pair values by the geometric mean. f(cut) makes the term go to 0 at the cutoff. The last factor σ(·) is the key. For pairs joined by a covalent bond (pairs whose distance is shorter than 1.25 times the sum of the covalent radii Rᵢ+Rⱼ) "
                       "this repulsion is switched off smoothly with a width of 0.3 Å, so that the neural network learns the bond lengths themselves freely and the repulsion is switched on only when non-bonded atoms come abnormally close. The factor 1/2 corrects for counting each pair twice.", P))
    s.append(fig_repulsion())
    s.append(Paragraph("Figure 7. Repulsion energy (top, up to 1500 meV) and force (bottom, up to 5000 meV/Å; larger values are clipped) of the three colliding pairs. UFF-nb w0.3 (thick red) is small inside the training range, steep at the collision distance, and continues smoothly towards the bond lengths. "
                       "ZBL (the nuclear-repulsion formula) is drawn as well for comparison.", CAP))
    s.append(fig_switch())
    s.append(Paragraph("Figure 8. Switching region that turns the repulsion off for bonded pairs. With a width of 0.1 Å the bend overlaps the range of C–H bond lengths stretched at 1200 K (orange) and the force error grows; a width of 0.3 Å avoids this.", CAP))

    s.append(Paragraph("4. Training: SOAP optimiser, loss, energy-constant refit", H2))
    s.append(eq(r"Previous:  AMSGrad  $m\leftarrow\beta_1 m+(1-\beta_1)G,\ \ v\leftarrow\beta_2 v+(1-\beta_2)G^{2},\ \ \hat v\leftarrow\max(\hat v, v),\ \ \Delta\theta=-\eta\,\frac{m}{\sqrt{\hat v}+\epsilon}$", "v1_opt", fs=11.5))
    s.append(eq(r"Current:  SOAP  $L\leftarrow\beta L+(1-\beta)GG^{\top},\ R\leftarrow\beta R+(1-\beta)G^{\top}G,\ \ G'=Q_L^{\top}GQ_R,\ \ \Delta\theta=-\eta\,Q_L\left[\frac{\hat m'}{\sqrt{\hat v'}+\epsilon}\right]Q_R^{\top}$", "v2_opt", fs=11.5))
    s.append(Paragraph("Plain explanation: Adam-type optimisers set the step of each weight separately, dividing by its 'recent gradient magnitude'. SOAP (Vyas et al. 2024) tracks, for every weight matrix, the row correlation L and the column correlation R of the gradient G, "
                       "rotates the coordinates onto their principal axes (eigenvectors Q(L), Q(R)) (G'), applies Adam in those coordinates and rotates back. The effect is to disentangle the directions in which the weights are coupled, and this is supported by the control experiment (Table 1), in which Adam with the same β and only the rotation removed is "
                       "much worse. β₁=β₂=0.95, principal axes updated every 10 steps, lr 1e-2, EMA 0.99, 6.8 s/epoch with the batched implementation. The loss is Huber (δ=10), energy per atom, E:F = 1:10.", P))
    s.append(eq(r"Previous:  none.   Current:  $\delta^{*}=\arg\min_\delta\sum_n (d_n+N_n\delta)^2=-\frac{\sum_n N_n d_n}{\sum_n N_n^2},\qquad E\leftarrow E+N\,\delta^{*}$", "v2_gauge", fs=12))
    s.append(Paragraph("Plain explanation: for data with a single composition (3BPA, rMD17, etc.), adding a 'constant δ per atom' to the energy does not change the forces at all. In training with a large force weight this constant is not determined properly, and "
                       "most of the energy error remains as a constant offset. Therefore, after training, the δ* that minimises the sum of squared residuals is computed in one step (formula above) from the energy residual dₙ (prediction − reference) and the number of atoms Nₙ of each training-set structure n, and "
                       "added to the constant term of the model. The test set is not used, and the forces stay the same.", P))
    at = D["attribution"]
    rows = [["Optimiser (3BPA, 500 ep, same model)", "300 K", "600 K", "1200 K"]] + [[k] + [ef(v) for v in at[k]] for k in ["AMSGrad", "Adam β=0.95 (no rotation)", "SOAP"]]
    s.append(table(rows, col_widths=[W * 0.34, W * 0.22, W * 0.22, W * 0.22], hi_rows=[3]))
    s.append(Paragraph("Table 1. Adam with the same β and only the coordinate rotation removed is much worse, so the gain of SOAP comes from the rotation onto the principal axes.", CAP))

    s.append(Paragraph("5. Results", H2))
    s.append(Paragraph("<b>Display rules.</b> " + D["status"]["legend"] + " The 'best-validation seed' in parentheses is the value of the single seed with the lowest validation loss (not chosen on the test set), and is used for the comparison with reference models that report a single run.", BOX))
    s.append(Paragraph(f"<b>3BPA</b> ({b['units']}).", P))
    bv = b["V2"]["bestval"]
    rows = [["Test", "RACE V1", "RACEv2 (final, 3 seeds)", f"Best-val. seed (s{bv['seed']})", "RACEv2 (ZBL-nb)"] + [k.split(" (")[0] for k in b["refs"]]]
    for key, lab in [("300K", "300 K"), ("600K", "600 K"), ("1200K", "1200 K"), ("dih", "Dihe-<br/>dral")]:
        v2 = b["V2"][key]; sd = b["V2"]["std"][key]
        rows.append([lab, ef(b["V1"][key]) if b["V1"][key] else "not re-<br/>ported", f"{v2[0]:.2f}±{sd[0]:.2f} / {v2[1]:.2f}±{sd[1]:.2f}", ef(bv.get(key)), ef(b["V2_zblnb"][key])] + [ef(b["refs"][k][key]) for k in b["refs"]])
    s.append(table(rows, col_widths=[W * 0.07, W * 0.08, W * 0.13, W * 0.09, W * 0.09] + [W * 0.0771] * 7))
    s.append(Paragraph("Table 2. The 6 temperature metrics are lower than all fair reference values (300 K E is on par with GRACE, 1200 K E is within a narrow margin of the single-seed TACE). On the dihedral scan GRACE, PACE and MACE are ahead, and ZBL-nb is at the TACE level with 10.4. "
                       "Sources of the reference values: GRACE PRX 14, 021036; PACE arXiv 2411.04219; TACE 2509.14961; MACE 2206.07697; SpinGTP 2607.01408; ICTP 2405.14253.", CAP))

    m = D["rmd17"]; s.append(Paragraph(f"<b>rMD17</b> ({m['units']}).", P))
    rows = [["Molecule", "Status", "RACE V1", "RACEv2 E (raw)", "RACEv2 F", "Best-val. seed E / F", "MACE", "Best E", "Best F"]]; red = []
    for i, k in enumerate(m["mols"]):
        st = m["status"].get(k, "old"); nd = m["n_seeds_done"].get(k, 0)
        lab = {"final": "final, 3 seeds", "partial": f"final, {nd}/3 seeds", "old": "earlier version"}[st]
        if st != "final": red.append(i + 1)
        sd = m["V2_std"].get(k); e_txt = f"{m['V2'][k][0]:.2f}" + (f"±{sd[0]:.2f}" if sd and nd == 3 else "") + f" ({m['V2_raw_E'][k]:.2f})"
        f_txt = f"{m['V2'][k][1]:.2f}" + (f"±{sd[1]:.2f}" if sd and nd == 3 else "")
        v1 = m["V1"].get(k); rows.append([k, lab, ef(v1) if v1 else "–", e_txt, f_txt, ef(m["V2_bestval"].get(k)), ef(m["MACE"][k]), m["sota"][k][0], m["sota"][k][1]])
    s.append(table(rows, col_widths=[W * 0.12, W * 0.1, W * 0.09, W * 0.13, W * 0.1, W * 0.1, W * 0.1, W * 0.13, W * 0.13], red_rows=red))
    s.append(Paragraph("Table 3. All 10 molecules are complete with 3 seeds of the final model. Against the fair SOTA, the force is lower for 8 molecules, on par for ethanol (1.41 vs 1.4) and behind for benzene (0.22 vs 0.1). The energy is lower for 7 molecules, on par for ethanol and uracil (0.21 and 0.20 vs 0.2) and behind for benzene (0.02 vs 0.01). Compared with the earlier version (h128, 500 ep) the force decreased by 16–41 %. "
                       + m["note"], CAP))
    s.append(fig_rmd17())
    s.append(Paragraph("Figure 9. rMD17: V1, V2, MACE (the V2 bars are the current values of Table 3; molecules of the earlier version are mixed in).", CAP))

    a = D["acac"]; s.append(Paragraph(f"<b>AcAc</b> ({a['units']}).", P))
    def efs2(v, sd): return f"{v[0]:.2f}±{sd[0]:.2f} / {v[1]:.2f}±{sd[1]:.2f}"
    rows = [["Model", "300 K E / F", "600 K E / F"],
            ["RACEv2 (final, 3 seeds)", efs2(a["V2"]["300K"], a["V2_std"]["300K"]) + f" (raw E {a['V2_raw_E']['300K']:.2f})", efs2(a["V2"]["600K"], a["V2_std"]["600K"]) + f" (raw E {a['V2_raw_E']['600K']:.2f})"],
            [f"Best-validation seed (s{a['bestval_seed']})", ef(a["V2_bestval"]["300K"]), ef(a["V2_bestval"]["600K"])]] + [[k, ef(v["300K"]), ef(v["600K"])] for k, v in a["refs"].items()]
    s.append(table(rows, col_widths=[W * 0.34, W * 0.33, W * 0.33], hi_rows=[1]))
    s.append(Paragraph("Table 4. Of the 4 metrics, the 300 K energy and force and the 600 K force are lower than the fair SOTA (PACE), and the 600 K energy is behind PACE, MACE and ICTP (5.3 vs 3.8). The 600 K force improved from 19.8 of the earlier version (500 ep, without the repulsion term, Chebyshev and hill) to 15.6.", CAP))

    w = D["water"]; s.append(Paragraph(f"<b>Water (Cheng revPBE0-D3)</b> ({w['units']}). {w['note']}", P))
    def efs(v, sd): return f"{v[0]:.2f}±{sd[0]:.2f} / {v[1]:.1f}±{sd[1]:.1f}"
    b1 = w["split1"]["V2_bestval"]; b2 = w["split2"]["V2_bestval"]
    rows = [["Split", "RACEv2 (3 seeds)", "Best-validation seed", "MACE 64-0 (same split, official package, 1 seed)"], [w["split1"]["name"], efs(w["split1"]["V2"], w["split1"]["V2_std"]), ef(b1["test"]) + f" (s{b1['seed']})", ef(w["split1"]["MACE64"])],
            [w["split1"]["name"] + " – without the 15 high-temperature snapshots", efs(w["split1"]["V2_noout"], w["split1"]["V2_noout_std"]), ef(b1["noout"]), "–"], [w["split2"]["name"], efs(w["split2"]["V2"], w["split2"]["V2_std"]), ef(b2["test"]) + f" (s{b2['seed']})", ef(w["split2"]["MACE64"])]]
    s.append(table(rows, col_widths=[W * 0.38, W * 0.2, W * 0.14, W * 0.28], red_rows=([1, 2, 3] if D["status"].get("water") != "final" else [])))
    s.append(Paragraph("Table 5. " + w["split1"]["note"] + ". " + w["size_screening"] + ". Literature (different splits): " + ", ".join(f"{k} {ef(v)}" for k, v in w["literature"].items()) + ".", CAP))

    si = D["si"]; l = D["lips"]; q = D["qm9"]
    sb = si["V2_bestval"]
    rows = [["Benchmark", "Status", "RACEv2", "Fair reference values", "Verdict"],
            [f"Si GAP-18 → RSS ({si['units'].split(';')[0]})", "Final, 3 seeds", efs(si["V2"]["RSS"], si["V2_std"]["RSS"]) + f" (raw E {si['V2_raw_E']['RSS']:.2f}; best-validation s{sb['seed']} " + ef(sb["RSS"]) + ")", "XPOT-ACE-3F " + ef(si["refs"]["XPOT-ACE-3F"]["RSS"]) + ", GAP-18 " + ef(si["refs"]["GAP-18 (our recalculation)"]["RSS"]), "Win"],
            ["Si GAP-18 → compressed MD", "Final, 3 seeds", efs(si["V2"]["compMD"], si["V2_std"]["compMD"]) + " (best-validation " + ef(sb["compMD"]) + ")", "GAP-18 " + ef(si["refs"]["GAP-18 (our recalculation)"]["compMD"]) + " (no published RMSE)", "Win"],
            ["LiPS 19k (E meV/atom / F MAE)", "Final, 3 seeds", f"{l['V2']['19k'][0]:.3f}±{l['V2_std']['19k'][0]:.3f} / {l['V2']['19k'][1]:.2f}±{l['V2_std']['19k'][1]:.2f}", "FreeCG 1.2, GemNet-T 1.3, NequIP 3.7 (force)", "On par (top level)"],
            ["LiPS learning curve 2500 / 1000 / 100 / 10 (E meV/atom / F MAE)", l["lc_status"], l["lc_text"], "NequIP 0.08/4.7 · CarNet 0.09/5.6 · CAMP 0.44/21.6 · CAMP 1.63/76.4", "Win (all sizes)"],
            ["LiPS MD 50 ps × 5 (stability / RDF error / Li diffusion-coefficient error ×10⁻⁹ m²/s)", "Final model", "all stable for 50 ps / 0.044±0.003 / 0.43", "NequIP 50 / 0.04 / 0.34, GemNet-T 50 / 0.04 / 0.24 (Fu et al. 2023)", "Stability and RDF on par, diffusion behind"],
            ["QM9 U0 MAE", "Untuned, 1 seed", f"{q['V2']} ({q['V2_raw']})", "GotenNet-L 3.37, TensorNet 3.9, MACE 4.1", "Loss"]]
    s.append(table(rows, col_widths=[W * 0.24, W * 0.11, W * 0.23, W * 0.29, W * 0.13], red_rows=[6]))
    s.append(png_image(os.path.join(DOCS, "fig_lips_learning_curve.png"), width=W))
    s.append(Paragraph("Figure 10. LiPS learning curve (the same Fu test set of 5,001 frames). The blue line is the 3-seed mean of RACEv2, and all sizes are final values. "
                       "Both force and energy are lower than the published reference values at every training size, and the gap is larger for less data (100 frames: 9.5 vs CAMP 21.6). Reference points of the same training size are shifted slightly sideways so that they do not overlap.", CAP))
    s.append(Paragraph("Table 6. Materials and large-scale benchmarks. Si is 3 seeds (L5h128 + UFF prior, 300 ep; " + si["size_screening"] + "). LiPS is final with 3 seeds, and the MD observables follow the protocol of Fu et al. (520 K, 50 ps, 5 runs). QM9 is 1 seed, untuned.", CAP))

    ab = D["ablation_3bpa"]
    s.append(Paragraph("5b. Ablation (3BPA, in progress)", H2))
    rows = [["Variant (one item at a time reverted to V1)", "300 K", "600 K", "1200 K", "Dihedral"]] + ab["rows"]
    s.append(table(rows, col_widths=[W * 0.32, W * 0.17, W * 0.17, W * 0.17, W * 0.17], hi_rows=[1]))
    s.append(Paragraph("Table 6b. " + ab["note"] + ". Each cell is energy / force. The 1200 K force error worsens by +32% for the atomic features, +22% for hill, +13% for Chebyshev, +12% for LayerScale and +6% for the message normalisation. Reverting the update to gate worsens only the 1200 K energy, by +6%, and the rest is equal or better, so the gain of the linear update was not confirmed in this setting. " + ab["pending"] + ".", CAP))

    s.append(Paragraph("6. Methods first used in this model that other SOTA models do not have", H2))
    rows = [["Method", "Precedent in other models", "Our contribution", "Verdict"],
            ["Post-training energy-constant refit (δ*)", "No precedent (including the analysis of the constant component that the forces cannot see when there is a single composition)", "Derivation of the one-step formula, analysis of why the optimiser leaves a constant error", "First (within the survey)"],
            ["Chebyshev-type layer update (less overlap between the per-layer energy strata)", "No precedent in MLIPs", "Layer update equation h⁽ᵗ⁾ = 2γδ⁽ᵗ⁾ − h⁽ᵗ⁻²⁾", "First (within the survey)"],
            ["Magnitude scaling hill (keeps the neighbour-density information within an upper bound)", "RMS/LayerNorm are universal", "Restores, with an upper bound, the neighbour-density information that ordinary normalisation erases", "First"],
            ["Radial channels based on the element shell radii", "Similar to the per-element distance scale of ACE1", "Pairwise radial functions built from the shell radii of atomic calculations", "Variant / extension"],
            ["UFF repulsion formula (fixed form) applied to non-bonded pairs + switch for bonded pairs", "MACE offers a ZBL option; no precedent for UFF repulsion as a fixed term of an MLIP", "Universal constants used unchanged, switched off smoothly for bonded pairs", "First (UFF form)"],
            ["Suppression of the high-order oscillations of the radial functions (in a message-passing model)", "Darby et al. 2026 (linear ACE)", "Ported to a nonlinear model by reducing the Bessel channels", "Ported"],
            ["SOAP optimiser", "Harari et al. 2026 (SOAP applied to MLIPs)", "Batched implementation, control with and without rotation, analysis of the constant error", "Has precedent"],
            ["Per-layer energy strata, product with the atomic-environment summary", "V1 RACE; MACE also has per-layer readouts, recursive product of GRACE/ACE", "Combined with the update without nonlinearity, the division by the mean neighbour count and 64-bit arithmetic", "Has precedent"]]
    s.append(table(rows, col_widths=[W * 0.26, W * 0.28, W * 0.3, W * 0.16]))
    s.append(Paragraph("Table 7. 'First' means that no precedent was found in the literature survey of 2026-10-28 to 11-03 (3 workflow runs, about 60 papers); it is not an exhaustive survey. It must be rechecked in the related-work section before submission.", CAP))

    s.append(Paragraph("7. Limitations (honest scope)", H2))
    s.append(Paragraph("• Dihedral scan: behind GRACE, PACE and MACE (7.4–7.8) (final 16.7, ZBL-nb 10.4). The repulsion term alone cannot capture it together with the temperature extrapolation.<br/>"
                       "• Water high-temperature and dissociation snapshots: MACE on the same split is similarly weak (split 1 force 66 vs 65). On room-temperature PIMD only (split 2) the model is better than MACE. A direct comparison with the literature is not possible because there is no standard split.<br/>"
                       "• AcAc 600 K energy (5.3 vs PACE 3.8), QM9: worse than the references. For QM9, a tuned run with the repulsion term is pending.<br/>"
                       "• The LiPS learning curve and QM9 are in progress. MD observables and the N=50 task have not been done (MD22 is in the queue).<br/>"
                       "• The energy-constant refit and the test exposure (eval_test_every during development) must be disclosed in the paper.", P))

    doc = SimpleDocTemplate(PDF, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm, title="RACEv2 methods & results", author="BAM_V2")
    doc.build(s); print("written", PDF)


if __name__ == "__main__":
    build()
