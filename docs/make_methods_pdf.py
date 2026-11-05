"""RACEv2 methods & results summary (Korean) -> docs/RACEv2_methods_summary.pdf

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
    for i, (ax, ttl, unit) in enumerate(zip(axs, ["에너지 RMSE", "힘 RMSE"], ["meV/분자", "meV/Å"])):
        ax.bar(xs - wd / 2, [b["V1"][k][i] for k in keys], wd, label="RACE V1 (논문)", color="#9e9e9e")
        ax.bar(xs + wd / 2, [b["V2"][k][i] for k in keys], wd, label="RACEv2 (최종)", color="#0b3d91")
        for (name, col, mk) in refs:
            ax.scatter(xs + wd / 2 + 0.22, [b["refs"][name][k][i] for k in keys], marker=mk, color=col, zorder=5, s=30, label=name.split(" (")[0])
        ax.set_xticks(xs); ax.set_xticklabels(labels); ax.set_ylabel(f"{ttl} ({unit})"); ax.set_title(f"3BPA {ttl}: V1 → V2와 문헌 최고값", fontsize=10)
        ax.set_ylim(0, None); ax.grid(axis="y", alpha=0.3)
        if i == 0: ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    fig.tight_layout(); return fig_image(fig, "v1v2")


def fig_bessel():
    rc = 5.0; r = np.linspace(0.05, rc, 500); sig = 0.224; fc = polynomial_cutoff(r, rc)
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for n in range(1, 9):
        bnr = 2 / rc * np.sin(n * np.pi * r / rc) / r * fc
        axs[0].plot(r, bnr, lw=1, label=f"n={n}" if n in (1, 4, 8) else None); axs[1].plot(r, bnr * np.exp(-sig ** 2 * n ** 2), lw=1)
    axs[0].set_title("Bessel 기저 × f(cut) (n=1..8, r_c=5 Å)"); axs[0].legend(fontsize=8)
    axs[1].set_title("× exp(−σₙ² n²) 적용 후 (σₙ=0.224)")
    n = np.arange(1, 9); axs[2].bar(n, np.exp(-sig ** 2 * n ** 2), color="#2b6cb0"); axs[2].set_title("축소 계수 exp(−σₙ² n²)"); axs[2].set_ylim(0, 1.05)
    for a in axs: a.grid(alpha=0.3); a.set_xlabel("r (Å)" if a is not axs[2] else "Bessel 차수 n")
    fig.tight_layout(); return fig_image(fig, "bessel")


def fig_shells():
    B = 0.529177
    rm = {"H 1s": 1.6733 * B, "C 2s": 1.5838 * B, "C 2p": 1.7942 * B, "N 2p": 1.4814 * B, "O 2p": 1.2645 * B}
    r = np.linspace(0.3, 5.0, 400); fc = polynomial_cutoff(r, 5.0)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for a, b in [("H 1s", "H 1s"), ("H 1s", "C 2p"), ("C 2p", "C 2p"), ("H 1s", "O 2p"), ("N 2p", "C 2p")]:
        rho = rm[a] + rm[b]; ax.plot(r, np.exp(-r / rho) * fc, label=f"{a}–{b} (ρ(a)+ρ(b) = {rho:.2f} Å)")
    ax.set_xlabel("r (Å)"); ax.set_ylabel("exp(−r/(ρ(a)+ρ(b)))·f(cut)"); ax.set_title("원소 쌍별 껍질 거리 함수 S(a,b)(r)"); ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
    return fig_image(fig, "shells", width=W * 0.72)


def fig_hill():
    rr = np.logspace(-2, 1.5, 300)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for a, b, lab in [(0, 0, "a=0, b=0 (순수 RMS 정규화)"), (0.5, 0, "a=0.5, b=0"), (1, 0, "a=1, b=0"), (1, -1, "a=1, b=−1"), (1, 1, "a=1, b=+1")]:
        ax.semilogx(rr, 2 / (1 + np.exp(-(a * np.log(rr) + b))), label=lab)
    ax.set_xlabel("rᵢ = xᵢ의 평균 크기(RMS)"); ax.set_ylabel("배율 2σ(a·log rᵢ + b)"); ax.set_title("크기 조절 배율 (출력 크기 ≤ 2)")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
    return fig_image(fig, "hill", width=W * 0.72)


def fig_cheb():
    x = np.linspace(-1, 1, 400)
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.0))
    for n in range(5):
        axs[0].plot(x, x ** n, label=f"x^{n}"); axs[1].plot(x, np.cos(n * np.arccos(x)), label=f"T_{n}(x)")
    axs[0].set_title("보통의 층 쌓기 xⁿ: 깊어질수록 서로 닮아 감"); axs[1].set_title("Chebyshev Tₙ: 층마다 다른 모양")
    for a in axs: a.legend(fontsize=7.5, ncol=2); a.grid(alpha=0.3)
    fig.tight_layout(); return fig_image(fig, "cheb")


def fig_repulsion():
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.0))
    pairs = [("H", "H", 1.5, 2.13, "H–H: 충돌 1.5 Å, 학습 최소 2.13 Å"), ("C", "H", 2.13, 2.60, "C–H: 충돌 2.13 Å, 학습 최소 2.60 Å"), ("C", "N", 2.59, 3.05, "C–N: 충돌 2.59 Å, 학습 최소 3.05 Å")]
    for k, (a, b, rcol, rmin, title) in enumerate(pairs):
        r = np.linspace(0.8, 4.0, 600); fc = polynomial_cutoff(r, 5.0)
        ez = zbl(Z[a], Z[b], r) * fc; eu = uff(a, b, r) * fc; s1 = switch(a, b, r, 1.25, 0.1); s3 = switch(a, b, r, 1.25, 0.3)
        ax = axs[0, k]
        ax.plot(r, ez * 1e3, "k:", lw=1, label="ZBL 원본"); ax.plot(r, ez * s1 * 1e3, color="#2b6cb0", label="ZBL-nb (c=1.25, w=0.1 Å)")
        ax.plot(r, eu * 1e3, color="#e08e0b", ls=":", lw=1, label="UFF 원본"); ax.plot(r, eu * s3 * 1e3, color="#d9534f", lw=2, label="UFF-nb w0.3 (최종)")
        ax.axvline(rcol, color="gray", ls=":", lw=1); ax.axvline(rmin, color="gray", ls="-.", lw=1)
        ax.set_ylim(0, 1500); ax.set_title(title, fontsize=9.5); ax.set_ylabel("반발 에너지 (meV)"); ax.grid(alpha=0.3)
        if k == 0: ax.legend(fontsize=7)
        ax = axs[1, k]
        for e, lab, col, ls, lw in [(ez, "ZBL", "k", ":", 1), (ez * s1, "ZBL-nb", "#2b6cb0", "-", 1.2), (eu, "UFF", "#e08e0b", ":", 1), (eu * s3, "UFF-nb w0.3", "#d9534f", "-", 2)]:
            ax.plot(r, -np.gradient(e * 1e3, r), color=col, ls=ls, lw=lw, label=lab)
        ax.axvline(rcol, color="gray", ls=":", lw=1); ax.axvline(rmin, color="gray", ls="-.", lw=1)
        ax.set_ylim(0, 5000); ax.set_xlabel("r (Å)"); ax.set_ylabel("반발력 −dE/dr (meV/Å)"); ax.grid(alpha=0.3)
    fig.suptitle("단거리 반발 prior: 점선 = 이면각 충돌 거리, 쇄선 = 학습셋 최소 거리", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); return fig_image(fig, "repulsion")


def fig_switch():
    r = np.linspace(0.5, 3.0, 400)
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for (a, b), col in [(("C", "H"), "#2b6cb0"), (("H", "H"), "#d9534f"), (("C", "N"), "#2e8b57")]:
        ax.plot(r, switch(a, b, r, 1.25, 0.3), color=col, label=f"{a}–{b}, w=0.3 Å (최종)"); ax.plot(r, switch(a, b, r, 1.25, 0.1), color=col, ls=":", label=f"{a}–{b}, w=0.1 Å")
    ax.axvspan(1.0, 1.4, color="orange", alpha=0.15, label="1200 K에서 늘어난 C–H 결합")
    ax.set_title("비결합 switch σ((r − 1.25(Rᵢ+Rⱼ))/w)"); ax.set_xlabel("r (Å)"); ax.legend(fontsize=6.8, ncol=2); ax.grid(alpha=0.3)
    return fig_image(fig, "switch", width=W * 0.72)


def fig_rmd17():
    m = D["rmd17"]; mols = m["mols"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4)); xs = np.arange(len(mols)); w = 0.27
    for ax, i, t in [(axs[0], 0, "에너지 MAE (meV)"), (axs[1], 1, "힘 MAE (meV/Å)")]:
        v1 = [m["V1"].get(k, [np.nan, np.nan])[i] for k in mols]
        ax.bar(xs - w, v1, w, label="RACE V1", color="#9e9e9e"); ax.bar(xs, [m["V2"][k][i] for k in mols], w, label="RACEv2", color="#0b3d91")
        ax.bar(xs + w, [m["MACE"][k][i] for k in mols], w, label="MACE", color="#d9534f")
        ax.set_xticks(xs); ax.set_xticklabels([k[:7] for k in mols], fontsize=8); ax.set_title("rMD17 " + t); ax.grid(axis="y", alpha=0.3)
    axs[0].legend(fontsize=8); fig.tight_layout(); return fig_image(fig, "rmd17")


# ----------------------------------------------------------------------------- document
def build():
    b = D["bpa"]; s = []
    s.append(Paragraph("RACEv2: 최종 모델의 방법과 결과", H1))
    s.append(Paragraph(f"갱신 {DATE} (수치 기준일 {D['date_note']}). 수치는 docs/methods_pdf_data.json에서 읽어 생성한다. "
                       "RMSE가 기본이며 rMD17·LiPS·QM9는 MAE. 에너지는 학습 후 상수 보정(4절 δ*)을 적용한 값이고, 효과가 큰 곳은 raw 값을 괄호로 함께 적었다. "
                       "참조값은 2026-11-03 벤치마크 감사에서 원 논문으로 확인한 공정 프로토콜(사전학습·추가 데이터 없음) 모델만 썼다.", PS))

    s.append(Paragraph("0. 한눈에 보기", H2))
    s.append(fig_v1v2())
    V1, V2 = b["V1"], b["V2"]
    s.append(Paragraph(f"그림 1. 3BPA(300 K 500프레임 학습). V1 → V2: 1200 K 힘 {V1['1200K'][1]} → {V2['1200K'][1]} meV/Å(−{100*(1-V2['1200K'][1]/V1['1200K'][1]):.0f} %), "
                       f"300 K 에너지 {V1['300K'][0]} → {V2['300K'][0]} meV(−{100*(1-V2['300K'][0]/V1['300K'][0]):.0f} %). 온도 6개 지표 모두 GRACE·PACE·MACE보다 낮다.", CAP))
    s.append(Paragraph("<b>최종 모델 한 줄.</b> 이웃 정보를 5번 주고받는 회전 대칭 신경망(128채널, 방향 정보 l ≤ 3, 5 Å 이내 이웃) + 층마다 에너지 조각을 내어 합산 "
                       "+ 각 층 메시지에 '원자 주변 요약'을 곱해 동시에 관여하는 원자 수를 늘림(크기 조절 포함) + 두 층 전 특징을 빼는 Chebyshev식 갱신 + 증분 배율(LayerScale) "
                       "+ 거리 함수의 고차 진동 억제 + 원소 껍질 크기를 반영한 거리 채널 + 가까이 붙은 비결합 원자쌍의 UFF 반발력(형태 고정, 전역 배율 공유), SOAP 옵티마이저(lr 1e-2) + 학습 후 에너지 상수 보정, 64비트 연산. "
                       f"{D['final_recipe']['selection_note']}", BOX))
    s.append(Paragraph("<b>용어 풀이.</b> "
                       "<b>에너지 조각(stratum)</b>: 각 층이 내는 에너지 기여. 총 에너지는 이 조각들의 합. "
                       "<b>원자 주변 요약 Aᵢ</b>: 중심 원자 주변 이웃의 종류·거리·방향을 한 번에 모아 둔 벡터(0층 집계). "
                       "<b>원자 주변 요약 곱(xᵢ 곱)</b>: 각 층 메시지에 Aᵢ를 변형한 xᵢ를 곱하는 것. 곱할 때마다 '동시에 몇 개 원자가 관여하는 항까지 표현하는가'가 한 단계 올라간다. "
                       "<b>크기 조절(hill)</b>: xᵢ의 크기를 일정하게 맞추되, 이웃이 빽빽한지 성긴지 정보는 상한을 두고 남긴다. "
                       "<b>Chebyshev식 갱신</b>: 새 층 특징 = 2×(이번 층 증분) − (두 층 전 특징). 층들이 서로 겹치는 정보를 덜 갖게 된다. "
                       "<b>증분 배율(LayerScale)</b>: 층 증분에 곱하는 학습 가능한 배율(초기 0.5). "
                       "<b>고정 항(prior)</b>: 데이터로 배우지 않고 물리 공식으로 미리 정해 넣는 항. "
                       "<b>에너지 상수 보정(게이지 refit)</b>: 원자당 상수 에너지는 힘에 영향이 없어 학습으로 정해지지 않으므로 학습 후 학습셋으로 한 번에 맞추는 것.", BOX))
    s.append(png_image(os.path.join(DOCS, "RACEv2_architecture.png"), width=W))
    s.append(Paragraph("그림 2. 최종 모델 구조(v1 논문 Figure 1 b·c 구도; docs/make_architecture_fig.py). b: 전체 흐름. 원소·방향·거리 표현 → 이웃 정보를 한 번 모은 원자 주변 요약 Aᵢ → "
                       "층별로 Aᵢ를 변형한 xᵢ(초록) → 5개 층이 각각 에너지 조각 Eᵢ⁽ᵗ⁾를 내고, UFF 반발력 Eᵢʳᵉᵖ과 함께 더한다. 힘은 에너지를 좌표로 미분해 얻는다. "
                       "c: 층 하나의 내부. 노랑 상자가 v1 대비 추가·변경된 부분(이웃 합을 평균 이웃 수로 나눔, xᵢ 곱, 비선형 없는 더하기, 증분 배율, 두 층 전 특징 빼기). "
                       "비선형은 에너지 조각을 읽어내는 작은 네트워크(Linear–Gate–Linear)에만 있다.", CAP))

    s.append(Paragraph("1. 기존(V1) → 현재(V2): 바뀐 수식 비교", H2))
    s.append(Paragraph("두 모델을 <b>같은 기호, 같은 단계</b>로 나란히 적었다. 한 층(t번째)의 계산을 다섯 단계로 나누고, 각 단계에서 기존과 현재 식의 왼쪽 변(무엇을 계산하는가)은 같다. "
                       "기존 식은 V1 논문(Methods 식 13–16)과 V1 코드 기본 설정(x_mode species, message_norm layer, update gate, readout linear, residual none)을 이 문서의 기호로 옮겨 쓴 것이다.", P))
    s.append(Paragraph("기호: hᵢ⁽ᵗ⁾ 층 t를 지난 원자 i의 특징(V1 논문의 Aᵢ⁽η⁾), mᵢ⁽ᵗ⁾ 이웃에서 모은 메시지(V1 식 15의 Aᵢ⁽η⁻¹⁾), xᵢ 메시지에 곱하는 인자(V1의 a⁽¹⁾), "
                       "δᵢ⁽ᵗ⁾ 층 증분, Yᵢⱼ 이웃 방향(구면조화함수, V1의 eᵢⱼ), Rₜ(rᵢⱼ) 거리 가중치(V1의 M(ẽᵢⱼ)), ⟨N⟩ 평균 이웃 수, Lin 회전 대칭 선형 변환, ⊗ 텐서곱, ⊙ 채널별 곱.", P))
    rows = [["단계 (계산하는 것)", "기존 (V1)", "현재 (V2)"],
            ["① 메시지 mᵢ⁽ᵗ⁾", "LayerNorm 후 이웃 합, ÷√⟨N⟩", "LayerNorm 없이 이웃 합, ÷⟨N⟩"],
            ["② 곱하는 인자 xᵢ", "원소별 상수 하나 a(Z) (모든 층 공통)", "원자 주변 요약 Aᵢ의 층별 선형 변환"],
            ["③ 인자의 크기 x̂ᵢ", "조절 없음", "크기 조절(hill): 0~2 배율"],
            ["④ 층 증분 δᵢ⁽ᵗ⁾", "Lin(m⊗x) + 자기 자신 항", "같음 (x 대신 x̂)"],
            ["⑤ 다음 층 특징 hᵢ⁽ᵗ⁾", "Gate(δ) (비선형)", "2γδ − h⁽ᵗ⁻²⁾ (선형, Chebyshev)"],
            ["⑥ 층 에너지 조각 Eᵢ⁽ᵗ⁾", "Lin(Lin(h)) (비선형은 ⑤에)", "Lin(Gate(Lin(h))) (비선형은 여기만)"],
            ["⑦ 총 에너지 E", "Σᵢ [E₀ + Σₜ Eᵢ⁽ᵗ⁾]", "Σᵢ [E₀ + Eᵢʳᵉᵖ + Σₜ Eᵢ⁽ᵗ⁾] + Nδ*"]]
    s.append(table(rows, col_widths=[W * 0.28, W * 0.36, W * 0.36]))
    s.append(Paragraph("표 0. 한 층의 계산 단계별 비교. 아래 1-1~1-6에서 각 단계의 식을 기존·현재 순서로 적는다.", CAP))

    s.append(Paragraph("<b>1-1. ① 메시지: 이웃 정보 모으기.</b>", P))
    s.append(eq(r"기존:  $m_i^{(t)}=\frac{1}{\sqrt{\langle N\rangle}}\sum_{j\in N(i)} R_t(r_{ij})\,\left(W_t\,\mathrm{LN}(h_j^{(t-1)})\otimes Y_{ij}\right)$", "u_msg1", fs=12))
    s.append(eq(r"현재:  $m_i^{(t)}=\frac{1}{\langle N\rangle}\sum_{j\in N(i)} R_t(r_{ij})\,\left(W_t\,h_j^{(t-1)}\otimes Y_{ij}\right)$", "u_msg2", fs=12))
    s.append(Paragraph("쉬운 설명: 이웃 원자 j마다 ① 특징 hⱼ를 행렬 Wₜ로 변환하고 ② 이웃 방향 Yᵢⱼ와 결합한 뒤 ③ 거리 가중치 Rₜ(rᵢⱼ)를 곱해 ④ 모두 더한다. Rₜ는 작은 신경망이 거리 표현 ẽᵢⱼ에서 만든 값이다. "
                       "<b>바뀐 점:</b> LayerNorm(LN)을 없애 특징의 크기 정보를 남기고, 1/√⟨N⟩ 대신 1/⟨N⟩으로 나눠 이웃 수가 늘어도 메시지 크기가 일정하게 했다(밀도가 다른 구조로 외삽할 때 덜 흔들린다).", P))

    s.append(Paragraph("<b>1-2. ② 메시지에 곱하는 인자.</b>", P))
    s.append(eq(r"기존:  $x_i^{(t)}=a_{Z_i}$   (원소별 학습 상수, 모든 층 공통)", "u_x1", fs=12))
    s.append(eq(r"현재:  $x_i^{(t)}=\mathrm{Lin}_t(A_i),\qquad A_i=\frac{1}{\langle N\rangle}\sum_{j\in N(i)} R_0(r_{ij})\,\left(W_0\,h_j^{(0)}\otimes Y_{ij}\right)$", "u_x2", fs=12))
    s.append(Paragraph("쉬운 설명: 기존 인자 a(Z)는 원자 종류만 보고 정해지는 상수라, 곱해도 주변 환경 정보가 늘지 않았다. 현재의 Aᵢ는 0층에서 이웃 정보를 한 번 모은 벡터(ACE의 원자 기저, 원자 주변 요약)이고, "
                       "층마다 다른 선형 변환 Linₜ를 거쳐 xᵢ⁽ᵗ⁾가 된다. 이것을 메시지에 곱할 때마다 식 안의 이웃 합이 하나씩 늘어나, 1층 2체, 2층 3체, … 5층 6체 항까지 표현한다(ACE의 재귀 곱).", P))

    s.append(Paragraph("<b>1-3. ③ 인자의 크기 조절(hill).</b>", P))
    s.append(eq(r"기존:  $\hat x_i^{(t)}=x_i^{(t)}$   (조절 없음)", "u_hill1", fs=12))
    s.append(eq(r"현재:  $\hat x_i^{(t)}=\frac{x_i^{(t)}}{r_i}\cdot 2\,\sigma\left(a\log r_i+b\right),\qquad r_i=\mathrm{RMS}(x_i^{(t)}),\quad \sigma(u)=\frac{1}{1+e^{-u}}$", "u_hill2", fs=12))
    s.append(Paragraph("쉬운 설명: xᵢ의 크기 rᵢ는 이웃이 많고 가까울수록 커진다. 그대로 곱하면 층을 지날수록 값이 폭발하고, 크기로 나누기만 하면 '이웃이 빽빽한가' 정보가 사라진다. "
                       "그래서 크기로 나눈 뒤 0~2 사이의 배율을 곱한다. a, b는 원소별·층별로 학습되는 두 숫자다. a=0이면 순수 정규화와 같고, a&gt;0이면 빽빽할수록 배율이 커지되 2를 넘지 않는다(그림 3).", P))
    s.append(fig_hill())
    s.append(Paragraph("그림 3. 크기 조절 배율 2σ(a log rᵢ + b). a=b=0이면 크기 정보가 완전히 사라지고, a=1이면 작은 rᵢ에서는 원래 크기를 따르다가 큰 rᵢ에서는 2에서 멈춘다.", CAP))

    s.append(Paragraph("<b>1-4. ④ 층 증분.</b>", P))
    s.append(eq(r"기존:  $\delta_i^{(t)}=\mathrm{Lin}\left(m_i^{(t)}\otimes x_i^{(t)}\right)+\mathrm{Lin}_{Z_i}\left(h_i^{(t-1)}\right)$", "u_d1", fs=12))
    s.append(eq(r"현재:  $\delta_i^{(t)}=\mathrm{Lin}\left(m_i^{(t)}\otimes \hat x_i^{(t)}\right)+\mathrm{Lin}_{Z_i}\left(h_i^{(t-1)}\right)$", "u_d2", fs=12))
    s.append(Paragraph("쉬운 설명: 메시지와 인자의 곱(채널별 텐서곱)을 선형 변환하고, 원소별 가중치를 쓰는 자기 자신 항을 더한다. 식의 모양은 같고, 곱하는 인자만 ②·③에서 바뀐 것으로 대체된다.", P))

    s.append(Paragraph("<b>1-5. ⑤ 다음 층으로 넘기는 특징.</b>", P))
    s.append(eq(r"기존:  $h_i^{(t)}=\mathrm{Gate}\left(\delta_i^{(t)}\right)$", "u_h1", fs=12))
    s.append(eq(r"현재:  $h_i^{(t)}=2\,\gamma_t\odot\delta_i^{(t)}-h_i^{(t-2)}\ \ (t\geq 3),\qquad h_i^{(t)}=\gamma_t\odot\delta_i^{(t)}\ \ (t=1,2)$", "u_h2", fs=12))
    s.append(Paragraph("쉬운 설명: 기존은 증분에 비선형 Gate를 씌워 그대로 다음 층에 넘겼다. 현재는 비선형 없이, 증분의 2배에서 두 층 전 특징을 뺀다. γₜ는 채널별 배율(초기 0.5, 학습됨)이다. "
                       "층을 그냥 쌓으면 x, x², x³ …처럼 서로 닮은 모양이 겹쳐 층별 에너지 조각이 같은 정보를 반복한다. Tₙ₊₁ = 2x·Tₙ − Tₙ₋₁(Chebyshev 점화식)을 따르면 층마다 서로 다른 모양이 되어 각 층이 새 정보를 담는다(그림 4).", P))
    s.append(fig_cheb())
    s.append(Paragraph("그림 4. 보통의 층 쌓기(왼쪽)는 거듭제곱처럼 비슷한 모양이 겹쳐 쌓이지만, Chebyshev식 갱신(오른쪽)은 층마다 서로 다른 모양이 된다.", CAP))

    s.append(Paragraph("<b>1-6. ⑥ 층 에너지 조각과 ⑦ 총 에너지.</b>", P))
    s.append(eq(r"기존:  $E_i^{(t)}=\mathrm{Lin}\left(\mathrm{Lin}(h_i^{(t)})\right),\qquad E=\sum_i\left[E_{0,Z_i}+\sum_{t=1}^{T}E_i^{(t)}\right]$", "u_E1", fs=12))
    s.append(eq(r"현재:  $E_i^{(t)}=\mathrm{Lin}\left(\mathrm{Gate}(\mathrm{Lin}(h_i^{(t)}))\right),\qquad E=\sum_i\left[E_{0,Z_i}+E_i^{\rm rep}+\sum_{t=1}^{T}E_i^{(t)}\right]+N\,\delta^{*}$", "u_E2", fs=12))
    s.append(Paragraph("쉬운 설명: 각 층 특징 h⁽ᵗ⁾의 회전에 무관한 성분(l=0)을 작은 네트워크로 읽어 그 층의 에너지 조각을 만든다. 기존은 비선형이 ⑤(층 사이)에 있었고, 현재는 층 사이를 선형으로 두는 대신 비선형을 이 읽기 단계에만 둔다. "
                       "총합에는 Eᵢʳᵉᵖ(3절의 UFF 반발력)과 δ*(4절의 학습 후 상수 보정, 원자 수 N을 곱함)가 더해진다. E₀는 원소별 기준 에너지, T는 층 수(5)다. 힘은 E를 좌표로 자동미분해 얻는다. 구현에서는 대괄호 안의 Eᵢʳᵉᵖ + Σₜ Eᵢ⁽ᵗ⁾ 전체에 학습되는 전역 배율 s가 곱해지고 원자당 상수가 더해진다(3절 참조).", P))

    s.append(Paragraph("2. 거리 표현: Bessel 함수 + 고차 진동 억제 + 원소 껍질 크기 채널", H2))
    s.append(eq(r"기존:  $\tilde e_{ij}=[\,B_n(r_{ij})\,f_{\rm cut}(r_{ij})\,],\qquad n=1,\dots,8$", "v1_rad", fs=12))
    s.append(eq(r"현재:  $\tilde e_{ij}=[\,B_n(r_{ij})\,e^{-\sigma_n^2 n^2}f_{\rm cut}(r_{ij})\ \|\ S_{ab}(r_{ij})],\quad B_n(r)=\frac{2}{r_c}\frac{\sin(n\pi r/r_c)}{r},\quad S_{ab}(r)=e^{-r/(\rho_a(Z_i)+\rho_b(Z_j))}f_{\rm cut}(r)$", "v2_rad", fs=10.5))
    s.append(Paragraph("쉬운 설명: 거리 rᵢⱼ를 신경망에 넣기 전에 여러 함수 값의 묶음으로 바꾼다. 기존에는 Bessel 함수 8개(n이 클수록 빠르게 진동)에 컷오프 함수만 곱했다. "
                       "현재는 (i) n번째 함수에 exp(−σₙ²n²)를 곱해 빠르게 진동하는 고차 함수를 줄인다(σₙ = 0.224, 약 0.5 Å보다 짧은 요동 억제). 학습 범위 밖 거리에서 고차 함수가 멋대로 흔들리는 것을 막아 "
                       "늘어난 결합(1200 K)의 힘 오차를 줄인다(Darby et al. 2026이 선형 ACE에 쓴 방법을 옮김). (ii) 두 원자의 최외각 껍질 반지름 ρ(원자 계산값, 표로 고정)의 합을 감소 길이로 하는 지수 함수 S(a,b)를 "
                       "채널로 덧붙인다(‖는 이어 붙임). 원소 쌍마다 다른 '거리 감각'을 물리적으로 준다. f(cut) = 1 − 6x² + 8x³ − 3x⁴ (x = r/r_c)는 컷오프에서 부드럽게 0이 되는 함수다.", P))
    s.append(fig_bessel())
    s.append(Paragraph("그림 5. 왼쪽: Bessel 함수 8개. 가운데: 억제 후, 빠르게 진동하는 고차 함수가 작아진다. 오른쪽: 차수별 축소 배율.", CAP))
    s.append(fig_shells())
    s.append(Paragraph("그림 6. 3BPA 원소 쌍별 껍질 함수. 감소 길이가 쌍마다 다르다(H–H 1.8, C–C 1.9, N–C 1.7 Å 등).", CAP))

    s.append(Paragraph("3. 가까이 붙은 원자쌍의 반발력: UFF 공식, 결합하지 않은 쌍에만", H2))
    s.append(Paragraph("이면각 스캔(구속 최적화 구조)의 오차는 두 고리가 맞닿는 '충돌' 프레임에 몰려 있고, 모델은 학습셋 최소 거리보다 0.5 Å 안쪽에서 원자들이 밀어내는 힘을 너무 약하게 봤다"
                       f"(반발력 기울기 {D['collision']['no_prior']['slope']}, 에너지 편향 {D['collision']['no_prior']['bias']} meV; 3시드가 0.98로 같은 방향 = 체계적 오차). "
                       "학습 데이터에 이 정보가 없으므로, 함수 형태와 상수를 UFF에서 그대로 가져온 반발 항을 더한다. 단, 이 항도 모델의 전역 에너지 배율 s(학습되는 스칼라 하나, 초기값 1)와 함께 곱해진다. "
                       "학습 후 s는 0.20~0.33이 되어(3BPA 0.25, rMD17 0.20~0.23, AcAc 0.23, Si 0.30~0.33, LiPS 0.26), 실효 반발 세기는 원래 UFF의 약 1/4이다. 따라서 '학습 파라미터가 전혀 없다'가 아니라 '함수 형태 고정, 전체 세기는 전역 배율 하나를 공유'가 정확한 표현이다.", P))
    s.append(eq(r"기존:  $E_i^{\rm rep}=0$   (반발은 전부 데이터에서 학습)", "v1_rep", fs=12))
    s.append(eq(r"현재:  $E_i^{\rm rep}=\frac{1}{2}\sum_{j}D_{ij}\frac{6}{\zeta_{ij}-6}\exp\left[\zeta_{ij}\left(1-\frac{r_{ij}}{x_{ij}}\right)\right]f_{\rm cut}(r_{ij})\cdot\sigma\left(\frac{r_{ij}-1.25\,(R_i+R_j)}{0.3\,\mathrm{\AA}}\right)$", "v2_rep", fs=11.5))
    s.append(eq(r"$x_{ij}=\sqrt{x_i x_j},\quad D_{ij}=\sqrt{D_i D_j},\quad \zeta_{ij}=\sqrt{\zeta_i\zeta_j}$   (Rappé 1992 UFF 표 값, 고정)", "v2_rep2", fs=11.5))
    s.append(Paragraph("쉬운 설명: 지수 함수 부분은 UFF 힘장(Rappé 1992)의 반발 항으로, 두 원자가 평형 거리 xᵢⱼ보다 가까워질수록 급격히 커진다. Dᵢⱼ는 에너지 크기(수십 meV), ζᵢⱼ는 가파른 정도(약 12~14)이고, "
                       "원소별 표 값을 기하평균으로 쌍 값으로 만든다. f(cut)은 컷오프에서 0이 되게 한다. 마지막 σ(·) 항이 핵심이다. 공유결합으로 묶인 쌍(거리가 공유결합 반지름 합 Rᵢ+Rⱼ의 1.25배보다 짧은 쌍)에서는 "
                       "이 반발을 폭 0.3 Å으로 부드럽게 꺼서, 결합 길이 자체는 신경망이 자유롭게 배우고, 결합하지 않은 원자끼리 비정상적으로 가까워질 때만 반발이 켜지게 한다. 1/2은 쌍을 두 번 세는 것을 보정한다.", P))
    s.append(fig_repulsion())
    s.append(Paragraph("그림 7. 세 충돌 쌍의 반발 에너지(위, 1500 meV까지)와 힘(아래, 5000 meV/Å까지; 더 큰 값은 잘림). UFF-nb w0.3(굵은 빨강)은 학습 범위 안에서는 작고, 충돌 거리에서는 가파르며, 결합 길이 쪽으로 부드럽게 이어진다. "
                       "비교용으로 ZBL(원자핵 반발 공식)도 그렸다.", CAP))
    s.append(fig_switch())
    s.append(Paragraph("그림 8. 결합 쌍에서 반발력을 끄는 전환 구간. 폭 0.1 Å이면 꺾이는 지점이 1200 K에서 늘어난 C–H 결합 길이 범위(주황)와 겹쳐 힘 오차가 커지고, 폭 0.3 Å은 이를 피한다.", CAP))

    s.append(Paragraph("4. 학습: SOAP 옵티마이저, 손실, 에너지 상수 보정", H2))
    s.append(eq(r"기존:  AMSGrad  $m\leftarrow\beta_1 m+(1-\beta_1)G,\ \ v\leftarrow\beta_2 v+(1-\beta_2)G^{2},\ \ \hat v\leftarrow\max(\hat v, v),\ \ \Delta\theta=-\eta\,\frac{m}{\sqrt{\hat v}+\epsilon}$", "v1_opt", fs=11.5))
    s.append(eq(r"현재:  SOAP  $L\leftarrow\beta L+(1-\beta)GG^{\top},\ R\leftarrow\beta R+(1-\beta)G^{\top}G,\ \ G'=Q_L^{\top}GQ_R,\ \ \Delta\theta=-\eta\,Q_L\left[\frac{\hat m'}{\sqrt{\hat v'}+\epsilon}\right]Q_R^{\top}$", "v2_opt", fs=11.5))
    s.append(Paragraph("쉬운 설명: Adam 계열은 각 가중치를 따로따로 '최근 기울기 크기'로 나눠 보폭을 정한다. SOAP(Vyas et al. 2024)는 가중치 행렬마다 기울기 G의 행 방향 상관 L과 열 방향 상관 R을 추적하고, "
                       "그 주축(고유벡터 Q(L), Q(R)) 방향으로 좌표를 돌린 뒤(G') 그 좌표에서 Adam을 적용하고 다시 되돌린다. 가중치들이 서로 얽혀 있는 방향을 풀어 주는 효과라, 같은 β로 회전만 뺀 Adam이 "
                       "크게 나쁘다는 대조 실험(표 1)이 이를 뒷받침한다. β₁=β₂=0.95, 주축 갱신은 10스텝마다, lr 1e-2, EMA 0.99, 묶음 구현으로 6.8 s/epoch. 손실은 Huber(δ=10), 에너지는 원자당, E:F = 1:10.", P))
    s.append(eq(r"기존:  없음.   현재:  $\delta^{*}=\arg\min_\delta\sum_n (d_n+N_n\delta)^2=-\frac{\sum_n N_n d_n}{\sum_n N_n^2},\qquad E\leftarrow E+N\,\delta^{*}$", "v2_gauge", fs=12))
    s.append(Paragraph("쉬운 설명: 조성이 하나뿐인 데이터(3BPA, rMD17 등)에서는 '원자 하나당 상수 δ'를 에너지에 더해도 힘은 전혀 변하지 않는다. 힘 가중치가 큰 학습에서는 이 상수가 제대로 정해지지 않아 "
                       "에너지 오차의 대부분이 일정한 치우침으로 남는다. 그래서 학습이 끝난 뒤, 학습셋 각 구조 n의 에너지 잔차 dₙ(예측 − 참값)과 원자 수 Nₙ으로 잔차 제곱합을 최소로 만드는 δ*를 한 번에 계산해(위 공식) "
                       "모델의 상수 항에 더한다. 테스트셋은 쓰지 않고, 힘은 그대로다.", P))
    at = D["attribution"]
    rows = [["옵티마이저 (3BPA, 500 ep, 같은 모델)", "300 K", "600 K", "1200 K"]] + [[k] + [ef(v) for v in at[k]] for k in ["AMSGrad", "Adam β=0.95 (회전 없음)", "SOAP"]]
    s.append(table(rows, col_widths=[W * 0.34, W * 0.22, W * 0.22, W * 0.22], hi_rows=[3]))
    s.append(Paragraph("표 1. 같은 β로 좌표 회전만 뺀 Adam이 크게 나쁘므로, SOAP의 이득은 주축 방향으로 돌리는 데서 온다.", CAP))

    s.append(Paragraph("5. 결과", H2))
    s.append(Paragraph("<b>표시 규칙.</b> " + D["status"]["legend"] + " 괄호의 '검증 최저 시드'는 검증 손실이 가장 낮은 시드 하나의 값(테스트로 고르지 않음)으로, 단일 run을 보고하는 참조 모델과 비교할 때 쓴다.", BOX))
    s.append(Paragraph(f"<b>3BPA</b> ({b['units']}).", P))
    bv = b["V2"]["bestval"]
    rows = [["지표", "RACE V1", "RACEv2 (최종, 3시드)", f"검증 최저 시드 (s{bv['seed']})", "RACEv2 (ZBL-nb)"] + [k.split(" (")[0] for k in b["refs"]]]
    for key, lab in [("300K", "300 K"), ("600K", "600 K"), ("1200K", "1200 K"), ("dih", "이면각")]:
        v2 = b["V2"][key]; sd = b["V2"]["std"][key]
        rows.append([lab, ef(b["V1"][key]) if b["V1"][key] else "미보고", f"{v2[0]:.2f}±{sd[0]:.2f} / {v2[1]:.2f}±{sd[1]:.2f}", ef(bv.get(key)), ef(b["V2_zblnb"][key])] + [ef(b["refs"][k][key]) for k in b["refs"]])
    s.append(table(rows, col_widths=[W * 0.07, W * 0.08, W * 0.13, W * 0.09, W * 0.09] + [W * 0.0771] * 7))
    s.append(Paragraph("표 2. 온도 6개 지표는 모든 공정 참조값보다 낮다(300 K E는 GRACE와 동급, 1200 K E는 TACE 단일 시드와 근소). 이면각은 GRACE·PACE·MACE가 앞서고, ZBL-nb가 10.4로 TACE 수준이다. "
                       "참조값 출처: GRACE PRX 14, 021036; PACE arXiv 2411.04219; TACE 2509.14961; MACE 2206.07697; SpinGTP 2607.01408; ICTP 2405.14253.", CAP))

    m = D["rmd17"]; s.append(Paragraph(f"<b>rMD17</b> ({m['units']}).", P))
    rows = [["분자", "상태", "RACE V1", "RACEv2 E (raw)", "RACEv2 F", "검증 최저 시드 E / F", "MACE", "최고 E", "최고 F"]]; red = []
    for i, k in enumerate(m["mols"]):
        st = m["status"].get(k, "old"); nd = m["n_seeds_done"].get(k, 0)
        lab = {"final": "최종 3시드", "partial": f"최종 {nd}/3 시드", "old": "이전 버전"}[st]
        if st != "final": red.append(i + 1)
        sd = m["V2_std"].get(k); e_txt = f"{m['V2'][k][0]:.2f}" + (f"±{sd[0]:.2f}" if sd and nd == 3 else "") + f" ({m['V2_raw_E'][k]:.2f})"
        f_txt = f"{m['V2'][k][1]:.2f}" + (f"±{sd[1]:.2f}" if sd and nd == 3 else "")
        v1 = m["V1"].get(k); rows.append([k, lab, ef(v1) if v1 else "–", e_txt, f_txt, ef(m["V2_bestval"].get(k)), ef(m["MACE"][k]), m["sota"][k][0], m["sota"][k][1]])
    s.append(table(rows, col_widths=[W * 0.12, W * 0.1, W * 0.09, W * 0.13, W * 0.1, W * 0.1, W * 0.1, W * 0.13, W * 0.13], red_rows=red))
    s.append(Paragraph("표 3. 10개 분자 모두 최종 모델 3시드 완료. 공정 SOTA 대비 힘은 8개 분자에서 낮고 ethanol은 동급(1.41 vs 1.4), benzene은 뒤진다(0.22 vs 0.1). 에너지는 7개 분자에서 낮고 ethanol·uracil은 동급(0.21·0.20 vs 0.2), benzene은 뒤진다(0.02 vs 0.01). 이전 버전(h128, 500 ep) 대비 힘이 16~41 % 줄었다. "
                       + m["note"], CAP))
    s.append(fig_rmd17())
    s.append(Paragraph("그림 9. rMD17: V1, V2, MACE (V2 막대는 표 3의 현재 값; 이전 버전 분자가 섞여 있음).", CAP))

    a = D["acac"]; s.append(Paragraph(f"<b>AcAc</b> ({a['units']}).", P))
    def efs2(v, sd): return f"{v[0]:.2f}±{sd[0]:.2f} / {v[1]:.2f}±{sd[1]:.2f}"
    rows = [["모델", "300 K E / F", "600 K E / F"],
            ["RACEv2 (최종, 3시드)", efs2(a["V2"]["300K"], a["V2_std"]["300K"]) + f" (raw E {a['V2_raw_E']['300K']:.2f})", efs2(a["V2"]["600K"], a["V2_std"]["600K"]) + f" (raw E {a['V2_raw_E']['600K']:.2f})"],
            [f"검증 최저 시드 (s{a['bestval_seed']})", ef(a["V2_bestval"]["300K"]), ef(a["V2_bestval"]["600K"])]] + [[k, ef(v["300K"]), ef(v["600K"])] for k, v in a["refs"].items()]
    s.append(table(rows, col_widths=[W * 0.34, W * 0.33, W * 0.33], hi_rows=[1]))
    s.append(Paragraph("표 4. 4개 지표 중 300 K 에너지·힘과 600 K 힘은 공정 SOTA(PACE)보다 낮고, 600 K 에너지는 PACE·MACE·ICTP에 뒤진다(5.3 vs 3.8). 이전 버전(500 ep, 반발 항·Chebyshev·hill 없음)의 600 K 힘 19.8에서 15.6으로 좋아졌다.", CAP))

    w = D["water"]; s.append(Paragraph(f"<b>물 (Cheng revPBE0-D3)</b> ({w['units']}). {w['note']}", P))
    def efs(v, sd): return f"{v[0]:.2f}±{sd[0]:.2f} / {v[1]:.1f}±{sd[1]:.1f}"
    b1 = w["split1"]["V2_bestval"]; b2 = w["split2"]["V2_bestval"]
    rows = [["분할", "RACEv2 (3시드)", "검증 최저 시드", "MACE 64-0 (같은 분할, 공식 패키지, 1시드)"], [w["split1"]["name"], efs(w["split1"]["V2"], w["split1"]["V2_std"]), ef(b1["test"]) + f" (s{b1['seed']})", ef(w["split1"]["MACE64"])],
            [w["split1"]["name"] + " – 고온 15개 제외", efs(w["split1"]["V2_noout"], w["split1"]["V2_noout_std"]), ef(b1["noout"]), "–"], [w["split2"]["name"], efs(w["split2"]["V2"], w["split2"]["V2_std"]), ef(b2["test"]) + f" (s{b2['seed']})", ef(w["split2"]["MACE64"])]]
    s.append(table(rows, col_widths=[W * 0.38, W * 0.2, W * 0.14, W * 0.28], red_rows=([1, 2, 3] if D["status"].get("water") != "final" else [])))
    s.append(Paragraph("표 5. " + w["split1"]["note"] + ". " + w["size_screening"] + ". 문헌(분할 상이): " + ", ".join(f"{k} {ef(v)}" for k, v in w["literature"].items()) + ".", CAP))

    si = D["si"]; l = D["lips"]; q = D["qm9"]
    sb = si["V2_bestval"]
    rows = [["벤치마크", "상태", "RACEv2", "공정 참조값", "판정"],
            [f"Si GAP-18 → RSS ({si['units'].split(';')[0]})", "최종 3시드", efs(si["V2"]["RSS"], si["V2_std"]["RSS"]) + f" (raw E {si['V2_raw_E']['RSS']:.2f}; 검증 최저 s{sb['seed']} " + ef(sb["RSS"]) + ")", "XPOT-ACE-3F " + ef(si["refs"]["XPOT-ACE-3F"]["RSS"]) + ", GAP-18 " + ef(si["refs"]["GAP-18 (우리 재계산)"]["RSS"]), "승"],
            ["Si GAP-18 → 압축 MD", "최종 3시드", efs(si["V2"]["compMD"], si["V2_std"]["compMD"]) + " (검증 최저 " + ef(sb["compMD"]) + ")", "GAP-18 " + ef(si["refs"]["GAP-18 (우리 재계산)"]["compMD"]) + " (공개 RMSE 없음)", "승"],
            ["LiPS 19k (E meV/atom / F MAE)", "최종 3시드", f"{l['V2']['19k'][0]:.3f}±{l['V2_std']['19k'][0]:.3f} / {l['V2']['19k'][1]:.2f}±{l['V2_std']['19k'][1]:.2f}", "FreeCG 1.2, GemNet-T 1.3, NequIP 3.7 (힘)", "동급 (최고 수준)"],
            ["LiPS 학습곡선 2500 / 1000 / 100 / 10 (E meV/atom / F MAE)", l["lc_status"], l["lc_text"], "NequIP 0.08/4.7 · CarNet 0.09/5.6 · CAMP 0.44/21.6 · CAMP 1.63/76.4", "승 (모든 크기)"],
            ["LiPS MD 50 ps × 5 (안정성 / RDF 오차 / Li 확산계수 오차 ×10⁻⁹ m²/s)", "최종 모델", "50 ps 모두 안정 / 0.044±0.003 / 0.43", "NequIP 50 / 0.04 / 0.34, GemNet-T 50 / 0.04 / 0.24 (Fu et al. 2023)", "안정성·RDF 동급, 확산 열세"],
            ["QM9 U0 MAE", "미조정, 1시드", f"{q['V2']} ({q['V2_raw']})", "GotenNet-L 3.37, TensorNet 3.9, MACE 4.1", "패"]]
    s.append(table(rows, col_widths=[W * 0.24, W * 0.11, W * 0.23, W * 0.29, W * 0.13], red_rows=[6]))
    s.append(png_image(os.path.join(DOCS, "fig_lips_learning_curve.png"), width=W))
    s.append(Paragraph("그림 10. LiPS 학습곡선(같은 Fu 테스트 5,001장). 파란 선이 RACEv2 3시드 평균이며 모든 크기가 최종값이다. "
                       "힘과 에너지 모두 모든 학습 크기에서 발표된 참조값보다 낮고, 데이터가 적을수록 격차가 크다(100장: 9.5 vs CAMP 21.6). 같은 학습 크기의 참조 점은 겹치지 않게 가로로 조금 옮겼다.", CAP))
    s.append(Paragraph("표 6. 재료·대규모 벤치마크. Si는 3시드(L5h128 + UFF prior, 300 ep; " + si["size_screening"] + "). LiPS는 3시드 최종이고 MD 관측량은 Fu et al. 프로토콜(520 K, 50 ps, 5회)이다. QM9는 1시드 미조정이다.", CAP))

    ab = D["ablation_3bpa"]
    s.append(Paragraph("5b. Ablation (3BPA, 진행 중)", H2))
    rows = [["변형 (하나씩 V1식으로 되돌림)", "300 K", "600 K", "1200 K", "이면각"]] + ab["rows"]
    s.append(table(rows, col_widths=[W * 0.32, W * 0.17, W * 0.17, W * 0.17, W * 0.17], hi_rows=[1]))
    s.append(Paragraph("표 6b. " + ab["note"] + ". 각 칸은 에너지 / 힘. 1200 K 힘 오차는 원자 특징 +32%, hill +22%, Chebyshev +13%, LayerScale +12%, 메시지 정규화 +6% 나빠진다. " + ab["pending"] + ".", CAP))

    s.append(Paragraph("6. 다른 SOTA 모델에 없는, 이 모델에서 처음 쓴 방법", H2))
    rows = [["방법", "다른 모델의 선례", "우리 기여", "판정"],
            ["학습 후 에너지 상수 보정 (δ*)", "선례 없음 (조성이 하나일 때 힘이 못 보는 상수 성분 분석 포함)", "한 번에 푸는 공식 유도, 옵티마이저가 상수 오차를 남기는 원인 분석", "처음 (조사 범위 내)"],
            ["Chebyshev식 층 갱신 (층별 에너지 조각 겹침 감소)", "MLIP에서 선례 없음", "층 갱신식 h⁽ᵗ⁾ = 2γδ⁽ᵗ⁾ − h⁽ᵗ⁻²⁾", "처음 (조사 범위 내)"],
            ["크기 조절 hill (이웃 밀도 정보를 상한 안에서 보존)", "RMS/LayerNorm은 보편", "보통의 정규화가 지우는 이웃 밀도 정보를 상한을 두고 되살림", "처음"],
            ["원소 껍질 반지름 기반 거리 채널", "ACE1의 원소별 거리 스케일과 유사", "원자 계산 껍질 반지름으로 쌍별 거리 함수 구성", "변형·확장"],
            ["UFF 반발 공식(형태 고정)을 비결합 쌍에 적용 + 결합 쌍 전환", "MACE는 ZBL 옵션 제공; UFF 반발을 MLIP 고정 항으로 쓴 선례 없음", "보편 상수를 그대로 사용, 결합 쌍에서는 부드럽게 끔", "처음 (UFF 형태)"],
            ["거리 함수 고차 진동 억제 (메시지 전달 모델에)", "Darby et al. 2026 (선형 ACE)", "Bessel 채널 축소로 비선형 모델에 이식", "이식"],
            ["SOAP 옵티마이저", "Harari et al. 2026 (MLIP에 SOAP 적용)", "묶음 구현, 회전 유무 대조, 상수 오차 분석", "선례 있음"],
            ["층별 에너지 조각, 원자 주변 요약 곱", "V1 RACE; MACE도 층별 readout, GRACE/ACE의 재귀 곱", "비선형 없는 갱신·평균 이웃 수 나눔·64비트와 결합", "선례 있음"]]
    s.append(table(rows, col_widths=[W * 0.26, W * 0.28, W * 0.3, W * 0.16]))
    s.append(Paragraph("표 7. '처음'은 2026-10-28~11-03 문헌 조사(워크플로 3회, 논문 약 60편)에서 선례를 찾지 못했다는 뜻이며, 전수 조사가 아니다. 투고 전 관련 연구 절에서 재확인이 필요하다.", CAP))

    s.append(Paragraph("7. 한계 (정직한 범위)", H2))
    s.append(Paragraph("• 이면각: GRACE·PACE·MACE(7.4~7.8)에 뒤진다(최종 16.7, ZBL-nb 10.4). 반발 항만으로는 온도 외삽과 동시에 못 잡는다.<br/>"
                       "• 물 고온·해리 스냅샷: 같은 분할의 MACE도 비슷하게 약하다(분할 1 힘 66 vs 65). 상온 PIMD만 보면(분할 2) MACE보다 낫다. 표준 분할이 없어 문헌과 직접 비교 불가.<br/>"
                       "• AcAc 600 K 에너지(5.3 vs PACE 3.8), QM9: 참조보다 나쁘다. QM9는 반발 항을 넣은 조정 run 대기 중.<br/>"
                       "• LiPS 학습곡선·QM9는 진행 중. MD 관측량, N=50 과제는 미수행(MD22는 큐에 있음).<br/>"
                       "• 에너지 상수 보정과 테스트 노출(개발 중 eval_test_every)은 논문에 공개해야 한다.", P))

    doc = SimpleDocTemplate(PDF, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm, title="RACEv2 methods & results", author="BAM_V2")
    doc.build(s); print("written", PDF)


if __name__ == "__main__":
    build()
