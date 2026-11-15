"""Korean single-message figures of the OMol25 pilot (public OpenFF part, neutral molecules, 100 epochs)."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, BASE, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
C = {'race': '#2a78d6', 'racelr': '#eb6834', 'racelr_q': '#1baf7a'}
NAME = {'race': 'RACE (기존)', 'racelr': 'RACE_LR\n(전하 정답 없이)', 'racelr_q': 'RACE_LR\n+ 전하 손실'}
plt.rcParams.update({'font.family': 'Malgun Gothic', 'font.size': 12, 'axes.unicode_minus': False,
                     'axes.edgecolor': BASE, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False})
S = json.load(open('runs/omol/summary_e100.json'))
OUT = 'runs/local_4g/plots'
models = ['race', 'racelr', 'racelr_q']


def head(fig, title, sub):
    fig.text(0.02, 0.97, title, fontsize=16, color=INK, weight='bold', va='top')
    fig.text(0.02, 0.905, sub, fontsize=11.5, color=INK2, va='top')


def bars(key, fname, title, sub, ylabel, fmt, ref=None, ref_label=None, ylim=None):
    fig, ax = plt.subplots(figsize=(9, 5.8))
    fig.subplots_adjust(top=0.78, left=0.11, right=0.97, bottom=0.14)
    head(fig, title, sub)
    vals = [S[f'omoloff10k100/{m}'].get(key, np.nan) for m in models]
    x = np.arange(len(models))
    ax.bar(x, vals, width=0.55, color=[C[m] for m in models], zorder=3)
    for xi, v in zip(x, vals):
        ax.annotate(fmt(v), (xi, v), xytext=(0, -8), textcoords='offset points', ha='center', va='top',
                    color='#ffffff', fontsize=13, weight='bold', zorder=5)
    if ref is not None:
        ax.axhline(ref, ls='--', color=INK, lw=2, zorder=4)
        ax.annotate(ref_label, (0.0, ref), xycoords=('axes fraction', 'data'), xytext=(4, 5),
                    textcoords='offset points', ha='left', va='bottom', color=INK, fontsize=11)
    ax.axhline(0, color=BASE, lw=1.2, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([NAME[m] for m in models], fontsize=12.5, color=INK)
    ax.set_ylabel(ylabel); ax.grid(axis='x', visible=False)
    if ylim:
        ax.set_ylim(*ylim)
    fig.savefig(f'{OUT}/{fname}', dpi=150)


r = S['omoloff10k100/race']
bars('E', 'ko_omol_1_energy.png', '에너지 오차: 장거리 항이 도움이 되었는가',
     'OMol25 공개 사본(중성 분자만), 학습 1만 개, 100 에폭, 시드 1개. 막대가 낮을수록 좋음.\n'
     '점선 = 기존 RACE. 중성 분자만 있는 데이터에서는 장거리 항이 에너지를 좋게 하지 못함',
     '에너지 오차 (meV/원자)', lambda v: f'{v:.1f}', ref=r['E'], ref_label='기존 RACE 수준', ylim=(0, 100))
bars('Fmed', 'ko_omol_2_force.png', '힘 오차: 세 모델이 비슷한가',
     '구조별 힘 오차의 중앙값. 막대가 낮을수록 좋음. 점선 = 기존 RACE.\n'
     '전하 손실을 넣은 RACE_LR이 약 2 % 낮지만, 시드 하나라 차이는 잡음 수준일 수 있음',
     '힘 오차, 중앙값 (meV/Å)', lambda v: f'{v:.1f}', ref=r['Fmed'], ref_label='기존 RACE 수준', ylim=(0, 180))
fig, ax = plt.subplots(figsize=(9, 5.8))
fig.subplots_adjust(top=0.78, left=0.11, right=0.97, bottom=0.14)
head(fig, '전하 정확도: 모델 전하가 DFT 전하와 맞는가',
     '모델 전하와 DFT(Löwdin) 전하의 상관계수. 1이면 완벽하게 일치(점선 = 정답).\n'
     '전하 정답 없이 배우면 오히려 반대 경향(음수). 전하 손실을 넣어야 맞음')
m2 = ['racelr', 'racelr_q']
vals = [S[f'omoloff10k100/{m}']['qr'] for m in m2]
x = np.arange(2)
ax.bar(x, vals, width=0.5, color=[C[m] for m in m2], zorder=3)
for xi, v in zip(x, vals):
    if v >= 0.2:
        ax.annotate(f'{v:+.2f}', (xi, v), xytext=(0, -8), textcoords='offset points', ha='center', va='top',
                    color='#ffffff', fontsize=13, weight='bold', zorder=5)
    else:
        ax.annotate(f'{v:+.2f}', (xi, v), xytext=(0, -5), textcoords='offset points', ha='center', va='top',
                    color=INK, fontsize=13)
ax.axhline(1.0, ls='--', color=INK, lw=2, zorder=4)
ax.annotate('정답: 1.0 (완전 일치)', (0.0, 1.0), xycoords=('axes fraction', 'data'), xytext=(4, 5),
            textcoords='offset points', ha='left', va='bottom', color=INK, fontsize=11)
ax.axhline(0, color=BASE, lw=1.2, zorder=2)
ax.set_xticks(x); ax.set_xticklabels([NAME[m] for m in m2], fontsize=12.5, color=INK)
ax.set_ylabel('상관계수 (DFT 전하 대비)'); ax.set_ylim(-0.4, 1.2); ax.grid(axis='x', visible=False)
fig.savefig(f'{OUT}/ko_omol_3_charges.png', dpi=150)
print('saved')
