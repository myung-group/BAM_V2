"""Korean single-message figures for Au2/MgO (1000 training structures, 60 epochs, periodic)."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, BASE, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
C_RACE, C_LR, C_DFT = '#2a78d6', '#eb6834', '#898781'
plt.rcParams.update({'font.family': 'Malgun Gothic', 'font.size': 12, 'axes.unicode_minus': False,
                     'axes.edgecolor': BASE, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False})
S = json.load(open('runs/local_4g/summary_aumgo_n1000.json'))
DOP, UND = 'Au2-MgO-Al-doped', 'Au2-MgO-undoped'
OUT = 'runs/local_4g/plots'


def frame(title, sub):
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    fig.subplots_adjust(top=0.76, left=0.11, right=0.97, bottom=0.12)
    fig.text(0.02, 0.97, title, fontsize=15, color=INK, weight='bold', va='top')
    fig.text(0.02, 0.90, sub, fontsize=11.5, color=INK2, va='top')
    return fig, ax


def bars(ax, groups, series, fmt):
    w = 0.36
    for j, (name, col, vals) in enumerate(series):
        xs = np.arange(len(groups)) + (j - 0.5) * (w + 0.03)
        ax.bar(xs, vals, width=w, color=col, label=name, zorder=3)
        for x, v in zip(xs, vals):
            ax.annotate(fmt(v), (x, v), xytext=(0, 4 if v >= 0 else -4), textcoords='offset points', ha='center',
                        va='bottom' if v >= 0 else 'top', color=INK, fontsize=12.5, weight='bold')
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, fontsize=12.5, color=INK)
    ax.grid(axis='x', visible=False)
    ax.legend(frameon=False, fontsize=11.5, loc='upper right', labelcolor=INK)


# 1. energy error by subset
fig, ax = frame('Au2/MgO: 10 Å 아래 Al 도핑을 모델이 에너지에 반영하는가',
                '시험 분할 에너지 오차(110원자 구조). 낮을수록 좋음. RACE는 5 Å 밖의 도핑을 못 봐서 도핑 슬랩 오차가 3배.\n'
                'RACE_LR은 도핑 슬랩 오차를 절반으로 줄임 (학습 1000개, 60 에폭, 시드 1개)')
r, l = S['aumgo_race_n1000'], S['aumgo_racelr_pauling_q_n1000']
bars(ax, ['도핑 없는 슬랩', 'Al 도핑 슬랩'],
     [('RACE (기존)', C_RACE, [r[UND]['E'], r[DOP]['E']]), ('RACE_LR', C_LR, [l[UND]['E'], l[DOP]['E']])],
     lambda v: f'{v:.0f}')
ax.set_ylabel('에너지 오차 RMSE (meV)'); ax.set_ylim(0, 185)
fig.savefig(f'{OUT}/ko_aumgo_1_energy.png', dpi=150)

# 2. Au2 charge: DFT vs model, undoped and doped
q = S['au2_charge']
fig, ax = frame('Au2 전하: 도핑이 Au2를 더 음(-)으로 만드는 효과를 재현하는가',
                f'Au 원자 두 개의 전하 합 평균. 회색 = DFT(Hirshfeld, 정답), 주황 = RACE_LR.\n'
                f'DFT는 도핑으로 {q[DOP]["dft"] - q[UND]["dft"]:+.2f} e 변함. 모델은 방향은 맞지만 크기가 '
                f'{q[DOP]["model"] - q[UND]["model"]:+.2f} e로 작음 (약 13 %)')
bars(ax, ['도핑 없는 슬랩', 'Al 도핑 슬랩'],
     [('DFT (정답)', C_DFT, [q[UND]['dft'], q[DOP]['dft']]), ('RACE_LR', C_LR, [q[UND]['model'], q[DOP]['model']])],
     lambda v: f'{v:+.2f}')
ax.axhline(0, color=BASE, lw=1.2, zorder=2)
ax.set_ylabel('Au2 전하 (e)'); ax.set_ylim(-0.3, 0.18)
ax.legend(frameon=False, fontsize=11.5, loc='upper right', labelcolor=INK)
fig.savefig(f'{OUT}/ko_aumgo_2_au2_charge.png', dpi=150)
print('saved')
