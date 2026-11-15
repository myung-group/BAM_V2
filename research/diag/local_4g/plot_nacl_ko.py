"""Korean single-message figures for NaCl: data size and spin input (Na8Cl8+ force error), MD stability."""
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, BASE, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
C_RACE, C_LR = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.family': 'Malgun Gothic', 'font.size': 12, 'axes.unicode_minus': False,
                     'axes.edgecolor': BASE, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False})
S = json.load(open('runs/local_4g/summary_nacl_spin.json'))
OUT = 'runs/local_4g/plots'


def frame(title, sub):
    fig, ax = plt.subplots(figsize=(9.8, 6.0))
    fig.subplots_adjust(top=0.76, left=0.11, right=0.97, bottom=0.14)
    fig.text(0.02, 0.97, title, fontsize=15, color=INK, weight='bold', va='top')
    fig.text(0.02, 0.90, sub, fontsize=11.5, color=INK2, va='top')
    return fig, ax


# 1. Na8Cl8+ force error vs data size and spin input
conds = [('학습 1000개', 'nacl_race', 'nacl_racelr_pauling_q'),
         ('학습 4500개', 'nacl_race_full', 'nacl_racelr_pauling_q_full'),
         ('4500개 + 스핀 입력', 'nacl_race_cs_full', 'nacl_racelr_pauling_q_cs_full')]
fig, ax = frame('Na8Cl8+을 맞추려면 무엇이 필요한가: 스핀 입력',
                'Na8Cl8+ 힘 오차. 낮을수록 좋음. 두 구조 모두 총전하 +1이라 총전하로는 구분 불가.\n'
                'Na8Cl8+은 전자 수가 홀수(이중항). 스핀을 넣자 오차가 약 20배 줄어 닫힌 껍질 Na9Cl8+ 수준(점선)에 근접')
w = 0.36
for j, (name, col, idx) in enumerate([('RACE', C_RACE, 1), ('RACE_LR', C_LR, 2)]):
    vals = [S[c[idx]]['Cl8Na8+']['F'] for c in conds]
    xs = np.arange(3) + (j - 0.5) * (w + 0.03)
    ax.bar(xs, vals, width=w, color=col, label=name, zorder=3)
    for x, v in zip(xs, vals):
        ax.annotate(f'{v:.1f}', (x, v), xytext=(0, 4), textcoords='offset points', ha='center', va='bottom',
                    color=INK, fontsize=12.5, weight='bold')
ref = np.mean([S['nacl_race_full']['Cl8Na9+']['F'], S['nacl_racelr_pauling_q_full']['Cl8Na9+']['F']])
ax.axhline(ref, ls='--', color=INK, lw=2, zorder=2)
ax.plot([], [], ls='--', color=INK, lw=2, label=f'Na9Cl8+(닫힌 껍질) 수준, 약 {ref:.0f}')
ax.set_xticks(range(3)); ax.set_xticklabels([c[0] for c in conds], fontsize=12.5, color=INK)
ax.set_ylabel('Na8Cl8+ 힘 오차 RMSE (meV/Å)'); ax.set_ylim(0, 122); ax.grid(axis='x', visible=False)
ax.legend(frameon=False, fontsize=11.5, loc='upper right', labelcolor=INK)
fig.savefig(f'{OUT}/ko_nacl_1_spin.png', dpi=150)

# 2. MD stability: stable runs out of 3 velocity seeds
runs = [('RACE', 'nacl_race', C_RACE), ('RACE_LR', 'nacl_racelr_pauling_q', C_LR)]
fig, ax = frame('데이터를 늘리면 MD가 안정해지는가',
                'NaCl 2 ps NVE MD, 초기 속도 시드 3개 중 끝까지 안정한 횟수. 점선 = 목표(3번 모두 안정).\n'
                '학습 1000개는 두 모델 모두 같은 시드 하나에서 붕괴, 4500개는 모두 안정 (에너지 흔들림 0.01 meV)')
for j, (name, base, col) in enumerate(runs):
    vals = []
    for suffix in ('', '_full'):
        ok = 0
        for f in sorted(glob.glob(f'runs/local_4g/md/{base}{suffix}_v*.npz')):
            ok += str(np.load(f)['stopped']) == 'None'
        vals.append(ok)
    xs = np.arange(2) + (j - 0.5) * (w + 0.03)
    ax.bar(xs, vals, width=w, color=col, label=name, zorder=3)
    for x, v in zip(xs, vals):
        ax.annotate(f'{v} / 3', (x, v), xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', color=INK,
                    fontsize=12.5, weight='bold')
ax.axhline(3, ls='--', color=INK, lw=2, zorder=4)
ax.set_xticks(range(2)); ax.set_xticklabels(['학습 1000개', '학습 4500개'], fontsize=12.5, color=INK)
ax.set_ylabel('안정하게 끝난 MD 수 (3개 중)'); ax.set_ylim(0, 3.9); ax.set_yticks([0, 1, 2, 3])
ax.grid(axis='x', visible=False)
ax.legend(frameon=False, fontsize=11.5, loc='upper left', labelcolor=INK)
fig.savefig(f'{OUT}/ko_nacl_2_md.png', dpi=150)
print('saved')
