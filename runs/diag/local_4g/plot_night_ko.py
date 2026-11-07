"""Korean single-message figures of the overnight local queue: seed repeats, short-range switch, charge-loss weight."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, BASE, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
C_LR, C_RACE = '#eb6834', '#2a78d6'
C_SYS = {'carbon_chain': '#2a78d6', 'nacl': '#1baf7a'}
NAME = {'carbon_chain': '탄소 사슬\n(공유 결합)', 'nacl': 'NaCl 클러스터\n(이온 결합)'}
plt.rcParams.update({'font.family': 'Malgun Gothic', 'font.size': 12, 'axes.unicode_minus': False,
                     'axes.edgecolor': BASE, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False})
S = json.load(open('runs/local_4g/summary_night.json'))
OUT = 'runs/local_4g/plots'
SYS = ['carbon_chain', 'nacl']


def head(fig, title, sub):
    fig.text(0.02, 0.97, title, fontsize=15, color=INK, weight='bold', va='top')
    fig.text(0.02, 0.90, sub, fontsize=11.5, color=INK2, va='top')


def frame(title, sub):
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    fig.subplots_adjust(top=0.76, left=0.12, right=0.88, bottom=0.15)
    head(fig, title, sub)
    return fig, ax


# ------------------------------------------------ 1. seeds: RACE_LR force change vs RACE, per seed
fig, ax = frame('시드 3개로 반복해도 RACE_LR이 RACE보다 나은가',
                '막대 = RACE 대비 힘 오차 변화(시드 3개 평균), 점 = 시드별 값, 점선 0 % = RACE와 같음.\n'
                '아래로 갈수록 좋음. 탄소 사슬 약 -49 %, NaCl 약 -11 %: 세 시드 모두 같은 방향')
for k, s in enumerate(SYS):
    race = np.mean([S[f'{s}/race/s{x}']['F'] for x in (42, 43, 44)])
    rel = [(S[f'{s}/racelr_pauling_q/s{x}']['F'] / race - 1) * 100 for x in (42, 43, 44)]
    ax.bar(k, np.mean(rel), width=0.5, color=C_SYS[s], zorder=3)
    ax.scatter(np.full(3, k) + np.array([-0.08, 0, 0.08]), rel, s=60, color=SURF, edgecolor=INK, zorder=5, lw=1.5)
    ax.annotate(f'평균 {np.mean(rel):+.0f} %', (k + 0.25, np.mean(rel)), xytext=(8, 0), textcoords='offset points',
                ha='left', va='center', color=INK, fontsize=13, weight='bold')
ax.axhline(0, ls='--', color=INK, lw=2, zorder=4)
ax.annotate('RACE와 같음 (0 %)', (1.0, 0), xycoords=('axes fraction', 'data'), xytext=(0, 5), textcoords='offset points',
            ha='right', va='bottom', color=INK, fontsize=11)
ax.set_xticks([0, 1]); ax.set_xticklabels([NAME[s] for s in SYS], fontsize=12.5, color=INK)
ax.set_ylabel('RACE 대비 힘 오차 변화 (%)'); ax.set_ylim(-62, 12); ax.grid(axis='x', visible=False)
fig.savefig(f'{OUT}/ko_night_1_seeds.png', dpi=150)

# ------------------------------------------------ 2. short-range switch
fig, ax = frame('단거리 스위치 [3, 5] Å를 켜면 어떻게 되나',
                '스위치 = 5 Å 안쪽 원자쌍의 정전기를 끔. 막대 = 스위치 없는 RACE_LR 대비 힘 오차 변화.\n'
                '점선 0 % = 변화 없음, 아래로 갈수록 좋음. 이온성 NaCl에서는 장거리 항의 이득이 사라짐')
for k, s in enumerate(SYS):
    base = S[f'{s}/racelr_pauling_q/s42']['F']
    v = (S[f'{s}/racelr_pauling_q_sw35']['F'] / base - 1) * 100
    ax.bar(k, v, width=0.5, color=C_SYS[s], zorder=3)
    ax.annotate(f'{v:+.0f} %', (k, v), xytext=(0, 5 if v >= 0 else -5), textcoords='offset points', ha='center',
                va='bottom' if v >= 0 else 'top', color=INK, fontsize=13, weight='bold')
    rr = (S[f'{s}/race/s42']['F'] / base - 1) * 100
    ax.plot([k - 0.3, k + 0.3], [rr, rr], color=C_RACE if False else MUTED, lw=2, ls=':', zorder=4)
    ax.annotate('RACE 수준', (k + 0.3, rr), xytext=(4, 0), textcoords='offset points', ha='left', va='center',
                color=INK2, fontsize=10.5)
ax.axhline(0, ls='--', color=INK, lw=2, zorder=4)
ax.set_xticks([0, 1]); ax.set_xticklabels([NAME[s] for s in SYS], fontsize=12.5, color=INK)
ax.set_ylabel('스위치 없을 때 대비 힘 오차 변화 (%)'); ax.set_ylim(-15, 140); ax.grid(axis='x', visible=False)
fig.savefig(f'{OUT}/ko_night_2_switch.png', dpi=150)

# ------------------------------------------------ 3. charge-loss weight
fig, ax = frame('전하 손실 가중치는 얼마가 좋은가',
                '가중치 0.1을 기준(0 %, 점선)으로 한 힘 오차 변화. 아래로 갈수록 좋음.\n'
                '0.01은 둘 다 나쁨. 공유 결합 분자는 0.1, 이온 결합 NaCl은 1.0이 가장 좋음 (시드 1개)')
ws = [0.01, 0.1, 1.0]
for s in SYS:
    base = S[f'{s}/racelr_pauling_q/s42']['F']
    v = [(S[f'{s}/racelr_pauling_q_w001']['F'] / base - 1) * 100, 0.0,
         (S[f'{s}/racelr_pauling_q_w1']['F'] / base - 1) * 100]
    ax.plot(ws, v, color=C_SYS[s], lw=2.5, marker='o', ms=9, mec=SURF, mew=2, zorder=4)
    ax.annotate(NAME[s].replace('\n', ' '), (ws[-1], v[-1]), xytext=(8, 0), textcoords='offset points', ha='left',
                va='center', color=INK, fontsize=11.5)
    for w, y in zip(ws, v):
        if w != 0.1:
            ax.annotate(f'{y:+.0f} %', (w, y), xytext=(0, 9), textcoords='offset points', ha='center', va='bottom',
                        color=INK2, fontsize=10.5)
ax.axhline(0, ls='--', color=INK, lw=2, zorder=3)
ax.set_xscale('log'); ax.set_xticks(ws); ax.set_xticklabels(['0.01', '0.1 (기준)', '1.0'], fontsize=12, color=INK)
ax.set_xlim(0.006, 4.0); ax.set_ylabel('가중치 0.1 대비 힘 오차 변화 (%)'); ax.set_ylim(-20, 30)
ax.set_xlabel('전하 손실 가중치')
fig.savefig(f'{OUT}/ko_night_3_weight.png', dpi=150)
print('saved')
