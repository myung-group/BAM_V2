"""Korean single-message figures: force error at cutoff 5 A vs 3 A, RACE vs RACE_LR (one figure per system).
Ideal: shortening the cutoff does not raise the error (dashed line at each model's 5 A value)."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, BASE, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
C = {'race': '#2a78d6', 'racelr_pauling_q': '#eb6834'}
NAME = {'race': 'RACE (기존)', 'racelr_pauling_q': 'RACE_LR\n(Pauling + 전하 손실)'}
plt.rcParams.update({'font.family': 'Malgun Gothic', 'font.size': 12, 'axes.unicode_minus': False,
                     'axes.edgecolor': BASE, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'hatch.color': '#ffffff', 'hatch.linewidth': 1.2})
S = json.load(open('runs/local_4g/summary_cut3.json'))
SYS = {'carbon_chain': ('탄소 사슬 (공유 결합)',
                        '컷오프를 줄이면 두 모델 모두 크게 나빠짐. 같은 3 Å끼리는 RACE_LR이 36 % 낮지만,\n'
                        '나빠진 비율은 RACE_LR이 더 큼: 이 분자에서는 장거리 항이 3–5 Å 정보를 대신하지 못함'),
       'nacl': ('NaCl 클러스터 (이온 결합)',
                'RACE_LR은 컷오프를 줄여도 거의 그대로(+2.5 %), RACE는 +9 %.\n'
                '3 Å RACE_LR이 5 Å RACE보다도 나음: 이온성 계에서는 장거리 정전기가 짧은 컷오프를 보충함')}
for s, (title, sub) in SYS.items():
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    fig.subplots_adjust(top=0.76, left=0.10, right=0.92, bottom=0.16)
    fig.text(0.02, 0.97, f'컷오프 5 Å → 3 Å, 힘 오차가 얼마나 커지나: {title}', fontsize=14.5, color=INK,
             weight='bold', va='top')
    fig.text(0.02, 0.90, sub, fontsize=11.5, color=INK2, va='top')
    w, gap = 0.34, 0.03
    for k, m in enumerate(['race', 'racelr_pauling_q']):
        f5, f3 = S[f'{s}/{m}/5']['F'], S[f'{s}/{m}/3']['F']
        x5, x3 = k - (w + gap) / 2, k + (w + gap) / 2
        ax.bar(x5, f5, width=w, color=C[m], zorder=3)
        ax.bar(x3, f3, width=w, color=C[m], hatch='//', zorder=3)
        ax.plot([x3 - w / 2, x3 + w / 2], [f5, f5], ls='--', color=INK, lw=2, zorder=5)
        for x, v, lab in ((x5, f5, '5 Å'), (x3, f3, '3 Å')):
            ax.annotate(f'{v:.1f}', (x, v), xytext=(0, 4), textcoords='offset points', ha='center', va='bottom',
                        color=INK, fontsize=12.5, weight='bold')
            ax.text(x, -0.035, lab, transform=ax.get_xaxis_transform(), ha='center', va='top', color=INK2, fontsize=11)
        ax.annotate(f'{(f3 / f5 - 1) * 100:+.0f} %', (x3 + w / 2, f3), xytext=(6, -2), textcoords='offset points',
                    ha='left', va='top', color=INK2, fontsize=11)
    ax.set_xticks([0, 1]); ax.set_xticklabels([NAME[m] for m in ['race', 'racelr_pauling_q']], fontsize=12.5,
                                              color=INK)
    ax.tick_params(axis='x', pad=22)
    ax.set_ylabel('힘 오차 RMSE (meV/Å)')
    ax.set_ylim(0, max(S[f'{s}/{m}/{c}']['F'] for m in ['race', 'racelr_pauling_q'] for c in '35') * 1.22)
    ax.grid(axis='x', visible=False)
    ax.text(0.99, 0.98, '점선 = 이상적인 경우\n(컷오프를 줄여도 5 Å 때 오차 그대로)\n빗금 막대 = 3 Å', transform=ax.transAxes,
            ha='right', va='top', color=INK2, fontsize=10.5)
    fig.savefig(f'runs/local_4g/plots/ko_cutoff_{s}.png', dpi=150)
print('saved')
