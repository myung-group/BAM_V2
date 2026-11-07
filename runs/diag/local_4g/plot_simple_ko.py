"""Simplified Korean figures of the local physics checks (one message per figure)."""
import json
import numpy as np
import ase.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

fam = next(f for f in ('Malgun Gothic', 'Noto Sans CJK KR', 'Noto Sans CJK JP')
           if any(f == x.name for x in font_manager.fontManager.ttflist))
INK, INK2, MUTED, GRID, BASE, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
C_RACE, C_GLOB, C_FRAG = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.family': fam, 'font.size': 12, 'axes.unicode_minus': False, 'axes.edgecolor': BASE,
                     'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.facecolor': SURF,
                     'figure.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False})
K = 14.399645
OUT = 'runs/local_4g/plots'


def head(fig, title, sub):
    fig.text(0.02, 0.97, title, fontsize=16, color=INK, weight='bold', va='top')
    fig.text(0.02, 0.905, sub, fontsize=11.5, color=INK2, va='top')


# ---------------------------------------------------------------- 1. energy between separated ions
a0 = ase.io.read('data_local/4ghdnnp/nacl/test.xyz', 0)
o = json.load(open('runs/local_4g/nacl_racelr_pauling_q/dissociation_Cl1.json'))
d = np.array(o['d'])
u = a0.positions[1] - a0.positions.mean(0); u /= np.linalg.norm(u)
rest = np.delete(a0.positions, 1, 0); com = rest.mean(0); R = []
for dd in d:
    lo, hi = 0.0, 40.0
    for _ in range(60):
        t = 0.5 * (lo + hi)
        if np.linalg.norm(a0.positions[1] + t * u - rest, axis=1).min() < dd:
            lo = t
        else:
            hi = t
    R.append(np.linalg.norm(a0.positions[1] + hi * u - com))
ideal = -2 * K / np.array(R); ideal -= ideal[-1]
fig, ax = plt.subplots(figsize=(10, 6.2))
fig.subplots_adjust(top=0.80, left=0.09, right=0.97, bottom=0.11)
head(fig, '멀리 떨어진 이온 사이의 끌림을 모델이 계산하는가',
     'NaCl 클러스터(+1)에서 Cl- 하나를 떼어 냄. 남은 덩어리(+2)와 Cl-(-1)는 서로 당기므로,\n'
     '정답은 거리가 멀어질수록 에너지가 1/거리로 천천히 0에 가까워지는 곡선(검은 점선)')
ax.axvspan(d[0], 5.0, color=GRID, alpha=0.6, lw=0)
ax.text(3.75, 0.42, '모델이 직접\n보는 범위', ha='center', va='top', color=INK2, fontsize=10.5)
ax.plot(d, ideal, ls='--', color=INK, lw=2)
for key, c in (('race', C_RACE), ('lr_global', C_GLOB), ('lr_frag', C_FRAG)):
    e = np.array(o[key]); ax.plot(d, e - e[-1], color=c, lw=2.6)
ax.axhline(0, color=BASE, lw=1)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], ls='--', color=INK, lw=2, label='정답: 쿨롱 법칙 (1/거리)'),
                   Line2D([], [], color=C_FRAG, lw=2.6, label='RACE_LR + 조각별 전하 고정  → 정답을 따라감'),
                   Line2D([], [], color=C_GLOB, lw=2.6, label='RACE_LR + 전체 전하만 고정  → 전하가 새서 끌림이 약함'),
                   Line2D([], [], color=C_RACE, lw=2.6, label='RACE (기존)  → 5 Å 밖은 못 봄 (평평)')],
          loc='lower right', frameon=False, fontsize=11.5, labelcolor=INK)
ax.text(0.01, 0.02, '아래로 갈수록 더 강하게 당김', transform=ax.transAxes, color=INK2, fontsize=10.5)
ax.set_xlabel('떼어 낸 거리 (Å)')
ax.set_ylabel('에너지 변화 (eV)')
ax.set_ylim(-3.0, 0.55)
fig.savefig(f'{OUT}/ko_1_ion_attraction.png', dpi=150)

# ---------------------------------------------------------------- 2. charge of the separated atom
oc = json.load(open('runs/local_4g/nacl_racelr_pauling_q/dissociation_Cl1.json'))
on = json.load(open('runs/local_4g/nacl_racelr_pauling_q/dissociation_Na11.json'))
groups = ['Cl-을 30 Å 떼어 냄', 'Na+을 30 Å 떼어 냄']
vals = {'정답': [-1.0, 1.0], '전체 전하만 고정': [oc['q_global'][-1], on['q_global'][-1]],
        '조각별 전하 고정': [oc['q_frag'][-1], on['q_frag'][-1]]}
cols = {'정답': '#898781', '전체 전하만 고정': C_GLOB, '조각별 전하 고정': C_FRAG}
fig, ax = plt.subplots(figsize=(10, 6.0))
fig.subplots_adjust(top=0.80, left=0.09, right=0.97, bottom=0.10)
head(fig, '멀리 떨어진 원자가 자기 전하를 그대로 가지고 있는가',
       '30 Å이나 떨어진 이온은 전하가 정확히 -1 또는 +1이어야 함 (회색 막대 = 정답).\n'
       '조각별 전하 고정은 정답과 같고, 전체 전하만 고정하면 전하가 남은 덩어리 쪽으로 샘')
x = np.arange(2); w = 0.26
for k, (name, v) in enumerate(vals.items()):
    xs = x + (k - 1) * (w + 0.02)
    ax.bar(xs, v, width=w, color=cols[name], label=name, zorder=3)
    for xi, vi in zip(xs, v):
        ax.text(xi, vi + (0.06 if vi >= 0 else -0.06), f'{vi:+.2f}'.replace('-', '-'), ha='center', va='bottom' if vi >= 0 else 'top',
                color=INK, fontsize=12)
ax.axhline(0, color=BASE, lw=1.2, zorder=2)
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=12.5, color=INK)
ax.set_ylabel('떨어진 원자의 전하 (e)')
ax.set_ylim(-1.35, 1.35)
ax.grid(axis='x', visible=False)
ax.legend(frameon=False, loc='upper left', fontsize=11.5, labelcolor=INK2)
fig.savefig(f'{OUT}/ko_2_charge_of_separated_atom.png', dpi=150)

# ---------------------------------------------------------------- 3. MD stability
runs = [('C10H3+', 'carbon_chain'), ('Na9Cl8+', 'nacl')]
fig, ax = plt.subplots(figsize=(10, 5.6))
fig.subplots_adjust(top=0.78, left=0.19, right=0.97, bottom=0.15)
head(fig, '원자를 움직여 보는 시뮬레이션(MD)이 끝까지 안정한가',
     '목표: 2 ps 동안 총에너지가 일정하게 유지되는 것 (점선). 막대 = 안정하게 버틴 시간')
ylab, yv, ycol, notes = [], [], [], []
for name, s in runs:
    for m, c, mlab in (('race', C_RACE, 'RACE'), ('racelr_pauling_q', C_GLOB, 'RACE_LR')):
        dd = np.load(f'runs/local_4g/md/{s}_{m}.npz')
        t_end = float(dd['t_ps'][-1]) if str(dd['stopped']) != 'None' else 2.0
        et = (dd['epot'] + dd['ekin']) * 1e3
        ok = str(dd['stopped']) == 'None'
        ylab.append(f'{name}  {mlab}'); yv.append(t_end); ycol.append(c)
        notes.append(f'에너지 오차 {np.ptp(et - et[0]):.0f} meV 이내, 안정' if ok else
                     f'{t_end:.2f} ps에 붕괴: 학습에 없던 구조로 들어감')
yy = np.arange(len(yv))[::-1]
ax.barh(yy, yv, height=0.55, color=ycol, zorder=3)
ax.axvline(2.0, ls='--', color=INK, lw=2, zorder=4)
ax.text(2.0, len(yv) - 0.35, ' 목표 2 ps', color=INK, fontsize=11.5, va='bottom')
for yi, v, n in zip(yy, yv, notes):
    inside = v > 1.2
    ax.text(0.05 if inside else v + 0.04, yi, n, va='center', color='#ffffff' if inside else INK, fontsize=11.5,
            zorder=5)
ax.set_yticks(yy); ax.set_yticklabels(ylab, fontsize=12.5, color=INK)
ax.set_xlim(0, 2.4); ax.set_xlabel('안정하게 버틴 시간 (ps)')
ax.grid(axis='y', visible=False)
ax.text(0.99, -0.16, 'NaCl은 학습 데이터(구조 1000개)가 좁은 시험용 작은 모델이라 범위 밖에서 무너짐. 전하 계산의 문제는 아님',
        transform=ax.transAxes, ha='right', va='top', color=INK2, fontsize=10.5)
fig.savefig(f'{OUT}/ko_3_md_stability.png', dpi=150)
print('font', fam)
