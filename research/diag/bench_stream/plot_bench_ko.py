"""Korean single-message figures of the pickle vs stream benchmark (research/diag/bench_stream/out/*.json)."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, BASE, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
PK, ST = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.family': 'Malgun Gothic', 'font.size': 12, 'axes.unicode_minus': False,
                     'axes.edgecolor': BASE, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False})
D = 'research/diag/bench_stream/out'
SETS = [('omol10k', 'OMol 분자 1만 개\n(평균 16원자)'), ('aumgo', 'Au2/MgO 주기계 4,500개\n(110원자, 지금 코드)'),
        ('aumgo_vesin', 'Au2/MgO 주기계\n(이웃 목록 vesin)')]
R = {k: json.load(open(f'{D}/{k}.json')) for k, _ in SETS}


def rate(r, key):
    return float(np.mean([x['structures_per_s'] for x in r['C'][key]]))


def head(fig, title, sub):
    fig.text(0.02, 0.97, title, fontsize=16, color=INK, weight='bold', va='top')
    fig.text(0.02, 0.905, sub, fontsize=11.5, color=INK2, va='top')


def grouped(fname, title, sub, ylabel, vals_of, ref, ref_label, fmt, ylim=None):
    ws = [k for k in R[SETS[0][0]]['C'] if k.startswith('stream_w')]
    keys = ['pickle'] + ws
    names = ['피클'] + [f'스트리밍\n워커 {k[8:]}' for k in ws]
    fig, ax = plt.subplots(figsize=(12, 6.2))
    fig.subplots_adjust(top=0.78, left=0.09, right=0.98, bottom=0.2)
    head(fig, title, sub)
    wbar = 0.8 / len(keys)
    for si, (s, lab) in enumerate(SETS):
        vs = vals_of(R[s], keys)
        for ki, (k, v) in enumerate(zip(keys, vs)):
            x = si + (ki - (len(keys) - 1) / 2) * wbar
            ax.bar(x, v, width=wbar * 0.92, color=PK if k == 'pickle' else ST, zorder=3,
                   alpha=1.0 if k == 'pickle' else 0.55 + 0.45 * ki / len(keys))
            ax.annotate(fmt(v), (x, v), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                        color=INK, fontsize=10)
            ax.annotate(names[ki], (x, 0), xytext=(0, -6), textcoords='offset points', ha='center', va='top',
                        color=INK2, fontsize=9)
    ax.axhline(ref, ls='--', color=INK, lw=2, zorder=4)
    ax.text(0.01, 0.97, '- - -  ' + ref_label, transform=ax.transAxes, ha='left', va='top', color=INK, fontsize=11.5,
            weight='bold')
    ax.set_xticks(range(len(SETS))); ax.set_xticklabels([l for _, l in SETS], fontsize=12.5, color=INK)
    ax.tick_params(axis='x', pad=34, length=0)
    ax.set_ylabel(ylabel); ax.grid(axis='x', visible=False)
    if ylim:
        ax.set_ylim(*ylim)
    fig.savefig(f'{D}/{fname}', dpi=150)
    plt.close(fig)


# 1. training speed relative to the GPU-only ideal (same batches already in memory)
def rel(r, keys):
    out = []
    for k in keys:
        ideal = rate(r, 'gpu_only_pickle_batches' if k == 'pickle' else 'gpu_only_stream_batches')
        out.append(100 * rate(r, k) / ideal)
    return out


grouped('fig1_train_speed_ko.png', '학습 속도: 피클과 스트리밍이 같은가?',
        '100 % = 같은 배치를 미리 메모리에 올려 두고 GPU만 돌린 속도 (데이터를 기다리는 시간 0).\n'
        '막대가 점선에 붙어 있으면 데이터 읽기가 학습을 늦추지 않는다는 뜻입니다.',
        '이상값 대비 학습 속도 (%)', rel, 100, '이상값 100 % (데이터 대기 없음)', lambda v: f'{v:.0f}',
        ylim=(0, 135))


# 2. loader alone vs what the GPU consumes: headroom factor
def headroom(r, keys):
    g = rate(r, 'gpu_only_pickle_batches')
    return [r['B'][k] / g for k in keys]


grouped('fig2_loader_headroom_ko.png', '데이터 로더의 여유: GPU가 몇 배 빨라져도 버티나?',
        '로더 혼자 1초에 만드는 구조 수 / 이 PC의 GPU(SM 50 %)가 1초에 학습하는 구조 수.\n'
        '1보다 크면 GPU가 데이터를 기다리지 않습니다. GH200은 이 PC보다 몇 배 빠르므로 여유가 그만큼 필요합니다.',
        '로더 속도 / GPU 학습 속도 (배)', headroom, 1, '1배 (이보다 낮으면 GPU가 기다림)', lambda v: f'{v:.1f}')
print('ok')
