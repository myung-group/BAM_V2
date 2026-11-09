"""OOD uncertainty summary for the 3BPA setting of the BAM V1 paper (train 300 K; test 300 / 600 / 1200 K).

Reads the per-structure dumps of bayes_lastlayer.py (per_structure_<set>.npz with F_sig_tot_mean, sigE_tot, ...)
and soap_laplace.py (F_sig_mean, sigE, ...) and prints, per method and test set: force RMSE of the network,
mean sigma, rms z, 95 % coverage, Spearman(|err|, sigma) over components, and the OOD-detection AUROC of the
structure-level sigma (300 K = in-distribution, 600 K / 1200 K = out-of-distribution; the oBN25 metric of the paper,
Table 5).  Writes a Korean figure: structure-mean force sigma vs structure force RMSE per temperature.

    python runs/diag/ood/ood_summary.py --blr <dir> --lap <name>=<dir> ... --out <dir> [--sets test_300K test_600K test_1200K]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def auroc(neg, pos):
    """P(score_pos > score_neg) by the Mann-Whitney U statistic (ties count one half)."""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    allv = np.concatenate([neg, pos])
    ranks = np.empty_like(allv)
    order = np.argsort(allv, kind='mergesort'); sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    r_pos = ranks[len(neg):].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load(dir_, name, kind):
    z = np.load(Path(dir_) / f'per_structure_{name}.npz')
    if kind == 'blr':
        return {'F_err': z['F_err'], 'F_sig': z['F_sig_tot'], 'F_err_rms': z['F_err_rms'], 'F_sig_mean': z['F_sig_tot_mean'],
                'E_err': z['mE'] - z['yE'], 'E_sig': z['sigE_tot']}
    if kind == 'blr_epi':       # the epistemic (posterior) part alone, without the fitted noise
        return {'F_err': z['F_err'], 'F_sig': z['F_sig_epi'], 'F_err_rms': z['F_err_rms'], 'F_sig_mean': z['F_sig_epi_mean'],
                'E_err': z['mE'] - z['yE'], 'E_sig': z['sigE_epi']}
    return {'F_err': z['F_err'], 'F_sig': z['F_sig'], 'F_err_rms': z['F_err_rms'], 'F_sig_mean': z['F_sig_mean'],
            'E_err': z['netE'] - z['yE'], 'E_sig': z['sigE']}


def stats(d):
    z = d['F_err'] / d['F_sig']
    return {'F_rmse': 1e3 * float(np.sqrt(np.mean(d['F_err'] ** 2))), 'F_sigma': 1e3 * float(np.mean(d['F_sig'])),
            'F_rms_z': float(np.sqrt(np.mean(z ** 2))), 'F_cov95': float(np.mean(np.abs(z) < 1.96)),
            'F_spearman': float(spearmanr(np.abs(d['F_err']), d['F_sig']).correlation),
            'F_spearman_struct': float(spearmanr(d['F_err_rms'], d['F_sig_mean']).correlation),
            'E_rmse': 1e3 * float(np.sqrt(np.mean(d['E_err'] ** 2))), 'E_sigma': 1e3 * float(np.mean(d['E_sig'])),
            'E_rms_z': float(np.sqrt(np.mean((d['E_err'] / d['E_sig']) ** 2)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--blr', default=None); ap.add_argument('--lap', nargs='*', default=[])
    ap.add_argument('--sets', nargs='+', default=['test_300K', 'test_600K', 'test_1200K'])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    methods = {}
    if a.blr:
        methods['② 마지막 층 BLR'] = ('blr', a.blr)
        methods['② BLR, 인식론적 항만'] = ('blr_epi', a.blr)
    for item in a.lap:
        name, d = item.split('=', 1)
        methods[name] = ('lap', d)
    report = {}
    hdr = f"{'method':>28} {'set':>10} {'F rmse':>7} {'sigma':>7} {'rms z':>6} {'cov95':>6} {'rho':>6} {'rho_s':>6} {'E rmse':>7} {'E sig':>7} {'E z':>6}"
    print(hdr)
    data = {}
    for m, (kind, d) in methods.items():
        report[m] = {}
        for sname in a.sets:
            dd = load(d, sname, kind); data[(m, sname)] = dd
            st = stats(dd); report[m][sname] = st
            print(f"{m:>28} {sname:>10} {st['F_rmse']:7.1f} {st['F_sigma']:7.1f} {st['F_rms_z']:6.2f} {st['F_cov95']:6.2f} "
                  f"{st['F_spearman']:6.2f} {st['F_spearman_struct']:6.2f} {st['E_rmse']:7.1f} {st['E_sigma']:7.1f} {st['E_rms_z']:6.2f}")
        idn = a.sets[0]
        for sname in a.sets[1:]:
            aF = auroc(data[(m, idn)]['F_sig_mean'], data[(m, sname)]['F_sig_mean'])
            aE = auroc(data[(m, idn)]['E_sig'], data[(m, sname)]['E_sig'])
            report[m][f'auroc_{sname}'] = {'F_sigma': aF, 'E_sigma': aE}
            print(f"{m:>28} OOD AUROC {idn} vs {sname}: sigma_F {aF:.3f}, sigma_E {aE:.3f}")
    json.dump(report, open(out / 'ood_summary.json', 'w'), indent=1)
    # figure: structure-mean force sigma vs structure force RMSE, one panel per method, colour per temperature
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fam = next((f for f in ('Noto Sans CJK KR', 'Noto Sans CJK JP', 'Noto Sans CJK HK', 'NanumGothic', 'DejaVu Sans')
                if any(f == x.name for x in font_manager.fontManager.ttflist)), 'DejaVu Sans')
    plt.rcParams.update({'font.family': fam, 'axes.unicode_minus': False})
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 4.0), squeeze=False)
    colors = {'test_300K': '#1f77b4', 'test_600K': '#ff7f0e', 'test_1200K': '#d62728'}
    for ax, m in zip(axes[0], methods):
        lo, hi = np.inf, 0
        for sname in a.sets:
            dd = data[(m, sname)]
            x, y = 1e3 * dd['F_sig_mean'], 1e3 * dd['F_err_rms']
            ax.scatter(x, y, s=6, alpha=0.5, color=colors.get(sname, 'k'), label=sname.replace('test_', '').replace('K', ' K'))
            lo, hi = min(lo, x.min(), y.min()), max(hi, x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='이상적 (σ = 오차)')
        ax.set_xscale('log'); ax.set_yscale('log')
        from matplotlib.ticker import ScalarFormatter, NullFormatter
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(ScalarFormatter()); axis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('구조 평균 힘 σ (meV/Å)'); ax.set_ylabel('구조 힘 RMSE (meV/Å)'); ax.set_title(m)
        ax.legend(fontsize=8)
    fig.suptitle('3BPA: 300 K 학습, 600 / 1200 K 외삽 — 불확정도 대 오차')
    fig.tight_layout(); fig.savefig(out / 'ood_sigma_vs_error.png', dpi=150)
    print(f"written {out / 'ood_summary.json'}, {out / 'ood_sigma_vs_error.png'}")


if __name__ == '__main__':
    main()
