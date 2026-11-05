"""Compare the per-atom errors of several trained runs (seeds) saved by ``analyze_errors.py``.

For every split and for the force error e_k (run k, per atom component):
  * MSE_k, mean MSE over runs
  * bias / variance split:  mean_k MSE_k = |mean_k e_k|^2 + mean_k |e_k - mean_k e_k|^2
    (bias = the part all seeds share = systematic model-class / data limitation;
     variance = seed-dependent part = what averaging / regularisation could remove)
  * pairwise correlation of the error vectors between runs
  * RMSE of the k-run ensemble mean prediction (= sqrt of the bias term)
  * the same for the energy error per frame
  * bias / variance per species and per novelty bin (edges from the json of the first run)

Numpy only: safe to run on the login node.

Usage:
    python -m bam_omat24.analyze_errors_compare <prefix1> <prefix2> ... [--out json]
    (prefix = the --out prefix given to analyze_errors; reads <prefix>.npz / .json)
"""
import argparse
import json

import numpy as np

SPECIES_NAMES = {1: "H", 6: "C", 7: "N", 8: "O"}


def _rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = [np.load(p + ".npz") for p in args.prefixes]
    rep0 = json.load(open(args.prefixes[0] + ".json"))
    K = len(runs)
    z = runs[0]["z_of_species"]
    sets = sorted({k.split("/")[0] for k in runs[0].files if "/" in k},
                  key=lambda s: ["train", "valid", "test_300K", "test_600K", "test_1200K"].index(s)
                  if s in ("train", "valid", "test_300K", "test_600K", "test_1200K") else 99)
    report = {}
    for s in sets:
        F_ref = runs[0][f"{s}/F_ref"]
        E_ref = runs[0][f"{s}/E_ref"]
        species = runs[0][f"{s}/species"]
        nov = runs[0][f"{s}/novelty"]
        errs = []
        dEs = []
        for r in runs:
            assert np.allclose(r[f"{s}/F_ref"], F_ref)
            errs.append(r[f"{s}/F_s"].sum(0) - F_ref)                          # (F, N, 3)
            natoms = r[f"{s}/natoms"]
            # E_s already excludes the shift; the shift is identical across seeds only if trained the
            # same way, so use the stored per-run total: E_s.sum + shift*natoms is not stored -> use
            # json E_rmse for the absolute number and the de-meaned error here
            dE = r[f"{s}/E_s"].sum(0) - E_ref
            dEs.append(dE - dE.mean())
        e = np.stack(errs)                                                       # (K, F, N, 3)
        mean_e = e.mean(0)
        mse_k = np.mean(e ** 2, axis=(1, 2, 3))
        bias2 = np.mean(mean_e ** 2)
        var = np.mean((e - mean_e[None]) ** 2)
        corr = np.corrcoef(e.reshape(K, -1))
        off = corr[~np.eye(K, dtype=bool)]
        dE = np.stack(dEs)
        mean_dE = dE.mean(0)
        e_bias2 = np.mean(mean_dE ** 2); e_var = np.mean((dE - mean_dE[None]) ** 2)
        e_corr = np.corrcoef(dE)[~np.eye(K, dtype=bool)]
        rep = dict(n_runs=K, F_rmse_runs_meVA=[float(np.sqrt(m)) * 1e3 for m in mse_k],
                   F_rmse_mean_run_meVA=float(np.sqrt(mse_k.mean())) * 1e3,
                   F_rmse_ensemble_meVA=float(np.sqrt(bias2)) * 1e3,
                   F_bias_fraction=float(bias2 / (bias2 + var)), F_err_corr_mean=float(off.mean()),
                   E_rmse_demeaned_runs_meV=[_rmse(d) * 1e3 for d in dE],
                   E_rmse_demeaned_ensemble_meV=float(np.sqrt(e_bias2)) * 1e3,
                   E_bias_fraction=float(e_bias2 / (e_bias2 + e_var)), E_err_corr_mean=float(e_corr.mean()))
        print(f"\n[{s}]  runs: F RMSE {np.round(rep['F_rmse_runs_meVA'], 2)} meV/A   mean-run {rep['F_rmse_mean_run_meVA']:.2f}")
        print(f"  force error: bias^2 fraction {rep['F_bias_fraction']:.3f}  (seed-shared part),  "
              f"mean pairwise corr {rep['F_err_corr_mean']:.3f},  {K}-ensemble RMSE {rep['F_rmse_ensemble_meVA']:.2f}")
        print(f"  energy error (de-meaned): runs {np.round(rep['E_rmse_demeaned_runs_meV'], 2)} meV, "
              f"bias^2 fraction {rep['E_bias_fraction']:.3f}, corr {rep['E_err_corr_mean']:.3f}, "
              f"ensemble {rep['E_rmse_demeaned_ensemble_meV']:.2f}")
        # per species
        rows = []
        for sidx in sorted(set(species.tolist())):
            m = species == sidx
            es = e[:, :, m]
            me = es.mean(0)
            b2 = np.mean(me ** 2); v = np.mean((es - me[None]) ** 2)
            nm = SPECIES_NAMES.get(int(z[sidx]), "?")
            rows.append(dict(species=nm, F_rmse_mean_run_meVA=float(np.sqrt(b2 + v)) * 1e3,
                             F_rmse_ensemble_meVA=float(np.sqrt(b2)) * 1e3, bias_fraction=float(b2 / (b2 + v))))
            print(f"    {nm}: mean-run {rows[-1]['F_rmse_mean_run_meVA']:7.2f}  ensemble {rows[-1]['F_rmse_ensemble_meVA']:7.2f}"
                  f"  bias fraction {rows[-1]['bias_fraction']:.3f}")
        rep["per_species"] = rows
        # per novelty bin
        edges = [float(b["bin"].split(",")[1].rstrip("A)")) for b in rep0[s]["novelty_bins"] if not b["bin"].endswith("inf)A")]
        edges = sorted(set(edges))
        idx = np.digitize(nov, edges)
        rows = []
        for b in range(len(edges) + 1):
            m = idx == b
            if m.sum() == 0:
                continue
            es = e[:, m]
            me = es.mean(0)
            b2 = np.mean(me ** 2); v = np.mean((es - me[None]) ** 2)
            lo = "-inf" if b == 0 else f"{edges[b-1]:.3f}"; hi = "inf" if b == len(edges) else f"{edges[b]:.3f}"
            rows.append(dict(bin=f"[{lo},{hi})", frac=float(m.mean()), F_rmse_mean_run_meVA=float(np.sqrt(b2 + v)) * 1e3,
                             F_rmse_ensemble_meVA=float(np.sqrt(b2)) * 1e3, bias_fraction=float(b2 / (b2 + v))))
            print(f"    novelty {rows[-1]['bin']:>16s} frac {m.mean():.3f}: mean-run {rows[-1]['F_rmse_mean_run_meVA']:7.2f}"
                  f"  ensemble {rows[-1]['F_rmse_ensemble_meVA']:7.2f}  bias fraction {rows[-1]['bias_fraction']:.3f}")
        rep["per_novelty_bin"] = rows
        report[s] = rep
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
