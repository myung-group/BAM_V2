"""LiPS MD observables exactly as in MDsim/observable.ipynb (Fu et al., TMLR 2023, Table 6).

    python -m bam_omat24.md_lips_analysis <md dir with run*/> --ref /scratch/t589a02/datasets_ext/lips_src/lips.xyz

Per run: stability = first 1-ps window (80 saved frames) whose RDF MAE x 6 vs the reference exceeds 1.0
(else the full 50 ps); RDF error over the stable part; Li diffusivity (if stable > 3200 frames = 40 ps):
smoothed MSD over frames [400:collapse:4] (drop 5 ps, 0.05-ps spacing), value at index 700 (35 ps window),
x 20 x 1e-8 -> m^2/s; error = |D - D_ref| with D_ref from the reference trajectory (frames 2500::25).
RDF: minimum-image distances over the 27 neighbouring cells, 500 bins on (1e-6, 6] A, normalised by the
ideal-gas shell count with rho = N^2/V (the notebook's definition).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read

XLIM, NBINS = 6.0, 500
BINS = np.linspace(1e-6, XLIM, NBINS + 1)


def dist_matrix_min(cell, pos):
    """pos (T, N, 3) -> (T, N, N) minimum distance over the 27 cell shifts (MDsim definition)."""
    rng = np.arange(-1, 2)
    combos = np.stack(np.meshgrid(rng, rng, rng, indexing="xy"), -1).reshape(-1, 3)
    shifts = combos @ cell                                             # (27, 3)
    d = pos[:, :, None, None, :] - (pos[:, None, :, None, :] + shifts[None, None, None])
    return np.sqrt((d ** 2).sum(-1)).min(-1)


def frame_hists(pos, cell, chunk=20):
    """per-frame pair-distance histograms (T, NBINS) and the per-frame pair count N^2 (zeros included)."""
    out = []
    for i in range(0, len(pos), chunk):
        dm = dist_matrix_min(cell, pos[i:i + chunk])
        for f in dm:
            v = f.reshape(-1)
            out.append(np.histogram(v[v != 0], BINS)[0])
    return np.asarray(out, dtype=np.float64), pos.shape[1] ** 2


def rdf_from_hists(h, n_pairs, n_frames, cell):
    vol = abs(np.dot(np.cross(cell[0], cell[1]), cell[2]))
    rho = n_frames * n_pairs / vol
    return h / (rho * 4 / 3 * np.pi * (BINS[1:] ** 3 - BINS[:-1] ** 3))


def rdf(pos, cell):
    h, npair = frame_hists(pos, cell)
    return rdf_from_hists(h.sum(0), npair, len(pos), cell)


def mae(x, y):
    return np.abs(x - y).mean() * XLIM


def smoothed_diff(xyz):
    """xyz (T, N, 3) -> D(t) averaged over all time origins (MDsim get_smoothed_diff)."""
    T = xyz.shape[0]
    seq = T - 1
    diff = np.zeros(seq)
    for i in range(seq):
        p = xyz[i:]
        p = p - p.mean(1, keepdims=True)
        msd = ((p[1:] - p[0][None]) ** 2).sum(-1).mean(-1)
        diff[: seq - i] += msd / np.arange(1, T - i) / 6
    return diff / np.arange(seq, 0, -1)


def unwrap_traj(pos, cell):
    inv = np.linalg.inv(cell)
    frac = pos @ inv
    d = np.diff(frac, axis=0)
    d -= np.round(d)
    frac_u = np.concatenate([frac[:1], frac[:1] + np.cumsum(d, axis=0)], axis=0)
    return frac_u @ cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("md_dir")
    ap.add_argument("--ref", default="/scratch/t589a02/datasets_ext/lips_src/lips.xyz")
    ap.add_argument("--ref-cache", default="/scratch/t589a02/datasets_ext/lips/md_reference.npz")
    args = ap.parse_args()
    cache = Path(args.ref_cache)
    if cache.exists():
        c = np.load(cache); gt_rdf, gt_diff, cell = c["rdf"], c["diff"], c["cell"]
    else:
        frames = read(args.ref, index=":")
        cell = np.asarray(frames[0].cell.array)
        z = frames[0].numbers
        pos = unwrap_traj(np.stack([a.positions for a in frames]), cell)
        gt_rdf = rdf(pos, cell)
        gt_diff = smoothed_diff(pos[2500::25][:, z == 3]) * 20 * 1e-8
        np.savez(cache, rdf=gt_rdf, diff=gt_diff, cell=cell)
    print(f"reference: D_ref(idx 700) = {gt_diff[700]:.3e} m^2/s  (paper 1.35e-9)")
    rows = []
    for run in sorted(Path(args.md_dir).glob("run*")):
        meta = json.load(open(run / "meta.json"))
        z = np.array(meta["numbers"])
        chunks = sorted(run.glob("traj_*.npz"))
        if not chunks:
            continue
        traj = np.concatenate([np.load(c)["pos"] for c in chunks]).astype(np.float64)
        ncheck = 80
        fh, npair = frame_hists(traj, cell)
        csum = np.concatenate([np.zeros((1, NBINS)), np.cumsum(fh, 0)])
        collapse = len(traj)
        for i in range(1, len(traj) - ncheck):
            if mae(rdf_from_hists(csum[i + ncheck] - csum[i], npair, ncheck, cell), gt_rdf) > 1.0:
                collapse = i
                break
        r = rdf_from_hists(csum[collapse] - csum[0], npair, collapse, cell)
        row = {"run": run.name, "frames": len(traj), "stable_ps": collapse * 0.0125, "rdf_mae": float(mae(r, gt_rdf))}
        if collapse > 3200:
            d = smoothed_diff(traj[400:collapse:4][:, z == 3]) * 20 * 1e-8
            row["D"] = float(d[700]); row["D_err_1e-9"] = float(abs(d[700] - gt_diff[700]) * 1e9)
        rows.append(row)
        print(row, flush=True)
    if rows:
        st = np.array([r["stable_ps"] for r in rows]); rm = np.array([r["rdf_mae"] for r in rows])
        de = np.array([r["D_err_1e-9"] for r in rows if "D_err_1e-9" in r])
        summ = {"n_runs": len(rows), "stability_ps": [float(st.mean()), float(st.std())], "rdf_mae": [float(rm.mean()), float(rm.std())],
                "diffusivity_mae_1e-9": float(np.abs(np.mean([r["D"] for r in rows if "D" in r]) - gt_diff[700]) * 1e9) if len(de) else None,
                "per_run_D_err_1e-9": de.tolist(), "D_ref": float(gt_diff[700])}
        json.dump({"runs": rows, "summary": summ}, open(Path(args.md_dir) / "observables.json", "w"), indent=1)
        print("SUMMARY", summ)


if __name__ == "__main__":
    main()
