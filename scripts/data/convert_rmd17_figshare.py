"""Official rMD17 (figshare 10.6084/m9.figshare.12672038 v4) npz + split csv -> per-molecule extxyz.

usage: python -I convert_rmd17_figshare.py <figshare_dir> <out_dir> [split ...]   (split = 01..05, default 01)

Per molecule and split k writes
  train_split{k}.xyz     : the 1000 frames of index_train_{k}.csv (order of the csv)
  test_split{k}.xyz      : the 1000 frames of index_test_{k}.csv
  rest_split{k}.xyz      : all 99,000 frames not in index_train_{k}.csv
Energies kcal/mol -> eV, forces kcal/mol/A -> eV/A (1 kcal/mol = 4.184 / 96.48533212 eV).
The npz files are loaded with allow_pickle=False (data only).
"""
import os, sys
import numpy as np

KCAL = 4.184 / 96.48533212          # eV per kcal/mol
SYM = {1: "H", 6: "C", 7: "N", 8: "O"}
MOLS = ["aspirin", "azobenzene", "benzene", "ethanol", "malonaldehyde",
        "naphthalene", "paracetamol", "salicylic", "toluene", "uracil"]


def read_index(path):
    idx = np.loadtxt(path, dtype=np.int64, ndmin=1)
    assert idx.ndim == 1 and len(np.unique(idx)) == len(idx), path
    return idx


def write(path, z, R, E, F, idx):
    sym = [SYM[int(a)] for a in z]
    with open(path, "w") as fo:
        for i in idx:
            fo.write(f"{len(z)}\n")
            fo.write(f'Properties=species:S:1:pos:R:3:forces:R:3 energy={float(E[i])!r} rmd17_index={int(i)} pbc="F F F"\n')
            for s, p, f in zip(sym, R[i], F[i]):
                fo.write(f"{s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {f[0]:.10f} {f[1]:.10f} {f[2]:.10f}\n")


def main():
    src, out = sys.argv[1], sys.argv[2]
    splits = sys.argv[3:] or ["01"]
    idx = {k: (read_index(f"{src}/index_train_{k}.csv"), read_index(f"{src}/index_test_{k}.csv")) for k in splits}
    for k, (tr, te) in idx.items():
        assert len(tr) == 1000 and len(te) == 1000 and not set(tr) & set(te), k
    for m in MOLS:
        d = np.load(f"{src}/rmd17_{m}.npz", allow_pickle=False)
        z, R = d["nuclear_charges"], d["coords"]
        E, F = d["energies"] * KCAL, d["forces"] * KCAL
        n = len(E)
        assert R.shape == (n, len(z), 3) and F.shape == R.shape, m
        os.makedirs(f"{out}/{m}", exist_ok=True)
        for k, (tr, te) in idx.items():
            assert tr.max() < n and te.max() < n, (m, k)
            rest = np.setdiff1d(np.arange(n), tr)
            write(f"{out}/{m}/train_split{k}.xyz", z, R, E, F, tr)
            write(f"{out}/{m}/test_split{k}.xyz", z, R, E, F, te)
            write(f"{out}/{m}/rest_split{k}.xyz", z, R, E, F, rest)
            print(f"{m} split {k}: n={n} natoms={len(z)} train {len(tr)} test {len(te)} rest {len(rest)} "
                  f"E range {E.min():.3f}..{E.max():.3f} eV  |F| max {np.abs(F).max():.3f} eV/A", flush=True)


if __name__ == "__main__":
    main()
