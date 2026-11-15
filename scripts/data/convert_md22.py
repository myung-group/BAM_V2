"""MD22 (Chmiela et al., Sci. Adv. 2023) npz -> extxyz train/test with the standard training-set sizes.

    python convert_md22.py [molecule ...]       (default: all seven)

Protocol (MD22 paper Table 1; MACE eval. paper arXiv 2305.14247 uses the same sizes):
  train N = 6000 Ac-Ala3-NHMe, 8000 DHA, 8000 stachyose, 3000 AT-AT, 2000 AT-AT-CG-CG,
            600 buckyball-catcher, 800 double-walled_nanotube; test = all remaining frames.
  Random split, numpy seed 42; validation (5 %) is drawn from the training set by the trainer (valid_frac).
Units: kcal/mol -> eV (x 0.0433641), kcal/mol/A -> eV/A. Energies are total energies.
Writes md22/<mol>/{train,test}.xyz and md22/<mol>/split.json (indices into the npz).
"""
import json
import sys
from pathlib import Path

import numpy as np

KCAL = 0.0433641153087705
NTRAIN = {"Ac-Ala3-NHMe": 6000, "DHA": 8000, "stachyose": 8000, "AT-AT": 3000, "AT-AT-CG-CG": 2000,
          "buckyball-catcher": 600, "double-walled_nanotube": 800}
SYM = {1: "H", 6: "C", 7: "N", 8: "O"}
ROOT = Path(__file__).resolve().parent / "md22"


def write_xyz(path, z, R, E, F, idx):
    with open(path, "w") as f:
        for k in idx:
            f.write(f"{len(z)}\n")
            f.write(f'Properties=species:S:1:pos:R:3:forces:R:3 energy={E[k]:.10f} md22_index={int(k)} pbc="F F F"\n')
            for a in range(len(z)):
                x, y, w = R[k, a]
                fx, fy, fz = F[k, a]
                f.write(f"{SYM[int(z[a])]} {x:.8f} {y:.8f} {w:.8f} {fx:.8f} {fy:.8f} {fz:.8f}\n")


def convert(mol):
    out = ROOT / mol
    if (out / "test.xyz").exists() and (out / "split.json").exists():
        print(mol, "already converted")
        return
    d = np.load(ROOT / f"md22_{mol}.npz")
    assert str(d["e_unit"]) == "kcal/mol" and str(d["r_unit"]) == "Ang", (d["e_unit"], d["r_unit"])
    z, R = d["z"], d["R"]
    E = d["E"].reshape(-1) * KCAL
    F = d["F"] * KCAL
    n = len(E)
    perm = np.random.RandomState(42).permutation(n)
    tr, te = np.sort(perm[:NTRAIN[mol]]), np.sort(perm[NTRAIN[mol]:])
    out.mkdir(parents=True, exist_ok=True)
    write_xyz(out / "train.xyz", z, R, E, F, tr)
    write_xyz(out / "test.xyz", z, R, E, F, te)
    json.dump({"n_frames": int(n), "n_atoms": int(len(z)), "train": tr.tolist(), "test": te.tolist(), "seed": 42,
               "source": f"md22_{mol}.npz", "theory": str(d["theory"])}, open(out / "split.json", "w"))
    print(f"{mol}: {n} frames x {len(z)} atoms -> train {len(tr)} / test {len(te)}")


if __name__ == "__main__":
    for m in (sys.argv[1:] or list(NTRAIN)):
        convert(m)
