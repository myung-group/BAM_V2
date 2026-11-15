"""4G-HDNNP benchmark systems (Ko, Finkler, Goedecker, Behler, Nat. Commun. 12, 398 (2021)) -> extended XYZ.

usage: python -I scripts/data/convert_4ghdnnp.py <raw_dir> <out_dir>
  raw_dir: the directory that holds datasets.tar.gz extracted (datasets/<System>/input.data)
  out_dir: one sub-directory per system is written (carbon_chain, ag_cluster, nacl, aumgo)

Source: Materials Cloud Archive 2020.137, DOI 10.24435/materialscloud:f3-yh (record bk6kw-4b895), CC BY 4.0,
datasets.tar.gz md5 0aa11ea305b3af6a47bb3c577c2729bc.  The archive is read as data only (no script is run).

Input format (RuNNer input.data; readme.txt of the record): one structure per begin ... end block with
  lattice x y z                              (3 lines, periodic systems only)
  atom x y z element q_atom e_atom fx fy fz  (q_atom = Hirshfeld charge, e_atom unused = 0)
  energy E                                   (total DFT energy, FHI-aims PBE light)
  charge Q                                   (total charge of the structure)
All in atomic units: Bohr, Hartree, e, Hartree/Bohr.

Output per system, in eV, Angstrom, eV/Angstrom:
  all.xyz     every structure in archive order
  train.xyz   90 %, test.xyz 10 %: RandomState(42).permutation(n), first round(0.9 n) = train.  The paper used a
              random 90/10 split but did not publish the indices, so this split is ours.
  split.json  the train / test indices (0-based archive order) and the counts per subset and charge
  e0.json     per-element reference energies (eV), least squares fit of E = sum_Z n_Z E0_Z on the training split
              only (minimum-norm solution if the compositions do not determine every element).  The archive has
              no isolated-atom energies; these offsets only keep the float32 targets of the trainer well resolved.
extxyz keys: info energy (eV), charge (int, total charge in e: use "total_charge_key": "charge"), subset (charge
state / structure family), source_index (block index in input.data); arrays forces (eV/A), hirshfeld_charges (e).
Periodic structures (lattice lines present, Au2 on MgO(001)) get the cell and pbc = (T, T, T): the DFT was a 3D
periodic slab with vacuum along c (3x3x1 k points).  Molecules and clusters get pbc = (F, F, F) and no cell.
"""
import json
import os
import sys
from collections import Counter

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
import ase.io

HARTREE = 27.211386245988          # eV (CODATA 2018)
BOHR = 0.529177210903              # Angstrom (CODATA 2018)

SYSTEMS = {                        # archive directory -> output directory
    "Carbon_chain": "carbon_chain",
    "Ag_cluster": "ag_cluster",
    "NaCl": "nacl",
    "AuMgO": "aumgo",
}
SEED, TRAIN_FRAC = 42, 0.9


def read_runner(path):
    """Yield dicts with lattice (3x3 or None), symbols, pos, q_atom, forces, energy, charge (atomic units)."""
    s = None
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            tok = line.split()
            if not tok:
                continue
            key = tok[0]
            if key == "begin":
                assert s is None, f"{path}:{ln}: nested begin"
                s = dict(lattice=[], symbols=[], pos=[], q=[], forces=[], energy=None, charge=None)
            elif key == "end":
                assert s is not None and s["energy"] is not None and s["charge"] is not None and s["symbols"], \
                    f"{path}:{ln}: incomplete structure"
                assert len(s["lattice"]) in (0, 3), f"{path}:{ln}: {len(s['lattice'])} lattice lines"
                yield s
                s = None
            elif key == "lattice":
                s["lattice"].append([float(x) for x in tok[1:4]])
            elif key == "atom":
                assert len(tok) == 10, f"{path}:{ln}: {line!r}"
                s["pos"].append([float(x) for x in tok[1:4]])
                s["symbols"].append(tok[4])
                s["q"].append(float(tok[5]))
                assert float(tok[6]) == 0.0, f"{path}:{ln}: nonzero atomic energy"
                s["forces"].append([float(x) for x in tok[7:10]])
            elif key == "energy":
                s["energy"] = float(tok[1])
            elif key == "charge":
                s["charge"] = float(tok[1])
            elif key == "comment":
                pass
            else:
                raise ValueError(f"{path}:{ln}: unknown keyword {key!r}")
    assert s is None, f"{path}: missing final end"


def subset_label(system, symbols, charge):
    c = Counter(symbols)
    formula = "".join(f"{el}{c[el] if c[el] > 1 else ''}" for el in sorted(c))
    if system == "AuMgO":
        return "Au2-MgO-Al-doped" if c.get("Al", 0) else "Au2-MgO-undoped"
    sign = "" if charge == 0 else ("+" if charge > 0 else "-")
    return formula + sign


def to_atoms(system, i, s):
    q_tot = s["charge"]
    assert abs(q_tot - round(q_tot)) < 1e-8, (system, i, q_tot)
    q_tot = int(round(q_tot))
    pos = np.asarray(s["pos"]) * BOHR
    if s["lattice"]:
        atoms = Atoms(s["symbols"], positions=pos, cell=np.asarray(s["lattice"]) * BOHR, pbc=True)
    else:
        atoms = Atoms(s["symbols"], positions=pos, pbc=False)
    atoms.arrays["hirshfeld_charges"] = np.asarray(s["q"])
    atoms.info.update(charge=q_tot, subset=subset_label(system, s["symbols"], q_tot), source_index=i)
    atoms.calc = SinglePointCalculator(atoms, energy=s["energy"] * HARTREE,
                                       forces=np.asarray(s["forces"]) * (HARTREE / BOHR))
    return atoms


def fit_e0(frames, idx):
    """Per-element reference energies (eV) from the training frames (least squares, minimum norm)."""
    elems = sorted({z for i in idx for z in frames[i].get_atomic_numbers()})
    X = np.array([[np.sum(frames[i].get_atomic_numbers() == z) for z in elems] for i in idx], float)
    y = np.array([frames[i].get_potential_energy() for i in idx])
    coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return dict(atomic_numbers=[int(z) for z in elems], atom_energies=[float(c) for c in coef], rank=int(rank),
                n_elements=len(elems), residual_std_eV=float(resid.std()), residual_max_abs_eV=float(abs(resid).max()))


def counts(frames, idx):
    return dict(Counter(f"{frames[i].info['subset']} (Q={frames[i].info['charge']:+d})" for i in idx))


def main():
    raw, out = sys.argv[1], sys.argv[2]
    for system, name in SYSTEMS.items():
        frames = [to_atoms(system, i, s) for i, s in
                  enumerate(read_runner(os.path.join(raw, "datasets", system, "input.data")))]
        n = len(frames)
        perm = np.random.RandomState(SEED).permutation(n)
        n_train = int(round(TRAIN_FRAC * n))
        train, test = np.sort(perm[:n_train]), np.sort(perm[n_train:])
        d = os.path.join(out, name)
        os.makedirs(d, exist_ok=True)
        ase.io.write(os.path.join(d, "all.xyz"), frames, format="extxyz")
        ase.io.write(os.path.join(d, "train.xyz"), [frames[i] for i in train], format="extxyz")
        ase.io.write(os.path.join(d, "test.xyz"), [frames[i] for i in test], format="extxyz")
        with open(os.path.join(d, "split.json"), "w") as f:
            json.dump(dict(source=f"datasets/{system}/input.data", n=n, seed=SEED,
                           rule="RandomState(42).permutation(n); first round(0.9 n) = train; indices sorted",
                           n_train=len(train), n_test=len(test), counts_all=counts(frames, range(n)),
                           counts_train=counts(frames, train), counts_test=counts(frames, test),
                           train=train.tolist(), test=test.tolist()), f)
        with open(os.path.join(d, "e0.json"), "w") as f:
            json.dump(fit_e0(frames, train), f, indent=1)
        print(f"{system}: {n} structures -> {d} (train {len(train)}, test {len(test)}); {counts(frames, range(n))}")


if __name__ == "__main__":
    main()
