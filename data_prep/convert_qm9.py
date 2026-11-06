"""QM9 (figshare 10.6084/m9.figshare.c.978904: dsgdb9nsd.xyz.tar.bz2 + uncharacterized.txt) -> extxyz splits.

usage: python3 -I convert_qm9.py <xyz_dir> <uncharacterized.txt> <out_dir> [seed]

Standard protocol (DimeNet / PaiNN / Allegro / MACE tables): drop the 3,054 uncharacterized molecules
(130,831 left), random split 110,000 train / 10,000 valid / 10,831 test, one model per target.
Target here: U0 (internal energy at 0 K, Hartree -> eV).  QM9 has no forces: zero forces are written
and the configs use force_weight 0.  Fortran exponents "*^" in coordinates are converted.
"""
import glob, os, re, sys
import numpy as np

HARTREE = 27.211386245988  # eV


def parse(path):
    with open(path) as f:
        lines = f.read().splitlines()
    n = int(lines[0])
    props = lines[1].split()
    idx, u0 = int(props[1]), float(props[12]) * HARTREE
    atoms = []
    for ln in lines[2:2 + n]:
        t = ln.replace("*^", "e").split()
        atoms.append((t[0], float(t[1]), float(t[2]), float(t[3])))
    return idx, u0, atoms


def write(path, mols):
    with open(path, "w") as fo:
        for idx, u0, atoms in mols:
            fo.write(f"{len(atoms)}\n")
            fo.write(f'Properties=species:S:1:pos:R:3:forces:R:3 energy={u0!r} qm9_index={idx} pbc="F F F"\n')
            for s, x, y, z in atoms:
                fo.write(f"{s} {x:.10f} {y:.10f} {z:.10f} 0.0 0.0 0.0\n")


def main():
    src, unchar, out = sys.argv[1], sys.argv[2], sys.argv[3]
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    os.makedirs(out, exist_ok=True)
    bad = set()
    with open(unchar) as f:
        for ln in f:
            m = re.match(r"^\s+(\d+)\s+\S", ln)
            if m:
                bad.add(int(m.group(1)))
    assert len(bad) == 3054, len(bad)
    files = sorted(glob.glob(f"{src}/dsgdb9nsd_*.xyz"))
    assert len(files) == 133885, len(files)
    mols = [m for m in map(parse, files) if m[0] not in bad]
    assert len(mols) == 130831, len(mols)
    perm = np.random.RandomState(seed).permutation(len(mols))
    parts = {"train": perm[:110000], "valid": perm[110000:120000], "test": perm[120000:]}
    for name, p in parts.items():
        write(f"{out}/{name}.xyz", [mols[i] for i in p])
        print(name, len(p))
    np.savez(f"{out}/splits_seed{seed}.npz", **{k: np.array([mols[i][0] for i in v]) for k, v in parts.items()})


if __name__ == "__main__":
    main()
