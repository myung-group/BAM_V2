"""QM9 extra targets (U, H, G, ZPVE, Cv) with exactly the same split as the U0 run.

usage: python3 -I convert_qm9_props.py <dsgdb9nsd.xyz.tar.bz2> <qm9 dir holding splits_seed42.npz> <out_root>

Reads the raw tarball as one stream (no unpacked file tree), keeps the molecules listed in the U0 split
file (uncharacterized ones are already excluded there) and writes <out_root>/qm9_<target>/{train,valid,test}.xyz
in the same order as the U0 files.  QM9 line 2: gdb idx A B C mu alpha homo lumo gap r2 zpve U0 U H G Cv.
Units: U, H, G, ZPVE Hartree -> eV; Cv kept in cal/(mol K) (stored in the "energy" field, so a logged
"meV" value for Cv is 1e3 x cal/(mol K)).  Zero forces are written (energy-only training, force_weight 0).
"""
import os
import sys
import tarfile

import numpy as np

HARTREE = 27.211386245988
TARGETS = {"U": (13, HARTREE), "H": (14, HARTREE), "G": (15, HARTREE), "ZPVE": (11, HARTREE), "Cv": (16, 1.0)}


def parse(text):
    lines = text.splitlines()
    n = int(lines[0])
    props = lines[1].split()
    atoms = []
    for ln in lines[2:2 + n]:
        t = ln.replace("*^", "e").split()
        atoms.append((t[0], float(t[1]), float(t[2]), float(t[3])))
    return int(props[1]), props, atoms


def main():
    tar_path, qm9_dir, out_root = sys.argv[1], sys.argv[2], sys.argv[3]
    split = np.load(os.path.join(qm9_dir, "splits_seed42.npz"))
    want = set(int(i) for k in ("train", "valid", "test") for i in split[k])
    mols = {}
    with tarfile.open(tar_path, "r:bz2") as tf:
        for m in tf:
            if not m.isfile():
                continue
            idx, props, atoms = parse(tf.extractfile(m).read().decode())
            if idx in want:
                mols[idx] = (props, atoms)
    assert len(mols) == 130831, len(mols)
    for tgt, (col, unit) in TARGETS.items():
        out = os.path.join(out_root, f"qm9_{tgt}")
        os.makedirs(out, exist_ok=True)
        for part in ("train", "valid", "test"):
            with open(os.path.join(out, f"{part}.xyz"), "w") as fo:
                for idx in split[part]:
                    props, atoms = mols[int(idx)]
                    val = float(props[col].replace("*^", "e")) * unit
                    fo.write(f"{len(atoms)}\n")
                    fo.write(f'Properties=species:S:1:pos:R:3:forces:R:3 energy={val!r} qm9_index={int(idx)} pbc="F F F"\n')
                    for s, x, y, z in atoms:
                        fo.write(f"{s} {x:.10f} {y:.10f} {z:.10f} 0.0 0.0 0.0\n")
        print(tgt, {p: len(split[p]) for p in ("train", "valid", "test")}, "->", out)


if __name__ == "__main__":
    main()
