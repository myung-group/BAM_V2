"""LiPS (Materials Cloud 2022.45 lips.xyz, md5 16f91b4a..., 25,001 frames, 83 atoms) -> fair-protocol splits.

usage: venv_tools/bin/python -I convert_lips.py <lips.xyz> <out_dir>

Protocol F (Fu et al., "Forces are not Enough", TMLR 2023; MDsim preprocessing/lips.py):
  pool, test = train_test_split(arange(25001), train_size=20000, test_size=5001, random_state=123)
  train, val = train_test_split(pool, train_size=19000, test_size=1000, random_state=123)
  -> fu_train.xyz (19000), fu_valid.xyz (1000), fu_test.xyz (5001)
Protocol L (NequIP Table VI / CAMP learning curve, train 10/100/1000/2500, valid 1000, test 5000):
  drawn from the Fu pool (train+val, 20000 frames) with RandomState(0); test = fu_test (5001, disjoint).
  -> lc_valid.xyz (1000, shared), lc{N}_train.xyz (nested: lc10 c lc100 c lc1000 c lc2500)
Frames are copied verbatim (text), so energies / forces / cell are untouched.
Writes splits.npz (all indices) and e0.json (mean energy per atom of each training set, eV/atom).
"""
import json, os, sys
import numpy as np
from sklearn.model_selection import train_test_split

NAT = 83


def read_frames(path):
    with open(path) as f:
        lines = f.readlines()
    per = NAT + 2
    assert len(lines) % per == 0, len(lines)
    frames = [lines[i:i + per] for i in range(0, len(lines), per)]
    for fr in frames:
        assert int(fr[0]) == NAT
    return frames


def energy(fr):
    for tok in fr[1].split():
        if tok.startswith("energy="):
            return float(tok.split("=")[1])
    raise ValueError(fr[1])


def write(path, frames, idx):
    with open(path, "w") as fo:
        for i in idx:
            fo.writelines(frames[i])


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    frames = read_frames(src)
    n = len(frames)
    assert n == 25001, n
    E = np.array([energy(fr) for fr in frames])

    pool, test = train_test_split(np.arange(n), train_size=20000, test_size=n - 20000, random_state=123)
    train, val = train_test_split(pool, train_size=19000, test_size=1000, random_state=123)
    assert len(set(train) | set(val) | set(test)) == n

    rng = np.random.RandomState(0)
    perm = rng.permutation(pool)
    lc_val = perm[:1000]
    lc = {N: perm[1000:1000 + N] for N in (10, 100, 1000, 2500)}

    sets = {"fu_train": train, "fu_valid": val, "fu_test": test, "lc_valid": lc_val,
            **{f"lc{N}_train": v for N, v in lc.items()}}
    for name, idx in sets.items():
        write(f"{out}/{name}.xyz", frames, idx)
    np.savez(f"{out}/splits.npz", **sets)
    e0 = {name: float(E[idx].mean() / NAT) for name, idx in sets.items() if name.endswith("train")}
    with open(f"{out}/e0.json", "w") as f:
        json.dump(e0, f, indent=1)
    for name, idx in sets.items():
        print(name, len(idx), idx[:5])
    print(json.dumps(e0, indent=1))


if __name__ == "__main__":
    main()
