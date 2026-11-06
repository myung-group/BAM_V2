"""ColabFit rMD17 parquet -> per-molecule extxyz: train_1000.xyz (seeded random 1000) and test_2000.xyz
(seeded random 2000 of the remaining frames).  Energies eV, forces eV/A (as stored by ColabFit)."""
import sys, json, numpy as np, pyarrow.parquet as pq
root = sys.argv[1]; out = sys.argv[2]
cs = {r["id"]: r["name"] for r in pq.read_table(f"{root}/cs/cs_0.parquet").to_pylist()}
cmap = {}
for r in pq.read_table(f"{root}/cs_co_map/cs_co_map_0.parquet").to_pylist():
    cmap[r["configuration_id"]] = cs[r["configuration_set_id"]]
cols = ["configuration_id", "energy", "atomic_forces", "positions", "atomic_numbers", "configuration_metadata"]
per = {m: [] for m in cs.values()}
for f in (f"{root}/co/co_0.parquet", f"{root}/co/co_1.parquet"):
    pf = pq.ParquetFile(f)
    for i in range(pf.num_row_groups):
        for r in pf.read_row_group(i, columns=cols).to_pylist():
            m = cmap.get(r["configuration_id"])
            if m is None: continue
            idx = json.loads(r["configuration_metadata"]).get("MD17-index", -1)
            per[m].append((idx, r["energy"], r["atomic_numbers"], r["positions"], r["atomic_forces"]))
SYM = {1: "H", 6: "C", 7: "N", 8: "O"}
def write(path, rows):
    with open(path, "w") as fo:
        for idx, e, z, pos, frc in rows:
            fo.write(f"{len(z)}\n")
            fo.write(f'Properties=species:S:1:pos:R:3:forces:R:3 energy={e!r} md17_index={idx} pbc="F F F"\n')
            for zi, p, fc in zip(z, pos, frc):
                fo.write(f"{SYM[zi]} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {fc[0]:.10f} {fc[1]:.10f} {fc[2]:.10f}\n")
for m, rows in per.items():
    rows.sort(key=lambda r: r[0])
    rng = np.random.RandomState(1234)
    perm = rng.permutation(len(rows))
    tr = [rows[i] for i in perm[:1000]]; te = [rows[i] for i in perm[1000:3000]]
    import os; os.makedirs(f"{out}/{m}", exist_ok=True)
    write(f"{out}/{m}/train_1000.xyz", tr); write(f"{out}/{m}/test_2000.xyz", te)
    nb = 0; zs = sorted(set(rows[0][2]))
    print(m, len(rows), "frames; natoms", len(rows[0][2]), "elements", zs)
