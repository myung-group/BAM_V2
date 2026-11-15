"""Checks of race_v2_torch that need no JAX reference (login-node size: a few small structures, single thread).

  1. Clebsch-Gordan tensors (l <= 3) and spherical harmonics against e3nn-torch (wigner_3j, spherical_harmonics
     'component'), whose conventions coincide with e3nn-jax.
  2. Normalised-silu constant: grid value and the Gauss-Hermite integral it approximates.
  3. With a converted model: invariance of E under rotation / translation / permutation, equivariance of F and
     stress, F = -dE/dx and stress = (1/V) dE/d(strain) by central finite differences, supercell extensivity.

    race_v2_torch/tools/tpy race_v2_torch/tests/selfcheck.py --model model.pt
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from race_v2_torch import o3  # noqa: E402

torch.set_default_dtype(torch.float64)
ap = argparse.ArgumentParser()
ap.add_argument('--model', default=None)
a = ap.parse_args()
rows = []


def report(name, val, tol):
    ok = val <= tol
    rows.append((name, val, tol, ok))
    print(f"  {'ok  ' if ok else 'FAIL'} {name:58s} {val:10.3e}  (tol {tol:.0e})", flush=True)


print("1. Clebsch-Gordan / spherical harmonics vs e3nn-torch")
from e3nn import o3 as eo3  # noqa: E402  (tests only)
worst = 0.0
for l1 in range(4):
    for l2 in range(4):
        for l3 in range(abs(l1 - l2), min(l1 + l2, 3) + 1):
            c = torch.as_tensor(o3.clebsch_gordan(l1, l2, l3))
            w = eo3.wigner_3j(l1, l2, l3, dtype=torch.float64)
            worst = max(worst, float((c - w).abs().max()))
report("max |CG_port - e3nn.wigner_3j| over l1,l2,l3 <= 3", worst, 1e-13)
v = torch.randn(200, 3)
y_port = o3.spherical_harmonics(3, v)
y_e3 = eo3.spherical_harmonics([0, 1, 2, 3], v, normalize=True, normalization='component')
report("max |Y_port - e3nn Y| (lmax 3, component)", float((y_port - y_e3).abs().max()), 1e-13)
R = eo3.rand_matrix()
D = eo3.Irreps('0e+1o+2e+3o').D_from_matrix(R)
report("SH equivariance  Y(Rv) - D(R) Y(v)", float((o3.spherical_harmonics(3, v @ R.T) - y_port @ D.T).abs().max()), 1e-13)

print("2. normalised silu constant")
c = o3.normalize_constant('silu')
xg, wg = np.polynomial.hermite_e.hermegauss(200)
f = xg / (1 + np.exp(-xg))
c_int = math.sqrt(float(np.sum(wg * f ** 2) / math.sqrt(2 * math.pi)))
print(f"  c_silu(grid normalspace(1_000_001)) = {c!r}, Gauss-Hermite integral {c_int!r}")
report("|c_grid - c_integral| / c (expected ~2e-5: JAX uses the grid value)", abs(c - c_int) / c, 1e-4)

if a.model:
    from race_v2_torch import RACEv2
    from race_v2_torch.calculator import RACEv2Calculator
    import ase
    from ase.build import bulk
    print(f"3. model checks with {a.model}")
    t0 = time.time()
    model = RACEv2.load(a.model)
    calc = RACEv2Calculator(model)
    z = model.atomic_numbers
    rng = np.random.RandomState(0)
    # small periodic cell: rocksalt-like MgO with 4 atoms, rattled and sheared
    at = bulk('MgO', 'rocksalt', a=4.2, cubic=False) * (1, 1, 2)
    at.set_cell(at.cell.array @ (np.eye(3) + 0.03 * rng.randn(3, 3)), scale_atoms=True)
    at.positions += 0.08 * rng.randn(len(at), 3)
    at.calc = calc
    e = at.get_potential_energy(); f = at.get_forces(); s = at.get_stress()
    print(f"  MgO x{len(at)}: E = {e:.10f} eV, |F|max = {np.abs(f).max():.4f}, stress = {np.round(s, 5)}")
    # rotation (positions and cell)
    Rm = eo3.rand_matrix().numpy()
    at2 = at.copy(); at2.set_cell(at.cell.array @ Rm.T, scale_atoms=True); at2.calc = calc
    report("rotation: |E(R) - E|", abs(at2.get_potential_energy() - e), 1e-10)
    report("rotation: |F(R) - R F|", float(np.abs(at2.get_forces() - f @ Rm.T).max()), 1e-10)
    sv = lambda v: np.array([[v[0], v[5], v[4]], [v[5], v[1], v[3]], [v[4], v[3], v[2]]])
    report("rotation: |S(R) - R S R^T|", float(np.abs(sv(at2.get_stress()) - Rm @ sv(s) @ Rm.T).max()), 1e-10)
    at3 = at.copy(); at3.positions += np.array([0.37, -1.1, 2.3]); at3.calc = calc
    report("translation: |dE|", abs(at3.get_potential_energy() - e), 1e-10)
    p = rng.permutation(len(at))
    at4 = at[p]; at4.calc = calc
    report("permutation: |dE|, |F_p - F[p]|", max(abs(at4.get_potential_energy() - e),
                                                  float(np.abs(at4.get_forces() - f[p]).max())), 1e-10)
    # finite differences
    h = 1e-4
    fd = np.zeros_like(f)
    for i in range(len(at)):
        for k in range(3):
            ap_, am_ = at.copy(), at.copy()
            ap_.positions[i, k] += h; am_.positions[i, k] -= h
            ap_.calc = calc; am_.calc = calc
            fd[i, k] = -(ap_.get_potential_energy() - am_.get_potential_energy()) / (2 * h)
    report("F vs central finite differences (h 1e-4) max abs", float(np.abs(fd - f).max()), 1e-6)
    sfd = np.zeros((3, 3))
    for u in range(3):
        for w in range(3):
            eps = np.zeros((3, 3)); eps[u, w] = h
            vals = []
            for sgn in (1, -1):
                b = at.copy(); b.set_cell(at.cell.array @ (np.eye(3) + sgn * eps).T, scale_atoms=True); b.calc = calc
                vals.append(b.get_potential_energy())
            sfd[u, w] = (vals[0] - vals[1]) / (2 * h) / at.get_volume()
    sfd = 0.5 * (sfd + sfd.T)
    report("stress vs finite strain (h 1e-4) max abs [eV/A^3]", float(np.abs(sv(s) - sfd).max()), 1e-7)
    sup = at * (2, 1, 1); sup.calc = calc
    report("supercell 2x1x1: |E_sup - 2 E| / atom", abs(sup.get_potential_energy() - 2 * e) / len(sup), 1e-10)
    # molecule (no pbc) with H, C, O if present
    zs = [zz for zz in (1, 6, 8) if zz in z]
    if len(zs) == 3:
        mol = ase.Atoms('CH2O', positions=[[0, 0, 0], [0.0, 0.94, 0.58], [0.0, -0.94, 0.58], [0, 0, -1.21]])
        mol.positions += 0.02 * rng.randn(4, 3)
        mol.calc = calc
        em, fm = mol.get_potential_energy(), mol.get_forces()
        m2 = mol.copy(); m2.positions = mol.positions @ Rm.T; m2.calc = calc
        report("molecule CH2O: rotation |dE|, |dF|", max(abs(m2.get_potential_energy() - em),
                                                          float(np.abs(m2.get_forces() - fm @ Rm.T).max())), 1e-10)
        report("molecule CH2O: sum of forces", float(np.abs(fm.sum(0)).max()), 1e-10)
    # edge chunking (checkpointed) gives the same numbers
    at5 = at.copy(); at5.calc = RACEv2Calculator(model, edge_chunk=7)
    report("edge_chunk=7 vs unchunked: |dE|, |dF|", max(abs(at5.get_potential_energy() - e),
                                                         float(np.abs(at5.get_forces() - f).max())), 1e-11)
    model.edge_chunk = 0
    print(f"  model checks took {time.time() - t0:.1f} s")

n_fail = sum(not r[3] for r in rows)
print(f"{len(rows)} checks, {n_fail} failed")
sys.exit(1 if n_fail else 0)
