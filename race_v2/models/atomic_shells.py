"""Spherical, spin-unpolarised LDA (Slater exchange + Perdew–Zunger 81 correlation) atom
solver, used once to tabulate the radial extent and energy of the valence shells of
every element (Z = 1 .. 94):

    bam_omat24/atomic_shells.json  = { "26": { "config": {"1s": 2, ...},
                                               "shells": { "s": {"n": 4, "occ": 2, "r_mean": 3.2, "r_peak": 2.6, "eps": -0.19},
                                                           "p": {...}, "d": {...}, "f": null },
                                               "n_valence": 8 }, ... }

For each l the *outermost occupied* shell of that angular momentum is reported
(for Fe: 4s, 3p, 3d; for Gd: 6s, 5p, 5d, 4f), with <r> and the radius of the maximum of
P(r)^2 in Bohr and the Kohn–Sham eigenvalue in Hartree.  The eigenvalue tells the
consumer whether the shell is core-like (deep) or valence.  Non-relativistic, so the 6s/7s
radii of heavy elements are a few % too large and the 4f/5f a few % too small compared to
Dirac–Fock, which is irrelevant for a learned descriptor.

Method: radial Kohn–Sham equation on a shifted exponential grid r = r0 (e^x - 1),
P(r) = sqrt(r') Q(x),   -1/2 Q'' + [ r'^2 (V + l(l+1)/(2r^2) - eps) + 1/8 ] Q = 0
(the 1/8 is the Schwarzian term, exact for this grid where r' = r'' = r'''),
discretised with 3-point finite differences; after the similarity transform with
S^{-1/2}, S = diag(r'^2), this is a symmetric tridiagonal eigenproblem solved with
``scipy.linalg.eigh_tridiagonal`` (lowest states only).  A pure logarithmic grid with
r_min = 1e-6 loses the valence eigenvalues to round-off (matrix entries 1e16), hence the
shift.  Density mixing 0.3, converged to 1e-7 Ha in the eigenvalues.  Checks against the
NIST LDA atomic reference data (spin-unpolarised, PZ81): H 1s -0.233471, He 1s -0.570425,
Ar 3p -0.3822 Ha, and the hydrogenic U 1s -4232 Ha.

Pure numpy / scipy (no JAX): run it as a script (``python -I bam_omat24/atomic_shells.py``),
not through the package (``bam_omat24/__init__`` imports JAX, which must not run on the
login node).

Usage:
    python -I bam_omat24/atomic_shells.py [--zmax 94] [--out bam_omat24/atomic_shells.json]
"""
import argparse
import json
import math
import time

import numpy as np
from scipy.linalg import eigh_tridiagonal

L_NAMES = "spdf"
# Madelung (n + l) filling order
MADELUNG = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (3, 2), (4, 1), (5, 0), (4, 2), (5, 1),
            (6, 0), (4, 3), (5, 2), (6, 1), (7, 0), (5, 3), (6, 2), (7, 1)]
# experimental ground-state exceptions to the Madelung rule: Z -> {shell: occupation}
EXCEPTIONS = {
    24: {"3d": 5, "4s": 1}, 29: {"3d": 10, "4s": 1},
    41: {"4d": 4, "5s": 1}, 42: {"4d": 5, "5s": 1}, 44: {"4d": 7, "5s": 1}, 45: {"4d": 8, "5s": 1},
    46: {"4d": 10, "5s": 0}, 47: {"4d": 10, "5s": 1},
    57: {"4f": 0, "5d": 1}, 58: {"4f": 1, "5d": 1}, 64: {"4f": 7, "5d": 1},
    78: {"5d": 9, "6s": 1}, 79: {"5d": 10, "6s": 1},
    89: {"5f": 0, "6d": 1}, 90: {"5f": 0, "6d": 2}, 91: {"5f": 2, "6d": 1}, 92: {"5f": 3, "6d": 1},
    93: {"5f": 4, "6d": 1},
}
NOBLE = [0, 2, 10, 18, 36, 54, 86]


def shell_name(n, l):
    return f"{n}{L_NAMES[l]}"


def configuration(Z):
    """Ground-state occupations {(n, l): f} (Madelung + exceptions)."""
    occ = {}
    left = Z
    for n, l in MADELUNG:
        if left <= 0:
            break
        cap = 2 * (2 * l + 1)
        f = min(cap, left)
        occ[(n, l)] = f
        left -= f
    for name, f in EXCEPTIONS.get(Z, {}).items():
        n, l = int(name[0]), L_NAMES.index(name[1])
        if f == 0:
            occ.pop((n, l), None)
        else:
            occ[(n, l)] = f
    assert sum(occ.values()) == Z, (Z, occ)
    return occ


def xc_pz81(rho):
    """Slater exchange + Perdew–Zunger 81 correlation (spin-unpolarised). Returns V_xc."""
    rho = np.maximum(rho, 1e-30)
    vx = -(3.0 / math.pi) ** (1.0 / 3.0) * rho ** (1.0 / 3.0)
    rs = (3.0 / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
    vc = np.empty_like(rho)
    hi = rs >= 1.0
    g, b1, b2 = -0.1423, 1.0529, 0.3334
    sq = np.sqrt(rs[hi])
    ec = g / (1.0 + b1 * sq + b2 * rs[hi])
    vc[hi] = ec * (1.0 + 7.0 / 6.0 * b1 * sq + 4.0 / 3.0 * b2 * rs[hi]) / (1.0 + b1 * sq + b2 * rs[hi])
    lo = ~hi
    A, B, C, D = 0.0311, -0.048, 0.0020, -0.0116
    lr = np.log(rs[lo])
    vc[lo] = A * lr + (B - A / 3.0) + 2.0 / 3.0 * C * rs[lo] * lr + (2.0 * D - C) / 3.0 * rs[lo]
    return vx + vc


class Atom:
    def __init__(self, Z, n_grid=3000, r0=0.05, r_max=60.0):
        """Shifted exponential grid r = r0 (e^x - 1), x in (0, X): spacing r0*h near the nucleus
        (~100 points inside the 1s radius of U) and exponential further out.  The Jacobian
        r' = r + r0 >= r0 keeps the similarity-transformed matrix well conditioned."""
        self.Z = Z
        X = math.log(r_max / r0 + 1.0)
        x = np.linspace(0.0, X, n_grid + 2)[1:-1]                      # interior points: P(0) = P(r_max) = 0
        self.h = x[1] - x[0]
        self.r = r0 * (np.exp(x) - 1.0)
        self.rp = self.r + r0                                          # dr/dx
        self.occ = configuration(Z)
        self.lmax = max(l for _, l in self.occ)
        self.n_per_l = {l: sum(1 for (n, ll) in self.occ if ll == l) for l in range(self.lmax + 1)}

    def solve_l(self, V, l):
        """Lowest n_per_l[l] states of angular momentum l for potential V(r): (eps, P) with
        P normalised as int P^2 dr = 1."""
        r, rp, h = self.r, self.rp, self.h
        k = self.n_per_l[l]
        diag = 1.0 / h ** 2 + 0.125 + rp ** 2 * (V + l * (l + 1) / (2.0 * r ** 2))
        off = np.full(len(r) - 1, -0.5 / h ** 2)
        s = 1.0 / rp                                                   # S^{-1/2}, S = diag(r'^2)
        d_t = diag * s * s
        e_t = off * s[:-1] * s[1:]
        eps, y = eigh_tridiagonal(d_t, e_t, select="i", select_range=(0, k - 1))
        Q = y * s[:, None]                                             # Q = S^{-1/2} y
        P = np.sqrt(rp)[:, None] * Q
        norm = np.sqrt(np.sum(P ** 2 * rp[:, None], axis=0) * h)      # int P^2 dr = sum P^2 r' h
        return eps, P / norm

    def hartree(self, n_r):
        """V_H from the radial density n(r) = 4 pi r^2 rho = sum f P^2 (int n dr = N_e)."""
        r, rp, h = self.r, self.rp, self.h
        w = n_r * rp * h                                               # n dr
        q_in = np.cumsum(w) - 0.5 * w                                  # int_0^r n dr'
        out = np.cumsum((w / r)[::-1])[::-1] - 0.5 * w / r             # int_r^inf n / r' dr'
        return q_in / r + out

    def scf(self, V0=None, mix=0.3, tol=1e-7, max_iter=300):
        r = self.r
        Z = self.Z
        if V0 is None:
            # screened-nucleus start: -Z/r at short range, -1/r far away
            V = -Z / r + (Z - 1) / r * (1.0 - np.exp(-1.6 * Z ** (1 / 3) * r))
        else:
            V = V0
        eps_old = None
        n_old = None
        for it in range(max_iter):
            n_r = np.zeros_like(r)
            eps_all = {}
            P_all = {}
            for l in range(self.lmax + 1):
                eps, P = self.solve_l(V, l)
                for i_n in range(self.n_per_l[l]):
                    n = i_n + l + 1
                    f = self.occ.get((n, l), 0.0)
                    eps_all[(n, l)] = eps[i_n]
                    P_all[(n, l)] = P[:, i_n]
                    n_r += f * P[:, i_n] ** 2
            if n_old is not None:
                n_r = (1 - mix) * n_old + mix * n_r
            n_old = n_r
            rho = n_r / (4.0 * math.pi * r ** 2)
            V = -Z / r + self.hartree(n_r) + xc_pz81(rho)
            if eps_old is not None:
                d = max(abs(eps_all[k] - eps_old[k]) for k in eps_all)
                if d < tol and it > 5:
                    break
            eps_old = eps_all
        self.eps, self.P, self.V, self.n_iter = eps_all, P_all, V, it + 1
        return self

    def shell_report(self):
        r, rp, h = self.r, self.rp, self.h
        out = {}
        for l in range(4):
            shells = [(n, ll) for (n, ll) in self.occ if ll == l]
            if not shells:
                out[L_NAMES[l]] = None
                continue
            n = max(n for n, _ in shells)
            P = self.P[(n, l)]
            r_mean = float(np.sum(P ** 2 * r * rp) * h)
            r_peak = float(r[np.argmax(P ** 2)])
            out[L_NAMES[l]] = dict(n=n, occ=int(self.occ[(n, l)]), r_mean=r_mean, r_peak=r_peak,
                                   eps=float(self.eps[(n, l)]))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zmax", type=int, default=94)
    ap.add_argument("--out", default="bam_omat24/atomic_shells.json")
    ap.add_argument("--n-grid", type=int, default=3000)
    args = ap.parse_args()
    table = {}
    V_prev = None
    t0 = time.time()
    for Z in range(1, args.zmax + 1):
        atom = Atom(Z, n_grid=args.n_grid)
        atom.scf(V0=V_prev)
        V_prev = atom.V
        rep = atom.shell_report()
        core = max(c for c in NOBLE if c < Z)
        table[str(Z)] = dict(config={shell_name(n, l): f for (n, l), f in sorted(atom.occ.items())},
                             shells=rep, n_valence=Z - core, n_iter=atom.n_iter)
        hi = " ".join(f"{v['n']}{k}:<r>={v['r_mean']:.3f},eps={v['eps']:.4f}"
                      for k, v in rep.items() if v is not None)
        print(f"Z={Z:3d} iters={atom.n_iter:3d}  {hi}", flush=True)
    json.dump(table, open(args.out, "w"), indent=1)
    print(f"written {args.out} ({len(table)} elements, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
