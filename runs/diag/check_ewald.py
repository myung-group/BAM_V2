"""Ewald settings check for a periodic dataset (RACE_LR long_range ewald_alpha / ewald_n_real / ewald_n_recip).

    python -I runs/diag/check_ewald.py <periodic.xyz> [--n 20] [--seed 0] [--tol-mev 1.0]
        [--alpha 0.3 --n-real 2 --n-recip 6]

For n structures of the file: random neutral charges (normal, std 0.5 e, mean removed) with Gaussian widths = covalent
radii, energy 1/2 q^T G q (Gaussian-charge Ewald of bam_omat24.long_range) with the given settings vs a reference
(the same alpha with n_real + 3 and 2 n_recip, checked against a second alpha).  Then a grid over alpha / n_real /
n_recip: the cheapest setting whose worst error is below --tol-mev, with cost = N^2 (2 n_real + 1)^3 +
N (2 n_recip + 1)^3 for the largest structure.  Runs on CPU (JAX_PLATFORMS=cpu is fine).
"""
import argparse
import itertools

import numpy as np
import ase.io
from ase.data import covalent_radii


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('xyz')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tol-mev', type=float, default=1.0)
    ap.add_argument('--alpha', type=float, default=0.3)
    ap.add_argument('--n-real', type=int, default=2)
    ap.add_argument('--n-recip', type=int, default=6)
    args = ap.parse_args()
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from bam_omat24.long_range import ewald_matrix
    frames = [a for a in ase.io.read(args.xyz, ':') if a.pbc.all()]
    rng = np.random.RandomState(args.seed)
    pick = [frames[i] for i in rng.choice(len(frames), min(args.n, len(frames)), replace=False)]
    sys_ = []
    for a in pick:
        q = rng.normal(0, 0.5, len(a)); q -= q.mean()
        sys_.append((jnp.asarray(a.positions), jnp.asarray(a.cell.array), jnp.asarray(covalent_radii[a.numbers]), q))
    lengths = np.array([np.linalg.norm(a.cell.array, axis=1) for a in pick])
    print(f'{len(pick)} of {len(frames)} periodic structures; atoms {min(len(a) for a in pick)}-{max(len(a) for a in pick)}; '
          f'cell lengths {lengths.min():.2f}-{lengths.max():.2f} A')

    def energies(alpha, nr, nk):
        return np.array([0.5 * float(q @ np.asarray(ewald_matrix(p, c, s, alpha, nr, nk)) @ q) for p, c, s, q in sys_])
    ref = energies(args.alpha, args.n_real + 3, 2 * args.n_recip)
    ref2 = energies(1.5 * args.alpha, args.n_real + 3, 2 * args.n_recip)
    print(f'reference check (alpha x1.5): max |dE| {np.abs(ref2 - ref).max() * 1e3:.4f} meV '
          '(must be far below the tolerance, else enlarge the reference)')
    e = energies(args.alpha, args.n_real, args.n_recip)
    err = np.abs(e - ref) * 1e3
    print(f'given alpha {args.alpha} n_real {args.n_real} n_recip {args.n_recip}: max |dE| {err.max():.4f} meV, '
          f'median {np.median(err):.4f} meV')
    nmax = max(len(a) for a in pick)
    best = []
    for alpha, nr, nk in itertools.product([0.2, 0.3, 0.4, 0.5, 0.6, 0.8], [1, 2, 3], [4, 6, 8, 10, 12]):
        cost = nmax ** 2 * (2 * nr + 1) ** 3 + nmax * (2 * nk + 1) ** 3
        best.append((cost, alpha, nr, nk))
    best.sort()
    for cost, alpha, nr, nk in best:
        m = (np.abs(energies(alpha, nr, nk) - ref) * 1e3).max()
        if m < args.tol_mev:
            print(f'cheapest within {args.tol_mev} meV: alpha {alpha} n_real {nr} n_recip {nk} (max {m:.4f} meV, '
                  f'cost {cost:.3g}) -> "long_range": {{"ewald_alpha": {alpha}, "ewald_n_real": {nr}, "ewald_n_recip": {nk}}}')
            break
    else:
        print('no setting of the grid is within the tolerance')


if __name__ == '__main__':
    main()
