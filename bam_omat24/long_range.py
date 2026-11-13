"""Long-range electrostatics for RACE_LR (2026-11-06; docs/RACE_LR_design.md): Gaussian-charge Coulomb matrices
(direct sum for molecules / clusters, Ewald sum for periodic cells) and charge equilibration (Qeq) with a total-charge
constraint per graph (or per fragment of a graph), and an optional short-range switch of the energy.  Pure jax.numpy, differentiable with respect to positions, cell and the Qeq parameters.

Units: Å, e, eV;  K_COULOMB = e^2 / (4 pi eps0) = 14.399645 eV Å.

Gaussian charges of width sigma_i interact as  K erf(r / (sqrt(2) s_ij)) / r,  s_ij = sqrt(sigma_i^2 + sigma_j^2);  the
self energy of one Gaussian is  K q^2 / (2 sqrt(pi) sigma).  Qeq energy of one graph
    E(q) = chi . q + 1/2 q^T A q,   A_ii = J_i + K / (sqrt(pi) sigma_i),   A_ij = gamma_ij (Coulomb, periodic or not)
minimised subject to sum_i q_i = Q (Lagrange multiplier per graph).
"""
from __future__ import annotations

import math

import jax
import numpy as np
import jax.numpy as jnp

K_COULOMB = 14.399645
SQRT_PI = math.sqrt(math.pi)


def _pair_sigma(sigma):
    return jnp.sqrt(sigma[:, None] ** 2 + sigma[None, :] ** 2)


def coulomb_direct(pos, sigma, same):
    """Open-boundary Gaussian-charge Coulomb matrix (zero diagonal).  pos (N,3), sigma (N,), same (N,N) bool = pairs
    of the same graph (and real nodes)."""
    d = pos[None, :, :] - pos[:, None, :]
    r2 = jnp.sum(d * d, axis=-1)
    off = same & ~jnp.eye(pos.shape[0], dtype=bool)
    r = jnp.sqrt(jnp.where(off, r2, 1.0))                         # safe sqrt (no NaN gradient on the diagonal)
    g = K_COULOMB * jax.scipy.special.erf(r / (math.sqrt(2.0) * _pair_sigma(sigma))) / r
    return jnp.where(off, g, 0.0)


def sr_switch(r, r_on: float, r_off: float):
    """Short-range switch for the electrostatic ENERGY (2026-11-07): 0 for r <= r_on, 1 for r >= r_off, C2 smoothstep
    x^3 (10 - 15 x + 6 x^2) in between.  Pairs closer than r_off are partly left to the learned short-range strata
    (no double counting there); the Qeq charges themselves are solved with the unswitched matrix."""
    x = jnp.clip((r - r_on) / (r_off - r_on), 0.0, 1.0)
    return x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)


def short_range_part(pos, sigma, same, r_on: float, r_off: float, cell_rows=None, n_img: int = 0):
    """The part (1 - switch(r)) K erf(r / (sqrt(2) s_ij)) / r of the Gaussian-charge Coulomb matrix (zero diagonal of
    the n = 0 image), summed over the lattice images |n_k| <= n_img for periodic graphs (cell_rows (N, 3, 3), the cell
    of each atom's graph; None = molecules).  Subtracting it from the Coulomb / Ewald matrix gives the switched energy
    matrix.  Only pairs with r < r_off contribute, so n_img = 1 suffices when r_off is below the shortest cell
    height."""
    if cell_rows is None:
        d = (pos[None, :, :] - pos[:, None, :])[:, :, None, :]                     # (N, N, 1, 3)
    else:
        rng = np.arange(-n_img, n_img + 1)
        shifts = jnp.asarray(np.stack(np.meshgrid(rng, rng, rng, indexing='ij'), -1).reshape(-1, 3), pos.dtype)
        T = jnp.einsum('sc,icd->isd', shifts, cell_rows)
        d = pos[None, :, None, :] - pos[:, None, None, :] + T[:, None, :, :]       # (N, N, S, 3)
    r2 = jnp.sum(d * d, axis=-1)
    use = (r2 > 1e-12) & (r2 < r_off * r_off) & same[:, :, None]
    r = jnp.sqrt(jnp.where(use, r2, 1.0))
    s = _pair_sigma(sigma)[:, :, None]
    g = K_COULOMB * jax.scipy.special.erf(r / (math.sqrt(2.0) * s)) / r * (1.0 - sr_switch(r, r_on, r_off))
    return jnp.sum(jnp.where(use, g, 0.0), axis=-1)


def ewald_matrix(pos, cell, sigma, alpha, n_real: int = 2, n_recip: int = 6):
    """Periodic Gaussian-charge Coulomb matrix of ONE cell (N atoms); see ewald_matrix_rows."""
    return ewald_matrix_rows(pos, jnp.broadcast_to(cell, (pos.shape[0], 3, 3)), sigma, alpha, n_real, n_recip)


def ewald_matrix_rows(pos, cell_rows, sigma, alpha, n_real: int = 2, n_recip: int = 6):
    """Periodic Gaussian-charge Coulomb matrix, Ewald split with parameter alpha (1/Å).  cell_rows (N, 3, 3): the
    cell of each atom's graph (a batch of periodic graphs); entries between atoms of different graphs are
    meaningless and must be masked by the caller.
        gamma_ij = sum_n K [erf(r/(√2 s_ij)) - erf(alpha r)] / r        (real space, images |n_k| <= n_real)
                 + (4 pi K / V) sum_{m != 0} exp(-k^2 / (4 alpha^2)) / k^2 cos(k . r_ij)     (|m_k| <= n_recip)
                 - pi K / (V alpha^2)                                   (neutralising background)
                 - delta_ij 2 K alpha / sqrt(pi)                       (self term of the erf(alpha r) part)
    The i = j image n = 0 is excluded (the Gaussian self energy is in the diagonal of the Qeq matrix).  The
    reciprocal part is evaluated as C C^T + S S^T with C_im = sqrt(w_m) cos(k_m . r_i) (memory N x M, not N^2 M).
    Independent of alpha when n_real / n_recip are large enough (tests/test_race_lr.py)."""
    N = pos.shape[0]
    rng_r = jnp.arange(-n_real, n_real + 1)
    shifts = jnp.stack(jnp.meshgrid(rng_r, rng_r, rng_r, indexing='ij'), -1).reshape(-1, 3).astype(pos.dtype)
    T = jnp.einsum('sc,icd->isd', shifts, cell_rows)                 # (N, S, 3) translations of row i's cell
    d = pos[None, :, None, :] - pos[:, None, None, :] + T[:, None, :, :]          # (N, N, S, 3)
    r2 = jnp.sum(d * d, axis=-1)
    zero = (r2 < 1e-12)
    r = jnp.sqrt(jnp.where(zero, 1.0, r2))
    s = _pair_sigma(sigma)[:, :, None]
    real = jnp.where(zero, 0.0, (jax.scipy.special.erf(r / (math.sqrt(2.0) * s)) - jax.scipy.special.erf(alpha * r)) / r)
    gamma = K_COULOMB * jnp.sum(real, axis=-1)
    rng_m = np.arange(-n_recip, n_recip + 1)                        # numpy: a static constant also under jit
    m = np.stack(np.meshgrid(rng_m, rng_m, rng_m, indexing='ij'), -1).reshape(-1, 3)
    m = jnp.asarray(m[np.any(m != 0, axis=-1)], dtype=pos.dtype)    # (M, 3), m = 0 excluded
    vol = jnp.abs(jnp.linalg.det(cell_rows))                        # (N,)
    recip = 2.0 * math.pi * jnp.swapaxes(jnp.linalg.inv(cell_rows), -1, -2)       # (N, 3, 3) rows b_k
    k = jnp.einsum('mc,icd->imd', m, recip)                         # (N, M, 3)
    k2 = jnp.sum(k * k, axis=-1)
    # sqrt(4 pi K / V * exp(-k^2 / 4 alpha^2) / k^2) written without a sqrt of the exponential: in float32 the
    # exponential underflows to 0 at large k, and d sqrt(x)/dx at x = 0 is infinite (NaN gradients, e.g. stress)
    w = (jnp.sqrt(4.0 * math.pi * K_COULOMB / vol)[:, None] * jnp.exp(-k2 / (8.0 * alpha ** 2))
         * jax.lax.rsqrt(k2))                                       # (N, M)
    ph = jnp.einsum('imd,id->im', k, pos)
    C, S = w * jnp.cos(ph), w * jnp.sin(ph)
    gamma = gamma + C @ C.T + S @ S.T
    gamma = gamma - math.pi * K_COULOMB / (vol[:, None] * alpha ** 2)
    gamma = gamma - jnp.eye(N, dtype=pos.dtype) * (2.0 * K_COULOMB * alpha / SQRT_PI)
    return gamma


def qeq_solve(chi, hard, sigma, gamma, node_graph, node_mask, total_charge, n_graphs, gamma_energy=None):
    """Charge equilibration for a batch of graphs.  chi, hard, sigma (N,); gamma (N, N) Coulomb matrix (pairs of
    different graphs / padding must be 0); node_graph (N,) graph index; node_mask (N,) real nodes;
    total_charge (G,).  Returns (q (N,), per-atom energy e_i = chi_i q_i + 1/2 q_i (A q)_i).
    node_graph / total_charge / n_graphs are the constraint GROUPS: graphs, or (graph, fragment) pairs for per-fragment
    total charges (RACE_LR fragments); gamma still couples every pair of the same graph.
    gamma_energy: the Coulomb matrix used in the energy (e.g. with a short-range switch); default gamma.  The charges
    always come from gamma."""
    N = chi.shape[0]
    diag = jnp.where(node_mask, hard + K_COULOMB / (SQRT_PI * sigma), 1.0)
    A = jnp.where(node_mask[:, None] & node_mask[None, :], gamma, 0.0) + jnp.diag(diag)
    C = ((node_graph[None, :] == jnp.arange(n_graphs)[:, None]) & node_mask[None, :]).astype(chi.dtype)   # (G, N)
    empty = (jnp.sum(C, axis=1) == 0).astype(chi.dtype)            # graphs without real nodes: lambda = 0
    M = jnp.block([[A, C.T], [C, jnp.diag(empty)]])
    rhs = jnp.concatenate([jnp.where(node_mask, -chi, 0.0), jnp.where(empty > 0, 0.0, total_charge)])
    sol = jnp.linalg.solve(M, rhs)
    q = jnp.where(node_mask, sol[:N], 0.0)
    if gamma_energy is not None:
        A = jnp.where(node_mask[:, None] & node_mask[None, :], gamma_energy, 0.0) + jnp.diag(diag)
    Aq = (A - jnp.diag(jnp.where(node_mask, 0.0, 1.0))) @ q
    e = jnp.where(node_mask, chi * q + 0.5 * q * Aq, 0.0)
    return q, e


def ct_solve(chi, hard, sigma, gamma, cond, node_mask, q0, gamma_energy=None):
    """Charge equilibration with distance-limited charge transfer (split-charge form, 2026-11-13), ONE structure (or one
    block of real + padding atoms).  Charge flows only along pairs with a conductance cond_ij >= 0 (symmetric, zero on
    the diagonal and for padding), starting from reference charges q0 (an integer charge per molecule):

        q = q0 + B p,   p = argmin over the pair charges of  E(q) + 1/2 sum_pairs p_ij^2 / cond_ij
        =>  (I + L A) q = q0 - L chi,   L = diag(cond 1) - cond   (graph Laplacian),
            A = gamma + diag(J + K / (sqrt(pi) sigma))           (as qeq_solve)

    Every connected component of cond keeps the total of q0 exactly (1^T L = 0); atoms without conductance keep q0;
    for cond -> infinity inside a component the charges tend to the Qeq charges of that component.  Returns
    (q (N,), per-atom energy e_i = chi_i q_i + 1/2 q_i (A_e q)_i with A_e from gamma_energy (default gamma))."""
    N = chi.shape[0]
    m = node_mask
    mm = m[:, None] & m[None, :]
    diag = jnp.where(m, hard + K_COULOMB / (SQRT_PI * sigma), 1.0)
    A = jnp.where(mm, gamma, 0.0) + jnp.diag(diag)
    C = jnp.where(mm, cond, 0.0) * (1.0 - jnp.eye(N, dtype=chi.dtype))
    L = jnp.diag(jnp.sum(C, axis=1)) - C
    chi_m = jnp.where(m, chi, 0.0)
    rhs = jnp.where(m, q0, 0.0) - L @ chi_m
    q = jnp.linalg.solve(jnp.eye(N, dtype=chi.dtype) + L @ A, rhs)
    q = jnp.where(m, q, 0.0)
    Ae = jnp.where(mm, gamma if gamma_energy is None else gamma_energy, 0.0) + jnp.diag(jnp.where(m, diag, 0.0))
    e = jnp.where(m, chi_m * q + 0.5 * q * (Ae @ q), 0.0)
    return q, e
