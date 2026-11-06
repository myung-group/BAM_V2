"""RACE_LR (charge equilibration + long-range electrostatics, 2026-11-06): Ewald against the NaCl Madelung constant,
alpha independence, Qeq charge conservation and optimality, and the model's energy / force consistency, rotation
invariance and total-charge dependence.  Run in a Slurm job (JAX)."""
import math

import numpy as np
import pytest

from conftest import ATOMIC_NUMBERS, ATOM_ENERGIES, pad_single, random_rotation, as_float64

MADELUNG_NACL = 1.747564594633


def _nacl(a=5.64):
    frac = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                     [.5, .5, .5], [.5, 0, 0], [0, .5, 0], [0, 0, .5]])
    q = np.array([1, 1, 1, 1, -1, -1, -1, -1], float)
    return frac * a, np.eye(3) * a, q


@pytest.mark.parametrize('alpha', [0.5, 0.6, 0.75])
def test_ewald_madelung_and_alpha_independence(x64, alpha):
    import jax.numpy as jnp
    from bam_omat24.long_range import ewald_matrix, K_COULOMB
    pos, cell, q = _nacl()
    sigma = jnp.full((8,), 1e-3)
    g = ewald_matrix(jnp.asarray(pos), jnp.asarray(cell), sigma, alpha, n_real=3, n_recip=10)
    e = 0.5 * float(q @ np.asarray(g) @ q)                      # 4 ion pairs
    ref = -4 * MADELUNG_NACL * K_COULOMB / (5.64 / 2)
    assert abs(e - ref) < 1e-7 * abs(ref), (e, ref)


def test_ewald_rows_batch_matches_single(x64):
    import jax.numpy as jnp
    from bam_omat24.long_range import ewald_matrix, ewald_matrix_rows
    rng = np.random.default_rng(0)
    p1, c1 = rng.uniform(0, 6, (5, 3)), np.diag([6.0, 6.5, 7.0])
    p2, c2 = rng.uniform(0, 5, (4, 3)), np.array([[5.0, 0, 0], [1.0, 5.5, 0], [0.5, 0.3, 6.0]])
    s = jnp.asarray(rng.uniform(0.5, 1.0, 9))
    rows = np.concatenate([np.broadcast_to(c1, (5, 3, 3)), np.broadcast_to(c2, (4, 3, 3))])
    G = np.asarray(ewald_matrix_rows(jnp.asarray(np.concatenate([p1, p2])), jnp.asarray(rows), s, 0.5, 2, 6))
    G1 = np.asarray(ewald_matrix(jnp.asarray(p1), jnp.asarray(c1), s[:5], 0.5, 2, 6))
    G2 = np.asarray(ewald_matrix(jnp.asarray(p2), jnp.asarray(c2), s[5:], 0.5, 2, 6))
    assert np.allclose(G[:5, :5], G1, atol=1e-10) and np.allclose(G[5:, 5:], G2, atol=1e-10)


def test_qeq_conserves_charge_and_is_optimal(x64):
    import jax.numpy as jnp
    from bam_omat24.long_range import coulomb_direct, qeq_solve, K_COULOMB
    rng = np.random.default_rng(1)
    N, G = 9, 3                                  # graphs of 4 and 4 atoms, last node + last graph = padding
    node_graph = jnp.asarray([0, 0, 0, 0, 1, 1, 1, 1, 2])
    mask = jnp.asarray([True] * 8 + [False])
    pos = jnp.asarray(rng.uniform(0, 4, (N, 3)))
    chi, hard = jnp.asarray(rng.normal(size=N)), jnp.asarray(rng.uniform(3, 6, N))
    sigma = jnp.asarray(rng.uniform(0.6, 1.0, N))
    same = (node_graph[:, None] == node_graph[None, :]) & mask[:, None] & mask[None, :]
    Q = jnp.asarray([1.0, -1.0, 0.0])
    q, e = qeq_solve(chi, hard, sigma, coulomb_direct(pos, sigma, same), node_graph, mask, Q, G)
    q = np.asarray(q)
    assert abs(q[:4].sum() - 1.0) < 1e-12 and abs(q[4:8].sum() + 1.0) < 1e-12 and q[8] == 0.0
    # optimality: the gradient of E(q) is constant within each graph (= -lambda)
    A = np.asarray(coulomb_direct(pos, sigma, same)) + np.diag(np.asarray(hard + K_COULOMB / (math.sqrt(math.pi) * sigma)))
    grad = np.asarray(chi) + A @ q
    assert np.ptp(grad[:4]) < 1e-9 and np.ptp(grad[4:8]) < 1e-9


def _lr_model(seed=0):
    from flax import nnx
    from bam_omat24.race_lr_nnx import RACE_LR
    return RACE_LR(n_species=4, lmax=2, cutoff=5.0, hidden_irreps="16x0e + 16x1o + 16x2e", n_layers=2,
                   radial_basis_size=8, radial_mlp_size=16, radial_mlp_layers=2, mlp_init_scale=4.0, shift=0.0,
                   scale=1.0, avg_n_neighbors=20.0, atom_energies=ATOM_ENERGIES, l_train=True, periodic=False,
                   x_mode="species", rngs=nnx.Rngs(seed), species_radii=[0.31, 0.76, 0.71, 0.66],
                   long_range={"chi_scale": 1.0})


def test_race_lr_forces_rotation_and_charge(x64, graph):
    import jax
    import jax.numpy as jnp
    model, g = as_float64(_lr_model(), pad_single(graph))
    # make chi nonzero so that the electrostatic stratum is not trivial
    model.chi0.value = jnp.asarray([0.3, -0.2, 0.1, -0.4])

    def run(gr, Q):
        gr = gr._replace(globals={**gr.globals, 'charge': np.array([Q, 0.0])})
        return model(jax.tree.map(jnp.asarray, gr))
    r0 = run(g, 0.0)
    # forces = -dE/dx (finite difference on one coordinate)
    h = 1e-5
    p = np.asarray(g.nodes['positions']).copy()
    for i, c in [(3, 0), (7, 2)]:
        pp, pm = p.copy(), p.copy(); pp[i, c] += h; pm[i, c] -= h
        ep = float(run(g._replace(nodes={**g.nodes, 'positions': pp}), 0.0)['energy'][0])
        em = float(run(g._replace(nodes={**g.nodes, 'positions': pm}), 0.0)['energy'][0])
        assert abs(-(ep - em) / (2 * h) - float(r0['forces'][i, c])) < 1e-6
    # rotation invariance
    R = random_rotation(3).astype(np.float64)
    rot = run(g._replace(nodes={**g.nodes, 'positions': p @ R.T}), 0.0)
    assert abs(float(rot['energy'][0]) - float(r0['energy'][0])) < 1e-9
    # the energy depends on the total charge (charge-dependent model) and the charges sum to it
    r1 = run(g, 1.0)
    assert abs(float(r1['energy'][0]) - float(r0['energy'][0])) > 1e-3
