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
    # rotation invariance. random_rotation returns float32 (orthogonal only to ~5e-8, which alone moves E by ~2e-9
    # here, for RACE as well), so snap it to the nearest exact float64 rotation (polar factor) before the 1e-9 check.
    u, _, vt = np.linalg.svd(random_rotation(3).astype(np.float64))
    R = u @ vt
    rot = run(g._replace(nodes={**g.nodes, 'positions': p @ R.T}), 0.0)
    assert abs(float(rot['energy'][0]) - float(r0['energy'][0])) < 1e-9
    # the energy depends on the total charge (charge-dependent model) and the charges sum to it
    r1 = run(g, 1.0)
    assert abs(float(r1['energy'][0]) - float(r0['energy'][0])) > 1e-3
    n = int(g.n_node[0])
    q1 = np.asarray(r1['charges'])
    assert abs(q1[:n].sum() - 1.0) < 1e-10 and np.all(q1[n:] == 0.0)
    assert abs(np.asarray(r0['charges'])[:n].sum()) < 1e-10


def test_race_output_keys_unchanged(graph):
    """energy_forces_stress adds 'charges' only for models that return them: plain RACE has no such key."""
    import jax
    import jax.numpy as jnp
    from conftest import make_model
    out = make_model()(jax.tree.map(jnp.asarray, pad_single(graph)))
    assert 'charges' not in out
    assert {'energy', 'forces', 'stress', 'node_features'} <= set(out)


def test_chi_init_pauling():
    """chi_init 'pauling': chi0 = 3 eV x (Pauling - mean); the default stays zero."""
    import jax.numpy as jnp
    from flax import nnx
    from bam_omat24.race_lr_nnx import RACE_LR, PAULING
    kw = dict(n_species=4, lmax=1, cutoff=5.0, hidden_irreps="8x0e + 8x1o", n_layers=1, radial_basis_size=4,
              radial_mlp_size=8, radial_mlp_layers=1, mlp_init_scale=4.0, shift=0.0, scale=1.0, avg_n_neighbors=10.0,
              atom_energies=ATOM_ENERGIES, l_train=True, periodic=False, x_mode="species", rngs=nnx.Rngs(0),
              species_radii=[0.31, 0.76, 0.71, 0.66], species_z=ATOMIC_NUMBERS)
    assert np.all(np.asarray(RACE_LR(**kw, long_range={}).chi0[...]) == 0.0)
    m = RACE_LR(**kw, long_range={"chi_init": "pauling"})
    v = np.array([PAULING[z] for z in ATOMIC_NUMBERS])
    assert np.allclose(np.asarray(m.chi0[...]), 3.0 * (v - v.mean()), atol=1e-6)
    with pytest.raises(ValueError):
        RACE_LR(**{**kw, 'species_z': [1, 6, 7, 2]}, long_range={"chi_init": "pauling"})   # He: no value


def test_charge_loss_term(x64, graph):
    """charge_weight * mean_i (q_i - q_ref_i)^2 over the real atoms is added to the loss; weight 0 = unchanged."""
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from bam_omat24.training_utils import compute_loss
    model, g = as_float64(_lr_model(), pad_single(graph))
    model.chi0.value = jnp.asarray([0.3, -0.2, 0.1, -0.4])
    n = int(g.n_node[0])
    rng = np.random.default_rng(0)
    ref = np.zeros(g.nodes['positions'].shape[0]); ref[:n] = rng.normal(0, 0.2, n); ref[n:] = 5.0  # padding ignored
    g = g._replace(nodes={**g.nodes, 'ref_charges': ref}, globals={**g.globals, 'charge': np.array([0.0, 0.0])})
    g = jax.tree.map(jnp.asarray, g)
    graphdef, params = nnx.split(model)
    l0, a0 = compute_loss(graphdef, params, g, 1.0, 10.0, 0.0, 10.0, periodic=False)
    l1, a1 = compute_loss(graphdef, params, g, 1.0, 10.0, 0.0, 10.0, periodic=False, charge_weight=0.5)
    q = np.asarray(model(g)['charges'])[:n]
    expect = np.mean((q - ref[:n]) ** 2)
    assert abs(float(a1['q_loss']) - expect) < 1e-10 and float(a0['q_loss']) == 0.0
    assert abs(float(l1) - float(l0) - 0.5 * expect) < 1e-10
    # gradient of the charge term reaches the electronegativity parameters
    gq = jax.grad(lambda p: compute_loss(graphdef, p, g, 0.0, 0.0, 0.0, 10.0, periodic=False, charge_weight=1.0)[0])(params)
    assert float(jnp.abs(gq['chi0'][...]).sum()) > 0.0


def test_ref_charges_in_graph(graph, frame):
    """charge_label_key puts the per-atom labels into nodes['ref_charges']; without it the node keys are unchanged."""
    from conftest import preprocess_graph, atomic_numbers_to_indices, CUTOFF
    a = frame.copy(); a.calc = frame.calc
    a.arrays['hirshfeld_charges'] = np.linspace(-0.5, 0.5, len(a))
    g0 = preprocess_graph(a, atomic_numbers_to_indices(ATOMIC_NUMBERS), ATOM_ENERGIES, CUTOFF)
    g1 = preprocess_graph(a, atomic_numbers_to_indices(ATOMIC_NUMBERS), ATOM_ENERGIES, CUTOFF,
                          charge_label_key='hirshfeld_charges')
    assert 'ref_charges' not in g0.nodes and set(g1.nodes) == set(g0.nodes) | {'ref_charges'}
    assert np.allclose(g1.nodes['ref_charges'], a.arrays['hirshfeld_charges'])
