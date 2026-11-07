"""RACE_LR (charge equilibration + long-range electrostatics, 2026-11-06): Ewald against the NaCl Madelung constant,
alpha independence, Qeq charge conservation and optimality, and the model's energy / force consistency, rotation
invariance and total-charge dependence.  Run in a Slurm job (JAX)."""
import math

import numpy as np
import pytest

from conftest import ATOMIC_NUMBERS, ATOM_ENERGIES, pad_single, random_rotation, as_float64
# imported at module level: train_3bpa forces float32 when it is first imported, which would undo the x64 fixture
from bam_omat24.train_3bpa import element_shell_radii, COVALENT_RADII  # noqa: E402
from test_oam_stress import ZS, _cell, _graph, _f64  # noqa: E402

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


def _lr_model(seed=0, lr=None):
    from flax import nnx
    from bam_omat24.race_lr_nnx import RACE_LR
    return RACE_LR(n_species=4, lmax=2, cutoff=5.0, hidden_irreps="16x0e + 16x1o + 16x2e", n_layers=2,
                   radial_basis_size=8, radial_mlp_size=16, radial_mlp_layers=2, mlp_init_scale=4.0, shift=0.0,
                   scale=1.0, avg_n_neighbors=20.0, atom_energies=ATOM_ENERGIES, l_train=True, periodic=False,
                   x_mode="species", rngs=nnx.Rngs(seed), species_radii=[0.31, 0.76, 0.71, 0.66],
                   species_z=ATOMIC_NUMBERS, long_range={"chi_scale": 1.0, **(lr or {})})


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
    """chi_init 'pauling' (the default since 2026-11-07): chi0 = 3 eV x (Pauling - mean); 'zero' when asked."""
    import jax.numpy as jnp
    from flax import nnx
    from bam_omat24.race_lr_nnx import RACE_LR, PAULING
    kw = dict(n_species=4, lmax=1, cutoff=5.0, hidden_irreps="8x0e + 8x1o", n_layers=1, radial_basis_size=4,
              radial_mlp_size=8, radial_mlp_layers=1, mlp_init_scale=4.0, shift=0.0, scale=1.0, avg_n_neighbors=10.0,
              atom_energies=ATOM_ENERGIES, l_train=True, periodic=False, x_mode="species", rngs=nnx.Rngs(0),
              species_radii=[0.31, 0.76, 0.71, 0.66], species_z=ATOMIC_NUMBERS)
    assert np.all(np.asarray(RACE_LR(**kw, long_range={"chi_init": "zero"}).chi0[...]) == 0.0)
    m = RACE_LR(**kw, long_range={})
    v = np.array([PAULING[z] for z in ATOMIC_NUMBERS])
    assert np.allclose(np.asarray(m.chi0[...]), 3.0 * (v - v.mean()), atol=1e-6)
    with pytest.raises(ValueError):                                                   # He: no value
        RACE_LR(**{**kw, 'species_z': [1, 6, 7, 2]}, long_range={"chi_init": "pauling", "chi_init_missing": "error"})
    m2 = RACE_LR(**{**kw, 'species_z': [1, 6, 7, 2]}, long_range={})                     # default: mean for He
    v2 = np.array([2.20, 2.55, 3.04, np.mean([2.20, 2.55, 3.04])])
    assert np.allclose(np.asarray(m2.chi0[...]), 3.0 * (v2 - v2.mean()), atol=1e-6)
    with pytest.raises(ValueError):                                                   # Pauling needs species_z
        RACE_LR(**{k: v for k, v in kw.items() if k != 'species_z'}, long_range={})


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


def _cs_model(qeq, seed=0):
    from flax import nnx
    from bam_omat24.race_lr_nnx import RACE_LR
    return RACE_LR(n_species=4, lmax=2, cutoff=5.0, hidden_irreps="16x0e + 16x1o + 16x2e", n_layers=2,
                   radial_basis_size=8, radial_mlp_size=16, radial_mlp_layers=2, mlp_init_scale=4.0, shift=0.0,
                   scale=1.0, avg_n_neighbors=20.0, atom_energies=ATOM_ENERGIES, l_train=True, periodic=False,
                   x_mode="species", rngs=nnx.Rngs(seed), species_radii=[0.31, 0.76, 0.71, 0.66],
                   species_z=ATOMIC_NUMBERS, long_range={"qeq": qeq, "charge_spin_embedding": True})


def test_charge_spin_embedding(x64, graph):
    """Embedding only (qeq false): equals RACE at init; once trained it depends on Q and 2S+1 and stays
    rotation invariant with forces = -dE/dx."""
    import jax
    import jax.numpy as jnp
    from conftest import make_model
    g = pad_single(graph)
    m_cs, g64 = as_float64(_cs_model(qeq=False), g)
    m_race, _ = as_float64(make_model(seed=0), g)

    def run(model, gr, Q=0.0, S=1.0):
        gr = gr._replace(globals={**gr.globals, 'charge': np.array([Q, 0.0]), 'spin': np.array([S, 1.0])})
        return model(jax.tree.map(jnp.asarray, gr))
    e_race = float(m_race(jax.tree.map(jnp.asarray, g64))['energy'][0])
    assert abs(float(run(m_cs, g64, 1.0, 3.0)['energy'][0]) - e_race) < 1e-9          # zero-initialised output
    assert 'charges' not in run(m_cs, g64)
    rng = np.random.default_rng(0)
    m_cs.cs_mlp2.kernel.value = jnp.asarray(rng.normal(0, 0.3, m_cs.cs_mlp2.kernel.value.shape))
    e00, e10, e03 = (float(run(m_cs, g64, Q, S)['energy'][0]) for Q, S in [(0, 1), (1, 1), (0, 3)])
    assert abs(e10 - e00) > 1e-4 and abs(e03 - e00) > 1e-4
    r0 = run(m_cs, g64, 1.0, 2.0)
    u, _, vt = np.linalg.svd(random_rotation(3).astype(np.float64)); R = u @ vt
    p = np.asarray(g64.nodes['positions'])
    rot = run(m_cs, g64._replace(nodes={**g64.nodes, 'positions': p @ R.T}), 1.0, 2.0)
    assert abs(float(rot['energy'][0]) - float(r0['energy'][0])) < 1e-9
    h = 1e-5
    pp, pm = p.copy(), p.copy(); pp[5, 1] += h; pm[5, 1] -= h
    ep = float(run(m_cs, g64._replace(nodes={**g64.nodes, 'positions': pp}), 1.0, 2.0)['energy'][0])
    em = float(run(m_cs, g64._replace(nodes={**g64.nodes, 'positions': pm}), 1.0, 2.0)['energy'][0])
    assert abs(-(ep - em) / (2 * h) - float(r0['forces'][5, 1])) < 1e-6


def test_charge_spin_embedding_with_qeq_and_options(x64, graph, frame):
    import jax
    import jax.numpy as jnp
    from bam_omat24.race_lr_nnx import RACE_LR
    from conftest import preprocess_graph, atomic_numbers_to_indices, CUTOFF
    m, g = as_float64(_cs_model(qeq=True), pad_single(graph))
    g = g._replace(globals={**g.globals, 'charge': np.array([1.0, 0.0]), 'spin': np.array([2.0, 1.0])})
    out = m(jax.tree.map(jnp.asarray, g))
    assert abs(float(np.asarray(out['charges'])[:int(g.n_node[0])].sum()) - 1.0) < 1e-10
    with pytest.raises(ValueError):
        RACE_LR(**{k: v for k, v in dict(n_species=4, lmax=1, cutoff=5.0, hidden_irreps="8x0e", n_layers=1,
                radial_basis_size=4, radial_mlp_size=8, radial_mlp_layers=1, mlp_init_scale=4.0, shift=0.0,
                scale=1.0, avg_n_neighbors=10.0, atom_energies=ATOM_ENERGIES, l_train=True, periodic=False,
                x_mode="species").items()}, rngs=__import__('flax').nnx.Rngs(0),
                long_range={"qeq": False, "charge_spin_embedding": False})
    a = frame.copy(); a.calc = frame.calc; a.info['spin'] = 3
    g1 = preprocess_graph(a, atomic_numbers_to_indices(ATOMIC_NUMBERS), ATOM_ENERGIES, CUTOFF, spin_key='spin')
    g0 = preprocess_graph(a, atomic_numbers_to_indices(ATOMIC_NUMBERS), ATOM_ENERGIES, CUTOFF)
    assert float(g1.globals['spin'][0]) == 3.0 and 'spin' not in g0.globals


def _periodic_lr_model(ewald):
    from flax import nnx
    from bam_omat24.race_lr_nnx import RACE_LR
    return RACE_LR(n_species=len(ZS), lmax=2, cutoff=5.0, hidden_irreps="16x0e + 16x1o + 16x2e", n_layers=2,
                   radial_basis_size=8, radial_mlp_size=16, radial_mlp_layers=2, mlp_init_scale=4.0, shift=0.0,
                   scale=1.0, avg_n_neighbors=30.0, atom_energies=[0.0] * len(ZS), l_train=True, periodic=True,
                   x_mode="basis", x_norm="hill", residual="chebyshev", layer_scale=0.5, update="linear",
                   readout_mode="gated", message_norm="avg", radial_prior={"type": "gaussian", "sigma": 0.224},
                   radial_shells=True, species_shell_radii=element_shell_radii(ZS), pair_repulsion="uff",
                   pair_repulsion_switch=1.25, pair_repulsion_switch_width=0.3, species_z=ZS,
                   species_radii=[COVALENT_RADII[z] for z in ZS], rngs=nnx.Rngs(0),
                   long_range={"charge_spin_embedding": True, "chi_init": "pauling", **ewald})


def test_race_lr_periodic_stress_forces_translation(x64):
    """Periodic RACE_LR (Ewald + embedding, Q = +1, 2S+1 = 2): stress = (1/V) dE/d(strain), forces = -dE/dx, and
    the energy is unchanged when one atom moves by a lattice vector (converged Ewald sums)."""
    import jax
    import jax.numpy as jnp
    a = _cell(1)
    model, _ = as_float64(_periodic_lr_model({"ewald_alpha": 0.5, "ewald_n_real": 3, "ewald_n_recip": 10}), _graph(a))
    rng = np.random.default_rng(0)
    model.cs_mlp2.kernel.value = jnp.asarray(rng.normal(0, 0.3, model.cs_mlp2.kernel.value.shape))

    def prep(atoms):
        g = _f64(_graph(atoms))
        return g._replace(globals={**g.globals, 'charge': np.array([1.0]), 'spin': np.array([2.0])})
    g = prep(a)
    out = model(g)
    assert abs(float(np.asarray(out['charges']).sum()) - 1.0) < 1e-10
    stress = np.asarray(out["stress"])[0]
    cell0 = np.asarray(g.globals["cell"][0]); pos0 = np.asarray(g.nodes["positions"])
    vol = abs(np.linalg.det(cell0))

    def e_of(pos, cell):
        gg = g._replace(nodes={**g.nodes, "positions": pos},
                        globals={**g.globals, "cell": cell[None], "volume": np.array([abs(np.linalg.det(cell))])})
        return float(np.asarray(model(gg)["energy"])[0])
    h = 1e-5
    fd = []
    for (u, v) in [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]:
        eps = np.zeros((3, 3)); eps[u, v] += h / 2; eps[v, u] += h / 2
        fd.append((e_of(pos0 @ (np.eye(3) + eps).T, cell0 @ (np.eye(3) + eps).T)
                   - e_of(pos0 @ (np.eye(3) - eps).T, cell0 @ (np.eye(3) - eps).T)) / (2 * h) / vol)
    assert np.allclose(stress, np.array(fd), rtol=1e-4, atol=1e-7), (stress, fd)
    pp, pm = pos0.copy(), pos0.copy(); pp[2, 0] += h; pm[2, 0] -= h
    assert abs(-(e_of(pp, cell0) - e_of(pm, cell0)) / (2 * h) - float(out['forces'][2, 0])) < 1e-6
    from ase.calculators.singlepoint import SinglePointCalculator
    b = a.copy(); b.positions[4] += a.cell[0] - a.cell[2]
    b.calc = SinglePointCalculator(b, energy=-50.0, forces=np.zeros((len(b), 3)), stress=np.zeros(6))
    e_b = float(np.asarray(model(prep(b))["energy"])[0])
    assert abs(e_b - float(np.asarray(out["energy"])[0])) < 1e-6, (e_b, float(out["energy"][0]))


def test_race_lr_periodic_under_jit(x64):
    """The Ewald stratum must trace under jax.jit (the trainer jits the step): same energy and stress as eager."""
    import jax
    a = _cell(2)
    model, _ = as_float64(_periodic_lr_model({"ewald_alpha": 0.3, "ewald_n_real": 2, "ewald_n_recip": 6}), _graph(a))
    g = _f64(_graph(a))
    g = g._replace(globals={**g.globals, 'charge': np.array([0.0]), 'spin': np.array([1.0])})
    eager = model(g)
    jitted = jax.jit(lambda gg: model(gg))(g)
    assert abs(float(eager['energy'][0]) - float(jitted['energy'][0])) < 1e-9
    assert np.allclose(np.asarray(eager['stress']), np.asarray(jitted['stress']), atol=1e-10)


def test_race_lr_periodic_float32_gradients_finite():
    """float32, a batch with several padding graphs (zero cells) and large k vectors (exp(-k^2/4 alpha^2) underflows):
    the gradients of energy, forces and stress with respect to every parameter stay finite."""
    import jax
    import jax.numpy as jnp
    import jraph
    from flax import nnx
    gs = [_graph(_cell(s)) for s in (0, 1)]
    b = jraph.batch(gs)
    b = jraph.pad_with_graphs(b, int(b.n_node.sum()) + 5, int(b.n_edge.sum()) + 40, 6)
    b = b._replace(globals={**b.globals, 'charge': np.zeros(6, np.float32), 'spin': np.ones(6, np.float32)})
    b = jax.tree.map(jnp.asarray, b)
    m = _periodic_lr_model({"ewald_alpha": 0.3, "ewald_n_real": 2, "ewald_n_recip": 10})
    gd, p = nnx.split(m, nnx.Param)

    def loss(p):
        out = nnx.merge(gd, p)(b)
        return jnp.sum(out['energy'][:2]) + jnp.sum(out['forces'] ** 2) + 1e3 * jnp.sum(out['stress'][:2] ** 2)
    g = jax.grad(loss)(p)
    bad = [jax.tree_util.keystr(k) for k, v in jax.tree_util.tree_leaves_with_path(g) if not bool(jnp.all(jnp.isfinite(v)))]
    assert not bad, bad



def _two_fragments(graph, q_frag):
    """The 3BPA graph split into two fragments (first 10 atoms = 0, the rest = 1) with the given fragment charges."""
    n = int(graph.n_node.sum())
    frag = np.zeros(graph.nodes['positions'].shape[0], np.int32); frag[10:n] = 1
    return graph._replace(nodes={**graph.nodes, 'fragment': frag},
                          globals={**graph.globals, 'charge': np.array([sum(q_frag), 0.0]),
                                   'fragment_charges': np.array([q_frag, [0.0, 0.0]])})


def test_fragment_charge_constraints(x64, graph):
    """Per-fragment constraints: each fragment carries exactly its charge (here +1 / -1, total 0), unlike the global
    constraint, and forces = -dE/dx still hold."""
    import jax
    import jax.numpy as jnp
    model, g = as_float64(_lr_model(), pad_single(graph))
    n = int(g.n_node[0])
    run = lambda gr: model(jax.tree.map(jnp.asarray, gr))
    gf = _two_fragments(g, [1.0, -1.0])
    out = run(gf)
    q = np.asarray(out['charges'])
    assert abs(q[:10].sum() - 1.0) < 1e-10 and abs(q[10:n].sum() + 1.0) < 1e-10 and np.all(q[n:] == 0.0)
    q0 = np.asarray(run(g._replace(globals={**g.globals, 'charge': np.array([0.0, 0.0])}))['charges'])
    assert abs(q0[:10].sum() - 1.0) > 0.1                    # the global constraint does not fix the fragment charge
    p = np.asarray(gf.nodes['positions']); h = 1e-5
    pp, pm = p.copy(), p.copy(); pp[12, 1] += h; pm[12, 1] -= h
    ep = float(run(gf._replace(nodes={**gf.nodes, 'positions': pp}))['energy'][0])
    em = float(run(gf._replace(nodes={**gf.nodes, 'positions': pm}))['energy'][0])
    assert abs(-(ep - em) / (2 * h) - float(out['forces'][12, 1])) < 1e-6


def test_fragments_in_graph(frame):
    from conftest import preprocess_graph, atomic_numbers_to_indices, CUTOFF
    ai = atomic_numbers_to_indices(ATOMIC_NUMBERS)
    a = frame.copy(); a.calc = frame.calc
    a.arrays['is_ligand'] = (np.arange(len(a)) >= 10).astype(int)
    a.info.update(charge=1, fragment_charges=[2, -1])
    g = preprocess_graph(a, ai, ATOM_ENERGIES, CUTOFF, charge_key='charge', fragments=('is_ligand', 'fragment_charges', 2))
    assert g.nodes['fragment'].tolist() == a.arrays['is_ligand'].tolist()
    assert g.globals['fragment_charges'].tolist() == [[2.0, -1.0]]
    b = frame.copy(); b.calc = frame.calc; b.info['charge'] = -1                    # no fragment array: one fragment
    gb = preprocess_graph(b, ai, ATOM_ENERGIES, CUTOFF, charge_key='charge', fragments=('is_ligand', 'fragment_charges', 2))
    assert gb.globals['fragment_charges'].tolist() == [[-1.0, 0.0]]
    a.info['fragment_charges'] = [2, 0]                                             # does not sum to the charge
    with pytest.raises(ValueError):
        preprocess_graph(a, ai, ATOM_ENERGIES, CUTOFF, charge_key='charge', fragments=('is_ligand', 'fragment_charges', 2))
    del a.info['fragment_charges']                                                  # two fragments, no charges
    with pytest.raises(ValueError):
        preprocess_graph(a, ai, ATOM_ENERGIES, CUTOFF, charge_key='charge', fragments=('is_ligand', 'fragment_charges', 2))


def test_short_range_switch_molecular(x64, graph):
    """sr_switch: no effect when every pair is beyond r_off; otherwise a different energy with the same charges, and
    forces = -dE/dx."""
    import jax
    import jax.numpy as jnp
    g = pad_single(graph)
    g = g._replace(globals={**g.globals, 'charge': np.array([1.0, 0.0])})
    run = lambda m, gr: m(jax.tree.map(jnp.asarray, gr))
    m0, g64 = as_float64(_lr_model(), g)
    m_far, _ = as_float64(_lr_model(lr={"sr_switch": [0.01, 0.02]}), g)
    m_sw, _ = as_float64(_lr_model(lr={"sr_switch": [2.0, 4.0]}), g)
    r0, rf, rs = run(m0, g64), run(m_far, g64), run(m_sw, g64)
    assert abs(float(rf['energy'][0]) - float(r0['energy'][0])) < 1e-9
    assert abs(float(rs['energy'][0]) - float(r0['energy'][0])) > 1e-4
    assert np.allclose(np.asarray(rs['charges']), np.asarray(r0['charges']), atol=1e-12)
    p = np.asarray(g64.nodes['positions']); h = 1e-5
    for i, c in [(3, 0), (20, 2)]:
        pp, pm = p.copy(), p.copy(); pp[i, c] += h; pm[i, c] -= h
        ep = float(run(m_sw, g64._replace(nodes={**g64.nodes, 'positions': pp}))['energy'][0])
        em = float(run(m_sw, g64._replace(nodes={**g64.nodes, 'positions': pm}))['energy'][0])
        assert abs(-(ep - em) / (2 * h) - float(rs['forces'][i, c])) < 1e-6


def test_short_range_switch_periodic_stress(x64):
    """Periodic sr_switch (images |n| <= 1): stress = (1/V) dE/d(strain) and forces = -dE/dx."""
    a = _cell(1)
    model, _ = as_float64(_periodic_lr_model({"ewald_alpha": 0.5, "ewald_n_real": 3, "ewald_n_recip": 10,
                                              "sr_switch": [1.5, 3.5]}), _graph(a))
    g = _f64(_graph(a))
    g = g._replace(globals={**g.globals, 'charge': np.array([0.0]), 'spin': np.array([1.0])})
    out = model(g)
    stress = np.asarray(out["stress"])[0]
    cell0 = np.asarray(g.globals["cell"][0]); pos0 = np.asarray(g.nodes["positions"])

    def e_of(pos, cell):
        gg = g._replace(nodes={**g.nodes, "positions": pos},
                        globals={**g.globals, "cell": cell[None], "volume": np.array([abs(np.linalg.det(cell))])})
        return float(np.asarray(model(gg)["energy"])[0])
    h, vol, fd = 1e-5, abs(np.linalg.det(cell0)), []
    for (u, v) in [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]:
        eps = np.zeros((3, 3)); eps[u, v] += h / 2; eps[v, u] += h / 2
        fd.append((e_of(pos0 @ (np.eye(3) + eps).T, cell0 @ (np.eye(3) + eps).T)
                   - e_of(pos0 @ (np.eye(3) - eps).T, cell0 @ (np.eye(3) - eps).T)) / (2 * h) / vol)
    assert np.allclose(stress, np.array(fd), rtol=1e-4, atol=1e-7), (stress, fd)
    pp, pm = pos0.copy(), pos0.copy(); pp[6, 2] += h; pm[6, 2] -= h
    assert abs(-(e_of(pp, cell0) - e_of(pm, cell0)) / (2 * h) - float(out['forces'][6, 2])) < 1e-6


def test_predict_api_and_species_refit(tmp_path, frame):
    """bam_omat24.predict: the Predictor reproduces model + E0 of a checkpoint, returns Qeq charges that sum to the
    total charge, and the training-split species refit recovers per-element energy offsets."""
    import io
    import jax
    import jax.numpy as jnp
    from bam_omat24 import train_3bpa as T
    from bam_omat24.training_utils import save_checkpoint
    from bam_omat24.predict import load, species_refit
    from bam_omat24.data_3bpa import load_frames
    cfg = {"model": "RACE_LR", "atomic_numbers": ATOMIC_NUMBERS, "atom_energies": ATOM_ENERGIES, "cutoff": 5.0,
           "hidden_size": 8, "lmax": 1, "n_layers": 1, "radial_mlp_size": 8, "radial_mlp_layers": 1,
           "avg_n_neighbors": 15.0, "tp_backend": "e3nn", "total_charge_key": "charge", "x64": False,
           "long_range": {"chi_init": "pauling"}, "node_bucket_size": 16, "edge_bucket_size": 256}
    stats = {'shift_raw': 0.0, 'scale': 1.0, 'avg_n_neighbors': 15.0, 'n_graphs': 1, 'n_atoms': 1}
    graphdef, params, *_ = T.build_model({**cfg, 'learning_rate': 1e-3}, 4, ATOM_ENERGIES, stats, io.StringIO())
    ck = tmp_path / 'ckpt.pkl'
    save_checkpoint({'config': cfg, 'ema_params': params}, str(ck))
    frames = load_frames(T.Path(__file__).resolve().parents[1] / 'dataset_3BPA' / 'train_300K.xyz')[:6]
    for i, a in enumerate(frames):
        a.info['charge'] = [0, 1, -1, 0, 1, 0][i]
    p = load(ck)
    e, f = p.predict(frames)
    from bam_omat24.data_3bpa import preprocess_graph, atomic_numbers_to_indices
    from flax import nnx
    model = nnx.merge(graphdef, params)
    a = frames[1]
    g = pad_single(preprocess_graph(a, atomic_numbers_to_indices(ATOMIC_NUMBERS), ATOM_ENERGIES, 5.0, targets=False,
                                    charge_key='charge'))
    g = g._replace(globals={**g.globals, 'charge': np.array([1.0, 0.0])})
    ref = float(model(jax.tree.map(jnp.asarray, g))['energy'][0]) + sum(
        ATOM_ENERGIES[ATOMIC_NUMBERS.index(z)] for z in a.numbers)
    assert abs(e[1] - ref) < 1e-3 * max(1.0, abs(ref)) and f[1].shape == (len(a), 3)
    q = p.charges(frames)
    assert all(abs(qi.sum() - c) < 1e-4 for qi, c in zip(q, [0, 1, -1, 0, 1, 0]))
    # species refit: labels = prediction + known per-element offsets -> the refit recovers them (single composition
    # here, so only the composition total is identifiable: compare the corrected energies)
    from ase.calculators.singlepoint import SinglePointCalculator
    off = np.array([0.01, -0.02, 0.03, 0.005])
    for a, ei, fi in zip(frames, e, f):
        a.calc = SinglePointCalculator(a, energy=float(ei - p.counts([a])[0] @ off), forces=fi)
    r = species_refit(ck, frames)
    assert r['rmse_after_eV'] < 1e-4
    e2, _ = load(ck).predict(frames)
    assert np.allclose(e2, [a.get_potential_energy() for a in frames], atol=1e-3)


def test_predict_atom_budget_batches(tmp_path):
    """Predictor cuts batches at max_atoms (the Qeq solve is dense per batch): the results do not depend on it, also
    for frames of about 1000 atoms."""
    import io
    import ase
    from bam_omat24 import train_3bpa as T
    from bam_omat24.training_utils import save_checkpoint
    from bam_omat24.predict import load
    from bam_omat24.data_3bpa import load_frames
    cfg = {"model": "RACE_LR", "atomic_numbers": ATOMIC_NUMBERS, "atom_energies": ATOM_ENERGIES, "cutoff": 5.0,
           "hidden_size": 8, "lmax": 1, "n_layers": 1, "radial_mlp_size": 8, "radial_mlp_layers": 1,
           "avg_n_neighbors": 15.0, "tp_backend": "e3nn", "total_charge_key": "charge", "x64": False,
           "node_bucket_size": 64, "edge_bucket_size": 1024, "learning_rate": 1e-3}
    stats = {'shift_raw': 0.0, 'scale': 1.0, 'avg_n_neighbors': 15.0, 'n_graphs': 1, 'n_atoms': 1}
    _, params, *_ = T.build_model(cfg, 4, ATOM_ENERGIES, stats, io.StringIO())
    ck = tmp_path / 'ckpt.pkl'
    save_checkpoint({'config': cfg, 'ema_params': params}, str(ck))
    mol = load_frames(T.Path(__file__).resolve().parents[1] / 'dataset_3BPA' / 'train_300K.xyz')[:3]
    big = []
    for k, q in enumerate([0, 1]):                          # two "complexes" of 37 molecules (999 atoms), 12 A apart
        a = ase.Atoms()
        for i in range(37):
            m = mol[(i + k) % 3].copy()
            m.positions += 12.0 * np.array([i % 4, (i // 4) % 4, i // 16])
            a += m
        a.info['charge'] = q
        big.append(a)
    frames = big + [m.copy() for m in mol]
    for m in frames[2:]:
        m.info['charge'] = 0
    e1, f1 = load(ck, max_atoms=100000).predict(frames)
    p_small = load(ck, max_atoms=1000)
    assert len(p_small._chunks(frames)) == 3                 # [big0], [big1], [3 small]
    e2, f2 = p_small.predict(frames)
    assert np.allclose(e1, e2, atol=1e-3) and all(np.allclose(a, b, atol=1e-4) for a, b in zip(f1, f2))
    q = p_small.charges(frames[:2])
    assert abs(q[0].sum()) < 1e-3 and abs(q[1].sum() - 1.0) < 1e-3 and len(q[0]) == 999


def _batch_of(graphs, extra_nodes, extra_edges, n_pad_graphs):
    import jraph
    b = jraph.batch(graphs)
    return jraph.pad_with_graphs(b, int(b.n_node.sum()) + extra_nodes, int(b.n_edge.sum()) + extra_edges,
                                 len(graphs) + n_pad_graphs)


def test_qeq_block_equals_dense_molecular(x64, graph):
    """long_range.qeq_block: per-structure blocks give the dense batch result (energies, forces, charges), with and
    without fragments and the short-range switch; a structure larger than the block makes the energy NaN."""
    import copy
    import jax
    import jax.numpy as jnp
    from flax import nnx
    g1 = graph
    g2 = graph._replace(nodes={**graph.nodes, 'positions': np.asarray(graph.nodes['positions']) + 0.05})
    b = _batch_of([g1, g2], 5, 40, 2)
    G = b.n_node.shape[0]
    n = int(graph.n_node[0])
    frag = np.zeros(b.nodes['positions'].shape[0], np.int32); frag[10:n] = 1; frag[n + 5:2 * n] = 1
    variants = {
        'plain': dict(charge=np.array([1.0, -1.0] + [0.0] * (G - 2))),
        'fragments': dict(charge=np.array([0.0, 1.0] + [0.0] * (G - 2)),
                          fragment_charges=np.array([[1.0, -1.0], [2.0, -1.0]] + [[0.0, 0.0]] * (G - 2))),
    }
    for lr in ({}, {"sr_switch": [2.0, 4.0]}):
        dense, _ = as_float64(_lr_model(lr=lr), b)
        block = copy.deepcopy(dense); block.lr = {**dense.lr, 'qeq_block': 30}
        for name, glb in variants.items():
            nodes = {**b.nodes, **({'fragment': frag} if 'fragment_charges' in glb else {})}
            bb = jax.tree.map(jnp.asarray, b._replace(nodes={**nodes, 'positions': np.asarray(b.nodes['positions'], np.float64)},
                                                      globals={**b.globals, **glb}))
            rd, rb = dense(bb), block(bb)
            for k in ('energy', 'forces', 'charges'):
                assert np.allclose(np.asarray(rd[k]), np.asarray(rb[k]), atol=1e-9), (lr, name, k)
    small = copy.deepcopy(dense); small.lr = {**dense.lr, 'qeq_block': 20}           # 27-atom structures > 20
    bb = jax.tree.map(jnp.asarray, b._replace(nodes={**b.nodes, 'positions': np.asarray(b.nodes['positions'], np.float64)},
                                              globals={**b.globals, **variants['plain']}))
    assert np.isnan(float(small(bb)['energy'][0]))


def test_qeq_block_equals_dense_periodic(x64):
    """Periodic batch of two different cells plus padding graphs: blocks (per-graph Ewald) equal the dense rows."""
    import copy
    import jax
    import jax.numpy as jnp
    gs = [_f64(_graph(_cell(s))) for s in (0, 1)]
    b = _batch_of(gs, 3, 30, 2)
    G = b.n_node.shape[0]
    b = b._replace(globals={**b.globals, 'charge': np.array([1.0, 0.0] + [0.0] * (G - 2)),
                            'spin': np.ones(G)})
    for lr in ({"ewald_alpha": 0.4, "ewald_n_real": 2, "ewald_n_recip": 8},
               {"ewald_alpha": 0.4, "ewald_n_real": 2, "ewald_n_recip": 8, "sr_switch": [1.5, 3.5]}):
        dense, _ = as_float64(_periodic_lr_model(lr), gs[0])
        block = copy.deepcopy(dense); block.lr = {**dense.lr, 'qeq_block': 12}
        bb = jax.tree.map(jnp.asarray, b)
        rd, rb = dense(bb), block(bb)
        for k in ('energy', 'forces', 'stress', 'charges'):
            assert np.allclose(np.asarray(rd[k])[:2] if k in ('energy', 'stress') else np.asarray(rd[k]),
                               np.asarray(rb[k])[:2] if k in ('energy', 'stress') else np.asarray(rb[k]), atol=1e-8), (lr, k)
