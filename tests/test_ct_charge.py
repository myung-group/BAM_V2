"""Distance-limited charge transfer and bounded Qeq widths of RACE_LR (2026-11-13; long_range ct_mode "conductance",
sigma_bounds; graph_fields.formal_charges; long_range.ct_solve).  The JAX tests run in a Slurm job."""
import numpy as np
import pytest

from conftest import ATOMIC_NUMBERS, ATOM_ENERGIES

# imported at module level: train_3bpa forces float32 when it is first imported, which would undo the x64 fixture
from bam_omat24.train_3bpa import COVALENT_RADII  # noqa: E402,F401

R_OH, ANG = 0.9572, np.deg2rad(104.52)
H1 = np.array([R_OH * np.sin(ANG / 2), 0.0, R_OH * np.cos(ANG / 2)])
H2 = H1 * np.array([-1.0, 1.0, 1.0])


def _pairs(pos, cutoff=5.0):
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    i, j = np.nonzero((d < cutoff) & (d > 0))
    return i, j


# ------------------------------------------------------------------------------------------- JAX-free: formal charges
def test_formal_charges_round_per_molecule():
    from bam_omat24.graph_fields import formal_charges, molecules
    nums = np.array([8, 1, 1, 11])
    # Na-O 2.3 A is below 1.2 (r_Na + r_O) = 2.78 A: one molecule; at 3.5 A two
    pos = np.array([[0, 0, 0], H1, H2, [0, 0, -2.3]])
    assert molecules(nums, pos, *_pairs(pos))[1] == 1
    pos = np.array([[0, 0, 0], H1, H2, [0, 0, -3.5]])
    src, dst = _pairs(pos)
    lab, k = molecules(nums, pos, src, dst)
    assert k == 2 and lab[0] == lab[1] == lab[2] != lab[3]
    # contact ion pair: the labels move 0.16 e to the water; rounded per molecule: Na +1, water 0
    q0, src_ = formal_charges(nums, pos, src, dst, 1.0, info={'lowdin_charges': [-0.5, 0.33, 0.33, 0.84]})
    assert src_ == 'lowdin_charges' and np.allclose(q0, [0, 0, 0, 1])
    # NBO first when there
    q0, src_ = formal_charges(nums, pos, src, dst, 1.0, info={'nbo_charges': [-0.9, 0.45, 0.45, 1.0],
                                                             'lowdin_charges': [0, 0, 0, 1]})
    assert src_ == 'nbo_charges' and np.allclose(q0, [0, 0, 0, 1])
    # given per atom (inference)
    q0, src_ = formal_charges(nums, pos, src, dst, 1.0, arrays={'formal_charges': np.array([0, 0, 0, 1])})
    assert src_ == 'given' and np.allclose(q0, [0, 0, 0, 1])
    # largest remainder: two half charges that round to 0 + 0 for a total of +1 -> the larger remainder gets it
    q0, _ = formal_charges(nums, pos, src, dst, 1.0, info={'lowdin_charges': [0.1, 0.1, 0.2, 0.45]})
    assert np.isclose(q0.sum(), 1.0) and np.isclose(q0[3], 1.0)
    # no labels: neutral -> zeros; one molecule -> spread; several molecules with a charge -> Q / N (no integer split)
    assert np.allclose(formal_charges(nums, pos, src, dst, 0.0)[0], 0.0)
    q0, src_ = formal_charges(nums[:3], pos[:3], *_pairs(pos[:3]), -1.0)
    assert np.allclose(q0, -1.0 / 3) and src_ == 'single molecule'
    q0, src_ = formal_charges(nums, pos, src, dst, 1.0)
    assert np.allclose(q0, 0.25) and src_.startswith('uniform')


# ------------------------------------------------------------------------------------------------------- ct_solve
def test_ct_solve_conserves_components_and_tends_to_qeq(x64):
    import jax.numpy as jnp
    from bam_omat24.long_range import coulomb_direct, ct_solve, qeq_solve
    rng = np.random.default_rng(0)
    N = 6
    pos = jnp.asarray(rng.uniform(0, 3, (N, 3)))
    chi, hard = jnp.asarray(rng.normal(size=N)), jnp.asarray(rng.uniform(3, 6, N))
    sig = jnp.asarray(rng.uniform(0.6, 1.0, N))
    mask = jnp.ones(N, bool)
    gam = coulomb_direct(pos, sig, mask[:, None] & mask[None, :])
    # two components {0,1,2} and {3,4,5}: each keeps the sum of its q0
    C = np.zeros((N, N)); C[:3, :3] = 2.0; C[3:, 3:] = 2.0; np.fill_diagonal(C, 0.0)
    q0 = jnp.asarray([1.0, 0, 0, -0.5, -0.5, 0.0])
    q, e = ct_solve(chi, hard, sig, gam, jnp.asarray(C), mask, q0)
    q = np.asarray(q)
    assert abs(q[:3].sum() - 1.0) < 1e-12 and abs(q[3:].sum() + 1.0) < 1e-12
    # no conductance: q = q0
    q, _ = ct_solve(chi, hard, sig, gam, jnp.zeros((N, N)), mask, q0)
    assert np.allclose(np.asarray(q), np.asarray(q0), atol=1e-12)
    # very large conductance in one component: the Qeq charges of the structure
    Cbig = np.full((N, N), 1e7); np.fill_diagonal(Cbig, 0.0)
    q, _ = ct_solve(chi, hard, sig, gam, jnp.asarray(Cbig), mask, jnp.full(N, 0.5 / N))
    qref, _ = qeq_solve(chi, hard, sig, gam, jnp.zeros(N, jnp.int32), mask, jnp.asarray([0.5]), 1)
    assert np.allclose(np.asarray(q), np.asarray(qref), atol=1e-5)


def test_ct_solve_float32_inputs_keep_the_charge(x64):
    """a large, strongly coupled structure with float32 inputs (a float32 model under jax_enable_x64): the total
    charge is kept to float32 rounding (the float32 LU of I + L A lost it completely before the float64 solve)"""
    import jax.numpy as jnp
    from bam_omat24.long_range import coulomb_direct, ct_solve
    rng = np.random.default_rng(3)
    N = 300
    pos = rng.uniform(0, 14, (N, 3))
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    C = np.where((d < 3.0) & (d > 0), 10.0, 0.0)
    sig = rng.uniform(0.15, 1.0, N)
    mask = jnp.ones(N, bool)
    gam = coulomb_direct(jnp.asarray(pos), jnp.asarray(sig), mask[:, None] & mask[None, :])
    q0 = np.zeros(N); q0[:7] = 1.0
    f32 = lambda x: jnp.asarray(x, jnp.float32)
    q, e = ct_solve(f32(rng.normal(0, 8, N)), f32(rng.uniform(0, 8, N)), f32(sig), f32(gam), f32(C), mask, f32(q0))
    assert q.dtype == jnp.float32
    assert abs(float(jnp.sum(q)) - 7.0) < 1e-3 and float(jnp.max(jnp.abs(q))) < 20.0


# ------------------------------------------------------------------------------------------------------- the model
CT = {'qeq_block': 8, 'ct_mode': 'conductance', 'sigma_bounds': [0.5, 2.0], 'ct_init': 0.0}


def _model(lr):
    from flax import nnx
    from bam_omat24.race_lr_nnx import RACE_LR
    return RACE_LR(n_species=4, lmax=2, cutoff=5.0, hidden_irreps="16x0e + 16x1o + 16x2e", n_layers=2,
                   radial_basis_size=8, radial_mlp_size=16, radial_mlp_layers=2, mlp_init_scale=4.0, shift=0.0,
                   scale=1.0, avg_n_neighbors=20.0, atom_energies=ATOM_ENERGIES, l_train=True, periodic=False,
                   x_mode="species", rngs=nnx.Rngs(0), species_radii=[0.31, 0.76, 0.71, 0.66],
                   species_z=ATOMIC_NUMBERS, long_range={"chi_scale": 1.0, **lr})


def _graph(R, q_c=1.0):
    """water + a C atom (stand-in ion) on the C2 axis at distance R, total charge q_c on the C atom"""
    import ase
    from bam_omat24.data_3bpa import preprocess_graph, atomic_numbers_to_indices
    from bam_omat24.graph_fields import formal_charges
    from conftest import pad_single
    a = ase.Atoms('OHHC', positions=[[0, 0, 0], H1, H2, [0, 0, -R]])
    a.info['charge'] = q_c
    g = preprocess_graph(a, atomic_numbers_to_indices(ATOMIC_NUMBERS), ATOM_ENERGIES, 5.0, targets=False,
                         charge_key='charge')
    q0, _ = formal_charges(a.numbers, a.positions, g.receivers, g.senders, q_c,
                           arrays={'formal_charges': np.array([0, 0, 0, q_c])})
    g = g._replace(nodes={**g.nodes, 'q0': q0.astype(np.float64),
                          'positions': np.asarray(g.nodes['positions'], np.float64)},
                   globals={k: np.asarray(v, np.float64) for k, v in g.globals.items()})
    return pad_single(g)


def _f64(model):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    gd, st = nnx.split(model)
    st = jax.tree.map(lambda x: x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x, st)
    return nnx.merge(gd, st)


def _run(model, g):
    import jax
    import jax.numpy as jnp
    return model(jax.tree.map(jnp.asarray, g))


def test_sigma_bounds_start_at_covalent_radii(x64):
    import jax.numpy as jnp
    m = _f64(_model(CT))
    s = np.asarray(m._sigma(jnp.arange(4), jnp.float64))
    assert np.allclose(s, [0.31, 0.76, 0.71, 0.66], atol=1e-6)
    m.log_sigma.value = jnp.asarray([50.0, -50.0, 0.0, 0.0])
    s = np.asarray(m._sigma(jnp.arange(4), jnp.float64))
    assert np.allclose(s[:2], [0.31 * 2.0, 0.76 * 0.5], atol=1e-6)


def test_ct_model_integer_charges_apart_transfer_in_contact_and_forces(x64):
    import jax.numpy as jnp
    m = _f64(_model(CT))
    m.chi0.value = jnp.asarray([0.3, -2.0, 0.1, 1.5])          # make the transfer non-trivial
    far = _run(m, _graph(7.0))
    q = np.asarray(far['charges'])
    assert abs(q[3] - 1.0) < 1e-12 and abs(q[:3].sum()) < 1e-12        # beyond ct_switch[1] = 4.5 A: integers
    near = _run(m, _graph(2.2))
    q = np.asarray(near['charges'])
    assert abs(q[:4].sum() - 1.0) < 1e-10 and abs(q[3] - 1.0) > 1e-4   # in contact: some charge moves, total kept
    # forces = -dE/dx in the switching region of the conductance (C ... O at 3.9 A, C ... H in between)
    g = _graph(3.9)
    r0 = _run(m, g)
    p = np.asarray(g.nodes['positions']).copy()
    h = 1e-5
    for i, c in [(3, 2), (1, 0), (0, 2)]:
        pp, pm = p.copy(), p.copy(); pp[i, c] += h; pm[i, c] -= h
        ep = float(_run(m, g._replace(nodes={**g.nodes, 'positions': pp}))['energy'][0])
        em = float(_run(m, g._replace(nodes={**g.nodes, 'positions': pm}))['energy'][0])
        assert abs(-(ep - em) / (2 * h) - float(r0['forces'][i, c])) < 1e-6
    # the energy and charges are continuous across the end of the switch (4.5 A)
    e1, e2 = float(_run(m, _graph(4.4999))['energy'][0]), float(_run(m, _graph(4.5001))['energy'][0])
    assert abs(e1 - e2) < 1e-4


def test_ct_options_need_qeq_block():
    with pytest.raises(ValueError):
        _model({'ct_mode': 'conductance'})
    with pytest.raises(ValueError):
        _model({'sigma_bounds': [1.2, 2.0]})


def test_ct_cond_max_caps_the_conductance(x64):
    """long_range ct_cond_max: conductances <= C; below the cap almost unchanged; with huge g (no cap: 1e10) the
    capped model gives the free-flow charges and a well-conditioned solve"""
    import jax.numpy as jnp
    m0 = _f64(_model(CT))
    m1 = _f64(_model({**CT, 'ct_cond_max': 100.0}))
    for m in (m0, m1):
        m.chi0.value = jnp.asarray([0.3, -2.0, 0.1, 1.5])
    g = _graph(2.2)
    B = 8
    data = jax_tree(g)
    n = int(np.asarray(g.n_node).sum())
    mask = np.zeros(int(np.asarray(g.nodes['species']).shape[0]), bool); mask[:n] = True
    node_graph = jnp.repeat(jnp.arange(g.n_node.shape[0]), jnp.asarray(g.n_node), total_repeat_length=mask.size)
    slot = jnp.where(jnp.asarray(mask), jnp.arange(mask.size), B)
    args = (jnp.asarray(g.nodes['positions']), None, data, node_graph, jnp.asarray(mask), slot, g.n_node.shape[0], B,
            jnp.float64)
    c0, c1 = np.asarray(m0._ct_conductance(*args)), np.asarray(m1._ct_conductance(*args))
    assert c1.max() <= 100.0 + 1e-9 and np.abs(c1 - c0).max() / max(c0.max(), 1e-12) < 0.02   # g = 0: cond ~1
    for m in (m0, m1):
        m.ct_log_g.value = jnp.full((4,), 12.0)                         # the v2local drift
    c0, c1 = np.asarray(m0._ct_conductance(*args)), np.asarray(m1._ct_conductance(*args))
    assert c0.max() > 1e9 and c1.max() <= 100.0 + 1e-9 and c1.max() > 90.0
    q1 = np.asarray(_run(m1, g)['charges'])[:4]
    m1.ct_log_g.value = jnp.full((4,), 3.0)                            # still capped at ~100: the same charges
    q2 = np.asarray(_run(m1, g)['charges'])[:4]
    assert abs(q1.sum() - 1.0) < 1e-10 and np.abs(q1 - q2).max() < 0.02


def test_ct_decay_fades_the_charge_transfer(x64):
    """long_range ct_decay [beta, r0]: cond = exp(g_i + g_j - beta (r - r0)) (1 - switch); with a saturated
    conductance the charge of the ion no longer drops as a step just before ct_switch[1] but fades over ~1 A"""
    import jax.numpy as jnp
    # 1) the conductance formula (g = 0, no cap): c1 = c0 exp(-beta (r - r0))
    m0, m1 = _f64(_model(CT)), _f64(_model({**CT, 'ct_decay': [3.0, 2.0]}))
    g = _graph(3.2)
    B = 8
    data = jax_tree(g)
    n = int(np.asarray(g.n_node).sum())
    mask = np.zeros(int(np.asarray(g.nodes['species']).shape[0]), bool); mask[:n] = True
    node_graph = jnp.repeat(jnp.arange(g.n_node.shape[0]), jnp.asarray(g.n_node), total_repeat_length=mask.size)
    slot = jnp.where(jnp.asarray(mask), jnp.arange(mask.size), B)
    pos = np.asarray(g.nodes['positions'])
    args = (jnp.asarray(pos), None, data, node_graph, jnp.asarray(mask), slot, g.n_node.shape[0], B, jnp.float64)
    c0, c1 = np.asarray(m0._ct_conductance(*args))[0], np.asarray(m1._ct_conductance(*args))[0]
    d = np.linalg.norm(pos[:4, None] - pos[None, :4], axis=-1)
    on = c0[:4, :4] > 1e-12
    assert on.sum() > 0
    assert np.allclose(c1[:4, :4][on], (c0[:4, :4] * np.exp(-3.0 * (d - 2.0)))[on], rtol=1e-6)
    # 2) saturated conductance under the cap (the v2local case): charge of the ion along the approach line
    base = {**CT, 'ct_cond_max': 100.0}
    m0, m1 = _f64(_model(base)), _f64(_model({**base, 'ct_decay': [3.0, 2.0]}))
    for m in (m0, m1):
        m.chi0.value = jnp.asarray([0.3, -2.0, 0.1, 1.5])
        m.ct_log_g.value = jnp.full((4,), 3.0)
    Rs = np.round(np.arange(3.6, 4.61, 0.04), 3)
    q = {k: np.array([float(np.asarray(_run(m, _graph(R))['charges'])[3]) for R in Rs]) for k, m in (('0', m0), ('1', m1))}
    assert abs(q['0'][0] - 1.0) > 0.02                          # a real charge transfer inside the switch
    assert np.allclose(q['0'][Rs >= 4.5], 1.0, atol=1e-9) and np.allclose(q['1'][Rs >= 4.5], 1.0, atol=1e-9)
    j0, j1 = np.abs(np.diff(q['0'])).max(), np.abs(np.diff(q['1'])).max()
    assert j1 < 0.5 * j0, (j0, j1)


def jax_tree(g):
    import jax
    import jax.numpy as jnp
    return jax.tree.map(jnp.asarray, g)
