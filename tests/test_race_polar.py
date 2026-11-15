"""RACE_POLAR (RACEv2 + Thole-type polarizable electrostatics, race_v2/models/race_polar.py): rotation behaviour,
forces and stress against finite differences, total charge, and independence of the graphs of a mixed (periodic +
open-boundary) batch.  The electrostatics themselves are checked against BAM-torch in tests/test_thole.py."""
import numpy as np
import jax
import jraph
import pytest
from flax import nnx

from conftest import ATOMIC_NUMBERS, ATOM_ENERGIES
from race_v2.training.train import COVALENT_RADII  # noqa: F401  (imports training.train before the x64 fixture)
from race_v2.data import stream_io as S
from race_v2.data.stream_data import _to_graph, _cast

CUTOFF = 5.0
IDX = {z: i for i, z in enumerate(ATOMIC_NUMBERS)}
FIELDS = dict(charge_key='charge', spin_key='spin', mixed=True, task=0)


def _model(periodic, seed=0, polar=None, noise=0.3):
    from race_v2.models.race_polar import RACE_POLAR
    import jax.numpy as jnp
    pol = {'max_cell_length': 9.0, 'max_cell_volume': 450.0, 'cg_tolerance': 1e-12, 'cg_max_iterations': 200,
           'multipole_init_scale': 1.0, 'max_polarizability_per_element': {'H': 1.0}, **(polar or {})}
    m = RACE_POLAR(n_species=4, lmax=2, cutoff=CUTOFF, hidden_irreps="16x0e + 16x1o + 16x2e", n_layers=2,
                   radial_basis_size=8, radial_mlp_size=16, radial_mlp_layers=2, mlp_init_scale=4.0, shift=0.0,
                   scale=1.0, avg_n_neighbors=20.0, atom_energies=ATOM_ENERGIES, l_train=True, periodic=periodic,
                   x_mode="species", rngs=nnx.Rngs(seed), species_radii=[0.31, 0.76, 0.71, 0.66],
                   species_z=ATOMIC_NUMBERS, long_range={'charge_spin_embedding': True, 'n_tasks': 3}, polar=pol)
    # the quadrupole and polarizability heads start at zero: perturb the polar heads so that every term is active
    graphdef, params = nnx.split(m, nnx.Param)
    key = jax.random.PRNGKey(seed + 1)
    flat = dict(nnx.to_flat_state(params)) if hasattr(nnx, 'to_flat_state') else None
    leaves, treedef = jax.tree.flatten_with_path(params)
    out = []
    for i, (path, x) in enumerate(leaves):
        name = '/'.join(str(getattr(k, 'key', getattr(k, 'name', k))) for k in path)
        x = jnp.asarray(x, jnp.float64)
        if 'pol_' in name:
            x = x + noise * jax.random.normal(jax.random.fold_in(key, i), x.shape, jnp.float64)
        out.append(x)
    params = jax.tree.unflatten(jax.tree.structure(params), out)
    return nnx.merge(graphdef, params)


def _structure(kind, seed, charge=0.0):
    rng = np.random.default_rng(seed)
    n = 10 if kind == 'p' else 9
    z = rng.choice(ATOMIC_NUMBERS, n)
    if kind == 'p':
        cell = np.diag([7.0, 7.4, 7.9]) + rng.normal(0, 0.3, (3, 3))
        # atoms on a jittered grid: no close contacts
        frac = (np.stack(np.meshgrid(*[np.arange(3)] * 3, indexing='ij'), -1).reshape(-1, 3)[rng.permutation(27)[:n]]
                + 0.5 + rng.normal(0, 0.08, (n, 3))) / 3.0
        pos, pbc = frac @ cell, np.ones(3, bool)
    else:
        cell, pbc = np.zeros((3, 3)), np.zeros(3, bool)
        g = np.stack(np.meshgrid(*[np.arange(3)] * 3, indexing='ij'), -1).reshape(-1, 3)[rng.permutation(27)[:n]]
        pos = g * 1.6 + rng.normal(0, 0.15, (n, 3))
    return {'numbers': z, 'positions': pos, 'cell': cell, 'pbc': pbc, 'energy': 0.0, 'forces': np.zeros((n, 3)),
            'stress': np.zeros(6) if kind == 'p' else None, 'info': {'charge': charge, 'spin': 1}, 'arrays': {}}


def _graph(st):
    d = _cast(S.graph_arrays(st, IDX, np.zeros(4), CUTOFF, dict(FIELDS)), np.float64)
    # graph_arrays stores float32 coordinates; the finite-difference checks need the float64 ones
    d['positions'] = np.asarray(st['positions'], np.float64)
    if st['pbc'].any():
        d['cell'] = np.asarray(st['cell'], np.float64)[None]
        d['volume'] = np.array([abs(np.linalg.det(st['cell']))])
    return _to_graph(d)


def _batch(sts, extra_nodes=3, extra_edges=11):
    gs = [_graph(st) for st in sts]
    b = jraph.batch_np(gs)
    return jraph.pad_with_graphs(b, int(b.n_node.sum()) + extra_nodes, int(b.n_edge.sum()) + extra_edges, len(gs) + 1)


def test_mixed_batch_charges_forces_stress(x64):
    sts = [_structure('p', 0), _structure('m', 1, charge=1.0), _structure('p', 2, charge=-1.0), _structure('m', 3)]
    model = _model(periodic=True)
    batch = _batch(sts)
    out = model(batch)
    n = [len(s['numbers']) for s in sts]
    q = np.asarray(out['charges'])
    off = np.concatenate([[0], np.cumsum(n)])
    for g, st in enumerate(sts):                      # exact total charge of every graph; padding nodes zero
        assert abs(q[off[g]:off[g + 1]].sum() - st['info']['charge']) < 1e-10
    assert np.all(q[off[-1]:] == 0.0) and np.all(np.isfinite(np.asarray(out['energy'])))
    assert float(np.abs(q[:off[-1]]).max()) > 1e-3     # the electrostatics are active

    # every graph alone gives the same energy, forces, stress and dipole as in the batch
    for g, st in enumerate(sts):
        o1 = model(_batch([st], extra_nodes=2, extra_edges=5))
        assert abs(float(o1['energy'][0]) - float(out['energy'][g])) < 1e-8
        assert np.abs(np.asarray(o1['forces'])[:n[g]] - np.asarray(out['forces'])[off[g]:off[g + 1]]).max() < 1e-8
        assert np.abs(np.asarray(o1['dipole'])[0] - np.asarray(out['dipole'])[g]).max() < 1e-8
        if st['pbc'].all():
            assert np.abs(np.asarray(o1['stress'])[0] - np.asarray(out['stress'])[g]).max() < 1e-9

    # forces = -dE/dx (central differences on a few coordinates of every graph)
    def energy(sts2):
        return np.asarray(model(_batch(sts2))['energy'])[:len(sts2)].sum()
    h = 1e-5
    f = np.asarray(out['forces'])
    for g, i, k in [(0, 2, 0), (1, 4, 1), (2, 7, 2), (3, 0, 0)]:
        ep = []
        for sgn in (+1, -1):
            s2 = [dict(s, positions=s['positions'].copy()) for s in sts]
            s2[g]['positions'][i, k] += sgn * h
            ep.append(energy(s2))
        fd = -(ep[0] - ep[1]) / (2 * h)
        assert abs(fd - f[off[g] + i, k]) < 2e-6 * max(1.0, abs(fd)), (g, i, k, fd, f[off[g] + i, k])

    # stress of a periodic graph = (1/V) dE/d(strain) (Voigt xx, yy, zz, yz, xz, xy)
    st = sts[2]
    vol = abs(np.linalg.det(st['cell']))
    s_model = np.asarray(model(_batch([st]))['stress'])[0]
    for v, (a, b) in enumerate([(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]):
        es = []
        for sgn in (+1, -1):
            eps = np.zeros((3, 3)); eps[a, b] += 0.5 * sgn * h; eps[b, a] += 0.5 * sgn * h
            d = np.eye(3) + eps
            es.append(energy([dict(st, positions=st['positions'] @ d, cell=st['cell'] @ d)]))
        fd = (es[0] - es[1]) / (2 * h) / vol
        assert abs(fd - s_model[v]) < 2e-6 * max(1.0, abs(fd)), (v, fd, s_model[v])


def test_rotation_and_translation(x64):
    model = _model(periodic=True)
    R, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
    R = R * np.sign(np.linalg.det(R))
    t = np.array([3.1, -2.2, 0.7])
    for st in (_structure('m', 5, charge=1.0), _structure('p', 6)):
        o = model(_batch([st]))
        n = len(st['numbers'])
        s2 = dict(st, positions=(st['positions'] + t) @ R.T, cell=st['cell'] @ R.T)
        o2 = model(_batch([s2]))
        assert abs(float(o2['energy'][0]) - float(o['energy'][0])) < 1e-8
        assert np.abs(np.asarray(o2['forces'])[:n] - np.asarray(o['forces'])[:n] @ R.T).max() < 1e-8
        assert np.abs(np.asarray(o2['charges'])[:n] - np.asarray(o['charges'])[:n]).max() < 1e-9
        if not st['pbc'].any():       # dipole about the centre of mass: a vector, translation invariant also for Q != 0
            assert np.abs(np.asarray(o2['dipole'])[0] - np.asarray(o['dipole'])[0] @ R.T).max() < 1e-8
            assert float(np.abs(np.asarray(o['dipole'])[0]).max()) > 1e-4


def test_open_boundary_model_and_supercell(x64):
    # non-periodic model (no reciprocal sums at all) = the mixed model on the same molecule
    st = _structure('m', 7, charge=-1.0)
    mp, mo = _model(periodic=True), _model(periodic=False)
    mo = nnx.merge(nnx.split(mo, nnx.Param)[0], nnx.split(mp, nnx.Param)[1])       # the same parameters
    g = _graph(st)
    g_open = g._replace(edges=None, globals={k: v for k, v in g.globals.items()
                                             if k not in ('cell', 'cutoff', 'volume', 'stress', 'pbc', 'stress_w')})
    e_mixed = float(mp(_batch([st]))['energy'][0])
    b = jraph.pad_with_graphs(g_open, int(g_open.n_node.sum()) + 2, int(g_open.n_edge.sum()) + 5, 2)
    assert abs(float(mo(b)['energy'][0]) - e_mixed) < 1e-8
    # a periodic cell and its 2 x 1 x 1 supercell: twice the energy (reciprocal vectors of another cell)
    sp = _structure('p', 8)
    big = dict(sp, numbers=np.concatenate([sp['numbers']] * 2),
               positions=np.concatenate([sp['positions'], sp['positions'] + sp['cell'][0]]),
               cell=sp['cell'] * np.array([[2.0], [1.0], [1.0]]), forces=np.zeros((20, 3)))
    m2 = _model(periodic=True, polar={'max_cell_length': 18.0, 'max_cell_volume': 900.0})
    e1 = float(m2(_batch([sp]))['energy'][0])
    e2 = float(m2(_batch([big]))['energy'][0])
    assert abs(e2 - 2.0 * e1) < 1e-6 * max(1.0, abs(e1)), (e1, e2)


def test_dipole_and_charge_loss_terms(x64):
    import jax.numpy as jnp
    from race_v2.training.utils import compute_loss
    st = _structure('m', 9)
    f = dict(FIELDS, dipole=True, opt_charge_labels=True)
    st['info']['multipoles'] = [[0.0] * 3, [0.3, -0.2, 0.1], [0.0] * 6]
    st['info']['nbo_charges'] = np.linspace(-0.4, 0.4, len(st['numbers']))
    g = _to_graph(_cast(S.graph_arrays(st, IDX, np.zeros(4), CUTOFF, f, {'charge_label_key': 'nbo_charges'}), np.float64))
    assert np.allclose(g.globals['dipole'][0], np.array([0.3, -0.2, 0.1]) * S.BOHR) and g.globals['dipole_w'][0] == 1.0
    b = jraph.pad_with_graphs(g, int(g.n_node.sum()) + 2, int(g.n_edge.sum()) + 5, 2)
    model = _model(periodic=True)
    out = model(b)
    kw = dict(energy_weight=1.0, force_weight=1.0, stress_weight=0.0, huber_delta=10.0, periodic=True)
    gd, pr = nnx.split(model, nnx.Param)
    l0, _ = compute_loss(gd, pr, b, **kw)
    l1, aux = compute_loss(gd, pr, b, charge_weight=2.0, dipole_weight=3.0, **kw)
    n = len(st['numbers'])
    dq = np.asarray(out['charges'])[:n] - st['info']['nbo_charges']
    dd = np.asarray(out['dipole'])[0] - np.array([0.3, -0.2, 0.1]) * S.BOHR
    # labels are stored in float32
    assert abs(float(aux['q_loss']) - np.mean(dq ** 2)) < 1e-6
    assert abs(float(aux['d_loss']) - np.mean(dd ** 2)) < 1e-6
    assert abs(float(l1 - l0) - (2.0 * np.mean(dq ** 2) + 3.0 * np.mean(dd ** 2))) < 1e-5


# ----------------------------------------------------------------------------------------------- phase 2
P2 = {'charge_constraint': 'electrode', 'max_groups': 8, 'charge_transfer': 'contact', 'slab_correction': True}
F2 = dict(FIELDS, groups=8, slab=True, dipole=True)


def _structure2(kind, seed, charge=0.0):
    """_structure with charge groups: the first 4 atoms are the reservoir, the rest pairs; the first pair carries
    +1 (fixed charge 0.5 per atom), so the reservoir target is charge - 1."""
    st = _structure(kind, seed, charge)
    n = len(st['numbers'])
    em = np.arange(n) < 4
    cg = np.where(em, -1, (np.arange(n) - 4) // 2)
    fc = np.where(cg == 0, 0.5, 0.0).astype(np.float32)
    st['groups'] = (em, cg, fc)
    if kind == 'p':
        st['info']['dipole'] = np.array([0.3, -0.1, 0.7])
    else:
        st['info']['multipoles'] = [[0.0] * 3, [0.2, 0.1, -0.3], [0.0] * 6]
    return st


def _graph2(st):
    opts = {'slab': True, 'dipole_sign': -1.0} if st['pbc'].any() else {}
    d = _cast(S.graph_arrays(st, IDX, np.zeros(4), CUTOFF, dict(F2), opts), np.float64)
    d['positions'] = np.asarray(st['positions'], np.float64)
    if st['pbc'].any():
        d['cell'] = np.asarray(st['cell'], np.float64)[None]
        d['volume'] = np.array([abs(np.linalg.det(st['cell']))])
    return _to_graph(d)


def _batch2(sts, extra_nodes=3, extra_edges=11):
    b = jraph.batch_np([_graph2(st) for st in sts])
    return jraph.pad_with_graphs(b, int(b.n_node.sum()) + extra_nodes, int(b.n_edge.sum()) + extra_edges, len(sts) + 1)


def test_phase2_groups_contact_slab(x64):
    sts = [_structure2('p', 0), _structure2('m', 1, charge=1.0), _structure2('m', 3, charge=-1.0)]
    n = [len(s['numbers']) for s in sts]
    off = np.concatenate([[0], np.cumsum(n)])
    g0 = _graph2(sts[0])
    assert g0.globals['slab'][0] == 1 and g0.nodes['charge_group'].max() == 3 and g0.globals['dipole_w'][0] == 1
    assert np.allclose(g0.globals['dipole'][0], [-0.3, 0.1, -0.7])            # VASP sign
    assert _graph2(sts[1]).globals['slab'][0] == 0

    # without contact transfer every group has exactly its target charge
    m_fix = _model(periodic=True, polar=dict(P2, charge_transfer='none'))
    q = np.asarray(m_fix(_batch2(sts))['charges'])
    for g, st in enumerate(sts):
        qg, lg = q[off[g]:off[g + 1]], np.asarray(_graph2(st).nodes['charge_group'])
        assert abs(qg[lg == 1].sum() - 1.0) < 1e-6 and abs(qg[lg == 2].sum()) < 1e-6
        assert abs(qg[lg == 0].sum() - (st['info']['charge'] - 1.0)) < 1e-6

    # contact transfer: charge moves between groups in contact, the total charge of every graph stays exact
    model = _model(periodic=True, polar=P2)
    batch = _batch2(sts)
    out = model(batch)
    q2 = np.asarray(out['charges'])
    assert np.abs(q2 - q).max() > 1e-3
    for g, st in enumerate(sts):
        assert abs(q2[off[g]:off[g + 1]].sum() - st['info']['charge']) < 1e-6
    assert float(np.abs(np.asarray(model.node_energies(
        batch.nodes['positions'], batch.globals['cell'], batch.globals['cutoff'], batch)['polar']['node_slab'])).max()) > 1e-6

    # each graph alone = in the batch; forces and stress = finite differences with everything switched on
    def energy(sts2):
        return np.asarray(model(_batch2(sts2))['energy'])[:len(sts2)].sum()
    for g, st in enumerate(sts):
        o1 = model(_batch2([st], extra_nodes=2, extra_edges=5))
        assert abs(float(o1['energy'][0]) - float(out['energy'][g])) < 1e-8
        assert np.abs(np.asarray(o1['charges'])[:n[g]] - q2[off[g]:off[g + 1]]).max() < 1e-9
    h = 1e-5
    f = np.asarray(out['forces'])
    for g, i, k in [(0, 1, 2), (0, 6, 0), (1, 5, 1), (2, 2, 2)]:
        ep = []
        for sgn in (+1, -1):
            s2 = [dict(s, positions=s['positions'].copy()) for s in sts]
            s2[g]['positions'][i, k] += sgn * h
            ep.append(energy(s2))
        fd = -(ep[0] - ep[1]) / (2 * h)
        assert abs(fd - f[off[g] + i, k]) < 5e-6 * max(1.0, abs(fd)), (g, i, k, fd, f[off[g] + i, k])
    st = sts[0]
    vol = abs(np.linalg.det(st['cell']))
    s_model = np.asarray(out['stress'])[0]
    for v, (a, b) in enumerate([(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]):
        es = []
        for sgn in (+1, -1):
            eps = np.zeros((3, 3)); eps[a, b] += 0.5 * sgn * h; eps[b, a] += 0.5 * sgn * h
            dmat = np.eye(3) + eps
            es.append(energy([dict(st, positions=st['positions'] @ dmat, cell=st['cell'] @ dmat)]))
        fd = (es[0] - es[1]) / (2 * h) / vol
        assert abs(fd - s_model[v]) < 5e-6 * max(1.0, abs(fd)), (v, fd, s_model[v])

    # dipole loss: the slab contributes its normal component only, the molecules all three
    from race_v2.training.utils import compute_loss
    gd, pr = nnx.split(model, nnx.Param)
    kw = dict(energy_weight=1.0, force_weight=1.0, stress_weight=0.0, huber_delta=10.0, periodic=True)
    _, aux = compute_loss(gd, pr, batch, dipole_weight=1.0, **kw)
    dn = float(out['dipole_normal'][0]) - float(np.dot(np.asarray(out['slab_normal'])[0], [-0.3, 0.1, -0.7]))
    dm = [np.asarray(out['dipole'])[g] - np.array([0.2, 0.1, -0.3]) * S.BOHR for g in (1, 2)]
    ref = (dn ** 2 + np.sum(np.square(dm))) / 7.0
    assert abs(float(aux['d_loss']) - ref) < 1e-6 * max(1.0, ref)
