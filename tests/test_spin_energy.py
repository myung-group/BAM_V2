"""Spin-polarisation energy in invariant spin variables (RACE_LR long_range spin_energy, 2026-11-14):
    E_spin = sum_i [Delta_i mu_i + 1/2 W_Z mu_i^2] + sum_{i<j} J_ij(r) mu_i mu_j c_ij
mu_i >= 0 local moments (sum >= 2S; 0 on restricted singlets), c_ij in [-1, 1] pair correlations (symmetric).
Checked: zero-initialised it changes nothing; a restricted singlet gets exactly 0; c_ij = c_ji; the pair sum equals
the documented formula; a symmetric dimer represents both the parallel and the broken-symmetry arrangement
(E(c=+1) - E(c=-1) = 2 J mu^2); forces = -dE/dx; graphs of a batch are independent; the spin loss terms equal the
documented formulas.  Run in a Slurm job (JAX)."""
import numpy as np
import pytest

from conftest import ATOMIC_NUMBERS  # noqa: F401
from race_v2.training.train import COVALENT_RADII  # noqa: F401  (imports training.train before the x64 fixture)
from test_race_polar_qeq import _polar_model, _molecule, KW, IDX, FIELDS, CUTOFF, _name  # noqa: E402

SE = {'spin_head': True, 'spin_energy': True}
HEADS = ['sp_delta0', 'sp_w', 'sp_j', 'sp_delta.kernel', 'sp_mu.kernel', 'sp_c2.kernel', 'sp_head2.kernel']


def _batch(sts, extra_nodes=3, extra_edges=11):
    """as test_race_polar_qeq._batch, with the spin labels / unrestricted flag of the rows (stream_io spin_labels)"""
    import jraph
    from race_v2.data import stream_io as S
    from race_v2.data.stream_data import _to_graph, _cast
    gs = []
    for st in sts:
        d = _cast(S.graph_arrays(st, IDX, np.zeros(4), CUTOFF, dict(FIELDS, spin_labels={'key': 'lowdin_spins'})),
                  np.float64)
        d['positions'] = np.asarray(st['positions'], np.float64)
        gs.append(_to_graph(d))
    b = jraph.batch_np(gs)
    return jraph.pad_with_graphs(b, int(b.n_node.sum()) + extra_nodes, int(b.n_edge.sum()) + extra_edges, len(gs) + 1)


def _noisy(m, names=HEADS, scale=0.3, seed=0):
    import jax.numpy as jnp
    rng = np.random.default_rng(seed)
    for n in names:
        obj, attr = (n.split('.') + [None])[:2]
        p = getattr(m, obj) if attr is None else getattr(getattr(m, obj), attr)
        p.value = jnp.asarray(p.value) + jnp.asarray(rng.normal(0, scale, np.shape(p.value)))
    return m


def _mols(spins=(3, 1, 2), unres=(True, True, True), labels=False, seed=0):
    rng = np.random.default_rng(seed)
    sts = [_molecule(20 + k) for k in range(len(spins))]
    for st, s, u in zip(sts, spins, unres):
        st['info']['spin'] = s
        st['info']['unrestricted'] = u
        if labels:
            n = len(st['numbers'])
            v = rng.normal(0, 0.4, n)
            v[rng.random(n) < 0.4] = 0.0
            st['info']['lowdin_spins'] = v - (v.sum() - (s - 1)) / n
    return sts


def _jt(b):
    import jax
    import jax.numpy as jnp
    return jax.tree.map(jnp.asarray, b)


def test_options():
    with pytest.raises(ValueError):                    # needs the spin head
        _polar_model(0, lr={'spin_energy': True})
    with pytest.raises(ValueError):
        _polar_model(0, lr={**SE, 'spin_w_init': 'big'})
    with pytest.raises(ValueError):                    # switch beyond the cutoff
        _polar_model(0, lr={**SE, 'spin_pair_switch': [4.0, KW['cutoff'] + 1.0]})
    m = _polar_model(0, lr=SE)
    assert m.lr['spin_pair_switch'] == [KW['cutoff'] - 2.0, KW['cutoff']]
    w = np.asarray(m.sp_w.value)
    assert abs(w[0] - (-0.0717 * 27.211386245988)) < 1e-4 and abs(w[1] - (-0.0227 * 27.211386245988)) < 1e-4


def test_zero_init_is_the_old_model(x64):
    import jax
    from flax import nnx
    old = _noisy(_polar_model(1, lr={'spin_head': True}), ['sp_head2.kernel'])
    new = _polar_model(1, lr={**SE, 'spin_w_init': 'zero'})
    _, ps = nnx.split(old, nnx.Param)
    gd, pd = nnx.split(new, nnx.Param)
    by = {_name(p): x for p, x in jax.tree.flatten_with_path(ps)[0]}
    leaves = jax.tree.flatten_with_path(pd)[0]
    assert sum(_name(p) not in by for p, _ in leaves) >= 6
    new = nnx.merge(gd, jax.tree.unflatten(jax.tree.structure(pd), [by.get(_name(p), x) for p, x in leaves]))
    b = _jt(_batch(_mols()))
    o0, o1 = old(b), new(b)
    assert np.abs(np.asarray(o0['energy']) - np.asarray(o1['energy'])).max() < 1e-12
    assert np.abs(np.asarray(o0['forces']) - np.asarray(o1['forces'])).max() < 1e-12
    assert np.all(np.asarray(o1['spin_node_energies']) == 0.0)
    assert np.abs(np.asarray(o0['spins']) - np.asarray(o1['spins'])).max() == 0.0


def test_restricted_singlet_is_zero(x64):
    m = _noisy(_polar_model(2, lr=SE), scale=0.5)
    sts = _mols((1, 1, 3), unres=(False, True, False))     # the flag of a triplet is ignored (always unrestricted)
    b = _batch(sts)
    o = m(_jt(b))
    n = [len(s['numbers']) for s in sts]
    off = np.concatenate([[0], np.cumsum(n)])
    mu, es = np.asarray(o['spin_moments']), np.asarray(o['spin_node_energies'])
    assert np.all(mu[off[0]:off[1]] == 0.0) and np.all(es[off[0]:off[1]] == 0.0)
    assert np.abs(mu[off[1]:off[2]]).max() > 1e-3 and np.abs(es[off[1]:off[2]]).max() > 1e-6
    assert mu[off[2]:off[3]].sum() >= 2.0 - 1e-10                     # sum mu >= 2S
    assert np.all(mu[off[3]:] == 0.0)                                 # padding
    s2 = np.asarray(o['spin_s2'])
    assert abs(s2[0]) < 1e-12 and abs(s2[2] - (2.0 + 0.5 * (mu[off[2]:off[3]].sum() - 2.0))) < 1e-10


def test_pair_formula_and_symmetry(x64):
    from race_v2.models.long_range import sr_switch
    m = _noisy(_polar_model(3, lr=SE), scale=0.5)
    m.sp_delta0.value = 0 * m.sp_delta0.value; m.sp_w.value = 0 * m.sp_w.value
    m.sp_delta.kernel.value = 0 * m.sp_delta.kernel.value; m.sp_delta.bias.value = 0 * m.sp_delta.bias.value
    b = _batch(_mols((1, 3)))
    o = m(_jt(b))
    e_pair = float(np.sum(np.asarray(o['spin_node_energies'])))
    mu, c = np.asarray(o['spin_moments']), np.asarray(o['spin_corr'])
    sp, P = np.asarray(b.nodes['species']), np.asarray(b.nodes['positions'])
    T = np.asarray(m.sp_j.value); A = 0.5 * (T + T.T)
    kap = np.log1p(np.exp(np.asarray(m.sp_kappa.value))); rc = np.asarray(m.cov_r)
    r_on, r_off = m.lr['spin_pair_switch']
    rcv, snd = np.asarray(b.receivers), np.asarray(b.senders)
    import jraph
    msk = np.asarray(jraph.get_node_padding_mask(b))
    cdir = {}
    for e, (r_, s_) in enumerate(zip(rcv.tolist(), snd.tolist())):
        if msk[r_] and msk[s_] and r_ != s_:
            cdir[(r_, s_)] = c[e]
    assert all(abs(cdir[(i, j)] - cdir[(j, i)]) < 1e-12 for (i, j) in cdir)        # c_ij = c_ji
    ref = 0.0
    for (i, j), cij in cdir.items():
        if i < j:
            d = np.linalg.norm(P[i] - P[j])
            J = A[sp[i], sp[j]] * np.exp(-(kap[sp[i]] + kap[sp[j]]) * (d - rc[sp[i]] - rc[sp[j]]))
            ref += J * (1.0 - float(sr_switch(d, r_on, r_off))) * mu[i] * mu[j] * cij
    assert len(cdir) > 10 and abs(e_pair - ref) < 1e-10 * max(1.0, abs(ref)) and abs(ref) > 1e-6


def test_symmetric_dimer_broken_symmetry(x64):
    """two equivalent atoms: the network gives them the same mu, and c alone decides parallel / antiparallel"""
    import jax.numpy as jnp
    m = _polar_model(4, lr=SE)
    m.sp_mu.bias.value = jnp.asarray([1.0])                    # mu = softplus(1) on both atoms
    m.sp_j.value = jnp.full_like(m.sp_j.value, 0.2)            # J > 0: antiparallel favoured
    m.sp_w.value = 0 * m.sp_w.value
    st = {'numbers': np.array([6, 6]), 'positions': np.array([[0.0, 0.0, 0.0], [2.2, 0.0, 0.0]]),
          'cell': np.zeros((3, 3)), 'pbc': np.zeros(3, bool), 'energy': 0.0, 'forces': np.zeros((2, 3)), 'stress': None,
          'info': {'charge': 0.0, 'spin': 1, 'unrestricted': True}, 'arrays': {}}
    es = {}
    for sign in (+1.0, -1.0):
        m.sp_c2.bias.value = jnp.asarray([sign * 12.0])        # c = tanh(+-12) = +-1
        o = m(_jt(_batch([st])))
        mu = np.asarray(o['spin_moments'])[:2]
        assert abs(mu[0] - mu[1]) < 1e-12 and abs(mu[0] - np.log1p(np.e)) < 1e-12
        es[sign] = float(np.asarray(o['spin_node_energies'])[:2].sum())
    rc = np.asarray(m.cov_r)[1]
    kap = 2 * np.log1p(np.exp(float(np.asarray(m.sp_kappa.value)[1])))
    r_on, r_off = m.lr['spin_pair_switch']
    from race_v2.models.long_range import sr_switch
    J = 0.2 * np.exp(-kap * (2.2 - 2 * rc)) * (1.0 - float(sr_switch(2.2, r_on, r_off)))
    assert abs((es[1.0] - es[-1.0]) - 2 * J * np.log1p(np.e) ** 2) < 1e-9 and es[-1.0] < es[1.0]


def test_forces_and_batch_independence(x64):
    m = _noisy(_polar_model(5, lr=SE), scale=0.4)
    sts = _mols((3, 1, 2), unres=(True, True, True))
    out = m(_jt(_batch(sts)))
    assert np.abs(np.asarray(out['spin_node_energies'])).max() > 1e-4
    n = [len(s['numbers']) for s in sts]
    off = np.concatenate([[0], np.cumsum(n)])
    for g, st in enumerate(sts):
        o1 = m(_jt(_batch([st], extra_nodes=2, extra_edges=5)))
        assert abs(float(o1['energy'][0]) - float(out['energy'][g])) < 1e-8
        assert np.abs(np.asarray(o1['forces'])[:n[g]] - np.asarray(out['forces'])[off[g]:off[g + 1]]).max() < 1e-8

    def energy(s2):
        return float(np.asarray(m(_jt(_batch(s2)))['energy'])[:len(s2)].sum())
    h, f = 1e-5, np.asarray(out['forces'])
    for g, i, k in [(0, 2, 0), (1, 4, 1), (2, 7, 2), (1, 0, 1)]:
        ep = []
        for sgn in (+1, -1):
            s2 = [dict(s, positions=s['positions'].copy()) for s in sts]
            s2[g]['positions'][i, k] += sgn * h
            ep.append(energy(s2))
        fd = -(ep[0] - ep[1]) / (2 * h)
        assert abs(fd - f[off[g] + i, k]) < 1e-6 * max(1.0, abs(fd)), (g, i, k, fd, f[off[g] + i, k])


def test_spin_loss_terms(x64):
    import jraph
    from flax import nnx
    from race_v2.training.utils import compute_loss
    m = _noisy(_polar_model(6, lr=SE), scale=0.4)
    sts = _mols((1, 3, 1), unres=(True, True, False), labels=True)
    sts[2]['info'].pop('lowdin_spins')                 # a restricted singlet without labels: target 0, weight 1
    b = _jt(_batch(sts))
    gd, params = nnx.split(m, nnx.Param)
    kw = dict(energy_weight=0.0, force_weight=0.0, stress_weight=0.0, huber_delta=0.02, periodic=False)
    _, a1 = compute_loss(gd, params, b, spin_weight=1.0, **kw)
    res = nnx.merge(gd, params)(b)
    ref = np.asarray(b.nodes['ref_spins']); nw = np.asarray(jraph.get_node_padding_mask(b), float) * np.asarray(
        b.nodes['ref_spins_w'])
    mom = np.sum(nw * (np.asarray(res['spin_moments']) - np.abs(ref)) ** 2) / nw.sum()
    big = (np.abs(ref) >= 0.1) * nw
    rcv, snd = np.asarray(b.receivers), np.asarray(b.senders)
    ew = big[rcv] * big[snd] * (rcv != snd)
    cor = np.sum(ew * (np.asarray(res['spin_corr']) - np.sign(ref[rcv] * ref[snd])) ** 2) / max(ew.sum(), 1.0)
    S = np.asarray(b.globals['spin']).reshape(-1)
    ng = np.repeat(np.arange(len(S)), np.asarray(b.n_node))
    nws = nw * (S[ng] > 1.5)
    sgn = np.sum(nws * (np.asarray(res['spins']) - ref) ** 2) / nws.sum()
    assert ew.sum() > 3 and nws.sum() > 3 and np.all(np.asarray(b.globals['unrestricted']).reshape(-1)[:3] == [1, 1, 0])
    assert abs(float(a1['sp_loss']) - (mom + cor + sgn)) < 1e-9


NB = {'spin_head': True, 'spin_nbo_head': True}


def _nbo_labels(sts, seed=1):
    """NBO labels = the Lowdin labels plus a random zero-sum local change"""
    rng = np.random.default_rng(seed)
    for st in sts:
        if 'lowdin_spins' in st['info']:
            v = np.asarray(st['info']['lowdin_spins'], float)
            dlt = rng.normal(0, 0.05, len(v))
            st['info']['nbo_spins'] = v + dlt - dlt.mean()
    return sts


def test_nbo_head_identity_conservation_range(x64):
    import jraph
    m = _polar_model(7, lr=NB)
    assert m.lr['spin_nbo_switch'] == [3.0, 4.5]            # the ct switch of LR_DEFAULTS
    m = _noisy(m, ['sp_head2.kernel'], scale=0.4)
    sts = _mols((3, 1, 2))
    b = _batch(sts)
    o = m(_jt(b))
    assert np.abs(np.asarray(o['spins_nbo']) - np.asarray(o['spins'])).max() < 1e-14     # zero output layer
    m = _noisy(m, ['sp_t2.kernel', 'sp_t2.bias'], scale=0.5, seed=3)
    o = m(_jt(b))
    sn, s, t = np.asarray(o['spins_nbo']), np.asarray(o['spins']), np.asarray(o['spin_nbo_t'])
    n = [len(x['numbers']) for x in sts]
    off = np.concatenate([[0], np.cumsum(n)])
    for g in range(len(sts)):
        assert abs(sn[off[g]:off[g + 1]].sum() - s[off[g]:off[g + 1]].sum()) < 1e-12
    assert np.abs(sn - s).max() > 1e-4
    rcv, snd, P = np.asarray(b.receivers), np.asarray(b.senders), np.asarray(b.nodes['positions'])
    msk = np.asarray(jraph.get_node_padding_mask(b))
    tdir = {}
    for e, (r_, s_) in enumerate(zip(rcv.tolist(), snd.tolist())):
        if msk[r_] and msk[s_] and r_ != s_:
            tdir[(r_, s_)] = t[e]
            if np.linalg.norm(P[r_] - P[s_]) >= 4.5:
                assert t[e] == 0.0                        # nothing moves beyond the switch
    assert all(abs(tdir[(i, j)] - tdir[(j, i)]) < 1e-12 for (i, j) in tdir)


def test_nbo_loss_term(x64):
    import jraph
    from flax import nnx
    from race_v2.training.utils import compute_loss
    from race_v2.models.race_lr import RACE_LR
    m = _noisy(_polar_model(8, lr=NB), ['sp_head2.kernel', 'sp_t2.kernel', 'sp_t2.bias'], scale=0.4)
    sts = _nbo_labels(_mols((1, 3, 2), labels=True))
    sts[2]['info'].pop('nbo_spins')                   # a row without NBO labels: weight 0 for that term
    import race_v2.data.stream_io as S
    from race_v2.data.stream_data import _to_graph, _cast
    gs = []
    for st in sts:
        d = _cast(S.graph_arrays(st, IDX, np.zeros(4), CUTOFF, dict(FIELDS, spin_labels={'key': 'lowdin_spins',
                                                                                          'nbo_key': 'nbo_spins'})),
                  np.float64)
        d['positions'] = np.asarray(st['positions'], np.float64)
        gs.append(_to_graph(d))
    bb = jraph.batch_np(gs)
    b = _jt(jraph.pad_with_graphs(bb, int(bb.n_node.sum()) + 3, int(bb.n_edge.sum()) + 11, len(gs) + 1))
    gd, params = nnx.split(m, nnx.Param)
    kw = dict(energy_weight=0.0, force_weight=0.0, stress_weight=0.0, huber_delta=0.02, periodic=False)
    _, a1 = compute_loss(gd, params, b, spin_weight=1.0, **kw)
    res = nnx.merge(gd, params)(b)
    ref = np.asarray(b.nodes['ref_spins'])
    nw = np.asarray(jraph.get_node_padding_mask(b), float) * np.asarray(b.nodes['ref_spins_w'])
    signed = np.sum(nw * (np.asarray(res['spins']) - ref) ** 2) / nw.sum()
    nwn = nw * np.asarray(b.nodes['ref_spins_nbo_w'])
    pn = np.asarray(RACE_LR.spin_redistribute(b.nodes['ref_spins'], res['spin_nbo_t'], b))
    nbo = np.sum(nwn * (pn - np.asarray(b.nodes['ref_spins_nbo'])) ** 2) / nwn.sum()
    assert nwn.sum() > 5 and nwn.sum() < nw.sum() and nbo > 0
    assert abs(float(a1['sp_loss']) - (signed + nbo)) < 1e-9


def test_charge_nbo_head(x64):
    import jraph
    from flax import nnx
    from race_v2.training.utils import compute_loss
    from race_v2.models.race_lr import RACE_LR
    import race_v2.data.stream_io as S
    from race_v2.data.stream_data import _to_graph, _cast
    m = _polar_model(9, lr={'charge_nbo_head': True})
    rng = np.random.default_rng(2)
    sts = [_molecule(30), _molecule(31, charge=-1.0), _molecule(32, charge=1.0)]
    for st in sts:
        n = len(st['numbers'])
        q = rng.normal(0, 0.3, n); q = q - (q.sum() - st['info']['charge']) / n
        d = rng.normal(0, 0.1, n); d = d - d.mean()
        st['info'].update(lowdin_charges=q, nbo_charges=q + d)
    sts[2]['info'].pop('nbo_charges')
    gs = []
    for st in sts:
        dd = _cast(S.graph_arrays(st, IDX, np.zeros(4), CUTOFF, dict(FIELDS, charge_label_key='lowdin_charges',
                                                                      charge_nbo_labels={'key': 'nbo_charges'})),
                   np.float64)
        dd['positions'] = np.asarray(st['positions'], np.float64)
        gs.append(_to_graph(dd))
    bb = jraph.batch_np(gs)
    b = _jt(jraph.pad_with_graphs(bb, int(bb.n_node.sum()) + 3, int(bb.n_edge.sum()) + 11, len(gs) + 1))
    o = m(b)
    assert np.abs(np.asarray(o['charges_nbo']) - np.asarray(o['charges'])).max() < 1e-14      # zero output layer
    m = _noisy(m, ['cq_t2.kernel', 'cq_t2.bias'], scale=0.5, seed=4)
    o = m(b)
    qn, q, t = np.asarray(o['charges_nbo']), np.asarray(o['charges']), np.asarray(o['charge_nbo_t'])
    n = [len(x['numbers']) for x in sts]
    off = np.concatenate([[0], np.cumsum(n)])
    for g in range(len(sts)):                                     # total charge of every graph kept
        assert abs(qn[off[g]:off[g + 1]].sum() - q[off[g]:off[g + 1]].sum()) < 1e-12
    assert np.abs(qn - q).max() > 1e-4
    rcv, snd, P = np.asarray(b.receivers), np.asarray(b.senders), np.asarray(b.nodes['positions'])
    for e, (r_, s_) in enumerate(zip(rcv.tolist(), snd.tolist())):
        if r_ < off[-1] and s_ < off[-1] and r_ != s_ and np.linalg.norm(P[r_] - P[s_]) >= 4.5:
            assert t[e] == 0.0                                    # nothing moves beyond ct_switch[1]
    gd, params = nnx.split(m, nnx.Param)
    kw = dict(energy_weight=0.0, force_weight=0.0, stress_weight=0.0, huber_delta=0.02, periodic=False)
    _, a1 = compute_loss(gd, params, b, charge_weight=1.0, **kw)
    res = nnx.merge(gd, params)(b)
    mask = np.asarray(jraph.get_node_padding_mask(b), float)
    rl = np.asarray(b.nodes['ref_charges'])
    base = np.sum(mask * (np.asarray(res['charges']) - rl) ** 2) / mask.sum()
    wq = mask * np.asarray(b.nodes['ref_charges_nbo_w'])
    pq = np.asarray(RACE_LR.spin_redistribute(b.nodes['ref_charges'], res['charge_nbo_t'], b))
    nb = np.sum(wq * (pq - np.asarray(b.nodes['ref_charges_nbo'])) ** 2) / wq.sum()
    assert wq.sum() > 5 and wq.sum() < mask.sum() and nb > 0
    assert abs(float(a1['q_loss']) - (base + nb)) < 1e-9


def test_nbo_t_bounded(x64):
    """long_range nbo_t_max: |t| <= t_max for any raw output (spin and charge NBO heads); None: unbounded"""
    sts = _mols((3, 1, 2))
    b = _jt(_batch(sts))
    for tm, bound in ((0.5, True), (None, False)):
        m = _polar_model(9, lr={**NB, 'charge_nbo_head': True, 'nbo_t_max': tm})
        m = _noisy(m, ['sp_head2.kernel', 'sp_t2.kernel', 'sp_t2.bias', 'cq_t2.kernel', 'cq_t2.bias'], scale=50.0, seed=4)
        o = m(b)
        tmax = max(np.abs(np.asarray(o['spin_nbo_t'])).max(), np.abs(np.asarray(o['charge_nbo_t'])).max())
        if bound:
            assert tmax <= 0.5 + 1e-12 and tmax > 0.1
        else:
            assert tmax > 1.0
