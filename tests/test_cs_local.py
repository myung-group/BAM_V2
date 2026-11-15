"""Local charge / spin conditioning (RACE_LR long_range cs_local, 2026-11-14):
    Qbar_i = sum_j s(r_ij) q0_j,   2Sbar_i = sum_j s(r_ij) s0_j   replace the graph totals Q, 2S of the conditioning.
Checked: a compact structure (all atoms closer than r_on) gives exactly the model without cs_local; two separated
fragments give exactly the sum of the fragments (the graph-total conditioning does not); the energy is continuous
where a molecule splits (graph_fields.molecules) and smooth in the switch region (forces = -dE/dx, also with the
spin energy); formal_spins rules; the stream fields carry s0.  Run in a Slurm job (JAX)."""
import numpy as np
import pytest

from conftest import ATOMIC_NUMBERS  # noqa: F401
from bam_omat24.train_3bpa import COVALENT_RADII  # noqa: F401  (imports train_3bpa before the x64 fixture)
from test_race_polar_qeq import _polar_model, _perturbed, _with_params_of, KW, IDX, FIELDS, CUTOFF  # noqa: E402

CS = {'cs_encoding': 'table', 'cs_layers': 'all', 'cs_species_gate': True}
LOCAL = [4.0, 5.0]
FC = {'label_keys': ['nbo_charges', 'lowdin_charges'], 'bond_factor': 1.2, 'spins': {'label_key': 'lowdin_spins'}}


def _st(numbers, pos, charge=0, spin=1, arrays=None):
    pos = np.asarray(pos, np.float64)
    return {'numbers': np.asarray(numbers), 'positions': pos, 'cell': np.zeros((3, 3)), 'pbc': np.zeros(3, bool),
            'energy': 0.0, 'forces': np.zeros((len(pos), 3)), 'stress': None,
            'info': {'charge': charge, 'spin': spin}, 'arrays': dict(arrays or {})}


def _batch(sts, fields=None, extra_nodes=3, extra_edges=11):
    import jraph
    from bam_omat24 import stream_io as S
    from bam_omat24.stream_data import _to_graph, _cast
    gs = []
    for st in sts:
        d = _cast(S.graph_arrays(st, IDX, np.zeros(4), CUTOFF, dict(fields or FIELDS)), np.float64)
        d['positions'] = np.asarray(st['positions'], np.float64)
        gs.append(_to_graph(d))
    b = jraph.batch_np(gs)
    return jraph.pad_with_graphs(b, int(b.n_node.sum()) + extra_nodes, int(b.n_edge.sum()) + extra_edges, len(gs) + 1)


def _jt(b):
    import jax
    import jax.numpy as jnp
    return jax.tree.map(jnp.asarray, b)


def _lr_model(seed, local=True):
    """RACE_LR with the conditioning only (no Qeq stratum: no Coulomb / polarisation between fragments), cs params noisy"""
    from flax import nnx
    from bam_omat24.race_lr_nnx import RACE_LR
    lr = {'qeq': False, 'charge_spin_embedding': True, 'n_tasks': 2, **CS, 'cs_local': LOCAL if local else None}
    return _perturbed(RACE_LR(**KW, rngs=nnx.Rngs(seed), long_range=lr), seed, noise=0.5, parts=('cs_', 'task'))


def _compact(seed, n=5):
    rng = np.random.default_rng(seed)
    z = rng.choice(ATOMIC_NUMBERS, n)
    v = rng.normal(0, 1, (n, 3))
    pos = 1.3 * v / np.linalg.norm(v, axis=1, keepdims=True) * rng.uniform(0.4, 1.0, (n, 1))   # inside r = 1.3 A
    return z, pos


def test_options():
    from flax import nnx
    from bam_omat24.race_lr_nnx import RACE_LR
    with pytest.raises(ValueError):                    # needs the charge / spin embedding
        RACE_LR(**KW, rngs=nnx.Rngs(0), long_range={'qeq': True, 'cs_local': LOCAL})
    with pytest.raises(ValueError):                    # beyond the cutoff
        RACE_LR(**KW, rngs=nnx.Rngs(0), long_range={'charge_spin_embedding': True, 'cs_local': [4.0, CUTOFF + 1]})


def test_compact_structure_is_the_global_model(x64):
    """all atoms closer than r_on: Qbar = Q, 2Sbar = 2S, so cs_local changes nothing (also the polar model with the
    spin energy, so the spin pair features are covered)"""
    sts = []
    for k, (q, s) in enumerate([(0, 1), (1, 2), (-1, 3), (2, 1)]):
        z, pos = _compact(10 + k)
        sts.append(_st(z, pos, q, s))
    new = _lr_model(1, local=True)
    old = _with_params_of(new, _lr_model(1, local=False))
    b = _jt(_batch(sts))                     # no q0 / s0 nodes: the graph totals spread evenly
    o0, o1 = old(b), new(b)
    assert np.abs(np.asarray(o0['energy']) - np.asarray(o1['energy'])).max() < 1e-10
    assert np.abs(np.asarray(o0['forces']) - np.asarray(o1['forces'])).max() < 1e-10
    se = {'spin_head': True, 'spin_energy': True, **CS}
    newp = _perturbed(_polar_model(2, lr={**se, 'cs_local': LOCAL}), 2, noise=0.4, parts=('cs_', 'sp_'))
    oldp = _with_params_of(newp, _polar_model(2, lr=se))
    for st in sts:
        st['info']['unrestricted'] = True
    b = _jt(_batch(sts))
    o0, o1 = oldp(b), newp(b)
    assert np.abs(np.asarray(o1['spin_node_energies'])).max() > 1e-6
    assert np.abs(np.asarray(o0['energy']) - np.asarray(o1['energy'])).max() < 1e-10
    assert np.abs(np.asarray(o0['forces']) - np.asarray(o1['forces'])).max() < 1e-10


def _fragments(d=20.0):
    """A = CH N cation (2S = 1), B = OH2 anion (2S = 1), B shifted by d along x"""
    za, pa = [6, 1, 7], [[0, 0, 0], [-1.05, 0, 0], [0.3, 1.25, 0]]
    zb, pb = [8, 1, 1], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]
    A = _st(za, pa, 1, 2)
    B = _st(zb, pb, -1, 2)
    pd = np.asarray(pb, float) + [d, 0, 0]
    AB = _st(za + zb, np.concatenate([pa, pd]), 0, 1,
             arrays={'formal_charges': np.array([1.0, 0, 0, -1.0, 0, 0]),
                     'formal_spins': np.array([1.0, 0, 0, 1.0, 0, 0])})
    return A, B, AB


def test_separated_fragments_are_the_sum(x64):
    fields = dict(FIELDS, formal_charges=FC)
    A, B, AB = _fragments(20.0)
    for local in (True, False):
        m = _lr_model(3, local=local)
        e = np.asarray(m(_jt(_batch([A, B, AB], fields)))['energy'])[:3]
        gap = abs(float(e[2] - e[0] - e[1]))
        if local:
            assert gap < 1e-10, gap
        else:                         # the graph totals (Q = 0, 2S = 0) differ from those of the fragments
            assert gap > 1e-4, gap


def test_continuous_where_a_molecule_splits(x64):
    """H-C ... O-H (CH+ and OH-, given per-atom formal charges): the molecules split at C-O = 1.2 (0.76 + 0.66) A =
    1.704 A; q0 jumps there (0 -> +-1/2 per atom), Qbar does not (all atoms within r_on)"""
    m = _lr_model(4, local=True)
    fields = dict(FIELDS, formal_charges=FC)

    def st(d):
        pos = [[-1.1, 0, 0], [0, 0, 0], [d, 0.05, 0], [d + 0.97, 0.1, 0]]
        return _st([1, 6, 8, 1], pos, 0, 1, arrays={'formal_charges': np.array([1.0, 0, -1.0, 0]),
                                                     'formal_spins': np.zeros(4)})
    dc, eps = 1.2 * (0.76 + 0.66), 1e-4
    o = m(_jt(_batch([st(dc - eps), st(dc + eps)], fields)))
    e = np.asarray(o['energy'])[:2]
    f = np.asarray(o['forces'])
    fx_o = 0.5 * (f[2, 0] + f[3, 0] + f[6, 0] + f[7, 0])         # force on the moving O-H along x (both points)
    assert abs((e[1] - e[0]) + fx_o * 2 * eps) < 1e-6, (e, fx_o)


def test_forces_in_the_switch_region(x64):
    """finite differences of the energy against the forces, fragments in the cs_local switch region (4 - 5 A), polar
    model with the spin energy (local S_z in the pair features)"""
    se = {'spin_head': True, 'spin_energy': True, **CS, 'cs_local': LOCAL}
    m = _perturbed(_polar_model(5, lr=se), 5, noise=0.4, parts=('cs_', 'sp_'))
    fields = dict(FIELDS, formal_charges=FC)
    _, _, AB = _fragments(4.4)
    AB['info']['spin'] = 3
    AB['info']['unrestricted'] = True
    out = m(_jt(_batch([AB], fields)))
    f = np.asarray(out['forces'])

    def energy(pos):
        s = dict(AB, positions=pos)
        return float(np.asarray(m(_jt(_batch([s], fields)))['energy'])[0])
    h = 1e-5
    for i, k in [(0, 0), (3, 0), (4, 1), (2, 2), (5, 0)]:
        ep = []
        for sgn in (+1, -1):
            p = AB['positions'].copy()
            p[i, k] += sgn * h
            ep.append(energy(p))
        fd = -(ep[0] - ep[1]) / (2 * h)
        assert abs(fd - f[i, k]) < 1e-6 * max(1.0, abs(fd)), (i, k, fd, f[i, k])


def test_formal_spins_rules():
    from bam_omat24.graph_fields import formal_spins
    from ase import Atoms
    from ase.neighborlist import neighbor_list

    def pairs(z, pos):
        a = Atoms(z, positions=pos)
        i, j = neighbor_list('ij', a, CUTOFF)
        return i, j
    # one molecule: the multiplicity
    z, pos = [6, 1, 1], [[0, 0, 0], [1.09, 0, 0], [-0.3, 1.05, 0]]
    i, j = pairs(z, pos)
    s0, src = formal_spins(z, np.asarray(pos, float), i, j, 3, np.zeros(3))
    assert src == 'single molecule' and abs(s0.sum() - 2.0) < 1e-12
    # two separated radicals (CH3 ... CH3) of an overall singlet: 2S_m = 1 each (parity rule)
    z = [6, 1, 1, 1] * 2
    m1 = np.array([[0, 0, 0], [1.09, 0, 0], [-0.36, 1.03, 0], [-0.36, -0.51, 0.89]])
    pos = np.concatenate([m1, m1 + [4.5, 0, 0]])
    i, j = pairs(z, pos)
    s0, src = formal_spins(z, pos, i, j, 1, np.zeros(8))
    assert src == 'parity' and abs(s0[:4].sum() - 1) < 1e-12 and abs(s0[4:].sum() - 1) < 1e-12
    # labels: a triplet carbene CH2 next to water, Lowdin spins 1.9 on C and 0.05 on each H, 0 on water
    z = [6, 1, 1, 8, 1, 1]
    pos = np.array([[0, 0, 0], [1.08, 0, 0], [-0.4, 1.0, 0], [4.0, 0, 0], [4.96, 0, 0], [3.76, 0.93, 0]])
    i, j = pairs(z, pos)
    s0, src = formal_spins(z, pos, i, j, 3, np.zeros(6), info={'lowdin_spins': [1.9, 0.05, 0.05, 0, 0, 0]})
    assert src == 'lowdin_spins' and abs(s0[:3].sum() - 2) < 1e-12 and abs(s0[3:].sum()) < 1e-12
    # the rest of 2S on the largest molecule (no labels): quintet with CH2 (even) and water (even)
    s0, src = formal_spins(z, pos, i, j, 5, np.zeros(6))
    assert src == 'parity' and abs(s0.sum() - 4) < 1e-12
    # given per-atom spins win
    s0, src = formal_spins(z, pos, i, j, 3, np.zeros(6), arrays={'formal_spins': np.array([2.0, 0, 0, 0, 0, 0])})
    assert src == 'given' and abs(s0[:3].sum() - 2) < 1e-12


def test_stream_fields_carry_s0():
    from bam_omat24 import stream_io as S
    _, _, AB = _fragments(20.0)
    d = S.graph_arrays(AB, IDX, np.zeros(4), CUTOFF, dict(FIELDS, formal_charges=FC))
    assert np.allclose(d['n:q0'], [1 / 3] * 3 + [-1 / 3] * 3)
    assert np.allclose(d['n:s0'], [1 / 3] * 6)
