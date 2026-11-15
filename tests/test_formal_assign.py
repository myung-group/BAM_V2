"""race_v2.data.formal_assign: integer formal charges per molecule from the bonds (no labels), and
graph_fields.formal_charges(auto=True) that uses them for structures of several molecules."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'race_v2', 'data'))
from formal_assign import assign_formal_charges  # noqa: E402
from graph_fields import formal_charges  # noqa: E402

Z = {'H': 1, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Na': 11, 'Mg': 12, 'P': 15, 'S': 16, 'Cl': 17, 'Fe': 26}
TET = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / np.sqrt(3.0)
OCT = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)


def _perp(d):
    p = np.cross(d, [0.0, 0.0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0.0, 0.0])
    return p / np.linalg.norm(p)


def star(c, lig, dirs, dist):
    return [(c, (0.0, 0.0, 0.0))] + [(lig, tuple(dist * u)) for u in dirs]


def water_at(o, out):
    """water with O at o and both H pointing along out (away from a cation)"""
    out = out / np.linalg.norm(out)
    p = _perp(out)
    a = np.deg2rad(52.0)
    return [('O', tuple(o)), ('H', tuple(o + 0.96 * (np.cos(a) * out + np.sin(a) * p))),
            ('H', tuple(o + 0.96 * (np.cos(a) * out - np.sin(a) * p)))]


def hydrated(ion, dirs, r):
    at = [(ion, (0.0, 0.0, 0.0))]
    for u in dirs:
        at += water_at(r * u, u)
    return at


ring = [(1.39 * np.cos(t), 1.39 * np.sin(t), 0.0) for t in np.arange(6) * np.pi / 3]
ringH = [(2.47 * np.cos(t), 2.47 * np.sin(t), 0.0) for t in np.arange(6) * np.pi / 3]
METHYL_H = [(-0.36, 1.03, 0.0), (-0.36, -0.51, 0.89), (-0.36, -0.51, -0.89)]
MOLS = {   # formula of formal_assign._formula -> (atoms, expected charge)
    'H2O': ([('O', (0, 0, 0)), ('H', (0.757, 0.586, 0)), ('H', (-0.757, 0.586, 0))], 0),
    'HO': ([('O', (0, 0, 0)), ('H', (0.97, 0, 0))], -1),
    'H3O': ([('O', (0, 0, 0))] + [('H', tuple(0.98 * u)) for u in TET[:3]], +1),
    'H4N': (star('N', 'H', TET, 1.03), +1),
    'O4S': (star('S', 'O', TET, 1.49), -2),
    'O4P': (star('P', 'O', TET, 1.54), -3),
    'ClO4': (star('Cl', 'O', TET, 1.45), -1),
    'BF4': (star('B', 'F', TET, 1.40), -1),
    'F6P': (star('P', 'F', OCT, 1.60), -1),
    'F6S': (star('S', 'F', OCT, 1.56), 0),
    'NO3': ([('N', (0, 0, 0))] + [('O', (1.25 * np.cos(t), 1.25 * np.sin(t), 0)) for t in (0, 2.094, 4.189)], -1),
    'CH3NO2': ([('C', (0, 0, 0)), ('N', (1.49, 0, 0)), ('O', (2.10, 1.06, 0)), ('O', (2.10, -1.06, 0))]
               + [('H', h) for h in METHYL_H], 0),
    'C2H3O2': ([('C', (0, 0, 0)), ('C', (1.52, 0, 0)), ('O', (2.15, 1.08, 0)), ('O', (2.15, -1.08, 0))]
               + [('H', h) for h in METHYL_H], -1),
    'CH6N': ([('C', (0, 0, 0)), ('N', (1.49, 0, 0)), ('H', (1.83, 0.97, 0)), ('H', (1.83, -0.485, 0.84)),
              ('H', (1.83, -0.485, -0.84))] + [('H', h) for h in METHYL_H], +1),
    'C6H6': ([('C', p) for p in ring] + [('H', p) for p in ringH], 0),
    'C5H6N': ([('N', ring[0])] + [('C', p) for p in ring[1:]] + [('H', (2.40, 0, 0))] + [('H', p) for p in ringH[1:]],
              +1),
    'CH6N3': ([('C', (0, 0, 0))] + sum([[('N', (1.33 * np.cos(t), 1.33 * np.sin(t), 0)),
                                         ('H', (1.33 * np.cos(t) + 1.01 * np.cos(t + 1.05), 1.33 * np.sin(t)
                                                + 1.01 * np.sin(t + 1.05), 0)),
                                         ('H', (1.33 * np.cos(t) + 1.01 * np.cos(t - 1.05), 1.33 * np.sin(t)
                                                + 1.01 * np.sin(t - 1.05), 0))] for t in (0, 2.094, 4.189)], []),
              +1),
    'H8NaO4': (hydrated('Na', TET, 2.40), +1),
    'Cl': ([('Cl', (0, 0, 0))], -1),
    'Mg': ([('Mg', (0, 0, 0))], +2),
}


def assemble(names, spacing=14.0):
    zs, ps = [], []
    for k, name in enumerate(names):
        atoms = MOLS[name][0]
        off = spacing * np.array([k % 5, (k // 5) % 5, k // 25], float)
        zs += [Z[s] for s, _ in atoms]
        ps += [np.asarray(p, float) + off for _, p in atoms]
    return np.array(zs), np.array(ps)


def pairs(pos, cutoff=6.0):
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    i, j = np.nonzero((d < cutoff) & (d > 0))
    return i, j


@pytest.mark.parametrize('name', sorted(MOLS))
def test_single_molecule_rules(name):
    """each molecule alone, the rules only (no total charge to balance towards)"""
    z, p = assemble([name])
    s, d = pairs(p)
    q, rep = assign_formal_charges(z, p, s, d, None)
    assert int(q.sum()) == MOLS[name][1], (name, q, rep)
    assert rep['molecules'][0]['formula'] == name


def test_all_molecules_together():
    names = sorted(MOLS)
    z, p = assemble(names)
    s, d = pairs(p)
    Q = sum(MOLS[n][1] for n in names)
    q, rep = assign_formal_charges(z, p, s, d, Q)
    got = {m['formula']: m['charge'] for m in rep['molecules']}
    for n in names:
        assert got[n] == MOLS[n][1], (n, got[n])
    assert not any(m['flags'] for m in rep['molecules']) and not rep['flags'], rep


def test_salt_in_water_q0():
    """NaCl in water, Q = 0: auto gives Na(H2O)4 +1 and Cl -1; the old rule (auto False) makes every molecule
    neutral"""
    z, p = assemble(['H8NaO4', 'Cl', 'H2O', 'H2O'], spacing=9.0)
    s, d = pairs(p)
    q0, src = formal_charges(z, p, s, d, 0.0, auto=True)
    assert src == 'auto'
    assert abs(q0[:13].sum() - 1.0) < 1e-9 and abs(q0[13] + 1.0) < 1e-9 and abs(q0[14:].sum()) < 1e-9
    q0_old, src_old = formal_charges(z, p, s, d, 0.0)
    assert src_old == 'neutral' and np.all(q0_old == 0)
    q0_user, src_user = formal_charges(z, p, s, d, 0.0, arrays={'formal_charges': np.zeros(len(z))}, auto=True)
    assert src_user == 'given' and np.all(q0_user == 0)


def test_metal_complex_and_balance():
    """Fe(H2O)6 with Q = +3: the default Fe(II) moves to Fe(III) and the molecule is flagged; a lone H with Q = +1
    becomes a proton"""
    fe = hydrated('Fe', OCT, 2.10)
    MOLS['H12FeO6'] = (fe, 3)
    try:
        z, p = assemble(['H12FeO6', 'H2O'])
        s, d = pairs(p)
        q, rep = assign_formal_charges(z, p, s, d, 3)
        m = {x['formula']: x for x in rep['molecules']}
        assert m['H12FeO6']['charge'] == 3 and 'adjusted to the total charge' in m['H12FeO6']['flags']
    finally:
        MOLS.pop('H12FeO6')
    z, p = assemble(['C6H6'])
    z, p = np.r_[z, 1], np.r_[p, [[20.0, 0, 0]]]
    s, d = pairs(p)
    q, rep = assign_formal_charges(z, p, s, d, 1)
    assert q[-1] == 1 and int(q.sum()) == 1


def test_cache_returns_copies():
    z, p = assemble(['C2H3O2', 'CH6N'])
    s, d = pairs(p)
    q1, _ = assign_formal_charges(z, p, s, d, 0)
    q1[:] = 99
    q2, _ = assign_formal_charges(z, p + 0.01, s, d, 0)
    assert int(q2.sum()) == 0 and q2.max() < 99
