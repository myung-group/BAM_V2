"""Integer formal charges per atom from the geometry alone (no charge labels), for the per-molecule reference charges q0
of RACE_LR (graph_fields.formal_charges with auto=True: a charged or salt-containing structure of several molecules
without labels).  Only the sum per molecule enters q0; the per-atom split is a by-product.

    q, report = assign_formal_charges(numbers, positions, src, dst, total_charge, shift_vectors=None)

Rules (numpy + networkx, no RDKit):
  bonds         d_ij < bond_factor (r_cov,i + r_cov,j), the same bonds as graph_fields.molecules().  A hydrogen
                bonded to several atoms keeps its nearest non-metal partner only (short hydrogen bonds).
  metals        every element outside the main-group non-metals.  Metal-ligand bonds are cut before the Lewis
                analysis of the ligands; the metal gets its common ion charge (alkali +1, alkaline earth +2, Al +3,
                Zn +2, ...), a transition metal a default oxidation state that may move to match the total charge.
  valence       each non-metal takes the smallest neutral valence >= its degree (P 3/5, S 2/4/6, halogens 1/3/5/7);
                above the largest: N, O, P, S, As, Se with one bond more -> onium (+1); otherwise "ate" (-(d - V):
                BF4-, PF6-, SiF6 2-).
  pi bonds      the unsaturations V - d are paired along the bonds short enough for a pi bond (PI_FACTOR; a biaryl
                or N-methyl single bond is not) by a maximum matching of the unit copies (forced leaves first, the
                rest by networkx, shorter bonds preferred).
  hypervalent   P, S, Se, As, halogens raise the valence by 2 while that pairs more units (sulfate -2, perchlorate -1,
                phosphate -3, sulfone 0).
  onium         an N / O / P / S / As / Se at its neutral valence becomes onium (+1, one more pi bond) when it is
                bonded to two unpaired heteroatoms (nitro 0, nitrate -1), or when that pairs an unpaired carbon
                (pyridinium, imidazolium, guanidinium, iminium +1).
  leftovers     an unpaired unit on N, O, S, Se, halogens is a lone pair (-1); on P / S / As / Se a missing bond
                (phosphonium, sulfonium +1); on a carbon bonded to a metal a carbanion ligand (-1); on any other
                carbon ambiguous (radical, 0 by default).
  total charge  the ambiguous centres move so that the sum is Q: a lone pair may be a radical (OH., NO2., Cl.), an
                unpaired carbon a cation or anion, an isolated H a proton or hydride, a metal another oxidation
                state.  Among the solutions the number of odd-electron molecules closest to the target wins (0 for a
                singlet or with transition metals, else multiplicity - 1), then the fewest changes (_balance, exact).
                Without any solution the rest goes to the largest molecule, with a flag.  A gas-phase homolysis
                singlet (two radicals) comes out as an ion pair: the rules prefer the solution chemistry.

A user value (atoms.arrays['formal_charges']) or charge labels win over this module (graph_fields.formal_charges).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

try:
    from graph_fields import _COV_RADII, molecules
except ImportError:                                   # imported as race_v2.data.formal_assign
    from race_v2.data.graph_fields import _COV_RADII, molecules

# ------------------------------------------------------------------------------------------------ element tables
NONMETAL = {1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 33, 34, 35, 36, 52, 53, 54, 85, 86}
NOBLE = {2, 10, 18, 36, 54, 86}
VALENCE = {1: (1,), 5: (3,), 6: (4,), 7: (3,), 8: (2,), 9: (1,), 14: (4,), 15: (3, 5), 16: (2, 4, 6),
           17: (1, 3, 5, 7), 33: (3, 5), 34: (2, 4, 6), 35: (1, 3, 5, 7), 52: (2, 4, 6), 53: (1, 3, 5, 7),
           85: (1, 3, 5, 7)}
ONIUM = {7, 8, 15, 16, 33, 34}                         # +1 with one more bond
LONE_PAIR = {7, 8, 9, 16, 17, 34, 35, 52, 53, 85}      # an unpaired unit is a lone pair: -1
ISOLATED = {9: -1, 17: -1, 35: -1, 53: -1, 85: -1}     # single non-metal atoms (others 0; H ambiguous)
FIXED_METAL = {3: 1, 11: 1, 19: 1, 37: 1, 55: 1, 87: 1, 4: 2, 12: 2, 20: 2, 38: 2, 56: 2, 88: 2,
               13: 3, 30: 2, 48: 2, 47: 1, 31: 3, 49: 3, 57: 3, 39: 3, 21: 3, 81: 1, 82: 2, 50: 2, 83: 3}
ODD_WEIGHT = 0.4        # _balance: weight of the odd-electron count against the cost of the changes
SHORT_FACTOR = 0.88     # an onium double bond to a lone-pair atom: d < 0.88 (r_cov,i + r_cov,j) (N=O+ of TEMPO+)
PI_FACTOR = 0.97       # a bond can carry pi bonds when d < 0.97 (r_cov,i + r_cov,j): C-C < 1.47 A (benzene 1.39
#                         yes, biaryl 1.48 no), C-O < 1.38 A, N-O < 1.33 A, S-O < 1.66 A
DEFAULT_TM = {22: 4, 23: 3, 24: 3, 25: 2, 26: 2, 27: 2, 28: 2, 29: 2, 40: 4, 41: 5, 42: 4, 43: 4, 44: 2, 45: 3,
              46: 2, 72: 4, 73: 5, 74: 4, 75: 4, 76: 2, 77: 3, 78: 2, 79: 1, 80: 2}


def _formula(zs):
    from ase.data import chemical_symbols
    c = Counter(chemical_symbols[int(x)] for x in zs)
    keys = [s for s in ('C', 'H') if s in c] + sorted(s for s in c if s not in ('C', 'H'))
    return ''.join(s + (str(c[s]) if c[s] > 1 else '') for s in keys)


# ------------------------------------------------------------------------------------------------ matching
def _pair_units(u: dict, nbr, weight=None):
    """Pair the units u[i] (atom -> count, one molecule) along the pi-capable bonds nbr, at most two pi bonds per
    bond; u is reduced in place to the unpaired units.  weight[(i, j)] (i < j): preference among maximum matchings
    (shorter bonds relative to the single-bond length)."""
    pi = Counter()

    def partners(i):
        return [j for j in nbr[i] if u.get(j, 0) > 0 and pi[(min(i, j), max(i, j))] < 2]

    todo = [i for i in u if u[i] > 0]
    while todo:                                   # forced: an atom whose units can go to one neighbour only
        nxt = []
        for i in todo:
            if u[i] <= 0:
                continue
            p = partners(i)
            if len(p) == 1:
                j = p[0]
                key = (min(i, j), max(i, j))
                k = min(u[i], u[j], 2 - pi[key])
                u[i] -= k
                u[j] -= k
                pi[key] += k
                nxt += [x for x in nbr[j] if u.get(x, 0) > 0] + ([i] if u[i] > 0 else [])
        todo = list(dict.fromkeys(nxt))
    rest = {i for i in u if u[i] > 0 and partners(i)}
    if rest:
        import networkx as nx
        g = nx.Graph()
        for i in rest:
            for j in nbr[i]:
                if j > i and j in rest:
                    w = 1.0 if weight is None else weight[(i, j)]
                    for a in range(u[i]):
                        for b in range(u[j]):
                            g.add_edge((i, a), (j, b), weight=w)
        for (i, _), (j, _) in nx.max_weight_matching(g, maxcardinality=True):
            u[i] -= 1
            u[j] -= 1


# ------------------------------------------------------------------------------------------------ main
def assign_formal_charges(numbers, positions, src, dst, total_charge=0.0, shift_vectors=None,
                          bond_factor: float = 1.2, multiplicity=None):
    """Integer formal charge per atom (N,) and a report {'molecules': [...], 'flags': [...]} (module docstring).
    total_charge None: the rules alone, without moving the ambiguous centres towards a total charge.
    multiplicity (2S + 1 of the whole structure): with several molecules and no transition metal, the balancing
    prefers multiplicity - 1 odd-electron molecules (radicals); a singlet prefers closed shells (ions)."""
    z = np.asarray(numbers, np.int64)
    n = len(z)
    if n == 0:
        return np.zeros(0, np.int64), {'molecules': [], 'flags': []}
    P = np.asarray(positions, np.float64)
    src, dst = np.asarray(src, np.int64), np.asarray(dst, np.int64)
    d = P[dst] - P[src]
    if shift_vectors is not None:
        d = d + np.asarray(shift_vectors, np.float64)
    r = np.linalg.norm(d, axis=-1)
    rc = np.array([_COV_RADII.get(int(x), 1.5) for x in z])
    bond = (r < bond_factor * (rc[src] + rc[dst])) & (src != dst)
    bs, bd, br = src[bond], dst[bond], r[bond]
    # MD calls this every step with the same bonds: the result depends on the bond graph and Q only
    key = (z.tobytes(), np.unique(np.stack([np.minimum(bs, bd), np.maximum(bs, bd)], 1), axis=0).tobytes(),
           None if total_charge is None else int(round(float(total_charge))),
           None if multiplicity is None else int(round(float(multiplicity))))
    if key in _CACHE:
        q, rep = _CACHE[key]
        return q.copy(), rep
    out = _assign(z, P, src, dst, bs, bd, br, total_charge, shift_vectors, bond_factor, multiplicity)
    if len(_CACHE) >= 32:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = out
    return out[0].copy(), out[1]


_CACHE = {}


def _assign(z, P, src, dst, bs, bd, br, total_charge, shift_vectors, bond_factor, multiplicity):
    n = len(z)
    lab, k = molecules(z, P, src, dst, shift_vectors, bond_factor)

    metal = np.array([int(x) not in NONMETAL for x in z])
    pairs = {}
    for a, b, rr in zip(bs, bd, br):
        if not (metal[a] or metal[b]):
            key = (int(min(a, b)), int(max(a, b)))
            pairs[key] = min(rr, pairs.get(key, np.inf))
    best_h = {}
    for (a, b), rr in pairs.items():
        for h, o in ((a, b), (b, a)):
            if z[h] == 1 and (h not in best_h or rr < best_h[h][1]):
                best_h[h] = (o, rr)
    nbr = [set() for _ in range(n)]
    nbr_pi = [set() for _ in range(n)]      # bonds short enough for a pi bond: d < PI_FACTOR (r_cov,i + r_cov,j)
    weight = {}
    for (a, b), rr in pairs.items():
        if (z[a] == 1 and best_h[a][0] != b) or (z[b] == 1 and best_h[b][0] != a):
            continue
        nbr[a].add(b)
        nbr[b].add(a)
        ref = _COV_RADII.get(int(z[a]), 1.5) + _COV_RADII.get(int(z[b]), 1.5)
        if rr < PI_FACTOR * ref:
            nbr_pi[a].add(b)
            nbr_pi[b].add(a)
            # shorter bonds first; between equal ones keep the lone pair (anion) on the heteroatom
            weight[(a, b)] = ref / rr + (0.02 if z[a] not in LONE_PAIR and z[b] not in LONE_PAIR else 0.0)
    near_metal = np.zeros(n, bool)
    for a, b in zip(bs, bd):
        if metal[a] != metal[b]:
            near_metal[b if metal[a] else a] = True
    deg = np.array([len(s) for s in nbr])

    q = np.zeros(n, np.int64)
    V = np.zeros(n, np.int64)
    amb = []                # (atom, [(delta, cost), ...]): centres that may move towards the total charge
    metal_amb = []          # (atom, kind) of the transition metals (oxidation state assumed)
    lewis = np.zeros(n, bool)             # atoms of the Lewis analysis
    for i in range(n):
        zi = int(z[i])
        if metal[i]:
            if zi in FIXED_METAL:
                q[i] = FIXED_METAL[zi]
                amb.append((i, [(-k_, 1.5 * k_) for k_ in range(1, int(q[i]) + 1)]))     # low-valent (Al+, Mg+)
            else:
                q[i] = DEFAULT_TM.get(zi, 2)
                amb.append((i, [(s * k_, 1.0 * k_) for k_ in (1, 2) for s in (1, -1) if q[i] + s * k_ >= 0]))
                metal_amb.append((i, 'metal'))
            continue
        if zi in NOBLE:
            continue
        if deg[i] == 0 and not near_metal[i]:
            q[i] = ISOLATED.get(zi, 0)
            if zi == 1:
                amb.append((i, [(1, 2.0), (-1, 2.5)]))      # H. default (a stretched bond); H+ / H- rarely
            elif q[i] < 0:
                amb.append((i, [(1, 1.0)]))                                              # halogen atom
            continue
        lewis[i] = True
        opts = VALENCE.get(zi, (int(deg[i]),))
        ok = [v for v in opts if v >= deg[i]]
        if ok:
            V[i] = ok[0]
        elif zi in ONIUM and len(opts) == 1 and deg[i] == opts[0] + 1:      # NR4+, H3O+ (PF6-, SF6: not onium)
            V[i], q[i] = deg[i], 1
        else:
            V[i], q[i] = deg[i], -(int(deg[i]) - max(opts))

    for m in range(k):
        atoms = [int(i) for i in np.flatnonzero((lab == m) & lewis)]
        if not atoms:
            continue

        def solve():
            u = {i: int(V[i] - deg[i]) for i in atoms}
            _pair_units(u, nbr_pi, weight)
            return u

        def n_un(u, carbon=False):
            return sum(v for i, v in u.items() if v > 0 and (not carbon or z[i] == 6))

        u = solve()
        # hypervalent: raise the valence by 2 while that pairs more units
        for i in atoms:
            if q[i] != 0 or len(VALENCE.get(int(z[i]), ())) < 2:
                continue
            while u[i] == 0:
                higher = [v for v in VALENCE[int(z[i])] if v > V[i]]
                if not higher or sum(1 for j in nbr_pi[i] if u.get(j, 0) > 0) < higher[0] - V[i]:
                    break
                old = V[i]
                V[i] = higher[0]
                u2 = solve()
                if n_un(u2) < n_un(u):
                    u = u2
                else:
                    V[i] = old
                    break
        # onium next to two unpaired heteroatoms (nitro, nitrate), or to one through a very short bond (oxoammonium)
        for i in atoms:
            if int(z[i]) in ONIUM and q[i] == 0 and V[i] == deg[i] and u[i] == 0:
                free = [j for j in nbr_pi[i] if u.get(j, 0) > 0 and z[j] in LONE_PAIR]
                short = [j for j in free if weight[(min(i, j), max(i, j))] > 1.0 / SHORT_FACTOR]
                if len(free) >= 2 or short:
                    V[i], q[i] = V[i] + 1, 1
                    u = solve()
        # onium that pairs an unpaired carbon (pyridinium, guanidinium, iminium)
        while n_un(u, carbon=True) > 0:
            # at the neutral valence (pyridinium N), or a carbonyl-type O next to an unpaired carbon (C#O+, acylium)
            cand = [i for i in atoms if int(z[i]) in ONIUM and q[i] == 0 and V[i] <= VALENCE[int(z[i])][0]
                    and any(V[j] > deg[j] for j in nbr_pi[i])
                    and (V[i] == deg[i] or any(u.get(j, 0) > 0 and z[j] == 6 for j in nbr_pi[i]))]
            done = False
            for i in cand:
                V[i], q[i] = V[i] + 1, 1
                u2 = solve()
                if n_un(u2, carbon=True) < n_un(u, carbon=True):
                    u, done = u2, True
                    break
                V[i], q[i] = V[i] - 1, 0
            if not done:
                break
        for i in atoms:
            if u[i] <= 0:
                continue
            zi = int(z[i])
            uu = int(u[i])
            if zi in LONE_PAIR and not (zi in ONIUM and len(VALENCE.get(zi, ())) >= 2 and uu == 1
                                        and V[i] > VALENCE[zi][0]):
                q[i] -= uu                        # lone pair (anion); or a radical
                amb.append((i, [(k_, 1.0 * k_) for k_ in range(1, uu + 1)]))
            elif zi in ONIUM:                     # P / S / As / Se one bond short of the valence chosen
                q[i] += 1
                amb.append((i, [(-1, 1.0)]))
            elif zi == 6 and near_metal[i]:
                q[i] -= uu                        # carbanion ligand (Cp-, aryl, CN-)
                amb.append((i, [(k_, 1.0 * k_) for k_ in range(1, uu + 1)]))
            else:                                 # radical by default; carbocation / carbanion
                amb.append((i, [(s * k_, 1.0 * k_) for k_ in range(1, min(uu, 2) + 1) for s in (1, -1)]))

    flags = []
    moved = {}
    if total_charge is not None:              # None: the rules alone (diagnostics)
        moved = _balance(z, q, lab, k, amb, int(round(float(total_charge))), multiplicity, flags)
    amb_m = {i for i, kind in metal_amb}
    mols = []
    for m in range(k):
        at = sorted(np.flatnonzero(lab == m).tolist())
        f = []
        if any(i in moved for i in at):
            f.append('adjusted to the total charge')
        if amb_m & set(at):
            f.append('metal oxidation state assumed')
        qm = int(q[at].sum())
        if (int(z[at].sum()) - qm) % 2:
            f.append('odd electron count (radical)')
        mols.append({'molecule': m, 'n_atoms': len(at), 'formula': _formula(z[at]), 'charge': qm, 'flags': f})
    return q, {'molecules': mols, 'flags': flags}


def _balance(z, q, lab, k, centres, Q, multiplicity, flags):
    """Move the ambiguous centres (atom, [(delta, cost), ...]) so that the charges sum to Q, preferring (1) the
    number of odd-electron molecules closest to the target (0 for a singlet or when transition metals can carry
    the spin; else multiplicity - 1), (2) the smallest total cost.  Exact over molecules (knapsack per molecule, then
    a dynamic programme over the molecules).  Changes q in place; returns {atom: delta}."""
    D = Q - int(q.sum())
    elec = np.bincount(lab, weights=z.astype(np.float64), minlength=k) - np.bincount(lab, weights=q, minlength=k)
    elec = np.round(elec).astype(np.int64)
    has_tm = any(int(x) in DEFAULT_TM or (int(x) not in NONMETAL and int(x) not in FIXED_METAL) for x in z)
    mult = 1 if multiplicity is None else int(round(float(multiplicity)))
    target = 0 if (mult <= 1 or has_tm) else mult - 1
    per_mol = {}
    for i, opts in centres:
        per_mol.setdefault(int(lab[i]), []).append((i, opts))
    odd_fixed = int(sum(elec[m] % 2 for m in range(k) if m not in per_mol))
    if D == 0 and not per_mol:
        return {}
    cap = abs(D) + 4
    # knapsack per molecule: delta -> (cost, ((atom, delta), ...))
    mol_opts = {}
    for m, cs in per_mol.items():
        best = {0: (0.0, ())}
        for i, opts in cs[:16]:
            nb = dict(best)
            for dl, (c, pk) in best.items():
                for od, oc in opts:
                    nd = dl + od
                    if abs(nd) <= cap and (nd not in nb or c + oc < nb[nd][0]):
                        nb[nd] = (c + oc, pk + ((i, od),))
            best = nb
        mol_opts[m] = best
    # dynamic programme over the molecules: (delta, n_odd) -> (cost, ((molecule, delta), ...))
    states = {(0, 0): (0.0, ())}
    for m, best in mol_opts.items():
        ns = {}
        for (dl, no), (c, ch) in states.items():
            for od, (oc, _) in best.items():
                nd = dl + od
                if abs(nd) > cap:
                    continue
                key = (nd, no + int((elec[m] - od) % 2))
                if key not in ns or c + oc < ns[key][0]:
                    ns[key] = (c + oc, ch + ((m, od),))
        states = ns
    # score: cost + ODD_WEIGHT |odd-electron molecules - target| (the parity only tips near-equal solutions: a
    # stretched X-H of a singlet stays two radicals, X. + H., rather than X- + H+)
    ok = [(c + ODD_WEIGHT * abs(odd_fixed + no - target), ch) for (dl, no), (c, ch) in states.items() if dl == D]
    moved = {}
    if ok:
        _, ch = min(ok, key=lambda t: t[0])
        for m, od in ch:
            if od:
                for i, dl in mol_opts[m][od][1]:
                    q[i] += dl
                    moved[i] = dl
    else:
        big = int(np.argmax(np.bincount(lab, minlength=k)))
        i = int(np.flatnonzero(lab == big)[0])
        q[i] += D
        moved[i] = D
        flags.append(f'total charge {Q}: {D:+d} put on the largest molecule (no rule matched)')
    return moved


def report_text(report, max_lines: int = 30) -> str:
    """Short text of the report: charged or flagged molecules one per line, the neutral ones summarised."""
    lines, neutral = [], Counter()
    for m in report['molecules']:
        if m['charge'] == 0 and not m['flags']:
            neutral[m['formula']] += 1
            continue
        lines.append(f"  {m['formula']:<16s} {m['charge']:+d}  ({m['n_atoms']} atoms)"
                     + (f"  [{'; '.join(m['flags'])}]" if m['flags'] else ''))
    out = lines[:max_lines] + ([f'  ... {len(lines) - max_lines} more'] if len(lines) > max_lines else [])
    if neutral:
        out.append('  neutral: ' + ', '.join(f'{c} x {f}' for f, c in neutral.most_common(8)))
    return '\n'.join(out + [f'  ! {f}' for f in report['flags']])
