"""Optional per-graph fields of the training graphs (total charge, spin, per-fragment charges, reference charges),
JAX-free so that the in-memory path (data_3bpa.preprocess_graph) and the streaming workers (stream_io.graph_arrays)
build them with the same code.  Importable as ``bam_omat24.graph_fields`` or as a top-level module.

    nodes, glob = extra_fields(info, arrays, n_atoms, charge_key=..., spin_key=..., charge_label_key=...,
                               fragments=(fragment_key, fragment_charges_key, n_fragments), task=...)

task: the labelling convention of the structure (RACE_LR long_range.n_tasks, row of the task table): an int = the
same task for every structure of the config, or a str = info key holding the index per structure (mixed datasets;
a structure without the key is task 0).

info: per-structure values (ase.Atoms.info); arrays: per-atom arrays (ase.Atoms.arrays).  A per-atom key missing from
arrays is looked up in info as well (aselmdb rows keep everything in key_value_pairs / data).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _per_atom(info, arrays, key):
    v = arrays.get(key) if arrays is not None else None
    if v is None and info is not None:
        v = info.get(key)
    return v


def extra_fields(info, arrays, n_atoms: int, charge_key: Optional[str] = None, spin_key: Optional[str] = None,
                 charge_label_key: Optional[str] = None, fragments: Optional[Sequence] = None, task=None,
                 charge_label_optional: bool = False, charge_label_scale: float = 1.0):
    """charge_label_optional: a structure without the labels gets NaN labels instead of an error; the charge loss
    skips NaN labels (training_utils.masked_charge_loss), so such a structure still trains energy and forces."""
    info = info or {}
    nodes, glob = {}, {}
    if task is not None and task is not False:   # task index (RACE_LR task table); only when asked
        t = info.get(task, 0) if isinstance(task, str) else task
        if isinstance(t, bool) or float(t) != int(round(float(t))) or int(round(float(t))) < 0:
            raise ValueError(f"task index must be a non-negative integer, got {t!r}")
        glob["task"] = np.array([int(round(float(t)))], dtype=np.int32)
    if charge_key:     # total charge of the structure (RACE_LR); only when asked, so other graphs keep their keys
        glob["charge"] = np.array([float(info.get(charge_key, 0.0))], dtype=np.float32)
    if spin_key:       # spin multiplicity 2S+1 of the structure (RACE_LR charge/spin embedding); default 1
        glob["spin"] = np.array([float(info.get(spin_key, 1.0))], dtype=np.float32)
    if fragments:      # (fragment array key, fragment charges info key, max fragments): per-fragment Qeq constraints
        fkey, fqkey, n_frag = fragments[0], fragments[1], int(fragments[2])
        q_tot = float(info.get(charge_key, 0.0)) if charge_key else 0.0
        frag = _per_atom(info, arrays, fkey)
        frag = np.zeros(n_atoms, np.int32) if frag is None else np.asarray(frag).astype(np.int32).reshape(-1)
        if frag.min() < 0 or frag.max() >= n_frag:
            raise ValueError(f"fragment ids {frag.min()}..{frag.max()} outside 0..{n_frag - 1}")
        if fqkey in info:
            fq = np.asarray(info[fqkey], np.float64).reshape(-1)
            if len(fq) > n_frag:
                raise ValueError(f"{len(fq)} fragment charges > n_fragments {n_frag}")
            fq = np.concatenate([fq, np.zeros(n_frag - len(fq))])
            if charge_key and abs(fq.sum() - q_tot) > 1e-6:
                raise ValueError(f"fragment charges {fq.tolist()} do not sum to the total charge {q_tot}")
        elif len(np.unique(frag)) == 1:       # one fragment only (e.g. a ligand or protein part): it carries Q
            fq = np.zeros(n_frag); fq[frag[0]] = q_tot
        else:
            raise ValueError(f"structure with fragments {np.unique(frag).tolist()} but no info[{fqkey!r}]")
        nodes["fragment"] = frag
        glob["fragment_charges"] = fq.astype(np.float32)[None]
    if charge_label_key:   # per-atom reference charges (RACE_LR optional charge loss), e.g. Hirshfeld; only when asked
        v = _per_atom(info, arrays, charge_label_key)
        if v is None:
            if not charge_label_optional:
                raise KeyError(f"no per-atom charge labels {charge_label_key!r}")
            v = np.full(n_atoms, np.nan)
        nodes["ref_charges"] = np.asarray(v, dtype=np.float32).reshape(-1)
        if len(nodes["ref_charges"]) != n_atoms:
            raise ValueError(f"{charge_label_key!r}: {len(nodes['ref_charges'])} values for {n_atoms} atoms")
        sc = float(charge_label_scale)
        if sc != 1.0 and np.all(np.isfinite(nodes["ref_charges"])):
            # scaled about the mean, so the total charge is kept (e.g. NBO x 0.75: NBO point-charge dipoles are
            # about 1 / 0.79 of the DFT dipoles on the OMol25 validation molecules)
            r = nodes["ref_charges"].astype(np.float64)
            nodes["ref_charges"] = (sc * r + (1.0 - sc) * r.mean()).astype(np.float32)
    return nodes, glob


def field_keys(charge_key=None, spin_key=None, charge_label_key=None, fragments=None, task=None,
               charge_label_optional=False, formal_charges=None, charge_label_scale=1.0):
    """(info keys, per-atom keys) a structure must carry for these fields."""
    info_k, arr_k = [], []
    if formal_charges:      # labels that set the molecule charges (formal_charges()), and charges given per atom
        arr_k += list(formal_charges.get('label_keys', ('nbo_charges', 'lowdin_charges'))) + ['formal_charges']
        if formal_charges.get('spins'):     # long_range cs_local: molecule spins (formal_spins())
            arr_k += [formal_charges['spins'].get('label_key', 'lowdin_spins'), 'formal_spins']
    if isinstance(task, str):
        info_k.append(task)
    if charge_key:
        info_k.append(charge_key)
    if spin_key:
        info_k.append(spin_key)
    if fragments:
        arr_k.append(fragments[0]); info_k.append(fragments[1])
    if charge_label_key:
        arr_k.append(charge_label_key)
    return info_k, arr_k


# ----------------------------------------------------------------------------------------- formal charges (RACE_LR ct)
# Covalent radii (Cordero et al. 2008), A; the same table as train_3bpa.COVALENT_RADII (copied: this module is JAX-free)
_COV_RADII = {1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
                  10: 0.58, 11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02,
                  18: 1.06, 19: 2.03, 20: 1.76, 21: 1.70, 22: 1.60, 23: 1.53, 24: 1.39, 25: 1.39,
                  26: 1.32, 27: 1.26, 28: 1.24, 29: 1.32, 30: 1.22, 31: 1.22, 32: 1.20, 33: 1.19,
                  34: 1.20, 35: 1.20, 36: 1.16, 53: 1.39,
                  # completed to Z = 96 (Cordero et al. 2008) for the element descriptor, 2026-10-27
                  37: 2.20, 38: 1.95, 39: 1.90, 40: 1.75, 41: 1.64, 42: 1.54, 43: 1.47, 44: 1.46, 45: 1.42,
                  46: 1.39, 47: 1.45, 48: 1.44, 49: 1.42, 50: 1.39, 51: 1.39, 52: 1.38, 54: 1.40, 55: 2.44,
                  56: 2.15, 57: 2.07, 58: 2.04, 59: 2.03, 60: 2.01, 61: 1.99, 62: 1.98, 63: 1.98, 64: 1.96,
                  65: 1.94, 66: 1.92, 67: 1.92, 68: 1.89, 69: 1.90, 70: 1.87, 71: 1.87, 72: 1.75, 73: 1.70,
                  74: 1.62, 75: 1.51, 76: 1.44, 77: 1.41, 78: 1.36, 79: 1.36, 80: 1.32, 81: 1.45, 82: 1.46,
                  83: 1.48, 84: 1.40, 85: 1.50, 86: 1.50, 87: 2.60, 88: 2.21, 89: 2.15, 90: 2.06, 91: 2.00,
                  92: 1.96, 93: 1.90, 94: 1.87, 95: 1.80, 96: 1.69}


def molecules(numbers, positions, src, dst, shift_vectors=None, bond_factor: float = 1.2):
    """Molecule index of every atom: connected components of the bonds d_ij < bond_factor (r_cov,i + r_cov,j) among the
    given neighbour pairs (src, dst; shift_vectors (E, 3) in A for periodic images, None for open structures)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(numbers)
    if n == 0:
        return np.zeros(0, np.int64), 0
    src, dst = np.asarray(src, np.int64), np.asarray(dst, np.int64)
    d = positions[dst] - positions[src]
    if shift_vectors is not None:
        d = d + shift_vectors
    r = np.linalg.norm(d, axis=-1)
    rc = np.array([_COV_RADII.get(int(z), 1.5) for z in numbers])
    bond = r < bond_factor * (rc[src] + rc[dst])
    g = coo_matrix((np.ones(int(bond.sum())), (src[bond], dst[bond])), shape=(n, n))
    k, lab = connected_components(g, directed=False)
    return lab.astype(np.int64), int(k)


def formal_charges(numbers, positions, src, dst, total_charge: float, info=None, arrays=None, shift_vectors=None,
                   label_keys=('nbo_charges', 'lowdin_charges'), bond_factor: float = 1.2, given_key='formal_charges'):
    """Per-atom reference charges q0 for the distance-limited charge transfer of RACE_LR (long_range ct_mode
    "conductance"): every molecule (molecules()) carries an integer charge Q_m, spread evenly over its atoms.
      arrays[given_key] (per-atom, e.g. formal charges of the user)  -> summed per molecule;
      else the first available per-atom label of label_keys          -> summed per molecule and rounded;
      else one molecule, or a neutral structure                       -> Q on the molecule / zero everywhere;
      else                                                            -> Q / N on every atom (no integer split).
    The integers are corrected to sum to the total charge by the largest-remainder rule.  Returns (q0 (N,), source)."""
    n = len(numbers)
    Q = float(total_charge)
    lab, k = molecules(numbers, positions, src, dst, shift_vectors, bond_factor)
    v, source = None, None
    if arrays is not None and given_key in arrays:
        v, source = np.asarray(arrays[given_key], np.float64).reshape(-1), 'given'
    else:
        for key in label_keys:
            x = _per_atom(info, arrays, key)
            if x is not None and len(np.asarray(x).reshape(-1)) == n and np.all(np.isfinite(np.asarray(x, float))):
                v, source = np.asarray(x, np.float64).reshape(-1), key
                break
    if v is None:
        if k <= 1 or abs(Q) < 1e-9:
            qm = np.zeros(k)
            if k >= 1 and abs(Q) >= 1e-9:
                qm[0] = Q
            source = 'single molecule' if k <= 1 else 'neutral'
        else:
            return np.full(n, Q / max(n, 1)), 'uniform (no labels)'
    else:
        s = np.bincount(lab, weights=v, minlength=k)
        qm = np.round(s)
        diff = int(round(Q - qm.sum()))
        if diff != 0:                       # largest remainder: move the molecules closest to the next integer
            rem = s - qm
            order = np.argsort(-rem if diff > 0 else rem)
            for j in order[:abs(diff)]:
                qm[j] += np.sign(diff)
    cnt = np.bincount(lab, minlength=k).astype(np.float64)
    return (qm / np.maximum(cnt, 1.0))[lab], source


def formal_spins(numbers, positions, src, dst, multiplicity: float, q0, info=None, arrays=None, shift_vectors=None,
                 label_key='lowdin_spins', bond_factor: float = 1.2, given_key='formal_spins'):
    """Per-atom reference spins s0 for the local charge / spin conditioning of RACE_LR (long_range cs_local): every
    molecule (molecules(), the same as formal_charges) carries an integer 2S_m >= 0, spread evenly over its atoms.
      one molecule                                     -> 2S_m = multiplicity - 1;
      arrays[given_key] (per atom, e.g. of the user)   -> summed per molecule;
      else the per-atom spin label label_key           -> |sum per molecule|, rounded to the parity of the molecule's
                                                          electron count (sum Z - Q_m, Q_m from q0);
      else                                             -> the lowest 2S_m of that parity (0 or 1), the rest of 2S
                                                          (2S - sum of those, if > 0) on the largest molecule.
    Two separated radicals of an overall singlet keep 2S_m = 1 each (the sum of 2S_m need not be 2S).
    Returns (s0 (N,), source)."""
    n = len(numbers)
    two_s = int(round(float(multiplicity))) - 1
    lab, k = molecules(numbers, positions, src, dst, shift_vectors, bond_factor)
    if k <= 1:
        return np.full(n, two_s / max(n, 1), np.float64), 'single molecule'
    z = np.asarray(numbers, np.float64)
    qm = np.round(np.bincount(lab, weights=np.asarray(q0, np.float64).reshape(-1), minlength=k))
    par = np.mod(np.round(np.bincount(lab, weights=z, minlength=k) - qm).astype(np.int64), 2)
    v, source = None, None
    if arrays is not None and given_key in arrays:
        sm, source = np.round(np.bincount(lab, weights=np.asarray(arrays[given_key], np.float64).reshape(-1),
                                          minlength=k)), 'given'
    else:
        x = _per_atom(info, arrays, label_key)
        if x is not None and len(np.asarray(x).reshape(-1)) == n and np.all(np.isfinite(np.asarray(x, float))):
            v, source = np.asarray(x, np.float64).reshape(-1), label_key
        if v is not None:
            t = np.abs(np.bincount(lab, weights=v, minlength=k))
            sm = 2.0 * np.floor((t - par) / 2.0 + 0.5) + par        # nearest value of the right parity
            sm = np.maximum(sm, par)
        else:
            sm, source = par.astype(np.float64), 'parity'
            rest = two_s - int(sm.sum())
            if rest > 0:
                cnt0 = np.bincount(lab, minlength=k)
                sm[int(np.argmax(cnt0))] += rest
    cnt = np.bincount(lab, minlength=k).astype(np.float64)
    return (np.asarray(sm, np.float64) / np.maximum(cnt, 1.0))[lab], source
