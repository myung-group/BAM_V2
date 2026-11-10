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
                 charge_label_key: Optional[str] = None, fragments: Optional[Sequence] = None, task=None):
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
            raise KeyError(f"no per-atom charge labels {charge_label_key!r}")
        nodes["ref_charges"] = np.asarray(v, dtype=np.float32).reshape(-1)
        if len(nodes["ref_charges"]) != n_atoms:
            raise ValueError(f"{charge_label_key!r}: {len(nodes['ref_charges'])} values for {n_atoms} atoms")
    return nodes, glob


def field_keys(charge_key=None, spin_key=None, charge_label_key=None, fragments=None, task=None):
    """(info keys, per-atom keys) a structure must carry for these fields."""
    info_k, arr_k = [], []
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
