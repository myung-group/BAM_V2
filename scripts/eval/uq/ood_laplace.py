#!/usr/bin/env python
"""Out-of-distribution detection with the SOAP-eigenbasis Laplace posterior (race_v2/eval/soap_laplace.py) of a
trained periodic RACEv2 potential.

    cd /scratch/t510a13/race_v2/work
    python scripts/eval/uq/ood_laplace.py --ckpt runs/mbd/snap/<ckpt>.pkl --out runs/uq/<tag> [--n-fit 400]
        [--n-eval 1000] [--samples 16] [--ggn-probes 2] [--lam 1e3 | grid] [--max-atoms 200] [--seed 0]
        [--sets mptrj_val,salex_val,omol25,opoly26,oc25]

Steps
  1. model from the checkpoint's config, run in float64 (parameters cast; the SOAP state is float64 already), EMA
     weights as the posterior mean, eigenbases Q_L / Q_R of the SOAP state of the checkpoint.
  2. curvature: GGN diagonal in the SOAP basis (soap_laplace.ggn_diag, Hutchinson probes) over --n-fit training
     structures (half MPtrj, half sAlex).  Likelihood: energy PER ATOM with --sigma-e (eV/atom) and force components
     with --sigma-f (eV/A).  Posterior precision = kappa * n_train * nu + lam (--kappa 1 = untempered, the module's
     --kappa-times-ntrain convention; nu = mean GGN diagonal per structure).
  3. per evaluation set: --n-eval random structures, energy and forces at the EMA weights and at --samples weight
     samples (the same samples for every set) -> <out>/<set>.npz (with --lam grid: <out>/lam_<value>/<set>.npz):
       n_atoms, n_edges, e_mean (EMA energy per atom, relative to the model's E0), sigE (std over samples of the
       energy per atom), sigF / sigFmax (mean / max over atoms of the norm of the per-component force std),
       e_abs_err (|E error| per atom; meaningless for other DFT levels), f_mae, file, key.
  4. summary.json and the log: AUROC and FPR at 95 % TPR for every (ID, OOD) pair and score, score quantiles per set,
     Spearman(sigE, |E error|) and Spearman(sigF, force MAE) for the ID sets.
  5. --wbm-relaxed <dir> adds the set 'wbm' (model-relaxed WBM structures, <dir>/relaxed_*of*.jsonl; --n-wbm N for a
     subset): the same scores plus material_id, wbm_step and, with <dir>/../discovery.csv.gz and --wbm-summary,
     e_form_err = predicted - reference formation energy per atom; report: Spearman(score, |error|), AUROC of each
     score for |error| > 0.05 / 0.1 eV/atom, median score and MAE per WBM step, mean |error| in the top 10 % and
     bottom 50 % of sigE, AUROC of WBM against the ID sets and of the OOD sets against WBM.
Molecules (no pbc) get a cubic box of max extent + 2 * cutoff + 1 A and zero shift vectors, so no image is a
neighbour.  Structures with elements outside the model's list or with more than --max-atoms atoms are skipped
(counted in summary.json).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

MPTRJ = '/scratch/t510a13/data/mptrj_lmdb'
OAM = '/scratch/t510a13/data/oam_lmdb'
ID_SETS = {'mptrj_val': f'{MPTRJ}/val', 'salex_val': f'{OAM}/salex_val/val'}
OOD_SETS = {'omol25': '/appsdata/t510a13/omol25/neutral_val', 'opoly26': '/appsdata/t510a13/opoly26/val',
            'oc25': '/appsdata/t510a13/oc25/val'}
FIT_SETS = {'mptrj_train': f'{MPTRJ}/train', 'salex_train': f'{OAM}/salex_train/train'}
SCORES = ('sigE', 'sigF', 'sigFmax')
T0 = time.time()


def log(msg):
    print(f"[{time.strftime('%F %T')} +{time.time() - T0:6.0f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------- statistics (numpy)
def _ranks(x):
    """Average ranks (ties share the mean rank)."""
    x = np.asarray(x, np.float64)
    order = np.argsort(x, kind='mergesort')
    r = np.empty(len(x)); r[order] = np.arange(1, len(x) + 1)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    return (np.bincount(inv, weights=r) / cnt)[inv]


def auroc(s_id, s_ood):
    """P(score of an OOD structure > score of an ID structure), ties 1/2 (rank-sum formula)."""
    s_id, s_ood = np.asarray(s_id, np.float64), np.asarray(s_ood, np.float64)
    s_id, s_ood = s_id[np.isfinite(s_id)], s_ood[np.isfinite(s_ood)]
    n0, n1 = len(s_id), len(s_ood)
    if n0 == 0 or n1 == 0:
        return None
    r = _ranks(np.concatenate([s_id, s_ood]))
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def fpr_at_tpr(s_id, s_ood, tpr=0.95):
    """Fraction of ID structures above the threshold that flags `tpr` of the OOD structures."""
    s_id, s_ood = np.asarray(s_id, np.float64), np.asarray(s_ood, np.float64)
    s_id, s_ood = s_id[np.isfinite(s_id)], s_ood[np.isfinite(s_ood)]
    if len(s_id) == 0 or len(s_ood) == 0:
        return None
    thr = np.quantile(s_ood, 1.0 - tpr)
    return float(np.mean(s_id >= thr))


def spearman(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return None
    rx, ry = _ranks(x[ok]), _ranks(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _f(x):
    """JSON-safe float."""
    return None if x is None or not np.isfinite(x) else float(x)


def quantiles(x):
    x = np.asarray(x, np.float64); x = x[np.isfinite(x)]
    if len(x) == 0:
        return {'n': 0, 'median': None, 'p05': None, 'p95': None, 'mean': None}
    q = np.quantile(x, [0.05, 0.5, 0.95])
    return {'n': int(len(x)), 'median': float(q[1]), 'p05': float(q[0]), 'p95': float(q[2]), 'mean': float(x.mean())}


# ----------------------------------------------------------------------------------------------- structures -> graphs
def build_graph(st, S, cast, z_to_idx, e0, cutoff, fields):
    """Structure dict (stream_io) -> float64 graph arrays of stream_io.graph_arrays with the periodic fields always
    present.  No pbc: cubic box of max extent + 2 * cutoff + 1 A, pbc False, shifts zero.  ValueError('bad_cell'):
    a periodic direction with a degenerate cell."""
    pos = np.asarray(st['positions'], np.float64)
    pbc = np.zeros(3, bool) if st.get('pbc') is None else np.asarray(st['pbc'], bool).reshape(3)
    cell = np.zeros((3, 3)) if st.get('cell') is None else np.asarray(st['cell'], np.float64).reshape(3, 3)
    st = dict(st)
    if pbc.any():
        if not np.all(np.isfinite(cell)) or abs(np.linalg.det(cell)) < 1e-3:
            raise ValueError('bad_cell')
        st.update(positions=pos, cell=cell, pbc=pbc)
        ga = S.graph_arrays(st, z_to_idx, e0, cutoff, fields)
    else:
        L = float(np.ptp(pos, axis=0).max()) + 2.0 * cutoff + 1.0
        pos = pos - 0.5 * (pos.max(0) + pos.min(0)) + 0.5 * L          # centred in the box (translation only)
        cell = np.eye(3) * L
        st.update(positions=pos, cell=cell, pbc=np.zeros(3, bool))
        try:
            ga = S.graph_arrays(st, z_to_idx, e0, cutoff, fields)       # open-boundary search (matscipy)
        except Exception:
            # no matscipy: the same box with pbc True through ASE; the box is large enough that every shift is zero
            st['pbc'] = np.ones(3, bool)
            ga = S.graph_arrays(st, z_to_idx, e0, cutoff, fields)
            if np.any(ga['Sij'] != 0):
                raise RuntimeError('periodic image inside the cutoff of a boxed molecule')
    if 'Sij' not in ga:
        ga['Sij'] = np.zeros((len(ga['senders']), 3))
        ga['cutoff'] = np.array([cutoff])
        ga['volume'] = np.array([abs(np.linalg.det(cell))])
        ga['stress'] = np.zeros((1, 6))
    ga = cast(ga, np.float64)
    ga['positions'] = pos                       # graph_arrays rounds to float32; keep the float64 input
    ga['cell'] = cell[None]
    return ga, bool(pbc.any())


def make_accept(skip, S, cast, z_to_idx, e0, cutoff, fields, max_atoms):
    e0 = np.asarray(e0, np.float64)

    def accept(st):
        z, pos = st.get('numbers'), st.get('positions')
        if z is None or pos is None or len(z) == 0 or not np.all(np.isfinite(pos)):
            skip['missing'] += 1; return None
        if len(z) > max_atoms:
            skip['max_atoms'] += 1; return None
        if any(int(x) not in z_to_idx for x in np.unique(z)):
            skip['species'] += 1; return None
        try:
            ga, periodic = build_graph(st, S, cast, z_to_idx, e0, cutoff, fields)
        except ValueError as ex:
            if str(ex) != 'bad_cell':
                raise
            skip['bad_cell'] += 1; return None
        except Exception as ex:
            if skip['graph_error'] == 0:
                log(f"  graph construction failed ({type(ex).__name__}: {str(ex)[:200]}); structure skipped")
            skip['graph_error'] += 1; return None
        n = len(z)
        e0s = float(e0[[z_to_idx[int(x)] for x in z]].sum())
        e, f = st.get('energy'), st.get('forces')
        e_lab = (float(e) - e0s) / n if (e is not None and np.isfinite(e)) else np.nan
        if f is not None and (np.shape(f) != (n, 3) or not np.all(np.isfinite(f))):
            f = None
        return {'ga': ga, 'n': n, 'ne': int(ga['n_edge'][0]), 'e_lab': e_lab,
                'f_lab': None if f is None else np.asarray(f, np.float64), 'periodic': periodic}
    return accept


def read_random(S, path, n_want, rng, accept, max_files=16, scan_factor=30):
    """n_want random accepted structures of an aselmdb directory (or file): up to max_files random files, random row
    ids of those files, read in blocks until n_want are accepted or scan_factor * n_want rows were looked at."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist")
    _, files = S._source_files('aselmdb:' + path)
    if not files:
        raise RuntimeError(f"{path}: no *.aselmdb file below this path (only aselmdb is read here)")
    pick = np.sort(rng.permutation(len(files))[:max_files])
    src = S.AselmdbSource(files, lru=max_files + 1)
    try:
        cand = []
        for fi in pick:
            ids = np.asarray(src.row_ids(int(fi)), np.int64)
            cand.append(np.stack([np.full(len(ids), fi, np.int64), ids], 1))
        cand = np.concatenate(cand)
        total = len(cand)
        cand = cand[rng.permutation(total)[:scan_factor * n_want]]
        kept, i = [], 0
        while len(kept) < n_want and i < len(cand):
            blk = cand[i:i + 128]; i += len(blk)
            for fi in np.unique(blk[:, 0]):
                keys = blk[blk[:, 0] == fi, 1]
                for k, st in zip(keys, src.read_block(int(fi), keys)):
                    if len(kept) >= n_want:
                        break
                    r = accept(st)
                    if r is not None:
                        r['file'] = os.path.basename(files[int(fi)]); r['key'] = int(k)
                        kept.append(r)
    finally:
        for env in src._envs.values():
            env.close()
    return kept, {'files': len(files), 'files_used': int(len(pick)), 'rows_in_used_files': int(total),
                  'rows_scanned': int(i)}


def read_wbm(dirpath, accept, n_max, rng):
    """Model-relaxed WBM structures: <dir>/relaxed_*of*.jsonl, one JSON per line with material_id, numbers,
    final_cell (3x3), final_positions (Cartesian), as scripts/eval/mbd/rmsd_sample.py reads them.  pbc True."""
    files = sorted(Path(dirpath).glob('relaxed_*of*.jsonl'))
    if not files:
        raise FileNotFoundError(f"{dirpath}: no relaxed_*of*.jsonl file")
    res, n_bad = {}, 0
    for p in files:
        with open(p) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    n_bad += 1; continue
                if not isinstance(r, dict) or r.get('final_positions') is None or r.get('final_cell') is None \
                        or r.get('numbers') is None or 'material_id' not in r:
                    n_bad += 1; continue
                res[str(r['material_id'])] = (r, p.name)
    ids = sorted(res)
    n_all = len(ids)
    if n_max and n_all > n_max:
        ids = [ids[i] for i in np.sort(rng.choice(n_all, n_max, replace=False))]
    kept = []
    for mid in ids:
        r, fn = res[mid]
        st = {'numbers': np.asarray(r['numbers'], np.int64),
              'positions': np.asarray(r['final_positions'], np.float64).reshape(-1, 3),
              'cell': np.asarray(r['final_cell'], np.float64).reshape(3, 3), 'pbc': np.ones(3, bool),
              'energy': None, 'forces': None, 'stress': None, 'info': {}, 'arrays': {}}
        if len(st['numbers']) != len(st['positions']):
            n_bad += 1; continue
        rec = accept(st)
        if rec is not None:
            rec.update(file=fn, key=len(kept), mid=mid)
            kept.append(rec)
    return kept, {'files': len(files), 'structures_in_files': n_all, 'rows_unusable': n_bad, 'rows_scanned': len(ids)}


def read_csv_cols(path, cols, keep=None):
    """{first column: [other columns as strings]} of a (gzipped) csv, standard library only; keep: ids to retain."""
    import csv
    import gzip
    op = gzip.open if str(path).endswith('.gz') else open
    with op(path, 'rt', newline='') as fh:
        rd = csv.DictReader(fh)
        missing = [c for c in cols if c not in (rd.fieldnames or [])]
        if missing:
            raise KeyError(f"{path}: no column {missing}")
        return {r[cols[0]]: [r[c] for c in cols[1:]] for r in rd if keep is None or r[cols[0]] in keep}


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return np.nan


def wbm_extra(recs, wbm_dir, summary_csv):
    """Per-structure WBM arrays: material_id, wbm_step, and (when the files exist) e_form_err = predicted - reference
    formation energy per atom (discovery.csv.gz next to the relaxed directory, WBM summary) and unique_prototype."""
    mids = [r['mid'] for r in recs]

    def step(m):
        try:
            return int(m.split('-')[1])
        except (IndexError, ValueError):
            return -1
    ex = {'material_id': np.array(mids), 'wbm_step': np.array([step(m) for m in mids])}
    disc = Path(wbm_dir).resolve().parent / 'discovery.csv.gz'
    try:
        if not disc.exists():
            raise FileNotFoundError(f"{disc} is missing")
        if not Path(summary_csv).exists():
            raise FileNotFoundError(f"{summary_csv} is missing")
        keep = set(mids)
        pred = read_csv_cols(disc, ['material_id', 'e_form_per_atom_racev2'], keep)
        ref = read_csv_cols(summary_csv, ['material_id', 'e_form_per_atom_mp2020_corrected', 'unique_prototype'], keep)
        ex['e_form_pred'] = np.array([_num(pred[m][0]) if m in pred else np.nan for m in mids])
        ex['e_form_ref'] = np.array([_num(ref[m][0]) if m in ref else np.nan for m in mids])
        ex['e_form_err'] = ex['e_form_pred'] - ex['e_form_ref']
        ex['unique_prototype'] = np.array([m in ref and str(ref[m][1]).strip().lower() == 'true' for m in mids])
        log(f"wbm: formation-energy errors for {int(np.isfinite(ex['e_form_err']).sum())} of {len(mids)} structures "
            f"({disc})")
    except Exception as exc:
        log(f"wbm: error analysis skipped ({type(exc).__name__}: {exc}); scores are still computed")
    return ex


def wbm_report(arr):
    """Do the high-uncertainty WBM structures have the large formation-energy errors?"""
    err = np.abs(arr['e_form_err']); ok = np.isfinite(err)
    rep = {'n_with_error': int(ok.sum()), 'MAE': _f(err[ok].mean()) if ok.any() else None}
    for k in SCORES:
        rep[f'spearman_{k}_abserr'] = spearman(arr[k], err)
    for thr in (0.05, 0.1):
        hi, lo = ok & (err > thr), ok & (err <= thr)
        rep[f'auroc_abserr_gt_{thr:g}'] = {'n_pos': int(hi.sum()), 'n_neg': int(lo.sum()),
                                           **{k: auroc(arr[k][lo], arr[k][hi]) for k in SCORES}}
    rep['steps'] = {}
    for s in range(1, 6):
        m = arr['wbm_step'] == s
        rep['steps'][str(s)] = {'n': int(m.sum()), 'MAE': _f(err[m & ok].mean()) if np.any(m & ok) else None,
                                **{f'median_{k}': quantiles(arr[k][m])['median'] for k in SCORES}}
    sE = arr['sigE']; v = ok & np.isfinite(sE)
    if v.sum() >= 10:
        q50, q90 = np.quantile(sE[v], [0.5, 0.9])
        rep['mean_abserr_top10pct_sigE'] = _f(err[v & (sE >= q90)].mean())
        rep['mean_abserr_bottom50pct_sigE'] = _f(err[v & (sE <= q50)].mean())
    return rep


def _ladder(n, base):
    """Smallest base * 2^k >= n."""
    r = int(base)
    while r < n:
        r *= 2
    return r


def pack(recs, nb, eb, gb):
    """Greedy packs of structures (sorted by edge count) under nb nodes / eb edges / gb graphs -> [(ids, shape)].
    Two shapes for the packs, (nb, eb / 4) and (nb, eb); a structure above the budget runs alone on a 2^k ladder."""
    order = np.argsort([r['ne'] for r in recs], kind='stable')
    out, cur, cn, ce = [], [], 0, 0

    def flush():
        nonlocal cur, cn, ce
        if cur:
            out.append((cur, (nb, eb // 4 if ce <= eb // 4 else eb)))
        cur, cn, ce = [], 0, 0
    for i in order:
        n, ne = recs[i]['n'], recs[i]['ne']
        if n + 1 > nb or ne > eb:
            out.append(([int(i)], (_ladder(n + 1, nb), _ladder(max(ne, 1), eb))))
            continue
        if cur and (cn + n + 1 > nb or ce + ne > eb or len(cur) >= gb):
            flush()
        cur.append(int(i)); cn += n; ce += ne
    flush()
    return out


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ckpt', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--n-fit', type=int, default=400, help='training structures of the GGN curvature')
    ap.add_argument('--n-eval', type=int, default=1000, help='structures per evaluation set')
    ap.add_argument('--samples', type=int, default=16)
    ap.add_argument('--ggn-probes', type=int, default=2, help='Hutchinson probes per structure (0: exact, slow)')
    ap.add_argument('--lam', default='1e3', help="prior precision, or 'grid' (1e2, 1e3, 1e4; all reported)")
    ap.add_argument('--max-atoms', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sets', default=','.join(list(ID_SETS) + list(OOD_SETS)))
    ap.add_argument('--sigma-e', type=float, default=0.02, help='energy noise of the likelihood, eV per atom')
    ap.add_argument('--sigma-f', type=float, default=0.07, help='force noise of the likelihood, eV/A')
    ap.add_argument('--n-train', type=float, default=22.4e6, help='structures per fine-tuning data epoch')
    ap.add_argument('--kappa', type=float, default=1.0, help='tempering, relative to n_train (1: untempered)')
    ap.add_argument('--batch-nodes', type=int, default=1024, help='padded nodes of an evaluation batch')
    ap.add_argument('--batch-edges', type=int, default=49152, help='padded edges of an evaluation batch')
    ap.add_argument('--batch-graphs', type=int, default=64, help='structures per evaluation batch at most')
    ap.add_argument('--max-files', type=int, default=16, help='aselmdb files sampled per set')
    ap.add_argument('--reuse-nu', action='store_true', help='read <out>/ggn_nu.npz of an earlier run of this --out')
    ap.add_argument('--wbm-relaxed', default=None,
                    help="evaluation set 'wbm': directory with relaxed_*of*.jsonl (model-relaxed WBM structures); "
                         'errors from <dir>/../discovery.csv.gz and --wbm-summary')
    ap.add_argument('--wbm-summary', default='/scratch/t510a13/probabilistic_symmetrization/bundle/scratch/t589a02/'
                                             'mp_data/wbm/2023-12-13-wbm-summary.csv.gz')
    ap.add_argument('--n-wbm', type=int, default=0, help='WBM structures (seeded random subset; 0: all)')
    a = ap.parse_args()
    if a.samples < 2:
        ap.error('--samples must be at least 2')
    lams = [1e2, 1e3, 1e4] if a.lam == 'grid' else [float(a.lam)]
    names = [s for s in a.sets.replace(':', ',').split(',') if s]
    unknown = [s for s in names if s not in ID_SETS and s not in OOD_SETS]
    if unknown:
        ap.error(f"unknown sets {unknown}; known: {list(ID_SETS) + list(OOD_SETS)}")
    sys.path.insert(0, os.getcwd())                 # race_v2 is a symlink in the working directory
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log(f"ood_laplace: {vars(a)}")

    import jax
    import jax.numpy as jnp
    import jraph
    from flax import nnx
    from race_v2.training import train as T          # switches jax_enable_x64 off at import: switch it on afterwards
    from race_v2.data import stream_io as S
    from race_v2.data.stream_data import _to_graph, _cast
    from race_v2.data.loaders import fixed_collate_fn
    from race_v2.training.utils import load_checkpoint
    from race_v2.eval.soap_laplace import find_soap_state, per_leaf_state, leaf_specs, sample_params, ggn_diag
    jax.config.update('jax_enable_x64', True)
    log(f"jax {jax.__version__}, devices {jax.devices()}")

    # ---- 1. model (float64), EMA weights, SOAP state
    ck = load_checkpoint(a.ckpt)
    if ck.get('config') is None:
        raise SystemExit('the checkpoint has no config')
    cfg = dict(ck['config'])
    trained_x64 = bool(cfg.get('x64'))
    cfg['x64'] = True                               # parameters and data in float64 (as predict.py BAM_PREDICT_X64)
    if not cfg.get('periodic'):
        raise SystemExit('this experiment needs a periodic model (config periodic true)')
    zs = [int(z) for z in cfg['atomic_numbers']]
    if 'atom_energies' not in cfg:
        raise SystemExit('config without atom_energies')
    e0 = np.asarray(cfg['atom_energies'], np.float64)
    cutoff = float(cfg.get('cutoff', 5.0))
    z_to_idx = {z: i for i, z in enumerate(zs)}
    if cfg.get('use_dataset_stats') and 'avg_n_neighbors' not in cfg:
        raise SystemExit('use_dataset_stats without avg_n_neighbors in the config')
    stats = {'shift_raw': 0.0, 'scale': 1.0, 'avg_n_neighbors': float(cfg.get('avg_n_neighbors', 25.0)),
             'n_graphs': 1, 'n_atoms': 1}
    assert jax.config.jax_enable_x64
    graphdef, *_ = T.build_model({'learning_rate': 1e-3, **cfg}, len(zs), list(e0), stats, io.StringIO())
    fields = {k: v for k, v in dict(charge_key=cfg.get('total_charge_key'), spin_key=cfg.get('total_spin_key'),
                                    task=cfg.get('task'), fragments=T.fragments_of(cfg)).items() if v}
    if fields:
        log(f"WARNING: the config asks for extra graph fields {fields}; structures without them may fail")

    st = find_soap_state(ck['opt_state'])
    leaves, treedef, cols = per_leaf_state(cfg, ck['params'], st, len(zs))

    def f64(x):
        x = np.asarray(x)
        return x.astype(np.float64) if np.issubdtype(x.dtype, np.floating) else x
    ema = [f64(x) for x in treedef.flatten_up_to(ck['ema_params'])]
    specs = leaf_specs(cfg, ema)
    cols = [[np.asarray(x) if hasattr(x, 'shape') else x for x in c] for c in cols]
    del ck, st, leaves
    soap_idx = [i for i, sp in enumerate(specs) if sp.kind == 'soap']
    adam_idx = [i for i, sp in enumerate(specs) if sp.kind == 'adam']
    n_w = int(sum(np.size(ema[i]) for i in soap_idx + adam_idx))
    log(f"model: {len(zs)} species, cutoff {cutoff} A, trained x64 {trained_x64} -> run in float64; SOAP state: "
        f"{len(soap_idx)} SOAP leaves, {len(adam_idx)} Adam leaves, {len(specs) - len(soap_idx) - len(adam_idx)} "
        f"fixed; {n_w:,} weights in the posterior")

    def to_dev(lv):
        return jax.tree.unflatten(treedef, [jnp.asarray(x) for x in lv])

    def _energy(m, pos, b, mask):
        ne = m.node_energies(pos, b.globals['cell'], b.globals['cutoff'], b)['node_energies'].reshape(-1)
        return jnp.where(mask, ne, 0.0)

    def _ef1(state, b):
        # ggn_diag output function, one structure per padded batch: energy PER ATOM (E0 excluded), forces, node mask
        m = nnx.merge(graphdef, state)
        mask = jraph.get_node_padding_mask(b)
        en, g = jax.value_and_grad(lambda pos: jnp.sum(_energy(m, pos, b, mask)))(b.nodes['positions'])
        return en / jnp.maximum(jnp.sum(mask), 1), -jnp.where(jnp.isfinite(g), g, 0.0), mask

    def _efb(state, b):
        # several structures per padded batch: energies per graph (E0 excluded; last entry = padding graph), forces
        m = nnx.merge(graphdef, state)
        mask = jraph.get_node_padding_mask(b)
        n_g, n_pad = b.n_node.shape[0], b.nodes['positions'].shape[0]
        gid = jnp.repeat(jnp.arange(n_g), b.n_node, total_repeat_length=n_pad)

        def e(pos):
            ne = _energy(m, pos, b, mask)
            return jnp.sum(ne), jax.ops.segment_sum(ne, gid, num_segments=n_g)
        (_, eg), g = jax.value_and_grad(e, has_aux=True)(b.nodes['positions'])
        return eg, -jnp.where(jnp.isfinite(g), g, 0.0)
    efb = jax.jit(_efb)

    summary = {'args': vars(a), 'ckpt': str(a.ckpt), 'cutoff': cutoff, 'n_species': len(zs), 'n_weights': n_w,
               'trained_x64': trained_x64, 'curvature': {}, 'lams': {}, 'paths': {**ID_SETS, **OOD_SETS, **FIT_SETS}}

    def write_summary():
        summary['time_s'] = time.time() - T0
        tmp = out / 'summary.json.tmp'
        json.dump(summary, open(tmp, 'w'), indent=1, default=str)
        os.replace(tmp, out / 'summary.json')

    def new_skip():
        return {'missing': 0, 'max_atoms': 0, 'species': 0, 'bad_cell': 0, 'graph_error': 0}

    # ---- 2. curvature: GGN diagonal in the SOAP basis
    nu_file = out / 'ggn_nu.npz'
    if a.reuse_nu and nu_file.exists():
        z = np.load(nu_file)
        nu_new = list(cols[1])
        for i in soap_idx + adam_idx:
            nu_new[i] = np.asarray(z[f'leaf{i}'], np.float64).reshape(np.shape(cols[1][i]))
        summary['curvature'] = {'reused': str(nu_file)}
        log(f"curvature: reused {nu_file}")
    else:
        fit, t0 = [], time.time()
        parts = list(FIT_SETS.items())
        for j, (nm, path) in enumerate(parts):
            want = a.n_fit // len(parts) + (a.n_fit % len(parts) if j == 0 else 0)
            skip = new_skip()
            recs, info = read_random(S, path, want, np.random.default_rng([a.seed, 100 + j]),
                                     make_accept(skip, S, _cast, z_to_idx, e0, cutoff, fields or None, a.max_atoms),
                                     a.max_files)
            log(f"fit set {nm}: {len(recs)} of {want} structures ({info}), skipped {skip}, {time.time() - t0:.0f} s")
            summary['curvature'][nm] = {'n': len(recs), 'skipped': skip, **info}
            fit += recs
        if not fit:
            raise SystemExit('no structure for the curvature fit')
        # one structure per padded batch on a coupled ladder (64 * 2^k nodes, 8192 * 2^k edges): few compilations
        batches, shapes = [], {}
        for r in fit:
            k = 0
            while r['n'] + 1 > 64 * 2 ** k or r['ne'] > 8192 * 2 ** k:
                k += 1
            shp = (64 * 2 ** k, 8192 * 2 ** k)
            shapes[shp] = shapes.get(shp, 0) + 1
            batches.append(fixed_collate_fn([_to_graph(r['ga'])], 1, shp[0], shp[1]))
        del fit
        log(f"GGN curvature: {len(batches)} structures, "
            f"{'exact' if a.ggn_probes == 0 else f'{a.ggn_probes} Hutchinson probes'}, sigma_E {a.sigma_e} eV/atom, "
            f"sigma_F {a.sigma_f} eV/A, padded shapes {shapes} (one compilation each)")
        nu_s, nu_a = ggn_diag(_ef1, to_dev(ema), batches, specs, cols, treedef, 1.0 / a.sigma_e ** 2,
                              1.0 / a.sigma_f ** 2, a.ggn_probes, seed=a.seed, log=log)
        del batches
        nu_new = list(cols[1])
        for k, i in enumerate(soap_idx):
            nu_new[i] = nu_s[k].reshape(np.shape(cols[1][i]))
        for k, i in enumerate(adam_idx):
            nu_new[i] = nu_a[k].reshape(np.shape(cols[1][i]))
        np.savez(nu_file, **{f'leaf{i}': nu_new[i] for i in soap_idx + adam_idx})
        summary['curvature'].update(shapes={str(k): v for k, v in shapes.items()}, time_s=time.time() - t0)
    cols = [cols[0], nu_new] + list(cols[2:])
    kappa = a.kappa * a.n_train
    nu_all = np.concatenate([np.asarray(nu_new[i], np.float64).ravel() for i in soap_idx + adam_idx])
    if not np.all(np.isfinite(nu_all)):
        raise SystemExit(f"non-finite GGN diagonal ({int(np.sum(~np.isfinite(nu_all)))} entries)")
    q = np.quantile(kappa * nu_all, [0.1, 0.5, 0.9, 0.999, 1.0])
    log(f"posterior precision kappa * n_train * nu (kappa {a.kappa:g}, n_train {a.n_train:g}): 10 % {q[0]:.2e}, "
        f"median {q[1]:.2e}, 90 % {q[2]:.2e}, 99.9 % {q[3]:.2e}, max {q[4]:.2e}; lam {lams}")
    summary['curvature'].update(kappa_effective=kappa, precision_quantiles_10_50_90_999_max=[float(x) for x in q],
                                sigma_e_per_atom=a.sigma_e, sigma_f=a.sigma_f)

    # ---- weight samples: the same for every set; the same random numbers for every lam
    theta_sq = float(sum(np.sum(ema[i] ** 2) for i in soap_idx + adam_idx))
    samples, lam_tag = {}, {}
    for lam in lams:
        rng = np.random.default_rng(a.seed + 1)
        samples[lam] = [sample_params(ema, specs, cols, kappa, lam, rng) for _ in range(a.samples)]
        d_sq = float(sum(np.sum((samples[lam][0][i] - ema[i]) ** 2) for i in soap_idx + adam_idx))
        gamma = float(np.sum(kappa * nu_all / (kappa * nu_all + lam)))
        lam_tag[lam] = f'{lam:g}'
        summary['lams'][lam_tag[lam]] = {'lam': lam, 'gamma_well_determined': gamma,
                                         'rel_perturbation_rms': float(np.sqrt(d_sq / max(theta_sq, 1e-300))),
                                         'sets': {}, 'pairs': {}}
        log(f"lam {lam:g}: {a.samples} weight samples, |theta_s - theta| / |theta| = "
            f"{np.sqrt(d_sq / max(theta_sq, 1e-300)):.3e}, well-determined directions {gamma:.0f} of {n_w}")
    write_summary()

    # ---- 3. evaluation sets
    ema_state = to_dev(ema)
    seen_shapes = set()
    results = {lam: {} for lam in lams}             # lam -> set -> dict of per-structure arrays

    def run(state, batches, recs, bad):
        E = np.full(len(recs), np.nan); F = [None] * len(recs)
        for bi, (ids, b) in enumerate(batches):
            if bi in bad:
                continue
            shp = (b.nodes['positions'].shape[0], b.senders.shape[0])
            t1 = time.time()
            try:
                eg, f = efb(state, b)
                eg, f = np.asarray(eg, np.float64), np.asarray(f, np.float64)
            except Exception as ex:
                bad[bi] = f"{type(ex).__name__}: {str(ex)[:300]}"
                log(f"  batch {bi} ({len(ids)} structures, shape {shp}) failed and is dropped: {bad[bi]}")
                continue
            if shp not in seen_shapes:
                seen_shapes.add(shp)
                log(f"  new padded shape {shp}: first call {time.time() - t1:.0f} s (compilation)")
            off = 0
            for j, i in enumerate(ids):
                n = recs[i]['n']
                E[i] = eg[j] / n; F[i] = f[off:off + n]; off += n
        return E, F

    def eval_set(name, loader, n_req, extra_fn=None, report_fn=None):
        """loader(accept) -> (records, info).  extra_fn(records) -> further per-structure arrays for the npz;
        report_fn(arrays) -> a 'wbm' entry of the set's report (only when the arrays hold e_form_err)."""
        t0 = time.time()
        skip = new_skip()
        recs, info = loader(make_accept(skip, S, _cast, z_to_idx, e0, cutoff, fields or None, a.max_atoms))
        extra = extra_fn(recs) if (extra_fn and recs) else {}
        if not recs:
            raise RuntimeError(f"no usable structure ({info}, skipped {skip})")
        packs = pack(recs, a.batch_nodes, a.batch_edges, a.batch_graphs)
        batches = [(ids, fixed_collate_fn([_to_graph(recs[i]['ga']) for i in ids], a.batch_graphs, shp[0], shp[1]))
                   for ids, shp in packs]
        for r in recs:
            del r['ga']
        n_at = np.array([r['n'] for r in recs]); n_ed = np.array([r['ne'] for r in recs])
        log(f"{name}: {len(recs)} of {n_req} structures ({info}), skipped {skip}; atoms median "
            f"{int(np.median(n_at))} max {n_at.max()}, edges median {int(np.median(n_ed))} max {n_ed.max()}, "
            f"{int(np.sum([r['periodic'] for r in recs]))} periodic; {len(batches)} batches; {time.time() - t0:.0f} s")
        bad = {}
        E0_, F0_ = run(ema_state, batches, recs, bad)
        log(f"{name}: EMA weights done, {time.time() - t0:.0f} s")
        e_lab = np.array([r['e_lab'] for r in recs])
        f_mae = np.array([np.mean(np.abs(F0_[i] - r['f_lab'])) if (F0_[i] is not None and r['f_lab'] is not None)
                          else np.nan for i, r in enumerate(recs)])
        base = {'skipped': skip, **info, 'n': len(recs), 'n_requested': n_req,
                'n_periodic': int(np.sum([r['periodic'] for r in recs]))}
        for lam in lams:
            N = len(recs)
            cnt = np.zeros(N); s1 = np.zeros(N); s2 = np.zeros(N)
            f1 = [None if F0_[i] is None else np.zeros_like(F0_[i]) for i in range(N)]
            f2 = [None if F0_[i] is None else np.zeros_like(F0_[i]) for i in range(N)]
            for s, lv in enumerate(samples[lam]):
                Es, Fs = run(to_dev(lv), batches, recs, bad)
                for i in range(N):
                    if F0_[i] is None or Fs[i] is None or not (np.isfinite(Es[i]) and np.all(np.isfinite(Fs[i]))):
                        continue
                    d = Es[i] - E0_[i]; df = Fs[i] - F0_[i]        # deviations from the EMA prediction
                    cnt[i] += 1; s1[i] += d; s2[i] += d * d; f1[i] += df; f2[i] += df * df
                if s == 0 or (s + 1) % 4 == 0 or s + 1 == len(samples[lam]):
                    log(f"{name} lam {lam:g}: sample {s + 1}/{len(samples[lam])}, {time.time() - t0:.0f} s")
            sigE = np.full(N, np.nan); sigF = np.full(N, np.nan); sigFmax = np.full(N, np.nan)
            nll_sum, nll_n = 0.0, 0
            for i in range(N):
                if cnt[i] < 2:
                    continue
                c = cnt[i]
                sigE[i] = np.sqrt(max(s2[i] - s1[i] ** 2 / c, 0.0) / (c - 1))
                var = np.maximum(f2[i] - f1[i] ** 2 / c, 0.0) / (c - 1)        # (n, 3) per-component variance
                nrm = np.sqrt(var.sum(-1))
                sigF[i], sigFmax[i] = nrm.mean(), nrm.max()
                if recs[i]['f_lab'] is not None:
                    v = np.maximum(var, 1e-18); err = F0_[i] - recs[i]['f_lab']
                    nll_sum += float(np.sum(0.5 * (np.log(2 * np.pi * v) + err ** 2 / v))); nll_n += err.size
            arr = {'n_atoms': n_at, 'n_edges': n_ed, 'e_mean': E0_, 'sigE': sigE, 'sigF': sigF, 'sigFmax': sigFmax,
                   'e_label': e_lab, 'e_abs_err': np.abs(E0_ - e_lab), 'f_mae': f_mae, 'n_samples': cnt,
                   'periodic': np.array([r['periodic'] for r in recs]),
                   'file': np.array([r['file'] for r in recs]), 'key': np.array([r['key'] for r in recs])}
            arr.update(extra)
            d = out if len(lams) == 1 else out / f'lam_{lam_tag[lam]}'
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / f'{name}.npz', **{('src_file' if k == 'file' else k): v for k, v in arr.items()})
            results[lam][name] = arr
            rep = dict(base, n_scored=int(np.sum(np.isfinite(sigE))), failed_batches=dict(bad),
                       scores={k: quantiles(arr[k]) for k in SCORES},
                       E_mae_per_atom=_f(np.nanmean(arr['e_abs_err'])) if np.any(np.isfinite(arr['e_abs_err'])) else None,
                       F_mae=_f(np.nanmean(f_mae)) if np.any(np.isfinite(f_mae)) else None,
                       F_nll_sample_std=_f(nll_sum / nll_n) if nll_n else None,
                       spearman_sigE_absEerr=spearman(sigE, arr['e_abs_err']),
                       spearman_sigF_Fmae=spearman(sigF, f_mae), time_s=time.time() - t0)
            if report_fn is not None and 'e_form_err' in arr:
                rep['wbm'] = report_fn(arr)
            summary['lams'][lam_tag[lam]]['sets'][name] = rep
            sc = rep['scores']
            log(f"{name} lam {lam:g}: n {rep['n_scored']}; " + '; '.join(
                f"{k} median {sc[k]['median']:.3e} [5 % {sc[k]['p05']:.3e}, 95 % {sc[k]['p95']:.3e}]"
                for k in SCORES if sc[k]['n']) + f"; E MAE/atom {rep['E_mae_per_atom']}, F MAE {rep['F_mae']}")

    all_sets = {**ID_SETS, **OOD_SETS}
    jobs = []
    for name in names:
        si = list(all_sets).index(name)
        jobs.append((name, a.n_eval, None, None,
                     lambda acc, p=all_sets[name], si=si: read_random(
                         S, p, a.n_eval, np.random.default_rng([a.seed, 200 + si]), acc, a.max_files)))
    if a.wbm_relaxed:
        jobs.append(('wbm', a.n_wbm or 'all', lambda recs: wbm_extra(recs, a.wbm_relaxed, a.wbm_summary), wbm_report,
                     lambda acc: read_wbm(a.wbm_relaxed, acc, a.n_wbm, np.random.default_rng([a.seed, 300]))))
        summary['paths']['wbm'] = str(a.wbm_relaxed)
    for name, n_req, extra_fn, report_fn, loader in jobs:
        try:
            eval_set(name, loader, n_req, extra_fn, report_fn)
        except Exception as ex:
            txt = f"{type(ex).__name__}: {ex}"
            log(f"{name}: FAILED, skipped: {txt}\n{traceback.format_exc()}")
            for lam in lams:
                summary['lams'][lam_tag[lam]]['sets'].setdefault(name, {})['error'] = txt
        write_summary()

    # ---- 4. report
    for lam in lams:
        rl, sl = results[lam], summary['lams'][lam_tag[lam]]
        log(f"==== lam {lam:g}: AUROC (OOD has the larger score) and FPR at 95 % TPR")
        ids_ok = [n for n in names if n in ID_SETS and n in rl]
        oods_ok = [n for n in names if n in OOD_SETS and n in rl]
        pairs = [(i, o) for i in ids_ok for o in oods_ok]
        if 'wbm' in rl:       # WBM as the "OOD" side against the ID sets, and as the reference side against the OOD sets
            pairs += [(i, 'wbm') for i in ids_ok] + [('wbm', o) for o in oods_ok]
        for i_name, o_name in pairs:
            pr = {}
            for k in SCORES:
                pr[k] = {'auroc': auroc(rl[i_name][k], rl[o_name][k]),
                         'fpr95': fpr_at_tpr(rl[i_name][k], rl[o_name][k])}
            sl['pairs'][f'{i_name}|{o_name}'] = pr
            log(f"  {i_name:>10s} vs {o_name:<8s} " + '  '.join(
                f"{k}: AUROC {pr[k]['auroc']:.3f} FPR95 {pr[k]['fpr95']:.3f}" if pr[k]['auroc'] is not None
                else f"{k}: n/a" for k in SCORES))
        w = sl['sets'].get('wbm', {}).get('wbm')
        if w:
            log(f"  wbm ({w['n_with_error']} with errors, MAE {w['MAE']}): Spearman with |error| "
                + ', '.join(f"{k} {w[f'spearman_{k}_abserr']}" for k in SCORES))
            for thr in ('0.05', '0.1'):
                r = w[f'auroc_abserr_gt_{thr}']
                log(f"  wbm AUROC for |error| > {thr} eV/atom ({r['n_pos']} of {r['n_pos'] + r['n_neg']}): "
                    + ', '.join(f"{k} {r[k]}" for k in SCORES))
            for s, r in w['steps'].items():
                log(f"  wbm step {s}: n {r['n']}, MAE {r['MAE']}, median "
                    + ', '.join(f"{k} {r[f'median_{k}']}" for k in SCORES))
            log(f"  wbm mean |error|: top 10 % sigE {w.get('mean_abserr_top10pct_sigE')}, bottom 50 % sigE "
                f"{w.get('mean_abserr_bottom50pct_sigE')}")
        for i_name in ids_ok:
            r = sl['sets'][i_name]
            log(f"  {i_name}: Spearman(sigE, |E err|/atom) {r['spearman_sigE_absEerr']}, Spearman(sigF, force MAE) "
                f"{r['spearman_sigF_Fmae']}, force NLL (sample std only) {r['F_nll_sample_std']}")
    if len(lams) > 1:
        nll = {lam_tag[l]: summary['lams'][lam_tag[l]]['sets'].get('mptrj_val', {}).get('F_nll_sample_std')
               for l in lams}
        ok = {k: v for k, v in nll.items() if v is not None}
        summary['lam_best_by_mptrj_val_force_nll'] = min(ok, key=ok.get) if ok else None
        log(f"lam grid: MPtrj-val force NLL {nll}; lowest: {summary['lam_best_by_mptrj_val_force_nll']} "
            f"(all lam values are reported)")
    write_summary()
    log(f"written {out / 'summary.json'}")


if __name__ == '__main__':
    main()
