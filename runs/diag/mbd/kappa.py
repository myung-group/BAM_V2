"""Thermal conductivity of the 103 PhononDB structures with a RACEv2 checkpoint, and the kappa SRME metric
(Matbench Discovery phonons task; 2026-11-11).

    python runs/diag/mbd/kappa.py run   --ckpt <ckpt.pkl> --structures <phononDB-PBE-103-structures.extxyz> \
                                        --out-dir <dir> [--shard 0 --num-shards 4] [--limit N]
    python runs/diag/mbd/kappa.py score --out-dir <dir> --reference <kappas-phononDB-PBE-noNAC.json.gz>

Protocol, following matbench_discovery/phonons (pipeline.py KappaSettings defaults, adapters/standard.py and
thermal_conductivity.py, fetched 2026-11-11; themselves adapted from MPA2suite/k_SRME):
  relax      FIRE on FrechetCellFilter(mask = no tilt) with FixSymmetry(symprec 0.01), fmax 1e-4 eV/A, <= 300 steps
  FC2 / FC3  phono3py finite displacements of 0.03 A in the supercells of the dataset (atoms.info fc2_supercell /
             fc3_supercell), symprec 1e-5, automatic plus-minus, primitive_matrix "auto"; fc2 and fc3 symmetrised
  kappa      Wigner transport equation in RTA (transport_type "MS-SMM19"), isotope scattering, boundary mean free
             path 1e6 um, 300 K, the q mesh of the dataset
  skipped    FC3 and kappa when the relaxation changed the space group or FC2 gives imaginary modes
             (frequency < 0 away from Gamma or for optical modes at Gamma, < -0.01 THz for acoustic modes at Gamma)
  SRME       2 * sum_modes |kappa_pred - kappa_ref| / sum(weights) / (kappa_pred + kappa_ref) per material, 2.0 for
             skipped or failed materials; the reported value is the mean over the 103 materials.
Differences from the upstream runner: forces of the displaced supercells are evaluated in batches (same forces),
and the space group comes from spglib instead of moyo.  One process per GPU; a rerun skips finished materials.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

NO_TILT_MASK = [True, True, True, False, False, False]
IMAG_THRESHOLD = -0.01
SRME_MAX = 2.0


def _bam_dir() -> str:
    d = os.environ.get('BAM_OMAT24_DIR')
    if d:
        return d
    for base in (Path.cwd(), *Path(__file__).resolve().parents):
        if (base / 'bam_omat24' / 'relax_batched.py').exists():
            return str(base / 'bam_omat24')
    raise SystemExit('bam_omat24 not found: run from the work folder or set BAM_OMAT24_DIR')


def spg_number(atoms, symprec: float) -> int:
    import spglib
    ds = spglib.get_symmetry_dataset((atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers), symprec=symprec)
    return int(ds.number)


def has_imaginary(freqs: np.ndarray) -> bool:
    return bool(np.all(np.isnan(freqs)) or np.any(freqs[0, 3:] < 0) or np.any(freqs[0, :3] < IMAG_THRESHOLD)
                or np.any(freqs[1:] < 0))


def kappa_avg(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, float)
    if k.shape[-2:] == (3, 3):
        return np.trace(k, axis1=-2, axis2=-1) / 3
    if k.shape[-1] == 6:
        return k[..., :3].mean(axis=-1)
    if k.shape[-1] == 3:
        return k.mean(axis=-1)
    raise ValueError(f'kappa of shape {k.shape}')


def mode_kappa_total(mode_kappa_p, mode_kappa_c, cv):
    with np.errstate(divide='ignore', invalid='ignore'):
        c = 2 * ((mode_kappa_c * cv[:, :, :, None, None]) / (cv[:, :, :, None, None] + cv[:, :, None, :, None])).sum(axis=2)
    c[np.isnan(c)] = 0
    return c + mode_kappa_p


class Forces:
    """Batched forces of many supercells of equal size (the displaced supercells of one material)."""

    def __init__(self, ev, RB, atoms_per_call: int, max_graphs: int, workers: int):
        self.ev, self.RB, self.budget, self.max_graphs = ev, RB, int(atoms_per_call), int(max_graphs)
        self.prepare = ev.prepare_fn()
        self.pool = None
        if workers > 1:
            import concurrent.futures as cf
            import multiprocessing as mp
            self.pool = cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'))

    def one(self, atoms):
        return self.ev.evaluate([self.prepare(atoms)])[0]

    def many(self, supercells) -> np.ndarray:
        """supercells: list of PhonopyAtoms or None -> forces (len, n_atoms, 3), zeros for None."""
        from ase import Atoms
        idx = [i for i, s in enumerate(supercells) if s is not None]
        n_at = len(next(s for s in supercells if s is not None))
        out = np.zeros((len(supercells), n_at, 3))
        frames = [Atoms(supercells[i].symbols, cell=supercells[i].cell, positions=supercells[i].positions, pbc=True)
                  for i in idx]
        per = max(1, min(self.max_graphs, self.budget // n_at))
        payloads = (list(self.pool.map(self.prepare, frames, chunksize=4)) if self.pool is not None
                    else [self.prepare(a) for a in frames])
        for k in range(0, len(idx), per):
            res = self.ev.evaluate(payloads[k:k + per])
            for i, (_e, f, _s) in zip(idx[k:k + per], res):
                out[i] = f
        if not np.all(np.isfinite(out)):
            raise FloatingPointError('non-finite displacement forces')
        return out

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()


def run_one(atoms, forces: Forces, RB) -> dict:
    from ase.calculators.calculator import Calculator, all_changes
    from ase.constraints import FixSymmetry
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from phono3py.api_phono3py import Phono3py
    from phonopy.structure.atoms import PhonopyAtoms

    class Calc(Calculator):
        implemented_properties = ['energy', 'forces', 'stress']

        def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            e, f, s = forces.one(self.atoms)
            self.results = {'energy': e, 'forces': f, 'stress': s}

    mid = str(atoms.info.get('material_id'))
    rec = {'material_id': mid, 'name': str(atoms.info.get('name', atoms.get_chemical_formula())), 'errors': []}
    t0 = time.time()
    stage = 'Relax'
    try:
        info = dict(atoms.info)
        rec['init_spg_num'] = spg_number(atoms, 1e-5)
        atoms = atoms.copy()
        atoms.calc = Calc()
        atoms.set_constraint(FixSymmetry(atoms, symprec=0.01))
        opt = FIRE(FrechetCellFilter(atoms, mask=NO_TILT_MASK), logfile=None)
        opt.run(fmax=1e-4, steps=300)
        rec['n_steps'] = int(opt.get_number_of_steps())
        rec['reached_max_steps'] = rec['n_steps'] >= 300
        rec['max_stress'] = np.abs(atoms.get_stress(voigt=True)).tolist()
        atoms.calc = None
        atoms.constraints = None
        rec['final_spg_num'] = spg_number(atoms, 1e-5)
        rec['broken_symmetry'] = rec['init_spg_num'] != rec['final_spg_num']
        rec['relaxed_cell'] = atoms.cell.array.tolist()
        rec['relaxed_positions'] = atoms.positions.tolist()
        rec['numbers'] = atoms.numbers.tolist()
        rec['t_relax'] = time.time() - t0

        stage = 'ForceConstant'
        ph3 = Phono3py(unitcell=PhonopyAtoms(atoms.symbols, cell=atoms.cell, positions=atoms.positions),
                       supercell_matrix=np.asarray(info['fc3_supercell']),
                       phonon_supercell_matrix=np.asarray(info['fc2_supercell']),
                       primitive_matrix='auto', symprec=1e-5)
        ph3.mesh_numbers = tuple(info['q_point_mesh'])
        ph3.generate_displacements(distance=0.03, is_plusminus='auto')
        t1 = time.time()
        d2 = ph3.phonon_supercells_with_displacements
        rec['n_fc2'], rec['n_atoms_fc2'] = len(d2), len(ph3.phonon_supercell)
        ph3.phonon_forces = forces.many(d2)
        ph3.produce_fc2(symmetrize_fc2=True)
        ph3.init_phph_interaction(symmetrize_fc3q=False)
        ph3.run_phonon_solver()
        freqs, _, _ = ph3.get_phonon_data()
        rec['has_imag_ph_modes'] = has_imaginary(freqs)
        rec['min_freq'] = float(np.nanmin(freqs))
        rec['t_fc2'] = time.time() - t1
        if rec['has_imag_ph_modes'] or rec['broken_symmetry']:
            rec['conductivity_skipped'] = True
            return rec
        t2 = time.time()
        d3 = ph3.supercells_with_displacements
        rec['n_fc3'], rec['n_atoms_fc3'] = len(d3), len(ph3.supercell)
        ph3.forces = forces.many(d3)
        ph3.produce_fc3(symmetrize_fc3r=True)
        rec['t_fc3'] = time.time() - t2

        stage = 'Conductivity'
        t3 = time.time()
        ph3.init_phph_interaction(symmetrize_fc3q=False)
        ph3.run_thermal_conductivity(temperatures=[300.0], is_isotope=True, transport_type='MS-SMM19',
                                     boundary_mfp=1e6)
        kappa = ph3.thermal_conductivity
        extra, cv = kappa.get_extra_kappa_output(), kappa.mode_heat_capacities
        if extra is None or cv is None:
            raise RuntimeError('missing kappa output')
        k_tot = np.array(extra['kappa_TOT_RTA'][0])
        mode_tot = mode_kappa_total(np.array(extra['mode_kappa_P_RTA'][0]), np.array(extra['mode_kappa_C'][0]),
                                    np.array(cv))
        rec.update(kappa_tot_rta=k_tot.tolist(), kappa_p_rta=np.array(extra['kappa_P_RTA'][0]).tolist(),
                   kappa_c=np.array(extra['kappa_C'][0]).tolist(), kappa_tot_avg=kappa_avg(k_tot).tolist(),
                   mode_kappa_tot_avg=kappa_avg(mode_tot).tolist(), weights=np.array(kappa.grid_weights).tolist(),
                   q_points=np.array(kappa.qpoints).tolist(), ph_freqs=np.array(kappa.frequencies).tolist())
        rec['t_kappa'] = time.time() - t3
    except Exception as exc:                       # every failure is recorded and scored as SRME 2
        rec['errors'].append(f'{stage}Error: {type(exc).__name__}: {exc}')
        rec['traceback'] = traceback.format_exc(limit=6)
    rec['t_total'] = time.time() - t0
    return rec


def cmd_run(a):
    bam = _bam_dir()
    sys.path.insert(0, bam); sys.path.insert(0, str(Path(bam).parent))
    import relax_batched as RB
    import stream_worker
    import ase.io
    stream_worker.clean_main()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    frames = ase.io.read(a.structures, index=':', format='extxyz')
    todo = [(i, at) for i, at in enumerate(frames) if i % a.num_shards == a.shard]
    if a.limit:
        todo = todo[:a.limit]
    ev = RB.RaceEvaluator(a.ckpt, max_graphs=a.max_graphs)
    forces = Forces(ev, RB, a.atoms_per_call, a.max_graphs, a.workers)
    print(f'[kappa] {len(todo)} of {len(frames)} structures, shard {a.shard}/{a.num_shards}, ckpt {a.ckpt}', flush=True)
    for i, at in todo:
        mid = str(at.info.get('material_id', i))
        path = out / f'{mid}.json'
        if path.exists() and not a.redo:
            continue
        rec = run_one(at, forces, RB)
        tmp = path.with_suffix('.tmp')
        json.dump(rec, open(tmp, 'w')); os.replace(tmp, path)
        k = rec.get('kappa_tot_avg')
        print(f"[kappa] {mid} {rec['name']}: steps {rec.get('n_steps')} spg {rec.get('init_spg_num')}->"
              f"{rec.get('final_spg_num')} imag {rec.get('has_imag_ph_modes')} fc2 {rec.get('n_fc2')} fc3 "
              f"{rec.get('n_fc3')} kappa {None if k is None else round(float(k[0]), 3)} "
              f"t {rec['t_total']:.0f} s {'; '.join(rec['errors'])}", flush=True)
    forces.close()


def srme_one(rec, ref):
    """(SRME, reason or None).  ref: record of the reference file."""
    if rec is None:
        return SRME_MAX, 'missing'
    if rec.get('errors'):
        return SRME_MAX, rec['errors'][0][:60]
    if rec.get('has_imag_ph_modes'):
        return SRME_MAX, 'imaginary modes'
    exp = rec.get('init_spg_num', ref.get('init_spg_num'))
    if rec.get('final_spg_num') is not None and exp is not None and rec['final_spg_num'] != exp:
        return SRME_MAX, 'broken symmetry'
    try:
        mp, mt = np.asarray(rec['mode_kappa_tot_avg'], float), np.asarray(ref['mode_kappa_tot_avg'], float)
        kp, kt = np.atleast_1d(np.asarray(rec['kappa_tot_avg'], float)), np.atleast_1d(np.asarray(ref['kappa_tot_avg'], float))
        w = np.asarray(rec['weights'], float)
        if mp.shape != mt.shape or kp.shape != kt.shape or w.sum() <= 0 or not np.all(np.isfinite(mp)):
            return SRME_MAX, f'shape {mp.shape} vs {mt.shape}'
        micro = np.abs(mp - mt).sum(axis=tuple(range(1, mp.ndim))) / w.sum()
        s = SRME_MAX * micro / (kp + kt)
        if not (np.isfinite(s[0]) and 0 <= s[0] <= SRME_MAX):
            return SRME_MAX, 'invalid score'
        return float(s[0]), None
    except (KeyError, TypeError, ValueError) as exc:
        return SRME_MAX, f'{type(exc).__name__}: {exc}'[:60]


def cmd_score(a):
    ref = {r['material_id']: r for r in json.load(gzip.open(a.reference))}
    out = Path(a.out_dir)
    rows, reasons = [], {}
    for mid, r in ref.items():
        p = out / f'{mid}.json'
        rec = json.load(open(p)) if p.exists() else None
        s, why = srme_one(rec, r)
        kp = None if rec is None or rec.get('kappa_tot_avg') is None else float(np.atleast_1d(rec['kappa_tot_avg'])[0])
        kt = float(np.atleast_1d(r['kappa_tot_avg'])[0])
        rows.append((mid, r.get('name'), s, kp, kt, why))
        if why:
            reasons[why] = reasons.get(why, 0) + 1
    srme = np.array([x[2] for x in rows])
    ok = [(x[3], x[4]) for x in rows if x[5] is None]
    sre = [2 * abs(p - t) / (p + t) for p, t in ok]
    res = {'n': len(rows), 'kappa_SRME': float(srme.mean()), 'n_scored': len(ok), 'n_censored': len(rows) - len(ok),
           'censored_reasons': reasons, 'kappa_SRE_of_scored': float(np.mean(sre)) if sre else None,
           'kappa_SRME_of_scored': float(np.mean([x[2] for x in rows if x[5] is None])) if ok else None}
    json.dump({'summary': res, 'rows': rows}, open(out / 'kappa_srme.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--ckpt', required=True); r.add_argument('--structures', required=True)
    r.add_argument('--out-dir', required=True)
    r.add_argument('--shard', type=int, default=0); r.add_argument('--num-shards', type=int, default=1)
    r.add_argument('--limit', type=int, default=0); r.add_argument('--redo', action='store_true')
    r.add_argument('--atoms-per-call', type=int, default=1024); r.add_argument('--max-graphs', type=int, default=16)
    r.add_argument('--workers', type=int, default=16)
    s = sub.add_parser('score')
    s.add_argument('--out-dir', required=True); s.add_argument('--reference', required=True)
    a = ap.parse_args()
    (cmd_run if a.cmd == 'run' else cmd_score)(a)


if __name__ == '__main__':
    main()
