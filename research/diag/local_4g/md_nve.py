"""Short NVE MD with a trained checkpoint (bam_omat24.predict.ase_calculator): energy conservation and stability.

    python runs/diag/local_4g/md_nve.py <ckpt> <xyz> [--frame 0] [--steps 4000] [--dt 0.5] [--T 300] [--seed 0]
Prints the total-energy drift (linear fit, meV/ps per atom), the RMS fluctuation of the total energy, the
temperature, the minimum interatomic distance over the run and, for RACE_LR, the total Qeq charge.
"""
import argparse
import sys
import time

import numpy as np
import ase.io
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

sys.path.insert(0, '.')
from bam_omat24.predict import ase_calculator, Predictor     # noqa: E402


class Runaway(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('xyz')
    ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--dt', type=float, default=0.5)
    ap.add_argument('--T', type=float, default=300.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None, help='npz with the time series (t, Epot, Ekin, min distance, sum q)')
    args = ap.parse_args()
    a = ase.io.read(args.xyz, args.frame)
    info = {k: a.info[k] for k in ('charge', 'spin') if k in a.info}
    a.calc = None
    pred = Predictor(args.ckpt, species_refit=False, batch_size=1)
    is_lr = pred.cfg.get('model') == 'RACE_LR'
    a.calc = ase_calculator(pred)
    a.info.update(info)
    MaxwellBoltzmannDistribution(a, temperature_K=args.T, rng=np.random.RandomState(args.seed))
    Stationary(a); ZeroRotation(a)
    md = VelocityVerlet(a, timestep=args.dt * units.fs)
    rec = []

    def log():
        d = a.get_all_distances(); np.fill_diagonal(d, 99.0)
        sq = float(pred.charges([a])[0].sum()) if is_lr else np.nan
        rec.append((a.get_potential_energy(), a.get_kinetic_energy(), d.min(), sq))
        T_now = a.get_kinetic_energy() / (1.5 * units.kB * len(a))
        if T_now > 5000.0 or d.min() < 0.8:      # runaway: stop and keep the record up to here
            raise Runaway(f'step {len(rec) * 10}: T {T_now:.0f} K, min distance {d.min():.2f} A')
    md.attach(log, interval=10)
    t0 = time.time()
    stopped = None
    try:
        md.run(args.steps)
    except Runaway as exc:
        stopped = str(exc)
    wall = time.time() - t0
    r = np.array(rec)
    etot = r[:, 0] + r[:, 1]
    t_ps = np.arange(len(r)) * 10 * args.dt / 1000.0
    slope = np.polyfit(t_ps, etot, 1)[0]
    temp = r[:, 1] / (1.5 * units.kB * len(a))
    line = (f'{a.get_chemical_formula()} Q={info.get("charge", 0)} {pred.cfg.get("model")}: {args.steps} steps x '
            f'{args.dt} fs, drift {slope / len(a) * 1e3:+.3f} meV/ps/atom, Etot RMS {etot.std() * 1e3:.2f} meV, '
            f'T {temp.mean():.0f} +- {temp.std():.0f} K, min distance {r[:, 2].min():.2f} A, '
            f'{wall / max(len(rec) * 10, 1) * 1e3:.0f} ms/step' + (f'; STOPPED (runaway) at {stopped}' if stopped else ''))
    if is_lr:
        q = pred.charges([a])[0]
        line += f', final sum q {q.sum():+.6f}, |q| max {np.abs(q).max():.3f}'
    print(line)
    if args.out:
        np.savez(args.out, t_ps=t_ps, epot=r[:, 0], ekin=r[:, 1], dmin=r[:, 2], sum_q=r[:, 3], stopped=str(stopped),
                 n_atoms=len(a), charge=info.get('charge', 0), model=pred.cfg.get('model'))


if __name__ == '__main__':
    main()
