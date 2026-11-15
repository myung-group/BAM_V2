"""NVT molecular dynamics with a trained RACE checkpoint (MD-observable benchmarks, Fu et al. TMLR 2023 protocol).

    python -m race_v2.inference.md_run <train config json> --ckpt <pkl> --init-xyz <test.xyz> --out <dir>
           [--run 0] [--n-runs 5] [--init-seed 123] [--steps 200000] [--dt 0.25] [--temp 520]
           [--ttime 20] [--save-freq 50] [--chunk 2000]

Integrator: the single Nose-Hoover thermostat of MDsim (kyonofx/MDsim, mdsim/md/integrator.py, Q = 3 N kT
(ttime*dt)^2), ported line by line, so the LiPS settings (520 K, 0.25 fs, ttime 20, 200k steps, save every 50
steps = 0.0125 ps, 5 random test configurations, Maxwell-Boltzmann velocities at T) match the benchmark.
Positions are NOT wrapped (unwrapped trajectories for the diffusivity).  Restartable: <out>/run<k>/state.npz
holds positions, velocities, zeta and the step; traj chunks are appended as traj_<start>.npz.
Writes <out>/run<k>/thermo.log with "time[ps] Etot Epot Ekin T" (MDsim format).
"""
import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from ase import units
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from race_v2.training import train as T
from race_v2.data.xyz_graphs import preprocess_graph, atomic_numbers_to_indices
from race_v2.data.loaders import default_collate_fn
from race_v2.training.utils import load_checkpoint


class RaceModel:
    def __init__(self, cfg, ckpt_path):
        if cfg.get("x64"):
            jax.config.update("jax_enable_x64", True)
        (self.atomic_numbers, self.atom_energies, _tr, _va, _te, stats) = T.build_datasets(cfg, sys.stderr)
        graphdef, _p0, _o, _s, _g = T.build_model(cfg, len(self.atomic_numbers), self.atom_energies, stats, sys.stderr)
        ck = load_checkpoint(ckpt_path)
        self.model = nnx.merge(graphdef, jax.tree.map(jnp.asarray, ck["ema_params"]))
        self.idx = atomic_numbers_to_indices(self.atomic_numbers)
        self.cutoff = float(T._get(cfg, "cutoff", 5.0))
        self.x64 = bool(cfg.get("x64"))
        self.collate = functools.partial(default_collate_fn, batch_size=1,
                                         node_bucket_size=int(T._get(cfg, "node_bucket_size", 128)),
                                         edge_bucket_size=int(T._get(cfg, "edge_bucket_size", 4096)))
        model = self.model

        @jax.jit
        def step(batch):
            r = model(batch)
            return r["energy"].reshape(-1)[0], r["forces"]
        self._step = step

    def __call__(self, atoms):
        g = preprocess_graph(atoms, self.idx, self.atom_energies, self.cutoff, targets=False)
        if self.x64:
            g = T._cast_graphs([g], np.float64)[0]
        e, f = self._step(self.collate([g]))
        e0 = float(np.sum(np.asarray(self.atom_energies, dtype=np.float64)[[self.idx[int(z)] for z in atoms.numbers]]))
        return float(e) + e0, np.asarray(f, dtype=np.float64)[: len(atoms)]


class NoseHoover:
    """MDsim NoseHoover (single thermostat), same update order."""

    def __init__(self, atoms, model, dt_fs, temp_k, ttime, zeta=0.0):
        self.atoms, self.model = atoms, model
        self.dt = dt_fs * units.fs
        self.T = temp_k * units.kB
        n = len(atoms)
        self.target_ekin = 0.5 * (3.0 * n) * self.T
        self.Q = 3.0 * n * self.T * (ttime * self.dt) ** 2
        self.zeta = zeta
        self.m = atoms.get_masses().reshape(-1, 1)
        self.epot, self.f = model(atoms)

    def step(self):
        a = self.atoms
        accel = self.f / self.m
        vel = a.get_velocities()
        a.set_positions(a.get_positions() + vel * self.dt + (accel - self.zeta * vel) * (0.5 * self.dt ** 2))
        ke0 = a.get_kinetic_energy()
        vel_half = vel + 0.5 * self.dt * (accel - self.zeta * vel)
        a.set_velocities(vel_half)
        self.epot, self.f = self.model(a)
        accel = self.f / self.m
        self.zeta += 0.5 * self.dt * (1 / self.Q) * (ke0 - self.target_ekin)
        self.zeta += 0.5 * self.dt * (1 / self.Q) * (a.get_kinetic_energy() - self.target_ekin)
        a.set_velocities((a.get_velocities() + 0.5 * self.dt * accel) / (1 + 0.5 * self.dt * self.zeta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--init-xyz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--init-seed", type=int, default=123)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--temp", type=float, default=520.0)
    ap.add_argument("--ttime", type=float, default=20.0)
    ap.add_argument("--save-freq", type=int, default=50)
    ap.add_argument("--chunk", type=int, default=2000, help="frames per saved trajectory chunk")
    ap.add_argument("--max-hours", type=float, default=23.0)
    args = ap.parse_args()
    t_start = time.time()
    cfg = json.load(open(args.config))
    model = RaceModel(cfg, args.ckpt)
    out = Path(args.out) / f"run{args.run}"
    out.mkdir(parents=True, exist_ok=True)
    frames = read(args.init_xyz, index=":")
    init_idx = np.random.RandomState(args.init_seed).choice(len(frames), size=args.n_runs, replace=False)[args.run]
    atoms = frames[int(init_idx)].copy()
    atoms.calc = None
    state = out / "state.npz"
    if state.exists():
        s = np.load(state)
        atoms.set_positions(s["pos"]); atoms.set_velocities(s["vel"]); zeta = float(s["zeta"]); step0 = int(s["step"])
        print(f"resume at step {step0}", flush=True)
    else:
        MaxwellBoltzmannDistribution(atoms, temperature_K=args.temp, rng=np.random.RandomState(args.init_seed + args.run))
        Stationary(atoms)
        zeta, step0 = 0.0, 0
        json.dump({"init_idx": int(init_idx), "init_xyz": args.init_xyz, "ckpt": args.ckpt, "config": args.config,
                   "dt_fs": args.dt, "temp_K": args.temp, "ttime": args.ttime, "steps": args.steps,
                   "save_freq": args.save_freq, "numbers": atoms.numbers.tolist(),
                   "cell": np.asarray(atoms.cell.array).tolist()}, open(out / "meta.json", "w"), indent=1)
        with open(out / "thermo.log", "w") as f:
            f.write("time[ps] Etot[eV] Epot[eV] Ekin[eV] T[K]\n")
    md = NoseHoover(atoms, model, args.dt, args.temp, args.ttime, zeta)
    buf, buf_start = [], step0
    log = open(out / "thermo.log", "a")
    t0 = time.time()
    step = step0
    while step < args.steps:
        md.step()
        step += 1
        if step % args.save_freq == 0:
            ek = atoms.get_kinetic_energy()
            tk = ek / (1.5 * len(atoms) * units.kB)
            log.write(f"{step * args.dt / 1000:.4f} {md.epot + ek:.6f} {md.epot:.6f} {ek:.6f} {tk:.2f}\n")
            buf.append(atoms.get_positions().astype(np.float32))
            if not np.all(np.isfinite(buf[-1])):
                print(f"non-finite positions at step {step}; stopping", flush=True)
                break
        if len(buf) >= args.chunk or step == args.steps or (time.time() - t_start) > args.max_hours * 3600:
            if buf:
                np.savez(out / f"traj_{buf_start:08d}.npz", pos=np.stack(buf), first_step=buf_start + args.save_freq,
                         save_freq=args.save_freq)
            np.savez(state, pos=atoms.get_positions(), vel=atoms.get_velocities(), zeta=md.zeta, step=step)
            log.flush()
            rate = (step - step0) / (time.time() - t0)
            print(f"step {step}  {rate:.1f} steps/s  T={atoms.get_kinetic_energy() / (1.5 * len(atoms) * units.kB):.0f} K", flush=True)
            buf, buf_start = [], step
            if (time.time() - t_start) > args.max_hours * 3600:
                print("time budget reached; resume later", flush=True)
                break
    log.close()


if __name__ == "__main__":
    main()
