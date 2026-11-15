"""pair_style race mode halo, weak scaling: the same LiPS frame replicated so that every GPU holds `--base` reps of it.
Periodic replicas of one configuration have identical per-replica energy and identical forces on equivalent atoms,
so every run is checked against the 1-GPU run (E per base cell, max |F - F_1GPU[equivalent atom]|).

    python research/diag/lammps_weak.py --mpi <lmp_mpi> --plugin <so> --export <dir> --xyz <extxyz> --base 4 4 4 \
        --runs 1:1:1x1x1 2:1:2x1x1 4:1:2x2x1 8:2:2x2x2 [--steps 200]          (ranks:nodes:replication)
"""
import argparse
import os
import re
import subprocess
import time

import numpy as np

WRAP = """#!/bin/bash
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
exec "$@"
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mpi', required=True); ap.add_argument('--plugin', required=True)
    ap.add_argument('--export', required=True); ap.add_argument('--xyz', required=True)
    ap.add_argument('--base', type=int, nargs=3, default=[4, 4, 4]); ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--runs', nargs='+', default=['1:1:1x1x1', '2:1:2x1x1', '4:1:2x2x1', '8:2:2x2x2'])
    a = ap.parse_args()
    import ase.io
    from ase.data import chemical_symbols
    base = ase.io.read(a.xyz, index=0).repeat(tuple(a.base))
    nb = len(base)
    zs = sorted(set(int(z) for z in base.numbers)); syms = [chemical_symbols[z] for z in zs]
    ref = None
    for spec in a.runs:
        nr, nn, rep = spec.split(':'); nr, nn = int(nr), int(nn); rep = tuple(int(v) for v in rep.split('x'))
        work = os.path.join(a.export, f'lammps_weak_{rep[0]}{rep[1]}{rep[2]}_{nr}'); os.makedirs(work, exist_ok=True)
        at = base.repeat(rep)                                  # index = t * nb + i  (ase repeat order)
        ase.io.write(os.path.join(work, 'frame.data'), at, format='lammps-data', specorder=syms, masses=True,
                     atom_style='atomic')
        open(os.path.join(work, 'wrap.sh'), 'w').write(WRAP)
        inp = f"""units metal
atom_style atomic
atom_modify map yes
boundary p p p
read_data frame.data
pair_style race {a.plugin} {os.path.abspath(a.export)} gpu auto mode halo
pair_coeff * * {' '.join(syms)}
neighbor 1.0 bin
thermo_style custom step pe
thermo_modify format float %22.12f
dump d all custom 1 forces.dump id fx fy fz
dump_modify d format float %20.14f sort id
run 0
undump d
velocity all create 520.0 4928459 dist gaussian
timestep 0.00025
fix 1 all nve
thermo_style custom step temp pe etotal
thermo 100
run {a.steps}
"""
        open(os.path.join(work, 'in.weak'), 'w').write(inp)
        per = nr // nn
        cmd = ['srun', '-n', str(nr), '-N', str(nn), f'--ntasks-per-node={per}', f'--cpus-per-task={288 // per}',
               '--gpu-bind=none', 'bash', 'wrap.sh', a.mpi, '-in', 'in.weak', '-log', 'log.weak', '-screen', 'none']
        t0 = time.time()
        r = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
        if r.returncode:
            print('FAILED', spec, r.stderr[-1500:]); 
            lg = os.path.join(work, 'log.weak')
            if os.path.exists(lg): print(open(lg).read()[-1500:])
            continue
        txt = open(os.path.join(work, 'log.weak')).read()
        E = float(re.search(r"Step\s+PotEng\s*\n\s*0\s+(\S+)", txt).group(1))
        perf = float(re.findall(r"Performance: .*?([\d.]+) timesteps/s", txt)[-1])
        fl = open(os.path.join(work, 'forces.dump')).read().split('ITEM: ATOMS id fx fy fz')[1].split('ITEM')[0]
        F = np.array([[float(x) for x in l.split()[1:]] for l in fl.strip().splitlines()])
        et = [float(m[3]) for m in re.findall(r"^\s*(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*$", txt, re.M)]
        ncell = len(at) // nb
        tim = [l.strip() for l in txt.splitlines() if 'halo timing' in l]
        if ref is None:
            ref = (E / ncell, F)
        dE = abs(E / ncell - ref[0]); dF = np.abs(F - ref[1][np.arange(len(at)) % nb]).max()
        print(f"{nr} GPU(s) / {nn} node(s), {len(at)} atoms: E/cell {E / ncell:.10f} |dE/cell| {dE:.2e} eV, "
              f"max |dF| {dF:.2e} eV/A | {perf:.3f} steps/s = {perf * len(at) / 1e3:.1f} k atom-steps/s | "
              f"NVE Etot span {(max(et) - min(et)) if et else float('nan'):.2e} eV | wall {time.time() - t0:.0f} s",
              flush=True)
        for l in tim[:1]:
            print('   ', l, flush=True)


if __name__ == '__main__':
    main()
