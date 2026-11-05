"""pair_style race across ranks / GPUs: run 0 of a LiPS supercell with (1 rank owner) as the reference vs
(1 rank ghost), (N ranks ghost, one GPU each); then NVE timings.  Run inside a Slurm GPU job.

    python runs/diag/lammps_multi.py --serial <lmp> --mpi <lmp_mpi> --plugin <so> --export <dir> --xyz <extxyz>
        --rep 2 2 2 --ranks 2 4 [--nodes 1] [--steps 200]
"""
import argparse
import os
import re
import subprocess
import time

import numpy as np

WRAP = """#!/bin/bash
# one visible GPU per rank (node-local rank), so every PJRT client sees exactly its own device
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID:-0}
exec "$@"
"""


def write_inputs(work, plugin, export, syms, mode, steps):
    inp = f"""units metal
atom_style atomic
atom_modify map yes
boundary p p p
read_data frame.data
pair_style race {plugin} {os.path.abspath(export)} gpu auto mode {mode}
pair_coeff * * {' '.join(syms)}
neighbor 1.0 bin
thermo_style custom step pe pxx pyy pzz pxy pxz pyz
thermo_modify format float %20.12f
dump d all custom 1 forces_{mode}.dump id fx fy fz
dump_modify d format float %20.14f sort id
run 0
undump d
"""
    if steps:
        inp += f"""velocity all create 520.0 4928459 dist gaussian
timestep 0.00025
fix 1 all nve
thermo_style custom step temp pe etotal
thermo 100
run {steps}
"""
    path = os.path.join(work, f'in.{mode}.{steps}')
    open(path, 'w').write(inp)
    return os.path.basename(path)


def parse(work, log, mode):
    txt = open(os.path.join(work, log)).read()
    m = re.search(r"Step\s+PotEng\s+Pxx.*\n\s*0\s+(.+)\n", txt)
    vals = [float(v) for v in m.group(1).split()]
    fl = open(os.path.join(work, f'forces_{mode}.dump')).read().split('ITEM: ATOMS id fx fy fz')[1].split('ITEM')[0]
    F = np.array([[float(x) for x in line.split()[1:]] for line in fl.strip().splitlines()])
    perf = re.findall(r"Performance: .*?([\d.]+) timesteps/s", txt)
    et = [float(r[3]) for r in re.findall(r"^\s*(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*$", txt, re.M)]
    return vals[0], np.array(vals[1:7]), F, (float(perf[-1]) if perf else None), et


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--serial', required=True); ap.add_argument('--mpi', required=True)
    ap.add_argument('--plugin', required=True); ap.add_argument('--export', required=True)
    ap.add_argument('--xyz', required=True); ap.add_argument('--rep', type=int, nargs=3, default=[2, 2, 2])
    ap.add_argument('--ranks', type=int, nargs='+', default=[2, 4]); ap.add_argument('--nodes', type=int, default=1)
    ap.add_argument('--steps', type=int, default=200); ap.add_argument('--tag', default='multi')
    ap.add_argument('--skip-serial-ghost', action='store_true')
    a = ap.parse_args()
    import ase.io
    from ase.data import chemical_symbols
    work = os.path.join(a.export, f'lammps_{a.tag}_{a.rep[0]}{a.rep[1]}{a.rep[2]}'); os.makedirs(work, exist_ok=True)
    at = ase.io.read(a.xyz, index=0).repeat(tuple(a.rep))
    zs = sorted(set(int(z) for z in at.numbers)); syms = [chemical_symbols[z] for z in zs]
    ase.io.write(os.path.join(work, 'frame.data'), at, format='lammps-data', specorder=syms, masses=True,
                 atom_style='atomic')
    open(os.path.join(work, 'wrap.sh'), 'w').write(WRAP)
    print(f'{len(at)} atoms, cell lengths {np.round(at.cell.lengths(), 2)}', flush=True)

    def run(cmd, log):
        t0 = time.time()
        r = subprocess.run(cmd + ['-log', log, '-screen', 'none'], cwd=work, capture_output=True, text=True)
        if r.returncode:
            print('FAILED', ' '.join(cmd)); print(r.stdout[-2500:]); print(r.stderr[-2500:])
            tail = open(os.path.join(work, log)).read()[-2500:] if os.path.exists(os.path.join(work, log)) else ''
            print(tail); return False
        print(f'  ran {" ".join(cmd[-3:])} in {time.time() - t0:.0f} s', flush=True)
        return True

    # reference: 1 rank, owner mode (validated against Python, runs/diag/lammps_validate.py)
    inp = write_inputs(work, a.plugin, a.export, syms, 'owner', a.steps)
    env_cmd = ['bash', 'wrap.sh', a.serial, '-in', inp]
    if not run(env_cmd, 'log.owner'):
        return
    E0, P0, F0, perf0, et0 = parse(work, 'log.owner', 'owner')
    print(f'reference 1 rank owner: E {E0:.10f} eV, |P|max {np.abs(P0).max():.1f} bar, {perf0} steps/s', flush=True)
    cases = [] if a.skip_serial_ghost else [('ghost', 1)]
    cases += [('ghost', n) for n in a.ranks]
    for mode, n in cases:
        inp = write_inputs(work, a.plugin, a.export, syms, mode, a.steps)
        if n == 1:
            cmd = ['bash', 'wrap.sh', a.serial, '-in', inp]
        else:
            per_node = n // a.nodes
            cmd = ['srun', '-n', str(n), '-N', str(a.nodes), f'--ntasks-per-node={per_node}',
                   f'--cpus-per-task={288 // per_node}', '--gpu-bind=none', 'bash', 'wrap.sh', a.mpi, '-in', inp]
        log = f'log.{mode}.{n}'
        if not run(cmd, log):
            continue
        E, P, F, perf, et = parse(work, log, mode)
        drift = (max(et) - min(et)) if et else float('nan')
        print(f'{mode} {n} rank(s): |dE| {abs(E - E0):.2e} eV | max |dF| {np.abs(F - F0).max():.2e} eV/A | '
              f'max |dP| {np.abs(P - P0).max():.2e} bar | {perf} steps/s (owner 1 rank {perf0}) | '
              f'NVE Etot span {drift:.2e} eV', flush=True)


if __name__ == '__main__':
    main()
