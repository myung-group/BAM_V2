"""LAMMPS step 4: pair_style race vs Python on a LiPS test frame (run 0: energy, forces, pressure tensor), then a
1 ps NVE run (energy drift, speed).  Run on a GPU node after lammps/build_lammps.sh.

    python research/diag/lammps_validate.py <lmp binary> <plugin.so> <export_dir> <config.json> <ckpt.pkl> <extxyz>
"""
import os
import re
import subprocess
import sys

import numpy as np

EV_A3_TO_BAR = 1.602176634e6


def main():
    lmp, plugin, exp_dir, cfg_path, ckpt, xyz = sys.argv[1:7]
    import ase.io
    from ase.data import atomic_numbers, chemical_symbols
    work = os.path.join(exp_dir, 'lammps_check'); os.makedirs(work, exist_ok=True)
    a = ase.io.read(xyz, index=0)
    zs = sorted(set(int(z) for z in a.numbers))
    syms = [chemical_symbols[z] for z in zs]
    ase.io.write(os.path.join(work, 'frame.data'), a, format='lammps-data', specorder=syms, masses=True,
                 atom_style='atomic')
    inp = f"""units metal
atom_style atomic
atom_modify map yes
boundary p p p
read_data frame.data
pair_style race {plugin} {os.path.abspath(exp_dir)} 0
pair_coeff * * {' '.join(syms)}
neighbor 1.0 bin
thermo_style custom step pe pxx pyy pzz pxy pxz pyz
thermo_modify format float %20.12f
dump d all custom 1 forces.dump id fx fy fz
dump_modify d format float %20.14f sort id
run 0
undump d
velocity all create 520.0 4928459 dist gaussian
timestep 0.00025
fix 1 all nve
thermo_style custom step temp pe ke etotal
thermo 400
run 4000
"""
    open(os.path.join(work, 'in.racev2'), 'w').write(inp)
    r = subprocess.run([lmp, '-in', 'in.racev2', '-log', 'log.lammps', '-screen', 'none'], cwd=work,
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:]); sys.exit(1)
    log = open(os.path.join(work, 'log.lammps')).read()
    # run 0 thermo line: step pe pxx pyy pzz pxy pxz pyz
    m = re.search(r"Step\s+PotEng\s+Pxx.*\n\s*0\s+(.+)\n", log)
    vals = [float(v) for v in m.group(1).split()]
    pe, p = vals[0], np.array(vals[1:7])                     # bar, LAMMPS order xx yy zz xy xz yz
    fl = open(os.path.join(work, 'forces.dump')).read().split('ITEM: ATOMS id fx fy fz')[1].split('ITEM')[0]
    F = np.array([[float(x) for x in line.split()[1:]] for line in fl.strip().splitlines()])

    # LAMMPS needs a lower-triangular cell: the data writer rotated the frame, so the reference is computed on the
    # frame read back from the data file (energies are rotation invariant, forces / pressure are not)
    a = ase.io.read(os.path.join(work, 'frame.data'), format='lammps-data', atom_style='atomic',
                    Z_of_type={k + 1: z for k, z in enumerate(zs)})
    a.pbc = True
    # reference: the exported function through JAX (same graph construction as the Python check) + E0
    import jax.numpy as jnp
    from race_v2.export.lammps import build, make_fn, graph_inputs
    cfg, model, zs_m, e0 = build(cfg_path, ckpt, 'e3nn')
    import jax
    fn = jax.jit(make_fn(model, float(cfg.get('cutoff', 5.0)), 128, 4096))
    ins = graph_inputs(a, zs_m, float(cfg.get('cutoff', 5.0)), 128, 4096)
    e, ne, g = fn(*[jnp.asarray(x) for x in ins])  # ins[6] = n_local = n
    n, ned = int(ins[4]), int(ins[5])
    rec, snd, rv, g = ins[1][:ned], ins[2][:ned], ins[3][:ned], np.asarray(g)[:ned]
    Fr = np.zeros((n, 3)); np.add.at(Fr, rec, g); np.add.at(Fr, snd, -g)
    Er = float(e) + sum(e0[zs_m.index(int(z))] for z in a.numbers)
    S = (rv[:, :, None] * g[:, None, :]).sum(0) / a.get_volume()      # ASE stress (eV/A^3)
    P_ref = -EV_A3_TO_BAR * np.array([S[0, 0], S[1, 1], S[2, 2], S[0, 1], S[0, 2], S[1, 2]])
    print(f"run 0: E lammps {pe:.10f} ref {Er:.10f} |dE| {abs(pe - Er):.2e} eV | max |dF| {np.abs(F - Fr).max():.2e} eV/A "
          f"| max |dP| {np.abs(p - P_ref).max():.2e} bar (max |P| {np.abs(P_ref).max():.1f} bar)")
    # NVE drift and speed
    rows = re.findall(r"^\s*(\d+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*$", log, re.M)
    et = np.array([float(x[4]) for x in rows if int(x[0]) >= 0])
    if et.size > 1:
        print(f"NVE 1 ps (dt 0.25 fs, 520 K start): Etot first {et[0]:.6f} last {et[-1]:.6f} max dev {np.abs(et - et[0]).max():.2e} eV "
              f"({1e3 * np.abs(et - et[0]).max() / n:.3f} meV/atom)")
    perf = re.findall(r"Performance: (.*)", log)
    if perf:
        print('LAMMPS performance:', perf[-1])


if __name__ == '__main__':
    main()
