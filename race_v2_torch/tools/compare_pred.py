"""compare_pred.py <model.pt> <jax_pred.npz> [n] [rattle stdev]: energies/forces of the first n PhononDB structures with the torch
port against JAX predictions saved by mergecheck_tools/predict_dump.py (e: total energies, f: all forces)."""
import sys, time
import numpy as np, ase.io
from race_v2_torch.calculator import RACEv2Calculator
ref = np.load(sys.argv[2]); n = int(sys.argv[3]) if len(sys.argv) > 3 else len(ref['e'])
fr = ase.io.read('/scratch/t510a13/data/mbd/phonons/2024-11-09-phononDB-PBE-103-structures.extxyz', f':{n}')
if len(sys.argv) > 4:                                # same displacement as scripts/eval/bench/ref_rattled.py
    for a in fr:
        a.rattle(stdev=float(sys.argv[4]), seed=1)
calc = RACEv2Calculator(sys.argv[1]); E, F = [], []; t0 = time.time()
for a in fr:
    a.calc = calc; E.append(a.get_potential_energy()); F.append(a.get_forces())
E = np.array(E); F = np.concatenate(F); nat = np.array([len(a) for a in fr])
rf = ref['f'][:len(F)]
print(f'{n} structures ({nat.sum()} atoms), {time.time() - t0:.0f} s')
print('max |dE| %.3e eV  (per atom max %.3e eV/atom), max |dF| %.3e eV/A, max |F| %.2f eV/A' %
      (abs(E - ref['e'][:n]).max(), (abs(E - ref['e'][:n]) / nat).max(), abs(F - rf).max(), abs(rf).max()))
print('E torch', E[:3], '\nE jax  ', ref['e'][:3])
