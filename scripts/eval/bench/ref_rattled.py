"""ref_rattled.py <ckpt> <out.npz> [n]: JAX energies / forces of the first n PhononDB structures rattled by 0.05 A
(seed 1), for the comparison with the PyTorch port (run with BAM_PREDICT_X64=1 for a float64 reference)."""
import os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
import ase.io
from race_v2.inference.predict import Predictor
n = int(sys.argv[3]) if len(sys.argv) > 3 else 16
fr = ase.io.read('/scratch/t510a13/data/mbd/phonons/2024-11-09-phononDB-PBE-103-structures.extxyz', f':{n}')
for a in fr:
    a.rattle(stdev=0.05, seed=1)
p = Predictor(sys.argv[1], species_refit=False)
e, f = p.predict(fr)
np.savez(sys.argv[2], e=np.asarray(e), f=np.concatenate([np.asarray(x) for x in f]), x64=bool(p.cfg.get('x64')))
print('saved', sys.argv[2], 'x64', p.cfg.get('x64'), 'E[:3]', np.asarray(e)[:3], 'max |F|', float(np.abs(np.concatenate(f)).max()))
