"""mem_bench.py <ckpt> <out.json>: time and peak GPU memory of one energy+force evaluation versus system size (one GPU).
Supercells of a PhononDB structure (rattled 0.02 A), sizes up to --max-atoms; stops at the first failure (out of memory).
Precision follows the checkpoint config, or float64 with BAM_PREDICT_X64=1."""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.getcwd())
import ase.io
ckpt, out = sys.argv[1], sys.argv[2]
sizes = [int(x) for x in (sys.argv[3].split(',') if len(sys.argv) > 3 else '64,128,256,512,1024,2048,4096,8192,16384'.split(','))]
import jax
from race_v2.inference.predict import Predictor
p = Predictor(ckpt, species_refit=False, batch_size=1, max_atoms=10 ** 9)
dev = jax.devices()[0]
P = '/scratch/t510a13/data/mbd/phonons/2024-11-09-phononDB-PBE-103-structures.extxyz'
rows = []
for name, idx in (('first', 0), ('dense', 40)):
    base = ase.io.read(P, str(idx))
    for n in sizes:
        k = max(1, round((n / len(base)) ** (1 / 3)))
        a = base.repeat((k, k, k)); a.rattle(stdev=0.02, seed=1)
        from ase.neighborlist import neighbor_list
        ne = len(neighbor_list('i', a, float(p.cfg['cutoff'])))
        rec = dict(base=name, formula=base.get_chemical_formula(), n_atoms=len(a), n_edges=int(ne), edges_per_atom=ne / len(a),
                   dtype='float64' if p.cfg.get('x64') else 'float32')
        try:
            t = []
            for r in range(3):
                t0 = time.time(); e, f = p.predict([a]); t.append(time.time() - t0)
            ms = dev.memory_stats() or {}
            rec.update(first_call_s=t[0], call_s=min(t[1:]), peak_gb=ms.get('peak_bytes_in_use', float('nan')) / 1e9,
                       in_use_gb=ms.get('bytes_in_use', float('nan')) / 1e9, e_per_atom=float(e[0]) / len(a))
            print(json.dumps(rec), flush=True); rows.append(rec)
        except Exception as ex:                                              # noqa: BLE001
            rec.update(error=f'{type(ex).__name__}: {str(ex)[:160]}'); print(json.dumps(rec), flush=True); rows.append(rec)
            break
json.dump(rows, open(out, 'w'), indent=1)
