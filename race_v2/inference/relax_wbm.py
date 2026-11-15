"""WBM-style relaxation of many structures with a RACEv2 checkpoint (Matbench Discovery protocol: FIRE, at most 500
steps, fmax 0.05 eV/A on the FrechetCellFilter gradient), batched over structures (relax_batched.py).

    python -m race_v2.inference.relax_wbm --ckpt runs/<name>_cue/ckpt_<name>_cue.pkl \
        --input <2024-08-04-wbm-initial-atoms.extxyz.zip | file.extxyz> --out-dir <dir> \
        [--shard 0 --num-shards 8] [--limit N] [--max-structures 64] [--groups 2] [--host-workers 16]

One process drives one GPU; run one process per GPU with --shard i --num-shards N (CRC32 of the material id, the
sharding of BAM-GAMP-jax scripts/relax_wbm.py).  Output: <out-dir>/relaxed_<i>of<N>.jsonl, one JSON object per
structure with the keys of that script (material_id, n_atoms, n_steps, converged, initial_energy, final_energy,
numbers, final_positions, final_cell, final_forces, final_stress, wall_seconds, energy_convention), appended and
flushed as structures finish; a rerun skips the material ids already in the file.  Failures go to
<out-dir>/failed_<i>of<N>.jsonl.  Energies are total energies on the scale of the training label (model + E0).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile
import zlib
from pathlib import Path


def _shard(mid: str, n: int) -> int:
    return zlib.crc32(mid.encode('utf-8')) % max(n, 1)


def iter_structures(path, keep):
    """(material_id, ase.Atoms) of an extxyz zip (one member per structure, as Matbench Discovery ships WBM) or of a
    plain extxyz file (info['material_id']); keep(material_id) is applied before parsing where the id is known."""
    import ase.io
    if str(path).endswith('.zip'):
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                mid = Path(name).stem
                if not name.endswith('.extxyz') or not keep(mid):
                    continue
                atoms = ase.io.read(io.StringIO(zf.read(name).decode()), format='extxyz')
                yield str(atoms.info.get('material_id', mid)), atoms
    else:
        for i, atoms in enumerate(ase.io.iread(str(path), format='extxyz')):
            mid = str(atoms.info.get('material_id', i))
            if keep(mid):
                yield mid, atoms


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ckpt', required=True); ap.add_argument('--input', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--fmax', type=float, default=0.05); ap.add_argument('--max-steps', type=int, default=500)
    ap.add_argument('--max-structures', type=int, default=64); ap.add_argument('--groups', type=int, default=2)
    ap.add_argument('--atom-budget', type=int, default=3000); ap.add_argument('--edge-budget', type=int, default=120000)
    ap.add_argument('--host-workers', type=int, default=16); ap.add_argument('--log-every', type=float, default=60.0)
    a = ap.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.path.join(os.path.dirname(here), 'data'), here):       # stream_worker is in race_v2/data
        if d not in sys.path:
            sys.path.insert(0, d)
    import relax_batched as RB                   # top-level module: the host workers must stay JAX-free
    import stream_worker
    stream_worker.clean_main()                   # spawned workers do not re-import this (JAX) module

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = f'{a.shard}of{a.num_shards}'
    res_path, fail_path = out / f'relaxed_{tag}.jsonl', out / f'failed_{tag}.jsonl'
    done = set()
    if res_path.exists():
        with open(res_path) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)['material_id'])
                except (json.JSONDecodeError, KeyError):
                    pass                          # a line cut by a kill: that structure is relaxed again

    def log(msg):
        print(msg, flush=True)

    t0 = time.time()
    ev = RB.RaceEvaluator(a.ckpt, max_graphs=a.max_structures)
    log(f"[relax] ckpt {a.ckpt}  cutoff {ev.cutoff}  dtype {ev.dtype.__name__}  shard {tag}  resume {len(done)} done  "
        f"model ready in {time.time() - t0:.0f} s")
    drv = RB.BatchedRelaxDriver(ev, fmax=a.fmax, max_steps=a.max_steps, max_structures=a.max_structures,
                                groups=a.groups, atom_budget=a.atom_budget, edge_budget=a.edge_budget,
                                host_workers=a.host_workers, log=log, log_every=a.log_every)

    def items():
        n = 0
        for mid, atoms in iter_structures(a.input, lambda m: _shard(m, a.num_shards) == a.shard and m not in done):
            if a.limit is not None and n >= a.limit:
                return
            n += 1
            yield mid, atoms

    n_ok = n_fail = 0
    with open(res_path, 'a') as fres, open(fail_path, 'a') as ffail:
        for oc in drv.run(items()):
            if oc.result is not None:
                r = dict(oc.result); r['material_id'] = oc.key; r['energy_convention'] = 'dft_total_uncorrected'
                fres.write(json.dumps(r) + '\n'); fres.flush(); n_ok += 1
            else:
                ffail.write(json.dumps({'material_id': oc.key, 'error': oc.error}) + '\n'); ffail.flush(); n_fail += 1
    log(f"[relax] shard {tag}: {n_ok} relaxed, {n_fail} failed, {time.time() - t0:.0f} s, shapes {sorted(ev.shapes)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
