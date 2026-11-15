"""Pull one atom out of a charged cluster beyond the cutoff (local 4G-HDNNP checkpoints; no DFT reference there, so
this checks the FORM of the tail, not accuracy).

    python research/diag/local_4g/dissociation.py <system> <index of the atom to pull> <q of that atom as a fragment>
        [--frame 0] [--rmax 30]
For each model: RACE, RACE_LR with the global constraint (total charge only) and RACE_LR with fragment constraints
(the pulled atom = fragment 1 with charge q, the rest = fragment 0 with Q - q).  Prints E(d) - E(d_max), the charge
on the pulled atom and, for the fragment run, the monopole reference K q (Q - q) / d.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import ase.io

sys.path.insert(0, '.')
from race_v2.inference.predict import load            # noqa: E402

K = 14.399645


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('system')
    ap.add_argument('atom', type=int)
    ap.add_argument('q_atom', type=float)
    ap.add_argument('--frame', type=int, default=0)
    ap.add_argument('--rmax', type=float, default=30.0)
    ap.add_argument('--race', default=None)
    ap.add_argument('--racelr', default=None)
    args = ap.parse_args()
    s = args.system
    a0 = ase.io.read(f'data_local/4ghdnnp/{s}/test.xyz', args.frame)
    Q = int(a0.info['charge'])
    com = a0.positions.mean(0)
    i = args.atom
    u = a0.positions[i] - com; u /= np.linalg.norm(u)
    d0 = np.linalg.norm(a0.positions[i] - np.delete(a0.positions, i, 0), axis=1).min()
    ds = np.concatenate([np.linspace(d0, 8.0, 12), np.linspace(9.0, args.rmax, 12)])
    frames = []
    for d in ds:
        a = a0.copy()
        # move atom i along u until its nearest-neighbour distance is d (rest fixed)
        lo, hi = 0.0, args.rmax + 10
        for _ in range(60):
            t = 0.5 * (lo + hi)
            p = a0.positions[i] + t * u
            if np.linalg.norm(p - np.delete(a0.positions, i, 0), axis=1).min() < d:
                lo = t
            else:
                hi = t
        a.positions[i] = a0.positions[i] + hi * u
        a.arrays['frag'] = (np.arange(len(a)) == i).astype(int)
        a.info.update(charge=Q, fragment_charges=[Q - args.q_atom, args.q_atom])
        a.calc = None
        frames.append(a)
    race = args.race or f'runs/local_4g/{s}_race/ckpt.pkl'
    racelr = args.racelr or f'runs/local_4g/{s}_racelr_pauling_q/ckpt.pkl'
    out = {'d': ds.tolist()}
    p = load(race, species_refit=False)
    out['race'] = (p.predict(frames)[0]).tolist()
    p = load(racelr, species_refit=False)
    e, _ = p.predict(frames); out['lr_global'] = e.tolist()
    out['q_global'] = [float(q[i]) for q in p.charges(frames)]
    cfg = dict(p.cfg); cfg.update(fragment_key='frag', fragment_charges_key='fragment_charges', n_fragments=2)
    pf = load(racelr, config=cfg, species_refit=False)
    e, _ = pf.predict(frames); out['lr_frag'] = e.tolist()
    out['q_frag'] = [float(q[i]) for q in pf.charges(frames)]
    ref = K * args.q_atom * (Q - args.q_atom) / ds
    print(f'{s} frame {args.frame}: pull atom {i} ({a0[i].symbol}), Q = {Q}, fragment charges ({Q - args.q_atom:+g}, '
          f'{args.q_atom:+g}); energies relative to d = {ds[-1]:.1f} A (eV)')
    print('   d(A)  RACE      LR global q_i     LR fragment q_i   K q1 q2/d - ref(dmax)')
    for k, d in enumerate(ds):
        print(f'{d:7.2f} {out["race"][k] - out["race"][-1]:+8.3f} {out["lr_global"][k] - out["lr_global"][-1]:+9.3f} '
              f'{out["q_global"][k]:+6.3f} {out["lr_frag"][k] - out["lr_frag"][-1]:+9.3f} {out["q_frag"][k]:+6.3f} '
              f'{ref[k] - ref[-1]:+9.3f}')
    dst = Path(racelr).parent / f'dissociation_{a0[i].symbol}{i}.json'
    json.dump(out, open(dst, 'w'))
    print('->', dst)


if __name__ == '__main__':
    main()
