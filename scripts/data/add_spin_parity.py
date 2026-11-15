"""Add the spin multiplicity 2S+1 to extxyz frames from the electron-count parity: N_e = sum Z - total charge,
2S+1 = 1 for even N_e and 2 for odd N_e (the lowest spin state).  For datasets that store no spin (e.g. the 4G-HDNNP
files); this ASSUMES the reference DFT used the lowest multiplicity.

usage: python -I scripts/data/add_spin_parity.py <in.xyz> <out.xyz> [--charge-key charge] [--spin-key spin]
Prints the counts per (charge, multiplicity).  The input file is not modified.
"""
import argparse
from collections import Counter

import ase.io


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp')
    ap.add_argument('out')
    ap.add_argument('--charge-key', default='charge')
    ap.add_argument('--spin-key', default='spin')
    args = ap.parse_args()
    frames = ase.io.read(args.inp, ':')
    cnt = Counter()
    for a in frames:
        q = int(round(float(a.info.get(args.charge_key, 0))))
        ne = int(a.numbers.sum()) - q
        a.info[args.spin_key] = 1 if ne % 2 == 0 else 2
        cnt[(a.info.get('subset', ''), q, a.info[args.spin_key])] += 1
    ase.io.write(args.out, frames, format='extxyz')
    for (sub, q, s), n in sorted(cnt.items()):
        print(f'{args.out}: {sub} charge {q:+d} multiplicity {s}: {n}')


if __name__ == '__main__':
    main()
