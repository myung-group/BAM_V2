"""Where the time of one MD force call goes (md_diatomics.RaceV2Calculator), and whether a neighbour list with a skin
gives the same energies, forces and stresses.  One GPU, a few minutes; same environment as md_diatomics.py.

    python md_profile.py --ckpt <pkl> [--systems A B ...] [--repeats 40] [--skin 0.5]

Per system (first frame of the DynaMat reference, positions displaced a little at every repeat):
  neighbours   neighbor_pairs (BAM_NEIGHBORS backend)
  payload      race_payload without the neighbour search (array conversions)
  collate      _cast + _to_graph + fixed_collate_fn (numpy padding)
  call         jitted model call with the numpy batch + results back on the host
  device       the same call with the batch already on the device (GPU computation alone)
  pos-only     device batch reused, only the positions sent again (what a skin list would do between rebuilds)
  total        RaceEvaluator.evaluate (what the calculator does at every MD step)
Skin check: the list of cutoff + skin with the model cutoff unchanged against the exact list.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import md_diatomics as X  # noqa: E402


def tm(fn, n):
    ts = []
    for i in range(n):
        t0 = time.perf_counter(); fn(i); ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--ref-file')
    ap.add_argument('--systems', nargs='*', default=['bulkLiMgAlZnSn_600K_J_Schmidt_VASP', 'bulkAu_1500K_Kapil',
                                                     'bulkAg_600K_Kapil', 'anthracene_293K_Sharma_S'])
    ap.add_argument('--repeats', type=int, default=40); ap.add_argument('--skin', type=float, default=0.5)
    a = ap.parse_args()
    X._paths()
    import jax
    import jax.numpy as jnp
    import relax_batched as RB
    from neighbors import neighbor_pairs
    from matbench_discovery.md import default_md_reference_path, read_reference_trajectory
    from race_v2.data.loaders import fixed_collate_fn
    from race_v2.data.stream_data import _cast, _to_graph
    ref = a.ref_file or default_md_reference_path()
    ev = RB.RaceEvaluator(a.ckpt, max_graphs=2)
    prep = ev.prepare_fn()
    z2i = {int(z): i for i, z in enumerate(ev.atomic_numbers)}
    print(f"dtype {np.dtype(ev.dtype).name}  cutoff {ev.cutoff}  neighbours {os.environ.get('BAM_NEIGHBORS', 'ase')}  "
          f"device {jax.devices()[0].device_kind}  repeats {a.repeats}", flush=True)
    rng = np.random.default_rng(0)
    for name in a.systems:
        atoms0 = read_reference_trajectory(ref, name, max_frames=4)[0].frame_as_atoms(0)
        frames = []
        for _ in range(a.repeats + 3):
            at = atoms0.copy(); at.positions += rng.normal(0, 0.01, at.positions.shape); frames.append(at)

        def batch_of(p):
            n, e = int(p['n_node'][0]), int(p['n_edge'][0])
            shape = (RB._rung(n + 1, ev.node_rung), RB._rung(max(e, 1), ev.edge_rung))
            g = _to_graph(_cast({k: v for k, v in p.items() if k != 'numbers'}, ev.dtype))
            return fixed_collate_fn([g], ev.max_graphs, shape[0], shape[1]), shape

        pay = [prep(at) for at in frames]
        b0, shape = batch_of(pay[0])
        out = ev._step(b0); np.asarray(out['forces'])                       # compilation
        n, e = int(pay[0]['n_node'][0]), int(pay[0]['n_edge'][0])
        t_nl = tm(lambda i: neighbor_pairs(frames[i], ev.cutoff), a.repeats)
        t_prep = tm(lambda i: prep(frames[i]), a.repeats)
        t_col = tm(lambda i: batch_of(pay[i]), a.repeats)
        batches = [batch_of(p)[0] for p in pay[:a.repeats] if batch_of(p)[1] == shape]

        def call(i):
            o = ev._step(batches[i % len(batches)])
            return np.asarray(o['energy']), np.asarray(o['forces']), np.asarray(o['stress'])
        t_call = tm(call, a.repeats)
        dev = jax.tree.map(jnp.asarray, b0)
        jax.block_until_ready(ev._step(dev)['forces'])

        def device(i):
            jax.block_until_ready(ev._step(dev))
        t_dev = tm(device, a.repeats)
        pos_pad = np.array(b0.nodes['positions'])

        def pos_only(i):
            pp = pos_pad.copy(); pp[:n] = frames[i].positions
            o = ev._step(dev._replace(nodes={**dev.nodes, 'positions': jnp.asarray(pp.astype(ev.dtype))}))
            return np.asarray(o['energy']), np.asarray(o['forces']), np.asarray(o['stress'])
        pos_only(0)
        t_pos = tm(pos_only, a.repeats)
        t_tot = tm(lambda i: ev.evaluate([prep(frames[i])]), a.repeats)
        print(f"\n{name}: atoms {n}, edges {e} ({e / n:.1f} per atom), padded shape {shape} "
              f"({100 * (shape[1] / e - 1):.0f} % edge padding)")
        print(f"  ms per call (median): neighbours {t_nl:.1f} | payload {t_prep - t_nl:.1f} | collate {t_col:.1f} | "
              f"call {t_call:.1f} | device {t_dev:.1f} | pos-only {t_pos:.1f} | total {t_tot:.1f}", flush=True)
        # skin list against the exact list (same structure)
        at = frames[0]
        e_ref, f_ref, s_ref = ev.evaluate([prep(at)])[0]
        ps = RB.race_payload(at, cutoff=ev.cutoff + a.skin, z_to_index=z2i)
        ps['cutoff'] = np.array([ev.cutoff], np.float64)
        e_s, f_s, s_s = ev.evaluate([ps])[0]
        print(f"  skin {a.skin} A: edges {int(ps['n_edge'][0])} (+{100 * (int(ps['n_edge'][0]) / e - 1):.0f} %), "
              f"|dE| {abs(e_s - e_ref):.3e} eV, max |dF| {np.abs(f_s - f_ref).max():.3e} eV/A, "
              f"max |dS| {np.abs(s_s - s_ref).max():.3e} eV/A^3   (|F| max {np.abs(f_ref).max():.3f})", flush=True)
        # the same structure twice with the exact list: run-to-run noise of the GPU sums
        e_2, f_2, s_2 = ev.evaluate([prep(at)])[0]
        print(f"  repeat of the exact list: |dE| {abs(e_2 - e_ref):.3e}, max |dF| {np.abs(f_2 - f_ref).max():.3e}, "
              f"max |dS| {np.abs(s_2 - s_ref).max():.3e}", flush=True)
    print(f"\npadded shapes compiled: {sorted(ev.shapes)}")


if __name__ == '__main__':
    main()
