"""Staged (per-layer halo exchange) RACEv2 vs the monolithic export function.

1. the monolithic function after the node_energies split reproduces the case files written before the split;
2. a periodic frame as LAMMPS would see it on one rank in halo mode (owned atoms + one shell of image ghosts,
   edges only for owned receivers, ghost rows refreshed from their owners between layers, ghost cotangents added
   back to the owners) gives the same energy and forces as the monolithic owner-mode graph.

    python runs/diag/staged_check.py <config.json> <ckpt.pkl> <extxyz> <old_case_dir> [backend] [rep]
"""
import sys

import numpy as np
import jax
import jax.numpy as jnp


def main():
    cfg_path, ckpt, xyz, case_dir = sys.argv[1:5]
    backend = sys.argv[5] if len(sys.argv) > 5 else 'cue'
    rep = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    from bam_omat24.export_lammps import build, make_fn, graph_inputs, make_stage_fns, staged_energy_forces
    import ase.io
    from ase.neighborlist import neighbor_list
    cfg, model, zs, e0 = build(cfg_path, ckpt, backend, 'float64')
    cutoff = float(cfg.get('cutoff', 5.0))

    # 1. monolithic export function vs the cases written before the refactor
    for k in range(3):
        with open(f'{case_dir}/case_{k}.bin', 'rb') as fh:
            n_pad, e_pad, n, e, nl = np.fromfile(fh, np.int32, 5)
            sp = np.fromfile(fh, np.int32, n_pad); rc = np.fromfile(fh, np.int32, e_pad)
            sd = np.fromfile(fh, np.int32, e_pad); rv = np.fromfile(fh, np.float64, 3 * e_pad).reshape(e_pad, 3)
            et = np.fromfile(fh, np.float64, 1)[0]; ea = np.fromfile(fh, np.float64, n_pad)
            g = np.fromfile(fh, np.float64, 3 * e_pad).reshape(e_pad, 3)
        fn = jax.jit(make_fn(model, cutoff, int(n_pad), int(e_pad)))
        e1, ea1, g1 = fn(jnp.asarray(sp), jnp.asarray(rc), jnp.asarray(sd), jnp.asarray(rv), jnp.int32(n), jnp.int32(e),
                         jnp.int32(nl))
        print(f"monolithic after split vs pre-split case {k}: |dE| {abs(float(e1) - et):.2e} "
              f"|dE_atom| {np.abs(np.asarray(ea1)[:n] - ea[:n]).max():.2e} |d dE/dr| {np.abs(np.asarray(g1)[:e] - g[:e]).max():.2e}",
              flush=True)

    # 2. staged halo exchange on one rank vs monolithic owner mode
    a = ase.io.read(xyz, index=0).repeat((rep, 1, 1))
    n = len(a)
    i, j, S, D = neighbor_list('ijSD', a, cutoff)
    ghosts, gid = [], {}
    snd = np.empty(len(i), np.int64)
    for q in range(len(i)):
        if not S[q].any():
            snd[q] = j[q]
        else:
            key = (int(j[q]), *map(int, S[q]))
            if key not in gid:
                gid[key] = n + len(ghosts); ghosts.append(int(j[q]))
            snd[q] = gid[key]
    n_real, ne = n + len(ghosts), len(i)
    n_pad = int(2 ** np.ceil(np.log2(n_real + 1))); e_pad = int(2 ** np.ceil(np.log2(ne)))
    idx = {z: s for s, z in enumerate(sorted(zs))}
    sp = np.zeros(n_pad, np.int32); sp[:n] = [idx[int(z)] for z in a.numbers]
    sp[n:n_real] = sp[np.array(ghosts)]
    rc = np.full(e_pad, n_real, np.int32); sd = np.full(e_pad, n_real, np.int32); rv = np.zeros((e_pad, 3))
    rc[:ne], sd[:ne], rv[:ne] = i, snd, D
    print(f"halo graph: {n} owned + {len(ghosts)} ghosts, {ne} edges (pad {n_pad}/{e_pad})", flush=True)
    fns = make_stage_fns(model, cutoff, n_pad, e_pad)
    fns['prelude'] = jax.jit(fns['prelude']); fns['prelude_vjp'] = jax.jit(fns['prelude_vjp'])
    fns['layers'] = [jax.jit(f) for f in fns['layers']]; fns['layers_vjp'] = [jax.jit(f) for f in fns['layers_vjp']]
    e_st, g_st = staged_energy_forces(fns, jnp.asarray(sp), jnp.asarray(rc), jnp.asarray(sd), jnp.asarray(rv),
                                      jnp.int32(n_real), jnp.int32(ne), jnp.int32(n), np.array(ghosts))
    g_st = np.asarray(g_st)[:ne]
    F = np.zeros((n_real, 3)); np.add.at(F, i, g_st); np.add.at(F, snd, -g_st)
    np.add.at(F, np.array(ghosts), F[n:n_real]); F = F[:n]
    # monolithic owner mode
    key = (int(2 ** np.ceil(np.log2(n + 1))), int(2 ** np.ceil(np.log2(ne))))
    ins = graph_inputs(a, zs, cutoff, *key)
    fn = jax.jit(make_fn(model, cutoff, *key))
    e_m, _, g_m = fn(*[jnp.asarray(x) for x in ins])
    g_m = np.asarray(g_m)[:int(ins[5])]
    Fm = np.zeros((n, 3)); np.add.at(Fm, ins[1][:int(ins[5])], g_m); np.add.at(Fm, ins[2][:int(ins[5])], -g_m)
    print(f"staged halo vs monolithic: E {float(e_st):.10f} vs {float(e_m):.10f} |dE| {abs(float(e_st) - float(e_m)):.2e} | "
          f"max |dF| {np.abs(F - Fm).max():.2e} eV/A (max |F| {np.abs(Fm).max():.3f})", flush=True)
    # 4. owned-rows-only variant (k_pad >= owned atoms): same energy and forces
    k_pad = int(2 ** np.ceil(np.log2(n + 1)))
    fk = make_stage_fns(model, cutoff, n_pad, e_pad, k_pad=k_pad)
    fk['prelude'] = jax.jit(fk['prelude']); fk['prelude_vjp'] = jax.jit(fk['prelude_vjp'])
    fk['layers'] = [jax.jit(f) for f in fk['layers']]; fk['layers_vjp'] = [jax.jit(f) for f in fk['layers_vjp']]
    e_k, g_k = staged_energy_forces(fk, jnp.asarray(sp), jnp.asarray(rc), jnp.asarray(sd), jnp.asarray(rv),
                                    jnp.int32(n_real), jnp.int32(ne), jnp.int32(n), np.array(ghosts))
    g_k = np.asarray(g_k)[:ne]
    print(f"staged owned-rows-only (k_pad {k_pad} of n_pad {n_pad}) vs full rows: |dE| {abs(float(e_k) - float(e_st)):.2e} | "
          f"max |d dE/dr| {np.abs(g_k - g_st).max():.2e}", flush=True)
    # 3. saved-residual variant (forward returns the VJP residuals, backward consumes them)
    fns['layers_res'] = [(jax.jit(fw), jax.jit(bw), h) for fw, bw, h in fns['layers_res']]
    e_r, g_r = staged_energy_forces(fns, jnp.asarray(sp), jnp.asarray(rc), jnp.asarray(sd), jnp.asarray(rv),
                                    jnp.int32(n_real), jnp.int32(ne), jnp.int32(n), np.array(ghosts), residuals=True)
    g_r = np.asarray(g_r)[:ne]
    print(f"staged residual mode vs recompute mode: |dE| {abs(float(e_r) - float(e_st)):.2e} | "
          f"max |d dE/dr| {np.abs(g_r - g_st).max():.2e} | residual arrays per layer "
          f"{[h['n_res'] for _, _, h in fns['layers_res']]}", flush=True)


if __name__ == '__main__':
    main()
