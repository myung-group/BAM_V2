"""GPU time of the staged (halo) functions vs the monolithic export function on one halo graph (LiPS supercell,
owned atoms + one shell of image ghosts, edges for owned receivers).  Which part dominates a halo step?

    python runs/diag/staged_profile.py <config.json> <ckpt.pkl> <extxyz> [backend] [rep]
"""
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp


def bench(f, *a, n=10):
    out = f(*a); jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(n):
        out = f(*a)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / n * 1e3


def main():
    cfg_path, ckpt, xyz = sys.argv[1:4]
    backend = sys.argv[4] if len(sys.argv) > 4 else 'cue'
    rep = int(sys.argv[5]) if len(sys.argv) > 5 else 2
    from bam_omat24.export_lammps import build, make_fn, make_stage_fns
    import ase.io
    from ase.neighborlist import neighbor_list
    cfg, model, zs, e0 = build(cfg_path, ckpt, backend, 'float64')
    cutoff = float(cfg.get('cutoff', 5.0))
    a = ase.io.read(xyz, index=0).repeat((rep, rep, rep))
    n = len(a)
    i, j, S, D = neighbor_list('ijSD', a, cutoff)
    gid, ghosts = {}, []
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
    sp = np.zeros(n_pad, np.int32); sp[:n] = [idx[int(z)] for z in a.numbers]; sp[n:n_real] = sp[np.array(ghosts)]
    rc = np.full(e_pad, n_real, np.int32); sd = np.full(e_pad, n_real, np.int32); rv = np.zeros((e_pad, 3))
    rc[:ne], sd[:ne], rv[:ne] = i, snd, D
    g = (jnp.asarray(sp), jnp.asarray(rc), jnp.asarray(sd), jnp.asarray(rv), jnp.int32(n_real), jnp.int32(ne), jnp.int32(n))
    print(f"halo graph: {n} owned + {len(ghosts)} ghosts, {ne} edges, bucket {n_pad}/{e_pad}, backend {backend}", flush=True)
    mono = jax.jit(make_fn(model, cutoff, n_pad, e_pad))
    print(f"monolithic energy+dE/dr (same bucket, all nodes):  {bench(mono, *g):8.2f} ms", flush=True)
    fns = make_stage_fns(model, cutoff, n_pad, e_pad)
    dims, da = fns['dims'], fns['dim_a']
    pre = jax.jit(fns['prelude']); h0, A, _ = pre(*g)
    print(f"prelude:                                  {bench(pre, *g):8.2f} ms", flush=True)
    prev = jax.jit(fns['prelude_vjp'])
    print(f"prelude_vjp:                              {bench(prev, *g, jnp.zeros((n_pad, da)), jnp.zeros((e_pad, 3))):8.2f} ms")
    for li in (0, 2):
        h = jnp.zeros((n_pad, dims[li])) + 0.01
        hpp = jnp.zeros((n_pad, dims[li - 1])) if (fns['cheb'] and li >= 2) else jnp.zeros((n_pad, 1))
        lf = jax.jit(fns['layers'][li]); lv = jax.jit(fns['layers_vjp'][li])
        ct = jnp.ones((n_pad, dims[li + 1])) * 1e-3
        print(f"layer {li}:                                  {bench(lf, h, hpp, A, *g):8.2f} ms", flush=True)
        print(f"layer {li} vjp (+accumulators):              "
              f"{bench(lv, h, hpp, A, *g, ct, jnp.zeros_like(h), jnp.zeros((n_pad, da)), jnp.zeros((e_pad, 3))):8.2f} ms",
              flush=True)

    # prelude-only cost inside a layer: the layer function with the layer step removed
    from bam_omat24.export_lammps import _graph
    import e3nn_jax as e3nn
    from bam_omat24.race_nnx import _dense

    def ctx_only(sp_, rc_, sd_, rv_, nr, er, nl):
        gg, pos0, cell, cut = _graph(sp_, rc_, sd_, rv_, nr, er, n_pad, e_pad, cutoff)
        ctx = model._prelude(pos0, cell, cut, gg, atomic_basis=e3nn.IrrepsArray(_dense(model._prelude(
            pos0, cell, cut, gg)['atomic_basis']).irreps, A), with_pre_strata=False)
        return jax.tree.map(lambda x: jnp.sum(x.array if hasattr(x, 'array') else x) if x is not None else 0.0,
                            {k: ctx[k] for k in ('radial_basis', 'sh', 'shell_env') if k in ctx})
    print(f"prelude part recomputed in every layer:   {bench(jax.jit(ctx_only), *g):8.2f} ms", flush=True)


if __name__ == '__main__':
    main()
