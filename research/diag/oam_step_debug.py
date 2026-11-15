"""Training-step NaN hunt for the OAM subset runs: N steps on real fixed-shape batches (as the trainer builds them),
SOAP vs AMSGrad, reporting the first step whose loss, gradient, update or parameters are not finite, with the
offending leaves, and the energies of real vs padding graphs at that step.

    python research/diag/oam_step_debug.py research/input/input_3bpa_omat24_<run>_cue.json [n_steps] [name:key=val,key=val ...]
variants default: soap (config as is) and amsgrad; values are parsed as JSON when possible.
"""
import functools
import json
import math
import sys

import numpy as np
import jax
import jax.numpy as jnp
import jraph
import optax
from flax import nnx

from race_v2.training import train as T
from race_v2.data.loaders import BucketedDataLoader, fixed_collate_fn
from race_v2.training.utils import compute_loss


def bad_leaves(tree, n=5):
    leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
    return [jax.tree_util.keystr(p) for p, v in leaves if not bool(jnp.all(jnp.isfinite(v)))][:n]


def main():
    cfg0 = json.load(open(sys.argv[1]))
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    variants = {}
    for spec in sys.argv[3:]:
        name, _, kv = spec.partition(':')
        d = {}
        for item in filter(None, kv.split(',')):
            k, v = item.split('=')
            try:
                d[k] = json.loads(v)
            except json.JSONDecodeError:
                d[k] = v
        variants[name] = d
    if not variants:
        variants = {'soap': {}, 'amsgrad': {'optimizer': 'amsgrad'}}
    fout = sys.stdout
    (zs, e0, tr, va, tests, stats) = T.build_datasets(cfg0, fout)
    bs = int(cfg0['batch_size'])
    ntr = np.array([int(g.n_node[0]) for g in tr.graphs]); etr = np.array([int(g.n_edge[0]) for g in tr.graphs])
    pad_n = int(math.ceil(max(bs * ntr.mean(), ntr.max() + 1) / 64) * 64)
    pad_e = int(math.ceil(max(bs * etr.mean(), etr.max()) / 1024) * 1024)
    print('pads', pad_n, pad_e, flush=True)
    sw = float(cfg0.get('stress_weight', 10.0))
    for opt, over in variants.items():
        cfg = dict(cfg0); cfg.update(over)
        print(f"[{opt}] overrides {over}", flush=True)
        jax.config.update('jax_default_matmul_precision', cfg.get('matmul_precision', None))
        if cfg.get('soap_state_dtype', None) == 'float64':
            jax.config.update('jax_enable_x64', True)        # optimizer state only; params / graphs float32
        if cfg.get('x64', False) and not np.asarray(tr.graphs[0].nodes['positions']).dtype == np.float64:
            # float64 as in train_3bpa (params cast in build_model, graphs cast here); keep x64 variants last
            jax.config.update('jax_enable_x64', True)
            tr.graphs = T._cast_graphs(tr.graphs, np.float64)
        graphdef, params, optimizer, schedule, gc = T.build_model(cfg, len(zs), e0, stats, fout)
        opt_state = optimizer.init(params)
        loader = BucketedDataLoader(tr, batch_size=int(cfg.get('pack_max_graphs', 64)), n_buckets=8, shuffle=True,
                                    drop_last=False, rngs=nnx.Rngs(0), max_nodes_per_batch=pad_n,
                                    max_edges_per_batch=pad_e)
        loader.collate_fn = functools.partial(fixed_collate_fn, n_node_pad=pad_n, n_edge_pad=pad_e)

        @jax.jit
        def step(params, opt_state, batch):
            def lf(p):
                loss, aux = compute_loss(graphdef, p, batch, 1.0, 10.0, sw, 10.0, 'huber', True, 'scalar6', 0.0)
                return loss
            loss, g = jax.value_and_grad(lf)(params)
            g = optax.clip_by_global_norm(1.0).update(g, optax.EmptyState())[0]
            upd, new_state = optimizer.update(g, opt_state, params)
            new_params = optax.apply_updates(params, upd)
            e = nnx.merge(graphdef, params)(batch)['energy']
            return new_params, new_state, loss, g, upd, e

        p0 = jax.tree.map(lambda x: x, params)
        zero = jax.tree.map(lambda x: jnp.ones(x.shape, bool), params)      # gradient exactly 0 at every step so far
        upd_zero = jax.jit(lambda z, g: jax.tree.map(lambda a, b: a & (b == 0), z, g))

        @jax.jit
        def drift(p, p0, z):
            d = jax.tree.map(lambda a, b, m: (jnp.max(jnp.where(m, jnp.abs(a - b), 0.0)) if a.size
                                              else jnp.zeros((), a.dtype)).astype(jnp.float32), p, p0, z)
            n = jax.tree.map(lambda m: jnp.sum(m), z)
            return jnp.max(jnp.stack(jax.tree.leaves(d))), jnp.sum(jnp.stack(jax.tree.leaves(n)))

        import time
        t_first = t_last = None
        clean = True
        for k, batch in enumerate(loader):
            if k >= n_steps:
                break
            new_params, new_state, loss, g, upd, e = step(params, opt_state, batch)
            zero = upd_zero(zero, g)
            gm = np.asarray(jraph.get_graph_padding_mask(batch)); e = np.asarray(e)
            if k == 5:
                t_first = time.time()
            t_last = time.time()
            problems = {}
            if not np.isfinite(float(loss)):
                problems['loss'] = float(loss)
            for name, tree in (('grad', g), ('update', upd), ('params', new_params)):
                b = bad_leaves(tree)
                if b:
                    problems[name] = b
            n_real = int(gm.sum())
            line = (f"[{opt}] step {k} graphs {n_real} loss {float(loss):.4g} E nan real {int(np.isnan(e[gm]).sum())} "
                    f"pad {int(np.isnan(e[~gm]).sum())}/{int((~gm).sum())} max|E| real {np.nanmax(np.abs(e[gm])):.3g}")
            if problems or k % 10 == 0:
                dz, nz = drift(new_params, p0, zero)
                line += f" | zero-grad entries {int(nz)} max drift {float(dz):.3g}"
                print(line, problems if problems else '', flush=True)
            if problems:
                clean = False
                break
            params, opt_state = new_params, new_state
        if clean:
            dz, nz = drift(params, p0, zero)
            ms = 1e3 * (t_last - t_first) / max(k - 6, 1) if t_first else float('nan')
            print(f"[{opt}] {k} steps finite | zero-grad entries {int(nz)} max drift {float(dz):.3g} | "
                  f"{ms:.0f} ms/step | param dtype {jax.tree.leaves(params)[0].dtype} "
                  f"loss dtype {loss.dtype}", flush=True)
    print('done')


if __name__ == '__main__':
    main()
