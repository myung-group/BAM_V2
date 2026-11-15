"""Which operations of the float32 model silently run in float64 when jax_enable_x64 is on (needed for a float64
SOAP state, soap_state_dtype)?  Traces loss + gradient of one float32 batch and lists every equation with a float64
output, grouped by source line.

    python research/diag/x64_audit.py research/input/input_3bpa_omat24_<run>_cue.json
"""
import collections
import functools
import json
import math
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax._src import source_info_util

from race_v2.training import train as T
from race_v2.data.loaders import fixed_collate_fn
from race_v2.training.utils import compute_loss

jax.config.update('jax_enable_x64', True)        # after the imports: train_3bpa switches it off at import


def sub_jaxprs(params):
    for v in params.values():
        for x in (v if isinstance(v, (list, tuple)) else [v]):
            if hasattr(x, 'jaxpr') and hasattr(x.jaxpr, 'eqns'):      # ClosedJaxpr
                yield x.jaxpr
            elif hasattr(x, 'eqns'):                                   # Jaxpr
                yield x


def walk(jaxpr, hits, counts):
    for eqn in jaxpr.eqns:
        counts['all'] += 1
        if any(getattr(getattr(v, 'aval', None), 'dtype', None) == np.float64 for v in eqn.outvars):
            counts['f64'] += 1
            hits[(eqn.primitive.name, source_info_util.summarize(eqn.source_info))] += 1
        for sj in sub_jaxprs(eqn.params):
            walk(sj, hits, counts)


def main():
    cfg = json.load(open(sys.argv[1]))
    cfg['soap_state_dtype'] = 'float64'
    cfg['x64'] = False
    (zs, e0, tr, va, tests, stats) = T.build_datasets(cfg, sys.stdout)
    graphdef, params, *_ = T.build_model(cfg, len(zs), e0, stats, sys.stdout)
    print('param dtypes', collections.Counter(str(x.dtype) for x in jax.tree.leaves(params)))
    gs = tr.graphs[:16]
    nn_ = sum(int(g.n_node[0]) for g in gs); ne_ = sum(int(g.n_edge[0]) for g in gs)
    batch = fixed_collate_fn(gs, 64, n_node_pad=(nn_ // 64 + 2) * 64, n_edge_pad=(ne_ // 1024 + 2) * 1024)
    print('batch dtypes', collections.Counter(str(np.asarray(x).dtype) for x in jax.tree.leaves(batch)))
    batch = jax.tree.map(jnp.asarray, batch)      # device arrays as in the trainer (numpy n_node + int64 index fails)

    def lf(p):
        loss, aux = compute_loss(graphdef, p, batch, 1.0, 10.0, 10.0, 10.0, 'huber', True, 'scalar6', 0.0)
        return loss
    closed = jax.make_jaxpr(jax.value_and_grad(lf))(params)
    print('outputs', [str(v.aval) for v in closed.jaxpr.outvars][:3])
    hits, counts = collections.Counter(), collections.Counter()
    walk(closed.jaxpr, hits, counts)
    print(f"equations {counts['all']}, with float64 output {counts['f64']}")
    by_line = collections.Counter()
    for (prim, src), n in hits.items():
        by_line[src] += n
    for src, n in by_line.most_common(40):
        prims = sorted({p for (p, s2) in hits if s2 == src})
        print(f"{n:6d}  {src}  {prims}")


if __name__ == '__main__':
    main()
