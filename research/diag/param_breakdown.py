"""Where are the parameters?  Builds the model of each config (no checkpoint, no data needed beyond the cache) and
splits the parameter count into species-indexed leaves (a dimension equal to n_species) and shared leaves, per
top-level module, plus the largest leaves.

    python runs/diag/param_breakdown.py <config.json> [...]
"""
import collections
import json
import sys

import jax

from bam_omat24 import train_3bpa as T


def main():
    for path in sys.argv[1:]:
        cfg = json.load(open(path))
        zs = cfg['atomic_numbers']; ns = len(zs)
        stats = {'shift': 0.0, 'shift_raw': 0.0, 'scale': 1.0, 'avg_n_neighbors': float(cfg.get('avg_n_neighbors', 20.0))}
        try:
            graphdef, params, *_ = T.build_model(cfg, ns, cfg['atom_energies'], stats, open('/dev/null', 'w'))
        except Exception as ex:                     # build_model may want full dataset stats
            (zs2, e0, tr, va, te, st) = T.build_datasets(cfg, open('/dev/null', 'w'))
            graphdef, params, *_ = T.build_model(cfg, ns, e0, st, open('/dev/null', 'w'))
        leaves = jax.tree_util.tree_flatten_with_path(params)[0]
        tot = sum(v.size for _, v in leaves)
        by_mod = collections.Counter(); sp = 0
        rows = []
        for p, v in leaves:
            k = jax.tree_util.keystr(p)
            top = k.split(']')[0].strip("['")
            if top in ('layers', 'x_readouts', 'f_readouts', 'x_projs'):
                top = top + '[' + k.split('][')[1].split(']')[0] + ']'
            species_idx = ns in v.shape or (v.ndim >= 1 and v.shape[0] % ns == 0 and v.shape[0] // ns in (1, 2, 3, 4, 5, 6, 7, 8))
            by_mod[(top, 'species' if ns in v.shape else 'shared')] += v.size
            sp += v.size if ns in v.shape else 0
            rows.append((v.size, k, tuple(v.shape)))
        print(f"== {path}: {tot / 1e6:.2f} M parameters, n_species {ns}, species-indexed (a dim == {ns}) "
              f"{sp / 1e6:.2f} M ({100 * sp / tot:.0f}%)")
        mods = collections.defaultdict(dict)
        for (top, kind), n in by_mod.items():
            mods[top][kind] = n
        for top, d in sorted(mods.items(), key=lambda x: -sum(x[1].values()))[:14]:
            print(f"   {top:28s} shared {d.get('shared', 0) / 1e6:7.3f} M   species {d.get('species', 0) / 1e6:7.3f} M")
        print('   largest leaves:')
        for n, k, shp in sorted(rows, reverse=True)[:8]:
            print(f"     {n / 1e6:7.3f} M  {shp}  {k}")


if __name__ == '__main__':
    main()
