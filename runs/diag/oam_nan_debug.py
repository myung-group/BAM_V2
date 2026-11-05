"""Locate the NaN of the OAM subset runs (oamfix_*: E_LOSS nan from epoch 1, F_LOSS frozen).

    python runs/diag/oam_nan_debug.py input/input_3bpa_omat24_oamfix_salex_v2_s42_cue.json

One batch of 16 training graphs; forward and loss gradient for: fixed vs bucket padding, stress loss on/off, and
the model with single additions switched off.  Then jax_debug_nans on the first failing case for the traceback.
"""
import copy
import json
import sys
import traceback

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from bam_omat24 import train_3bpa as T
from bam_omat24.data_nnx import fixed_collate_fn, default_collate_fn
from bam_omat24.training_utils import compute_loss


def nan_report(tag, out, batch):
    import jraph
    gm = np.asarray(jraph.get_graph_padding_mask(batch)); nm = np.asarray(jraph.get_node_padding_mask(batch))
    e = np.asarray(out['energy']); f = np.asarray(out['forces']); s = np.asarray(out['stress'])
    print(f"[{tag}] energy nan real {np.isnan(e[gm]).sum()}/{gm.sum()} pad {np.isnan(e[~gm]).sum()}/{(~gm).sum()} | "
          f"forces nan real {np.isnan(f[nm]).sum()} | stress nan real {np.isnan(s[gm]).sum()} | "
          f"E real range {np.nanmin(e[gm]) if gm.any() else 0:.3g}..{np.nanmax(e[gm]) if gm.any() else 0:.3g}", flush=True)
    st = out.get('strata')
    if st is not None:
        st = np.asarray(st)
        print(f"   strata nan per stratum (real nodes): {[int(np.isnan(st[k][nm]).sum()) for k in range(st.shape[0])]}", flush=True)
    return bool(np.isnan(e[gm]).any())


def grad_report(tag, graphdef, params, batch, periodic):
    def lf(p):
        loss, aux = compute_loss(graphdef, p, batch, 1.0, 10.0, 10.0, 10.0, 'huber', periodic, 'scalar6', 0.0)
        return loss
    loss, g = jax.value_and_grad(lf)(params)
    leaves = jax.tree_util.tree_flatten_with_path(g)[0]
    bad = [jax.tree_util.keystr(p) for p, v in leaves if not np.all(np.isfinite(np.asarray(v)))]
    print(f"[{tag}] loss {float(loss):.4g}  non-finite grad leaves {len(bad)}/{len(leaves)}: {bad[:6]}", flush=True)
    return bool(bad) or not np.isfinite(float(loss))


def main():
    cfg0 = json.load(open(sys.argv[1]))
    fout = sys.stdout
    (zs, e0, tr, va, tests, stats) = T.build_datasets(cfg0, fout)
    graphs = tr.graphs[:16]
    nn_ = sum(int(g.n_node[0]) for g in graphs); ne_ = sum(int(g.n_edge[0]) for g in graphs)
    print('batch: 16 graphs', nn_, 'nodes', ne_, 'edges; species', sorted({int(s) for g in graphs for s in g.nodes['species']}))
    pads = {'fixed': fixed_collate_fn(graphs, 64, n_node_pad=(nn_ // 64 + 2) * 64, n_edge_pad=(ne_ // 1024 + 2) * 1024),
            'bucket': default_collate_fn(graphs, 16, node_bucket_size=64, edge_bucket_size=2048)}
    variants = {
        'full': {},
        'no_uff': {'pair_repulsion': None},
        'no_hill': {'x_norm': None},
        'no_shells': {'radial_shells': False},
        'no_prior': {'radial_prior': None},
        'no_cheb': {'residual': 'none', 'layer_scale': None},
        'v1': {'x_mode': 'species', 'x_norm': None, 'message_norm': 'layer', 'residual': 'none', 'layer_scale': None,
               'update': 'gate', 'readout_mode': 'linear', 'radial_prior': None, 'radial_shells': False,
               'pair_repulsion': None},
    }
    first_bad = None
    for vname, flips in variants.items():
        cfg = copy.deepcopy(cfg0); cfg.update(flips)
        graphdef, params, *_ = T.build_model(cfg, len(zs), e0, stats, fout)
        model = nnx.merge(graphdef, params)
        for pname, batch in pads.items():
            bad_f = nan_report(f'{vname}/{pname}', model(batch), batch)
            bad_g = grad_report(f'{vname}/{pname}/stress', graphdef, params, batch, True)
            bad_g2 = grad_report(f'{vname}/{pname}/nostress', graphdef, params, batch, False) if bad_g else False
            if (bad_f or bad_g) and first_bad is None:
                first_bad = (vname, pname, graphdef, params, batch, bad_f)
    if first_bad is not None:
        vname, pname, graphdef, params, batch, bad_f = first_bad
        print(f'--- jax_debug_nans on {vname}/{pname} ({"forward" if bad_f else "gradient"})', flush=True)
        jax.config.update('jax_debug_nans', True)
        try:
            if bad_f:
                nnx.merge(graphdef, params)(batch)
            else:
                grad_report('debug', graphdef, params, batch, True)
        except Exception as ex:
            tb = traceback.format_exception(ex)
            keep = [l for l in tb if 'bam_omat24' in l or 'Error' in l or 'nan' in l.lower()]
            print(''.join(keep[-30:]))
    print('done')


if __name__ == '__main__':
    main()
