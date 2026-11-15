"""Minimal reproducer: cue fused edge TP (gather senders / scatter receivers) value in a plain forward
vs inside value_and_grad, vs the e3nn reference; effect of edge ordering."""
import json, sys, io
import numpy as np, jax, jax.numpy as jnp
import e3nn_jax as e3nn
from race_v2.training import train as T
from race_v2.data.loaders import default_collate_fn
from race_v2.models.race import cue_edge_tensor_product
jax.config.update("jax_enable_x64", True)
IRR_IN = sys.argv[1]
for mol in sys.argv[2:]:
    cfg = json.load(open(f"research/input/input_3bpa_omat24_rmd17_{mol}_v9a_s42_cue.json"))
    f = io.StringIO()
    an, ae, tr, va, te, st = T.build_datasets(cfg, f)
    g0 = tr.graphs[0]
    r, s = np.asarray(g0.receivers), np.asarray(g0.senders)
    print(f"== {mol}: n_node {int(g0.n_node[0])} n_edge {int(g0.n_edge[0])} receivers sorted {bool(np.all(np.diff(r) >= 0))} "
          f"senders sorted {bool(np.all(np.diff(s) >= 0))}")
    b = default_collate_fn(T._cast_graphs(tr.graphs[:5], np.float64), 5, 16, 256)
    snd, rcv = jnp.asarray(b.senders), jnp.asarray(b.receivers)
    n_nodes, n_edges = b.nodes["positions"].shape[0], snd.shape[0]
    print(f"   batch: {n_nodes} nodes, {n_edges} edges; receivers sorted {bool(np.all(np.diff(np.asarray(rcv)) >= 0))}")
    irr_in = e3nn.Irreps(IRR_IN); irr_out = e3nn.Irreps("128x0e+128x1o+128x2e+128x3o"); sh_irr = e3nn.Irreps.spherical_harmonics(3)
    rng = np.random.RandomState(0)
    x = jnp.asarray(rng.normal(size=(n_nodes, irr_in.dim)))
    vec = jnp.asarray(b.nodes["positions"])[snd] - jnp.asarray(b.nodes["positions"])[rcv]
    sh = e3nn.spherical_harmonics(sh_irr, vec + 1e-9, normalize=True, normalization="component")
    tp_irr = e3nn.tensor_product(irr_in, sh_irr, filter_ir_out=irr_in)
    from race_v2.models.race import _cue_descriptor
    nw = _cue_descriptor(irr_in, sh_irr, irr_out).polynomial.inputs[0].size
    w = jnp.asarray(rng.normal(size=(n_edges, nw)))
    def agg(x, w, sh, snd, rcv):
        return cue_edge_tensor_product(irr_in, sh_irr, irr_out, x, sh.array, w, snd, rcv).array
    def loss(x, w, sh, snd, rcv):
        out = agg(x, w, sh, snd, rcv); return jnp.sum(out ** 2), out
    out_fwd = jax.jit(agg)(x, w, sh, snd, rcv)
    (_, out_vg), _g = jax.jit(jax.value_and_grad(loss, has_aux=True))(x, w, sh, snd, rcv)
    out_eager = agg(x, w, sh, snd, rcv)
    print(f"   |fwd - value_and_grad primal| max {float(jnp.max(jnp.abs(out_fwd - out_vg))):.3e}; |fwd - eager| {float(jnp.max(jnp.abs(out_fwd - out_eager))):.3e}")
    # same with edges sorted by receiver
    perm = jnp.argsort(rcv, stable=True)
    sh_p = e3nn.IrrepsArray(sh.irreps, sh.array[perm])
    out_fwd_p = jax.jit(agg)(x, w[perm], sh_p, snd[perm], rcv[perm])
    (_, out_vg_p), _ = jax.jit(jax.value_and_grad(loss, has_aux=True))(x, w[perm], sh_p, snd[perm], rcv[perm])
    print(f"   sorted-by-receiver: |fwd - vg primal| {float(jnp.max(jnp.abs(out_fwd_p - out_vg_p))):.3e}; |fwd(sorted) - fwd(orig)| {float(jnp.max(jnp.abs(out_fwd_p - out_fwd))):.3e}")
    # gradient w.r.t. positions through sh and the radial weights (the model's training path)
    pos = jnp.asarray(b.nodes["positions"])
    def through_pos(pos):
        v = pos[snd] - pos[rcv]
        shp = e3nn.spherical_harmonics(sh_irr, v + 1e-9, normalize=True, normalization="component")
        rr = jnp.linalg.norm(v + 1e-9, axis=-1, keepdims=True)
        wp = w * jnp.cos(rr)
        out = agg(x, wp, shp, snd, rcv)
        return jnp.sum(out ** 2), out
    o_f = jax.jit(lambda p: through_pos(p)[1])(pos)
    (_, o_g), gp = jax.jit(jax.value_and_grad(through_pos, has_aux=True))(pos)
    print(f"   positions path: |fwd - vg primal| max {float(jnp.max(jnp.abs(o_f - o_g))):.3e}")
    # two identical calls (atomic basis + layer-0 message) multiplied, grad w.r.t. positions
    def twice(pos):
        v = pos[snd] - pos[rcv]
        shp = e3nn.spherical_harmonics(sh_irr, v + 1e-9, normalize=True, normalization="component")
        rr = jnp.linalg.norm(v + 1e-9, axis=-1, keepdims=True)
        wp = w * jnp.cos(rr)
        a1 = agg(x, wp, shp, snd, rcv); a2 = agg(x, wp, shp, snd, rcv)
        out = a1 * a2
        return jnp.sum(out), out
    t_f = jax.jit(lambda p: twice(p)[1])(pos)
    (_, t_g), _ = jax.jit(jax.value_and_grad(twice, has_aux=True))(pos)
    print(f"   two identical calls: |fwd - vg primal| max {float(jnp.max(jnp.abs(t_f - t_g))):.3e} (scale {float(jnp.max(jnp.abs(t_f))):.2e})")
