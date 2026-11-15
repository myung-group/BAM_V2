"""Water checkpoint: per-frame test errors (eager, one frame per call) vs the sharded evaluation,
and the frames / atoms that carry the force-RMSE tail."""
import json, sys, io, functools
import numpy as np, jax, jax.numpy as jnp
from flax import nnx
from ase.io import read
from race_v2.training import train as T
from race_v2.data.loaders import default_collate_fn, setup_mesh
from race_v2.training.utils import load_checkpoint, make_sharded_evaluate_step, evaluate_sharded, replicate_pytree
cfg = json.load(open(sys.argv[1])); ck = sys.argv[2]
jax.config.update("jax_enable_x64", True)
f = io.StringIO()
an, ae, tr, va, te, st = T.build_datasets(cfg, f)
name, ds = next(iter(te.items()))
ds.graphs = T._cast_graphs(ds.graphs, np.float64)
gd, *_ = T.build_model(cfg, len(an), ae, st, f)
c = load_checkpoint(ck); params = jax.tree.map(jnp.asarray, c["ema_params"])
print("checkpoint epoch", c["epoch"])
nb, eb = int(cfg.get("node_bucket_size", 128)), int(cfg.get("edge_bucket_size", 4096))
model = nnx.merge(gd, params)
fwd = jax.jit(lambda p, b: (lambda o: (o["energy"], o["forces"]))(nnx.merge(gd, p)(b)))
frames = read(cfg["test_paths"][0] if "test_paths" in cfg else cfg["test_path"], ":")
dE, fse, fae, rows = [], [], [], []
for k, g in enumerate(ds.graphs):
    b = default_collate_fn([g], 1, nb, eb)
    e, fr = fwd(params, b)
    n = int(g.n_node[0])
    de = float(np.asarray(e)[0]) - float(np.asarray(g.globals["energy"]).ravel()[0])
    df = np.asarray(fr)[:n] - np.asarray(g.nodes["forces"], np.float64)
    dE.append(de); fse.append((df ** 2).sum()); fae.append(np.abs(df).sum())
    a = frames[k]
    from ase.neighborlist import neighbor_list
    dmin = neighbor_list("d", a, 1.2).min()
    rho = len(a) / 3 * 18.015 / (a.get_volume() * 0.6022)
    per_atom = np.sqrt((df ** 2).sum(1))
    rows.append((k, de * 1e3, np.sqrt((df ** 2).mean()) * 1e3, per_atom.max() * 1e3, int(np.argmax(per_atom)),
                 a.get_chemical_symbols()[int(np.argmax(per_atom))], np.abs(a.get_forces()).max(), dmin, rho))
dE = np.array(dE); N = sum(int(g.n_node[0]) for g in ds.graphs)
print(f"eager per-frame: E RMSE {np.sqrt(np.mean(dE ** 2)) * 1e3:.1f} meV/frame ({np.sqrt(np.mean(dE ** 2)) * 1e3 / 64:.2f} meV/H2O), "
      f"E MAE {np.mean(np.abs(dE)) * 1e3:.1f}, F RMSE {np.sqrt(sum(fse) / (3 * N)) * 1e3:.2f}, F MAE {sum(fae) / (3 * N) * 1e3:.2f} meV/A")
rows.sort(key=lambda r: -r[2])
print("frames by force RMSE: idx | dE meV | F RMSE | worst atom err (idx sym) | max|F| ref eV/A | min dist A | density")
for r in rows[:12]:
    print(f"  {r[0]:4d} | {r[1]:9.1f} | {r[2]:7.1f} | {r[3]:8.1f} ({r[4]} {r[5]}) | {r[6]:6.2f} | {r[7]:.3f} | {r[8]:.3f}")
fr_rmse = np.array([r[2] for r in rows]); print("F RMSE per frame quantiles 50/90/99/max:", np.round(np.quantile(fr_rmse, [0.5, 0.9, 0.99, 1.0]), 1))
de_s = sorted(rows, key=lambda r: -abs(r[1]))
print("frames by |dE|:", [(r[0], round(r[1], 1), round(r[8], 3)) for r in de_s[:10]])
mesh, nd = setup_mesh()
es = make_sharded_evaluate_step(mesh, huber_delta=float(cfg.get("huber_delta", 10.0)), loss_type="huber", periodic=False,
                                cov_mode="scalar6", beta_nll=0.0)
col = functools.partial(default_collate_fn, node_bucket_size=nb, edge_bucket_size=eb)
m = evaluate_sharded(graphdef=gd, params=replicate_pytree(params, mesh), files_pkl=[ds], batch_size=int(cfg.get("batch_size", 2)),
                     n_devices=nd, mesh=mesh, energy_weight=1.0, force_weight=10.0, stress_weight=0.0,
                     huber_delta=float(cfg.get("huber_delta", 10.0)), loss_type="huber", eval_step=es,
                     params_already_replicated=True, periodic=False, cov_mode="scalar6", beta_nll=0.0, collate_fn=col)
print(f"evaluate_sharded:  E RMSE {m['energy_rmse_total'] * 1e3:.1f} meV/frame, E MAE {m['energy_mae_total'] * 1e3:.1f}, "
      f"F RMSE {m['force_rmse'] * 1e3:.2f}, F MAE {m['force_mae'] * 1e3:.2f}")
