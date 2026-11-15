"""Where does the 3BPA force / energy error of a trained RACE sit?

Per-atom decomposition of the error on every split (train, valid, test_300K /
600K / 1200K), using the fixed 27-atom ordering of the 3BPA frames:

  * per species (H, C, N, O): F RMSE and share of the squared error
  * per atomic site (27 sites): F RMSE, mean reference |F|, mean novelty
  * per stratum: least-squares fit  e ~= sum_s a_s F_s  of the error on the
    stratum forces (a_s < 0: stratum s over-shoots by |a_s|), total and per species
  * novelty bins: per-atom novelty = RMS distance between the atom's 26-vector of
    inter-atomic distances and the closest training-frame vector of the same site
    (0 = a geometry seen in training); F RMSE per bin, share of the squared error
  * bond-strain bins: max |r_ij - <r_ij>_train| over the bonded neighbours j
  * direction: share of the squared force error inside the span of the bond
    vectors of the atom (H: along the bond; 3-bonded heavy atom: in the bond plane)
    vs. the orthogonal complement (bends / torsions / out-of-plane)
  * per frame: energy error vs frame novelty (max atom novelty) and vs the
    distance of the dihedral triple to the nearest training triple

Everything is also saved as an .npz (per-atom strata forces, reference forces,
positions, novelty, strain, per-frame energies) so that several runs (seeds) can
be compared with ``analyze_errors_compare.py`` (bias / variance split, ensemble).

Usage:
    python -m race_v2.eval.analyze_errors <train config json> [--ckpt path] [--out prefix]
"""
import argparse
import functools
import json
import sys
from pathlib import Path

import jax
jax.devices()
import jax.numpy as jnp
import jraph
import numpy as np
from flax import nnx

from race_v2.data.xyz_graphs import GraphListDataset
from race_v2.data.loaders import BucketedDataLoader, default_collate_fn
from race_v2.training.utils import load_checkpoint
from race_v2.training import train as T

SPECIES_NAMES = {1: "H", 6: "C", 7: "N", 8: "O"}
BOND_CUT = 1.75   # Å, bonded pair threshold on the mean training geometry


def _rmse(x):
    return float(np.sqrt(np.mean(np.square(x)))) if np.size(x) else float("nan")


def _tag_frames(graphs):
    """Copy graphs with a per-graph frame index in the globals (survives batching / padding)."""
    out = []
    for i, g in enumerate(graphs):
        glob = dict(g.globals)
        glob["frame_id"] = np.array([i], dtype=np.int32)
        out.append(g._replace(globals=glob))
    return out


def _read_dihedrals(path, n):
    """dihedral triples from the extxyz comment lines (NaN when absent)."""
    out = np.full((n, 3), np.nan)
    try:
        import ase.io
        for i, atoms in enumerate(ase.io.iread(str(path), index=":")):
            d = atoms.info.get("dihedrals")
            if d is not None and len(d) == 3:
                out[i] = np.asarray(d, dtype=float)
    except Exception as exc:  # pragma: no cover
        print(f"  (dihedrals not read from {path}: {exc})")
    return out


def _pairwise(pos):
    """(F, N, N) distance matrices."""
    d = pos[:, :, None, :] - pos[:, None, :, :]
    return np.sqrt(np.sum(d * d, axis=-1))


def _site_descriptors(D):
    """(F, N, N-1): the distances of each site to the other sites, fixed order."""
    F, N, _ = D.shape
    mask = ~np.eye(N, dtype=bool)
    return D[:, mask].reshape(F, N, N - 1)


def _novelty(desc, desc_train, chunk=256):
    """RMS distance of each site descriptor to its closest training descriptor.
    desc (F, N, K), desc_train (Ft, N, K) -> (F, N)."""
    F, N, K = desc.shape
    out = np.empty((F, N))
    for i0 in range(0, F, chunk):
        a = desc[i0:i0 + chunk]                                   # (f, N, K)
        # ||a - b||^2 = |a|^2 + |b|^2 - 2 a.b     per site
        a2 = np.sum(a * a, -1)[:, :, None]                        # (f, N, 1)
        b2 = np.sum(desc_train * desc_train, -1).T[None]          # (1, N, Ft)
        ab = np.einsum("fnk,tnk->fnt", a, desc_train)             # (f, N, Ft)
        d2 = np.maximum(a2 + b2 - 2 * ab, 0.0)
        out[i0:i0 + chunk] = np.sqrt(d2.min(-1) / K)
    return out


def _bond_span_share(err, pos, bonds):
    """Share of the squared error inside the span of the bond vectors, per atom.
    err, pos (F, N, 3); bonds: list of neighbour-index lists per site."""
    F, N, _ = pos.shape
    inside = np.zeros((F, N))
    for i in range(N):
        nb = bonds[i]
        if not nb:
            inside[:, i] = np.nan
            continue
        v = pos[:, nb, :] - pos[:, i:i + 1, :]                    # (F, nb, 3)
        for f in range(F):
            q, _ = np.linalg.qr(v[f].T)                           # (3, r) orthonormal basis
            p = q @ (q.T @ err[f, i])
            inside[f, i] = np.sum(p * p)
    return inside


def analyse_set(name, F_s, F_ref, E_s, E_ref, natoms, shift, pos, species, z_of_species,
                train_desc, train_D_mean, bonds, dihedrals, train_dihedrals, report, out):
    S = F_s.shape[0]
    nF, N = pos.shape[:2]
    F_tot = F_s.sum(0)                                            # (F, N, 3)
    err = F_tot - F_ref
    e2 = np.sum(err * err, -1)                                    # (F, N) squared error per atom
    E_tot = E_s.sum(0) + shift * natoms
    dE = E_tot - E_ref
    f_rmse = _rmse(err) * 1e3
    e_rmse = _rmse(dE) * 1e3
    rep = dict(n_frames=int(nF), F_rmse_meVA=f_rmse, E_rmse_meV=e_rmse)
    print(f"\n[{name}] frames {nF}  F RMSE {f_rmse:7.2f} meV/A   E RMSE {e_rmse:7.2f} meV")

    # ---- per species ---------------------------------------------------------
    tot_e2 = e2.sum()
    rows = []
    print(f"  {'species':>8s} {'n_atoms':>7s} {'F_RMSE':>8s} {'err share':>9s} {'|F_ref| rms':>11s} {'rel err':>8s}")
    for sidx in sorted(set(species.tolist())):
        m = species == sidx
        r = _rmse(err[:, m]) * 1e3
        fref = _rmse(F_ref[:, m]) * 1e3
        share = e2[:, m].sum() / tot_e2
        nm = SPECIES_NAMES.get(int(z_of_species[sidx]), str(z_of_species[sidx]))
        rows.append(dict(species=nm, n_sites=int(m.sum()), F_rmse_meVA=r, err_share=float(share),
                         F_ref_rms_meVA=fref, rel_err=r / fref))
        print(f"  {nm:>8s} {int(m.sum()):>7d} {r:8.2f} {share:9.3f} {fref:11.1f} {r / fref:8.3f}")
    rep["per_species"] = rows

    # ---- per site ------------------------------------------------------------
    nov = _novelty(_site_descriptors(_pairwise(pos)), train_desc)          # (F, N)
    D = _pairwise(pos)
    strain = np.zeros((nF, N))
    for i in range(N):
        nb = bonds[i]
        strain[:, i] = np.max(np.abs(D[:, i, nb] - train_D_mean[i, nb][None]), -1) if nb else 0.0
    site_rows = []
    for i in range(N):
        nm = SPECIES_NAMES.get(int(z_of_species[species[i]]), "?")
        site_rows.append(dict(site=i, species=nm, n_bonds=len(bonds[i]), F_rmse_meVA=_rmse(err[:, i]) * 1e3,
                              F_ref_rms_meVA=_rmse(F_ref[:, i]) * 1e3, err_share=float(e2[:, i].sum() / tot_e2),
                              novelty_mean_A=float(nov[:, i].mean()), strain_mean_A=float(strain[:, i].mean())))
    rep["per_site"] = site_rows
    top = sorted(site_rows, key=lambda r: -r["F_rmse_meVA"])[:8]
    print(f"  worst sites: " + ", ".join(f"{r['species']}{r['site']}({r['n_bonds']}b) {r['F_rmse_meVA']:.0f}" for r in top))

    # ---- per stratum: regress the error on the stratum forces ------------------
    X = F_s.reshape(S, -1).T                                       # (F*N*3, S)
    y = err.reshape(-1)
    a, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1.0 - np.sum((y - X @ a) ** 2) / np.sum((y - y.mean()) ** 2)
    rep["strata_fit"] = dict(coef=a.tolist(), r2=float(r2),
                             F_s_rms_meVA=[_rmse(F_s[s]) * 1e3 for s in range(S)])
    print(f"  err ~ sum_s a_s F_s : a = {np.array2string(a, precision=4)}   R^2 = {r2:.3f}"
          f"   (|F_s| rms = {[round(_rmse(F_s[s]) * 1e3) for s in range(S)]})")
    per_sp = {}
    for sidx in sorted(set(species.tolist())):
        m = species == sidx
        Xs = F_s[:, :, m].reshape(S, -1).T
        ys = err[:, m].reshape(-1)
        a_s, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
        r2s = 1.0 - np.sum((ys - Xs @ a_s) ** 2) / np.sum((ys - ys.mean()) ** 2)
        nm = SPECIES_NAMES.get(int(z_of_species[sidx]), "?")
        per_sp[nm] = dict(coef=a_s.tolist(), r2=float(r2s))
        print(f"     {nm}: a = {np.array2string(a_s, precision=4)}  R^2 = {r2s:.3f}")
    rep["strata_fit_per_species"] = per_sp

    # ---- novelty / strain bins -------------------------------------------------
    def binned(values, edges, label, unit):
        idx = np.digitize(values, edges)
        rows = []
        print(f"  {label:>22s} {'atoms':>7s} {'frac':>6s} {'F_RMSE':>8s} {'err share':>9s}")
        for b in range(len(edges) + 1):
            m = idx == b
            lo = "-inf" if b == 0 else f"{edges[b - 1]:.3f}"
            hi = "inf" if b == len(edges) else f"{edges[b]:.3f}"
            if m.sum() == 0:
                continue
            r = _rmse(err[m]) * 1e3
            rows.append(dict(bin=f"[{lo},{hi}){unit}", n=int(m.sum()), frac=float(m.mean()),
                             F_rmse_meVA=r, err_share=float(e2[m].sum() / tot_e2)))
            print(f"  {rows[-1]['bin']:>22s} {int(m.sum()):>7d} {m.mean():6.3f} {r:8.2f} {rows[-1]['err_share']:9.3f}")
        return rows

    rep["novelty_bins"] = binned(nov, out["novelty_edges"], "novelty bin", "A")
    rep["strain_bins"] = binned(strain, out["strain_edges"], "bond-strain bin", "A")
    rep["novelty_quantiles_A"] = np.quantile(nov, [0.5, 0.9, 0.99]).tolist()
    rep["corr_err_novelty"] = float(np.corrcoef(np.sqrt(e2).ravel(), nov.ravel())[0, 1])
    print(f"  novelty quantiles 50/90/99 % = {np.round(rep['novelty_quantiles_A'], 3)} A;"
          f"  corr(|err|, novelty) = {rep['corr_err_novelty']:.3f}")

    # ---- direction: inside the bond span vs orthogonal ----------------------------
    inside = _bond_span_share(err, pos, bonds)
    dir_rows = {}
    for key, m in (("H (1 bond)", np.array([len(bonds[i]) == 1 for i in range(N)])),
                   ("heavy 2 bonds", np.array([len(bonds[i]) == 2 for i in range(N)])),
                   ("heavy 3 bonds", np.array([len(bonds[i]) == 3 for i in range(N)]))):
        if m.sum() == 0:
            continue
        share = np.nansum(inside[:, m]) / e2[:, m].sum()
        dir_rows[key] = dict(n_sites=int(m.sum()), err_share_in_bond_span=float(share),
                             F_rmse_meVA=_rmse(err[:, m]) * 1e3)
        print(f"  {key:>14s}: {int(m.sum()):2d} sites, F RMSE {dir_rows[key]['F_rmse_meVA']:7.2f},"
              f" squared error inside bond span {share:.3f} (random: {min(len(bonds[int(np.flatnonzero(m)[0])]), 3) / 3:.3f})")
    rep["direction"] = dir_rows

    # ---- per frame: energy -------------------------------------------------------
    frame_nov = nov.max(1)
    frame_f = np.sqrt(e2.mean(1))
    rep["corr_absdE_frame_novelty"] = float(np.corrcoef(np.abs(dE), frame_nov)[0, 1])
    rep["corr_absdE_frame_Frmse"] = float(np.corrcoef(np.abs(dE), frame_f)[0, 1])
    rep["dE_mean_meV"] = float(dE.mean() * 1e3)
    line = (f"  energy: bias {dE.mean() * 1e3:+.2f} meV, corr(|dE|, frame novelty) = {rep['corr_absdE_frame_novelty']:.3f},"
            f" corr(|dE|, frame F RMSE) = {rep['corr_absdE_frame_Frmse']:.3f}")
    if np.all(np.isfinite(dihedrals)) and np.all(np.isfinite(train_dihedrals)):
        # angular distance of the dihedral triple to the nearest training triple
        diff = dihedrals[:, None, :] - train_dihedrals[None, :, :]
        diff = np.abs((diff + 180.0) % 360.0 - 180.0)
        dih_nov = np.sqrt(np.sum(diff ** 2, -1) / 3).min(1)      # (F,) degrees
        rep["corr_absdE_dihedral_novelty"] = float(np.corrcoef(np.abs(dE), dih_nov)[0, 1])
        rep["corr_Frmse_dihedral_novelty"] = float(np.corrcoef(frame_f, dih_nov)[0, 1])
        rep["dihedral_novelty_median_deg"] = float(np.median(dih_nov))
        line += (f"\n  dihedral novelty median {np.median(dih_nov):.1f} deg, corr(|dE|, dih nov) = "
                 f"{rep['corr_absdE_dihedral_novelty']:.3f}, corr(F RMSE, dih nov) = {rep['corr_Frmse_dihedral_novelty']:.3f}")
    else:
        dih_nov = np.full(nF, np.nan)
    print(line)
    report[name] = rep
    return dict(F_s=F_s.astype(np.float32), F_ref=F_ref.astype(np.float32), pos=pos.astype(np.float32),
                E_s=E_s.astype(np.float64), E_ref=E_ref.astype(np.float64), natoms=natoms,   # float32 spacing ~1 meV at 3BPA totals
                novelty=nov.astype(np.float32), strain=strain.astype(np.float32),
                dihedrals=dihedrals.astype(np.float32), dihedral_novelty=dih_nov.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--out", default=None, help="prefix: writes <prefix>.json and <prefix>.npz")
    ap.add_argument("--raw", action="store_true", help="use raw instead of EMA parameters")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    if cfg.get("x64"):
        jax.config.update("jax_enable_x64", True)      # training.train switches x64 off on import
    ckpt_path = args.ckpt or cfg.get("ckpt_name", "ckpt_best.pkl")
    ckpt = load_checkpoint(ckpt_path)
    params = ckpt["params"] if args.raw else ckpt["ema_params"]
    params = jax.tree.map(jnp.asarray, params)

    fout = sys.stdout
    (atomic_numbers, atom_energies, train_ds, valid_ds, test_sets, stats) = T.build_datasets(cfg, fout)
    if cfg.get("x64"):
        for ds in [train_ds, valid_ds] + list(test_sets.values()):
            ds.graphs = T._cast_graphs(ds.graphs, np.float64)
    graphdef, _p0, _opt, _sched, _gc = T.build_model(cfg, len(atomic_numbers), atom_energies, stats, fout)
    model = nnx.merge(graphdef, params)
    shift = float(np.asarray(model.shift[...]))
    z_of_species = np.array(sorted(int(z) for z in atomic_numbers))
    print(f"checkpoint: {ckpt_path} (epoch {ckpt.get('epoch')}, val loss {ckpt.get('best_val_loss'):.3e})")

    collate = functools.partial(default_collate_fn, node_bucket_size=int(cfg.get("node_bucket_size", 16)),
                                edge_bucket_size=int(cfg.get("edge_bucket_size", 256)))

    @jax.jit
    def strata_step(batch):
        return model.strata_energies_forces(batch)                # (S, G), (S, N, 3)

    def run(graphs):
        graphs = _tag_frames(graphs)
        loader = BucketedDataLoader(GraphListDataset(graphs), batch_size=args.batch_size, n_buckets=1,
                                    shuffle=False, drop_last=False, rngs=nnx.Rngs(0))
        loader.collate_fn = collate
        Fs, Fr, P, Sp, Fid, Es, Er, Nat, Gid = [], [], [], [], [], [], [], [], []
        for batch in loader:
            gmask = np.asarray(jraph.get_graph_padding_mask(batch))
            nmask = np.asarray(jraph.get_node_padding_mask(batch))
            E_s, F_s = strata_step(batch)
            Fs.append(np.asarray(F_s)[:, nmask]); Fr.append(np.asarray(batch.nodes["forces"])[nmask])
            P.append(np.asarray(batch.nodes["positions"])[nmask]); Sp.append(np.asarray(batch.nodes["species"])[nmask])
            gidx = np.repeat(np.arange(batch.n_node.shape[0]), np.asarray(batch.n_node))
            Fid.append(np.asarray(batch.globals["frame_id"]).reshape(-1)[gidx[nmask]])
            Es.append(np.asarray(E_s)[:, gmask]); Er.append(np.asarray(batch.globals["energy"]).reshape(-1)[gmask])
            Nat.append(np.asarray(batch.n_node)[gmask]); Gid.append(np.asarray(batch.globals["frame_id"]).reshape(-1)[gmask])
        F_s = np.concatenate(Fs, 1); F_ref = np.concatenate(Fr); pos = np.concatenate(P); sp = np.concatenate(Sp)
        fid = np.concatenate(Fid); E_s = np.concatenate(Es, 1); E_ref = np.concatenate(Er); natoms = np.concatenate(Nat)
        gid = np.concatenate(Gid)
        N = int(natoms[0]); assert np.all(natoms == N), "analysis assumes a fixed molecule"
        nF = len(gid)
        # reorder to frame order and reshape to (.., F, N, ..)
        order = np.argsort(fid, kind="stable")
        F_s = F_s[:, order].reshape(F_s.shape[0], nF, N, 3); F_ref = F_ref[order].reshape(nF, N, 3)
        pos = pos[order].reshape(nF, N, 3); sp = sp[order].reshape(nF, N)
        assert np.all(sp == sp[0]), "species ordering differs between frames"
        gorder = np.argsort(gid, kind="stable")
        return F_s, F_ref, pos, sp[0], E_s[:, gorder], E_ref[gorder], natoms[gorder].astype(np.float64)

    # ---- training-set reference geometry ------------------------------------------
    train_graphs = train_ds.graphs if args.max_frames is None else train_ds.graphs[: args.max_frames]
    train_pos = np.stack([np.asarray(g.nodes["positions"]) for g in train_graphs])
    train_D = _pairwise(train_pos)
    train_D_mean = train_D.mean(0)
    train_desc = _site_descriptors(train_D)
    N = train_pos.shape[1]
    bonds = [[int(j) for j in np.flatnonzero((train_D_mean[i] < BOND_CUT) & (np.arange(N) != i))] for i in range(N)]
    # in-sample novelty is 0 by construction -> leave-one-out for the bin edges
    loo = np.empty(train_desc.shape[:2])
    for f in range(train_desc.shape[0]):
        others = np.delete(train_desc, f, axis=0)
        loo[f] = _novelty(train_desc[f:f + 1], others)[0]
    out = dict(novelty_edges=np.quantile(loo, [0.5, 0.9, 0.99]).tolist() + [2 * float(np.quantile(loo, 0.99))],
               strain_edges=[0.05, 0.1, 0.2, 0.3])
    print(f"training leave-one-out novelty quantiles 50/90/99 % = {np.round(np.quantile(loo, [0.5, 0.9, 0.99]), 3)} A"
          f" -> novelty bin edges {np.round(out['novelty_edges'], 3)}")
    print("bonds (mean training geometry, < %.2f A): " % BOND_CUT +
          " ".join(f"{SPECIES_NAMES.get(int(z_of_species[int(np.asarray(train_graphs[0].nodes['species'])[i])]), '?')}{i}:{len(bonds[i])}" for i in range(N)))

    # training dihedrals (the training frames are a shuffled subset of the file)
    train_path = Path(cfg.get("train_path", "research/dataset_3BPA/train_300K.xyz"))
    all_train_dih = _read_dihedrals(train_path, 500)
    # match training graphs to file frames via energies (unique in practice)
    file_E = None
    try:
        import ase.io
        file_E = np.array([a.get_potential_energy() for a in ase.io.iread(str(train_path), index=":")])
    except Exception:
        pass

    report = {}
    arrays = {}
    datasets = {"train": train_ds, "valid": valid_ds, **test_sets}      # "train" first: defines train_dih_ref
    test_paths = list(cfg.get("test_paths", []))
    train_dih_ref = np.full((1, 3), np.nan)
    for name, ds in datasets.items():
        graphs = ds.graphs if args.max_frames is None else ds.graphs[: args.max_frames]
        F_s, F_ref, pos, species, E_s, E_ref, natoms = run(graphs)
        # dihedrals: test files are in file order; train/valid are subsets -> match by energy
        dih = np.full((len(graphs), 3), np.nan)
        if name in ("train", "valid"):
            if file_E is not None:
                e0 = np.sum(np.asarray(atom_energies)[species])
                for k, g in enumerate(graphs):
                    j = np.argmin(np.abs(file_E - (float(np.asarray(g.globals["energy"])[0]) + e0)))
                    dih[k] = all_train_dih[j]
        else:
            for p in test_paths:
                if Path(p).stem.replace("test_", "") == name.replace("test_", ""):
                    dih = _read_dihedrals(p, len(graphs))
        if name == "train":
            train_dih_ref = dih.copy()
        arrays[name] = analyse_set(name, F_s, F_ref, E_s, E_ref, natoms, shift, pos, species, z_of_species,
                                   train_desc, train_D_mean, bonds, dih, train_dih_ref, report, out)
        arrays[name]["species"] = species

    if args.out:
        json.dump(report, open(args.out + ".json", "w"), indent=1)
        np.savez_compressed(args.out + ".npz", **{f"{k}/{kk}": v for k, d in arrays.items() for kk, v in d.items()},
                            bonds=np.array(json.dumps(bonds)), z_of_species=z_of_species)
        print(f"\nwritten {args.out}.json / .npz")


if __name__ == "__main__":
    main()
