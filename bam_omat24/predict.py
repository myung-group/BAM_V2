"""Inference API for trained RACE / RACE_LR checkpoints (2026-11-07), for external evaluation pipelines (e.g. the
ctx-RACE evaluate.py adapter): total energies and forces of ase.Atoms frames, with the total charge, spin and
fragments read as the training config says.

    from bam_omat24.predict import load
    pred = load('runs/<name>/ckpt.pkl')                  # config from the checkpoint (or config=dict / path)
                                                         # batch_size=16 frames, max_atoms=2048 atoms per batch
    energies, forces = pred.predict(frames)              # energies (n_frames,) eV, forces: list of (n_atoms, 3) eV/Å
    charges = pred.charges(frames)                       # RACE_LR only: list of (n_atoms,) Qeq charges in e

Inputs per frame (key names from the config): info[total_charge_key] (e.g. "charge"), info[total_spin_key] (e.g.
"spin", default 1), and for per-fragment constraints arrays[fragment_key] (e.g. "is_ligand") with
info[fragment_charges_key].  No reference energies or forces are needed.
Energies are TOTAL energies: the model output plus the config's atom_energies (E0) per atom, plus an optional
per-element correction from a training-split refit:

    python -m bam_omat24.predict refit runs/<name>/ckpt.pkl --xyz <training split.xyz> [--max-frames 20000]
        -> runs/<name>/ckpt.species_refit.json, applied by load(...) when present (species_refit=True).
The refit is a least-squares fit of the training residuals on the element counts (minimum norm), training data only
(fair protocol).  It corrects the per-composition energy offsets that a single per-atom gauge cannot.
"""
from __future__ import annotations

import argparse
import functools
import io
import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


def _refit_path(ckpt_path) -> Path:
    return Path(ckpt_path).with_suffix('.species_refit.json')


class Predictor:
    def __init__(self, ckpt_path, config=None, species_refit: bool = True, batch_size: int = 16,
                 max_atoms: int = 2048):
        import jax
        from bam_omat24 import train_3bpa as T
        from bam_omat24.training_utils import load_checkpoint
        ckpt = load_checkpoint(str(ckpt_path))
        if config is None:
            config = ckpt.get('config')
        elif not isinstance(config, dict):
            config = json.load(open(config))
        if config is None:
            raise ValueError('the checkpoint has no config; pass config=')
        cfg = self.cfg = dict(config)
        if cfg.get('x64'):
            jax.config.update('jax_enable_x64', True)
        self.atomic_numbers = [int(z) for z in cfg['atomic_numbers']]
        if 'atom_energies' not in cfg:
            raise ValueError("config without atom_energies (e0_mode computed from the data): pass a config that lists "
                             "them")
        self.atom_energies = np.asarray(cfg['atom_energies'], np.float64)
        if cfg.get('use_dataset_stats') and 'avg_n_neighbors' not in cfg:
            raise ValueError('use_dataset_stats without avg_n_neighbors in the config: add the training value')
        stats = {'shift_raw': 0.0, 'scale': 1.0, 'avg_n_neighbors': float(cfg.get('avg_n_neighbors', 25.0)),
                 'n_graphs': 1, 'n_atoms': 1}
        bcfg = {'learning_rate': 1e-3, **cfg}       # build_model also creates the optimiser; not used here
        graphdef, *_ = T.build_model(bcfg, len(self.atomic_numbers), list(self.atom_energies), stats, io.StringIO())
        from flax import nnx
        import jax.numpy as jnp
        self.model = nnx.merge(graphdef, jax.tree.map(jnp.asarray, ckpt['ema_params']))
        self.batch_size = int(batch_size)
        # RACE_LR solves the Qeq system densely over a whole batch (N_batch^2 memory, N_batch^3 time), so batches
        # are cut at max_atoms real atoms (a larger single frame runs alone)
        self.max_atoms = int(max_atoms)
        self.delta = np.zeros(len(self.atomic_numbers))
        rp = _refit_path(ckpt_path)
        if species_refit and rp.exists():
            r = json.load(open(rp))
            if [int(z) for z in r['atomic_numbers']] != self.atomic_numbers:
                raise ValueError(f'{rp}: element list differs from the config')
            self.delta = np.asarray(r['delta_eV'], np.float64)
        self._step = jax.jit(lambda b: self.model(b))
        self._T = T

    def _graphs(self, frames):
        from bam_omat24.data_3bpa import preprocess_graph, atomic_numbers_to_indices
        from bam_omat24.analyze_errors import _tag_frames
        T, cfg = self._T, self.cfg
        ai = atomic_numbers_to_indices(self.atomic_numbers)
        kw = dict(charge_key=T._get(cfg, 'total_charge_key', None), spin_key=T._get(cfg, 'total_spin_key', None),
                  fragments=T.fragments_of(cfg), task=T._get(cfg, 'task', None))
        graphs = [preprocess_graph(a, ai, list(self.atom_energies), float(T._get(cfg, 'cutoff', 5.0)),
                                   targets=False, **kw) for a in frames]
        m = self.model
        if getattr(m, 'ct_cond', False) or getattr(m, 'cs_local', False) or getattr(m, 'spin_energy', False):
            # RACE_LR inputs that training builds in stream_io (2026-11-15, as md_water/race_jax_calc.py): integer
            # charges q0 / spins s0 per molecule (atoms.arrays['formal_charges'] / ['formal_spins'], else labels,
            # else graph_fields.formal_charges(auto): rules from the bonds), and the restricted / unrestricted flag.
            # Without them the model falls back to Q / N and (2S) / N on every atom.
            from bam_omat24.graph_fields import formal_charges, formal_spins
            ck, sk = kw['charge_key'], kw['spin_key']
            for a, g in zip(frames, graphs):
                S = None
                if np.all(a.pbc) and isinstance(g.edges, dict) and 'Sij' in g.edges:
                    S = -np.asarray(g.edges['Sij'], np.float64) @ np.asarray(a.cell.array, np.float64)
                Q = float(a.info.get(ck, 0.0)) if ck else 0.0
                mult = float(a.info.get(sk, 1.0)) if sk else 1.0
                rcv, snd = np.asarray(g.receivers), np.asarray(g.senders)
                if getattr(m, 'ct_cond', False) or getattr(m, 'cs_local', False):
                    q0, _ = formal_charges(a.numbers, a.get_positions(), rcv, snd, Q, a.info, a.arrays, S, auto=True,
                                           multiplicity=mult)
                    g.nodes['q0'] = q0.astype(np.float32)
                    if getattr(m, 'cs_local', False):
                        s0, _ = formal_spins(a.numbers, a.get_positions(), rcv, snd, mult, q0, None, a.arrays, S)
                        g.nodes['s0'] = s0.astype(np.float32)
                if getattr(m, 'spin_energy', False) and isinstance(g.globals, dict):
                    u = bool(a.info.get('unrestricted', mult > 1))
                    g.globals['unrestricted'] = np.array([1.0 if (u or mult > 1) else 0.0], np.float32)
        if cfg.get('x64'):
            graphs = T._cast_graphs(graphs, np.float64)
        return _tag_frames(graphs)

    def _chunks(self, frames):
        out, cur, n_at = [], [], 0
        for i, a in enumerate(frames):
            if cur and (len(cur) >= self.batch_size or n_at + len(a) > self.max_atoms):
                out.append(cur); cur, n_at = [], 0
            cur.append(i); n_at += len(a)
        if cur:
            out.append(cur)
        return out

    def _run(self, frames, want):
        n = len(frames)
        energies, forces, charges = np.full(n, np.nan), [None] * n, [None] * n
        for idx in self._chunks(frames):
            e, f, q = self._run_batch([frames[i] for i in idx], want)
            for j, i in enumerate(idx):
                energies[i], forces[i], charges[i] = e[j], f[j], q[j]
        return energies, forces, charges

    def _run_batch(self, frames, want):
        import jraph
        from flax import nnx
        from bam_omat24.data_3bpa import GraphListDataset
        from bam_omat24.data_nnx import BucketedDataLoader, default_collate_fn
        T, cfg = self._T, self.cfg
        n_tot = sum(len(a) for a in frames)
        # one batch per chunk: lift the loader's node cap (default 256) above the chunk size
        loader = BucketedDataLoader(GraphListDataset(self._graphs(frames)), batch_size=len(frames), n_buckets=1,
                                    shuffle=False, drop_last=False, rngs=nnx.Rngs(0), max_nodes_per_batch=n_tot + 1)
        loader.collate_fn = functools.partial(default_collate_fn,
                                              node_bucket_size=int(T._get(cfg, 'node_bucket_size', 128)),
                                              edge_bucket_size=int(T._get(cfg, 'edge_bucket_size', 4096)))
        n = len(frames)
        energies, forces, charges = np.full(n, np.nan), [None] * n, [None] * n
        for batch in loader:
            out = self._step(batch)
            gmask = np.asarray(jraph.get_graph_padding_mask(batch))
            fid = np.asarray(batch.globals['frame_id']).reshape(-1)
            off = np.concatenate([[0], np.cumsum(np.asarray(batch.n_node))])
            E = np.asarray(out['energy']).reshape(-1)
            F = np.asarray(out['forces'])
            q = np.asarray(out['charges']) if ('charges' in out and want == 'charges') else None
            for g in np.flatnonzero(gmask):
                i = int(fid[g])
                energies[i] = float(E[g])
                forces[i] = F[off[g]:off[g + 1]].astype(np.float64)
                if q is not None:
                    charges[i] = q[off[g]:off[g + 1]].astype(np.float64)
        return energies, forces, charges

    def counts(self, frames) -> np.ndarray:
        return np.array([[int(np.sum(a.numbers == z)) for z in self.atomic_numbers] for a in frames], float)

    def predict(self, frames: Sequence, total: bool = True):
        """(energies (n_frames,) eV, forces list of (n_atoms, 3) eV/Å).  total=False: the raw model output (energy
        relative to the E0 of the config, no refit)."""
        frames = list(frames)
        e, f, _ = self._run(frames, 'energy')
        if total:
            c = self.counts(frames)
            e = e + c @ self.atom_energies + c @ self.delta
        return e, f

    def charges(self, frames: Sequence) -> List[np.ndarray]:
        _, _, q = self._run(list(frames), 'charges')
        if any(x is None for x in q):
            raise ValueError('the model returns no charges (not a RACE_LR checkpoint with the Qeq stratum)')
        return q


def load(path, **kw) -> Predictor:
    return Predictor(path, **kw)


def species_refit(ckpt_path, frames, max_frames: Optional[int] = None, seed: int = 0) -> dict:
    """Per-element energy correction from training frames (with reference energies): least squares of
    (E_pred - E_ref) on the element counts, minimum norm.  Writes <ckpt>.species_refit.json and returns it."""
    frames = list(frames)
    if max_frames and len(frames) > max_frames:
        idx = np.random.RandomState(seed).choice(len(frames), max_frames, replace=False)
        frames = [frames[i] for i in sorted(idx)]
    p = Predictor(ckpt_path, species_refit=False)
    e, _ = p.predict(frames, total=True)
    ref = np.array([a.get_potential_energy() for a in frames])
    X = p.counts(frames)
    res = e - ref
    coef, _, rank, _ = np.linalg.lstsq(X, res, rcond=None)
    out = dict(atomic_numbers=p.atomic_numbers, delta_eV=[float(-c) for c in coef], rank=int(rank),
               n_frames=len(frames), rmse_before_eV=float(np.sqrt(np.mean(res ** 2))),
               rmse_after_eV=float(np.sqrt(np.mean((res - X @ coef) ** 2))),
               note='training-split refit; delta is added per atom of each element to the predicted total energy')
    json.dump(out, open(_refit_path(ckpt_path), 'w'), indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('refit', help='per-element energy refit on training frames')
    r.add_argument('ckpt')
    r.add_argument('--xyz', required=True, action='append')
    r.add_argument('--max-frames', type=int, default=20000)
    e = sub.add_parser('eval', help='energy / force errors of frames with reference labels')
    e.add_argument('ckpt')
    e.add_argument('--xyz', required=True)
    args = ap.parse_args()
    import ase.io
    if args.cmd == 'refit':
        frames = [a for x in args.xyz for a in ase.io.read(x, ':')]
        out = species_refit(args.ckpt, frames, args.max_frames)
        print(json.dumps({k: v for k, v in out.items() if k != 'delta_eV'}, indent=1))
    else:
        frames = ase.io.read(args.xyz, ':')
        p = load(args.ckpt)
        en, fo = p.predict(frames)
        ref = np.array([a.get_potential_energy() for a in frames])
        fe = np.concatenate([(f - a.get_forces()).ravel() for f, a in zip(fo, frames)])
        print(f'{len(frames)} frames: E RMSE {np.sqrt(np.mean((en - ref) ** 2)) * 1e3:.2f} meV, '
              f'F RMSE {np.sqrt(np.mean(fe ** 2)) * 1e3:.2f} meV/A (species refit '
              f'{"on" if np.any(p.delta) else "off"})')



def ase_calculator(ckpt_path, **kw):
    """ASE calculator on a Predictor (energy, forces, and charges for RACE_LR): for MD and geometry scans.
    The Atoms object carries info['charge'] / ['spin'] / fragment data under the config's key names."""
    from ase.calculators.calculator import Calculator, all_changes

    pred = ckpt_path if isinstance(ckpt_path, Predictor) else Predictor(ckpt_path, **kw)

    class RACECalculator(Calculator):
        implemented_properties = ['energy', 'free_energy', 'forces', 'charges']

        def __init__(self):
            super().__init__()
            self.predictor = pred

        def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            a = self.atoms.copy()
            a.info.update(atoms.info)
            e, f = pred.predict([a])
            self.results = {'energy': float(e[0]), 'free_energy': float(e[0]), 'forces': f[0]}
            if 'charges' in properties:
                self.results['charges'] = pred.charges([a])[0]

    return RACECalculator()


if __name__ == '__main__':
    main()
