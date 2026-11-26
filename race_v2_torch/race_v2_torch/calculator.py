"""ASE calculator for race_v2_torch models (energy, free_energy, forces, stress, energies).

    from race_v2_torch.calculator import RACEv2Calculator
    atoms.calc = RACEv2Calculator('model.pt')            # device='cpu' | 'cuda'
    atoms.calc = RACEv2Calculator('model.pt', device='cuda', tp_kernel='cue', dtype='float32')     # fast path (2026-11-24)
    atoms.get_potential_energy(); atoms.get_forces(); atoms.get_stress()

Energies are TOTAL energies, model output + sum of the config's atom_energies (E0) per atom in float64, as
race_v2/inference/predict.py does.  Neighbour list: ase.neighborlist.neighbor_list('ijS'), the backend of the training graphs
(race_v2/data/neighbors.py, BAM_NEIGHBORS=ase): receiver i, sender j, Sij = -S.

``jax_compat=True`` rounds positions, cell and volume to float32 before the float64 evaluation, as
race_v2/data_3bpa.preprocess_graph does for the JAX model (use it to reproduce predict.py numbers exactly; the
default evaluates the given float64 geometry).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from .model import RACEv2

# Core exclusion of the RACE v2.1 materials models (repository README, "Inference: core exclusion"): minimum distances
# of the MPtrj + sAlex + OMat24 training data, cap 0.5 (R_i + R_j), width 0.1 A.
def _default_table_path() -> str:
    """The table is package data of race_v2 (race_v2/resources).  It is located without importing race_v2 (that would
    import JAX): find_spec of a top-level name only searches the path.  Installed (pip, same distribution as race_v2)
    or in the repository with race_v2 on the path: race_v2's own folder; else the repository layout next to this file."""
    import importlib.util
    name = 'mindist_train_mptrj_salex_omat24.json'
    candidates = []
    try:
        spec = importlib.util.find_spec('race_v2')
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.submodule_search_locations:
        candidates += [os.path.join(p, 'resources', name) for p in spec.submodule_search_locations]
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    candidates.append(os.path.join(repo, 'race_v2', 'resources', name))
    return next((p for p in candidates if os.path.exists(p)), candidates[-1])


DEFAULT_CORE_EXCLUSION = {'table': [_default_table_path()], 'cap': 0.5, 'width': 0.1}


def neighbor_graph(atoms, cutoff: float):
    """(receivers i, senders j, Sij) with the edge vector x_j - (x_i + Sij . cell)."""
    import ase.neighborlist
    if np.any(atoms.pbc):
        i, j, S = ase.neighborlist.neighbor_list('ijS', atoms, cutoff)
    else:
        i, j = ase.neighborlist.neighbor_list('ij', atoms, cutoff)
        S = np.zeros((len(i), 3), dtype=np.int64)
    return i.astype(np.int64), j.astype(np.int64), (-S).astype(np.float64)


class RACEv2Calculator(Calculator):
    implemented_properties = ['energy', 'free_energy', 'forces', 'stress', 'energies']

    def __init__(self, model, device: str = 'cpu', jax_compat: bool = False, e0: str = 'float64',
                 edge_chunk: int = 0, dtype: str = 'float64', tp_kernel: str = 'torch', core_exclusion=True, **kw):
        """dtype: 'float64' (default; the thermal conductivity needs it, kappa SRME 0.229 in float64 and 0.861 in
        float32 for RACE v2.0) or 'float32'.
        core_exclusion: True (default) = on, with the r_in table stored in the model file when it has one, else the
        table of the RACE v2.1 training data (DEFAULT_CORE_EXCLUSION); False = off; a dict = that specification
        ({"table": ..., "cap": 0.5, "width": 0.1}).  The same defaults as race_v2.inference.predict.ase_calculator."""
        super().__init__(**kw)
        if dtype not in ('float64', 'float32'):
            raise ValueError("dtype must be 'float64' or 'float32'")
        self.dtype = torch.float64 if dtype == 'float64' else torch.float32
        self.model = RACEv2.load(model, map_location=device) if not isinstance(model, RACEv2) else model
        self.model = self.model.to(device).to(self.dtype).eval()
        if isinstance(core_exclusion, dict):
            self.model.set_core_exclusion(core_exclusion)
        elif core_exclusion is False:
            self.model.set_core_exclusion(None)
        elif core_exclusion is True and self.model.core_r_in is None:
            if not os.path.exists(DEFAULT_CORE_EXCLUSION['table'][0]):
                raise FileNotFoundError(f"core exclusion table {DEFAULT_CORE_EXCLUSION['table'][0]} not found (package "
                                        "data race_v2/resources); pass core_exclusion={...} or core_exclusion=False")
            self.model.set_core_exclusion(DEFAULT_CORE_EXCLUSION)
        self.model.use_tp_kernel(tp_kernel)
        self.model.edge_chunk = int(edge_chunk)
        self.device = device
        self.jax_compat = bool(jax_compat)
        self.e0 = e0
        self.z_to_index = {z: k for k, z in enumerate(self.model.atomic_numbers)}

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        a = self.atoms
        try:
            species = np.array([self.z_to_index[int(z)] for z in a.numbers], dtype=np.int64)
        except KeyError as e:
            raise ValueError(f"element Z={e.args[0]} is not in the model's atomic_numbers") from None
        recv, send, sij = neighbor_graph(a, self.model.cutoff)
        pos = np.asarray(a.positions, np.float64)
        cell = np.asarray(a.cell.array, np.float64)
        periodic = bool(np.any(a.pbc))
        vol = abs(float(np.linalg.det(cell))) if periodic else 0.0
        if self.jax_compat:
            pos = pos.astype(np.float32).astype(np.float64)
            cell = cell.astype(np.float32).astype(np.float64)
            vol = float(np.float32(a.get_volume())) if periodic else 0.0
        dev, dt = self.device, self.dtype
        T = lambda x, d=dt: torch.as_tensor(x, dtype=d, device=dev)
        want_stress = self.model.periodic and bool(np.all(a.pbc))     # same backward pass as the forces
        pbc_model = self.model.periodic          # a molecular (periodic false) model ignores cells, as in JAX
        out = self.model(T(pos), T(species, torch.long), T(send, torch.long), T(recv, torch.long),
                         shifts=(T(sij) if pbc_model else None), cell=(T(cell[None]) if pbc_model else None),
                         n_node=T([len(a)], torch.long), n_edge=T([len(send)], torch.long),
                         cutoff=(T([self.model.cutoff]) if pbc_model else None),
                         volume=(T([vol]) if periodic else None), compute_forces=True,
                         compute_stress=want_stress, e0=(self.e0 if dt == torch.float64 else None))
        if dt == torch.float64:
            e = float(out['energy'][0])
            energies = out['node_energies_total'].detach().cpu().numpy()
        else:
            # float32 model: the E0 of every atom is added in float64 on the host (a float32 total energy of a
            # large cell would lose the digits of the model part)
            e0_tab = np.asarray(self.model.atom_energies.detach().cpu().numpy(), np.float64)
            if self.e0 == 'float32':
                e0_tab = e0_tab.astype(np.float32).astype(np.float64)
            energies = out['node_energies_total'].detach().cpu().numpy().astype(np.float64) + e0_tab[species]
            e = float(energies.sum())
        self.results = {'energy': e, 'free_energy': e,
                        'forces': out['forces'].detach().cpu().numpy().astype(np.float64), 'energies': energies}
        if want_stress:
            self.results['stress'] = out['stress'][0].detach().cpu().numpy().astype(np.float64)
