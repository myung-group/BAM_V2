"""ASE calculator for racev2_torch models (energy, free_energy, forces, stress, energies).

    from racev2_torch.calculator import RACEv2Calculator
    atoms.calc = RACEv2Calculator('model.pt')            # device='cpu' | 'cuda'
    atoms.get_potential_energy(); atoms.get_forces(); atoms.get_stress()

Energies are TOTAL energies, model output + sum of the config's atom_energies (E0) per atom in float64, as
bam_omat24/predict.py does.  Neighbour list: ase.neighborlist.neighbor_list('ijS'), the backend of the training graphs
(bam_omat24/neighbors.py, BAM_NEIGHBORS=ase): receiver i, sender j, Sij = -S.

``jax_compat=True`` rounds positions, cell and volume to float32 before the float64 evaluation, as
bam_omat24/data_3bpa.preprocess_graph does for the JAX model (use it to reproduce predict.py numbers exactly; the
default evaluates the given float64 geometry).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from .model import RACEv2


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
                 edge_chunk: int = 0, **kw):
        super().__init__(**kw)
        self.model = RACEv2.load(model, map_location=device) if not isinstance(model, RACEv2) else model
        self.model = self.model.to(device).eval()
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
        dev, dt = self.device, torch.float64
        T = lambda x, d=dt: torch.as_tensor(x, dtype=d, device=dev)
        want_stress = self.model.periodic and bool(np.all(a.pbc))     # same backward pass as the forces
        pbc_model = self.model.periodic          # a molecular (periodic false) model ignores cells, as in JAX
        out = self.model(T(pos), T(species, torch.long), T(send, torch.long), T(recv, torch.long),
                         shifts=(T(sij) if pbc_model else None), cell=(T(cell[None]) if pbc_model else None),
                         n_node=T([len(a)], torch.long), n_edge=T([len(send)], torch.long),
                         cutoff=(T([self.model.cutoff]) if pbc_model else None),
                         volume=(T([vol]) if periodic else None), compute_forces=True,
                         compute_stress=want_stress, e0=self.e0)
        e = float(out['energy'][0])
        self.results = {'energy': e, 'free_energy': e, 'forces': out['forces'].detach().cpu().numpy(),
                        'energies': out['node_energies_total'].detach().cpu().numpy()}
        if want_stress:
            self.results['stress'] = out['stress'][0].detach().cpu().numpy()
