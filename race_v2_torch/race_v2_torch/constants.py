"""Element tables used by RACEv2 (fixed, not trained): UFF van der Waals constants, covalent radii, valence-shell
radii.  Generated from the JAX sources by tools/make_element_tables.py (see data/element_tables.json)."""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import List, Sequence


@functools.lru_cache(maxsize=1)
def tables() -> dict:
    return json.load(open(Path(__file__).with_name('data') / 'element_tables.json'))


def uff_rows(species_z: Sequence[int]) -> List[List[float]]:
    """(n_species, 3): x_I [A], D_I [eV], zeta_I (models.race.UFF_VDW)."""
    t = tables()['uff_vdw']
    missing = [z for z in species_z if str(int(z)) not in t]
    if missing:
        raise ValueError(f"pair_repulsion 'uff': no UFF vdW parameters for Z={missing}")
    return [list(t[str(int(z))]) for z in species_z]


def covalent_radii(species_z: Sequence[int]) -> List[float]:
    """(n_species,) Cordero covalent radii [A] (training.train.COVALENT_RADII)."""
    t = tables()['covalent_radii']
    return [float(t[str(int(z))]) for z in species_z]


def shell_radii(species_z: Sequence[int]) -> List[List[float]]:
    """(n_species, 4): <r> of the outermost s, p, d, f shell [A], 0 = absent (training.train.element_shell_radii)."""
    t = tables()['shell_radii']
    return [list(t[str(int(z))]) for z in species_z]
