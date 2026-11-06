"""RACE_LR: RACEv2 with charge-dependent long-range electrostatics (2026-11-06; docs/RACE_LR_design.md).
RACE itself (race_nnx.py) is unchanged; this subclass adds one physical stratum:

    E = sum_i [ local RACEv2 strata + repulsion ]_i  +  E_Qeq({q_i})

    chi_i, J_i = MLP(scalar features of the last layer) + per-species offsets   (electronegativity, hardness > 0)
    q = argmin_q  chi . q + 1/2 q^T A q   s.t.  sum_i q_i = Q_graph   (charge equilibration, one linear solve)
    A_ij = Gaussian-charge Coulomb (direct sum, or Ewald for periodic graphs), A_ii = J_i + K / (sqrt(pi) sigma_i)

The total charge Q of each graph comes from data.globals['charge'] (0 if absent; train_3bpa reads it from the extxyz
info key given by cfg "total_charge_key").  Charges are latent: trained from energies and forces only (no charge
labels), as in latent-charge / 4G-HDNNP-type models.  Forces and stress follow from autodiff through the linear
solve.  Config: "model": "RACE_LR", "long_range": {"ewald_alpha": 0.3, "ewald_n_real": 2, "ewald_n_recip": 6,
"head_hidden": 32, "chi_init": "zero" | "pauling", "chi_init_scale": 3.0, "hard_init": 5.0}.  Optional charge
loss on reference charges (e.g. Hirshfeld): cfg "charge_label_key" + "charge_weight" (train_3bpa; default off).  Periodic graphs: Ewald per graph (dense, for small cells; see design
doc for limits).
"""
from __future__ import annotations

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import jraph
from flax import nnx

from bam_omat24.long_range import coulomb_direct, ewald_matrix_rows, qeq_solve
from bam_omat24.race_nnx import RACE, _dense

LR_DEFAULTS = {"ewald_alpha": 0.3, "ewald_n_real": 2, "ewald_n_recip": 6, "head_hidden": 32, "chi_scale": 1.0,
               "hard_init": 5.0, "chi_init": "zero", "chi_init_scale": 3.0}

# Pauling electronegativities (CRC Handbook values as in common tables; noble gases He, Ne, Ar have none).  Used only
# by "chi_init": "pauling": chi0_Z = chi_init_scale * (chi_P(Z) - mean over the species), in eV.  chi_init_scale = 3 eV
# per Pauling unit is the approximate slope of the Mulliken scale (chi_M in eV ~ 3 chi_P).
PAULING = {
    1: 2.20, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90,
    15: 2.19, 16: 2.58, 17: 3.16, 19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55, 26: 1.83,
    27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65, 31: 1.81, 32: 2.01, 33: 2.18, 34: 2.55, 35: 2.96, 36: 3.00, 37: 0.82,
    38: 0.95, 39: 1.22, 40: 1.33, 41: 1.60, 42: 2.16, 43: 1.90, 44: 2.20, 45: 2.28, 46: 2.20, 47: 1.93, 48: 1.69,
    49: 1.78, 50: 1.96, 51: 2.05, 52: 2.10, 53: 2.66, 54: 2.60, 55: 0.79, 56: 0.89, 57: 1.10, 58: 1.12, 59: 1.13,
    60: 1.14, 62: 1.17, 64: 1.20, 66: 1.22, 67: 1.23, 68: 1.24, 69: 1.25, 71: 1.27, 72: 1.30, 73: 1.50, 74: 2.36,
    75: 1.90, 76: 2.20, 77: 2.20, 78: 2.28, 79: 2.54, 80: 2.00, 81: 1.62, 82: 2.33, 83: 2.02, 84: 2.00, 85: 2.20,
}


class RACE_LR(RACE):
    def __init__(self, *args, long_range=None, **kw):
        super().__init__(*args, **kw)
        lr = dict(LR_DEFAULTS)
        lr.update(long_range or {})
        self.lr = lr
        rngs = kw['rngs']
        n_species = kw['n_species']
        n0 = sum(mul for mul, ir in e3nn.Irreps(kw['hidden_irreps']) if ir == e3nn.Irrep('0e'))
        h = int(lr['head_hidden'])
        self.lr_head1 = nnx.Linear(n0, h, rngs=rngs)
        self.lr_head2 = nnx.Linear(h, 2, rngs=rngs, kernel_init=nnx.initializers.zeros_init())
        chi_init = str(lr['chi_init'])
        if chi_init == 'zero':
            chi0 = jnp.zeros((n_species,), jnp.float32)
        elif chi_init == 'pauling':
            zs = self.species_z
            if zs is None or len(zs) != n_species:
                raise ValueError("chi_init 'pauling' needs species_z (one atomic number per species)")
            missing = [z for z in zs if z not in PAULING]
            if missing:
                raise ValueError(f"chi_init 'pauling': no Pauling electronegativity for Z={missing}")
            v = jnp.asarray([PAULING[z] for z in zs], jnp.float32)
            chi0 = float(lr['chi_init_scale']) * (v - jnp.mean(v))
        else:
            raise ValueError(f"long_range chi_init must be 'zero' or 'pauling', got {chi_init!r}")
        self.chi0 = nnx.Param(chi0)
        # hardness J = softplus(raw) > 0, initialised at hard_init (eV/e^2); Gaussian widths from covalent radii
        raw = jnp.log(jnp.expm1(jnp.asarray(float(lr['hard_init']))))
        self.hard0 = nnx.Param(jnp.full((n_species,), raw, jnp.float32))
        radii = kw.get('species_radii') or [1.0] * n_species
        self.log_sigma = nnx.Param(jnp.log(jnp.asarray(radii, jnp.float32)))

    def node_energies(self, positions, cell, cutoff, data, *args, **kw):
        out = super().node_energies(positions, cell, cutoff, data, *args, **kw)
        species = data.nodes['species']
        n_nodes = species.shape[0]
        n_graphs = data.n_node.shape[0]
        node_mask = (jraph.get_node_padding_mask(data) if n_graphs > 1 else jnp.ones((n_nodes,), bool))
        node_graph = jnp.repeat(jnp.arange(n_graphs), data.n_node, total_repeat_length=n_nodes)
        s = _dense(out['node_features']).filter(keep='0e').array
        dt = s.dtype
        z = self.lr_head2(jax.nn.silu(self.lr_head1(s)))
        chi = (self.chi0[...][species] + float(self.lr['chi_scale']) * z[:, 0]).astype(dt)
        hard = jax.nn.softplus(self.hard0[...][species] + z[:, 1]).astype(dt)
        sigma = jnp.exp(self.log_sigma[...][species]).astype(dt)
        Q = data.globals.get('charge') if isinstance(data.globals, dict) else None
        Q = jnp.zeros((n_graphs,), dt) if Q is None else jnp.asarray(Q, dt).reshape(n_graphs)
        same = (node_graph[:, None] == node_graph[None, :]) & node_mask[:, None] & node_mask[None, :]
        if self.periodic and cell is not None:
            gamma = self._ewald_batch(positions, cell, sigma, node_graph, same, n_graphs)
        else:
            gamma = coulomb_direct(positions, sigma, same)
        q, e = qeq_solve(chi, hard, sigma, gamma, node_graph, node_mask, Q, n_graphs)
        out = dict(out)
        out['node_energies'] = out['node_energies'] + e.reshape(out['node_energies'].shape).astype(out['node_energies'].dtype)
        out['charges'] = q
        return out

    def _ewald_batch(self, positions, cell, sigma, node_graph, same, n_graphs):
        """Ewald matrix of a batch of periodic graphs: every row uses the cell of its own graph; pairs of different
        graphs are masked (memory N^2 x images + N x k-vectors; small cells, see the design doc)."""
        cells = cell.reshape(n_graphs, 3, 3)
        lr = self.lr
        gamma = ewald_matrix_rows(positions, cells[node_graph], sigma, float(lr['ewald_alpha']),
                                  int(lr['ewald_n_real']), int(lr['ewald_n_recip']))
        return jnp.where(same, gamma, 0.0)
