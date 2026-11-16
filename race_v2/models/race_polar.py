"""RACE_POLAR: RACEv2 with Thole-type polarizable electrostatics (2026-11-11; docs/HANDOFF_multidataset_thole.md on branch racev2).

JAX port of the "polar" block of BAM-torch (bam_ufo/model/thole.py, TholeElectrostaticsBlock, mode "pbc_split");
the electrostatics are in race_v2/models/thole.py.  RACE and RACE_LR are unchanged; this subclass of RACE_LR (for its
charge / spin embedding and task table; the Qeq stratum stays off) adds one physical stratum:

    E = sum_i [ local RACEv2 strata + repulsion ]_i + E_elec({q_i, mu_i, Theta_i, alpha_i})

Heads on the (normalised) last-layer node features h_i, as in the torch block:
    q_i^raw, mu_i   = multipole_init_scale * Linear(h -> 1x0e + 1x1o)        permanent Gaussian charge and dipole
    Theta_i         = multipole_init_scale * Linear(h -> 1x2e), zero init    traceless quadrupole (needs 2e features)
    alpha_i         = min + (max_Z - min) sigmoid(Linear(h -> 0e) + bias)    polarizability, Linear zero init,
                                                                             bias such that alpha = polarizability_init
    f_i             = MLP(Linear(h -> 16x0e)) / T                            Fukui logits, 1/T learned (softplus)
    q_i             = q_i^raw + softmax_graph(f)_i (Q_graph - sum q^raw)     exact total charge of every graph
Feature normalisation "smooth": every irrep type is multiplied by rsqrt(1 + mean square / scale^2) per atom.

Config: "model": "RACE_POLAR", "polar": {thole.POLAR_DEFAULTS; for periodic data "max_cell_length" and
"max_cell_volume" (or "k_grid" and "n_k") size the reciprocal sums: training.train fills them from the stream size
tables}, "long_range": {"charge_spin_embedding": true, "n_tasks": N} as for RACE_LR (qeq is forced off).
"polar": {"charges": "qeq"} (2026-11-11) = RACE_LR's charge equilibration + this block: q_i are the Qeq charges
(long_range options chi_init, qeq_block, fragments; the heads q^raw and f are unused), Qeq adds only its on-site
energy chi q + 1/2 (J + self) q^2, and E_elec below carries the Coulomb energy of q with mu, Theta and the induced
dipoles.
Boundary conditions per graph: data.globals['pbc'] (stream option "mixed"); without it every graph is periodic
when the model is periodic and open otherwise.  The open-boundary pair sums (dense N x N per batch) are built
only when open graphs can occur ("open_boundary": "auto").  The total charge comes from data.globals['charge'] (0 if absent).

Phase 2 options of "polar" (2026-11-11), as in the BAM-torch input.json:
    "charge_constraint": "electrode"   target charge per charge group (data.nodes['charge_group'], ['fixed_charge'],
                                       from the *_meta_v2 files: stream sources "meta_dir"; group 0 = reservoir /
                                       electrode atoms takes the rest of the graph's charge); "graph" (default)
    "charge_transfer": "contact"       groups in contact exchange charge (contact_radius, transfer_cutoff,
                                       contact_pairs, contact_eps; thole.contact_charges)
    "slab_correction": true            slab dipole correction on the graphs with data.globals['slab'] (stream
                                       sources "slab": true), slab_axis; neutral cells only

"dispersion": true (2026-11-15): pairwise C6 dispersion from the atom-in-molecule polarizabilities alpha_i of this
block, the long-range van der Waals attraction the 6 A network cutoff leaves out (OMol25 wB97M-V has VV10):
    C6_ii = kappa_Z alpha_i^2   (London: C6 ~ 3/4 I alpha^2; kappa per element, softplus, init disp_kappa_init eV)
    C6_ij = 2 C6_ii C6_jj / (alpha_j / alpha_i C6_ii + alpha_i / alpha_j C6_jj)            (Tkatchenko-Scheffler)
    E     = - sum_{i<j} C6_ij / (r^6 + R0_ij^6),  R0_ij = a1 (R_i + R_j) + disp_a2,  R_i = 2.54 alpha_i^(1/7) bohr
            (Becke-Johnson damping, vdW radius from alpha as Fedorov-Tkatchenko 2018; a1 learned, init disp_a1)
Open-boundary graphs: all pairs of the structure (blocks of long_range.qeq_block atoms, no cutoff).  Periodic
graphs: lattice images |n_k| <= disp_n_img, smoothly cut between disp_cutoff - 2 A and disp_cutoff, plus the
uniform-density tail -2 pi / (3 V) sum_ij C6_ij / (disp_cutoff - 1)^3.  alpha keeps its full gradient: alpha depends on
the positions, and cutting it (a stop_gradient tried first) made the forces non-conservative (finite differences
off by 6e-4 eV/A).  The dispersion therefore also trains alpha.  Output 'dispersion_node_energies'.
Outputs next to the RACE ones: 'charges' (N,), 'dipole' (G, 3): open-boundary graphs about their centre of mass
(ORCA convention, the OMol25 / OPoly26 labels), periodic graphs sum_i q_i r_i + mu_i (fragments unwrapped with the
slab correction); 'dipole_normal' (G,) its component along 'slab_normal' (G, 3), the label of slab sources (OC25).
"""
from __future__ import annotations

import functools
import math

import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
import jraph
import numpy as np
from flax import nnx

from race_v2.models import thole
from race_v2.models.linear import Linear as e3nn_nnx_Linear
from race_v2.models.race_lr import RACE_LR
from race_v2.models.race import _dense
from race_v2.training.utils import _traceless_symmetric_from_2e


@functools.lru_cache(maxsize=8)
def _grid(k_grid):
    return thole.half_grid(k_grid)


def _zero_linear(lin):
    for name, v in list(vars(lin).items()):
        if isinstance(v, nnx.Param):
            setattr(lin, name, nnx.Param(jnp.zeros_like(v[...])))
    return lin


class RACE_POLAR(RACE_LR):
    _allow_plain = True          # RACE_LR without Qeq / embedding / tasks is allowed here (the polar stratum is new)

    def __init__(self, *args, polar=None, long_range=None, **kw):
        lr = dict(long_range or {})
        mode = (polar or {}).get('charges', thole.POLAR_DEFAULTS['charges'])
        if mode not in ('fukui', 'qeq'):
            raise ValueError("polar.charges must be 'fukui' or 'qeq'")
        if mode == 'qeq':
            # Qeq + Thole: the charges are the Qeq solution (environment-dependent chi and hardness, total charge or
            # fragment constraints, long_range options as for RACE_LR); Qeq contributes its on-site energy only and
            # the Thole block computes the Coulomb energy of these charges with the dipoles, quadrupoles and
            # induced dipoles (no double counting).  The energy is not variational in q; forces are exact (autodiff
            # through the linear solve).
            if lr.get('qeq') is False:
                raise ValueError("RACE_POLAR with polar.charges 'qeq' needs the Qeq stratum (long_range.qeq)")
            if lr.get('coulomb_energy'):
                raise ValueError("RACE_POLAR with polar.charges 'qeq': long_range.coulomb_energy must be false")
            if (polar or {}).get('charge_constraint', 'graph') != 'graph':
                raise ValueError("polar.charges 'qeq' supports charge_constraint 'graph' only (use the fragment "
                                 "constraints of RACE_LR for per-fragment charges)")
            lr['qeq'], lr['coulomb_energy'] = True, False
        else:
            if lr.get('qeq'):
                raise ValueError("RACE_POLAR: long_range.qeq must be false unless polar.charges is 'qeq'")
            lr['qeq'] = False
        super().__init__(*args, long_range=lr, **kw)
        p = dict(thole.POLAR_DEFAULTS)
        unknown = [k for k in (polar or {}) if k not in p and not k.startswith('_')]
        if unknown:
            raise ValueError(f"polar: unknown options {unknown}; known {sorted(p)}")
        p.update({k: v for k, v in (polar or {}).items() if not k.startswith('_')})
        if p['feature_norm'] not in ('smooth', 'none'):
            raise ValueError("polar.feature_norm must be 'smooth' or 'none'")
        if p['charge_constraint'] not in ('graph', 'electrode'):
            raise ValueError("polar.charge_constraint must be 'graph' or 'electrode'")
        if p['charge_transfer'] not in ('none', 'contact'):
            raise ValueError("polar.charge_transfer must be 'none' or 'contact'")
        if p['charge_transfer'] == 'contact' and p['charge_constraint'] != 'electrode':
            raise ValueError("polar.charge_transfer 'contact' needs charge_constraint 'electrode' (charge groups)")
        if p['slab_correction'] and not kw.get('periodic'):
            raise ValueError("polar.slab_correction needs a periodic model")
        rngs, n_species = kw['rngs'], kw['n_species']
        irreps = self.hidden_irreps
        self.pol_quad = bool(p['quadrupoles'])
        if self.pol_quad and not any(ir == e3nn.Irrep('2e') for _, ir in irreps):
            raise ValueError(f"polar.quadrupoles needs 2e node features (hidden irreps {irreps})")
        if not any(ir == e3nn.Irrep('1o') for _, ir in irreps):
            raise ValueError(f"RACE_POLAR needs 1o node features for the dipoles (hidden irreps {irreps})")
        self.pol_multipole = e3nn_nnx_Linear(irreps_in=irreps, irreps_out=e3nn.Irreps("1x0e + 1x1o"),
                                             force_irreps_out=True, rngs=rngs)
        if self.pol_quad:
            self.pol_quadrupole = _zero_linear(e3nn_nnx_Linear(
                irreps_in=irreps, irreps_out=e3nn.Irreps("1x2e"), force_irreps_out=True, rngs=rngs))
        self.pol_alpha = _zero_linear(e3nn_nnx_Linear(
            irreps_in=irreps, irreps_out=e3nn.Irreps("1x0e"), force_irreps_out=True, rngs=rngs))
        a_min, a_max, a0 = (float(p[k]) for k in ('min_polarizability', 'max_polarizability', 'polarizability_init'))
        if not a_min < a0 < a_max:
            raise ValueError("polar: need min_polarizability < polarizability_init < max_polarizability")
        self.pol_alpha_bias = nnx.Param(jnp.asarray(math.log(a0 - a_min) - math.log(a_max - a0), jnp.float32))
        caps = [a_max] * n_species
        per_el = p.get('max_polarizability_per_element') or {}
        if per_el:
            from ase.data import atomic_numbers as _Z
            if self.species_z is None:
                raise ValueError("polar.max_polarizability_per_element needs species_z")
            for el, v in per_el.items():
                z = _Z[el] if isinstance(el, str) and not el.isdigit() else int(el)
                if z in self.species_z:
                    caps[self.species_z.index(z)] = float(v)
        self.pol_caps = tuple(caps)
        if self.species_z is not None:
            from ase.data import atomic_masses
            self.pol_masses = tuple(float(atomic_masses[z]) for z in self.species_z)
        else:
            self.pol_masses = None
        h = int(p['fukui_hidden'])
        self.pol_fukui0 = e3nn_nnx_Linear(irreps_in=irreps, irreps_out=e3nn.Irreps("16x0e"), force_irreps_out=True,
                                          rngs=rngs)
        self.pol_fukui1 = nnx.Linear(16, h, rngs=rngs)
        self.pol_fukui2 = nnx.Linear(h, 1, use_bias=False, rngs=rngs)
        self.pol_fukui_logit = nnx.Param(jnp.zeros((), jnp.float32))      # log inverse temperature (softplus), T = 1
        # reciprocal-space budget (static shapes): explicit k_grid / n_k, or from the largest cell of the data
        es = thole.ewald_settings(p, self.cutoff)
        self.pol_k_grid, self.pol_n_k = None, 0
        if self.periodic:
            if p.get('k_grid') and p.get('n_k'):
                self.pol_k_grid, self.pol_n_k = tuple(int(x) for x in p['k_grid']), int(p['n_k'])
            elif p.get('max_cell_length') and p.get('max_cell_volume'):
                self.pol_k_grid, self.pol_n_k = thole.k_budget(es['k_max'], p['max_cell_length'],
                                                               p['max_cell_volume'])
            else:
                raise ValueError("RACE_POLAR with periodic data needs polar.max_cell_length and "
                                 "polar.max_cell_volume (largest cell of the data), or polar.k_grid and polar.n_k")
        # open-boundary pair sums: "auto" = whenever a batch can hold open graphs (a non-periodic model, or data with
        # the per-graph flag globals['pbc']); true / false force it
        ob = p['open_boundary']
        self.pol_open = None if ob == 'auto' else bool(ob)
        if p['dispersion']:
            if not self.lr.get('qeq_block'):
                raise ValueError("polar.dispersion needs long_range.qeq_block (largest structure, atoms)")
            k0, a10 = float(p['disp_kappa_init']), float(p['disp_a1'])
            if k0 <= 0.0 or a10 <= 0.0 or float(p['disp_cutoff']) <= 2.0:
                raise ValueError("polar.dispersion: disp_kappa_init > 0, disp_a1 > 0, disp_cutoff > 2 A")
            self.disp_kappa = nnx.Param(jnp.full((n_species,), math.log(math.expm1(k0)), jnp.float32))
            self.disp_a1 = nnx.Param(jnp.asarray(math.log(math.expm1(a10)), jnp.float32))
        self.polar = p

    # ---------------------------------------------------------------------------------------------- dispersion
    def _dispersion(self, positions, cells, alpha, species, data, node_graph, node_mask, n_graphs, pbc, dt):
        """(N,) per-atom C6 dispersion energies (polar.dispersion; half of every pair on each atom)."""
        from race_v2.models.long_range import sr_switch
        p = self.polar
        B = int(self.lr['qeq_block'])
        N = positions.shape[0]
        n_node = jnp.asarray(data.n_node)
        local = jnp.arange(N) - (jnp.cumsum(n_node) - n_node)[node_graph]
        valid = node_mask & (local < B)
        slot = jnp.where(valid, local, B)

        def blocks(x, fill):
            out = jnp.full((n_graphs, B + 1) + x.shape[1:], fill, x.dtype).at[node_graph, slot].set(x)
            return out[:, :B]
        a = jnp.where(node_mask, alpha, 1.0)
        c6 = jax.nn.softplus(self.disp_kappa[...].astype(dt))[species] * a * a              # eV A^6
        rv = 0.529177 * 2.54 * (a / 0.148185) ** (1.0 / 7.0)                                 # A
        # padding slots get C = alpha = 1 (never 0): a 0 / (0 + tiny) there gave NaN gradients in float32 (the tiny
        # squared underflows; NaN * 0 survives the mask) -- pre-flight 2026-11-15
        c6 = jnp.where(valid, c6, 1.0)
        P, A, C, Rv, M = blocks(positions, 0.0), blocks(a, 1.0), blocks(c6, 1.0), blocks(rv, 1.0), blocks(valid, False)
        cij = 2.0 * C[:, :, None] * C[:, None, :] / (
            A[:, None, :] / A[:, :, None] * C[:, :, None] + A[:, :, None] / A[:, None, :] * C[:, None, :])
        r0 = jax.nn.softplus(self.disp_a1[...].astype(dt)) * (Rv[:, :, None] + Rv[:, None, :]) + float(p['disp_a2'])
        r06 = r0 ** 6
        mm = M[:, :, None] & M[:, None, :]
        off = mm & ~jnp.eye(B, dtype=bool)[None]

        def pair_e(d, keep):                                    # d (G, B, B, 3): -C6 / (r^6 + R0^6) on the kept pairs
            r2 = jnp.sum(d * d, axis=-1)
            return jnp.where(keep, -cij / (r2 ** 3 + r06), 0.0), r2
        e_open, _ = pair_e(P[:, :, None, :] - P[:, None, :, :], off)
        e_open = 0.5 * jnp.sum(e_open, axis=2)                                               # (G, B)
        e_blk = e_open
        if cells is not None:                                   # periodic graphs: images, smooth cutoff, tail
            rc = float(p['disp_cutoff'])
            n = int(p['disp_n_img'])
            e_per = jnp.zeros_like(e_open)
            for i in range(-n, n + 1):
                for j in range(-n, n + 1):
                    for k in range(-n, n + 1):
                        shift = jnp.einsum('c,gcd->gd', jnp.asarray([i, j, k], dt), cells)            # (G, 3)
                        d = P[:, :, None, :] - P[:, None, :, :] - shift[:, None, None, :]
                        keep = off if (i, j, k) == (0, 0, 0) else mm
                        e, r2 = pair_e(d, keep)
                        r = jnp.sqrt(jnp.maximum(r2, 1e-12))
                        e_per = e_per + 0.5 * jnp.sum(e * (1.0 - sr_switch(r, rc - 2.0, rc)), axis=2)
            vol = jnp.abs(jnp.linalg.det(cells))                                                     # (G,)
            tail = -2.0 * math.pi / (3.0 * (rc - 1.0) ** 3) * jnp.sum(jnp.where(mm, cij, 0.0), axis=2) / vol[:, None]
            e_per = e_per + tail                  # per atom i: -2 pi / (3 V rc^3) sum_j C6_ij (half of every pair)
            e_blk = jnp.where(pbc[:, None], e_per, e_open)
            # images |n| <= disp_n_img reach every pair inside rc only when n_img x (smallest cell height) >= rc
            # (wrapped positions); otherwise pairs inside the cutoff are missing: NaN, as a Qeq overflow (2026-11-15)
            vol_h = jnp.abs(jnp.linalg.det(cells))
            areas = jnp.stack([jnp.linalg.norm(jnp.cross(cells[:, 1], cells[:, 2]), axis=-1),
                               jnp.linalg.norm(jnp.cross(cells[:, 2], cells[:, 0]), axis=-1),
                               jnp.linalg.norm(jnp.cross(cells[:, 0], cells[:, 1]), axis=-1)], axis=-1)
            h_min = jnp.min(vol_h[:, None] / jnp.maximum(areas, 1e-12), axis=-1)
            real = jax.ops.segment_sum(node_mask.astype(dt), node_graph, num_segments=n_graphs) > 0   # not padding
            short = pbc & (n * h_min < rc) & real
            e_blk = jnp.where(short[:, None], jnp.nan, e_blk)
        e = e_blk[node_graph, jnp.minimum(slot, B - 1)]
        e = jnp.where(valid, e, 0.0)
        # atoms beyond qeq_block get no dispersion: a NaN energy instead of a silent omission (as the Qeq solve)
        return jnp.where(jnp.any(node_mask & (local >= B)), jnp.nan, e)

    # ---------------------------------------------------------------------------------------------- heads
    def _polar_features(self, feats: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        if self.polar['feature_norm'] == 'none':
            return feats
        scale = float(self.polar['feature_norm_scale'])
        arr = feats.array
        types = {}
        for (mul, ir), sl in zip(feats.irreps, feats.irreps.slices()):
            types.setdefault(str(ir), []).append(sl)
        out = arr
        for sls in types.values():
            count = sum(s.stop - s.start for s in sls)
            ms = sum(jnp.sum((arr[:, s] / scale) ** 2, axis=-1, keepdims=True) for s in sls) / count
            gain = jax.lax.rsqrt(1.0 + ms)
            for s in sls:
                out = out.at[:, s].set(arr[:, s] * gain)
        return e3nn.IrrepsArray(feats.irreps, out)

    def polar_heads(self, feats, species):
        """(raw charge, Fukui logits, mu, theta or None, alpha) of every node."""
        p = self.polar
        f = self._polar_features(_dense(feats))
        dt = f.array.dtype
        s = float(p['multipole_init_scale'])
        mp = _dense(self.pol_multipole(f)).array * s
        q_raw, mu = mp[:, 0], mp[:, 1:4]
        theta = None
        if self.pol_quad:
            # orthonormal basis of the traceless symmetric matrices (the 2e map has Frobenius norm sqrt 2)
            theta = _traceless_symmetric_from_2e(_dense(self.pol_quadrupole(f)).array * s) / math.sqrt(2.0)
        raw = _dense(self.pol_alpha(f)).array[:, 0] + self.pol_alpha_bias[...].astype(dt)
        a_min = float(p['min_polarizability'])
        upper = jnp.asarray(self.pol_caps, dt)[species]
        alpha = a_min + (upper - a_min) * jax.nn.sigmoid(raw)
        hid = jax.nn.silu(self.pol_fukui1(_dense(self.pol_fukui0(f)).array))
        inv_t = jax.nn.softplus(self.pol_fukui_logit[...] + math.log(math.expm1(1.0)))
        logits = (self.pol_fukui2(hid)[:, 0] * inv_t).astype(dt)
        return q_raw, logits, mu, theta, alpha

    def polar_charges(self, q_raw, logits, positions, cells, data, node_graph, node_mask, edge_mask, n_graphs, Q):
        """Charges with the constraint of the config: "graph" = the total charge of every graph; "electrode" = the
        target charge of every charge group (data.nodes['charge_group'] / ['fixed_charge'], reservoir group 0 takes
        the rest of the graph's charge), relaxed between groups in contact with charge_transfer "contact"."""
        p = self.polar
        nodes = data.nodes if isinstance(data.nodes, dict) else {}
        if p['charge_constraint'] != 'electrode' or 'charge_group' not in nodes:
            return thole.equilibrate(q_raw, logits, node_graph, n_graphs, Q, node_mask)
        F = int(p['max_groups'])
        dt = q_raw.dtype
        group, targets = thole.constraint_targets(nodes['charge_group'], jnp.asarray(nodes['fixed_charge'], dt),
                                                  node_graph, node_mask, Q, F)
        if p['charge_transfer'] != 'contact':
            return thole.equilibrate(q_raw, logits, group, n_graphs * F, targets.reshape(-1), node_mask)
        d = positions[data.receivers] - positions[data.senders]
        if cells is not None and isinstance(data.edges, dict) and 'Sij' in data.edges:
            d = d + jnp.einsum('ei,eij->ej', data.edges['Sij'].astype(dt), cells[node_graph[data.receivers]])
        dist = jnp.sqrt(jnp.where(edge_mask, jnp.sum(d * d, axis=-1), 1.0))
        return thole.contact_charges(q_raw, logits, group, targets, node_mask, data.receivers, data.senders, dist,
                                     edge_mask, contact_radius=float(p['contact_radius']),
                                     cutoff=float(p['transfer_cutoff']), pairs=p['contact_pairs'],
                                     eps=float(p['contact_eps']))

    # ---------------------------------------------------------------------------------------------- energy
    def node_energies(self, positions, cell, cutoff, data, *args, **kw):
        out = super().node_energies(positions, cell, cutoff, data, *args, **kw)
        species = data.nodes['species']
        n_nodes = species.shape[0]
        n_graphs = data.n_node.shape[0]
        padded = n_graphs > 1
        node_mask = jraph.get_node_padding_mask(data) if padded else jnp.ones((n_nodes,), bool)
        graph_mask = jraph.get_graph_padding_mask(data) if padded else jnp.ones((n_graphs,), bool)
        n_edges = data.senders.shape[0]
        edge_mask = jraph.get_edge_padding_mask(data) if padded else jnp.ones((n_edges,), bool)
        node_graph = jnp.repeat(jnp.arange(n_graphs), data.n_node, total_repeat_length=n_nodes)
        dt = out['node_energies'].dtype
        Q, _ = self._graph_globals(data, dt)
        q_raw, logits, mu, theta, alpha = self.polar_heads(out['node_features'], species)
        glb = data.globals if isinstance(data.globals, dict) else {}
        nodes = data.nodes if isinstance(data.nodes, dict) else {}
        periodic = self.periodic and cell is not None
        if 'pbc' in glb:
            pbc = (jnp.asarray(glb['pbc']).reshape(n_graphs) > 0.5) & graph_mask
        else:
            pbc = graph_mask if periodic else jnp.zeros((n_graphs,), bool)
        if not periodic:
            pbc = jnp.zeros((n_graphs,), bool)
        cells = cell.reshape(n_graphs, 3, 3).astype(dt) if periodic else None
        sij = data.edges['Sij'] if (periodic and isinstance(data.edges, dict) and 'Sij' in data.edges) else None
        pos = positions.astype(dt)
        if self.polar['charges'] == 'qeq':
            q = out['charges'].astype(dt)          # RACE_LR Qeq charges (exact total / fragment charges)
        else:
            q = self.polar_charges(q_raw.astype(dt), logits, pos, cells, data, node_graph, node_mask, edge_mask,
                                   n_graphs, Q)
        # slab dipole correction: graphs flagged globals['slab']; charges enter the dipole at their unwrapped positions
        slab_on, pos_field = None, None
        if self.polar['slab_correction'] and periodic and 'slab' in glb:
            slab_on = (jnp.asarray(glb['slab']).reshape(n_graphs) > 0.5) & pbc
            if 'unwrap' in nodes:
                pos_field = pos + jnp.einsum('ni,nij->nj', jnp.asarray(nodes['unwrap'], dt), cells[node_graph])
        res = thole.thole_energy(
            positions.astype(dt), cells, node_graph, node_mask, pbc, data.senders, data.receivers, sij, edge_mask,
            q.astype(dt), mu.astype(dt), None if theta is None else theta.astype(dt), alpha.astype(dt),
            cutoff=self.cutoff, polar=self.polar, grid_m=_grid(self.pol_k_grid) if periodic else None,
            n_k=self.pol_n_k if periodic else 0,
            open_dense=((not periodic) or 'pbc' in glb) if self.pol_open is None else self.pol_open,
            slab_on=slab_on, pos_field=pos_field)
        # graph dipole: open-boundary graphs about their centre of mass (p - Q R_com), periodic graphs as stored
        nm = node_mask.astype(dt)
        w = nm if self.pol_masses is None else jnp.asarray(self.pol_masses, dt)[species] * nm
        seg = functools.partial(jax.ops.segment_sum, segment_ids=node_graph, num_segments=n_graphs)
        com = seg(w[:, None] * positions.astype(dt)) / jnp.maximum(seg(w), 1e-12)[:, None]
        net = seg(q * nm)
        dipole = res['total_dipole'] - jnp.where(pbc[:, None], 0.0, net[:, None] * com)
        out = dict(out)
        out['node_energies'] = out['node_energies'] + res['node_energy'].reshape(out['node_energies'].shape)
        if self.polar['dispersion']:
            e_d = self._dispersion(positions.astype(dt), cells, alpha.astype(dt), species, data, node_graph, node_mask,
                                   n_graphs, pbc, dt)
            out['node_energies'] = out['node_energies'] + e_d.reshape(out['node_energies'].shape)
            out['dispersion_node_energies'] = e_d
        out['charges'] = q * nm
        out['dipole'] = dipole
        # traceless graph quadrupole about the centre of mass (2026-11-13), the convention of the OMol25 labels:
        # Cartesian second moment sum_i [q_i r r + mu_i r + r mu_i] + (2/3) Theta_i (Theta: Buckingham, traceless)
        # with mu the permanent + induced dipoles, minus its trace / 3; components xx yy zz xy xz yz, e A^2.
        # Open-boundary graphs only (zero for periodic graphs).
        rc = (positions.astype(dt) - com[node_graph]) * nm[:, None]
        mu_all = res['dipoles'].astype(dt) * nm[:, None]
        M = (q * nm)[:, None, None] * rc[:, :, None] * rc[:, None, :] + mu_all[:, :, None] * rc[:, None, :] \
            + rc[:, :, None] * mu_all[:, None, :]
        if theta is not None:
            M = M + (2.0 / 3.0) * theta.astype(dt) * nm[:, None, None]
        Mg = seg(M)
        Mg = Mg - jnp.trace(Mg, axis1=1, axis2=2)[:, None, None] / 3.0 * jnp.eye(3, dtype=dt)
        quad = jnp.stack([Mg[:, 0, 0], Mg[:, 1, 1], Mg[:, 2, 2], Mg[:, 0, 1], Mg[:, 0, 2], Mg[:, 1, 2]], axis=-1)
        out['quadrupole'] = jnp.where(pbc[:, None], 0.0, quad)
        # along the slab normal (periodic graphs: the label of a dipole-corrected slab calculation)
        normals = res['normal_dipole_axis']
        out['dipole_normal'] = jnp.sum(dipole * normals, axis=-1)
        out['slab_normal'] = normals
        out['polar'] = {k: res[k] for k in ('node_permanent', 'node_polarization', 'node_slab', 'induced_dipoles',
                                            'dipoles')}
        out['polar']['polarizabilities'] = alpha * nm
        return out
