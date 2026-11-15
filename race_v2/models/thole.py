"""Thole-type self-consistent induced-dipole electrostatics in JAX (2026-11-11).

Port of the ``mode="pbc_split"`` path of BAM-torch ``bam_ufo/model/thole.py`` (TholeElectrostaticsBlock), phase 1:
permanent Gaussian charges, dipoles and (optional) quadrupoles of width sigma, isotropic atomic polarizabilities
alpha, and induced dipoles from

    (alpha^-1 + shift - T) mu_ind = E_perm          (conjugate gradient, implicit differentiation)

The energy is the Hartree energy of the total Gaussian density, with the induced self energy replaced by the
polarization cost |mu_i|^2 / (2 alpha_i):

    E = k_e/2 sum_i [ q_i phi_i - mu_i . E_i + Theta_i : grad grad phi_i / 3 ]
        + k_e sum_i [ |mu_ind,i|^2 / (2 alpha_i) - mu_ind,i . E_perm,i - 1/2 mu_ind,i . (T mu_ind)_i ]

Widths (a single Gaussian has width s; a pair has a = sqrt(2 (s1^2 + s2^2))): permanent-permanent a_0 = 2 sigma;
induced Gaussian s_i = thole_width alpha_i^(1/3) / 2 (optionally capped smoothly at eta / 2); induced-permanent
a_i = sqrt(2 (s_i^2 + sigma^2)); induced-induced a_ij = sqrt(2 (s_i^2 + s_j^2)).

Every kernel erf(r/a)/r is split at one Ewald width eta = max(2 sigma, cutoff / sqrt(ln 1/tol)) (TTM-3F rule):
  long range   the eta-Gaussian density.  Periodic graphs: one reciprocal-space sum per graph (tin-foil);
               open-boundary graphs: the direct sum of the eta kernel over all pairs of the graph.
  short range  (model kernel - eta kernel) over the model's neighbour list (cutoff): exact for the permanent
               multipoles (zero when eta = 2 sigma), weighted by the polynomial envelope for the induction field
               and the induced-induced tensor.

Differences from the torch code (static shapes for jit; the energy functional is the same, checked numerically in
tests/test_thole.py against values of the torch block):
  * reciprocal vectors: a fixed integer half grid (|m_k| <= k_grid[k]); every periodic graph keeps its n_k
    vectors of smallest |k| inside the sphere |k| <= k_max, padded with zero weight.  A graph with more than n_k
    vectors inside the sphere gets NaN energies (raise polar.n_k).
  * open-boundary graphs: dense N x N pair arrays of the padded batch (pairs of the same open graph only).
  * one CG for the whole batch (global residual), not a grouped CG per graph.
Phase 2 (2026-11-11): the slab dipole correction (thole_energy slab_on / pos_field), electrode / charge-group
constraints (constraint_targets) and the contact charge relaxation (contact_charges), checked against the torch
functions in tests/test_thole_phase2.py.  Not ported: the reactive charge solve, split-charge transfer,
charge_constraint "component", external fields.

Units: Angstrom, e, e Angstrom, Angstrom^3, eV.  Edge convention of this package: receiver i, sender j, model edge
vector x_j - (x_i + Sij . cell); here d = x_i - x_j + Sij . cell points from the image of j to i.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

K_COULOMB = 14.399645351950543
SQRT_PI = math.sqrt(math.pi)
EWALD_TOLERANCE = 1e-8

POLAR_DEFAULTS = {
    "sigma": 1.0, "thole_width": 2.0, "min_polarizability": 0.001, "max_polarizability": 4.0,
    "polarizability_init": 0.7, "max_polarizability_per_element": None, "multipole_init_scale": 0.1,
    "feature_norm": "smooth", "feature_norm_scale": 1.0, "fukui_hidden": 64,
    "induction": True, "mutual_induction": True, "quadrupoles": True, "cap_induced_width": True,
    "cg_max_iterations": 25, "cg_tolerance": 1e-6, "diagonal_shift": 1e-6, "envelope_p": 6,
    "ewald_k_tolerance": None, "k_grid": None, "n_k": None, "max_cell_length": None, "max_cell_volume": None,
    "open_boundary": "auto",
    # phase 2 (2026-11-11): charge-group constraints, contact charge relaxation, slab dipole correction
    "charge_constraint": "graph", "max_groups": 128, "charge_transfer": "none", "contact_radius": 2.5,
    "transfer_cutoff": 4.0, "contact_pairs": "charged", "contact_eps": 1e-6,
    "slab_correction": False, "slab_axis": 2,
    # 2026-11-11: where the charges of the Thole block come from.  "fukui" (default): raw charge + Fukui partition of
    # the missing total charge (the torch block).  "qeq": the charge-equilibration charges of RACE_LR
    # (long_range chi / hardness heads; the pair Coulomb energy is then counted here only, see models/race_polar.py).
    "charges": "fukui",
    # 2026-11-15: pairwise C6 dispersion from the Thole polarizabilities (models.race_polar.RACE_POLAR._dispersion)
    "dispersion": False, "disp_kappa_init": 7.5, "disp_a1": 0.8, "disp_a2": 0.0,
    "disp_cutoff": 12.0, "disp_n_img": 1,
}


# ------------------------------------------------------------------------------------------------- settings
def ewald_settings(polar: dict, cutoff: float) -> dict:
    """Derived constants of the split: eta, k_max, the induced-width cap and the reciprocal self-term fraction."""
    sigma = float(polar["sigma"])
    decay = math.sqrt(math.log(1.0 / EWALD_TOLERANCE))
    eta = max(2.0 * sigma, float(cutoff) / decay)
    tol = polar.get("ewald_k_tolerance")
    if tol is not None and not 0.0 < float(tol) < 1.0:
        raise ValueError("polar.ewald_k_tolerance must be in (0, 1)")
    k_decay = decay if tol is None else math.sqrt(math.log(1.0 / float(tol)))
    k_max = 2.0 * k_decay / eta
    y = 0.5 * eta * k_max
    represented = math.erf(y) - 2.0 * y * math.exp(-y * y) / SQRT_PI
    return {"sigma": sigma, "width": 2.0 * sigma, "eta": eta, "k_max": k_max,
            "correction": 4.0 * represented / (3.0 * eta ** 3 * SQRT_PI),
            "cap": 0.5 * eta if polar.get("cap_induced_width") else None}


def k_budget(k_max: float, max_cell_length, max_cell_volume, margin: float = 1.15):
    """(k_grid, n_k) that cover every cell with lengths <= max_cell_length (a number or three) and volume <=
    max_cell_volume: |m_k| <= k_max |a_k| / 2 pi, and half the sphere holds V k_max^3 / (12 pi^2) vectors."""
    L = np.broadcast_to(np.asarray(max_cell_length, float), (3,))
    grid = tuple(int(math.ceil(k_max * float(x) / (2.0 * math.pi))) for x in L)
    n_k = int(math.ceil(margin * float(max_cell_volume) * k_max ** 3 / (12.0 * math.pi ** 2))) + 16
    return grid, n_k


def half_grid(k_grid) -> np.ndarray:
    """Integer vectors m of the half space (m0 > 0, or m0 = 0 and m1 > 0, or m0 = m1 = 0 and m2 > 0)."""
    r = [np.arange(-int(n), int(n) + 1) for n in k_grid]
    m = np.stack(np.meshgrid(*r, indexing="ij"), -1).reshape(-1, 3)
    half = (m[:, 0] > 0) | ((m[:, 0] == 0) & (m[:, 1] > 0)) | ((m[:, 0] == 0) & (m[:, 1] == 0) & (m[:, 2] > 0))
    return m[half]


# ------------------------------------------------------------------------------------------------- kernels
_TERMS = 28


def _series(m: int):
    c = [(-1.0) ** n / (math.factorial(n) * (2 * n + 1)) for n in range(_TERMS)]
    return [math.prod(range(n - m + 1, n + 1)) * c[n] for n in range(m, _TERMS)]


def gaussian_radial(r2, a, order: int = 2):
    """[T, D T, ..., D^order T] of T = erf(r/a)/r with D = (1/r) d/dr, as functions of r^2 (finite at r = 0):
    T' / r = D T ("iso"), (T'' - T'/r) / r^2 = D^2 T ("outer").  Direct recursion
    D^(m+1) T = [(-2/a^2)^m p e^(-r^2/a^2) - (2m+1) D^m T] / r^2, p = 2 / (a sqrt pi), for r^2 >= 2 a^2 and the
    power series of erf(x)/x below (float32 relative error < 1e-6 for orders <= 4)."""
    a = jnp.asarray(a, r2.dtype)
    a2 = a * a
    x2 = r2 / a2
    small = x2 < 2.0
    y = jnp.where(small, x2, 0.0)
    rr = jnp.where(small, 2.0 * a2, r2)            # both branches stay finite for autodiff
    r = jnp.sqrt(rr)
    p = 2.0 / (a * SQRT_PI)
    pe = p * jnp.exp(-rr / a2)
    c = jax.scipy.special.erf(r / a) / r
    out = []
    for m in range(order + 1):
        if m:
            c = ((-2.0 / a2) ** (m - 1) * pe - (2 * m - 1) * c) / rr
        s = jnp.zeros_like(y)
        for v in reversed(_series(m)):             # Horner
            s = s * y + v
        out.append(jnp.where(small, p * (2.0 / a2) ** m * s, c))
    return out


def envelope(r, r_max: float, p: int):
    x = r / r_max
    e = (1.0 - ((p + 1.0) * (p + 2.0) / 2.0) * x ** p + p * (p + 2.0) * x ** (p + 1)
         - (p * (p + 1.0) / 2.0) * x ** (p + 2))
    return jnp.where(x < 1.0, e, 0.0)


def induced_widths(alpha, thole_width: float, cap=None):
    """Gaussian width of the induced dipole density, thole_width alpha^(1/3) / 2, with the smooth cap
    s (1 + (s/c)^16)^(-1/16) (never above c)."""
    s = 0.5 * thole_width * jnp.cbrt(alpha)
    if cap is None:
        return s
    return s * (1.0 + (s / cap) ** 16) ** (-1.0 / 16.0)


def quadrupole_self_gradient(q, theta, a):
    """grad grad phi of an atom's own Gaussian charge and quadrupole at its centre (pair width a)."""
    eye = jnp.eye(3, dtype=theta.dtype)
    return (q * (-4.0 / (3.0 * a ** 3 * SQRT_PI)))[:, None, None] * eye + (
        2.0 / 3.0 * 8.0 / (5.0 * a ** 5 * SQRT_PI)) * theta


# ------------------------------------------------------------------------------------------------- pair sums
class _Edges:
    """Pair arrays over an edge list: shape (E, ...), receiver i = recv, sender j = send."""

    def __init__(self, recv, send, n):
        self.recv, self.send, self.n = recv, send, n

    def gj(self, x):
        return x[self.send]

    def gi(self, x):
        return x[self.recv]

    def si(self, x):
        return jax.ops.segment_sum(x, self.recv, num_segments=self.n)


class _Dense:
    """Pair arrays over all pairs: shape (N, N, ...), [i, j] = receiver i, sender j."""

    def gj(self, x):
        return x[None]

    def gi(self, x):
        return x[:, None]

    def si(self, x):
        return jnp.sum(x, axis=1)


def _bc(c, like):
    return c.reshape(c.shape + (1,) * (like.ndim - c.ndim))


def multipole_sums(P, d, q, mu, theta, t0, c1, c2, c3=None, c4=None):
    """Potential, field and (with c4) field gradient at each receiver of the senders' Gaussian multipoles
    rho_j = q g - mu . grad g + (1/3) Theta : grad grad g, for the radial factors D^m T per pair (t0 None: no
    potential; theta None: charges and dipoles only).  d points from sender to receiver.

        phi += q t0 - c1 mu.d + (1/3) c2 d.Theta.d
        E   += (-q c1 + c2 mu.d) d + c1 mu - (1/3) [2 c2 Theta d + c3 (d.Theta.d) d]
        grad grad phi += (q c1 - c2 mu.d + c3 dQd/3) I + (q c2 - c3 mu.d + c4 dQd/3) d d
                         - c2 (mu d + d mu) + (2/3) [c2 Theta + c3 (Theta d d + d d Theta)]
    """
    q_j, mu_j = P.gj(q), P.gj(mu)
    md = jnp.sum(mu_j * d, axis=-1)
    field = (-q_j * c1 + c2 * md)[..., None] * d + c1[..., None] * mu_j
    v = None if t0 is None else q_j * t0 - c1 * md
    grad = None
    if theta is not None:
        th_j = P.gj(theta)
        qd = jnp.einsum("...ab,...b->...a", th_j, d)
        dqd = jnp.sum(qd * d, axis=-1)
        if v is not None:
            v = v + c2 * dqd / 3.0
        field = field - (2.0 * c2[..., None] * qd + (c3 * dqd)[..., None] * d) / 3.0
        if c4 is not None:
            eye = jnp.eye(3, dtype=d.dtype)
            dd = d[..., :, None] * d[..., None, :]
            mud = mu_j[..., :, None] * d[..., None, :]
            qdd = qd[..., :, None] * d[..., None, :]
            grad = ((q_j * c1 - c2 * md + c3 * dqd / 3.0)[..., None, None] * eye
                    + (q_j * c2 - c3 * md + c4 * dqd / 3.0)[..., None, None] * dd
                    - c2[..., None, None] * (mud + jnp.swapaxes(mud, -1, -2))
                    + (2.0 / 3.0) * (c2[..., None, None] * th_j
                                     + c3[..., None, None] * (qdd + jnp.swapaxes(qdd, -1, -2))))
            grad = P.si(grad)
    return (None if v is None else P.si(v)), P.si(field), grad


def tensor_sum(P, d, x, iso, outer):
    """(T x)_i = sum_j [iso x_j + outer (d . x_j) d] over the pairs."""
    xj = P.gj(x)
    return P.si(iso[..., None] * xj + (outer * jnp.sum(d * xj, axis=-1))[..., None] * d)


# ------------------------------------------------------------------------------------------------- reciprocal
def reciprocal_vectors(cell, pbc, k_max: float, grid_m: np.ndarray, n_k: int):
    """k-vectors of every graph: (k (G, K, 3), k2 (G, K), mask (G, K), overflow (G,)).  grid_m: half_grid;
    the K = min(n_k, len(grid_m)) vectors of smallest |k| with |k| <= k_max are kept (mask 1), the rest have
    k = 0, k2 = 1, mask 0.  Non-periodic graphs: mask 0 everywhere."""
    dt = cell.dtype
    eye = jnp.eye(3, dtype=dt)
    safe = jnp.where(pbc[:, None, None], cell, eye)
    bmat = 2.0 * math.pi * jnp.swapaxes(jnp.linalg.inv(safe), -1, -2)
    m = jnp.asarray(grid_m, dt)
    k = jnp.einsum("mi,gij->gmj", m, bmat)                           # (G, M, 3)
    k2 = jnp.sum(k * k, axis=-1)
    keep = (k2 <= k_max * k_max) & pbc[:, None]
    K = int(min(int(n_k), grid_m.shape[0]))
    score = jax.lax.stop_gradient(jnp.where(keep, -k2, -jnp.inf))
    _, idx = jax.lax.top_k(score, K)
    mask = jnp.take_along_axis(keep, idx, axis=1)
    k = jnp.where(mask[..., None], jnp.take_along_axis(k, idx[..., None], axis=1), 0.0)
    k2 = jnp.where(mask, jnp.take_along_axis(k2, idx, axis=1), 1.0)
    overflow = jnp.sum(keep, axis=1) > K
    return k, k2, mask.astype(dt), overflow


# ------------------------------------------------------------------------------------------------- main
def thole_energy(positions, cell, node_graph, node_mask, pbc, senders, receivers, sij, edge_mask,
                 q, mu, theta, alpha, *, cutoff: float, polar: dict, grid_m=None, n_k: int = 0,
                 open_dense: bool = True, slab_on=None, pos_field=None):
    """Electrostatic node energies of a padded batch.

    positions (N, 3); cell (G, 3, 3) or None; node_graph (N,) int; node_mask (N,) bool; pbc (G,) bool (periodic
    graphs; needs cell, grid_m and n_k); senders / receivers (E,), sij (E, 3) or None (shifts in cell units),
    edge_mask (E,) bool; q (N,), mu (N, 3), theta (N, 3, 3) or None, alpha (N,).  Padding nodes must have
    q = mu = theta = 0 (they are masked here as well) and any positive alpha.
    open_dense: evaluate the open-boundary graphs (dense N x N pair arrays); False when every graph is periodic.
    slab_on (G,) bool: graphs with the slab dipole correction E_slab = 2 pi k_e (M . n)^2 / V (Yeh-Berkowitz /
    Bengtsson; n = unit normal of the plane of the two cell vectors other than polar.slab_axis, M the total dipole
    of charges and all dipoles): its uniform field acts on the induced dipoles self-consistently (rank-one term
    of the CG operator).  Neutral cells only.  pos_field (N, 3): the positions that enter M (fragments unwrapped;
    default positions).

    Returns a dict: node_energy (N,), node_permanent, node_polarization, node_slab, induced_dipoles (N, 3), dipoles
    (permanent + induced), total_dipole (G, 3) = sum_i q_i r_i + mu_i (pos_field), normal_dipole (G,) = its
    component along the slab normal (the Cartesian slab axis for open graphs), potential (N,), field (N, 3) (what
    drives the induced dipoles, in e/A^2), overflow (G,) bool.
    """
    es = ewald_settings(polar, cutoff)
    sigma, width, eta, k_max = es["sigma"], es["width"], es["eta"], es["k_max"]
    N = positions.shape[0]
    G = pbc.shape[0]
    dt = positions.dtype
    use_theta = theta is not None
    order = 4 if use_theta else 2
    nm = node_mask.astype(dt)
    q = q * nm
    mu = mu * nm[:, None]
    if use_theta:
        theta = theta * nm[:, None, None]
    alpha = jnp.where(node_mask, alpha, 1.0)
    pbc = pbc.astype(bool)
    pbc_node = pbc[node_graph] & node_mask
    open_node = (~pbc[node_graph]) & node_mask
    periodic = cell is not None and grid_m is not None and n_k > 0
    induction = bool(polar["induction"])
    mutual = bool(polar["mutual_induction"]) and induction

    def seg(x):
        return jax.ops.segment_sum(x, node_graph, num_segments=G)

    v_perm = jnp.zeros((N,), dt)
    field_perm = jnp.zeros((N, 3), dt)
    field_ind = jnp.zeros((N, 3), dt)
    grad_perm = jnp.zeros((N, 3, 3), dt) if use_theta else None
    overflow = jnp.zeros((G,), bool)
    t_parts = []                                  # callables x -> (T x) contributions

    # ---- long range, periodic graphs: reciprocal sums at width eta -------------------------------------
    if periodic:
        k, k2, kmask, overflow = reciprocal_vectors(cell, pbc, k_max, grid_m, n_k)
        vol = jnp.abs(jnp.linalg.det(jnp.where(pbc[:, None, None], cell, jnp.eye(3, dtype=dt))))
        pref = 8.0 * math.pi / vol                                   # (G,)
        kn = k[node_graph]                                           # (N, K, 3)
        ang = jnp.einsum("nd,nkd->nk", positions, kn)
        cos_t, sin_t = jnp.cos(ang), jnp.sin(ang)
        m_k = jnp.einsum("nd,nkd->nk", mu, kn)
        q_k = q[:, None]
        if use_theta:
            q_k = q_k - jnp.einsum("nab,nka,nkb->nk", theta, kn, kn) / 3.0
        s_re = seg(q_k * cos_t - m_k * sin_t)                        # (G, K)
        s_im = -seg(q_k * sin_t + m_k * cos_t)
        re_i = s_re[node_graph] * cos_t - s_im[node_graph] * sin_t
        im_i = s_re[node_graph] * sin_t + s_im[node_graph] * cos_t
        w = (kmask * jnp.exp(-0.25 * eta * eta * k2) / k2 * pref[:, None])[node_graph]      # (N, K), pref included
        v_perm = v_perm + jnp.sum(re_i * w, axis=1)
        f_rec = jnp.einsum("nk,nkd->nd", im_i * w, kn)
        field_perm = field_perm + f_rec
        field_ind = field_ind + f_rec + es["correction"] * mu * pbc_node.astype(dt)[:, None]
        if use_theta:
            grad_perm = grad_perm - jnp.einsum("nk,nka,nkb->nab", re_i * w, kn, kn)
        w_sqrt = (jnp.sqrt(pref)[:, None] * kmask * jnp.exp(-0.125 * eta * eta * k2) * jax.lax.rsqrt(k2))[node_graph]
        ph_c, ph_s = cos_t * w_sqrt, sin_t * w_sqrt
        corr_node = es["correction"] * pbc_node.astype(dt)

        def t_rec(x):
            kx = jnp.einsum("nd,nkd->nk", x, kn)
            sc, ss = seg(ph_c * kx), seg(ph_s * kx)
            return corr_node[:, None] * x - jnp.einsum(
                "nk,nkd->nd", ph_c * sc[node_graph] + ph_s * ss[node_graph], kn)
        t_parts.append(t_rec)
        if eta > width:
            # the tin-foil sum at width eta carries the background -pi Q eta^2 / V; the real-space difference
            # kernel below carries the k = 0 term of that shift, removed here (charged cells)
            net = seg(q)
            v_perm = v_perm - (pbc.astype(dt) * math.pi * (eta * eta - width * width) * net / vol)[node_graph]

    # ---- long range, open-boundary graphs: the eta kernel over all pairs of the graph -----------------
    if open_dense:
        D = _Dense()
        d_o = positions[:, None, :] - positions[None, :, :]
        pm = ((node_graph[:, None] == node_graph[None, :]) & open_node[:, None] & open_node[None, :]
              & ~jnp.eye(N, dtype=bool))
        r2_o = jnp.where(pm, jnp.sum(d_o * d_o, axis=-1), 1.0)
        co = [c * pm.astype(dt) for c in gaussian_radial(r2_o, eta, order)]
        v_o, f_o, g_o = multipole_sums(D, d_o, q, mu, theta, co[0], co[1], co[2],
                                       co[3] if use_theta else None, co[4] if use_theta else None)
        on = open_node.astype(dt)
        v_perm = v_perm + v_o + on * q * (2.0 / (eta * SQRT_PI))
        field_perm = field_perm + f_o + on[:, None] * mu * (-4.0 / (3.0 * eta ** 3 * SQRT_PI))
        field_ind = field_ind + f_o
        if use_theta:
            grad_perm = grad_perm + g_o + on[:, None, None] * quadrupole_self_gradient(q, theta, eta)
        t_parts.append(lambda x: tensor_sum(D, d_o, x, co[1], co[2]))

    # ---- short range: model kernels minus the eta kernel over the neighbour list ----------------------
    E = _Edges(receivers, senders, N)
    d = positions[receivers] - positions[senders]
    if sij is not None and cell is not None:
        d = d + jnp.einsum("ei,eij->ej", sij.astype(dt), cell[node_graph[receivers]])
    em = edge_mask.astype(dt)
    r2 = jnp.where(edge_mask, jnp.sum(d * d, axis=-1), 1.0)
    r = jnp.sqrt(r2)
    env = envelope(r, float(cutoff), int(polar["envelope_p"])) * em
    ce = gaussian_radial(r2, eta, order)
    if eta > width:
        cw = gaussian_radial(r2, width, order)
        df = [(a - b) * em for a, b in zip(cw, ce)]
        v_e, f_e, g_e = multipole_sums(E, d, q, mu, theta, df[0], df[1], df[2],
                                       df[3] if use_theta else None, df[4] if use_theta else None)
        v_perm = v_perm + v_e + q * (2.0 / SQRT_PI) * (1.0 / width - 1.0 / eta)
        field_perm = field_perm + f_e + mu * (-4.0 / (3.0 * SQRT_PI) * (1.0 / width ** 3 - 1.0 / eta ** 3))
        if use_theta:
            grad_perm = grad_perm + g_e + (quadrupole_self_gradient(q, theta, width)
                                           - quadrupole_self_gradient(q, theta, eta))
    s2 = induced_widths(alpha, float(polar["thole_width"]), es["cap"]) ** 2
    a_ind = jnp.sqrt(2.0 * (s2 + sigma * sigma))
    ci = gaussian_radial(r2, a_ind[receivers], 3 if use_theta else 2)
    _, f_e, _ = multipole_sums(E, d, q, mu, theta, None, env * (ci[1] - ce[1]), env * (ci[2] - ce[2]),
                               env * (ci[3] - ce[3]) if use_theta else None, None)
    field_ind = field_ind + f_e
    a_pair = jnp.sqrt(2.0 * (s2[receivers] + s2[senders]))
    cp = gaussian_radial(r2, a_pair, 2)
    iso_e, outer_e = env * (cp[1] - ce[1]), env * (cp[2] - ce[2])
    t_parts.append(lambda x: tensor_sum(E, d, x, iso_e, outer_e))

    def apply_t(x):
        out = jnp.zeros_like(x)
        for f in t_parts:
            out = out + f(x)
        return out

    # ---- slab dipole correction: uniform field -4 pi k_e M_n n / V -------------------------------------
    axis = int(polar.get("slab_axis", 2))
    unit = jnp.zeros((3,), dt).at[axis].set(1.0)
    normals = jnp.broadcast_to(unit, (G, 3))
    if cell is not None:
        cs = jnp.where(pbc[:, None, None], cell, jnp.eye(3, dtype=dt))
        nv = jnp.cross(cs[:, (axis + 1) % 3], cs[:, (axis + 2) % 3])
        nv = nv / jnp.linalg.norm(nv, axis=-1, keepdims=True)
        sgn = jnp.sign(jnp.sum(nv * cs[:, axis], axis=-1, keepdims=True))
        normals = jnp.where(pbc[:, None], nv * jnp.where(sgn == 0, 1.0, sgn), unit)
    node_normal = normals[node_graph]
    pf = positions if pos_field is None else pos_field
    slab = slab_on is not None and cell is not None
    if slab:
        on = slab_on.astype(bool) & pbc
        vol_s = jnp.abs(jnp.linalg.det(jnp.where(on[:, None, None], cell, jnp.eye(3, dtype=dt))))
        coeff = 4.0 * math.pi / vol_s * on.astype(dt)                 # (G,)
        m0_node = q * jnp.sum(pf * node_normal, axis=-1) + jnp.sum(mu * node_normal, axis=-1)

    # ---- induced dipoles -------------------------------------------------------------------------------
    rhs = field_ind * nm[:, None]
    rhs_solve = rhs
    if slab:
        rhs_solve = (rhs - (coeff * seg(m0_node))[node_graph][:, None] * node_normal) * nm[:, None]
    inv_alpha = 1.0 / alpha + float(polar["diagonal_shift"])
    if not induction:
        mu_ind = jnp.zeros_like(rhs)
    elif not mutual:
        mu_ind = rhs_solve / inv_alpha[:, None]
        if slab:        # diagonal + rank one per graph: Sherman-Morrison
            u = node_normal / inv_alpha[:, None] * nm[:, None]
            s_y, s_u = seg(jnp.sum(mu_ind * node_normal, axis=-1)), seg(jnp.sum(u * node_normal, axis=-1))
            mu_ind = mu_ind - (coeff * s_y / (1.0 + coeff * s_u))[node_graph][:, None] * u
    else:
        def matvec(x):
            out = inv_alpha[:, None] * x - apply_t(x)
            if slab:
                out = out + (coeff * seg(jnp.sum(x * node_normal, axis=-1)))[node_graph][:, None] * node_normal
            return out * nm[:, None]
        mu_ind, _ = jax.scipy.sparse.linalg.cg(
            matvec, rhs_solve, tol=float(polar["cg_tolerance"]), atol=0.0, maxiter=int(polar["cg_max_iterations"]),
            M=lambda x: x / inv_alpha[:, None])

    node_perm = 0.5 * K_COULOMB * (q * v_perm - jnp.sum(mu * field_perm, axis=-1))
    if use_theta:
        node_perm = node_perm + 0.5 * K_COULOMB * jnp.sum(theta * grad_perm, axis=(-1, -2)) / 3.0
    if induction:
        t_mu = apply_t(mu_ind) if mutual else jnp.zeros_like(mu_ind)
        node_pol = K_COULOMB * (0.5 * jnp.sum(mu_ind * mu_ind, axis=-1) * inv_alpha
                                - jnp.sum(mu_ind * rhs, axis=-1) - 0.5 * jnp.sum(mu_ind * t_mu, axis=-1))
    else:
        node_pol = jnp.zeros((N,), dt)
    node_slab = jnp.zeros((N,), dt)
    if slab:
        m_node = (m0_node + jnp.sum(mu_ind * node_normal, axis=-1)) * nm
        node_slab = 0.5 * K_COULOMB * (coeff * seg(m_node))[node_graph] * m_node
    node_e = (node_perm + node_pol + node_slab) * nm
    node_e = jnp.where(overflow[node_graph], jnp.nan, node_e)
    dip = q[:, None] * pf + mu + mu_ind
    total_dipole = seg(dip * nm[:, None])
    return {"node_energy": node_e, "node_permanent": node_perm * nm, "node_polarization": node_pol * nm,
            "node_slab": node_slab * nm, "normal_dipole": jnp.sum(total_dipole * normals, axis=-1),
            "normal_dipole_axis": normals,
            "induced_dipoles": mu_ind, "dipoles": mu + mu_ind, "total_dipole": total_dipole,
            "potential": K_COULOMB * v_perm, "field": rhs, "overflow": overflow}


# ------------------------------------------------------------------------------------------------- charges
def equilibrate(q, logits, group, n_groups: int, targets, node_mask):
    """Charges that sum to targets in every group: q + w_i (target - sum q), w = softmax of the Fukui logits over
    the atoms of the group (BAM-torch charge_offset "fukui", fukui_activation "softmax")."""
    q = jnp.where(node_mask, q, 0.0)
    lg = jnp.where(node_mask, logits, -jnp.inf)
    mx = jax.ops.segment_max(lg, group, num_segments=n_groups)
    mx = jax.lax.stop_gradient(jnp.where(jnp.isfinite(mx), mx, 0.0))
    un = jnp.where(node_mask, jnp.exp(jnp.where(node_mask, logits, 0.0) - mx[group]), 0.0)
    den = jax.ops.segment_sum(un, group, num_segments=n_groups)
    w = un / jnp.where(den > 0, den, 1.0)[group]
    sum_q = jax.ops.segment_sum(q, group, num_segments=n_groups)
    return q + w * (targets - sum_q)[group]


def constraint_targets(local_group, fixed_charge, node_graph, node_mask, total_charge, max_groups: int):
    """Constraint groups of charge_constraint "electrode" (BAM-torch electrostatics.charge_constraint_groups with
    explicit charge_group labels).  local_group (N,) int: 0 = the reservoir group of the graph (electrode atoms; it
    takes the residual of the graph's total charge), 1.. = the other groups (molecules, ions), whose target is the
    sum of their atoms' fixed_charge.  Returns (group (N,) = graph * max_groups + local, targets (G, max_groups))."""
    G = total_charge.shape[0]
    F = int(max_groups)
    lg = jnp.clip(jnp.asarray(local_group, jnp.int32), 0, F - 1)
    group = node_graph * F + lg
    fixed = jnp.where(node_mask & (lg > 0), fixed_charge, 0.0)
    t = jax.ops.segment_sum(fixed, group, num_segments=G * F).reshape(G, F)
    t = t.at[:, 0].set(total_charge - jnp.sum(t[:, 1:], axis=1))
    return group, t


def contact_charges(raw, logits, group, targets, node_mask, recv, send, dist, edge_mask, *, contact_radius: float,
                    cutoff: float, pairs: str = "charged", eps: float = 1e-6):
    """Local, smooth relaxation of the group charge constraints (BAM-torch utils/contact_charges.contact_charges,
    offset "fukui", activation "softmax").  Without contacts every group has its target charge
    (q = raw + w (target - sum raw), w the softmax of the Fukui logits within the group); groups in contact (edges
    between different groups of a graph, quintic switch flat through contact_radius and zero at cutoff, conductance
    s / (1 - s + eps)) exchange charge, and a strong contact makes them one group.  pairs "charged": only contacts
    with a group of non-zero target on at least one side.  For every graph one (F - 1) x (F - 1) solve with the
    constant-potential mode removed (Householder), F = targets.shape[1].

    group (N,) = graph * F + local (constraint_targets); targets (G, F).  In float32 use eps of about 1e-3 (the
    conductance of a full contact is 1 / eps, and the solve loses that many digits)."""
    if pairs not in ("charged", "all"):
        raise ValueError(f"contact_pairs must be 'charged' or 'all', got {pairs!r}")
    if not 0.0 < contact_radius < cutoff:
        raise ValueError("need 0 < contact_radius < transfer_cutoff")
    G, F = targets.shape
    dt = raw.dtype
    n_seg = G * F
    x = jnp.where(node_mask, raw, 0.0)
    lg = jnp.where(node_mask, logits, -jnp.inf)
    mx = jax.ops.segment_max(lg, group, num_segments=n_seg)
    exists = jnp.isfinite(mx)
    mx = jax.lax.stop_gradient(jnp.where(exists, mx, 0.0))
    u = jnp.where(node_mask, jnp.exp(jnp.where(node_mask, logits, 0.0) - mx[group]), 0.0)
    z = jax.ops.segment_sum(u, group, num_segments=n_seg)
    z_safe = jnp.where(exists, z, 1.0)
    w = u / z_safe[group]
    log_a = jnp.where(exists, mx + jnp.log(z_safe), -jnp.inf).reshape(G, F)
    exists = exists.reshape(G, F)
    residual = jnp.where(exists, targets - jax.ops.segment_sum(x, group, num_segments=n_seg).reshape(G, F), 0.0)
    q_fragment = x + w * residual.reshape(-1)[group]
    # conductance of the inter-group edges
    t = jnp.clip((dist - contact_radius) / (cutoff - contact_radius), 0.0, 1.0)
    sw = jnp.clip(1.0 - t ** 3 * (10.0 - 15.0 * t + 6.0 * t * t), 0.0, 1.0)
    gi, gj = group[recv], group[send]
    inter = edge_mask & (gi != gj)
    if pairs == "charged":
        charged = (jnp.abs(targets) > 1e-6).reshape(-1)
        inter = inter & (charged[gi] | charged[gj])
    cond = jnp.where(inter, sw / (1.0 - sw + eps), 0.0)
    graph_e = gi // F
    adj = jax.ops.segment_sum(cond, graph_e * (F * F) + (gi % F) * F + (gj % F),
                              num_segments=G * F * F).reshape(G, F, F)
    adj = 0.5 * (adj + jnp.swapaxes(adj, -1, -2))

    def solve_graph(adj, log_a, exists, residual):
        any_group = jnp.any(exists)
        a = jax.nn.softmax(jnp.where(exists, log_a, -jnp.inf))
        a = jnp.where(any_group, jnp.where(exists, a, 0.0), 1.0 / F)       # padding graph: any finite weights
        a = jnp.where(exists | ~any_group, jnp.maximum(a, 1e-12), 0.0)
        a = a / jnp.sum(a)
        pos = a > 0
        den = a[:, None] + a[None, :]
        adj = adj * (a[:, None] * a[None, :]) / jnp.where(den > 0, den, 1.0)
        lap = jnp.diag(jnp.sum(adj, axis=1)) - adj
        root_safe = jnp.sqrt(jnp.where(pos, a, 1.0))          # no sqrt at 0 (infinite derivative -> NaN gradients)
        root = jnp.where(pos, root_safe, 0.0)
        v = root + jnp.zeros((F,), dt).at[0].set(1.0)
        eye = jnp.eye(F, dtype=dt)
        basis = (eye - 2.0 * v[:, None] * v[None, :] / jnp.sum(v * v))[:, 1:]
        matrix = eye + lap / (root_safe[:, None] * root_safe[None, :])
        reduced = basis.T @ matrix @ basis
        rhs = basis.T @ (residual / root_safe)
        variable = basis @ jnp.linalg.solve(reduced, rhs)
        correction = a * jnp.sum(residual) + root * variable - residual
        return jnp.where(jnp.sum(adj, axis=1) > 0, correction, 0.0)

    delta = jax.vmap(solve_graph)(adj, log_a, exists, residual)
    return q_fragment + w * delta.reshape(-1)[group]
