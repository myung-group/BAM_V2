# RACE v3 (RACE_POLAR) — formulas as implemented

Config keys / values relied on (first JSON of `loss_log.txt`, lines 1-451; the JSON after the restart at line 475 differs only in `n_epochs` 185 -> 170 and the note):

1. `model RACE_POLAR`, `cutoff 6.0`, `hidden_size 128`, `lmax 3`, `n_layers 5`, `radial_basis_size 12`, `radial_mlp_size 64`, `radial_mlp_layers 3`, `radial_polynomial_p 6`, `mlp_init_scale 4`, `radial_shells true`, `radial_prior {gaussian, sigma 0.224}`
2. `x_mode basis`, `x_norm spectral`, `x_readout_norm_eps 0.003`, `update linear`, `message_norm avg`, `avg_n_neighbors 32.584`, `readout_mode gated`, `residual chebyshev`, `layer_scale 0.5`, `tp_backend cue`, `cov_mode scalar6`, `species_rank 32`, `species_rank_prune false`
3. `pair_repulsion uff`, `pair_repulsion_switch 1.25`, `pair_repulsion_switch_width 0.3`, `e0_mode isolated`, `atom_energies [...]`, `use_dataset_stats false` (shift 0, scale 1 at init), `periodic false`, `stress_loss false`, `x64 false`
4. `long_range`: `charge_spin_embedding true`, `n_tasks 8`, `chi_init pauling`, `chi_init_missing mean`, `qeq_block 352`, `sigma_bounds [0.5, 2.0]`, `cs_encoding table`, `cs_layers all`, `cs_species_gate true`, `cs_max_charge 8`, `cs_max_spin 12`, `cs_local [4.5, 6.0]`
5. `long_range` (ct): `ct_mode conductance`, `ct_switch [3.0, 4.5]`, `ct_init 1.15`, `ct_cond_max 100`, `ct_decay [4.0, 2.0]`, `ct_decay_learn true`, `ct_decay_min 2.5`, `ct_label_keys [nbo_charges, lowdin_charges]`, `ct_bond_factor 1.2`
6. `long_range` (spin / NBO): `spin_head true`, `spin_energy true`, `spin_w_init sdftb`, `spin_pair true`, `spin_nbo_head true`, `charge_nbo_head true`, `nbo_t_max 0.5`; defaults used: `head_hidden 32`, `cs_hidden 32`, `hard_init 5.0`, `chi_init_scale 3.0`, `chi_scale 1.0`, `spin_mu_bias -4.0`, `spin_kappa_init 0.5`, `spin_pair_switch None -> [4, 6]`, `spin_nbo_switch None -> ct_switch`, `sr_switch None`
7. `polar`: `sigma 1.0`, `thole_width 2.0`, `polarizability_init 0.7`, `min_polarizability 0.001`, `max_polarizability 4.0`, `max_polarizability_per_element {H: 1.0}`, `quadrupoles true`, `cap_induced_width true`, `cg_max_iterations 25`, `cg_tolerance 1e-4`, `multipole_init_scale 0.1`, `charges qeq`, `dispersion true`; defaults used (not in the JSON): `induction True`, `mutual_induction True`, `diagonal_shift 1e-6`, `envelope_p 6`, `feature_norm smooth`, `feature_norm_scale 1.0`, `disp_kappa_init 7.5`, `disp_a1 0.8`, `disp_a2 0.0`, `disp_cutoff 12.0`, `disp_n_img 1`
8. Loss: `loss_type huber`, `energy_weight 10`, `force_weight 1`, `huber_delta_energy 0.05`, `huber_delta_force 0.25`, `huber_delta_stress 0.05` (unused), `charge_weight 0.001`, `charge_label_key lowdin_charges`, `charge_label_optional true`, `charge_nbo_label_key nbo_charges`, `dipole_weight 1.875e-5`, `spin_weight 6.25e-4`, `spin_label_key lowdin_spins`, `spin_nbo_label_key nbo_spins`, `total_charge_key charge`, `total_spin_key spin`
9. Optimiser: `optimizer soap`, `soap_b1 0.95`, `soap_b2 0.95`, `soap_precondition_frequency 10`, `soap_impl grouped`, `soap_species_adam any_axis`, `soap_state_dtype float64`, `learning_rate 0.01`, `lr_schedule cosine`, `lr_warmup_epochs 2`, `lr_cosine_min_scale 0.05`, `grad_clip 1.0`, `ema_decay 0.999`, `ema_warmup true`, `gauge_refit true`, `seed 42`
10. Data: `data_format stream`, `batch_size 48`, `stream.graphs_per_pack 48`, `stream.pack_max_graphs 192`, `stream.n_buckets 1`, `stream.steps_per_epoch 1306`, `stream.dipole true`, `n_epochs 185 (170 after the restart at epoch 3)`; log lines: `12 device(s), 48 graphs per device`, `bucket 0: 2688 nodes, 86016 edges, <= 192 graphs`

Conventions. Code root: `/scratch/t510a13/BAM_V2_run_v3/bam_omat24/` (HEAD `d90bdc5`, `git diff --stat HEAD -- bam_omat24` empty). File references below are relative to it unless a full path is given. Units: Angstrom, e, eV; `K = 14.399645 eV A` (`long_range.py:20`; `thole.py:49` uses 14.399645351950543). Edge (i, j): receiver i, sender j, both directions in the list. `switch(r; a, b)` is the C2 smoothstep `x^3 (10 - 15 x + 6 x^2)`, `x = clip((r - a)/(b - a), 0, 1)`: 0 for r <= a, 1 for r >= b (`long_range.py:39-44`). VERIFIED = read from the cited lines; INFERRED = reconstructed.

Total energy as assembled (VERIFIED; `race_nnx.py:2093-2102`, `race_lr_nnx.py:525,538`, `race_polar_nnx.py:362,366`, `utils.py:392-402,420`):

```
E_i   = scale * ( e_rep,i + sum_{t=0..4} [ e_node + e_x + e_f ]_{t,i} ) + shift      (local RACE strata)
        + e_qeq,i        (on-site Qeq energy only)
        + e_spin,i
        + e_elec,i       (Thole block: Coulomb of q, mu, Theta + polarisation)
        + e_disp,i
E     = sum_i E_i                      (training: label is E_DFT - sum_i E0[Z_i])
E_inf = sum_i E_i + sum_i E0[Z_i]      (l_train False; E0 = config atom_energies, stop_gradient)
F     = - d E / d positions            (one jax.grad through everything; NaN gradient entries are set to 0)
```

`shift` and `scale` are trainable scalar parameters (init 0 and 1), `race_nnx.py:1379-1380`. Note that `scale` multiplies the UFF repulsion too, but not the Qeq / spin / Thole / dispersion terms.

---

## 1. Backbone

**(a) Formulas**

Edge features (VERIFIED; `utils.py:17-58, 233-258`; `race_nnx.py:1797-1812, 1460-1482, 1831-1833`):

```
r_ij_vec = x_j - x_i (sender - receiver),  r = |r_ij_vec|
B_n(r)   = (2 / r_c) sin(n pi r / r_c) / r,  n = 1..12,  r_c = 6 A
f_cut(r) = 1 - ((p+1)(p+2)/2) u^p + p (p+2) u^(p+1) - (p (p+1)/2) u^(p+2),  u = r / r_c,  p = 6   (0 for u >= 1)
           = 1 - 28 u^6 + 48 u^7 - 21 u^8
Bessel channels   b_n(r)  = B_n(r) f_cut(r) * exp(-(0.224 n)^2)            (radial_prior gaussian, fixed rescaling)
shell channels    s_ab(r) = exp(-r / (rho_a(Z_i) + rho_b(Z_j))) f_cut(r),  a, b in {s, p, d, f}   (16 channels, 0 if a shell is absent)
radial input      R_in(r) = [b_1..b_12, s_ss..s_ff]                         (28 numbers; all zero on r = 0 padding edges)
Y(r_hat)          = real spherical harmonics l = 0..3, "component" normalisation, unit vector
```

Embedding and conditioning (VERIFIED; `race_nnx.py:1843-1844`, `race_lr_nnx.py:403-447`):

```
e_i     = W_emb onehot(Z_i)                                (128x0e)
h_i^(0) = e_i + z_cs,i * (1 + G0[Z_i]) + task_row[0]       (section 4; task table row 0 since no task is set)
```

Atomic basis (computed once, from the bare species embedding e_j, before the charge/spin conditioning is added; uses the layer-0 `linear_1` and radial MLP) (VERIFIED; `race_nnx.py:1919-1928, 800-847`):

```
A_i = (1 / n_avg) sum_{j in N(i)} R^(0)(r_ij) (.) [ W_1^(0) e_j (x) Y(r_hat_ij) ],   n_avg = 32.584
```

Layer t = 0..4 (VERIFIED; `race_nnx.py:800-847, 859-972, 1944-2053`):

```
h~^(t)   = h^(t) + cs term of layer t (t >= 1, section 4)
m_i      = (1 / n_avg) sum_{j in N(i)} sum_paths R^(t)_path(r_ij) [ W_1 h~_j (x) Y(r_hat_ij) ]_path     (cue channelwise TP,
           fused gather / TP / scatter; R^(t) = MLP(R_in): 28 -> 64 -> 64 -> 64 -> n_paths*128, SiLU, no bias)
x_i      = W^_t A_i,   W^_t = W_t with every output-irrep block divided by its largest singular value   (x_norm spectral)
p_i      = W_2 [ (W_m m_i) (x)_channelwise x_i ]         (per channel u: all CG paths (l1, l2) -> l3 inside 0e,1o,2e,3o)
delta_i  = gamma (.) ( p_i + S[Z_i] h~_i )               (gamma: per-channel LayerScale, layers t >= 1 only, init 0.5;
                                                          update "linear": no gate / non-linearity on the features)
h^(t+1)  = delta                     for t = 0, 1
h^(t+1)  = 2 delta - h^(t-1)         for t >= 2          (residual "chebyshev"; h^(t-1) = input of layer t-1, unconditioned)
```

Stratum energy of layer t (VERIFIED; `race_nnx.py:955-968, 425-498, 1535-1558, 1984, 2035-2053`; `utils.py:61-108`):

```
e_node = w . gate_silu( W_r1 h^(t+1) + b )|_{64x0e} + b'                       (readout_mode gated: only 0e outputs of W_r1 are kept)
e_x    = MLP_x[Z_i]( LN_e0(x_i) / sqrt(n_avg) )                                 (x readout)
e_f    = MLP_f[Z_i]( LN_1e-6(h~^(t)_i) / sqrt(n_avg) )                          (f readout, on the layer INPUT)
MLP[Z](v) = w3 . gate_silu( W2 ( W1[Z] v ) + b2 ) + b3,   W1[Z]: v -> 128x0e (species-indexed, no bias), W2: 128x0e -> 64x0e
LN_eps (separated layer norm):  scalars:     (v0 - mean_c v0) / sqrt( mean_c (v0 - mean)^2 + eps )
                                non-scalars: v / sqrt( mean over all l>0 components of v^2 + eps )
x readout: eps = x_readout_norm_eps^2 = (3e-3)^2 = 9e-6;  f readout: eps = 1e-6
stratum gain = 1 (uniform)
```

Species tables with `species_rank 32` (VERIFIED; `linear_nnx.py:224-230, 362-370`; used by the skip `S[Z]`, the first Linear of the x / f readouts and `x_linear`, `race_nnx.py:1414-1439, 755-764`):

```
W[Z] = 32^(-1/2) sum_{k=1..32} c[Z, k] B[k]       (c: (n_species, 32), B: (32, mul_in, mul_out); biases stay free tables)
```

UFF pair repulsion, lowest stratum (VERIFIED; `race_nnx.py:1886-1901, 1907`; table `race_nnx.py:275-378`):

```
x_ij = sqrt(x_i x_j), D_ij = sqrt(D_i D_j), zeta_ij = sqrt(zeta_i zeta_j)          (UFF Table 1; D in eV)
e_ij = D_ij * 6 / (zeta_ij - 6) * exp( zeta_ij (1 - r / x_ij) ) * f_cut(r) * sigmoid( (r - 1.25 (R_i + R_j)) / 0.3 )
e_rep,i = 1/2 sum_{j in N(i)} e_ij                                                 (R = Cordero covalent radii; 0 on r = 0 edges)
```

**(b) Symbols.** `h` node features `128x0e+128x1o+128x2e+128x3o` (layer-0 input is `128x0e`); `W_*` learned equivariant linear maps (e3nn-style `Linear`); `S[Z]`, `W1[Z]` learned per-element (rank-32 factorised); `gamma` learned per channel; `R^(t)` learned radial MLP per layer; `rho_a(Z)` fixed (outermost-shell `<r>` from `atomic_shells.json`, `train_3bpa.py:326-333`); UFF constants fixed; `E0[Z]` fixed (config); `shift`, `scale` learned scalars.

**(c) v3 values.** As in config lines 1-3 above. `x_readout mlp`, `f_readout true`, `stratum_gain uniform`, `embed_mode onehot`, `pair_repulsion_scale 1.0` (defaults).

**(e) Notes.**
- `x_norm`: "spectral" is active; "hill" and "rms" exist (`race_nnx.py:222-237`) but are off, and no `x_hill` parameter exists in v3.
- The x and f readouts only read the scalar (0e) part: their first Linear maps to `128x0e`, so the l > 0 parts of x enter `e_x` not at all (they enter only the product `m (x) x`).
- `x_readout_norm_eps` is squared before use (`race_nnx.py:1546`); the f readout keeps `1e-6`.
- `species_rank_prune false`: the unused `x_linear` table (x_mode "species") still exists with rank 32 and receives no gradient.
- `cov_mode scalar6` only concerns the uncertainty heads (`frc_logvar`, `energy_logvar`, `stress_logvar`, `race_nnx.py:2104-2127`); with `loss_type huber` they are computed but never enter the loss.
- Pre-norm of messages is off (`message_norm avg`): sum / n_avg, not / sqrt(n_avg); the readouts divide by sqrt(n_avg).
- NOT DETERMINED: the path-normalisation constants of the cuEquivariance channelwise tensor product and of the e3nn `FunctionalLinear` (`path_weight`); they come from the libraries, which were not read.
- NOT DETERMINED: numerical values of `rho_a(Z)` (file `atomic_shells.json` not opened).

---

## 2. Charge equilibration (Qeq)

**(a) Formulas** (VERIFIED; heads `race_lr_nnx.py:485-490, 654-661`; matrix `long_range.py:28-36`; energy `long_range.py:161-171`; mode switch `race_polar_nnx.py:92-105`, `race_lr_nnx.py:734-735`)

```
s_i     = 0e scalars of the last-layer features h^(5)_i (128 numbers, not normalised)
z_i     = W_b silu( W_a s_i + b_a ) + b_b                    (128 -> 32 -> 2, W_b zero-initialised)
chi_i   = chi0[Z_i] + chi_scale * z_i[0]                     chi_scale = 1
J_i     = softplus( hard0[Z_i] + z_i[1] )                    > 0
sigma_i = r_cov(Z_i) * ( lo + (hi - lo) sigmoid(u[Z_i]) ),   [lo, hi] = [0.5, 2.0]        (per ELEMENT only)

A_ii    = J_i + K / (sqrt(pi) sigma_i)
A_ij    = K erf( r_ij / (sqrt(2) sqrt(sigma_i^2 + sigma_j^2)) ) / r_ij      (all pairs of the structure, no cutoff, open boundary)

E_solve(q) = sum_i chi_i q_i + 1/2 sum_ij q_i A_ij q_j        (functional that defines q; see section 3 for the constraint)

energy kept from this stratum (polar.charges "qeq" forces coulomb_energy False -> the pair matrix in the energy is zero):
e_qeq,i = chi_i q_i + 1/2 ( J_i + K / (sqrt(pi) sigma_i) ) q_i^2
```

The pair Coulomb energy of q is computed once, in the Thole block (section 7), with a different (fixed) Gaussian width `polar.sigma = 1 A`.

**(b) Symbols.** `chi` (eV/e) and `J` (eV/e^2): predicted per atom by the network head plus learned per-element offsets `chi0[Z]`, `hard0[Z]`. `sigma_i` (A): learned per element (`u[Z]`, parameter name `log_sigma`), bounded to `[0.5, 2] x r_cov`; it is NOT predicted per atom. `q_i` (e): solved. `r_cov` fixed (Cordero, `train_3bpa.py:263-275`).

**(c) v3 values / initialisation.** `chi0[Z] = 3.0 * (chi_Pauling(Z) - mean over the 94 species)`, elements without a Pauling value start at the mean (`race_lr_nnx.py:196-210`); `hard0` such that `J = 5.0 eV/e^2`; `u0 = ln((1 - lo)/(hi - 1)) = ln 0.5`, i.e. `sigma = r_cov` at the start (`race_lr_nnx.py:216-224`); `head_hidden 32`; `qeq_block 352`.

**How it is solved** (VERIFIED; `race_lr_nnx.py:696-765`): the atoms of every structure are gathered into a padded block of `B = 352` slots; matrices are built per block (`G x B x B`) and one dense `jnp.linalg.solve` per structure is vmapped over the batch. A structure with more than 352 atoms gives NaN energies. With `ct_mode conductance` the system solved is the one of section 3 (non-symmetric, no Lagrange multiplier); the plain constrained Qeq (`long_range.py:114-135`, KKT system with one multiplier per graph / fragment) exists but is off.

**Forces** (VERIFIED for the mechanism, `utils.py:361-393`; `race_polar_nnx.py:93-97`): one `jax.grad` of the total energy with respect to positions, differentiating through the linear solve. The energy is not variational in q (q minimises `E_solve` + the ct penalty, the energy uses `e_qeq` + the Thole Coulomb energy), so the response term dq/dr is needed and is included.

**(e) Notes.**
- Self energy appears twice with two widths: `K q^2 / (2 sqrt(pi) sigma_i)` here (element width) and `K q^2 / (2 sqrt(pi) * 1 A)` inside the Thole permanent energy (section 7). INFERRED from `long_range.py:161,169-170` and `thole.py:369,456`.
- The solve runs in float64 although the model is float32: `soap_state_dtype float64` switches `jax_enable_x64` on (`train_3bpa.py:1039-1041`), and `ct_solve` then casts its inputs to float64 (`long_range.py:153-157`). At inference this depends on whether x64 is enabled.
- The raw charge head `q_raw` and the Fukui heads of RACE_POLAR exist as parameters but are unused in "qeq" mode (`race_polar_nnx.py:336-340`).

---

## 3. Charge transfer by conductance (ct)

**(a) Formulas** (VERIFIED; `long_range.py:138-171`; `race_lr_nnx.py:663-694, 747-756`)

```
pair conductance on the neighbour-list edges (r_ij < 6 A; zero beyond 4.5 A):
  g_ij    = g[Z_i] + g[Z_j]
  s_ij    = ln C - softplus( ln C - g_ij )                   C = ct_cond_max = 100 e^2/eV   (smooth cap: exp(s) <= C)
  beta_ij = beta_min + softplus( b[Z_i] + b[Z_j] )            beta_min = ct_decay_min = 2.5 1/A
  c_ij    = exp( s_ij - beta_ij (r_ij - r0) ) * ( 1 - switch(r_ij; 3.0, 4.5) ),   r0 = 2.0 A

graph Laplacian   L = diag(sum_j c_ij) - c
charges           q = q0 + B p,   p = argmin_p [ E_solve(q0 + B p) + 1/2 sum_{pairs} p_ij^2 / c_ij ]      (split-charge form)
  <=>             ( I + L A ) q = q0 - L chi                  (solved; A as in section 2)
```

**What is constrained, precisely** (VERIFIED from `long_range.py:143-148`): `1^T L = 0`, so every connected component of the conductance graph (atoms linked by edges shorter than 4.5 A) keeps the sum of its `q0` exactly. Charges are therefore conserved per conductance-connected cluster, not per molecule: molecules (bonds, section 12) only define `q0`. Two molecules closer than 4.5 A exchange charge; an atom without any conductance keeps `q0_i`; for `c -> infinity` inside a component the charges tend to the Qeq charges of that component. The graph total is conserved because `sum q0 = Q`.

**(b) Symbols.** `g[Z]` (parameter `ct_log_g`) and `b[Z]` (parameter `ct_decay_b`) learned per element; `beta_ij` is the additive combination of two per-element numbers, not a free pair table. `c_ij` in e^2/eV. `q0` from the data (section 12); without node field `q0` the code uses `Q/N` on every atom (`race_lr_nnx.py:751-754`).

**(c) v3 values / initialisation.** `g = 1.15` (so `g_ij = 2.3`, `exp = 10`; after the cap `s = ln 100 - softplus(ln 100 - 2.3) ~ 2.205`, conductance ~9.1 at r = r0); `b0 = 1/2 ln(expm1(4.0 - 2.5)) ~ 0.624` so that `beta_ij = 4.0 1/A` everywhere at the start (`race_lr_nnx.py:244-245`).

**(e) Notes.**
- The decay is applied AFTER the cap (`race_lr_nnx.py:681-690`), so for `r < r0 = 2 A` the factor `exp(-beta (r - r0)) > 1` and the conductance exceeds `ct_cond_max` (e.g. x e^4 ~ 55 at r = 1 A with beta = 4): the cap holds only for r >= 2 A. This confirms the known issue.
- `I + L A` is not symmetric; solved by dense LU in float64 (section 2).
- Without the decay the code comment reports a step in the charge transfer at 4.5 A (`race_lr_nnx.py:56-63`); with it the transfer fades over about `4.4 / beta` A.
- Per-atom gathers `g[sp]` then per edge (`race_lr_nnx.py:677-680`): a performance detail, same mathematics.

---

## 4. Local charge / spin conditioning (cs_local)

**(a) Formulas** (VERIFIED; `race_lr_nnx.py:386-401, 403-447, 449-474, 363-372`)

```
w(r)     = 1 - switch(r; 4.5, 6.0)                              (1 below 4.5 A, 0 at 6.0 A)
Qbar_i   = q0_i + sum_{j in N(i)} w(r_ij) q0_j
2Sbar_i  = s0_i + sum_{j in N(i)} w(r_ij) s0_j

pre_i    = W_c1 [Qbar_i, 2Sbar_i] + b_c1                         (2 -> 32)
           + T_q( clip(Qbar_i, -8, 8) + 8 )                      (table 17 x 32, rows linearly interpolated)
           + T_s( clip(2Sbar_i, 0, 11) )                         (table 12 x 32, rows linearly interpolated)
hid_i    = silu(pre_i)
layer 0:  h_i^(0) = e_i + ( W_c2 hid_i + b_c2 ) * (1 + G[Z_i, 0])                     (added to the 128 layer-0 scalars)
layer t >= 1:  0e block of h_i^(t)  +=  ( hid_i V_t ) * (1 + G[Z_i, t])               (V_t: 32 -> 128, one per layer)
```

`W_c2`, `T_q`, `T_s`, `V_t`, `G` are zero-initialised, so the model starts as the unconditioned one.

**What it replaces.** Without `cs_local` the inputs are the graph totals `(Q, 2S+1 - 1)` broadcast to all atoms (`race_lr_nnx.py:418-419`); with it every atom gets its own `(Qbar_i, 2Sbar_i)` built from the per-atom reference values `q0`, `s0` (section 12). The same local `Sbar` is the pair feature of the spin energy (section 5).

**Why size consistent** (INFERRED from the construction; stated in the code comment `race_lr_nnx.py:76-87`): `Qbar_i` and `2Sbar_i` only use atoms inside the 6 A neighbour list, and `q0` / `s0` of a molecule do not depend on other molecules, so the conditioning of a fragment is unchanged when another fragment further than 6 A away is added. For a structure whose atoms are all within 4.5 A of each other `Qbar_i = Q` and `2Sbar_i = 2S` (identical to the graph-total model). For larger molecules `Qbar_i` is a local partial sum of the evenly spread molecular charge, not the molecular charge.

**(c) v3 values.** `cs_local [4.5, 6.0]`, `cs_hidden 32`, `cs_encoding table`, `cs_layers all`, `cs_species_gate true`, `cs_max_charge 8`, `cs_max_spin 12`.

**(e) Notes.**
- The atomic basis `A_i` is built from the bare species embedding, before the conditioning is added (`race_nnx.py:1919-1928` runs inside `super()._prelude`, `race_lr_nnx.py:404`); the conditioning enters through the layer inputs only.
- Size consistency is lost in the fallbacks of section 12 (`q0 = Q/N` uniform) and, independently, in the spin head / spin energy, which still use the graph total 2S (section 5).
- The task row (`n_tasks 8`, row 0 only) is a learned constant vector added to the layer-0 scalars (`race_lr_nnx.py:432-438`).

---

## 5. Spin

**(a) Spin head** (VERIFIED; `race_lr_nnx.py:638-652`)

```
hid_i          = silu( W_s1 s_i + b_s1 )                       (128 -> 32; s_i as in section 2)
(raw_i, l_i)   = W_s2 hid_i + b_s2                              (W_s2 zero-initialised)
w_i            = softmax over the atoms of the graph of l_i
m_i            = raw_i + w_i ( 2S - sum_k raw_k )               (sums exactly to 2S = multiplicity - 1 of the graph)
```

`m_i` (output `spins`) is an auxiliary output trained on `lowdin_spins`; it does not enter the energy.

**Spin energy** (VERIFIED; `race_lr_nnx.py:587-632, 579-585`)

```
mu_i^raw = softplus( w_mu . hid_i + b_mu )                                        (b_mu init -4: mu ~ 0.018)
mu_i     = u_g * [ mu_i^raw + w_i * max(0, 2S - sum_k mu_k^raw) ]                 (sum mu >= 2S; w_i = spin-head softmax weights)
u_g      = 1 if 2S+1 > 1.5, else clip(globals['unrestricted'], 0, 1)              (0 if the field is absent)
Delta_i  = delta0[Z_i] + w_D . hid_i + b_D                                        (zero init)
c_ij     = tanh( MLP_c([hid_i + hid_j, hid_i * hid_j, Sbar_i]) )                  (65 -> 32 -> 1, output layer zero; Sbar_i = 1/2 * 2Sbar_i of the RECEIVER i)
J_ij     = A_{Zi Zj} exp( min( -(kappa_Zi + kappa_Zj)(r_ij - r_cov,i - r_cov,j), 20 ) ) * ( 1 - switch(r_ij; 4, 6) )
           A = (T + T^T)/2 (zero init),  kappa_Z = softplus(.) (init 0.5 1/A)
e_spin,i = Delta_i mu_i + 1/2 W[Z_i] mu_i^2 + 1/2 sum_{j in N(i)} J_ij mu_i mu_j c_ij
E_spin   = sum_i [ Delta_i mu_i + 1/2 W_Z mu_i^2 ] + sum_{i<j} J_ij mu_i mu_j (c_ij + c_ji)/2
<S^2> estimate (output spin_s2) = S_z (S_z + 1) + 1/2 ( sum_i mu_i - 2 S_z ),   S_z = S
```

**(b) Symbols.** `mu_i >= 0` local moment (electrons), `c_ij in [-1, 1]` pair correlation, both predicted by the network; learned per element: `delta0[Z]` (eV), `W[Z]` (eV per electron^2), `kappa[Z]` (1/A); learned per element pair: `A` (eV); `r_cov` fixed.

**(c) v3 values / initialisation.** `spin_w_init sdftb`: `W = -0.0717, -0.0227, -0.026, -0.028 Ha x 27.211386 = -1.951, -0.618, -0.707, -0.762 eV` for H, C, N, O and 0 for all other elements (`race_lr_nnx.py:129-130, 306-316`); `spin_pair_switch` default `[cutoff - 2, cutoff] = [4, 6]`; `spin_mu_bias -4`; `spin_kappa_init 0.5`.

**Labels and loss terms**: section 10.

**Restricted vs unrestricted** (VERIFIED; `race_lr_nnx.py:579-585`, `stream_io.py:349-351`): the row flag is `unrestricted = 1 if (multiplicity > 1 or info['unrestricted'])`. A restricted singlet has `u_g = 0`, hence `mu = 0` and `E_spin = 0` exactly. Every open-shell graph and every singlet flagged unrestricted has `u_g = 1`.

**(e) Notes.**
- Kink: `max(0, 2S - sum mu)` is a hard `jnp.maximum` (`race_lr_nnx.py:597`); the energy has a discontinuous derivative where `sum_k mu_k^raw = 2S`.
- The lift uses the graph-total 2S and a softmax over the whole graph, and `u_g` is a per-graph flag: these parts are not local (not size consistent for separated fragments), unlike the `cs_local` conditioning.
- `c_ij` is not exactly symmetric with `cs_local`: the third feature is the local `Sbar` of the receiver; the energy uses the average of `c_ij` and `c_ji`.
- The exponent of `J` is capped at 20.

---

## 6. NBO heads

**(a) Formulas** (VERIFIED; `race_lr_nnx.py:527-532, 542-545, 548-577`)

```
redistribution map   R_t[v]_i = v_i + sum_{j in N(i)} t_ij ( v_j - v_i )            (conserves sum_i v_i since t_ij = t_ji)
t_ij = t_max tanh( MLP([a_i + a_j, a_i * a_j, r_ij]) / t_max ) * ( 1 - switch(r_ij; 3.0, 4.5) )     (65 -> 32 -> 1, output layer zero)

charges:  a_i = silu( W_q s_i + b_q )   (own 128 -> 32 layer),  q^NBO_i = R_{t^q}[q]_i        (q = the ct / Qeq charges)
spins:    a_i = hid_i (spin-head hidden layer),                 m^NBO_i = R_{t^s}[m]_i        (m = spin-head populations)
```

**(b) What is what.**
- `charges` (the ct / Qeq charges `q`) are the Lowdin-like charges: trained against `lowdin_charges` (`charge_label_key`) and they are the charges that enter the electrostatics (Thole block) and the dipole.
- `charges_nbo`, `spins_nbo` are auxiliary outputs; they do not enter the energy.
- In the loss the maps are applied to the LABELS, not to the predictions: `R_{t^q}[q^Lowdin_ref]` vs `q^NBO_ref` and `R_{t^s}[m^Lowdin_ref]` vs `m^NBO_ref` (section 10). These terms therefore train only `t` (and, through its input features, the backbone), not `q` or `m`.
- NBO charges do enter the model indirectly at training time: `ct_label_keys = [nbo_charges, lowdin_charges]`, so the integer molecule charges `q0` come from the summed NBO charges when a row has them (section 12).

**(c) v3 values.** `nbo_t_max 0.5` (|t| <= 0.5); switch = `ct_switch [3.0, 4.5]` for both heads (`spin_nbo_switch` default).

---

## 7. Polarisation (Thole block)

**Active branch.** The config has no `induction` / `mutual_induction` keys; `RACE_POLAR` starts from `thole.POLAR_DEFAULTS` and overwrites only the keys given (`race_polar_nnx.py:111-115`, `thole.py:53-72`), so `induction = True` and `mutual_induction = True`: the induced dipoles are solved self-consistently (`thole.py:306-307, 446-454`). The one-shot response (`mutual_induction False`, `thole.py:440-445`) and no induction exist but are off. NOT DETERMINED: which tool printed the summary with `null` (no such print exists in this package; a summary that reads `cfg['polar'].get(...)` shows null for defaulted keys).

**(a) Heads** (VERIFIED; `race_polar_nnx.py:248-284, 133-165`)

```
f_i      = h^(5)_i with every irrep type scaled by rsqrt( 1 + mean square of that type )          (feature_norm smooth, scale 1)
mu_i     = 0.1 * Linear(f_i -> 1x1o)                           permanent dipole (e A)
Theta_i  = 0.1 * sym_traceless( Linear(f_i -> 1x2e) ) / sqrt(2)     permanent traceless quadrupole (e A^2), Linear zero-initialised
alpha_i  = a_min + ( a_max[Z_i] - a_min ) sigmoid( Linear(f_i -> 0e) + b_alpha )                  (A^3), Linear zero-initialised
q_i      = ct / Qeq charges of sections 2-3
```

**Kernels and widths** (VERIFIED; `thole.py:76-90, 119-158, 396-405`)

```
T_a(r) = erf(r / a) / r,   c_m^a(r) = D^m T_a,  D = (1/r) d/dr          (c_1 = T'/r, c_2 = (T'' - T'/r)/r^2, ...)
permanent-permanent width   a_0  = 2 sigma = 2.0 A                       (polar.sigma = 1.0 A, same for every atom)
split width                 eta  = max(2 sigma, cutoff / sqrt(ln 1e8)) = max(2.0, 1.398) = 2.0 A  (= a_0)
induced width               s_i  = (thole_width / 2) alpha_i^(1/3) = alpha_i^(1/3),  capped: s (1 + (s / c)^16)^(-1/16),  c = eta/2 = 1.0 A
induced-permanent           a_i  = sqrt( 2 (s_i^2 + sigma^2) )
induced-induced             a_ij = sqrt( 2 (s_i^2 + s_j^2) )
env(r) = polynomial envelope of section 1 with p = 6, r_max = 6 A
```

**Fields, open-boundary graphs** (VERIFIED; `thole.py:202-241, 359-374, 377-405`; `d = x_i - x_j`):

```
phi_i   = sum_{j != i} [ q_j c_0 - c_1 mu_j.d + (1/3) c_2 d.Theta_j.d ]_eta  +  q_i * 2 / (eta sqrt(pi))
E_i     = sum_{j != i} [ (-q_j c_1 + c_2 mu_j.d) d + c_1 mu_j - (1/3)(2 c_2 Theta_j d + c_3 (d.Theta_j.d) d) ]_eta  -  mu_i * 4 / (3 eta^3 sqrt(pi))
G_i     = grad grad phi_i  (multipole_sums with c_1..c_4, plus the self term of quadrupole_self_gradient)
          sums over ALL atoms j of the same structure (dense, no cutoff)

F_i     = sum_{j != i} [ field of (q_j, mu_j, Theta_j) ]_eta                                   (same as E_i without the self term)
          + sum_{j in N(i)} env(r_ij) [ field of (q_j, mu_j, Theta_j) with radial factors c_m^{a_i} - c_m^{eta} ]     (a_i of the receiver)
(T v)_i = sum_{j != i} [ c_1 v_j + c_2 (d.v_j) d ]_eta
          + sum_{j in N(i)} env(r_ij) [ (c_1^{a_ij} - c_1^eta) v_j + (c_2^{a_ij} - c_2^eta) (d.v_j) d ]
```

Since `eta = a_0`, the short-range correction of the permanent-permanent part vanishes (`thole.py:386-395` is skipped): the permanent multipoles interact with the single kernel `erf(r / 2 A) / r` at all distances.

**Induced dipoles and energy** (VERIFIED; `thole.py:433-470`)

```
( 1/alpha_i + eps_d ) mu_ind,i - (T mu_ind)_i = F_i ,   eps_d = diagonal_shift = 1e-6
   solved by conjugate gradient, one CG for the whole batch, preconditioner alpha, tol 1e-4 (relative), at most 25 iterations

e_perm,i = (K/2) [ q_i phi_i - mu_i . E_i + (1/3) Theta_i : G_i ]
e_pol,i  = K [ 1/2 (1/alpha_i + eps_d) |mu_ind,i|^2 - mu_ind,i . F_i - 1/2 mu_ind,i . (T mu_ind)_i ]
e_elec,i = e_perm,i + e_pol,i
graph dipole = sum_i [ q_i r_i + mu_i + mu_ind,i ] - Q_graph R_com       (R_com mass weighted; open-boundary graphs; race_polar_nnx.py:354-360)
```

At the converged solution `sum_i e_pol,i = -(K/2) sum_i mu_ind,i . F_i` (INFERRED, algebra).

**(b) Symbols.** `alpha_i` (A^3), `mu_i`, `Theta_i`: predicted per atom by the network. `sigma = 1 A`, `thole_width = 2`, caps: fixed. Fields in e/A^2, potentials in e/A, energies via `K`.

**(c) v3 values.** `a_min 0.001`, `a_max 4.0` (H: 1.0), `b_alpha = ln(0.7 - 0.001) - ln(4.0 - 0.7)`, i.e. `alpha = 0.7 A^3` at the start for all elements except H, where the same bias gives `0.001 + 0.999 * 0.1748 ~ 0.176 A^3` (`race_polar_nnx.py:140-154, 277-280`).

**(e) Notes.**
- The damping is Gaussian (erf) smearing with widths from `alpha^(1/3)`, not Thole's exponential damping function; "thole_width" is the proportionality constant `s_i = thole_width alpha_i^(1/3) / 2`.
- The smooth cap makes `s_i <= 1 A`, active for `alpha > ~1 A^3`; beyond the 6 A cutoff (and wherever `env -> 0`) all induced kernels fall back to the `eta = 2 A` kernel.
- CG is truncated at 25 iterations / 1e-4, so the induced dipoles are approximately converged. Differentiation: `jax.scipy.sparse.linalg.cg` (implicit differentiation of the linear solve; stated in `thole.py:7`; INFERRED for the JAX internals).
- Permanent self terms are inside `e_perm` (`q_i^2 K / (2 sqrt(pi) * 1 A)` etc.), in addition to the Qeq on-site term (section 2).
- The dense open-boundary sums are N x N over the padded batch (pairs of the same graph only), `thole.py:359-374`.

---

## 8. Dispersion

**(a) Formulas** (VERIFIED; `race_polar_nnx.py:193-245`; parameters `182-189`)

```
C6_ii = softplus(kappa[Z_i]) * alpha_i^2                                   (eV A^6)
R_i   = 0.529177 * 2.54 * ( alpha_i / 0.148185 )^(1/7)                      (A; alpha in A^3 -> bohr^3 by / 0.148185)
C6_ij = 2 C6_ii C6_jj / ( (alpha_j / alpha_i) C6_ii + (alpha_i / alpha_j) C6_jj )            (Tkatchenko-Scheffler)
R0_ij = softplus(a1) * (R_i + R_j) + a2
e_disp,i = -1/2 sum_{j != i, same structure} C6_ij / ( r_ij^6 + R0_ij^6 )                    (Becke-Johnson-type damping)
```

**(b) Symbols.** `alpha_i`: the Thole polarisability of section 7 (per atom, from the network; full gradient, so the dispersion also trains alpha and contributes dalpha/dr to the forces). `kappa[Z]` learned per element; `a1` one learned scalar; `a2` fixed.

**(c) v3 values.** `softplus(kappa) = 7.5 eV` and `softplus(a1) = 0.8` at the start, `a2 = 0.0` (defaults `thole.py:70-71`).

**Range.** Open-boundary structures (v3 training): all pairs of the structure in blocks of `qeq_block = 352`, no cutoff and no switching; `disp_cutoff 12.0` is NOT used there. Periodic branch (off in training): images `|n_k| <= disp_n_img = 1`, factor `1 - switch(r; disp_cutoff - 2, disp_cutoff)`, plus the tail `-(2 pi / (3 V (disp_cutoff - 1)^3)) sum_j C6_ij` per atom (`race_polar_nnx.py:227-243`).

**(e) NaN guards.** Padding slots get `alpha = C6 = R = 1` instead of 0 (a `0 / (0 + tiny)` gave NaN gradients in float32, `race_polar_nnx.py:207-213`); pairs are masked with `where`; in the periodic branch `r = sqrt(max(r^2, 1e-12))`. Atoms beyond slot 352 are dropped from the dispersion sum silently (the Qeq block makes the energy NaN in that case anyway).

---

## 9. Electrostatics: open vs periodic

Open boundary (v3: `periodic false`, so every graph is open; VERIFIED `race_polar_nnx.py:326-333, 352`, `race_lr_nnx.py:732-733`): the Qeq matrix is the direct Gaussian-charge Coulomb sum over all pairs of a structure (`coulomb_direct`, no cutoff); the Thole block uses dense all-pair sums with the `eta = 2 A` kernel plus neighbour-list corrections for the induced widths; dispersion over all pairs. No Ewald, no cutoff on any 1/r term.

Periodic branch (present, off): Qeq matrix by Ewald per graph (`ewald_alpha 0.3`, `n_real 2`, `n_recip 6`, neutralising background, `long_range.py:73-111`); Thole block by one reciprocal-space sum per graph at width `eta` with a fixed k budget (`thole.py:320-356`), optional slab dipole correction; dispersion with images and tail (section 8).

---

## 10. Loss

**(a) Per-step loss** (VERIFIED; `training_utils.py:57-60, 273-319, 405-409, 516-530, 606-678, 886-922`). `n_dev = 12` devices; `G`, `N` = real graphs and atoms of the whole step (all devices).

```
huber_d(x) = x^2 / 2  if |x| <= d,  else d (|x| - d/2)

energy:  dE_g = ( E_pred,g - E_ref,g ) / N_g                         (per atom; energy_loss_per "atom")
         dE_g <- 5 tanh( dE_g / 5 )                                  (clip, eV/atom)
         L_E  = (1/G) sum_g huber_0.05( dE_g )
forces:  dF_i <- dF_i * 10 tanh(|dF_i| / 10) / |dF_i|                (norm clip, eV/A; |.| with + 1e-24 under the root)
         L_F  = (1/(3N)) sum_i sum_{x,y,z} huber_0.25( dF_i,x )

L = 10 L_E + 1 L_F + 0.001 L_q + 1.875e-5 L_dip + 6.25e-4 L_spin        (stress term off: stress_loss false)

L_q    = (1/N) sum_i l_i ( q_i - q_i^Lowdin )^2                                              (global normalisation)
         + sum_dev [ sum_i w_i^qn ( R_{t^q}[q^Lowdin_ref]_i - q_i^NBO )^2 / max(sum_i w_i^qn, 1) ]_dev
L_dip  = sum_dev [ sum_g w_g |p_g - p_g^ref|^2 / max(3 sum_g w_g, 1) ]_dev                   (e A; about the centre of mass)
L_spin = sum_dev [ L_mom + L_cor + L_sgn + L_nbo ]_dev
  L_mom = sum_i w_i ( mu_i - |m_i^ref| )^2 / max(sum_i w_i, 1)
  L_cor = sum_{edges} e_ij ( c_ij - sign(m_i^ref m_j^ref) )^2 / max(sum e_ij, 1),   e_ij = w_i w_j [|m_i^ref| >= 0.1][|m_j^ref| >= 0.1]
  L_sgn = sum_i w_i [2S+1 > 1.5] ( m_i - m_i^ref )^2 / max(sum_i w_i [2S+1 > 1.5], 1)
  L_nbo = sum_i w_i w_i^sn ( R_{t^s}[m^ref]_i - m_i^NBO,ref )^2 / max(sum_i w_i w_i^sn, 1)
```

**Reduction across GPUs** (VERIFIED; `training_utils.py:886-888, 918-922`; comments `637-638, 647-648`): every device computes its loss with the global counts `G`, `N` and the per-device losses and gradients are summed (`psum`). `L_E`, `L_F` and the first (Lowdin) part of `L_q` are therefore global means, independent of the device count. The terms normalised by their own per-device weight sums (NBO-charge part of `L_q`, `L_dip`, all of `L_spin`) are per-device means that are then summed, so their effective weight is the config weight times the number of devices:

```
effective (12 GPUs):  NBO-charge map 0.001 * 12 = 0.012,   dipole 1.875e-5 * 12 = 2.25e-4,   spin 6.25e-4 * 12 = 7.5e-3
Lowdin charge term: 0.001, not scaled
```

**Labels and weights** (VERIFIED; `stream_io.py:277-287, 318-351, 409-417`; `graph_fields.py:65-71`; `training_utils.py:173-186`):
- Energy label `E_DFT - sum_i E0[Z_i]` (float64 difference stored as float32); forces as given.
- Lowdin charges: a row without `lowdin_charges` gets NaN labels (`charge_label_optional`), masked by `l_i = isfinite`; the denominator stays the number of all real atoms `N`.
- NBO charges: weight `w^qn = 1` only for rows with finite `nbo_charges` of the right length, else 0.
- Spins: rows with finite `lowdin_spins` get them with weight 1; a row without them that is a singlet and not flagged unrestricted gets label 0 with weight 1; any other row without labels gets weight 0. NBO spins: weight 1 only where present.
- Dipole: ORCA `multipoles[1]` x 0.529177 (e A), weight 0 if absent.

**(e) Notes.**
- The tanh clipping of the residuals before the Huber function (`clip_residuals=True` is the default and is never switched off, `training_utils.py:250, 308-319, 506-507`) is part of the training and validation loss; the reported RMSE / MAE use raw residuals.
- Zero spin labels on restricted singlets give no gradient: `mu = 0` there by construction (section 5), `L_sgn` excludes singlets, `L_cor` needs `|m_ref| >= 0.1`. They only enlarge the denominator of `L_mom` (and of `L_nbo` only if NBO spin labels exist).
- The signed head `m_i` is trained on open-shell graphs only.
- Validation loss = `10 L_E + L_F` only (no auxiliary terms), on EMA weights; it selects the best checkpoint (`train_3bpa.py:1494-1495, 1526-1542`; `training_utils.py:1185-1187`).
- The `AUX` column of the log is the sum of the weighted auxiliary terms; the `LOSS` column is `10 L_E + L_F` (`train_3bpa.py:1487-1489, 1498`).

---

## 11. Training procedure

Optimiser chain (VERIFIED; `train_3bpa.py:816-871, 926-940, 958`; `soap_grouped.py:199`; log lines 461-463, 936-939):

```
g  <- psum over devices of the gradients
g  <- clip_by_global_norm(g, 1.0)
u  <- SOAP(g):  b1 = 0.95, b2 = 0.95, eps = 1e-8, precondition_frequency = 10, max_precond_dim = 10000, shampoo_beta = None (-> b2),
                grouped implementation, state in float64; 0-D / 1-D leaves are updated by Adam (4299 parameters in the log
                summary); leaves with >= 2 dimensions that have an axis of length n_species (94) are also updated by Adam
                with the same b1 / b2 / eps instead of SOAP (soap_species_adam any_axis, soap_grouped.py:80-93)
u  <- - lr * f(t) * u,   lr = 0.01,  no weight decay
f(t) = 0.05 + 0.95 t / 2612                                           t < 2612          (2 epochs x 1306 steps, linear warm-up)
f(t) = 0.05 + 0.95 * 1/2 ( 1 + cos( pi (t - 2612) / (T - 2612) ) )    t >= 2612         (cosine to 0.05 at T)
T = 185 * 1306 = 241610 for steps 0..3918, then T = 170 * 1306 = 222020 (restart at epoch 3 with n_epochs 170)
```

NOT DETERMINED: how many parameters fall under the species-table Adam rule; the log summary line ("both 19,915,364, adam 4299") comes from the per-leaf classifier `soap_optax.soap_leaf_summary`, which does not know that rule.

Check: epoch 167 (t = 218102) gives `f = 0.050747`, `lr = 5.0747e-4`, as printed in the log (INFERRED check against log line of epoch 167).

EMA (VERIFIED; `training_utils.py:937-955`): `ema <- d ema + (1 - d) params`, `d = min(0.999, (1 + t)/(10 + t))` at optimiser step t (`ema_warmup`); EMA starts from the initial parameters. Validation, tests and checkpoints' reported metrics use the EMA weights.

Batch (VERIFIED from the log lines 452-456 and `stream_plan.py:1-13`): 3 processes x 4 GPUs = 12 devices; each device gets one "pack" per step, filled greedily in stream order until 2688 nodes, 86016 edges or 192 graphs; the budgets are sized for 48 graphs on average, i.e. nominally 576 graphs per step. INFERRED from the log counts: (3,936,920 - 49,613 dropped) / 6997 steps = about 556 graphs per step. One "epoch" is 1306 steps (virtual; a pass over the data is 6997 steps), so 170 epochs are about 31.7 passes.

Precision (VERIFIED; `train_3bpa.py:1039-1041, 792-798`): model parameters and data float32 (`x64 false`); `jax_enable_x64` is on because of `soap_state_dtype float64`: SOAP statistics in float64 and the ct solve in float64 (section 2).

Other numbers: seed 42 (model init; data stream seed 42; step RNG seed + 1000); 19,919,663 parameters; data: 3,936,920 training structures (215,949,005 atoms), validation 10,000 structures of shard 79, test `neutral_val` 5000 structures (max), evaluated every 16 epochs; neighbour list ASE/matscipy, cutoff 6 A; `avg_n_neighbors 32.584` from the full size table. Epochs: config 185, changed to 170 at the restart after epoch 3 (log line 684 and the note); the log copy ends at epoch 167.

Gauge refit (VERIFIED for the code, `training_utils.py:418-419, 1212, 1216-1220`; `train_3bpa.py:632-638, 1586-1599`):

```
on a fixed seeded subset of the training split (stream.gauge_structures default 20000, seed + 1), with the best-checkpoint EMA weights:
d_n   = E_pred,n - E_ref,n        (total energy residual of structure n, N_n atoms)
delta = - sum_n N_n d_n / sum_n N_n^2          (least squares of sum_n (d_n + N_n delta)^2)
shift <- shift + delta                          (the per-atom constant; forces unchanged)
```

The shifted EMA parameters are written to `<ckpt>.gauge.pkl` and the test / validation metrics are reported again under `gauge`. NOT DETERMINED: the fitted value and whether the run reached epoch 170; `loss_log.txt` ends at epoch 167 and contains no "Training finished" / "Gauge refit" line.

---

## 12. Formal charges q0 and formal spins s0 at training time

**Molecules** (VERIFIED; `graph_fields.py:121-138`): connected components of the graph of bonds `r_ij < bond_factor (r_cov,i + r_cov,j)`, `bond_factor = ct_bond_factor = 1.2`, among the neighbour pairs of the 6 A list (Cordero radii; 1.5 A for an element missing from the table; scipy `connected_components`).

**q0** (VERIFIED; `graph_fields.py:141-180`; called from `stream_io.py:352-363` with `label_keys = [nbo_charges, lowdin_charges]`). Order of sources:

```
1. arrays['formal_charges'] present (per atom, user)           -> v = that array
2. else the first of nbo_charges, lowdin_charges that is present, finite and of length N   -> v = that label
3. else (no per-atom values):
     one molecule (k <= 1) or neutral structure (|Q| < 1e-9)   -> Q on the single molecule / 0 on every molecule
     several molecules and Q != 0                               -> q0_i = Q / N on every atom (no integer split), returned directly
for 1 and 2:  Q_m = round( sum_{i in m} v_i );  if sum_m Q_m != Q, the |Q - sum Q_m| molecules with the largest
              remainder (in the direction of the difference) are moved by +-1   (largest-remainder rule)
q0_i = Q_m(i) / N_m(i)            (the molecule charge spread evenly over its atoms)
```

Note: the rounding and the largest-remainder correction are applied to given `formal_charges` as well (case 1). Case 3 "neutral" gives 0 on every molecule, also for a neutral ion pair without labels. The automatic assignment (`formal_assign.py`) is not part of this training code (not present in `BAM_V2_run_v3`).

**s0** (VERIFIED; `graph_fields.py:183-222`; `stream_io.py:364-372` with `label_key = lowdin_spins`):

```
one molecule                         -> 2S_m = multiplicity - 1
else, parity_m = ( sum_{i in m} Z_i - round(sum_{i in m} q0_i) ) mod 2
  arrays['formal_spins'] present     -> 2S_m = round( sum_{i in m} v_i )
  else lowdin_spins present          -> t = | sum_{i in m} m_i |,  2S_m = max( 2 floor((t - parity)/2 + 1/2) + parity, parity )
  else                               -> 2S_m = parity_m; if 2S - sum_m 2S_m > 0 the rest goes to the largest molecule
s0_i = 2S_m(i) / N_m(i)
```

The sum of `2S_m` need not equal `2S` (two separated radicals of a singlet keep `2S_m = 1` each).

**Inference-only automatic assignment** (`/scratch/t510a13/BAM_V2_wt_thole/bam_omat24/formal_assign.py`; constants VERIFIED at lines 58-60, 116-117):

```
ODD_WEIGHT = 0.4, SHORT_FACTOR = 0.88, PI_FACTOR = 0.97, bond_factor = 1.2 (default of assign_formal_charges)
rule order (module docstring lines 7-32; code order lines 152-376):
1 bonds (d < bond_factor sum r_cov; H keeps its nearest non-metal)   2 metals (bonds cut, common ion charge / default oxidation state)
3 valence (smallest neutral valence >= degree; onium +1 / "ate")      4 pi bonds (matching on bonds with d < PI_FACTOR sum r_cov)
5 hypervalent (P, S, Se, As, halogens: valence + 2)                   6 onium (two unpaired heteroatoms, or d < SHORT_FACTOR; pairing a carbon)
7 leftovers (lone pair -1, missing bond +1, carbanion on metal, radical 0)
8 total charge (_balance: cost + ODD_WEIGHT |odd-electron molecules - target|; else the rest on the largest molecule, flagged)
```
