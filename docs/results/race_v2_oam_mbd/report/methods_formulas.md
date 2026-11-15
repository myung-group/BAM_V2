# RACE v2 (model `RACE`) — formulas as implemented

> File names and line numbers in this note refer to the code the models were trained with: branch `racev2`, commit
> `3545c1d`, package `bam_omat24`. On the merge branch the same files are under `race_v2/` (table in `docs/LAYOUT.md`:
> `race_nnx.py` -> `race_v2/models/race.py`, `train_3bpa.py` -> `race_v2/training/train.py`,
> `training_utils.py` -> `race_v2/training/utils.py`, `soap_grouped.py` -> `race_v2/training/soap_grouped.py`).

Model configuration of the Matbench Discovery runs: MPtrj-only (`mptrj_v1set_sr32_soap_s42`), OMat24 pretraining (`omat24_v1set_sr32_soap_pre`) and MPtrj+sAlex fine-tuning (`oam_v1set_sr32_soap_ft`). The three configs share every model key; they differ only in `atom_energies`, `avg_n_neighbors`, `x64` / `soap_state_dtype`, learning rate / schedule keys, data and `init_params_from` (key-by-key comparison of the three JSON files in `/scratch/t510a13/race_v2/work/runs/omat_stream/`).

Config keys relied on (`oam_v1set_sr32_soap_ft.json`, line numbers of that file):

1. Model: `model RACE` (2), `cutoff 7.0` (3), 89 species Z = 1-83, 89-94 (4-94), `hidden_size 128`, `lmax 3`, `n_layers 5` (186-188), `radial_basis_size 12`, `radial_mlp_size 64`, `radial_mlp_layers 3`, `radial_polynomial_p 6.0`, `mlp_init_scale 4.0` (189-193), `x_mode basis` (206), `cov_mode scalar6` (207), `use_dataset_stats false` (208), `tp_backend cue` (215), `radial_shells true` (217), `update linear` (218), `readout_mode gated` (219), `message_norm avg` (220), `radial_prior {gaussian, sigma 0.224}` (222-225), `residual chebyshev` (226), `layer_scale 0.5` (227), **`x_norm hill`** (228), **`pair_repulsion uff`**, `pair_repulsion_switch 1.25`, `pair_repulsion_switch_width 0.3` (236-238), `e0_mode isolated` (240), `periodic true` (241), `avg_n_neighbors 63.25` (249; 92.0 in the MPtrj-only config), `species_rank 32`, `species_rank_prune true` (283-284). Defaults used (not in the JSON; `train_3bpa.py:649-706`): `x_readout mlp`, `x_readout_norm true`, `f_readout true`, `stratum_gain uniform`, `embed_mode onehot`, `pair_repulsion_scale 1.0`.
2. Loss: `loss_type huber` (205), `energy_weight 1`, `force_weight 1` (200-201), `stress_loss true`, `stress_weight 1` (242-243), `huber_delta_energy 0.05`, `huber_delta_force 0.25`, `huber_delta_stress 0.05` (285-287); `huber_delta 0.01` (202) is overridden by the three per-term values (`train_3bpa.py:973-979`).
3. Optimiser: `optimizer soap`, `soap_b1 0.95`, `soap_b2 0.95`, `soap_precondition_frequency 10` (229-232), `soap_impl grouped` (235), `soap_species_adam true` (247), `soap_state_dtype float64` (288), `learning_rate 0.005` (194), `lr_schedule cosine` (282), `lr_warmup_epochs 1` (248), `lr_cosine_epochs 100000` (289), `lr_cooldown_from_epoch 27` (292), `n_epochs 38` (199), `grad_clip 1.0` (204), `ema_decay 0.999` (203), `x64 false` (221), `seed 42` (216), `init_params_from runs/mbd/snap/omat_pre_final.pkl`, `init_from_ema true` (290-291). `lr_schedule_patience / _factor / _min_scale` (195-197) are plateau keys and are inert with `lr_schedule cosine` (`train_3bpa.py:899-901`).

Conventions. Code root `/scratch/t510a13/BAM_V2_merged/bam_omat24/` (branch `racev2`, HEAD `3545c1d`, `git status --short -- bam_omat24` empty); file references are relative to it. Units: Angstrom, eV. Edge $(i, j)$: receiver $i$, sender $j$, both directions in the list. $n_s = 89$ species, $C = 128$ channels, $K = 32$. VERIFIED = read from the cited lines; INFERRED = derived from them; NOT VERIFIED = not confirmed.

Consistency check of this whole document against the run (INFERRED): the parameter count derived from the structure described in sections 3-5 is exactly the logged `Model parameters: 17,757,551` (table in section 5). The count requires `x_norm hill` (890 parameters), 23 tensor-product paths per channel in layers 1-4, and the pruned tables.

Verification status (things this document does not establish):
- NOT VERIFIED: that the jobs imported this code tree at this commit (only the parameter count and the printed `Model:` line agree).
- NOT VERIFIED: normalisation constants inside the libraries (cuEquivariance channelwise tensor-product path normalisation; e3nn `FunctionalLinear` `path_weight` / `weight_std`; e3nn `tensor_product` and `gate` / `scalar_activation` normalisation; exact spherical-harmonics convention). The libraries were not opened.
- NOT VERIFIED: which checkpoint (best-validation EMA or final-epoch EMA) was used for the Matbench Discovery evaluation; sign/units of the stress labels inside the LMDB sources (the loader passes them through, `stream_io.py:58-61, 96`).
- The MPtrj-only log read here ends at epoch 20 of 80 and the fine-tuning log at epoch 34 of 38 (no "Training finished" line at the time of reading); the pretraining log is complete (58 epochs).

---

## 1. Total energy, forces, stress

**(a) Formulas** (VERIFIED; `race_nnx.py:1723-1732, 2028-2037, 2077-2086, 2200-2204`; `utils.py:315-339, 369-372, 388-414, 440`)

$$E_i = s\,\Big(e^{\rm rep}_i + \sum_{t=0}^{4}\big[e^{\rm node}_{t,i} + e^{x}_{t,i} + e^{f}_{t,i}\big]\Big) + b$$

$$E = \sum_{i} E_i \quad(\text{training target: } E_{\rm DFT} - \textstyle\sum_i E_0[Z_i]),\qquad E_{\rm inference} = \sum_i \big(E_i + E_0[Z_i]\big)$$

$$\mathbf F_i = -\frac{\partial E}{\partial \mathbf x_i}\Big|_{\mathbf h},\qquad \sigma_{uv} = \frac{1}{V}\Big[\sum_a \frac{\partial E}{\partial h_{au}}\,h_{av} + \sum_i \frac{\partial E}{\partial x_{iu}}\,x_{iv}\Big]$$

Voigt vector returned: $(\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{yz},\sigma_{xz},\sigma_{xy})$ = elements `[0,4,8,5,2,1]` of the flattened $3\times3$ (`utils.py:413`), in eV/A$^3$.

**(b) Symbols.** $s$ = `scale`, $b$ = `shift`: trainable scalars, initial values 1 and 0 (`race_nnx.py:1370-1371`; `use_dataset_stats false`, `train_3bpa.py:615-618`; log line `Model init: shift=0.000000 scale=1.000000`). $E_0[Z]$ = config `atom_energies`, fixed, not trainable (`race_nnx.py:1375-1384`). $\mathbf h$ = cell matrix (rows = lattice vectors), $V = |\det \mathbf h|$ (`stream_io.py:95`). $e^{\rm rep}$: section 2; $e^{\rm node}, e^x, e^f$: sections 3-4. `stratum_gain uniform`: every layer energy has gain 1 (`race_nnx.py:2020-2021`).

**(e) Notes.**
- One `jax.grad` of $\sum_i E_i$ with respect to (positions, cell) gives both gradients; the per-graph `cutoff` is passed as an argument but not differentiated (`utils.py:334-339`). NaN gradient entries are set to 0 (`utils.py:388, 398`).
- INFERRED sign convention: the bracket equals $\partial E/\partial\varepsilon_{uv}$ for a homogeneous strain $\mathbf x \to \mathbf x(1+\varepsilon)$, $\mathbf h \to \mathbf h(1+\varepsilon)$, so $\sigma = +\frac{1}{V}\,\partial E/\partial \varepsilon$ (tension positive, the ASE convention named in the docstring `train_3bpa.py:169-170`). The $3\times3$ matrix is not symmetrised; the upper-triangle elements are taken.
- $E$ depends on the cell only through the image shifts of the edge vectors (section 2).
- Training uses `l_train=True` (`train_3bpa.py:638`): $E_0$ is not added by the model and is subtracted from the labels by the loader (`stream_io.py:85, 89`); with `l_train False`, $E_0[Z_i]$ is added with `stop_gradient` (`utils.py:369-372`).
- $s$ multiplies the fixed UFF repulsion too (`race_nnx.py:1898, 2077-2086`).
- The MPtrj-only config has a different `atom_energies` table than the two OAM configs (first entry -3.6687 vs -3.0698 eV).

---

## 2. Inputs: graph, edge features, embedding

**(a) Formulas** (VERIFIED; `utils.py:17-58, 205-256`; `race_nnx.py:1740-1751, 1759-1771, 1787-1802, 1821-1825, 1833-1834, 1451-1473, 819-820`; `neighbors.py:56-60`; `stream_io.py:83-94`)

$$\mathbf r_{ij} = \mathbf x_j - \mathbf x_i - \mathbf s_{ij}\,\mathbf h,\qquad r = |\mathbf r_{ij}|,\qquad r < r_c = 7\ \text{A}$$

$$B_n(r) = \frac{2}{r_c}\,\frac{\sin(n\pi r/r_c)}{r},\qquad n = 1,\dots,12$$

$$f_{\rm cut}(r) = 1 - \tfrac{(p+1)(p+2)}{2}u^{p} + p(p+2)\,u^{p+1} - \tfrac{p(p+1)}{2}u^{p+2}\;\overset{p=6}{=}\;1 - 28u^6 + 48u^7 - 21u^8,\qquad u = r/r_c\ \ (0 \text{ for } u \ge 1)$$

$$\beta_n(r) = B_n(r)\,f_{\rm cut}(r)\,e^{-(0.224\,n)^2}\qquad\text{(Bessel channels with the fixed radial-prior rescaling)}$$

$$\varsigma_{ab}(r) = \exp\!\Big(-\frac{r}{\rho_a(Z_i) + \rho_b(Z_j)}\Big)\,f_{\rm cut}(r),\qquad a, b \in \{s, p, d, f\}\quad(16 \text{ channels; } 0 \text{ if either shell is absent})$$

$$\mathbf R^{\rm in}_{ij} = [\beta_1,\dots,\beta_{12},\ \varsigma_{ss},\dots,\varsigma_{ff}] \in \mathbb R^{28},\qquad Y(\hat{\mathbf r}_{ij}):\ \text{real spherical harmonics } l = 0..3\ (16 \text{ components})$$

$$\mathbf e_i = W_{\rm emb}\,\mathrm{onehot}(Z_i)\ \in 128{\times}0e$$

UFF pair repulsion, the lowest energy term (VERIFIED; `race_nnx.py:1855-1859, 1876-1891, 1897`; table `race_nnx.py:275-378`; radii `train_3bpa.py:252-264, 674`):

$$e_{ij} = D_{ij}\,\frac{6}{\zeta_{ij}-6}\,\exp\!\Big[\zeta_{ij}\Big(1 - \frac{r}{x_{ij}}\Big)\Big]\,f_{\rm cut}(r)\;\mathrm{sigmoid}\!\Big(\frac{r - 1.25\,(R_i + R_j)}{0.3}\Big),\qquad e^{\rm rep}_i = \frac12\sum_{j\in N(i)} e_{ij}$$

with $x_{ij} = \sqrt{x_i x_j}$, $D_{ij} = \sqrt{D_i D_j}$, $\zeta_{ij} = \sqrt{\zeta_i\zeta_j}$.

**(b) Symbols.** $\mathbf s_{ij}$ = stored edge attribute `Sij` = minus the ASE cell shift (`stream_io.py:92`). $\rho_a(Z)$: fixed, $\langle r\rangle$ of the outermost occupied shell of angular momentum $a$ from the LDA atom table `atomic_shells.json` times 0.529177 A/Bohr, 0 = absent (`train_3bpa.py:267-269, 315-322, 690-691`; solver `atomic_shells.py:177-191`; e.g. Fe: s 1.587, p 0.456, d 0.599 A). $x_I, D_I, \zeta_I$: UFF van der Waals constants ($D$ in eV); $R$: Cordero covalent radii; all fixed. $W_{\rm emb}$: learned free $(89, 128)$ matrix (`race_nnx.py:1434-1438`).

**(c) Values.** $r_c = 7.0$, 12 Bessel functions, $p = 6$, prior factors $e^{-0.050176\,n^2}$ (0.951 for $n = 1$, $7.3\times10^{-4}$ for $n = 12$), shell channels not rescaled (`race_nnx.py:1468, 1471`). $n_{\rm avg}$ = `avg_n_neighbors` = 63.25 (OAM runs) or 92.0 (MPtrj-only): a fixed constant from the config, not computed from the data (`race_nnx.py:1375`).

**(e) Notes.**
- Neighbour list: ASE `neighbor_list("ijS")` with the config cutoff (log: `neighbour backend ase`). In the periodic branch the Bessel functions and their envelope use the per-graph global `cutoff` written by the loader (= 7.0, clamped to $\ge 1$; `utils.py:225-238`, `stream_io.py:94`); the shell envelopes and the UFF term use the model attribute (7.0; `race_nnx.py:1771`).
- Spherical harmonics: `e3nn.spherical_harmonics(s2_irreps(3), r, normalize=True, normalization="component")` (`utils.py:251-256`): unit-vector argument; "component" is the e3nn convention $\sum_m Y_{lm}^2 = 2l+1$ (definition NOT VERIFIED in the library).
- Padding edges ($r = 0$) get zero radial and angular features (`race_nnx.py:1821-1825`); the UFF term is masked for $r = 0$ (`race_nnx.py:1891`).
- $n_{\rm avg}$ enters in two places: the neighbour sum is divided by $n_{\rm avg}$ (`message_norm avg`, `race_nnx.py:816-818, 837`), and the x / f readout inputs by $\sqrt{n_{\rm avg}}$ (`race_nnx.py:481`).
- OFF in our config: `radial_species_scale`, `radial_slater`, `radial_shell_scale`, `radial_gaussians`, `angular_prior`, `layer_cutoff`, `layer_lmax`, `shell_sk_gate`, `radial_species_mix`, `embed_mode descriptor/both`, `pair_term`, `bond_order` (`train_3bpa.py:649-706` defaults; `Model:` log line).
- The constructor docstring says `radial_shells` is "Non-periodic only" (`race_nnx.py:1125-1128`); the code does handle periodic cells (image shifts in `r_norm`, `race_nnx.py:1763-1768, 1297-1298`).

---

## 3. One message-passing layer (x_mode basis, residual chebyshev, layer_scale 0.5, tp_backend cue)

**(a) Formulas.** $\mathbf h^{(t)}$ = input of layer $t$; $\mathbf h^{(0)}_i = \mathbf e_i$ ($128{\times}0e$), $\mathbf h^{(t\ge1)} \in 128{\times}(0e + 1o + 2e + 3o)$ (dimension 2048).

Atomic basis, computed once with the layer-0 `linear_1` and radial MLP (VERIFIED; `race_nnx.py:1909-1918, 798-845`):

$$\mathbf A_i = \frac{1}{n_{\rm avg}}\sum_{j\in N(i)} \mathbf R^{(0)}(\mathbf R^{\rm in}_{ij})\odot\big[W_1^{(0)}\mathbf e_j \otimes Y(\hat{\mathbf r}_{ij})\big]$$

Layer $t = 0,\dots,4$ (VERIFIED; `race_nnx.py:1953-1961, 228-237, 857-914, 922-923, 931-939, 1728-1729`):

$$\mathbf m^{(t)}_i = \frac{1}{n_{\rm avg}}\sum_{j\in N(i)}\ \sum_{\rm paths} R^{(t)}_{\rm path}(\mathbf R^{\rm in}_{ij})\,\big[W_1^{(t)}\mathbf h^{(t)}_j \otimes Y(\hat{\mathbf r}_{ij})\big]_{\rm path}$$

$$\mathbf x^{(t)}_i = W_x^{(t)}\mathbf A_i,\qquad q_i = \tfrac{1}{2048}\textstyle\sum_c \big(x^{(t)}_{i,c}\big)^2 + 10^{-6},\qquad \hat{\mathbf x}^{(t)}_i = \frac{\mathbf x^{(t)}_i}{\sqrt{q_i}}\; 2\,\mathrm{sigmoid}\!\big(\tfrac12 a_{t,Z_i}\ln q_i + b_{t,Z_i}\big)$$

$$\mathbf p^{(t)}_i = W_2^{(t)}\Big[\big(W_m^{(t)}\mathbf m^{(t)}_i\big)\otimes_{\rm channelwise}\hat{\mathbf x}^{(t)}_i\Big],\qquad \boldsymbol\delta^{(t)}_i = \boldsymbol\gamma^{(t)}\odot\Big(\mathbf p^{(t)}_i + S^{(t)}[Z_i]\,\mathbf h^{(t)}_i\Big)$$

$$\mathbf h^{(t+1)} = \boldsymbol\delta^{(t)}\ \ (t = 0, 1),\qquad \mathbf h^{(t+1)} = 2\,\boldsymbol\delta^{(t)} - \mathbf h^{(t-1)}\ \ (t = 2, 3, 4)$$

**(b) Symbols.** $W_1^{(t)}$: equivariant linear map input irreps $\to$ input irreps (`race_nnx.py:621-625`). $\mathbf R^{(t)}$: radial MLP, one weight per path and channel (`race_nnx.py:627-635`). $W_m^{(t)}$: linear map from the message irreps to the hidden irreps (`649-654`). $\otimes_{\rm channelwise}$: for each of the 128 channels $u$, all Clebsch-Gordan paths $(l_1, l_2)\to l_3$ of $m_u \otimes \hat x_u$ with output in $\{0e, 1o, 2e, 3o\}$ (`901-910`, `655-659`). $W_2^{(t)}$: linear map from the product irreps to the hidden irreps (`744-749`). $S^{(t)}[Z]$: per-element skip matrix, rank-32 factorised (section 5; `753-762`). $\boldsymbol\gamma^{(t)}$: learned per-channel LayerScale, 512 values, initial value 0.5, layers $t \ge 1$ only; layer 0 has no $\gamma$ (`582-583, 1505`). $W_x^{(t)}$: per-layer bias-free linear map of $\mathbf A_i$ (`1576-1589`). $a_{t,Z}, b_{t,Z}$: learned, one pair per layer and element, initial value 0 (`x_hill`, shape $(5, 2, 89)$, `1633-1635`).

**(c) Values.** Radial MLP: $28 \to 64 \to 64 \to 64 \to n_w$, SiLU, no bias; $n_w = 4\cdot128 = 512$ in layer 0 and $23\cdot128 = 2944$ in layers 1-4 (INFERRED: 23 = number of triples $(l_1, l_2, l_3) \le 3$ with the triangle rule and $l_1+l_2+l_3$ even; confirmed by the parameter count). Initial weights $\mathcal N(0, 4/n_{\rm in})$ for the first three layers, $\mathcal N(0, 1/n_{\rm in})$ for the last (`utils.py:121-124, 160`; `mlp_init_scale 4`). The channelwise product $m \otimes \hat x$ also has 23 paths per channel.

**(e) Notes — which normalisations are ON.**
- ON: `x_norm hill` (`_hill_normalize`, `race_nnx.py:228-237, 1958-1961`). With $a = b = 0$ (initial state) the gate is 1 and $\hat{\mathbf x}$ is $\mathbf x$ divided by its per-node RMS over all 2048 components; in general $|\hat x_{\rm rms}| \le 2$. This key is not printed in the `Model:` log line; it is in the JSON (line 228) and is passed by `train_3bpa.py:701`.
- OFF: spectral normalisation (`_spectral_normalize`, `race_nnx.py:240-260`), `x_norm rms`, `feature_norm`, `tie_operator`.
- OFF: layer norm of the messages. `message_norm avg`: no normalisation of $\mathbf h_j$, sum divided by $n_{\rm avg}$ (`race_nnx.py:812-818`). The separated layer norm is used only inside the x / f readouts (section 4).
- OFF: gate non-linearity on the features (`update linear`, `race_nnx.py:916-921`): $\boldsymbol\delta$ is linear in $(\mathbf m \otimes \hat{\mathbf x},\ \mathbf h)$; the non-linearities are the Hill gate, the radial MLP and the readouts.
- Chebyshev recurrence: `prev_features` is the input of layer $t-1$, passed for $t \ge 2$ only (`race_nnx.py:1728-1729, 935-938`); stochastic depth is OFF (`drop_scale None`).
- `tp_backend cue`: gather, tensor product and scatter of the message are one fused cuEquivariance "channelwise tensor product" kernel (`race_nnx.py:99-126, 832-837`); the node-level product $m \otimes \hat x$ uses the e3nn tensor product with the channel moved to a batch axis (`907-910`).
- Layer 0 recomputes its message inside the layer call with the same inputs as $\mathbf A_i$, so $\mathbf m^{(0)}_i = \mathbf A_i$ (INFERRED from `race_nnx.py:873, 1916-1918`).
- `in_layer_order 1`: no symmetric contraction / in-layer square (OFF, `race_nnx.py:676-678`).

---

## 4. Readout

**(a) Formulas** (VERIFIED; `race_nnx.py:768-796, 953-966, 425-496, 1526-1548, 1968, 2019`; `utils.py:61-106`)

$$e^{\rm node}_{t,i} = \mathbf w_t\cdot\mathrm{silu}\big(W_{r1}^{(t)}\mathbf h^{(t+1)}_i + \mathbf b_t\big) + b'_t,\qquad W_{r1}^{(t)}: \text{hidden} \to 64{\times}0e$$

$$e^{x}_{t,i} = \mathrm{MLP}^{x}_t[Z_i]\Big(\frac{\mathrm{LN}(\hat{\mathbf x}^{(t)}_i)}{\sqrt{n_{\rm avg}}}\Big),\qquad e^{f}_{t,i} = \mathrm{MLP}^{f}_t[Z_i]\Big(\frac{\mathrm{LN}(\mathbf h^{(t)}_i)}{\sqrt{n_{\rm avg}}}\Big)$$

$$\mathrm{MLP}[Z](\mathbf v) = \mathbf w_3\cdot\mathrm{silu}\big(W_2\,(W_1[Z]\,\mathbf v) + \mathbf b_2\big) + b_3,\qquad W_1[Z]: \mathbf v \to 128{\times}0e\ (\text{no bias}),\quad W_2: 128 \to 64$$

$$\mathrm{LN}:\quad v^{0e}_c \to \frac{v^{0e}_c - \bar v}{\sqrt{\frac{1}{128}\sum_c (v^{0e}_c - \bar v)^2 + 10^{-6}}},\qquad \mathbf v^{l>0} \to \frac{\mathbf v^{l>0}}{\sqrt{\text{mean over all } l>0 \text{ components of } v^2 + 10^{-6}}}$$

**(e) Notes.**
- `readout_mode gated`: the first Linear of the node readout keeps only $0e$ outputs (`race_nnx.py:768-770`), so the "gate" is a SiLU on 64 scalars (`e3nn.gate`; its activation normalisation constant is NOT VERIFIED). `readout_species False`: the last Linear is shared by all elements.
- Per-layer energies that exist: $e^{\rm node}$ (reads the layer output $\mathbf h^{(t+1)}$), $e^x$ (reads the Hill-normalised $\hat{\mathbf x}^{(t)}$) and $e^f$ (reads the layer input $\mathbf h^{(t)}$), for each of the 5 layers, plus the UFF term: 16 terms in total (`race_nnx.py:2028-2037`). Only the $0e$ part of $\hat{\mathbf x}$ and $\mathbf h^{(t)}$ enters $e^x$, $e^f$.
- INFERRED: $e^f_{0,i}$ depends on $Z_i$ only (its input is the species embedding), i.e. it is a learned per-element constant.
- The layer-norm epsilon is fixed at $10^{-6}$ (`utils.py:73`); there is no `x_readout_norm_eps` option in this code.
- `cov_mode scalar6` concerns only three uncertainty heads on the final features $\mathbf h^{(5)}$: `frc_logvar` (hidden $\to 6{\times}0e$: softplus + 0.04 on three diagonal entries and three off-diagonal entries of a $3\times3$ Cholesky factor), `energy_logvar` ($\to 1{\times}0e$) and `stress_logvar` ($\to 6{\times}0e$), all softplus + 0.04 (`race_nnx.py:1667-1699, 2088-2111`). They do not contribute to the energy. They are read only by `loss_type nll / gauss_nll` (`training_utils.py:284-287`); with `loss_type huber` they never enter the loss, so their 1,664 weights receive zero gradient.
- OFF: `noise_head`, `dens`, `force_encoding`, `strata_resum`, `x_readout linear/off`.

---

## 5. Species-rank factorisation (`species_rank 32`, `species_rank_prune true`)

**(a) Formula** (VERIFIED; `linear_nnx.py:224-235, 297-316, 352-370`)

$$W[Z] = K^{-1/2}\sum_{k=1}^{K} c[Z, k]\,B[k],\qquad c \in \mathbb R^{n_s\times K},\quad B \in \mathbb R^{K\times C_{\rm in}\times C_{\rm out}}\ \text{per irrep block}$$

Initialisation: $c[Z,k] \sim \mathcal N(0, 1)$; $B[k] \sim$ `weight_std` $\cdot\,\mathcal N(0,1)$ with the e3nn `weight_std` of the block (`linear_nnx.py:307-316`), so each $W[Z]$ has the variance of a free weight. Biases of an indexed Linear would stay free tables; no indexed Linear has a bias in our config.

**Which tables** (VERIFIED; `race_nnx.py:1405-1430, 1516-1517, 753-762, 446-455, 1536, 1547`):

| table | blocks | rank |
|---|---|---|
| skip $S^{(t)}[Z]$, layer 0 | $0e \to 0e$ | 32 |
| skip, layers 1-3 | 4 blocks ($0e, 1o, 2e, 3o$) | 32 |
| skip, layer 4 | $0e$: 32; $1o, 2e, 3o$: **1** (pruned, `species_rank_lpos`) | 32 / 1 |
| first Linear $W_1[Z]$ of the x readout, 5 layers | $0e \to 0e$ | 32 |
| first Linear $W_1[Z]$ of the f readout, 5 layers | $0e \to 0e$ | 32 |
| `x_linear` (the per-species x of `x_mode species`; unused with `x_mode basis`) | $0e \to 0e$ | **1** (pruned) |

Not factorised: the species embedding $W_{\rm emb}$ (a plain Linear on the one-hot vector) and `x_hill`.

**Parameter counts** (INFERRED from the constructor; sum equals the logged 17,757,551). One rank-32 block: $32\cdot128^2 + 89\cdot32 = 527{,}136$ (a free table would be $89\cdot128^2 = 1{,}458{,}176$); one rank-1 block: $128^2 + 89 = 16{,}473$.

| group | count |
|---|---|
| shift, scale | 2 |
| species embedding $89\times128$ | 11,392 |
| `x_linear` (rank 1) | 16,473 |
| `x_hill` $5\times2\times89$ | 890 |
| uncertainty heads $768 + 128 + 768$ | 1,664 |
| layer 0: $W_1$ 16,384 + radial MLP 42,752 + $W_m$ 65,536 + $W_2$ 376,832 + skip 527,136 + node readout 8,321 | 1,036,961 |
| layers 1-3, each: $\gamma$ 512 + $W_1$ 65,536 + radial MLP 198,400 + $W_m$ 376,832 + $W_2$ 376,832 + skip 2,108,544 + node readout 8,321 | 3 x 3,134,977 |
| layer 4: as layers 1-3 with skip 576,555 | 1,602,988 |
| x readouts, 5 x (527,136 + 8,256 + 65) | 2,677,285 |
| f readouts, 5 x (527,136 + 8,256 + 65) | 2,677,285 |
| x projections $W_x^{(t)}$, 5 x 65,536 | 327,680 |
| **total** | **17,757,551** |

The species-indexed tables hold 12,717,156 parameters (72 %): 24 rank-32 blocks and 4 rank-1 blocks.

**(e) Notes.**
- `species_rank` must differ from $n_s$ (`race_nnx.py:1413-1415`), because the optimiser identifies element tables by a leading dimension equal to $n_s$ (section 7).
- Pruning rationale in the code: `x_linear` and the $l > 0$ skip blocks of the last layer cannot reach the energy (`race_nnx.py:1409-1412`); `x_linear` is still evaluated (`race_nnx.py:1851`) but its result is replaced in every layer (`1955`).
- `species_rank_prune` is not valid with `dens` / `noise_head` (`race_nnx.py:1418-1419`); both are OFF.

---

## 6. Loss

**(a) Per-step loss** (VERIFIED; `training_utils.py:57-60, 225-240, 259-279, 351-356, 443-469, 750-753, 778-784`). $G$, $N$ = real graphs and atoms of the whole step (all devices); $N_g$ = atoms of graph $g$.

$$h_d(x) = \tfrac12 x^2\ \ (|x| \le d),\qquad h_d(x) = d\,\big(|x| - \tfrac12 d\big)\ \ (|x| > d)$$

$$\Delta E_g = \frac{E^{\rm pred}_g - E^{\rm ref}_g}{N_g},\qquad L_E = \frac1G\sum_g h_{0.05}\big(5\tanh(\Delta E_g/5)\big)$$

$$\Delta\tilde{\mathbf F}_i = \Delta\mathbf F_i\,\frac{10\tanh(n_i/10)}{n_i},\quad n_i = \sqrt{|\Delta\mathbf F_i|^2 + 10^{-24}},\qquad L_F = \frac{1}{3N}\sum_i\sum_{c\in\{x,y,z\}} h_{0.25}\big(\Delta\tilde F_{i,c}\big)$$

$$L_S = \frac1G\sum_g\sum_{v=1}^{6} h_{0.05}\big(0.5\tanh(\Delta\sigma_{g,v}/0.5)\big),\qquad L = 1\cdot L_E + 1\cdot L_F + 1\cdot L_S$$

**(c) Values.** Deltas 0.05 eV/atom, 0.25 eV/A (per Cartesian component), 0.05 eV/A$^3$ (per Voigt component); weights 1 / 1 / 1; `energy_loss_per atom` (default). No auxiliary terms: deep supervision, stratum decorrelation, noisy nodes, DeNS, Taylor consistency, `radial_l2`, charge loss are all OFF (`train_3bpa.py:1144-1166` with the config defaults; `aux` column of the log is 0).

**(e) Notes.**
- The tanh clipping of the residuals (5 eV/atom, 10 eV/A on the force-error norm, 0.5 eV/A$^3$) is always on: `clip_residuals=True` is the default and no caller changes it (`training_utils.py:202, 259-279, 445, 877`). It is part of the training and of the validation loss; reported MAE / RMSE use raw residuals (`training_utils.py:359-373`).
- $L_S$ is a sum over the 6 Voigt components divided by the number of graphs, not a mean over components (`training_utils.py:355, 463`); $L_F$ is a mean over $3N$ components.
- Multi-GPU: each device divides its sums by the global $G$, $N$ and the per-device losses and gradients are summed (`psum`), so the step equals the single-device step on the union batch (`training_utils.py:750-753, 778-784`).
- Validation loss = the same $L_E + L_F + L_S$ (same deltas, same clipping, weights of the current epoch), accumulated over the validation split, evaluated with the EMA parameters after every (virtual) epoch (`training_utils.py:872-878, 1035-1039`; `train_3bpa.py:1408-1409`). Validation split: MPtrj `val`, fixed subset of 20,000 structures (config `stream.valid.max_structures`).
- Best-checkpoint rule: the checkpoint file is overwritten whenever `val_loss < best_val_loss` (`train_3bpa.py:1440-1456`); `<ckpt>.last.pkl` is written every epoch (`ckpt_last_every 1`, `1469-1479`). At the end the best checkpoint's EMA parameters are evaluated, and with `report_final_ema true` also the final-epoch EMA parameters (`1489-1520`).
- The `LOSS` column of the training log is $L_E + L_F$ without the stress term; the three validation terms are printed on the extra line `valid loss terms` (`train_3bpa.py:1401-1403, 1419-1425`).
- Labels: energy $E_{\rm DFT} - \sum_i E_0[Z_i]$ (float32 of a float64 difference), forces and the 6-component stress as stored; a $3\times3$ stress is converted with the index pairs $(xx, yy, zz, yz, xz, xy)$ (`stream_io.py:48-49, 58-61, 85-97`).

---

## 7. Training procedure

**Optimiser chain** (VERIFIED; `train_3bpa.py:732-735, 750-786, 837-866, 899-901`; `training_utils.py:794-798`; `soap_grouped.py:199-203, 215`; log line `SOAP: {'b1': 0.95, 'b2': 0.95, 'precondition_frequency': 10, 'max_precond_dim': 10000, 'shampoo_beta': None}`):

$$g \leftarrow \textstyle\sum_{\rm devices} \nabla L,\qquad g \leftarrow g\cdot\min\!\big(1,\ 1/\lVert g\rVert_2\big),\qquad u = \mathrm{SOAP}(g),\qquad \theta \leftarrow \theta - \eta\,f(t)\,u$$

SOAP per matrix leaf $G$ (`soap_grouped.py:320-338, 351-402`), $b_1 = b_2 = 0.95$, $\epsilon = 10^{-8}$, Kronecker-factor decay $\beta_s = b_2$ (`shampoo_beta None`):

$$L \leftarrow \beta_s L + (1-\beta_s)\,GG^{\top},\quad R \leftarrow \beta_s R + (1-\beta_s)\,G^{\top}G,\quad G' = Q_L^{\top} G\,Q_R$$

$$M \leftarrow b_1 M + (1-b_1)\,G',\quad V \leftarrow b_2 V + (1-b_2)\,G'^2,\quad u = \frac{\sqrt{1-b_2^{\,t}}}{1-b_1^{\,t}}\;Q_L\,\frac{M}{\sqrt V + \epsilon}\,Q_R^{\top}$$

- First step: $Q_L, Q_R$ = eigenvectors of $L, R$ and the update is zero. Every 10 steps ($t \bmod 10 = 0$): one power-iteration step $Q \leftarrow \mathrm{QR}(F\,Q)$ per factor (columns sorted by $\mathrm{diag}(Q^{\top} F Q)$, Householder QR, the default `soap_orth`), $M$ re-expressed in the new basis.
- `max_precond_dim 10000`: no leaf side exceeds it (log: `'both': 17754526, 'left': 0, 'right': 0`). Leaves with 3 or more dimensions are treated as independent matrices per leading index (`soap_optax.py:53-58, 194`).
- Adam instead of SOAP (same $b_1$, $b_2$, $\epsilon$, bias-corrected, `soap_grouped.py:312-317`): (i) all 0-D and 1-D leaves (`soap_optax.py:189-190`): 3,025 parameters, as in the log (`'adam': 3025` = shift, scale, 4 x 512 LayerScale, all readout biases); (ii) with `soap_species_adam true`, every leaf with at least 2 dimensions whose **leading** dimension is $n_s = 89$ (`soap_grouped.py:73-77, 92-94`; `train_3bpa.py:765-771`). INFERRED: these are the 28 coefficient tables $c$ (68,708 parameters) and the species embedding $(89, 128)$ (11,392), so 83,125 parameters use Adam and 17,674,426 use SOAP. The log summary does not know rule (ii) (`train_3bpa.py:783`).
- `x_hill` $(5, 2, 89)$ has leading dimension 5, so it is updated by SOAP as five $2\times89$ matrices whose species side is rotated (`soap_optax.py:63-69`); the `any_axis` variant that would move it to Adam is not used. `soap_grouped.py:86-90` warns that this rotation lets the entries of elements absent from the data drift. NOT VERIFIED: whether all 89 elements occur in each training set.
- No weight decay (`weight_decay=0.0` in the log). The plateau scheduler is inert (factor 1 always, `train_3bpa.py:899-901`).

**Learning-rate factor** (VERIFIED; `train_3bpa.py:839-861`; optax `warmup_cosine_decay_schedule`, formula checked against the logged values below). $n$ = steps per epoch, $W$ = `lr_warmup_epochs` $\cdot n$, $T_c$ = `lr_cosine_epochs` $\cdot n$ (default `n_epochs` $\cdot n$), $T$ = `n_epochs` $\cdot n$, $s_0$ = `lr_cooldown_from_epoch` $\cdot n$, $\mu$ = `lr_cosine_min_scale` = 0.01 (default; the key is not in our configs):

$$f_{\cos}(t) = \mu + (1-\mu)\,\frac{t}{W}\ \ (t < W),\qquad f_{\cos}(t) = \mu + (1-\mu)\,\tfrac12\Big[1 + \cos\Big(\pi\,\frac{t - W}{T_c - W}\Big)\Big]\ \ (W \le t \le T_c)$$

$$f(t) = f_{\cos}(t)\ \ (t < s_0),\qquad f(t) = f_{\cos}(s_0) + \big(\mu - f_{\cos}(s_0)\big)\,\mathrm{clip}\Big(\frac{t - s_0}{T - s_0},\,0,\,1\Big)\ \ (t \ge s_0)$$

| run | $\eta$ | $n$ | $W$ | $T_c$ | $s_0$ | $T$ | realised shape |
|---|---|---|---|---|---|---|---|
| MPtrj-only | 0.01 | 6,965 (one data epoch) | 13,930 | $= T$ | none | 557,200 (80 epochs) | warm-up, cosine to 0.01 |
| OMat24 pretraining | 0.01 | 5,000 (virtual) | 10,000 | $5\times10^8$ | 200,000 | 290,000 (58 epochs) | warm-up, constant, linear decay to 0.01 |
| OAM fine-tuning | 0.005 | 5,000 (virtual) | 5,000 | $5\times10^8$ | 135,000 | 190,000 (38 epochs) | warm-up, constant, linear decay to 0.01 |

Checks against the logs (INFERRED): MPtrj-only, $t = 20{,}895$: $f = 0.99960$ (logged `factor 0.99960`); epoch 20, $t = 139{,}300$: $\eta f = 0.0087552$ (logged 0.0087551). Fine-tuning, epoch 34, $t = 170{,}000$: $f = 1 - 0.99\cdot35/55 = 0.37$, $\eta f = 0.00185$ (logged 0.00185). Earlier launches of the same runs used other `n_epochs` / cool-down values (fine-tuning: 44 epochs with cool-down from 31; pretraining: cool-down from epoch 52; MPtrj-only: 40 epochs for the first 2 epochs), each replaced by a restart before the affected part of the schedule was reached, except that the first two MPtrj-only epochs are warm-up, which does not depend on $T$.

**EMA** (VERIFIED; `training_utils.py:807-810`; `train_3bpa.py:982, 1138, 1181-1183`): $\bar\theta \leftarrow 0.999\,\bar\theta + 0.001\,\theta$ after every optimiser step, constant decay, **no warm-up**; $\bar\theta$ starts from the initial parameters (fine-tuning: from the EMA parameters of the pretraining checkpoint, epoch 58, with a fresh optimiser state; log line `Initialised parameters from runs/mbd/snap/omat_pre_final.pkl (EMA params, epoch 58)`). Validation and checkpoint selection use $\bar\theta$.

**Precision** (VERIFIED; `train_3bpa.py:709-714, 952-954, 1033-1037`; `soap_grouped.py:208-213, 224-232`): OAM runs (`x64 false`, `soap_state_dtype float64`): parameters, data and model arithmetic float32; `jax_enable_x64` is switched on only so that the SOAP moments, Kronecker factors and eigenbases are float64 (log: SOAP state 770 MiB). MPtrj-only run (`x64 true`, no `soap_state_dtype`): parameters and data cast to float64, SOAP state in the parameter dtype (log: `float64: parameters and data cast to float64`, SOAP state 1541 MiB).

**Other numbers** (from the logs). Seed 42 (model initialisation; step RNG seed + 1000). Devices: MPtrj-only 16 GPUs, 8 graphs per pack; pretraining 20 then 24 GPUs; fine-tuning 24 GPUs, 32 graphs per pack on average (pack limits 704 nodes, 65,536 edges, 128 graphs). The log line `8 graphs per device, 192 per step` is computed from `batch_size` and is not the stream pack size; INFERRED from the log counts, the fine-tuning step holds about 725 graphs (22,400,585 structures / 30,887 steps per data epoch). Fine-tuning data per data epoch: MPtrj train x 8 (1,498,053 structures) + sAlex train x 1 (10,416,161). `gauge_refit false`: no shift refit after training.

---

## 8. Differences from the LR-POLAR document

Reference: `/scratch/t510a13/BAM_V2_wt_thole/docs/results_omol4m_v3/report/v3_methods_formulas.md` (model `RACE_POLAR`, lines 1-121 and 386-475 read). Same in both: 5 layers, 128 channels, $l_{\max} = 3$, 12 Bessel functions, envelope $p = 6$, shell channels, Gaussian radial prior $\sigma = 0.224$, radial MLP 64 x 3, `x_mode basis`, `update linear`, `message_norm avg`, gated readouts with x and f readouts, Chebyshev residual, LayerScale 0.5, `tp_backend cue`, `cov_mode scalar6`, `species_rank 32`, UFF repulsion with switch 1.25 / 0.3, Huber deltas 0.05 / 0.25, SOAP $b_1 = b_2 = 0.95$ with frequency 10, gradient clip 1.0, EMA decay 0.999.

| item | LR-POLAR document | this model |
|---|---|---|
| model class | `RACE_POLAR` (backbone + Qeq charges, charge transfer, spin, Thole electrostatics, dispersion) | `RACE`: local terms only; **no** long-range, charge, spin, polarisation or dispersion terms; no charge / spin / task conditioning of $\mathbf h$ (there $\mathbf h^{(0)} = \mathbf e + z_{cs}(1+G_0) + $ task row) |
| cutoff | 6.0 A | 7.0 A |
| radial basis | same formulas with $r_c = 6$ | same formulas with $r_c = 7$ |
| `avg_n_neighbors` | 32.584 | 63.25 (OAM), 92.0 (MPtrj-only) |
| x normalisation | `x_norm spectral`: every block of $W_x$ divided by its largest singular value, no per-node rescaling | `x_norm hill`: per-node RMS normalisation times a learned bounded gate (890 parameters); spectral normalisation OFF |
| x readout layer-norm epsilon | `x_readout_norm_eps 0.003` (epsilon $9\times10^{-6}$) | fixed $10^{-6}$; the option does not exist in this code |
| species tables | `species_rank_prune false` (unused `x_linear` kept at rank 32) | `species_rank_prune true`: `x_linear` and the $l>0$ skip blocks of layer 4 at rank 1 |
| species count | 94 (stated for the optimiser rule) | 89 |
| periodicity | `periodic false`, molecules, no stress | `periodic true`, cell shifts, stress by autodiff |
| energy assembly | local part + Qeq + spin + Thole + dispersion; `scale` not applied to those | $s\cdot(\text{UFF} + 15 \text{ layer terms}) + b$ only |
| loss | $10\,L_E + L_F$ + charge, dipole, spin terms; `stress_loss false` | $L_E + L_F + L_S$, weights 1 / 1 / 1; no auxiliary terms |
| Huber deltas | 0.05, 0.25 (stress delta unused) | 0.05, 0.25, 0.05 (stress used) |
| validation loss | $10\,L_E + L_F$ | $L_E + L_F + L_S$ |
| species Adam rule | `soap_species_adam any_axis` (species size on any axis) | `soap_species_adam true` (leading axis only; `x_hill` stays on SOAP) |
| EMA | `ema_warmup true`: $d = \min(0.999, (1+t)/(10+t))$ | constant 0.999, no warm-up (no such option in this code) |
| LR schedule | lr 0.01, 2 warm-up epochs, cosine to `lr_cosine_min_scale 0.05` | lr 0.01 / 0.01 / 0.005; minimum factor 0.01 (default); MPtrj-only: cosine; OAM runs: warm-up, constant, linear cool-down (`lr_cosine_epochs`, `lr_cooldown_from_epoch`) |
| precision | float32 model + float64 SOAP state | the same for the OAM runs; full float64 for the MPtrj-only run |
| gauge refit | `gauge_refit true` | `gauge_refit false` |
| parameters | 19,919,663 | 17,757,551 |
| code | `/scratch/t510a13/BAM_V2_run_v3/bam_omat24`, HEAD `d90bdc5` | `/scratch/t510a13/BAM_V2_merged/bam_omat24`, branch `racev2`, HEAD `3545c1d` |
