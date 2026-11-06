# RACEv2 to PyTorch inference port: scope (2026-11-06)

Read-only survey. No code was run (no JAX or torch imports). Sources read:

- `bam_omat24/race_nnx.py`, `bam_omat24/linear_nnx.py`, `bam_omat24/utils.py`, `bam_omat24/train_3bpa.py` (`build_model`), `bam_omat24/export_lammps.py`
- the final-recipe config `input/input_3bpa_omat24_abl3_full_s42_cue.json`; the LiPS config `input_3bpa_omat24_lipsF_fu_s42_cue.json` uses the same recipe with `periodic: true`
- `README.md` (Changes table, LAMMPS section)
- `bam_torch_cue/*` and `bam_torch/*`
- `/home01/t589a02/prog/BAM-torch`: its own V1 RACE, `pair_style bam`, and an ML-IAP unified example
- installed sources: e3nn-jax in `~/.conda/envs/bam_cue`; torch 2.14.1+cu126, e3nn 0.4.4 and mace-torch 0.3.17 in `~/.conda/envs/mace`

## 0. Bottom line

- **No PyTorch RACEv2 exists.** `bam_torch_cue/race.py` and `bam_torch/race.py` are approximate ports of RACE **V1**. They use x = per-species scalar, layer norm with 1/sqrt⟨N⟩, a gated update, and no residual. They are not numerically faithful even to V1: the activation constants and readout structure differ (see §1.3). Reusable parts:
  - Bessel basis, polynomial cutoff, separated layer norm, radial MLP, scatter, and the autograd force/stress code (`bam_torch_cue/utils.py`)
  - the fused cuet edge tensor product with gather/scatter indices (`bam_torch_cue/nequip_torch.py:479-541`)
  - the ML-IAP unified adapter template with ghost exchange (`BAM-torch/examples/tutorial/example-3BPA_300K-LAMMPS-mliap/lammps_mliap_bam.py`, class `LAMMPS_MP`)
- **Missing for RACEv2:**
  - x_mode basis (A_i and per-layer x projections)
  - the channelwise equivariant product m_i ⊗ x_i (23 CG paths)
  - hill normalisation
  - LayerScale γ
  - the Chebyshev residual (needs h from two layers back)
  - linear update and message_norm avg
  - gated stratum readout (readout1 → normalised silu → readout)
  - x and f readouts with the correct normalised activation
  - radial regularity prior
  - shell-radius radial channels
  - UFF repulsion with switch
  - E0, scale and shift handling
  - a weight converter
- **Recommended LAMMPS path:** `pair_style mliap unified` with KOKKOS (the MACE route). It exchanges ghost features after every layer inside autograd. That is the same algorithm as the existing validated `pair_style race mode halo`, but with a single backward pass and no custom C++.
- **Effort:** about **10–15 person-days** (most likely ~12): 5–6.5 for a bit-faithful float64 torch model with converter and validation, 1.5–2 for the cue fast path, 3.5–6.5 for the ML-IAP adapter, the KOKKOS LAMMPS build and multi-GPU validation.
- **Weigh this first:** BAM_V2 already runs RACEv2 multi-GPU and multi-node through PJRT halo mode. That mode agrees with the owner mode to within |ΔE| ≤ 1e-9 eV and |ΔF| ≤ 1e-14 eV/Å, with 88 % weak scaling on 8 GPUs. A torch port buys three things:
  - a standard interface, so no PJRT or custom pair style to maintain
  - one autograd backward instead of staged VJPs that recompute each layer (likely closes much of the 3.3× halo-vs-owner gap)
  - a familiar deployment path for MACE users

  It costs a second implementation that must track every future recipe change.

## 1. Component table

Final recipe, as resolved by `train_3bpa.build_model`:

- hidden_irreps = `128x0e+128x1o+128x2e+128x3o`, sh = `1x0e+1x1o+1x2e+1x3o`
- L = 5 layers, cutoff 5 Å, polynomial cutoff p = 2
- 8 Bessel + 16 shell channels = 24 radial inputs; radial MLP [24, 64, 64, 64, n_w], silu, no bias
- `x_mode=basis`, `x_norm=hill`, `residual=chebyshev`, `layer_scale=0.5` (layers ≥ 1), `update=linear`, `message_norm=avg` (⟨N⟩ = 16.723 for 3BPA, 25.198 for LiPS), `readout_mode=gated`
- defaults left unchanged: `x_readout=mlp`, `x_readout_norm=true`, `f_readout=true`, `readout_species=false`, `stratum_gain=uniform`, `in_layer_order=1`, `cov_mode=scalar6` (variance heads; not needed for inference)
- `radial_prior` gaussian σ = 0.224, `radial_shells=true`, `pair_repulsion=uff` (switch 1.25, width 0.3)
- x64 (float64), `tp_backend=cue`

Line numbers refer to `bam_omat24/race_nnx.py` unless noted. "cue" means `bam_torch_cue`.

| # | RACEv2 component (JAX location) | Torch status | Torch location / note |
|---|---|---|---|
| 1 | Edge vectors r_ij = x_sender − x_receiver, periodic shift `Sij·cell` (utils.py:209-232; `_prelude` 1732-1740) | exists | cue/utils.py:315-330. In ML-IAP the vectors come from LAMMPS; the sign does not matter for E (O(3)-invariant model) but must be consistent in the chain rule (pin with a finite-difference test) |
| 2 | Bessel basis `2/rc · sin(nπr/rc)/r` (utils.py:17-37) | exists, matches | cue/utils.py:83-108. Same formula; cutoff per edge clamped to ≥ 1.0 (utils.py:236). Not the √(2/rc)·sinc form in nequip_torch.py:80 |
| 3 | Polynomial cutoff p = 2 (utils.py:40-57) | exists, matches | cue/utils.py:111-124; `(xn<1).float()` mask: cast to the model dtype for float64 |
| 4 | Radial regularity prior: channel n × 1/exp(σ²n²), n = 1..8, shell channels × 1 (1425-1447; applied in `aggregate` 812-813, every layer including the layer-0 A_i) | missing | static vector; recompute from cfg or (better) dump `radial_scales` |
| 5 | Shell-radius channels: exp(−r/(ρ_a(recv)+ρ_b(send))) for 4×4 shell pairs (s,p,d,f), 0 where a shell is absent, × f_cut (1757-1772); ρ from `atomic_shells.json` via `element_shell_radii` (train_3bpa.py:262-269), Bohr → Å | missing | ordering a-major (receiver shell) × b (sender shell); concatenated after the 8 Bessel channels. Dump the (n_species, 4) table |
| 6 | r = 0 edge masking (1791-1795) | missing (n/a) | only needed for padded graphs; ML-IAP has no padding |
| 7 | Spherical harmonics, e3nn-jax `s2_irreps(3)`, normalize=True, "component" (utils.py:249-254) | partial | cue/utils.py:344 uses `cuet.SphericalHarmonics` (normalisation not checked against e3nn-jax); bam_torch/utils.py uses e3nn torch `o3.spherical_harmonics(..., 'component')`, expected identical (e3nn-jax follows e3nn conventions) but **verify to 1e-14** |
| 8 | Species embedding: vanilla Linear `4x0e → 128x0e`, no bias, one-hot input (1408-1412, 1803-1804) | exists (pattern) | cue/race.py:240; one-hot `.float()` must become the model dtype |
| 9 | Equivariant Linear, vanilla and species-indexed (`linear_nnx.Linear`, 129-422; FunctionalLinear normalisation) | partial | cue/linear.py (cuet.Linear, ir_mul layout, `weight_classes`); bam_torch/linear.py (e3nn o3.Linear + weight table). Conventions in §1.2; recommend a plain-torch block matmul in mul_ir layout (exact, trivially convertible) |
| 10 | Radial MLP: `DenseLinear` (in, out), no bias, silu between layers (utils.py:108-171) | exists | cue/utils.py:207-236; torch `nn.Linear.weight` is (out, in), so **transpose** |
| 11 | Message: `linear_1(h)` → channelwise CG TP with Y(r_ij) weighted by R(r_ij) → scatter to receiver → **/⟨N⟩** (message_norm avg, no layer norm) (`aggregate` 791-838; `cue_edge_tensor_product` 99-126) | partial | cue/nequip_torch.py:479-541 has the fused `cuet.SegmentedPolynomial` with input/output indices. It builds the descriptor with an extra `True` argument and `* epsilon`: RACEv2 must use exactly `cue.descriptors.channelwise_tensor_product(irreps_in, sh, msg_irreps)` (race_nnx.py:62), no epsilon, mul_ir ↔ ir_mul transposes as in race_nnx.py:78-96. cue/race.py:114-167 uses `ChannelWiseTensorProduct` + `/sqrt(⟨N⟩)` (V1) |
| 12 | Atomic basis A_i = `layers[0].aggregate(embedding)` (computed once, 1879-1892); irreps = layer-0 tp_irreps = `128x0e+128x1o+128x2e+128x3o` | missing | identical to layer 0's own aggregated message (computed twice in JAX; torch can reuse it) |
| 13 | x_mode basis: x_i = `x_projs[t](A_i)`, vanilla Linear, no bias (1546-1559, 1923-1925) | missing | 5 Linears tp_irreps(layer 0) → hidden |
| 14 | Hill normalisation: x̂ = x/√ms · 2σ(a_{t,Z}·½log ms + b_{t,Z}), ms = mean(x²) + 1e-6 over **all** 512·… components of the node (228-237); `x_hill` (L, 2, n_species) (1603-1605) | missing | plain `torch.sigmoid` (not normalised) |
| 15 | Equivariant product m_i ⊗ x_i: `linear_m(m)` → mul_to_axis(128) → e3nn-jax `tensor_product` of two mul-1 copies with all paths into `{0e,1o,2e,3o}` → axis_to_mul (642-657, 894-903). **Pure e3nn-jax even with tp_backend cue.** | missing | 23 CG paths per channel, regrouped to `512x0e+768x1o+896x2e+768x3o` (4/6/7/6 paths), element order **(channel, path)**, channel-major (axis_to_mul). Node-wise op: an einsum with dumped CG tensors is fast enough |
| 16 | Update: δ = `linear_2(m⊗x) + skip_Z(h)`, skip species-indexed; no gate (update=linear) (904-907) | partial | V1 torch has linear_2 + indexed skip, followed by a gate (wrong for v2) |
| 17 | LayerScale: δ ← γ ⊙ δ, γ has 512 entries (one per irrep instance, mul-major over the 4 blocks), layers ≥ 1 only (393-401, 578-579, 915-916) | missing | |
| 18 | Chebyshev residual: h_{t+1} = 2δ_t − h_{t−1} for t ≥ 2; δ alone for t = 0, 1 (924-932); history `feats_hist[i-1]` = input of layer i−1 (1695-1700) | missing | readout reads h_{t+1} |
| 19 | Gated stratum readout: `readout1` hidden → `64x0e` (+bias) → **normalised** silu (`e3nn.gate` on scalars) → `readout` 64x0e → 0e (+bias) (761-789, 946-959) | missing / wrong | cue/race.py:142 has a single Linear (readout1 commented out); bam_torch/race.py has no activation |
| 20 | x readout per layer (`RACEXReadout`, 425-494): separated LN(x̂)/√⟨N⟩ → indexed Linear hidden → 128x0e → Linear → 64x0e (+b) → **normalised** silu → Linear → 0e (+b) | partial | cue/race.py:30-83 has the right order but plain `F.silu` (no normalisation constant); bam_torch/race.py:68-80 puts silu after readout1 (wrong order) |
| 21 | f readout per layer: same `RACEXReadout` on the layer **input** h_t (layer 0: 128x0e embedding, separated LN still applied) (1510-1518, 1938) | partial | same as 20 |
| 22 | Separated layer norm (utils.py:60-105) | exists, matches | cue/utils.py:131-200 (layout-agnostic within ranges; eps 1e-6) |
| 23 | UFF repulsion: E = ½Σ_ij D_ij·6/(ζ_ij−6)·exp(ζ_ij(1−r/x_ij))·f_cut·σ((r − 1.25(R_i+R_j))/0.3), geometric combination rules, R = covalent radii (`COVALENT_RADII`, train_3bpa.py ~195-211); table 275-378; energy 1846-1867 | missing | fixed constants; dump x, D, ζ and R per species |
| 24 | Strata sum: Σ_t (e_node + e_x + e_f) + E_rep, then × `scale` + `shift` (1990-2007, 2046-2056); gauge refit folded into `shift` (training_utils.py:1035-1039) | partial | V1 torch has scale/shift; E_rep is also multiplied by the learned `scale` |
| 25 | E0: `atom_energies[species]` added when `l_train=False` (utils.py:365-368); export adds E0 in C++ | exists (pattern) | |
| 26 | Forces/stress by autograd (utils.py:270-442) | exists | cue/utils.py:353-457 (position-based); ML-IAP needs the edge-vector version (dE/dr_ij), as in the BAM-torch adapter |
| 27 | Variance heads (frc/energy/stress_logvar), `x_linear` (species x; unused when x_mode = basis) | not needed | parameters are in the checkpoint; skip them |

### 1.1 Irreps ordering and layout (must match exactly)

- **e3nn-jax data layout is mul_ir** (per block `(mul, 2l+1)`). cuequivariance kernels use ir_mul. JAX transposes around the cue call only (race_nnx.py:78-96). `bam_torch_cue` uses ir_mul everywhere. Recommendation: keep **mul_ir** in torch, so weight and feature layouts match JAX one-to-one, and transpose only around the cue edge TP, as JAX does.
- **Edge TP output irreps.** e3nn-jax `tensor_product` loops (block of x) × (block of Y) × (l_out); each path gives mul_1·mul_2 = 128 channels; the result is then `sort()` + `regroup()` (stable), see e3nn_jax `_src/tensor_products.py:97-133`.
  - Layer 0: 128x0e ⊗ Y gives 4 paths, so tp_irreps = `128x0e+128x1o+128x2e+128x3o`.
  - Layers 1–4: hidden ⊗ Y gives 23 paths, regrouped `512x0e+768x1o+896x2e+768x3o`, **path-major** within each block (128 channels per path).
  - cue's `channelwise_tensor_product` output order equals this: `tests/test_tp_backend_cue.py` passes with identical parameters. The radial MLP output (n_w = 128 × #paths = 512 or 2944) follows the same order.
- **x product (row 15)** uses the mul-1 trick: tensor_product of `1x0e+1x1o+1x2e+1x3o` with itself, giving `4x0e+6x1o+7x2e+6x3o`, then axis_to_mul. The merged block index is therefore **channel-major (c·n_paths + p)**, the opposite of the edge TP. `linear_2`'s weight rows (512/768/896/768 × 128) follow this order.
- **CG coefficients.** e3nn-jax uses `e3nn.clebsch_gordan(l1,l2,l3) · sqrt(2l_out+1)` (component normalisation). e3nn torch `o3.wigner_3j` should be the same tensor, but a single sign difference on one (l1,l2,l3) would break 1e-10 agreement. **Dump the 23 CG tensors from e3nn-jax** in the converter job and use them in torch.
- **Spherical harmonics.** Both e3nn libraries use "component" normalisation, ‖Y^l‖² = 2l+1, with l = 1 ∝ (x, y, z) in e3nn convention. Verify e3nn-torch or cuet SH against dumped e3nn-jax values on random vectors. If they differ, port e3nn-jax's lmax ≤ 3 polynomials verbatim.

### 1.2 Linear layers (`linear_nnx.Linear` over e3nn-jax `FunctionalLinear`)

- Config defaults (e3nn_jax `_src/config.py`): irrep_normalization component, path_normalization "element", **gradient_normalization "path"**. So the stored weights are N(0,1)-scale and the forward multiplies by path_weight = sqrt(1/fan_in), with fan_in = Σ mul_in over instructions feeding that output block.
  - Inputs are `regroup()`ed, so each output block has exactly one input block of the same irrep: path_weight = 1/sqrt(mul_in). For example 1/sqrt(512) for the 0e block of `linear_2` in layers ≥ 1.
  - y_{w,i} = path_weight · Σ_u W[u,w] x[u,i]; W shape **(mul_in, mul_out)**.
  - Biases (0e outputs only) have path_weight 1 and shape (mul_out,).
- Indexed (species) linears have a leading axis: (n_species, mul_in, mul_out). Species order = cfg `atomic_numbers` order (3BPA: H, C, N, O; LiPS: Li, P, S).
- `force_irreps_out=True` keeps unreachable output blocks as zeros. Example: `x_readout1` hidden → 128x0e uses only the 0e input block.
- **Parameter names** are sanitised instruction names (linear_nnx.py:304-330): `w[0,0] 128x0e,128x0e` becomes attribute `weight_w_0_0_128x0e_128x0e`, and `b[0] 64x0e` becomes `weight_b_0_64x0e`. The names contain the regrouped irreps strings (e.g. `512x0e`), so the converter can recover each block's shape and fan-in from the name alone.
- With cuet.Linear instead (cue/linear.py), cue's own linear normalisation and weight flattening must be checked against the above. A plain-torch block matmul avoids this. Linears are node-wise and cheap; the edge TP dominates.
- Indexed linears in torch: a `W[species]` gather costs O(N·128·128) memory per block (the problem the JAX `BAM_INDEXED_LINEAR=ragged` path solves). Sort nodes by species and do one matmul per species (n_species ≤ 4 here) instead.

### 1.3 Scalars and other conventions that must match bit-for-bit

- **Normalised activations.** e3nn-jax `gate` / `scalar_activation` default `normalize_act=True`: silu(x)/c with c = sqrt(mean(silu(x)²)) over `normalspace(1_000_001)`, a deterministic quantile grid (e3nn_jax `_src/activation.py:49-70`).
  - e3nn torch `normalize2mom` and `bam_torch_cue/nequip_torch.py:58-77` use 10⁶ seeded `randn` samples instead. That gives a different constant, off at about the 1e-4 relative level, which alone breaks 1e-10 agreement.
  - **Dump c_silu from the JAX env (x64 on) and hard-code it.** It is used in every gated stratum readout and in every x/f readout.
  - `hill` uses a plain sigmoid. With natural-parity irreps there are no odd scalars, so no tanh path.
- **⟨N⟩:** a static float from cfg (16.723 / 25.198). Messages are divided by ⟨N⟩ (message_norm avg). The x and f readouts divide by sqrt(⟨N⟩) after separated layer norm (race_nnx.py:478-481).
- **eps values:** 1e-6 in separated layer norm and hill.
- **Cutoff:** model cutoff 5.0. In the periodic and LAMMPS-export paths the Bessel and cutoff functions take the cutoff from `globals['cutoff']`, which must equal the model cutoff. The shell channels and UFF use `self.cutoff` (`r_c`).
- **scale, shift:** trained Params. `gauge_refit` adds δ* to `shift` in the saved ckpt (use the gauge-refit checkpoint the paper numbers come from).
- **E0:** `atom_energies` from cfg (training targets are E − ΣE0 with `l_train=True`).
- **dtype:** float64 throughout (x64). Bitwise equality with JAX is not expected: summation order differs, and the cue kernels use atomics. Expected agreement is about 1e-13 relative, so 1e-10 absolute is a safe target for E and F.
- **Reference runs:** the JAX reference must run with `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` (known float64 miscompile).

## 2. Weight conversion

**Source.** `pickle.load(ckpt)['ema_params']` is a Flax NNX `State`, the `nnx.split(model, nnx.Param)` param tree. Unpickling needs flax and jax, which initialise XLA, so the converter must run in a Slurm job (the no-JAX-on-login-node rule); CPU (`JAX_PLATFORMS=cpu`) is enough.

**Converter design.** Run in the bam_cue env and write `racev2_weights.npz` plus `racev2_meta.json`. Torch then loads them with no JAX or flax dependency.

1. Build the model exactly as `export_lammps.build` does (`T.build_model(cfg, ...)`, `nnx.merge(graphdef, ema_params)`), so all static constants are resolved by the original code.
2. Flatten with `jax.tree_util.tree_flatten_with_path` into `"layers/3/linear_2/weight_w_0_0_512x0e_128x0e" → array`.
3. Write the metadata:
   - hidden, sh and per-layer tp_irreps strings
   - for every Linear: its instruction list (i_in, i_out, path_shape, path_weight)
   - `radial_scales`, `species_shell_radii`, `species_radii_t`, the UFF rows for `species_z`
   - `avg_n_neighbors`, `atom_energies`, cutoff, p, n_layers, the dtype
   - c_silu
   - the 23 CG tensors and the edge-TP path list in cue output order (from `e.outputs[0].irreps` and the descriptor)
4. Write per-block probes (inputs/outputs of SH, radial basis, one layer, x product, hill, readouts on a small 3BPA frame), so torch blocks can be unit-tested in isolation.

**Tree to torch mapping:**

| NNX path | torch target | transform |
|---|---|---|
| `species_embedding/weight_w_0_0_4x0e_128x0e` (4, 128) | embedding W | ×1/√4 at forward (path_weight) |
| `layers/t/linear_1/*` (vanilla, in → in) | per-block W | path_weight 1/√mul_in |
| `layers/t/radial_mlp/layers/k/weights` (in, out) | `nn.Linear.weight` | **transpose**, no bias |
| `layers/t/linear_m/*` (tp_irreps → hidden) | per-block W | |
| `layers/t/linear_2/*` (x-product irreps → hidden) | per-block W | row order = (channel, path), see §1.1 |
| `layers/t/skip/*` (n_species, mul_in, mul_out) | indexed W | |
| `layers/t/gamma` (512,) (t ≥ 1) | γ | expand to components per block (mul-major) |
| `layers/t/readout1/*`, `layers/t/readout/*` (+ biases) | gated readout | |
| `x_projs/t/*` (layer-0 tp_irreps → hidden, no bias) | x projection | |
| `x_hill` (L, 2, n_species) | hill a, b | |
| `x_readouts/t/{x_readout1 (indexed), x_readout2, x_readout3}`, `f_readouts/t/...` | readouts | |
| `shift`, `scale` | scalars | |
| `x_linear`, `frc_logvar`, `energy_logvar`, `stress_logvar` | ignored | |

**Risky parts, highest first:**

1. Block-internal ordering of the merged irreps: path-major (edge TP) versus channel-major (x product). A wrong order still runs and still gives finite energies. Only a probe comparison catches it.
2. The activation normalisation constant (§1.3).
3. CG and SH sign or normalisation conventions (dump them; do not re-derive).
4. Constants that are not Params (radial prior, shells, UFF, ⟨N⟩, E0, covalent radii): the converter must write them, not torch.
5. Choosing the checkpoint (gauge-refit EMA versus raw `params`) and the species order.
6. cuet linears or SH, if used instead of plain torch (§1.2).

## 3. LAMMPS integration options

The model needs n_layers × r_c = 25 Å of context per atom. With domain decomposition there are only two choices: a 25 Å ghost shell (the existing `mode ghost`, out of memory above ~1k atoms per rank), or a per-layer exchange of node features (the existing `mode halo`).

The Chebyshev term reads h from two layers back only **node-wise**: h_{t+1} = 2γ⊙δ_t − h_{t−1}. Messages only read the current h_t of senders. So it needs no extra communication, provided the history keeps the post-exchange tensors (as `export_lammps.staged_energy_forces`, lines 460-514, does). In the backward pass, the cotangent of h_{t−1} collects both uses (sender in layer t−1 and Chebyshev term in layer t+1) before its reverse exchange. Autograd does this automatically when the same exchanged tensor object is reused.

| Option | How multi-GPU works | Fit for RACEv2 | Work |
|---|---|---|---|
| **A. ML-IAP unified + KOKKOS** (`pair_style mliap unified`, LAMMPS ≥ Jul 2025; MACE `lammps_mliap_mace.py`; BAM-torch V1 adapter `lammps_mliap_bam.py`) | LAMMPS hands over pair vectors (i owned, j owned or ghost, one cutoff shell; `rcutfac = rc/2`) and `forward_exchange` / `reverse_exchange` on device (KOKKOS only); a `torch.autograd.Function` exchanges after each layer, and its backward is the reverse comm | **Best.** Exactly the validated halo algorithm. 4 forward + 4 reverse exchanges per step (after layers 0–3). One autograd backward (no layer recompute). Eager torch, so cue kernels and torch.compile work. Energy summed over owned atoms only. LAMMPS derives the virial from the pair forces | adapter ~1 d (template exists); needs a KOKKOS + ML-IAP + PYTHON LAMMPS build with cython, cupy, embedded python and CUDA-aware MPI |
| B. `pair_style nequip` / `allegro`-style (TorchScript or AOTInductor, one model, libtorch C++) | ghost atoms are part of the graph; a message-passing model needs ghost cutoff ≥ L·rc (Allegro avoids this by being strictly local; RACEv2 is not) | **Poor.** Same memory wall as `mode ghost` (25 Å shell). BAM-torch `pair_style bam` (`ML-BAM/pair_bam.cpp`) is this design; it also limits kernels to TorchScript-able ones | small C++ if adapted from pair_bam, but does not scale |
| C. SevenNet-style parallel (`e3gnn/parallel`: one TorchScript module per layer; C++ pair style does forward and reverse comm of node features between layers and runs the backward) | per-layer comm in C++ | **Good but redundant.** `lammps/pair_race.cpp` mode halo already does exactly this with PJRT executables; swapping PJRT for libtorch modules reuses its comm logic | 3–5 d C++ (libtorch link, per-layer modules and their backward or VJP); keeps a custom pair style to maintain |

**Recommendation: A.** Notes for the adapter:

- RACEv2 edge vectors are x_sender − x_receiver, so with sender = `pair_j`, receiver = `pair_i`, use `vectors = data.rij` (no negation, unlike the V1 adapter). Pin this with a finite-difference force check; E is parity-invariant, so a sign error shows up only in the forces.
- Return per-atom energies with E0 added, and pair forces dE/dr_ij.
- `reverse_exchange` must **zero the ghost rows** after adding them to the owners, as `rev_comm` does in export_lammps.py:476-480. Otherwise cotangents leak through the garbage ghost rows of δ. A 1-rank versus N-rank force comparison at 1e-10 catches this.
- A_i and x_i are node-wise and need no exchange. Ghost rows of every layer output are overwritten by the exchange.

**Alternative without a torch port:** an ML-IAP unified adapter that calls the existing staged JAX functions (`make_stage_fns`) through DLPack, with the exchanges between stages. That is cheaper (~3–4 d) if the only goal is "standard LAMMPS interface". It keeps the staged-VJP recompute cost and adds JAX-in-LAMMPS-python process issues (two CUDA runtimes with cupy and Kokkos).

## 4. Effort and plan

Steps that import JAX or torch, or need a GPU, run as Slurm jobs on `normal` (from glogin, under /scratch). Each sbatch needs the user's approval: one small test job first, no resubmit loops. Dev env: the existing `mace` env (aarch64, torch 2.14.1+cu126, e3nn 0.4.4), plus `cuequivariance-torch` and `cuequivariance-ops-torch-cu12` at the same version as the JAX side (0.13; check that an aarch64 wheel exists), plus `cupy-cuda12x` for the KOKKOS coupling.

| Step | Content | Validation | Days |
|---|---|---|---|
| 1. Converter | JAX job: build the model as `export_lammps.build`; dump npz + meta + CG tables + c_silu + per-block probes on one 3BPA frame and one LiPS frame (periodic) | probe files reproduce `model(data)` E/F inside JAX | 1 |
| 2. Reference torch model (plain torch, mul_ir, float64, CPU and GPU) | radial (Bessel, cutoff, prior, shells), SH, Linear vanilla/indexed, edge TP by CG einsum + scatter, A_i, x projections + hill, x product, linear update, γ, Chebyshev, gated readouts, x/f readouts, UFF, strata sum, scale/shift/E0; `forward(vectors, species, edges)` and the position-based wrapper | block-by-block against the probes at 1e-12 | 3–4 |
| 3. End-to-end check | 3BPA: 20 frames each from test_300K/600K/1200K; LiPS: 20 test frames (periodic, stress) | max \|ΔE\| < 1e-10 eV, max \|ΔF\| < 1e-10 eV/Å, ΔP < 1e-6 bar; rotation-equivariance test; the reported 3BPA RMSE reproduced | 1–1.5 |
| 4. cue fast path | edge TP through `cuet.SegmentedPolynomial(cue.descriptors.channelwise_tensor_product(in, sh, out).polynomial, method="uniform_1d")` with `input_indices={1: senders}`, `output_indices={0: receivers}` and mul_ir ↔ ir_mul transposes (as race_nnx.py:99-126); per-species sorted matmuls for the indexed linears; optional `torch.compile` | same tolerances as step 3; speed and memory against JAX on LiPS-83/664/5312 | 1.5–2 |
| 5. ML-IAP adapter | `MLIAPUnified` subclass: `elems → species`, `rij → vectors`, `pair_j/pair_i → senders/receivers`, LAMMPS_MP exchange after layers 0..L−2 with post-exchange history for Chebyshev, local-only energy sum, E0, `update_pair_forces_gpu`; `__reduce__` rebuild like the V1 adapter | without LAMMPS: emulate ghosts on one process with ghost_owner maps (copy of `staged_energy_forces` in torch), compare with the JAX owner mode at 1e-10 | 1–1.5 |
| 6. LAMMPS build | stable_30Sep2026 with `PKG_KOKKOS` (CUDA, `Kokkos_ARCH_HOPPER90` + `ARMV9_GRACE`), `PKG_ML-IAP`, `PKG_ML-SNAP`, `MLIAP_ENABLE_PYTHON`, `PKG_PYTHON`, cython; Cray PrgEnv-gnu + cray-mpich with GPU support (`MPICH_GPU_SUPPORT_ENABLED=1`, GTL) for CUDA-aware exchange; build on glogin with ≤ 16 make jobs or in a GPU job | `lmp -h` lists `mliap`; a MACE ML-IAP smoke test if available | 1–3 (most uncertain) |
| 7. LAMMPS validation | `run 0` on 1 rank against Python; 1/2/4/8 ranks (2 nodes) against 1 rank; NVE 1 ps drift; weak scaling against the README PJRT table | \|ΔE\| ≤ 1e-9 eV, \|ΔF\| ≤ 1e-10 eV/Å, ΔP ≤ 1e-6 bar; drift ≤ 0.01 meV/atom/ps | 1.5–2 |
| **Total** | | | **10–15 (≈ 12)** |

Optional extras, not in the total:
- float32 inference check against float64, 0.5 d. Not validated in JAX either: the README reports no speed gain for float32 PJRT export.
- Keeping the torch model in sync with JAX: add a CI-style probe test (step 1 probes + step 2 model) run on every recipe change.

**Main schedule risks:**
- the KOKKOS + python + cupy LAMMPS build on Cray aarch64
- whether `reverse_exchange` semantics zero the ghost rows
- whether a cuequivariance-torch aarch64 wheel exists at the JAX-matching version
- the ordering and normalisation items in §1.1 and §1.3, which are cheap to fix once the probes exist
