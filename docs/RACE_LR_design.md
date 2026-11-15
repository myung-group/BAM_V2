# RACE_LR: charge-dependent long-range extension of RACEv2 (design, 2026-11-06)

User request (2026-11-06): extend the model to charge-dependent and long-range interactions **without changing the
existing model**, so that it can be tried on other problems. RACE (`race_v2/models/race.py`) is unchanged; the
extension is a subclass selected with `"model": "RACE_LR"`.

Status:
- Step 1 (code and unit tests): the CPU tests pass on the user's local PC (2026-11-06, `docs/local_test_report.md`).
- Step 2 data: converted (`docs/DATASETS.md` §16). Step 2 training and step 3: planned.

## 1. Model

$$E = \sum_i\Big(\sum_t E_i^{(t)} + E_i^{\mathrm{rep}}\Big)\ \text{(RACEv2, unchanged)} \;+\; E^{\mathrm{Qeq}}(\{q_i\}),\qquad
E^{\mathrm{Qeq}} = \chi^\top q + \tfrac12 q^\top A q,$$

$$q = \arg\min_q E^{\mathrm{Qeq}}\quad\text{s.t.}\quad \textstyle\sum_{i\in g} q_i = Q_g\ \text{for every graph } g .$$

- **Electronegativity and hardness.**
  - $\chi_i = \chi^0_{Z_i} + c\,\mathrm{MLP}(h_i^{0e})_1$ and $J_i = \mathrm{softplus}(J^0_{Z_i} + \mathrm{MLP}(h_i^{0e})_2) > 0$.
  - $h^{0e}$ are the scalar features of the last RACE layer, so they are environment dependent (4G-HDNNP style).
  - The MLP output layer is initialised at zero.
- **Matrix A.** $A_{ii} = J_i + K/(\sqrt\pi\,\sigma_i)$ (hardness plus the Gaussian self energy), and
  $A_{ij} = K\,\mathrm{erf}(r_{ij}/\sqrt{2(\sigma_i^2+\sigma_j^2)})/r_{ij}$ (Gaussian charges, no singularity at short
  range).
  - $\sigma_Z$ is learned and initialised at the covalent radius. $K$ = 14.399645 eV Å.
  - Molecules and clusters use the direct sum over all pairs of the graph (`long_range.coulomb_direct`).
  - Periodic graphs use an Ewald sum per graph (`long_range.ewald_matrix_rows`), built from three parts:
    - real space, $K[\mathrm{erf}(r/\sqrt2 s) - \mathrm{erf}(\alpha r)]/r$ over images $|n|\le n_{\mathrm{real}}$;
    - reciprocal space as $CC^\top + SS^\top$ with $C_{im} = \sqrt{w_m}\cos(k_m\cdot r_i)$, so memory is $N\times M$;
    - the neutralising background and the self term.
- **Solution.** One linear solve per batch with one Lagrange multiplier per graph (`long_range.qeq_solve`). Padding
  nodes get q = 0, and graphs without real nodes get λ = 0. A is positive definite (Gaussian Coulomb plus J > 0).
- **Charge dependence.** The total charge $Q_g$ comes from `data.globals['charge']`. In the trainer it is read from
  the extxyz info key `total_charge_key` (graphs and caches only change when that key is set). Spin is not included
  yet; add it as a second global with an embedding if a spin-resolved dataset is used.
- **Latent charges.** Training uses energies, forces and stress only, so no charge labels are needed. Autodiff goes
  through the linear solve, which is the implicit derivative of the Qeq minimum. Stress follows from the existing
  formula `cellgrad^T cell + Σ grad ⊗ pos`, because the Ewald matrix depends on both positions and cell.
- **Per-atom energy.** $e_i = \chi_i q_i + \tfrac12 q_i (Aq)_i$, added to the RACE node energies. The model also
  returns `charges` (passed through `utils.energy_forces_stress` only when the model provides them).

Why this fits RACE: the locally resolved strata stay as they are, and all non-local physics enters through one
explicit, physically constrained stratum. This keeps the local/long-range separation interpretable and lets a paper
claim it cleanly.

**Options added 2026-11-06, all off by default.** They were added after the charge diagnostics of the local test
(`docs/local_test_report.md` §8): the latent charges were nearly uniform and too small.
- `long_range.chi_init: "pauling"` initialises χ0_Z as `chi_init_scale` × (Pauling χ_Z − mean over the species).
  The default scale is 3 eV, about the slope of the Mulliken scale. The default `"zero"` is unchanged.
- `charge_label_key` (e.g. `"hirshfeld_charges"`) with `charge_weight` w adds w · mean_i (q_i − q_ref_i)² in e² to
  the loss. The reference charges are labels of the same dataset; 4G-HDNNP fits them in a first step. With this
  option the charges are no longer purely latent, so report it as a separate variant.

- `long_range.charge_spin_embedding: true` adds a zero-initialised MLP(Q, 2S+1 − 1) to the layer-0 scalars. This
  is the cheap global charge and spin input, also for pretraining on OMol25. The spin comes from cfg
  `total_spin_key`.
- `long_range.qeq: false` switches the Qeq stratum off, giving an embedding-only variant.
- `long_range.n_tasks: N` (default 0, off) reserves a task embedding table of shape (N, layer-0 width),
  zero-initialised. Row `globals['task']` (0 if the data carry no task) is added to the layer-0 scalars, at the
  same place as the charge and spin embedding. A task is one labelling convention, that is the level of theory of
  one dataset. A single-dataset run trains row 0 only, and the other rows receive an exactly zero gradient. The
  slot exists so that a model pretrained on OMol25 can later take other datasets without a change of the parameter
  tree (`docs/HANDOFF_uq_omol25.md` §7). The data side (a task index per stream source) is not written yet.
- `chi_init_missing: "mean"` starts elements without a Pauling value at the mean.
- **Spin input matters** (local test, report §14).
  - Na8Cl8+ (doublet) and Na9Cl8+ (singlet) have the same total charge.
  - Without spin, both RACE and RACE_LR fail on Na8Cl8+ once trained on the full split. With
    `charge_spin_embedding` and the multiplicity the force error drops from 108 to 4.8 meV/Å.
  - Use `total_spin_key` whenever the data provide it (OMol25 does).
- **Zero χ is a stationary point for neutral data** (found 2026-11-07). For Q = 0 the Qeq energy is −½ χᵀBχ,
  so every χ parameter has zero gradient at χ = 0. A neutral-only dataset with `chi_init: "zero"` never develops
  charges. Use `"pauling"` there. The default is unchanged for reproducibility.
- In the local 4G test (report §10), the Hirshfeld charge loss is what makes the charges physical (r ≥ 0.98, right
  magnitude). It also gives the best energies and forces on the carbon chain and NaCl.

**Options added 2026-11-07, for the ctx-RACE use case** (protein–ligand ΔE_int; the user decided to pretrain RACE_LR
from scratch on OMol25):
- **`chi_init` defaults to `"pauling"`** (user decision), with `chi_init_missing: "mean"`. Configs written before
  this date pin `"chi_init": "zero"`, so they reproduce.
- **Per-fragment charge constraints.** cfg `fragment_key` (per-atom integer ids, e.g. `"is_ligand"`),
  `fragment_charges_key` (info list of the fragment charges, default `"fragment_charges"`) and `n_fragments`
  (default 2).
  - Each fragment keeps its own integer charge, so no fractional charge flows between distant fragments (a known
    Qeq weakness). Coulomb still couples all atoms.
  - A structure with a single fragment, such as a ligand or protein part, needs no fragment charges: it carries the
    total charge.
  - Fragment charges that do not sum to the total charge raise an error.
- **`long_range.sr_switch: [r_on, r_off]`.** The Coulomb energy of pairs closer than r_off is switched off smoothly
  (C2 smoothstep, fully off below r_on), so the learned short-range strata and the Coulomb term do not double count
  there.
  - The Qeq charges are solved with the unswitched matrix.
  - Periodic graphs use the lattice images |n| ≤ 1, so r_off must be below the shortest cell height.
- **Inference API `race_v2/inference/predict.py`.** `load(ckpt) -> Predictor`.
  - `Predictor.predict(frames)` returns total energies (eV) and per-frame forces (eV/Å). `charges(frames)` returns
    the Qeq charges.
  - Each frame gives `info["charge"]`, `info["spin"]` and `arrays["is_ligand"]` under the key names of the training
    config.
  - `python -m race_v2.inference.predict refit <ckpt> --xyz <train.xyz>` fits a per-element energy correction on training
    frames and writes it to `<ckpt>.species_refit.json`. It is applied when present.
  - Checked on a trained checkpoint: it reproduces the trainer's test numbers exactly, and the refit gives the
    per-composition result of `by_state.py`.
  - Batching: at most `batch_size` (16) frames and `max_atoms` (2048) real atoms per batch, because the Qeq solve is
    dense per batch. A larger frame runs alone, and the loader's 256-node cap is lifted.
    - RTX 4060 Ti, reduced float64 model: about 120 ms per 1000-atom frame.
    - Peak GPU memory is 2.3 GB at a 1100-atom cap and 4.3 GB at 2200.

- **`long_range.qeq_block: B`** (2026-11-07) solves Qeq per structure instead of densely over the whole batch.
  - B is the largest structure, in atoms.
  - The atoms are gathered into (G, B) blocks, and the Coulomb / Ewald matrix, the short-range switch and the
    constrained solve are vmapped per block.
  - Memory is G·B² instead of N², and time G·B³ instead of N³.
  - The results equal the dense path to 1e-9 (tests: molecular with and without fragments and switch; periodic
    with different cells and padding).
  - A structure with more than B atoms makes the energy NaN, so the failure is loud.
  - RTX 4060 Ti measurements:
    - Float32, 64 small molecules or 4 × 1000 atoms: the two paths are the same speed; the Qeq matrix is small next
      to the model.
    - Float64, 8 × 1000 atoms in one batch (8081 atoms): the dense path runs out of memory (a 6.5 GB allocation
      above the 7.2 GB cap), while the block path runs at 166 ms per structure and 5.3 GB peak.
  - Use it for OMol-scale training (B = 352, the OMol maximum is 350 atoms) and for evaluating large complexes.

## 2. Limits of the first version

- **Dense matrices.** The Qeq matrix is N_pad × N_pad per batch, and the Ewald real-space part is N² × images. This
  is fine for molecules and clusters, and for periodic cells up to a few hundred atoms per batch.
  - For larger systems, use conjugate gradient with Ewald matrix-vector products, or PME.
  - The species tables use `n_real` / `n_recip` from the config (defaults α = 0.3 Å⁻¹, n_real = 2, n_recip = 6).
    Check convergence per dataset with the α-independence test.
  - Au2/MgO (9.05 × 9.05 × 26.46 Å, 110 atoms), checked 2026-11-06 with the Hirshfeld charges: the defaults are within
    0.008 meV of α 0.3 / n_real 4 / n_recip 24, and α 0.2–0.4 agree to < 1e-4 meV.
  - With random charges (std 0.5 e) the defaults err by up to 1.9 meV there, so use α 0.3 / n_real 1 / n_recip 8
    (0.06 meV). Small bulk cells of about 5 Å are fine with the defaults (0.006 meV).
  - Check every periodic dataset with `scripts/data/check_ewald.py <xyz>`. It prints the cheapest converged setting.
- **Slab dipole.** The Ewald sum is plain 3D periodic, with no slab dipole correction. This matches the DFT
  reference as far as the record shows: the paper names no dipole correction and FHI-aims applies none by default.
  - The Al-doped Au2/MgO slab has M_z = 1.2–3.7 eÅ. A Yeh–Berkowitz correction would be 65–584 meV, with a
    standard deviation of 92 meV, so adding one would change the target physics. Keep it off unless the DFT
    settings show otherwise.
- **No charges in the short-range features yet.** 4G-HDNNP feeds the charges back into the short-range descriptors.
  This is possible as a later option: a second pass, or q as an extra scalar input of the layers.
- **Not supported yet:** the streaming loader (`stream_io.graph_arrays`) and the LAMMPS export.
  - For LAMMPS, the model would export per-atom χ, J and σ (or the solved q).
  - The long-range part would then run in LAMMPS (`kspace_style pppm` with a Qeq fix), as other ML-Qeq models do.
  - The halo mode covers only the local part.

## 3. Validation plan

Step 1 (unit tests, `tests/test_race_lr.py`, job `research/diag/race_lr_check.sh`, about 5 min on one GPU):
- Ewald vs the NaCl Madelung constant 1.747564594633 (relative error 1e-7), for three values of α
  (α-independence).
- Batched Ewald with different cells per graph equals the single-cell result.
- Qeq conserves charge per graph, keeps padding at zero, and is optimal (constant gradient within each graph).
- RACE_LR forces equal −∇E (finite differences), the energy is rotation invariant, and the energy changes with the
  total charge.
- The existing periodic and stress tests still pass after the data / trainer edits.

Step 2 (small long-range benchmarks; fair protocol, train split only), with RACEv2 as the baseline:
- The four systems of the 4G-HDNNP paper (Ko et al., Nat. Commun. 12, 398 (2021)): C10H2/C10H3⁺ (charge state),
  Ag₃⁺/Ag₃⁻ clusters, NaCl clusters with a missing ion, and Au₂ on MgO(001) (periodic).
  - Their data are in the paper's archive (locate the Materials Cloud record; verify license and md5).
  - These test exactly the non-local charge transfer that a local model cannot represent.
- Dimer and ion-pair dissociation curves as a sanity check: correct 1/r tail.

Step 3:
- SPICE (charged molecules; Zenodo) and OMol25 (charge and spin; Hugging Face, check the license). The total charge
  is in the info fields.
- Report: the energy and force error by charge state, and the long-range tail.
- Then the LAMMPS long-range coupling (section 2).

## 4. Configuration example

```json
"model": "RACE_LR",
"long_range": {"ewald_alpha": 0.3, "ewald_n_real": 2, "ewald_n_recip": 6, "head_hidden": 32, "chi_scale": 1.0,
               "hard_init": 5.0},
"total_charge_key": "charge"
```

Everything else is the RACEv2 recipe of the dataset (copy its config).
