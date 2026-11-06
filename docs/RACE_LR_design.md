# RACE_LR: charge-dependent long-range extension of RACEv2 (design, 2026-11-06)

User request (2026-11-06): extend the model to charge-dependent and long-range interactions **without changing the
existing model**, so that it can be tried on other problems. RACE (`bam_omat24/race_nnx.py`) is unchanged; the
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

## 2. Limits of the first version

- **Dense matrices.** The Qeq matrix is N_pad × N_pad per batch, and the Ewald real-space part is N² × images. This
  is fine for molecules and clusters, and for periodic cells up to a few hundred atoms per batch.
  - For larger systems, use conjugate gradient with Ewald matrix-vector products, or PME.
  - The species tables use `n_real` / `n_recip` from the config (defaults α = 0.3 Å⁻¹, n_real = 2, n_recip = 6).
    Check convergence per dataset with the α-independence test.
  - Au2/MgO (9.05 × 9.05 × 26.46 Å, 110 atoms), checked 2026-11-06 with the Hirshfeld charges: the defaults are within
    0.008 meV of α 0.3 / n_real 4 / n_recip 24, and α 0.2–0.4 agree to < 1e-4 meV.
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

Step 1 (unit tests, `tests/test_race_lr.py`, job `runs/diag/race_lr_check.sh`, about 5 min on one GPU):
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
