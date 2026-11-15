# SOAP-curvature Laplace posterior for RACEv2 (design, 2026-11-06)

Goal: a posterior over **all** weights at almost no extra cost, built from the curvature statistics that the SOAP
optimizer already keeps. It complements the exact last-layer Bayesian regression (`race_v2/eval/bayes_lastlayer.py`),
which represents only the uncertainty of the readout weights. Status: theory and code skeleton
(`race_v2/eval/soap_laplace.py`), **not run** (allocation exhausted). Order of work at the end.

## 1. Laplace approximation

For the training loss $\mathcal L(\theta) = \sum_n \ell_n(\theta)$, read as a negative log-likelihood, and a Gaussian
prior $\mathcal N(0, \lambda^{-1} I)$, the Laplace posterior around the trained weights $\theta^*$ is

$$p(\theta\mid D) \approx \mathcal N\big(\theta^*,\ (\kappa H + \lambda I)^{-1}\big),\qquad H = \sum_n J_n^\top \Lambda_n J_n
\quad\text{(generalised Gauss–Newton)},$$

where $J_n$ is the Jacobian of the outputs (energy, force components) with respect to $\theta$, and $\Lambda_n$ is
the Hessian of $\ell_n$ with respect to the outputs.
- **Huber loss.** Huber has second derivative 1 inside $\delta$ and 0 outside. With δ = 10 in internal units,
  essentially every residual is inside, so the GGN equals that of the weighted Gaussian likelihood.
- **κ.** The single scalar $\kappa$ converts the training-loss scale into a likelihood: noise precision times the
  loss normalisation. It is a hyperparameter, like $\lambda$.

## 2. What SOAP stores and how it relates to the GGN

For every matrix-shaped weight $W \in \mathbb R^{m\times n}$ (species slices of 3-D tables are separate matrices),
SOAP (`race_v2/training/soap_grouped.py`) keeps:
- Kronecker factors $L = \mathrm{EMA}_{\beta_s}(G G^\top)$ and $R = \mathrm{EMA}_{\beta_s}(G^\top G)$ of the
  minibatch gradients $G$;
- their eigenbases $Q_L$ and $Q_R$, refreshed every 10 steps;
- Adam moments **in the rotated basis**: $\nu = \mathrm{EMA}_{\beta_2}\big((Q_L^\top G Q_R)^{\odot 2}\big)$.

This is the structure of eigenvalue-corrected K-FAC (EKFAC, George et al. 2018): a Kronecker-factored eigenbasis
with a diagonal of second moments measured in that basis. Near convergence $\mathbb E[G] \approx 0$, so $\nu$
estimates the variance of the minibatch gradient in the rotated basis:

$$\nu_{ab} \approx \frac{1}{B}\,\big(Q^\top F_{\mathrm{emp}} Q\big)_{(ab),(ab)},\qquad
F_{\mathrm{emp}} = \frac1N\sum_n g_n g_n^\top,$$

with $B$ the structures per step. For the (weighted) Gaussian likelihood, the empirical Fisher of the summed loss is
$N F_{\mathrm{emp}} \approx H$ at the optimum, so in the SOAP eigenbasis

$$\mathrm{diag}\big(Q^\top H Q\big) \approx N B\,\nu .$$

The Laplace precision of each weight matrix is therefore diagonal in the SOAP basis:

$$\Lambda_W = (Q_L\otimes Q_R)\,\mathrm{diag}\big(\kappa' \nu + \lambda\big)\,(Q_L\otimes Q_R)^\top,\qquad \kappa' = \kappa N B .$$

Leaves updated by plain Adam (1-D leaves, and the species tables under `soap_species_adam`) give the diagonal
version, with $\nu$ the ordinary Adam second moment. Leaves of kind `skip` are held fixed.

**Sampling** costs two matrix products per matrix:
$\delta W = Q_L\,\big(\varepsilon \oslash \sqrt{\kappa'\nu+\lambda}\big)\,Q_R^\top$ with $\varepsilon \sim \mathcal N(0, I)$.

## 3. Caveats (to be handled, not ignored)

1. **Noisy ν.** With $\beta_2 = 0.95$, ν averages only the last ~20 steps. Better: one extra pass over the training
   set at $\theta^*$ with the bases fixed, accumulating $(Q_L^\top G_t Q_R)^{\odot 2}$ over all batches (no update).
   This needs a gradient function from `train_3bpa` and is the first refinement.
2. **EMA weights.** ν was collected along the raw-weight trajectory, but the reported model uses EMA weights.
   Centre the posterior on the EMA weights and accept the small mismatch, or recollect ν at the EMA weights
   (refinement 1 does this).
3. **Empirical vs true Fisher.** The empirical Fisher equals the GGN only at a well-specified optimum; it
   over-weights directions with large residuals. A cross-check on a small model: GGN-vector products via
   JVP/VJP versus the SOAP diagonal.
4. **Energy-gauge null direction.** A constant energy per atom has zero force curvature; on single-stoichiometry
   data the energy curvature is tiny. Only $\lambda$ controls this direction, so report the per-structure energy σ
   with and without that direction (or fix it via the closed-form refit).
5. **Hyperparameters κ′ and λ.**
   - Option (a): maximise the validation predictive log-likelihood on a grid.
   - Option (b): the Laplace marginal likelihood
     $\log p(D\mid\theta^*) - \tfrac12\sum\log(1+\kappa'\nu/\lambda) - \tfrac\lambda2\lVert\theta^*\rVert^2$. In the
     eigenbasis this is closed form (Immer et al. 2021).
   - Use (b) for the paper, (a) as a check.
6. **Predictive.**
   - Monte Carlo over S = 8–32 weight samples (energies and forces of each sample) plus the aleatoric noise from
     the BLR evidence fit.
   - Alternatively the linearised (GLM) predictive, using one JVP per sample instead of a full re-evaluation.
   - MC is simpler; GLM is better calibrated in the literature (Immer et al. 2021).

## 4. Why this is a methodology contribution

- The posterior comes "for free" from the optimizer that already produces the accuracy: no extra training and
  no ensemble.
- The SOAP eigenbasis is exactly where the Kronecker/EKFAC approximation is most accurate.
- Together with the exact last-layer regression, it separates readout uncertainty from feature uncertainty.
- The test is calibration under extrapolation: 3BPA 600 K, 1200 K and the dihedral scan, compared with the 3-seed
  deep ensemble, using `calib()` of `bayes_lastlayer.py` (z-RMS, NLL, coverage, Spearman |err|–σ).

## 5. Order of work (when compute is available)

1. Quick check of `bayes_lastlayer.py` (`research/diag/bayes_lastlayer.sh`, MAX_TEST=200), then the full run.
2. `soap_laplace.py` on the 3BPA final checkpoint (abl3_full s42):
   - check that sampled weights stay finite and that energy/force predictions at κ′ → ∞ equal the network;
   - grid-fit κ′ and λ on validation;
   - calibration on the test sets against the ensemble and the BLR.
3. Refinement 1 (ν recollected at θ*), then the marginal-likelihood fit (5b).

## 6. Status (2026-11-09)

Run on the local 4G checkpoints (docs/local_test_report.md sections 18-19), not yet on 3BPA/OMat24.

- Steps 2 and 3 done: the sampler is exact in its limits; caveat 3 turned out to be decisive. Both the optimizer ν
  and the recollected ν (refinement 1) are empirical Fishers with median 1e-15: the posterior is either the
  isotropic prior (with the global `scale` frozen) or over-confident. Negative result.
- **Refinement 2 (`--ggn`): the GGN diagonal in the SOAP basis.** $
u_i = rac1N\sum_s\sum_k w_k
  ig[(Q_L\otimes Q_R)^	op J_{s,k}ig]_i^2$ with $J_{s,k}$ the Jacobian of output $k$ (energy, forces) of
  structure $s$ wrt the weights and $w_k = 1/\sigma_k^2$ the Gaussian likelihood precisions (defaults: the
  network's validation RMSEs). Estimated either exactly (one VJP per output, 1 + 3 n_atoms per structure) or by
  Hutchinson probes $v\sim\mathcal N(0,\mathrm{diag}\,w)$: $\mathbb E[(J^	op v)^{\odot 2}] = \mathrm{diag}(J^	op W J)$,
  one VJP per probe. The untempered posterior precision is $N_{
m train}\,
u + \lambda$ (`--kappa-times-ntrain`,
  κ = 1). This is the EKFAC-style diagonal GGN that V1's torch Laplace obtained from curvlinops, here in the
  optimizer's own eigenbasis.
