# Ablation protocol (pre-registered 2026-11-05, before any ablation result)

Goal: attribute the V1 → V2 gain to each component of the final recipe (leave-one-out from the final recipe).

Reference = final recipe (r37 for 3BPA, rmd17f h256 for rMD17, siF for Si). Each variant changes exactly one setting:

| tag | removed component | config change |
|---|---|---|
| full | – (reference at the same epoch budget) | – |
| msgnorm | ① message: back to LayerNorm + 1/√⟨N⟩ | message_norm: layer |
| xspecies | ② atom-centred factor x_i = Lin_t(A_i) → per-species constant (V1) | x_mode: species, x_norm: null |
| nohill | ③ hill magnitude control | x_norm: null |
| nocheb | ⑤ Chebyshev 3-term recurrence (h⁽ᵗ⁾ = γδ only) | residual: none |
| nols | ⑤ LayerScale γ | layer_scale: null |
| gate | ⑤ linear update → V1 gated update | update: gate |
| noprior | Gaussian radial regularity prior | radial_prior: null |
| noshell | shell-radius radial channels | radial_shells: false |
| nouff | UFF-nb repulsion | pair_repulsion: null |
| amsgrad | SOAP → AMSGrad (V1 optimiser) | optimizer: amsgrad |
| (gauge) | closed-form energy gauge refit | no run: raw vs gauge from every summary |

Budgets (fixed before results; shorter than the finals, so absolute numbers are higher than the 1500-ep tables):
- 3BPA: 500 ep × 3 seeds (42/43/44); test 300/600/1200 K + dihedral β slices (train_dih.sh).
- rMD17 aspirin: 500 ep × 1 seed (42).
- Si GAP-18: 150 ep × 1 seed (42); RSS and compression-MD external tests.

Reporting: best-validation checkpoint + gauge (raw also given), mean ± std over seeds where 3 seeds exist; the
effect of a component = variant − full at the same budget. No variant is re-tuned; a variant that diverges is
reported as such. Caveat recorded in advance: 500-ep (150-ep) orderings may differ from converged orderings.
