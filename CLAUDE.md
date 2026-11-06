# BAM_V2 / RACEv2

Read `docs/HANDOFF.md` first: project state, the user's rules, environment, data locations, and the ordered next
steps. Results are in `README.md` (English) and `docs/RACEv2_methods_summary.pdf` (Korean).

Short version of the rules:
- Talk to the user in Korean. Write README and its figures in English. Keep the methods PDF (Korean, from
  `docs/methods_pdf_data.json`) and README in sync with every new result.
- Fair protocol: official training split only. No pretraining, extra data or augmentation.
- Work on branch `racev2`. Commits are fine; the user pushes. Never touch the user's own jobs.
- GPU runs always use `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`. No JAX or heavy work on login nodes.
- Before any job submission, show the exact command and wait for approval.
- Update the "State" and "Next steps" sections of `docs/HANDOFF.md` when something finishes.
