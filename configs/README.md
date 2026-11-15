# Representative configs

Only the configs of the main workflows are here; every other config of the project history is in `research/input/`.
The files are copies of the configs that were actually run (contents unchanged, so `log_file` / `ckpt_name` inside
still carry the original run names).  Run with `python -m race_v2.training.train <config>` from a work folder.

| File | Workflow | Original name |
|---|---|---|
| `mptrj/race_v2_mptrj.json` | MPtrj-only training (float64, SOAP, 80 epochs) | `runs/omat_stream/mptrj_v1set_sr32_soap_s42.json` |
| `mptrj/race_v2_mptrj_cooldown.json` | MPtrj cool-down that gave `v1soapCD20_final_e40.pkl` | `runs/omat_stream/mptrj_v1set_sr32_soap_cd20_s42.json` |
| `omat_pretrain/race_v2_omat24_pretrain.json` | OMat24 pretraining with the stream loader (float32) | `runs/omat_stream/omat24_v1set_sr32_soap_pre.json` |
| `omat_pretrain/omat24_100k_L5x64_screening.json`, `omat24_100k_L5x128_screening.json` | OMat24 100k size screening; the L5x64 file is the base of the stream smoke launcher in `scripts/train/omat_stream/` | `input/input_3bpa_omat24_sz100k_L5x{64,128}_omat24_v2_s42_cue.json` |
| `oam_finetune/race_v2_oam_finetune.json` | OAM (OMat24 -> MPtrj + sAlex) fine-tuning, stream parts | `runs/omat_stream/oam_v1set_sr32_soap_ft.json` |
| `omol25_lr_polar/race_v2_lr_polar_omol25_4m.json` | RACE v2 LR-POLAR on OMol25 4M (internal run label v3) | `input/omol/omol4m_v3_polarqeq_c6_cs_ct_L5x128_r12p6_spec_hub_spinE_nbo_csloc_decL_disp_xrn3_lr1e-2_soap_g48x12_s42.json` |
| `base/race_v2_base_recipe.json` | recipe template read by the config generators `scripts/train/make_*_configs.py` | `input/input_3bpa_omat24_abl3_full_s42_cue.json` |
| `smoke/oam_finetune_stream_parts_smoke.json` | smoke of the stream `parts` fine-tuning (2 epochs, 4 GPUs) | `runs/omat_stream/oam_ft_smoke2.json` |
| `smoke/cpu_race_smoke.json`, `smoke/cpu_race_lr_smoke.json` | CPU smokes of RACE and RACE_LR | `input/input_local_race_smoke.json`, `input/input_local_racelr_smoke.json` |
