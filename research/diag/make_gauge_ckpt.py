"""Write runs/<run>_cue/ckpt_<run>_cue.gauge.pkl from runs/<run>_cue/gauge_refit.json (single stoichiometry:
per-frame constant c -> per-atom shift c / N).  usage: python runs/diag/make_gauge_ckpt.py N_atoms|auto run [run ...]  (auto: atoms of the first training frame)"""
import json, sys, pickle
from bam_omat24.training_utils import load_checkpoint, save_checkpoint, gauge_shift_params
for run in sys.argv[2:]:
    if sys.argv[1] == "auto":
        cfg = json.load(open(f"input/input_3bpa_omat24_{run}_cue.json"))
        n_at = int(open(cfg["train_path"]).readline())
    else:
        n_at = int(sys.argv[1])
    c = json.load(open(f"runs/{run}_cue/gauge_refit.json"))["shift_delta_meV_per_frame"]
    ck = load_checkpoint(f"runs/{run}_cue/ckpt_{run}_cue.pkl")
    delta = c * 1e-3 / n_at
    ck["ema_params"] = gauge_shift_params(ck["ema_params"], delta); ck["gauge_shift_delta"] = delta
    save_checkpoint(ck, f"runs/{run}_cue/ckpt_{run}_cue.gauge.pkl")
    print(run, f"per-atom shift {delta*1e3:+.4f} meV")
