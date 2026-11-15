#!/bin/bash
#SBATCH -p normal
#SBATCH --comment="field=sciai;appl=jax"
#SBATCH -o runs/slurm/%x_%j.out
# Si final (siF_s42/43/44, 300 ep): if the validation force RMSE was still falling at the end, raise n_epochs
# and continue from the .last.pkl checkpoint (restart: true).  Rule (validation only): min over the last 50
# epochs < 0.98 x min over epochs 200-250  ->  extend by EXTRA epochs (repeatable: the rule is re-applied).
# usage: sbatch -J siX --dependency=afterany:<siF jobid> --gres=gpu:3 -t 12:00:00 research/diag/si_extend.sh
cd /scratch/t510a13/race_v2/work
EXTRA=${EXTRA:-200}
PY=~/.conda/envs/bam_cue/bin/python
todo=()
for s in 42 43 44; do
  name=siF_s$s
  dec=$($PY - "$name" "$EXTRA" <<'EOF'
import sys, re, json, numpy as np
name, extra = sys.argv[1], int(sys.argv[2])
cfgp = f"research/input/input_3bpa_omat24_{name}_cue.json"; cfg = json.load(open(cfgp))
rows = []
for l in open(f"runs/{name}_cue/loss_{name}_cue.out"):
    m = re.match(r"\d\d/\d\d/\d{4} \d\d:\d\d:\d\d\s+(\d+)\s+[\d.]+\s+\|", l)
    if m:
        v = l.split("|")[2].split(); rows.append((int(m.group(1)), float(v[2])))
a = np.array(rows); n_end = int(cfg["n_epochs"])
done = a[:, 0].max() if len(a) else 0
if done < n_end:
    print(f"incomplete {done}/{n_end}"); sys.exit(0)
ref = a[(a[:, 0] > n_end - 100) & (a[:, 0] <= n_end - 50), 1].min()
last = a[a[:, 0] > n_end - 50, 1].min()
if last < 0.98 * ref:
    cfg["n_epochs"] = n_end + extra; json.dump(cfg, open(cfgp, "w"), indent=4)
    print(f"extend {ref:.2f}->{last:.2f} n_epochs {n_end}->{n_end + extra}")
else:
    print(f"converged {ref:.2f}->{last:.2f} at {n_end}")
EOF
)
  echo "$name: $dec" | tee -a research/diag/si_extend.log
  case "$dec" in extend*) todo+=($name);; esac
done
[ ${#todo[@]} -eq 0 ] && { echo "nothing to extend"; exit 0; }
i=0
for NAME in "${todo[@]}"; do
  CUDA_VISIBLE_DEVICES=$i bash runs/run_cue.sh $NAME &
  i=$((i+1))
done
wait
for NAME in "${todo[@]}"; do echo "$NAME: $(tail -1 runs/${NAME}_cue/stdout.txt)"; done
