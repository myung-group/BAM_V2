#!/bin/bash
#SBATCH -p normal
#SBATCH --comment="field=sciai;appl=jax"
#SBATCH -o runs/slurm/%x_%j.out
# Print the trained global energy scale / shift of final checkpoints (the UFF repulsion stratum is
# multiplied by `scale` inside RACE.node_energies). Needs JAX to unpickle, so it runs on a compute node.
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
cd /scratch/t510a13/race_v2/work
python - <<'EOF' | tee research/diag/check_scale.log
import pickle, numpy as np, jax
names = ['r37_uffw03_s42', 'r37_uffw03_s43', 'r37_uffw03_s44', 'siF_s42', 'siF_s43', 'siF_s44',
         'rmd17f_aspirin_uffw03_s42', 'rmd17f_ethanol_uffw03_s42', 'rmd17f_benzene_uffw03_s42',
         'acacf_uffw03_s42', 'lipsF_fu_s42', 'wF2_s42']
for n in names:
    try:
        c = pickle.load(open(f'runs/{n}_cue/ckpt_{n}_cue.pkl', 'rb'))
        p = c['ema_params']
        leaves = jax.tree_util.tree_flatten_with_path(p)[0]
        hits = []
        for path, v in leaves:
            keys = [getattr(k, 'key', getattr(k, 'name', getattr(k, 'idx', k))) for k in path]
            top = [str(k) for k in keys if str(k) not in ('value', 'raw_value')]
            if len(top) == 1 and top[0] in ('scale', 'shift'):
                hits.append(f"{top[0]}={np.asarray(v).ravel()}")
        print(f"{n:30s} " + "  ".join(hits) if hits else f"{n:30s} no top-level scale/shift; first paths: {[str(l[0]) for l in leaves[:3]]}")
    except Exception as e:
        print(n, 'ERR', type(e).__name__, e)
EOF
