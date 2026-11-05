#!/bin/bash
#SBATCH -p normal
#SBATCH --comment="field=sciai;appl=jax"
#SBATCH -o runs/slurm/%x_%j.out
# Print the trained global energy scale / shift of final checkpoints (the UFF repulsion stratum is
# multiplied by `scale` inside RACE.node_energies). Needs JAX to unpickle, so it runs on a compute node.
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
cd /scratch/t589a02/BAM_V2
python - <<'EOF' | tee runs/diag/check_scale_noprior.log
import pickle, numpy as np, jax
names = ['r28_soap_s42', 'r28_soap_s43', 'wF1_s43', 'w7_h64_lr1e3', 'acac_soap_s42', 'si_full_hill_soap_s42', 'lips_fu_hill_soap_s42', 'r33_zblnb_s42',
         'rmd17o_aspirin_soap_s42']
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
