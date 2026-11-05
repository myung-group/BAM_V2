#!/bin/bash
#SBATCH -p normal
#SBATCH --comment="field=sciai;appl=jax"
#SBATCH --gres=gpu:1
#SBATCH -t 00:40:00
#SBATCH -J ew_dih
#SBATCH -o runs/slurm/%x_%j.out
# (1) dihedral + train errors of the SOAP energy-weight sweep (r27_soap_ew*, 500 ep) -> does energy weight fix the collision bias?
# (2) parameter-norm probe of lr 1e-2 vs 3e-3 vs v9a checkpoints (radial MLP extrapolation hypothesis)
cd /scratch/t589a02/BAM_V2
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
B=/scratch/t589a02/datasets_ext/BOTNet-datasets/dataset_3BPA
for N in r27_soap_ew3_s42 r27_soap_ew10_s42 r27_soap_ew30_s42 r27_soap_ew100_s42; do
  python -m bam_omat24.dump_frame_errors input/input_3bpa_omat24_${N}_cue.json --ckpt runs/${N}_cue/ckpt_${N}_cue.pkl --per-atom --index-key none --batch-size 32 \
    --xyz dataset_3BPA/train_300K.xyz --xyz $B/test_dih.xyz --out runs/diag/wd/$N 2>&1 | grep -E "MAE|Error|Traceback"
done
python - <<'PY'
import pickle, json, numpy as np, jax
def norms(path):
    ck=pickle.load(open(path,'rb')); p=ck['ema_params']
    out={}
    for kp,v in jax.tree_util.tree_leaves_with_path(p):
        k='/'.join(str(getattr(x,'key',getattr(x,'idx',x))) for x in kp); a=np.asarray(v)
        out[k]=dict(shape=list(a.shape), fro=float(np.linalg.norm(a)), max=float(np.abs(a).max()))
    return out
R={n:norms(f'runs/{n}_cue/ckpt_{n}_cue.gauge.pkl') for n in ['r28_soap_s42','r28_soap_lr3e3_s42','r31_base','r31_no_shells']}
R['v9a_rp_s42']=norms('runs/v9a_rp_s42_cue/ckpt_v9a_rp_s42_cue.pkl')
json.dump(R,open('runs/diag/wd/param_norms.json','w'))
print('param norms written', {k:len(v) for k,v in R.items()})
PY
