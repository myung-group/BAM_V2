#!/bin/bash
#SBATCH -p normal
#SBATCH --comment="field=sciai;appl=jax"
#SBATCH --gres=gpu:3
#SBATCH -t 02:00:00
#SBATCH -J soapgchk
#SBATCH -o runs/slurm/%x_%j.out
# grouped SOAP: equivalence tests (GPU 0) and 30-epoch timing runs, Householder (GPU 1) / CholeskyQR3 (GPU 2)
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
(CUDA_VISIBLE_DEVICES=0 python -m pytest -q -p no:warnings tests/test_soap_grouped.py tests/test_soap.py > research/diag/soap_grouped_pytest.txt 2>&1) &
CUDA_VISIBLE_DEVICES=1 bash runs/run_cue.sh r29_soapg_s42 &
CUDA_VISIBLE_DEVICES=2 bash runs/run_cue.sh r29_soapg_chol_s42 &
wait
tail -3 research/diag/soap_grouped_pytest.txt
for n in r29_soapg_s42 r29_soapg_chol_s42; do tail -1 runs/${n}_cue/stdout.txt; done
