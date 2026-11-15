#!/bin/bash
# pair_style race mode halo (per-layer halo exchange) on one node (2026-11-06): export with staged functions, then
# LiPS 2x2x2 (664 atoms) and 4x4x4 (5312 atoms): run 0 + NVE, 1 rank owner (reference) vs halo on 1, 2, 4 ranks.
# usage: sbatch -J rv2_halo -p normal --comment="field=sciai;appl=jax" -N 1 --ntasks-per-node=4 --gres=gpu:4 \
#          -t 02:00:00 research/diag/lammps_halo.sh
#SBATCH -o runs/slurm/%x_%j.out
ulimit -c 0
cd /scratch/t510a13/race_v2/work
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
module load PrgEnv-gnu cray-mpich >/dev/null 2>&1
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
export BAM_INDEXED_LINEAR=ragged
export RACEV2_TIMING=1
OUT=${OUT:-runs/lammps_export/lips_cue_halo4}
if [ ! -f $OUT/racev2_lammps_meta.txt ] || ! grep -q staged_bucket $OUT/racev2_lammps_meta.txt; then
CUDA_VISIBLE_DEVICES=0 python -m race_v2.export.lammps research/input/input_3bpa_omat24_lipsF_fu_s42_cue.json \
  --ckpt runs/lipsF_fu_s42_cue/ckpt_lipsF_fu_s42_cue.gauge.pkl --out $OUT --backend cue \
  --buckets 128:4096 1024:32768 8192:262144 --staged-buckets 1024:8192 1024:16384 1024:32768 2048:8192 2048:16384 2048:32768 4096:32768 2048:65536 4096:65536 4096:131072 6144:131072 8192:131072 8192:262144 12288:262144 16384:524288 \
  2>&1 | grep -a -E "exported|recorded|written|Error|error|Traceback"
fi
export LD_LIBRARY_PATH=$(ls -d $SP/nvidia/*/lib | tr '\n' ':')$SP/cuequivariance_ops/lib:${LD_LIBRARY_PATH}
export PATH=$SP/nvidia/cuda_nvcc/bin:$PATH
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_cuda_data_dir=$SP/nvidia/cuda_nvcc"
for REP in "2 2 2" "4 4 4"; do
python research/diag/lammps_multi.py --serial /scratch/t510a13/src/lammps-racev2-build/lmp_racev2 \
  --mpi /scratch/t510a13/src/lammps-racev2-mpi-build/lmp_racev2_mpi \
  --plugin $SP/jax_plugins/xla_cuda12/xla_cuda_plugin.so --export $OUT --modes halo --tag halo \
  --xyz /scratch/t510a13/data/datasets_ext/lips/fu_test.xyz --rep $REP --ranks 4 --steps 200 --skip-serial-ghost
done
