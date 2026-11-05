# BAM(V2)

E(3)-equivariant graph neural network interatomic potentials with uncertainty quantification, available in three backends:

| Package | Framework | Equivariance library | Best for |
|---|---|---|---|
| `bam_omat24` | JAX + Flax NNX | e3nn-jax | Large-scale training (TPU/multi-host) |
| `bam_torch` | PyTorch | e3nn | GPU training, standard workflows |
| `bam_torch_cue` | PyTorch | cuequivariance | CUDA-optimized, fastest GPU inference |

All backends implement the same model hierarchy — **NequIP → RACE**  — and share the same training interface.

## RACEv2 (branch `racev2`, 2026-11)

RACEv2는 `bam_omat24`(JAX) 위에 만든 RACE 개선판이다. 모든 결과는 각 데이터셋의 공식 학습셋만 사용했고, 사전학습·추가 데이터·데이터 증강은 쓰지 않았다.
자세한 수식·표·그림은 [`docs/RACEv2_methods_summary.pdf`](docs/RACEv2_methods_summary.pdf), 구조 그림은 [`docs/RACEv2_architecture.png`](docs/RACEv2_architecture.png)에 있다.

![RACEv2 architecture](docs/RACEv2_architecture.png)

### V1 대비 바뀐 점 (최종 레시피)

| 항목 | RACE V1 | RACEv2 |
|---|---|---|
| 원자 특징 x | 원소 임베딩 (`x_mode=species`) | 이웃 요약에서 층마다 선형 사영 (`x_mode=basis`) |
| x 크기 조절 | 없음 | hill: x/r · 2σ(a log r + b), a·b는 층×원소별 학습값 (`x_norm=hill`) |
| 층 갱신 | 단순 잔차 | Chebyshev식 h⁽ᵗ⁾ = 2γ⊙δ⁽ᵗ⁾ − h⁽ᵗ⁻²⁾, LayerScale 0.5 (`residual=chebyshev`) |
| 메시지 정규화 | LayerNorm + 1/√⟨N⟩ | 평균 이웃 수로 나눔 (`message_norm=avg`) |
| 갱신 / readout | gate / linear | linear / gated |
| 거리 함수 | Bessel | Bessel + 원소 껍질 반지름 채널, Gaussian 고차 진동 억제 prior (σ=0.224) |
| 반발 항 | 없음 | UFF 비결합 반발(형태 고정, 결합 쌍은 부드럽게 끔), 전역 배율 s 학습 |
| 학습 | Adam | SOAP (lr 1e-2, β 0.95), Huber δ=10, EMA 0.99, float64, 학습 후 에너지 상수 보정 δ* |

기본 크기는 5층, lmax 3, 컷오프 5 Å, hidden 128(rMD17·MD22 256, 물 64).

### 결과 요약 (3시드 평균, 검증 기준 체크포인트)

**3BPA** (RMSE, E meV/분자 / F meV/Å, 300 K 500프레임 학습)

| 모델 | 300 K | 600 K | 1200 K | 이면각 |
|---|---|---|---|---|
| **RACEv2** | **1.56 / 6.31** | **6.05 / 15.92** | **22.20 / 45.16** | 16.67 / 19.09 |
| RACE V1 | 3.4 / 12.1 | 11.7 / 31.8 | 37.5 / 115.3 | — |
| GRACE (PRX 2024) | 1.7 / 7.6 | 7.3 / 19.2 | 25.8 / 61.7 | 7.4 / 15.3 |
| TACE | 2.69 / 8.68 | 8.10 / 19.88 | 23.42 / 54.26 | 11.0 / 16.45 |
| MACE | 3.0 / 8.8 | 9.7 / 21.8 | 29.8 / 62.0 | 7.8 / 16.5 |

**rMD17** (MAE, 공식 분할, 학습 1000, L5h256): 힘은 10개 분자 중 8승 1무 1패, 에너지는 7승 2무 1패(발표된 최고값 대비).

| 분자 | aspirin | azobenzene | benzene | ethanol | malonaldehyde | naphthalene | paracetamol | salicylic | toluene | uracil |
|---|---|---|---|---|---|---|---|---|---|---|
| E (meV) | 1.19 | 0.36 | 0.02 | 0.21 | 0.36 | 0.12 | 0.61 | 0.36 | 0.13 | 0.20 |
| F (meV/Å) | 4.01 | 1.41 | 0.22 | 1.41 | 2.49 | 0.65 | 2.72 | 2.03 | 0.78 | 1.43 |
| MACE F | 6.6 | 3.0 | 0.3 | 2.1 | 4.1 | 1.6 | 4.8 | 3.1 | 1.5 | 2.1 |

**기타 벤치마크**

| 벤치마크 | RACEv2 | 참조값 |
|---|---|---|
| AcAc 300 K / 600 K (RMSE E/F) | 0.46/3.06 · 5.32/15.55 | PACE 0.73/4.2 · 3.8/16.9 |
| Si GAP-18 → RSS (RMSE meV/atom, meV/Å) | 15.65 / 106.7 | XPOT-ACE 20.5 / 139 |
| Si GAP-18 → 압축 MD | 4.47 / 69.0 | GAP-18 15.7 / 125 |
| LiPS 19k (MAE meV/atom, meV/Å) | 0.020 / 1.20 | FreeCG 힘 1.2, GemNet-T 1.3 |
| LiPS 학습곡선 2500/1000/100/10 (힘 MAE) | 2.02 / 3.17 / 9.46 / 34.1 | NequIP 4.7, CarNet 5.6, CAMP 21.6 / 76.4 |
| LiPS MD 50 ps × 5 (안정성 / RDF / Li 확산 오차) | 50 ps / 0.044 / 0.43 | NequIP 50 / 0.04 / 0.34, GemNet-T 0.24 |
| 물 분할 1 / 2 (RMSE meV/H₂O, meV/Å, 반발 항 재계산 진행 중) | 3.38/63.6 · 0.64/21.96 | 같은 분할 MACE 64-0 4.5/64.8 · 0.9/23.2 |

진행 중: 3BPA ablation(4/10 변형 완료, PDF 표 6b), 물 반발 항 재계산, MD22, QM9.

### 실행

```bash
# 학습 (설정 예: input/input_3bpa_omat24_abl3_full_s42_cue.json)
python -m bam_omat24.train_3bpa input/<config>.json
# LiPS MD (Fu et al. 2023 프로토콜) 와 관측량 분석
python -m bam_omat24.md_run --help
python -m bam_omat24.md_lips_analysis <md_output_dir>
# 문서·그림 재생성 (docs/fonts/ 에 NanumGothic 폰트 필요, 저장소에는 미포함)
python docs/make_methods_pdf.py
```

HANGANG Slurm 스크립트 예시는 `runs/diag/*.sh`에 있다. GPU 노드에서는 `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`가 필요하다(float64 GEMM 오컴파일 회피).

## Overview

A modified version of RACE is an E(3)-equivariant graph neural network architecture for learning interatomic potentials. This implementation provides:

- **Three model tiers**: NequIP (baseline), RACE (per-layer readouts + x-features)
- **Uncertainty quantification**: Learnable variance heads with Cholesky NLL loss
- **Fine-tuning**: Delta-learning wrapper (`FineTuningPotential`) around frozen foundation models
- **Multi-GPU/multi-node**: Distributed training via JAX pmap (JAX backend) or PyTorch DDP (Torch backends)
- **EMA weights**: Exponential moving average for stable training

## Features

- E(3)-equivariant message passing with spherical harmonics
- Bessel radial basis functions with polynomial cutoff
- Per-species embeddings and skip connections
- Separated layer normalization for equivariant features
- Huber loss and Cholesky NLL loss for robust training
- Learning rate scheduling with reduce-on-plateau
- Checkpoint saving/loading

## Installation

```bash
# Clone the repository
git clone https://github.com/myung-group/BAM_V2.git
cd BAM_V2
```

Install only the backend(s) you need:

```bash
# JAX backend (bam_omat24, bam_3bpa)
pip install -e ".[jax]"

# PyTorch backend with e3nn (bam_torch)
pip install -e ".[torch]"

# PyTorch backend with cuequivariance — requires CUDA (bam_torch_cue)
pip install -e ".[torch-cue]"

# All backends
pip install -e ".[all]"
```

### Requirements

**Shared**
- Python >= 3.12
- ASE >= 3.24.0
- matscipy >= 1.2.0
- numpy >= 2.2.6

**JAX backend** (`bam_omat24`, `bam_3bpa`)
- JAX >= 0.8.2
- Flax >= 0.12.0
- e3nn-jax >= 0.20.7
- jraph >= 0.0.6
- optax >= 0.2.4

**PyTorch backend** (`bam_torch`)
- PyTorch >= 2.10
- e3nn >= 0.5.0

**cuequivariance backend** (`bam_torch_cue`)
- PyTorch >= 2.10
- cuequivariance >= 0.9.0
- cuequivariance-torch >= 0.9.0
- einops >= 0.7.0

## Usage

### JAX backend (`bam_omat24`)

1. Prepare your dataset in extended XYZ format with energy and forces.
2. Create a configuration file (see `input/input_3bpa.json` for an example):

```json
{
    "cutoff": 5.0,
    "train_path": "data/train.xyz",
    "atomic_numbers": [1, 6, 7, 8],
    "atom_energies": [-13.587, -1029.489, -1484.981, -2041.982],
    "hidden_size": 128,
    "lmax": 3,
    "n_layers": 5,
    "batch_size": 5,
    "n_epochs": 1000,
    "learning_rate": 0.01
}
```

3. Run training:

```bash
python -m bam_3bpa.train_3bpa
```

### PyTorch backends (`bam_torch` / `bam_torch_cue`)

Create a JSON config selecting your model and loss type:

```json
{
    "model": "RACE",
    "lmax": 3,
    "cutoff": 5.0,
    "hidden_irreps": "64x0e + 64x1o + 64x2e",
    "n_layers": 5,
    "radial_basis_size": 8,
    "radial_mlp_size": 64,
    "radial_mlp_layers": 3,
    "avg_n_neighbors": 25.0,
    "periodic": true,
    "loss_type": "nll"
}
```

Train on a single GPU:

```bash
# e3nn backend
python -m bam_torch.train input.json

# cuequivariance backend (requires CUDA)
python -m bam_torch_cue.train input.json
```

Multi-GPU training via `torchrun`:

```bash
torchrun --nproc_per_node=4 -m bam_torch.train input.json
```

Fine-tune a pretrained RACE checkpoint with delta-learning:

```bash
python -m bam_torch.train input_finetune.json --finetune path/to/checkpoint.pt
```

### Model Architecture

All three model tiers share the same backbone:

1. **Species embedding**: Per-species linear projection to hidden irreps
2. **Interaction layers**: Equivariant convolutions with:
   - Linear transformation of node features
   - Tensor product with spherical harmonics of edge vectors
   - Radial MLP for distance-dependent weights
   - Gated nonlinearity
   - Per-species skip connections
3. **Readout**: Linear projections to scalar per-atom energies

**NequIP**: single readout after all layers.  
**RACE**: per-layer energy readout accumulated across layers, plus a second tensor product with scalar x-features.  
**BRACE** (planned, JAX): MACE-style symmetric contractions via `e3nn.reduced_symmetric_tensor_product_basis` — not implemented yet.

**FineTuningPotential** wraps a frozen RACE checkpoint with trainable affine (scale + shift) and variance heads for delta-learning on new systems.

### Equivariance backends

| | `bam_torch` | `bam_torch_cue` |
|---|---|---|
| Library | e3nn | cuequivariance (NVIDIA) |
| Device | CPU + GPU | CUDA only |
| Tensor product | `e3nn.o3.TensorProduct` | `cuet.ChannelWiseTensorProduct` |
| Linear | `e3nn.o3.Linear` | `cuet.Linear` |
| Layout | e3nn default | `ir_mul` |
| Relative speed | baseline | ~10× faster on CUDA |

### Configuration Options

| Parameter           | Description                        | Default | Backends     |
| ------------------- | ---------------------------------- | ------- | ------------ |
| `cutoff`            | Radial cutoff distance (Å)         | 5.0     | all          |
| `hidden_size`       | Hidden feature dimension (JAX)     | 128     | JAX          |
| `hidden_irreps`     | Hidden irreps string (Torch)       | —       | Torch        |
| `lmax`              | Maximum angular momentum           | 3       | all          |
| `n_layers`          | Number of interaction layers       | 5       | all          |
| `radial_basis_size` | Number of Bessel basis functions   | 8       | all          |
| `radial_mlp_size`   | Hidden size of radial MLP          | 64      | all          |
| `radial_mlp_layers` | Number of radial MLP layers        | 3       | all          |
| `loss_type`         | `"huber"` or `"nll"` (Cholesky)   | huber   | Torch        |
| `learning_rate`     | Initial learning rate              | 0.01    | all          |
| `energy_weight`     | Loss weight for energy             | 1.0     | all          |
| `force_weight`      | Loss weight for forces             | 1.0     | all          |
| `use_scan`          | Use `jax.lax.scan` for training    | true    | JAX          |
| `grad_clip`         | Gradient clipping norm             | 1.0     | all          |
| `ema_decay`         | EMA decay rate                     | 0.99    | all          |

RACE options added in `bam_omat24` (JAX). Defaults reproduce the original behaviour; see
`docs/RACE_omat24_review.md` for details.

| Parameter           | Description                                                                 | Default     |
| ------------------- | --------------------------------------------------------------------------- | ----------- |
| `x_mode`            | `"species"` (per-element scalar x-feature) or `"node_scalar"` (per-layer projection of the node's own 0e features) | `species` |
| `cov_mode`          | Force covariance: `"scalar6"` (6 invariant scalars → Cholesky, **not** rotation equivariant) or `"irreps"` (`1x0e + 1x2e`, Σ = (aI + B)², equivariant) | `scalar6` |
| `loss_type`         | `"huber"`, `"nll"` (legacy Huber-whitened), `"gauss_nll"` (exact Gaussian NLL) | `huber`   |
| `beta_nll`          | β of β-NLL weighting (only `gauss_nll`)                                     | 0.0         |
| `use_dataset_stats` | Initialise `shift` (per-atom mean energy), `scale` (force RMS) and `avg_n_neighbors` from the training set | false |
| `e0_mode`           | `"isolated"` (config `atom_energies`) or `"regression"` (per-species least squares on the training energies) | `isolated` |
| `loss_schedule`     | List of stages `{"until_epoch": N, "energy_weight": .., "force_weight": ..}` (best-checkpoint tracking restarts at each stage) | none |
| `node_bucket_size` / `edge_bucket_size` | Padding granularity of batches (smaller = less wasted compute, more JIT shapes) | 128 / 4096 |

3BPA with the `bam_omat24` backend:

```bash
python -m bam_omat24.train_3bpa input/input_3bpa_omat24_baseline.json
pytest tests            # equivariance / forces / stats / x_mode
```


### Comparison with other codes

In order to verify correctness of the implementation, we compare performance on
the 3BPA dataset to two different PyTorch NequIP implementations:

1. The results from Musaelian et al. in ["Learning local equivariant

representations for large-scale atomistic
dynamics"]([https://www.nature.com/articles/s41467-023-36329-y](https://www.nature.com/articles/s41467-023-36329-y)), using a version
of their `[nequip](https://github.com/mir-group/nequip)` repository.
2. The
results from Batatia et al. in ["MACE: Higher Order Equivariant Message
Passing Neural Networks for Fast and Accurate Force
Fields"](https://arxiv.org/abs/2206.07697), using a version of their `[mace](https://github.com/ACEsuit/mace)` repository.

We use the same hyperparameters as [1], which can be viewed in `[input/input_3bpa.json](input/input_3bpa.json)`, with the following exceptions:

- Instead of hidden irreps of `64x0e + 64x0o + 64x1o + 64x1e + 64x2e + 64x2o +  64x3o + 64x3e`, we opt for `128x0e + 128x1o + 128x2e + 128x3o` for simplicity
 with the same feature dimensions.
- We use Kaiming normal, i.e. sampling from $\mathcal{N}(0, \mathrm{std})$ with $\mathrm{std} =
 \sqrt{\frac{4.0}{\mathrm{fan_in}}}$.
- Isolated atom energies are added to each predicted node energy.

Resulting energy (E) and force (F) RMSE in meV and meV/Å respectively.


| Code     | `[nequip](https://github.com/mir-group/nequip)` | `[mace](https://github.com/ACEsuit/mace)` | `BAM-nequip` (this repo) | `BAM-race` (this repo) |
| -------- | ----------------------------------------------- | ----------------------------------------- | ------------------------ | ---------------------- |
| 300 K E  | 3.3 (0.1)                                       | 3.1 (0.1)                                 | 3.7                      | **3.2**                |
| 300 K F  | 10.8 (0.2)                                      | 11.3 (0.2)                                | 9.6                      | **8.8**                |
| 600 K E  | 11.2 (0.1)                                      | 11.3 (0.3)                                | 11.1                     | **10.1**               |
| 600 K F  | 26.4 (0.1)                                      | 27.3 (0.3)                                | 23.5                     | **22.3**               |
| 1200 K E | 38.5 (1.6)                                      | 40.8 (1.3)                                | 33.5                     | **31.4**               |
| 1200 K F | 76.2 (1.1)                                      | 86.4 (1.5)                                | **67.6**                 | 72.9                   |


### Example: Inference

**JAX backend** (`bam_omat24`):

```python
from bam_omat24.race_nnx import RACE
from flax import nnx

model = RACE(
    n_species=95,
    lmax=3,
    cutoff=5.0,
    hidden_size=128,
    n_layers=5,
    rngs=nnx.Rngs(42),
)
energies, forces, stress = model(graph_data)
```

**PyTorch backend** (`bam_torch`):

```python
from bam_torch.race import RACE

model = RACE(config).cuda()
out = model(graph_data)  # dict with energy, forces, stress (+ variance)
```

**cuequivariance backend** (`bam_torch_cue`):

```python
from bam_torch_cue.race import RACE

model = RACE(config).cuda()  # same API as bam_torch
out = model(graph_data)
```

## Citation

If you use this code, please cite the original NequIP paper:

```bibtex
@article{batzner2022nequip,
  title={E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials},
  author={Batzner, Simon and Musaelian, Albert and Sun, Lixin and Geiger, Mario and Mailoa, Jonathan P and Kornbluth, Mordechai and Molinari, Nicola and Smidt, Tess E and Kozinsky, Boris},
  journal={Nature Communications},
  volume={13},
  number={1},
  pages={2453},
  year={2022},
  publisher={Nature Publishing Group}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [e3nn-jax](https://github.com/e3nn/e3nn-jax) for equivariant neural network primitives (JAX backend)
- [e3nn](https://github.com/e3nn/e3nn) for equivariant neural network primitives (PyTorch backend)
- [cuequivariance](https://github.com/NVIDIA/cuEquivariance) for CUDA-optimized equivariant operations (`bam_torch_cue`)
- [jraph](https://github.com/deepmind/jraph) for graph neural network utilities
- Original [NequIP](https://github.com/mir-group/nequip) implementation
- [mariogeiger/nequip-jax](https://github.com/mariogeiger/nequip-jax): NequIP-style model in JAX
- [teddykoker/nequip-eqx](https://github.com/teddykoker/nequip-eqx): NequIP-style model in JAX and equinox

