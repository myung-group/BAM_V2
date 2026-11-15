# RACEv2 → LAMMPS (방법 2: 모델을 내보내 C++에서 직접 실행) 계획, 2026-11-06

## 목표

학습된 JAX 모델(pkl)을 GPU용 컴파일 가능한 파일로 내보내고, LAMMPS의 C++ pair style에서 파이썬 없이 실행한다.
NequIP / Allegro의 `pair_style nequip/allegro`(TorchScript + libtorch)와 같은 구조에서, libtorch 대신 XLA의 PJRT C API를 쓴다.

## 왜 이 경로인가 (환경 확인 결과)

| 필요한 것 | 현재 환경 (bam_cue) | 비고 |
|---|---|---|
| JAX 함수를 파일로 내보내기 | `jax.export` (jax 0.12) | StableHLO 바이트코드, 버전 호환 보장 형식 |
| C/C++에서 GPU로 실행 | `jax_plugins/xla_cuda12/xla_cuda_plugin.so` (`GetPjrtApi` 공개) | pip로 이미 설치된 PJRT CUDA 플러그인. 별도 XLA 빌드 불필요 |
| cue 커널 | `cuequivariance_ops_jax/lib/libcue_ops_jax.so` (FFI 핸들러 C 심볼) | 내보낸 모델에 custom_call로 남으면 C++에서 같은 이름으로 다시 등록해야 함 |
| e3nn 백엔드 | 같은 파라미터로 cue와 같은 결과 (tests/test_tp_backend_cue.py) | 순수 XLA라 custom_call 없음 → 1차 목표 |

HANGANG에는 LAMMPS 모듈이 없어서 LAMMPS는 소스에서 빌드한다(GPU 노드가 aarch64이므로 glogin에서, 병렬 빌드 16 이하).

## 핵심 설계: 원자 쌍 벡터를 입력으로

LAMMPS가 이미 만든 이웃 목록의 쌍 벡터 r_ij = x_j − x_i 를 그대로 모델에 넣고, 에너지와 dE/dr_ij를 돌려받는다.
모델 코드는 바꾸지 않는다. 원자 위치 0, 셀 = 단위행렬, Sij = −r_ij 로 주면 RACEv2가 계산하는 간선 벡터가 정확히 r_ij가 된다
(RACEv2는 절대 위치를 쓰지 않음).

- 힘: F_i += g_ij, F_j −= g_ij (g = dE/dr_ij)
- 비리얼: W = −Σ r_ij ⊗ g_ij, 응력 = −W/V
- 패딩: 고정 크기 묶음(원자 n_pad, 간선 e_pad)으로 내보내고, 남는 간선은 길이 0으로 패딩 원자에 붙인다
  (길이 0 간선은 메시지를 보내지 않도록 2026-11-06에 고침). 크기 묶음 몇 개를 미리 내보내고 실행 시 가장 작은 것을 고른다.
- 원소 기준 에너지 E0는 메타데이터에 넣고 C++에서 더한다.

## 단계

1. **내보내기와 검증 (파이썬, GPU 작업 1개)** — `race_v2/export/lammps.py`, `research/diag/export_lammps.sh`
   - LiPS 최종 모델(주기계, float64)을 e3nn / cue 백엔드로 각각 내보낸다.
   - 내보낸 함수를 다시 불러 LiPS 시험 구조 20개에서 원래 모델과 에너지·힘·응력을 비교한다(float64라 1e-10 수준 기대).
   - cue 버전의 custom_call 이름 목록을 기록한다.
2. **C++ 실행기 (LAMMPS 없이)** — PJRT C API로 `.mlirbc`를 컴파일·실행하는 작은 라이브러리와 시험 프로그램.
   - 필요한 것: `pjrt_c_api.h` (openxla/xla 저장소, jaxlib 0.12와 맞는 버전; 데이터 이동 노드에서 받음).
   - 시험: 같은 LiPS 구조를 C++에서 계산해 1단계 결과와 비교.
   - cue: `libcue_ops_jax.so`의 핸들러를 PJRT FFI 확장으로 등록해 cue 버전도 실행되는지 확인.
3. **LAMMPS pair style `racev2` (단일 MPI 프로세스, GPU 1장)**
   - full 이웃 목록 → 간선 목록. 이웃이 ghost 원자이면 `atom->map(tag)`로 실제 원자 번호로 바꿔, 학습 때와 같은 주기 그래프를 만든다.
     프로세스가 하나면 모든 ghost가 실제 원자의 상(image)이므로 5층 메시지 전달도 정확하다.
   - 에너지, 힘, 비리얼(위 식) 계산. `run 0`과 NVE MD로 파이썬 결과·에너지 보존 확인. 우리 JAX MD와 속도 비교.
4. **여러 프로세스 / 여러 GPU** (필요할 때)
   - 공간을 나누면 경계에서 5층 × 5 Å = 25 Å 까지의 정보가 필요하다.
   - 쉬운 방법: ghost 거리를 25 Å로 늘리고 ghost 원자도 그래프에 넣되 에너지는 실제 원자만 합산. 계산량이 커진다.
   - 효율적인 방법: 층마다 경계 원자의 특징값을 MPI로 교환(모델을 층 단위로 나눠 내보내야 함). 구현 부담이 크다.

## 위험 요소

- PJRT C API 헤더 버전이 플러그인과 맞지 않으면 실행 실패 → jaxlib 0.12.0.1의 XLA 커밋에 맞는 헤더를 쓴다.
- 크기 묶음 밖의 큰 구조 → 묶음 목록을 넉넉히 만들거나 실행 시 새로 내보내기 요청.
- float64 고정. float32 추론은 정확도 영향 확인 후 별도 내보내기(SOAP 문제는 학습에만 해당).
- cue custom_call 등록이 안 되면 e3nn 버전으로 진행(속도만 손해).
