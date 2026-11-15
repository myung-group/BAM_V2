# 인계 문서: SOAP 옵티마이저 속도 관련 수정과 측정 (2026-11-11, HANGANG)

다른 Claude Code 세션이 이 문서만 읽고 무엇이 바뀌었고 무엇이 확인됐는지 알 수 있게 쓴다.
대상 코드는 브랜치 `thole-merge`(작업 폴더 `/scratch/t510a13/BAM_V2_wt_thole`)다. GitHub `racev2`에는 아직 없다.
이전 분석은 `docs/RACE_SOAP_speedup_plan_2026-10-29.md`에 있고, 이 문서는 그 뒤의 일을 다룬다.

## 0. 한 줄 요약

- SOAP 알고리즘 자체(`soap_grouped.py`, `soap_optax.py`)는 **고치지 않았다.**
- 고친 것은 두 가지다. (1) 기저 갱신을 GPU들이 나눠 계산하는 기존 코드를 학습기에 연결했다(옵션, 기본 꺼짐).
  (2) "모델 float32 + SOAP 상태 float64" 모드가 실제로는 float64로 돌던 버그를 고쳤다.
- 이득은 (2)가 압도적이다: **스텝 시간 1.82배, GPU 메모리 절반.** (1)은 4장에서 2 %다.

## 1. 시간이 어디에 드는가 (측정)

모델: racelr_q, 5층 × 64채널, lmax 3, 파라미터 1,228만 개, OMol25 4M, GPU 1장, 64개 구조/스텝.
작업 1258956(`jobs/soap_profile.sh`). 값은 200스텝 구간 2~3번째의 평균이다.

| 설정 | 스텝당 시간 | GPU 메모리 | 비고 |
|---|---|---|---|
| SOAP float64, Householder QR, 갱신 10스텝마다 (기준) | 210 ms | 42 GB | 지금까지의 레시피 |
| `soap_orth: "cholqr"` | 195 ms (−6.9 %) | 42 GB | 600스텝 뒤 검증 힘 RMSE 307.9 (기준 313.6) |
| `soap_precondition_frequency: 50` | 197 ms (−6.3 %) | 42 GB | 수식이 바뀜. 308.0 |
| 갱신 없음 (주기 10^6, 진단용) | 194 ms (−7.5 %) | 42 GB | 학습이 느려짐: 346.6 |
| AMSGrad float64 | 182 ms | 41 GB | SOAP 전체 비용 = 210 − 182 = 28 ms |
| AMSGrad float32 | 91 ms | 20 GB | float64가 모델 계산을 2배로 만든다 |
| SOAP float64, 128개 구조/스텝 | 342 ms | 80 GB | 처리량 +22 % (296 → 363 구조/초) |

스텝 210 ms의 분해:

| 부분 | 시간 | 비중 |
|---|---|---|
| 모델 계산 (float32였다면) | 약 91 ms | 43 % |
| float64로 계산해서 늘어난 부분 | 약 91 ms | 43 % |
| SOAP 회전 계산 (매 스텝) | 약 12 ms | 6 % |
| SOAP 기저 갱신 (10스텝마다, 평균) | 약 16 ms | 8 % |

- 이전 문서의 "GH200에서 FP64와 FP32 속도가 비슷하다"는 QR 등 선형대수 연산에 대한 측정이었다. 모델 전체(cuequivariance
  텐서곱 포함)에는 해당하지 않는다.
- 4장 한 프로세스(GPU당 64개)에서는 스텝이 약 290 ms다. 1장보다 80 ms 긴 부분은 SOAP가 아니라 주 프로세스의 배치 조립이다.

## 2. 수정 1: 기저 갱신의 GPU 분담 (`soap_distributed_refresh`)

- **무엇이 문제였나.** 옵티마이저 갱신은 `shard_map` 안에서 복제된 상태로 돈다(`training_utils.make_sharded_train_step`).
  그래서 GPU마다 같은 QR을 중복 계산한다. `soap_grouped.py`에는 장치별로 1/D씩 나눠 계산하고 all_gather하는 경로
  (`_orth_partitioned`, 인자 `axis_name` / `axis_size`)가 이미 있었지만 `training/train.py`가 넘기지 않아 쓰이지 않았다.
- **수정** (커밋 `7897ce8`, `race_v2/training/train.py` `build_model`의 SOAP 분기): 설정 키
  `"soap_distributed_refresh": true`이고 장치가 2개 이상이면 `axis_name='dp', axis_size=n_devices`를
  `scale_by_soap_grouped`에 넘긴다. 기본값 false. `soap_impl: "grouped"`에서만 의미가 있다.
- **측정** (4장 한 프로세스, GPU당 64개, 200스텝 시간): 기준 58.1초, 분담 57.0초(−2.0 %), 분담 + cholqr 56.9초(−2.1 %).
- **확인된 것.** 오류 없이 돌고, 600스텝 뒤 손실이 기준과 같은 수준이다(학습 손실 0.352 / 0.358 / 0.354).
- **확인 안 된 것.** 수식이 정확히 같은지. 같은 설정을 두 번 돌려도 약 1 % 차이가 나므로 손실 비교로는 증명되지 않는다.
  단위 테스트가 없다(`tests/test_soap_grouped.py`는 `axis_name`을 다루지 않는다). 여러 프로세스(다중 노드)에서의
  all_gather는 돌려 보지 않았다.
- **판단.** 이득이 2 %뿐이라 지금 도는 학습에는 켜지 않았다.

## 3. 수정 2: float32 모델 + float64 SOAP 상태 (`soap_state_dtype`)

### 3.1 증상

`"x64": false, "soap_state_dtype": "float64"`는 "SOAP의 모멘트·인자·고유기저만 float64, 파라미터와 데이터는 float32"라는
옵션이다(SOAP는 순수 float32에서 발산한 기록이 있다: OAM 30~80스텝, 3BPA 107~337에폭). 그런데 이 모드의 실행이
시간(210 ms)도 메모리(44 GB)도 float64와 같았다. 파라미터와 배치는 float32인데 모델 출력(에너지, 힘)이 float64였다.

### 3.2 원인

이 모드는 `jax_enable_x64`를 켠다. 그러면 정밀도를 지정하지 않은 생성자가 float64 배열을 만들고, float32 값과 만나는 순간
float64로 승격된다. 뿌리는 `race_v2/models/blocks.py`의 네 곳이었고 거기서 모든 층으로 번졌다.

| 위치 | 원래 코드 | 번진 범위 |
|---|---|---|
| `bessel_basis` | `jnp.linspace(1.0, num_basis, num_basis)` | 방사 기저 → 간선 특징 → 이후 전부 |
| `compute_node_edge_features` | `jax.nn.one_hot(species, n_species)` | 원자 특징 → 선형층 → 이후 전부 |
| `separated_layer_norm` | `jnp.zeros((n, 1))`, `xi_s / jnp.maximum(cnt, 1)` (정수 배열 → float64) | 정규화된 특징 |
| `energy_forces_stress` | `frc_variation`의 패딩 채움 값 | 부수 출력 하나 |

### 3.3 수정

커밋 `a26057c`. 네 곳 모두 입력의 dtype을 따르게 했다(5줄). 순수 float64 실행과 순수 float32 실행의 결과는 바뀌지 않는다.
`thole.py` / `models/race_polar.py`(Thole 블록)에는 고칠 것이 없었다.

### 3.4 확인 (작업 1260206 추적, 1260218 실행)

계산 그래프 추적: 순전파·손실·기울기가 모두 float32. float64를 새로 만드는 연산이 105개 → 1개(손실 기록용 스칼라).

GPU 1장, 64개 구조/스텝, 2,000스텝(400스텝 × 5):

| 실행 | 스텝당 시간 | GPU 메모리 | 검증 손실 | 검증 힘 MAE (meV/Å) | 학습 손실 |
|---|---|---|---|---|---|
| SOAP, 모델 float64 (기준) | 201 ms | 42.2 GB | 0.1999 | 80.9 | 0.276 |
| **SOAP 상태 float64 + 모델 float32** | **110 ms** | **20.8 GB** | 0.1994 | 80.9 | 0.279 |
| SOAP, 전부 float32 | 110 ms | 19.1 GB | 0.2018 | 81.4 | 0.282 |
| 합친 모델(Qeq + Thole), 상태 float64 + 모델 float32, 48개/스텝 | 121 ms | 22.8 GB | 0.2135 | 84.1 | 0.313 |

- 속도 1.82배, 메모리 절반, 에폭별 검증 손실이 기준과 같이 움직인다.
- SOAP 상태를 float64로 두는 추가 비용은 없다(전부 float32와 시간이 같다).
- 테스트 78개 통과(Thole, RACE_POLAR, 합친 모델, 스트림, RACE_LR, 전하 라벨 옵션, grouped SOAP).
- 참고: 합친 모델은 float64, GPU당 48개, 2노드에서 메모리가 최대 81 GB였다(작업 1259021). float32가 아니면 여유가 없다.

### 3.5 확인 안 된 것

- **긴 학습에서의 안정성.** 2,000스텝까지만 봤다. 전부 float32인 대조군도 이 길이에서는 발산하지 않았다(검증 힘 RMSE가
  한 번 튀었다가 돌아옴). 과거의 발산은 더 뒤에서 났으므로, "상태 float64면 안전하다"는 아직 증명되지 않았다.
  첫 긴 실행은 작업 1260633(합친 모델, 2노드, 80에폭)이다. 발산하면 체크포인트에서 `"x64": true`로 이어 갈 수 있을
  것으로 보지만(파라미터를 float64로 올림) 해 보지 않았다.
- **이 모드에서의 재개(restart).** 작업 1260633의 앞부분 사전 점검이 처음 확인한다.
- **SOAP 고유기저 Laplace**(`soap_laplace.py`)가 float32 파라미터 + float64 상태의 체크포인트를 받는지.
- 손실 함수 안의 스칼라 자리표시자 `jnp.zeros(())`는 float64로 남아 있다. 기록용 값 하나에만 영향을 준다.

## 4. 쓰는 법

float32 모델 + float64 SOAP 상태 (커밋 `a26057c` 이후의 코드에서만 효과가 있다):

```json
"x64": false,
"optimizer": "soap", "soap_impl": "grouped",
"soap_state_dtype": "float64"
```

- 설정 변환기에서는 `--set x64=false --set soap_state_dtype='"float64"'` (`jobs/omol4m_stream_config.py`).
  합친 모델 설정 생성기는 `S64M32=1 bash jobs/omol4m_make_hybrid_config.sh`.
- RACE_POLAR의 `polar.cg_tolerance`는 float32에서 1e-4를 쓴다(`jobs/omol_polar_config.py`가 `x64`를 보고 정한다).
- 그 밖의 SOAP 옵션: `"soap_orth": "cholqr"`(1장에서 −7 %, 수식 거의 같음), `"soap_distributed_refresh": true`
  (여러 장에서 −2 %), `"soap_precondition_frequency"`(늘리면 수식이 바뀌고 cholqr보다 이득이 없다),
  `"soap_species_adam": "any_axis"`(범용 원소 목록에서 필수).

## 5. 승격을 찾는 도구

새 코드를 넣은 뒤 float32 모드가 깨졌는지 1분 안에 확인할 수 있다.

```bash
cd /scratch/t510a13 && sbatch jobs/dtype_trace.sh          # CFG=<config.json> REPO=<repo> 로 바꿀 수 있다
```

- `jobs/dtype_trace.py`: 학습기와 같은 방식으로 로더와 모델을 만들고, 순전파·손실·기울기를 `jax.make_jaxpr`로 추적해
  "float64가 아닌 입력에서 float64를 만드는 연산"을 코드 줄별로 집계한다. 실제 계산은 하지 않는다(CPU, 약 30초).
- 출력의 `output dtypes`가 float32이고 집계가 0~1줄이면 정상이다.
- **규칙:** 모델·손실 코드에서 `jnp.zeros / ones / full / eye / arange / linspace / array / asarray`, `jax.nn.one_hot`을 쓸 때는
  입력의 dtype을 넘긴다. 정수 배열로 실수를 나누지 않는다(파이썬 정수는 괜찮다).

## 6. HANGANG에서 속도를 잴 때 주의

- 사용자당 노드별 프로세스 + 스레드 한도가 8,192개다. XLA와 LLVM은 보이는 코어 수만큼 스레드를 만든다. JAX 프로세스
  여러 개를 한 노드에서 동시에 띄울 때는 `taskset`으로 코어를 나눠 준다(GPU당 72코어). 안 그러면
  `pthread_create failed`로 죽는다(작업 1258661).
- 코어를 고정하면 1장 실행이 약 15 % 빨라진다(주 프로세스의 배치 조립).
- GPU 1장만 받은 작업의 코어 번호는 0부터 시작하지 않는다. 그런 작업에서 `taskset -c 0-31`은 실패한다(작업 1259693).
- 같은 설정을 두 번 돌려도 초기 손실이 약 1 % 다르다. 600스텝짜리 비교에서 그보다 작은 차이는 읽지 않는다.

## 7. 관련 커밋, 파일, 작업

| 항목 | 위치 |
|---|---|
| GPU 분담 옵션 | 커밋 `7897ce8`, `race_v2/training/train.py` |
| float32 승격 수정 | 커밋 `a26057c`, `race_v2/models/blocks.py` |
| SOAP 구현 (수정 없음) | `race_v2/training/soap_grouped.py`, `race_v2/training/soap_optax.py` |
| 측정 작업 스크립트 | `/scratch/t510a13/jobs/soap_profile.sh`, `f32_check.sh`, `dtype_trace.sh`, `dtype_trace.py` |
| 측정 로그 | `/scratch/t510a13/logs/soap_prof_1258956.out`, `f32_check_1260218.out`, `dtype_trace_1260206.out` |
| float32로 도는 첫 긴 학습 | 작업 1260633, 코드 `/scratch/t510a13/BAM_V2_run_hybrid_f32`(커밋 `a26057c`로 고정) |
| float64로 도는 본 학습 | 작업 1257815, 코드 `/scratch/t510a13/BAM_V2`(학습 중에는 수정 금지) |
