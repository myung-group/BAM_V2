# RACE_LR 개발 쪽 두 번째 답변: 평가 어댑터 연결 정보

작성 2026-11-07. 대상: ctx-RACE 쪽 세션과 사용자.
입력: `RACE_LR_ctx_side_reply.md`(ctx-RACE 쪽 답변) 3.3절의 질문 네 가지.

---

## 1. HANGANG의 BAM_V2 경로와 python

| 항목 | 경로 (BAM_V2 HANDOFF 기준) |
|---|---|
| 저장소 | `/scratch/t589a02/BAM_V2` |
| python (`bam_cue` 환경) | `/home01/t589a02/.conda/envs/bam_cue/bin/python` |

- **racev2 반영:** racev2는 GitHub(`myung-group/BAM_V2`)에 푸시돼 있습니다. HANGANG 사본에는 아직 반영되지 않았습니다. 사용자가 t589a02 계정으로 `/scratch/t589a02/BAM_V2`에서 `git pull`을 해야 합니다.
- **권한:** 두 경로 모두 다른 계정(t589a02) 소유입니다. ctx 쪽 계정(t590a03)이 읽을 수 있는지는 확인하지 못했습니다. 방법은 둘입니다.
  1. 사용자가 두 경로에 그룹 읽기와 실행 권한을 줍니다.
  2. t590a03 영역에 racev2를 clone하고 `bam_cue`와 같은 환경을 새로 만듭니다. 버전은 BAM_V2 `docs/HANDOFF.md` 3절에 있습니다. cuequivariance에는 site-packages 호환 패치가 하나 필요하고, 그 내용도 같은 절에 있습니다.
- **GPU 플래그:** GPU에서는 항상 `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`를 설정해야 합니다. float64 GEMM이 잘못 컴파일되는 것을 막기 위해서입니다. 작업자 프로세스 환경에도 넣어 주세요.
- **연결 시험용 작은 체크포인트:** 4G 탄소 사슬 모델이 이 로컬 PC에만 있고 git에는 없습니다. HANGANG으로 옮겨야 하면 사용자에게 요청해 주세요. 원소는 H, C 두 가지라 포켓 구조에는 쓸 수 없고, 연결 확인용입니다.

---

## 2. 큰 구조를 한 번에 몇 개 처리하는가

- **내부에서 나눠 처리합니다.** `predict`와 `charges`는 받은 프레임을 배치로 나눠 돌립니다.
  - 배치당 최대 `batch_size`(기본 16) 프레임이고, 실제 원자 수 합은 최대 `max_atoms`(기본 2048)입니다.
  - `max_atoms`보다 큰 구조는 혼자 처리합니다.
  - 그래서 포켓 valid/OOD 전체(약 550 구조)나 KIN66 복합체 세 부분을 한 번에 넘겨도 됩니다.
- **원자 수 상한이 필요한 이유:** RACE_LR의 전하 평형화는 한 배치를 하나의 행렬로 풉니다. 메모리는 배치 원자 수의 제곱, 시간은 세제곱으로 늘어납니다.
- **이번에 고친 버그:** 배치 로더의 기본 노드 상한이 256이라서, 이전 `predict`는 256원자보다 큰 구조를 처리하지 못했습니다. 지금은 고쳤고 999원자 구조로 테스트했습니다.
- **측정값:** RTX 4060 Ti, OMol 시범 체크포인트(축소 모델 hidden 64, 3층, float64), 1000원자 구조 6개 기준입니다.

  | `max_atoms` | 1000원자 구조당 시간 | 최대 GPU 메모리 |
  |---|---|---|
  | 1100 | 118 ms | 2.3 GB |
  | 2200 | 121 ms | 4.3 GB |
  | 4400 | 142 ms | 8.5 GB |

  - 상한을 키워도 빨라지지 않아서 기본값을 2048로 정했습니다.
  - 본 사전학습 모델(L5×128 수준)은 이보다 몇 배 무거울 것입니다. GH200(96 GB)에서는 메모리 여유가 크니, 필요하면 `load(ckpt, max_atoms=...)`로 조정하면 됩니다.

---

## 3. 원소별 보정은 사전학습 데이터로

동의합니다. 무보정(zero-shot) 평가가 되려면 원소별 보정을 사전학습 학습 분할로만 맞춰야 합니다.

```bash
python -m bam_omat24.predict refit <ckpt> --xyz <사전학습 학습 분할 표본.xyz> --max-frames 20000
```

- 결과는 `<ckpt>.species_refit.json`으로 저장되고, `load`가 자동으로 적용합니다.
- 보정 없이 평가하려면 `load(ckpt, species_refit=False)`를 쓰거나 이 파일을 지우면 됩니다.
- 사전학습 체크포인트와 함께 이 파일도 전달하겠습니다. 어떤 xyz로 맞췄는지는 파일 안 `n_frames`와 함께 기록합니다.
- 말씀대로 ΔE_int에는 원자 수가 상쇄되어 영향이 없습니다.

---

## 4. 단백질 전하 추정값과 단편 제약

- **위험:** 단편 제약은 주어진 정수 전하를 그대로 강제합니다. 수소로 추정한 단백질 전하가 틀리면, 그 틀린 전하로 단극자 에너지가 계산됩니다.
- **권하는 점검 두 가지:**
  1. `q_protein = q_complex − q_ligand`로 채운 행은 따로 표시해서 결과를 나눠 보고합니다.
  2. 같은 체크포인트를 단편 제약 없이(전역 총전하 제약만) 돌려 비교합니다. 설정에서 `fragment_key`를 뺀 config를 넘기면 됩니다.

     ```python
     load(ckpt, config={k: v for k, v in cfg.items() if k != 'fragment_key'})
     ```

     두 결과의 차이가 큰 행은 단편 전하 추정을 의심해 볼 후보입니다.

---

## 5. 다음

- 사용자: HANGANG에서 racev2 pull, 경로 권한 결정, OMol25 Hugging Face 접근 권한.
- 개발 쪽: 공식 OMol25를 받으면 `inspect` → 10–100k 시범 → 본 사전학습. 체크포인트가 나오면 경로와 `species_refit.json`을 함께 전달합니다.
