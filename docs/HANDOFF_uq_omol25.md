# 인계 문서: 불확정도(UQ) 방법 현황과 OMol25 전체 데이터셋 계획 (2026-11-10)

다른 Claude Code 세션이 이 문서만 읽고 이어서 진행할 수 있도록 쓴다. 먼저 `docs/HANDOFF.md`(규칙·환경·경로)를
읽고, 그 다음 이 문서를 읽는다. 세부 수치는 `docs/local_test_report.md` §17–20, 방법은 `docs/SOAP_Laplace.md`.
이 학습이 무슨 연구를 위한 것인지는 `docs/RESEARCH_PURPOSE.md`에 있다(단백질–리간드 상호작용 에너지).

## 0. 사용자 결정 (2026-11-10)

- **OMol25를 전체 데이터셋으로 해보기로 함.** ("전체 데이터셋으로 해볼까 해") 파일럿 부분집합이 아니라 공식 학습
  분할(약 1억 구조) 전체를 목표로 한다. 다만 §4의 순서(접근 → inspect → 변환 → 스트리밍 스모크 → 소규모 파일럿 → 전체)는
  지키고, 전체 학습은 다중 GPU/다중 노드가 전제이므로 그 전에 다중 호스트 로더(T5)부터 검증한다.
- **OMol25를 먼저 학습하고, 나중에 다른 데이터셋(OMat24 / MPtrj / Alexandria / OC20)을 이어서 또는 트렁크를 고정한 채
  멀티헤드로 붙일 수 있게 해 두기로 함.** 그래서 OMol25 설정은 처음부터 범용 원소 목록(Z 1..94)과 과제 임베딩 자리를
  갖는다(§7, 구현·검증됨). 전하와 스핀은 OMol25 학습에 입력으로 들어간다.
- 불확정도는 전체 데이터로 적합할 필요가 없다. 사후(post-hoc) 방법이라 학습 한 번 뒤에 학습 구조 수천 개(②) /
  수백 개(③)로 적합하고, 평가는 공식 검증 세트(Val Comp)에서 1–2천 개를 뽑아 한다(§3).

## 1. 불확정도 방법 네 가지와 상태

| 번호 | 방법 | 구현 | 상태 |
|---|---|---|---|
| ① | 모델 분산(MVE류) | 기존 | 사용 안 함 |
| ② | 마지막 층 베이지안 선형회귀(BLR) | `bam_omat24/bayes_lastlayer.py` | 완료. 주기 구조·응력 지원. 3BPA·탄소 사슬·AuMgO·OMat24 스크립트 있음 |
| ③ | SOAP 고유기저 Laplace (GGN 곡률, 선형화 예측) | `bam_omat24/soap_laplace.py` | 완료(로컬). HANGANG 정식 체크포인트에서는 아직 안 돌림 |
| ④ | 딥 앙상블 | `bayes_lastlayer.py`의 `--seeds`로 3시드 결합 | 보조용 |

### ② 마지막 층 BLR
- 네트워크의 마지막 층 특징(`READOUT_TAP`)에 대한 베이지안 선형회귀. 에너지·힘(·응력) 야코비안으로 정규방정식을 풀고,
  증거(evidence) 최대화로 사전/노이즈 σ를 정한다. 어떤 옵티마이저 체크포인트든 적용 가능.
- 예측 σ = 인식론적 항(특징 공간의 거리) + 노이즈 항(학습 잔차에서 맞춘 상수). **노이즈 항이 지배**해서 σ의 크기가
  학습 잔차에 고정되는 것이 한계다(§2).
- 결과 덤프: `per_structure_<set>.npz` (yE, mE, netE, sigE_epi, sigE_tot, F_err, F_sig_epi, F_sig_tot …).

### ③ SOAP 고유기저 Laplace
- 모든 가중치에 대한 Laplace 근사. SOAP 옵티마이저의 Kronecker 고유기저 (Q_L⊗Q_R) 위에서 곡률을 대각으로 둔다.
  여기서 SOAP는 **옵티마이저**(Shampoo 계열)이지 SOAP 기술자가 아니다. 그래서 **SOAP 옵티마이저로 학습한
  체크포인트가 필요**하다(현 레시피는 SOAP f64).
- 곡률: 옵티마이저의 ν(경험적 Fisher)는 실패(§18). `--ggn`으로 GGN 대각을 직접 모은다. 정확식(출력마다 VJP 1회)과
  Hutchinson(구조당 프로브 2회, `--ggn-probes 2`)이 있고 둘은 Spearman 0.999로 일치. 학습 구조 450–500개면 수렴.
- 사후 정밀도 n_train·ν + λ (`--kappa 1 --kappa-times-ntrain`). λ는 검증 NLL로 고르거나 `--lam-evidence`(닫힌형
  증거 최대화) / `--lam-per-leaf`(MacKay 고정점)로 정한다.
- 예측: `--predictive glm`(선형화, 샘플마다 JVP 1회, `--jvp-chunk 4`)이 MC보다 모든 면에서 낫다. 비용은 샘플 16개
  기준 추론의 약 16배.
- 노이즈 σ_E, σ_F 기본값 = 네트워크 검증 RMSE.

## 2. 검증 결과 요약

### 2.1 분포 내 (report §19; 탄소 사슬·AuMgO, 로컬 GPU)
- ③ GLM은 자기 보정된다: 탄소 사슬 σ_F ≈ RMSE, rms z ≈ 1, 95 % 포함률 0.95–0.97, 힘 순위 Spearman은 ②와 동급
  (0.3–0.6). AuMgO에서 처음으로 쓸 만한 에너지 σ(z 0.4–0.8).
- 증거 최적 λ(0.1–0.3)는 검증으로 고른 λ(10–100)보다 훨씬 작다. 둘 다 기록돼 있다.

### 2.2 BAM V1 논문의 3BPA OOD 설정 (report §20; 로컬 축소 모델)
설정: 300 K 학습(450+50), 시험 300 / 600 / 1200 K(온도별 400개). 축소 모델 hidden 64, lmax 2, 3층, 300 에폭.
네트워크 힘 RMSE 11.3 / 27.0 / 80.7 meV/Å (논문 RACE 12.1 / 31.8 / 115.3), 에너지 3.9 / 13.8 / 37.1 meV.

| 방법 | σ_F (300/600/1200 K) | rms z | 95 % 포함률 | OOD AUROC (600 / 1200 K) |
|---|---|---|---|---|
| ② BLR | 3.5 / 3.5 / 3.6 | 3.2 / 7.6 / 22.3 | 0.64 / 0.42 / 0.21 | 0.99 / 1.00 |
| ③ GLM (κ=n, λ=10) | 10.7 / 12.5 / 16.3 | 0.90 / 1.71 / 3.64 | 0.96 / 0.82 / 0.58 | 0.98 / 1.00 |
| ③ MC | 10.5 / 11.6 / 13.6 | 1.06 / 2.10 / 4.58 | 0.93 / 0.78 / 0.51 | 0.98 / 1.00 |

읽기:
- OOD 탐지(논문의 AUROC 지표)는 세 방법 모두 거의 완벽. 논문의 DE·IVON 1.00, SWAG 0.58, MVE 0.54 (oBN25 기준).
- ②는 순위는 완벽하지만 σ 크기가 온도와 무관하게 고정(노이즈 항 지배). 인식론적 항만 보면 0.17→0.21→0.32로
  움직이나 절대값이 30배쯤 작다.
- ③ 선형화만 σ가 외삽과 함께 커진다(로그 기울기 약 1/3). 300 K에서는 보정이 맞고, 1200 K에서는 3–4배 과신.
- 논문 그림 2(oBN25 |Δy| 대 σ 산점도)와 형태를 대응시키면 ②는 MVE·SWAG형(수직선), ③ GLM은 DE·IVON형(대각선 쪽)이다.
  논문에는 3BPA의 σ–오차 그림이 없다. 3BPA는 표 4(RMSE)와 능동학습 표만 있다. 비교 그림:
  `docs/figures/ood_3bpa_local_sigma_vs_error.png`, 수치 `docs/figures/ood_3bpa_local_summary.json`.
- 개선 후보(미착수): ②의 순위 + ③의 크기 결합, 또는 ③의 예측 분포에 ②의 노이즈 항 추가. 논문식 분위수 띠를
  우리 그림에 추가.

### 2.3 OMol25에서 ② vs ③ 판단 기준
분자냐 결정이냐는 무관하다. 갈리는 점은 (i) 외삽에서 σ가 커져야 하는가(→ ③), (ii) 비용·제약(② 는 아무
체크포인트, 추론 1회; ③은 SOAP 체크포인트, 추론 16배). 추천: 기본은 ②, 화학 조성 외삽 감지가 필요한 곳은 ③ GLM
또는 ②의 σ 상위 몇 %에만 ③을 선택적으로 적용. 이 판단은 OMol25 체크포인트에서 둘을 같이 돌려 확정한다(§3).

## 3. OMol25에서 UQ를 돌리는 방법 (전체 학습 뒤)

1. 체크포인트 하나(SOAP f64 레시피면 ②·③ 모두 가능; AMSGrad f32로 학습했다면 ②만 가능).
2. ②: 학습 구조 2–5천 개를 균등 추출해 적합. `runs/diag/bayes_lastlayer_oam.sh`를 본떠 스크립트 작성
   (스트리밍 소스 `aselmdb:` 지원 여부는 `bayes_lastlayer.py`가 xyz를 읽으므로 xyz 부분집합을 먼저 만든다:
   `data_prep/convert_omol25.py subset`).
3. ③: 학습 구조 500개로 `--ggn 500 --ggn-probes 2 --predictive glm --jvp-chunk 4 --kappa 1 --kappa-times-ntrain
   --lam 1 10 --lam-evidence --samples 16` (`runs/diag/ood/ood_3bpa_hangang.sh`의 인자 그대로).
4. 평가: 공식 검증 세트 Val Comp(학습에 없는 분자식, 정답 공개)에서 1–2천 개를 뽑고, 분포 내 대조군으로 학습
   분할에서 뽑은 1–2천 개(학습에 쓰지 않은 행)를 둔다. `runs/diag/ood/ood_summary.py --blr <dir> --lap glm=<dir>
   --sets <names>`로 표·AUROC·그림을 만든다. `load()`의 세트 이름만 맞추면 된다.
5. 비용: ② 수 분, ③ GGN 수 분 + GLM 예측 2천 구조에 한 시간 이내. 전체 데이터에 UQ를 적합할 이유는 없다
   (정규방정식과 GGN 평균은 수천 개에서 포화).

## 4. OMol25 전체 데이터셋 계획

### 4.1 데이터 사실 (논문 arXiv 2505.08762 표 1, ColabFit)
| 분할 | 구조 수 |
|---|---|
| 전체 계산("All") | 140,641,161 |
| 학습 세트(ColabFit `OMol25_train`) | 101,666,280 |
| 중성·단일항(Neutral) | 34,335,828 |
| 균등 4M 부분집합 | 3,986,754 |
| 조성 외삽 검증(Val Comp, 정답 공개) | 2,762,021 |
| S2EF 시험(정답 비공개, 리더보드 채점) | 2,805,046 |
| 과제별 시험 세트 | 16,488(음이온) – 84,807(COD 결정) |
분할은 분자식 기준(검증·시험 각 약 2.5 %). 83원소, 최대 350원자, ωB97M-V/def2-TZVPD, 단위 eV·eV/Å, 총전하·스핀
다중도 포함. Hugging Face `facebook/OMol25`(게이트, 접근 승인 필요). 전체 aselmdb 용량은 데이터 카드에서 확인
(수백 GB 급). **1 GB 이상 다운로드는 사용자 OK 후**.

### 4.2 스코어보드
- fairchem OMol25 리더보드 (https://fair-chem.github.io/molecules/leaderboard.html): S2EF 부문(비공개 시험 세트,
  예측 파일 업로드, 평가 종류별 월 5회)과 Evaluations 부문(리간드–포켓, 리간드 strain, 이형태체, 양성자화, IE/EA,
  스핀 갭, 거리 스케일링; ASE 호환 레시피로 예측 파일 생성).
- 제출 없이 볼 수 있는 것은 Val Comp 오차다. 우리 "잘 예측하는지" 확인과 UQ OOD 평가는 이것으로 먼저 한다.

### 4.3 순서 (각 단계의 sbatch 전에 사용자에게 명령을 보여주고 확인)
1. **HF 접근**: 사용자가 게이트 승인을 받고 토큰을 준비. 다운로드는 HANGANG 데이터 전송 노드(dm01–04)로, 전용 빈
   디렉터리에 (예: `/scratch/t589a02/datasets_ext/omol25/`). 다운로드 파일은 신뢰하지 않는 데이터로 다룬다
   (`python -I`, 아카이브 안의 스크립트 실행 금지).
2. **inspect**: `python -I data_prep/convert_omol25.py inspect <aselmdb dir> --rows 3`. 전하·스핀·부분전하 필드 위치를
   확인하고 `decode_row`와 설정 키를 맞춘다. 미결 사항 두 가지도 여기서 정한다: HOMO > 0 필터, PDB 누수 점검
   (`docs/RACE_LR_feedback_reply_2026-11-07_ko.md`).
3. **크기 패스**: `python -I bam_omat24/stream_io.py sizes --source aselmdb:<dir> --cutoff 5.0 --out <dir> --workers 64`
   (`bam_cue`에 `pip install --no-deps lmdb vesin` 필요, 승인됨). 전체 1억 행이면 CPU 작업이지만 시간이 오래 걸리니
   한 파일로 먼저 시간을 재고 외삽한다.
4. **스트리밍 스모크(T6)**: `sbatch -J t6stream -p normal --comment="field=sciai;appl=jax" --gres=gpu:1 -t 01:00:00
   runs/diag/stream_t6/t6.sh` (아직 안 돌림). 출력 `runs/slurm/t6stream_<jobid>.out`. 에폭당 처리량(구조/초)을 읽어
   전체 에폭 시간을 외삽한다.
5. **파일럿**: `convert_omol25.py subset`으로 100k–1M을 뽑아 `runs/diag/make_omol_configs.py`(race / race_cs /
   racelr / racelr_q × soap_f64 / amsgrad_f32)로 설정을 만들고 30 에폭. 기본값이 범용 원소 목록 + `n_tasks` 8이며
   설정 이름에 `_u94`가 붙는다(§7; 예전 방식은 `--elements subset --n-tasks 0`). **전체 학습에 쓸 설정은 반드시 범용
   목록으로 만든다**(나중에 못 바꾼다). 로컬 OpenFF 부분집합 결과(report §11):
   AMSGrad f32가 SOAP f64보다 약 10배 빠르고 정확도는 같거나 약간 낫다. 전체 학습에 AMSGrad f32를 쓰면 ③은 못 쓰고
   ②만 쓸 수 있다는 점을 사용자에게 알리고 결정받는다.
6. **다중 호스트 검증(T5)**: 전체 학습은 단일 GPU로 불가능하다. 100k OMat24에서 L5x128이 GH200 한 장에 에폭당
   520 s(AMSGrad) / 1010 s(SOAP f64)였으므로, 구조당 비용이 비슷하다고 보면 1억 구조는 한 장에 **에폭당 6–12일**,
   4M도 6–12시간이다. `BAM_DISTRIBUTED=1`과 스트리밍 로더의 다중 호스트 경로(코드만 있고 미검증,
   `docs/HANDOFF_stream_loader.md`)를 2프로세스 → 2노드 순으로 검증한다. 블록 대각 Qeq(`long_range.qeq_block`, 최대
   구조 352)는 RACE_LR 변형에서만 필요.
7. **전체 학습**: 노드 수·에폭 수·체크포인트 주기를 사용자와 정한 뒤 제출. `/scratch`는 15일 미접근 파일을 지우므로
   체크포인트를 다른 곳에 복사.
8. **평가와 UQ**: Val Comp 오차 → 리더보드 S2EF/Evaluations 제출(사용자 계정) → §3의 UQ.

### 4.4 알려진 제약
- 학습 분할만 사용(공정 프로토콜). 사전학습·증강·앙상블로 보고 수치를 만들지 않는다.
- SOAP는 float32에서 발산한다. 전체 학습을 SOAP로 하려면 f64(2배 느림) 또는 `soap_state_dtype`.
- `runs/`는 git 무시. 진단 스크립트는 `git add -f`. 커밋은 `racev2`에만. 푸시는 SSH URL
  `git@github.com:myung-group/BAM_V2.git`로.

## 5. 당장 남아 있는 HANGANG 작업 (사용자가 실행; 각 sbatch 전에 확인)

1. `git pull origin racev2`; `pip install --no-deps vesin lmdb` (승인됨).
2. T6 스트리밍 스모크 (§4.3-4).
3. 3BPA OOD 체인을 정식 레시피 체크포인트(`runs/abl3_full_s42_cue/ckpt_*.gauge.pkl`)에서:
   `sbatch -J ood3bpa -p normal --comment="field=sciai;appl=jax" --gres=gpu:1 -t 04:00:00 runs/diag/ood/ood_3bpa_hangang.sh`
   (빠른 확인은 `MAX_TEST=400`을 앞에). 출력 `runs/bayes/ood_3bpa_abl3_full_s42[_quick]/`. "F(mean)" 행을 논문 RACE-LA
   (F 18.2 / 51.0 / 171.8, E 4.8 / 15.3 / 60.8)와 비교하고, 축소 모델의 결론(§2.2)이 유지되는지 본다.
4. 선택: `runs/diag/bayes_lastlayer.sh`, `runs/diag/bayes_lastlayer_oam.sh`, `runs/diag/soap_laplace.sh`.

## 6. 관련 파일
| 파일 | 내용 |
|---|---|
| `bam_omat24/bayes_lastlayer.py`, `tests/test_bayes_lastlayer.py` | ② |
| `bam_omat24/soap_laplace.py`, `tests/test_soap_laplace.py`, `docs/SOAP_Laplace.md` | ③ (§6에 상태) |
| `runs/diag/ood/ood_summary.py`, `ood_3bpa_hangang.sh` | OOD 표·AUROC·그림, HANGANG 체인 |
| `runs/diag/soap_laplace.sh`, `bayes_lastlayer*.sh` | 단일 데이터셋 체인 |
| `input/local_3bpa/race_s42.json` | 로컬 축소 3BPA 설정 |
| `docs/local_test_report.md` §17–20 | 수치와 해석 |
| `data_prep/convert_omol25.py`, `runs/diag/make_omol_configs.py`, `runs/diag/stream_t6/` | OMol25 도구 |
| `docs/HANDOFF_stream_loader.md`, `docs/stream_loader_design.md` | 스트리밍 로더 |
| `bam_omat24/race_lr_nnx.py` (`n_tasks`), `runs/diag/omol_universal/check_untouched.py` | 과제 임베딩 자리, 범용 원소 목록 점검 (§7) |

## 7. OMol25를 먼저 학습하고 나중에 범용(다중 데이터셋) 모델로 잇기 (사용자 결정 2026-11-10)

사용자 질문과 결정: OMol / OMat / MPtrj / Alexandria / OC20을 아우르는 모델을 만들 수 있는가 → 가능(UMA 방식). 일단
OMol25로 학습하고, 필요해질 때 그 모델을 버리지 않고 **이어서 학습하거나 고정한 채 멀티헤드를 붙이는** 길을 열어 두기로
했다. `init_params_from`은 모델 설정(파라미터 트리)이 같아야 하므로, OMol25 학습을 시작하기 전에 아래 세 가지를 고정한다.

### 7.1 OMol25 학습 전에 고정하는 것 (구현됨)
| 항목 | 설정 | 구현 |
|---|---|---|
| 원소 목록 | Z 1..94 (H–Pu). OMol25는 Z ≤ 83, OMat24 / MPtrj / Alexandria는 Pu까지 89개 | `make_omol_configs.py --elements universal` (기본값, 이름에 `_u94`). 부분집합에 없는 원소는 E0 = 0.0 |
| cutoff | 5.0 Å 유지 | 설정 생성기가 항상 5.0 |
| 과제 임베딩 자리 | `long_range.n_tasks: 8` | `race_lr_nnx.py`: 0으로 초기화한 (n_tasks, 층-0 폭) 표. `globals['task']` 행을 층-0 스칼라에 더한다 (전하·스핀 임베딩과 같은 자리). `make_omol_configs.py --n-tasks 8` (기본값; RACE_LR 계열 arm만, 순수 `race` arm에는 자리가 없다) |

- 과제(task) = 라벨 규약 하나(데이터셋의 DFT 수준). **OMol25 = 과제 0** (데이터에 `task` 키가 없으면 0). 번호 예약 제안:
  1 OMat24 / MPtrj / sAlex (PBE, PBE+U; 필요하면 둘로 나눔), 2 OC20 (RPBE), 3 이후 여유.
- **전하·스핀은 OMol25 학습에 들어간다.** `race_cs` / `racelr` / `racelr_q` arm은 `charge_spin_embedding: true`이고 모든
  설정이 `total_charge_key: charge`, `total_spin_key: spin`을 읽는다. 순수 `race` arm만 입력으로 쓰지 않는다(비교 기준).
  주기계 데이터에는 키가 없으므로 기본값 Q = 0, 2S+1 = 1이 들어간다.
- 없는 원소의 종별 표 행과 과제 1.. 행은 기울기가 정확히 0이라 초기값 그대로 남는다(가중치 감쇠 0 전제. 감쇠를 켜면 종별
  표의 없는 원소 행이 줄어드니 켜지 말 것).

### 7.2 검증 (로컬, OpenFF 부분집합 10k, 원소 7종, L3x64 lmax 2, 2 에폭)
같은 부분집합에서 원소 목록만 바꿔 비교했다 (`race_cs`, 시드 42; 스크립트 `runs/diag/omol_universal/check_untouched.py`).

| | 원소 7종 (부분집합), 과제 자리 없음 | Z 1..94, `n_tasks` 8 |
|---|---|---|
| 파라미터 수 (L3x64) | 859,029 | 5,854,559 |
| AMSGrad f32 에폭 시간 | 10.6 s | 11.1 s |
| SOAP f64 에폭 시간 | 107.4 s | 111.8 s |
| 2 에폭 뒤 검증 힘 MAE, AMSGrad | 203.7 meV/Å | 206.3 meV/Å |
| 2 에폭 뒤 검증 힘 MAE, SOAP | 168.4 meV/Å | 170.1 meV/Å |

- 속도와 정확도는 사실상 같다(에폭 시간 +4–5 %). 파라미터가 느는 것은 종별 표(원소마다 폭×폭 행렬: skip, readout) 때문이며
  원소 수에 **선형**이다. 생산 크기(L5x128, lmax 3)로 모델을 만들어 센 값: 7종 8.24M, **83종(OMol25 자체) 43.12M,
  94종 48.17M (+12 %)**, 과제 표는 +1,024개. 즉 OMol25만 써도 이미 4천만 개 규모이고 범용 목록의 추가 비용은 12 %다.
- 1 에폭 / 2 에폭 체크포인트 비교(같은 시드): 없는 원소 87종의 모든 종별 표(16개)와 과제 1–7 행이 params·EMA 모두에서
  비트 단위로 초기값 그대로, 과제 0 행만 학습됨. AMSGrad f32와 SOAP f64 둘 다 통과.
- **SOAP에서 발견한 문제와 수정**: `soap_species_adam: true`는 선행 축이 원소인 표만 Adam으로 보낸다. `x_hill`
  (n_layers, 2, n_species)은 원소가 마지막 축이라 SOAP가 회전시키고, 그 고유기저가 있는 원소와 없는 원소를 섞어 없는 원소
  값이 한 에폭에 최대 0.11 움직였다. `soap_species_adam: "any_axis"`(어느 축이든 n_species이면 Adam)를 추가했고
  범용 목록 설정에는 자동으로 들어간다. 단위 테스트 `tests/test_soap_grouped.py::test_species_adam_any_axis_keeps_absent_species`.
- 테스트: `tests/test_race_lr.py::test_task_embedding_slot`(자리가 없는 모델과 초기에 동일, `task` 없으면 0번, 쓰지 않는
  행의 기울기 정확히 0, 범위 밖 번호는 마지막 행으로 클립, 힘 유한차분 일치). `test_race_lr.py` 24개, `test_soap_grouped.py` /
  `test_soap_laplace.py` 8개 통과(CPU).
- 로그의 E0 줄은 이제 "rank 7/94 (… 87 elements of atomic_numbers are absent)"로 나온다(예전 문구 "single stoichiometry"는
  이 경우 틀린 말이었다).
- 확인하지 않은 것: 스트리밍 로더(`data_format: stream`) 경로에서의 범용 목록(코드상 원소 목록은 같은 방식으로 쓰이지만
  돌려 보지 않음), 생산 크기에서의 실제 학습, 과제 번호가 0이 아닌 데이터로의 학습.
- 이어 받기 확인: 범용 목록 2 에폭 체크포인트를 `init_params_from`으로 받아 1 에폭 더 학습하면 검증 힘 MAE가 206.3에서
  180.7 meV/Å로 이어서 내려간다(AMSGrad f32; 옵티마이저 상태는 새로 시작). 같은 데이터로 한 확인이며 다른 데이터셋으로의
  전이는 아직 안 해 봤다.

### 7.3 나중에 잇는 두 경로 (일부 구현됨, 2026-11-10)
- **A. 트렁크 고정 + 새 헤드**: OMol25 성능이 변하지 않는다. 새 데이터셋에 대한 계산은 ②(마지막 층 선형 회귀)와 같은
  꼴이라 싸다. 다만 OMol25에 없던 원소(악티늄족 등)와 주기계 환경은 트렁크가 본 적이 없으므로 정확도 한계가 있다.
- **B. 이어서 학습**: 새 데이터셋에 과제 행을 주고 트렁크까지 학습. OMol25를 일정 비율 섞어(replay) 망각을 막는다.
  스트리밍 로더가 여러 소스를 지원하므로 혼합 자체는 가능하다.
- 코드 상태:

  | 항목 | 상태 |
  |---|---|
  | 1. 데이터 쪽 과제 키 | **구현됨.** 설정 `task`: 정수(그 설정의 모든 구조에 같은 번호, 시험 세트 포함) 또는 문자열(구조별 번호가 든 info 키, 없으면 0). 기본값 없음 = 과제 0. 메모리 내 로더와 스트림 로더 둘 다 `globals['task']`로 간다. 캐시 이름에 `_t-<task>`가 붙는다 |
  | 2. 과제별 E0 | 코드는 그대로다(`atom_energies` 한 벌). §7.4의 시험에서는 **데이터를 만들 때 각 데이터셋의 E0를 빼서** 기준 에너지로 쓰고 설정의 `atom_energies`를 0으로 뒀다. 이 방식이면 코드 변경이 필요 없다 |
  | 3a. 파라미터 고정 | **구현됨.** 설정 `train_only`: 파라미터 경로(`jax.tree_util.keystr`, 예 `['layers'][0]['skip']['weight_w_0_0_64x0e_64x0e'].value`)에 대한 정규식 또는 그 목록. 맞지 않는 잎은 기울기와 갱신이 0이고 EMA도 그대로다. 하나도 안 맞으면 오류. **표 안의 행 단위 고정은 없다** |
  | 3b. 데이터셋별 readout 헤드 | 없음 |
  | 4. 주기계 Qeq arm | 변동 없음. Ewald 설정과 블록 대각 Qeq가 필요(§4.3-6). `race_cs`(Qeq 없음)가 범용 트렁크로는 단순하다 |
  | 5. 일부 잎만 받는 체크포인트 로딩 | 없음. `init_params_from`은 파라미터 트리가 같아야 한다 |

### 7.4 소규모 전이 시험: OMol25 → OMat24 (로컬, 2026-11-10)

목적은 이어 받기 배관이 실제로 도는지 확인하는 것이다. 모델이 작고(L3x64 lmax 2, `race_cs`, Z 1..94, `n_tasks` 8)
데이터가 각 1만 구조, 시드 1개라서 **정확도 결론을 내는 시험이 아니다.**

- 1단계: OpenFF가 재배포한 OMol25 부분집합 1만 구조(원소 7종)로 40 에폭, 비주기, AMSGrad f32.
  설정 `input/omol/tr_u94_race_cs_amsgrad_f32_s42.json`.
- 2단계: OMat24 검증 분할의 `rattled-300-subsampled` 조각에서 뽑은 1만 구조(과제 1, 원소 85종), 40 에폭, 주기. 네 갈래.
  시험 2,000 구조는 학습과 검증에 쓰지 않은 행에서 뽑았다.
  설정과 데이터는 `runs/diag/transfer/prepare.py`가 만든다. 분자는 진공 상자(변 = 분자 크기 + 2 × 컷오프 + 2 Å)에 넣어
  주기 모델 하나로 두 데이터를 읽는다. 에너지는 각 데이터셋의 E0(학습 분할만으로 맞춘 원소별 최소제곱 값)를 뺀 값이다.
- 평가는 `bam_omat24.eval_sets` 하나로 두 시험 세트(OMat24 2,000, OMol25 2,932 구조)에 대해 했다.
  표는 `runs/diag/transfer/collect.py`가 만든다(`runs/diag/transfer/results.json`).

| 갈래 | OMat24 E MAE (meV/원자) | OMat24 F MAE (meV/Å) | OMol25 E MAE (meV/원자) | OMol25 F MAE (meV/Å) |
|---|---|---|---|---|
| 1단계 모델 그대로 | 1060.2 | 2013.2 | 79.2 | 209.8 |
| 처음부터 (`tr2_scratch`) | 93.3 | 229.5 | 336.1 | 818.9 |
| 이어서 학습 (`tr2_cont`) | 99.3 | 237.4 | 767.8 | 751.7 |
| 이어서 + OMol 1만 혼합 (`tr2_replay`) | 111.5 | 254.1 | 74.8 | 206.0 |
| 일부 고정 (`tr2_frozen`) | 161.4 | 348.8 | 127.7 | 655.2 |

확인된 것:
- **배관이 돈다.** 비주기로 학습한 체크포인트를 주기 설정으로 받아 과제 1 데이터로 학습하고, 두 데이터셋을 한 모델로 평가했다.
- **상자에 넣은 분자의 주기 평가가 비주기 평가와 같다.** 1단계 모델의 OMol25 시험 오차가 학습기의 비주기 값과 일치한다
  (E 79.15 meV/원자, F 209.80 meV/Å, F RMSE 817.07 meV/Å).
- **혼합(replay)이 망각을 막는다.** OMol25 오차가 1단계와 같은 수준으로 남는다(F 206.0 대 209.8). 혼합 없이 이어서 학습하면
  OMol25 힘 오차가 3.6배가 된다(751.7).
- **과제 행.** 처음부터 학습한 갈래는 과제 1 행만 0이 아니다. 이어 받은 갈래는 0과 1 행이 0이 아니다.
- **없는 원소 행.** 두 데이터셋 어디에도 없는 원소 9종의 종별 표 행은 원시 파라미터에서 1단계 값과 비트 단위로 같다.
  EMA 평균에서는 float32 반올림으로 최대 3.3e-6(상대 8e-7) 움직인다.
- **고정.** `tr2_frozen`에서 `train_only` 밖의 잎 55개가 1단계 값과 비트 단위로 같고, 안의 잎 55개는 모두 변했다.

이 시험에서 얻지 못한 것:
- **사전학습의 이득은 없었다.** `tr2_cont`는 모든 에폭에서 `tr2_scratch`보다 약간 나빴다(검증 F MAE 5 에폭 342.8 대 320.6,
  40 에폭 241.0 대 234.7 meV/Å). 1단계 데이터가 원소 7종의 중성 유기분자 1만 개뿐이라 85원소 결정에 줄 것이 거의 없다.
  전체 OMol25(83원소)에서는 다를 수 있지만 이 시험으로는 알 수 없다.
- **`tr2_frozen`은 "트렁크 고정"이 아니다.** 종별 표를 풀어 둔 탓에 파라미터의 93 %(5,416,447 / 5,854,559)가 학습됐다.
  OMat24는 가장 나쁘고 OMol25도 망가졌다(F 655.2). 종별 표는 두 데이터셋이 공유하는 원소 행을 함께 쓰므로, OMol25를
  그대로 보존하려면 데이터셋별 헤드(3b)나 행 단위 고정이 필요하다.
- 혼합 갈래는 OMat24에서 조금 손해를 본다(F 254.1 대 229.5). 에폭당 스텝이 두 배라 시간도 두 배다(1077 s 대 537 s).

이어 받을 때 지켜야 할 것(이 시험에서 정리됨):
- **`avg_n_neighbors`를 사전학습 값으로 고정한다.** `message_norm: "avg"`는 메시지를 이 상수로 나누는데, 이 값은 체크포인트가
  아니라 설정에서 온다. 이 시험에서는 OMol 14.137을 넣었다(OMat24 조각의 값은 약 19.8).
- `shift`와 `scale`은 파라미터라서 체크포인트와 함께 온다. `shift` 하나를 모든 과제가 함께 쓴다. 2단계 설정은 `gauge_refit: false`.
- 옵티마이저 상태와 스케줄은 새로 시작한다.
