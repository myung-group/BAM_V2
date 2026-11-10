# 인계 문서: 불확정도(UQ) 방법 현황과 OMol25 전체 데이터셋 계획 (2026-11-10)

다른 Claude Code 세션이 이 문서만 읽고 이어서 진행할 수 있도록 쓴다. 먼저 `docs/HANDOFF.md`(규칙·환경·경로)를
읽고, 그 다음 이 문서를 읽는다. 세부 수치는 `docs/local_test_report.md` §17–20, 방법은 `docs/SOAP_Laplace.md`.

## 0. 사용자 결정 (2026-11-10)

- **OMol25를 전체 데이터셋으로 해보기로 함.** ("전체 데이터셋으로 해볼까 해") 파일럿 부분집합이 아니라 공식 학습
  분할(약 1억 구조) 전체를 목표로 한다. 다만 §4의 순서(접근 → inspect → 변환 → 스트리밍 스모크 → 소규모 파일럿 → 전체)는
  지키고, 전체 학습은 다중 GPU/다중 노드가 전제이므로 그 전에 다중 호스트 로더(T5)부터 검증한다.
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
   racelr / racelr_q × soap_f64 / amsgrad_f32)로 설정을 만들고 30 에폭. 로컬 OpenFF 부분집합 결과(report §11):
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
