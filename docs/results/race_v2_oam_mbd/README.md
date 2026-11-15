# RACE v2 Matbench Discovery 결과 모음 (MPtrj-only, OAM)

2026-11-11~16 계산 창(KISTI HANGANG, exclusive06, 11/16 09:00 종료)에서 RACE v2를 Matbench Discovery 기준으로 학습하고
평가한 결과다. 방법론(수식)과 결과를 정리한 문서는 `report/RACEv2_MBD_methods_results_ko.pdf`(13쪽)다.

학습은 브랜치 `racev2`(패키지 옛 이름 `bam_omat24`)의 코드로 했다. 주 브랜치 `main`에서는 패키지 이름이 `race_v2`이고,
옛 이름과 새 이름의 대응은 `docs/LAYOUT.md`에 있다. 체크포인트, 크기표, 평가 원본은 여기에 넣지 않았다(5절).

## 1. 결과

WBM 전체(고유 원형 215,488개), 보정 없음, float64 추론. 최종 체크포인트는 미리 정한 마지막 에폭이다.

| 모델 | 파라미터 | 에폭 | F1 | MAE (eV/원자) | R2 | RMSD (Å) | κ SRME | CPS | 리더보드 순위 |
|---|---|---|---|---|---|---|---|---|---|
| RACE v2 OAM | 17.8M | 사전학습 2 + 파인튜닝 6.2 | 0.893 | 0.0214 | 0.872 | 0.0643 | 0.229 | 0.858 | 49개 중 20위 수준 |
| RACE v2 MPtrj-only | 17.8M | 40 | 0.777 | 0.040 | 0.800 | 0.0785 | 0.375 | 0.761 | MPtrj-only 19개 중 6위 수준 |
| BAM-MP-DeNS (이전 모델) | - | - | 0.761 | 0.043 | 0.797 | 0.0832 | 0.766 | 0.672 | |
| BAM-OAM (이전 모델) | 10.5M | 4 + 64 | - | 0.027 | - | - | - | - | |

- 리더보드 값은 2026-11-15에 읽었다(`scripts/eval/mbd/mbd_leaderboard_2026-11-12.tsv`). 다른 모델과의 비교표는
  `figures/sota_table_racev2_oam.png`.
- 불확실성(SOAP 고유기저 Laplace): 힘 σ의 최대값으로 OMol25, OPoly26, OC25를 MPtrj·sAlex 검증 세트와 AUROC 0.96으로
  구분한다. WBM에서 오차가 큰 구조를 가려내는 AUROC는 0.56~0.59로 약하다. σ의 절대 크기는 오차보다 작다(에너지 약 16배).
- GPU 하나(GH200)에서 한 번에 계산할 수 있는 크기: float32 간선 41만~56만 개, float64 17만~26만 개.

## 2. 학습 설정 요약

- 모델: RACE, 7 Å, 128 채널, lmax 3, 5층, 원소별 표 랭크 32, 파라미터 17,757,551개. float32 모델 + float64 SOAP 상태.
- 로스: Huber(에너지 0.05 eV/원자, 힘 0.25 eV/Å, 응력 0.05 eV/Å³), 가중치 1 : 1 : 1, EMA 0.999.
- 옵티마이저: SOAP(b1 = b2 = 0.95, 전처리 갱신 10 스텝마다), 기울기 자르기 1.0.
- MPtrj-only: 20 에폭 뒤 20 에폭 선형 감쇠. 설정 `configs/mptrj/`.
- OAM: OMat24 2 에폭 사전학습(`configs/omat_pretrain/`) → MPtrj × 8 + sAlex × 1 파인튜닝(`configs/oam_finetune/`).
  가상 에폭 하나는 5,000 스텝이다. 반복 사본은 에폭 전체에 흩어 읽는다(`stream.train.parts`).

## 3. 폴더 내용

| 경로 | 내용 |
|---|---|
| `report/RACEv2_MBD_methods_results_ko.pdf` | 방법론과 결과 문서 |
| `report/build_report.py`, `pdfkit.py`, `assets/` | PDF를 만드는 스크립트와 수식 그림 |
| `report/methods_formulas.md` | 코드에서 옮긴 수식 정리 (파일 이름과 줄 번호는 브랜치 `racev2`, 커밋 3545c1d 기준) |
| `report/final_results.json`, `wbm_points.json` | PDF에 들어가는 최종 수치와 3k 표본 추적값 |
| `report/plot_*.py`, `table_sota.py` | 그림과 비교표를 만드는 스크립트 |
| `figures/` | 문서의 그림: 학습 곡선, 사전학습, 파인튜닝, y=x 산점도, OOD, 비교표 |

PDF 다시 만들기(로그인 노드, reportlab과 matplotlib만 씀):

```bash
cd docs/results/race_v2_oam_mbd/report
python -I build_report.py <NanumGothic 폰트 폴더> ../figures RACEv2_MBD_methods_results_ko.pdf final_results.json
```

## 4. 평가를 다시 하는 방법

실행 폴더(HANGANG에서는 `/scratch/t510a13/race_v2/work`)에 `race_v2 -> <저장소>/race_v2`,
`scripts -> <저장소>/scripts` 심볼릭 링크를 두고 실행한다.

| 할 일 | 스크립트 |
|---|---|
| 체크포인트 하나의 전체 평가 (float64: 3k 표본, κ, 전체 WBM, RMSD, CPS) | `bash scripts/eval/mbd/eval_x64.sh <체크포인트> <태그>` (작업 5개 제출) |
| 학습 중 에폭마다 3k 표본 평가 | `scripts/eval/mbd/eval_loop.sh` |
| 검증 오차 표 | `python -I scripts/eval/mbd/val_table.py <실행 이름>` |
| 리더보드 순위 | `python -I scripts/eval/mbd/rank_mbd.py F1=.. MAE=.. R2=.. RMSD=.. kSRME=..` |
| Laplace 불확실성과 OOD | `scripts/eval/uq/ood_laplace.sh`, 그림은 `scripts/eval/uq/plot_ood.py` |
| 메모리·속도 측정, PyTorch 비교용 기준값 | `scripts/eval/bench/bench.sh` |
| PyTorch 변환본과 JAX 비교 | `race_v2_torch/tools/compare_pred.py` |

- κ와 정밀 이완은 `BAM_PREDICT_X64=1`로 추론한다(`docs/inference_precision.md`). float32로는 κ SRME가 0.861로 나온다.
- 이번 학습에서 쓴 일회용 스크립트(파인튜닝 시작, 재시작, 체크포인트 백업)는 `research/diag/oam_run_2026-11/`에 기록으로 남겼다.

## 5. 여기에 없는 것 (HANGANG)

| 내용 | 위치 |
|---|---|
| 체크포인트 (최종, 사전학습 1·2 에폭, 파인튜닝 중간), PyTorch 모델 | `/appsdata/t510a13/race_v2_backup_2026-11-15/{checkpoints, torch}` |
| 7 Å 크기표, 다음 작업 목록 | `/appsdata/t510a13/race_v2_backup_2026-11-15/{oam_sizes, NEXT_TASKS.md}` |
| 학습 로그, 구조별 예측값, OOD 결과 | `/appsdata/t510a13/archive/race_v2_oam_mbd_20261116.tar.gz` |
| 이어서 작업하기 위한 묶음 (코드, 설정, 핵심 체크포인트, 메모) | `/appsdata/t510a13/archive/race_v2_resume_20261116.tar.gz` |
| 사전처리 결과 설명 | `scripts/data/prep_results/README.md`, 내보내기 폴더의 `prep/README.md` |

## 6. 알려진 한계

- 사전학습이 2 에폭으로 짧다. 상위 모델은 4~6 에폭 이상, 파라미터 3천만 개 이상이다.
- 응력 로스가 Voigt 성분 6개의 합이라 의도보다 6배 크게 들어간다.
- PyTorch 변환본과 JAX float64의 차이가 에너지 4e-7 eV, 힘 2e-6 eV/Å다(5 Å 모델에서는 1e-14). 원인은 확인하지 않았다.
- Laplace σ는 사전분포 정밀도를 맞추지 않아 오차 막대로 쓸 수 없다. 원자당 에너지 σ는 구조 크기에 따라 달라지므로
  OOD 점수로는 힘 σ나 전체 에너지 σ를 쓴다.
