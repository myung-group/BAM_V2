# 연구 목적: 왜 RACE_LR을 OMol25로 사전학습하는가 (2026-11-10)

다른 Claude Code 세션이 "이 모델을 무엇에 쓰려고 학습하는가"를 이 문서만 읽고 알 수 있게 쓴다.
규칙과 환경은 `docs/HANDOFF.md`, 실행 계획은 `docs/HANDOFF_uq_omol25.md` §4에 있다.
수치는 모두 ctx-RACE 쪽 인계 문서(§8)에서 옮긴 것이다. 이 저장소에서 다시 계산한 값이 아니다.

## 0. 한 줄 요약

최종 목표는 **단백질–리간드 결합의 상호작용 에너지 ΔE_int를 DFT 수준으로 맞추는 MLIP**이다.
기존 사전학습 모델은 총전하 입력이 없고 단거리라서 충전된 복합체에서 크게 틀린다.
그래서 총전하·스핀 입력과 장거리 정전기를 갖춘 RACE_LR(RACEv2 기반, JAX)을 OMol25로 처음부터 사전학습한다.

## 1. 배경: ctx-RACE 프로젝트

- 계산하는 양: `ΔE_int = E(complex) − E(protein pocket) − E(ligand)`. 같은 기하, 기체상, 단위 kcal/mol.
  포켓은 리간드에서 8 Å 이내 잔기를 자르고 캡을 씌운 클러스터다(약 100–1,000 원자).
- 기존 모델: BAM-torch(PyTorch)의 RACE 사전학습 체크포인트 `model_best.pkl`. OMol 데이터,
  ωB97M-V/def2-TZVPD, 5층, 컷오프 6 Å, 약 1,050만 파라미터, **총전하·스핀 입력 없음**.
- ctx-RACE는 구조 예측 모델(K-Fold)의 임베딩을 RACE에 문맥으로 넣으면 ΔE_int가 좋아지는지 본 프로젝트였다.
  결론은 "QM 에너지에는 도움이 안 된다"였다. 그 뒤 역할을 나눴다.
  - MLIP + 새 전하·장거리 모델: 에너지, 힘, ΔE_int. **이 저장소의 RACE_LR이 이 역할이다.**
  - K-Fold: 구조/포즈 예측, 실험 친화도(ΔG) 보정 항의 입력.

## 2. 왜 새 모델이 필요한가 (증거)

| 관찰 | 수치 (ΔE_int MAE, kcal/mol) |
|---|---|
| 중성 복합체는 잘 맞는다 | KIN66(전부 중성) 3.86, 공개 최고 수준 |
| 충전 복합체는 크게 틀린다 | PLA15(전부 충전) 24.1. 전하 인식 모델은 3.7–9.3 |
| 오차의 부호가 전하 곱을 따른다 | 포켓 OOD 인력쌍 +34.7, 반발쌍 −37.5 |
| 단일 항 a·q_prot·q_lig로 대부분 설명된다 | 포켓 OOD 전체 18.4 → 8.0 |
| 그 계수는 세트마다 다르다 | 34.6 / 24.8 / 14.3. 상수가 아니라 기하로 계산해야 한다 |

- 원인 두 가지: 모델이 총전하를 모른다. 유효 거리가 짧아 6–12 Å의 이온쌍 상호작용이 빠져 있다.
- 미세조정으로는 고쳐지지 않았다. 모델이 배울 수 있는 것은 전역 상수 이동뿐이라 중성 복합체가 나빠졌다
  (KIN66 부호 평균 −3.1 → −6.5).
- 기존 PyTorch 모델의 문제는 성능이 아니라 구조다(총전하 입력 없음, 단거리).

## 3. 사용자 결정의 흐름

| 날짜 | 결정 |
|---|---|
| 2026-11-06 | RACEv2에 전하 평형화와 장거리 정전기를 더한 RACE_LR을 로컬에서 검증 (`docs/LOCAL_TEST_PROMPT.md`, `docs/local_test_report.md`) |
| 2026-11-06–07 | 기존 PyTorch 체크포인트 위에 LR 모듈만 얹는 경로는 버림. **JAX RACEv2 계열로 OMol25를 처음부터 사전학습**하고 총전하·스핀을 처음부터 입력으로 넣음 |
| 2026-11-10 | 부분집합이 아니라 **OMol25 전체 학습 분할**로 진행 |
| 2026-11-10 | 나중에 OMat24 / MPtrj / Alexandria / OC20을 잇거나 헤드를 붙일 수 있게 범용 원소 목록(Z 1..94)과 과제 임베딩 자리를 미리 둠. 지금 목표는 아니고 길만 열어 둔 것 |

OMol25를 고른 이유:
- 레벨이 ωB97M-V/def2-TZVPD로 ctx-RACE의 포켓 DFT 라벨과 정확히 같다. 레벨이 다른 데이터를 섞으면 backbone이
  그 레벨로 끌려가고, 헤드를 나눠도 막히지 않았다.
- 총전하와 스핀 다중도가 있다. 생체분자 부분도 들어 있다고 알려져 있다(공식 데이터에서 확인 필요).
- 포켓 데이터(약 2.1만 구조)만으로는 단거리 퍼텐셜을 새로 만들 수 없다.

## 4. 사전학습 모델이 갖춰야 할 것 (ctx-RACE 요구사항)

| 요구 | 상태 |
|---|---|
| 원소 15종 이상: H, B, C, N, O, F, Mg, Si, P, S, Cl, Ca, Zn, Br, I | 범용 목록 Z 1..94로 충족 |
| 총전하·스핀 입력 | 구현됨 (`long_range.charge_spin_embedding`, `total_charge_key`, `total_spin_key`) |
| 단편별 총전하 제약 (단백질 / 리간드) | 구현됨 (`fragment_key`, `fragment_charges_key`) |
| 단거리 스위치 | 구현됨 (`long_range.sr_switch`). 로컬 NaCl에서는 [3, 5] Å가 이득을 없앴다 |
| χ 초기화 Pauling 기본값 | 구현됨 |
| 전하 손실은 한 가지 방식으로 통일 (Löwdin 권장. Mulliken은 이 기저에서 불안정) | 공식 데이터 inspect 뒤 결정 |
| 같은 데이터·레시피의 RACE 대조군 (LR 없음) | 파일럿 설정의 arm으로 준비됨 (`race`, `race_cs`, `racelr`, `racelr_q`) |
| HOMO > 0 다중 음이온 필터, PDB 누수 점검 | 미결. 공식 데이터 inspect 뒤 |
| 추론 API | 구현됨 (`race_v2/inference/predict.py`: `load` / `predict` / `charges` / `refit`) |

세부는 `docs/RACE_LR_design.md`, `docs/RACE_LR_feedback_reply_2026-11-07_ko.md`,
`docs/RACE_LR_reply2_2026-11-07_ko.md`에 있다.

## 5. 성공 기준

최종 판단은 OMol25 검증 오차가 아니라 아래 표다(ΔE_int MAE, kcal/mol). pre_zero는 기존 PyTorch 체크포인트다.

| 지표 | pre_zero | 넘어야 할 기준 |
|---|---|---|
| 포켓 OOD 충전 리간드 | 25.8 | 10.2 (단순 쿨롱 기준선) |
| 포켓 OOD 전체 | 18.4 | 8.0 |
| 포켓 OOD 중성 리간드 | 3.5 | 유지 |
| KIN66 (전부 중성) | 3.86 | 유지 |
| PLA15 (전부 충전) | 24.1 | 3.7–9.3 (전하 인식 모델) |
| PL-REX 충전쌍 | 63.0 | 35.2 (고정 계수 보정) |

- 인력쌍과 반발쌍의 부호 오차가 둘 다 줄어야 한다.
- 힘: 포켓 OOD 98.4 meV/Å, 재라벨 SPICE held-out 96.9 meV/Å보다 나빠지지 않아야 한다.
- 실험 친화도(ΔG)는 이 모델의 성공 기준이 아니다. 기체상 ΔE_bind는 ΔG 정보를 거의 담지 못했다.
- OMol25 자체의 확인은 Val Comp 오차와 fairchem 리더보드로 한다. 리더보드의 Evaluations 부문에 리간드–포켓
  항목이 있다(`docs/HANDOFF_uq_omol25.md` §4.2).

## 6. 사전학습 뒤의 순서

1. 무보정 평가: 새 사전학습 모델, RACE 대조군, pre_zero를 §5의 표로 함께 본다.
   ctx-RACE 쪽 평가 코드가 `race_v2.inference.predict:load`를 별도 프로세스로 부르는 어댑터를 갖고 있다. 실제 백엔드로는 아직 안 돌렸다.
2. 포켓 미세조정: ctx-RACE의 A5 레시피(lr 1e-5, warmup 100, cosine, clip norm 1, EMA, 상대에너지 + ΔE_int Huber).
3. 미세조정 뒤 KIN66 부호 오차가 음수 쪽으로 밀리는지 확인한다.

## 7. 이 저장소의 다른 연구 줄기와의 관계

| 줄기 | 목적 | 문서 |
|---|---|---|
| RACEv2 논문 | 국소 분해능이 높은 MLIP. 3BPA, rMD17 등 벤치마크 | `docs/HANDOFF.md` §1, `README.md` |
| RACE_LR + OMol25 사전학습 | 이 문서. 단백질–리간드 ΔE_int | `docs/RACE_LR_design.md`, `docs/HANDOFF_uq_omol25.md` §4 |
| 불확정도 (② BLR, ③ SOAP Laplace) | 사후 불확정도와 OOD 판별. BAM V1 논문의 3BPA OOD 설정으로 검증. OMol25에서는 전체 학습 뒤 Val Comp로 평가 | `docs/HANDOFF_uq_omol25.md` §1–3 |
| 범용 모델 (OMat24 등) | 지금 목표 아님. OMol25 모델을 버리지 않고 이을 수 있게만 해 둠 | `docs/HANDOFF_uq_omol25.md` §7 |

- 공정 프로토콜(`docs/HANDOFF.md` §2)은 보고 수치에 대한 규칙이다. OMol25에서도 공식 학습 분할만 쓴다
  (`docs/HANDOFF_uq_omol25.md` §4.4). OMol25로 사전학습 모델을 만드는 일 자체는 이 규칙과 충돌하지 않는다.
- ctx-RACE 저장소(BAM-torch)는 커밋 금지다. 이 저장소(BAM_V2)는 racev2 브랜치 커밋이 허용된다.

## 8. 근거 문서

이 문서는 사용자가 전달한 ctx-RACE 쪽 문서 세 개를 요약한 것이다. 원본은 이 저장소에 없다.

| 문서 | 내용 |
|---|---|
| `HANDOFF_charge_lr_model.md` (2026-11-06, 11-07 갱신) | 왜 전하·장거리 모델이 필요한지, 데이터, 기준선, 평가 |
| `RACE_LR_pretrain_feedback.md` (2026-11-07) | 처음부터 사전학습할 때의 요구사항과 성공 기준 |
| `RACE_LR_ctx_side_reply.md` (2026-11-07) | 단편 전하가 들어간 포켓 데이터셋, 평가 어댑터 |

- ctx-RACE의 코드, 데이터, 결정 기록은 HANGANG의 다른 계정 경로에 있다고 적혀 있다
  (`/scratch/t590a03/BAM-torch`, `/scratch/t590a03/data/ctx_race/`, 결정 기록 `docs/ctx_race/DECISIONS.md` D28–D56).
  지금 계정에서 읽을 수 있는지는 확인하지 않았다.
- 포켓 데이터셋: `dataset_dft_spice_relab_frag` (train 21,091 구조 중 포켓 1,620, valid 560, ood 540).
  평가셋: KIN66, PLA15, PL-REX, OpenFE PLB, Schindler FEP.

## 9. 기록에 없는 것

- 단백질–리간드 연구의 상위 목표(대상 단백질, 논문 계획)는 기록에 없다. 필요하면 사용자에게 묻는다.
- 불확정도를 단백질–리간드 쪽에서 어떻게 쓸지는 정해진 바 없다.
- OMol25 전체 학습의 모델 크기, 노드 수, 에폭 수는 아직 정하지 않았다(`docs/HANDOFF_uq_omol25.md` §4.3).
