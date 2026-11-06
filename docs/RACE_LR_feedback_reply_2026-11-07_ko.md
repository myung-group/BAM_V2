# RACE_LR 개발 쪽 답변: 사전학습 요구사항 반영 상태

작성 2026-11-07. 대상은 ctx-RACE 쪽 세션과 사용자입니다.
입력 문서는 `RACE_LR_pretrain_feedback.md`와 `HANDOFF_charge_lr_model.md`(2026-11-07 갱신본)입니다.
코드는 BAM_V2 저장소 브랜치 `racev2`에 있습니다(JAX). 아래에서 "완료"는 코드와 테스트가 끝났다는 뜻입니다.
OMol25 사전학습 자체는 아직 하지 않았습니다.

---

## 1. 요구사항별 상태

| 피드백 항목 | 상태 | 내용 |
|---|---|---|
| 3.5 단편별 총전하 제약 | **완료** | 단편마다 정수 전하를 강제하고, 단편 사이 쿨롱 결합은 유지합니다. 테스트에서 단편 전하 +1 / −1이 1e-10 이내로 지켜졌습니다. 전역 제약만 쓰면 같은 단편의 합이 0.1 e 넘게 어긋났습니다. |
| 3.5 단거리 스위치 | **완료** | `long_range.sr_switch: [r_on, r_off]`. r_off보다 가까운 원자쌍의 쿨롱 에너지를 C² smoothstep으로 부드럽게 끕니다. 전하는 스위치 없는 행렬로 풉니다. 분자와 주기계 모두에서 힘과 응력이 미분과 일치합니다. |
| 3.5 χ 초기화 Pauling 기본값 | **완료** | 기본값을 `"pauling"`으로 바꿨습니다. 이전 설정에는 `"chi_init": "zero"`를 명시해서 기존 결과가 그대로 재현됩니다. |
| 3.5 원소별 에너지 보정 | **완료 (사후 보정)** | 학습 분할로 원소별 보정을 맞추는 도구를 만들었습니다(아래 2절). 학습할 때의 E0는 원래부터 원소별 최소제곱입니다. |
| 3.5 사전학습에 전하 손실 | **준비됨** | `charge_label_key` + `charge_weight`. OMol25의 Löwdin 전하로 통일하는 것을 권하는 데 동의합니다. Mulliken은 쓰지 않습니다. |
| 5 평가 어댑터용 `load` / `predict` | **완료** | `bam_omat24/predict.py` (아래 2절). |
| 3.1 공식 OMol25 | 대기 | Hugging Face 승인(약관 동의와 토큰)이 필요해서 사용자 계정으로만 받을 수 있습니다. 다운로드는 HANGANG 데이터 이동 노드에서 하고, 저장은 `/appsdata`에 합니다. |
| 3.2 원소 15종 | 사전학습 때 확인 | OMol25는 83개 원소를 다룹니다. 설정의 `atomic_numbers`에 15종(H, B, C, N, O, F, Mg, Si, P, S, Cl, Ca, Zn, Br, I)이 모두 들어가는지, 각 원소의 학습 구조 수가 충분한지 변환 단계에서 셉니다. |
| 3.3 HOMO > 0 다중 음이온 제외 | 데이터 확인 뒤 | 공식 OMol25 행에 HOMO 필드가 있는지 아직 모릅니다. 변환기의 `inspect` 모드로 확인한 뒤 필터를 넣겠습니다. |
| 3.6 데이터 누수 점검 | 데이터 확인 뒤 | 메타데이터에 PDB ID나 리간드 정보가 있으면, KIN66, PL-REX, OpenFE, Schindler 복합체와 겹치는 구조를 빼거나 따로 보고합니다. |
| 3.7 RACE 대조군 | 계획에 포함 | 시범과 본 학습 모두 같은 데이터와 레시피의 RACE(LR 없음)를 함께 학습합니다. `make_omol_configs.py`의 `race` 비교군입니다. |
| 3.8 옵티마이저 재측정 | HPC에서 | 로컬 결과 "AMSGrad float32가 10배 빠름"은 RTX 4060 Ti 기준입니다. GH200에서 다시 재겠습니다. |

---

## 2. ctx-RACE 평가 어댑터용 인터페이스

```python
from bam_omat24.predict import load

pred = load('runs/<name>/ckpt.pkl')      # 설정은 체크포인트 안에 들어 있습니다
energies, forces = pred.predict(frames)  # energies: (n_frames,) eV 총에너지, forces: (n_atoms, 3) eV/Å의 목록
charges = pred.charges(frames)           # RACE_LR일 때만: 프레임별 (n_atoms,) 전하, 단위 e
```

- **입력:** `ase.Atoms` 목록입니다. 정답 에너지나 힘은 필요 없습니다.
- **각 프레임에서 읽는 값:** 키 이름은 학습 설정을 따릅니다.
  - `info["charge"]`: 그 부분의 총전하 (`total_charge_key`)
  - `info["spin"]`: 스핀 다중도, 없으면 1 (`total_spin_key`)
  - `arrays["is_ligand"]`: 단편 번호 0 = 단백질, 1 = 리간드 (`fragment_key`)
  - `info["fragment_charges"]`: `[q_단백질, q_리간드]` (`fragment_charges_key`)
- **단편 전하 규칙:**
  - complex 프레임에는 `info["fragment_charges"]`가 필요합니다. 두 값의 합은 `info["charge"]`와 같아야 하고, 다르면 오류가 납니다.
  - protein만 있거나 ligand만 있는 프레임은 원자가 한 단편에만 속하므로 `fragment_charges`가 없어도 됩니다. 그 단편이 총전하를 가집니다.
  - **ctx-RACE 쪽에서 할 일:** 포켓 데이터의 complex 프레임에 `fragment_charges`를 넣어 주셔야 합니다. 같은 system과 frame의 protein / ligand 부분의 `info["charge"]`에서 가져오면 됩니다.
- **원소별 보정:** 학습 분할로 한 번 맞추면 이후 `predict`에 자동으로 적용됩니다.

  ```bash
  python -m bam_omat24.predict refit runs/<name>/ckpt.pkl --xyz <학습 분할.xyz>
  ```

- **검증:** 학습된 탄소 사슬 체크포인트에서 `predict`가 학습 코드의 시험 결과와 정확히 같은 값(에너지 10.77 meV, 힘 9.21 meV/Å)을 냈습니다. 원소별 보정 뒤 에너지는 2.23 meV로, 분석 스크립트의 값과 같습니다.
- **실행 환경:** JAX입니다. 체크포인트 설정이 `tp_backend: "cue"`이면 cuequivariance-jax가 있는 환경이 필요합니다. HANGANG에서는 BAM_V2의 `bam_cue` conda 환경입니다. PyTorch 평가 스크립트에서 부를 때는 별도 프로세스로 실행하는 어댑터가 안전합니다.

---

## 3. 포켓 데이터에 쓸 설정 예시

```json
"model": "RACE_LR",
"total_charge_key": "charge",
"total_spin_key": "spin",
"fragment_key": "is_ligand",
"fragment_charges_key": "fragment_charges",
"n_fragments": 2,
"long_range": {"chi_init": "pauling", "charge_spin_embedding": true, "sr_switch": [4.0, 6.0]},
"charge_label_key": "ref_charges",
"charge_weight": 0.1
```

- `sr_switch`는 사전학습 모델의 단거리 범위(cutoff 6 Å)에 맞춘 예시입니다. 값은 사전학습 시범에서 정합니다.
- 포켓에는 전하 라벨이 없으므로, 포켓으로 미세조정할 때는 `charge_label_key`와 `charge_weight`를 뺍니다. 피드백 6절과 같습니다.

---

## 4. 로컬 결과 중 판단에 쓸 것 (요약)

- **4G-HDNNP 네 계:** Pauling 초기화와 전하 손실을 함께 쓰면 전하가 DFT와 맞고(상관 0.98–1.00), 에너지와 힘도 가장 좋았습니다.

  | 계 | 힘 RMSE (meV/Å), RACE → RACE_LR 최종 변형 |
  |---|---|
  | 탄소 사슬 | 20.2 → 9.2 |
  | NaCl | 43.3 → 35.8 |

- **OMol25 공개 사본(중성 분자만, 학습 1만 개, 30 에폭):** 전하 손실은 전하만 맞췄고(상관 0.93), 에너지와 힘은 좋아지지 않았습니다. 전하 상태가 섞인 공식 데이터로 다시 판단해야 합니다.
- **중성 데이터와 χ 초기화:** 중성 데이터에서 χ를 0으로 초기화하면 전하가 생기지 않습니다. 이번 기본값 변경으로 해결했습니다.
- **주의:** 모두 축소 모델과 시드 하나의 결과입니다.

---

## 5. 남은 일과 순서

1. 사용자가 공식 OMol25 Hugging Face 접근 권한을 받습니다.
2. HANGANG 데이터 이동 노드에서 다운로드합니다. 쿼터는 `lfs quota -gh <group> /appsdata`로 먼저 확인합니다.
3. `convert_omol25.py inspect`로 행 구조를 확인합니다. 확인할 필드는 전하, 스핀, Löwdin, HOMO, PDB 메타데이터입니다.
4. HOMO 필터와 누수 점검을 넣고 10–100k 시범을 돌립니다. 비교군은 RACE, RACE_LR, RACE_LR + 전하 손실이고, 옵티마이저 두 가지를 GH200에서 재측정합니다.
5. 본 사전학습 전에 스트리밍 로더를 연결하고, 전하 평형화를 구조별 블록 풀이로 바꿉니다. 지금은 배치 전체를 한 행렬로 풀어서 수백 원자 구조가 많으면 비쌉니다.
6. 사전학습 뒤 ctx-RACE 쪽 평가를 합니다. 순서는 무보정 평가, 포켓 미세조정(A5 레시피), D53 재발 확인이고, 이때 2절의 `predict`를 씁니다.
