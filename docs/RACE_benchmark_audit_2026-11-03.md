# RACEv2 벤치마크 감사 보고서: 데이터셋 적절성과 SOTA 비교 검증 (2026-11-03)

**범위**: 3BPA, AcAc, rMD17, water, Si GAP-18, LiPS, QM9
**보고 규칙**
- 모든 수치는 1차 출처(논문 표, 본문, 그림)나 로컬 run 파일에서 확인한 값입니다. 어디서 읽었는지(표, 본문, 그림)를 함께 적었습니다.
- 출처에서 확인하지 못한 값에는 **[미검증]**을 붙였습니다.
- 공정 프로토콜은 "각 벤치마크의 자체 training set만 사용, 사전학습·추가 데이터·증강·앙상블 없음, from-scratch"를 뜻합니다. 이를 어긴 모델은 **[비교불가]** 블록으로 따로 모았습니다.

---

## 1. 요약

### 전반 판정
- **데이터셋 자체는 7개 모두 "조건부 적절"입니다.** 쓰면 안 될 데이터셋은 없습니다. 다만 데이터셋마다 무엇을 입증할 수 있는지가 다릅니다.
  - **외삽(extrapolation)을 실제로 시험하는 것**: 3BPA(T-set, dihedral), AcAc 600K, Si RSS/compression.
  - **보간(interpolation) 전용**: rMD17, LiPS-19k, water, QM9.
  - **locality는 어떤 분자 벤치마크로도 입증되지 않습니다.** 3BPA 분자의 지름은 약 9 Å(MACE 본문)이고, 5층 × 5 Å 모델의 receptive field가 분자 전체를 덮습니다.
- **"가장 강한 SOTA와 비교하고 있는가"에 대한 답은 대부분 "아니오"입니다.** PACE, GotenNet, FreeCG, ICTP, CarNet, TensorNet 최종본 같은 최신 from-scratch 모델이 비교표에서 여러 곳 빠져 있고, 인용한 값 일부는 구버전 preprint 값입니다.
- **데이터셋 공통의 심각한 문제가 두 가지 있습니다.**
  - (i) 사후 energy gauge refit을 거의 모든 데이터셋에 적용했지만 논문 어디에도 공개하지 않았습니다. 데이터셋에 따라 효과가 0.01 meV 수준부터 7배까지 차이 납니다.
  - (ii) 보고할 checkpoint와 recipe를 test 수치를 보면서 골랐습니다. 3BPA UFF-nb, water의 final_ema 행이 해당합니다.

### 데이터셋별 한 줄 판정 (정정 후)

| 데이터셋 | 정정된 판정 |
|---|---|
| **3BPA** | 프로토콜은 표준과 일치합니다. MD 6개 지표는 실제 SOTA(GRACE/TACE)와 비교하고 있습니다. 다만 대표 recipe인 UFF-nb는 test를 본 뒤 고른 것이고, 사전에 등록한 규칙으로 고른 ZBL-nb를 쓰면 600K E는 동급, 1200K E는 TACE보다 나쁩니다. **dihedral E/F는 패배**입니다. "전 지표 SOTA"라고 쓰면 안 됩니다. |
| **AcAc** | **잘못된 SOTA와 비교하고 있습니다.** MACE가 아니라 PACE(TMLR 2024; 0.73/4.2/3.8/16.9)가 4개 지표 모두 SOTA입니다. 정정 후에는 300K E만 승리(gauge refit 필요), 300K F는 동급, 600K E/F는 패배입니다. |
| **rMD17** | **SOTA 집합이 불완전합니다.** ACE(g)만이 아니라 GotenNet, FreeCG, PACE를 넣어야 합니다. 정정 후 force는 aspirin, azobenzene, paracetamol 3개만 승리합니다. 에너지는 gauge refit 없이는 10개 분자 모두 ACE(g)에 밀립니다. 일괄 "rMD17 SOTA"는 불가합니다. |
| **water** | 표준 split이 없어서 **문헌 표와 직접 비교하는 것이 성립하지 않습니다.** 헤드라인 수치(3.8/71.4, 0.50/23.5)도 validation으로 선택한 값이 아닙니다. 같은 split에서 다시 학습한 MACE 64-0과의 비교만 유효합니다(split 1 패배 또는 노이즈 범위, split 2 F 소폭 우세). |
| **Si GAP-18** | RSS에서는 올바른 SOTA(XPOT-ACE-3F 20.5/139)와 비교하고 있고 14.9/112로 앞섭니다. 하지만 1 seed이고, AMSGrad로 학습하면 21.5/145로 뒤집힙니다. compression에는 공개된 RMSE가 없습니다. MPNN baseline이 하나도 없습니다. |
| **LiPS** | 19k에서는 GemNet-T가 아니라 **FreeCG 1.2와 동률**입니다. MD 관측량이 없어서 "force MAE only"로만 비교할 수 있습니다. learning curve는 모든 크기에서 우세하지만, n=1000의 비교 기준은 CarNet 5.6이고 NequIP 인용값은 Nat Commun 버전으로 바꿔야 합니다. |
| **QM9** | **잘못된 SOTA와 비교하고 있습니다.** 실제 SOTA는 MACE 4.1이 아니라 GotenNet-L 3.37이고, TensorNet 최종값은 3.9입니다. 우리 5.51(raw 5.96)은 약 2.1 meV 뒤집니다. 이 데이터셋은 보조 지표로만 써야 합니다. |

---

## 2. 데이터셋별 분석

### 2.1 3BPA

**(a) 적절성: 조건부 적절**
- 2024–2026년에도 표준 벤치마크입니다. GRACE (PRX 2024), ICTP (NeurIPS 2024), TACE, SpinGTP, Bayesian RACE, BOBOP가 모두 같은 과제(train_300K → 300/600/1200K + dihedral)를 보고합니다.
- 로컬 파일 검증 결과:
  - BOTNet-datasets와 md5가 동일합니다.
  - frame 수는 500 / 1669 / 2138 / 2139 / 7047입니다. dihedral은 β=120/150/180이 각각 2347/2350/2350입니다.
  - test_300K frame 1376은 train frame 200과 에너지가 같지만, 기하 구조가 3.6 Å 다르므로 우연의 일치입니다. 누수가 아닙니다.
- 한계:
  - 분자 하나, 기능함수 하나(ωB97X/6-31G(d))뿐입니다.
  - 분자 지름이 약 9 Å라서 **locality 주장의 근거가 될 수 없습니다.**
  - 300K E는 포화되어 있습니다(1.56 meV ≈ 0.06 meV/atom).
  - test frame들이 MD로 서로 상관되어 있습니다.
  - OOD validation set이 없어서 모두가 test에 맞춰 튜닝하는 구조입니다.

**(b) 표준 프로토콜과 우리 프로토콜**

일치하는 부분:
- 450/50 split. MACE, Allegro, ICTP와 같고, TACE의 10% validation과도 같습니다.
- test 크기 1669/2138/2139/7047은 TACE v3 Methods와 같습니다.
- 지표: RMSE, 분자당 total meV, meV/Å. `energy_rmse_total` = per-atom 값 × 27로 확인했습니다.
- best-val EMA checkpoint 사용. gauge refit 효과는 0.01 meV 이하입니다.

차이와 그 영향:
1. **recipe를 test를 보고 선택했습니다.**
   - 약 38라운드 동안 test_600K/1200K/dih 수치로 승격 여부를 판단했습니다("promote if 1200K F −10%").
   - 사전 등록 규칙("1200K F ≤ +10% of base, lowest dihedral E")이 고른 것은 **ZBL-nb c1.25(r33)**입니다. **UFF-nb w0.3(r37)은 사후에 "best temperature metrics"를 이유로 선택**한 것입니다.
   - 예상 winner's-curse bias는 seed std의 1–2배입니다. 1200K 기준으로 E 약 0.4–0.8 meV, F 약 0.7–1.4 meV/Å입니다.
2. **"w0.3"은 가중치가 아닙니다.** `pair_repulsion_switch_width = 0.3 Å`(sigmoid 폭)이고 scale은 1.0, switch는 1.25×(R_i+R_j)입니다. 논문에서 "weight 0.3"으로 쓰면 오기입니다.
3. **UFF/ZBL pair repulsion은 외부 경험적 prior입니다.** 규칙상 허용되지만, NN-xTB를 같은 이유로 제외하고 있으므로 반드시 공개하고 recipe A(prior 없음)를 함께 보여야 합니다.
4. **recipe를 열별로 섞으면 안 됩니다.** ZBL은 dih E에서, UFF는 T-set에서 가장 좋습니다. 열마다 가장 좋은 recipe를 골라 한 행을 만드는 것은 cherry-picking입니다.
5. **seed와 split.** 3 seed이고, split_seed는 42로 고정했습니다(train_3bpa.py:332). 따라서 std에는 초기화 분산만 들어 있습니다. 반면 TACE는 seed 1개(std 없음), GRACE와 ICTP는 5회입니다.
6. **dihedral set 크기.** 7047개로 일치한다고 확인된 것은 TACE뿐입니다. MACE, GRACE, SpinGTP는 개수를 밝히지 않았으므로 "reference가 set 크기를 명시하지 않음"으로 표기해야 합니다.

**(c) 정정된 SOTA 비교표** (RMSE, E는 분자당 total meV, F는 meV/Å)

| 지표 | 공정 SOTA (모델, 값, 출처) | UFF-nb (사후 선택, 3 seed) | ZBL-nb (사전 등록, 3 seed) | 판정 |
|---|---|---|---|---|
| 300K E | GRACE/ACE(g) 1.7(0.1), 5 fits ([2311.16326 Tab. II](https://arxiv.org/html/2311.16326)) | 1.56±0.06 | 1.62 | **근소 우세/동급.** Welch t≈2.5, seed 44는 1.63. 헤드라인 주장 금지 |
| 300K F | GRACE 7.6(0.2) | 6.31±0.11 | 6.53 | **승** |
| 600K E | GRACE 7.3(0.8) | 6.05±0.27 | 7.44 | UFF **승**, ZBL **동급** |
| 600K F | GRACE 19.2(0.4) | 15.92±0.34 | 16.89 | **승** |
| 1200K E | TACE spe-cart o3 23.42, 1 seed ([2509.14961v3 Tab. 1](https://arxiv.org/pdf/2509.14961v3)). 다중 seed 최고: GRACE 25.8(1.6) | 22.20±0.42 | 24.68 | UFF는 단일 seed TACE 대비 근소 우세(bias 범위 안), ZBL은 TACE에 **패**, GRACE에는 승 |
| 1200K F | TACE o3 54.26, 1 seed. 다중 seed 최고: SpinGTP 61.4(3.2) ([2607.01408 Tab. 1](https://arxiv.org/pdf/2607.01408)) | 45.16±0.68 | 50.79 | **승** |
| Dihedral E | GRACE 7.4(0.8). 이어서 PACE 7.6(0.4), TACE o2 7.6, MACE 7.8(0.6) | 16.67±0.72 | 10.44 | **패** |
| Dihedral F | GRACE 15.3(0.8) = SpinGTP 15.3(0.7). 이어서 PACE 16.0(0.5), MACE 16.5(1.7) | 19.09±0.29 | 17.42 | **패** (ZBL은 ICTP_full 17.52보다만 근소하게 앞섬) |

참고로 recipe A(prior 없음)는 1.65/6.46, 6.57/16.43, 25.31/47.54, dih 21.22/21.21입니다.

비교표에 넣어야 할 공정 baseline:
- **PACE** ([2411.04219 Tab. 2](https://arxiv.org/pdf/2411.04219), 3 seed): 2.4(0.1)/9.1(0.1), 7.9(0.2)/21.4(0.3), 29.6(0.4)/60.7(2.0), dih 7.6(0.4)/16.0(0.5)
- **ICTP_full** ([2405.14253 Tab. 2](https://arxiv.org/pdf/2405.14253), 5 runs, 450/50): 2.70/9.45, 10.74/22.99, 29.80/62.82, dih 9.82/17.52
- **SpinGTP** (MACE 기반, float32, 2000+250 epoch): 2.9(0.1)/9.0(0.3), 9.6(0.3)/21.0(0.5), 29.0(1.1)/61.4(3.2), dih 8.9(2.5)/15.3(0.7)
- **TopNets** ([2406.03164 Tab. 6](https://arxiv.org/pdf/2406.03164)): 2.5/8.9, 9.5/21.8, 29.2/62.3. dihedral 없음.
- **MACE-Gaunt**: 2.9/9.2, 10.6/22.2, 30.4/63.1, dih 9.9/17.7
- 그 밖에 Cartesian-3j, NEMP, BOBOP, Bayesian RACE(3.4/12.1, 11.7/31.8, 37.5/115.3)는 모두 우리보다 약합니다.

**[비교불가] 블록**

| 모델 | 비교불가 사유 | 수치 |
|---|---|---|
| MMEA ([2511.06696 Tab. 3](https://arxiv.org/pdf/2511.06696)) | MACE-OFF fine-tune | 2.7/7.5, 6.5/15.4, 17.1/39.7, dih 5.6/10.6 |
| ELoRA | MACE-OFF fine-tune | 3.0/7.5, 6.5/15.5, 17.6/42.0, dih 5.9/11.4 |
| MACE-OFF full fine-tune | 사전학습 | 3.3/7.8, 7.3/16.6, 20.3/48.7, dih 7.3/12.3 |
| "Beyond Adam" ([2512.05489](https://arxiv.org/html/2512.05489)) | AdamW로 MACE-OFF fine-tune. E는 meV/atom 단위 | F 8.1/15.5/38.1, dih 11.3 |
| FTA ([2511.08782](https://arxiv.org/pdf/2511.08782)) | DFT field 입력과 electron-density loss 사용. 단위 표기도 자체 모순 | — |
| NN-xTB | GFN2-xTB baseline 사용. PDF는 403으로 열리지 않음 | 1200K F 35.4 **[미검증, 검색 snippet]** |

**(d) 반드시 고칠 것 / 논문에 명시할 것**
- 대표 recipe를 하나로 고정합니다. 두 가지 선택지가 있습니다.
  - 사전 등록 규칙을 따른다면 ZBL-nb입니다.
  - UFF-nb를 쓴다면 "test_600K/1200K/dih 수치를 보고 선택했다"고 명시해야 합니다.
  - 어느 쪽이든 다른 recipe는 ablation으로만 보여줍니다.
- "SOTA on all 3BPA tasks" 또는 "below MACE on all columns"라는 표현은 금지합니다. dihedral이 한계라는 점을 명시합니다.
- TACE 비교에는 "single seed"라는 주석을 붙입니다.
- dihedral은 β별 값과 MAE를 함께 보고합니다. 근거는 다음과 같습니다(s42, `eval_dihbeta_gauge.json`).
  - β=120: RMSE 14.20, MAE 6.43, F 16.14
  - β=150: RMSE 15.94, MAE 5.56, F 17.56
  - β=180: RMSE 21.55, MAE 6.66, F 23.70
  - 전체: RMSE 17.51, MAE 6.22. RMSE가 MAE의 약 2.8배이므로 꼬리가 무거운 분포입니다.
  - dihedral 에너지는 training 최소값보다 최대 0.51 eV 낮습니다.
- 개발 과정에서 test를 보았다는 사실을 공개합니다. 오염되지 않은 확인용으로는 recipe를 동결한 뒤 AcAc, rMD17, MD22를 각 1회만 돌리는 방법을 씁니다.
  - train_mixedT는 이 용도로 적합하지 않습니다. 같은 test 파일을 쓰고, 500개 중 31개 에너지가 train_300K와 동일하기 때문입니다.
- 학습 예산 차이를 표로 정리합니다. 우리 1500 epoch, MACE AMSGrad, SpinGTP 2250 epoch, Allegro/NequIP 최대 7일 또는 100k epoch, GRACE L-BFGS-B. 손실 가중도 다릅니다: 우리 F:E = 10, MACE 1000:1, SpinGTP stage 2는 E 25.

---

### 2.2 AcAc

**(a) 적절성: 조건부 적절 (보조 벤치마크)**
- 15원자 분자이고 training은 500 frame입니다. 300K로 학습해 600K, dihedral, H-transfer로 외삽하므로 논문 범위와는 잘 맞습니다.
- 다만 사용하는 논문이 적습니다. BOTNet, MACE, ICTP, PACE 정도이고 2025년 이후는 fine-tune 논문뿐입니다.
- 300K E는 포화 상태입니다(약 0.05 meV/atom). 600K E는 소수 outlier가 지배합니다.
- H-transfer와 dihedral은 모든 reference가 그림으로만 보고합니다(min을 0으로 정렬). 숫자로 된 baseline이 없습니다.

**(b) 프로토콜 차이**
- 일치하는 부분: 450/50 split, test 650/650, 분자당 total meV RMSE, float64, 5 Å, batch 5, lr 0.01, EMA 0.99. E0는 dataset의 isolated_atoms.xyz 값입니다.
- **gauge refit 의존도가 큽니다.**
  - SOAP 기준 300K E는 refit 전 1.45/1.40/0.87(평균 1.24±0.32)에서 refit 후 0.58/0.67/0.59(0.61±0.05)로 바뀝니다.
  - 반면 AMSGrad v9b1500은 refit 없이도 0.66/0.70/0.74(평균 0.70)입니다. 따라서 "에너지 이득 전부가 refit에서 나온다"는 말은 SOAP 500 epoch 설정에만 해당합니다.
- **학습이 수렴하지 않았습니다.** best checkpoint가 epoch 495/495/500(전체 500)에 있습니다. 비교 대상은 ICTP 2000 epoch, PACE 5000 epoch입니다.
- **파라미터 수**: 우리 6,401,425개, MACE 2,803,984개, ICTP 2.65–2.77M개. 우리가 약 2.3배 큽니다.
- **variant 선택**: 7개 variant 모두 test로 평가했습니다. v9b의 σ는 docs에 "v9a 결과를 본 뒤 정한 값"이라고 적혀 있습니다. validation 기준으로 고르면 SOAP가 선택되므로 SOAP를 대표로 쓰는 것은 정당화할 수 있습니다.
- split_seed가 모든 seed에서 동일합니다. ICTP는 seed마다 split을 다시 뽑습니다.

**(c) 정정된 SOTA 비교표** (RMSE; E는 meV/분자, F는 meV/Å; 우리 값은 SOAP+gauge, 3 seed, ddof=1)

| 지표 | 공정 SOTA | 우리 | 판정 |
|---|---|---|---|
| 300K E | **PACE 0.73(0.02)** ([2411.04219 Tab. 3](https://arxiv.org/pdf/2411.04219), 3 seed, 450/50). 이어서 ICTP_full 0.75±0.04, MACE^a 0.75±0.05 | 0.61±0.05 (raw 1.24±0.32; v9b1500 raw 0.70) | refit 기준 **승**, SOAP raw는 **패** |
| 300K F | **PACE 4.2(0.1)**. 이어서 MACE^a 5.00±0.17 | 4.25±0.58 | **동급** |
| 600K E | **PACE 3.8(0.6)**. 이어서 ICTP_sym 4.43±0.34, MACE 4.6(0.3) | 6.69±0.27 (v9b1500: 4.71±0.50) | **패** (BOTNet 6.2, NequIP 6.04에도 뒤짐) |
| 600K F | **PACE 16.9(0.3)**. 이어서 MACE 22.4(0.9), ICTP_sym 22.90 | 19.75±0.82 | **패** (MACE와 ICTP에는 승) |
| H-transfer / dihedral | 숫자 baseline 없음 (그림만 있음) | 미보고. 내부 기록상 SOAP s42 dihedral 59.7 meV, 정렬 방식은 미확인 | 비교 불가 |

- PACE 관련 주의: Table 3 캡션과 본문은 RMSE라고 하지만 Table 4 캡션은 "Mean absolute errors"라고 되어 있어 논문 내부가 일치하지 않습니다. 인용할 때 주석을 답니다. PACE는 5000 epoch, E:F = 15:1000입니다.
- ICTP의 MACE^a는 ICTP 학습 설정으로 다시 돌린 MACE입니다. run 수는 명시되어 있지 않습니다.
- **[비교불가]** MMEA (MACE-OFF fine-tune): 0.7/4.2/3.2/13.0. ELoRA: 0.8/4.5/3.9/13.6.

**(d) 고칠 것 / 명시할 것**
- MACE 대신 PACE와 ICTP를 기준으로 margin을 계산합니다(0.73 / 4.2 / 3.8 / 16.9).
- `paper_outline_2026-10-30.md`의 "improves … AcAc"는 과장이므로 수정합니다.
- refit 전과 후 에너지를 모두 보고합니다.
- variant 선택 규칙(validation loss로 SOAP 선택)을 명시합니다. 600K 수치로 variant를 고르는 것은 금지합니다.
- dihedral과 H-transfer를 빼면 선택적 보고가 됩니다. 3BPA에서 SOAP가 dihedral을 악화시킨다는 기록이 이미 있기 때문입니다. reference와 같은 방식(min 정렬, DFT 대비 그림, barrier 오차)으로 넣습니다.
- 50-config 과제도 고려합니다. ICTP Tab. A5의 MACE 값은 4.22/28.38입니다.
- BOTNet은 저널판을 인용합니다(Nature Mach. Intell., doi 10.1038/s42256-024-00956-x).

---

### 2.3 rMD17

**(a) 적절성: 조건부 적절 (in-distribution 정확도 전용)**
- 여전히 표준입니다. GotenNet 2025, ICTP 2024, PACE 2024 등이 사용합니다.
- 하지만 포화되어 있습니다. MACE §5.3.1과 ICTP §5 모두 "comparison less meaningful"이라고 씁니다.
- 500 K 궤적 하나에서 뽑은 상관된 frame이므로 보간만 시험합니다.
- 소량 데이터 주장에는 **Ntrain=50** 설정이 더 적합합니다. ACE(g) 50-config 예시: aspirin 11.7/38.2.

**(b) 프로토콜 차이**
- **test set 정의.**
  - 우리 헤드라인은 공식 index_test_01(1000개)이고, MACE, PACE, 기존 RACE와 같은 방식입니다.
  - ACE(g), GotenNet, NequIP, Allegro, ICTP, TACE-v1은 나머지 약 99k frame에서 평가합니다.
  - 우리 99k 평가는 **seed 42만** 있습니다.
- **gauge refit 의존도가 극단적입니다.** raw와 gauge 3-seed 평균(meV)은 다음과 같습니다.

  | 분자 | raw | gauge |
  |---|---|---|
  | aspirin | 1.95 | 1.60 |
  | azobenzene | 1.86 | 0.50 |
  | benzene | 0.26 | 0.04 |
  | ethanol | 0.54 | 0.31 |
  | malonaldehyde | 0.61 | 0.49 |
  | naphthalene | 0.77 | 0.18 |
  | paracetamol | 1.10 | 0.82 |
  | salicylic | 0.86 | 0.50 |
  | toluene | 0.57 | 0.20 |
  | uracil | 0.68 | 0.29 |

  raw 기준으로는 10개 분자 모두 ACE(g)보다 나쁩니다. refit은 training split만 사용하므로 누수는 아니지만, 500 epoch에서 수렴하지 않은 offset을 가리고 있습니다.
- **ACE(g)의 training 규모**: 1000개를 "randomly selected for training"했고 validation은 언급하지 않았습니다. 우리는 950개입니다.
- **정밀도**: MACE, Allegro, ICTP는 float32, 우리는 float64입니다.
- **cutoff**: Allegro는 7 Å(naphthalene은 9 Å), 우리는 5 Å입니다.
- **반올림**: baseline 값은 소수 첫째 자리까지만 있으므로 같은 정밀도로 반올림한 뒤 비교해야 합니다.
- 분자 간 config 차이는 avg_n_neighbors뿐입니다. 분자별 튜닝은 없었습니다. eval_test_every로 test가 노출되었다는 점은 공개합니다.

**(c) 정정된 SOTA 비교표** (MAE; E는 meV/분자, F는 meV/Å)

kcal/mol로 보고한 값은 ×43.364로 환산했습니다(GotenNet, FreeCG). 우리 값은 test-1000, 3 seed, gauge 적용이고 force는 refit의 영향을 받지 않습니다.

| 분자 | E 공정 SOTA | 우리 E (raw) | E 판정 | F 공정 SOTA | 우리 F | F 판정 |
|---|---|---|---|---|---|---|
| aspirin | GotenNet 1.55 | 1.60 (1.95) | 근소 패 | FreeCG 5.26 | 4.79±0.08 | **승** |
| azobenzene | ACE(g) = PACE = LiTEN 0.5 | 0.50 (1.86) | 동급 (refit 의존) | GotenNet 2.09 | 1.92±0.06 | **승** |
| benzene | ACE(g) 0.01 | 0.04 (0.26) | 패 | ACE(g) 0.1 | 0.37±0.01 | 패 |
| ethanol | ACE(g) 0.2 | 0.31 (0.54) | 패 | ACE(g) 1.4 (PACE 1.8) | 2.07±0.10 | 패 |
| malonaldehyde | ACE(g) = PACE 0.5 | 0.49 (0.61) | 동급 | ACE(g) 3.1 | 3.11±0.03 | 동급 |
| naphthalene | GotenNet 0.17 | 0.18 (0.77) | 동급 | ACE(g) = Allegro = PACE 0.9 | 0.99±0.03 | 근소 패 |
| paracetamol | ACE(g) = PACE 0.9 | 0.82 (1.10). 99k s42: 0.877 | 동급 (반올림하면 0.9) | FreeCG 3.64 | 3.33±0.08 | **승** |
| salicylic | ACE(g) = PACE 0.5 | 0.50 (0.86) | 동급 | ACE(g) 2.7 | 2.77±0.11 | 동급/근소 패 |
| toluene | GotenNet 0.19 | 0.20 (0.57) | 동급 | FreeCG 1.04 | 1.12±0.04 | 근소 패 |
| uracil | ACE(g) 0.2 | 0.29 (0.68) | 패 | ACE(g) 1.7 (PACE 2.0) | 2.04±0.07 | 패 |

출처:
- ACE(g): [2311.16326 Tab. I](https://arxiv.org/abs/2311.16326)
- GotenNet: [ICLR 2025 Tab. 4](https://proceedings.iclr.cc/paper_files/paper/2025/file/64d4ff4fff788cdffe236f9ce8b09400-Paper-Conference.pdf), 5 split 평균, 12층, 3000 epoch
- FreeCG: [2407.02263v4 Tab. 2](https://arxiv.org/pdf/2407.02263v4). preprint이고 test set 정의를 명시하지 않았습니다.
- PACE: [2411.04219v1 Tab. 1](https://arxiv.org/pdf/2411.04219v1). MACE와 같은 split과 seed, test 1000개라서 **우리 test-1000 수치와 가장 직접 비교할 수 있습니다.** rMD17의 seed 수는 명시되어 있지 않습니다.

**신뢰도가 낮은 항목**
- MACE-MARA ([2602.02671 Tab. 1](https://arxiv.org/pdf/2602.02671)): ethanol E 0.11, malonaldehyde E 0.39, benzene F 0.28. baseline 열에 전사 오류가 있고 출처 설명이 서로 모순되며 test 크기와 seed가 없습니다. 이름은 언급하되 근거를 들어 배제합니다.
- MACE+PAN+PGS: 표에 라벨이 잘못 붙어 있습니다.

**[비교불가]**
- NN-xTB (xTB prior): force 기준 paracetamol 2.98, salicylic 2.16, ethanol 1.50, malonaldehyde 3.00, naphthalene 0.82.
- MD-ET: CCSD(T) 데이터와 QCML 사전학습 사용.

**(d) 고칠 것 / 명시할 것**
- GotenNet, FreeCG, PACE 열을 추가합니다. "rMD17 SOTA"라는 일괄 표현은 금지합니다. 정당한 표현은 "큰 분자(aspirin, azobenzene, paracetamol) force에서 최고, 나머지는 동급 또는 열세"입니다.
- 99k 평가를 3 seed 모두로 확장합니다. test-1000 열(MACE, PACE와 비교)과 99k 열(ACE(g), GotenNet, NequIP, Allegro와 비교)을 분리합니다.
- 에너지는 raw와 gauge를 함께 보고합니다.
- 가능하면 여러 공식 split(01–05)을 쓰거나 seed마다 split을 바꿉니다.
- Ntrain=50 과제를 추가합니다.
- 이 벤치마크를 외삽이나 locality 근거로 쓰지 않습니다.

---

### 2.4 Water (Cheng revPBE0-D3, 1593 frame)

**(a) 적절성: 조건부 적절 (보조 벤치마크, 같은 split에서 다시 학습한 비교만 유효)**
- 응집상 local 정확도를 시험하기에는 좋은 데이터입니다. 하지만 **표준 split이 없습니다.** 논문마다 쓰는 방식이 다릅니다.
  - CACE: validation 오차, training 90%
  - REANN/Cheng 원저: random 80/20
  - PET: 1303/100/190
  - CAMP와 TACE: 90:5:5 (TACE는 1432/79/81)
  - MTS-Allegro: 1000/100/493
  - MACE와 NEMP: split 미명시
- split에 따른 분산이 모델 간 격차보다 큽니다. 같은 MACE 64-0 recipe로도 F가 23.2부터 64.8까지 나오고, 문헌에서도 36.2와 46.8이 있습니다.
- 외삽 벤치마크가 아닙니다.

**(b) 프로토콜 문제**
- **헤드라인 수치는 validation으로 선택한 값이 아닙니다.**

  | split | 헤드라인 (출처) | validation으로 선택한 값 |
  |---|---|---|
  | split 1 | "E 3.8 / F 71.4": w7 run의 final_ema(3.78/71.4)이거나, w5의 E(3.85, 이 run의 F는 82.7)와 w6의 F(71.4)를 조합한 것 | w7 best-val 4.29 (gauge 4.23) / 69.5. valid F 최저 run(w6) 기준이면 4.53 / 71.4 |
  | split 2 | "E 0.50 / F 23.5": final_ema | best-val 0.70 (gauge 0.59) / 21.9 |

  두 경우 모두 test를 보고 checkpoint나 run을 고른 것과 같습니다.
- **split 2의 test는 100% PIMD입니다**(frame 1393–1592). 공개된 30–37 meV/Å 값 옆에 놓으면 유리하게 왜곡된 비교가 됩니다. TACE 30.31보다 약 22% 낮게 보이는 효과가 납니다.
- **split 1의 test**는 classical 103개, PIMD 56개로 구성됩니다. frame 383 하나가 F 제곱오차의 약 71%를 차지합니다. bootstrap 95% CI는 [40, 128]입니다.
- **MACE 재학습 세부사항**
  - mace-torch 0.3.17, 문서 recipe 사용. 단 stage 2 가중치는 1000/100으로, 논문의 1000/10과 다릅니다.
  - split 1에서는 patience 30 때문에 stage 2가 약 epoch 284에 시작되었습니다. 따라서 "SWA from 400"은 사실이 아닙니다.
  - split 1 결과: stage 2 1.5 meV/atom(=4.5 meV/H₂O)/64.8, stage 1 2.0/60.1.
  - split 2 결과: stage 2 0.3 meV/atom(0.75–1.05 meV/H₂O)/23.2. 로그가 0.1 단위로 반올림되어 있어 E 비교를 확정할 수 없습니다.
  - E0 설정이 average인지 least-squares인지는 두 검증자의 판단이 엇갈립니다. **[확인 필요]**
  - split 2의 validation 집합은 RACE(seed 42)와 MACE(seed 123)가 서로 다릅니다.
- **MACE 에너지 격차의 원인**: split 1 MACE의 validation F는 39.4로 공개값과 비슷하므로 force 격차는 test draw 탓입니다. 그러나 에너지는 train/val에서도 공개값의 약 3배이므로, test 난이도가 아니라 recipe나 재현성 차이일 가능성이 큽니다.

**(c) 비교표**

같은 split에서 다시 학습한 비교만 유효합니다(RMSE, E meV/H₂O, F meV/Å, 모두 1 seed).

| split | RACE (validation 선택) | MACE 64-0 재학습 (stage 2) | 판정 |
|---|---|---|---|
| split 1 | 4.29 / 69.5 | 4.5 / 64.8 (stage 1: 6.0 / 60.1) | F는 패이지만 CI [40, 128] 안이므로 유의하지 않음 |
| split 2 | 0.70 (gauge 0.59) / 21.9 | 0.75–1.05 / 23.2 | F 근소 우세, E는 판정 불가 |

문헌값은 참고용입니다. split이 서로 달라 직접 비교할 수 없습니다.

| 모델 | E (meV/H₂O) | F RMSE | 비고 |
|---|---|---|---|
| TACE 64-2 ([2509.14961v3 Tab. 2](https://arxiv.org/pdf/2509.14961v3)) | 2.16 | **30.31** | 1 seed, 1432/79/81 |
| CarNet-3L ([2510.04015](https://arxiv.org/html/2510.04015)) | 1.62 | 31 | 90:5:5, seed 미명시 |
| TACE 64-0 | 1.74 | 33.16 | |
| CAMP ([2411.12096](https://arxiv.org/html/2411.12096), npj 2025) | 1.77 | 34(0.1) | 5 models |
| MACE 192-2 ([2305.14247 Tab. V](https://arxiv.org/pdf/2305.14247v2)) | 1.9 | 36.2 | MAE 1.2/18.5, split 미명시 |
| NEMP 출판본 ([Chem. Sci. 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC12766320/)) | 2.4(0.1) | 36.8(0.5) | 4 runs |
| MLANet | 1.41 | 60 | 과적합 자인 |
| PET ([2406.17747 SI Tab. I](https://arxiv.org/pdf/2406.17747)) | — | RMSE 60 / MAE 17 | test인지 validation인지 본문과 캡션이 모호 |
| PET ([2305.19302](https://arxiv.org/html/2305.19302v3)) | — | F MAE 14.4 | 본문 값 |
| MTS-Allegro ([2310.13756](https://arxiv.org/abs/2310.13756)) | — | 76.9 (Allegro 77.4) | test 약 493개로 큰 random test. 우리 split 1의 69–71이 이상치가 아님을 뒷받침 |

사전학습이나 추가 입력을 쓴 모델은 이 벤치마크에서 발견되지 않았습니다.

**(d) 고칠 것 / 명시할 것**
- 보고 규칙을 고정합니다(best-val). 이에 따라 split 1은 4.29/69.5, split 2는 0.70(또는 0.59)/21.9로 정정합니다.
- 문헌 표와 head-to-head로 비교하는 것을 금지하고, 참고용 블록으로 분리합니다.
- split 2를 공개값 옆에 두는 것을 금지합니다.
- MACE 192-2, CAMP, TACE 64-2를 우리 split에서 다시 학습합니다.
- 여러 seed로 random 90:5:5 또는 80/20(Cheng/REANN 원 프로토콜)을 돌립니다. 스냅샷과 quench 쌍은 같은 group으로 묶습니다.
- MACE 에너지는 저장된 예측에서 다시 계산합니다. 로그는 반올림되어 있습니다.
- 어떤 stage를 보고하는지 명시합니다.
- dataset 개수를 명시합니다. ColabFit에는 1588개로 올라와 있고 우리 로컬 파일은 1593개입니다.

---

### 2.5 Si GAP-18

**(a) 적절성: 조건부 적절 (niche)**
- RSS(100개, 1–64원자, GAP-18로 0–32 GPa에서 relax)는 진짜 OOD입니다. compression MD(100 × 64원자)도 마찬가지입니다.
- 라벨은 GAP-18과 같은 PW91/CASTEP 설정입니다.
- 원저자 Morrow & Deringer는 "not intended as a competitive benchmark"라고 썼습니다.
- 이 test들에서 MPNN(MACE, NequIP, Allegro, GRACE) 결과는 하나도 없습니다. 따라서 여기서 SOTA는 "ACE/GAP 대비"라는 뜻입니다.

**(b) 프로토콜 차이**
- 우리는 2474 config 중 5%를 validation으로 빼서 약 2350개로 학습했습니다. du Toit은 GAP-18 전체로 학습하고 별도 testing DB로 HPO를 했습니다. 이 차이는 우리에게 약간 불리합니다.
- 지표 정의(구조별 E/N RMSE, 성분별 F)는 du Toit의 GAP-18 34.9/177을 정확히 재현합니다.
- **결과가 optimizer에 따라 뒤집힙니다.** SOAP는 RSS 14.85/112.0인데 AMSGrad는 21.46/145.2(gauge 21.16)입니다. 1 seed에서 SOTA 판정이 갈립니다.
- **compression 에너지가 수렴하지 않았고 checkpoint에 민감합니다.**
  - EMA 기준: epoch 100에서 17.8, epoch 200에서 7.8, epoch 300(final)에서 5.92 meV/atom.
  - best-val(epoch 295): 4.20, gauge 적용 시 3.83.
- σ_n = 0.224는 3BPA의 σ = 0.5 Å를 그대로 가져온 값입니다. Si test에 맞춰 튜닝한 것이 아닙니다(audit의 지적은 오류).
- GAP-18의 RSS 평균 부호 오차는 −24.4 meV/atom으로, MSE의 49%를 차지합니다. 하지만 우리 모델도 같은 부호의 offset이 있습니다(필요 shift +9.26 meV/atom). 따라서 GAP-18이 불공정하게 불리한 것은 아닙니다. 그래도 주 비교 기준은 XPOT-ACE-3F로 둡니다.

**(c) 정정된 SOTA 비교표** (우리 값은 1 seed, SOAP, best-val)

| 지표 | 공정 SOTA | 우리 | 판정 |
|---|---|---|---|
| RSS E RMSE (meV/atom) | XPOT-ACE-3F 20.5 ([du Toit JCTC 2024 Tab. 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC11603601/), 단일 fit) | 14.85 (raw 15.04) | 승. 1 seed이고 optimizer에 따라 뒤집힘 |
| RSS F RMSE (meV/Å) | XPOT-ACE-3F 139 | 112.0 | 승. 같은 단서 |
| RSS MAE (E/F) | Darby linear ACE 약 20 / 약 0.126 eV/Å ([2601.15072v2 Fig. S6](https://arxiv.org/pdf/2601.15072v2), 그림 판독, 근사값). GAP-18 28.81 / 129.7 (우리 재계산) | 11.39 / 84.4 | 승 |
| compression E/F RMSE | **공개 RMSE 없음.** 우리가 계산한 GAP-18 15.72 / 125.1 | 3.83 (raw 4.20, final 5.92) / 69.8 | GAP-18보다 승 |
| compression MAE | Darby 약 4.35 / 약 0.079 eV/Å (그림, best-of-σ). GAP-18 10.06 / 87.3 | 2.94 / 48.8 (final EMA E MAE는 5.33) | best-val 기준 승, final 기준 E 패 |
| MQ-MD, Si-GAP-18 test | 3F 4.1 / 97, GAP-18 1.6 / 83 | 미평가 | 데이터 없음 |

Darby et al.은 합성 dimer(100 eV, 가중치 0.01)를 추가해 학습했습니다. 작은 비-DFT 추가이므로 주석을 답니다.

**(d) 고칠 것 / 명시할 것**
- compression을 du Toit의 "MQ-MD" 열에 넣지 않습니다. 둘은 서로 다른 set입니다.
- 로컬 `phonons_gaps_si.zip`에는 MQ-MD 구조가 없습니다(SOTA 보고서의 오류). George 2020 원 데이터를 별도로 확보해야 합니다.
- 3 seed 이상 돌리고, optimizer 선택 규칙을 명시합니다(3BPA에서 고정한 SOAP). AMSGrad 결과도 함께 보고합니다.
- 학습을 수렴할 때까지 돌리거나, best와 final을 함께 보고합니다.
- MACE나 NequIP를 같은 프로토콜로 학습해 baseline으로 넣습니다.
- Si-GAP-18 testing DB(Zenodo 1250555)를 확보합니다. 포함된 파일이 du Toit이 쓴 것과 같은지 확인해야 합니다.
- 물성(phonon, 탄성, vacancy, surface) 평가를 추가합니다.
- 우리 strat/extrap split에서 GAP-18 값을 보고하지 않습니다. GAP-18의 training 데이터이기 때문입니다.
- MAE 대 MAE로 비교합니다. "우리 RMSE가 그들 MAE보다 낮다"는 논법은 쓰지 않습니다.

---

### 2.6 LiPS

**(a) 적절성: 조건부 적절 (주기적 고체의 정확도, 데이터 효율, MD 안정성 시험용)**
- 50 ps, 520 K, 2 fs 간격의 AIMD 궤적 하나에서 나온 25,001 frame입니다.
- **19k random split은 사실상 중복 보간입니다.** test frame의 94.6%가 training frame과 1 step 이내이고, 최대 거리도 3 step입니다.
- 정보가 있는 것은 learning curve의 n=10과 n=100입니다. 중앙값 거리가 각각 751 step, 82 step입니다.
- Fu et al.의 핵심 주장은 "force MAE만으로는 부족하다"이고, 그래서 MD 관측량을 함께 봅니다.
- LiPS-25나 "LiPS 20"과 혼동하지 않도록 합니다. 다른 데이터셋입니다.

**(b) 프로토콜 차이**
- Fu split(MDsim, random_state=123, 19000/1000/5001)은 정확히 재현했습니다.
- learning curve:
  - NequIP: validation 100, test 미명시.
  - CAMP와 CarNet: 각자의 random split(valid 1000, test 5000).
  - 우리: validation 1000, test는 Fu의 5001 frame, 1 seed, subset draw 1회.
- **보고 규칙이 정해져 있지 않습니다.** run마다 best raw, best+gauge, final_ema 세 값이 있습니다. n=100 에너지를 예로 들면 best raw 0.475, gauge 0.168, final_ema 0.166입니다.
- 19k는 수렴하지 않았습니다. best checkpoint가 epoch 149(전체 150)이고, LR이 한 번도 감소하지 않았습니다.
- 파라미터 수: 우리 6,403,503개, CAMP 125k개 미만. 용량 효과일 가능성이 있습니다.
- MD 관측량이 하나도 없습니다.

**(c) 정정된 SOTA 비교표** (MAE; E meV/atom, F meV/Å; 우리 값은 1 seed)

| 지표 | 공정 SOTA | 우리 (best+gauge / best raw / final_ema) | 판정 |
|---|---|---|---|
| 19k F | **FreeCG 1.2** ([2407.02263v4 App. Fig. 9](https://arxiv.org/pdf/2407.02263v4), Fu split, conservative, 그림의 막대 라벨 값). GemNet-T 1.3 ([Fu Tab. 6](https://arxiv.org/pdf/2210.07237v2)) | 1.20 | **동급** |
| 19k E | 공개값 없음 | 0.020 / 0.078 / 0.053 | 비교 불가 |
| 19k MD (stability/RDF/D) | stability와 RDF는 포화(50 ps, 0.04). D MAE: E2GNN 0.21 (direct force), GemNet-T 0.24 | 없음 | **미수행** |
| n=10 E/F | CAMP 1.63(0.16) / 76.4(3.3) (npj 2025 Tab. 1, 5 models) | E 0.668 / 1.094 / 0.669, F 33.69 | **승** (run 완료, "unfinished" 아님) |
| n=100 E/F | NequIP = CAMP 0.44. F는 CAMP 21.6(0.6) | E 0.168 / 0.475 / 0.166, F 9.52 | F **승**. E는 best raw 기준 **패**, gauge나 final 기준 승. 규칙을 고정해야 함 |
| n=1000 E/F | **CarNet-3L 0.09 / 5.6** ([2510.04015v3 Tab. I](https://arxiv.org/pdf/2510.04015v3), 자체 split, seed 미명시) | E 0.052 / 0.075, F 3.23 | **승** (기존 주장한 약 2.3배가 아니라 약 1.7배) |
| n=2500 E/F | NequIP 0.08 / 4.7 ([Nat Commun 2022 Tab. 4](https://pmc.ncbi.nlm.nih.gov/articles/PMC9068614/)) | E 0.035 / 0.043, F 2.04 | **승** |

- CAMP의 19k 값 3.6과 D MAE 0.18은 Fu split이 아니라 CAMP 자체 split에서 나왔습니다.
- CAMP의 0.18은 본문의 D = 1.08 vs 1.35와 맞지 않습니다. 이 값으로 계산하면 오차는 약 0.27이어야 합니다. **[내부 미검증]**
- GemNet-dT(1.4)는 non-conservative 모델이므로 표시합니다.

**(d) 고칠 것 / 명시할 것**
- NequIP 인용을 Nat Commun 버전으로 교체합니다(F 97.8/25.8/7.7/4.7).
- CAMP 표의 NequIP 열은 NequIP 논문에서 복사한 것입니다. 독립된 재학습 결과로 표기하지 않습니다.
- FreeCG와 CarNet을 추가합니다.
- 19k는 "최고 공개값과 동급, force MAE only"로 표현합니다. 아니면 Fu의 MD 프로토콜(5 × 50 ps, 0.25 fs, 520 K NH)을 compute 노드에서 돌립니다.
- 3 seed 이상 돌리고, n=10과 n=100은 subset draw도 여러 번 합니다.
- 보고 규칙을 사전에 고정합니다.
- 용량을 맞춘 ablation(작은 RACEv2)을 추가합니다.
- Fu 논문의 fairchem PBC 버그 수정 후 개정판 Table 6을 직접 확인합니다(OpenReview 접근 불가). 영향은 MD 지표에 한정될 것으로 추정되지만 추론일 뿐입니다.

---

### 2.7 QM9 (U0)

**(a) 적절성: 조건부 적절 (보조 지표, 논문 주장과는 범위 밖)**
- 평형 구조와 에너지만 있는 property regression 벤치마크입니다. MACE eval 논문은 이를 "not directly related to force field fitting"이라고 표현합니다.
- 우리 test force MAE가 295 meV/Å인데, 이 데이터셋은 PES 품질을 측정하지 않으므로 이 값은 의미가 없습니다.
- training 110k는 small/medium data가 아닙니다. 벤치마크도 포화되어 있습니다.

**(b) 프로토콜**
- 일치하는 부분:
  - uncharacterized 3054개 제거 후 130,831개, 110k/10k/10,831 분할. 크기와 필터링이 Allegro, TensorNet, Wigner, GotenNet과 같습니다.
  - validation은 별도 파일 valid.xyz입니다.
  - U0 변환에 27.211386245988을 썼습니다.
- 다른 부분:
  - split 순열이 우리 seed 42입니다. GotenNet은 seed 1(TorchMD-Net make_splits), DimeNet 계열은 또 다른 partition을 씁니다.
  - 1 seed, 300 epoch로 수렴하지 않았습니다. 600 epoch run은 epoch 400에서 raw 5.64였고, epoch별 EMA 변동이 ±0.3–0.5 meV입니다.
  - gauge refit으로 5.96에서 5.51이 되었습니다(−8%).
  - Huber loss를 썼습니다. 비교 대상은 대부분 MSE입니다.
  - zero-force 변형(u0f, u0f4)은 추가 감독 신호를 쓰는 것이므로 별도 "-wF" 행으로 분리합니다.

**(c) 정정된 SOTA 비교표** (MAE, meV/분자)

| 순위 | 모델 | U0 | 출처와 비고 | 우리 5.51 (raw 5.96) 대비 |
|---|---|---|---|---|
| 1 | **GotenNet-L** | **3.37** | [ICLR 2025 Tab. 1/9](https://proceedings.iclr.cc/paper_files/paper/2025/file/64d4ff4fff788cdffe236f9ce8b09400-Paper-Conference.pdf). 단일 run. repo config 기준 110k/10k/rest, seed 1 | **패** (약 2.1 meV) |
| | GotenNet-B / S | 3.43 / 3.71 | 같은 출처 | 패 |
| | EST + GATA | 3.70 | [2505.23086 **v3** Tab. 11](https://arxiv.org/pdf/2505.23086). v1/v2에서는 3.94. GotenNet 블록을 사용 | 패 |
| | TorchMD-Net 2.0 TensorNet | 3.8(2) | [2402.17660](https://arxiv.org/pdf/2402.17660) 본문 | 패 |
| | TensorNet 3L | **3.9(1)** | [2306.06482v2 Tab. 1](https://arxiv.org/html/2306.06482), 4 splits | 패 |
| | MACE | 4.1 | 2305.14247 Tab. VI. -wF 사용 여부 불명 | 패 |
| | FreeCG | 4.11 | 2407.02263v4 Tab. 4 | 패 |
| | ViSNet | 4.23 | arXiv v3 = Nat Commun | 패 |
| | Geoformer / Allegro | 4.43 / 4.7(0.2) | | 패 |
| | beSEN (geometry-only) / SaVeNet-B (다른 partition) | 5.3 / 5.43 | | 패 |
| | ENINet | 5.52 | | 동급 |
| | eSEN / EST / SE3Set / PaiNN | 5.6 / 5.64 / 5.74 / 5.85 | | gauge 기준 승 |

- MGNN 4.1은 검색 snippet으로만 확인했습니다. **[미검증]**
- GotenNet을 독립 재현한 결과(2605.27662 Tab. 2, 4 seeds)는 약 0.0103±0.0054입니다. 단위가 표기되어 있지 않고 학습 예산도 불명이라 참고용입니다.

**[비교불가]**
- OrbNet-Equi 3.5 (xTB 특징 사용)
- beSEN surrogate 2.8 / density 입력 4.9 (DFT 밀도 사용)
- SliDe 4.28, DenoiseVAE 4.31, Frad 5.33, Transformer-M 9.37 등 (사전학습)
- NoisyNodes 7.3 (denoising 보조 목적함수)

**(d) 고칠 것 / 명시할 것**
- TensorNet 4.3을 3.9로 바꾸고, MACE 대신 GotenNet-L을 SOTA로 표기합니다.
- raw와 gauge를 함께 보고합니다.
- 3 split 또는 3 seed 이상, 수렴할 때까지 학습합니다. variant 선택은 validation으로만 합니다.
- "보조적 화학공간 점검"으로 위치를 정합니다. learning curve(1k–10k)가 논문 범위와 더 잘 맞습니다.

---

## 3. 우리가 인용하던 참조값 중 틀린 것

| # | 데이터셋 | 현재 인용 | 정정 |
|---|---|---|---|
| 1 | 3BPA | NequIP 3.3/10.8, 11.2/26.4, 38.5/76.2, **dih 23.2/23.1** | 두 구현을 섞은 행입니다. T-set 값은 nequip-code(Allegro 2204.05249, dihedral 없음)이고, dih는 mace-code 재구현(T-set 3.1/11.3, 11.3/27.3, 40.8/86.4) 값입니다. 하나로 통일하거나 각주를 답니다. |
| 2 | 3BPA | GRACE, TACE, BOBOP 표에서 가져온 baseline | 2차 전사 오류가 있습니다. GRACE 표: MACE 300K E std 0.5 (원본 0.2), BOTNet 1200K F 81.5 (원본 81.1), NequIP dih F 23.3 (원본 23.1). BOBOP: MACE 1200K F 62.8 (원본 62.0). TACE의 CACE 1200K F 139.6 (ICTP/BOBOP는 113.8). 모두 원본에서 가져옵니다. |
| 3 | 3BPA | "UFF w0.3 = 가중치 0.3" | switch width 0.3 Å입니다(scale 1.0). |
| 4 | 3BPA | "ICTP가 MACE를 재현하지 못함" | 사실이지만, ICTP의 재학습은 ICTP식 학습 설정을 쓴 것입니다. 표현을 신중하게 합니다. |
| 5 | 3BPA | TACE 제목 | v3에서 "Spectral/Spatial Tensor Atomic Cluster Expansion with Universal Embeddings in Cartesian Space"로 바뀌었습니다. 1 seed입니다. |
| 6 | AcAc | MACE 0.9/5.1/4.6/22.4 = SOTA | 값은 정확하지만 SOTA가 아닙니다. 실제 SOTA는 PACE 0.73/4.2/3.8/16.9입니다. ICTP도 MACE보다 우수합니다. |
| 7 | AcAc | wizwand의 "MACE re-run 0.75/5.0" | 출처는 ICTP Tab. 3의 MACE^a입니다. 논문을 직접 인용하고, run 수는 미명시로 적습니다. |
| 8 | AcAc | NequIP std | MACE 표(0.04/0.38/1.26/3.29)와 BOTNet 표(0.05/0.46/1.54/4.03)가 다릅니다. ddof 차이로 추정됩니다. 출처를 하나로 고정합니다. |
| 9 | rMD17 | NequIP 열 (GRACE 표 경유) | NequIP arXiv v3 Table II 값입니다. Nat Commun 2022 Tab. 2 값(2.3/8.2, 0.7/2.9, 0.04/0.3, 0.4/2.8, 0.8/5.1, 0.2/1.3, 1.4/5.9, 0.7/4.0, 0.3/1.6, 0.4/3.1)으로 교체합니다. |
| 10 | rMD17 | MACE 계열 표의 naphthalene E | MACE Tab. 1에서 NequIP(0.9)과 Allegro(0.2)가 뒤바뀌었습니다. 원본은 NequIP 0.2, Allegro 0.5입니다. 이 오류가 GotenNet, FreeCG, LiTEN 등으로 전파되었습니다. |
| 11 | rMD17 | MACE를 ACE(g)와 같은 test로 비교 | MACE는 test 1000개, ACE(g)는 99k개입니다. 같은 test로 취급하면 안 됩니다. |
| 12 | rMD17 | "TACE 미검증" | v1 Tab. 7에만 있고 v2/v3에서는 삭제되었습니다. 어떤 지표에서도 최고가 아닙니다. |
| 13 | water | NEMP 2.3/37.3 | 출판본(Chem. Sci. 2026)은 2.4(0.1)/36.8(0.5)입니다. |
| 14 | water | REANN 2.0/47 | 어떤 REANN 원 출처와도 맞지 않습니다. 2.4/47(JCP 2022) 또는 2.4/53.2(PRL 2021)를 씁니다. |
| 15 | water | NequIP 2.8/45 | NequIP 논문 값이 아니라 CACE 저자의 재학습 값이고 **validation 오차**입니다. |
| 16 | water | CACE 1.8/47 | CACE T=1의 1.77/47이고 **validation 오차**입니다. |
| 17 | water | ACE 1.7 meV/atom (5.196 meV/H₂O) / 99 | ACEpotentials.jl의 **training 오차**입니다. 제외합니다. |
| 18 | water | TACE "32-water" | 캡션 오타입니다. 실제는 1593 frame입니다. |
| 19 | water | split 2 = "MACE 원래 설정" (HF 카드) | 근거가 없습니다. 우리 MACE 재학습(23.2)이 MACE Tab. V(37.1)를 재현하지 못합니다. |
| 20 | water | PET RMSE 60 / MAE 17 "미검증" | 2406.17747 SI Tab. I에서 **확인**했습니다. |
| 21 | Si | (인용값 자체는 정확) | compression MD는 du Toit의 MQ-MD와 다른 set입니다. 로컬 zip에 MQ-MD 구조가 없습니다. |
| 22 | LiPS | NequIP Tab. VI 2.03/109.9, 0.44/28.4, 0.12/8.3, 0.08/4.9 | arXiv v3 값입니다. Nat Commun Tab. 4의 F 97.8/25.8/7.7/4.7(E는 동일)로 바꿉니다. |
| 23 | LiPS | CAMP 표의 NequIP 열을 "valid 1000 / test 5000"으로 기술 | NequIP 논문에서 복사한 값입니다(valid 100). |
| 24 | LiPS | CAMP 19k 3.6, D 0.18 | Fu split이 아니라 CAMP 자체 split입니다. 0.18은 본문 D 값과 맞지 않습니다. CAMP에는 RDF MAE가 없습니다. E2GNN의 RDF는 0.05입니다. |
| 25 | LiPS | "GemNet-T 1.3이 최고" | FreeCG 1.2(Fu split)가 있습니다. |
| 26 | QM9 | TensorNet 4.3 | v1 값입니다. 최종값은 3.9(1)입니다(U/H/G도 3.9/4.0/5.7). |
| 27 | QM9 | MACE 4.1 = SOTA | SOTA는 GotenNet-L 3.37입니다. MACE 4.1은 -wF를 썼을 가능성이 있습니다. |
| 28 | QM9 | PaiNN 5.9, TorchMD-NET 6.2, DimeNet++ 6.3 | 원본은 5.85 / 6.15 / 6.32입니다. |

---

## 4. 빠진 벤치마크와 보고 일관성 문제

### 4.1 빠진 벤치마크 (리뷰어가 요구할 가능성이 높은 순서)
1. **MD 안정성과 관측량.** 가장 큰 공백입니다. 레포에 MD driver가 없습니다.
   - LiPS: Fu 프로토콜.
   - 3BPA: 1200 K 안정성.
   - water: RDF와 밀도.
   - rMD17 또는 MD22: 안정성.
2. **MD22** ([2209.14865](https://arxiv.org/pdf/2209.14865)). 분자가 receptive field보다 큰 유일한 분자 벤치마크이고, 따라서 "highly locally resolved" 주장을 뒷받침할 수 있는 유일한 분자 벤치마크입니다. GotenNet이 MACE와 비교한 baseline이 있습니다(수치는 이번 감사에서 추출하지 않았습니다).
3. **rMD17 Ntrain=50, AcAc Ntrain=50.** 소량 데이터 주장의 핵심입니다.
4. **AcAc dihedral과 H-transfer, 3BPA β별 dihedral.** reference 논문들은 이를 benchmark의 일부로 다룹니다.
5. **Si**: MQ-MD와 Si-GAP-18 testing DB, 그리고 물성(phonon, 탄성, 결함).
6. **같은 프로토콜의 MPNN baseline**: Si에 MACE/NequIP, water에 MACE 192-2/CAMP/TACE.
7. **비용과 정확도의 trade-off.** 시간당 처리량, GPU 시간, 파라미터 수(6.4M vs MACE 2.8M, CAMP 125k 미만). `bench_speed.py`는 있지만 결과가 없습니다. 작은 RACEv2로 용량 ablation을 합니다.
8. 다성분 bulk 외삽(예: LiPS-25). 현재 재료 쪽 외삽 근거가 Si 하나뿐입니다.
9. SPICE/MACE-OFF는 대규모 데이터라 범위 밖입니다. 추가하지 말고, "화학 간 전이는 범위 밖"이라고 명시합니다.
10. 3BPA train_mixedT는 우선순위가 낮습니다. 비교할 baseline이 거의 없고, 오염되지 않은 확인용도 아닙니다.

### 4.2 보고 일관성 문제
1. **checkpoint와 보고 규칙이 데이터셋마다 다릅니다.**
   - 3BPA, AcAc, rMD17, Si, QM9: best-val + gauge.
   - water: final_ema이거나 run을 섞은 조합.
   - LiPS: 세 가지 후보가 공존합니다.
   - 규칙 하나를 사전에 고정하고 모든 데이터셋에 같게 적용해야 합니다.
2. **gauge refit을 공개하지 않았고, 효과가 데이터셋마다 크게 다릅니다.**

   | 데이터셋 | refit 효과 |
   |---|---|
   | 3BPA | 0.01 meV 이하 |
   | Si | 0.2–0.4 meV/atom |
   | QM9 | −8% |
   | AcAc 300K | 약 2배 |
   | rMD17 | benzene에서 약 7배 |
   | LiPS n=100 | best raw 기준으로는 패배 |

   모든 에너지 표에 raw 열을 넣어야 합니다.
3. **seed 수가 다릅니다.** 3BPA, AcAc, rMD17은 3 seed이고 Si, LiPS, water, QM9는 1 seed입니다. split_seed는 42로 고정이라 split 분산이 std에 들어가지 않습니다. std는 ddof=1입니다. MACE(추정)와 Allegro는 population std를 씁니다.
4. **test set 정의가 섞여 있습니다.** rMD17(1000 vs 99k), LiPS(Fu 5001 vs CAMP/CarNet 5000), water(split 2는 PIMD만), QM9(seed 42 vs GotenNet seed 1).
5. **지표와 단위가 섞여 있습니다.**
   - RMSE: 3BPA, AcAc, water, Si. MAE: rMD17, LiPS, QM9.
   - 에너지 단위: meV/분자, meV/H₂O, meV/atom.
   - 환산이 필요한 값: CAMP/TACE water는 ×3, GotenNet/FreeCG/ViSNet/QuinNet은 ×43.364.
   - 모든 표 머리에 지표, 단위, test 정의를 명시합니다.
   - RMSE가 꼬리에 지배되는 열(3BPA 1200K와 dih, AcAc 600K, water, Si compression)에는 MAE를 일관되게 함께 넣습니다.
6. **recipe가 데이터셋마다 다른데, 논문은 단일 방법처럼 서술합니다.**
   - 3BPA: UFF-nb, 1500 epoch.
   - AcAc, rMD17, Si, LiPS: SOAP, prior 없음, epoch 수는 각각 다름.
   - water: E:F 100:1, 64 채널.
   - QM9: energy-only.
   - 데이터셋별 하이퍼파라미터와 그것을 고른 규칙을 표로 정리합니다.
7. **모든 데이터셋에서 test가 개발 중에 노출되었습니다**(eval_test_every). 3BPA 약 38라운드, AcAc 7개 variant, Si SOAP/AMSGrad, water 약 20 run, QM9 4개 variant입니다. 통합 공개 문단 하나가 필요합니다.
8. **baseline 관리**: 구버전 preprint 값과 2차 전사 오류가 반복됩니다. 같은 모델 계열(PACE, GotenNet, FreeCG, CarNet, ICTP)을 보고가 있는 모든 표에 일관되게 넣습니다.
9. **비교불가 모델 표기가 일관되지 않습니다.** MMEA/ELoRA, Beyond Adam, NN-xTB, FTA, OrbNet-Equi, beSEN-density를 모든 표에서 같은 형식의 블록으로 정리합니다. 우리 쪽의 외부 prior(UFF/ZBL)와 zero-force 라벨도 같은 기준으로 공개합니다.
10. **주장 범위와 증거가 맞지 않습니다.**
    - 외삽 근거는 3BPA T-set, AcAc 600K, Si RSS/compression뿐입니다. 그리고 각각에 약점이 하나씩 있습니다: 3BPA dihedral 패배, AcAc 600K 패배, Si는 1 seed.
    - locality 근거는 같은 split의 MPNN baseline이 없는 bulk 데이터셋에만 기대고 있습니다.

---

## 5. 투고 전 필수 작업 (우선순위 순)

**P0. 지금 바로 할 일 (계산 불필요 또는 소량)**
1. 보고 규칙을 하나로 고정하고 모든 수치를 다시 집계합니다. best-val EMA를 권장하고, raw와 gauge를 둘 다 보고합니다. water 헤드라인을 정정합니다(4.29/69.5, 0.70/21.9). LiPS n=10은 완료 상태로 갱신합니다.
2. 비교표를 정정합니다. 3절의 28개 항목을 반영하고 PACE, GotenNet, FreeCG, CarNet, ICTP, TensorNet 3.9, NEMP 출판본, NequIP Nat Commun 값을 넣습니다. 비교불가 블록을 통일합니다.
3. 주장 문구를 하향합니다.
   - 3BPA: "dihedral 제외 MD 6개 지표 우세, dihedral은 한계". TACE에는 single-seed 주석.
   - AcAc: 300K E만 우세.
   - rMD17: 분자별로 기술.
   - water: 같은 split 비교만.
   - LiPS 19k: 동급.
   - QM9: 열세.
   - `paper_outline_2026-10-30.md`를 수정합니다.
4. 3BPA 대표 recipe를 결정합니다. ZBL-nb(사전 등록)로 가거나, UFF-nb를 쓰면서 test 기반 선택이었음을 공개합니다. "w0.3" 표기를 정정합니다.
5. 공개 문단 두 개를 씁니다. 하나는 test 노출과 recipe 선택 과정, 다른 하나는 gauge refit 방법과 효과입니다. 데이터셋별 하이퍼파라미터와 학습 예산 표도 만듭니다.

**P1. 핵심 재실행 (compute 노드)**
6. seed 확충. Si, LiPS(n=10/100은 subset draw도 여러 번), QM9, water를 3 seed 이상으로 돌립니다. 가능하면 seed마다 split도 바꿉니다.
7. rMD17 99k 평가를 3 seed 모두로 확장합니다.
8. 수렴할 때까지 학습합니다. AcAc(best가 마지막 epoch), Si(compression E가 아직 감소 중), LiPS 19k, QM9.
9. water에서 MACE 192-2와 CAMP를 우리 split으로 다시 학습하고, 여러 seed의 90:5:5 또는 80/20 split 행을 추가합니다.
10. LiPS Fu MD 프로토콜(stability, RDF, Li 확산)을 돌립니다. 이를 위해 MD driver를 구현해야 합니다.

**P2. 주장을 강화하는 추가 벤치마크**
11. MD22. locality 주장을 뒷받침할 수 있는 유일한 분자 근거입니다.
12. rMD17과 AcAc의 Ntrain=50.
13. AcAc dihedral과 H-transfer, 3BPA β별 dihedral과 MAE 보고.
14. Si: MACE/NequIP baseline, MQ-MD와 GAP-18 testing DB 확보, 물성 평가.
15. 비용 벤치마크(`bench_speed.py`)와 파라미터 수를 맞춘 작은 RACEv2 ablation.

**P3. 선택**
16. 3BPA train_mixedT(우선순위 낮음), LiPS-25(재료 외삽), QM9 learning curve.
17. 수동 확인 항목: Fu TMLR 개정판 Table 6, NN-xTB 원문, MACE water 재학습의 E0 설정(로그 재확인), PACE Table 3/4의 RMSE/MAE 표기 불일치.

---
*근거 자료*: 각 절에 인용한 URL과 로컬 run summary(`/scratch/t589a02/BAM_V2/runs/*/loss_*.summary.json`). 논문에서 추출한 텍스트는 `/tmp/claude-500030602/-scratch-t589a02-BAM-V2/5ef853ce-88a8-4dac-9612-afaeb18bb68a/scratchpad/`에 있습니다.