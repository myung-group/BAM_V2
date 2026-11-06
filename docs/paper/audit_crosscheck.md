# RACEv2 보고 수치 교차검증: README.md / methods_pdf_data.json ↔ 벤치마크 감사(2026-11-03)

- 작성일: 2026-11-06. 읽기 전용 분석입니다. 프로젝트 파일은 수정하지 않았습니다.
- 대상
  - `README.md` (branch racev2, 4743c65)
  - `docs/methods_pdf_data.json` (date_note 2026-11-04)
  - `docs/RACE_benchmark_audit_2026-11-03.md` (이하 "감사")
  - `runs/*/loss_*.summary.json`, `runs/r37_uffw03_s4x_cue/dih.txt`
- 재계산 스크립트와 원시 출력
  - `/tmp/claude-500030602/-scratch-t589a02-BAM-V2/5ef853ce-88a8-4dac-9612-afaeb18bb68a/scratchpad/recompute.py`
  - `/tmp/claude-500030602/-scratch-t589a02-BAM-V2/5ef853ce-88a8-4dac-9612-afaeb18bb68a/scratchpad/recompute_out.json`
- 규약: 에너지는 `summary['gauge'][set]`, 힘은 `summary[set]`(best-val EMA checkpoint)을 쓰고 3 seed(42/43/44)를 평균했습니다. std는 따로 표기하지 않으면 ddof=1입니다.
- 감사 이후 바뀐 점: 감사는 이전 run을 기준으로 작성되었습니다(rMD17 `rmd17o_*_soap`, AcAc SOAP 500 ep, Si 1 seed). 현재 README는 최종 통일 레시피 run을 씁니다(`rmd17f_*_uffw03`, `acacf_uffw03`, `siF`, `lipsF`, `wFu*`). 그래서 감사의 **우리 쪽 수치와 승패 판정** 일부는 낡았습니다. 반면 **문헌 참조값과 보고 규칙**은 그대로 유효합니다. 아래 판정은 모두 이 구분을 따릅니다.

---

## 0. 요약 (건수)

| 항목 | 건수 |
|---|---|
| 점검한 문헌 참조값(README 약 74개, JSON 약 200개) 중 감사 값과 **수치가 다른 것** | **5건** (아래 1.A의 A1–A5) |
| 수치는 맞지만 **라벨·출처·주석 정정이 필요한 것** | **12건** (1.B) |
| 감사에 없어 **검증할 수 없는 참조값** | 9개 묶음 (1.C) |
| README 우리 수치 중 summary와 **반올림 이상 차이** | **0건**. 다만 water split 2가 **반발항 없는 run(wF2)**이라 split 1(wFu1)과 레시피가 섞여 있습니다 (2.2) |
| methods JSON 우리 수치의 **낡음과 불일치** | 5건 (2.3) |
| README 표기 오류(볼드, std 규약, 서술) | 5건 (2.4) |
| 보고 규칙 20개 | 준수 4 / 부분 준수 8 / 미준수 8 (3절) |
| 근거가 없거나 단서가 필요한 README 주장 | 10건 (4절) |

---

## 1. 문헌 참조값 대조

### 1.A 감사 값과 수치가 다른 것 (반드시 정정)

| # | 위치 | 현재 값 | 감사 값과 출처(감사에 적힌 그대로) | 교체 문구 |
|---|---|---|---|---|
| A1 | README "Other benchmarks", Water 행 (README:74). JSON `water.split2.MACE64[0]` | MACE 64-0 split 2 E **0.9** | "split 2 결과: stage 2 0.3 meV/atom(**0.75–1.05 meV/H₂O**)/23.2. 로그가 0.1 단위로 반올림되어 있어 E 비교를 확정할 수 없습니다" (감사 §2.4(b), (c)) | README: `MACE 64-0 retrained on the same splits (stage 2, 1 seed): 4.5/64.8 · 0.75–1.05/23.2 (split-2 energy from logs rounded to 0.1 meV/atom; not decidable)`. JSON: `"MACE64": ["0.75–1.05", 23.2]`처럼 문자열로 두거나, 값 0.9를 유지하고 `"MACE64_note": "E 0.75–1.05 (로그 0.1 meV/atom 반올림), 판정 불가"`를 추가 |
| A2 | JSON `qm9.refs.PaiNN` | **5.9** | 원본 **5.85** (감사 §3 #28; §2.7(c) "eSEN / EST / SE3Set / PaiNN 5.6 / 5.64 / 5.74 / 5.85") | `"PaiNN": 5.85` |
| A3 | JSON `acac.refs."ICTP_full"` | 300K [0.75, null], 600K **[4.43, 22.9]** | 300K E 0.75±0.04는 **ICTP_full**이고, 600K E 4.43±0.34와 600K F 22.90은 **ICTP_sym**입니다 (감사 §2.2(c)). 한 행에 두 변형이 섞여 있습니다 | 두 행으로 나눕니다. `"ICTP_full": {"300K": [0.75, null]}`, `"ICTP_sym": {"600K": [4.43, 22.9]}`. 또는 키를 `"ICTP (300K E: full; 600K: sym)"`로 바꿉니다 |
| A4 | README rMD17 본문 (README:61–62). JSON `rmd17.note` | "The reference models were tested on the remaining ~99k frames". JSON: "ACE(g)/PACE/GotenNet/FreeCG는 테스트가 나머지 99k 프레임" | PACE는 "MACE와 같은 split과 seed, **test 1000개**라서 우리 test-1000 수치와 가장 직접 비교할 수 있습니다". FreeCG는 "test set 정의를 명시하지 않았습니다". 99k에서 평가한 것은 ACE(g), GotenNet, NequIP, Allegro, ICTP, TACE-v1입니다 (감사 §2.3(b), (c), §3 #11) | README: `ACE(g) and GotenNet were tested on the remaining ~99k frames, PACE (like MACE) on the official 1000 test frames used here, and FreeCG does not state its test set.` JSON: `"note": "ACE(g)·GotenNet은 나머지 약 99k 프레임, PACE·MACE는 우리와 같은 공식 test 1000, FreeCG는 test 정의 미명시."` |
| A5 | JSON `lips.refs."CAMP (2500)"` | [0.08, **4.9**] | 감사는 CAMP의 n=2500 값을 적지 않았습니다. 대신 0.08/4.9는 감사 §3 #22의 **NequIP arXiv v3 Tab. VI n=2500 값**(0.08/4.9)과 같고, #23에는 "CAMP 표의 NequIP 열은 NequIP 논문에서 복사한 것"이라고 되어 있습니다. CAMP 자체 값이 아니라 CAMP 표의 NequIP 열을 옮겨 적었을 가능성이 큽니다 **[원문 확인 필요]** | CAMP npj 2025 Tab. 1에서 CAMP 행의 n=2500 값을 확인해 교체합니다. 확인 전에는 이 항목을 삭제하거나 `"CAMP 표의 NequIP(arXiv v3) 열 — CAMP 자체 값 아님"`으로 표기합니다 |

### 1.B 수치는 맞지만 라벨·주석·누락을 정정해야 하는 것

| # | 위치 | 문제 (감사 근거) | 교체 문구 |
|---|---|---|---|
| B1 | README 3BPA 표, TACE 행 (README:38) | seed 1개라는 주석과 변형 이름이 없습니다. 감사 §2.1(d)는 "TACE 비교에는 'single seed'라는 주석을 붙입니다"라고 하고, 1200K 값 23.42/54.26은 spe-cart o3 기준입니다(2509.14961v3 Tab. 1). JSON에는 이미 "TACE spe-cart ν=3 (1 seed)"로 되어 있습니다 | `TACE spe-cart ν=3 (1 seed)` |
| B2 | README 3BPA 표 | 감사가 "넣어야 할 공정 baseline"으로 지정한 PACE, ICTP_full, SpinGTP가 빠졌습니다(JSON에는 있음). 사전 등록 규칙이 고른 ZBL-nb와 recipe A(prior 없음)도 없습니다 (§2.1(c), (d)) | 아래 "제안 3BPA 표" 참고 |
| B3 | README 3BPA 표, TACE 이면각 11.0/16.45 | 감사 이면각 행에는 "TACE o2 7.6"이 있습니다. o3 행만 보여주면 TACE의 이면각을 과소평가하게 됩니다 | 표 아래 주석: `TACE ν=2 reaches 7.6 meV on the dihedral energy (arXiv 2509.14961v3).` |
| B4 | README 3BPA 표, GRACE 1200 K F 61.7 | 감사 표에 GRACE의 1200 K F 값은 없습니다. "다중 seed 최고: SpinGTP 61.4"와는 모순되지 않지만 직접 검증된 값은 아닙니다 | 2311.16326 Tab. II에서 확인한 뒤 `GRACE/ACE(g) (PRX 2024, 5 fits)`로 라벨을 정합니다(JSON과 같게) |
| B5 | README Other benchmarks, Si RSS (README:69) | "XPOT-ACE"라고만 되어 있습니다. 감사: "XPOT-ACE-3F 20.5 (du Toit JCTC 2024 Tab. 1, 단일 fit)" | `XPOT-ACE-3F 20.5 / 139 (du Toit et al. JCTC 2024, single fit)` |
| B6 | README Other benchmarks, Si compressed MD (README:70) | "GAP-18 15.7 / 125"가 문헌값처럼 보입니다. 감사: "**공개 RMSE 없음.** 우리가 계산한 GAP-18 15.72 / 125.1". 또 이 set은 du Toit의 MQ-MD와 다릅니다 (§2.5(c), (d), §3 #21) | `GAP-18 evaluated by us 15.7 / 125 (no published RMSE on this set; not the MQ-MD set of du Toit et al.)` |
| B7 | README Other benchmarks, AcAc (README:68) | PACE 논문은 Tab. 3에서는 RMSE, Tab. 4 캡션에서는 MAE라고 해 내부가 일치하지 않습니다. 학습은 5000 epoch입니다. ICTP와 MACE도 함께 써야 합니다 (§2.2(c)) | `PACE 0.73/4.2 · 3.8/16.9 (TMLR 2024 Tab. 3, 3 seeds, 5000 epochs; the paper labels Tab. 4 as MAE) · ICTP 0.75 (full) · 4.43/22.90 (sym) · MACE 0.9/5.1 · 4.6/22.4` |
| B8 | README Other benchmarks, LiPS MD (README:73). JSON `lips.md.refs` | 감사 §2.6(c): "D MAE: E2GNN 0.21 (direct force), GemNet-T 0.24". E2GNN이 빠졌습니다. 또 Fu 개정판 Table 6은 아직 직접 확인하지 않았습니다(§2.6(d), §5 P3 #17) | `NequIP 50 / 0.04 / 0.34, GemNet-T 50 / 0.04 / 0.24, E2GNN D error 0.21 (direct forces); Fu et al. Table 6, revised version not yet checked` |
| B9 | JSON `qm9.refs.MACE` | 감사 #27: "MACE 4.1은 -wF를 썼을 가능성이 있습니다" | `"MACE (‑wF 사용 여부 불명)": 4.1` |
| B10 | JSON `si.refs."GAP-18 (우리 재계산)"` RSS 34.9/177 | RSS 34.9/177은 du Toit가 보고한 GAP-18 값이고, 우리 지표 정의가 이를 정확히 재현합니다(§2.5(b)). compMD 15.7/125만 우리가 계산한 값입니다 | 두 항목으로 나눕니다. `"GAP-18 (du Toit Tab. 1; 우리 재계산 일치)": {"RSS": [34.9, 177]}`, `"GAP-18 (우리 계산, 공개값 없음)": {"compMD": [15.72, 125.1]}` |
| B11 | JSON `final_recipe.name` "RACEv2 (UFF-nb w0.3)". `make_methods_pdf.py` 그림 범례 "UFF-nb w0.3" | 감사 #3: "w0.3"은 가중치가 아니라 `pair_repulsion_switch_width = 0.3 Å`입니다(scale 1.0, switch 1.25×(R_i+R_j)). 실제 config(r37, acacf, siF, lipsF, wFu, rmd17f)도 이 값입니다 | `"RACEv2 (UFF-nb, 전환 폭 0.3 Å)"`. 범례는 `UFF-nb (폭 0.3 Å)` |
| B12 | README 3BPA 표, "RACE V1" 행 (3.4/12.1, 11.7/31.8, 37.5/115.3) | 감사 §2.1(c)는 이 값을 "Bayesian RACE"라고 부릅니다. 값은 같습니다 | 주석: `RACE V1: published values (Bayesian RACE paper), single model.` 출처 논문을 명시합니다 |

### 1.C 감사가 다루지 않아 검증할 수 없는 참조값

다음 값은 감사 문서에 없습니다. 맞다 틀리다를 판정할 수 없으므로 원문으로 확인해야 합니다.
- README/JSON 3BPA: TACE 300 K/600 K(2.69/8.68, 8.10/19.88)와 이면각(11.0/16.45). MACE 300/600/1200 K E와 600 K F(3.0/8.8, 9.7/21.8, 29.8).
- JSON `rmd17.MACE` 10개 분자 20개 값, `rmd17.V1` 14개 값.
- JSON `acac.refs.NequIP` 0.81/5.9/27.8. 600 K E 6.04는 감사와 일치합니다. 감사 #8은 NequIP std가 출처마다 다르니 하나로 고정하라고 합니다.
- JSON `si.refs."XPOT-ACE-2"` 23.1/150.
- JSON `lips.refs."NequIP (19k, Fu)"` 3.7. `lips.md.refs`의 GemNet-dT 0.28, DimeNet 0.30, SchNet 0.38, PaiNN·SphereNet 0.40, NequIP 0.34. 감사가 언급한 것은 GemNet-dT가 non-conservative라는 점뿐입니다.
- JSON `qm9.V1` 5.61.

### 1.D 감사와 일치하는 참조값 (확인 완료)

- **3BPA**: GRACE 1.7/7.6, 7.3/19.2, 25.8, dih 7.4/15.3. TACE 1200 K 23.42/54.26. MACE 1200 K F 62.0(BOBOP의 62.8이 아님)과 dih 7.8/16.5. JSON의 PACE, SpinGTP, ICTP_full 전부. JSON NequIP 행은 감사 #1의 mace-code 재구현 값(3.1/11.3, 11.3/27.3, 40.8/86.4, dih 23.2/23.1)으로 이미 통일되어 있어 정정할 필요가 없습니다.
- **rMD17 "Best published"** 20개 값(README:48–57, JSON `rmd17.sota`)이 모두 감사 §2.3(c)와 같습니다.
- **AcAc**: PACE 0.73/4.2/3.8/16.9. MACE 0.9/5.1/4.6/22.4(감사 #6, 값은 정확하나 SOTA는 아님).
- **Water 문헌 블록**(JSON): TACE 64-2 2.16/30.31, CarNet-3L 1.62/31, CAMP 1.77/34, MACE 192-2 1.9/36.2, NEMP 출판본 2.4/36.8. 감사 #13–#17의 오류값(NEMP 2.3/37.3, REANN 2.0/47, NequIP 2.8/45, CACE 1.8/47, ACE 1.7)은 더 이상 쓰지 않습니다.
- **Si**: XPOT-ACE-3F 20.5/139.
- **LiPS**: FreeCG 1.2, GemNet-T 1.3, CarNet 0.09/5.6, NequIP Nat Commun 0.08/4.7(감사 #22 반영됨), CAMP n=100 0.44/21.6, n=10 1.63/76.4.
- **QM9**: GotenNet-L 3.37, TensorNet 3.9(감사 #26 반영됨), FreeCG 4.11, Allegro 4.7.

---

## 2. 우리 수치 점검 (summary JSON 재계산)

### 2.1 README 표와 대조: 반올림 이상 차이 없음

| 표 | run | 재계산 (E gauge / F; 괄호는 raw E) | README | 일치 |
|---|---|---|---|---|
| 3BPA 300 K | r37_uffw03 | 1.560 (1.566) / 6.308 | 1.56 / 6.31 | ✓ |
| 3BPA 600 K | | 6.045 (6.047) / 15.922 | 6.05 / 15.92 | ✓ |
| 3BPA 1200 K | | 22.196 (22.200) / 45.162 | 22.20 / 45.16 | ✓ |
| 3BPA dihedral | dih.txt × 3 | 16.673 / 19.090 | 16.67 / 19.09 | ✓ |
| Ablation 최종 | abl3_full | 1.870/6.702, 6.525/16.634, 22.890/46.418, dih 16.740/19.283 | 1.87/6.70, 6.52/16.63, 22.89/46.42, 16.74/19.28 | ✓ |
| Ablation xspecies | | 2.550/9.521, 9.796/22.693, 29.861/61.436, 15.363/19.950 | 일치 | ✓ |
| Ablation noprior | | 2.054/7.215, 7.338/18.928, 28.220/60.115, 28.057/23.323 | 일치 | ✓ |
| Ablation nohill | | 2.205/7.364, 7.596/19.077, 26.677/56.674, 21.043/20.170 | 2.20/7.36 … (2.205를 2.20으로 내림, 반올림 경계) | ✓ |
| Ablation nocheb | | 1.936/6.847, 6.813/17.325, 27.754/52.394, 17.790/20.163 | 일치 | ✓ |
| Ablation nols | | 2.266/7.242, 7.189/18.274, 28.146/51.767, 17.733/20.297 | 일치 | ✓ |
| Ablation msgnorm | | 2.124/6.808, 6.779/17.001, 25.732/49.406, 19.443/20.183 | 일치 | ✓ |
| Ablation gate | | 1.799/6.390, 6.414/16.257, 24.145/46.910, 15.067/18.597 | 일치 | ✓ |
| "1200 K force change" 열 | | +32.4 / +29.5 / +22.1 / +12.9 / +11.5 / +6.4 / +1.1 % | +32/+30/+22/+13/+12/+6/+1 % | ✓ (nols 11.5 → 12) |
| AcAc 300 K | acacf_uffw03 | 0.460 (0.461) / 3.063 | 0.46/3.06 | ✓ |
| AcAc 600 K | | 5.324 (5.324) / 15.549 | 5.32/15.55 | ✓ |
| Si RSS | siF | 15.646 (15.720) / 106.734 | 15.65/106.7 | ✓ |
| Si compMD | | 4.472 (4.650) / 68.975 | 4.47/69.0 | ✓ |
| rMD17 aspirin | rmd17f_aspirin | 1.190 (1.189) / 4.012 | 1.19/4.01 | ✓ |
| rMD17 benzene | rmd17f_benzene | 0.022 (0.027) / 0.225 | 0.02/0.22 | ✓ |
| rMD17 나머지 8개 | rmd17f_* | azo 0.358/1.411, eth 0.206/1.414, mal 0.364/2.492, naph 0.116/0.649, para 0.614/2.722, sal 0.363/2.035, tol 0.130/0.780, ura 0.203/1.431 | 전부 일치 | ✓ |
| LiPS 19k | lipsF_fu | 0.020 (**0.100**) / 1.199 | 0.020/1.20 | ✓ |
| LiPS LC 2500/1000/100/10 F | lipsF_lc* | 2.022 / 3.167 / 9.458 / 34.122 | 2.02/3.17/9.46/34.1 | ✓ |
| Water split 1 | **wFu1** (UFF) | 3.376 (3.410) / 63.59 | 3.38/63.6 | ✓ |
| Water split 2 | **wF2** (반발항 없음) | 0.639 (0.724) / 21.961 | 0.64/21.96 | ✓ (단, 2.2 참고) |

README의 rMD17 승·무·패 계산과 V1 대비 감소율(E 64–87 %, F 58–75 %)도 재계산값과 일치합니다(4.2 참고).

### 2.2 Water 행에 레시피가 섞여 있음 (정정 필요)

- README water 행은 split 1이 **wFu1**(UFF-nb, 최종 레시피)이고 split 2가 **wF2**(반발항 없음)입니다. 두 split에서 다른 레시피를 쓰고 있습니다.
- 최종 레시피 split 2 run **wFu2**(3 seed)는 2026-11-05에 이미 완료되었습니다. 값은 **0.573 (raw 0.657) / 21.84**입니다. seed별 E는 0.535/0.600/0.584이고, F는 21.79/21.84/21.89입니다.
- 행 머리의 "rerun with repulsion in progress"는 두 split 모두 완료되었으므로 더 이상 맞지 않습니다.
- 교체 문구:
  `| Water split 1 / 2 (RMSE, E meV/H₂O / F meV/Å; 3 seeds; split 2 test = last 200 frames, all PIMD) | 3.38/63.6 · 0.57/21.84 | MACE 64-0 retrained on the same splits (stage 2, 1 seed): 4.5/64.8 · 0.75–1.05/23.2 |`

### 2.3 methods_pdf_data.json의 우리 수치: 낡음과 불일치

| # | 키 | 현재 | summary 기준 | 조치 |
|---|---|---|---|---|
| J1 | `water.split1.V2` / `V2_seeds` / `V2_std` | 3.8/66.39 (seed E 4.23, 2.96, 4.21) | 반발항 없는 이전 run입니다. 2.96과 4.21은 wF1 s43/s44이고, 4.23/69.5는 감사가 말한 w7 best-val인데 `runs/wF1_s42_cue`는 없습니다. 최종 레시피 wFu1은 **3.376/63.59** (seed 3.364/3.118/3.646, 62.49/60.16/68.12)입니다 | wFu1으로 바꾸고 빨간 글씨를 해제합니다. `status.water`를 "final"로 바꿉니다 |
| J2 | `water.split2.V2` | 0.64/21.96 (wF2) | wFu2 **0.573/21.84** | 위와 같습니다 |
| J3 | `water.units` "E:F 100:1" | | config(wF*, wFu*)는 `energy_weight 1.0, force_weight 10.0`으로 3BPA와 같습니다. 감사 §4.2 #6도 "E:F 100:1"이라고 적었지만 현재 config와 맞지 않습니다 **[loss 정의 확인 필요]** | 확인한 뒤 `"E:F = 1:10 (3BPA와 동일)"` 등으로 정정합니다. lr 1e-3은 맞습니다 |
| J4 | `lips.V2` 학습곡선, `lips.units` "1시드", `status.lips` "old" | 2500 0.035/2.04, 1000 0.052/3.23, 100 0.168/9.52, 10 0.668/33.7 (이전 1 seed) | 3 seed 최종 lipsF: 0.035/2.02, 0.052/3.17, 0.165/9.46, 0.688/34.1. `lc_text`에는 이미 이 값이 들어 있습니다 | `lips.V2`를 3 seed 값으로 바꾸고, `units`를 "3시드", `status.lips`를 "final"로 바꿉니다 |
| J5 | std 규약 | 3BPA `V2_std`는 **ddof=1**(0.06/0.11 …). rMD17, AcAc, Si, water `V2_std`는 **ddof=0**(Si RSS 1.6/1.5는 ddof=0이고, ddof=1이면 1.95/1.84. AcAc 600 K F 0.62는 ddof=1이면 0.75. water split 1 E 0.59는 ddof=1이면 0.73) | | 하나로 통일하고(감사 §4.2 #3은 ddof=1을 기준으로 함) units 문구에 "표준편차 ddof=1"을 적습니다 |

참고로 JSON의 `bpa.V2`, `bpa.V2.bestval`(s42), `rmd17.V2`, `rmd17.V2_raw_E`, `acac.V2`, `acac.V2_raw_E`, `si.V2`, `si.V2_raw_E`, `lips.V2["19k"]`, `ablation_3bpa.rows`는 모두 summary와 일치합니다.

### 2.4 README 표기 오류 (수치는 맞음)

1. **Ablation 볼드가 틀렸습니다** (README:82). 볼드가 "Final recipe" 행에 붙어 있지만, gate 행이 300 K(1.80/6.39), 600 K(6.41/16.26), 이면각 E(15.07)와 F(18.60)에서 더 낮습니다. 이면각 E 볼드도 xspecies 15.36에 붙어 있지만 gate 15.07이 더 낮습니다. 열별 최솟값에 볼드를 붙이면 다음과 같습니다.
   - 300 K: **1.80 / 6.39** (gate)
   - 600 K: **6.41 / 16.26** (gate)
   - 1200 K: **22.89 / 46.42** (final)
   - Dihedral: **15.07 / 18.60** (gate)
   - 볼드를 고치거나, 볼드를 빼고 "reference" 열만 둡니다.
2. **std 규약이 섞여 있습니다** (README:92–93). "seed spread of 2.8"과 "0.96 to 3.41"은 **population std(ddof=0)**입니다. ddof=1로 계산하면 xspecies 이면각 E 3.46, final 1200 K F 1.17, nols 1200 K F 4.18입니다. 반면 JSON 3BPA std는 ddof=1입니다.
   - 교체 문구: `…within the seed spread (sample std over 3 seeds: 3.5)… raises the seed spread (sample std) of the 1200 K force error from 1.17 to 4.18.`
3. **"the gap grows with temperature"** (README:92)는 절대 차이에만 맞습니다. 상대 차이로 보면 xspecies의 F 증가율이 +42 %(300 K) → +36 %(600 K) → +32 %(1200 K)로 오히려 줄고, E는 +36/+50/+30 %로 단조롭지 않습니다. msgnorm의 E도 +13/+4/+12 %입니다.
   - 교체 문구: `the absolute gap grows with temperature`
4. **Ablation 에너지는 gauge refit에 크게 의존합니다**(500 ep). 3BPA 본 결과(1500 ep, refit 효과 0.006 meV 이하)와 달리, ablation에서는 raw와 gauge가 크게 다릅니다.

   | 변형 | raw | gauge |
   |---|---|---|
   | final 300 K | 3.34 | 1.87 |
   | msgnorm 300 K | 6.32 | 2.12 |
   | nols 300 K | 4.77 | 2.27 |
   | nocheb 300 K | 4.00 | 1.94 |

   raw 기준 300 K E는 xspecies 2.67, noprior 2.71, nohill 2.88으로 **최종 레시피(3.34)보다 낮습니다**. 600 K와 1200 K는 raw 기준으로도 모든 변형이 최종 레시피보다 나쁘므로, README:91의 주장 자체는 유지됩니다. raw 열을 추가하거나 다음 문장을 넣습니다.
   - `At 500 epochs the energy constant is not converged: the refit lowers the 300 K energy RMSE of the final recipe from 3.34 to 1.87 meV, and by 0.1–4.2 meV for the variants; without the refit three variants have a lower 300 K energy than the final recipe.`
5. **rMD17 표 머리** "1000 training frames" (README:41)는 실제로 950개 학습, 50개 validation입니다(JSON units에는 "검증 50 포함", 감사 §2.3(b)는 "우리는 950개").
   - 교체 문구: `(MAE, E meV/molecule / F meV/Å, official split 01: 950 training + 50 validation frames from the 1000-frame training set, 1000 test frames, L5h256)`

---

## 3. 보고 규칙 준수 여부

| # | 감사 규칙 (위치) | README | methods JSON/PDF | 판정 | 추가할 문장·열 |
|---|---|---|---|---|---|
| R1 | checkpoint 규칙 하나를 사전에 고정 (§4.2 #1, §5 P0-1) | "checkpoint selected on validation"이 Results 머리에 있습니다. water는 이제 best-val입니다(감사 당시 final_ema 문제는 해결) | 각 units에 "검증 기준 체크포인트" | **준수**. Si는 best와 final을 함께 보고하는 것 미준수 | Si 행에 `final-epoch EMA: RSS 15.51/107.3, compMD 4.06/69.2`를 추가합니다(감사 §2.5(d)) |
| R2 | 모든 에너지 표에 raw 열 (§4.2 #2, §5 P0-1) | **raw가 하나도 없습니다.** 효과가 큰 곳: LiPS 19k 0.100→0.020 (5배), LC n=10 1.33→0.69, n=100 0.33→0.17, ablation 300 K 3.34→1.87, Si compMD 4.65→4.47, water split 2 0.66→0.57 | rMD17, AcAc, Si, QM9는 `V2_raw_E` 있음. 3BPA, LiPS, water는 없음 | **미준수** (README), **부분 준수** (JSON) | 모든 에너지 셀을 `gauge (raw)` 형식으로 씁니다. 예: `LiPS 19k 0.020 (raw 0.100) / 1.20` |
| R3 | gauge refit 방법과 효과 공개 문단 (§5 P0-5) | Training 행에 "closed-form energy-constant refit δ* after training" 한 줄만 있습니다 | PDF 본문에 설명(make_methods_pdf.py:369) | **부분 준수** | 아래 문단 (a) |
| R4 | test 노출 공개 (§4.2 #7) | 없음. 최종 config 전부 `eval_test_every` 100이고, LiPS 19k는 30, LC10은 600입니다 | `final_recipe.selection_note`에 lr을 test 1200 K/이면각으로 고른 과정이 적혀 있음 | **미준수** (README), **부분 준수** (PDF) | 아래 문단 (b) |
| R5 | 3BPA 대표 recipe 결정 과정 공개, ZBL-nb 병기 (§2.1(d), §5 P0-4) | 없음 | `V2_zblnb` 수치는 있지만 "사후 선택"이라는 문구가 없음 | **미준수** | 아래 문단 (c)와 제안 3BPA 표 |
| R6 | "w0.3"를 가중치로 쓰지 않기 (§3 #3) | README에는 "w0.3"이 없음 | `final_recipe.name`, 그림 범례에 "UFF-nb w0.3"이 있어 모호함 | **부분 준수** | B11 |
| R7 | TACE single-seed 주석 (§2.1(d)) | 없음 | 있음 | **부분 준수** | B1 |
| R8 | 이면각을 β별로, MAE와 함께 보고 (§2.1(d)) | 없음 | 없음 | **미준수** | 3 seed 평균, gauge 기준(dih.txt). 아래 표 |
| R9 | "SOTA on all 3BPA" 금지, dihedral 한계 명시 (§2.1(d)) | 그런 문구는 없지만, 300 K E(근소 우세 또는 동급)와 1200 K E(단일 seed TACE 대비 bias 범위 안)에 볼드가 붙어 있고, dihedral 한계를 언급하지 않습니다 | — | **부분 준수** | 제안 3BPA 표 아래 주석 |
| R10 | 표 머리마다 지표, 단위, test 정의 (§4.2 #5) | 3BPA, rMD17, Si, LiPS, water, ablation은 있습니다. **AcAc 행에는 단위가 없습니다** ("RMSE E/F"만 있음). test 크기는 대부분 없습니다 | units 필드 있음 | **부분 준수** | AcAc 행: `(RMSE, E meV/molecule / F meV/Å, test 650/650)` |
| R11 | 꼬리에 지배되는 열에 MAE 병기: 3BPA 1200 K와 dih, AcAc 600 K, water, Si compression (§4.2 #5) | 없음 | 없음 | **미준수** | 3 seed MAE(gauge E / F). 아래 표 |
| R12 | seed 수, std 규약, split_seed 고정 사실 (§4.2 #3) | 3 seed는 적혀 있지만 std가 없습니다. 본문에는 ddof=0 std를 씁니다. split_seed를 고정해 std가 초기화 분산만 담는다는 설명이 없습니다 | std가 ddof 0과 1로 섞임 (J5) | **미준수** | `± is the sample standard deviation (ddof = 1) over 3 initialisation seeds on one fixed split; split variance is not included.` |
| R13 | test set 정의 분리: rMD17 test-1000 vs 99k (§2.3(d)) | 한 표에 섞여 있습니다. "99k gives nearly identical values"라는 주장은 **최종 모델로 확인되지 않았습니다**. 99k 평가는 이전 모델 `rmd17o_*_soap_s42`의 aspirin, ethanol, paracetamol, uracil만 있습니다(`runs/diag/post_eval2_rmd17_gpu*.txt`) | 같음 | **미준수** | 4.3 |
| R14 | 비교불가 블록 통일, 외부 prior 공개 (§4.2 #9, §2.1(b)-3) | "no pretraining, no extra data, no data augmentation"이 있고, UFF는 Repulsion 행에 있습니다. 하지만 외부 경험적 prior라는 명시와 recipe A(prior 없음) 결과는 없습니다. MMEA, ELoRA, NN-xTB 같은 비교불가 블록도 없습니다 | 없음 | **부분 준수** | 아래 문단 (d) |
| R15 | 데이터셋별 하이퍼파라미터와 학습 예산 표 (§4.2 #6, §2.1(d)) | "Default size" 한 줄뿐입니다. Training 행은 lr 1e-2를 일반값처럼 적었지만 water는 1e-3입니다. epoch도 3BPA/AcAc/rMD17 1500, Si 300, LiPS 19k 150, LC10 3000, water 400으로 다릅니다 | 데이터셋별 units에 흩어져 있음 | **미준수** | 열: dataset, channels, lr, epochs, E:F, eval_test_every, n params. 참고: 3BPA 6.4M vs MACE 2.8M (감사 §2.2(b)) |
| R16 | water: 같은 split 비교만, split 2 test가 PIMD뿐이라는 점, MACE stage 명시 (§2.4(d)) | 같은 split 비교만 하는 것은 **준수**. PIMD와 stage는 미기재 | split 2 이름에 "전부 상온 PIMD" 있음. stage 미기재 | **부분 준수** | 2.2의 교체 행 |
| R17 | LiPS 19k는 "동급, force MAE only" (§2.6(d)) | "FreeCG force 1.2"로 나란히 놓았을 뿐 동급이라는 말이 없습니다. MD 관측량은 이제 있습니다(감사 당시 공백 해소) | — | **부분 준수** | `1.20 (tie with FreeCG 1.2)` |
| R18 | Si 3 seed 이상 (§2.5(d)) | siF 3 seed | 3시드 | **준수** | — |
| R19 | 문헌 오류값 제거 (§3) | — | water와 NequIP 오류값 제거됨 | **준수** (A1–A5 제외) | — |
| R20 | water 헤드라인을 best-val로 정정 (§2.4(d)) | best-val (wFu1/wF2) | best-val | **준수** | — |

합계(20개 규칙):
- 준수 4: R1, R18, R19, R20
- 부분 준수 8: R3, R6, R7, R9, R10, R14, R16, R17
- 미준수 8: R2, R4, R5, R8, R11, R12, R13, R15

### R8과 R11에 쓸 수치 (3 seed 평균, gauge E, best-val; 단위는 각 표와 같음)

| set | E RMSE | E MAE | F RMSE | F MAE |
|---|---|---|---|---|
| 3BPA 300 K | 1.56 | 1.08 | 6.31 | 3.93 |
| 3BPA 600 K | 6.05 | 3.92 | 15.92 | 9.23 |
| 3BPA 1200 K | 22.20 | 14.19 | 45.16 | 24.62 |
| 3BPA dih 전체 | 16.67 | 5.94 | 19.09 | 6.28 |
| dih β=120° | 13.91 | 6.17 | 16.52 | 7.19 |
| dih β=150° | 15.59 | 5.40 | 17.43 | 5.98 |
| dih β=180° | 19.81 | 6.25 | 22.71 | 5.68 |
| AcAc 300 K | 0.46 | 0.33 | 3.06 | 1.93 |
| AcAc 600 K | 5.32 | **1.69** | 15.55 | 5.97 |
| Si RSS | 15.65 | 10.86 | 106.7 | 80.95 |
| Si compMD | 4.47 | 3.77 | 69.0 | 48.47 |
| water split 1 (wFu1) | 3.38 | 1.37 | 63.6 | 20.77 |
| water split 2 (wFu2) | 0.57 | 0.46 | 21.84 | 13.48 |

AcAc 600 K는 E RMSE/MAE가 3.1이고 3BPA 이면각은 2.8입니다. 두 열 모두 소수 outlier가 지배하므로 MAE를 함께 보고할 근거가 됩니다.

### 추가할 공개 문단 (README 영어 원문)

(a) gauge refit
> After training, a single per-atom energy constant δ* is refitted in closed form on the training split only (forces are unchanged). Its effect depends on the dataset: ≤ 0.01 meV/molecule on 3BPA and AcAc at 1500 epochs, up to 0.007 meV on rMD17, 0.07–0.18 meV/atom on Si, but a factor of 5 on the LiPS 19k energy MAE (0.100 → 0.020 meV/atom) and up to 4 meV/molecule in the 500-epoch ablation. All energy cells show the refitted value with the raw value in parentheses.

(b) test 노출
> The test sets were evaluated during training (every 100 epochs; LiPS every 30–600) and were seen during development of the recipe (about 38 screening rounds on 3BPA). Checkpoints were selected on validation only, but recipe choices were not: see the 3BPA note.

(c) 3BPA recipe 선택
> The UFF-nb repulsion (switch width 0.3 Å) was chosen after inspecting the 600 K, 1200 K and dihedral test errors. The rule fixed in advance ("1200 K force ≤ +10 % of the base, lowest dihedral energy") selects the ZBL-nb variant, shown as the second row; the expected selection bias is 1–2 seed standard deviations (about 0.4–0.8 meV and 0.7–1.4 meV/Å at 1200 K).

(d) 외부 prior와 공정 프로토콜
> The UFF repulsion is an external empirical prior (fixed functional form and radii, one learned global scale); results without it are given in the ablation. Water has no official split (two random/sequential splits are used); Si uses 95 % of the GAP-18 training set. Models that use pretraining (MACE-OFF fine-tunes such as MMEA, ELoRA) or semi-empirical baselines (NN-xTB) are not compared.

---

## 4. 근거가 없거나 단서가 필요한 README 주장

| # | README 위치와 주장 | 감사 판단 | 제안 |
|---|---|---|---|
| C1 | 제목 "Highly locally resolved MLIP" (README:1) | "locality는 어떤 분자 벤치마크로도 입증되지 않습니다. 3BPA 지름 약 9 Å, 5층 × 5 Å receptive field가 분자 전체를 덮음"(§1, §2.1(a)). 현재 locality를 뒷받침할 수 있는 후보는 MD22뿐입니다(§4.1 #2) | 제목은 사용자가 정한 것이므로 유지하되 본문에 단서를 답니다. `The locality claim is not tested by the molecular benchmarks below (3BPA, AcAc, rMD17 molecules fit inside the 5 × 5 Å receptive field); MD22 is in progress for this purpose.` |
| C2 | rMD17 승패 "wins 7, ties 2, loses 1 … wins 8, ties 1, loses 1" (README:59) | 산술은 재계산과 맞습니다. 참조 정밀도로 반올림하면 E는 benzene 패, ethanol·uracil 무이고, F는 benzene 패, ethanol 무입니다. 하지만 (i) test set이 섞여 있고(A4), (ii) split 01 하나에 3 seed이며, (iii) test가 학습 중 노출되었습니다. 감사는 일괄 "rMD17 SOTA" 표현을 금지했습니다(§2.3(d)). 또 감사의 분자별 판정(force 3개만 승)은 이전 모델 기준이라 낡았습니다 | 문장을 분자별 서술로 바꾸고 test 정의를 적습니다. `On split 01 (3 seeds) RACEv2 has the lowest error among the listed values for 7 molecules in energy and 8 in forces; benzene is worse in both, ethanol and uracil are equal at the reported precision. Only PACE (and MACE) share our test set; ACE(g) and GotenNet use the remaining ~99k frames, FreeCG does not state its test set.` MACE-MARA를 뺀 이유를 각주로 답니다(감사 §2.3: 전사 오류, test 크기와 seed 없음) |
| C3 | "evaluating on the ~99k frames gives nearly identical values" (README:62) | 최종 모델(`rmd17f_*`)은 99k 평가가 **없습니다**. 근거는 이전 모델 s42의 4개 분자뿐입니다(감사: "우리 99k 평가는 seed 42만") | `For an earlier model (seed 42; aspirin, ethanol, paracetamol, uracil) the ~99k evaluation agreed closely with test-1000; the final models have not been evaluated on the ~99k frames yet.` 그리고 최종 모델 3 seed 99k 평가를 P1 작업으로 둡니다 |
| C4 | 3BPA 볼드: 300 K E 1.56, 1200 K E 22.20 (README:35) | 300 K E는 "근소 우세/동급, Welch t≈2.5, seed 44는 1.63, 헤드라인 주장 금지"입니다. 1200 K E는 단일 seed TACE 23.42 대비 "bias 범위 안"입니다. UFF-nb 자체가 사후 선택입니다(§2.1(b)-1, (c)) | 이 두 셀의 볼드를 빼거나 `≈`로 표기하고 각주 (c)를 답니다 |
| C5 | 3BPA 이면각 (README:35) | 이면각 E/F는 **패배**입니다. 16.67/19.09는 GRACE, PACE, TACE, SpinGTP, ICTP, MACE 모두에 뒤집니다(감사 §2.1(c)) | 표 아래: `The dihedral scan is a limitation: every listed reference except V1 has lower dihedral energy and force errors (7.4–11.0 meV, 15.3–17.5 meV/Å).` |
| C6 | "trained on the official training split of its dataset only: no pretraining, no extra data, no data augmentation" (README:8–9) | UFF는 외부 prior이므로 공개해야 합니다(§2.1(b)-3). water에는 공식 split이 없습니다(§2.4(a)). Si는 GAP-18의 95 %로 학습합니다 | 문단 (d)로 보완합니다 |
| C7 | AcAc 행 (README:68) | 정정 후 판정은 "300 K E만 승"입니다(이전 모델 기준). 최종 모델 기준 PACE 대비 300 K E/F 승, 600 K F 승(15.55 vs 16.9)이지만, **600 K E는 패**입니다(5.32 vs PACE 3.8, ICTP_sym 4.43, MACE 4.6). 600 K E MAE는 1.69입니다. dihedral과 H-transfer를 빼면 선택적 보고가 된다고 감사가 경고했습니다(§2.2(d)) | 행에 `600 K energy worse than PACE, ICTP and MACE (RMSE dominated by a few frames; MAE 1.69)`를 추가하고, `AcAc dihedral/H-transfer not yet reported`를 명시합니다 |
| C8 | LiPS MD 행 (README:73) | 우리 Li 확산 오차 0.43은 NequIP 0.34, GemNet-T 0.24, E2GNN 0.21보다 **나쁩니다**. README는 이를 말하지 않습니다 | `Li diffusivity error is higher than NequIP, GemNet-T and E2GNN (0.43 vs 0.21–0.34 ×10⁻⁹ m²/s).` 이 MD가 최종 lipsF 모델로 돌린 것인지 확인해야 합니다(JSON `status.lips`가 "old") |
| C9 | Si (README:69–70) | 3 seed 승리는 이제 근거가 있습니다. 다만 MPNN baseline이 없어 SOTA는 "ACE/GAP 대비"라는 뜻입니다. 이전 1 seed에서 AMSGrad를 쓰면 판정이 뒤집혔다는 기록도 있습니다(§2.5(a), (b)) | `Only ACE/GAP baselines exist on these test sets; no MPNN has been evaluated with this protocol.` |
| C10 | Water (README:74) | split 1 F 63.6은 MACE stage 2 64.8보다 낮지만 MACE stage 1(60.1)보다 높습니다. 감사는 split 1 F 차이가 CI [40, 128] 안이라 유의하지 않다고 했습니다. split 2 E는 판정할 수 없습니다 | `Differences to MACE are within the bootstrap CI on split 1 (one test frame carries ~70 % of the force error); split-2 energies cannot be compared (MACE log rounding).` |

참고로 감사 범위 밖이지만 보이는 점이 하나 있습니다. OAM 비교(README:98–123)는 1 seed이고, V1 쪽은 옵티마이저도 AMSGrad로 바뀌어 있습니다("V1 reverts every RACEv2 addition (including AMSGrad)"). "RACEv2 is better on every metric"는 아키텍처와 옵티마이저 효과가 섞인 1 seed 결과로 서술하는 편이 정확합니다.

---

## 5. 제안 README 3BPA 표 (B1, B2, B3, C4, C5, R5 반영)

```
### 3BPA (RMSE, E in meV/molecule / F in meV/Å; train 450 + valid 50 frames at 300 K; test 1669 / 2138 / 2139 / 7047 frames; 3 seeds, sample std in the methods PDF)

| Model | 300 K | 600 K | 1200 K | Dihedral |
|---|---|---|---|---|
| **RACEv2** (UFF-nb, chosen after test, see note) | 1.56 / **6.31** | **6.05 / 15.92** | 22.20 / **45.16** | 16.67 / 19.09 |
| RACEv2 ZBL-nb (rule fixed in advance) | 1.62 / 6.53 | 7.44 / 16.89 | 24.68 / 50.79 | 10.44 / 17.42 |
| RACE V1 (published) | 3.4 / 12.1 | 11.7 / 31.8 | 37.5 / 115.3 | — |
| GRACE/ACE(g) (PRX 2024, 5 fits) | 1.7 / 7.6 | 7.3 / 19.2 | 25.8 / 61.7 | **7.4 / 15.3** |
| PACE (TMLR 2024, 3 seeds) | 2.4 / 9.1 | 7.9 / 21.4 | 29.6 / 60.7 | 7.6 / 16.0 |
| TACE spe-cart ν=3 (1 seed) | 2.69 / 8.68 | 8.10 / 19.88 | 23.42 / 54.26 | 11.0 / 16.45 |
| SpinGTP | 2.9 / 9.0 | 9.6 / 21.0 | 29.0 / 61.4 | 8.9 / **15.3** |
| ICTP_full (5 runs) | 2.70 / 9.45 | 10.74 / 22.99 | 29.80 / 62.82 | 9.82 / 17.52 |
| MACE | 3.0 / 8.8 | 9.7 / 21.8 | 29.8 / 62.0 | 7.8 / 16.5 |

300 K energy: within the seed spread of GRACE (not a significant difference). 1200 K energy: below the single-seed TACE value
by less than the expected selection bias. The dihedral scan is a limitation: every listed reference except V1 is better.
TACE ν=2 reaches 7.6 meV on the dihedral energy. MACE, GRACE and SpinGTP do not state the dihedral set size.
```

---

## 6. 정정 우선순위 (상위 10)

1. **Water 행 레시피 혼합과 낡은 문구 정정.** split 2를 wFu2 0.57/21.84로 바꾸고 "in progress"를 지웁니다. MACE split 2 E를 0.75–1.05로 고칩니다(A1, 2.2). JSON의 water split 1 값도 wFu1 3.38/63.6으로 바꿉니다(J1, J2).
2. **rMD17 test set 서술 정정.** PACE는 test-1000, FreeCG는 미명시입니다(A4). "99k nearly identical"은 최종 모델로 확인되지 않았습니다(C3).
3. **3BPA 표에 PACE, ICTP, SpinGTP와 ZBL-nb 행을 추가합니다.** TACE에 1 seed 주석을 달고, 300 K E와 1200 K E 볼드를 빼고, 이면각 한계 문장을 넣습니다(B1–B3, C4, C5).
4. **3BPA UFF-nb 사후 선택과 test 노출 공개 문단**을 넣습니다(R4, R5).
5. **raw 에너지 병기.** 특히 LiPS 19k(0.100 → 0.020)와 ablation(3.34 → 1.87)입니다(R2, 2.4-4).
6. **JSON의 AcAc "ICTP_full" 600K 값은 ICTP_sym입니다**(A3). AcAc 600 K E 패배를 명시합니다(C7).
7. **JSON의 QM9 PaiNN 5.9를 5.85로**(A2). LiPS "CAMP (2500)" 0.08/4.9는 NequIP 열을 옮겨 적었을 가능성이 있어 원문을 확인합니다(A5).
8. **Ablation 볼드를 고치고 std 규약을 통일합니다**(ddof=0인 2.8과 0.96→3.41, ddof=1인 JSON 3BPA). "gap grows"는 절대 차이 기준이라고 한정합니다(2.4-1~3, J5).
9. **Si 라벨 정정.** XPOT-ACE-3F, single fit으로 쓰고, GAP-18 compMD는 우리 계산이며 공개값이 없다고 적습니다(B5, B6, B10). LiPS MD에 E2GNN 0.21을 넣고, 우리 확산 오차가 가장 크다는 점을 명시합니다(B8, C8).
10. **locality 단서와 외부 prior, 공식 split 문구를 보완합니다**(C1, C6). JSON의 "w0.3"을 "전환 폭 0.3 Å"로 바꾸고(B11), JSON의 LiPS 학습곡선을 3 seed 값으로 갱신합니다(J4).
