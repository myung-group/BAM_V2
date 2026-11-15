# RACEv2 물(Cheng revPBE0-D3) 성능 개선: 문헌 조사와 검증 결과 보고

## 1. 요약

- **결론부터: 17개 후보 중 "물 성능을 실제로 올린다"는 주장이 검증을 통과한 방법은 없습니다(survivor 0개).** 각 후보는 문헌 수치나 우리 로그로 재검토했고, 남은 것은 결정적 실험의 후보입니다. 채택이 확정된 방법은 없습니다.
- **헤드라인 수치는 대부분 프레임 하나에서 나옵니다.** test F RMSE ~71–80 meV/Å 중 squared error의 71.5%/71.2%를 원본 프레임 383 하나가 차지합니다. 이 프레임의 F RMSE는 756/764 meV/Å이고 median 프레임은 23.5/24.0입니다. 383을 빼면 37.9/38.6입니다. 최저 valid F 33.8은 이미 MACE 64-0(37.1), CAMP(34) 수준입니다(분할이 달라 참고용). 실제로 남은 갭은 세 가지입니다. (a) reactive tail, (b) bulk 에너지 갭: full-test E MAE 1.62 meV/H2O(383 제외 1.31) 대 MACE 1.2, (c) force memorisation과 그에 따른 checkpoint 선택 문제.
- **방향 1: 평가 방식부터 고정합니다.** 기존 dump로 bootstrap CI를 계산하고 classical/PIMD/extreme로 층화하되, 헤드라인은 full-set RMSE로 둡니다. 지금 방식으로는 2–3 meV/Å 수준의 진짜 개선이 보이지 않습니다.
- **방향 2: learning curve(25/50/100%)를 돌립니다.** reactive tail이 데이터를 늘리면 줄어드는지 확인하는 실험으로, 다른 모든 후보가 낼 수 있는 효과의 상한을 정해 줍니다.
- **방향 3: best-val checkpoint로 되돌린 뒤 빠르게 anneal합니다(M03 수정판).** F 최적점(ep 33–57)과 E 최적점(그 이후)이 어긋나는 문제를 겨냥합니다. ep42의 post-gauge valid E 491 meV/frame를 줄이면서 F를 유지할 수 있는지 봅니다. 3BPA는 1500 ep 전체로 재검해야 합니다.
- **방향 4(조건부):** learning curve가 "데이터에 민감"하다고 나오면 reactive 프레임 상향가중을, no-training gate를 통과하면 latent 장거리항(LES)을 시도합니다. 두 방법 모두 물 데이터셋에 대한 직접 근거는 없습니다.
- **공정성에 대해 결정해 주셔야 할 점:** 현재 물 최선 레시피는 E:F 100:1이고 3BPA는 1:10입니다. loss 가중치를 데이터셋별 하이퍼파라미터로 볼지 정해 주셔야 합니다(5절).

## 2. 우선순위 표

> 아래 표는 **결정적 실험의 우선순위**입니다. 채택 권고가 아닙니다. P1–P3은 검증 의견(vote)의 수정사항을 반영한 형태입니다. P5–P8은 평가자 판정이 "weak"였고 반박 투표는 아직 없었습니다.

| # | 방법 | 무엇을 고치나(진단) | 문헌 근거(수치, 출처) | 기존 데이터셋 위험 | 공정성 | 구현 비용 | 결정적 실험 |
|---|---|---|---|---|---|---|---|
| P1 | **평가 인프라(M01 수정판, Phase 0만)**: per-frame bootstrap 95% CI, RMSE/MAE/Q95/max-frame, classical/PIMD/extreme/defect 층화. 헤드라인은 항상 full-set RMSE | 측정 문제. 383 하나가 F² 오차의 71%를 차지해 메서드 간 차이가 보이지 않음. 진단 (3)의 선택 노이즈 | 분할만 바꿔도 MACE가 36.2에서 46.8로 변함([CACE](https://arxiv.org/abs/2402.07472)). PET는 valid RMSE ~40, test 60이고 "RMSE metric is dominated by a few outliers"라고 씀([PET SI](https://arxiv.org/abs/2406.17747)). Q95/ECDF 권고([Pernot](https://arxiv.org/abs/2004.02524)) | 없음(모델 변경 없음) | ok. 단, 규칙을 결과 전에 고정하고 noout이나 PIMD-only를 헤드라인으로 쓰지 않을 것 | 낮음. GPU 평가 1 job | 기존 체크포인트에 `dump_frame_errors --per-atom` 실행. 스크리닝 규칙은 **양면**: bulk(MAE 또는 median-frame F)가 개선되고 tail(extreme stratum RMSE 또는 max-frame)이 악화되지 않아야 함. 3BPA(1200 K, dihedral)에도 같은 규칙 적용 |
| P2 | **Learning curve**: 같은 valid/test에 train 25/50/100% | 진단 (2) tail이 데이터 부족 때문인지 모델 한계 때문인지 구분. 다른 후보들의 상한 결정 | NequIP가 Cheng water에서 learning curve를 수행함([NequIP](https://arxiv.org/abs/2101.03164)). 수치는 읽지 않음. 비평가(critic)가 제안 | 없음 | ok(자기 train의 부분집합) | 짧은 run 2개 | 100%가 50%보다 extreme/classical stratum F에서 CI 밖으로 낮으면 P4로, 아니면 tail을 겨냥한 학습 측 방법의 기대값을 낮게 봄 |
| P3 | **Best-val 롤백 + 빠른 anneal(M03 수정판)**: best-val params/EMA/opt_state로 돌아간 뒤 30 ep 동안 1e-3배까지 기하급수 감쇠. 통합안은 trigger가 아닌 고정 모양(상수 LR, 마지막 15–20%에서 감쇠, epoch 예산은 데이터셋별) | 진단 (3) checkpoint 선택과 E/F 최적 epoch 불일치. 진단 (2) 초기 에너지 under-fit | PET C.5: "We start fast decaying of the learning rate at the epoch with the best validation error". **수치 없음, 그림만 있음.** baseline 기술도 "constant LR"과 "StepLR"로 불일치([PET](https://arxiv.org/abs/2305.19302)). 로컬: water2_hill valid total-E 645.87(ep42, ≈10.1 meV/H2O) → 411.39(ep300). ep42 post-gauge 491.44. test F는 ep42 70.90, ep300 80.95 | **3BPA 위험 실재.** 원안 trigger(W=50, 1%)는 3BPA r28에서 ep 494, r29에서 ep 471에 발화하고 물은 ep 81. 3BPA cosine+SWA는 기각된 바 있음 | ok(규칙 하나를 전 데이터셋에) | 낮음. Step 0은 코드 없이 재시작만 | 아래 Round 1 J3 참조. 같은 epoch의 plateau 체크포인트와 비교해 경과 epoch 효과를 분리 |
| P4 | **Reactive 프레임 상향가중(조건부, P2 결과에 따름)**: 프레임 단위, 데이터셋 무관 규칙(예: 공유결합 반경 합으로 정규화한 최소 쌍거리의 희소도에 비례하는 가중) | 진단 (2)와 383 tail. 로컬(best-val ckpt): train F가 reactive 34.3, non-reactive 20.3, O-O<2.3 Å 프레임 46.4 meV/Å | **문헌 근거 없음.** 비평가가 짚은 빈칸이며, 관련 문헌은 선택 기반([2407.10361](https://arxiv.org/pdf/2407.10361), [2201.09829](https://ar5iv.arxiv.org/html/2201.09829), 수치 미확인). 반대 방향(down-weight)인 M16은 13/13 reactive 프레임을 지워서 기각 | 3BPA에서 per-node strain/force 가중은 1200 K +5–7%, Boltzmann tilt는 기각. 프레임 단위는 미시험 | ok(같은 프레임, 가중만). 통합 규칙 필수 | 낮음–중간(`globals['loss_weight']` gw 경로 존재) | Round 2a |
| P5 | **Latent 장거리 정전기(M10)**: LES → uQiqiu를 별도 stratum으로. **no-training gate 우선** | 진단 (2) 중 전하 분리 결함(H3O+/OH−) 프레임 에너지. 383 같은 초근거리 접촉은 σ=1 Å 스미어링 때문에 못 고침 | Park et al.(RPBE-D3 물, Cheng과 **다른 세트**): NequIP 16.81 → les 15.74 → uQiqiu 12.51(−26%)([2605.05746](https://arxiv.org/html/2605.05746v2)). Cheng 물에서 T=1 CACE는 "smaller reduction"([LES](https://arxiv.org/html/2408.15165)). 벌크 물에서 장거리항은 "largely isotropic and tend to average out"([Kim & Cheng](https://arxiv.org/html/2512.18029)). 자가이온화 이온쌍 프로파일에 장거리 정전기가 필요하다는 보고는 초록 수준([Calegari Andrade et al.](https://collaborate.princeton.edu/en/publications/probing-the-self-ionization-of-liquid-water-with-ab-initio-deep-p/)) | 통합하려면 모든 데이터셋에서 켜야 함. 3BPA 1200 K에서 latent charge가 외삽될 위험 | ok(전하 라벨 불필요) | 중간 | Gate: 잔차를 결함 간 거리에 회귀하고 SPC/E 장 피팅. 설명 분산 5% 미만이면 중단 |
| P6 | **강체 분해 진단(M11 Step 0)**: force 잔차를 trans/rot/vib로 분해 | 다른 후보의 방향 결정. 잔차가 vib에 있으면 분자내/reactive, trans/rot에 있으면 분자간/장거리 | Bigi et al.: 분자당 net force RMS 0.97 eV/Å, 잔여 3.93([2412.11569](https://arxiv.org/html/2412.11569)). 로컬 라벨 F²에서 trans+rot 비중 1.2%(valid)/1.5%(test) | 없음(진단). 이후 loss로 쓰더라도 고립 분자에는 정확히 no-op | ok | 낮음 | trans+rot가 잔차 F²(383 제외)의 25% 미만이면 loss 버전 폐기 |
| P7 | **SOAP b2 0.99(M13)**, shampoo_beta는 0.95로 고정 | 진단 (1) 암묵적 정규화(reactive 배치의 gradient spike 완화) | SOAP 논문: 작은 배치에서 b2=0.99가 "small improvement"이고 0.999는 미시험([2409.11321](https://arxiv.org/html/2409.11321)). Harari: 물 NequIP에서 SOAP(b2 0.95) MAE 0.604/20.9, AdamW 0.773/25.7([2607.02499](https://arxiv.org/html/2607.02499)) | 데이터셋별 하이퍼파라미터면 낮음 | ok | config만 | min valid F가 3% 이상 개선되지 않으면 폐기(Harari seed std ≈ 3%) |
| P8 | **좁은 모델(M02)**: 32 ch, `layer_lmax [3,3,2,2,1]`, hidden `64x(0e+1o+2e)`. **`64x0e` 금지**(각도 정보가 전부 사라짐) | 진단 (1) | MACE 64-0 37.1, 192-2 36.2([MACE](https://arxiv.org/abs/2305.14247)). Cheng 물에서 NequIP l=0은 "significantly" 나쁨([NequIP](https://arxiv.org/abs/2101.03164)). TACE 64-2가 64-0보다 F가 낫다는 수치는 미검증 | 데이터셋별이면 없음(3BPA는 128 ch, lmax 3 유지) | ok | config만 | ep 300에 min valid F가 control보다 1.0 meV/Å 이상 낮지 않으면 폐기 |

### 보류 / 기각

| ID | 방법 | 판정 | 이유(검증 의견 포함) |
|---|---|---|---|
| M01(원안) | 평가 개편을 "물 개선 방법"으로 제시 | 방법으로는 기각, P1로 재분류 | 두 투표 모두 refute. 모델을 바꾸지 않습니다. "MAE가 MACE 64-0 수준"이라는 주장은 outlier를 뺀 수치(20.56)를 쓴 것이고, full-test F MAE는 21.35–22.23으로 20.7보다 높습니다. bulk 에너지 갭(E MAE 1.62 대 1.2)은 tail로 설명되지 않습니다. fold std가 3보다 크다는 결정 규칙은 미리 정해진 결론입니다(재표집 결과 fold RMSE std ~41). worst-frame ID로 fold를 층화하면 모델 출력이 분할 설계에 들어가므로 금지해야 합니다 |
| M03(원안) | validation trigger로 fast anneal + stop_after | 원안 기각, P3로 수정 | 두 투표 모두 refute. trigger는 원래 주장인 ep ~700–850보다 일찍(3BPA ep 471–494) 발화합니다. production trigger는 best 이후 ~50 ep에 발화하므로 Step 0와 다른 상태를 anneal하게 됩니다. 에너지 개선분 대부분은 LR 감소가 아니라 경과 epoch와 gauge refit에서 왔습니다(ep42 pre-gauge 646 → post 491). 예상 이득 ~1 meV/Å은 epoch 간 산포(35.6/39.6/36.2) 안에 있습니다 |
| M04 | 단위 정규화 선택 지표 S=10·E[meV/atom]+F | 보류(P1의 리포팅 옵션) | 현 레시피(100:1)에서는 w5_L5_h64, w6_h64_lr3e3 모두 선택 epoch가 같아 사실상 no-op입니다. 1:10 레시피에서만 E와 F를 맞바꿉니다 |
| M05 | 노이즈 바닥 기반 정지(discrepancy) / flooding | 기각(flooding만 보류) | valid 최소점에서 train F ≈ valid F ≈ 32–37이고, train ≈ 20일 때 valid는 38–39.5입니다. Morozov 정지는 최소점을 지나쳐서 멈춥니다. Huber floor는 방향이 반대입니다. flooding은 회귀 문제 근거가 없고([Ishida](https://arxiv.org/abs/2002.08709)는 분류만 다룸) LiPS(σ̂ ≈ 1.05)에 위험합니다 |
| M06 | MACE식 stage two(E-heavy, lr÷10, SWA) | 기각 | w4_2st에서 이미 사실상 시험했습니다. 전환 시 LR 3.28e-3이 2.25e-4까지 떨어지는 동안 valid E는 ~293 meV/frame에서 평평했습니다. 3BPA에서는 E×100 two-stage가 +300%, lr 1e-3 E30 fine-tune이 1200 K +10%/+7%였습니다. SWA는 no-soup 규칙과 충돌할 수 있습니다 |
| M07 | LR 연동 force weight 감쇠(NEMP/REANN) | 기각 | late force drift는 E:F가 0.1이든 100이든 나타납니다(LR과 시간에 연동). NEMP도 3BPA에는 "constant force weight of 0.2"를 썼고([NEMP](https://arxiv.org/abs/2508.16086)) ablation이 없습니다. 통합 방법론 기준에 걸립니다 |
| M08 | PET 적응형 loss 가중 | 기각 | 물에서의 고정점(~40–130:1)은 이미 시험한 100:1과 같습니다. 3BPA에서는 같은 wE가 유효 비율 ~10⁴배 차이를 만듭니다. SoftAdapt도 잘 고른 고정 가중을 이기지 못합니다([2403.18122](https://arxiv.org/abs/2403.18122)) |
| M09 | 층 내 대칭곱 ν=2–3 | 기각(통합 시) / 보류(데이터셋별, 결정 필요) | 3BPA에서 네 번의 독립 시험 모두 300 K는 개선, 1200 K는 악화(ν2 all-layers 1200 K E +16%, F +3%)였습니다. RACE의 basis-x에는 이미 층 내 곱(m⊗A)이 있습니다. 물에서 ν를 ablation한 문헌은 없습니다 |
| M12 | 2×2×2 replica 누설 검사 + 수용장 제한 | 기각 | edge-local MPNN에서는 covering graph 항등식(E₂₂₂=8E₁₁₁) 때문에 구성상 null입니다(코드 독해로만 확인). 2층으로 줄이면 물 F가 40–42로 악화되고, 3BPA layer cutoff는 1200 K 39.4/82.0이었습니다 |
| M14 | γ²-가중 decoupled decay(regularity prior를 실제 penalty로) | 보류 | 물 근거가 없습니다([Darby](https://arxiv.org/html/2601.15072)는 linear ACE의 aspirin과 Si만 다룸). 3BPA에서 σ를 키우면 악화됐습니다. "SOAP가 1/γ rescaling을 무력화한다"는 전제는 측정되지 않았습니다. 시험한다면 config만으로 되는 Stage 0(σ_n 0.35/0.5, Bessel 6)부터 |
| M15 | ZBL 재설계(학습형 floor 진폭 / 폭 0.3 Å 이상 스위치) | 물에는 기각, 3BPA 쪽 별도 과제로 보류 | 물 ZBL은 epoch마다 부호가 바뀌어 robust한 효과가 없습니다(비교도 epoch를 맞추지 않음). 폭 0.1 Å 스위치가 O-H 0.97 Å에서 ~2 eV/Å의 인력을 만드는 것은 3BPA의 문제입니다. 학습형 진폭은 ×0.3(dihedral 18.39/20.99) 쪽으로 퇴행할 가능성이 큽니다 |
| M16 | robust/heteroscedastic force loss, z-score 프레임 가중 | 기각 | 오프라인 Lam 규칙이 13개 프레임을 flag했는데 13/13이 reactive였고(7개는 O-O<2.3), 383의 학습 유사 프레임(888, 1172)도 포함됐습니다. 3BPA에서는 Huber δ를 0.1에서 10으로 바꿨을 때 1200 K F가 −13% 개선됐습니다(robust할수록 나쁨) |
| M17 | MD 안전 근거리 prior(support-gated) | 보류(안전 기본값으로만) | 구성상 in-support 지표에 영향이 없어 RMSE 개선을 주장할 수 없습니다. core-only ZBL은 압축된 O-H에 고정 에너지를 더할 뿐 383의 거리(H-H 0.70, O-O 1.45)에는 닿지 않습니다 |

## 3. 우리가 이미 해 본 것과 새 제안의 차이

| 이미 시험(물) | 결과 요약 |
|---|---|
| SOAP 대 AMSGrad | SOAP는 최소점(~35.6, ep41–42)에 더 빨리 도달하고 그 뒤 과적합합니다. 3BPA와 rMD17에서 보인 이득이 물에서는 작습니다 |
| A+hill 대 v9a | SOAP에서 v9a ≥ A+hill(final test 5.58/75.81 대 5.78/80.95) |
| Gauge refit | 초기 체크포인트에서만 크게 작용(10.84 → 7.35). two-stage에서는 4.56 → 4.49 |
| Two-stage 1:10 → 100:1/1000:1 | valid E는 개선되지만 full-test E는 ~4.6–4.9에 머물고 F는 약간 악화 |
| 처음부터 100:1 | valid E 개선, test E ~4.7(train E 0.53) |
| Radial prior 제거 | 에너지가 약간 악화, force는 동일 |
| Cutoff 6 Å | F −1.4%, E 효과 없음. two-stage와 결합하면 이득 없음 |
| 폭/깊이 | 64 ch가 E에 약간 유리하고 2층은 F에 불리. L5 h64가 최선(valid ~2.9/36.9, min F 35.78) |
| lr 3e-3 | min valid F 33.78(ep57). 최선이지만 run 미완 |
| WD 1e-4/1e-3, EMA 0.999 | 이득 없음(미완) |
| ZBL(전쌍, ×1/×0.3, 스위치 없음) | 30–40 ep까지 뚜렷한 효과 없음. non-bonded 스위치는 물에서 미시험 |
| Checkpoint 선택(best-val 대 final EMA) | 1:10에서는 best-val이 F 좋음/E 나쁨, final이 E 좋음/F ~80 |

3BPA에서만 닫힌 항목 가운데 물과 관련 있는 것: 곱(ν=2/3), per-node/Boltzmann 재가중, SAM, cosine+SWA, augmentation 계열 전부, stochastic depth.

**새 제안이 다른 점:**
- P1은 기존 시도들이 1-seed, 1-split, outlier 지배 지표로 판정됐다는 점을 고칩니다. 이 때문에 cutoff 6 Å, WD 같은 ±1–2 meV/Å 결론은 사실상 판별 불가였을 수 있습니다.
- P2는 지금까지 없던 데이터 스케일링 진단입니다.
- P3는 감속의 위치(best-val에서 롤백)와 깊이(30 ep 안에 1e-3배)가 기존 plateau의 ×0.8 반복과 다릅니다. 기존 로그에서는 LR을 서서히 내리면 memorisation이 가속됐습니다.
- P4는 down-weight(M16)나 node 단위 가중(3BPA 기각)과 반대로, **프레임 단위 상향가중**입니다.
- P5와 P6은 물에서 한 번도 평가된 적이 없습니다. LES는 3BPA 이유로만 탈락했었습니다.

## 4. 실행 계획

- **물 기준 레시피(현재 최선):** L5, 64 ch, lmax 3, cutoff 5, SOAP lr 3e-3(b1=b2=0.95), batch 2, Huber δ 10, E:F 100:1, EMA 0.99, gauge refit(`research/input/input_3bpa_omat24_w6_h64_lr3e3_cue.json` 계열).
- **3BPA 통합 레시피:** v9a + SOAP lr 1e-2 + ZBL non-bonded(c=1.25). 500 ep 1-seed 참조값은 2.20/6.92, 7.52/17.50, 25.94/52.36, dihedral 10.34/18.29(r33_zblnb 3-seed 진행 중).
- 5-job 한도를 지킵니다.

### Round 0: 학습 없음, GPU 평가 1 job, 나머지는 compute 노드에서 numpy

1. 체크포인트 w5_L5_h64, w6_h64_lr3e3, w4_2st_lr3e3(best), water2_v9a_soap의 train_only/valid/test에 `dump_frame_errors.py --per-atom --index-key orig_idx`를 실행합니다.
2. P1: full RMSE/MAE/Q95/max-frame과 per-frame bootstrap 95% CI를 계산하고, classical/PIMD/extreme(O-O<2.3 또는 H-H<1.2)/defect 개수별로 층화합니다.
   - **결정:** 이후 모든 스크리닝은 CI 폭보다 큰 차이만 인정합니다. bulk와 tail 양면 규칙을 적용합니다.
3. P6: 잔차를 trans/rot/vib로 분해합니다. trans+rot가 25% 미만이면 M11 loss 버전을 폐기합니다.
4. P5 gate:
   - (a) per-frame 에너지 잔차를 defect 수와 H3O+–OH− 최소 거리에 회귀합니다.
   - (b) SPC/E 전하의 Ewald 장으로 force 잔차를 c·qᵢEᵢ로 피팅합니다.
   - 설명 분산이 5% 미만이고 거리 경향이 없으면 LES를 폐기합니다(383/482 포함과 제외 둘 다 계산).
5. **추가 검사(비평가와 이 보고서가 추가):** 383과 쌍둥이 384(train), 482와 481의 per-frame 오차와 라벨 일관성을 비교합니다. 쌍둥이가 학습에 있는데도 383의 F RMSE가 756인 이유가 라벨 문제인지 외삽 문제인지 진단하는 목적입니다. **test 프레임을 제거하는 데 쓰면 안 됩니다.**
6. rMD17/AcAc/Si의 σ̂(net force) 계산.

### Round 1: 물, 1 seed(s42)

| Job | 설정 | 측정 | Kill 기준 |
|---|---|---|---|
| J1/J2 | P2: train 50% / 25%(twin group 단위 무작위 추출, 같은 valid/test). 기준 레시피와 epoch 수 동일 | stratum별 F/E와 CI | 정의상 kill 없음. extreme/classical F가 50%에서 100%로 갈 때 CI 밖으로 개선되면 Round 2a로 |
| J3 | P3 Step 0: w4_2st_lr3e3 ep57 best ckpt(저장돼 있으면)와 water2_hill ep42 ckpt에서 재시작. patience 1, factor 0.5, min_scale 1e-3, 30 ep | post-gauge valid E, valid F, test F를 세 가지로(전체, 383/482 제외, PIMD), 같은 epoch의 plateau ckpt와 비교 | post-gauge valid E가 491보다 20% 이상 낮지 않거나 valid F가 기준−1 meV/Å을 넘으면 폐기. 통과 시 2 seed로 재확인 |
| J4 | P7: `soap_b2` 0.99, `soap_shampoo_beta` 0.95, 150 ep | min valid F와 그 epoch, train–valid gap, CI 기반 test | min valid F 개선이 3% 미만이거나 valid E가 악화되면 폐기 |
| J5 | P8: `hidden_size` 32 또는 `layer_lmax` [3,3,2,2,1](w6_h64_lr3e3가 미완이면 control 재실행으로 대체) | 동일 | ep 300에 control보다 1.0 meV/Å 이상 낮지 않으면 폐기 |

교차 검증: J1, J2, J4, J5는 데이터셋별 하이퍼파라미터라 3BPA 재실행이 필요 없습니다. J3를 통합 스케줄로 승격하면 Round 2c를 실행합니다.

### Round 2: 조건부

- **2a. reactive 프레임 상향가중**(J1/J2가 데이터 민감으로 나온 경우)
  - 통합 규칙: 프레임 가중 = 1 + α·rarity(rmin/(Rᵢ+Rⱼ)), α ∈ {1, 3}. 정규화와 rarity 계산 방식은 모든 데이터셋에서 동일하게 합니다.
  - Kill: extreme stratum F가 CI 밖으로 개선되지 않거나, bulk MAE가 1% 넘게 악화되면 폐기.
  - 교차 검증: 3BPA 500 ep(ZBL-nb) 300/600/1200 K + dihedral, rMD17 aspirin/ethanol, AcAc. 1-seed std 이내여야 하고 dihedral은 10.3/18.3보다 나쁘면 안 됩니다.
- **2b. LES**(Round 0 gate를 통과한 경우)
  - 물에서 les, les_uQiqiu(head zero-init) 2 run.
  - Kill: min valid F 개선이 3% 미만이고 결함 프레임 E RMSE도 개선되지 않으면 폐기.
  - 교차 검증: 3BPA 500 ep + dihedral(1200 K 지표가 3% 넘게 악화되면 폐기), rMD17 aspirin/ethanol, AcAc, Si에서 q ≈ 0 확인.
- **2c. 통합 스케줄**(J3 통과 시): 상수 LR 후 마지막 15–20%에서 1e-3배로 감쇠.
  - **3BPA는 1500 ep 전체를 ZBL-nb로 실행합니다.** 500 ep로는 피해가 보이지 않습니다.
  - ±2% 이내여야 하고 dihedral은 10.3/18.3보다 나쁘지 않아야 합니다.
  - rMD17 aspirin/ethanol, AcAc 600 K E를 확인하고, LiPS/QM9는 감쇠 시점만 기록합니다.
- 메서드가 정해진 뒤 헤드라인용으로 grouped fold를 최대 3개 만듭니다. twin을 묶고 물리 기술자로만 층화하며, seed와 규칙을 미리 고정합니다.

## 5. 벤치마크 프로토콜 관련 사실

- **데이터:**
  - 1593 프레임 × 64 H2O(192 원자), revPBE0-D3, CP2K. 앞 1000개는 고온 classical(0.7–1.2 g/mL에서 평형 후 steepest descent로 quench, FPS로 선택), 뒤 593개는 상온 PIMD입니다([2006.13316](https://arxiv.org/abs/2006.13316), [1811.08630](https://arxiv.org/abs/1811.08630)).
  - 400 Ry 컷오프입니다([2512.23939](https://arxiv.org/html/2512.23939v2)).
  - 라벨은 TotEnergy와 force뿐이고 virial은 없습니다.
- **표준 분할이 없습니다**([repo](https://github.com/BingqingCheng/ab-initio-thermodynamics-of-water)).
  - NequIP: valid 100, 고정 test 190
  - BPNN: 1303/290
  - PET: 1303/100/190
  - CACE: 90%, validation 오차 보고
  - CAMP: 90:5:5(1433)
  - TACE: 1432/79/81
  - REANN: 80/20
  - MACE: "different train test splits"
  - NEMP: 미기재
  - "MACE = 앞 1393 / 뒤 200"은 제3자 카드의 주장으로 미검증입니다([HF card](https://huggingface.co/datasets/OneScience-Sugon/water)). 뒤 200개는 전부 결함 없는 PIMD라 MACE의 MAE/RMSE 비 0.51과 맞지 않을 가능성이 있습니다(추론).
- **분할 효과:** MACE가 CACE 분할에서 1.9/46.8, 자체 논문에서 36.2([CACE](https://arxiv.org/abs/2402.07472)). PET는 valid ~40, test 60([PET SI](https://arxiv.org/abs/2406.17747)).
- **문헌 수치(E meV/H2O / F meV/Å RMSE):**
  - MACE 64-0: 1.9/37.1(MAE 1.2/20.7). 192-2: 1.9/36.2(MAE 1.2/18.5)([MACE Table V](https://arxiv.org/abs/2305.14247))
  - NequIP: MACE 표에는 MAE 2.5/21만 있고, 2.8/45는 CACE와 NEMP 표에서 나온 수치
  - REANN: 2.4/53.2(MACE 표). NEMP 표의 2.0/47은 REANN 자체 표와 불일치
  - CACE T=1: 1.77/47
  - NEMP: 2.3/37.3([2508.16086](https://arxiv.org/abs/2508.16086))
  - CAMP: 0.59 meV/atom/34([2411.12096](https://arxiv.org/abs/2411.12096))
  - TACE 64-0/64-2: 0.58/33.16, 0.72/30.31(meV/atom)([2509.14961](https://arxiv.org/abs/2509.14961)). 1 meV/atom = 3 meV/H2O. TACE 표 캡션에 "32-water system"이라고 적혀 있어 데이터 동일성이 의심됩니다
  - PET: F MAE 14.4(Rc 4.25). 다른 PET 모델은 F RMSE 60, MAE 17
  - 우리 수치는 full-test F MAE 21.35–22.23, E MAE 1.62
- **학습 레시피 차이:**
  - MACE 물: 6 Å, 2층, AMSGrad 0.01, λE=10/λF=1000에서 1000/10으로 전환, docs 기준 batch 2, SWA는 800 ep 중 400부터([docs](https://mace-docs.readthedocs.io/en/latest/examples/training_examples.html))
  - MACE 분자: 1/1000 고정, 256 ch, 5 Å([2206.07697](https://arxiv.org/abs/2206.07697))
  - NEMP: 물은 force weight 10에서 0.1로 감쇠, 3BPA는 고정 0.2
  - CAMP: 최대 10000 ep, patience 400
  - 정리하면 **물에서 강한 계열들은 폭, L, cutoff, lr, batch, 때로는 loss schedule까지 데이터셋별로 바꿉니다.** 우리도 물 100:1과 3BPA 1:10의 차이를 하이퍼파라미터로 인정할지 결정해 주셔야 합니다.
- **라벨 노이즈:** 우리 split의 net force는 평균 2.29 meV/Å/atom이고, 이를 iid로 가정한 σ̂_F ≈ 20입니다. Beck & Marsalek은 같은 GPW 400 Ry 조건에서 2.14 meV/Å/atom을 보고했고, 재계산 라벨에서는 테스트 F가 38.1에서 33.8로 바뀌었습니다(다른 데이터셋)([2603.20451](https://arxiv.org/html/2603.20451v2)).
- **로컬 분할:** train 1290(classical 803 / PIMD 487), valid 144, test 159.
  - outlier 383/482의 쌍둥이(384/481)가 train에 있습니다.
  - 모든 valid/test 쌍거리는 train support 안에 있습니다(train 최소 O-O 1.364, H-H 0.549, O-H 0.741 Å).
  - 에너지가 동일한 중복 프레임이 최소 5개 있습니다.
- **장거리항:** Cheng 물에서 MACE, TACE, CAMP 모두 장거리항이 없습니다. 이 데이터셋에서 BOTNet, Allegro, SO3krates, eSEN, GRACE, ICTP, HotPP 결과는 찾지 못했습니다.

## 6. 검증되지 않은 주장과 빠진 영역

**미검증 핵심 가정**

- **σ̂_F ≈ 20 meV/Å 노이즈 바닥**: M05, M11, M13, M16과 "PIMD 포화" 판단이 여기에 의존합니다. per-atom iid 가정이 시험된 적 없고, 분자 내 상관 노이즈(egg-box)가 있으면 추정이 치우칩니다. LiPS σ̂ ≈ 1.05, "VASP가 net force를 제거한다"(Si)도 근거가 제시되지 않았습니다.
- **코드 독해로만 한 주장**: M12의 covering-graph 항등식, M02의 "64x0e+16x1o 구성 실패"와 hidden lmax 1에서의 Y³ 소실.
- **측정되지 않은 분석**: M14의 SOAP가 1/γ rescaling을 무력화한다는 주장.
- **원 논문 미확인**:
  - PET fast anneal의 물 이득 수치(그림만 있고 baseline 기술도 불일치)
  - MACE `--swa`가 실제 가중평균인지 여부
  - Tompa et al. "two-stage 무이득"의 데이터셋
  - DeePMD prefactor 공식(검색 요약에서 옴), DeePMD 6.3/92
  - M10의 SPICE 2.35/72.43 → 1.88/61.13(2507.14302에서 찾지 못함), Kim & Cheng 21.0 대 21.1, Baldwin et al.
  - M17의 MACE-MP-0 "holes" 인용문, DP-ZBL 수치
  - TACE 물 수치(32-water 의심), NEMP 출판본 2.4/36.8, CACE 파라미터 수, MACE 3BPA L=0 수치
- **로컬 분석 미재현**:
  - M11의 분자당 net force ≈ 0.70 eV/Å가 Bigi et al.의 0.97과 불일치(원인 미해결)
  - M15 물 ZBL 비교는 epoch 비매칭(76 ep 대 256 ep). DFT 쌍투영 force(O-O −2.5 대 ZBL +12.8 eV/Å)도 재실행하지 않음
  - M13 "batch 1은 사실상 이미 시험" 주장에 로그 근거 없음
  - M04는 rMD17과 Si에서 replay하지 않음
- **분할 관련**: M01의 twin 통계(184/999)와 "HF 카드 분할 반박"은 추론입니다.

**빠진 영역(비평가 지적)**

- **Reactive 프레임 상향가중이나 oversampling**: 지금까지 down-weight만 검토했습니다. P4로 반영했고 문헌 근거는 없습니다.
- **Learning curve 진단**: P2로 반영했습니다.
- **MD 관측량 평가**(안정성, RDF, 밀도, 확산; 라벨 불필요하므로 공정): Fu et al.은 force 정확도와 시뮬레이션 지표가 잘 맞지 않고 물에서 불안정한 모델이 있다고 보고했습니다([2210.07237](https://arxiv.org/pdf/2210.07237), 검색 요약 수준). M15와 M17의 가치는 MD에서만 드러납니다.
- **Ewald/Fourier message passing**([2303.04791](https://arxiv.org/abs/2303.04791)): 근거 없이 기각됐으므로 M10 사다리의 한 단으로 gate와 함께 넣어야 합니다.
- **Kalman 계열 옵티마이저(RLEKF)**: DeePMD 물(다른 데이터셋), Adam 대비 test F 68.1 대 87.6([2212.06989](https://arxiv.org/abs/2212.06989)). 우선순위는 낮습니다.
- **보조 direct-force head(multi-task, 예측은 conservative 유지)**: Bigi et al.이 Cheng 물에서 conservative와 direct를 다뤘습니다. staged 레시피는 no-pretraining 규칙과 충돌할 수 있어 결정이 필요합니다.
- **SAM**: 3BPA에서 이미 기각됐으므로(1200 K F 117.5/89.8 대 80.3) 재제안하지 않습니다.
- **TACE/CAMP 1432/79/81 분할**을 추가 비교 행으로 넣는 안은 아직 제안되지 않았습니다.
- **닫힌 각도**: virial 학습(라벨 없음), stochastic depth(3BPA에서 깨짐, strata 합과 비호환).

**결정이 필요한 공정성 쟁점**

1. 물 E:F 100:1과 3BPA 1:10을 데이터셋별 하이퍼파라미터로 볼지
2. SWA가 no-soup 규칙에 걸리는지
3. ν(body order)를 데이터셋별 하이퍼파라미터로 볼지
4. 분석적 D3 차감이 Δ-learning에 해당하는지(현재는 기각 쪽)