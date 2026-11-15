# 다음 사용 기간 계획: OMol25 전체 데이터로 RACE v3 학습 (2026-11-15 작성)

이번 창(2026-11-16 09:00 종료)에서 RACE v3는 OMol25 4M 부분집합으로 학습했다. 같은 계산량으로 전체 데이터(약 1억 구조)를
쓰면 짧은 거리 정확도 격차(UMA-s-1.2.1과 에너지 3-6배, 힘 2-3배)를 가장 크게 줄일 수 있다. 이 문서는 그 준비 사항이다.
다운로드·설치·작업 제출은 사용자가 말할 때까지 하지 않는다.

## 1. 데이터 (공개 URL, HF 승인 불필요: dl.fbaipublicfiles.com)

| 분할 | 구조 수 | 압축 크기 | URL (`https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/...`) |
|---|---|---|---|
| OMol-0 train 전체 | 101,666,280 | 456 GB | `250514/train.tar.gz` |
| OMol-0 train 4M (보유) | 3,986,754 | 19 GB | `250514/train_4M.tar.gz` |
| OMol-0 neutral train | 34,335,828 | 101 GB | `250514/neutral_train.tar.gz` |
| OMol-0 val (보유: `dl/val`) | 2,762,021 | 20 GB | `250514/val.tar.gz` |
| OMol-0 test | 2,805,046 | 8 GB | `250514/test.tar.gz` |
| OMol-1 train 전체 (2026-01 갱신) | 139,836,891 | 592 GB | `260123/train.tar.gz` |

- 출처: `/appsdata/t510a13/omol25/hf_card/DATASET.md`.
- **권장: OMol-0 전체.** eSEN/UMA 논문 기준선과 같은 데이터라 비교가 공정하다.
  - OMol-1은 더 크지만, 기준선 모델이 어느 판으로 학습했는지 먼저 확인해야 한다.
- 저장공간: `/appsdata` 그룹 할당량 20 TB에서 104 GB를 쓰는 중이다.
  - 압축 파일과 풀린 파일을 합쳐 약 0.9 TB가 필요하고, 풀고 나면 tar는 지운다.
- 받는 곳: dm01-04(x86, 데이터 전송 노드)에서 `wget -c`. 명령은 `jobs/omol_full_download.txt`에 있다.

## 2. 사전 처리 (CPU, 학습 전에 끝낼 것)

| 단계 | 내용 | 예상 시간 |
|---|---|---|
| 압축 풀기 | dm 노드 또는 CPU 작업 | 1-2 h |
| 크기표 (cutoff 6) | `stream_io.py sizes`, 64 workers (4M: 97 s) | 약 45 min |
| E0 재적합 | `omol4m_e0.py`, 학습 shard에서 표본 추출 | 약 30 min |
| 레이블 점검 | Lowdin/NBO 전하·스핀, multipoles 보유율 (`md_water/spin_label_check.py`) | 약 30 min |
| 형식전하 점검 | `jobs/formal_assign_check.sh` 방식 (레이블 q0와 규칙 일치율) | 약 15 min |

- 스크립트 초안은 `jobs/omol_full_prep.sh`(미제출, 새 작업)에 있다. 제출 전에 작은 시험 작업을 먼저 돌린다.

## 3. 계산량

- v3(L5x128, polar + dispersion + cs_local + 학습형 ct_decay)의 측정 속도: 3노드(GH200 12개)에서 **초당 약 1,700 구조**.
- 이번 창의 v3는 170 에폭 x 75만 = 약 1.28억 구조를 봤다. 같은 계산량이면 **전체 데이터를 1.3바퀴** 돈다.
- 다음 창 후보:

| 노드 | 100시간에 보는 구조 수 | OMol-0 전체 기준 |
|---|---|---|
| 3 | 6.1억 | 약 6 에폭 |
| 6 | 12억 | 약 12 에폭 (OMol25 논문 수준) |
| 10 | 20억 | 약 20 에폭 |

- 계산이 부족하면 20-30M 부분집합으로 줄인다. 그래도 4M보다 크게 나아질 것으로 예상한다.
- 멀티노드는 지금 3노드 체인 스크립트(`jobs/v2local_3node_chain.sh`)를 그대로 쓴다.

## 4. 모델·학습 변경 (이번 창 검토에서 나온 것, 새 학습에서만 켠다)

1. 스핀 에너지 `mu += w*max(0, 2S - sum mu)`의 꺾임 → softplus 옵션 (힘 불연속 제거).
2. ct_decay가 r0(2 Å) 안쪽에서 상한을 넘는 문제 → r < r0에서는 상한으로 고정하는 옵션.
3. 가중 레이블 손실(전하·스핀·쌍극자)이 GPU 수에 비례하는 문제 → 전역 레이블 수로 정규화(psum). 그다음 설정의 "/16" 보정을 없앤다.
4. 비활성 기체(Ar2, Li+...Xe) 반발 과다·분산 과소 → 데이터 보강(rare-gas dimer 등)과 분산 계수 초깃값 점검.
5. 고립 이온·원자 기준 에너지(IP-EA 오차): OMol 공식 atom_refs(`references/iso_atom_elem_refs.yaml`) 사용 검토.
6. `parts`의 repeat 사본이 몰리는 문제 → 사본 번호를 정렬 키로 (OAM 쪽과 합의 후).
7. 평가: 레이블 없는 구조는 반드시 형식전하 자동 지정(`formal_assign`, 481c81e 이후 코드)으로 평가한다.
   - 이번 창 공식 val 200개에서 자동 지정 없이 486 meV, 자동 지정으로 274 meV가 나왔다.

## 5. 일정 (5일 창 기준)

| 시점 | 일 |
|---|---|
| 창 시작 전 | 다운로드·풀기(dm 노드), 크기표·E0·레이블 점검 |
| 0일 | 사전 점검(pre-flight) 30분, 본학습 시작 (cosine, warmup 1 에폭) |
| 1-4일 | 본학습. 하루 1회 중간 시험(내부 그룹, 물리, 이온-물, 크기 일관성, 공식 val 일부) |
| 마지막 반나절 | 최종 총 시험 (공식 val 전체, 공개 모델 비교 그림) |

## 6. 확인할 것

- 기준선 모델(UMA-s-1.2.1, eSEN)의 학습 데이터 판(OMol-0 / OMol-1)과 데이터 크기별 결과(논문 표) → 공정 비교 표.
- OMat 사전학습 체크포인트에서 시작할지: 구조를 맞춰야 한다(cutoff 6, spectral, 같은 원소 목록).
- 노드 배정: 그룹 규칙상 node0485-0498만 사용, 시험용 1노드 확보.

## 7. 비교 과제: BAM_v1 Thole 모델과 같은 데이터로 학습 (2026-11-15 추가)

- 비교 대상: BAM_v1 브랜치의 Thole 기반 BAM 모델.
  - 파라미터 약 4.5M, WBM 테스트셋 형성 에너지 MAE 약 33 meV/atom(학습자 보고).
- 학습 데이터 (같은 구성으로 RACE를 학습해서 비교한다):
  - OC25: 표면 slab (촉매).
  - MPtrj / sAlex: 응집상 (bulk).
  - OMol25 / OPoly26: 분자·기체·응집상 (생체분자에 적용 가능). OPoly26은 train+val 590만 + 20만, 66 GB.
- 방법:
  - 주 브랜치 `main`(racev2-thole-merge를 병합한 상태)의 다중 데이터 스트림(parts, 소스별 task·레이블, 주기·비주기 혼합)을 쓴다.
  - 파라미터를 약 4.5M으로 맞춘 모델과, 더 큰 모델 두 가지로 학습한다.
- 보고 항목:
  - WBM 형성 에너지 MAE (meV/atom, racev2 쪽 eval_wbm.sh)
  - OMol25 공식 val
  - 이온-물·염다리·크기 일관성
  - OC25 slab 오차
- 혼합 학습 전에 고칠 것 (오늘 검토 결과):
  - RACE_LR이 혼합 배치에서 그래프별 pbc를 무시하고 분자에 Ewald를 적용함
  - parts repeat 사본이 한데 몰림
  - 레이블 손실이 GPU 수에 비례함
- 데이터: 지금 OC25와 OPoly26은 없다. 다운로드는 사용자 지시 후에만 한다.
