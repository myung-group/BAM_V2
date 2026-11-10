# RACEv2 온보딩 — 계정 t510a13 (2026-11-10)

이전 계정(t589a02)에서 하던 연구를 t510a13 계정에서 이어가기 위한 정리. 기준 커밋 35fa074(브랜치 `racev2`), 기준 문서는
`docs/HANDOFF.md`(2026-11-06 작성)이며 아래 수치는 모두 그 시점 값이다. 이 문서를 쓰는 동안 작업 제출·무거운 계산은 하지 않았다.
`/scratch/t589a02/...` 경로와 작업 ID는 **이전 계정 기준** 표기이며 이 계정에서는 접근하지 않는다.

## 1. 프로젝트 요약

RACEv2는 JAX + Flax NNX로 구현한 E(3) 등변 메시지 전달 원자간 포텐셜(MLIP)이다(`bam_omat24/race_nnx.py`). 층마다 stratum
readout이 있어 총 에너지가 층별 stratum의 합이며, 깊이에 따라 body order가 커진다. RACE V1 대비 추가한 요소는 원자 중심 곱셈 인자
(`x_mode=basis`), hill 크기 제어, Chebyshev형 층 점화식 + LayerScale, 방사 기저의 Gaussian prior + 원소 shell 채널, UFF 비결합
반발 prior(학습되는 전역 스케일 하나)이고, 학습은 SOAP(float64) + 학습 후 에너지 상수의 닫힌형 재적합이다. 모든 보고 수치는 공식
학습 분할만 사용(사전학습·추가 데이터·증강·앙상블 없음). 논문 가제는 "Highly locally resolved machine learning interatomic
potential: RACEv2"이며 OAM/Matbench 결과가 나오기 전에는 범용 모델이라고 주장하지 않는다.

## 2. 지금까지 결과 핵심 수치 (3 seed 평균, validation 기준 체크포인트; README 기준)

| 항목 | RACEv2 | 비교 |
|---|---|---|
| 3BPA 300/600/1200 K (E meV/분자 / F meV/Å) | 1.56/6.31 · 6.05/15.92 · 22.20/45.16 | GRACE 1.7/7.6 · 7.3/19.2 · 25.8/61.7; TACE 2.69/8.68 · 8.10/19.88 · 23.42/54.26 |
| 3BPA dihedral | 16.67/19.09 (한계) | GRACE 7.4/15.3 |
| 3BPA 사전등록 규칙대로(ZBL-nb) | 1.62/6.53 · 7.44/16.89 · 24.68/50.79 · dih 10.44/17.42 | UFF-nb 선택은 test 확인 후 이루어짐(README 주석) |
| rMD17 (MAE, E/F), split 01 | aspirin 1.19/4.01, azobenzene 0.36/1.41, naphthalene 0.12/0.65 등 | 에너지 7개·힘 8개 분자에서 최저(벤젠은 열세) |
| AcAc 300/600 K | 0.46/3.06 · 5.32/15.55 | PACE 0.73/4.2 · 3.8/16.9 (600 K 에너지는 열세) |
| Si GAP-18 → RSS / 압축 MD | 15.65/106.7 · 4.47/69.0 | XPOT-ACE 20.5/139 |
| LiPS 19k (MAE) | 0.020/1.20 (raw E 0.100) | FreeCG F 1.2 |
| LiPS MD 50 ps×5 | 안정 50 ps, RDF 0.044, Li 확산 오차 0.43 | NequIP 0.34, GemNet-T 0.24 (확산은 열세) |
| Water split 1 / 2 | 3.38/63.6 · 0.57/21.84 | MACE 4.5/64.8 · 0.75–1.05/23.2 |
| 절제(3BPA 500 ep) | 1200 K 힘: V1 atomic feature +32 %, radial prior 제거 +30 %(dihedral +68 %), hill 제거 +22 % | 7/10 완료(noshell, nouff, amsgrad 남음) |
| OAM 10k subset (RACEv2 vs V1) | OMat24 E MAE 110 vs 160, F RMSE 490 vs 751 meV/Å | 3개 데이터셋 모두 RACEv2 우세, 에폭당 2–2.4배 느림 |
| 100k OMat24 크기 선별 | **L5×64 선택**(11.7 M, F RMSE 281.0); 모든 크기의 F MAE 148–155 meV/Å | AMSGrad 대조군이 SOAP와 동률, 1.5배 빠름 |
| LAMMPS `pair_style race` | owner 검증 2e-13 eV; halo 1/2/4 rank 일치 1e-14 eV/Å; 8 GPU 약한 확장성 88 % | halo는 1 GPU에서 owner보다 3.3배 느림 |

주의: 위 수치에는 test 모니터링 공개 사항(개발 중 약 38회 3BPA 선별 라운드)이 따른다. rMD17 최종 모델은 ~99k 프레임 평가 전이다.

## 3. 파일 구조 지도

```
/scratch/t510a13/race_v2/
  RESTORE.md, HANDOFF.md, DATASETS.md      복사본(원본은 BAM_V2/docs)
  BAM_V2_racev2.bundle                     git 번들(racev2, main)
  runs_text.tar.gz                         2,451개 항목, 모든 런의 텍스트 결과(미해제)
  data_splits.tar.gz                       108개 항목, 처리된 분할(미해제, 압축 325 MB)
  docs_fonts.tar.gz                        NanumGothic/NotoSansKR(미해제)
  env/                                     pip_freeze_bam_cue.txt, python_version.txt(3.12.15), DCE shim 패치
  manifests/                               checkpoints_core.txt(10개), checkpoints_all.txt(125개)
  claude_memory/                           이전 계정 메모리 19개(참고용)
  BAM_V2/
    CLAUDE.md                              규칙 요약
    README.md                              결과(영어)
    bam_omat24/                            RACEv2 코어: race_nnx.py, train_3bpa.py, soap_*.py, data_3bpa.py, export_lammps.py,
                                           long_range.py·race_lr_nnx.py(RACE_LR), stream_*.py, soap_laplace.py, bayes_lastlayer.py
    bam_3bpa/, bam_torch/, bam_torch_cue/  다른 백엔드(건드리지 않음)
    lammps/                                PJRT C++ 런타임, pair_race, build_lammps.sh
    input/                                 설정 681개(369개가 t589a02 경로 포함)
    runs/diag/                             Slurm 스크립트·진단(63개 .sh, git에 force-add; 61개가 t589a02 경로 포함)
    data_prep/, dataset_3BPA/, tests/
    docs/                                  HANDOFF, RACE_LR_design, paper/RACEv2_draft, stream_loader_design, SOAP_Laplace,
                                           torch_port_scope, LOCAL_TEST_PROMPT, DATASETS, benchmark audit, methods PDF 등
    runs/                                  git 무시. 현재 체크포인트 .gauge.pkl 10개만 있음(이미 전송된 것)
```

문서 요지: `paper/RACEv2_draft.md`(초록, 모델 §2, 학습 §3, 결과 §4, 절제 분석 §5, LAMMPS §6, 한계 §7, 보고 문구 §8; TODO/CHECK/PENDING 표시 남음),
`stream_loader_design.md`(OAM 대규모 학습용 pickle-free 스트리밍 로더; 6 Å 샤드 vs 5 Å, 1 노드에서 OMat24 1 에폭 ≈ 39 h → 중간 재개 필요),
`SOAP_Laplace.md`(SOAP 곡률로 전 가중치 Laplace 사후분포, 미실행), `torch_port_scope.md`(PyTorch RACEv2는 없음, 포팅 10–15 인일, PJRT halo 유지와 비교 판단 필요),
`LOCAL_TEST_PROMPT.md`(GPU 없는 로컬 PC에서 RACE_LR CPU 테스트와 4G-HDNNP 데이터 변환용 프롬프트).

## 4. 검증 안 된 코드와 다음 작업 순서 (HANDOFF §7–8 기준)

**검증 안 된 코드(작성만 하고 실행하지 않음)**

| 항목 | 상태 | 먼저 돌릴 검증 작업 |
|---|---|---|
| RACE_LR (Qeq + 장거리 정전기) | 코드·단위테스트 작성, 미실행 | `runs/diag/race_lr_check.sh` (GPU 1장 약 5분) |
| LAMMPS halo 저장 잔차 VJP / owned-row bucket | 컴파일만, 실행 안 함 | `runs/diag/lammps_halo_res.sh` |
| 스트리밍 로더 | JAX 없는 테스트 4/4 통과(로그인 노드), 그래프·로더 동등성 테스트는 skip | `runs/diag/stream_check.sh` (CPU; aselmdb에는 `pip install lmdb` 필요) |
| `stream` 데이터 형식의 `train_3bpa` 통합(2c') | 미착수 | — |
| SOAP-Laplace | 이론 + 골격, 미실행 | `runs/diag/soap_laplace.sh` |
| last-layer Bayes (`bayes_lastlayer.py`) | 미실행 | `runs/diag/bayes_lastlayer.sh` (`MAX_TEST=200` 먼저) |
| 논문 초안 | 텍스트 | — (TODO/CHECK 채우기) |

**다음 작업 순서**
1. 할당 복구 후 검증 작업: stream_check → lammps_halo_res → bayes_lastlayer → soap_laplace (각각 사용자 승인, 작은 테스트 먼저).
2. 남은 절제(noshell, nouff, amsgrad; 3 seed × 500 ep), 일반 Adam과 lr 5e-3 민감도(비사전등록 표시).
3. 100k 크기 선별은 완료. 옵티마이저·정밀도 선별(opt100k: Adam, AMSGrad float32, seed 43), V1 100k 비교.
4. RACE_LR: 단위테스트 → 4G-HDNNP 벤치마크 → SPICE/OMol25와 LAMMPS 장거리 결합.
5. 스트리밍 로더 통합(2c') 후 전체 OAM 학습(OMat24 사전학습 → MPtrj+sAlex 미세조정).
6. 논문 초안 TODO 채우기, 메커니즘 테스트(§5.4).
7. 이후: water/MD22, MD 관측량, 추론 속도 표, rMD17 ~99k 평가.

(HANDOFF의 sbatch 예시는 이전 계정 환경 기준이다. 이 계정의 `--comment` 값과 파티션은 §5에서 확인한다.)

## 5. 이 계정에서 시작하려면

확인 결과(2026-11-10, glogin03, aarch64, 계정 t510a13): conda 환경 `bam_cue` **없음**(`~/.conda` 자체가 없음). 모듈 `miniconda3/py314_26.7.1-1`은 **있음**.
`lfs quota`: /scratch 사용 17.54 GB, 쿼터 제한 없음(기본값).

**환경 재구성 계획(설치는 아직 안 함)**
1. `module load miniconda3/py314_26.7.1-1`; 환경은 홈 쿼터를 피하려고 `/scratch/t510a13` 아래에 만든다:
   `conda create -p /scratch/t510a13/envs/bam_cue python=3.12`(파이썬 3.12.15 기록; 마이너 버전은 가장 가까운 3.12.x).
2. GPU 노드가 aarch64이므로 **aarch64**에서 `pip install -r env/pip_freeze_bam_cue.txt`(jax/jaxlib/jax-cuda12-plugin 0.12.0.1,
   cuequivariance 0.13.0 + ops-jax-cu12, flax 0.12.10, optax 0.2.8, e3nn-jax 0.21.0, ase 3.29.0, numpy 2.5.4, scipy 1.18.1 등 81줄).
   설치는 소형 파일이 매우 많아 로그인 노드에 부담이 되므로, Slurm 작업(제출 전 승인 필요)이나 소규모로 나누어 사용자와 협의해 진행.
3. 레포를 `pip install -e . --no-deps`.
4. 패치: site-packages에서 `patch -p0 --dry-run < env/cuequivariance_jax_dce_shim.patch` 로 먼저 확인 후 적용
   (대상 `cuequivariance_jax/segmented_polynomials/segmented_polynomial.py`, 끝 4줄: DCE 규칙 시그니처 shim). 로그인 노드에서는 JAX를 import하지 않는다.
5. 항상 `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`. 검증은 GPU 작업에서 `pytest tests`.
6. 스크립트(`runs/diag/*.sh` 등)가 `~/.conda/envs/bam_cue`를 가정하면 `/scratch/t510a13/envs/bam_cue`로 바꿔야 한다(먼저 grep으로 확인).

**할당·파티션 확인용 명령(사용자가 실행; `!` 접두어로 이 세션에서도 가능)**
```
sacctmgr show assoc user=$USER format=account,partition,qos,grpTRES%30 -P
sshare -U -u $USER            # 또는 KSC 포털의 SRU 잔량
showappl                      # --comment 의 field/appl 값
sinfo -s                      # 파티션 목록
sbatch --test-only --comment="field=<field>;appl=<appl>" -p <partition> --gres=gpu:1 -t 00:05:00 --wrap "hostname"   # 작업 생성 없이 검사
```
이전 계정은 SRU 소진으로 2026-11-06에 sbatch가 거부됐다(문의: account@ksc.re.kr). 이 계정의 할당 여부는 위 명령으로 확인해야 한다.

**데이터 경로 변경 계획(파일은 아직 수정 안 함)**
- 설정(369개)의 `/scratch/t589a02/` 접두어는 모두 `datasets_ext` 아래이고(937곳), `runs/diag/*.sh` 61개에도 t589a02 경로가 있다.
- `data_splits.tar.gz`는 `scratch/t589a02/...` 구조라, 경로 단축을 위해 풀 때 두 단계를 벗긴다(108개 항목, 약 1 GB; 소량이지만 Slurm 또는 dm 노드 권장):
  `mkdir -p /scratch/t510a13/data && tar -xzf data_splits.tar.gz -C /scratch/t510a13/data --strip-components=2`
  → `/scratch/t510a13/data/datasets_ext/...`, `/scratch/t510a13/data/oam_data/refs/`.
- 그 뒤 한 번에 치환(RESTORE.md의 sed 예시와 같은 방식, 범위를 `/scratch/t589a02/` 전체로 넓힘):
  `sed -i "s#/scratch/t589a02/#/scratch/t510a13/data/#g" input/*.json` 와 필요한 스크립트. 실행 전 `git diff --stat`으로 확인하고, 다른 용도의 t589a02 경로(예: `/home01/t589a02`, 소스 코드 경로)는 따로 본다.
- 패키지에 **없는** 데이터: md22, qm9(변환기 스크립트만 있음), OAM 원시 샤드(`oam_data/*_train|val`, MPtrj raw; refs 3개만 있음), LAMMPS 소스/빌드, 3BPA 외 `rmd17`·`water`·`si` 는 있음. 필요 시 `docs/DATASETS.md`의 다운로드/변환 절차로 재구성.
- 대안: `/scratch/t589a02` 심볼릭 링크는 다른 계정 영역이므로 만들지 않는다.

**체크포인트 rsync**
- manifests는 레포 상대 경로(`runs/<name>_cue/ckpt_*.gauge.pkl`)이며 core 10개(11.3 GB), all 125개(302.6 GB). 이미 BAM_V2/runs 아래에 10개 `.gauge.pkl`이 있다(core와 같은 이름으로 보이나 크기·무결성은 미확인).
- 원 계정의 /scratch는 15일 미접근 파일을 삭제하므로, 원본이 아직 남아 있는지는 **이전 계정 소유자(사용자)** 가 dm 노드에서 확인해 전송해야 한다. 이 세션은 이전 계정 디렉터리에 접근하지 않는다.
- 전송 명령 형태(소유자 쪽 dm 노드에서 실행): `rsync -av --files-from=<manifest> ./ <대상>:/scratch/t510a13/race_v2/BAM_V2/`

**압축 풀기(미실행)**: `runs_text.tar.gz`(2,451개 항목)와 `docs_fonts.tar.gz`(4개)는 레포 루트에서 풀면 된다. runs_text는 파일 수가 많지 않지만 `runs/` 디렉터리와 섞이므로 먼저 겹치는 파일이 없는지 `tar -tzf`로 확인한다.

## 6. 사용자에게 확인할 질문

1. 이 계정의 SRU 할당과 사용 가능한 파티션·`--comment` 값(field/appl)은 무엇인가? (위 확인 명령 결과를 알려 주면 된다.)
2. 체크포인트: 원 계정에서 core 10개만 옮길지, 125개 전체(302.6 GB)를 옮길지? 이미 와 있는 `.gauge.pkl` 10개는 core 그대로인가?
3. 환경 설치를 어디서·어떻게 할까(Slurm 작업으로 제출해도 되는지, `/scratch/t510a13/envs`에 두는 것으로 괜찮은지; `/scratch`는 15일 미접근 삭제 대상이므로 환경 보관 방식)?
4. 데이터: `data_splits.tar.gz`를 `/scratch/t510a13/data`에 풀고 `input/*.json`·`runs/diag/*.sh` 경로를 sed로 바꿔도 되는가? 변경분을 커밋할지, 로컬 변경으로만 둘지? md22/qm9/OAM 원시 샤드는 필요한가?
5. 첫 작업으로 무엇을 할까: HANDOFF §8의 검증 작업(stream_check부터)인가, 남은 절제인가, RACE_LR 검증인가? 방법론(수학·물리 기반) 쪽 우선순위가 있는가?
6. 이 계정에서 계속 `racev2` 브랜치에 커밋하고 push는 사용자가 하는 방식(원격 `origin` URL 재설정 포함) 그대로 가는가?
