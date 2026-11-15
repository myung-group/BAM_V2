# 데이터셋 사전처리 결과

스트림 학습(`data_format: "stream"`)에 쓰는 사전처리 결과 중 작은 파일만 패키지 안에 보관한다.
큰 크기표(데이터셋당 수백 MB 이상)는 `/appsdata/t510a13/prep/` 아래에 그대로 있다.

## 데이터셋별 현황 (2026-11-16)

| 폴더 | 데이터셋 | DFT 수준 | 구조 수(train / valid) | cutoff | 평균 이웃 수 | 최대 원자 / 간선 | E0 원소 수 | E0 잔차 (eV/원자) |
|---|---|---|---|---|---|---|---|---|
| `omol4m_c5` | OMol25 4M | ωB97M-V/def2-TZVPD | 3,936,920 / 49,834 | 5 | 23.2 | 350 / 17,880 | 83 | 0.420 |
| `omol4m_c6` | OMol25 4M | 같음 | 3,936,920 / 49,834 | 6 | 32.6 | 350 / 27,172 | (`omol4m_c5`의 E0 사용) | - |
| `opoly26_c6` | OPoly26 | ωB97M-V/def2-TZVPD | 5,902,827 / 201,865 | 6 | 40.7 | 360 / 23,858 | 25 | 0.037 |
| `oc25_c6` | OC25 | VASP RPBE+D3 | 7,395,512 / 203,630 | 6 | 작업 1284769가 끝나면 추가 | | | |
| `omol_full_c6` | OMol25 전체 | ωB97M-V/def2-TZVPD | 101,666,280 / 공식 val | 6 | 작업 1284787이 끝나면 추가 | | | |

- RACE v2 LR-POLAR 학습(내부 실행 이름 v3; OMol25 4M, cutoff 6)은 `omol4m_c6`의 크기표와 `omol4m_c5`의 E0를 썼다.
- E0는 DFT 수준마다 따로 쓴다. OMol25와 OPoly26은 수준이 같아 섞어 쓸 수 있고, OC25(RPBE+D3)와
  MPtrj/sAlex(PBE, PBE+U)는 각자의 E0와 출력 헤드가 필요하다.
- OPoly26의 E0는 파일당 앞 20,000행(합계 160만 행) 표본으로 맞췄다. OMol25 4M은 전체 행으로 맞췄다.
- 전하·스핀 레이블: OMol25 4M은 87행, OPoly26 표본은 1,048행만 Löwdin 전하가 없다. OC25에는 레이블이 없다.

## 폴더 안의 파일

| 파일 | 내용 |
|---|---|
| `e0.json` | `atomic_numbers`, `atom_energies`(eV): 원소별 기준 에너지. 총에너지를 원소 개수에 최소제곱으로 맞춘 값 |
| `meta.json` | E0와 함께 구조 수, 잔차, 평균 이웃 수, 전하 레이블 키, train/valid 경로. `runs/diag/make_omol_configs.py`가 읽는 형식 |
| `sizes_<split>_meta.json` | 크기표 요약: 구조 수, 원자·간선 합, 평균 이웃 수, 최대 원자·간선 수, 주기 구조 수, 파일 목록 |

`meta.json`의 `rule` 문구에는 "(official train split, 4M subset)"이라고 적혀 있는데, 스크립트의 고정 문구라서
OPoly26 등 다른 데이터셋에서도 그대로 나온다. 실제 입력은 `source.files`를 본다.

## 크기표 원본 위치

| 데이터셋 | 경로 |
|---|---|
| OMol25 4M (cutoff 5 / 6) | `/appsdata/t510a13/omol25/prep_4M`, `/appsdata/t510a13/omol25/prep_4M_c6` |
| OPoly26 | `/appsdata/t510a13/prep/opoly26_c6.0` (`sizes_train` 341 MB) |
| OC25 | `/appsdata/t510a13/prep/oc25_c6.0` |
| OMol25 전체 | `/appsdata/t510a13/prep/omol_full_c6.0` |

## 만드는 방법

`scripts/`에 만든 스크립트의 사본이 있다(원본은 `/scratch/t510a13/jobs`).

```bash
cd /scratch/t510a13
NAME=opoly26 TRAIN=/appsdata/t510a13/opoly26/train VALID=/appsdata/t510a13/opoly26/val \
  sbatch --exclude=node[0449-0484,0499-0576] jobs/dataset_prep.sh
```

- `dataset_prep.sh`: 크기표(`bam_omat24/stream_io.py sizes`, 64 프로세스)와 E0 적합(`omol4m_e0.py`)을 차례로 돈다.
- 메모리 주의: 크기표는 프로세스 하나가 aselmdb 파일 하나를 통째로 메모리에 올린다(`stream_io.py`의
  `_count_task`). 파일이 큰 OMol25 전체(파일당 평균 7 GB, 최대 10.5 GB, 80개)는 GPU 1개 몫(195 GB)으로는
  메모리 부족으로 죽는다(작업 1284770). `--gres=gpu:4`(780 GB)로 제출해야 한다.
- 걸린 시간: OPoly26 크기표 178초 + E0 51초. OC25는 주기 구조라 시험에서 원자당 약 12배 느렸다(전체는 약 2시간 예상).
