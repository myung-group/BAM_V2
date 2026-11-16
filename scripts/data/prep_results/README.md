# Dataset preprocessing results

Only the small files of the preprocessing for stream training (`data_format: "stream"`) are kept inside the package.
The large size tables (hundreds of MB or more per dataset) stay under `/appsdata/t510a13/prep/` on KISTI HANGANG.

## Status per dataset (2026-11-16)

| Folder | Dataset | DFT level | Structures (train / valid) | cutoff | Mean neighbours | Max atoms / edges | E0 elements | E0 residual (eV/atom) |
|---|---|---|---|---|---|---|---|---|
| `omol4m_c5` | OMol25 4M | ωB97M-V/def2-TZVPD | 3,936,920 / 49,834 | 5 | 23.2 | 350 / 17,880 | 83 | 0.420 |
| `omol4m_c6` | OMol25 4M | same | 3,936,920 / 49,834 | 6 | 32.6 | 350 / 27,172 | (uses the E0 of `omol4m_c5`) | - |
| `opoly26_c6` | OPoly26 | ωB97M-V/def2-TZVPD | 5,902,827 / 201,865 | 6 | 40.7 | 360 / 23,858 | 25 | 0.037 |
| `oc25_c6` | OC25 | VASP RPBE+D3 | 7,395,512 / 203,630 | 6 | 47.2 | 300 / 35,068 | 72 | 0.224 |
| `omol_full_c6` | OMol25 full | ωB97M-V/def2-TZVPD | 101,666,280 / 2,762,021 (official val) | 6 | 31.8 | 350 / 29,006 | 83 | 0.345 |

- The RACE v2 LR-POLAR training (internal run name v3; OMol25 4M, cutoff 6) used the size tables of `omol4m_c6` and the
  E0 of `omol4m_c5`.
- E0 is kept separately for each DFT level. OMol25 and OPoly26 share a level and can be mixed; OC25 (RPBE+D3) and
  MPtrj/sAlex (PBE, PBE+U) each need their own E0 and output head.
- The E0 of OPoly26 and of the full OMol25 was fitted on a sample of the first 20,000 rows per file (1.6 million rows
  in total). OMol25 4M was fitted on all rows.
- OC25 (job 1284769): all structures are periodic and there are no charge or spin labels. Size tables 8,832 s (20
  files, so only 20 processes work), E0 from a sample of 20,000 rows per file (400,000 rows).
- Full OMol25 (job 1284787): the number of structures equals the 101,666,280 of the dataset documentation. Size tables
  3,125 s, peak memory 701 GB (`--gres=gpu:4`).
- Charge and spin labels: Löwdin charges are missing in only 87 rows of OMol25 4M and 1,048 rows of the OPoly26 sample.
  OC25 has no labels.

## Files in each folder

| File | Content |
|---|---|
| `e0.json` | `atomic_numbers`, `atom_energies` (eV): reference energy per element, a least-squares fit of the total energies to the element counts |
| `meta.json` | E0 together with the number of structures, the residual, the mean number of neighbours, the charge-label key and the train/valid paths. This is the format read by `scripts/train/make_omol_configs.py` |
| `sizes_<split>_meta.json` | summary of a size table: number of structures, sums of atoms and edges, mean number of neighbours, maximum atoms and edges, number of periodic structures, file list |

The `rule` text of `meta.json` says "(official train split, 4M subset)". It is a fixed string of the script, so it
appears for OPoly26 and the other datasets too. The actual input is in `source.files`.

## Where the full size tables are

| Dataset | Path |
|---|---|
| OMol25 4M (cutoff 5 / 6) | `/appsdata/t510a13/omol25/prep_4M`, `/appsdata/t510a13/omol25/prep_4M_c6` |
| OPoly26 | `/appsdata/t510a13/prep/opoly26_c6.0` (`sizes_train` 341 MB) |
| OC25 | `/appsdata/t510a13/prep/oc25_c6.0` |
| Full OMol25 | `/appsdata/t510a13/prep/omol_full_c6.0` |

## How they are made

Copies of the scripts are in `scripts/` (the originals are in `/scratch/t510a13/jobs`).

```bash
cd /scratch/t510a13
NAME=opoly26 TRAIN=/appsdata/t510a13/opoly26/train VALID=/appsdata/t510a13/opoly26/val \
  sbatch --exclude=node[0449-0484,0499-0576] jobs/dataset_prep.sh
```

- `dataset_prep.sh` runs the size tables (`race_v2/data/stream_io.py sizes`, 64 processes) and then the E0 fit
  (`omol4m_e0.py`).
- Memory: for the size tables one process loads a whole aselmdb file into memory (`_count_task` in `stream_io.py`).
  The full OMol25 (7 GB per file on average, 10.5 GB at most, 80 files) runs out of memory with the share of one GPU
  (195 GB; job 1284770). Submit it with `--gres=gpu:4` (780 GB).
- Time: OPoly26 size tables 178 s + E0 51 s. Full OMol25 55 minutes. OC25 is periodic, about 12 times slower per
  atom: 2 hours 31 minutes.
