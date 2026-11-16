# RACEv2 (BAM_V2) dataset provenance and re-preprocessing recipe

> **Note (2026-11-16).** This document was written during development. Paths under `research/` and the documents
> `docs/HANDOFF*.md` and `docs/local_test_report.md` it mentions were removed from `main`; they are on the `racev2`
> branch (`research/diag` = `runs/diag`, `research/input` = `input/` there). `<DATA_ROOT>`, `<HOME>` and
> `<scratchpad>` stand for directories of the machine the work was done on.

Compiled 2026-11-06 from the Claude Code session transcript
`<session transcript>`
(tool calls with their exact commands and results; timestamps below are UTC as in the transcript, file
mtimes are KST = UTC+9), plus the converter scripts and READMEs on disk. Data root on HANGANG:
`<DATA_ROOT>/datasets_ext` (below: `$DS`). Repo: `<DATA_ROOT>/BAM_V2`.

> **In this repository:** the converter scripts `convert_*.py` named below are copied to `scripts/data/` (on HANGANG they
> live in `$DS = <DATA_ROOT>/datasets_ext`). The processed splits themselves are in the transfer package
> (`data_splits.tar.gz`, see `docs/HANDOFF.md`); rebuilding from the raw sources is only needed if that package is not
> available, or to check it. The OAM subsets cannot be rebuilt without the BAM-GAMP shards (section 12), so copy the
> subset files instead.

Conventions in this file:

- Code blocks marked **verbatim** are copied character for character from the transcript (tool-call input).
- **NOT FOUND in transcript** marks things the record does not contain. Nothing there is guessed.
- "md5 (recorded)" = printed in the transcript at download time. "md5 (computed 2026-11-06)" = computed from the
  file on disk when this file was written (not in the transcript). Output md5s are listed in section 14 so a
  re-run can be checked byte for byte.
- Paths are hard-coded to `<DATA_ROOT>/...` in most commands. On another server, replace them with your own
  data root. `convert_md22.py` is the only converter that finds its data relative to its own location.
- `*_omat24_cutoff_*_e0_*.pkl` files next to the xyz files are graph caches the trainer writes
  (`race_v2/data/xyz_graphs.py`). They are not data and can be deleted; they are rebuilt on first use.

## 0. Summary

| dataset dir | raw source | converter | reproducible from the record? |
|---|---|---|---|
| repo `dataset_3BPA/` | in git (initial commit 016cd2d); md5-identical to BOTNet-datasets `dataset_3BPA/` | none (used as is) | yes |
| `BOTNet-datasets/dataset_3BPA/test_dih.xyz` | git clone github.com/davkovacs/BOTNet-datasets | none | yes |
| `BOTNet-datasets/dataset_acac` | same clone | none | yes |
| `rmd17_official` | figshare 10.6084/m9.figshare.12672038.v4 (npz + index csv, md5 recorded) | `convert_rmd17_figshare.py ... 01` | yes |
| `rmd17_xyz` (older, superseded) | ColabFit rMD17 on Hugging Face (`colabfit/rMD17`) | `convert_rmd17.py` (seed 1234) | yes (HF files may change; no md5 recorded) |
| `lips` | Materials Cloud record ycbvx-knj69 `lips.xyz` (md5 recorded) | `convert_lips.py` | yes |
| `md22` | quantum-machine.org/gdml/repo/datasets/md22_*.npz (no md5 recorded) | `convert_md22.py` | yes (md5 only computed now) |
| `qm9`, `qm9_{U,H,G,ZPVE,Cv}` | figshare files 3195389 + 3195404 (md5 recorded) | `convert_qm9.py`, `convert_qm9_props.py` | yes |
| `si_gap18` | Zenodo 1250555 (GAP-18 training set, md5 recorded) + Zenodo 7003068 (RSS / compression-MD tests) | inline python (verbatim below) | yes |
| `water_xyz` | git clone github.com/BingqingCheng/ab-initio-thermodynamics-of-water | inline python (verbatim below) | yes |
| `water_split1393` | same clone | inline python (verbatim below) | yes |
| `oam_subsets`, `oam_subsets_100k` | BAM-GAMP-jax shards in `<DATA_ROOT>/oam_data`, `<DATA_ROOT>/mp_data/jax_data_v3_raw` | `race_v2/data/oam_subset.py` | subsets: yes, given the shards. Shards: **partly** (converter and options are on disk; the raw OMat24/sAlex/MPtrj downloads and the MPtrj `jax_data_v3_raw` build are NOT FOUND in transcript) |
| `venv_tools` | `python -m venv` from conda env `bam_cue` + pip `pyarrow numpy scikit-learn` | n/a | yes |
| local `data_local/4ghdnnp/{carbon_chain,ag_cluster,nacl,aumgo}` | Materials Cloud 2020.137 (10.24435/materialscloud:f3-yh) `datasets.tar.gz` (md5 recorded) | `scripts/data/convert_4ghdnnp.py` | yes (section 16) |
| `tm23_dl`, `zuo2020_dl` | Materials Cloud tcrks-ymp88; github materialsvirtuallab/mlearn | none (inspected only) | not used in results |

## 1. Environments used for the conversions

- **`bam_cue`** conda env (`~/.conda/envs/bam_cue`, activated with
  `module load miniconda3/py314_26.7.1-1; source $(conda info --base)/etc/profile.d/conda.sh; conda activate bam_cue`):
  Python 3.12.15, ase 3.29.0, numpy 2.5.4, matscipy 1.3.1 (versions read 2026-11-06). Used for every ASE-based step
  (water, Si, OAM subsets, MD22, QM9 props).
- **`$DS/venv_tools`**: a helper venv for conversions that need only numpy / pyarrow / scikit-learn. `pyvenv.cfg`:
  `home = <HOME>/.conda/envs/bam_cue/bin`, Python 3.12.15, `include-system-site-packages = false`.
  Created 2026-10-28 09:08 (verbatim):

```bash
module load miniconda3/py314_26.7.1-1 2>/dev/null; source $(conda info --base)/etc/profile.d/conda.sh; conda activate bam_cue; python -m venv <DATA_ROOT>/datasets_ext/venv_tools && <DATA_ROOT>/datasets_ext/venv_tools/bin/pip install -q pyarrow numpy 2>&1 | tail -2; <DATA_ROOT>/datasets_ext/venv_tools/bin/python -c "import pyarrow, numpy; print(pyarrow.__version__, numpy.__version__)"
```
  (result: pyarrow 26.0.0, numpy 2.5.4). scikit-learn was added 2026-10-30 04:53 for LiPS (verbatim):

```bash
cd <DATA_ROOT>/datasets_ext && venv_tools/bin/pip install -q scikit-learn 2>&1 | tail -2; venv_tools/bin/python -I -c "import sklearn; print(sklearn.__version__)"
```
  (result: scikit-learn 1.9.1). The venv also holds scipy 1.18.1, joblib, threadpoolctl (sklearn deps),
  cloudpickle 3.1.2 and narwhals 2.28.0; the install of the last two is NOT FOUND in transcript.
  Used by: `convert_rmd17.py`, `convert_rmd17_figshare.py`, `convert_lips.py`, some analysis scripts.
- **`/usr/bin/python3`** (system Python 3.6.15, numpy 1.17.3): used for `convert_qm9.py` (`python3 -I convert_qm9.py ...`).
  The split uses `np.random.RandomState(42)`, whose stream is fixed across numpy versions.
- Reproducibility notes: `RandomState` (rMD17 ColabFit, water, QM9, MD22, LiPS learning curves) is stable across numpy
  versions. `np.random.default_rng` (Si split, OAM subsets) and `sklearn.train_test_split` (LiPS Fu split) have been
  stable in practice, but numpy does not formally guarantee the `Generator` stream; check the md5s in section 14.
  ASE writes the extxyz headers, so the byte-level md5 of ASE-written files can change with the ASE version even when
  the data are identical.

## 2. 3BPA (repo `dataset_3BPA/` and BOTNet `dataset_3BPA/test_dih.xyz`)

**Source.** The four training/test files are tracked in git in the repo (`git log -- dataset_3BPA` = `016cd2d Initial
commit`; the session started from them: "the data are included in the repository as `dataset_3BPA/*.xyz`"). Their origin before the repo
is NOT FOUND in transcript, but they are byte-identical to the BOTNet-datasets copies (md5 computed 2026-11-06):

| file | md5 (repo = BOTNet clone) |
|---|---|
| train_300K.xyz | f78c925a4dcd8997d3ac14b3b4fd6752 |
| test_300K.xyz | 2d8d0bd5183be87aeb263d84deb9a5a5 |
| test_600K.xyz | f25ad304df7e336918a65d5e84639174 |
| test_1200K.xyz | 9d4205b03aceb7edf7ca8e1a01f8e9d7 |
| BOTNet only: test_dih.xyz | 08dd7eccdae0f239ba0117346425ba60 |
| BOTNet only: iso_atoms.xyz | dcb566f6760236a728c9e14c049fc0b6 |
| BOTNet only: train_mixedT.xyz | 3e10b0941f209697c417f74437ea47b4 |

BOTNet-datasets: https://github.com/davkovacs/BOTNet-datasets, cloned commit
`29e6d467317e4b5967b7ea5cbee54de953fa0d45` (2022-08-30, "update ethanol dataset with N energy of revMD17"). Its README:
"datasets used for the paper The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials ... cite
https://arxiv.org/abs/2205.06643". `dataset_3BPA/README.md`: units Å, eV, eV/Å; keys `energy`, `forces`;
train_300K = 500 configs from 300 K MD. (The README names `test_dih_beta.xyz`, the file in the repo is `test_dih.xyz`.)

**Download (verbatim, 2026-10-28 06:40).** Note from the session: git works on the HANGANG login node, raw http did not.

```bash
mkdir -p <DATA_ROOT>/datasets_ext && cd <DATA_ROOT>/datasets_ext && timeout 300 git clone --depth 1 https://github.com/davkovacs/BOTNet-datasets 2>&1 | tail -2; ls BOTNet-datasets; ls BOTNet-datasets/dataset_3BPA BOTNet-datasets/dataset_acac 2>/dev/null; du -sh BOTNet-datasets 2>/dev/null; cd <DATA_ROOT>/BAM_V2
```
Result: `dataset_3BPA/ iso_atoms.xyz README.md test_1200K.xyz test_300K.xyz test_600K.xyz test_dih.xyz train_300K.xyz train_mixedT.xyz`, 44M total.

**Preprocessing.** None. Files are used as is. `test_dih.xyz`: 7047 frames × 27 atoms, header has `dihedrals="_JSON [a, b, g]"`.

**Use.**
- Training/test: every 3BPA config, e.g. the final recipe `research/input/input_3bpa_omat24_r37_uffw03_s4{2,3,4}_cue.json`
  (also `v9a_rp_*`, `r28_soap_*`, `abl3_*`, ~300 configs): `train_path dataset_3BPA/train_300K.xyz`,
  `test_paths` = test_300K / test_600K / test_1200K, `valid_frac 0.1`. Its `atom_energies` (H, C, N, O =
  −13.587222780835477, −1029.4889999855063, −1484.9814568572233, −2041.9816003861047 eV) are the `iso_atoms.xyz` energies.
- Dihedral slices: no training config; evaluated with `race_v2.eval.eval_sets` through `runs/sbatch_eval_sets.sh`, e.g.
  `sbatch runs/sbatch_eval_sets.sh dih "--group-key dihedrals --group-index 0" <run names> -- $DS/BOTNet-datasets/dataset_3BPA/test_dih.xyz`
  (`--group-index 1` = beta slices, tag `dihbeta`).

## 3. Acetylacetone (`BOTNet-datasets/dataset_acac`)

**Source / download.** Same BOTNet clone as section 2 (one `git clone`). README: ORCA 5.0, UHF PBE + D3,
def2-SVP; train_300K / train_600K = 500 decorrelated geometries from xTB MD; test_MD_300K / test_MD_600K;
test_H_transfer (NEB path, energy-only, 15 frames); test_dihedral (dihedral scan); isolated_atoms.xyz.
md5 (computed 2026-11-06): train_300K 7400a8ac2be5a9e4fff035f437b82221, train_600K e0c1c19010f1c271e09f4c10dd144fce,
test_MD_300K 6ab2486dd8beb23c9f93e41c7d15808e, test_MD_600K 16ff0cad2ffd67c0dce4b3e468a2db45,
test_H_transfer d65f3fd4774f259d35cbf8b6ccffeb39, test_dihedral dfd4afb920fbdfa04d70d69c9c01ee1a,
isolated_atoms 88727b896d5e05062468049ea5d093bb.

**Preprocessing.** None.

**Use.** `research/input/input_3bpa_omat24_acacf_uffw03_s4{2,3,4}_cue.json` (final), also `acac_soap_*`, `acac_v9a1500_*`, ...
(19 configs): train `train_300K.xyz` (`valid_frac 0.1`), tests `test_MD_300K.xyz`, `test_MD_600K.xyz`;
`atom_energies` = the `isolated_atoms.xyz` energies (H −13.568422178253735, C −1026.8538996116154, O −2037.796869412825 eV). `train_600K.xyz` is not used by any
config. `test_H_transfer.xyz` and `test_dihedral.xyz` were evaluated with `runs/sbatch_eval_sets.sh` (e.g. tag
`acac_extra`, and `--group-key dihedral_angle`).

## 4. rMD17, official split (`rmd17_official`, from `rmd17_figshare`)

**Source.** figshare article 12672038, DOI 10.6084/m9.figshare.12672038.v4 ("Revised MD17 dataset (rMD17)",
Christensen & von Lilienfeld). The 1.07 GB `rmd17.tar.bz2` was NOT used (see below); the per-molecule npz files and
the index csv files of the same article were downloaded instead. Files, download URLs and md5 (recorded, from
`$DS/rmd17_figshare/files.tsv`, all verified `OK` by `md5sum -c`):

```
readme.txt	https://ndownloader.figshare.com/files/24013628	29e6f250bb2d1c461363e24955b5be1e
rmd17_ethanol.npz	https://ndownloader.figshare.com/files/62265733	eb837fb8deb27d4e0d52f71a03aff776
rmd17_azobenzene.npz	https://ndownloader.figshare.com/files/62265754	be79df918468eb3579aa73a0becf7390
rmd17_toluene.npz	https://ndownloader.figshare.com/files/62265742	0f2913d51f8149c90ab28d697a076f64
rmd17_aspirin.npz	https://ndownloader.figshare.com/files/62265757	17fd6fb69066888613f7e16b358a7553
rmd17_benzene.npz	https://ndownloader.figshare.com/files/62265739	18c9242bc90fbf28215f6dd81e650f16
rmd17_naphthalene.npz	https://ndownloader.figshare.com/files/62265751	0efba19c9907e3852318b1e6008b3b9e
rmd17_malonaldehyde.npz	https://ndownloader.figshare.com/files/62265736	cdc1d70c0c34062ddde6e5071eb6fe21
rmd17_paracetamol.npz	https://ndownloader.figshare.com/files/62265760	ba10784f7b67635427085f6d7ec2dd97
rmd17_salicylic.npz	https://ndownloader.figshare.com/files/62265748	900fedf242da438400fa4293348d7dd1
rmd17_uracil.npz	https://ndownloader.figshare.com/files/62265745	992a4479c28a07e0cce6da964805be31
splits.zip	https://ndownloader.figshare.com/files/62265763	242c8ca7006b708d6a21c81225c13867
index_train_04.csv	https://ndownloader.figshare.com/files/62265766	3764257abf8121a0905f364468ba83da
index_test_02.csv	https://ndownloader.figshare.com/files/62265769	9dc098724c1680daca038b785ee4f602
index_train_02.csv	https://ndownloader.figshare.com/files/62265772	28dbc9a3d5e7c47e653b1cc01809df35
index_train_03.csv	https://ndownloader.figshare.com/files/62265775	8de68b1b288c790e633c9d05ff5f04dd
index_train_05.csv	https://ndownloader.figshare.com/files/62265778	4670362bd554c9c22581c936432236b3
index_test_01.csv	https://ndownloader.figshare.com/files/62265781	124c8a0849e84e244d6ce8424e1c05cf
index_test_05.csv	https://ndownloader.figshare.com/files/62265784	4892da12e88a3c30c9a15f1f95e6eaac
index_test_03.csv	https://ndownloader.figshare.com/files/62265787	fd96f4623291dc23dd80b24dd634ca0b
index_test_04.csv	https://ndownloader.figshare.com/files/62265790	f0acce759fa4fe46b3a0b2a450e7e260
index_train_01.csv	https://ndownloader.figshare.com/files/62265793	22264dcf2e42d6133362411abbbe4ce8
```

`readme.txt` (figshare): npz keys `nuclear_charges`, `coords` (Å), `energies` (kcal/mol), `forces` (kcal/mol/Å),
`old_indices`, `old_energies`, `old_forces`; azobenzene has 99,988 samples; "Five training and test splits are saved in
CSV format". It warns not to train on more than 1000 samples.

**Download (verbatim, 2026-10-29 04:11).** Script written to the scratchpad (since deleted; text below is the full
script):

```bash
D=<DATA_ROOT>/datasets_ext/rmd17_figshare; mkdir -p $D && cd $D && ls -A | head; 
cat > <scratchpad>/dl_rmd17.sh <<'EOF'
set -e
D=<DATA_ROOT>/datasets_ext/rmd17_figshare; cd $D
timeout 60 curl -sS -L "https://api.figshare.com/v2/articles/12672038" -o article.json
python3 - <<'PY' > files.tsv
import json; d=json.load(open("article.json"))
for f in d["files"]:
    if f["name"] != "rmd17.tar.bz2": print(f["name"], f["download_url"], f["computed_md5"], sep="\t")
PY
while IFS=$'\t' read -r name url md5; do
  for t in 1 2 3; do
    curl -sS -L --retry 3 -o "$name" "$url" && [ "$(md5sum "$name" | cut -d' ' -f1)" = "$md5" ] && { echo "OK $name"; break; } || echo "retry $name ($t)"
  done
done < files.tsv
md5sum -c <(awk -F'\t' '{print $3"  "$1}' files.tsv)
EOF
echo written
```
run as (verbatim):
```bash
bash <scratchpad>/dl_rmd17.sh > <scratchpad>/dl_rmd17.log 2>&1
```

Failed earlier attempt (kept for the record): `curl -sL -o rmd17.tar.bz2 https://figshare.com/ndownloader/files/23950376`
gave a 0-byte `$DS/rmd17.tar.bz2` (figshare answered HTTP 202). That empty file is still in `$DS` and can be deleted.

**Conversion.** `$DS/convert_rmd17_figshare.py` (file on disk, written 2026-10-29 04:12; one later edit with
`sed -i 's/energy={E\[i\]!r}/energy={float(E[i])!r}/'` so numpy-2 scalars print as plain floats; the version below is
the fixed one that produced the current files):

```python
"""Official rMD17 (figshare 10.6084/m9.figshare.12672038 v4) npz + split csv -> per-molecule extxyz.

usage: python -I convert_rmd17_figshare.py <figshare_dir> <out_dir> [split ...]   (split = 01..05, default 01)

Per molecule and split k writes
  train_split{k}.xyz     : the 1000 frames of index_train_{k}.csv (order of the csv)
  test_split{k}.xyz      : the 1000 frames of index_test_{k}.csv
  rest_split{k}.xyz      : all 99,000 frames not in index_train_{k}.csv
Energies kcal/mol -> eV, forces kcal/mol/A -> eV/A (1 kcal/mol = 4.184 / 96.48533212 eV).
The npz files are loaded with allow_pickle=False (data only).
"""
import os, sys
import numpy as np

KCAL = 4.184 / 96.48533212          # eV per kcal/mol
SYM = {1: "H", 6: "C", 7: "N", 8: "O"}
MOLS = ["aspirin", "azobenzene", "benzene", "ethanol", "malonaldehyde",
        "naphthalene", "paracetamol", "salicylic", "toluene", "uracil"]


def read_index(path):
    idx = np.loadtxt(path, dtype=np.int64, ndmin=1)
    assert idx.ndim == 1 and len(np.unique(idx)) == len(idx), path
    return idx


def write(path, z, R, E, F, idx):
    sym = [SYM[int(a)] for a in z]
    with open(path, "w") as fo:
        for i in idx:
            fo.write(f"{len(z)}\n")
            fo.write(f'Properties=species:S:1:pos:R:3:forces:R:3 energy={float(E[i])!r} rmd17_index={int(i)} pbc="F F F"\n')
            for s, p, f in zip(sym, R[i], F[i]):
                fo.write(f"{s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {f[0]:.10f} {f[1]:.10f} {f[2]:.10f}\n")


def main():
    src, out = sys.argv[1], sys.argv[2]
    splits = sys.argv[3:] or ["01"]
    idx = {k: (read_index(f"{src}/index_train_{k}.csv"), read_index(f"{src}/index_test_{k}.csv")) for k in splits}
    for k, (tr, te) in idx.items():
        assert len(tr) == 1000 and len(te) == 1000 and not set(tr) & set(te), k
    for m in MOLS:
        d = np.load(f"{src}/rmd17_{m}.npz", allow_pickle=False)
        z, R = d["nuclear_charges"], d["coords"]
        E, F = d["energies"] * KCAL, d["forces"] * KCAL
        n = len(E)
        assert R.shape == (n, len(z), 3) and F.shape == R.shape, m
        os.makedirs(f"{out}/{m}", exist_ok=True)
        for k, (tr, te) in idx.items():
            assert tr.max() < n and te.max() < n, (m, k)
            rest = np.setdiff1d(np.arange(n), tr)
            write(f"{out}/{m}/train_split{k}.xyz", z, R, E, F, tr)
            write(f"{out}/{m}/test_split{k}.xyz", z, R, E, F, te)
            write(f"{out}/{m}/rest_split{k}.xyz", z, R, E, F, rest)
            print(f"{m} split {k}: n={n} natoms={len(z)} train {len(tr)} test {len(te)} rest {len(rest)} "
                  f"E range {E.min():.3f}..{E.max():.3f} eV  |F| max {np.abs(F).max():.3f} eV/A", flush=True)


if __name__ == "__main__":
    main()
```
Command (verbatim, 2026-10-29 04:28, the final run after the sed fix):
```bash
cd <DATA_ROOT>/datasets_ext && <DATA_ROOT>/datasets_ext/venv_tools/bin/python -I convert_rmd17_figshare.py <DATA_ROOT>/datasets_ext/rmd17_figshare <DATA_ROOT>/datasets_ext/rmd17_official 01
```
Split: official split 01 only (`index_train_01.csv` / `index_test_01.csv`, 1000 / 1000 rows, used in csv order).
Units: kcal/mol → eV with 4.184/96.48533212. No filtering. Output per molecule (10 molecules):
`rmd17_official/<mol>/{train_split01,test_split01,rest_split01}.xyz` (rest = 99,000 frames not in train; 98,988 for
azobenzene). Header field `rmd17_index` = row in the npz.

Checks recorded: ColabFit `MD17-index` equals the npz `old_indices` (dE ≈ 1e-4 eV, dpos 0, dF 2e-8 eV/Å), not the npz row.

**Use.** `research/input/input_3bpa_omat24_rmd17f_<mol>_uffw03_s4{2,3,4}_cue.json` (final), also `rmd17o_<mol>_*`,
`rmdS_<mol>_L5h*`, `ablA_*` (aspirin ablations). Train `train_split01.xyz` (`valid_frac 0.05`), test `test_split01.xyz`;
`rest_split01.xyz` is not used by any config.

## 5. rMD17 from ColabFit (`rmd17_xyz`, older, superseded) and other rMD17 copies

**Source.** Hugging Face dataset `colabfit/rMD17` (ColabFit DS_8rafgy0ly6bt_0, DOI 10.60732/682fe04a; README cites the
original link 10.6084/m9.figshare.12672038.v3). Energies eV, forces eV/Å as stored by ColabFit.

**Download (verbatim, 2026-10-28 09:07):**
```bash
mkdir -p <DATA_ROOT>/datasets_ext/rmd17_colabfit && cd <DATA_ROOT>/datasets_ext/rmd17_colabfit && for f in README.md ds.parquet cs/cs_0.parquet co/co_0.parquet co/co_1.parquet cs_co_map/cs_co_map_0.parquet; do mkdir -p $(dirname $f); timeout 600 curl -sL -o $f "https://huggingface.co/datasets/colabfit/rMD17/resolve/main/$f"; echo "$f $(stat -c %s $f)"; done; head -c 1500 README.md
```
Sizes: co_0 523,387,868 B, co_1 523,625,563 B, cs_co_map_0 10,419,419 B. md5 not recorded; computed 2026-11-06:
co/co_0.parquet acff44468508118937821ca09fd79192, co/co_1.parquet 1cec17687014b775fca35d7f553686aa,
cs/cs_0.parquet 420102292ca41e7152740e49d30974e6, cs_co_map/cs_co_map_0.parquet 327ff5c6a73bad5153c71b6b562ccce0.

**Conversion.** `$DS/convert_rmd17.py` (file on disk = the heredoc written 2026-10-28 09:09, unchanged since):
```python
"""ColabFit rMD17 parquet -> per-molecule extxyz: train_1000.xyz (seeded random 1000) and test_2000.xyz
(seeded random 2000 of the remaining frames).  Energies eV, forces eV/A (as stored by ColabFit)."""
import sys, json, numpy as np, pyarrow.parquet as pq
root = sys.argv[1]; out = sys.argv[2]
cs = {r["id"]: r["name"] for r in pq.read_table(f"{root}/cs/cs_0.parquet").to_pylist()}
cmap = {}
for r in pq.read_table(f"{root}/cs_co_map/cs_co_map_0.parquet").to_pylist():
    cmap[r["configuration_id"]] = cs[r["configuration_set_id"]]
cols = ["configuration_id", "energy", "atomic_forces", "positions", "atomic_numbers", "configuration_metadata"]
per = {m: [] for m in cs.values()}
for f in (f"{root}/co/co_0.parquet", f"{root}/co/co_1.parquet"):
    pf = pq.ParquetFile(f)
    for i in range(pf.num_row_groups):
        for r in pf.read_row_group(i, columns=cols).to_pylist():
            m = cmap.get(r["configuration_id"])
            if m is None: continue
            idx = json.loads(r["configuration_metadata"]).get("MD17-index", -1)
            per[m].append((idx, r["energy"], r["atomic_numbers"], r["positions"], r["atomic_forces"]))
SYM = {1: "H", 6: "C", 7: "N", 8: "O"}
def write(path, rows):
    with open(path, "w") as fo:
        for idx, e, z, pos, frc in rows:
            fo.write(f"{len(z)}\n")
            fo.write(f'Properties=species:S:1:pos:R:3:forces:R:3 energy={e!r} md17_index={idx} pbc="F F F"\n')
            for zi, p, fc in zip(z, pos, frc):
                fo.write(f"{SYM[zi]} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {fc[0]:.10f} {fc[1]:.10f} {fc[2]:.10f}\n")
for m, rows in per.items():
    rows.sort(key=lambda r: r[0])
    rng = np.random.RandomState(1234)
    perm = rng.permutation(len(rows))
    tr = [rows[i] for i in perm[:1000]]; te = [rows[i] for i in perm[1000:3000]]
    import os; os.makedirs(f"{out}/{m}", exist_ok=True)
    write(f"{out}/{m}/train_1000.xyz", tr); write(f"{out}/{m}/test_2000.xyz", te)
    nb = 0; zs = sorted(set(rows[0][2]))
    print(m, len(rows), "frames; natoms", len(rows[0][2]), "elements", zs)
```
Command (verbatim, from the same tool call):
```bash
cd <DATA_ROOT>/datasets_ext && OMP_NUM_THREADS=1 timeout 1800 venv_tools/bin/python convert_rmd17.py rmd17_colabfit rmd17_xyz; ls rmd17_xyz; du -sh rmd17_xyz
```
Split: per molecule, rows sorted by ColabFit `MD17-index` (= original-MD17 index), `RandomState(1234).permutation`,
first 1000 → `train_1000.xyz`, next 2000 → `test_2000.xyz`. Not the official rMD17 split (that is why it was replaced
by section 4 on 2026-10-29). Frame counts per molecule after ColabFit: 99,976–99,995 (ColabFit deduplication).

**Use.** Only early runs: `research/input/input_3bpa_omat24_rmd17_<mol>_v9a_s42_cue.json` (+ `_fn_`, `_xn_`, `_g1_`, `_mix_`
variants). Not used for reported numbers.

**Other rMD17 copies (not used in configs).**
- `$DS/rmd17_frodo`: unpacked from `<DATA_ROOT>/rmd17_from_frodo.tar.gz`, a tarball the user brought from another
  machine (`frodo:/home/gbsim`; FAENet-era ASE `.traj` files for 7 molecules, 1000/1000 frames, plus `splits/*.csv`). How the
  tarball was made: NOT FOUND in transcript (its `rmd17/README.md` describes it). Only used for cross-checks on 2026-10-28.
- `$DS/rmd17.tar.bz2`: 0-byte failed download (section 4).

## 6. LiPS (`lips`, from `lips_src/lips.xyz`)

**Source.** Materials Cloud Archive record `ycbvx-knj69` (the converter docstring calls it "Materials Cloud 2022.45"),
file `lips.xyz`: 25,001 frames × 83 atoms (Li, P, S), periodic. md5 (recorded) `16f91b4aca0b5759026f8bd62790e5c6`.

**Download (verbatim, 2026-10-30 04:33):**
```bash
mkdir -p <DATA_ROOT>/datasets_ext/lips_src && cd <DATA_ROOT>/datasets_ext/lips_src && curl -sL -o lips.xyz "https://archive.materialscloud.org/records/ycbvx-knj69/files/lips.xyz?download=1" && md5sum lips.xyz && grep -c Lattice lips.xyz && head -2 lips.xyz | cut -c1-400
```

**Conversion.** `$DS/convert_lips.py` (on disk, written 2026-10-30 04:54, unchanged):
```python
"""LiPS (Materials Cloud 2022.45 lips.xyz, md5 16f91b4a..., 25,001 frames, 83 atoms) -> fair-protocol splits.

usage: venv_tools/bin/python -I convert_lips.py <lips.xyz> <out_dir>

Protocol F (Fu et al., "Forces are not Enough", TMLR 2023; MDsim preprocessing/lips.py):
  pool, test = train_test_split(arange(25001), train_size=20000, test_size=5001, random_state=123)
  train, val = train_test_split(pool, train_size=19000, test_size=1000, random_state=123)
  -> fu_train.xyz (19000), fu_valid.xyz (1000), fu_test.xyz (5001)
Protocol L (NequIP Table VI / CAMP learning curve, train 10/100/1000/2500, valid 1000, test 5000):
  drawn from the Fu pool (train+val, 20000 frames) with RandomState(0); test = fu_test (5001, disjoint).
  -> lc_valid.xyz (1000, shared), lc{N}_train.xyz (nested: lc10 c lc100 c lc1000 c lc2500)
Frames are copied verbatim (text), so energies / forces / cell are untouched.
Writes splits.npz (all indices) and e0.json (mean energy per atom of each training set, eV/atom).
"""
import json, os, sys
import numpy as np
from sklearn.model_selection import train_test_split

NAT = 83


def read_frames(path):
    with open(path) as f:
        lines = f.readlines()
    per = NAT + 2
    assert len(lines) % per == 0, len(lines)
    frames = [lines[i:i + per] for i in range(0, len(lines), per)]
    for fr in frames:
        assert int(fr[0]) == NAT
    return frames


def energy(fr):
    for tok in fr[1].split():
        if tok.startswith("energy="):
            return float(tok.split("=")[1])
    raise ValueError(fr[1])


def write(path, frames, idx):
    with open(path, "w") as fo:
        for i in idx:
            fo.writelines(frames[i])


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    frames = read_frames(src)
    n = len(frames)
    assert n == 25001, n
    E = np.array([energy(fr) for fr in frames])

    pool, test = train_test_split(np.arange(n), train_size=20000, test_size=n - 20000, random_state=123)
    train, val = train_test_split(pool, train_size=19000, test_size=1000, random_state=123)
    assert len(set(train) | set(val) | set(test)) == n

    rng = np.random.RandomState(0)
    perm = rng.permutation(pool)
    lc_val = perm[:1000]
    lc = {N: perm[1000:1000 + N] for N in (10, 100, 1000, 2500)}

    sets = {"fu_train": train, "fu_valid": val, "fu_test": test, "lc_valid": lc_val,
            **{f"lc{N}_train": v for N, v in lc.items()}}
    for name, idx in sets.items():
        write(f"{out}/{name}.xyz", frames, idx)
    np.savez(f"{out}/splits.npz", **sets)
    e0 = {name: float(E[idx].mean() / NAT) for name, idx in sets.items() if name.endswith("train")}
    with open(f"{out}/e0.json", "w") as f:
        json.dump(e0, f, indent=1)
    for name, idx in sets.items():
        print(name, len(idx), idx[:5])
    print(json.dumps(e0, indent=1))


if __name__ == "__main__":
    main()
```
Command (verbatim, 2026-10-30 04:54):
```bash
cd <DATA_ROOT>/datasets_ext && venv_tools/bin/python -I convert_lips.py lips_src/lips.xyz lips && ls -la lips && grep -c Lattice lips/fu_test.xyz
```
Splits: protocol F (Fu et al. TMLR 2023 / MDsim): `train_test_split(arange(25001), train_size=20000, test_size=5001,
random_state=123)`, then the 20000 pool → 19000 train / 1000 valid (random_state=123). Protocol L (learning curve):
`RandomState(0).permutation(pool)`; first 1000 → `lc_valid`, next N → `lc{N}_train` (nested, N = 10, 100, 1000, 2500);
test = `fu_test` (5001). Frames copied as text (no unit change). Outputs: `fu_train.xyz` (19000), `fu_valid.xyz`
(1000), `fu_test.xyz` (5001), `lc_valid.xyz`, `lc10/100/1000/2500_train.xyz`, `splits.npz`, `e0.json` (mean E/atom of
each training set: fu_train −4.308618082853139 eV/atom).

`lips/md_reference.npz` (2026-11-05) is not a split: it is the reference RDF / Li diffusivity cache written by
`race_v2/eval/md_lips_analysis.py` on its first run (`--ref $DS/lips_src/lips.xyz`, `--ref-cache $DS/lips/md_reference.npz`;
RDF over all 25,001 reference frames, D from frames `2500::25`). Delete it to have it rebuilt.

**Use.** `research/input/input_3bpa_omat24_lipsF_fu_s4{2,3,4}_cue.json` (fu_train / fu_valid / fu_test; `atom_energies` =
e0.json value for every element, `e0_mode isolated`) and
`lipsF_lc{10,100,1000,2500}_s4{2,3,4}` (lc*_train / lc_valid / fu_test); also `lips_*_hill_{soap,adam}_s42`, `lipsS_*`.

## 7. MD22 (`md22`)

**Source.** sGDML repository, `http://www.quantum-machine.org/gdml/repo/datasets/md22_<mol>.npz` for the 7 molecules.
md5 not recorded; computed 2026-11-06 on the files in `$DS/md22`:

| file | bytes (recorded) | md5 (computed 2026-11-06) |
|---|---|---|
| md22_Ac-Ala3-NHMe.npz | 144406075 | f130a28fe97a49c4c0d535e58b7fb4d8 |
| md22_DHA.npz | 125691777 | cbb86fed19f6d2698f41be5c90f86047 |
| md22_stachyose.npz | 72115728 | 7b3798cec7d6729c50ae6b9d963cd817 |
| md22_AT-AT.npz | 47370353 | 1b73fd567f84c5bdd58a4c1f382c0b7b |
| md22_AT-AT-CG-CG.npz | 47108832 | 43261f58a24ca5a54aa6b8ab4947a8d2 |
| md22_buckyball-catcher.npz | 41652224 | d010c8c076e7f11d20e7477f193458ac |
| md22_double-walled_nanotube.npz | 85745429 | 5ca2b6fe68ebdf3b6e382727448424af |

**Download (verbatim, 2026-11-04 14:56; run on data-mover node dm01 over ssh):**
```bash
timeout 600 ssh -o BatchMode=yes dm01 'mkdir -p <DATA_ROOT>/datasets_ext/md22 && cd <DATA_ROOT>/datasets_ext/md22 && for m in Ac-Ala3-NHMe DHA stachyose AT-AT AT-AT-CG-CG buckyball-catcher double-walled_nanotube; do curl -sL --max-time 300 -o md22_$m.npz http://www.quantum-machine.org/gdml/repo/datasets/md22_$m.npz; echo "$m $(stat -c %s md22_$m.npz) $(head -c 2 md22_$m.npz | od -An -c | tr -d " ")"; done' 2>&1 | tail -8
```

**Conversion.** `$DS/convert_md22.py` (on disk, written 2026-11-04 15:00, unchanged):
```python
"""MD22 (Chmiela et al., Sci. Adv. 2023) npz -> extxyz train/test with the standard training-set sizes.

    python convert_md22.py [molecule ...]       (default: all seven)

Protocol (MD22 paper Table 1; MACE eval. paper arXiv 2305.14247 uses the same sizes):
  train N = 6000 Ac-Ala3-NHMe, 8000 DHA, 8000 stachyose, 3000 AT-AT, 2000 AT-AT-CG-CG,
            600 buckyball-catcher, 800 double-walled_nanotube; test = all remaining frames.
  Random split, numpy seed 42; validation (5 %) is drawn from the training set by the trainer (valid_frac).
Units: kcal/mol -> eV (x 0.0433641), kcal/mol/A -> eV/A. Energies are total energies.
Writes md22/<mol>/{train,test}.xyz and md22/<mol>/split.json (indices into the npz).
"""
import json
import sys
from pathlib import Path

import numpy as np

KCAL = 0.0433641153087705
NTRAIN = {"Ac-Ala3-NHMe": 6000, "DHA": 8000, "stachyose": 8000, "AT-AT": 3000, "AT-AT-CG-CG": 2000,
          "buckyball-catcher": 600, "double-walled_nanotube": 800}
SYM = {1: "H", 6: "C", 7: "N", 8: "O"}
ROOT = Path(__file__).resolve().parent / "md22"


def write_xyz(path, z, R, E, F, idx):
    with open(path, "w") as f:
        for k in idx:
            f.write(f"{len(z)}\n")
            f.write(f'Properties=species:S:1:pos:R:3:forces:R:3 energy={E[k]:.10f} md22_index={int(k)} pbc="F F F"\n')
            for a in range(len(z)):
                x, y, w = R[k, a]
                fx, fy, fz = F[k, a]
                f.write(f"{SYM[int(z[a])]} {x:.8f} {y:.8f} {w:.8f} {fx:.8f} {fy:.8f} {fz:.8f}\n")


def convert(mol):
    out = ROOT / mol
    if (out / "test.xyz").exists() and (out / "split.json").exists():
        print(mol, "already converted")
        return
    d = np.load(ROOT / f"md22_{mol}.npz")
    assert str(d["e_unit"]) == "kcal/mol" and str(d["r_unit"]) == "Ang", (d["e_unit"], d["r_unit"])
    z, R = d["z"], d["R"]
    E = d["E"].reshape(-1) * KCAL
    F = d["F"] * KCAL
    n = len(E)
    perm = np.random.RandomState(42).permutation(n)
    tr, te = np.sort(perm[:NTRAIN[mol]]), np.sort(perm[NTRAIN[mol]:])
    out.mkdir(parents=True, exist_ok=True)
    write_xyz(out / "train.xyz", z, R, E, F, tr)
    write_xyz(out / "test.xyz", z, R, E, F, te)
    json.dump({"n_frames": int(n), "n_atoms": int(len(z)), "train": tr.tolist(), "test": te.tolist(), "seed": 42,
               "source": f"md22_{mol}.npz", "theory": str(d["theory"])}, open(out / "split.json", "w"))
    print(f"{mol}: {n} frames x {len(z)} atoms -> train {len(tr)} / test {len(te)}")


if __name__ == "__main__":
    for m in (sys.argv[1:] or list(NTRAIN)):
        convert(m)
```
Run inside Slurm job `md22prep` (job 1215622, script `scripts/data/md22_prep.sh`, queued from `runs/queue_1103b.txt` as
`-J md22prep --gres=gpu:4 -t 02:00:00 scripts/data/md22_prep.sh`) with the `bam_cue` python:
`cd <DATA_ROOT>/datasets_ext && python convert_md22.py` (no arguments = all seven). Log
(`runs/slurm/md22prep_1215622.out`):
```
Ac-Ala3-NHMe: 85109 frames x 42 atoms -> train 6000 / test 79109
DHA: 69753 frames x 56 atoms -> train 8000 / test 61753
stachyose: 27272 frames x 87 atoms -> train 8000 / test 19272
AT-AT: 20001 frames x 60 atoms -> train 3000 / test 17001
AT-AT-CG-CG: 10153 frames x 118 atoms -> train 2000 / test 8153
buckyball-catcher: 6102 frames x 148 atoms -> train 600 / test 5502
double-walled_nanotube: 5032 frames x 370 atoms -> train 800 / test 4232
```
Split: `RandomState(42).permutation(n)`, first N_train (standard MD22 sizes) → train, rest → test, both sorted.
Validation = 5 % of train drawn by the trainer (`valid_frac`). Units kcal/mol → eV (× 0.0433641153087705). Outputs:
`md22/<mol>/{train,test}.xyz`, `md22/<mol>/split.json` (indices into the npz).

**Use.** `research/input/input_3bpa_omat24_md22_<short>_s4{2,3,4}_cue.json` with short = ala3, dha, stach, atat, atatcgcg,
bucky, dwnt (`md22T_<short>` = 2-epoch timing runs).

## 8. QM9 (`qm9`, `qm9_U`, `qm9_H`, `qm9_G`, `qm9_ZPVE`, `qm9_Cv`)

**Source.** QM9 figshare collection 10.6084/m9.figshare.c.978904: `dsgdb9nsd.xyz.tar.bz2` (figshare file 3195389) and
`uncharacterized.txt` (file 3195404). md5 (recorded): `ad1ebd51ee7f5b3a6e32e974e5d54012  dsgdb9nsd.xyz.tar.bz2`
(86,144,227 B), `a361887bacb427b8a0ce7903d92a53b4  uncharacterized.txt` (486,752 B).

**Download (verbatim, 2026-10-30 04:34):**
```bash
mkdir -p <DATA_ROOT>/datasets_ext/qm9_src && cd <DATA_ROOT>/datasets_ext/qm9_src && curl -sL -o dsgdb9nsd.xyz.tar.bz2 https://ndownloader.figshare.com/files/3195389 ; curl -sL -o uncharacterized.txt https://ndownloader.figshare.com/files/3195404 ; ls -la; file *; md5sum *
```
**Unpack (verbatim, 2026-10-30 11:37)** (133,885 files in `qm9_src/xyz/`; an earlier identical attempt at 04:34 was
rejected by the user and not run):
```bash
cd <DATA_ROOT>/datasets_ext/qm9_src && mkdir -p xyz && tar -xjf dsgdb9nsd.xyz.tar.bz2 -C xyz && ls xyz | wc -l && sed -n 1,12p uncharacterized.txt && tail -3 uncharacterized.txt && sed -n 1,4p xyz/dsgdb9nsd_000001.xyz
```

**U0 conversion.** `$DS/convert_qm9.py` (on disk, written 2026-10-30 11:40, unchanged):
```python
"""QM9 (figshare 10.6084/m9.figshare.c.978904: dsgdb9nsd.xyz.tar.bz2 + uncharacterized.txt) -> extxyz splits.

usage: python3 -I convert_qm9.py <xyz_dir> <uncharacterized.txt> <out_dir> [seed]

Standard protocol (DimeNet / PaiNN / Allegro / MACE tables): drop the 3,054 uncharacterized molecules
(130,831 left), random split 110,000 train / 10,000 valid / 10,831 test, one model per target.
Target here: U0 (internal energy at 0 K, Hartree -> eV).  QM9 has no forces: zero forces are written
and the configs use force_weight 0.  Fortran exponents "*^" in coordinates are converted.
"""
import glob, os, re, sys
import numpy as np

HARTREE = 27.211386245988  # eV


def parse(path):
    with open(path) as f:
        lines = f.read().splitlines()
    n = int(lines[0])
    props = lines[1].split()
    idx, u0 = int(props[1]), float(props[12]) * HARTREE
    atoms = []
    for ln in lines[2:2 + n]:
        t = ln.replace("*^", "e").split()
        atoms.append((t[0], float(t[1]), float(t[2]), float(t[3])))
    return idx, u0, atoms


def write(path, mols):
    with open(path, "w") as fo:
        for idx, u0, atoms in mols:
            fo.write(f"{len(atoms)}\n")
            fo.write(f'Properties=species:S:1:pos:R:3:forces:R:3 energy={u0!r} qm9_index={idx} pbc="F F F"\n')
            for s, x, y, z in atoms:
                fo.write(f"{s} {x:.10f} {y:.10f} {z:.10f} 0.0 0.0 0.0\n")


def main():
    src, unchar, out = sys.argv[1], sys.argv[2], sys.argv[3]
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    os.makedirs(out, exist_ok=True)
    bad = set()
    with open(unchar) as f:
        for ln in f:
            m = re.match(r"^\s+(\d+)\s+\S", ln)
            if m:
                bad.add(int(m.group(1)))
    assert len(bad) == 3054, len(bad)
    files = sorted(glob.glob(f"{src}/dsgdb9nsd_*.xyz"))
    assert len(files) == 133885, len(files)
    mols = [m for m in map(parse, files) if m[0] not in bad]
    assert len(mols) == 130831, len(mols)
    perm = np.random.RandomState(seed).permutation(len(mols))
    parts = {"train": perm[:110000], "valid": perm[110000:120000], "test": perm[120000:]}
    for name, p in parts.items():
        write(f"{out}/{name}.xyz", [mols[i] for i in p])
        print(name, len(p))
    np.savez(f"{out}/splits_seed{seed}.npz", **{k: np.array([mols[i][0] for i in v]) for k, v in parts.items()})


if __name__ == "__main__":
    main()
```
Command (verbatim, 2026-10-30 11:40; `python3` = system `/usr/bin/python3`):
```bash
cd <DATA_ROOT>/datasets_ext && ls qm9_src/xyz | grep -v dsgdb9nsd_ | head; python3 -I convert_qm9.py qm9_src/xyz qm9_src/uncharacterized.txt qm9 && ls -la qm9 && head -3 qm9/test.xyz && ~/.conda/envs/bam_cue/bin/python -I -c "
from ase.io import read; import numpy as np
fr=read('<DATA_ROOT>/datasets_ext/qm9/train.xyz',':')
print(len(fr), sorted(set(np.concatenate([a.numbers for a in fr]))), np.mean([len(a) for a in fr]))"
```
Split: drop the 3,054 uncharacterized molecules (130,831 left, files sorted by name), `RandomState(42).permutation`,
110,000 train / 10,000 valid / 10,831 test. U0 = column 12 of line 2 (Hartree → eV with 27.211386245988). Zero forces
written. Outputs `qm9/{train,valid,test}.xyz`, `qm9/splits_seed42.npz` (gdb indices per part).

**Other targets.** `$DS/convert_qm9_props.py` (on disk, written 2026-11-04 15:13, unchanged):
```python
"""QM9 extra targets (U, H, G, ZPVE, Cv) with exactly the same split as the U0 run.

usage: python3 -I convert_qm9_props.py <dsgdb9nsd.xyz.tar.bz2> <qm9 dir holding splits_seed42.npz> <out_root>

Reads the raw tarball as one stream (no unpacked file tree), keeps the molecules listed in the U0 split
file (uncharacterized ones are already excluded there) and writes <out_root>/qm9_<target>/{train,valid,test}.xyz
in the same order as the U0 files.  QM9 line 2: gdb idx A B C mu alpha homo lumo gap r2 zpve U0 U H G Cv.
Units: U, H, G, ZPVE Hartree -> eV; Cv kept in cal/(mol K) (stored in the "energy" field, so a logged
"meV" value for Cv is 1e3 x cal/(mol K)).  Zero forces are written (energy-only training, force_weight 0).
"""
import os
import sys
import tarfile

import numpy as np

HARTREE = 27.211386245988
TARGETS = {"U": (13, HARTREE), "H": (14, HARTREE), "G": (15, HARTREE), "ZPVE": (11, HARTREE), "Cv": (16, 1.0)}


def parse(text):
    lines = text.splitlines()
    n = int(lines[0])
    props = lines[1].split()
    atoms = []
    for ln in lines[2:2 + n]:
        t = ln.replace("*^", "e").split()
        atoms.append((t[0], float(t[1]), float(t[2]), float(t[3])))
    return int(props[1]), props, atoms


def main():
    tar_path, qm9_dir, out_root = sys.argv[1], sys.argv[2], sys.argv[3]
    split = np.load(os.path.join(qm9_dir, "splits_seed42.npz"))
    want = set(int(i) for k in ("train", "valid", "test") for i in split[k])
    mols = {}
    with tarfile.open(tar_path, "r:bz2") as tf:
        for m in tf:
            if not m.isfile():
                continue
            idx, props, atoms = parse(tf.extractfile(m).read().decode())
            if idx in want:
                mols[idx] = (props, atoms)
    assert len(mols) == 130831, len(mols)
    for tgt, (col, unit) in TARGETS.items():
        out = os.path.join(out_root, f"qm9_{tgt}")
        os.makedirs(out, exist_ok=True)
        for part in ("train", "valid", "test"):
            with open(os.path.join(out, f"{part}.xyz"), "w") as fo:
                for idx in split[part]:
                    props, atoms = mols[int(idx)]
                    val = float(props[col].replace("*^", "e")) * unit
                    fo.write(f"{len(atoms)}\n")
                    fo.write(f'Properties=species:S:1:pos:R:3:forces:R:3 energy={val!r} qm9_index={int(idx)} pbc="F F F"\n')
                    for s, x, y, z in atoms:
                        fo.write(f"{s} {x:.10f} {y:.10f} {z:.10f} 0.0 0.0 0.0\n")
        print(tgt, {p: len(split[p]) for p in ("train", "valid", "test")}, "->", out)


if __name__ == "__main__":
    main()
```
Run inside Slurm job `qm9props` (job 1215668, `scripts/data/qm9_props_prep.sh`, queued from `runs/queue_1103b.txt` as
`-J qm9props --gres=gpu:1 -t 00:40:00 scripts/data/qm9_props_prep.sh`). The job script (verbatim, written 2026-11-04 15:13):
```bash
#!/bin/bash
#SBATCH -p normal
#SBATCH --comment="field=sciai;appl=jax"
#SBATCH -o runs/slurm/%x_%j.out
# Build QM9 U/H/G/ZPVE/Cv extxyz files (same split as U0) from the raw tarball (single-stream read).
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
cd <DATA_ROOT>/datasets_ext
python -I convert_qm9_props.py qm9_src/dsgdb9nsd.xyz.tar.bz2 qm9 . | tee <DATA_ROOT>/BAM_V2/runs/diag/qm9_props_prep.log
```
Same molecules and order as the U0 files (read from `qm9/splits_seed42.npz`). U, H, G, ZPVE in eV; Cv kept in
cal/(mol K) in the `energy` field. Output: `qm9_<T>/{train,valid,test}.xyz` (110000 / 10000 / 10831).

**Use.** U0: `research/input/input_3bpa_omat24_qm9F_s42_cue.json` (final; also `qm9_u0_hill_soap_s42`, `qm9_u0f*`, `qm9_u0t*`).
Other targets: `qm9F_{U,H,G,ZPVE,Cv}_s42`. Configs set `force_weight 0`; `qm9_u0_hill_soap_s42` sets
`atom_energies` = Hartree × 27.211386245988 of [−0.500273, −37.846772, −54.583861, −75.064579, −99.718730] for
H, C, N, O, F with `e0_mode regression`.

## 9. Si GAP-18 (`si_gap18`)

**Training data source.** Zenodo record 1250555 ("libAtoms/silicon-testing-framework v1.0"), file
`libAtoms/silicon-testing-framework-v1.0.zip`, 229,608,485 B, md5 (recorded, Zenodo checksum and local md5sum)
`bfae586f13d7894c35e2994529307c2d`. The GAP-18 training set inside it is
`libAtoms-silicon-testing-framework-fc252cb/models/GAP/gp_iter6_sparse9k.xml.xyz` (47,228,275 B, md5 recorded
`764eca58591fe015cbd0b840f782ff55`; 2475 frames, 171,815 atoms). The same file is in the Cambridge Apollo record
(DOI 10.17863/CAM.65004) `Si_PRX_GAP.zip` (downloaded to `$DS/si_prx_gap_apollo`, inner file md5 identical; zip md5
computed 2026-11-06 `9bc2b0a324e83047cdfdf90f2171bcd7`). The pointer to this file came from the
`load-atoms` Si-GAP-18 yaml (github.com/jla-gardner/load-atoms, cloned commit e291ca63...).

Download (verbatim, 2026-10-30 00:41):
```bash
D=<DATA_ROOT>/datasets_ext/si_gap18_zenodo; mkdir -p $D; cd $D; timeout 60 curl -sS "https://zenodo.org/api/records/1250555" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['metadata']['title']); [print(f['key'], f['size'], f['checksum'], f['links']['self']) for f in d['files']]"
```
```bash
D=<DATA_ROOT>/datasets_ext/si_gap18_zenodo; cd $D && timeout 1200 curl -sSL --retry 3 -o stf-v1.0.zip "https://zenodo.org/api/records/1250555/files/libAtoms/silicon-testing-framework-v1.0.zip/content" && md5sum stf-v1.0.zip && unzip -q -o stf-v1.0.zip "*/models/GAP/gp_iter6_sparse9k.xml.xyz" && find . -name "gp_iter6_sparse9k.xml.xyz" -exec ls -la {} \;
```
Apollo copy (verbatim, 2026-10-30 04:12; only used for the md5 cross-check):
```bash
D=<DATA_ROOT>/datasets_ext/si_prx_gap_apollo; mkdir -p $D; cd $D; timeout 900 curl -sSL -o Si_PRX_GAP.zip -w "%{http_code} %{size_download}\n" "https://www.repository.cam.ac.uk/bitstreams/f17c105a-8466-4729-9fab-94ac03e48106/download"; file Si_PRX_GAP.zip; unzip -l Si_PRX_GAP.zip 2>/dev/null | head -30
```

**Test-set source (RSS, compression MD).** Zenodo record 7003068, files `README.txt` and `structural_data.tar.gz`
(174,929,212 B; md5 not recorded, computed 2026-11-06 `18d45c84dced4dec2aec762c512c24f9`). README: `test_set_RSS.xyz` =
"100 structures produced with buildcell used for numerical error evaluation, described in Fig. 3a";
`test_set_compressionMD.xyz` = "100 64-atom structures ... from 10 independent MD trajectories driven by GAP-18 of
compression to 20 GPa ... Fig. 3b". Extracted members md5 (computed 2026-11-06): test_set_RSS.xyz
e5a09c798cd3db5e320e347bc2e51ab4, test_set_compressionMD.xyz f172afc96cadc46187aea6fcfeffd11a.

Download and extraction (verbatim, 2026-10-30 04:13 and 04:17):
```bash
D=<DATA_ROOT>/datasets_ext/si_ext_tests; mkdir -p $D/rss $D/mqmd; cd $D/rss; timeout 60 curl -sSL -o README.txt "https://zenodo.org/api/records/7003068/files/README.txt/content"; cat README.txt; timeout 900 curl -sSL -o structural_data.tar.gz "https://zenodo.org/api/records/7003068/files/structural_data.tar.gz/content"; ls -la; tar -tzf structural_data.tar.gz | head -40
```
```bash
cd <DATA_ROOT>/datasets_ext/si_ext_tests/rss; tar -xzf structural_data.tar.gz test_set_RSS.xyz test_set_compressionMD.xyz
```

**Conversion 1: stratified and extrapolation splits (verbatim, 2026-10-30 00:49, `bam_cue` python, unquoted heredoc
so `$S`/`$O` expand):**
```bash
module load miniconda3/py314_26.7.1-1 >/dev/null 2>&1; source $(conda info --base)/etc/profile.d/conda.sh; conda activate bam_cue; S=<DATA_ROOT>/datasets_ext/si_gap18_zenodo; O=<DATA_ROOT>/datasets_ext/si_gap18; mkdir -p $O; cd <DATA_ROOT>/datasets_ext && python -I - <<EOF
import ase.io, numpy as np, collections
from ase.calculators.singlepoint import SinglePointCalculator
fr = ase.io.read("$S/libAtoms-silicon-testing-framework-fc252cb/models/GAP/gp_iter6_sparse9k.xml.xyz", index=":")
EXTRAP = {"crack_111_1-10", "crack_110_1-10", "screw_disloc", "111adatom", "decohesion", "surface_111_3x3_das"}
clean, iso = [], None
for a in fr:
    e = a.info.get("dft_energy", a.info.get("DFT_energy")); f = a.arrays.get("dft_force", a.arrays.get("DFT_force"))
    ct = a.info.get("config_type", "?")
    b = ase.Atoms(a.numbers, a.positions, cell=a.cell, pbc=True)
    if ct == "isolated_atom":
        iso = float(e); continue
    if f is None:          # 60 'sh' frames are energy-only
        continue
    b.calc = SinglePointCalculator(b, energy=float(e), forces=np.asarray(f, float)); b.info["config_type"] = ct
    clean.append(b)
rng = np.random.default_rng(42)
strat_tr, strat_te, ex_tr, ex_te = [], [], [], []
by = collections.defaultdict(list)
for b in clean: by[b.info["config_type"]].append(b)
for ct, L in by.items():
    idx = rng.permutation(len(L)); nte = max(1, round(0.1 * len(L)))
    for k, i in enumerate(idx):
        (strat_te if k < nte else strat_tr).append(L[i])
    (ex_te if ct in EXTRAP else ex_tr).extend(L)
for name, L in (("train_strat", strat_tr), ("test_strat", strat_te), ("train_extrap", ex_tr), ("test_extrap", ex_te)):
    ase.io.write(f"$O/{name}.xyz", L, format="extxyz")
    print(name, len(L), "frames", sum(len(x) for x in L), "atoms")
print("isolated-atom energy", iso, "eV")
EOF
```
Result: `train_strat 2224 frames 153344 atoms`, `test_strat 250 frames 18470 atoms`, `train_extrap 2393 frames 163308
atoms`, `test_extrap 81 frames 8506 atoms`, `isolated-atom energy -158.54496821 eV`.
Rules: energy key `dft_energy` (or `DFT_energy`), forces `dft_force` (or `DFT_force`); the `isolated_atom` frame is
removed (its energy is the E0); 60 `sh` frames without forces are dropped (2474 frames kept). Stratified: per
`config_type` (in order of first appearance), `default_rng(42).permutation`, `max(1, round(0.1 n))` → test.
Extrapolation: config types crack_111_1-10, crack_110_1-10, screw_disloc, 111adatom, decohesion, surface_111_3x3_das
→ test, rest → train. All frames written with `pbc=True` and `config_type` in info.

**Conversion 2: RSS / compression tests and the full training set (verbatim, 2026-10-30 04:22):**
```bash
module load miniconda3/py314_26.7.1-1 >/dev/null 2>&1; source $(conda info --base)/etc/profile.d/conda.sh; conda activate bam_cue; O=<DATA_ROOT>/datasets_ext/si_gap18; R=<DATA_ROOT>/datasets_ext/si_ext_tests/rss; cd <DATA_ROOT>/datasets_ext && python -I - <<EOF
import ase.io, numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
for src, dst in (("$R/test_set_RSS.xyz", "$O/test_rss.xyz"), ("$R/test_set_compressionMD.xyz", "$O/test_compressionMD.xyz")):
    out = []
    for a in ase.io.read(src, index=":"):
        b = ase.Atoms(a.numbers, a.positions, cell=a.cell, pbc=a.pbc)
        b.calc = SinglePointCalculator(b, energy=float(a.info["dft_energy"]), forces=np.asarray(a.arrays["dft_forces"], float))
        out.append(b)
    ase.io.write(dst, out, format="extxyz"); print(dst, len(out), "pbc", set(tuple(x.pbc) for x in out))
full = ase.io.read("$O/train_strat.xyz", index=":") + ase.io.read("$O/test_strat.xyz", index=":")
ase.io.write("$O/train_full.xyz", full, format="extxyz"); print("train_full", len(full))
EOF
```
Result: `test_rss.xyz` 100, `test_compressionMD.xyz` 100, `train_full.xyz` 2474 (= train_strat + test_strat, i.e. all
GAP-18 frames with forces, isolated atom excluded). Labels are the `dft_energy` / `dft_forces` fields (the files also
carry GAP-18 predictions, which are not copied).

**Use.** Final: `research/input/input_3bpa_omat24_siF_s4{2,3,4}_cue.json` (train_full; tests test_rss + test_compressionMD;
`atom_energies` [−158.54496821], `e0_mode isolated`), also `si_full_hill_*`, `siS_*`, `ablS_*`. Split studies:
`si_strat_hill_{soap,adam}_s42`, `si_extrap_hill_{soap,adam}_s42`.

`si_gap18/core` (27 MB, 2026-11-04) is a core dump from a crashed process, not data. Delete it.

## 10. Liquid water, random split (`water_xyz`)

**Source.** git clone https://github.com/BingqingCheng/ab-initio-thermodynamics-of-water (cloned commit
`7b8b1a2b4270cffcab1bfd11a8603767b4966396`, 2020-06-14), file `training-set/dataset_1593.xyz` (libatoms extxyz, 1593
frames × 192 atoms = 64 H2O; `TotEnergy` and `force` in atomic units, Hartree / Bohr; md5 computed 2026-11-06
`4f64ba5262e170fd5cbaf8370dfdbef0`). `training-set/NOTE`: "1593 bulk liquid water configurations + energy + forces".

**Download (verbatim, 2026-10-28 09:06; the same call also made the failed rMD17 tar download):**
```bash
cd <DATA_ROOT>/datasets_ext && (timeout 600 curl -sL -o rmd17.tar.bz2 https://figshare.com/ndownloader/files/23950376; echo "rmd17 curl exit $?"; ls -la rmd17.tar.bz2; file rmd17.tar.bz2) ; (timeout 300 git clone --depth 1 https://github.com/BingqingCheng/ab-initio-thermodynamics-of-water water_cheng 2>&1 | tail -1; ls water_cheng/training-set 2>/dev/null | head; du -sh water_cheng 2>/dev/null)
```

**Conversion to train.xyz / test.xyz (verbatim, 2026-10-28 09:13).** This is the run that produced the current files.
A first run one minute earlier (09:12) treated the values as eV/Å and was overwritten by this one.
```bash
cd <DATA_ROOT>/datasets_ext && module load miniconda3/py314_26.7.1-1 2>/dev/null; source $(conda info --base)/etc/profile.d/conda.sh; conda activate bam_cue; OMP_NUM_THREADS=1 python - <<'EOF'
import ase.io, numpy as np
from ase.units import Bohr, Hartree
from ase.calculators.singlepoint import SinglePointCalculator
from matscipy.neighbours import neighbour_list
fr = ase.io.read("water_cheng/training-set/dataset_1593.xyz", index=":")
# the libatom file is in atomic units (Bohr, Hartree, Hartree/Bohr): box 23.47 Bohr = 12.42 A for 64 H2O
out = []
for a in fr:
    e = float(a.info["TotEnergy"]) * Hartree; f = np.asarray(a.arrays["force"], float) * Hartree / Bohr
    b = ase.Atoms(numbers=a.numbers, positions=a.positions * Bohr, cell=np.asarray(a.cell) * Bohr, pbc=True)
    b.calc = SinglePointCalculator(b, energy=e, forces=f); out.append(b)
rng = np.random.RandomState(1234); perm = rng.permutation(len(out)); n_tr = int(round(0.9 * len(out)))
tr = [out[i] for i in perm[:n_tr]]; te = [out[i] for i in perm[n_tr:]]
ase.io.write("water_xyz/train.xyz", tr, format="extxyz"); ase.io.write("water_xyz/test.xyz", te, format="extxyz")
d = np.linalg.norm(tr[0].positions[tr[0].numbers==8][:,None]-tr[0].positions[tr[0].numbers==1][None],axis=-1)
nb = np.mean([len(neighbour_list("i", a, 5.0)) / len(a) for a in tr[:50]])
print("train", len(tr), "test", len(te), "cell A", out[0].cell.lengths().round(3), "density g/cm3", round(64*18.015/(6.022e23*out[0].get_volume()*1e-24),3),
      "min O-H A", round(d.min(),3), "avg_n_neighbors(5A)", round(float(nb),3), "E/H2O eV", round(np.mean([a.get_potential_energy()/64 for a in out]),3),
      "F rms eV/A", round(float(np.sqrt(np.mean(np.concatenate([a.get_forces().ravel() for a in tr[:100]])**2))),3))
EOF
```
Result: train 1434, test 159, cell 12.417 Å, density 1.0 g/cm³, avg_n_neighbors(5 Å) 47.889. Units: positions and
cell × `ase.units.Bohr`, energy × `ase.units.Hartree`, forces × Hartree/Bohr (ase 3.29.0 constants). Split:
`RandomState(1234).permutation(1593)`, first `round(0.9 × 1593)` = 1434 → train, 159 → test.

**test_noout.xyz (verbatim, 2026-11-01 12:49):** the test set without its single most compressed frame (index 138 of
test.xyz = original frame 383, min interatomic distance 0.703 Å). The `sed` lines at the start print a literature PDF
from the scratchpad and do not affect the data.
```bash
S=<scratchpad>; sed -n 1416,1432p $S/nemp.txt | cut -c1-130; sed -n 1036,1046p $S/nemp.txt | cut -c1-130; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1; ~/.conda/envs/bam_cue/bin/python -I -c "
from ase.io import read,write; import numpy as np, matscipy.neighbours as mn
fr=read('<DATA_ROOT>/datasets_ext/water_xyz/test.xyz',':')
d=[mn.neighbour_list('d',a,3.0).min() for a in fr]; i=np.argsort(d)[:4]; print(i,np.array(d)[i])
write('<DATA_ROOT>/datasets_ext/water_xyz/test_noout.xyz',[a for k,a in enumerate(fr) if k!=i[0]],format='extxyz'); print(len(fr)-1)" 2>&1 | tail -3
```
Result: indices `[138 90 28 123]`, distances `[0.703 0.740 0.752 0.756]`; 158 frames written.

**train_only.xyz / valid.xyz (verbatim, 2026-11-01 12:58):** the trainer's internal 90/10 split written to fixed files.
```bash
cd <DATA_ROOT>/datasets_ext/water_xyz; head -2 train.xyz | cut -c1-400; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1; ~/.conda/envs/bam_cue/bin/python -I - <<'EOF' 2>&1 | grep -v -i openblas
from ase.io import read, write; import numpy as np, matscipy.neighbours as mn
tr=read('<DATA_ROOT>/datasets_ext/water_xyz/train.xyz',':'); te=read('<DATA_ROOT>/datasets_ext/water_xyz/test.xyz',':')
def mind(a): return mn.neighbour_list('d',a,2.0).min()
dtr=np.array([mind(a) for a in tr]); dte=np.array([mind(a) for a in te])
print('train min-dist: <0.72:',(dtr<0.72).sum(),'<0.76:',(dtr<0.76).sum(),'<0.80:',(dtr<0.80).sum(),'min %.3f'%dtr.min(),'median %.3f'%np.median(dtr))
print('test  min-dist: <0.72:',(dte<0.72).sum(),'<0.76:',(dte<0.76).sum(),'<0.80:',(dte<0.80).sum())
vol=np.array([a.get_volume() for a in tr]); print('train volume A^3: min %.0f max %.0f; density g/cm3 range %.3f-%.3f'%(vol.min(),vol.max(),64*18.015/6.022e23/(vol.max()*1e-24),64*18.015/6.022e23/(vol.min()*1e-24)))
print('info keys', list(tr[0].info.keys()), 'cell', np.round(tr[0].cell.lengths(),2))
# reproduce the loader split: RandomState(42).permutation, n_train = int(N*0.9)
perm=np.random.RandomState(42).permutation(len(tr)); n=int(len(tr)*0.9)
write('<DATA_ROOT>/datasets_ext/water_xyz/train_only.xyz',[tr[i] for i in perm[:n]],format='extxyz')
write('<DATA_ROOT>/datasets_ext/water_xyz/valid.xyz',[tr[i] for i in perm[n:]],format='extxyz')
print('wrote train_only',n,'valid',len(tr)-n)
E=np.array([a.get_potential_energy() for a in tr]); print('corr(E, volume) = %.3f'%np.corrcoef(E,vol)[0,1])
EOF
```
Result: `train_only` 1290, `valid` 144 (`RandomState(42).permutation(1434)`, first `int(1434 × 0.9)` = 1290).

**Use.** Final: `research/input/input_3bpa_omat24_wFu1_s4{2,3,4}_cue.json` (train_only / valid / test + test_noout); also
`w3_*` … `w7_*`, `wF1_*`, `wS_*` (31 configs). Early runs `water_v9a_s42`, `water2_*` use `train.xyz` with
`valid_frac 0.1` (this is the split `train_only`/`valid` reproduce).

## 11. Liquid water, first-1393 / last-200 split (`water_split1393`)

**Source.** Same `dataset_1593.xyz` as section 10. Motivation recorded in the session: a Hugging Face card
(OneScience-Sugon/water) claims the MACE water split is first 1393 / last 200 frames (all PIMD); marked unverified
for MACE itself in the session notes.

**Conversion (verbatim, 2026-11-02 09:04).** This run produced the current files; an earlier run at 08:59 wrote the
same split with `b.info={'orig_index':k,'energy':E}` and a leftover `force` array and was overwritten by this one.
```bash
cd <DATA_ROOT>/BAM_V2; squeue -h -u <user> -n w8s1393 -o %i | xargs -r scancel; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1; ~/.conda/envs/bam_cue/bin/python -I - <<'EOF' 2>&1 | grep -v -i openblas
import numpy as np, os
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator
B=0.529177; HA=27.211386
orig=read('<DATA_ROOT>/datasets_ext/water_cheng/training-set/dataset_1593.xyz',':')
out=[]
for k,a in enumerate(orig):
    b=a.copy(); b.set_positions(a.positions*B); b.set_cell(a.cell*B,scale_atoms=False); b.pbc=True
    E=float(a.info['TotEnergy'])*HA; F=np.asarray(a.arrays['force'])*HA/B
    b.info={'orig_index':k}; b.arrays.pop('force',None); b.calc=SinglePointCalculator(b,energy=E,forces=F); out.append(b)
te=read('<DATA_ROOT>/datasets_ext/water_xyz/test.xyz',':'); t=te[138]
k=[i for i,b in enumerate(out) if np.allclose(b.positions,t.positions,atol=1e-4)][0]
print('check orig',k,'dE %.2e eV, dF max %.2e eV/A, cell diff %.1e'%(out[k].get_potential_energy()-t.get_potential_energy(), np.abs(out[k].get_forces()-t.get_forces()).max(), np.abs(out[k].cell.array-t.cell.array).max()))
D='<DATA_ROOT>/datasets_ext/water_split1393'; os.makedirs(D,exist_ok=True)
write(f'{D}/train_first1393.xyz',out[:1393],format='extxyz'); write(f'{D}/test_last200.xyz',out[1393:],format='extxyz')
r=read(f'{D}/test_last200.xyz',':'); print('written train 1393 / test',len(r),'; reread energy ok:',abs(r[0].get_potential_energy()-out[1393].get_potential_energy())<1e-6, 'forces present:', r[0].get_forces().shape)
EOF
ls -la <DATA_ROOT>/datasets_ext/water_split1393/
```
Result: `check orig 383 dE 2.69e-05 eV, dF max 6.52e-06 eV/A, cell diff 5.3e-06` (against water_xyz test frame 138);
`train_first1393.xyz` = original frames 0–1392, `test_last200.xyz` = frames 1393–1592; `info` holds `orig_index`.
Units here use B = 0.529177, HA = 27.211386 (rounded constants, not `ase.units`), hence the 2.7e-5 eV / frame
difference to `water_xyz`.

**Use.** Final: `research/input/input_3bpa_omat24_wFu2_s4{2,3,4}_cue.json` (train_first1393, `valid_frac 0.05`, test
test_last200); also `wF2_*`, `w8_s1393_*`.

## 12. OAM subsets (`oam_subsets`, `oam_subsets_100k`)

**Subset script.** `race_v2/data/oam_subset.py` (repo, written 2026-11-05 13:59, committed in c91167e, unchanged since):
```python
"""Random subsets of the OAM training data (OMat24, MPtrj, sAlex) as extended XYZ for RACEv2 screening runs.

Reads the packed super-batch shards of BAM-GAMP-jax (``bam_jax.data.schema``: one pickle per shard, a list of
super-batches with positions, species index, cell, raw DFT energy, forces, stress) and writes
``<out>/<name>_{train,valid,test}.xyz`` plus ``<out>/<name>_meta.json``.

    python -m race_v2.data.oam_subset --name omat24 --train-root <DATA_ROOT>/oam_data/omat24_train \
        --eval-root <DATA_ROOT>/oam_data/omat24_val --atom-ref <DATA_ROOT>/oam_data/refs/omat24_train_ls.npz \
        --n-train 10000 --n-valid 1000 --n-test 1000 --out <DATA_ROOT>/datasets_ext/oam_subsets

Conventions of the shards (scripts/convert_aselmdb_to_shards.py): energy = raw total DFT energy (eV), forces eV/A,
stress eV/A^3 with ASE sign (positive = tensile) and ASE Voigt order (xx, yy, zz, yz, xz, xy).  Species are encoded
indices; ``species_encoding`` of the atom-reference npz maps Z -> index and is inverted here.

Sampling: shard files are shuffled (seed) over all shape buckets, and at most ``--per-shard`` graphs are drawn at
random from each opened shard, so the subset spans compositions, sizes and (for OMat24) the sub-datasets without
reading the whole set.  Train comes from the training split only; valid and test are disjoint draws from the
evaluation split (``--eval-root``).  The meta file records the element list, the per-element least-squares
reference energies of the full training split (``atom_ref``; used as E0) and the mean neighbour count at the
RACE cutoff.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np


def list_shards(root: Path) -> list[Path]:
    out = []
    for b in sorted(os.scandir(root), key=lambda e: e.name):
        if b.is_dir() and not b.name.startswith(('.', '_')):
            out += [Path(f.path) for f in os.scandir(b.path) if f.name.startswith('shard_') and f.name.endswith('.pkl')]
    return sorted(out)


def graphs_from_shard(path: Path, idx_to_z: np.ndarray):
    """Yield (numbers, positions, cell, energy, forces, stress) for every real graph of a shard."""
    with open(path, 'rb') as fh:
        d = pickle.load(fh)
    for sb in d['batches']:
        nm, gm = np.asarray(sb['node_mask']), np.asarray(sb['graph_mask'])
        batch = np.asarray(sb['batch'])
        for g in np.nonzero(gm)[0]:
            sel = nm & (batch == g)
            yield (idx_to_z[np.asarray(sb['species'])[sel]], np.asarray(sb['positions'], np.float64)[sel],
                   np.asarray(sb['cell'], np.float64)[g], float(sb['energy'][g]),
                   np.asarray(sb['forces'], np.float64)[sel], np.asarray(sb['stress'], np.float64)[g])


def draw(root: Path, n: int, per_shard: int, rng: np.random.Generator, idx_to_z: np.ndarray, skip=None):
    files = list_shards(root)
    rng.shuffle(files)
    out, used = [], []
    for f in files:
        if len(out) >= n:
            break
        gs = list(graphs_from_shard(f, idx_to_z))
        keep = rng.permutation(len(gs))
        k = 0
        for i in keep:
            if k >= per_shard or len(out) >= n:
                break
            key = (f.name, int(i))
            if skip is not None and key in skip:
                continue
            out.append(gs[i]); used.append(key); k += 1
    if len(out) < n:
        raise RuntimeError(f"{root}: only {len(out)} graphs drawn, {n} requested")
    return out, set(used)


def to_atoms(g):
    import ase
    from ase.calculators.singlepoint import SinglePointCalculator
    numbers, pos, cell, e, f, s = g
    a = ase.Atoms(numbers=numbers, positions=pos, cell=cell, pbc=True)
    a.calc = SinglePointCalculator(a, energy=e, forces=f, stress=s)
    return a


def mean_neighbors(atoms_list, cutoff: float, n_max: int = 500) -> float:
    from ase.neighborlist import neighbor_list
    counts = [len(neighbor_list('i', a, cutoff)) / len(a) for a in atoms_list[:n_max]]
    return float(np.mean(counts))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--name', required=True)
    ap.add_argument('--train-root', required=True, type=Path)
    ap.add_argument('--eval-root', required=True, type=Path)
    ap.add_argument('--atom-ref', required=True, type=Path)
    ap.add_argument('--n-train', type=int, default=10000)
    ap.add_argument('--n-valid', type=int, default=1000)
    ap.add_argument('--n-test', type=int, default=1000)
    ap.add_argument('--per-shard', type=int, default=50)
    ap.add_argument('--cutoff', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()

    import ase.io
    ref = np.load(args.atom_ref)
    enc = np.asarray(ref['species_encoding'])                 # Z -> index, -1 = not encoded
    zs = np.nonzero(enc >= 0)[0]
    idx_to_z = np.zeros(int(enc.max()) + 1, dtype=np.int64)
    idx_to_z[enc[zs]] = zs
    assert np.all(np.diff(enc[zs]) == 1) and enc[zs][0] == 0, "encoding is not Z-sorted"   # index order = sorted Z
    atom_ref = np.asarray(ref['atom_ref'], dtype=np.float64)

    rng = np.random.default_rng(args.seed)
    train, _ = draw(args.train_root, args.n_train, args.per_shard, rng, idx_to_z)
    ev, _ = draw(args.eval_root, args.n_valid + args.n_test, args.per_shard, rng, idx_to_z)
    ev = [ev[i] for i in rng.permutation(len(ev))]
    parts = {'train': train, 'valid': ev[:args.n_valid], 'test': ev[args.n_valid:]}

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {'name': args.name, 'train_root': str(args.train_root), 'eval_root': str(args.eval_root),
            'atom_ref_path': str(args.atom_ref), 'seed': args.seed, 'per_shard': args.per_shard,
            'atomic_numbers': [int(z) for z in zs], 'atom_energies': [float(x) for x in atom_ref],
            'units': 'energy eV (raw DFT), forces eV/A, stress eV/A^3 (ASE sign, Voigt xx yy zz yz xz xy)'}
    for k, gs in parts.items():
        atoms = [to_atoms(g) for g in gs]
        p = args.out / f'{args.name}_{k}.xyz'
        ase.io.write(p, atoms, format='extxyz')
        nat = np.array([len(a) for a in atoms])
        epa = np.array([a.get_potential_energy() / len(a) for a in atoms])
        meta[k] = {'path': str(p), 'n': len(atoms), 'n_atoms_mean': float(nat.mean()), 'n_atoms_max': int(nat.max()),
                   'E_per_atom_mean': float(epa.mean()),
                   'elements_present': sorted({int(z) for a in atoms for z in a.numbers})}
        if k == 'train':
            meta['avg_n_neighbors'] = mean_neighbors(atoms, args.cutoff)
        print(k, meta[k]['n'], 'atoms/graph', round(meta[k]['n_atoms_mean'], 1), 'E/atom', round(meta[k]['E_per_atom_mean'], 3),
              'elements', len(meta[k]['elements_present']), flush=True)
    print('avg_n_neighbors', round(meta['avg_n_neighbors'], 2))
    with open(args.out / f'{args.name}_meta.json', 'w') as fh:
        json.dump(meta, fh, indent=1)


if __name__ == '__main__':
    main()
```

**10k subsets** (`oam_subsets`: omat24 / mptrj / salex, 10000 train / 1000 valid / 1000 test each, seed 0,
per-shard 50, cutoff 5.0 for the neighbour count). Built in the Slurm job `research/diag/oam_smoke.sh` (written 2026-11-05
14:05; submitted as `sbatch -J oam_smoke -p normal --comment="field=sciai;appl=jax" --gres=gpu:3 -t 03:00:00
research/diag/oam_smoke.sh`, files dated 2026-11-05 23:16 KST). The data part of that script (verbatim):
```bash
module load miniconda3/py314_26.7.1-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bam_cue
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
OUT=<DATA_ROOT>/datasets_ext/oam_subsets
N_TRAIN=${N_TRAIN:-10000}
set -x
python -I race_v2/data/oam_subset.py --name omat24 --train-root <DATA_ROOT>/oam_data/omat24_train \
  --eval-root <DATA_ROOT>/oam_data/omat24_val --atom-ref <DATA_ROOT>/oam_data/refs/omat24_train_ls.npz \
  --n-train $N_TRAIN --n-valid 1000 --n-test 1000 --out $OUT > $OUT.omat24.log 2>&1 &
python -I race_v2/data/oam_subset.py --name mptrj --train-root <DATA_ROOT>/mp_data/jax_data_v3_raw/train \
  --eval-root <DATA_ROOT>/mp_data/jax_data_v3_raw/valid --atom-ref <DATA_ROOT>/oam_data/refs/mptrj_raw_ls.npz \
  --n-train $N_TRAIN --n-valid 1000 --n-test 1000 --out $OUT > $OUT.mptrj.log 2>&1 &
python -I race_v2/data/oam_subset.py --name salex --train-root <DATA_ROOT>/oam_data/salex_train \
  --eval-root <DATA_ROOT>/oam_data/salex_val --atom-ref <DATA_ROOT>/oam_data/refs/salex_train_ls.npz \
  --n-train $N_TRAIN --n-valid 1000 --n-test 1000 --out $OUT > $OUT.salex.log 2>&1 &
wait
```
Logs `$DS/oam_subsets.{omat24,mptrj,salex}.log`:
omat24 train 30.2 atoms/graph, E/atom −4.779, 84 elements, avg_n_neighbors 26.97; mptrj 41.1, −5.752, 86, 28.05;
salex 16.5, −5.303, 85, 33.7. Outputs `<name>_{train,valid,test}.xyz` and `<name>_meta.json` (89-element list and the
per-element least-squares reference energies of the full training split, used as `atom_energies`).

**100k subset** (`oam_subsets_100k`: OMat24 only, 100000 train / 2000 valid / 1000 test, seed 0, per-shard 50).
Built by `scripts/train/size_screen.sh` on its first run (job `szsmk`, 2026-11-05 22:08 UTC; files 2026-11-06 07:12 KST).
Data part (verbatim from the script on disk):
```bash
OUT=<DATA_ROOT>/datasets_ext/oam_subsets_100k
if [ ! -f $OUT/omat24_meta.json ]; then
  python -I race_v2/data/oam_subset.py --name omat24 --train-root <DATA_ROOT>/oam_data/omat24_train \
    --eval-root <DATA_ROOT>/oam_data/omat24_val --atom-ref <DATA_ROOT>/oam_data/refs/omat24_train_ls.npz \
    --n-train 100000 --n-valid 2000 --n-test 1000 --per-shard 50 --out $OUT 2>&1 | tail -n 5
fi
```
Its `avg_n_neighbors` (26.965) equals the 10k value, because both draw with seed 0 from the same shuffled shard list
and the count uses the first 500 training graphs.

**Use.** 10k: `research/input/input_3bpa_omat24_oam64_{omat24,mptrj,salex}_{v1,v2}_s42_cue.json` and `oam{c6,f50,spa,sw*,...}_*`
(15 configs per dataset; made by `scripts/train/make_oam_configs.py $DS/oam_subsets --tag <tag> ...`). 100k:
`sz100k_L{4,5,6}x{64,128,192}_omat24_v2_s42`, `szsmk_*` (made by `make_oam_configs.py` inside `size_screen.sh`).

**Where the shards come from.**
- OMat24 and sAlex shards (`<DATA_ROOT>/oam_data/{omat24,salex}_{train,val}`) were written by
  `~/prog/BAM-GAMP-jax/scripts/convert_aselmdb_to_shards.py` (converter version 2). Its options are pinned in each
  `convert_config.json` (read 2026-11-06):

  | shard dir | `src` (raw aselmdb) | files | key options |
  |---|---|---|---|
  | omat24_train | <DATA_ROOT>/oam_lmdb/omat24_train | 5300 | `--dataset omat24 --rows_per_task 10000` |
  | omat24_val | <DATA_ROOT>/oam_lmdb/omat24_val | 11 | `--dataset omat24 --rows_per_task 10000` |
  | salex_train | <DATA_ROOT>/oam_lmdb/salex_train/train | 350 | `--dataset salex --rows_per_task 20000` |
  | salex_val | <DATA_ROOT>/oam_lmdb/salex_val/val | 348 | `--dataset salex --rows_per_task 20000` |

  All four: cutoff 6.0, max_force 50 eV/Å, max_stress 80 GPa, max_energy_per_atom 0.0, min_energy_residual −5.0
  eV/atom with `scripts/data/energy_floor_ref_mptrj.json` (sha d8a71267a6b55795), max_atoms 512, seed 0,
  batches_per_shard 200, `atom_ref <DATA_ROOT>/mp_data/atom_ref.npz` (species_encoding sha bd91e2c24fdae3a5),
  shapes from `<DATA_ROOT>/mp_data/jax_data_v3/train/manifest.json` (sha 6ce85154f0bb081a). Energies are raw
  total DFT energies (eV), forces eV/Å, stress eV/Å³ with ASE sign and Voigt order (xx, yy, zz, yz, xz, xy).
  The script's docstring gives the run pattern (`srun ... python scripts/convert_aselmdb_to_shards.py --dataset omat24
  --src <omat24_train_dir> --dst <DATA_ROOT>/oam_data/omat24_train --rows_per_task 10000`). The actual runs
  (2026-10-29/30) are not in this transcript: **NOT FOUND in transcript** (the exact commands, job ids, and the
  `oam_data/refs/*_ls.npz` fits; the job script of 2026-11-05 points to "step d of the 2026-10-30 slot job (markers
  d_omat24_train, d_fit_omat24, ... in jobs_2026-10-30/markers/)", which belongs to another session or workflow).
- Raw OMat24 / sAlex aselmdb downloads into `<DATA_ROOT>/oam_lmdb`: **NOT FOUND in transcript** (no URL, no
  command). The transcript only shows the row format as verified "on the extracted OMat24 val, sAlex val and
  OMat24-1M files".
- MPtrj shards `<DATA_ROOT>/mp_data/jax_data_v3_raw/{train,valid}` and `oam_data/refs/mptrj_raw_ls.npz`:
  **NOT FOUND in transcript** (pre-existing on 2026-10-26; how they were made and from which MPtrj release is not
  recorded here).
- Reference energy files `oam_data/refs/{omat24_train,mptrj_raw,salex_train}_ls.npz` (keys `species_encoding`,
  `atom_ref`): producer NOT FOUND in transcript (the docstring of convert_aselmdb_to_shards.py mentions
  `fit_atom_ref.py`).

## 13. Directories not used for reported results

- **`tm23_dl`** (TM23, 2026-11-03): Materials Cloud record `tcrks-ymp88`; `README.md`, `files_description.md`, and
  `benchmarking_master_collection-20240316T202423Z-001.zip` saved as `tm23.zip` (936,628,742 B, md5 recorded
  `52b2c471591c8a8a88717f3fbdff6e37`), unzipped to `benchmarking_master_collection/`. Download (verbatim):
```bash
cd <DATA_ROOT>/datasets_ext/tm23_dl && curl -sL --max-time 60 -o README.md "https://archive.materialscloud.org/records/tcrks-ymp88/files/README.md?download=1" && curl -sL --max-time 60 -o files_description.md "https://archive.materialscloud.org/records/tcrks-ymp88/files/files_description.md?download=1" && cat README.md files_description.md; curl -sL --max-time 580 -o tm23.zip "https://archive.materialscloud.org/records/tcrks-ymp88/files/benchmarking_master_collection-20240316T202423Z-001.zip?download=1"; ls -la tm23.zip; md5sum tm23.zip
```
  `tm23_paper_tables.json` / `tm23_dataset_summary.json` are summaries the session wrote. TM23 was declined by the user.
- **`zuo2020_dl`** (Zuo et al. 2020 mlearn data, 2026-11-03): `training.json` and `test.json` for Cu, Ge, Li, Mo, Ni, Si
  from `https://raw.githubusercontent.com/materialsvirtuallab/mlearn/master/data/<el>/<f>` (verbatim):
```bash
cd <DATA_ROOT>/datasets_ext/zuo2020_dl && for el in Cu Ge Li Mo Ni Si; do mkdir -p $el; for f in training.json test.json; do curl -sL --max-time 120 -o $el/$f https://raw.githubusercontent.com/materialsvirtuallab/mlearn/master/data/$el/$f; done; done; ls -la Cu; ~/.conda/envs/bam_cue/bin/python -I - <<'EOF'
import json,glob,collections,numpy as np
for el in ['Ni','Cu','Li','Mo','Si','Ge']:
    out=[]
    for sp in ['training','test']:
        try: d=json.load(open(f'<DATA_ROOT>/datasets_ext/zuo2020_dl/{el}/{sp}.json'))
        except Exception as e: out.append(f"{sp}: ERR {e}"); continue
        n=len(d); nat=[len(x['structure']['sites']) for x in d]
        tags=collections.Counter(x.get('tag','?') for x in d)
        keys=list(d[0].keys()) if sp=='training' else None
        out.append(f"{sp}: {n} structures, atoms {min(nat)}-{max(nat)} (sum {sum(nat)}), tags {dict(tags)}" + (f" keys {keys}" if keys else ""))
    print(el, '|', ' | '.join(out))
EOF
```
  Inspected only.
- **`si_gap18_colabfit`**: Hugging Face `colabfit/Si_PRX_GAP`; only `README.md` arrived (parquet downloads failed:
  "Could not resolve host: us.aws.cdn.hf.co"). Unused.
- **`si_gap18_src`**: clones of `load-atoms` (to find the GAP-18 source file) and `XPOT` (to look for published Si
  splits). A clone of `libAtoms/silicon-testing-framework` timed out and left nothing. Unused as data.
- **`si_ext_tests/mqmd/phonons_gaps_si.zip`**: Zenodo 3924470 (`JaGeo/Phonons_GAPs_silicon-1.1.zip`), searched for an
  MD test set, none used.
- **`rmd17_frodo`**, **`rmd17.tar.bz2`**, **`rmd17_colabfit`** (only as source of `rmd17_xyz`): see section 5.
- **`qm9_src`**, **`lips_src`**, **`si_gap18_zenodo`**, **`si_prx_gap_apollo`**, **`si_ext_tests/rss`**,
  **`rmd17_figshare`**, **`water_cheng`**: raw sources of the sections above.

## 14. md5 of the produced files (computed 2026-11-06, for checking a re-run)

Paths are relative to `$DS`. Converters that copy text or format numbers themselves (rMD17, LiPS, QM9, MD22) should reproduce these bytes exactly.
ASE-written files (Si, water, OAM) can differ in bytes with another ASE version while holding the same numbers.

```
518687fe670a6ba5a013aed385a48ee4  rmd17_official/aspirin/train_split01.xyz
4128993c86f02e4f1a8304ba01e6e67e  rmd17_official/azobenzene/train_split01.xyz
15f0e18c793973f282f6ed34335cd700  rmd17_official/benzene/train_split01.xyz
4de7d36e2075364df7f616602791845e  rmd17_official/ethanol/train_split01.xyz
c046bb2005245762688794ad79525edd  rmd17_official/malonaldehyde/train_split01.xyz
9d7b0a258dac6727ab8c013f9e527a1e  rmd17_official/naphthalene/train_split01.xyz
e9cbe19d686ea7c72888332b94c9a0f6  rmd17_official/paracetamol/train_split01.xyz
bfb80627853436c5b2f0f993d1ce1bf9  rmd17_official/salicylic/train_split01.xyz
30c4a1d278225cf2fa8b2f823cfd176c  rmd17_official/toluene/train_split01.xyz
64c2b0eae073211f80c26630249be720  rmd17_official/uracil/train_split01.xyz
77ad770934580f8b1ec1bbbc54eba05b  rmd17_official/aspirin/test_split01.xyz
a12e84d5d7a571ca904939aa63b657ef  rmd17_official/azobenzene/test_split01.xyz
c144e535d3afb27793f5a0b953c2febf  rmd17_official/benzene/test_split01.xyz
e9a45d0e5886fb64bef91f4598ea62a2  rmd17_official/ethanol/test_split01.xyz
5b47938f2ec2a69c189d456a9223b11a  rmd17_official/malonaldehyde/test_split01.xyz
77f59b87a4764e315dd3786b431c221d  rmd17_official/naphthalene/test_split01.xyz
508b5d57b107fbecdd688a74f1b4bc10  rmd17_official/paracetamol/test_split01.xyz
3f4e860756dedaa663229690c172a658  rmd17_official/salicylic/test_split01.xyz
8dbb28cdb9be197fcbba4c3d2112778a  rmd17_official/toluene/test_split01.xyz
5c04f18e84580bf73ce7270def205294  rmd17_official/uracil/test_split01.xyz
67c13d0f8becc4daa80c9b4ead49272f  rmd17_xyz/aspirin/train_1000.xyz
49c61f976958bc36a88f3230419ca38d  rmd17_xyz/azobenzene/train_1000.xyz
41bb853fc0148223274fe94d07b0ed7a  rmd17_xyz/benzene/train_1000.xyz
623697fda4f9ad5b5a40312ab331531b  rmd17_xyz/ethanol/train_1000.xyz
4e7785d18c48e23bfbacc0fd81222bf6  rmd17_xyz/malonaldehyde/train_1000.xyz
baca7b31e577157fecdb55ea7c0dd944  rmd17_xyz/naphthalene/train_1000.xyz
ef7b2510fcec95cf10a6deac412b9592  rmd17_xyz/paracetamol/train_1000.xyz
76ea7cac8a4b1edcfe8bbcd103020969  rmd17_xyz/salicylic/train_1000.xyz
f8758ecffec9490446c14e71e6929bed  rmd17_xyz/toluene/train_1000.xyz
57b49f75cb4569fb0f9e1d21f18ec85d  rmd17_xyz/uracil/train_1000.xyz
4cbeaecb4f8abb2c2f84fbb0e71ef658  rmd17_xyz/aspirin/test_2000.xyz
a02a558fbb1e9a41e9951c03d2485cad  rmd17_xyz/azobenzene/test_2000.xyz
38a946e21893663b8abc0ecb4aafc639  rmd17_xyz/benzene/test_2000.xyz
84ebe4e100b81a59410c322c68b91a9f  rmd17_xyz/ethanol/test_2000.xyz
e87944ea5f18c3447df9417dfb49282a  rmd17_xyz/malonaldehyde/test_2000.xyz
c53179596b7fa96f924792b22bf9a6fd  rmd17_xyz/naphthalene/test_2000.xyz
d881c9937eb622f580ca78a07cf3977d  rmd17_xyz/paracetamol/test_2000.xyz
9528aebbf05afa7d3a1d69682e26c9a2  rmd17_xyz/salicylic/test_2000.xyz
68a1cdc062080b67b668312d2b832ae3  rmd17_xyz/toluene/test_2000.xyz
e8eead673fbacbf202619abb0fbd7cb4  rmd17_xyz/uracil/test_2000.xyz
6bddb1844958cafc5ad3038c60e1f1a9  lips/fu_test.xyz
17351dd082ba5b6aecb2bd172a378200  lips/fu_train.xyz
591bfc79e523c3793602fbb7eaaadab6  lips/fu_valid.xyz
26fd8a6e054642757a93cd2a3e371e35  lips/lc1000_train.xyz
e2ed5f2ec05dd54b363f2ac75055f093  lips/lc100_train.xyz
fd2d06cd90e0a1a41d49f7278adffe33  lips/lc10_train.xyz
48adc377ff434615231f7aa8cee9eace  lips/lc2500_train.xyz
c29aa623983343c4b0c1a4b8f9a3bb70  lips/lc_valid.xyz
ac3d568e3b9c3fc656d4ea77c5ad30b1  lips/splits.npz
788a709d7b1178c2cd11ae999b89109f  lips/e0.json
45b8e4797efee4f8e22d74f653091cf0  qm9/test.xyz
6c906f6b6aef3a2a72dcd9603a9e5576  qm9/train.xyz
0a266b1e3af5f8865603dcda7b84909d  qm9/valid.xyz
ac101a3a9585fb13d4f04d29afb30ff8  qm9/splits_seed42.npz
aff22c0565aaff7cfc1ee707c8624189  qm9_U/test.xyz
fcdb4d0a19e3fbf44959842fc49842e6  qm9_U/train.xyz
63f19273e4f5281919d744460af4e11e  qm9_U/valid.xyz
8dd7f018c1fc75d53becdf15e7d24f4f  qm9_H/test.xyz
7c11c58cb9f7fd2138ac8f08dc80b34c  qm9_H/train.xyz
7edab84decacf19a49d4670237399b16  qm9_H/valid.xyz
c1012fac099b7a620ecd5719e00a2331  qm9_G/test.xyz
7aed09bd16dc0ac414ffa7c1dce6fa50  qm9_G/train.xyz
a1e8038359ac8c718c7fb4c2b490d34e  qm9_G/valid.xyz
0fc004d393e3f6c0ef96ca99a8050863  qm9_ZPVE/test.xyz
b0f52fcfcf791eac54992a91701ded9b  qm9_ZPVE/train.xyz
5bfdd48d031678f7bcc3fd75f0bb18eb  qm9_ZPVE/valid.xyz
bac7b87734dd62fc6c2a893f28c5d119  qm9_Cv/test.xyz
8d31a42483d8c734f6456f6f63c099c2  qm9_Cv/train.xyz
4688de42a8761700c012e29b1379450a  qm9_Cv/valid.xyz
addfea8174684b2bf348ec96db9d846c  md22/Ac-Ala3-NHMe/train.xyz
492574fc2c48f956e2cb3ff66b506c5b  md22/AT-AT-CG-CG/train.xyz
e8b303ddf342baf8840c16748a61d3da  md22/AT-AT/train.xyz
f72c2c615489b31ba206ca7f6b1d42b9  md22/buckyball-catcher/train.xyz
30c24d16e7c07a11f4666cc8c2b22d91  md22/DHA/train.xyz
7abe93941fc6529625335322cb788453  md22/double-walled_nanotube/train.xyz
22be9fd5f928b08084dc588331667051  md22/stachyose/train.xyz
dd03c6cff6af50a73d033039ee389663  md22/Ac-Ala3-NHMe/split.json
96e890094b10a2b80e8bb9a08e7eefb6  md22/AT-AT-CG-CG/split.json
12322ea54c02bb9b6ef4ed3f39d14026  md22/AT-AT/split.json
f967b737b0547012e744a6c6aacc447a  md22/buckyball-catcher/split.json
3196f38220162a5d15c53220a3ef2994  md22/DHA/split.json
6f8d5791d1e99a863b3e7c4f08cd27d4  md22/double-walled_nanotube/split.json
fe2ef763a887d17dfce08b40577ef97e  md22/stachyose/split.json
c722bb255126e8dee185c15885b83870  si_gap18/test_compressionMD.xyz
c7868ebd71794336193f2659693e219c  si_gap18/test_extrap.xyz
7e94e41a89fa2350ad226e64c912ea74  si_gap18/test_rss.xyz
a6967ce7c5b67690cecfed12deb6fb20  si_gap18/test_strat.xyz
d64aadbcab8e33236d57ff558ab32e6c  si_gap18/train_extrap.xyz
fbc5cba9be7b7c3a40ec0e9faabbe271  si_gap18/train_full.xyz
20e274b3aa2d26985b1094c4a7e426d6  si_gap18/train_strat.xyz
8fe884b84cc78abf59f25c3f031d0d1c  water_xyz/test_noout.xyz
14a95989402841e497e18e2df33dca1d  water_xyz/test.xyz
0fa912a3fd29232769aa4a129c4fce1f  water_xyz/train_only.xyz
6491ca1c5919a27f69ab4149485b7670  water_xyz/train.xyz
124e2798b7442bc168a4c1d842847b29  water_xyz/valid.xyz
a02561cc242f627d83a7cdd46ee40967  water_split1393/test_last200.xyz
1ab44f6598cf39211d026143357ede36  water_split1393/train_first1393.xyz
dd5e53c06836191406bdc0d0865b6959  oam_subsets/mptrj_test.xyz
092cfc2feee85d03e61bf992f18cebe1  oam_subsets/mptrj_train.xyz
58e4f4d6d6c5037a00d46ec1d7773e43  oam_subsets/mptrj_valid.xyz
0a97aca72c7ab4f2b8431a6055ee1597  oam_subsets/omat24_test.xyz
76640ebe0bfcd782772013e9652cd94e  oam_subsets/omat24_train.xyz
2cebfd4d6d2c5d98b38dccc1a06825d8  oam_subsets/omat24_valid.xyz
3d9ff87aa95bf85f6947b7cbda659846  oam_subsets/salex_test.xyz
19251a468e1afce46400d8a913024385  oam_subsets/salex_train.xyz
6e2ca59c93d0a45feedc89d2b2e76d67  oam_subsets/salex_valid.xyz
6f6531f6c820a2ddf1f70c560d6e57bc  oam_subsets/mptrj_meta.json
3fb70b3573af3ec50b8930ed6867bdde  oam_subsets/omat24_meta.json
7d2156b4e1243f0ad32fb866908461bd  oam_subsets/salex_meta.json
32930aeb87ed1ece5c2b246c52d3e642  oam_subsets_100k/omat24_test.xyz
b6361f3dd9e0ea61572adb95e551d10b  oam_subsets_100k/omat24_train.xyz
4ba12a4c45dcd2dab2e1960827a50e23  oam_subsets_100k/omat24_valid.xyz
6e64a1a1c52c0b2bdb8bb9caad2d0648  oam_subsets_100k/omat24_meta.json
```
Not hashed (large; rebuild and compare frame counts): `rmd17_official/*/rest_split01.xyz` (unused) and
`md22/*/test.xyz`.

## 15. Gaps in the record

1. OAM raw data: the OMat24 / sAlex aselmdb downloads (`<DATA_ROOT>/oam_lmdb`), the shard conversion runs, the
   `refs/*_ls.npz` fits, and the whole MPtrj `jax_data_v3_raw` pipeline are NOT FOUND in transcript. The subset step
   is reproducible only from the existing shards; copy `oam_data/{omat24,salex}_{train,val}`, `oam_data/refs`,
   `mp_data/jax_data_v3_raw`, or simply the finished `oam_subsets*` xyz + meta files.
2. Repo `dataset_3BPA/`: its pre-repo origin is NOT FOUND in transcript, but the files equal the BOTNet copies byte for
   byte, so re-downloading BOTNet-datasets reproduces them.
3. No md5 was recorded at download time for MD22, ColabFit rMD17, the Zenodo 7003068 tarball, the water repo file,
   and the BOTNet files; the md5s above were computed on 2026-11-06 from the files on disk.
4. `venv_tools`: cloudpickle and narwhals installs NOT FOUND in transcript (not needed by any converter).
5. `rmd17_from_frodo.tar.gz` (user-provided) has no recorded origin beyond its own README; it is not used in results.

## 16. 4G-HDNNP systems (RACE_LR)

Benchmark data for RACE_LR step 2 (`docs/RACE_LR_design.md` §3). Converted on the local PC on 2026-11-06, not on
HANGANG, so the paths are relative to the repo root. `data_local/` is git-ignored.

**Source.** Ko, Finkler, Goedecker, Behler, "A fourth-generation high-dimensional neural network potential with
accurate electrostatics including non-local charge transfer", Nat. Commun. 12, 398 (2021), arXiv 2009.06484.
- Data: Materials Cloud Archive 2020.137, DOI 10.24435/materialscloud:f3-yh. Record page:
  https://archive.materialscloud.org/records/bk6kw-4b895, published 2020-11-04.
- The DOI 10.24435/materialscloud:dz-zg resolves to the same record.
- License: **CC BY 4.0**. Cite the paper and the record.
- DFT: FHI-aims 171221_1, PBE, light settings. NaCl, Ag3 and Au2/MgO are spin polarised. Au2/MgO uses a 3×3×1
  k-point grid. The per-atom charges are Hirshfeld charges.

**Files (md5 recorded at download; the record's own checksums agree):**

| file | size (bytes) | md5 |
|---|---|---|
| files_description.md | 233 | 832437d3cd7102d0a601fba567d7c9b8 |
| readme.txt | 2536 | 8a7eb0061688d4014848bc37009210f1 |
| datasets.tar.gz | 39726732 | 0aa11ea305b3af6a47bb3c577c2729bc |
| datasets/Carbon_chain/input.data (extracted) | 15761645 | 222436bccf3ce2f22d0a0ece9262e4f8 |
| datasets/Ag_cluster/input.data | 4231431 | 62a88547ce3cc6a744c03eefbc8b027a |
| datasets/NaCl/input.data | 11685000 | e49432d41cf703ff8d44390f00db7599 |
| datasets/AuMgO/input.data | 77130000 | 90112426e0825a0cc35be36bed3321a6 |

The archive holds only the four RuNNer `input.data` files, so there is no script to run. The format and the units
are described in the docstring of `scripts/data/convert_4ghdnnp.py`.

**Commands:**

```bash
mkdir -p data_local/4ghdnnp/raw && cd data_local/4ghdnnp/raw
for f in files_description.md readme.txt datasets.tar.gz; do
  curl -sfL -o $f "https://archive.materialscloud.org/records/bk6kw-4b895/files/$f?download=1"; done
md5sum * > md5sums.txt
tar xzf datasets.tar.gz --no-same-owner --no-same-permissions && md5sum datasets/*/input.data >> md5sums.txt
cd ../../.. && python -I scripts/data/convert_4ghdnnp.py data_local/4ghdnnp/raw data_local/4ghdnnp
```
The environment was the repo `.venv` (Python 3.12.15, ase 3.29.0, numpy 2.5.4). The conversion takes about 15 s.

**Unit conversions.**
- The archive is in atomic units: Bohr, Hartree, e and Hartree/Bohr.
- Energies are multiplied by 1 Ha = 27.211386245988 eV and positions and cells by 1 Bohr = 0.529177210903 Å
  (CODATA 2018).
- Forces are multiplied by 27.211386245988 / 0.529177210903 eV/Å per Ha/Bohr.

**Output.** Each system gets a directory `data_local/4ghdnnp/<system>/` with these files:
- `all.xyz`, `train.xyz` and `test.xyz`.
- `split.json` with the indices and the counts per subset.
- `e0.json` with the per-element offsets, described below.

The extxyz keys:
- info `energy` in eV.
- info `charge`, the total charge as an integer in e. Use it with `"total_charge_key": "charge"`.
- info `subset`, the charge state or structure family, and info `source_index`, the block index in `input.data`.
- arrays `forces` in eV/Å and `hirshfeld_charges` in e. The Hirshfeld charges sum to the total charge in every
  structure.
- Au2/MgO has the cell and `pbc="T T T"`, matching the 3D periodic slab of the DFT. The cell is 9.05 × 9.05 ×
  26.46 Å, with the slab and the vacuum along c. The other systems have no cell and `pbc="F F F"`.

**Split.**
- The paper used a random 90 % / 10 % train / test split ("In all cases 90% of the available reference data was
  used for training ... the remaining 10% ... as an independent test set") but did not publish the indices.
- Our split is `numpy.random.RandomState(42).permutation(n)`, with the first `round(0.9 n)` frames for training,
  per system and not stratified. The indices are in `split.json`.
- Validation for training comes from `valid_frac` of the trainer, taken from `train.xyz` only.

| system | subset (total charge) | n | n_atoms | train | test | E range (eV) | max abs F component (eV/Å) |
|---|---|---|---|---|---|---|---|
| carbon_chain | C10H2 (0) | 5008 | 12 | 4511 | 497 | −10384.640 … −10383.614 | 7.16 |
| carbon_chain | C10H3+ (+1) | 5011 | 13 | 4506 | 505 | −10393.885 … −10392.777 | 10.85 |
| ag_cluster | Ag3+ (+1) | 5014 | 3 | 4507 | 507 | −439152.900 … −439152.131 | 3.12 |
| ag_cluster | Ag3− (−1) | 5999 | 3 | 5405 | 594 | −439160.765 … −439159.356 | 2.40 |
| nacl | Na8Cl8+ (+1) | 2500 | 16 | 2246 | 254 | −135471.864 … −135471.305 | 2.37 |
| nacl | Na9Cl8+ (+1) | 2500 | 17 | 2254 | 246 | −139888.893 … −139888.490 | 2.21 |
| aumgo | Au2/MgO undoped (0) | 2500 | 110 | 2264 | 236 | −1476723.181 … −1476716.845 | 31.82 |
| aumgo | Au2/MgO Al-doped (0) | 2500 | 110 | 2236 | 264 | −1480190.765 … −1480185.480 | 24.62 |

Totals are 10019, 11013, 5000 and 5000 structures, matching the record's readme. NaCl has total charge +1 in both
compositions, as the readme states. The charge-state contrast there is the position of the extra Na.

**Reference energies (`e0.json`).**
- The archive has no isolated-atom energies.
- The trainer subtracts `atom_energies` and then stores the target in float32. At −439 000 eV (Ag3), float32
  resolves only about 0.03 eV, so an offset is needed.
- `e0.json` holds per-element energies from a least-squares fit of E = Σ_Z n_Z E0_Z on the **training split
  only**, which stays within the fair protocol.
- When the compositions do not determine every element, the fit takes the minimum-norm solution. That happens for
  Ag3 (a single composition) and for Au2/MgO (rank 2 of 4 elements: O, Mg, Al, Au).
- The values are therefore offsets, not physical atom energies. With these offsets the targets after subtraction
  are within a few eV.
- The fit residual std is 0.135 eV for carbon_chain, 3.85 eV for ag_cluster, 0.071 eV for nacl and 0.447 eV for
  aumgo. For Ag3 it is the Ag3+ / Ag3− energy gap.
- Isolated-atom DFT energies at the same level would be the physical alternative. They are not computed.
- Decision (2026-11-06): use the `e0.json` offsets as `atom_energies` for RACE_LR and the RACEv2 baseline alike.
  They are a constant per composition, so they do not change any energy difference within a composition, including
  the charge-state gaps.

**Check (2026-11-06).**
- Read back with `ase.io.read`: five structures per system (indices 0, 1, n/2, n−1 and 1234) were compared with an
  independent re-parse of `input.data`. Symbols and total charge are identical. Energy, positions, forces and cell
  agree to 5e-9 or better, which is the 8-decimal precision of ASE's extxyz writer.
- Train ∪ test = all, the two are disjoint, and `source_index` matches `split.json`.
- A RACE_LR forward pass on converted structures gave finite energies, forces and stress, conserved charge exactly,
  and ran on the periodic Au2/MgO cell (110 atoms) on CPU.

**md5 of the outputs (ASE 3.29.0 writer):**

```
2f6ee7ae158922f18d1566aa0a70413f  carbon_chain/all.xyz
7ea91416ff935bfd827a18fc8712ef22  carbon_chain/train.xyz
7f961dc68aeee6238613feadc898100c  carbon_chain/test.xyz
7129a1765b5f41a68a8ca47b14cd39a8  carbon_chain/split.json
36a7d68ff3313e149d04a187d4825de9  carbon_chain/e0.json
8e920cc7f37a4121f32d1b52b538c43c  ag_cluster/all.xyz
5947c9a0a1abe7fe69c6b0b80a62e0ee  ag_cluster/train.xyz
8883f48ec4b04f6a1abe7a4453b119b9  ag_cluster/test.xyz
bc6a1f1c8e3a73a5aac3210957a4f5df  ag_cluster/split.json
671a8d7b8b9a1c7ffcbe95275ca73e32  ag_cluster/e0.json
ede591fecbdd6bdf0e3d9a6ccbc7ab34  nacl/all.xyz
ccac97c4ffda8d89ed9e8bb5238c60d5  nacl/train.xyz
7e52a56659d33e86af4aa0f0944c3879  nacl/test.xyz
ce43898e4315f7ab33f34762a2c3dfd0  nacl/split.json
7cd7917c00abbf643cfb282bb9e1bf10  nacl/e0.json
269c12ceec15ae62b6d62be007791f62  aumgo/all.xyz
524e371f1856ada080b1b04a2583c8bc  aumgo/train.xyz
a4fcd2c32eb623e9b6aa8096639fbb6e  aumgo/test.xyz
5ac5c63b65992204f1b2296d3e360bf1  aumgo/split.json
875b924c3df0de2ae7a07d6e0e18f7f4  aumgo/e0.json
```

**Training subsets for local tests (`scripts/data/subset_4ghdnnp.py`, 2026-11-06).**
- Each system gets 500 structures of each subset label from `train.xyz`, drawn with RandomState(0) and written to
  `train_sub1000.xyz` and `subset_sub1000.json`.
- Au2/MgO also gets 50 of each, drawn from that file, in `train_sub100.xyz`.
- The test split is not subsampled.
- Command: `python -I scripts/data/subset_4ghdnnp.py data_local/4ghdnnp`. md5 of the outputs:

```
20314c6a67f43dbe504a2610df3a5c18  ag_cluster/train_sub1000.xyz
e69e09e655dab7c5f08d20752877e39a  carbon_chain/train_sub1000.xyz
fabdbc4ecf8ac648b2f51f9dab8a0a96  nacl/train_sub1000.xyz
b8b492c8386c09fc13e495ed8f2442bb  aumgo/train_sub1000.xyz
87b08a8475b107695a0b1c8beef76ae9  aumgo/train_sub100.xyz
```

## 17. OMat24 piece `rattled-300-subsampled` and the OMol25 → OMat24 transfer set (RACE_LR)

Data of the small-scale transfer test of 2026-11-10 (`docs/HANDOFF_uq_omol25.md` §7.4). Downloaded and converted on
the local PC, so the paths are relative to the repo root. `data_local/` is git-ignored.

**Source.** OMat24 (Barroso-Luque et al., arXiv 2410.12771), the public **validation** split, sub-dataset
`rattled-300-subsampled` (75 MB, the smallest piece of the split).
- URL: https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/val/rattled-300-subsampled.tar.gz
- No login needed. License of OMat24: CC BY 4.0 as remembered from the dataset card; **not re-read on 2026-11-10**.
- This is a validation piece of OMat24. A model trained on it must not be reported against the OMat24 validation set.

| file | size (bytes) | md5 |
|---|---|---|
| rattled-300-subsampled.tar.gz | 74995647 | 4a2a35a16eca98c9f8e1eb25e33fe308 |
| rattled-300-subsampled/val.aselmdb (extracted) | 116310016 | 378d885dbeac1b39afe79efc0d7d1400 |
| rattled-300-subsampled/metadata.npz (extracted) | 299410 | 09d4d8d616512dc31d01dddb2a7be893 |

The archive holds these two files only. It was listed before extraction and extracted with
`tar -xzf ... --no-same-owner --no-same-permissions` in its own directory
`data_local/omat24_val/rattled-300-subsampled_dl/`. 37,393 rows.

**Subset `data_local/omat24_val/r300_10k`.**

```
python -I scripts/data/convert_omol25.py subset \
  --train data_local/omat24_val/rattled-300-subsampled_dl/rattled-300-subsampled \
  --eval  data_local/omat24_val/rattled-300-subsampled_dl/rattled-300-subsampled \
  --out data_local/omat24_val/r300_10k --n-train 10000 --n-valid 1000 --n-test 2000 --seed 0 \
  --allow-periodic --neutral-if-missing
```

- `--train` and `--eval` are the same files, so the test rows are drawn from the rows not used for train or valid
  (the three sets are disjoint; rule text in `meta.json`).
- 85 elements, charge 0 and spin 1 written for every row (`--neutral-if-missing`), stress kept.
- E0: least squares on the 10,000 train rows only, rank 85 (`e0.json`). `avg_n_neighbors` 19.804 at 5.0 Å.

| file | structures | md5 |
|---|---|---|
| train.xyz | 10000 | 8d473d0577bf66d7584a9842a21d8de7 |
| valid.xyz | 1000 | 99213d1bd4be4d5b6b09b57984ecb5d2 |
| test.xyz | 2000 | 73aaf1860e075d8223a61c594a635235 |

**Transfer set `data_local/transfer_omol_omat`** (`research/diag/transfer/prepare.py`).

```
python -I research/diag/transfer/prepare.py --omol data_local/omol25_openff/sub10k --omat data_local/omat24_val/r300_10k \
  --pre research/input/omol/tr_u94_race_cs_amsgrad_f32_s42.json
```

- OMol25 side: `data_local/omol25_openff/sub10k` (the ANI-2x part redistributed by OpenFF, `docs/local_test_report.md`),
  each molecule in a cubic vacuum box (edge = extent + 2 × 5.0 Å + 2 Å, pbc true), info `task` = 0.
- OMat24 side: `r300_10k`, info `task` = 1.
- Energies are referenced, E − Σ E0(Z), with the E0 of the structure's own subset. The configs use `atom_energies` 0.
- `mix_{train,valid}.xyz` hold both tasks, shuffled (replay arm).

| file | structures | md5 |
|---|---|---|
| mix_train.xyz | 20000 | 32234529cdc2d74b6ac5af32ef5fb463 |
| mix_valid.xyz | 2000 | 0ce0438f50a7fea8cb469d92b45d5f7d |
| omat_test.xyz | 2000 | 4560f12b37eb3b4de5f923692ea68c5c |
| omat_train.xyz | 10000 | 7c81ca0cef0dcf530fc4ea25aa086984 |
| omat_valid.xyz | 1000 | 905853704f358e184389da1b3e5196ca |
| omol_test.xyz | 2932 | 9d5749da409fd18bd9155874b1691ccf |
| omol_train.xyz | 10000 | d0f8d4c1ec132e737039314f1a204dc2 |
| omol_valid.xyz | 1000 | 7a64870078e021e197af1ae5fd687140 |

The `*_omat24_cutoff_*.pkl` files next to them are graph caches and can be deleted.
