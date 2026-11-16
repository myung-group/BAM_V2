"""Size tables on another account or machine: rewrite the raw-data paths of sizes_meta.json and check the raw files.

The trainer compares the config ``sources`` with the ``spec`` strings of sizes_meta.json (exact string match) and
opens the files listed under ``files`` (absolute paths), so a size table made elsewhere needs its paths rewritten.
For every table this writes <out>/<name>/sizes_meta.json with the new paths and links sizes.npz of the original.

    python -I scripts/data/remap_size_tables.py --src <oam_sizes> --out <new dir> \
        --map /scratch/t510a13/data/oam_lmdb=/scratch/<me>/oam_lmdb [--map OLD=NEW ...] \
        [--tables omat24_val_c7,...] [--check] [--count --workers 64]

--check  (directory listings only): every source directory must hold exactly the files of the table, in the same
         order as the size pass listed them (sorted recursive listing of *.aselmdb).
--count  (opens every aselmdb; run it on a compute node): rows of all files = n + dropped of the table, and the
         keys of the table are row ids of their file.
Tables whose sources are not covered by a --map are skipped with a message.  Exit status 1 on any mismatch.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'race_v2' / 'data'))


def remap(s, maps):
    for old, new in maps:
        if old in s:
            return s.replace(old, new), True
    return s, False


def _rows(path):
    import stream_io as S
    env = S.open_lmdb(path)
    ids = S.list_row_ids(env)
    env.close()
    return np.asarray(ids, np.int64)


def _check_file(args):
    path, keys = args
    ids = _rows(path)
    return len(ids), bool(np.isin(keys, ids, assume_unique=False).all())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--map', action='append', default=[], help='OLD=NEW path prefix, first match wins')
    ap.add_argument('--tables', help='comma-separated table folders (default: all with sizes_meta.json)')
    ap.add_argument('--check', action='store_true'); ap.add_argument('--count', action='store_true')
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    maps = [tuple(m.split('=', 1)) for m in a.map]
    src, out = Path(a.src), Path(a.out)
    names = a.tables.split(',') if a.tables else sorted(p.parent.name for p in src.glob('*/sizes_meta.json'))
    bad = 0
    for name in names:
        meta = json.load(open(src / name / 'sizes_meta.json'))
        specs = [remap(s['spec'], maps) for s in meta['sources']]
        if not all(ok for _, ok in specs):
            print(f"{name}: SKIPPED, no --map for {[s['spec'] for s, (_, ok) in zip(meta['sources'], specs) if not ok]}")
            continue
        for s, (spec, _) in zip(meta['sources'], specs):
            s['spec'] = spec
            s['files'] = [remap(f, maps)[0] for f in s['files']]
        meta['files'] = [remap(f, maps)[0] for f in meta['files']]
        meta['remapped_from'] = str(src / name)
        msg = [f"{name}: n {meta['n']}, files {len(meta['files'])}"]
        if a.check or a.count:
            import stream_io as S
            listed = []
            for s in meta['sources']:
                kind, fl = S._source_files(s['spec'])
                if fl != s['files']:
                    bad += 1
                    here, want = set(fl), set(s['files'])
                    msg.append(f"\n  MISMATCH {s['spec']}: {len(fl)} files here, {len(s['files'])} in the table; "
                               f"missing {sorted(want - here)[:3]}, extra {sorted(here - want)[:3]}, "
                               f"same set {here == want}")
                listed += fl
            if listed == meta['files']:
                msg.append('file list and order OK')
            elif len(msg) == 1:
                bad += 1; msg.append('\n  MISMATCH: order of the global file list')
        if a.count and not any('MISMATCH' in m for m in msg):
            from multiprocessing import Pool
            z = np.load(src / name / 'sizes.npz')
            fi, key = z['file_index'].astype(np.int64), z['key'].astype(np.int64)
            order = np.argsort(fi, kind='stable'); fs = fi[order]
            cut = np.searchsorted(fs, np.arange(len(meta['files']) + 1))
            tasks = [(meta['files'][i], key[order[cut[i]:cut[i + 1]]]) for i in range(len(meta['files']))]
            with Pool(a.workers) as pool:
                res = pool.map(_check_file, tasks, chunksize=4)
            rows = sum(r for r, _ in res); dropped = sum((meta.get('dropped') or {}).values())
            nok = sum(ok for _, ok in res)
            good = rows == meta['n'] + dropped and nok == len(res)
            bad += not good
            msg.append(f"rows in the files {rows}, table n + dropped {meta['n']} + {dropped} = {meta['n'] + dropped}, "
                       f"files with all table keys present {nok}/{len(res)}: {'OK' if good else 'MISMATCH'}")
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        json.dump(meta, open(d / 'sizes_meta.json', 'w'), indent=1)
        link = d / 'sizes.npz'
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink((src / name / 'sizes.npz').resolve(), link)
        print('; '.join(msg).replace('; \n', '\n'), flush=True)
    print('RESULT:', 'all tables OK' if bad == 0 else f'{bad} mismatch(es)')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
