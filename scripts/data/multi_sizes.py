"""Size tables (n_atoms, n_edges, cell per structure) of a multi-dataset stream config, made ahead of the training
with many workers (JAX-free; docs/HANDOFF_multidataset_thole.md).  Needs "sizes" directories in the config
(make_multidataset_configs.py --sizes-dir).

    BAM_NEIGHBORS=vesin python -I scripts/data/multi_sizes.py research/input/multi/<name>.json [--workers 11] [--splits train valid]
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'race_v2' / 'data'))
import stream_io as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--workers', type=int, default=max(1, len(os.sched_getaffinity(0)) - 1))
    ap.add_argument('--splits', nargs='+', default=['valid', 'train'])
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    for split in a.splits:
        spec = cfg['stream'][split]
        if not isinstance(spec, dict) or not spec.get('sizes'):
            raise SystemExit(f'stream.{split} has no "sizes" directory (make the config with --sizes-dir)')
        out = Path(spec['sizes'])
        if (out / 'sizes_meta.json').exists():
            print(f'{split}: {out} exists, skipped')
            continue
        print(f'{split}: {len(spec["sources"])} sources -> {out}', flush=True)
        S.size_pass(list(spec['sources']), float(cfg['cutoff']), out, workers=a.workers, block=256)


if __name__ == '__main__':
    main()
