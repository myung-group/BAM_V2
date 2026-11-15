"""Which ops of the exported RACEv2 function need the most device memory?  Compiles the LAMMPS export function for
one bucket with XLA buffer-assignment dumps and lists the largest allocations with their HLO op and source line.

    BAM_INDEXED_LINEAR=ragged python research/diag/memory_probe.py <config.json> <ckpt.pkl> <n_pad> <e_pad> [backend]
"""
import glob
import os
import re
import sys
import tempfile

dump = tempfile.mkdtemp(prefix='xla_dump_', dir=os.environ.get('TMPDIR', '/tmp'))
os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + f' --xla_dump_to={dump} --xla_dump_hlo_as_text'

import jax
import jax.numpy as jnp

from race_v2.export.lammps import build, make_fn


def main():
    cfg_path, ckpt, n_pad, e_pad = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    backend = sys.argv[5] if len(sys.argv) > 5 else 'cue'
    cfg, model, zs, e0 = build(cfg_path, ckpt, backend, 'float64')
    fn = jax.jit(make_fn(model, float(cfg.get('cutoff', 5.0)), n_pad, e_pad))
    sds = (jax.ShapeDtypeStruct((n_pad,), jnp.int32), jax.ShapeDtypeStruct((e_pad,), jnp.int32),
           jax.ShapeDtypeStruct((e_pad,), jnp.int32), jax.ShapeDtypeStruct((e_pad, 3), jnp.float64),
           jax.ShapeDtypeStruct((), jnp.int32), jax.ShapeDtypeStruct((), jnp.int32), jax.ShapeDtypeStruct((), jnp.int32))
    comp = fn.lower(*sds).compile()
    ma = comp.memory_analysis()
    print(f"bucket {n_pad}/{e_pad} ({backend}, indexed linear {os.environ.get('BAM_INDEXED_LINEAR', 'gather')}): "
          f"temp {ma.temp_size_in_bytes / 2**30:.2f} GiB, args {ma.argument_size_in_bytes / 2**30:.3f} GiB, "
          f"output {ma.output_size_in_bytes / 2**30:.3f} GiB, generated code {ma.generated_code_size_in_bytes / 2**20:.1f} MiB",
          flush=True)
    files = sorted(glob.glob(os.path.join(dump, '*buffer-assignment*.txt')), key=os.path.getsize)
    if not files:
        print('no buffer assignment dump found in', dump); return
    txt = open(files[-1]).read()
    # value lines: "  value: <id ...> (size=..., offset=...): f64[...]{...}" ; allocation lines: "allocation N: size S ..."
    allocs = re.findall(r"allocation (\d+): size (\d+)(.*?)\n((?: value:.*\n)+)", txt)
    rows = []
    for aid, size, attrs, values in allocs:
        vals = [v.strip() for v in values.strip().splitlines()]
        rows.append((int(size), aid, attrs.strip()[:60], vals[:3]))
    rows.sort(reverse=True)
    for size, aid, attrs, vals in rows[:6]:
        print(f"allocation {aid}: {size / 2**30:.2f} GiB {attrs}")
        for v in vals:
            print('    ', v[:220])
    # largest single values (logical buffers) with their HLO instruction
    vals = re.findall(r"value: <(\d+) ([^ >]+)[^>]*> \(size=(\d+),offset=\d+\): ([^\n]+)", txt)
    vals = sorted({(int(sz), name, shp[:80]) for _, name, sz, shp in vals}, reverse=True)[:15]
    hlo = sorted(glob.glob(os.path.join(dump, '*after_optimizations.txt')), key=os.path.getsize)
    htxt = open(hlo[-1]).read() if hlo else ''
    print('largest logical buffers:')
    for sz, name, shp in vals:
        inst = name.split('{')[0]
        m = re.search(r"\n\s*%?" + re.escape(inst) + r" = [^\n]*", htxt)
        meta = ''
        if m:
            mm = re.search(r'op_name="([^"]+)"', m.group(0)); sf = re.search(r'source_file="([^"]+)" source_line=(\d+)', m.group(0))
            meta = (mm.group(1)[-90:] if mm else '') + (f"  [{os.path.basename(sf.group(1))}:{sf.group(2)}]" if sf else '')
        print(f"  {sz / 2**30:8.2f} GiB  {shp:60s} {inst[:40]:40s} {meta}")


if __name__ == '__main__':
    main()
