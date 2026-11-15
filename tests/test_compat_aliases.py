"""Compatibility aliases of the old package layout (bam_omat24 -> race_v2, 2026-11-16).

Old imports must give the SAME module objects as the new ones (so that monkey-patching through either name acts on
one module), `python -m bam_omat24.<old>` and `python bam_omat24/<old>.py` must reach the new code, and the JAX-free
modules must stay importable as top-level modules from the old folder without importing JAX.
"""
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [('bam_omat24.train_3bpa', 'race_v2.training.train'), ('bam_omat24.training_utils', 'race_v2.training.utils'),
         ('bam_omat24.data_3bpa', 'race_v2.data.xyz_graphs'), ('bam_omat24.data_nnx', 'race_v2.data.loaders'),
         ('bam_omat24.race_nnx', 'race_v2.models.race'), ('bam_omat24.race_lr_nnx', 'race_v2.models.race_lr'),
         ('bam_omat24.race_polar_nnx', 'race_v2.models.race_polar'), ('bam_omat24.utils', 'race_v2.models.blocks'),
         ('bam_omat24.linear_nnx', 'race_v2.models.linear'), ('bam_omat24.predict', 'race_v2.inference.predict'),
         ('bam_omat24.stream_io', 'race_v2.data.stream_io'), ('bam_omat24.thole', 'race_v2.models.thole'),
         ('bam_omat24.export_lammps', 'race_v2.export.lammps'), ('bam_omat24.soap_optax', 'race_v2.training.soap_optax')]


@pytest.mark.parametrize('old,new', PAIRS)
def test_old_module_is_new_module(old, new):
    assert importlib.import_module(old) is importlib.import_module(new)


def test_every_old_module_has_an_alias():
    import bam_omat24
    for name, old in bam_omat24.ALIASES.items():
        assert (ROOT / 'bam_omat24' / f'{name}.py').is_file(), name
        m = importlib.import_module(old)
        assert m.__name__.startswith('race_v2.'), (old, m.__name__)
        assert getattr(bam_omat24, name) is m


def test_package_reexports_and_old_function_name():
    import bam_omat24
    import race_v2
    assert bam_omat24.RACE is race_v2.RACE and bam_omat24.load_checkpoint is race_v2.load_checkpoint
    from bam_omat24.train_3bpa import build_model
    from race_v2.training.train import build_model as new_build
    assert build_model is new_build
    from bam_omat24.data_3bpa import load_3bpa_graphs
    from race_v2.data.xyz_graphs import load_xyz_graphs
    assert load_3bpa_graphs is load_xyz_graphs


def test_patching_through_the_old_name_reaches_the_new_module():
    import bam_omat24.race_nnx as old
    import race_v2.models.race as new
    saved = new.READOUT_TAP
    try:
        old.READOUT_TAP = 'marker'
        assert new.READOUT_TAP == 'marker'
    finally:
        new.READOUT_TAP = saved


def _run(args, **kw):
    env = dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT))
    env.pop('BAM_DISTRIBUTED', None)
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, cwd=ROOT, env=env, timeout=600, **kw)


def test_jax_free_top_level_import_from_the_old_folder():
    code = ("import sys; sys.path.insert(0, 'bam_omat24'); import neighbors, stream_plan, graph_fields, stream_io, "
            "stream_worker, formal_assign; assert 'jax' not in sys.modules and 'race_v2' not in sys.modules; "
            "print(stream_io.__file__)")
    r = _run(['-I', '-c', code])
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().endswith(os.path.join('race_v2', 'data', 'stream_io.py'))


@pytest.mark.parametrize('script', ['bam_omat24/stream_io.py', 'race_v2/data/stream_io.py', 'bam_omat24/graph_sizes.py'])
def test_run_by_path(script):
    r = _run(['-I', script, '--help'])
    assert r.returncode == 0 and 'usage' in r.stdout, r.stderr[-2000:]


@pytest.mark.parametrize('module', ['bam_omat24.predict', 'race_v2.inference.predict', 'bam_omat24.eval_sets'])
def test_run_as_module(module):
    r = _run(['-m', module, '--help'])
    assert r.returncode == 0 and 'usage' in r.stdout, (r.stdout[-500:], r.stderr[-2000:])
