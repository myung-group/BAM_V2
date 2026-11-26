"""Package data read at run time (2026-11-24): the core exclusion table of the ASE calculators' default and the element
tables of the PyTorch port must be inside their packages and listed in pyproject package-data, so that a non-editable
pip install (e.g. the Matbench Discovery runners installing git+https://...@commit) has them."""
import importlib.util
import os
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _package_data():
    return tomllib.loads((REPO / 'pyproject.toml').read_text())['tool']['setuptools']['package-data']


def test_core_exclusion_table_is_package_data():
    from race_v2.inference.predict import DEFAULT_CORE_EXCLUSION
    table = Path(DEFAULT_CORE_EXCLUSION['table'][0])
    assert table.exists(), table
    pkg = Path(importlib.util.find_spec('race_v2').origin).parent
    assert table.resolve().parent == (pkg / 'resources').resolve()
    assert 'resources/*.json' in _package_data()['race_v2']


def test_torch_calculator_finds_the_same_table():
    spec = importlib.util.spec_from_file_location(
        '_rv2t_calc', REPO / 'race_v2_torch' / 'race_v2_torch' / 'calculator.py',
        submodule_search_locations=None)
    # read the path function without importing torch: evaluate _default_table_path from the source
    src = (REPO / 'race_v2_torch' / 'race_v2_torch' / 'calculator.py').read_text()
    start = src.index('def _default_table_path')
    end = src.index('\n\n\nDEFAULT_CORE_EXCLUSION')
    ns = {'os': os, '__file__': str(spec.origin)}
    exec(src[start:end], ns)
    from race_v2.inference.predict import DEFAULT_CORE_EXCLUSION
    assert Path(ns['_default_table_path']()).resolve() == Path(DEFAULT_CORE_EXCLUSION['table'][0]).resolve()


def test_torch_element_tables_are_package_data():
    assert (REPO / 'race_v2_torch' / 'race_v2_torch' / 'data' / 'element_tables.json').exists()
    assert 'data/*.json' in _package_data()['race_v2_torch']
