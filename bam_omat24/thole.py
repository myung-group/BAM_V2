"""Compatibility alias (2026-11-16): ``bam_omat24.thole`` is now ``race_v2.models.thole`` (race_v2/models/thole.py).

Generated; do not edit.  Old imports, ``python -m bam_omat24.thole`` and ``python bam_omat24/thole.py`` keep working.
"""
import os as _os
import sys as _sys

_ROOT = _os.path.realpath(__file__)
for _ in range(2):
    _ROOT = _os.path.dirname(_ROOT)
_PATH = _ROOT                                   # the folder that holds the new package
_NEW, _FILE = 'race_v2.models.thole', _os.path.join(_ROOT, 'race_v2', 'models', 'thole.py')
if _PATH not in _sys.path:
    _sys.path.insert(0, _PATH)

if __name__ == '__main__':
    import runpy as _runpy
    if __package__:                                   # python -m bam_omat24.thole
        _runpy.run_module(_NEW, run_name='__main__', alter_sys=True)
    else:                                             # python bam_omat24/thole.py  (stand-alone script, no package import)
        _sys.argv[0] = _FILE
        _runpy.run_path(_FILE, run_name='__main__')
elif '.' in __name__:                                 # import bam_omat24.thole
    import importlib as _importlib
    _sys.modules[__name__] = _importlib.import_module(_NEW)
else:                                                 # top-level import after sys.path.insert(0, <old folder>)
    import importlib.util as _util
    _spec = _util.spec_from_file_location(__name__, _FILE)
    _mod = _util.module_from_spec(_spec)
    _sys.modules[__name__] = _mod
    _spec.loader.exec_module(_mod)
