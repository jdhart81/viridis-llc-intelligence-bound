"""Make this repository's ``code`` package importable under pytest.

The package name shadows Python's standard-library ``code`` module, which pytest
imports before collecting tests. Put the repository root first on ``sys.path`` and
drop the standard-library module from the import cache so ``import code.<module>``
resolves to this repository.
"""

import pathlib
import sys

ROOT = str(pathlib.Path(__file__).resolve().parent)
if ROOT in sys.path:
    sys.path.remove(ROOT)
sys.path.insert(0, ROOT)

_cached = sys.modules.get("code")
if _cached is not None and not hasattr(_cached, "__path__"):
    del sys.modules["code"]
