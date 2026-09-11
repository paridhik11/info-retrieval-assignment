"""
run_evaluation.py
=================

Part E convenience entry point.

The full evaluation and analysis harness lives in :mod:`src.evaluate` (a single
source of truth, driven entirely by the live retrieval code). This thin wrapper
exists only so the historical ``tests/run_evaluation.py`` path still works; it
simply delegates to ``src.evaluate.main``.

Preferred invocation::

    python -m src.evaluate

Equivalent::

    python tests/run_evaluation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from evaluate import main  # noqa: E402


if __name__ == "__main__":
    main()
