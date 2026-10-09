"""Import hook for the v5 layout: puts ``backend/`` and ``ml/`` on ``sys.path``.

The modules keep their flat names (``common``, ``server``, ``scoring``, ``adapters`` ...) because the deployed
model file pickles classes by module name (``scoring``). Entry points add the project root to ``sys.path``
and then ``import btl_paths``.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for folder in ('ml', 'backend'):
    path = str(ROOT / folder)
    if path not in sys.path:
        sys.path.insert(0, path)
