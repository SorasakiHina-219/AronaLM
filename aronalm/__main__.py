"""
``python -m aronalm`` entry point.

Delegates to :mod:`aronalm.server` so both invocation styles work:

    python -m aronalm.server --port 8765
    python -m aronalm --port 8765
"""

from __future__ import annotations

from aronalm.server import main

if __name__ == "__main__":
    raise SystemExit(main())
