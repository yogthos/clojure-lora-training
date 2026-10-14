#!/usr/bin/env python3
"""Mine Clojure repositories for before/after code training pairs.

Wrapper for ``src.codeflow.cli.mine_clojure_repos``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.codeflow.cli.mine_clojure_repos import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
