#!/usr/bin/env python3
"""Generate plan-first, iterative-REPL workflow training data.

Wrapper for ``src.codeflow.cli.generate_workflow_data``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.codeflow.cli.generate_workflow_data import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
