#!/usr/bin/env python3
"""Generate synthetic Clojure training data using the EpiCoder pipeline.

Wrapper for ``src.codeflow.cli.generate_synthetic_data``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.codeflow.cli.generate_synthetic_data import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
