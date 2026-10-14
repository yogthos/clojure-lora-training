#!/usr/bin/env python3
"""Assemble a Code Flow training dataset for LLaMA-Factory.

Wrapper for ``src.codeflow.cli.assemble_codeflow_dataset``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.codeflow.cli.assemble_codeflow_dataset import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
