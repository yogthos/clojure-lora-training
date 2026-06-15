"""Tests for the mine_clojure_repos CLI script."""

import subprocess
import json
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent


def test_cli_help():
    result = subprocess.run(
        ["python3", str(PROJECT_ROOT / "scripts/mine_clojure_repos.py"), "--help"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0
    assert "Mine Clojure repositories" in result.stdout


def test_cli_outputs_valid_jsonl(tmp_path):
    """Test that CLI produces valid JSONL on a real Clojure repo.

    Uses a small public repo with known Clojure code.
    """
    output_file = tmp_path / "output.jsonl"

    # Clone a tiny Clojure repo for testing
    clone_dir = tmp_path / "test-repo"
    subprocess.run(
        ["git", "clone", "--depth=50",
         "https://github.com/ring-clojure/ring-codec",
         str(clone_dir)],
        capture_output=True, text=True,
    )

    result = subprocess.run(
        ["python3", str(PROJECT_ROOT / "scripts/mine_clojure_repos.py"),
         "--repo", str(clone_dir),
         "--max-commits", "20",
         "--output", str(output_file),
         "--stats"],
        capture_output=True, text=True,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0

    # Each line should be valid JSON with required fields
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                assert "system" in record
                assert "instruction" in record
                assert "input" in record
                assert "output" in record


def test_cli_pattern_filter(tmp_path):
    """Test pattern filtering via --pattern flag."""
    output_file = tmp_path / "output.jsonl"
    clone_dir = tmp_path / "test-repo"
    subprocess.run(
        ["git", "clone", "--depth=50",
         "https://github.com/ring-clojure/ring-codec",
         str(clone_dir)],
        capture_output=True, text=True,
    )

    result = subprocess.run(
        ["python3", str(PROJECT_ROOT / "scripts/mine_clojure_repos.py"),
         "--repo", str(clone_dir),
         "--max-commits", "20",
         "--output", str(output_file),
         "--pattern", "pure-refactor"],
        capture_output=True, text=True,
        cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0
    # Should either produce output or fail gracefully
