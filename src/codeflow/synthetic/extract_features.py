"""Extract Clojure code features via LLM analysis.

Adapted from EpiCoder's extract/extract_features.py. Sends Clojure source code
to an LLM to identify and catalog language-specific constructs: macros,
protocols, multimethods, concurrency primitives, transducers, specs,
JVM interop patterns, and REPL-driven development patterns.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ...llm.provider import LLMProvider


from .prompts import EXTRACT_SYSTEM as _EXTRACT_SYSTEM_PROMPT


@dataclass
class ClojureFeature:
    """A single extracted Clojure code feature."""
    feature_type: str
    name: str
    description: str
    file_path: str = ""
    line_hint: int = 0
    complexity: str = "simple"

    @classmethod
    def from_dict(cls, d: dict) -> "ClojureFeature":
        return cls(
            feature_type=d.get("feature_type", "unknown"),
            name=d.get("name", ""),
            description=d.get("description", ""),
            file_path=d.get("file_path", ""),
            line_hint=d.get("line_hint", 0),
            complexity=d.get("complexity", "simple"),
        )

    def to_dict(self) -> dict:
        return {
            "feature_type": self.feature_type,
            "name": self.name,
            "description": self.description,
            "file_path": self.file_path,
            "line_hint": self.line_hint,
            "complexity": self.complexity,
        }


def _is_clojure_file(path: str) -> bool:
    """Check if a file path is a Clojure source file."""
    for ext in (".clj", ".cljs", ".cljc"):
        if path.endswith(ext):
            # Exclude test files by convention
            return not _is_test_file(path)
    return False


def _is_test_file(path: str) -> bool:
    return "test/" in path or path.endswith("_test.clj") or path.endswith("_test.cljc")


def _read_file_safe(path: Path, max_lines: int = 300) -> Optional[str]:
    """Read a file safely, returning None if unreadable or too large."""
    try:
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()
        if len(lines) > max_lines:
            # Take head + tail to capture ns form and body
            head = "\n".join(lines[:50])
            tail = "\n".join(lines[-max_lines + 50:])
            return f"{head}\n\n;; ... {len(lines) - max_lines} lines omitted ...\n\n{tail}"
        return content
    except Exception:
        return None


def collect_clojure_files(
    repo_path: str,
    exclude_patterns: Optional[List[str]] = None,
) -> List[Path]:
    """Walk a repo and collect all Clojure source files (excluding tests)."""
    if exclude_patterns is None:
        exclude_patterns = [
            "test/", "tests/", "_test.clj", "_test.cljs", "_test.cljc",
            "resources/", "target/", "node_modules/", ".git/",
        ]

    root = Path(repo_path)
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if not _is_clojure_file(str(path)):
            continue
        rel = str(path.relative_to(root))
        if any(p in rel for p in exclude_patterns):
            continue
        files.append(path)

    return sorted(files)


def extract_features_from_files(
    file_paths: List[Path],
    root_path: str,
    llm: LLMProvider,
    batch_size: int = 3,
) -> List[ClojureFeature]:
    """Extract Clojure features from a list of source files using LLM.

    Files are batched to fit within context windows.
    """
    all_features = []
    root = Path(root_path)

    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i + batch_size]
        batch_content = _build_batch_prompt(batch, root)

        if not batch_content.strip():
            continue

        try:
            result = llm.call(
                system_prompt=_EXTRACT_SYSTEM_PROMPT,
                user_prompt=batch_content,
                temperature=0.1,
                max_tokens=4096,
                require_json=True,
            )
            features = _parse_features(result, root)
            all_features.extend(features)
        except Exception as e:
            # Log and continue — don't lose all progress for one batch failure
            import logging
            logging.getLogger(__name__).warning(
                f"Feature extraction batch failed (files {i}-{i + batch_size}): {e}"
            )

    return all_features


def _build_batch_prompt(files: List[Path], root: Path) -> str:
    """Build a prompt with multiple Clojure source files."""
    parts = []
    for path in files:
        content = _read_file_safe(path)
        if content is None:
            continue
        rel_path = path.relative_to(root)
        parts.append(f"### {rel_path}\n```clojure\n{content}\n```")

    if not parts:
        return ""

    return (
        "Extract all Clojure-specific code features from the following source files. "
        "For each feature, include the file path, feature type, name, "
        "description, line hint, and complexity.\n\n" +
        "\n".join(parts)
    )


def _parse_features(result: str, root: Path) -> List[ClojureFeature]:
    """Parse LLM JSON response into ClojureFeature list."""
    if isinstance(result, dict):
        items = result.get("features", result.get("data", [result]))
        if isinstance(items, dict):
            items = [items]
    elif isinstance(result, str):
        try:
            items = json.loads(result)
            if isinstance(items, dict):
                items = [items]
        except json.JSONDecodeError:
            # Try extracting from markdown code block
            match = re.search(r'```(?:json)?\s*\n?(.*?)```', result, re.DOTALL)
            if match:
                try:
                    items = json.loads(match.group(1))
                except json.JSONDecodeError:
                    return []
            else:
                return []
    else:
        return []

    if not isinstance(items, list):
        return []

    features = []
    for item in items:
        feat = ClojureFeature.from_dict(item)
        features.append(feat)

    return features


def extract_features_from_repo(
    repo_path: str,
    llm: LLMProvider,
    max_files: int = 50,
    sample_strategy: str = "first",
) -> List[ClojureFeature]:
    """Extract features from a full repository.

    Args:
        repo_path: Path to the Clojure repository.
        llm: LLM provider instance.
        max_files: Maximum number of source files to analyze.
        sample_strategy: "first", "random", or "largest".

    Returns:
        List of extracted ClojureFeature objects.
    """
    import random

    all_files = collect_clojure_files(repo_path)
    if not all_files:
        return []

    if sample_strategy == "random":
        random.shuffle(all_files)
        selected = all_files[:max_files]
    elif sample_strategy == "largest":
        selected = sorted(
            all_files,
            key=lambda f: f.stat().st_size,
            reverse=True,
        )[:max_files]
    else:
        selected = all_files[:max_files]

    return extract_features_from_files(selected, repo_path, llm)
