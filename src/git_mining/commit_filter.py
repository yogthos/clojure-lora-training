"""Filter commits for Clojure-relevant training data."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class CommitInfo:
    """Minimal commit metadata for filtering."""
    hash: str
    message: str
    files: List[str] = field(default_factory=list)
    is_merge: bool = False


# Files that are build/config, not meaningful code changes
_NON_CODE_FILES = {
    "project.clj", "build.boot", "deps.edn", "build.clj",
    ".github/", "Jenkinsfile", "Dockerfile", "Makefile",
}

# Path prefixes/patterns that indicate config/docs, not source code
_NON_CODE_PREFIXES = ("resources/", "doc/", "docs/", "test-resources/", "dev-resources/")

# Extensions we consider for training
_CLOJURE_EXTENSIONS = {".clj", ".cljs", ".cljc", ".edn"}

# Minimum message length and quality heuristics
_MIN_MESSAGE_WORDS = 3


def is_clojure_file(path: str) -> bool:
    """Check if a file path is a Clojure source file (not build config)."""
    # Exclude build/config files by name
    for non_code in _NON_CODE_FILES:
        if path.endswith(non_code) or path.startswith(non_code):
            return False
    # Exclude config/doc paths
    for prefix in _NON_CODE_PREFIXES:
        if path.startswith(prefix):
            return False
    # Check extension
    for ext in _CLOJURE_EXTENSIONS:
        if path.endswith(ext):
            return True
    return False


def has_meaningful_message(message: str) -> bool:
    """Check if a commit message is substantive enough for training."""
    msg = message.strip()
    words = msg.split()
    if len(words) < _MIN_MESSAGE_WORDS:
        return False
    # Exclude trivial one-word-ish messages
    if len(msg) < 10:
        return False
    return True


def _has_clojure_code_files(files: List[str]) -> bool:
    """Check if any file in the commit is a Clojure code file."""
    return any(is_clojure_file(f) for f in files)


def filter_clojure_commits(commits: List[CommitInfo]) -> List[CommitInfo]:
    """Filter commits to only those useful for Clojure code LoRA training.

    Excludes:
    - Merge commits
    - Non-Clojure commits (no .clj/.cljs/.cljc/.edn files)
    - Trivial messages ("fix", "wip", etc.)
    - Build/config-only changes
    """
    result = []
    for c in commits:
        if c.is_merge:
            continue
        if not _has_clojure_code_files(c.files):
            continue
        if not has_meaningful_message(c.message):
            continue
        result.append(c)
    return result
