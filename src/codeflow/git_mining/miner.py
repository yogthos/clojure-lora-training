"""Mine Clojure git repositories for before/after code training pairs.

Usage:
    examples = mine_repository("/path/to/repo", repo_name="my-project")
    for ex in examples:
        print(ex.to_jsonl())
"""

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .commit_filter import (
    CommitInfo,
    filter_clojure_commits,
    has_meaningful_message,
    is_clojure_file,
)
from .diff_parser import parse_diff

from ...shared import _SYSTEM_PROMPT

# The well-known SHA of git's empty tree, used as the base for a repo's root
# commit (which has no parent) so its arc still produces a diff.
_EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

_CLOJURE_PATHSPECS = ["*.clj", "*.cljs", "*.cljc", "*.edn"]


@dataclass
class MinedExample:
    """A single training example mined from git history."""
    repo_name: str
    instruction: str
    before: Dict[str, str]  # filename -> content
    after: Dict[str, str]  # filename -> content
    diff: str
    changed_files: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to LLaMA-Factory compatible format."""
        return {
            "system": _SYSTEM_PROMPT,
            "instruction": self.instruction,
            "input": _format_file_tree(self.before),
            "output": _format_output_with_repl(
                self.diff, self.instruction, self.before, self.after
            ),
        }

    def to_jsonl(self) -> str:
        """Serialize as a single JSONL line."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


def _format_file_tree(files: Dict[str, str]) -> str:
    """Format a file tree into a readable multi-file context block."""
    parts = []
    for path, content in sorted(files.items()):
        parts.append(f"### {path}\n```clojure\n{content}\n```")
    return "\n\n".join(parts)


def _format_output_with_repl(
    diff: str,
    instruction: str,
    files_before: Dict[str, str],
    files_after: Dict[str, str],
) -> str:
    """Format output with nREPL session and diff.

    The output structure mirrors nREPL-driven development:
    1. REPL exploration (eval forms, inspect, iterate)
    2. Apply final changes as unified diff
    """
    parts = []

    # REPL session header
    parts.append(";; nREPL session:")
    parts.append(";; Evaluate and test changes interactively before applying to files.")
    parts.append("")

    # Generate REPL steps from the diff changes
    repl_steps = _extract_repl_steps(files_before, files_after)
    if repl_steps:
        parts.extend(repl_steps)
        parts.append("")

    # Apply section
    parts.append(";; apply:")
    parts.append(diff)

    return "\n".join(parts)


def _extract_repl_steps(
    before: Dict[str, str],
    after: Dict[str, str],
) -> List[str]:
    """Extract plausible REPL evaluation steps from before/after file diffs.

    Identifies new/changed top-level forms and generates ;; eval: blocks.
    Falls back to generic interactive-development steps when no new forms
    are detected but content changed.
    """
    steps = []
    any_change = False

    for path in sorted(after):
        before_content = before.get(path, "")
        after_content = after.get(path, "")

        if not before_content:
            any_change = True
            for form in _top_level_forms(after_content):
                steps.append(f";; eval: {form}")
                steps.append(";; result: ;; => defined")
            continue

        if before_content != after_content:
            any_change = True
            new_forms = [
                f for f in _top_level_forms(after_content)
                if f not in _top_level_forms(before_content)
            ]
            if new_forms:
                for form in new_forms:
                    steps.append(f";; eval: {form}")
                    steps.append(";; result: ;; => defined")
            else:
                # Content changed but same top-level forms (e.g., body edits).
                # Show evaluation of the changed function.
                changed_fns = _find_changed_functions(before_content, after_content)
                if changed_fns:
                    for fn_name in changed_fns:
                        steps.append(f";; eval: ({fn_name} <args>)\n;; Test the updated function interactively")
                        steps.append(";; result: ;; => (inspect result, iterate if needed)")
                else:
                    steps.append(f";; eval: <evaluate changed forms in {path}>\n;; Explore interactively with the REPL")
                    steps.append(";; result: ;; => (inspect and refine)")
        elif path not in after:
            any_change = True

    return steps


def _find_changed_functions(
    before_content: str,
    after_content: str,
) -> List[str]:
    """Find function names whose definitions differ between before and after."""
    before_fns = set()
    after_fns = set()
    for form in _top_level_forms(before_content):
        name = _extract_def_name(form)
        if name:
            before_fns.add(name)
    for form in _top_level_forms(after_content):
        name = _extract_def_name(form)
        if name:
            after_fns.add(name)

    # Return names present in both — they may have had body changes
    common = before_fns & after_fns
    return sorted(common) if common else sorted(after_fns - before_fns)


def _extract_def_name(form: str) -> str | None:
    """Extract the name from a def-like form. E.g., (defn parse [args] -> parse"""
    tok = form.replace("(", "").split()
    if len(tok) >= 3:
        return tok[1]
    return None


def _top_level_forms(content: str) -> List[str]:
    """Extract top-level forms from Clojure source.

    Simple heuristic: lines starting with '(' at column 0 that contain 'defn',
    'def', 'defmacro', etc.
    """
    forms = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("(") and not stripped.startswith("(ns "):
            # Keep defn, def, defmacro, defprotocol, defmulti as top-level
            if any(
                stripped.startswith(f"({kw} ")
                for kw in ("defn", "def", "defmacro", "defprotocol", "defmulti",
                           "defmethod", "defrecord", "deftype", "defonce")
            ):
                forms.append(stripped)
    return forms


def get_commit_list(
    repo_path: str,
    max_count: int | None = 1000,
    since: str | None = None,
) -> List[CommitInfo]:
    """Get list of commits from a git repository (newest first).

    max_count=None walks the full history (needed for lifecycle windowing).
    """
    cmd = [
        "git", "-C", repo_path, "log",
        "--pretty=format:%H%x00%P%x00%aI%x00%s",
        "--name-only",
    ]
    if max_count is not None:
        cmd.insert(4, f"--max-count={max_count}")
    if since:
        cmd.append(f"--since={since}")

    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        return []

    commits = []
    current_hash = None
    current_parents = None
    current_ts = None
    current_msg = None
    current_files = []

    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        if "\x00" in line:
            # Flush previous commit
            if current_hash is not None:
                commits.append(CommitInfo(
                    hash=current_hash,
                    message=current_msg or "",
                    files=[f for f in current_files if f],
                    is_merge=" " in (current_parents or ""),
                    timestamp=current_ts or "",
                ))
            parts = line.split("\x00")
            current_hash = parts[0]
            current_parents = parts[1] if len(parts) > 1 else ""
            current_ts = parts[2] if len(parts) > 2 else ""
            current_msg = parts[3] if len(parts) > 3 else ""
            current_files = []
        else:
            current_files.append(line.strip())

    # Flush final commit
    if current_hash is not None:
        commits.append(CommitInfo(
            hash=current_hash,
            message=current_msg or "",
            files=[f for f in current_files if f],
            is_merge=" " in (current_parents or ""),
            timestamp=current_ts or "",
        ))

    return commits


def get_commit_diff(repo_path: str, commit_hash: str) -> str:
    """Get the unified diff for a single commit."""
    cmd = [
        "git", "-C", repo_path, "show",
        "--no-ext-diff",  # ignore a user's diff.external (e.g. difftastic)
        "--format=",  # suppress commit info
        "--unified=3",
        commit_hash,
        "--", ".",  # only tracked files
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def get_file_content(repo_path: str, commit_hash: str, filepath: str) -> str:
    """Get file content at a specific commit."""
    cmd = ["git", "-C", repo_path, "show", f"{commit_hash}:{filepath}"]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        return ""
    return result.stdout


def get_range_diff(
    repo_path: str,
    base_ref: str,
    head_ref: str,
    pathspecs: Optional[List[str]] = None,
) -> str:
    """Cumulative unified diff between two refs (R_old -> R_new).

    Unlike get_commit_diff (single commit), this spans an arc of development.
    pathspecs restrict the diff to matching files (git globs match '/').
    """
    cmd = ["git", "-C", repo_path, "diff", "--no-ext-diff", "--unified=3",
           base_ref, head_ref]
    if pathspecs:
        cmd.append("--")
        cmd.extend(pathspecs)
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _base_ref(repo_path: str, commit_hash: str) -> str:
    """Parent of a commit, or git's empty tree if it's the root commit."""
    result = subprocess.run(
        ["git", "-C", repo_path, "rev-parse", "--verify", "-q", f"{commit_hash}^"],
        capture_output=True, text=True, errors="replace",
    )
    ref = result.stdout.strip()
    return ref if result.returncode == 0 and ref else _EMPTY_TREE_SHA


def get_lifecycle_commits(
    repo_path: str,
    low: float = 0.4,
    high: float = 0.8,
    max_commits: int | None = None,
    since: str | None = None,
) -> List[CommitInfo]:
    """Commits within the [low, high] percentile band of project lifecycle.

    IQuest §3.1: the 40-80% band is the mature/stable development phase —
    avoiding both early-project churn and late-stage fragmented maintenance.
    Returned oldest-first (chronological).
    """
    chrono = list(reversed(get_commit_list(repo_path, max_count=None, since=since)))
    n = len(chrono)
    if n == 0:
        return []
    lo = int(low * n)
    hi = int(high * n)
    window = chrono[lo:hi]
    if max_commits and len(window) > max_commits:
        window = window[:max_commits]
    return window


def _synthesize_instruction(messages: List[str]) -> str:
    """Combine an arc's commit messages into a single instruction.

    Keeps the substantive first line of each commit, drops merge/trivial
    messages, and dedupes. The arc's combined intent is what the patch
    accomplishes from R_old to R_new.
    """
    lines: List[str] = []
    for msg in messages:
        first = (msg or "").strip().splitlines()[0].strip() if msg else ""
        if not first:
            continue
        if first.startswith("Merge pull request") or first.startswith("Merge branch"):
            continue
        if not has_meaningful_message(first):
            continue
        if first not in lines:
            lines.append(first)
    return "; ".join(lines)


def mine_repository(
    repo_path: str,
    repo_name: str = "",
    max_commits: int = 500,
    since: str | None = None,
    lifecycle_window: Optional[tuple[float, float]] = (0.4, 0.8),
    triplet_span: int = 3,
) -> List[MinedExample]:
    """Mine a Clojure repository for code-flow (R_old, P, R_new) triplets.

    Follows the IQuest-Coder git-history recipe (Tech Report §3.1):
    1. Select commits from the project's mature phase via ``lifecycle_window``
       (default 40-80% percentile), not the most-recent N from HEAD.
    2. Walk the Clojure-relevant commits in non-overlapping arcs of
       ``triplet_span`` commits, forming a cumulative diff for each arc — a
       multi-iteration development span rather than a single commit.
    3. Keep only arcs with a non-empty Clojure diff and a meaningful combined
       instruction (endpoint quality filtering).

    triplet_span=1 reproduces single-commit granularity. lifecycle_window=None
    walks the full history (capped by max_commits).
    """
    if not repo_name:
        repo_name = Path(repo_path).name

    # Step 1: select commits (chronological, oldest-first).
    if lifecycle_window is not None:
        low, high = lifecycle_window
        commits = get_lifecycle_commits(
            repo_path, low=low, high=high, max_commits=max_commits, since=since
        )
    else:
        commits = list(reversed(
            get_commit_list(repo_path, max_count=max_commits, since=since)
        ))

    filtered = filter_clojure_commits(commits)

    # Step 2: walk non-overlapping arcs, forming a cumulative triplet each.
    span = max(1, triplet_span)
    examples: List[MinedExample] = []
    for i in range(0, len(filtered), span):
        arc = filtered[i:i + span]
        if not arc:
            continue
        start, end = arc[0], arc[-1]
        base_ref = _base_ref(repo_path, start.hash)
        head_ref = end.hash

        diff_text = get_range_diff(
            repo_path, base_ref, head_ref, pathspecs=_CLOJURE_PATHSPECS
        )
        if not diff_text:
            continue

        parsed = [df for df in parse_diff(diff_text) if is_clojure_file(df.path)]
        if not parsed:
            continue

        before_state: Dict[str, str] = {}
        after_state: Dict[str, str] = {}
        for df in parsed:
            if df.change_type == "deleted":
                before_state[df.path] = get_file_content(repo_path, base_ref, df.path)
            elif df.change_type == "added":
                after_state[df.path] = get_file_content(repo_path, head_ref, df.path)
            else:
                before_state[df.path] = get_file_content(repo_path, base_ref, df.path)
                after_state[df.path] = get_file_content(repo_path, head_ref, df.path)

        if not before_state and not after_state:
            continue

        # Step 3: endpoint quality — meaningful combined instruction.
        instruction = _synthesize_instruction([c.message for c in arc])
        if not instruction:
            continue

        examples.append(MinedExample(
            repo_name=repo_name,
            instruction=instruction,
            before=before_state,
            after=after_state,
            diff=diff_text,
            changed_files=[df.path for df in parsed],
        ))

    return examples
