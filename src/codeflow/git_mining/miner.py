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

from .commit_filter import CommitInfo, filter_clojure_commits
from .diff_parser import parse_diff
from .session_grouper import CommitWithDiff, group_by_pr_boundary

from ...shared import _SYSTEM_PROMPT


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
    max_count: int = 1000,
    since: str | None = None,
) -> List[CommitInfo]:
    """Get list of commits from a git repository."""
    cmd = [
        "git", "-C", repo_path, "log",
        f"--max-count={max_count}",
        "--pretty=format:%H%x00%P%x00%aI%x00%s",
        "--name-only",
    ]
    if since:
        cmd.append(f"--since={since}")

    result = subprocess.run(cmd, capture_output=True, text=True)
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
        "--format=",  # suppress commit info
        "--unified=3",
        commit_hash,
        "--", ".",  # only tracked files
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def get_file_content(repo_path: str, commit_hash: str, filepath: str) -> str:
    """Get file content at a specific commit."""
    cmd = ["git", "-C", repo_path, "show", f"{commit_hash}:{filepath}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return ""
    return result.stdout


def mine_repository(
    repo_path: str,
    repo_name: str = "",
    max_commits: int = 500,
    since: str | None = None,
) -> List[MinedExample]:
    """Mine a Clojure repository for before/after training examples.

    Pipeline:
    1. Get commit list
    2. Filter for Clojure-relevant commits
    3. Group into sessions
    4. For each commit: get diff, extract before/after state
    5. Produce MinedExample records
    """
    if not repo_name:
        repo_name = Path(repo_path).name

    # Get and filter commits
    raw_commits = get_commit_list(repo_path, max_count=max_commits, since=since)
    filtered = filter_clojure_commits(raw_commits)

    # Get diffs for filtered commits
    with_diff = []
    for commit in filtered:
        diff_text = get_commit_diff(repo_path, commit.hash)
        if not diff_text:
            continue
        parsed = parse_diff(diff_text)
        changed = [f.path for f in parsed]
        with_diff.append(CommitWithDiff(
            hash=commit.hash,
            message=commit.message,
            timestamp=commit.timestamp,
            diff_text=diff_text,
            changed_files=changed,
        ))

    # Group into sessions
    sessions = group_by_pr_boundary(with_diff)

    # Produce examples
    examples = []
    for session in sessions:
        if len(session.commits) < 1:
            continue

        # Use the session's combined changes
        for commit in session.commits:
            parsed = parse_diff(commit.diff_text)
            if not parsed:
                continue

            # Get before and after state for changed files
            before_state = {}
            after_state = {}
            for df in parsed:
                if df.change_type == "deleted":
                    before_state[df.path] = get_file_content(
                        repo_path, f"{commit.hash}~1", df.path
                    )
                elif df.change_type == "added":
                    after_state[df.path] = get_file_content(
                        repo_path, commit.hash, df.path
                    )
                else:
                    before_state[df.path] = get_file_content(
                        repo_path, f"{commit.hash}~1", df.path
                    )
                    after_state[df.path] = get_file_content(
                        repo_path, commit.hash, df.path
                    )

            if not before_state and not after_state:
                continue

            examples.append(MinedExample(
                repo_name=repo_name,
                instruction=commit.message,
                before=before_state,
                after=after_state,
                diff=commit.diff_text,
                changed_files=commit.changed_files,
            ))

    return examples
