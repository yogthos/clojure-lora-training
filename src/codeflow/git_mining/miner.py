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
from .context import (HistoryEntry, file_history, format_history, format_related,
                      related_forms)
from .forms import top_level_form_spans as _top_level_form_spans
from .gitcmd import _DIFF_FLAGS, _GIT, get_file_content

from ...shared import _TRANSITION_SYSTEM_PROMPT

# The well-known SHA of git's empty tree, used as the base for a repo's root
# commit (which has no parent) so its arc still produces a diff.
_EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

_CLOJURE_PATHSPECS = ["*.clj", "*.cljs", "*.cljc", "*.edn"]

# A chain's turns, human and assistant, stay within this many characters
# (~11k tokens) so it fits the training cutoff with its reasoning added.
CHAIN_CHARS = 40000

# Changed files are shown whole, smallest first, while they fit in this many
# characters (~4.5k tokens); the rest show only the changed forms.
WHOLE_FILE_BUDGET = 16000

@dataclass
class MinedExample:
    """A single training example mined from git history."""
    repo_name: str
    instruction: str
    before: Dict[str, str]  # filename -> content
    after: Dict[str, str]  # filename -> content
    diff: str
    changed_files: List[str] = field(default_factory=list)
    history: List[HistoryEntry] = field(default_factory=list)
    related: Dict[str, str] = field(default_factory=dict)   # path -> forms using the change
    # Where the arc sits in history, for linking arcs into chains.
    base_ref: str = field(default="", repr=False)
    head_ref: str = field(default="", repr=False)
    old_paths: List[str] = field(default_factory=list, repr=False)   # changed files at base_ref

    def to_dict(self) -> dict:
        """Convert to LLaMA-Factory compatible format.

        The example is a code-flow transition: given the before-state files and
        an instruction, the target output is the unified diff that applies the
        change. Git history has no REPL trace, so no eval blocks are fabricated.
        """
        return {
            "system": _TRANSITION_SYSTEM_PROMPT,
            "instruction": self.instruction,
            "input": self.input_text(),
            "output": self.diff,
            # Not a training field: groups records by repo for the train/val split.
            "repo": self.repo_name,
        }

    def input_text(self, with_history: bool = True) -> str:
        """Recent history, then the code being changed, then the code that
        uses it."""
        code = _format_changed_regions(self.before, self.diff, whole_budget=WHOLE_FILE_BUDGET)
        history = format_history(self.history) if with_history else ""
        parts = [history, code, format_related(self.related)]
        return "\n\n".join(p for p in parts if p)

    def human_turn(self, with_history: bool = True) -> str:
        """The request as the model sees it: instruction, then input (how
        LLaMA-Factory joins an alpaca record's fields)."""
        return f"{self.instruction}\n{self.input_text(with_history)}"

    def to_jsonl(self) -> str:
        """Serialize as a single JSONL line."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class MinedChain:
    """Consecutive arcs on the same files, as one conversation: each request
    is shown with the code as the previous change left it, so the model
    learns to build each change on the one before. Later turns leave out the
    commit history, which the conversation itself now carries."""
    repo_name: str
    turns: List[MinedExample]

    def conversations(self) -> List[dict]:
        conv = []
        for i, t in enumerate(self.turns):
            conv.append({"from": "human", "value": t.human_turn(with_history=i == 0)})
            conv.append({"from": "gpt", "value": t.diff})
        return conv

    def to_dict(self) -> dict:
        first = self.turns[0]
        conv = self.conversations()
        # The flat fields mirror the first turn: dedup and the grouped split
        # key on them.
        return {
            "system": _TRANSITION_SYSTEM_PROMPT,
            "instruction": first.instruction,
            "input": first.input_text(),
            "output": conv[1]["value"],
            "conversations": conv,
            "repo": self.repo_name,
        }

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def _turn_chars(ex: MinedExample, first: bool) -> int:
    return len(ex.human_turn(with_history=first)) + len(ex.diff)


def _continues(repo_path: str, prev: MinedExample, nxt: MinedExample) -> bool:
    """``nxt`` changes a file ``prev`` changed, and finds every file it
    changes exactly as ``prev`` left it (nothing unshown happened between)."""
    if not set(prev.changed_files) & set(nxt.old_paths):
        return False
    if prev.head_ref == nxt.base_ref:
        return True
    result = subprocess.run(
        [*_GIT, "-C", repo_path, "diff", "--quiet", prev.head_ref, nxt.base_ref,
         "--", *[_literal(p) for p in nxt.old_paths]],
        capture_output=True)
    return result.returncode == 0


def link_chains(repo_path: str, examples: List[MinedExample], chain_max: int,
                chain_chars: int) -> List["MinedExample | MinedChain"]:
    """Group consecutive continuing arcs into chains of up to ``chain_max``
    turns and ``chain_chars`` characters; an arc that links to nothing stays
    a single transition."""
    out: list = []
    run: List[MinedExample] = []
    size = 0

    def flush():
        if len(run) == 1:
            out.append(run[0])
        elif run:
            out.append(MinedChain(run[0].repo_name, list(run)))

    for ex in examples:
        if (run and len(run) < chain_max and _continues(repo_path, run[-1], ex)
                and size + _turn_chars(ex, False) <= chain_chars):
            run.append(ex)
            size += _turn_chars(ex, False)
        else:
            flush()
            run, size = [ex], _turn_chars(ex, True)
    flush()
    return out


def _format_file_tree(files: Dict[str, str]) -> str:
    """Format a file tree into a readable multi-file context block."""
    parts = []
    for path, content in sorted(files.items()):
        parts.append(f"### {path}\n```clojure\n{content}\n```")
    return "\n\n".join(parts)


def _format_changed_regions(before: Dict[str, str], diff: str, whole_budget: int = 0) -> str:
    """Render the before-state showing only the changed top-level forms.

    Files are shown whole instead, smallest first, while their total size
    stays within ``whole_budget`` characters.

    For each file, keep the forms whose lines overlap a diff hunk's old range,
    plus the ``ns`` form for context. Non-adjacent kept forms are separated by a
    ``;; ...`` elision marker. This bounds the input to the functions actually
    being edited instead of inlining whole files.
    """
    ranges_by_path: Dict[str, List[tuple[int, int]]] = {}
    for f in parse_diff(diff):
        rs = ranges_by_path.setdefault(f.path, [])
        for h in f.hunks:
            count = h.old_count if h.old_count else 1
            rs.append((h.old_start, h.old_start + count - 1))

    whole, used = set(), 0
    for path in sorted(before, key=lambda p: len(before[p])):
        if used + len(before[path]) > whole_budget:
            break
        whole.add(path)
        used += len(before[path])

    parts = []
    for path in sorted(before):
        content = before[path]
        if path in whole:
            parts.append(f"### {path}\n```clojure\n{content.rstrip()}\n```")
            continue
        ranges = ranges_by_path.get(path)
        spans = _top_level_form_spans(content) if ranges else []
        kept = set()
        for (rs, re_) in ranges or []:
            hit = [sp for sp in spans if sp[0] <= re_ and rs <= sp[1]]
            if not hit:
                # The hunk sits in a gap between forms (e.g. inserting a new
                # form): show the forms on either side as its anchor.
                before_gap = [sp for sp in spans if sp[1] < rs]
                after_gap = [sp for sp in spans if sp[0] > re_]
                hit = before_gap[-1:] + after_gap[:1]
            kept.update(hit)
        kept.update(sp for sp in spans if sp[2].lstrip().startswith("(ns "))
        if not kept:
            body = content  # fallback: no ranges matched or unparseable
        else:
            lines = content.split("\n")
            chunks, prev_end = [], None
            for (s, e, text) in sorted(kept):
                if prev_end is not None:
                    gap = lines[prev_end:s - 1]
                    if any(l.strip() for l in gap):
                        chunks.append(";; ...")  # something was left out
                    else:
                        chunks.extend(gap)  # just blank lines: keep spacing
                chunks.append(text)
                prev_end = e
            body = "\n".join(chunks)
        parts.append(f"### {path}\n```clojure\n{body}\n```")
    return "\n\n".join(parts)


def get_commit_list(
    repo_path: str,
    max_count: int | None = 1000,
    since: str | None = None,
) -> List[CommitInfo]:
    """Get list of commits from a git repository (newest first).

    max_count=None walks the full history (needed for lifecycle windowing).
    """
    cmd = [
        *_GIT, "-C", repo_path, "log",
        "--first-parent",  # mainline only: side-branch work arrives via its merge
        "--pretty=format:%H%x00%P%x00%aI%x00%s",
        "--name-only",
    ]
    if max_count is not None:
        cmd.append(f"--max-count={max_count}")
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
        *_GIT, "-C", repo_path, "show",
        *_DIFF_FLAGS,  # also ignores a user's diff.external (e.g. difftastic)
        "--format=",  # suppress commit info
        commit_hash,
        "--", ".",  # only tracked files
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        return ""
    # Don't strip: trailing ' ' lines are blank context the hunk counts include.
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
    cmd = [*_GIT, "-C", repo_path, "diff", *_DIFF_FLAGS, base_ref, head_ref]
    if pathspecs:
        cmd.append("--")
        cmd.extend(pathspecs)
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        return ""
    # Don't strip: trailing ' ' lines are blank context the hunk counts include.
    return result.stdout


def _base_ref(repo_path: str, commit_hash: str) -> str:
    """Parent of a commit, or git's empty tree if it's the root commit."""
    result = subprocess.run(
        [*_GIT, "-C", repo_path, "rev-parse", "--verify", "-q", f"{commit_hash}^"],
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


# A GNU-changelog-style entry: optional "* ", a path token (contains "/" or a
# source extension), then ": <description>". We keep only the description so the
# instruction reads like a request, not a raw commit log with embedded paths.
_CHANGELOG_RE = re.compile(r"^\*?\s*([\w./\-]+)\s*:\s*(.*)$")
_SRC_EXTS = (".clj", ".cljs", ".cljc", ".cljx", ".edn")


def _clean_changelog_prefix(line: str) -> str:
    """Strip a leading ``* path:`` changelog prefix, keeping the description.

    "* src/foo/bar.clj: add retry" -> "add retry". A normal "Title: detail"
    message (no path before the colon) is returned unchanged.
    """
    m = _CHANGELOG_RE.match(line)
    if not m:
        return line
    head, rest = m.group(1), m.group(2).strip()
    looks_like_path = "/" in head or head.endswith(_SRC_EXTS)
    if looks_like_path:
        return rest
    return line


def clean_instruction(instruction: str) -> str:
    """Re-clean an already-synthesized instruction string.

    Instructions are stored as ``"; "``-joined commit subjects. This re-applies
    the changelog-prefix cleaning to each segment so data mined before the fix
    can be repaired at assembly time without re-mining the repositories.
    """
    segs = [_clean_changelog_prefix(s.strip()) for s in instruction.split(";")]
    segs = [s.strip() for s in segs if s and s.strip()]
    return "; ".join(segs)


def _synthesize_instruction(messages: List[str]) -> str:
    """Combine an arc's commit messages into a single instruction.

    Keeps the substantive first line of each commit, strips changelog path
    prefixes, drops merge/trivial messages, and dedupes. The arc's combined
    intent is what the patch accomplishes from R_old to R_new.
    """
    lines: List[str] = []
    for msg in messages:
        first = (msg or "").strip().splitlines()[0].strip() if msg else ""
        if not first:
            continue
        if first.startswith("Merge pull request") or first.startswith("Merge branch"):
            continue
        first = _clean_changelog_prefix(first)
        if not first:
            continue
        if not has_meaningful_message(first):
            continue
        if first not in lines:
            lines.append(first)
    return "; ".join(lines)


def _build_arcs(commits: List[CommitInfo], span: int) -> List[List[CommitInfo]]:
    """Split chronological first-parent commits into arcs of up to ``span``.

    An arc is a run of consecutive accepted commits, so its cumulative diff
    contains exactly the changes its instruction describes. A merge, or a
    rejected commit that touches Clojure code (e.g. "wip"), ends the current
    run. Commits touching no Clojure code are skipped without breaking it —
    they can't contribute to the arc's Clojure-only diff.
    """
    accepted = {c.hash for c in filter_clojure_commits(commits)}
    arcs: List[List[CommitInfo]] = []
    run: List[CommitInfo] = []
    for c in commits:
        if c.hash in accepted:
            run.append(c)
            if len(run) == span:
                arcs.append(run)
                run = []
        elif c.is_merge or any(is_clojure_file(f) for f in c.files):
            if run:
                arcs.append(run)
            run = []
    if run:
        arcs.append(run)
    return arcs


def _literal(path: str) -> str:
    return f":(literal){path}"


def mine_repository(
    repo_path: str,
    repo_name: str = "",
    max_commits: int = 500,
    since: str | None = None,
    lifecycle_window: Optional[tuple[float, float]] = (0.4, 0.8),
    triplet_span: int = 3,
    chain_max: int = 1,
    chain_chars: int = CHAIN_CHARS,
) -> List["MinedExample | MinedChain"]:
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
    walks the full history (capped by max_commits). chain_max > 1 links
    consecutive arcs on the same files into MinedChains (see link_chains).
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

    # Step 2: walk arcs of consecutive accepted commits, a cumulative triplet each.
    span = max(1, triplet_span)
    examples: List[MinedExample] = []
    for arc in _build_arcs(commits, span):
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

        # The pathspec globs also match build/config files (project.clj,
        # deps.edn, resources/*.edn) that the input never shows. Re-diff over
        # exactly the shown source files so the target touches nothing else.
        shown = []
        for df in parsed:
            shown.append(df.path)
            if df.change_type == "renamed" and df.old_path:
                shown.append(df.old_path)  # both sides, so the rename is detected
        headers = sum(1 for l in diff_text.split("\n") if l.startswith("diff --git "))
        if headers != len(parsed):
            diff_text = get_range_diff(
                repo_path, base_ref, head_ref,
                pathspecs=[_literal(p) for p in dict.fromkeys(shown)],
            )
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
                # A renamed file only exists under its old path at base_ref.
                old = df.old_path if df.change_type == "renamed" and df.old_path else df.path
                before_state[df.path] = get_file_content(repo_path, base_ref, old)
                after_state[df.path] = get_file_content(repo_path, head_ref, df.path)

        if not before_state and not after_state:
            continue

        history, related = [], {}
        if base_ref != _EMPTY_TREE_SHA:
            old_paths = [df.old_path if df.change_type == "renamed" and df.old_path else df.path
                         for df in parsed if df.change_type != "added"]
            history = file_history(repo_path, base_ref, old_paths)
            related = related_forms(repo_path, base_ref, before_state, diff_text)

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
            history=history,
            related=related,
            base_ref=base_ref,
            head_ref=head_ref,
            old_paths=[df.old_path if df.change_type == "renamed" and df.old_path else df.path
                       for df in parsed if df.change_type != "added"],
        ))

    if chain_max > 1:
        return link_chains(repo_path, examples, chain_max, chain_chars)
    return examples
