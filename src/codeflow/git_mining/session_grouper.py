"""Group related commits into training sessions.

Commit sessions are sequences of related commits that form a coherent
development story: a feature addition, a bug fix, a refactoring.
These sessions become training examples.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class CommitSession:
    """A group of related commits forming a development narrative."""
    session_id: str
    repo_name: str
    commits: List['CommitWithDiff'] = field(default_factory=list)
    # The before state: tree of all files at the session start
    before_state: dict = field(default_factory=dict)
    # The after state: tree of all files at the session end
    after_state: dict = field(default_factory=dict)
    # Combined diff across all commits in the session
    combined_diff: str = ""
    # Extracted instruction from commit messages / PR description
    instruction: str = ""


@dataclass
class CommitWithDiff:
    """A single commit with its parsed diff."""
    hash: str
    message: str
    timestamp: str
    diff_text: str
    changed_files: List[str] = field(default_factory=list)


def group_by_pr_boundary(
    commits: List[CommitWithDiff],
) -> List[CommitSession]:
    """Group commits that belong to the same PR based on message patterns.

    A PR typically starts with the first substantive commit and ends
    before the next "Merge pull request" or after a commit with a
    PR-like summary message.
    """
    if not commits:
        return []

    sessions: List[CommitSession] = []
    current: Optional[CommitSession] = None

    for commit in commits:
        msg = commit.message.strip()

        # Merge commits signal PR boundaries
        if msg.startswith("Merge pull request") or msg.startswith("Merge branch"):
            if current is not None and current.commits:
                sessions.append(current)
                current = None
            continue

        # Start a new session if needed
        if current is None:
            current = CommitSession(
                session_id=f"session-{len(sessions):04d}",
                repo_name="",
            )

        current.commits.append(commit)

    # Flush final session
    if current is not None and current.commits:
        sessions.append(current)

    return sessions


def group_by_time_window(
    commits: List[CommitWithDiff],
    max_window_hours: int = 4,
) -> List[CommitSession]:
    """Group commits by time proximity.

    Commits within `max_window_hours` of each other are grouped into
    the same session.
    """
    if not commits:
        return []

    sessions: List[CommitSession] = []
    current: Optional[CommitSession] = None
    last_ts: Optional[datetime] = None

    for commit in commits:
        try:
            ts = _parse_timestamp(commit.timestamp)
        except ValueError:
            ts = None

        # If time gap too large, start new session
        if last_ts is not None and ts is not None:
            gap = (ts - last_ts).total_seconds() / 3600.0
            if gap > max_window_hours:
                if current is not None and current.commits:
                    sessions.append(current)
                current = None

        if current is None:
            current = CommitSession(
                session_id=f"tsession-{len(sessions):04d}",
                repo_name="",
            )

        current.commits.append(commit)
        last_ts = ts

    if current is not None and current.commits:
        sessions.append(current)

    return sessions


def group_by_prefix(
    commits: List[CommitWithDiff],
) -> List[CommitSession]:
    """Group consecutive commits with matching message prefixes.

    E.g., "WIP: refactor handler", "WIP: add tests", "WIP: cleanup"
    all get grouped together.
    """
    if not commits:
        return []

    sessions: List[CommitSession] = []
    current: Optional[CommitSession] = None
    current_prefix: Optional[str] = None

    for commit in commits:
        prefix = _extract_prefix(commit.message)

        if prefix is None:
            if current is not None and current.commits:
                sessions.append(current)
            current = None
            current_prefix = None
            continue

        if current_prefix is not None and prefix != current_prefix:
            sessions.append(current)
            current = None
            current_prefix = None

        if current is None:
            current = CommitSession(
                session_id=f"psession-{len(sessions):04d}",
                repo_name="",
            )
            current_prefix = prefix

        current.commits.append(commit)

    if current is not None and current.commits:
        sessions.append(current)

    return sessions


def _parse_timestamp(ts: str) -> datetime:
    """Parse ISO timestamp string."""
    ts = ts.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S %z",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(ts)


def _extract_prefix(message: str) -> Optional[str]:
    """Extract prefix from commit message if present.

    E.g., "WIP:" from "WIP: refactor handler",
    "[feature]" from "[feature] add validation"
    """
    msg = message.strip()
    if not msg:
        return None
    # Check for bracket prefix: [feature]
    if msg.startswith("["):
        end = msg.find("]")
        if end > 1:
            return msg[:end + 1].lower()
    # Check for colon prefix: WIP:, fix:, feat:, etc.
    colon = msg.find(":")
    if 1 <= colon <= 10:
        prefix = msg[:colon].lower()
        # Accept common prefixes, reject accidental colons
        common = {"wip", "fix", "feat", "refactor", "chore", "docs",
                  "test", "perf", "style", "build", "ci", "revert"}
        if prefix in common:
            return prefix
        # Also accept squash-style: "squash! previous message"
        if prefix.startswith("squash"):
            return prefix
    return None
