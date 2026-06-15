"""Parse unified diff output into structured data."""

from dataclasses import dataclass, field
import re
from typing import List


@dataclass
class DiffHunk:
    """A single hunk within a file diff."""
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[str] = field(default_factory=list)


@dataclass
class DiffFile:
    """A single file's changes from a diff."""
    path: str
    change_type: str = ""  # "added", "modified", "deleted", "renamed"
    old_path: str = ""
    hunks: List[DiffHunk] = field(default_factory=list)


# Regex patterns
_DIFF_HEADER = re.compile(r'^diff --git a/(.+) b/(.+)$')
_OLD_MODE = re.compile(r'^(old mode|deleted file mode|index)')
_NEW_MODE = re.compile(r'^(new mode|new file mode|index)')
_RENAME_FROM = re.compile(r'^rename from (.+)$')
_RENAME_TO = re.compile(r'^rename to (.+)$')
_HUNK_HEADER = re.compile(r'^@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@')


def parse_diff(diff_text: str) -> List[DiffFile]:
    """Parse a unified diff string into structured DiffFile objects.

    Returns empty list for empty input or text with no diff headers.
    """
    if not diff_text or not diff_text.strip():
        return []

    files: List[DiffFile] = []
    current_file: DiffFile | None = None
    current_hunk: DiffHunk | None = None
    rename_src: str | None = None

    for line in diff_text.splitlines():
        # File header: diff --git a/path b/path
        m = _DIFF_HEADER.match(line)
        if m:
            if current_file is not None:
                if current_file.hunks:
                    if not current_file.change_type:
                        current_file.change_type = "modified"
                    files.append(current_file)
                current_file = None
            current_file = DiffFile(path=m.group(2), old_path=m.group(1))
            current_hunk = None
            continue

        if current_file is None:
            continue

        # Rename tracking
        rm = _RENAME_FROM.match(line)
        if rm:
            rename_src = rm.group(1)
            continue
        rm = _RENAME_TO.match(line)
        if rm:
            if rename_src:
                current_file.change_type = "renamed"
                current_file.old_path = rename_src
                rename_src = None
            continue

        # Skip index/mode lines
        if _OLD_MODE.match(line) or _NEW_MODE.match(line):
            continue

        # --- and +++ lines
        if line.startswith('--- '):
            if '/dev/null' in line:
                current_file.change_type = "added"
            continue
        if line.startswith('+++ '):
            if '/dev/null' in line:
                current_file.change_type = "deleted"
            continue

        # Hunk header: @@ -old_start,old_count +new_start,new_count @@
        hm = _HUNK_HEADER.match(line)
        if hm:
            old_start = int(hm.group(1))
            old_count = int(hm.group(2)) if hm.group(2) else 1
            new_start = int(hm.group(3))
            new_count = int(hm.group(4)) if hm.group(4) else 1
            current_hunk = DiffHunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
            )
            current_file.hunks.append(current_hunk)
            continue

        # Content lines within hunk
        if current_hunk is not None:
            current_hunk.lines.append(line)

    # Append final file
    if current_file is not None and current_file.hunks:
        if not current_file.change_type:
            current_file.change_type = "modified"
        files.append(current_file)

    return files
