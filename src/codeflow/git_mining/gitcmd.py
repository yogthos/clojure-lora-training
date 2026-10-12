"""Git commands with output pinned against user config."""

import subprocess

# Pin the output format against user config: unquoted non-ASCII paths, no
# color, standard a/ b/ prefixes (diff.noprefix / diff.mnemonicPrefix would
# break header parsing), renames detected regardless of diff.renames.
_GIT = ["git", "-c", "core.quotePath=false", "-c", "color.ui=never"]
_DIFF_FLAGS = ["--no-color", "--no-ext-diff", "--src-prefix=a/", "--dst-prefix=b/",
               "--find-renames", "--unified=3"]


def get_file_content(repo_path: str, commit_hash: str, filepath: str) -> str:
    """Get file content at a specific commit."""
    cmd = [*_GIT, "-C", repo_path, "show", f"{commit_hash}:{filepath}"]
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        return ""
    return result.stdout
