#!/usr/bin/env python3
"""Upload local LoRA checkpoints to a HuggingFace repo, skipping ones already there.

Each local folder `checkpoints/checkpoint-<N>/` is uploaded as a subfolder
`checkpoint-<N>/` in the HF repo. A checkpoint is considered "already uploaded"
if the remote has both `adapter_config.json` and `adapter_model.safetensors`
under that subfolder AND their sizes match the local files (to catch partial
or interrupted uploads).

Usage:
    # One-time: log in (or set HF_TOKEN env var)
    huggingface-cli login

    python scripts/upload_checkpoints_to_hf.py \\
        --repo yogthos/qwen2.5-32b-lovecraft-lora \\
        --checkpoints-dir checkpoints

    # Dry run to see what would upload:
    python scripts/upload_checkpoints_to_hf.py --repo ... --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
REQUIRED_FILES = ("adapter_config.json", "adapter_model.safetensors")


def _checkpoint_num(name: str) -> int:
    m = CHECKPOINT_RE.match(name)
    assert m is not None, f"not a checkpoint dir name: {name}"
    return int(m.group(1))


def local_checkpoints(root: Path) -> list[Path]:
    dirs = [p for p in root.iterdir() if p.is_dir() and CHECKPOINT_RE.match(p.name)]
    dirs.sort(key=lambda p: _checkpoint_num(p.name))
    return dirs


def remote_file_sizes(api, repo_id: str) -> dict[str, dict[str, int]]:
    """Return {'checkpoint-N': {filename: size_bytes}} for remote checkpoint subfolders.

    Uses list_repo_tree which returns size per file (including the real size
    of LFS-tracked files, which is what we want to compare against local).
    """
    try:
        entries = api.list_repo_tree(
            repo_id=repo_id, repo_type="model", recursive=True
        )
    except Exception as e:
        print(f"Warning: could not list remote tree ({e}); assuming empty repo")
        return {}

    by_dir: dict[str, dict[str, int]] = {}
    for entry in entries:
        path = getattr(entry, "path", None)
        if not path or "/" not in path:
            continue
        head, tail = path.split("/", 1)
        if "/" in tail or not CHECKPOINT_RE.match(head):
            continue
        # Files have a size; directories do not. Skip tree entries with no size.
        size = getattr(entry, "size", None)
        if size is None:
            continue
        by_dir.setdefault(head, {})[tail] = int(size)
    return by_dir


def ensure_repo(api, repo_id: str) -> None:
    try:
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:  # older huggingface_hub
        from huggingface_hub.utils import RepositoryNotFoundError  # type: ignore[no-redef]

    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
    except RepositoryNotFoundError:
        print(f"Repo {repo_id} not found; creating it")
        api.create_repo(repo_id=repo_id, repo_type="model", private=False)


def remote_matches_local(
    local_dir: Path, remote_sizes: dict[str, int] | None
) -> tuple[bool, str]:
    """Decide whether the remote copy of this checkpoint is complete and intact.

    Returns (match, reason). match=True means: skip. match=False: upload.
    """
    if not remote_sizes:
        return False, "not on remote"
    missing = [f for f in REQUIRED_FILES if f not in remote_sizes]
    if missing:
        return False, f"remote missing {missing}"
    for fname in REQUIRED_FILES:
        local_size = (local_dir / fname).stat().st_size
        remote_size = remote_sizes[fname]
        if local_size != remote_size:
            return False, (
                f"size mismatch on {fname}: local={local_size} remote={remote_size}"
            )
    return True, "sizes match"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="HF repo id, e.g. user/model-name")
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Local directory containing checkpoint-* folders (default: checkpoints)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF access token (or set HF_TOKEN / run `huggingface-cli login`).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be uploaded without uploading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upload even if the remote folder already has all required files.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        type=int,
        help="Only upload these checkpoint numbers (e.g. --only 7000 10000).",
    )
    args = parser.parse_args()

    if not args.checkpoints_dir.is_dir():
        print(f"Error: {args.checkpoints_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)

    locals_ = local_checkpoints(args.checkpoints_dir)
    if not locals_:
        print(f"No checkpoint-* folders found under {args.checkpoints_dir}")
        return

    if args.only:
        wanted = set(args.only)
        locals_ = [p for p in locals_ if _checkpoint_num(p.name) in wanted]

    ensure_repo(api, args.repo)
    remote = {} if args.force else remote_file_sizes(api, args.repo)
    complete = [
        d for d, sizes in remote.items() if set(REQUIRED_FILES).issubset(sizes)
    ]
    print(
        f"Remote has {len(complete)} checkpoint(s) with all required files: "
        f"{', '.join(sorted(complete)) or '(none)'}"
    )

    to_upload = []
    for path in locals_:
        missing = [f for f in REQUIRED_FILES if not (path / f).exists()]
        if missing:
            print(f"Skip {path.name}: missing local files {missing}")
            continue
        if args.force:
            to_upload.append(path)
            continue
        match, reason = remote_matches_local(path, remote.get(path.name))
        if match:
            print(f"Skip {path.name}: {reason}")
            continue
        print(f"Queue {path.name}: {reason}")
        to_upload.append(path)

    if not to_upload:
        print("Nothing to upload.")
        return

    print(f"\nWill upload {len(to_upload)} checkpoint(s): "
          f"{', '.join(p.name for p in to_upload)}")
    if args.dry_run:
        return

    for path in to_upload:
        print(f"\n[upload] {path} -> {args.repo}/{path.name}/")
        api.upload_folder(
            folder_path=str(path),
            path_in_repo=path.name,
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"Add {path.name}",
        )
        print(f"[upload] done: {path.name}")

    print(f"\nUploaded {len(to_upload)} checkpoint(s) to {args.repo}")


if __name__ == "__main__":
    main()
