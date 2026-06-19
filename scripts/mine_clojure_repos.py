#!/usr/bin/env python3
"""Mine Clojure repositories for before/after code training pairs.

End-to-end workflow:
  1. Clone or use local repositories
  2. Walk git history filtering for Clojure-relevant commits
  3. Extract before/after file states + diffs
  4. Group into PR sessions
  5. Classify by Clojure development pattern
  6. Output LLaMA-Factory compatible JSONL training data

Usage:
    # Mine a single local repo
    python scripts/mine_clojure_repos.py --repo /path/to/repo

    # Mine from a list of repos (one per line, GitHub URLs or local paths)
    python scripts/mine_clojure_repos.py --repo-file repos.txt

    # Filter by specific patterns
    python scripts/mine_clojure_repos.py --repo /path/to/repo \\
        --pattern pure-refactor --pattern state-machine

    # Output to file (default: stdout)
    python scripts/mine_clojure_repos.py --repo /path/to/repo \\
        --output training_data.jsonl

    # Sample repos to get started (modify or replace):
    #   metabase/metabase, babashka/babashka, clj-kondo/clj-kondo,
    #   ring-clojure/ring, compojure/compojure, reagent-project/reagent,
    #   day8/re-frame, pedestal/pedestal, xtdb/xtdb
"""

import argparse
import json
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional
from collections import Counter

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.codeflow.git_mining.miner import mine_repository, MinedExample
from src.codeflow.git_mining.pattern_classifier import classify_diff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine Clojure repositories for training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    repo_group = parser.add_mutually_exclusive_group(required=True)
    repo_group.add_argument(
        "--repo", type=str,
        help="Path to a local Clojure repository",
    )
    repo_group.add_argument(
        "--repo-file", type=str,
        help="File with one repo path/URL per line",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="-",
        help="Output file (default: stdout)",
    )
    parser.add_argument(
        "--max-commits", type=int, default=500,
        help="Max commits to scan per repo (default: 500)",
    )
    parser.add_argument(
        "--since", type=str, default=None,
        help="Only commits after this date (e.g., '2024-01-01')",
    )
    parser.add_argument(
        "--pattern", type=str, action="append", dest="patterns",
        choices=[
            "pure-refactor", "state-machine", "side-effect-isolation",
            "macro", "protocol", "spec", "async", "multimethod",
        ],
        help="Only include examples matching these Clojure patterns",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print mining statistics to stderr",
    )
    parser.add_argument(
        "--clone-dir", type=str, default=None,
        help="Directory to clone repos into (default: system temp dir)",
    )
    return parser.parse_args()


def resolve_repos(args: argparse.Namespace) -> List[str]:
    """Resolve repository paths from arguments."""
    if args.repo:
        return [args.repo]

    repos = []
    with open(args.repo_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                repos.append(line)
    return repos


def clone_if_needed(repo: str, clone_dir: str) -> Optional[str]:
    """Clone a remote repo if needed, return local path or None."""
    path = Path(repo)
    if path.is_dir():
        return str(path.resolve())

    # Try as GitHub shorthand: "owner/repo" or full URL
    if "/" in repo and not repo.startswith(("http://", "https://", "git@")):
        url = f"https://github.com/{repo}.git"
    else:
        url = repo

    name = repo.split("/")[-1].replace(".git", "")
    target = Path(clone_dir) / name

    if target.exists():
        return str(target)

    try:
        subprocess.run(
            ["git", "clone", "--depth=1000", url, str(target)],
            check=True, capture_output=True, text=True,
        )
        return str(target)
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone {url}: {e.stderr}", file=sys.stderr)
        return None


def main():
    args = parse_args()
    repos = resolve_repos(args)

    clone_dir = args.clone_dir or tempfile.mkdtemp(prefix="clj-mine-")
    all_examples: List[MinedExample] = []
    stats = Counter()

    for repo in repos:
        local_path = clone_if_needed(repo, clone_dir)
        if not local_path:
            stats["failed_clone"] += 1
            continue

        repo_name = Path(local_path).name
        print(f"Mining {repo_name}...", file=sys.stderr)

        try:
            examples = mine_repository(
                local_path,
                repo_name=repo_name,
                max_commits=args.max_commits,
                since=args.since,
            )
        except Exception as e:
            print(f"Error mining {repo_name}: {e}", file=sys.stderr)
            stats["failed_mine"] += 1
            continue

        # Apply pattern filter if requested
        if args.patterns:
            filtered = []
            for ex in examples:
                classification = classify_diff(ex.diff, ex.changed_files)
                for p in args.patterns:
                    attr = f"is_{p.replace('-', '_')}"
                    if getattr(classification, attr, False):
                        filtered.append(ex)
                        break
            stats["filtered_out"] += len(examples) - len(filtered)
            examples = filtered

        all_examples.extend(examples)
        stats[f"repo_{repo_name}"] = len(examples)
        stats["total_examples"] += len(examples)

        print(f"  {repo_name}: {len(examples)} examples", file=sys.stderr)

    # Write output
    out = open(args.output, "w") if args.output != "-" else sys.stdout
    try:
        for ex in all_examples:
            out.write(ex.to_jsonl() + "\n")
    finally:
        if out is not sys.stdout:
            out.close()

    # Print stats
    if args.stats:
        print(f"\n--- Mining Statistics ---", file=sys.stderr)
        print(f"Repos processed: {len(repos)}", file=sys.stderr)
        print(f"Total examples: {stats['total_examples']}", file=sys.stderr)
        print(f"Failed clones: {stats['failed_clone']}", file=sys.stderr)
        print(f"Failed mines: {stats['failed_mine']}", file=sys.stderr)
        if args.patterns:
            print(f"Filtered out: {stats['filtered_out']}", file=sys.stderr)
        print(f"Output: {args.output if args.output != '-' else 'stdout'}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
