"""Add <think> reasoning to mined git transitions (thinking-mode training).

Runs between mining and assembly. Each transition gets the reasoning a
developer would have had before making its change (see
git_mining/rationale.py); a transition whose reasoning can't be written is
left out, since assembly would reject it anyway. Records that already open
with reasoning pass through.

A rerun resumes: records already written, and ones that already failed, are
skipped. Pass --fresh to start over.

Usage:
    LLM_PROVIDER=deepseek clj-lora-add-reasoning \\
        --input data/git-mining/full_v3.jsonl \\
        --output data/assemble/git/full_v3.jsonl
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.codeflow.assembly.validator import split_think
from src.codeflow.git_mining.rationale import add_reasoning, write_rationale
from src.llm.env import provider_from_env
from src.llm.provider import LLMAuthError
from src.shared import compute_dedup_key, load_jsonl, trim_torn_tail


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add reasoning to mined git transitions",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--input", "-i", action="append", required=True, metavar="FILE",
                   help="Mined JSONL (repeatable)")
    p.add_argument("--output", "-o", required=True, help="Output JSONL (appended to on resume)")
    p.add_argument("--workers", type=int, default=8, help="Concurrent LLM calls (default 8)")
    p.add_argument("--fresh", action="store_true", help="Discard previous output and progress")
    return p.parse_args()


def setup_llm():
    return provider_from_env(max_tokens=1024, temperature=0.4)


def main() -> int:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    attempted_path = out.with_name(out.name + ".attempted")
    if args.fresh:
        out.unlink(missing_ok=True)
        attempted_path.unlink(missing_ok=True)
    for p in (out, attempted_path):
        trim_torn_tail(p)

    done = set(attempted_path.read_text().split()) if attempted_path.exists() else set()
    done |= {compute_dedup_key(r) for r in load_jsonl(out)} if out.exists() else set()
    records = [r for path in args.input for r in load_jsonl(path)]
    todo = [r for r in records if compute_dedup_key(r) not in done]
    print(f"{len(records)} records, {len(records) - len(todo)} already done", file=sys.stderr)

    llm = setup_llm()
    kept = 0
    with open(out, "a") as f, open(attempted_path, "a") as af:
        def finish(rec, result):
            nonlocal kept
            if result is not None:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                kept += 1
            af.write(compute_dedup_key(rec) + "\n")
            af.flush()

        ready = [r for r in todo if split_think(r.get("output", ""))]
        for rec in ready:
            finish(rec, rec)
        pending = [r for r in todo if not split_think(r.get("output", ""))]

        def work(rec):
            text = write_rationale(rec, llm)
            return add_reasoning(rec, text) if text else None

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(work, r): r for r in pending}
            for n, fut in enumerate(as_completed(futures), 1):
                try:
                    result = fut.result()
                except LLMAuthError:
                    for other in futures:
                        other.cancel()
                    raise
                except Exception as e:   # a bad record shouldn't end a long run
                    print(f"  error: {e}", file=sys.stderr)
                    result = None
                finish(futures[fut], result)
                if n % 50 == 0 or n == len(pending):
                    print(f"  {n}/{len(pending)} processed", file=sys.stderr)

    print(f"Wrote {kept} records with reasoning to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
