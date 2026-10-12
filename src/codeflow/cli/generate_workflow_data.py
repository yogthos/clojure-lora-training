"""Generate plan-first, iterative-REPL workflow training data.

Pipeline:
  1. Load mined git records (commit-message intentions)
  2. Backtranslate substantive messages into vague user-style prompts
  3. For each prompt, generate a trace live: each step's eval runs in a Jolt
     nREPL session and the real result is shown before the next step
  4. Keep traces that reach a working end state; write multi-turn JSONL

Usage:
    LLM_PROVIDER=deepseek python scripts/generate_workflow_data.py \\
        --git-data data/git-mining/full_v3.jsonl \\
        --output data/synthetic/workflow.jsonl \\
        --target 200
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


from src.llm.env import provider_from_env
from src.shared import load_jsonl, trim_torn_tail
from src.codeflow.synthetic.jolt_eval import JoltSession, jolt_available
from src.codeflow.synthetic.live_trace import drop_counts
from src.codeflow.synthetic.prompt_mining import MinedPrompt, mine_prompts
from src.codeflow.synthetic.workflow_gen import generate_workflows_concurrent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate plan-first iterative-REPL workflow training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--git-data", action="append", required=True, metavar="FILE",
                   help="Mined git JSONL to source commit-message prompts from (repeatable)")
    p.add_argument("--output", "-o", required=True, help="Output JSONL path")
    p.add_argument("--target", type=int, default=200,
                   help="Number of prompts to process (kept count is lower after verification)")
    p.add_argument("--min-pass-rate", type=float, default=0.6,
                   help="Min fraction of a trace's evals that must run cleanly to keep it")
    p.add_argument("--jolt", default="jolt", metavar="PATH",
                   help="jolt binary that runs the REPL sessions (default: jolt on PATH)")
    p.add_argument("--workers", type=int, default=8,
                   help="Concurrent worker threads (the work is I/O-bound on the "
                        "LLM API and jolt). Default 8.")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed for source records")
    p.add_argument("--fresh", action="store_true",
                   help="Discard previous output, prompt cache and progress, and start over "
                        "(by default a rerun resumes: cached prompts are reused, attempted "
                        "prompts are skipped and new traces are appended)")
    return p.parse_args()


def setup_llm():
    """LLM provider from LLM_PROVIDER (deepseek|ollama)."""
    return provider_from_env(max_tokens=8192, temperature=0.5)


def session_factory(args):
    return lambda: JoltSession(jolt_path=args.jolt)


def _read_attempted(path: Path) -> set:
    """Instructions already attempted: one JSON string per line. Older runs
    wrote the raw text with newlines flattened; those lines are kept as-is."""
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().split("\n"):
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            value = line
        done.add(value if isinstance(value, str) else line)
    return done


def main() -> int:
    import random
    args = parse_args()
    llm = setup_llm()

    if not jolt_available(args.jolt):
        print(f"ERROR: jolt not found ({args.jolt}); traces are generated against "
              "a live REPL.", file=sys.stderr)
        return 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Sidecars that make a rerun resume instead of redoing (and re-paying for)
    # finished work: the mined prompts, and every prompt already attempted.
    prompts_path = out.with_name(out.name + ".prompts")
    attempted_path = out.with_name(out.name + ".attempted")
    if args.fresh:
        for p in (out, prompts_path, attempted_path):
            p.unlink(missing_ok=True)
    # An interrupted run can leave a half-written last line in any of these;
    # drop it before reading or appending.
    for p in (out, prompts_path, attempted_path):
        trim_torn_tail(p)

    records = []
    for path in args.git_data:
        records.extend(load_jsonl(path))
    random.Random(args.seed).shuffle(records)
    print(f"Loaded {len(records)} source records from {len(args.git_data)} file(s)",
          file=sys.stderr)

    prompts = [MinedPrompt(**d) for d in load_jsonl(prompts_path)] if prompts_path.exists() else []
    if len(prompts) < args.target:
        # Backtranslate prompts (oversample: verification drops some downstream).
        used = {p.source_instruction for p in prompts}
        fresh = [r for r in records if (r.get("instruction") or "").strip() not in used]
        new = mine_prompts(fresh, llm, max_prompts=args.target - len(prompts))
        with open(prompts_path, "a") as pf:
            for p in new:
                pf.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
        prompts += new
        print(f"Mined {len(new)} user-style prompts ({len(prompts)} total)", file=sys.stderr)
    prompts = prompts[:args.target]

    attempted = _read_attempted(attempted_path)
    todo = [p for p in prompts if p.source_instruction not in attempted]
    if len(todo) < len(prompts):
        print(f"Resuming: {len(prompts) - len(todo)} prompts already attempted", file=sys.stderr)

    # Stream each verified trace to disk as it completes, so a long run is
    # observable and survives interruption with partial results intact.
    f = open(out, "a")
    af = open(attempted_path, "a")

    def on_result(r):
        f.write(r.to_jsonl() + "\n")
        f.flush()

    def on_attempt(prompt, _result):
        af.write(json.dumps(prompt.source_instruction, ensure_ascii=False) + "\n")
        af.flush()

    def on_progress(done, total, kept):
        if done % 10 == 0 or done == total:
            print(f"  {done}/{total} prompts processed, {kept} kept "
                  f"({kept / done * 100:.0f}%)", file=sys.stderr)

    print(f"Generating with {args.workers} workers...", file=sys.stderr)
    try:
        results = generate_workflows_concurrent(
            todo, llm,
            max_workers=args.workers,
            session_factory=session_factory(args),
            min_pass_rate=args.min_pass_rate,
            on_result=on_result,
            on_progress=on_progress,
            on_attempt=on_attempt,
        )
    finally:
        f.close()
        af.close()

    kept = len(results)
    print(f"Generated {kept} workflows "
          f"({len(todo) - kept} dropped)", file=sys.stderr)
    for reason, n in sorted(drop_counts().items(), key=lambda kv: -kv[1]):
        print(f"  dropped {n}: {reason}", file=sys.stderr)
    print(f"Output: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
