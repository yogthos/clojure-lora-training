#!/usr/bin/env python3
"""Generate plan-first, iterative-REPL workflow training data.

Pipeline:
  1. Load mined git records (commit-message intentions)
  2. Backtranslate substantive messages into vague user-style prompts
  3. For each prompt: plan (goal/files/steps) -> workflow (iterative REPL build
     with real failure/recovery) -> execute & ground in babashka
  4. Keep traces that reach a correct end state; write LLaMA-Factory JSONL

Usage:
    LLM_PROVIDER=deepseek python scripts/generate_workflow_data.py \\
        --git-data data/git-mining/full_v3.jsonl \\
        --output data/synthetic/workflow.jsonl \\
        --target 200
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shared import load_jsonl
from src.codeflow.synthetic.bb_eval import bb_available
from src.codeflow.synthetic.prompt_mining import mine_prompts
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
                   help="Min fraction of a trace's forms that must run cleanly to keep it")
    p.add_argument("--no-verify-execution", action="store_true",
                   help="Skip babashka execution/grounding (results stay LLM-fabricated)")
    p.add_argument("--workers", type=int, default=8,
                   help="Concurrent worker threads (the work is I/O-bound on the "
                        "LLM API and babashka). Default 8.")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed for source records")
    return p.parse_args()


def setup_llm():
    """LLM provider from LLM_PROVIDER (deepseek|ollama)."""
    from src.llm.provider import LLMProviderConfig

    provider = os.environ.get("LLM_PROVIDER", "ollama")
    if provider == "deepseek":
        from src.llm.deepseek import DeepSeekProvider
        return DeepSeekProvider(LLMProviderConfig(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            max_tokens=8192, temperature=0.5, timeout=120,
        ))
    from src.llm.ollama import OllamaProvider
    return OllamaProvider(LLMProviderConfig(
        api_key="", base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:14b"),
        max_tokens=8192, temperature=0.5, timeout=120,
    ))


def main() -> int:
    import random
    args = parse_args()
    llm = setup_llm()

    verify = not args.no_verify_execution and bb_available()
    if not args.no_verify_execution and not verify:
        print("  WARNING: babashka (bb) not found — skipping execution verification",
              file=sys.stderr)

    records = []
    for path in args.git_data:
        records.extend(load_jsonl(path))
    random.Random(args.seed).shuffle(records)
    print(f"Loaded {len(records)} source records from {len(args.git_data)} file(s)",
          file=sys.stderr)

    # Backtranslate prompts (oversample: verification drops some downstream).
    prompts = mine_prompts(records, llm, max_prompts=args.target)
    print(f"Mined {len(prompts)} user-style prompts", file=sys.stderr)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Stream each verified trace to disk as it completes, so a long run is
    # observable and survives interruption with partial results intact.
    f = open(out, "w")

    def on_result(r):
        f.write(r.to_jsonl() + "\n")
        f.flush()

    def on_progress(done, total, kept):
        if done % 10 == 0 or done == total:
            print(f"  {done}/{total} prompts processed, {kept} kept "
                  f"({kept / done * 100:.0f}%)", file=sys.stderr)

    print(f"Generating with {args.workers} workers...", file=sys.stderr)
    try:
        results = generate_workflows_concurrent(
            prompts, llm,
            max_workers=args.workers,
            verify=verify,
            min_pass_rate=args.min_pass_rate,
            on_result=on_result,
            on_progress=on_progress,
        )
    finally:
        f.close()

    kept = len(results)
    print(f"Generated {kept} verified workflows "
          f"({len(prompts) - kept} dropped)", file=sys.stderr)
    print(f"Output: {out} ({kept} records)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
