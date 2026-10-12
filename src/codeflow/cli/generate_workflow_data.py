#!/usr/bin/env python3
"""Generate plan-first, iterative-REPL workflow training data.

Pipeline:
  1. Load mined git records (commit-message intentions)
  2. Backtranslate substantive messages into vague user-style prompts
  3. For each prompt: plan (goal/files/steps) -> workflow (iterative REPL build
     with real failure/recovery) -> execute & ground in a Jolt nREPL session
  4. Keep traces that reach a correct end state; write LLaMA-Factory JSONL

Usage:
    LLM_PROVIDER=deepseek python scripts/generate_workflow_data.py \\
        --git-data data/git-mining/full_v3.jsonl \\
        --output data/synthetic/workflow.jsonl \\
        --target 200
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.shared import load_jsonl
from src.codeflow.synthetic.jolt_eval import jolt_available
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
                   help="Min fraction of a trace's forms that must run cleanly to keep it")
    p.add_argument("--no-verify-execution", action="store_true",
                   help="Skip jolt execution/grounding (results stay LLM-fabricated)")
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

    verify = not args.no_verify_execution
    if verify and not jolt_available():
        # Without execution every ';; result:' stays LLM-fabricated: refuse
        # rather than silently producing ungrounded data.
        print("ERROR: jolt not found on PATH. Install it, or pass "
              "--no-verify-execution to keep fabricated results.", file=sys.stderr)
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

    attempted = set(attempted_path.read_text().splitlines()) if attempted_path.exists() else set()
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
        af.write(prompt.source_instruction.replace("\n", " ") + "\n")
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
            verify=verify,
            min_pass_rate=args.min_pass_rate,
            on_result=on_result,
            on_progress=on_progress,
            on_attempt=on_attempt,
        )
    finally:
        f.close()
        af.close()

    kept = len(results)
    print(f"Generated {kept} verified workflows "
          f"({len(todo) - kept} dropped)", file=sys.stderr)
    print(f"Output: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
