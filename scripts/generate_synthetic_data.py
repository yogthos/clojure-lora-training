#!/usr/bin/env python3
"""Generate synthetic Clojure training data using the EpiCoder pipeline.

Full pipeline:
  1. Extract Clojure features from reference repositories
  2. Build feature taxonomy tree
  3. Evolve tree (breadth → depth → detail)
  4. Generate coding tasks from tree nodes
  5. Generate REPL-driven code solutions (analysis → code passes)
  6. Select diverse coreset via k-center greedy
  7. Merge with mined git-history data
  8. Output LLaMA-Factory compatible JSONL

Usage:
    # Generate synthetic data only
    python scripts/generate_synthetic_data.py --output synth_train.jsonl

    # Full pipeline: mine repos + generate synthetic + merge
    python scripts/generate_synthetic_data.py \\
        --repo-file repos.txt \\
        --output combined_train.jsonl \\
        --target 2000

    # From pre-extracted features
    python scripts/generate_synthetic_data.py \\
        --features-file features.json \\
        --output synth_train.jsonl
"""

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.provider import LLMProvider
from src.synthetic.extract_features import (
    collect_clojure_files,
    extract_features_from_repo,
    ClojureFeature,
)
from src.synthetic.construct_tree import (
    build_baseline_tree,
    assign_features_to_tree,
    FeatureTree,
    tree_to_json,
    tree_from_json,
)
from src.synthetic.feature_evol import (
    evolve_tree,
    merge_evolved_trees,
    EvolConfig,
)
from src.synthetic.gen_question import (
    generate_tasks_from_tree,
    GeneratedTask,
    format_task_for_training,
)
from src.synthetic.gen_code import (
    generate_training_examples,
    CodeGenResult,
    validate_solution,
)
from src.synthetic.cluster import (
    select_coreset,
    select_by_feature_diversity,
    kcenter_greedy,
    embed_examples,
)
from src.synthetic.file_utils import (
    merge_jsonl,
    deduplicate_jsonl,
    shuffle_jsonl,
    split_jsonl,
    count_records,
    write_jsonl,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Clojure training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o", type=str, default="training_data.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "--target", type=int, default=500,
        help="Target number of training examples",
    )
    parser.add_argument(
        "--repo-file", type=str, default=None,
        help="File with repo paths for feature extraction",
    )
    parser.add_argument(
        "--features-file", type=str, default=None,
        help="Pre-extracted features JSON file (skip extraction)",
    )
    parser.add_argument(
        "--tree-file", type=str, default=None,
        help="Pre-built feature tree JSON (skip tree building)",
    )
    parser.add_argument(
        "--mined-data", type=str, default=None,
        help="Additional mined git-history JSONL to merge",
    )
    parser.add_argument(
        "--evol-iterations", type=int, default=1,
        help="Number of tree evolution passes",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--no-evol", action="store_true",
        help="Skip feature tree evolution",
    )
    parser.add_argument(
        "--no-coreset", action="store_true",
        help="Skip coreset selection",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.05,
        help="Fraction for validation split",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print generation statistics",
    )
    return parser.parse_args()


def setup_llm() -> "LLMProvider":
    """Create LLM provider from environment or default config.

    Uses Ollama as the default local provider. Set LLM_PROVIDER env var
    to "deepseek" or "ollama" to override.
    """
    import os
    from src.llm.provider import LLMProviderConfig

    provider_name = os.environ.get("LLM_PROVIDER", "ollama")

    if provider_name == "deepseek":
        from src.llm.deepseek import DeepSeekProvider
        config = LLMProviderConfig(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
            max_tokens=4096,
            temperature=0.7,
            timeout=120,
        )
        return DeepSeekProvider(config)

    # Default: Ollama
    from src.llm.ollama import OllamaProvider
    config = LLMProviderConfig(
        api_key="",
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:14b"),
        max_tokens=4096,
        temperature=0.7,
        timeout=120,
    )
    return OllamaProvider(config)


def step_extract_features(args, llm) -> List[dict]:
    """Step 1: Extract features from reference repos."""
    if args.features_file:
        with open(args.features_file) as f:
            return json.load(f)

    if not args.repo_file:
        # Use baseline features without repos
        return _baseline_features()

    with open(args.repo_file) as f:
        repos = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    all_features = []
    for repo_path in repos:
        if not Path(repo_path).is_dir():
            continue
        features = extract_features_from_repo(
            repo_path, llm,
            max_files=30,
            sample_strategy="largest",
        )
        all_features.extend([f.to_dict() for f in features])

    return all_features


def _baseline_features() -> List[dict]:
    """Generate baseline features covering common Clojure patterns."""
    return [
        {"feature_type": "macros", "name": "defmacro", "description": "Custom syntax extension via macros", "complexity": "moderate"},
        {"feature_type": "protocols", "name": "defprotocol", "description": "Protocol-based polymorphism", "complexity": "moderate"},
        {"feature_type": "multimethods", "name": "defmulti", "description": "Multiple dispatch based on dispatch function", "complexity": "complex"},
        {"feature_type": "atoms-refs", "name": "atom", "description": "Atomic mutable state with compare-and-swap", "complexity": "simple"},
        {"feature_type": "core-async", "name": "go-loop", "description": "Asynchronous event loop with channels", "complexity": "complex"},
        {"feature_type": "transducers", "name": "comp", "description": "Composable algorithmic transformations", "complexity": "moderate"},
        {"feature_type": "spec-validation", "name": "s/def", "description": "Runtime specification and validation", "complexity": "moderate"},
        {"feature_type": "jvm-interop", "name": ".method", "description": "Java interop via host method calls", "complexity": "simple"},
        {"feature_type": "ring-handlers", "name": "handler", "description": "Ring HTTP request/response handler", "complexity": "moderate"},
        {"feature_type": "middleware", "name": "wrap-", "description": "Handler middleware wrapping patterns", "complexity": "moderate"},
        {"feature_type": "interactive-eval", "name": "comment", "description": "Rich comment blocks for REPL exploration", "complexity": "simple"},
        {"feature_type": "sequences", "name": "lazy-seq", "description": "Lazy sequence generation and consumption", "complexity": "moderate"},
        {"feature_type": "custom-collections", "name": "zipper", "description": "Functional tree navigation and editing", "complexity": "complex"},
        {"feature_type": "exceptions", "name": "try-catch", "description": "Exception handling with try/catch/finally", "complexity": "simple"},
        {"feature_type": "component", "name": "Lifecycle", "description": "Component lifecycle management protocol", "complexity": "moderate"},
        {"feature_type": "sql", "name": "hugsql", "description": "SQL query definition with HugSQL adapters", "complexity": "moderate"},
        {"feature_type": "rich-comments", "name": "(comment", "description": "Rich comment forms with REPL test expressions", "complexity": "simple"},
        {"feature_type": "routing", "name": "defroutes", "description": "HTTP route definitions via Compojure", "complexity": "moderate"},
        {"feature_type": "unit-tests", "name": "deftest", "description": "Unit test definitions with clojure.test", "complexity": "simple"},
        {"feature_type": "property-tests", "name": "defspec", "description": "Property-based testing with test.check", "complexity": "moderate"},
    ]


def step_build_tree(args, features: List[dict]) -> FeatureTree:
    """Step 2: Build feature taxonomy tree."""
    if args.tree_file:
        with open(args.tree_file) as f:
            return tree_from_json(json.load(f))

    tree = build_baseline_tree()
    tree = assign_features_to_tree(features, tree)
    return tree


def step_evolve_tree(args, tree: FeatureTree, llm) -> FeatureTree:
    """Step 3: Evolve feature tree."""
    if args.no_evol:
        return tree

    config = EvolConfig()
    return evolve_tree(tree, llm, config, iterations=args.evol_iterations)


def step_generate_tasks(args, tree: FeatureTree, llm) -> List[GeneratedTask]:
    """Step 4: Generate coding tasks."""
    tasks = generate_tasks_from_tree(
        tree, llm,
        tasks_per_node=3,
        max_total=args.target,
    )
    return tasks


def step_generate_code(args, tasks: List[GeneratedTask], llm) -> List[CodeGenResult]:
    """Step 5: Generate REPL-driven code solutions."""
    task_dicts = [t.to_dict() for t in tasks]
    results = generate_training_examples(
        task_dicts, llm,
        max_examples=args.target,
    )

    # Validate and filter
    valid = [r for r in results if validate_solution(r.solution)]
    return valid


def step_select_coreset(
    args,
    synthetic: List[CodeGenResult],
    mined: List[dict],
) -> List[dict]:
    """Step 6: Select diverse coreset."""
    combined = []

    for r in synthetic:
        combined.append(r.to_training_example())

    if mined:
        combined.extend(mined)

    if args.no_coreset or len(combined) <= args.target:
        return combined

    return select_by_feature_diversity(
        combined,
        target_size=args.target,
        seed=args.seed,
    )


def main():
    args = parse_args()
    random.seed(args.seed)

    llm = setup_llm()

    # Step 1: Extract features
    features = step_extract_features(args, llm)
    if args.stats:
        print(f"[1/6] Extracted {len(features)} features", file=sys.stderr)

    # Step 2: Build tree
    tree = step_build_tree(args, features)
    if args.stats:
        print(f"[2/6] Built feature tree with {len(tree.nodes)} nodes",
              file=sys.stderr)

    # Step 3: Evolve tree
    tree = step_evolve_tree(args, tree, llm)
    if args.stats:
        print(f"[3/6] Evolved tree: {len(tree.nodes)} nodes",
              file=sys.stderr)

    # Step 4: Generate tasks
    tasks = step_generate_tasks(args, tree, llm)
    if args.stats:
        print(f"[4/6] Generated {len(tasks)} tasks", file=sys.stderr)
        type_counts = Counter(t.task_type for t in tasks)
        for ttype, count in type_counts.most_common():
            print(f"      {ttype}: {count}", file=sys.stderr)

    # Step 5: Generate code
    results = step_generate_code(args, tasks, llm)
    if args.stats:
        print(f"[5/6] Generated {len(results)} code solutions "
              f"({len(tasks) - len(results)} invalid filtered)",
              file=sys.stderr)

    # Step 6: Coreset selection and merge
    mined = []
    if args.mined_data and Path(args.mined_data).exists():
        mined = read_jsonl(args.mined_data)
        if args.stats:
            print(f"[6/6] Loaded {len(mined)} mined examples", file=sys.stderr)

    selected = step_select_coreset(args, results, mined)
    if args.stats:
        print(f"[6/6] Selected {len(selected)} examples for training",
              file=sys.stderr)

    # Write output
    tmp = args.output + ".tmp"
    write_jsonl(selected, tmp)

    deduped = args.output + ".dedup"
    deduplicate_jsonl(tmp, deduped)

    shuffled = args.output + ".shuffled"
    shuffle_jsonl(deduped, shuffled, seed=args.seed)

    # Final output
    import shutil
    shutil.move(shuffled, args.output)

    # Cleanup
    for p in [tmp, deduped, args.output + ".shuffled"]:
        if Path(p).exists() and p != args.output:
            Path(p).unlink(missing_ok=True)

    # Validation split
    if args.val_split > 0:
        train_path, val_path = split_jsonl(
            args.output,
            str(Path(args.output).parent / "splits"),
            train_ratio=1.0 - args.val_split,
            seed=args.seed,
        )
        if args.stats:
            print(f"Train: {count_records(train_path)} records → {train_path}",
                  file=sys.stderr)
            print(f"Val:   {count_records(val_path)} records → {val_path}",
                  file=sys.stderr)

    if args.stats:
        print(f"\nOutput: {args.output} ({count_records(args.output)} records)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
