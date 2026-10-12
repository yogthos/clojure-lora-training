"""Split the assembled codeflow dataset into train/val files for LLaMA-Factory.

Splits by groups of related records (same repo, input or instruction),
stratified by training objective and change type, verifies nothing leaks
across the boundary, and writes sharegpt records.

Usage:
    python scripts/split_codeflow_dataset.py \\
        --input data/codeflow_dataset.jsonl \\
        --train-out data/codeflow_train.jsonl \\
        --val-out data/codeflow_val.jsonl \\
        --val-frac 0.05
"""

import argparse
import sys
from collections import Counter
from pathlib import Path


from src.shared import load_jsonl, write_jsonl
from src.codeflow.assembly.curriculum import curriculum_order
from src.codeflow.assembly.sharegpt import to_sharegpt
from src.codeflow.assembly.splitter import (
    stratified_split,
    assert_no_leakage,
    record_stratum,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stratified, leakage-safe train/val split for the codeflow dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", "-i", required=True, help="Assembled dataset JSONL")
    p.add_argument("--train-out", required=True, help="Output train JSONL")
    p.add_argument("--val-out", required=True, help="Output val JSONL")
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="Fraction held out for validation (default 0.05)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-curriculum", action="store_true",
                   help="Shuffle train instead of ordering it easy -> hard")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    records = load_jsonl(args.input)
    if not records:
        print(f"ERROR: no records in {args.input}", file=sys.stderr)
        return 1

    train, val = stratified_split(records, val_frac=args.val_frac, seed=args.seed)
    assert_no_leakage(train, val)  # raises on overlap

    # The split needs the flat fields (repo, instruction); the files LLaMA-
    # Factory reads are sharegpt (configs/dataset_info.codeflow.json).
    # Easy -> hard, shuffled within phases; the training config disables
    # LLaMA-Factory's own shuffling so this order holds.
    if not args.no_curriculum:
        train = curriculum_order(train, seed=args.seed)
    write_jsonl([to_sharegpt(r) for r in train], args.train_out)
    write_jsonl([to_sharegpt(r) for r in val], args.val_out)

    print(f"Total {len(records)} -> train {len(train)} / val {len(val)} "
          f"(val_frac={args.val_frac}, no leakage)")
    for name, split in (("train", train), ("val", val)):
        strata = Counter(record_stratum(r) for r in split)
        mix = ", ".join(f"{obj}/{ct}={n}" for (obj, ct), n in sorted(strata.items()))
        print(f"  {name}: {mix}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
