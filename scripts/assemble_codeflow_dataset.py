#!/usr/bin/env python3
"""Assemble a Code Flow training dataset for LLaMA-Factory.

Merges git-mined and synthetic Clojure code-evolution data,
deduplicates, balances by change type, formats for LLaMA-Factory,
and validates the output.

Usage:
    assemble_codeflow_dataset \\
        --git-dir data/git-mining/output \\
        --synth-dir data/synthetic/output \\
        --output data/training/codeflow.jsonl \\
        --validation-report data/training/validation_report.json
"""

import argparse
import json
import sys
from pathlib import Path

from src.codeflow.assembly.assembler import assemble_dataset
from src.codeflow.assembly.formatter import format_jsonl_file
from src.codeflow.assembly.validator import validate_jsonl_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Assemble Code Flow training dataset for Clojure LoRA training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--git-dir",
        action="append",
        required=True,
        metavar="DIR",
        help="Directory containing git-mined JSONL data (repeatable)",
    )
    parser.add_argument(
        "--synth-dir",
        action="append",
        required=True,
        metavar="DIR",
        help="Directory containing synthetic JSONL data (repeatable)",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="FILE",
        help="Path for merged output JSONL file",
    )
    parser.add_argument(
        "--max-per-type",
        type=int,
        default=None,
        metavar="N",
        help="Max records per change type for balancing (default: size of smallest type)",
    )
    parser.add_argument(
        "--no-format",
        action="store_true",
        help="Skip LLaMA-Factory format standardization",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip dataset validation",
    )
    parser.add_argument(
        "--validation-report",
        metavar="FILE",
        default=None,
        help="Path for validation report JSON (only if --no-validate not set)",
    )

    return parser.parse_args()


def main() -> int:
    """Run the dataset assembly pipeline. Returns 0 on success, 1 on error."""
    args = parse_args()

    git_paths = [Path(d) for d in args.git_dir]
    synth_paths = [Path(d) for d in args.synth_dir]
    output_path = Path(args.output)

    # Validate inputs
    for p in git_paths + synth_paths:
        if not p.exists():
            print(f"ERROR: path does not exist: {p}", file=sys.stderr)
            return 1

    print(f"Assembling dataset...")
    print(f"  Git sources:  {len(git_paths)}")
    print(f"  Synth sources: {len(synth_paths)}")
    print(f"  Output:       {output_path}")
    if args.max_per_type:
        print(f"  Max/type:     {args.max_per_type}")

    # Step 1: Assemble (merge + deduplicate + balance)
    records = assemble_dataset(
        git_paths=git_paths,
        synth_paths=synth_paths,
        output_path=output_path,
        max_per_type=args.max_per_type,
    )
    print(f"  Assembled:    {len(records)} records")

    # Step 2: Format for LLaMA-Factory
    if not args.no_format:
        count = format_jsonl_file(output_path, output_path)
        print(f"  Formatted:    {count} records (LLaMA-Factory standard)")
    else:
        print("  Formatting:   skipped (--no-format)")

    # Step 3: Validate
    if not args.no_validate:
        report_path = Path(args.validation_report or str(output_path).replace(".jsonl", "_validation_report.json"))
        summary = validate_jsonl_file(output_path, report_path)
        print(f"  Validation:   {summary['valid']}/{summary['total']} valid (avg score: {summary['avg_score']})")
        if summary["invalid"] > 0:
            print(f"  WARNING: {summary['invalid']} records below quality threshold — see {report_path}")
            report_data = json.loads(report_path.read_text())
            for result in report_data.get("results", []):
                if not result["is_valid"]:
                    print(f"    record {result['index']}: score={result['total_score']} "
                          f"warnings={result['warnings']}")
    else:
        print("  Validation:   skipped (--no-validate)")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
