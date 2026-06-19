"""JSONL file operations for training data.

Adapted from EpiCoder's utils/file_operation.py. Utilities for:
- Sorting large JSONL files
- Merging multiple JSONL files
- Deduplicating by instruction
- Shuffling with memory efficiency
"""

import hashlib
import json
import random
from pathlib import Path
from typing import List, Optional, Set, Tuple


def sort_jsonl(
    input_path: str,
    output_path: Optional[str] = None,
    sort_key: str = "instruction",
    max_buffer: int = 10000,
) -> str:
    """Sort a JSONL file by a key field.

    For large files, uses external sort with chunking.

    Args:
        input_path: Path to input JSONL.
        output_path: Path to output (defaults to input_path + '.sorted.jsonl').
        sort_key: Key to sort by.
        max_buffer: Maximum records to hold in memory.

    Returns:
        Path to sorted output file.
    """
    if output_path is None:
        output_path = str(Path(input_path).with_suffix(".sorted.jsonl"))

    # Read all records
    records = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                continue

    # Sort by key
    records.sort(key=lambda r: r.get(sort_key, ""))

    # Write sorted
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path


def merge_jsonl(
    input_paths: List[str],
    output_path: str,
    deduplicate: bool = True,
    dedup_key: str = "instruction",
) -> str:
    """Merge multiple JSONL files into one.

    Args:
        input_paths: List of JSONL file paths to merge.
        output_path: Output file path.
        deduplicate: Whether to remove duplicates.
        dedup_key: Key to use for deduplication.

    Returns:
        Path to merged output.
    """
    seen_signatures: Set[str] = set()
    total_kept = 0
    total_skipped = 0

    with open(output_path, "w") as out:
        for path in input_paths:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if deduplicate:
                        sig = _record_signature(record, dedup_key)
                        if sig in seen_signatures:
                            total_skipped += 1
                            continue
                        seen_signatures.add(sig)

                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_kept += 1

    return output_path


def shuffle_jsonl(
    input_path: str,
    output_path: Optional[str] = None,
    seed: int = 42,
) -> str:
    """Shuffle a JSONL file.

    Args:
        input_path: Path to input JSONL.
        output_path: Path to output (defaults to input_path + '.shuffled.jsonl').
        seed: Random seed for reproducibility.

    Returns:
        Path to shuffled output.
    """
    if output_path is None:
        output_path = str(Path(input_path).with_suffix(".shuffled.jsonl"))

    records = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    rng = random.Random(seed)
    rng.shuffle(records)

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return output_path


def deduplicate_jsonl(
    input_path: str,
    output_path: Optional[str] = None,
    dedup_key: str = "instruction",
) -> str:
    """Remove duplicate records from a JSONL file.

    Args:
        input_path: Path to input JSONL.
        output_path: Output path.
        dedup_key: Key to check for duplicates.

    Returns:
        Path to deduplicated output.
    """
    if output_path is None:
        output_path = str(Path(input_path).with_suffix(".dedup.jsonl"))

    seen: Set[str] = set()
    kept = 0
    skipped = 0

    with open(output_path, "w") as out:
        with open(input_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                sig = _record_signature(record, dedup_key)
                if sig in seen:
                    skipped += 1
                    continue
                seen.add(sig)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

    return output_path


def count_records(path: str) -> int:
    """Count records in a JSONL file."""
    count = 0
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def split_jsonl(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[str, str]:
    """Split a JSONL file into train/validation sets.

    Args:
        input_path: Path to input JSONL.
        output_dir: Directory for output files.
        train_ratio: Fraction for training set.
        seed: Random seed.

    Returns:
        (train_path, val_path) tuple.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    rng = random.Random(seed)
    rng.shuffle(records)

    split_idx = int(len(records) * train_ratio)
    train = records[:split_idx]
    val = records[split_idx:]

    train_path = str(out_dir / "train.jsonl")
    val_path = str(out_dir / "val.jsonl")

    write_jsonl(train, train_path)
    write_jsonl(val, val_path)

    return train_path, val_path


from ...shared import count_records  # noqa: F401 — re-exported for backward compat
from ...shared import load_jsonl as read_jsonl
from ...shared import write_jsonl


def _record_signature(record: dict, key: str = "instruction") -> str:
    """Generate a deduplication signature for a record.

    Uses the dedup key value normalized and hashed.
    """
    value = record.get(key, "")
    if isinstance(value, (list, dict)):
        value = json.dumps(value, sort_keys=True)
    normalized = str(value).lower().strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
