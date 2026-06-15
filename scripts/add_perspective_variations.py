#!/usr/bin/env python3
"""Add perspective variations to existing training data.

This script augments existing training data with perspective variations
without regenerating everything from scratch.

Pipeline:
1. Load existing paragraphs.json (extract "original" entries only)
2. Generate perspective variations (first_person_plural, third_person, impersonal)
3. Create chunks from new perspective variations
4. Run RTT neutralization on new chunks
5. Merge new training examples with existing all.jsonl
6. Regenerate train/valid/test splits

Usage:
    python scripts/add_perspective_variations.py \
        --training-dir data/training/lovecraft \
        --author "H.P. Lovecraft" \
        --workers 4

    # Resume from saved perspective paragraphs
    python scripts/add_perspective_variations.py \
        --training-dir data/training/lovecraft \
        --author "H.P. Lovecraft" \
        --resume-from-perspectives
"""

import argparse
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from main training script
from scripts.generate_flat_training import (
    PERSPECTIVE_TRANSFORMS,
    call_deepseek,
    create_perspective_variation,
    validate_perspective_variation,
    get_nlp,
    split_into_sentences,
    neutralize_batch,
    get_rtt_neutralizer,
    check_lexical_bleed,
    format_training_example,
    create_input_variants,
    OverlapConfig,
)


def load_original_paragraphs(paragraphs_path: Path) -> List[str]:
    """Load only 'original' paragraphs from paragraphs.json."""
    with open(paragraphs_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    originals = []
    for item in data:
        if item.get('variation_type') == 'original':
            originals.append(item['text'])

    logger.info(f"Loaded {len(originals)} original paragraphs from {paragraphs_path}")
    return originals


def generate_perspective_variations(
    paragraphs: List[str],
    author: str,
    workers: int = 4,
    output_path: Optional[Path] = None,
) -> List[Tuple[str, str, str]]:
    """Generate perspective variations for original paragraphs.

    Args:
        paragraphs: List of original author paragraphs.
        author: Author name.
        workers: Number of parallel workers.
        output_path: Optional path to save intermediate results.

    Returns:
        List of (variation_text, variation_type, original_text) tuples.
        The original_text is needed because training output should be the
        original Lovecraft first-person text, not the perspective variation.
    """
    result = []
    perspective_counts = {key: 0 for key in PERSPECTIVE_TRANSFORMS.keys()}
    perspective_failed = 0
    start_time = time.time()

    # Prepare all perspective tasks (3 perspectives per paragraph)
    perspective_tasks = []
    for idx, para in enumerate(paragraphs):
        for perspective_key in PERSPECTIVE_TRANSFORMS.keys():
            perspective_tasks.append((idx, para, perspective_key))

    logger.info(f"Generating {len(perspective_tasks)} perspective variations...")
    logger.info(f"  Perspectives: {list(PERSPECTIVE_TRANSFORMS.keys())}")

    def process_perspective(task):
        idx, para, perspective_key = task
        varied = create_perspective_variation(para, author, perspective_key)
        # Return original paragraph along with variation for training output
        return idx, perspective_key, varied, para

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_perspective, task): task for task in perspective_tasks}

        for future in as_completed(futures):
            try:
                idx, perspective_key, varied, original = future.result()
                if varied:
                    # Store (variation, type, original) - original is used as training output
                    result.append((varied, f"perspective_{perspective_key}", original))
                    perspective_counts[perspective_key] += 1
                else:
                    perspective_failed += 1

                total_processed = sum(perspective_counts.values()) + perspective_failed
                if total_processed % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    success_rate = sum(perspective_counts.values()) / total_processed * 100 if total_processed > 0 else 0
                    logger.info(
                        f"Progress: {sum(perspective_counts.values())}/{len(perspective_tasks)} | "
                        f"Failed: {perspective_failed} | "
                        f"Success: {success_rate:.0f}% | "
                        f"Rate: {rate:.1f}/s"
                    )

                    # Save intermediate results periodically
                    if output_path and total_processed % 200 == 0:
                        save_perspective_paragraphs(result, output_path)

            except Exception as e:
                perspective_failed += 1
                logger.debug(f"Perspective task failed: {e}")

    elapsed = time.time() - start_time
    logger.info(
        f"Perspective generation complete: {sum(perspective_counts.values())} created, "
        f"{perspective_failed} failed in {elapsed:.1f}s"
    )
    logger.info(f"  Per perspective: {perspective_counts}")

    # Save final results
    if output_path:
        save_perspective_paragraphs(result, output_path)

    return result


def save_perspective_paragraphs(items: List[Tuple[str, str, str]], path: Path) -> None:
    """Save perspective paragraphs to JSON file.

    Each item is (variation_text, variation_type, original_text).
    """
    data = [{"text": text, "variation_type": vtype, "original": original}
            for text, vtype, original in items]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(items)} perspective paragraphs to {path}")


def load_perspective_paragraphs(path: Path) -> List[Tuple[str, str, str]]:
    """Load perspective paragraphs from JSON file.

    Returns list of (variation_text, variation_type, original_text) tuples.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    items = [(item["text"], item["variation_type"], item["original"]) for item in data]
    logger.info(f"Loaded {len(items)} perspective paragraphs from {path}")
    return items


def create_perspective_chunks(
    paragraphs: List[Tuple[str, str, str]],
    config: OverlapConfig
) -> List[Tuple[str, str, str]]:
    """Create chunks from perspective paragraphs (no overlap - kept separate).

    Perspective variations are kept as separate chunks because each represents
    the same content in a different voice.

    Args:
        paragraphs: List of (variation_text, variation_type, original_text) tuples.
        config: Overlap configuration.

    Returns:
        List of (variation_text, variation_type, original_text) tuples that meet size requirements.
    """
    nlp = get_nlp()
    chunks = []

    for para_text, vtype, original in paragraphs:
        word_count = len(para_text.split())
        # Only include if it meets minimum size
        if word_count >= config.min_words:
            chunks.append((para_text, vtype, original))
        elif word_count >= config.min_words * 0.7:
            # Include slightly smaller ones too
            chunks.append((para_text, vtype, original))

    logger.info(f"Created {len(chunks)} perspective chunks from {len(paragraphs)} paragraphs")

    # Log breakdown by type
    type_counts = {}
    for _, vtype, _ in chunks:
        type_counts[vtype] = type_counts.get(vtype, 0) + 1
    logger.info(f"  By type: {type_counts}")

    return chunks


def generate_training_examples(
    chunks: List[Tuple[str, str, str]],
    author: str,
    monotone: bool = True,
    start_idx: int = 0,
) -> List[dict]:
    """Generate training examples from perspective chunks.

    CRITICAL: For perspective variations, the training format is:
    - INPUT: RTT-neutralized perspective variation (impersonal/third-person/plural)
    - OUTPUT: Original Lovecraft first-person text

    This teaches the LoRA to convert ANY perspective to Lovecraft's first-person style.

    Args:
        chunks: List of (variation_text, variation_type, original_text) tuples.
        author: Author name.
        monotone: Whether to use monotone RTT flattening.
        start_idx: Starting source_idx for new examples.

    Returns:
        List of training example dicts.
    """
    examples = []
    failed_count = 0
    start_time = time.time()

    # Get batch size from neutralizer
    neutralizer, _ = get_rtt_neutralizer()
    batch_size = getattr(neutralizer, 'batch_size', 1)
    concurrent_batches = getattr(neutralizer, 'concurrent_batches', 1)
    super_batch_size = batch_size * concurrent_batches

    logger.info(f"Generating training examples for {len(chunks)} perspective chunks...")
    logger.info(f"Using batched RTT with super_batch_size={super_batch_size}")

    # Process in super-batches
    for batch_start in range(0, len(chunks), super_batch_size):
        batch_end = min(batch_start + super_batch_size, len(chunks))
        batch = chunks[batch_start:batch_end]

        # Extract VARIATION texts for batch neutralization (NOT original)
        # The variation is what we neutralize and use as INPUT
        batch_texts = [variation_text for variation_text, _, _ in batch]

        # Progress update
        elapsed = time.time() - start_time
        if batch_start > 0:
            rate = batch_start / elapsed
            eta = (len(chunks) - batch_start) / rate if rate > 0 else 0
            logger.info(
                f"[{batch_start}/{len(chunks)}] ({batch_start*100//len(chunks)}%) | "
                f"Examples: {len(examples)} | Failed: {failed_count} | "
                f"{rate:.2f}/s | ETA: {eta/60:.1f}m"
            )

        # Batch neutralization of VARIATION texts
        try:
            neutrals = neutralize_batch(batch_texts, monotone=monotone)
        except Exception as e:
            logger.warning(f"Batch RTT error: {e}")
            neutrals = [None] * len(batch)

        # Process results
        # CRITICAL: Use ORIGINAL text as output, not the variation
        for i, ((variation_text, vtype, original_text), neutral) in enumerate(zip(batch, neutrals)):
            if neutral:
                # Word count based on original (target output length)
                word_count = len(original_text.split())

                # Check lexical bleed between neutral input and original output
                is_valid, overlap_ratio = check_lexical_bleed(neutral, original_text)
                if not is_valid:
                    failed_count += 1
                    logger.debug(f"Lexical bleed ({overlap_ratio:.0%} overlap) for {vtype}")
                    continue

                # INPUT: neutralized variation, OUTPUT: original Lovecraft first-person
                example = format_training_example(
                    neutral_text=neutral,
                    styled_text=original_text,  # ORIGINAL, not variation!
                    author=author,
                    word_count=word_count,
                    variation_type=vtype,
                )
                example["source_idx"] = start_idx + batch_start + i
                example["many_to_one"] = False
                example["is_perspective_variation"] = True

                examples.append(example)
            else:
                failed_count += 1

    elapsed = time.time() - start_time
    logger.info(
        f"Generated {len(examples)} training examples, {failed_count} failed "
        f"in {elapsed:.1f}s"
    )

    return examples


def merge_training_data(
    existing_path: Path,
    new_examples: List[dict],
    output_path: Path,
) -> int:
    """Merge new training examples with existing all.jsonl.

    Args:
        existing_path: Path to existing all.jsonl.
        new_examples: List of new training example dicts.
        output_path: Path to write merged output.

    Returns:
        Total number of examples after merge.
    """
    # Load existing examples
    existing_examples = []
    with open(existing_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                existing_examples.append(json.loads(line))

    logger.info(f"Loaded {len(existing_examples)} existing examples")
    logger.info(f"Adding {len(new_examples)} new perspective examples")

    # Combine
    all_examples = existing_examples + new_examples

    # Write merged output
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + '\n')

    logger.info(f"Wrote {len(all_examples)} total examples to {output_path}")

    return len(all_examples)


def create_splits(
    all_examples: List[dict],
    output_dir: Path,
) -> Tuple[int, int, int]:
    """Create train/valid/test splits from all examples.

    Args:
        all_examples: List of all training example dicts.
        output_dir: Directory to write split files.

    Returns:
        Tuple of (train_count, valid_count, test_count).
    """
    # 80% train, 10% valid, 10% test
    random.seed(42)  # Reproducible splits

    shuffled = all_examples.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)

    train_examples = shuffled[:n_train]
    valid_examples = shuffled[n_train:n_train + n_valid]
    test_examples = shuffled[n_train + n_valid:]

    # Write train/valid/test files (text field only for mlx_lm.lora)
    for split_name, examples in [("train", train_examples), ("valid", valid_examples), ("test", test_examples)]:
        split_path = output_dir / f"{split_name}.jsonl"
        with open(split_path, 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps({"text": ex["text"]}) + '\n')
        logger.info(f"{split_name}: {len(examples)} examples -> {split_path}")

    return len(train_examples), len(valid_examples), len(test_examples)


def main():
    parser = argparse.ArgumentParser(
        description="Add perspective variations to existing training data"
    )
    parser.add_argument("--training-dir", required=True,
                        help="Path to existing training data directory")
    parser.add_argument("--author", required=True,
                        help="Author name")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers for variation generation")
    parser.add_argument("--resume-from-perspectives", action="store_true",
                        help="Resume from saved perspective_paragraphs.json")
    parser.add_argument("--skip-neutralization", action="store_true",
                        help="Skip RTT step (use if you have perspective_chunks.json)")
    parser.add_argument("--no-monotone", action="store_true",
                        help="Disable monotone flattening in RTT")
    parser.add_argument("--min-chunk-words", type=int, default=150,
                        help="Minimum words per chunk")
    parser.add_argument("--max-chunk-words", type=int, default=400,
                        help="Maximum words per chunk")
    parser.add_argument("--max-paragraphs", type=int, default=None,
                        help="Limit number of original paragraphs to process")

    args = parser.parse_args()

    training_dir = Path(args.training_dir)
    if not training_dir.exists():
        parser.error(f"Training directory does not exist: {training_dir}")

    overall_start = time.time()

    # Paths
    paragraphs_path = training_dir / "paragraphs.json"
    perspective_paragraphs_path = training_dir / "perspective_paragraphs.json"
    perspective_chunks_path = training_dir / "perspective_chunks.json"
    existing_all_path = training_dir / "all.jsonl"
    new_all_path = training_dir / "all_with_perspectives.jsonl"

    # =========================================================================
    # Step 1: Load or generate perspective variations
    # =========================================================================
    if args.resume_from_perspectives and perspective_paragraphs_path.exists():
        logger.info("=" * 60)
        logger.info("STEP 1: Loading existing perspective paragraphs")
        logger.info("=" * 60)
        perspective_paragraphs = load_perspective_paragraphs(perspective_paragraphs_path)
    else:
        logger.info("=" * 60)
        logger.info("STEP 1: Generating perspective variations")
        logger.info("=" * 60)

        # Load original paragraphs
        originals = load_original_paragraphs(paragraphs_path)

        if args.max_paragraphs:
            originals = originals[:args.max_paragraphs]
            logger.info(f"Limited to {len(originals)} paragraphs (--max-paragraphs)")

        # Generate perspective variations
        perspective_paragraphs = generate_perspective_variations(
            originals,
            author=args.author,
            workers=args.workers,
            output_path=perspective_paragraphs_path,
        )

    # =========================================================================
    # Step 2: Create chunks from perspective paragraphs
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 2: Creating perspective chunks")
    logger.info("=" * 60)

    overlap_config = OverlapConfig(
        min_words=args.min_chunk_words,
        max_words=args.max_chunk_words,
        overlap_sentences=0,  # No overlap for perspective variations
    )

    perspective_chunks = create_perspective_chunks(perspective_paragraphs, overlap_config)

    # Save perspective-only chunks (with original for reference)
    chunks_data = [{"text": text, "variation_type": vtype, "original": original}
                   for text, vtype, original in perspective_chunks]
    with open(perspective_chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(perspective_chunks)} perspective chunks")

    # Also update the main paragraphs.json and chunks.json with perspective variations
    # Note: We only store the variation text and type (not original) in the main files
    # since the original is already there as "original" type
    logger.info("Updating paragraphs.json with perspective variations...")
    with open(paragraphs_path, 'r', encoding='utf-8') as f:
        existing_paragraphs = json.load(f)

    # Add perspective paragraphs (variation only, original is already in the file)
    for text, vtype, _ in perspective_paragraphs:
        existing_paragraphs.append({"text": text, "variation_type": vtype})

    # Backup and save
    paragraphs_backup = training_dir / "paragraphs.json.backup"
    shutil.copy(paragraphs_path, paragraphs_backup)
    with open(paragraphs_path, 'w', encoding='utf-8') as f:
        json.dump(existing_paragraphs, f, indent=2, ensure_ascii=False)
    logger.info(f"Updated paragraphs.json: {len(existing_paragraphs)} total paragraphs")

    # Update chunks.json
    chunks_path = training_dir / "chunks.json"
    logger.info("Updating chunks.json with perspective chunks...")
    with open(chunks_path, 'r', encoding='utf-8') as f:
        existing_chunks = json.load(f)

    for text, vtype, _ in perspective_chunks:
        existing_chunks.append({"text": text, "variation_type": vtype})

    chunks_backup = training_dir / "chunks.json.backup"
    shutil.copy(chunks_path, chunks_backup)
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(existing_chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"Updated chunks.json: {len(existing_chunks)} total chunks")

    # =========================================================================
    # Step 3: Generate training examples via RTT neutralization
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 3: Generating training examples (RTT neutralization)")
    logger.info("=" * 60)

    # Find the max source_idx in existing data to continue from there
    max_existing_idx = 0
    with open(existing_all_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                idx = ex.get('source_idx', 0)
                if idx > max_existing_idx:
                    max_existing_idx = idx

    logger.info(f"Existing max source_idx: {max_existing_idx}")
    start_idx = max_existing_idx + 10000  # Leave gap for safety

    new_examples = generate_training_examples(
        perspective_chunks,
        author=args.author,
        monotone=not args.no_monotone,
        start_idx=start_idx,
    )

    # Save new examples separately for debugging
    new_examples_path = training_dir / "perspective_examples.jsonl"
    with open(new_examples_path, 'w', encoding='utf-8') as f:
        for ex in new_examples:
            f.write(json.dumps(ex) + '\n')
    logger.info(f"Saved {len(new_examples)} new examples to {new_examples_path}")

    # =========================================================================
    # Step 4: Merge with existing training data
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 4: Merging with existing training data")
    logger.info("=" * 60)

    total_count = merge_training_data(existing_all_path, new_examples, new_all_path)

    # =========================================================================
    # Step 5: Regenerate train/valid/test splits
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 5: Regenerating train/valid/test splits")
    logger.info("=" * 60)

    # Load merged data
    all_examples = []
    with open(new_all_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_examples.append(json.loads(line))

    # Backup existing splits
    for split in ["train", "valid", "test"]:
        src = training_dir / f"{split}.jsonl"
        dst = training_dir / f"{split}.jsonl.backup"
        if src.exists():
            shutil.copy(src, dst)
            logger.info(f"Backed up {src} -> {dst}")

    # Create new splits
    train_n, valid_n, test_n = create_splits(all_examples, training_dir)

    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - overall_start

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Author: {args.author}")
    logger.info(f"Perspective paragraphs generated: {len(perspective_paragraphs)}")
    logger.info(f"Perspective chunks created: {len(perspective_chunks)}")
    logger.info(f"New training examples: {len(new_examples)}")
    logger.info(f"Total training examples: {total_count}")
    logger.info(f"Splits: train={train_n}, valid={valid_n}, test={test_n}")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    logger.info("")
    logger.info("Output files:")
    logger.info(f"  {perspective_paragraphs_path}")
    logger.info(f"  {perspective_chunks_path}")
    logger.info(f"  {new_examples_path}")
    logger.info(f"  {new_all_path}")
    logger.info("")
    logger.info("To use the new data, rename:")
    logger.info(f"  mv {new_all_path} {existing_all_path}")
    logger.info("Or retrain with the updated train/valid/test.jsonl files")


if __name__ == "__main__":
    main()
