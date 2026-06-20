"""Stratified, leakage-safe train/val split for the codeflow dataset.

The final assembled dataset has only ``system``/``instruction``/``input``/
``output`` (source and change_type are stripped at format time). To split it
representatively we re-derive each record's stratum — its training objective
(transition vs workflow, read from which system-prompt pool it belongs to) and
its change type — and split within each stratum so both halves carry the same
mix. The dataset is already deduplicated, so a disjoint split is leakage-free;
``assert_no_leakage`` verifies that invariant defensively.
"""

import random
from typing import List, Tuple

from ...shared import compute_dedup_key, WORKFLOW_SYSTEM_PROMPTS
from .assembler import classify_example

_WORKFLOW_PROMPTS = set(WORKFLOW_SYSTEM_PROMPTS)


def record_stratum(record: dict) -> Tuple[str, str]:
    """Return (objective, change_type) for stratification.

    Objective is read from the system prompt: workflow if it is one of the
    workflow paraphrases, otherwise transition. Change type comes from the
    shared keyword classifier.
    """
    objective = "workflow" if record.get("system") in _WORKFLOW_PROMPTS else "transition"
    return objective, classify_example(record)


def assert_no_leakage(train: List[dict], val: List[dict]) -> None:
    """Raise ValueError if any record appears in both splits (by dedup key)."""
    train_keys = {compute_dedup_key(r) for r in train}
    overlap = train_keys & {compute_dedup_key(r) for r in val}
    if overlap:
        raise ValueError(f"train/val leakage: {len(overlap)} shared record(s)")


def stratified_split(
    records: List[dict],
    val_frac: float = 0.05,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """Split records into (train, val), stratified by (objective, change_type).

    Within each stratum the records are shuffled with ``seed`` and the first
    ``round(n * val_frac)`` go to validation. Strata too small to yield a
    validation record fall entirely into train.
    """
    if not records:
        return [], []

    by_stratum: dict = {}
    for r in records:
        by_stratum.setdefault(record_stratum(r), []).append(r)

    rng = random.Random(seed)
    train: List[dict] = []
    val: List[dict] = []
    for stratum in sorted(by_stratum):
        group = by_stratum[stratum][:]
        rng.shuffle(group)
        n_val = int(round(len(group) * val_frac))
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val
