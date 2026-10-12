"""Stratified, grouped train/val split for the codeflow dataset.

Records are split in groups so related examples never straddle the split: all
records from one repository, records with the same input (the same before-state
files), and records whose instructions match after normalization. A record-level
split let near-identical examples sit on both sides and made eval loss look
better than generalization is.

Within that constraint the split is stratified by each record's training
objective (transition vs workflow, read from the system-prompt pool) and change
type, so both halves carry roughly the same mix. ``assert_no_leakage`` checks
that no record or group appears on both sides.
"""

import hashlib
import random
import re
from collections import Counter
from typing import Dict, List, Tuple

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


def _group_keys(record: dict) -> List[str]:
    keys = []
    if record.get("repo"):
        keys.append("repo:" + record["repo"])
    if record.get("input", "").strip():
        keys.append("input:" + hashlib.sha256(record["input"].encode()).hexdigest())
    instruction = re.sub(r"[^a-z0-9]+", " ", record.get("instruction", "").lower()).strip()
    if instruction:
        keys.append("instruction:" + instruction)
    return keys


def record_groups(records: List[dict]) -> List[int]:
    """Group id per record: connected components over shared repo, identical
    input, or normalized instruction."""
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first: Dict[str, int] = {}
    for i, r in enumerate(records):
        for key in _group_keys(r):
            if key in first:
                parent[find(i)] = find(first[key])
            else:
                first[key] = i
    return [find(i) for i in range(len(records))]


def assert_no_leakage(train: List[dict], val: List[dict]) -> None:
    """Raise ValueError if a record, or a group of related records, is in both splits."""
    train_keys = {compute_dedup_key(r) for r in train}
    overlap = train_keys & {compute_dedup_key(r) for r in val}
    if overlap:
        raise ValueError(f"train/val leakage: {len(overlap)} shared record(s)")
    groups = record_groups(train + val)
    shared = set(groups[:len(train)]) & set(groups[len(train):])
    if shared:
        raise ValueError(f"train/val leakage: {len(shared)} group(s) of related records "
                         "(same repo, input or instruction) on both sides")


def stratified_split(
    records: List[dict],
    val_frac: float = 0.05,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """Split records into (train, val) by whole groups (see ``record_groups``).

    Groups are shuffled with ``seed`` and each goes to validation while its
    dominant stratum is under ``round(n * val_frac)`` records there. A stratum
    that would otherwise get no validation records gets its smallest group, as
    long as that leaves the stratum something to train on.
    """
    if not records:
        return [], []

    by_group: Dict[int, List[dict]] = {}
    for r, g in zip(records, record_groups(records)):
        by_group.setdefault(g, []).append(r)
    groups = [by_group[g] for g in sorted(by_group)]

    strata = Counter(record_stratum(r) for r in records)
    target = {s: int(round(n * val_frac)) for s, n in strata.items()}

    def dominant(group):
        return Counter(record_stratum(r) for r in group).most_common(1)[0][0]

    rng = random.Random(seed)
    rng.shuffle(groups)
    in_val = Counter()
    val_groups, train_groups = [], []
    for group in groups:
        s = dominant(group)
        if in_val[s] + len(group) <= target[s]:
            val_groups.append(group)
            in_val[s] += len(group)
        else:
            train_groups.append(group)

    for s in sorted(target):
        if target[s] == 0 or in_val[s]:
            continue
        candidates = [g for g in train_groups if dominant(g) == s]
        if len(candidates) < 2:
            continue
        smallest = min(candidates, key=len)
        train_groups.remove(smallest)
        val_groups.append(smallest)
        in_val[s] += len(smallest)

    train = [r for g in train_groups for r in g]
    val = [r for g in val_groups for r in g]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val
