"""Order training data by difficulty (IQuest-Coder's three-phase SFT curriculum).

Phase 0: small single transitions. Phase 1: larger transitions and history
chains, which need a change built on the one before. Phase 2: live REPL
traces, the longest-horizon work (plan, evaluate, read results, recover,
apply). Records are shuffled within each phase; the training config sets
``disable_shuffling`` so LLaMA-Factory keeps the order, which then repeats
each epoch.
"""

import random
from typing import List

from .assembler import count_changed_lines
from .near_dup import record_text

SMALL_CHANGE = 30   # changed lines; at or under this a transition is phase 0


def phase(record: dict) -> int:
    conv = record.get("conversations") or []
    if record.get("tools") or any(t["from"] == "function_call" for t in conv):
        return 2
    if len(conv) > 2:
        return 1
    return 0 if count_changed_lines(record_text(record)) <= SMALL_CHANGE else 1


def curriculum_order(records: List[dict], seed: int = 42) -> List[dict]:
    rng = random.Random(seed)
    phases: List[List[dict]] = [[], [], []]
    for r in records:
        phases[phase(r)].append(r)
    for p in phases:
        rng.shuffle(p)
    return [r for p in phases for r in p]
