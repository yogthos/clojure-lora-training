"""Near-duplicate detection over the code a record trains on (MinHash LSH).

Exact dedup misses the same change seen twice: a fork or a project that
imported another's history (datahike from datascript), a backport, a trace
generated twice for the same prompt. IQuest and EpiCoder both dedup with
n-gram MinHash; this does the same over each record's assistant content
(diffs, eval code), leaving the reasoning out since it is written per record.

Texts are compared as sets of 5-token shingles, whitespace-insensitive. 128
seeded 64-bit hashes give each text a signature; LSH (32 bands of 4) finds
candidate pairs, and a pair is a duplicate when its signatures agree on at
least ``threshold`` of their positions (the estimated Jaccard similarity).
"""

import hashlib
import re
from typing import Dict, List

import numpy as np

SHINGLE = 5
NUM_PERM = 128
BANDS, ROWS = 32, 4

_TOKEN = re.compile(r"[^\s()\[\]{}]+|[()\[\]{}]")
_SEEDS = np.array([int.from_bytes(hashlib.blake2b(f"perm{i}".encode(), digest_size=8).digest(),
                                  "little") for i in range(NUM_PERM)], dtype=np.uint64)


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer, wrapping uint64 arithmetic."""
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def _shingle_hashes(text: str) -> np.ndarray:
    tokens = _TOKEN.findall(text)
    if len(tokens) <= SHINGLE:
        grams = {" ".join(tokens)}
    else:
        grams = {" ".join(tokens[i:i + SHINGLE]) for i in range(len(tokens) - SHINGLE + 1)}
    return np.array([int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little")
                     for g in grams], dtype=np.uint64)


def signature(text: str) -> np.ndarray:
    h = _shingle_hashes(text)
    return _mix(h[:, None] ^ _SEEDS[None, :]).min(axis=0)


def near_duplicate_clusters(texts: List[str], threshold: float = 0.8) -> List[int]:
    """Cluster id per text: texts whose estimated Jaccard similarity is at
    least ``threshold`` share one (transitively)."""
    sigs = [signature(t) for t in texts]
    parent = list(range(len(texts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def similar(i, j):
        return float(np.mean(sigs[i] == sigs[j])) >= threshold

    for b in range(BANDS):
        buckets: Dict[bytes, List[int]] = {}
        for i, s in enumerate(sigs):
            buckets.setdefault(s[b * ROWS:(b + 1) * ROWS].tobytes(), []).append(i)
        for members in buckets.values():
            # Compare each member with the bucket's first and with its
            # predecessor: linear in the bucket, and enough to join clusters.
            for k in range(1, len(members)):
                for j in {members[0], members[k - 1]}:
                    i = members[k]
                    if find(i) != find(j) and similar(i, j):
                        parent[find(i)] = find(j)
    return [find(i) for i in range(len(texts))]


def record_text(record: dict) -> str:
    """The code a record trains on: every assistant turn, reasoning left out."""
    from .validator import split_think

    conv = record.get("conversations")
    turns = ([t["value"] for t in conv if t["from"] in ("gpt", "function_call")]
             if conv else [record.get("output", "")])
    out = []
    for t in turns:
        parts = split_think(t)
        out.append(parts[1] if parts else t)
    return "\n".join(out)
