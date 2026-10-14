"""Coreset selection via k-center greedy clustering.

Adapted from EpiCoder's cluster/main.py. Selects a diverse subset of
training examples that maximizes coverage of the feature space.

The k-center greedy algorithm starts with an empty coreset, then
iteratively adds the point farthest from its nearest coreset neighbor,
ensuring maximum diversity in the selected subset.
"""

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class ClusterConfig:
    """Configuration for coreset clustering."""
    target_size: int = 1000       # Desired coreset size
    embedding_dim: int = 768      # Embedding dimension for features
    random_seed: int = 42
    metric: str = "cosine"        # "cosine" or "euclidean"
    min_samples_per_cluster: int = 5


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two vectors."""
    dot = np.dot(a, b)
    return 1.0 - float(dot)


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return float(np.linalg.norm(a - b))


def kcenter_greedy(
    embeddings: np.ndarray,
    target_size: int,
    metric: str = "cosine",
    seed: int = 42,
) -> np.ndarray:
    """Select a diverse coreset using k-center greedy.

    Algorithm:
    1. Start with a random point as the first center
    2. Compute min distance from each point to nearest center in coreset
    3. Add the point with maximum min-distance to the coreset
    4. Repeat until coreset reaches target_size

    Args:
        embeddings: (n_samples, embedding_dim) array of feature vectors.
        target_size: Number of points to select.
        metric: Distance metric ("cosine" or "euclidean").
        seed: Random seed for first center selection.

    Returns:
        Array of selected indices (sorted).
    """
    n = len(embeddings)
    if n <= target_size:
        return np.arange(n)

    distance_fn = _cosine_distance if metric == "cosine" else _euclidean_distance

    rng = np.random.RandomState(seed)

    selected: List[int] = []
    min_distances = np.full(n, np.inf, dtype=np.float64)

    # Pick first center randomly
    first = rng.randint(0, n)
    selected.append(first)

    # Initialize distances to first center
    first_center = embeddings[first]
    for j in range(n):
        if j != first:
            min_distances[j] = distance_fn(first_center, embeddings[j])
        else:
            min_distances[j] = 0.0

    # Greedy selection
    while len(selected) < target_size:
        # Find the unselected point with maximum min-distance
        best_idx = -1
        best_dist = -1.0
        for j in range(n):
            if j in selected:
                continue
            if min_distances[j] > best_dist:
                best_dist = min_distances[j]
                best_idx = j

        if best_idx < 0:
            break  # Shouldn't happen

        selected.append(best_idx)

        # Update min distances with the new center
        new_center = embeddings[best_idx]
        for j in range(n):
            if j in selected:
                continue
            dist = distance_fn(new_center, embeddings[j])
            if dist < min_distances[j]:
                min_distances[j] = dist

    return np.array(sorted(selected), dtype=int)


def embed_examples(
    examples: List[dict],
    embed_fn,
    embedding_dim: int = 768,
) -> np.ndarray:
    """Embed training examples (instruction + input) with ``embed_fn``.

    There is no default: k-center over anything but semantic embeddings
    picks points at random while claiming diversity.

    Returns:
        (n_examples, embedding_dim) array.
    """
    if embed_fn is None:
        raise ValueError("embed_fn is required (a text -> vector embedding model)")
    if not examples:
        return np.zeros((0, embedding_dim))
    return np.array([embed_fn(ex.get("instruction", "") + " " + ex.get("input", ""))
                     for ex in examples])


def select_coreset(
    examples: List[dict],
    embed_fn,
    target_size: int = 1000,
    embedding_dim: int = 768,
    metric: str = "cosine",
    seed: int = 42,
) -> List[dict]:
    """Select a diverse coreset of training examples.

    Args:
        examples: Training examples to select from.
        target_size: Desired coreset size.
        embedding_dim: Dimension for embeddings.
        embed_fn: Text -> vector embedding function (required).
        metric: Distance metric.
        seed: Random seed.

    Returns:
        Selected subset of examples.
    """
    if len(examples) <= target_size:
        return examples

    embeddings = embed_examples(examples, embed_fn, embedding_dim)
    indices = kcenter_greedy(embeddings, target_size, metric=metric, seed=seed)

    return [examples[i] for i in indices]


def select_by_feature_diversity(
    examples: List[dict],
    extract_fn=None,
    target_size: int = 1000,
    seed: int = 42,
) -> List[dict]:
    """Select a diverse subset by feature coverage.

    Ensures selected examples cover a wide range of Clojure features
    and development patterns. Coverage picks one example per feature; the
    remaining slots are filled at random from the rest, so the result always
    has min(target_size, len(examples)) items — examples without features
    (e.g. mined data) are still eligible rather than silently dropped.

    Args:
        examples: Training examples, optionally with a "features" field.
        extract_fn: Function (example) → feature list. Defaults to the
            example's "features" (or "feature") field.
        target_size: Target coreset size.
        seed: Random seed.

    Returns:
        Selected examples, in their original order.
    """
    if len(examples) <= target_size:
        return examples

    rng = random.Random(seed)

    # If no feature extractor, try to use embedded feature field
    if extract_fn is None:
        def extract_fn(ex):
            feats = ex.get("features", ex.get("feature", []))
            if isinstance(feats, str):
                return [feats]
            return feats if isinstance(feats, list) else []

    # Build feature → example index
    feature_to_examples: Dict[str, List[int]] = {}
    for i, ex in enumerate(examples):
        for f in extract_fn(ex) or []:
            if isinstance(f, dict):
                f = f.get("name", "")
            if not isinstance(f, str) or not f.strip():
                continue
            feature_to_examples.setdefault(f.lower().strip(), []).append(i)

    # Greedy coverage selection: one example per feature, features visited in
    # random order (sorted first so the seed alone decides the order).
    selected: Set[int] = set()
    features = sorted(feature_to_examples)
    rng.shuffle(features)
    for feat in features:
        if len(selected) >= target_size:
            break
        candidates = [
            i for i in feature_to_examples[feat] if i not in selected
        ]
        if candidates:
            selected.add(rng.choice(candidates))

    # Fill the remaining slots from everything not yet selected.
    if len(selected) < target_size:
        rest = [i for i in range(len(examples)) if i not in selected]
        selected.update(rng.sample(rest, target_size - len(selected)))

    return [examples[i] for i in sorted(selected)]
