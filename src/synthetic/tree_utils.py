"""Tree sampling, merge, and frequency counting utilities.

Adapted from EpiCoder's utils/tree.py. Utilities for:
- Sampling nodes/features from a feature tree with diversity guarantees
- Merging multiple feature trees
- Computing feature frequency statistics
- Generating balanced training splits from tree coverage
"""

import random
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from .construct_tree import FeatureTree, FeatureTreeNode


def sample_nodes(
    tree: FeatureTree,
    count: int = 10,
    min_depth: int = 1,
    strategy: str = "proportional",
) -> List[FeatureTreeNode]:
    """Sample nodes from a feature tree.

    Args:
        tree: The feature taxonomy tree.
        count: Number of nodes to sample.
        min_depth: Minimum node depth to sample from.
        strategy: Sampling strategy:
            - "proportional": weighted by feature count
            - "uniform": each eligible node equally likely
            - "balanced": ensure coverage across root categories

    Returns:
        List of sampled FeatureTreeNode objects.
    """
    eligible = [
        node for name, node in tree.nodes.items()
        if node.depth >= min_depth
    ]

    if len(eligible) <= count:
        return eligible

    if strategy == "uniform":
        return random.sample(eligible, count)

    if strategy == "balanced":
        return _balanced_sample(tree, eligible, count)

    # "proportional": weighted by feature count
    weights = [max(len(n.features), 1) for n in eligible]
    total = sum(weights)
    if total == 0:
        return random.sample(eligible, count)

    probs = [w / total for w in weights]
    selected = random.choices(eligible, weights=probs, k=count)

    # Deduplicate
    seen_names = set()
    result = []
    for node in selected:
        if node.name not in seen_names:
            result.append(node)
            seen_names.add(node.name)
            if len(result) >= count:
                break

    return result


def _balanced_sample(
    tree: FeatureTree,
    eligible: List[FeatureTreeNode],
    count: int,
) -> List[FeatureTreeNode]:
    """Sample nodes ensuring coverage across root categories."""
    # Group eligible nodes by root category
    by_root: Dict[str, List[FeatureTreeNode]] = {}
    for node in eligible:
        root = node.name.split("/")[0] if "/" in node.name else node.name
        if root not in by_root:
            by_root[root] = []
        by_root[root].append(node)

    # Allocate slots across root categories
    root_cats = list(by_root.keys())
    slots_per_cat = max(1, count // max(len(root_cats), 1))
    remainder = count - slots_per_cat * len(root_cats)

    result = []
    for i, cat in enumerate(root_cats):
        n = slots_per_cat + (1 if i < remainder else 0)
        pool = by_root[cat]
        if pool:
            sampled = random.sample(pool, min(n, len(pool)))
            result.extend(sampled)

    return result[:count]


def sample_features(
    tree: FeatureTree,
    count: int = 20,
    min_depth: int = 1,
) -> List[dict]:
    """Sample individual features from across the tree.

    Ensures diversity by sampling from different nodes proportional
    to their feature count.
    """
    # Build node → features mapping
    node_features: List[Tuple[str, List[dict]]] = []
    for name, node in tree.nodes.items():
        if node.depth >= min_depth and node.features:
            node_features.append((name, node.features))

    if not node_features:
        return []

    # Allocate slots proportionally
    total_features = sum(len(f) for _, f in node_features)
    results = []
    for node_name, features in node_features:
        n = max(1, int(count * len(features) / total_features))
        sampled = random.sample(features, min(n, len(features)))
        results.extend(sampled)

    random.shuffle(results)
    return results[:count]


def compute_feature_frequency(
    trees: List[FeatureTree],
) -> Counter:
    """Compute feature type frequencies across multiple trees.

    Returns a Counter mapping feature_type → total count.
    """
    counter = Counter()
    for tree in trees:
        for node in tree.nodes.values():
            for feat in node.features:
                ftype = feat.get("feature_type", "unknown")
                counter[ftype] += 1
    return counter


def compute_node_coverage(
    tree: FeatureTree,
) -> Dict[str, int]:
    """Compute how many features each node has.

    Returns dict of node_name → feature_count.
    """
    return {
        name: len(node.features)
        for name, node in tree.nodes.items()
    }


def find_sparse_nodes(
    tree: FeatureTree,
    threshold: int = 3,
    min_depth: int = 1,
) -> List[str]:
    """Find nodes with few features that need enrichment.

    Returns list of node names with feature count below threshold.
    """
    return [
        name for name, node in tree.nodes.items()
        if node.depth >= min_depth and len(node.features) < threshold
    ]


def compute_tree_diversity(tree: FeatureTree) -> float:
    """Compute a diversity score for a feature tree.

    Higher score = more even distribution of features across nodes.
    Uses normalized entropy of the feature count distribution.
    """
    import math

    counts = [
        len(node.features)
        for node in tree.nodes.values()
        if node.depth >= 1
    ]

    if not counts:
        return 0.0

    total = sum(counts)
    if total == 0:
        return 0.0

    # Shannon entropy
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log(p)

    # Normalize by max entropy (log of number of nodes)
    max_entropy = math.log(len(counts)) if len(counts) > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


def generate_training_split(
    tree: FeatureTree,
    total_examples: int = 100,
    min_per_node: int = 2,
) -> Dict[str, int]:
    """Generate a balanced training example allocation across tree nodes.

    Ensures each node gets at least min_per_node examples, then
    allocates remaining slots proportionally by feature count.

    Returns dict of node_name → example_count.
    """
    eligible = {
        name: node for name, node in tree.nodes.items()
        if node.depth >= 1
    }

    if not eligible:
        return {}

    # Baseline allocation
    allocation = {name: min_per_node for name in eligible}

    # Remaining budget
    used = sum(allocation.values())
    remaining = total_examples - used

    if remaining <= 0:
        return allocation

    # Proportional allocation of remainder
    feature_counts = {
        name: max(len(node.features), 1)
        for name, node in eligible.items()
    }
    total = sum(feature_counts.values())

    for name in eligible:
        extra = int(remaining * feature_counts[name] / total)
        allocation[name] += extra

    return allocation
