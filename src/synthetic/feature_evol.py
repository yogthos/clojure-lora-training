"""Evolve Clojure feature trees via synthetic expansion.

Adapted from EpiCoder's evol/feature_evol.py. Takes a populated feature
tree and expands it with synthetic features across multiple evolution
strategies: breadth (new categories), depth (new subcategories),
and detail (new features within existing nodes).

Uses LLM to generate plausible new Clojure features that maintain
consistency with the existing taxonomy.
"""

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from copy import deepcopy

from ..llm.provider import LLMProvider
from .construct_tree import FeatureTree, FeatureTreeNode, build_baseline_tree


@dataclass
class EvolConfig:
    """Configuration for feature tree evolution."""
    max_breadth_nodes: int = 8   # Max new root categories per pass
    max_depth_nodes: int = 12    # Max new subcategories per pass
    max_detail_features: int = 20  # Max new leaf features per pass
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096


# Evolution strategy prompts
from .prompts import BREADTH_SYSTEM as _BREADTH_SYSTEM, DEPTH_SYSTEM as _DEPTH_SYSTEM, DETAIL_SYSTEM as _DETAIL_SYSTEM


def evolve_breadth(
    tree: FeatureTree,
    llm: LLMProvider,
    config: Optional[EvolConfig] = None,
) -> FeatureTree:
    """Evolve breadth: suggest new top-level categories via LLM."""
    if config is None:
        config = EvolConfig()

    existing = [
        {"id": name, "label": node.label, "description": node.description}
        for name, node in tree.nodes.items()
        if node.depth == 0
    ]

    prompt = (
        "Existing taxonomy categories:\n" +
        json.dumps(existing, indent=2) +
        "\n\nSuggest NEW top-level categories not already covered. "
        "Focus on Clojure-specific development domains and language features."
    )

    try:
        result = llm.call(
            system_prompt=_BREADTH_SYSTEM,
            user_prompt=prompt,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            require_json=True,
        )
        new_categories = json.loads(result) if isinstance(result, str) else result
        if isinstance(new_categories, dict):
            new_categories = [new_categories]
    except Exception:
        return tree

    # Add new categories to tree
    added = 0
    for cat in new_categories[:config.max_breadth_nodes]:
        cat_id = cat.get("category", "").lower().replace(" ", "-")
        if not cat_id or cat_id in tree.nodes:
            continue

        root_node = FeatureTreeNode(
            name=cat_id,
            label=cat.get("label", cat_id.replace("-", " ").title()),
            description=cat.get("description", ""),
            depth=0,
            node_type="category",
        )
        tree.nodes[cat_id] = root_node
        tree.root_count += 1

        # Add subcategories
        for sub in cat.get("subcategories", [])[:config.max_depth_nodes]:
            sub_id = sub.get("id", "").lower().replace(" ", "-")
            if not sub_id:
                continue
            sub_name = f"{cat_id}/{sub_id}"
            sub_node = FeatureTreeNode(
                name=sub_name,
                label=sub.get("label", sub_id.replace("-", " ").title()),
                description=sub.get("description", ""),
                parent=cat_id,
                depth=1,
                node_type="subcategory",
            )
            tree.nodes[sub_name] = sub_node
            root_node.children.append(sub_name)

        added += 1

    return tree


def evolve_depth(
    tree: FeatureTree,
    llm: LLMProvider,
    config: Optional[EvolConfig] = None,
) -> FeatureTree:
    """Evolve depth: suggest new subcategories for existing categories."""
    if config is None:
        config = EvolConfig()

    # Find categories with few subcategories to expand
    root_nodes = [
        (name, node) for name, node in tree.nodes.items()
        if node.depth == 0 and len(node.children) < 8
    ]

    for cat_name, cat_node in root_nodes[:5]:  # evolve at most 5
        existing_subs = [
            {
                "id": child_name.split("/")[-1] if "/" in child_name else child_name,
                "label": tree.nodes[child_name].label if child_name in tree.nodes else "",
            }
            for child_name in cat_node.children
        ]

        prompt = (
            f"Category: {cat_node.label}\n"
            f"Description: {cat_node.description}\n"
            f"Existing subcategories: {json.dumps(existing_subs, indent=2)}\n\n"
            f"Suggest 2-4 new subcategories that would expand this category "
            f"into related but distinct areas of Clojure development."
        )

        try:
            result = llm.call(
                system_prompt=_DEPTH_SYSTEM,
                user_prompt=prompt,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
                require_json=True,
            )
            suggestions = json.loads(result) if isinstance(result, str) else result
            if isinstance(suggestions, dict):
                suggestions = [suggestions]
        except Exception:
            continue

        added = 0
        for s in suggestions:
            for new_sub in s.get("new_subcategories", []):
                if added >= config.max_depth_nodes:
                    break
                sub_id = new_sub.get("id", "").lower().replace(" ", "-")
                if not sub_id:
                    continue
                sub_name = f"{cat_name}/{sub_id}"
                if sub_name in tree.nodes:
                    continue
                sub_node = FeatureTreeNode(
                    name=sub_name,
                    label=new_sub.get("label", sub_id.replace("-", " ").title()),
                    description=new_sub.get("description", ""),
                    parent=cat_name,
                    depth=1,
                    node_type="subcategory",
                )
                tree.nodes[sub_name] = sub_node
                cat_node.children.append(sub_name)
                added += 1

    return tree


def evolve_detail(
    tree: FeatureTree,
    llm: LLMProvider,
    config: Optional[EvolConfig] = None,
) -> FeatureTree:
    """Evolve detail: generate synthetic features for underpopulated nodes."""
    if config is None:
        config = EvolConfig()

    # Find nodes with few features to enrich
    sparse_nodes = [
        (name, node) for name, node in tree.nodes.items()
        if node.depth > 0 and len(node.features) < 5
    ]

    random.shuffle(sparse_nodes)

    total_added = 0
    for node_name, node in sparse_nodes:
        if total_added >= config.max_detail_features:
            break

        # Find parent context
        parent_label = ""
        if node.parent and node.parent in tree.nodes:
            parent_label = tree.nodes[node.parent].label

        prompt = (
            f"Category: {parent_label}\n"
            f"Subcategory: {node.label}\n"
            f"Description: {node.description}\n"
            f"Existing features: {json.dumps(node.features, indent=2) if node.features else '(none yet)'}\n\n"
            f"Generate 2-4 plausible Clojure code features that belong here."
        )

        try:
            result = llm.call(
                system_prompt=_DETAIL_SYSTEM,
                user_prompt=prompt,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
                require_json=True,
            )
            new_features = json.loads(result) if isinstance(result, str) else result
            if isinstance(new_features, dict):
                new_features = [new_features]
        except Exception:
            continue

        for feat in new_features:
            if not isinstance(feat, dict):
                continue
            feat.setdefault("feature_type", node_name.split("/")[-1] if "/" in node_name else node_name)
            feat.setdefault("complexity", "moderate")
            node.features.append(feat)
            total_added += 1

    return tree


def evolve_tree(
    tree: Optional[FeatureTree],
    llm: LLMProvider,
    config: Optional[EvolConfig] = None,
    iterations: int = 2,
) -> FeatureTree:
    """Run full evolution: breadth → depth → detail for N iterations.

    Args:
        tree: Starting feature tree (builds baseline if None).
        llm: LLM provider for generation.
        config: Evolution configuration.
        iterations: Number of evolution passes.

    Returns:
        Evolved FeatureTree with expanded taxonomy and features.
    """
    if tree is None:
        tree = build_baseline_tree()

    if config is None:
        config = EvolConfig()

    for i in range(iterations):
        tree = evolve_breadth(tree, llm, config)
        tree = evolve_depth(tree, llm, config)
        tree = evolve_detail(tree, llm, config)

    return tree


def merge_evolved_trees(trees: List[FeatureTree]) -> FeatureTree:
    """Merge multiple evolved trees into one.

    Features from the same nodes are deduplicated by name.
    New nodes from each tree are added to the merged tree.
    """
    if not trees:
        return build_baseline_tree()

    merged = deepcopy(trees[0])

    for tree in trees[1:]:
        for name, node in tree.nodes.items():
            if name not in merged.nodes:
                merged.nodes[name] = deepcopy(node)
            else:
                # Merge features, deduplicate by name
                existing_names = {
                    f.get("name", "") for f in merged.nodes[name].features
                }
                for feat in node.features:
                    if feat.get("name", "") not in existing_names:
                        merged.nodes[name].features.append(feat)
                        existing_names.add(feat.get("name", ""))

                # Merge children
                existing_children = set(merged.nodes[name].children)
                for child in node.children:
                    if child not in existing_children:
                        merged.nodes[name].children.append(child)
                        existing_children.add(child)

    merged.root_count = sum(
        1 for n in merged.nodes.values() if n.depth == 0
    )

    return merged


def sample_feature_names(
    tree: FeatureTree,
    count: int = 10,
    min_depth: int = 1,
) -> List[str]:
    """Sample feature names from across the taxonomy tree.

    Ensures diversity by sampling from different nodes proportional
    to their feature count.
    """
    # Collect all features from nodes at >= min_depth
    weighted = []
    for name, node in tree.nodes.items():
        if node.depth >= min_depth and node.features:
            for feat in node.features:
                fname = feat.get("name", "")
                if fname:
                    weighted.append(fname)

    if len(weighted) <= count:
        return weighted

    return random.sample(weighted, count)
