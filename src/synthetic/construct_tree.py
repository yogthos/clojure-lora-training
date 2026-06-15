"""Build a Clojure feature taxonomy tree from extracted features.

Adapted from EpiCoder's extract/construct_tree.py. Takes flat feature lists
from extract_features and builds a hierarchical taxonomy organized by:
1. Clojure language constructs (macros, protocols, multimethods, transducers, etc.)
2. Development patterns (REPL-driven, middleware, side-effect management)
3. Domain patterns (web, data, async, interop)

The tree is used for:
- Sampling diverse training examples across the taxonomy
- Generating synthetic features via evolution (feature_evol)
- Controlling code generation coverage
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from ..llm.provider import LLMProvider


# Top-level Clojure feature taxonomy categories
_CLOJURE_TAXONOMY = {
    "metaprogramming": {
        "label": "Metaprogramming",
        "description": "Macros, code generation, syntax extension",
        "subcategories": ["macros", "code-generation", "syntax-extension", "macro-utilities"],
    },
    "polymorphism": {
        "label": "Polymorphism & Dispatch",
        "description": "Protocols, multimethods, hierarchies, type-based dispatch",
        "subcategories": ["protocols", "multimethods", "records", "types", "hierarchies"],
    },
    "concurrency": {
        "label": "Concurrency & State",
        "description": "Atoms, refs, agents, core.async, futures, promises",
        "subcategories": ["atoms-refs", "core-async", "agents", "futures-promises", "stm"],
    },
    "data-transformation": {
        "label": "Data Transformation",
        "description": "Transducers, sequence operations, data manipulation, spec validation",
        "subcategories": ["transducers", "sequences", "spec-validation", "data-coercion", "collection-ops"],
    },
    "interop": {
        "label": "Java/Platform Interop",
        "description": "JVM interop, host calls, proxy, gen-class, native access",
        "subcategories": ["jvm-interop", "host-calls", "proxy-gen-class", "native"],
    },
    "web-http": {
        "label": "Web & HTTP",
        "description": "Ring handlers, middleware, routing, HTTP clients, servers",
        "subcategories": ["ring-handlers", "middleware", "routing", "http-clients", "servers", "websockets"],
    },
    "data-storage": {
        "label": "Data Storage & Retrieval",
        "description": "Database access, queries, caching, EDN serialization",
        "subcategories": ["sql", "queries", "caching", "serialization", "datomic"],
    },
    "repl-development": {
        "label": "REPL-Driven Development",
        "description": "Interactive evaluation, rich comments, REPL tooling, exploration patterns",
        "subcategories": ["interactive-eval", "rich-comments", "repl-tooling", "exploration", "hot-reload"],
    },
    "testing": {
        "label": "Testing & Verification",
        "description": "Unit tests, property-based testing, generative testing, assertions",
        "subcategories": ["unit-tests", "property-tests", "generative-tests", "asserts", "fixtures"],
    },
    "error-handling": {
        "label": "Error Handling & Resilience",
        "description": "Exceptions, error monads, validation chains, retries, circuit breakers",
        "subcategories": ["exceptions", "error-monads", "validation", "retry", "circuit-breaker"],
    },
    "configuration": {
        "label": "Configuration & Lifecycle",
        "description": "Component systems, integrant, mount, config management",
        "subcategories": ["component", "integrant", "mount", "config-management", "lifecycle"],
    },
    "data-structures": {
        "label": "Custom Data Structures",
        "description": "Custom collections, lazy sequences, zippers, trees, graphs",
        "subcategories": ["custom-collections", "lazy-sequences", "zippers", "trees", "graphs"],
    },
}


@dataclass
class FeatureTreeNode:
    """A node in the feature taxonomy tree."""
    name: str
    label: str = ""
    description: str = ""
    parent: Optional[str] = None
    depth: int = 0
    features: List[dict] = field(default_factory=list)
    children: List[str] = field(default_factory=list)  # child node names
    node_type: str = "category"  # "category", "subcategory", "leaf"


@dataclass
class FeatureTree:
    """A complete feature taxonomy tree."""
    name: str = ""
    description: str = ""
    nodes: Dict[str, FeatureTreeNode] = field(default_factory=dict)
    root_count: int = 0


def build_baseline_tree() -> FeatureTree:
    """Build the baseline Clojure feature taxonomy tree.

    This creates the initial tree structure from _CLOJURE_TAXONOMY
    before any features are assigned. The tree is then populated by
    assign_features_to_tree().
    """
    tree = FeatureTree(
        name="clojure-features",
        description="Clojure language features and development patterns taxonomy",
    )

    for cat_id, cat_def in _CLOJURE_TAXONOMY.items():
        # Root category node
        root_node = FeatureTreeNode(
            name=cat_id,
            label=cat_def["label"],
            description=cat_def["description"],
            depth=0,
            node_type="category",
        )
        tree.nodes[cat_id] = root_node
        tree.root_count += 1

        # Subcategory nodes
        for sub_id in cat_def["subcategories"]:
            sub_name = f"{cat_id}/{sub_id}"
            sub_node = FeatureTreeNode(
                name=sub_name,
                label=_subcategory_label(sub_id),
                description=f"{cat_def['label']} — {sub_id.replace('-', ' ').title()}",
                parent=cat_id,
                depth=1,
                node_type="subcategory",
            )
            tree.nodes[sub_name] = sub_node
            root_node.children.append(sub_name)

    return tree


def _subcategory_label(sub_id: str) -> str:
    """Generate a human-readable label for a subcategory slug."""
    return sub_id.replace("-", " ").title()


def assign_features_to_tree(
    features: List[dict],
    tree: Optional[FeatureTree] = None,
    llm: Optional[LLMProvider] = None,
) -> FeatureTree:
    """Assign extracted features to the taxonomy tree.

    Uses simple keyword matching for fast classification. Falls back
    to LLM classification if available and keyword matching is ambiguous.

    Args:
        features: List of feature dicts from extract_features.
        tree: Existing tree to populate (creates new baseline if None).
        llm: Optional LLM provider for ambiguous classification.

    Returns:
        Populated FeatureTree.
    """
    if tree is None:
        tree = build_baseline_tree()

    # Keyword-based routing table: feature_type → tree category
    _ROUTING = {
        "macros": "metaprogramming",
        "code-generation": "metaprogramming",
        "syntax-extension": "metaprogramming",
        "macro-utilities": "metaprogramming",
        "protocols": "polymorphism",
        "multimethods": "polymorphism",
        "records": "polymorphism",
        "types": "polymorphism",
        "hierarchies": "polymorphism",
        "atoms-refs": "concurrency",
        "concurrency": "concurrency",
        "atoms": "concurrency",
        "refs": "concurrency",
        "core-async": "concurrency",
        "async": "concurrency",
        "agents": "concurrency",
        "futures-promises": "concurrency",
        "transducers": "data-transformation",
        "sequences": "data-transformation",
        "spec-validation": "data-transformation",
        "specs": "data-transformation",
        "data-coercion": "data-transformation",
        "collection-ops": "data-transformation",
        "data-manipulation": "data-transformation",
        "jvm-interop": "interop",
        "host-calls": "interop",
        "proxy-gen-class": "interop",
        "ring-handlers": "web-http",
        "middleware": "web-http",
        "routing": "web-http",
        "http-clients": "web-http",
        "repl-driven": "repl-development",
        "interactive-eval": "repl-development",
        "rich-comments": "repl-development",
        "unit-tests": "testing",
        "property-tests": "testing",
        "testing": "testing",
        "exceptions": "error-handling",
        "error-handling": "error-handling",
        "validation": "error-handling",
        "sql": "data-storage",
        "data-storage": "data-storage",
        "queries": "data-storage",
        "component": "configuration",
        "configuration": "configuration",
        "integrant": "configuration",
        "lifecycle": "configuration",
        "custom-collections": "data-structures",
        "data-structures": "data-structures",
    }

    for feat in features:
        ftype = feat.get("feature_type", "").lower().strip()
        category = _ROUTING.get(ftype)

        if category is None:
            # Try partial matches
            for key, cat in _ROUTING.items():
                if key in ftype or ftype in key:
                    category = cat
                    break

        if category is None:
            category = "data-transformation"  # default bucket

        # Find the best subcategory node
        sub_node = _best_subcategory(category, feat, tree)
        if sub_node:
            tree.nodes[sub_node].features.append(feat)

    return tree


def _best_subcategory(
    category: str,
    feat: dict,
    tree: FeatureTree,
) -> Optional[str]:
    """Find the best subcategory node for a feature within its category."""
    cat_node = tree.nodes.get(category)
    if cat_node is None:
        return None

    # If no children, use the category node itself
    if not cat_node.children:
        return category

    ftype = feat.get("feature_type", "").lower()
    desc = feat.get("description", "").lower()
    name = feat.get("name", "").lower()

    scored = []
    for child_name in cat_node.children:
        child = tree.nodes.get(child_name)
        if child is None:
            continue
        child_label = child.label.lower()
        score = 0
        # Exact match on feature_type
        if child_label in ftype or ftype in child_label:
            score += 3
        # Match in description
        if any(w in desc for w in child_label.split()):
            score += 1
        # Match in name
        if child_label in name:
            score += 2
        scored.append((score, child_name))

    scored.sort(key=lambda x: -x[0])
    if scored and scored[0][0] > 0:
        return scored[0][1]
    return cat_node.children[0] if cat_node.children else category


def build_tree_with_llm(
    features: List[dict],
    llm: LLMProvider,
) -> FeatureTree:
    """Build a taxonomy tree using LLM for both structure and classification.

    The LLM analyzes all features and proposes a taxonomy structure that
    best organizes them. This is used when keyword routing is insufficient.
    """
    tree = build_baseline_tree()

    _SYSTEM = """You are a Clojure code taxonomy expert. Given a list of extracted
Clojure code features, classify each one into a hierarchical taxonomy.

Categories: metaprogramming, polymorphism, concurrency, data-transformation,
interop, web-http, data-storage, repl-development, testing, error-handling,
configuration, data-structures.

For each feature, output:
{"feature_index": N, "category": "...", "subcategory_hint": "..."}

Output JSON array only, no markdown."""

    features_json = json.dumps(features, indent=2)
    prompt = f"Classify these Clojure features:\n\n{features_json}"

    try:
        result = llm.call(
            system_prompt=_SYSTEM,
            user_prompt=prompt,
            temperature=0.1,
            max_tokens=4096,
            require_json=True,
        )
        classifications = json.loads(result) if isinstance(result, str) else result
        if isinstance(classifications, dict):
            classifications = [classifications]

        for cls in classifications:
            idx = cls.get("feature_index", -1)
            cat = cls.get("category", "")
            if 0 <= idx < len(features):
                sub_node = _best_subcategory(cat, features[idx], tree)
                if sub_node:
                    tree.nodes[sub_node].features.append(features[idx])
    except Exception:
        # Fall back to keyword routing
        tree = assign_features_to_tree(features, tree)

    return tree


def get_tree_statistics(tree: FeatureTree) -> dict:
    """Get summary statistics about a feature tree."""
    stats = {
        "total_nodes": len(tree.nodes),
        "total_root_categories": tree.root_count,
        "total_features": sum(
            len(node.features)
            for node in tree.nodes.values()
        ),
        "categories": {},
    }

    for name, node in tree.nodes.items():
        if node.depth == 0:
            total_in_category = len(node.features)
            for child_name in node.children:
                child = tree.nodes.get(child_name)
                if child:
                    total_in_category += len(child.features)
            stats["categories"][name] = {
                "label": node.label,
                "subcategory_count": len(node.children),
                "feature_count": total_in_category,
            }

    return stats


def tree_to_json(tree: FeatureTree) -> dict:
    """Serialize a FeatureTree to JSON."""
    return {
        "name": tree.name,
        "description": tree.description,
        "nodes": {
            name: {
                "label": node.label,
                "description": node.description,
                "parent": node.parent,
                "depth": node.depth,
                "feature_count": len(node.features),
                "features": node.features,
                "children": node.children,
                "node_type": node.node_type,
            }
            for name, node in tree.nodes.items()
        },
    }


def tree_from_json(data: dict) -> FeatureTree:
    """Deserialize a FeatureTree from JSON."""
    tree = FeatureTree(
        name=data.get("name", ""),
        description=data.get("description", ""),
    )
    for name, node_data in data.get("nodes", {}).items():
        node = FeatureTreeNode(
            name=name,
            label=node_data.get("label", ""),
            description=node_data.get("description", ""),
            parent=node_data.get("parent"),
            depth=node_data.get("depth", 0),
            features=node_data.get("features", []),
            children=node_data.get("children", []),
            node_type=node_data.get("node_type", "category"),
        )
        tree.nodes[name] = node
    tree.root_count = sum(
        1 for n in tree.nodes.values() if n.depth == 0
    )
    return tree
