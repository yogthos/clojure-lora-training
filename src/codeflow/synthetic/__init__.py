"""Clojure-specific code synthesis for Code Flow training data.

Generates instruction/response examples using feature trees and LLM providers.
"""

from .construct_tree import (
    FeatureTree,
    FeatureTreeNode,
    assign_features_to_tree,
    build_baseline_tree,
    build_tree_with_llm,
    get_tree_statistics,
    tree_from_json,
    tree_to_json,
)
from .feature_evol import (
    EvolConfig,
    evolve_breadth,
    evolve_depth,
    evolve_detail,
    evolve_tree,
    merge_evolved_trees,
    sample_feature_names,
)
from .gen_code import CodeGenResult

__all__ = [
    "CodeGenResult",
    "EvolConfig",
    "FeatureTree",
    "FeatureTreeNode",
    "assign_features_to_tree",
    "build_baseline_tree",
    "build_tree_with_llm",
    "evolve_breadth",
    "evolve_depth",
    "evolve_detail",
    "evolve_tree",
    "get_tree_statistics",
    "merge_evolved_trees",
    "sample_feature_names",
    "tree_from_json",
    "tree_to_json",
]