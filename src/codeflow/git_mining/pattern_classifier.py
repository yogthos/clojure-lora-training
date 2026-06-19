"""Classify commit diffs by Clojure-specific development patterns.

Detects: pure-function refactoring, state-machine changes, side-effect
isolation, macro creation, protocol changes, spec additions.
"""

from dataclasses import dataclass, field
from typing import List, Set
import re


@dataclass
class DiffClassification:
    """Classification of a commit diff by Clojure patterns."""
    is_pure_refactor: bool = False       # Input→output, no side effects
    is_state_machine: bool = False       # atom/ref/agent pattern changes
    is_side_effect_isolation: bool = False  # Moving I/O to effect namespaces
    is_macro_change: bool = False        # defmacro creation/modification
    is_protocol_change: bool = False     # defprotocol/defrecord changes
    is_spec_change: bool = False         # clojure.spec additions
    is_async_change: bool = False        # core.async channel/go-loop changes
    is_multimethod_change: bool = False  # defmulti/defmethod changes
    is_namespace_change: bool = False    # ns form requirement changes
    patterns_found: List[str] = field(default_factory=list)


_CONCURRENCY_FORMS = {
    "atom", "swap!", "reset!", "deref", "@/", "@",
    "ref", "dosync", "alter", "commute", "ensure",
    "agent", "send", "send-off", "send-via", "agent-error",
    "future", "promise", "deliver",
    "add-watch", "remove-watch",
}

_SIDE_EFFECT_FORMS = {
    "spit", "slurp", "println", "printf", "pr", "prn",
    "io!", "with-open", "delete-file",
    "jdbc/insert!", "jdbc/update!", "jdbc/delete!",
    "clj-http/client", "http/post", "http/get",
    ".write", ".close", ".flush",
}

_ASYNC_FORMS = {
    "go", "go-loop", "<!", ">!", "<!!", ">!!", "chan", "close!",
    "alts!", "alts!!", "onto-chan!", "thread", "put!", "take!",
    "pipe", "mult", "tap", "pub", "sub", "mix",
}

_SPEC_FORMS = {
    "s/def", "s/fdef", "s/valid?", "s/conform",
    "s/cat", "s/alt", "s/keys", "s/merge", "s/every", "s/coll-of",
}

_MULTIMETHOD_FORMS = {
    "defmulti", "defmethod",
}

_PROTOCOL_FORMS = {
    "defprotocol", "defrecord", "deftype", "extend-protocol", "extend-type",
    "reify", "satisfies?", "extends?",
}

_MACRO_FORMS = {
    "defmacro", "`", "~", "~@", "gensym", "macroexpand",
}


def classify_diff(diff_text: str, changed_files: List[str]) -> DiffClassification:
    """Classify a commit diff by Clojure development patterns.

    Analyzes both the diff content and the file paths touched to
    determine what kind of Clojure development activity the commit
    represents.
    """
    result = DiffClassification()

    # Check file-level signals first
    clj_files = [f for f in changed_files if _is_clojure_source(f)]
    if not clj_files:
        return result

    # Inline the diff content
    content = diff_text.lower()

    # Detect patterns by keyword presence
    _check_pure_refactor(result, content, clj_files)
    _check_state_machine(result, content)
    _check_side_effect_isolation(result, content, clj_files)
    _check_macro(result, content)
    _check_protocol(result, content)
    _check_spec(result, content)
    _check_async(result, content)
    _check_multimethod(result, content)

    return result


def _is_clojure_source(path: str) -> bool:
    """Check if a file path is a Clojure source file."""
    return path.endswith((".clj", ".cljs", ".cljc", ".edn"))


def _check_pure_refactor(
    result: DiffClassification,
    content: str,
    files: List[str],
) -> None:
    """Detect pure-function refactoring: structural changes without side effects."""
    # A pure refactor touches only .clj/.cljs files, not resources/config/docs
    source_only = all(_is_clojure_source(f) for f in files)

    # No side-effecting forms added
    has_side_effects = any(form in content for form in _SIDE_EFFECT_FORMS)
    has_io_files = any(
        f.endswith((".edn", ".xml", ".properties", ".json")) for f in files
    )

    # Check for structural refactoring indicators
    has_structure = any(
        kw in content for kw in ("->", "->>", "comp", "partial",
                                  "update", "assoc", "dissoc", "get-in",
                                  "map", "filter", "reduce", "mapcat")
    )

    if source_only and not has_side_effects and not has_io_files and has_structure:
        result.is_pure_refactor = True
        result.patterns_found.append("pure-refactor")


def _check_state_machine(result: DiffClassification, content: str) -> None:
    """Detect state-machine changes: atom/ref/agent patterns."""
    concurrency_count = sum(1 for form in _CONCURRENCY_FORMS if form in content)
    if concurrency_count >= 2:
        result.is_state_machine = True
        result.patterns_found.append("state-machine")
    elif "atom" in content and "swap!" in content:
        result.is_state_machine = True
        result.patterns_found.append("state-machine")


def _check_side_effect_isolation(
    result: DiffClassification,
    content: str,
    files: List[str],
) -> None:
    """Detect side-effect isolation: moving I/O to dedicated namespaces."""
    effect_file_patterns = {"effects", "io", "db", "http", "api", "gateway"}
    core_file_patterns = {"core", "handler", "service", "logic"}

    touched_effect = any(
        any(p in f.lower() for p in effect_file_patterns) for f in files
    )
    touched_core = any(
        any(p in f.lower() for p in core_file_patterns) for f in files
    )

    has_side_effects = any(form in content for form in _SIDE_EFFECT_FORMS)

    if has_side_effects and touched_effect and not touched_core:
        result.is_side_effect_isolation = True
        result.patterns_found.append("side-effect-isolation")


def _check_macro(result: DiffClassification, content: str) -> None:
    """Detect macro creation/modification."""
    if any(form in content for form in _MACRO_FORMS):
        result.is_macro_change = True
        result.patterns_found.append("macro")


def _check_protocol(result: DiffClassification, content: str) -> None:
    """Detect protocol/record/type changes."""
    if any(form in content for form in _PROTOCOL_FORMS):
        result.is_protocol_change = True
        result.patterns_found.append("protocol")


def _check_spec(result: DiffClassification, content: str) -> None:
    """Detect clojure.spec additions."""
    if any(form in content for form in _SPEC_FORMS):
        result.is_spec_change = True
        result.patterns_found.append("spec")


def _check_async(result: DiffClassification, content: str) -> None:
    """Detect core.async channel/go-loop changes."""
    if any(form in content for form in _ASYNC_FORMS):
        result.is_async_change = True
        result.patterns_found.append("async")


def _check_multimethod(result: DiffClassification, content: str) -> None:
    """Detect multimethod changes."""
    if any(form in content for form in _MULTIMETHOD_FORMS):
        result.is_multimethod_change = True
        result.patterns_found.append("multimethod")
