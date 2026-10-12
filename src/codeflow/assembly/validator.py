"""Dataset validator: check Clojure syntax, diff coherence, relevance scoring.

Validates Code Flow training examples and produces validation reports.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..clj_text import bracket_errors
from ..diffs import apply_section_errors, check_patch, extract_diff, split_lines
from ...shared import write_jsonl
from .assembler import load_jsonl

# ── Clojure-specific terms for relevance scoring ────────────────────────

_CLOJURE_TERMS = [
    "defn", "defmacro", "defprotocol", "defmulti", "defmethod",
    "defrecord", "deftype", "ns", "require", "use", "import",
    "let", "fn", "loop", "recur", "doseq", "for", "when-let",
    "if-let", "cond->", "->", "->>", "comp", "partial", "memoize",
    "atom", "swap!", "reset!", "ref", "alter", "dosync",
    "future", "promise", "deliver", "channel", "go", "<!",
    "core.async", "transducer", "reducer", "lazy-seq",
    "ring", "compojure", "reitit", "pedestal", "http-kit",
    "reagent", "re-frame", "rum", "helix",
    "datomic", "xtdb", "next.jdbc", "hugsql",
    "spec", "test.check", "clojure.test",
    "component", "integrant", "mount", "mycelium",
]


@dataclass
class ValidationResult:
    """Result of validating a single training example."""
    is_valid: bool = True
    total_score: float = 0.0
    syntax_errors: List[str] = field(default_factory=list)
    diff_errors: List[str] = field(default_factory=list)
    format_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    instruction_score: float = 0.0
    syntax_score: float = 0.0
    diff_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "total_score": self.total_score,
            "instruction_score": self.instruction_score,
            "syntax_score": self.syntax_score,
            "diff_score": self.diff_score,
            "syntax_errors": self.syntax_errors,
            "diff_errors": self.diff_errors,
            "format_errors": self.format_errors,
            "warnings": self.warnings,
        }


# ── Clojure syntax check ─────────────────────────────────────────────────

def check_clojure_syntax(code: str) -> List[str]:
    """Check that brackets in Clojure code pair up and strings are closed.

    Not a full parse: strings, character literals and comments are skipped,
    and ``()``, ``[]`` and ``{}`` must nest correctly.

    Returns list of error messages (empty = valid).
    """
    return bracket_errors(code)


# ── Diff structure check ─────────────────────────────────────────────────

def check_diff_structure(diff: str) -> List[str]:
    """Validate that the output's diff is a patch ``git apply`` accepts.

    ``diff`` may be a whole output (REPL trace + ``;; apply:`` + diff); only the
    part from the first ``diff --git`` header is checked. Hunk counts must match
    their bodies and at least one line must actually change.
    """
    if not diff.strip():
        return ["Empty diff"]

    patch = extract_diff(diff)
    if not patch:
        return ["Missing diff --git header"]

    errors = []
    problem = check_patch(patch)
    if problem:
        errors.append(f"Corrupt patch: {problem}")

    changed_lines = [
        l for l in patch.splitlines()
        if (l.startswith("+") and not l.startswith("+++"))
        or (l.startswith("-") and not l.startswith("---"))
    ]
    if not changed_lines:
        errors.append("Diff contains no changed lines (only context)")

    return errors


# ── Instruction relevance scoring ────────────────────────────────────────

def score_relevance(instruction: str) -> float:
    """Score instruction relevance for Clojure Code Flow training.

    Returns a float 0.0–1.0 where higher = more relevant.

    Factors:
    - Length: too short (< 10 chars) or too long (> 500 chars) penalized
    - Clojure terms: presence of Clojure-specific keywords
    - Task specificity: whether it describes a concrete coding task
    """
    if not instruction or not instruction.strip():
        return 0.0

    text = instruction.lower().strip()
    score = 0.0

    # Length scoring: optimal 30–200 chars
    length = len(text)
    if length < 10:
        score += 0.1
    elif length < 30:
        score += 0.3
    elif length <= 200:
        score += 0.5
    elif length <= 500:
        score += 0.4
    else:
        score += 0.3

    # Clojure term density
    term_count = sum(1 for term in _CLOJURE_TERMS if term in text)
    if term_count >= 4:
        score += 0.3
    elif term_count >= 2:
        score += 0.2
    elif term_count >= 1:
        score += 0.1

    # Task specificity: looks like an actual instruction rather than
    # a generic label or commit message stub
    task_indicators = [
        "refactor", "fix", "add", "implement", "change", "update",
        "create", "remove", "replace", "convert", "migrate", "optimize",
        "should", "need to", "want to", "please",
    ]
    if any(ind in text for ind in task_indicators):
        score += 0.2

    return min(score, 1.0)


# ── Turn structure ───────────────────────────────────────────────────────

_THINK_OPEN, _THINK_CLOSE = "<think>\n", "\n</think>\n\n"


def split_think(text: str) -> Optional[Tuple[str, str]]:
    """(reasoning, content) of an assistant turn that opens with a closed,
    non-empty ``<think>`` block, else None."""
    if not text.startswith(_THINK_OPEN) or _THINK_CLOSE not in text:
        return None
    reasoning, content = text[len(_THINK_OPEN):].split(_THINK_CLOSE, 1)
    return (reasoning, content) if reasoning.strip() else None


def _eval_code(content: str) -> Optional[str]:
    """The code of an ``eval`` tool call, or None if it isn't one."""
    try:
        call = json.loads(content)
    except ValueError:
        return None
    if not isinstance(call, dict) or call.get("name") != "eval":
        return None
    code = (call.get("arguments") or {}).get("code")
    return code if isinstance(code, str) and code.strip() else None


def _conversation_errors(conv: List[dict]) -> Tuple[List[str], List[str]]:
    """(format errors, eval code) for a multi-turn record.

    Order: human, then (function_call, observation) pairs, then a final gpt
    turn that applies a diff. Every assistant turn opens with reasoning.
    """
    errors: List[str] = []
    codes: List[str] = []
    roles = [t.get("from") for t in conv]
    pairs_ok = all(roles[i:i + 2] == ["function_call", "observation"] for i in range(1, len(roles) - 1, 2))
    if len(roles) < 4 or roles[0] != "human" or roles[-1] != "gpt" or len(roles) % 2 or not pairs_ok:
        errors.append(f"turns are not human, (function_call, observation)*, gpt: {roles}")
        return errors, codes
    for n, turn in enumerate(conv[1:], 1):
        if turn["from"] == "observation":
            continue
        parts = split_think(turn.get("value", ""))
        if parts is None:
            errors.append(f"assistant turn {n} has no reasoning (<think>...</think>)")
            continue
        if turn["from"] == "function_call":
            code = _eval_code(parts[1])
            if code is None:
                errors.append(f"turn {n} is not an eval tool call")
            else:
                codes.append(code)
        elif not parts[1].startswith(";; apply:\n"):
            errors.append("final turn does not start with ';; apply:'")
    return errors, codes


# ── Full example validation ──────────────────────────────────────────────

def validate_example(example: dict, min_score: float = 0.4) -> ValidationResult:
    """Validate a single training example.

    Checks:
    1. Instruction present and relevant
    2. Clojure syntax in output code blocks
    3. Diff structure coherence
    4. Combined score meets threshold

    Args:
        example: The training example dict.
        min_score: Minimum total_score to be considered valid.

    Returns:
        ValidationResult with detailed findings.
    """
    result = ValidationResult()
    instruction = example.get("instruction", "")
    output = example.get("output", "")

    # Instruction check
    if not instruction.strip():
        result.warnings.append("missing instruction")
        result.instruction_score = 0.0
    else:
        result.instruction_score = score_relevance(instruction)
        if result.instruction_score < 0.2:
            result.warnings.append(f"low instruction relevance: {result.instruction_score:.2f}")

    conv = example.get("conversations")
    if conv:
        result.format_errors, code_blocks = _conversation_errors(conv)
    else:
        code_blocks = _extract_code_blocks(output)
        if split_think(output) is None:
            result.format_errors.append("output has no reasoning (<think>...</think>)")

    # Clojure syntax in the evaluated code
    if code_blocks:
        all_syntax_ok = True
        for code in code_blocks:
            syntax_errs = check_clojure_syntax(code)
            if syntax_errs:
                all_syntax_ok = False
                result.syntax_errors.extend(syntax_errs)
        result.syntax_score = 0.5 if all_syntax_ok else 0.0
    else:
        result.syntax_score = 0.5  # No code blocks to check, not a failure

    # Diff structure check
    diff_errors = check_diff_structure(output) + apply_section_errors(output)
    if diff_errors:
        result.diff_errors.extend(diff_errors)
        result.diff_score = 0.0
    else:
        result.diff_score = 0.5

    # Combined score
    result.total_score = (
        result.instruction_score * 0.3
        + result.syntax_score * 0.3
        + result.diff_score * 0.4
    )
    result.total_score = round(result.total_score, 2)

    # A corrupt patch is unusable training data whatever else scores well.
    result.is_valid = (result.total_score >= min_score and not result.diff_errors
                       and not result.format_errors)
    return result


def _extract_code_blocks(text: str) -> List[str]:
    """Clojure code in an output: every ``;; eval:`` form (multi-line forms
    included) and every ```clojure fenced block."""
    from ..synthetic.verify import extract_eval_blocks

    blocks = [b.form for b in extract_eval_blocks(text) if b.form]

    in_clj_block = False
    clj_lines: List[str] = []
    for line in split_lines(text):
        if line.strip().startswith("```clojure"):
            in_clj_block = True
            clj_lines = []
            continue
        if line.strip() == "```" and in_clj_block:
            in_clj_block = False
            if clj_lines:
                blocks.append("\n".join(clj_lines))
            continue
        if in_clj_block:
            clj_lines.append(line)

    return blocks


# ── Batch validation ─────────────────────────────────────────────────────

def validate_jsonl_file(input_path: Path, output_path: Path,
                        min_score: float = 0.4,
                        filtered_path: Optional[Path] = None) -> dict:
    """Validate all examples in a JSONL file and write a report.

    Args:
        input_path: Path to JSONL file to validate.
        output_path: Path to write JSON validation report.
        min_score: Minimum score to consider valid.
        filtered_path: If given, write only the valid records here (may be
            input_path itself, to drop invalid records in place).

    Returns:
        Summary dict with total, valid, invalid counts and scores.
    """
    records = load_jsonl(input_path)

    results = []
    kept = []
    scores = []

    for i, rec in enumerate(records):
        vr = validate_example(rec, min_score=min_score)
        entry = {"index": i, **vr.to_dict()}
        results.append(entry)
        scores.append(vr.total_score)
        if vr.is_valid:
            kept.append(rec)
    valid_count = len(kept)
    if filtered_path is not None:
        write_jsonl(kept, filtered_path)

    summary = {
        "total": len(records),
        "valid": valid_count,
        "invalid": len(records) - valid_count,
        "avg_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "min_score_threshold": min_score,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    return summary
