"""Dataset validator: check Clojure syntax, diff coherence, relevance scoring.

Validates Code Flow training examples and produces validation reports.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

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
            "warnings": self.warnings,
        }


# ── Clojure syntax check ─────────────────────────────────────────────────

def check_clojure_syntax(code: str) -> List[str]:
    """Check basic Clojure syntax validity.

    Checks for balanced parentheses and basic form structure.
    Does NOT do full parsing — just checks that parens balance
    and common structural issues are caught.

    Returns list of error messages (empty = valid).
    """
    errors = []

    depth = 0
    in_string = False
    in_comment = False
    prev_char = ""
    for i, ch in enumerate(code):
        # End of line resets comment state
        if ch == '\n':
            in_comment = False
            prev_char = ch
            continue

        # Track strings
        if ch == '"' and prev_char != '\\':
            in_string = not in_string

        if in_string:
            prev_char = ch
            continue

        if in_comment:
            prev_char = ch
            continue

        # Start of a line comment
        if ch == ';':
            in_comment = True
            prev_char = ch
            continue

        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth < 0:
                # Try to give a useful line number
                line_num = code[:i].count('\n') + 1
                errors.append(f"Unmatched closing parenthesis at line {line_num}")
                depth = 0

        prev_char = ch

    if depth > 0:
        errors.append(f"Unmatched opening parenthesis: {depth} unclosed")
    elif depth < 0:
        errors.append(f"Unmatched closing parenthesis: {-depth} extra")

    return errors


# ── Diff structure check ─────────────────────────────────────────────────

def check_diff_structure(diff: str) -> List[str]:
    """Validate unified diff structure.

    Checks for:
    - At least one diff --git header
    - Corresponding @@ hunk headers
    - Lines with +/- prefixes indicating actual changes
    - Basic file path format
    """
    errors = []

    if not diff.strip():
        errors.append("Empty diff")
        return errors

    has_header = "diff --git" in diff
    has_hunk = "@@" in diff
    has_old = "-" in diff
    has_new = "+" in diff

    if not has_header:
        errors.append("Missing diff --git header")

    if not has_hunk:
        errors.append("Missing hunk header (@@ -line,count +line,count @@)")

    if not (has_old or has_new):
        errors.append("No changed lines found (+/- lines)")

    # Check hunk header format for each hunk
    for i, line in enumerate(diff.splitlines()):
        if line.startswith("@@"):
            # Validate: @@ -digits[,digits] +digits[,digits] @@
            import re
            if not re.match(r'^@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@', line):
                errors.append(f"Malformed hunk header at line {i + 1}: {line[:60]}")

    # Verify there's at least one meaningful change (not just context)
    changed_lines = [
        l for l in diff.splitlines()
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

    # Clojure syntax — check code blocks in output
    code_blocks = _extract_code_blocks(output)
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
    diff_errors = check_diff_structure(output)
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

    result.is_valid = result.total_score >= min_score
    return result


def _extract_code_blocks(text: str) -> List[str]:
    """Extract Clojure code blocks from text.

    Looks for:
    - ;; eval: <form> patterns
    - ```clojure ``` blocks
    - Lines that look like top-level Clojure forms
    """
    blocks = []

    # Extract ;; eval: forms
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(";; eval:"):
            form = stripped[len(";; eval:"):].strip()
            if form:
                blocks.append(form)

    # Extract ```clojure blocks
    in_clj_block = False
    clj_lines = []
    for line in text.splitlines():
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
                        min_score: float = 0.4) -> dict:
    """Validate all examples in a JSONL file and write a report.

    Args:
        input_path: Path to JSONL file to validate.
        output_path: Path to write JSON validation report.
        min_score: Minimum score to consider valid.

    Returns:
        Summary dict with total, valid, invalid counts and scores.
    """
    records = load_jsonl(input_path)

    results = []
    valid_count = 0
    scores = []

    for i, rec in enumerate(records):
        vr = validate_example(rec, min_score=min_score)
        entry = {"index": i, **vr.to_dict()}
        results.append(entry)
        scores.append(vr.total_score)
        if vr.is_valid:
            valid_count += 1

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
