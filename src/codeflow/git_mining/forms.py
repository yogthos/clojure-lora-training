"""Top-level Clojure forms in source text: where they are and what they define."""

from typing import Dict, List, Optional

from .diff_parser import parse_diff


def top_level_form_spans(content: str) -> List[tuple[int, int, str]]:
    """Top-level forms in Clojure source as (start_line, end_line, text).

    Lines are 1-indexed and inclusive. A form begins where bracket depth goes
    0 -> 1 and ends where it returns to 0, tracking strings, line comments, and
    character literals so delimiters inside them don't shift the depth.
    """
    spans: List[tuple[int, int]] = []
    depth = 0
    in_string = False
    escaped = False
    in_line_comment = False
    line_no = 1
    form_start: Optional[int] = None

    i, n = 0, len(content)
    while i < n:
        ch = content[i]
        if ch == "\n":
            in_line_comment = False
            line_no += 1
            i += 1
            continue
        if in_line_comment:
            i += 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == ";":
            in_line_comment = True
        elif ch == '"':
            in_string = True
        elif ch == "\\":  # character literal, e.g. \( — skip the next char
            i += 2
            continue
        elif ch in "([{":
            if depth == 0:
                form_start = line_no
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth <= 0:
                depth = 0
                if form_start is not None:
                    spans.append((form_start, line_no))
                    form_start = None
        i += 1

    if form_start is not None:  # unterminated form
        spans.append((form_start, line_no))

    lines = content.split("\n")
    return [(s, e, "\n".join(lines[s - 1:e])) for (s, e) in spans]


def _symbols(form: str, n: int) -> List[str]:
    """The first ``n`` tokens after the opening paren, skipping metadata
    (``^:private``, ``^String``, ``^{...}``)."""
    out, i, size = [], 1, len(form)
    while i < size and len(out) < n:
        c = form[i]
        if c.isspace() or c == ",":
            i += 1
        elif c == "^":
            i += 1
            if i < size and form[i] == "{":
                depth = 0
                while i < size:
                    depth += {"{": 1, "}": -1}.get(form[i], 0)
                    i += 1
                    if depth == 0:
                        break
            else:
                while i < size and not form[i].isspace():
                    i += 1
        else:
            j = i
            while j < size and not form[j].isspace() and form[j] not in "()[]{}\"":
                j += 1
            if j == i:
                break
            out.append(form[i:j])
            i = j
    return out


def form_name(form: str) -> Optional[str]:
    """What a top-level form defines: ``foo`` for ``(defn foo ...)``,
    ``area :circle`` for a defmethod, ``ns`` for the ns form; None otherwise."""
    if not form.startswith("("):
        return None
    toks = _symbols(form, 3)
    if not toks:
        return None
    head = toks[0].split("/")[-1]
    if head == "ns":
        return "ns"
    if not head.startswith("def") or len(toks) < 2:
        return None
    if head == "defmethod" and len(toks) >= 3:
        return f"{toks[1]} {toks[2]}"
    return toks[1]


def is_private(form: str) -> bool:
    """``defn-`` or ``^:private`` metadata: not visible from other namespaces."""
    toks = _symbols(form, 1)
    if toks and toks[0].endswith("-"):
        return True
    head = form[:200]
    return "^:private" in head or ":private true" in head


def changed_form_names(diff: str, contents: Dict[str, str],
                       side: str = "old") -> Dict[str, List[str]]:
    """Names of the top-level forms each file's hunks touch, in file order.

    ``side`` says which version ``contents`` holds: "old" (before the diff)
    or "new" (after it). Forms without a name (comment blocks, top-level
    calls) are left out.
    """
    out: Dict[str, List[str]] = {}
    for f in parse_diff(diff):
        content = contents.get(f.path)
        if content is None and f.old_path:
            content = contents.get(f.old_path)
        if not content:
            continue
        ranges = []
        for h in f.hunks:
            start, count = (h.old_start, h.old_count) if side == "old" else (h.new_start, h.new_count)
            if count:
                ranges.append((start, start + count - 1))
        names: List[str] = []
        for s, e, text in top_level_form_spans(content):
            if any(s <= re_ and rs <= e for rs, re_ in ranges):
                name = form_name(text)
                if name and name not in names:
                    names.append(name)
        if names:
            out[f.path] = names
    return out
