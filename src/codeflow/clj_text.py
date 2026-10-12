"""Lexical scanning of Clojure text: brackets outside strings, chars and comments."""

from typing import Iterator, List, Tuple

_CLOSE = {")": "(", "]": "[", "}": "{"}


def _brackets(text: str) -> Iterator[Tuple[int, str]]:
    """Yield (index, char) for each bracket that is code, plus a final
    (len, '"') if the text ends inside a string.

    Skips string contents, character literals (``\\(``, ``\\"``) and ``;``
    comments.
    """
    in_str, i, n = False, 0, len(text)
    while i < n:
        c = text[i]
        if in_str:
            if c == "\\":
                i += 1
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "\\":
            i += 1  # character literal: skip the escaped char
        elif c == ";":
            nl = text.find("\n", i)
            i = n if nl < 0 else nl
            continue
        elif c in "([{)]}":
            yield i, c
        i += 1
    if in_str:
        yield n, '"'


def balance(text: str) -> Tuple[int, bool]:
    """Net bracket depth of ``text`` and whether it ends inside a string."""
    depth, in_str = 0, False
    for _, c in _brackets(text):
        if c == '"':
            in_str = True
        elif c in _CLOSE:
            depth -= 1
        else:
            depth += 1
    return depth, in_str


def bracket_errors(text: str) -> List[str]:
    """Unmatched, mismatched and unclosed brackets, and unterminated strings."""
    errors, stack = [], []
    for i, c in _brackets(text):
        line = text.count("\n", 0, i) + 1
        if c == '"':
            errors.append(f"Unterminated string at end of input (line {line})")
        elif c not in _CLOSE:
            stack.append((c, line))
        elif not stack:
            errors.append(f"Unmatched closing {c!r} at line {line}")
        elif stack[-1][0] != _CLOSE[c]:
            opener, at = stack.pop()
            errors.append(f"Mismatched {c!r} at line {line} closes {opener!r} from line {at}")
        else:
            stack.pop()
    errors.extend(f"Unmatched opening {c!r} from line {at}: unclosed" for c, at in stack)
    return errors


_DELIMS = set("()[]{}\";,") | {" ", "\t", "\n", "\r", "\f"}


def _skip_form(text: str, i: int) -> int:
    """Index just past the form starting at ``i`` (end of text if unbalanced)."""
    n = len(text)
    while i < n and text[i] in "'`~@^#":   # reader prefixes: 'x #{} #"re" ~@x
        i += 1
    if i >= n:
        return n
    c = text[i]
    if c == '"':
        i += 1
        while i < n and text[i] != '"':
            i += 2 if text[i] == "\\" else 1
        return min(i + 1, n)
    if c == "\\":
        i += 2
        while i < n and text[i] not in _DELIMS:
            i += 1
        return i
    if c in "([{":
        depth = 0
        while i < n:
            c = text[i]
            if c == '"':
                i = _skip_form(text, i)
                continue
            if c == "\\":
                i += 2
                continue
            if c == ";":
                nl = text.find("\n", i)
                i = n if nl < 0 else nl
                continue
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return n
    while i < n and text[i] not in _DELIMS:
        i += 1
    return i


def top_level_forms(text: str) -> List[str]:
    """The top-level forms in ``text``, skipping whitespace, commas and comments."""
    forms, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace() or c == ",":
            i += 1
        elif c == ";":
            nl = text.find("\n", i)
            i = n if nl < 0 else nl
        elif c in ")]}":
            i += 1    # stray closer: not a form
        else:
            end = _skip_form(text, i)
            forms.append(text[i:end])
            i = max(end, i + 1)
    return forms
