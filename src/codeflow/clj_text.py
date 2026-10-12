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
