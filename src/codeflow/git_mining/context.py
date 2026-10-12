"""What a mined transition shows besides the code it changes.

The change alone teaches "given this code and this request, write this
patch". Two more things let the model learn how a codebase evolves (IQuest
Finding 1: repository transitions beat static snapshots):

- recent history: the commits that last touched these files, and which
  definitions each one changed, so the model sees where the code has been
  heading;
- related code: the forms in other namespaces that use the definitions being
  changed, the repository-level context a change has to stay consistent with.
"""

import re
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List

from .diff_parser import parse_diff
from .forms import changed_form_names, form_name, is_private, top_level_form_spans
from .gitcmd import _DIFF_FLAGS, _GIT, get_file_content

__all__ = ["HistoryEntry", "changed_form_names", "file_history", "form_name",
           "format_history", "format_related", "related_forms"]

HISTORY_COMMITS = 5
_FORMS_PER_FILE = 5

RELATED_CHARS = 6000        # total budget for related code
_RELATED_FILES = 3
_RELATED_FORMS = 4          # per file
_RELATED_FORM_CHARS = 2500  # a caller longer than this isn't worth the space

_CLJ_GLOBS = ["*.clj", "*.cljs", "*.cljc"]


@dataclass
class HistoryEntry:
    date: str                                  # YYYY-MM-DD
    subject: str
    forms: Dict[str, List[str]] = field(default_factory=dict)   # path -> changed definitions


def file_history(repo: str, ref: str, paths: List[str],
                 n: int = HISTORY_COMMITS) -> List[HistoryEntry]:
    """The last ``n`` mainline commits up to and including ``ref`` that
    touched ``paths``, oldest first."""
    if not paths:
        return []
    cmd = [*_GIT, "-C", repo, "log", "--first-parent", "-m", f"-n{n}",
           "--format=%x01%H%x00%aI%x00%s", *_DIFF_FLAGS, "--unified=0", ref,
           "--", *paths]
    out = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if out.returncode != 0:
        return []
    entries = []
    for chunk in out.stdout.split("\x01")[1:]:
        header, _, patch = chunk.partition("\n")
        parts = header.split("\x00")
        if len(parts) < 3:
            continue
        sha, date, subject = parts[0], parts[1][:10], parts[2].strip()
        touched = [f.path for f in parse_diff(patch) if f.change_type != "deleted"]
        contents = {p: get_file_content(repo, sha, p) for p in touched}
        forms = {p: names[:_FORMS_PER_FILE]
                 for p, names in changed_form_names(patch, contents, side="new").items()}
        entries.append(HistoryEntry(date, subject, forms))
    return entries[::-1]


def format_history(entries: List[HistoryEntry]) -> str:
    if not entries:
        return ""
    lines = ["## Recent history of these files"]
    for e in entries:
        where = "; ".join(f"{p}: {', '.join(names)}" for p, names in e.forms.items())
        lines.append(f"- {e.date} {e.subject}" + (f" ({where})" if where else ""))
    return "\n".join(lines)


def _ns_name(content: str) -> str:
    for _, _, text in top_level_form_spans(content)[:3]:
        if form_name(text) == "ns":
            parts = text[1:].split(None, 2)
            if len(parts) >= 2:
                return parts[1].rstrip(")")
    return ""


def _uses(name: str) -> "re.Pattern":
    """``name`` as a whole Clojure symbol, bare or namespace-qualified."""
    edge = r"[\w\-?!*+<>=.'/]"
    return re.compile(rf"(?<!{edge})(?:[\w.\-]+/)?{re.escape(name)}(?!{edge})")


def related_forms(repo: str, ref: str, before: Dict[str, str], diff: str,
                  budget: int = RELATED_CHARS) -> Dict[str, str]:
    """Forms in other namespaces that use the public definitions ``diff``
    changes, as of ``ref``: path -> those forms, separated by ``;; ...``."""
    targets = []                               # (ns, [public names])
    changed = changed_form_names(diff, before, side="old")
    for path, names in changed.items():
        ns = _ns_name(before.get(path, ""))
        spans = {form_name(t): t for _, _, t in top_level_form_spans(before.get(path, ""))}
        public = [n.split(" ")[0] for n in names
                  if n != "ns" and not is_private(spans.get(n, ""))]
        if ns and public:
            targets.append((ns, list(dict.fromkeys(public))))
    if not targets:
        return {}

    skip = set(before) | {f.path for f in parse_diff(diff)}
    found: Dict[str, List[str]] = {}
    for ns, names in targets:
        grep = subprocess.run([*_GIT, "-C", repo, "grep", "-l", "-F", "-w", ns, ref,
                               "--", *_CLJ_GLOBS],
                              capture_output=True, text=True, errors="replace")
        files = [line.split(":", 1)[1] for line in grep.stdout.splitlines() if ":" in line]
        pats = [_uses(n) for n in names]
        for path in files:
            if path in skip:
                continue
            content = get_file_content(repo, ref, path)
            if _ns_name(content) == ns:
                continue
            hits = [t for _, _, t in top_level_form_spans(content)
                    if form_name(t) != "ns" and len(t) <= _RELATED_FORM_CHARS
                    and any(p.search(t) for p in pats)]
            if hits:
                found.setdefault(path, []).extend(h for h in hits if h not in found.get(path, []))

    out: Dict[str, str] = {}
    used = 0
    for path, forms in sorted(found.items(), key=lambda kv: -len(kv[1]))[:_RELATED_FILES]:
        kept = []
        for f in forms[:_RELATED_FORMS]:
            if used + len(f) > budget:
                break
            kept.append(f)
            used += len(f)
        if kept:
            out[path] = "\n;; ...\n".join(kept)
    return out


def format_related(related: Dict[str, str]) -> str:
    if not related:
        return ""
    parts = ["## Related code (uses the changed definitions)"]
    parts += [f"### {p}\n```clojure\n{body}\n```" for p, body in related.items()]
    return "\n".join(parts)
