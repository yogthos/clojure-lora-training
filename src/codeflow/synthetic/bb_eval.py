"""Run Clojure forms through babashka and capture real results.

Used to ground synthetic REPL traces in actual execution instead of
LLM-fabricated ``;; result:`` lines. Forms are evaluated sequentially in a
single babashka process so state (defs, requires) persists across them, the
way a real nREPL session behaves.

Security note: this executes model-generated Clojure on the host. A subprocess
timeout bounds runaway loops, but babashka can still touch the filesystem and
shell. For untrusted input at scale, run inside a container (matching IQuest's
sandboxed-execution setup); here we rely on the timeout and a trusted pipeline.
"""

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Delimiters printed by the runner script to frame each form's output.
_R = "<<<R>>>"
_OUT = "<<<OUT>>>"
_VAL = "<<<VAL>>>"
_ERR = "<<<ERR>>>"

# Per-form cap on a captured value/stdout. A runaway form (a realized large or
# near-infinite seq) can print hundreds of MB; such a result is not a useful
# training signal, so it is truncated to a marker rather than carried whole.
_MAX_FIELD_CHARS = 8000


def _truncate(text: str, limit: int = _MAX_FIELD_CHARS) -> str:
    """Cap captured text, appending a marker noting the original length."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]} ...[truncated {len(text)} chars]"


@dataclass
class EvalResult:
    """Outcome of evaluating one form."""
    form: str
    value: str = ""      # REPL representation of the return value (pr-str)
    stdout: str = ""     # anything the form printed to *out*
    ok: bool = True
    error: str = ""


def bb_available(bb_path: str = "bb") -> bool:
    """True if the babashka binary is on PATH."""
    return shutil.which(bb_path) is not None


def _clj_string(s: str) -> str:
    """Render a Python string as a Clojure string literal."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _build_script(forms: List[str]) -> str:
    """Build a babashka script that evals each form and frames the result.

    Each form's own stdout is captured separately from its return value so a
    form that prints doesn't corrupt the value channel.
    """
    vec = " ".join(_clj_string(f) for f in forms)
    return (
        f"(doseq [s [{vec}]]\n"
        f'  (println "{_R}")\n'
        f"  (try\n"
        f"    (let [sw (java.io.StringWriter.)\n"
        f"          v (binding [*out* sw] (load-string s))]\n"
        f'      (print "{_OUT} ") (println (pr-str (str sw)))\n'
        f'      (print "{_VAL} ") (println (pr-str v)))\n'
        f"    (catch Throwable e\n"
        f'      (print "{_ERR} ") (println (pr-str (.getMessage e))))))\n'
    )


def _unescape(payload: str) -> str:
    """Decode a pr-str'd Clojure string back to raw text.

    Clojure's pr-str uses JSON-compatible escapes for the common cases, so
    json.loads handles it; fall back to a literal strip if it doesn't.
    """
    try:
        return json.loads(payload)
    except (ValueError, TypeError):
        return payload.strip().strip('"')


def _field(chunk: List[str], marker: str) -> Optional[str]:
    """Return the payload of the first line starting with ``marker``."""
    prefix = marker + " "
    for line in chunk:
        if line.startswith(prefix):
            return line[len(prefix):]
    return None


def _parse_output(stdout: str, forms: List[str]) -> List[EvalResult]:
    chunks: List[List[str]] = []
    cur: Optional[List[str]] = None
    for line in stdout.splitlines():
        if line.strip() == _R:
            if cur is not None:
                chunks.append(cur)
            cur = []
        elif cur is not None:
            cur.append(line)
    if cur is not None:
        chunks.append(cur)

    results: List[EvalResult] = []
    for i, form in enumerate(forms):
        if i >= len(chunks):
            results.append(EvalResult(form=form, ok=False, error="no output"))
            continue
        chunk = chunks[i]
        err = _field(chunk, _ERR)
        if err is not None:
            results.append(EvalResult(form=form, ok=False, error=_unescape(err)))
            continue
        val = _field(chunk, _VAL)
        out = _field(chunk, _OUT)
        results.append(EvalResult(
            form=form,
            value=_truncate((val or "").strip()),
            stdout=_truncate(_unescape(out)) if out else "",
            ok=True,
        ))
    return results


def eval_forms(
    forms: List[str],
    timeout: float = 30.0,
    bb_path: str = "bb",
) -> List[EvalResult]:
    """Evaluate forms sequentially in one babashka session.

    Returns one EvalResult per form, in order. A form that throws is marked
    not-ok with its message but does not abort later forms. A process timeout
    marks every form not-ok.
    """
    if not forms:
        return []

    script = _build_script(forms)
    # Run inside a throwaway working directory so a form that writes a relative
    # path (spit/io) can't pollute the caller's project; it's removed on exit.
    with tempfile.TemporaryDirectory() as workdir:
        script_path = Path(workdir) / "_eval.clj"
        script_path.write_text(script)
        try:
            proc = subprocess.run(
                [bb_path, str(script_path)],
                cwd=workdir,
                capture_output=True, text=True, errors="replace", timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return [EvalResult(form=f, ok=False, error="timeout") for f in forms]

        return _parse_output(proc.stdout, forms)
