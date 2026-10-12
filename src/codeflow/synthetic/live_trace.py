"""Generate a multi-turn REPL trace one step at a time against real results.

The generator proposes a single step (its reasoning plus one eval, or the final
diff), the eval runs in a persistent Jolt nREPL session, and the real result is
shown before the next step. Reasoning after a failure therefore responds to the
error that actually happened, not to an output the generator imagined.

A finished trace becomes a LLaMA-Factory sharegpt record: the request, then
``function_call`` turns (``<think>`` + an ``eval`` tool call) alternating with
``observation`` turns (the rendered nREPL result, which carries no loss), then a
final ``gpt`` turn with the reasoning and the ``;; apply:`` diff.
"""

import json
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ...llm.provider import LLMAuthError, LLMError, LLMProvider
from ..clj_text import top_level_forms
from ..diffs import new_file_contents, repair_trace_diff
from .jolt_eval import EvalResult, JoltSession, render_observation
from .prompts import LIVE_SYSTEM

EVAL_TOOL = {
    "type": "function",
    "function": {
        "name": "eval",
        "description": "Evaluate Clojure code in the project's nREPL session. Returns what "
                       "it printed and its value (=> value), or the error.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Clojure code to evaluate"}},
            "required": ["code"],
        },
    },
}

MIN_EVALS = 3
MAX_STEPS = 24

_drops: Counter = Counter()
_drops_lock = threading.Lock()


def _dropped(reason: str) -> None:
    with _drops_lock:
        _drops[reason] += 1


def drop_counts() -> Dict[str, int]:
    """Why traces were dropped in this process, for run summaries."""
    with _drops_lock:
        return dict(_drops)

# A short unit (8-200 chars) repeated back-to-back 10+ times: a degenerate
# sampling loop, e.g. "'zip/xml-zip 'zip/seq-zip 'zip/xml-zip ...".
_REPETITION_RE = re.compile(r"(.{8,200}?)\1{9,}", re.DOTALL)


# Reasoning that talks about the generator's instructions rather than the
# task. It's training text, and the trained agent never sees those rules.
_PROMPT_LEAK = re.compile(
    r"\b(the|my|these|per the|following the|as the) (instructions?|rules)\b"
    r"|\b\d+-eval\b|\beval minimum\b|\bminimum (of )?\d+ evals?\b"
    r"|\bsession is (empty|fresh)\b|\bno (existing )?(files|code) (is |are )?(shown|given|provided)\b",
    re.IGNORECASE)


def leaks_prompt(text: str) -> bool:
    return bool(_PROMPT_LEAK.search(text))


def has_repetition_loop(text: str) -> bool:
    """True if generated text degenerates into a repeating loop.

    ``;; result:`` lines are skipped (real REPL output can legitimately
    repeat), as are units with under 4 distinct characters (whitespace
    padding, ``"xxxx"`` test strings).
    """
    own = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith(";; result:"))
    return any(len(set(m.group(1))) >= 4 for m in _REPETITION_RE.finditer(own))


_DEF_HEADS = ("def", "defn", "defn-", "defmacro", "defmulti", "defmethod", "defrecord",
              "deftype", "defprotocol", "defonce", "require", "ns", "import", "use",
              "in-ns", "refer")


def is_definition(code: str) -> bool:
    """True if every top-level form only defines or loads something.

    Definitions return a var or nil, so they "succeed" trivially; the end
    state is judged on the last eval that demonstrates something.
    """
    forms = top_level_forms(code)
    if not forms:
        return False
    for form in forms:
        if not form.startswith("("):
            return False
        head = form[1:].lstrip().split(None, 1)
        if not head or head[0].rstrip(")") not in _DEF_HEADS:
            return False
    return True


def think(reasoning: str) -> str:
    """Qwen's rendering of a reasoning block ahead of a turn's content."""
    return f"<think>\n{reasoning.strip()}\n</think>\n\n"


@dataclass
class Step:
    thinking: str
    code: str
    result: EvalResult


@dataclass
class LiveTrace:
    request: str
    context: str
    steps: List[Step] = field(default_factory=list)
    final_thinking: str = ""
    diff: str = ""
    repo: str = ""

    @property
    def user_input(self) -> str:
        return f"Project: {self.context}" if self.context.strip() else ""

    def to_record(self) -> dict:
        human = self.request + (f"\n{self.user_input}" if self.user_input else "")
        conv = [{"from": "human", "value": human}]
        for s in self.steps:
            call = json.dumps({"name": "eval", "arguments": {"code": s.code}}, ensure_ascii=False)
            conv.append({"from": "function_call", "value": think(s.thinking) + call})
            conv.append({"from": "observation", "value": render_observation(s.result)})
        final = think(self.final_thinking) + ";; apply:\n" + self.diff
        conv.append({"from": "gpt", "value": final})
        return {
            "instruction": self.request,
            "input": self.user_input,
            "output": final,
            "conversations": conv,
            "tools": json.dumps([EVAL_TOOL], ensure_ascii=False),
            "repo": self.repo,
        }


def _transcript(request: str, context: str, steps: List[Step]) -> str:
    parts = [f"User request: {request}"]
    if context.strip():
        parts.append(f"Project context: {context}")
    parts.append("")
    if not steps:
        parts.append("This is the first step.")
    else:
        parts.append("Session so far:")
        for i, s in enumerate(steps, 1):
            parts += [f"--- step {i} ---", f"thinking: {s.thinking}", "eval:", s.code,
                      "result:", render_observation(s.result)]
    parts += ["", "Reply with the next step as a JSON object."]
    return "\n".join(parts)


_MALFORMED = ("it was not one JSON object with \"thinking\", \"action\" (eval or apply) "
              "and \"code\" or \"diff\"")
_LEAKY = ("the thinking talked about the instructions, eval counts or the session "
          "starting empty; write it as your own reasoning about the task")


def _parse_step(raw):
    """(step, None) or (None, why it was rejected)."""
    try:
        # strict=False: generators put raw newlines inside JSON strings. Only
        # the first object counts; a reply that runs on into more steps
        # didn't see their results.
        step = (json.JSONDecoder(strict=False).raw_decode(raw.lstrip())[0]
                if isinstance(raw, str) else raw)
    except ValueError:
        return None, _MALFORMED
    if isinstance(step, list):
        step = step[0] if step else None
    if not isinstance(step, dict):
        return None, _MALFORMED
    thinking = str(step.get("thinking") or "").strip()
    action = step.get("action")
    body = str(step.get("code" if action == "eval" else "diff") or "")
    if not thinking or action not in ("eval", "apply") or not body.strip():
        return None, _MALFORMED
    if leaks_prompt(thinking):
        return None, _LEAKY
    return {"thinking": thinking, "action": action, "body": body}, None


def _next_step(llm: LLMProvider, prompt: str) -> Optional[dict]:
    """One generated step; a rejected reply gets one retry, told what was wrong."""
    attempt = prompt
    for _ in range(2):
        raw = llm.call(system_prompt=LIVE_SYSTEM, user_prompt=attempt, temperature=0.5,
                       max_tokens=8192, require_json=True, reject_truncated=True)
        step, why = _parse_step(raw)
        if step is not None:
            return step
        attempt = f"{prompt}\n\nYour last reply was rejected: {why}. Reply again with one step."
    return None


def run_live_trace(
    request: str,
    context: str,
    llm: LLMProvider,
    repo: str = "",
    session_factory: Optional[Callable[[], JoltSession]] = None,
    max_steps: int = MAX_STEPS,
    min_evals: int = MIN_EVALS,
) -> Optional[LiveTrace]:
    """Run the generate-eval loop to a final diff; None if the trace is unusable.

    Dropped: a failed or malformed generation, a session lost to a hang or
    memory, applying before ``min_evals`` evals, no apply within ``max_steps``,
    a diff that isn't a valid patch after @@ count repair, or degenerate
    repeating output. LLMAuthError propagates.
    """
    session = (session_factory or JoltSession)()
    trace = LiveTrace(request=request, context=context, repo=repo)
    try:
        for _ in range(max_steps):
            try:
                step = _next_step(llm, _transcript(request, context, trace.steps))
            except LLMAuthError:
                raise
            except LLMError as e:
                _dropped(f"generation failed: {type(e).__name__}")
                return None
            if step is None:
                _dropped("malformed step")
                return None
            if has_repetition_loop(step["thinking"] + "\n" + step["body"]):
                _dropped("repetition loop")
                return None

            if step["action"] == "apply":
                if len(trace.steps) < min_evals:
                    _dropped(f"applied before {min_evals} evals")
                    return None
                repaired = repair_trace_diff(";; apply:\n" + step["body"])
                if repaired is None:
                    _dropped("diff is not a valid patch")
                    return None
                trace.final_thinking = step["thinking"]
                trace.diff = repaired[len(";; apply:\n"):]
                return trace

            result = session.eval(step["body"])
            if not session.alive:
                _dropped("session lost")
                return None
            trace.steps.append(Step(step["thinking"], step["body"], result))
        _dropped(f"no apply within {max_steps} steps")
        return None
    finally:
        session.close()


def _ends_clean(trace: LiveTrace) -> bool:
    """The end state is the last eval that isn't a definition: it must run.
    Failures before it are the error->fix arc the data is meant to teach."""
    checks = [s.result for s in trace.steps if not is_definition(s.code)]
    return bool(checks) and checks[-1].ok


def _ns_name(content: str) -> Optional[str]:
    forms = top_level_forms(content)
    if forms and forms[0].startswith("("):
        head = forms[0][1:].split(None, 2)
        if len(head) >= 2 and head[0] == "ns":
            return head[1].rstrip(")")
    return None


def _new_files_load(trace: LiveTrace, session_factory: Callable[..., JoltSession]) -> bool:
    """Every file the diff creates loads the way a project would load it: the
    files sit on the session's source path and each namespace is required, so
    a test file can require its source and a namespace must match its path."""
    files = new_file_contents(trace.diff)
    if not files:
        return True
    try:
        session = session_factory(files=files)
    except ValueError:
        return False      # a path outside the project
    try:
        for content in files.values():
            ns = _ns_name(content)
            code = f"(require '{ns})" if ns else content
            if not session.eval(code).ok:
                return False
        return True
    finally:
        session.close()


def accept_trace(trace: Optional[LiveTrace], session_factory: Callable[[], JoltSession],
                 min_pass_rate: float = 0.6) -> Optional[float]:
    """The trace's pass rate if it is worth training on, else None.

    Kept only if it reaches a working end state, at least ``min_pass_rate`` of
    its evals ran cleanly, and the files its diff creates load.
    """
    if trace is None:
        return None
    if not _ends_clean(trace):
        _dropped("last check failed")
        return None
    pass_rate = sum(s.result.ok for s in trace.steps) / len(trace.steps)
    if pass_rate < min_pass_rate:
        _dropped(f"pass rate under {min_pass_rate}")
        return None
    if not _new_files_load(trace, session_factory):
        _dropped("diff's new files don't load")
        return None
    return pass_rate
