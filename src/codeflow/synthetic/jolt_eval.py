"""Run Clojure forms in a sandboxed Jolt nREPL session and capture real results.

Grounds synthetic REPL traces in actual execution. Each call starts its own
``jolt nrepl-server`` (with the jolt-lang/nrepl middleware for sessions and
interruptible eval), clones one session, and sends every form as its own
``eval`` op — the same wire exchange an editor or the inference-time agent has,
so values, printed output and errors are what a real nREPL returns.

A server process per trace (startup is ~0.15s) keeps traces isolated: jolt's
eval namespace is process-global, so concurrent sessions in one server could
interleave ``(ns ..)`` changes, and defs from one trace would leak into the next.

Sandboxing: the server runs with a minimal environment (no API keys), in a
throwaway work directory, and on macOS under ``sandbox-exec`` with no network
beyond localhost, no reads of $HOME except the dependency caches, and no writes
outside the work directory.
"""

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .nrepl_client import NReplClient

# jolt-lang/nrepl: sessions + interruptible eval over jolt's built-in server.
NREPL_GIT_URL = "https://github.com/jolt-lang/nrepl"
NREPL_GIT_TAG = "v0.2.1"
NREPL_GIT_SHA = "aad8baef0dfe56d288792017841f37321d8160b1"

_DEPS_EDN = f"""{{:deps {{jolt-lang/nrepl {{:git/url "{NREPL_GIT_URL}"
                         :git/tag "{NREPL_GIT_TAG}" :git/sha "{NREPL_GIT_SHA}"}}}}
 :nrepl/middleware [nrepl.middleware/default-middleware]}}
"""

# Values are printed with this *print-length*, so an infinite seq returns
# instead of hanging the printer.
PRINT_LENGTH = 100

# Per-form cap on a captured value/output. A runaway form can print hundreds of
# MB; that is not a useful training signal.
_MAX_FIELD_CHARS = 8000

_STARTUP_TIMEOUT = 30.0
_INTERRUPT_GRACE = 3.0

_SANDBOX_PROFILE = """(version 1)
(allow default)
(deny network-outbound)
(allow network-outbound (remote ip "localhost:*"))
(deny file-write*)
(allow file-write* (subpath "{workdir}") (literal "/dev/null") (literal "/dev/tty"))
(deny file-read* (subpath "{home}"))
(allow file-read* (literal "{home}") (subpath "{home}/.jolt") (subpath "{home}/.m2")
       (subpath "{workdir}"))
"""


@dataclass
class EvalResult:
    """Outcome of evaluating one form over nREPL."""
    form: str
    value: str = ""      # printed return value (the last one, if the code held several forms)
    stdout: str = ""     # what the form printed to *out*
    ok: bool = True
    error: str = ""      # the nREPL err text (or why the form didn't run)
    ns: str = ""         # session namespace after the eval


def _truncate(text: str, limit: int = _MAX_FIELD_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]} ...[truncated {len(text)} chars]"


def jolt_available(jolt_path: str = "jolt") -> bool:
    return shutil.which(jolt_path) is not None


def _sandbox_available() -> bool:
    return sys.platform == "darwin" and shutil.which("sandbox-exec") is not None


_deps_lock = threading.Lock()
_deps_ready = False


def _prefetch_deps(jolt_path: str) -> None:
    """Resolve the nrepl library once, unsandboxed, before any generated code runs.

    The sandbox can't write the dependency caches, so a first sandboxed start
    would fail to fetch. Starting a server and stopping it runs no user code.
    """
    global _deps_ready
    with _deps_lock:
        if _deps_ready:
            return
        workdir = tempfile.mkdtemp(prefix="jolt-deps-")
        try:
            server = _JoltServer(jolt_path, sandbox=False, workdir=workdir)
            server.stop()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        _deps_ready = True


class _JoltServer:
    """A ``jolt nrepl-server`` process in its own work directory."""

    def __init__(self, jolt_path: str, sandbox: bool, workdir: str):
        self.workdir = workdir
        Path(workdir, "deps.edn").write_text(_DEPS_EDN)
        tmp = Path(workdir, "tmp")
        tmp.mkdir(exist_ok=True)
        jolt_bin = shutil.which(jolt_path) or jolt_path
        env = {
            "PATH": os.pathsep.join([str(Path(jolt_bin).parent), "/usr/bin", "/bin"]),
            "HOME": os.path.expanduser("~"),
            "TMPDIR": str(tmp),
        }
        cmd = [jolt_bin, "nrepl-server", "0"]
        if sandbox:
            real = os.path.realpath(workdir)
            profile = Path(workdir, "sandbox.sb")
            profile.write_text(_SANDBOX_PROFILE.format(
                workdir=real, home=os.path.realpath(env["HOME"])))
            cmd = ["sandbox-exec", "-f", str(profile)] + cmd
        self.proc = subprocess.Popen(
            cmd, cwd=workdir, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", start_new_session=True,
        )
        self.port = self._await_port()

    def _await_port(self) -> int:
        """Read the startup banner ("... started on port N ...") for the bound port."""
        found: dict = {}

        stdout = self.proc.stdout
        assert stdout is not None

        def read():
            lines = []
            for line in stdout:
                lines.append(line)
                if "started on port" in line:
                    found["port"] = int(line.split("started on port", 1)[1].split()[0])
                    break
            found["log"] = "".join(lines)
            # Keep draining so the server never blocks on a full stdout pipe.
            for _ in stdout:
                pass

        threading.Thread(target=read, daemon=True).start()
        deadline = time.monotonic() + _STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if "port" in found:
                return found["port"]
            if self.proc.poll() is not None and "log" in found:
                break
            time.sleep(0.02)
        self.stop()
        raise RuntimeError(f"jolt nrepl-server failed to start: {found.get('log', '')[-2000:]}")

    def stop(self) -> None:
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.proc.wait()


def _result(form: str, responses: List[dict]) -> EvalResult:
    values, out, err = [], [], []
    statuses, ns = set(), ""
    for r in responses:
        if "value" in r:
            values.append(r["value"])
        if "out" in r:
            out.append(r["out"])
        if "err" in r:
            err.append(r["err"])
        if "ns" in r:
            ns = r["ns"]
        statuses.update(r.get("status", []))
    failed = "eval-error" in statuses or "interrupted" in statuses
    error = "".join(err).strip()
    if failed and not error:
        error = next((r["ex"] for r in responses if "ex" in r), "eval error")
    return EvalResult(
        form=form,
        value=_truncate(values[-1]) if values else "",
        stdout=_truncate("".join(out)),
        ok=not failed,
        error=_truncate(error) if failed else "",
        ns=ns,
    )


def eval_forms(
    forms: List[str],
    timeout: float = 10.0,
    jolt_path: str = "jolt",
    sandbox: Optional[bool] = None,
) -> List[EvalResult]:
    """Evaluate ``forms`` in order in one fresh nREPL session.

    Each form gets ``timeout`` seconds; a form that runs over is interrupted and
    reported as failed, and the session carries on with the next form. If the
    interrupt doesn't take (a form blocked in a foreign call), the server is
    killed and the remaining forms are reported as not run.

    ``sandbox`` defaults to on wherever ``sandbox-exec`` exists (macOS).
    """
    if not forms:
        return []
    if sandbox is None:
        sandbox = _sandbox_available()
    if sandbox:
        _prefetch_deps(jolt_path)

    workdir = tempfile.mkdtemp(prefix="jolt-eval-")
    server = client = None
    results: List[EvalResult] = []
    try:
        server = _JoltServer(jolt_path, sandbox=sandbox, workdir=workdir)
        client = NReplClient("127.0.0.1", server.port)
        session = client.request({"op": "clone"}, _STARTUP_TIMEOUT)[-1]["new-session"]
        client.request({
            "op": "eval", "session": session,
            "code": f"(alter-var-root #'*print-length* (constantly {PRINT_LENGTH}))",
        }, _STARTUP_TIMEOUT)

        for i, form in enumerate(forms):
            msg_id = client.send({"op": "eval", "session": session, "code": form})
            try:
                results.append(_result(form, client.collect(msg_id, timeout)))
                continue
            except TimeoutError:
                pass
            client.send({"op": "interrupt", "session": session, "interrupt-id": msg_id})
            try:
                client.collect(msg_id, _INTERRUPT_GRACE)
            except (TimeoutError, ConnectionError):
                results.append(EvalResult(form=form, ok=False,
                                          error=f"timed out after {timeout}s"))
                results.extend(EvalResult(form=f, ok=False, error="not run: session lost")
                               for f in forms[i + 1:])
                break
            results.append(EvalResult(form=form, ok=False,
                                      error=f"timed out after {timeout}s (interrupted)"))
    except ConnectionError:
        done = len(results)
        results.extend(EvalResult(form=f, ok=False, error="not run: nREPL connection lost")
                       for f in forms[done:])
    finally:
        if client is not None:
            client.close()
        if server is not None:
            server.stop()
        shutil.rmtree(workdir, ignore_errors=True)
    return results
