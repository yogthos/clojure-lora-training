"""Run Clojure forms in a sandboxed Jolt nREPL session and capture real results.

Grounds synthetic REPL traces in actual execution. Each call starts its own
``jolt nrepl-server`` (with the jolt-lang/nrepl middleware for sessions and
interruptible eval), clones one session, and sends every form as its own
``eval`` op — the same wire exchange an editor or the inference-time agent has,
so values, printed output and errors are what a real nREPL returns.

A server process per trace (startup is ~0.15s) keeps traces isolated: jolt's
eval namespace is process-global, so concurrent sessions in one server could
interleave ``(ns ..)`` changes, and defs from one trace would leak into the next.

Sandboxing: the server runs with a minimal environment (PATH and JOLT_*
settings only, no API keys), in a
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
from typing import Callable, Dict, List, Optional

from .nrepl_client import NReplClient

# jolt-lang/nrepl: sessions + interruptible eval over jolt's built-in server.
NREPL_GIT_URL = "https://github.com/jolt-lang/nrepl"
NREPL_GIT_TAG = "v0.3.0"
NREPL_GIT_SHA = "d779043d18235a83025c2800b60c9080c1bc9adc"

_DEPS_EDN = f"""{{:deps {{jolt-lang/nrepl {{:git/url "{NREPL_GIT_URL}"
                         :git/tag "{NREPL_GIT_TAG}" :git/sha "{NREPL_GIT_SHA}"}}}}
 :nrepl/middleware [nrepl.middleware/default-middleware]
 :paths [{{paths}}]}}
"""


def _deps_edn(paths: List[str]) -> str:
    return _DEPS_EDN.replace("{paths}", " ".join(f'"{p}"' for p in paths))

# Values are printed with this *print-length*, so an infinite seq returns
# instead of hanging the printer.
PRINT_LENGTH = 100

# Per-form cap on a captured value/output. A runaway form can print hundreds of
# MB; that is not a useful training signal.
_MAX_FIELD_CHARS = 8000

# A server that outgrows this resident size is killed (runaway allocation).
DEFAULT_MAX_RSS_MB = 2048
# Chez grows the stack on the heap. jolt throws StackOverflowError past
# JOLT_STACK_LIMIT; peak RSS is ~2.5x the limit, so a quarter of the RSS cap
# overflows the stack, like the JVM, well before the watchdog fires. jolt
# releases without the setting ignore it and the watchdog catches recursion.
_STACK_LIMIT_FRACTION = 4
_RSS_POLL_INTERVAL = 0.25

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
       (subpath "{workdir}"){extra_reads})
"""


def _server_env(jolt_bin: str, tmpdir: str, max_rss_mb: Optional[int] = None) -> dict:
    """The server's environment: PATH (jolt's launcher looks for Chez there),
    jolt's own JOLT_* settings, and nothing else, so API keys stay out."""
    env = {k: v for k, v in os.environ.items() if k.startswith("JOLT_")}
    if max_rss_mb and "JOLT_STACK_LIMIT" not in env:
        env["JOLT_STACK_LIMIT"] = f"{max_rss_mb // _STACK_LIMIT_FRACTION}m"
    path = os.environ.get("PATH", "/usr/bin:/bin")
    env.update({
        "PATH": os.pathsep.join([str(Path(jolt_bin).parent), path]),
        "HOME": os.path.expanduser("~"),
        "TMPDIR": tmpdir,
    })
    return env


def _sandbox_profile(workdir: str, home: str, jolt_bin: str) -> str:
    """sandbox-exec profile. jolt's install root stays readable: a source
    checkout under $HOME loads its own .ss files at startup. A root that
    would expose all of $HOME is left out."""
    root = Path(os.path.realpath(jolt_bin)).parent.parent
    extra = ""
    if not Path(home).is_relative_to(root):
        extra = f' (subpath "{root}")'
        # The launcher cd's into root, which needs the directories above it.
        if root.is_relative_to(home):
            extra += "".join(f' (literal "{d}")' for d in root.parents
                             if d.is_relative_to(home) and d != Path(home))
    return _SANDBOX_PROFILE.format(workdir=workdir, home=home, extra_reads=extra)


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

    def __init__(self, jolt_path: str, sandbox: bool, workdir: str,
                 max_rss_mb: Optional[int] = None, paths: Optional[List[str]] = None):
        self.workdir = workdir
        self.killed_for_memory = False
        Path(workdir, "deps.edn").write_text(_deps_edn(paths or ["src"]))
        tmp = Path(workdir, "tmp")
        tmp.mkdir(exist_ok=True)
        jolt_bin = shutil.which(jolt_path) or jolt_path
        env = _server_env(jolt_bin, str(tmp), max_rss_mb)
        cmd = [jolt_bin, "nrepl-server", "0"]
        if sandbox:
            real = os.path.realpath(workdir)
            profile = Path(workdir, "sandbox.sb")
            profile.write_text(_sandbox_profile(
                real, os.path.realpath(env["HOME"]), jolt_bin))
            cmd = ["sandbox-exec", "-f", str(profile)] + cmd
        self.proc = subprocess.Popen(
            cmd, cwd=workdir, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", start_new_session=True,
        )
        self.port = self._await_port()
        if max_rss_mb:
            threading.Thread(target=self._watch_memory, args=(max_rss_mb,), daemon=True).start()

    def _rss_mb(self) -> Optional[float]:
        r = subprocess.run(["ps", "-o", "rss=", "-p", str(self.proc.pid)],
                           capture_output=True, text=True)
        try:
            return int(r.stdout.strip()) / 1024
        except ValueError:
            return None

    def _watch_memory(self, max_rss_mb: int) -> None:
        """Kill the server once it outgrows ``max_rss_mb``; the open eval then
        fails with a connection error, reported as out of memory."""
        while self.proc.poll() is None:
            rss = self._rss_mb()
            if rss is not None and rss > max_rss_mb:
                self.killed_for_memory = True
                self.stop()
                return
            time.sleep(_RSS_POLL_INTERVAL)

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


def render_observation(result: EvalResult) -> str:
    """The text an agent sees after an eval: printed output, then ``=> value``
    or the nREPL error. Training traces and the inference agent both use this,
    so the model is trained on exactly what it will be shown."""
    parts = []
    if result.stdout:
        parts.append(result.stdout.rstrip("\n"))
    parts.append(f"=> {result.value}" if result.ok else result.error)
    return "\n".join(parts)


def _check_paths(files: Dict[str, str]) -> List[str]:
    """Source roots for ``files`` (their top-level directories, plus src).
    Paths must stay inside the work directory."""
    roots = {"src"}
    for rel in files:
        parts = Path(rel).parts
        if Path(rel).is_absolute() or ".." in parts or not parts:
            raise ValueError(f"file path outside the session: {rel!r}")
        if len(parts) > 1:
            roots.add(parts[0])
    return sorted(roots)


class JoltSession:
    """One nREPL session in its own sandboxed ``jolt nrepl-server``.

    Forms evaluate one at a time, so a caller can decide the next form from
    the last result. A form that runs over ``timeout`` is interrupted and the
    session carries on. If the interrupt doesn't take, or the server outgrows
    ``max_rss_mb``, the session is lost and later evals report "not run".
    """

    def __init__(self, timeout: float = 10.0, jolt_path: str = "jolt",
                 sandbox: Optional[bool] = None,
                 max_rss_mb: Optional[int] = DEFAULT_MAX_RSS_MB,
                 files: Optional[Dict[str, str]] = None):
        self.timeout = timeout
        self.max_rss_mb = max_rss_mb
        self.alive = False
        self._lost_reason = "not run: session lost"
        if sandbox is None:
            sandbox = _sandbox_available()
        if sandbox:
            _prefetch_deps(jolt_path)
        paths = _check_paths(files or {})
        self._workdir = tempfile.mkdtemp(prefix="jolt-eval-")
        self._server: Optional[_JoltServer] = None
        self._client: Optional[NReplClient] = None
        try:
            for rel, content in (files or {}).items():
                target = Path(self._workdir, rel)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
            self._server = _JoltServer(jolt_path, sandbox=sandbox, workdir=self._workdir,
                                       max_rss_mb=max_rss_mb, paths=paths)
            self._client = NReplClient("127.0.0.1", self._server.port)
            self._session = self._client.request({"op": "clone"}, _STARTUP_TIMEOUT)[-1]["new-session"]
            self._client.request({
                "op": "eval", "session": self._session,
                "code": f"(alter-var-root #'*print-length* (constantly {PRINT_LENGTH}))",
            }, _STARTUP_TIMEOUT)
            self.alive = True
        except ConnectionError:
            self._lost_reason = "not run: nREPL connection lost"
        except BaseException:
            self.close()
            raise

    def eval(self, code: str) -> EvalResult:
        if not self.alive:
            return EvalResult(form=code, ok=False, error=self._lost_reason)
        client, server = self._client, self._server
        assert client is not None and server is not None
        try:
            msg_id = client.send({"op": "eval", "session": self._session, "code": code})
            try:
                return _result(code, client.collect(msg_id, self.timeout))
            except TimeoutError:
                pass
            client.send({"op": "interrupt", "session": self._session, "interrupt-id": msg_id})
            try:
                client.collect(msg_id, _INTERRUPT_GRACE)
            except (TimeoutError, ConnectionError):
                self.alive = False
                return EvalResult(form=code, ok=False, error=f"timed out after {self.timeout}s")
            return EvalResult(form=code, ok=False,
                              error=f"timed out after {self.timeout}s (interrupted)")
        except ConnectionError:
            self.alive = False
            if server.killed_for_memory:
                return EvalResult(form=code, ok=False,
                                  error=f"out of memory: server exceeded {self.max_rss_mb}MB")
            self._lost_reason = "not run: nREPL connection lost"
            return EvalResult(form=code, ok=False, error=self._lost_reason)

    def close(self) -> None:
        self.alive = False
        if self._client is not None:
            self._client.close()
        if self._server is not None:
            self._server.stop()
        shutil.rmtree(self._workdir, ignore_errors=True)

    def __enter__(self) -> "JoltSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def session_factory(jolt_path: str = "jolt") -> Callable[..., "JoltSession"]:
    """Opens sessions on one jolt binary; keyword arguments (``files=``) pass
    through to JoltSession."""
    return lambda **kw: JoltSession(jolt_path=jolt_path, **kw)


def eval_forms(
    forms: List[str],
    timeout: float = 10.0,
    jolt_path: str = "jolt",
    sandbox: Optional[bool] = None,
    max_rss_mb: Optional[int] = DEFAULT_MAX_RSS_MB,
) -> List[EvalResult]:
    """Evaluate ``forms`` in order in one fresh nREPL session (see JoltSession).

    ``sandbox`` defaults to on wherever ``sandbox-exec`` exists (macOS). A
    server that grows past ``max_rss_mb`` is killed; that form is reported out
    of memory and the rest as not run. Runaway recursion throws
    StackOverflowError first (see ``_STACK_LIMIT_FRACTION``).
    """
    if not forms:
        return []
    with JoltSession(timeout=timeout, jolt_path=jolt_path, sandbox=sandbox,
                     max_rss_mb=max_rss_mb) as session:
        return [session.eval(f) for f in forms]
