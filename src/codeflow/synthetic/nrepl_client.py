"""Minimal nREPL client: bencode over a socket.

Decoding runs on a reader thread that feeds a queue, so a per-request timeout
never leaves the decoder halfway through a message.
"""

import queue
import socket
import threading
import time
from typing import BinaryIO, Dict, Iterator, List, Optional


def bencode(x) -> bytes:
    """Encode str/bytes/int/list/dict as bencode (dict keys sorted, as the spec requires)."""
    if isinstance(x, bool):
        raise TypeError("bencode has no booleans")
    if isinstance(x, int):
        return b"i%de" % x
    if isinstance(x, str):
        x = x.encode("utf-8")
    if isinstance(x, bytes):
        return b"%d:%s" % (len(x), x)
    if isinstance(x, (list, tuple)):
        return b"l" + b"".join(bencode(v) for v in x) + b"e"
    if isinstance(x, dict):
        items = sorted((k.encode("utf-8") if isinstance(k, str) else k, v) for k, v in x.items())
        return b"d" + b"".join(bencode(k) + bencode(v) for k, v in items) + b"e"
    raise TypeError(f"can't bencode {type(x).__name__}")


def _read_exact(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise EOFError("connection closed mid-message")
    return data


def _decode(f: BinaryIO, first: bytes):
    if first == b"i":
        digits = b""
        while (c := _read_exact(f, 1)) != b"e":
            digits += c
        return int(digits)
    if first in (b"l", b"d"):
        items = []
        while (c := _read_exact(f, 1)) != b"e":
            items.append(_decode(f, c))
        if first == b"l":
            return items
        return dict(zip(items[::2], items[1::2]))
    digits = first
    while (c := _read_exact(f, 1)) != b":":
        digits += c
    return _read_exact(f, int(digits)).decode("utf-8", "replace")


def bdecode_stream(f: BinaryIO) -> Iterator:
    """Yield top-level bencoded values from a binary stream until EOF."""
    while True:
        first = f.read(1)
        if not first:
            return
        yield _decode(f, first)


class NReplClient:
    """One connection; responses are routed to waiters by request id."""

    def __init__(self, host: str, port: int, connect_timeout: float = 5.0):
        self._sock = socket.create_connection((host, port), timeout=connect_timeout)
        self._sock.settimeout(None)
        self._inbox: "queue.Queue[Optional[dict]]" = queue.Queue()
        self._ids = 0
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self):
        try:
            for msg in bdecode_stream(self._sock.makefile("rb")):
                self._inbox.put(msg)
        except (OSError, EOFError, ValueError):
            pass
        self._inbox.put(None)  # connection gone

    def send(self, msg: Dict) -> str:
        self._ids += 1
        msg = dict(msg, id=str(self._ids))
        self._sock.sendall(bencode(msg))
        return msg["id"]

    def collect(self, msg_id: str, timeout: float) -> List[dict]:
        """Responses for ``msg_id`` up to its "done" status.

        Raises TimeoutError if "done" doesn't arrive in time, ConnectionError if
        the server goes away. Responses for other ids are dropped.
        """
        deadline = time.monotonic() + timeout
        out = []
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"no 'done' for request {msg_id} within {timeout}s")
            try:
                resp = self._inbox.get(timeout=left)
            except queue.Empty:
                continue
            if resp is None:
                raise ConnectionError("nREPL connection closed")
            if resp.get("id") != msg_id:
                continue
            out.append(resp)
            if "done" in resp.get("status", []):
                return out

    def request(self, msg: Dict, timeout: float) -> List[dict]:
        return self.collect(self.send(msg), timeout)

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass
