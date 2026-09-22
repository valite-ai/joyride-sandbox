"""HTTP client for webhook delivery."""

from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
from http.client import HTTPConnection, HTTPException, HTTPSConnection
import json
import socket
import threading
import time
from typing import Any, Callable, Mapping
from urllib.error import HTTPError
from urllib.request import (
    HTTPHandler,
    HTTPRedirectHandler,
    HTTPSHandler,
    Request,
    build_opener,
)

from . import __version__

DELIVERED = "DELIVERED"
RETRYABLE = "RETRYABLE"
PERMANENT = "PERMANENT"


@dataclass(frozen=True, slots=True)
class Outcome:
    """Describe the result of one delivery attempt."""

    kind: str
    status: int | None = None
    error: str = ""
    retry_after: float | None = None


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _Deadline:
    """Shut down every socket of one attempt when the budget runs out.

    A socket timeout applies to each read, so a partner that sends one byte
    at a time can hold a request open forever without this.
    """

    def __init__(self) -> None:
        self.expired = False
        self._sockets: list[socket.socket] = []

    def expire(self) -> None:
        self.expired = True
        for sock in self._sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def create_connection(self, *args, **kwargs) -> socket.socket:
        sock = socket.create_connection(*args, **kwargs)
        self._sockets.append(sock)
        if self.expired:
            self.expire()
        return sock

    def connection(self, cls: type[HTTPConnection]) -> Callable[..., HTTPConnection]:
        def make(*args, **kwargs) -> HTTPConnection:
            conn = cls(*args, **kwargs)
            conn._create_connection = self.create_connection
            return conn

        return make


class _DeadlineHTTPHandler(HTTPHandler):
    def __init__(self, deadline: _Deadline) -> None:
        super().__init__()
        self._deadline = deadline

    def http_open(self, req):
        return self.do_open(self._deadline.connection(HTTPConnection), req)


class _DeadlineHTTPSHandler(HTTPSHandler):
    def __init__(self, deadline: _Deadline) -> None:
        super().__init__()
        self._deadline = deadline

    def https_open(self, req):
        return self.do_open(self._deadline.connection(HTTPSConnection), req)


class Client:
    """Send JSON webhook requests without following redirects."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock

    def post(
        self,
        url: str,
        payload: Any,
        headers: Mapping[str, str] | None = None,
        timeout: float = 10.0,
    ) -> Outcome:
        body = json.dumps(payload).encode("utf-8")
        request_headers = {
            "Content-Type": "application/json",
            "User-Agent": f"webhook-relay/{__version__}",
        }
        if headers:
            request_headers.update(headers)

        request = Request(url, data=body, headers=request_headers, method="POST")
        deadline = _Deadline()
        opener = build_opener(
            _NoRedirects(),
            _DeadlineHTTPHandler(deadline),
            _DeadlineHTTPSHandler(deadline),
        )
        timer = threading.Timer(timeout, deadline.expire)
        timer.daemon = True
        timer.start()
        try:
            with opener.open(request, timeout=timeout) as response:
                status = response.status
                return Outcome(self._kind(status), status)
        except HTTPError as exc:
            with exc:
                return Outcome(
                    self._kind(exc.code),
                    exc.code,
                    self._body_text(exc),
                    self._retry_after(exc.headers.get("Retry-After")),
                )
        except (HTTPException, OSError) as exc:
            if deadline.expired:
                return Outcome(RETRYABLE, error=f"delivery budget of {timeout}s exceeded")
            return Outcome(RETRYABLE, error=str(exc)[:4096])
        finally:
            timer.cancel()

    @staticmethod
    def _kind(status: int) -> str:
        if 200 <= status < 300:
            return DELIVERED
        if status in {408, 425, 429} or 500 <= status < 600:
            return RETRYABLE
        return PERMANENT

    @staticmethod
    def _body_text(response: Any) -> str:
        try:
            body = response.read(4096)
        except (HTTPException, OSError) as exc:
            return str(exc)[:4096]
        return body.decode("utf-8", errors="replace")

    def _retry_after(self, value: str | None) -> float | None:
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
            except (TypeError, ValueError, OverflowError):
                return None
            if retry_at.tzinfo is None:
                # RFC 5322 treats "-0000" as UTC, but Python returns a naive time.
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, retry_at.timestamp() - self._clock())
