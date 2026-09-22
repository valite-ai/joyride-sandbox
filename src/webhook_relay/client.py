"""HTTP client for webhook delivery."""

from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import json
import socket
import time
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

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


class Client:
    """Send JSON webhook requests without following redirects."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._opener = build_opener(_NoRedirects())

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
        try:
            with self._opener.open(request, timeout=timeout) as response:
                status = response.status
                return Outcome(self._kind(status), status)
        except HTTPError as exc:
            return Outcome(
                self._kind(exc.code),
                exc.code,
                self._body_text(exc),
                self._retry_after(exc.headers.get("Retry-After")),
            )
        except (TimeoutError, socket.timeout, URLError, OSError) as exc:
            return Outcome(RETRYABLE, error=str(exc)[:4096])

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
        except OSError as exc:
            return str(exc)[:4096]
        return body.decode("utf-8", errors="replace")

    def _retry_after(self, value: str | None) -> float | None:
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value).timestamp()
            except (TypeError, ValueError, OverflowError):
                return None
            return max(0.0, retry_at - self._clock())
