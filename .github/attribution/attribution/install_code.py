"""Report a finished install to the hosted service with a one-time code.

A team invite gives each teammate an install command that carries a one-time
code. After the install finishes, the CLI sends only that code, so the
walkthrough can show that capture is installed before the first push. An install
without a code sends nothing.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


_CODE = re.compile(r"^jri_[A-Za-z0-9_-]{32}$")
_LOGIN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
_TIMEOUT_SECONDS = 10
_MAX_RESPONSE_BYTES = 4096


class InstallCodeError(ValueError):
    """The hosted service did not accept the install code."""


class _NoRedirect(HTTPRedirectHandler):
    # A redirect could carry the code to another origin.
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _origin(value: str) -> str:
    candidate = value.strip().rstrip("/")
    parsed = urlsplit(candidate)
    try:
        port = parsed.port
    except ValueError:
        port = -1
    secure = parsed.scheme == "https" or (
        parsed.scheme == "http" and parsed.hostname in _LOOPBACK
    )
    if (
        not secure
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or port is not None and not 1 <= port <= 65535
    ):
        raise InstallCodeError(
            "--hosted-url must be an HTTPS origin without a path, query, or fragment."
        )
    return candidate


def _refusal(exc: HTTPError) -> str:
    try:
        payload = json.loads(exc.read(_MAX_RESPONSE_BYTES))
    except (OSError, ValueError):
        payload = None
    message = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(message, str) and message.strip():
        return message.strip()[:300]
    return f"The hosted service refused the install code (HTTP {exc.code})."


def redeem_install_code(hosted_url: str, code: str) -> dict[str, str | None]:
    """Send the code once and return the GitHub login and team that it names."""

    if not isinstance(code, str) or not _CODE.fullmatch(code):
        raise InstallCodeError(
            "The install code is malformed. Copy the command from your invite page again."
        )
    request = Request(
        _origin(hosted_url) + "/v1/install-codes/redeem",
        data=json.dumps({"code": code}).encode("utf-8"),
        method="POST",
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    try:
        with build_opener(_NoRedirect()).open(request, timeout=_TIMEOUT_SECONDS) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        raise InstallCodeError(_refusal(exc)) from exc
    except (URLError, OSError) as exc:
        raise InstallCodeError(f"The hosted service did not answer: {exc}") from exc
    try:
        payload = json.loads(raw) if len(raw) <= _MAX_RESPONSE_BYTES else None
    except ValueError:
        payload = None
    login = payload.get("login") if isinstance(payload, dict) else None
    team = payload.get("team") if isinstance(payload, dict) else None
    if not isinstance(login, str) or not _LOGIN.fullmatch(login):
        raise InstallCodeError("The hosted service sent an answer that Joyride cannot read.")
    return {"login": login, "team": team if isinstance(team, str) and team.strip() else None}
