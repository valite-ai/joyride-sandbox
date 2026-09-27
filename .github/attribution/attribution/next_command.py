"""Print what to change next, from the hosted next steps page.

``joyride next`` reads ``GET /v1/me/next`` with the API token in
``JOYRIDE_API_TOKEN`` and prints the top three suggestions. ``--hook`` prints
one short block for a SessionStart hook, which the harness adds to the context
of the new session: the top change for the session's repository that the
session's agent can act on. A hook must never disturb a session, so ``--hook``
prints nothing on any error.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Any, TextIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from . import __version__
from .hosted import _remote_identity
from .install_code import InstallCodeError, _NoRedirect, _origin


TOKEN_ENV = "JOYRIDE_API_TOKEN"
URL_ENV = "ATTRIBUTION_HOSTED_URL"
DEFAULT_URL = "https://joyride.build"
TOP = 3
# A suggestion across many sessions can name dozens of pull requests; the text names the first ones.
REFERENCES = 6
# The hook block stays under this many characters.
HOOK_LIMIT = 400
# The service answers within the hook's own limit of 10 seconds.
HOOK_TIMEOUT_SECONDS = 8
TIMEOUT_SECONDS = 30
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_MAX_HOOK_INPUT_BYTES = 1024 * 1024
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class NextError(ValueError):
    """The next steps could not be read."""


def fetch(origin: str, token: str, *, timeout: float, fast: bool = False) -> dict[str, Any]:
    """Return the next steps page of the token's owner.

    With ``fast`` the service may answer with the page as it last computed it,
    at most a day old, instead of a full analysis.
    """
    request = Request(f"{origin}/v1/me/next{'?fast=1' if fast else ''}", headers={
        "Authorization": f"Bearer {token}", "Accept": "application/json", "User-Agent": f"joyride-cli/{__version__}",
    })
    try:
        # A redirect could carry the token to another origin.
        with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        if exc.code == 401:
            raise NextError(f"{origin} refused the token in {TOKEN_ENV}. "
                            f"Create an API token at {origin}/settings?tab=api-keys.") from exc
        try:
            message = json.loads(exc.read(4096)).get("error")
        except (OSError, ValueError, AttributeError):
            message = None
        raise NextError(message.strip()[:300] if isinstance(message, str) and message.strip()
                        else f"{origin} refused the request (HTTP {exc.code}).") from exc
    except (URLError, OSError) as exc:
        raise NextError(f"{origin} did not answer: {exc}") from exc
    try:
        payload = json.loads(raw) if len(raw) <= _MAX_RESPONSE_BYTES else None
    except ValueError:
        payload = None
    if not isinstance(payload, dict) or not isinstance(payload.get("suggestions"), list):
        raise NextError(f"{origin} sent an answer that Joyride cannot read.")
    return payload


def checkout_repository(directory: str) -> str | None:
    """Return the owner/name of the GitHub origin of the checkout in ``directory``."""
    try:
        result = subprocess.run(["git", "-C", directory, "remote", "get-url", "origin"], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    identity = _remote_identity(result.stdout) if result.returncode == 0 else None
    return "/".join(identity) if identity else None


def matching(payload: dict[str, Any], repository: str | None) -> list[dict[str, Any]]:
    """The suggestions that name a pull request of ``repository``, or all of them."""
    return [
        item for item in payload["suggestions"] if isinstance(item, dict) and (repository is None or any(
            isinstance(pr, dict) and str(pr.get("repository", "")).lower() == repository.lower()
            for pr in item.get("prs") or []
        ))
    ]


def _line(value: Any) -> str:
    from .terminal import safe_text

    return safe_text(" ".join(str(value or "").split()))


def render(payload: dict[str, Any], repository: str | None, origin: str) -> str:
    """The top three suggestions as plain text."""
    days = (payload.get("window") or {}).get("range_days")
    window = f"the last {days} days" if days else "all time"
    where = f" in {repository}" if repository else ""
    suggestions = matching(payload, repository)[:TOP]
    # When the service names the repositories that the token check admitted, a
    # repository outside that list needs the App or a wider token, not more work.
    covered = (payload.get("coverage") or {}).get("repositories")
    if not suggestions and repository and isinstance(covered, list) \
            and repository.lower() not in {str(name).lower() for name in covered}:
        return (f"Joyride does not cover {repository} with this token: install the App on it or widen the token.\n"
                f"Tokens: {origin}/settings?tab=api-keys")
    if not suggestions:
        return f"Joyride has no suggestion{where} for {window}.\nThe page: {origin}/next"
    lines = [f"What to change next{where}, from your pull requests of {window}:"]
    for position, item in enumerate(suggestions, 1):
        dollars = item.get("dollars")
        cost = "cost unknown" if not isinstance(dollars, (int, float)) else f"${dollars:,.2f}"
        prs = [f"{pr['repository']}#{pr.get('pr_number')}" for pr in item.get("prs") or []
               if isinstance(pr, dict) and pr.get("repository")]
        more = [f"and {len(prs) - REFERENCES} more"] if len(prs) > REFERENCES else []
        lines += [
            "",
            f"{position}. {_line(item.get('title'))} ({', '.join([cost, *map(_line, prs[:REFERENCES]), *more])})",
            f"   {_line(item.get('what_happened'))}",
            f"   Next: {_line(item.get('next_step'))}",
        ]
    lines += ["", f"Evidence and actions: {origin}/next"]
    return "\n".join(lines)


def hook_block(payload: dict[str, Any], repository: str) -> str:
    """The title and the next step of the top change for ``repository``, under 400 characters.

    Only a change that the agent can act on prints: one whose action is a pull
    request or a prompt for the agent. A command, such as a trial at another
    effort, is the developer's decision.
    """
    top = next((item for item in matching(payload, repository) if item.get("kind") == "change"
                and any((item.get("actions") or {}).get(key) for key in ("pull_request", "agent_prompt"))), None)
    if top is None:
        return ""
    text = (f"Joyride advice from your recent sessions in {_line(repository)}: "
            f"{_line(top.get('title'))}. {_line(top.get('next_step'))}")
    return text if len(text) < HOOK_LIMIT else text[:HOOK_LIMIT - 2].rstrip() + "…"


def _session_directory(stdin: TextIO | None) -> str:
    """The session's directory from the hook payload on stdin, else the current directory."""
    if stdin is not None and not stdin.isatty():
        raw = stdin.read(_MAX_HOOK_INPUT_BYTES)
        payload = json.loads(raw) if raw.strip() else None
        if isinstance(payload, dict) and isinstance(payload.get("cwd"), str) and payload["cwd"]:
            return payload["cwd"]
    return os.getcwd()


def _hosted_origin() -> str:
    try:
        return _origin(os.environ.get(URL_ENV) or DEFAULT_URL)
    except InstallCodeError as exc:
        raise NextError(f"{URL_ENV} must be an HTTPS origin without a path, query, or fragment.") from exc


def run(repository: str | None, *, as_json: bool, hook: bool,
        stdout: TextIO | None = None, stdin: TextIO | None = None) -> int:
    """Print the next steps and return the exit status. In a hook, print nothing on any error."""
    output = stdout or sys.stdout
    if hook:
        try:
            token = os.environ.get(TOKEN_ENV, "").strip()
            if not token:
                return 0
            selected = repository or checkout_repository(_session_directory(stdin if stdin is not None else sys.stdin))
            if not selected:
                return 0
            block = hook_block(fetch(_hosted_origin(), token, timeout=HOOK_TIMEOUT_SECONDS, fast=True), selected)
            if block:
                output.write(block + "\n")
        except Exception:
            pass
        return 0
    if repository is not None and not _REPOSITORY.fullmatch(repository):
        raise NextError("--repo must be a GitHub owner/repository name, for example acme/api.")
    origin = _hosted_origin()
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        raise NextError(f"Set {TOKEN_ENV} to a Joyride API token. Create one at {origin}/settings?tab=api-keys.")
    payload = fetch(origin, token, timeout=TIMEOUT_SECONDS)
    if as_json:
        output.write(json.dumps({**payload, "suggestions": matching(payload, repository)}, indent=2,
                                ensure_ascii=False) + "\n")
    else:
        output.write(render(payload, repository, origin) + "\n")
    return 0
