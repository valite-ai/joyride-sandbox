"""Hand one native hook event to the local collector and exit.

The machine launcher runs this module for every Claude Code and Codex hook.
A full CLI start imports the capture code and runs Git before it can decide
anything, which costs about 100 ms on every tool call. This client imports
only the standard library modules it needs, sends the event to the collector
that already runs for cost capture, and waits only as long as the event
requires: a ``PreToolUse`` waits until its snapshot exists, because the tool
runs next, and every other event waits only for a receipt.

The collector runs the same code as the CLI. When the collector cannot take
the event before it was sent, the client runs that code in this process, so
no event is lost. Like the CLI hook, the client prints nothing and exits 0 on
every path.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import time


HOOK_ENDPOINT_PATH = "/v1/hook"
HOOK_INPUT_LIMIT = 2 * 1024 * 1024
WORKFLOW_PATH = ".github/workflows/attribution-footer.yml"
STATE_DIR_ENV = "ATTRIBUTION_TELEMETRY_DIR"
DISABLE_TELEMETRY_ENV = "HARNESS_ATTRIBUTION_DISABLE_TELEMETRY"
HARNESSES = ("codex", "claude-code")

# A PreToolUse must finish before the tool runs. The native hook timeout is
# 10 seconds, so the client stops waiting a little before the tool would.
PRE_TOOL_WAIT_SECONDS = 9.0
RECEIPT_WAIT_SECONDS = 2.0
CONNECT_SECONDS = 0.5
# The collector must answer, and a snapshot must end, this long before the
# client releases the tool.
DEADLINE_MARGIN_SECONDS = 0.25
_RESPONSE_LIMIT = 64 * 1024

# The hook code reads these variables, directly or through the Git processes
# that it starts. Git reads HOME and XDG_CONFIG_HOME for its global
# configuration, ignore rules, and attributes. A collector serves an event
# only when its own values match the client's, so an event never runs under
# another session's configuration. The hook code does not read CODEX_HOME or
# CLAUDE_CONFIG_DIR, and PATH differs between coding tools without changing
# what Git reads, so those do not split sessions.
_ENVIRONMENT_NAMES = frozenset({"HOME", "XDG_CONFIG_HOME", "XDG_STATE_HOME"})
_ENVIRONMENT_PREFIXES = (
    "ATTRIBUTION_",
    "HARNESS_ATTRIBUTION_",
    "_HARNESS_ATTRIBUTION_",
    "GIT_",
)
# Every other GIT_ variable counts, including ones that a newer Git adds.
# These change only prompts, editors, pagers, tracing, or network transport.
_GIT_INTERACTIVE_NAMES = frozenset(
    {
        "GIT_ASKPASS",
        "GIT_CURL_VERBOSE",
        "GIT_EDITOR",
        "GIT_FLUSH",
        "GIT_MERGE_VERBOSITY",
        "GIT_PAGER",
        "GIT_PROGRESS_DELAY",
        "GIT_REDACT_COOKIES",
        "GIT_SEQUENCE_EDITOR",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_SSH_VARIANT",
        "GIT_TERMINAL_PROMPT",
    }
)
_GIT_INTERACTIVE_PREFIXES = ("GIT_TRACE",)
# Git discovery variables make the walk below unreliable, so the client then
# lets the hook code decide whether the checkout captures.
_GIT_LOCATION_NAMES = ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR")


def clock() -> float:
    """Return the clock that every hook deadline uses.

    ``CLOCK_MONOTONIC`` counts from one system-wide starting point on macOS
    and Linux, so a deadline that the client sets means the same moment in
    the collector.
    """

    return time.clock_gettime(time.CLOCK_MONOTONIC)


def runtime_identity() -> str:
    """Return a fingerprint of the interpreter and the installed package files.

    An upgrade rewrites the package files, so a collector that started before
    it reports another identity and drains instead of serving old code.
    """

    package = os.path.dirname(os.path.abspath(__file__))
    digest = hashlib.sha256()
    digest.update(os.path.realpath(sys.executable).encode("utf-8", "surrogateescape"))
    digest.update(b"\0")
    digest.update(package.encode("utf-8", "surrogateescape"))
    for directory, names, files in os.walk(package):
        names[:] = sorted(name for name in names if name != "__pycache__")
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(directory, name)
            try:
                details = os.stat(path)
            except OSError:
                continue
            relative = os.path.relpath(path, package)
            digest.update(
                f"\0{relative}\0{details.st_size}\0{details.st_mtime_ns}".encode(
                    "utf-8", "surrogateescape"
                )
            )
    return digest.hexdigest()


def environment_key(environ: dict[str, str] | None = None) -> str:
    """Return a fingerprint of the variables that change what a hook does."""

    source = os.environ if environ is None else environ
    selected = sorted(
        (name, value)
        for name, value in source.items()
        if name in _ENVIRONMENT_NAMES
        or (
            name.startswith(_ENVIRONMENT_PREFIXES)
            and name not in _GIT_INTERACTIVE_NAMES
            and not name.startswith(_GIT_INTERACTIVE_PREFIXES)
        )
    )
    encoded = json.dumps(selected, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8", "surrogatepass")).hexdigest()


def default_state_dir(environ: dict[str, str] | None = None) -> str:
    """Mirror ``telemetry.default_state_dir`` without importing it."""

    source = os.environ if environ is None else environ
    override = source.get(STATE_DIR_ENV)
    if override:
        return os.path.realpath(os.path.expanduser(override))
    state_home = source.get("XDG_STATE_HOME")
    base = (
        os.path.expanduser(state_home)
        if state_home
        else os.path.join(os.path.expanduser("~"), ".local", "state")
    )
    return os.path.realpath(os.path.join(base, "harness-attribution", "telemetry"))


def worktree_root(path: str) -> str | None:
    """Return the nearest directory at or above ``path`` that holds ``.git``.

    Git discovers the checkout from the physical directory, so a symbolic
    link into a checkout is resolved before the walk.
    """

    current = os.path.realpath(path)
    while True:
        if os.path.lexists(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def _common_dir(root: str) -> str | None:
    marker = os.path.join(root, ".git")
    if os.path.isdir(marker):
        git_dir = marker
    else:
        try:
            with open(marker, encoding="utf-8") as file:
                line = file.readline(4096).strip()
        except (OSError, UnicodeError):
            return None
        if not line.startswith("gitdir:"):
            return None
        git_dir = os.path.join(root, line[len("gitdir:") :].strip())
    try:
        with open(os.path.join(git_dir, "commondir"), encoding="utf-8") as file:
            relative = file.readline(4096).strip()
    except FileNotFoundError:
        return git_dir
    except (OSError, UnicodeError):
        return None
    return os.path.join(git_dir, relative)


def may_capture(path: str, environ: dict[str, str] | None = None) -> bool:
    """Return false only when no hook code could capture in this checkout.

    The hook code ignores a checkout that has neither a repository install
    nor the hosted workflow. Both are files, so a few ``stat`` calls answer
    for the common case of machine hooks in an unrelated repository.
    """

    source = os.environ if environ is None else environ
    if any(source.get(name) for name in _GIT_LOCATION_NAMES):
        return True
    root = worktree_root(path)
    if root is None:
        return False
    if os.path.isfile(os.path.join(root, WORKFLOW_PATH)):
        return True
    common = _common_dir(root)
    if common is None:
        return True
    return os.path.exists(os.path.join(common, "attribution", "install.json"))


def _parse(arguments: list[str]) -> tuple[str, str | None, bool] | None:
    """Parse the private hook options, or return None for any other shape."""

    harness: str | None = None
    repo: str | None = None
    repository_hook = False
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in {"--harness", "--repo"}:
            if index + 1 >= len(arguments):
                return None
            value = arguments[index + 1]
            index += 2
        elif argument.startswith(("--harness=", "--repo=")):
            argument, _, value = argument.partition("=")
            index += 1
        elif argument == "--repository-hook":
            repository_hook = True
            index += 1
            continue
        else:
            return None
        if argument == "--harness":
            harness = value
        else:
            repo = value
    if harness not in HARNESSES:
        return None
    return harness, repo, repository_hook


def _collector_allowed(environ: dict[str, str]) -> bool:
    """Return whether this machine install runs the collector at all."""

    if environ.get(DISABLE_TELEMETRY_ENV) == "1":
        return False
    manifest = os.path.join(os.path.expanduser("~"), ".attribution", "user-install.json")
    try:
        with open(manifest, encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, UnicodeError, ValueError):
        return False
    return isinstance(value, dict) and value.get("telemetry_enabled") is True


_COLLECTOR_SERVICE = "harness-attribution-telemetry-v1"


def _read_collector(state_dir: str) -> tuple[str, int, str, str] | None:
    """Return the recorded host, port, token, and instance of the collector."""

    try:
        with open(os.path.join(state_dir, "collector-runtime.json"), encoding="utf-8") as file:
            runtime = json.load(file)
        with open(os.path.join(state_dir, "collector.token"), encoding="ascii") as file:
            token = file.read(4097).strip()
    except (OSError, UnicodeError, ValueError):
        return None
    if not isinstance(runtime, dict) or not token:
        return None
    host, port = runtime.get("host"), runtime.get("port")
    instance_id = runtime.get("instance_id")
    if (
        host not in {"127.0.0.1", "::1"}
        or not isinstance(port, int)
        or not 0 < port < 65536
        or not isinstance(instance_id, str)
        or not instance_id
    ):
        return None
    return host, port, token, instance_id


class _NotSent(Exception):
    """The collector never received the event, so another path may run it."""


def _remaining(release_at: float) -> float:
    remaining = release_at - clock()
    if remaining <= 0:
        raise TimeoutError("the hook's wait ended")
    return remaining


def _request(
    host: str,
    port: int,
    head: str,
    body: bytes,
    release_at: float,
) -> tuple[int, dict[str, object]]:
    """Send one request and read the answer until ``release_at``.

    Raise ``_NotSent`` only before the body left.
    """

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    connection = socket.socket(family, socket.SOCK_STREAM)
    try:
        try:
            connection.settimeout(min(CONNECT_SECONDS, _remaining(release_at)))
            connection.connect((host, port))
            connection.sendall(head.encode("latin-1") + body)
        except OSError as exc:
            raise _NotSent from exc
        chunks: list[bytes] = []
        size = 0
        while size <= _RESPONSE_LIMIT:
            connection.settimeout(_remaining(release_at))
            chunk = connection.recv(8192)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
    finally:
        connection.close()
    response = b"".join(chunks)
    head_bytes, _, content = response.partition(b"\r\n\r\n")
    status_line = head_bytes.split(b"\r\n", 1)[0].split()
    status = int(status_line[1]) if len(status_line) >= 2 and status_line[1].isdigit() else 0
    try:
        payload = json.loads(content) if content else {}
    except ValueError:
        payload = {}
    return status, payload if isinstance(payload, dict) else {}


def _identified(collector: tuple[str, int, str, str], release_at: float) -> bool:
    """Return whether the recorded collector itself answers on its port.

    A collector that crashed leaves its record behind, and any local process
    can bind the freed port. Only the unauthenticated identity request goes
    to the port before this check passes, so such a process never receives
    the token or a payload.
    """

    host, port, _token, instance_id = collector
    status, answer = _request(
        host,
        port,
        f"GET /identity HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n",
        b"",
        release_at,
    )
    return (
        status == 200
        and answer.get("service") == _COLLECTOR_SERVICE
        and answer.get("instance_id") == instance_id
    )


def _exchange(
    collector: tuple[str, int, str, str],
    path: str,
    body: bytes,
    release_at: float,
) -> tuple[int, dict[str, object]]:
    """Send one authenticated request to a collector that was identified."""

    host, port, token, _instance_id = collector
    head = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Authorization: Bearer {token}\r\n"
        "Content-Type: application/octet-stream\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    )
    return _request(host, port, head, body, release_at)


def _start_collector(state_dir: str) -> bool:
    try:
        from .telemetry import ensure_collector

        ensure_collector(state_dir, startup_timeout=2.0)
    except Exception:
        return False
    return True


def _stop_old_collector(collector: tuple[str, int, str, str]) -> None:
    try:
        _exchange(collector, "/shutdown", b"", clock() + RECEIPT_WAIT_SECONDS)
    except Exception:
        pass


def deliver(
    state_dir: str,
    metadata: dict[str, object],
    payload: bytes,
    *,
    start: bool = True,
    started: float | None = None,
) -> bool:
    """Send one event; return True when the collector took responsibility.

    False means that the collector never took the event, so the caller runs
    it in-process. True also covers a request that the collector received
    but did not answer in time, because running it again could record the
    event twice.

    A ``PreToolUse`` must end by ``started`` plus its wait, measured from the
    start of the hook, because the tool runs as soon as the client returns.
    The collector gets that deadline, so it never takes a baseline later.
    """

    wait = metadata.get("wait") is True
    started = clock() if started is None else started
    for attempt in range(2):
        now = clock()
        release_at = started + PRE_TOOL_WAIT_SECONDS if wait else now + RECEIPT_WAIT_SECONDS
        deadline = release_at - DEADLINE_MARGIN_SECONDS
        if deadline <= now:
            # No snapshot could end before the tool is released, and a run
            # in this process would take longer still.
            return True
        collector = _read_collector(state_dir)
        try:
            identified = collector is not None and _identified(
                collector, min(release_at, now + RECEIPT_WAIT_SECONDS)
            )
        except Exception:
            identified = False
        if collector is None or not identified:
            # No collector answers as the recorded one. Starting one replaces
            # the record; nothing is sent to whatever holds the old port.
            if attempt or not start or not _start_collector(state_dir):
                return False
            continue
        body = (
            json.dumps({**metadata, "deadline": deadline}, separators=(",", ":")).encode("ascii")
            + b"\n"
            + payload
        )
        try:
            status, response = _exchange(collector, HOOK_ENDPOINT_PATH, body, release_at)
        except _NotSent:
            if attempt or not start or not _start_collector(state_dir):
                return False
            continue
        except Exception:
            return True
        if status in {200, 202}:
            return True
        if status == 404:
            # A collector from before this client has no hook endpoint. Stop
            # it so that the next event starts a current one.
            _stop_old_collector(collector)
        return False
    return False


def _run_in_process(
    harness: str, payload: dict[str, object], repo: str, repository_hook: bool
) -> None:
    from .hook_service import run_native_hook

    run_native_hook(harness, payload, repo, repository_hook)


def main(argv: list[str] | None = None) -> int:
    started = clock()
    arguments = list(sys.argv[1:] if argv is None else argv)
    parsed = _parse(arguments[1:]) if arguments[:1] == ["_hook"] else None
    if parsed is None:
        # The launcher also serves ordinary commands such as ``--version``.
        from .cli import main as cli_main

        return cli_main(arguments)
    try:
        harness, selected_repo, repository_hook = parsed
        source = sys.stdin.buffer if hasattr(sys.stdin, "buffer") else sys.stdin
        raw = source.read(HOOK_INPUT_LIMIT + 1)
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        environ = dict(os.environ)
        if len(raw) > HOOK_INPUT_LIMIT or (
            environ.get("ATTRIBUTION_WRAPPED_CAPTURE") == "1"
            or environ.get("ATTRIBUTION_WRAPPER_SESSION_ID")
        ):
            return 0
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return 0
        cwd = payload.get("cwd")
        repo = os.path.abspath(
            selected_repo
            if selected_repo
            else cwd
            if isinstance(cwd, str) and cwd.strip()
            else os.getcwd()
        )
        if not may_capture(repo, environ):
            return 0
        if _collector_allowed(environ):
            metadata: dict[str, object] = {
                "harness": harness,
                "repo": repo,
                "repository_hook": repository_hook,
                "identity": runtime_identity(),
                "environment": environment_key(environ),
                "wait": payload.get("hook_event_name") == "PreToolUse",
            }
            if deliver(default_state_dir(environ), metadata, raw, started=started):
                return 0
        _run_in_process(harness, payload, repo, repository_hook)
    except Exception:
        # Hook observers must never block an edit, and they print nothing.
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
