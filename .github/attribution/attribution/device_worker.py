"""Claim import jobs for a connected computer and run them.

The collector runs the poll thread. Each job runs in a child process, so the
hook path of the collector never waits on Git reads or transcript parsing.
The child reads local history, uploads the allowlisted metadata, and reports
fixed error codes. No local path, session text, or token leaves this module.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import threading
import time
from typing import Any, Callable, Mapping

from .device import (
    DeviceClient,
    DeviceError,
    DeviceHTTPError,
    DeviceRevoked,
    clear_credential,
    load_credential,
)

POLL_SECONDS = 15
CHILD_TIMEOUT_SECONDS = 60 * 60
MAX_SCAN_FILES = 2000
MAX_SCAN_LINES = 20
_RETRY_ATTEMPTS = 3
_RETRY_SECONDS = 2.0
_SCAN_REPORT_SECONDS = 2.0
_MAX_JOB_BYTES = 64 * 1024
_MESSAGES = {
    "checkout_not_found": "No checkout of this repository was found on this computer.",
    "checkout_mismatch": "The folder you named belongs to another repository.",
    "access_denied": "Repository access could not be verified.",
    "authentication_required": "Sign in to Joyride again to continue the import.",
    "rate_limited": "GitHub's rate limit was reached. Try again later.",
    "service_error": "The Joyride service returned an error.",
    "import_failed": "The import stopped with an error on this computer.",
}


class CheckoutMismatch(ValueError):
    """The named folder exists but belongs to another repository."""


class _Cancelled(Exception):
    """The service asked the device to stop this job."""


def _registered_paths(state_dir: str | os.PathLike[str] | None) -> list[str]:
    """Return the repository paths that local capture registered, read-only."""

    from .telemetry import _state_dir_path

    database = _state_dir_path(state_dir) / "telemetry.sqlite3"
    if not database.is_file():
        return []
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                "SELECT DISTINCT repository_path FROM registered_sessions "
                "WHERE repository_path IS NOT NULL"
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error:
        return []
    return [row[0] for row in rows if isinstance(row[0], str)]


def _transcript_cwds(path: Path) -> list[str]:
    """Return the working directories named in the first lines of one transcript."""

    from .usage_fallback import MAX_LINE_BYTES

    found: list[str] = []
    try:
        with path.open("rb") as source:
            for _ in range(MAX_SCAN_LINES):
                raw = source.readline(MAX_LINE_BYTES + 1)
                if not raw:
                    break
                if len(raw) > MAX_LINE_BYTES:
                    continue
                try:
                    record = json.loads(raw)
                except (UnicodeDecodeError, ValueError):
                    continue
                if not isinstance(record, dict):
                    continue
                payload = record.get("payload")
                for value in (record.get("cwd"), payload.get("cwd") if isinstance(payload, dict) else None):
                    if isinstance(value, str) and value.startswith("/") and "\x00" not in value:
                        found.append(value)
    except OSError:
        return found
    return found


def _candidates(state_dir: str | os.PathLike[str] | None) -> Counter[str]:
    """Count the local directories that provider records and capture name."""

    from .history_import import _files, _roots

    hits: Counter[str] = Counter()
    for value in _registered_paths(state_dir):
        hits[value] += 1
    claude_roots, codex_roots = _roots()
    scanned = 0
    for path in _files([*claude_roots, *codex_roots], Counter()):
        scanned += 1
        if scanned > MAX_SCAN_FILES:
            break
        for value in _transcript_cwds(path):
            hits[value] += 1
    return hits


def _matching_root(directory: Path, full_name: str) -> Path | None:
    """Return the checkout root when ``directory`` belongs to the repository."""

    from .history_import import repository_roots

    try:
        return repository_roots(directory, full_name)[0]
    except ValueError:
        return None


def locate_checkout(
    full_name: str, hint: str | None = None, *, state_dir: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Find a local checkout of ``full_name`` without reading Git objects.

    A hint that names an existing folder of another repository raises
    ``CheckoutMismatch``. Otherwise the candidates from local records decide,
    and the root with the most hits wins. ``None`` means nothing matched.
    """

    from .history_import import _git

    if isinstance(hint, str) and hint.startswith("/") and "\x00" not in hint:
        directory = Path(hint)
        if directory.is_dir():
            root = _matching_root(directory, full_name)
            if root is None:
                raise CheckoutMismatch(_MESSAGES["checkout_mismatch"])
            return root
    hits_by_top: Counter[str] = Counter()
    for value, count in _candidates(state_dir).items():
        directory = Path(value)
        if not directory.is_dir():
            continue
        top = _git(directory, "rev-parse", "--show-toplevel")
        if top:
            hits_by_top[top] += count
    hits_by_root: Counter[Path] = Counter()
    for top, count in hits_by_top.items():
        root = _matching_root(Path(top), full_name)
        if root is not None:
            hits_by_root[root] += count
    if not hits_by_root:
        return None
    return max(sorted(hits_by_root), key=lambda root: hits_by_root[root])


def _since_date(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


def _retrying(call: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Repeat a device request after a rate limit, a temporary service error, or no answer."""

    from urllib.error import HTTPError

    last = _RETRY_ATTEMPTS - 1
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return call()
        except HTTPError as exc:
            if attempt == last:
                raise DeviceHTTPError(exc.code, "rate_limited" if exc.code == 429 else None,
                                      _MESSAGES["service_error"]) from None
        except (DeviceRevoked, DeviceHTTPError):
            raise
        except DeviceError:
            if attempt == last:
                raise
        time.sleep(_RETRY_SECONDS * (attempt + 1))
    raise DeviceError(_MESSAGES["service_error"])


def run_job(
    job: Mapping[str, Any], credential: Mapping[str, Any], *,
    client: DeviceClient | None = None, state_dir: str | os.PathLike[str] | None = None,
) -> str:
    """Run one claimed job and report its result. Return the final status."""

    from .history_import import discover, upload_with

    client = client if client is not None else DeviceClient(credential)
    job_id = int(job["id"])
    full_name = str(job["repository"]["full_name"])
    # The attempt number tells the service which claim reports. A child that
    # outlived its collector stops as soon as a later claim owns the job.
    attempt = job.get("attempt") if type(job.get("attempt")) is int and job.get("attempt") > 0 else None
    progress: dict[str, Any] = {}

    def report(phase: str) -> None:
        answer = _retrying(lambda: client.progress(job_id, phase, progress, attempt))
        if answer.get("cancel_requested") is True:
            raise _Cancelled

    def finish(status: str, code: str | None = None) -> str:
        try:
            _retrying(lambda: client.finish(
                job_id, status, error_code=code, error=_MESSAGES[code] if code else None,
                progress=progress, attempt=attempt,
            ))
        except DeviceError:
            pass
        return status

    def fail(code: str) -> str:
        return finish("failed", code)

    try:
        report("locating")
        try:
            checkout = locate_checkout(full_name, job.get("checkout_hint"), state_dir=state_dir)
        except CheckoutMismatch:
            return fail("checkout_mismatch")
        if checkout is None:
            return fail("checkout_not_found")
        report("pull_requests")
        while True:
            answer = _retrying(lambda: client.advance_pull_requests(job_id))
            record = answer.get("import")
            if isinstance(record, dict):
                checked = record.get("processed_count")
                progress["pull_requests_checked"] = checked if type(checked) is int and checked >= 0 else 0
            if answer.get("cancel_requested") is True:
                raise _Cancelled
            report("pull_requests")
            if not isinstance(record, dict) or record.get("status") == "complete":
                break
        report("sessions")
        last_scan_report = 0.0

        def on_scan(step: str, done: int, total: int) -> None:
            # The page draws a bar from these. A report every two seconds
            # keeps the request count small on a large history.
            nonlocal last_scan_report
            progress.update({"scan_step": step, "files_scanned": done, "files_total": total})
            now = time.monotonic()
            if step == "catalog" or done >= total or now - last_scan_report >= _SCAN_REPORT_SECONDS:
                last_scan_report = now
                report("sessions")

        envelope = discover(checkout, full_name, since=_since_date(job.get("since_at")), progress=on_scan)
        for key in ("scan_step", "files_scanned", "files_total"):
            progress.pop(key, None)
        envelope["sessions"].sort(key=lambda session: (
            session["last_activity_at"], session["started_at"], session["native_session_id"],
        ), reverse=True)
        progress.update({"sessions_found": len(envelope["sessions"]), "sessions_uploaded": 0,
                         "sessions_created": 0, "sessions_updated": 0})
        report("sessions")
        if envelope["sessions"]:
            def on_request(answer: dict[str, Any], sent: int, total: int) -> None:
                for name in ("accepted", "created", "updated"):
                    number = answer.get(name, 0)
                    if type(number) is int and number >= 0:
                        key = "sessions_uploaded" if name == "accepted" else f"sessions_{name}"
                        progress[key] = min(progress["sessions_found"], progress[key] + number)
                if answer.get("cancel_requested") is True:
                    raise _Cancelled
                if sent < total:
                    report("sessions")

            upload_with(envelope, lambda data: _retrying(lambda: client.upload_sessions(job_id, data)),
                        on_request)
        progress["sessions_uploaded"] = progress["sessions_found"]
        return finish("complete")
    except _Cancelled:
        return finish("cancelled")
    except DeviceRevoked:
        clear_credential(state_dir)
        raise
    except DeviceHTTPError as exc:
        if exc.code == "stale_attempt":
            return "stale"
        if exc.status == 403:
            return fail("access_denied")
        if exc.status == 401:
            return fail("authentication_required")
        if exc.code == "rate_limited":
            return fail("rate_limited")
        return fail("service_error")
    except DeviceError:
        return fail("service_error")
    except Exception:
        # A failure of any other kind ends the job now, so the page does not
        # wait for five expired leases to learn that the import stopped.
        return fail("import_failed")


class DevicePoller:
    """Poll the hosted service from the collector and run each claimed job."""

    def __init__(self, state_dir: str | os.PathLike[str]) -> None:
        self.state_dir = Path(state_dir)
        self.poll_seconds = float(POLL_SECONDS)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._child: subprocess.Popen[bytes] | None = None
        self._child_started = 0.0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="joyride-device-poller", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        child = self._child
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.cycle()
            self._stop.wait(self.poll_seconds)

    def cycle(self) -> None:
        """Run one poll: keep the device online, and start a claimed job."""

        credential = load_credential(self.state_dir)
        if credential is None:
            return
        child = self._child
        busy = child is not None and child.poll() is None
        if busy and time.monotonic() - self._child_started > CHILD_TIMEOUT_SECONDS:
            child.kill()
            child.wait(timeout=2)
            busy = False
        if child is not None and not busy:
            self._child = None
        try:
            answer = DeviceClient(credential).poll(busy)
        except DeviceRevoked:
            clear_credential(self.state_dir)
            return
        except (DeviceError, OSError):
            return
        wait = answer.get("poll_seconds")
        if type(wait) is int and 5 <= wait <= 300:
            self.poll_seconds = float(wait)
        job = answer.get("job")
        if busy or not isinstance(job, dict) or self._stop.is_set():
            return
        self._start_child(job)

    def _start_child(self, job: dict[str, Any]) -> None:
        from .runtime import current_runtime

        invocation = current_runtime().cli(["_device-job"])
        try:
            child = subprocess.Popen(
                list(invocation.argv), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, env=invocation.environment(), close_fds=True,
            )
        except OSError:
            return
        self._child = child
        self._child_started = time.monotonic()
        try:
            assert child.stdin is not None
            child.stdin.write(json.dumps(job, separators=(",", ":")).encode("utf-8"))
            child.stdin.close()
        except (OSError, ValueError):
            pass


def run_job_from_stdin(stream: Any) -> int:
    """Read one job from a stream and run it with the stored credential."""

    credential = load_credential()
    if credential is None:
        raise ValueError("This computer is not connected to Joyride.")
    raw = stream.read(_MAX_JOB_BYTES + 1)
    if len(raw) > _MAX_JOB_BYTES:
        raise ValueError("The job is too large.")
    job = json.loads(raw)
    if (not isinstance(job, dict) or type(job.get("id")) is not int
            or not isinstance(job.get("repository"), dict)
            or not isinstance(job["repository"].get("full_name"), str)):
        raise ValueError("The job is invalid.")
    return 0 if run_job(job, credential) == "complete" else 1
