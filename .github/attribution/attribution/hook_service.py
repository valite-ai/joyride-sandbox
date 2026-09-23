"""Run native hook events inside the long-lived local collector.

The collector keeps one worker thread for each Git worktree that sends
events. A worker takes the events of its worktree in arrival order, so a
``PostToolUse`` is recorded before the next ``PreToolUse`` takes its
snapshot. Worktrees run in parallel, as separate hook processes did.

Threads are safe here because the hook code keeps no shared mutable state:
it opens a new SQLite connection for each call, takes ``flock`` locks on
file descriptors that it opens itself (``flock`` locks from two descriptors
conflict even inside one process), passes working directories to
subprocesses instead of calling ``chdir``, and never changes the process
environment. Separate worker processes would each pay the interpreter start
and the imports that this service exists to avoid, and forking a threaded
process is unsafe on macOS.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import queue
import threading
from typing import Any

from .hook_client import HARNESSES, HOOK_INPUT_LIMIT, clock, worktree_root


# The metadata line is small; the payload keeps the CLI's own limit.
MAX_HOOK_REQUEST_BYTES = HOOK_INPUT_LIMIT + 64 * 1024
# The client waits 9 seconds for a PreToolUse and 2 seconds for a receipt.
# Answering first keeps a request from ending as a timeout that the client
# cannot interpret.
PRE_TOOL_ANSWER_SECONDS = 8.5
RECEIPT_ANSWER_SECONDS = 1.5
IDLE_WORKER_SECONDS = 60.0


def run_native_hook(
    harness: str,
    payload: Mapping[str, Any],
    repo: str | Path,
    repository_hook: bool,
    *,
    deadline: float | None = None,
) -> None:
    """Run one native hook event exactly as the ``_hook`` command does.

    ``deadline`` is a ``hook_client.clock`` value. A ``PreToolUse`` whose
    snapshot ends after it records nothing, because the tool may already run.
    """

    from .automation import handle_hook
    from .store import git_path_scope

    # The collector outlives worktree moves, so each event resolves the Git
    # directories of its checkout again.
    with git_path_scope():
        if repository_hook:
            from .user_install import user_hook_covers

            if user_hook_covers(harness, payload.get("hook_event_name")):
                return
        handle_hook(Path(repo), payload, harness, deadline=deadline)


@dataclass
class _Job:
    run: Callable[[], None]
    deadline: float | None = None
    done: threading.Event = field(default_factory=threading.Event)


class HookDispatcher:
    """Run submitted jobs in arrival order for each key, one thread per key."""

    def __init__(self, *, idle_seconds: float = IDLE_WORKER_SECONDS) -> None:
        self._idle_seconds = idle_seconds
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._queues: dict[str, queue.SimpleQueue[_Job]] = {}
        self._unfinished: dict[str, int] = {}
        self._pending = 0
        self._closing = False

    def submit(
        self, key: str, run: Callable[[], None], *, deadline: float | None = None
    ) -> _Job | None:
        """Queue one job, or return None when the job may run elsewhere.

        A closed dispatcher still queues a job for a key that has unfinished
        jobs, because running it elsewhere would overtake them. A job that has
        not started by its ``hook_client.clock`` deadline is skipped.
        """

        job = _Job(run, deadline)
        with self._lock:
            if self._closing and not self._unfinished.get(key):
                return None
            jobs = self._queues.get(key)
            if jobs is None:
                jobs = queue.SimpleQueue()
                self._queues[key] = jobs
                threading.Thread(
                    target=self._work,
                    args=(key, jobs),
                    name=f"joyride-hook-{len(self._queues)}",
                    daemon=True,
                ).start()
            self._pending += 1
            self._unfinished[key] = self._unfinished.get(key, 0) + 1
            jobs.put(job)
        return job

    def _work(self, key: str, jobs: queue.SimpleQueue[_Job]) -> None:
        while True:
            try:
                job = jobs.get(timeout=self._idle_seconds)
            except queue.Empty:
                with self._lock:
                    # ``submit`` puts under the same lock, so an empty queue
                    # here cannot hide a job that arrived after the timeout.
                    if jobs.empty():
                        del self._queues[key]
                        return
                continue
            try:
                if job.deadline is None or clock() <= job.deadline:
                    job.run()
            except Exception:
                # A hook never raises into its caller; one failed event must
                # not stop the worktree's later events either.
                pass
            finally:
                job.done.set()
                with self._lock:
                    self._pending -= 1
                    self._unfinished[key] -= 1
                    if not self._unfinished[key]:
                        del self._unfinished[key]
                    self._idle.notify_all()

    def close(self) -> None:
        """Refuse later jobs for every key without unfinished jobs."""

        with self._lock:
            self._closing = True

    def drain(self, timeout: float | None = None) -> bool:
        """Refuse new jobs and wait until every queued job finished."""

        self.close()
        return self.wait_idle(timeout)

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait until every queued job finished; ``None`` waits without limit."""

        with self._lock:
            return self._idle.wait_for(lambda: self._pending == 0, timeout=timeout)


class HookEndpoint:
    """Validate ``/v1/hook`` requests and hand their events to the dispatcher."""

    def __init__(
        self,
        identity: str,
        environment: str,
        *,
        dispatcher: HookDispatcher | None = None,
        run: Callable[..., None] = run_native_hook,
    ) -> None:
        self.identity = identity
        self.environment = environment
        self.dispatcher = dispatcher or HookDispatcher()
        self._run = run
        self._retiring = False

    def handle(self, body: bytes) -> tuple[int, dict[str, Any], bool]:
        """Return the HTTP status, the JSON answer, and whether to stop."""

        metadata_line, separator, raw = body.partition(b"\n")
        if not separator or len(raw) > HOOK_INPUT_LIMIT:
            return 400, {"error": "invalid_hook_request"}, False
        try:
            metadata = json.loads(metadata_line)
            payload = json.loads(raw)
        except (UnicodeDecodeError, ValueError, RecursionError):
            return 400, {"error": "invalid_hook_request"}, False
        if not isinstance(metadata, dict) or not isinstance(payload, dict):
            return 400, {"error": "invalid_hook_request"}, False
        harness = metadata.get("harness")
        repo = metadata.get("repo")
        repository_hook = metadata.get("repository_hook")
        wait = metadata.get("wait")
        requested = metadata.get("deadline")
        if (
            harness not in HARNESSES
            or not isinstance(repo, str)
            or not os.path.isabs(repo)
            or not isinstance(repository_hook, bool)
            or not isinstance(wait, bool)
            or not isinstance(requested, (int, float))
            or isinstance(requested, bool)
            or not math.isfinite(requested)
        ):
            return 400, {"error": "invalid_hook_request"}, False
        key = worktree_root(repo) or repo
        if metadata.get("environment") != self.environment:
            # The hook environment of a session does not change, so a session
            # with another environment never had an event in this collector.
            # Its client runs the event itself at once without overtaking any
            # event of its own session.
            return 409, {"error": "environment_mismatch"}, False
        # The client's deadline counts from the start of its hook, so the
        # time before this line already counts against it.
        budget = PRE_TOOL_ANSWER_SECONDS if wait else RECEIPT_ANSWER_SECONDS
        answer_by = min(float(requested), clock() + budget)
        if metadata.get("identity") != self.identity:
            # This collector runs older or newer code than the client.
            self._retiring = True
        if self._retiring:
            # The client runs a refused event with its own code, so every event
            # that this collector took must finish first. When that takes
            # longer than the client waits, the event joins the queue behind
            # them instead, and a later request stops the collector.
            if self.dispatcher.wait_idle(max(0.0, answer_by - clock())):
                self.dispatcher.close()
                return 409, {"error": "stale_collector"}, True
        # The client is released soon after ``answer_by``. A snapshot that
        # has not finished by then could already hold the tool's edits.
        deadline = answer_by if wait else None
        if wait and answer_by <= clock():
            return 202, {"status": "expired"}, False
        job = self.dispatcher.submit(
            key,
            lambda: self._run(harness, payload, repo, repository_hook, deadline=deadline),
            deadline=deadline,
        )
        if job is None:
            # The collector is stopping and holds no unfinished event of this
            # worktree, so the client's own run overtakes nothing.
            return 409, {"error": "collector_stopping"}, False
        if not wait:
            return 202, {"status": "queued"}, False
        if job.done.wait(max(0.0, answer_by - clock())):
            return 200, {"status": "done"}, False
        return 202, {"status": "expired"}, False
