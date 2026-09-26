"""Read local Codex history through the official app-server interface.

Only the small session allowlist returned by ``read_sessions`` may be uploaded.
The native paths stay local for usage enrichment by the history importer.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import select
import shutil
import subprocess
import time
from typing import Any

from .history_import import (
    MAX_FILE_BYTES, MAX_FILES, _BRANCH, _FAILED_TOOL_OUTPUT, _ID, _MODEL,
    _accepted_commit, _commit_summaries, _git, _in_repo,
)

_SOURCE_KINDS = [
    "cli", "vscode", "exec", "appServer", "subAgent", "subAgentReview",
    "subAgentCompact", "subAgentThreadSpawn", "subAgentOther", "unknown",
]
_TOOL_TYPES = {
    "commandExecution", "fileChange", "mcpToolCall", "dynamicToolCall",
    "collabAgentToolCall", "webSearch", "imageView", "imageGeneration",
}
# One real thread answered 44 MiB with its turns. A larger one is skipped on
# its own; its native file still supplies the session.
_MAX_MESSAGE_BYTES = 256 * 1024 * 1024
_MAX_TOTAL_BYTES = 1024 * 1024 * 1024
# One request may take this long. A real computer holds hundreds of threads
# for one repository, so the whole conversation gets a larger budget.
_TIMEOUT_SECONDS = 45
_TOTAL_SECONDS = 900


class _Unavailable(Exception):
    """The official interface cannot safely supply complete local history."""


class _Oversized(_Unavailable):
    """One answer passed the message limit and was discarded. The server still runs."""


class _AppServer:
    def __init__(self, home: Path):
        binary = shutil.which("codex")
        if binary is None:
            raise _Unavailable
        try:
            self.process = subprocess.Popen(
                [binary, "app-server", "--stdio"], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env={**os.environ, "CODEX_HOME": str(home)},
            )
        except OSError as exc:
            raise _Unavailable from exc
        self.buffer = bytearray()
        self.total = 0
        self.next_id = 0
        self.deadline = time.monotonic() + _TOTAL_SECONDS

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.stdout is not None:
            self.process.stdout.close()

    def _send(self, message: dict[str, Any]) -> None:
        try:
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps(message, separators=(",", ":")).encode() + b"\n")
            self.process.stdin.flush()
        except (OSError, BrokenPipeError) as exc:
            raise _Unavailable from exc

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.next_id += 1
        request_id = self.next_id
        self._send({"id": request_id, "method": method, "params": params})
        deadline = min(self.deadline, time.monotonic() + _TIMEOUT_SECONDS)
        while True:
            newline = self.buffer.find(b"\n")
            if newline >= 0:
                if newline > _MAX_MESSAGE_BYTES:
                    del self.buffer[:newline + 1]
                    raise _Oversized
                raw = bytes(self.buffer[:newline])
                del self.buffer[:newline + 1]
                try:
                    response = json.loads(raw)
                except (ValueError, RecursionError) as exc:
                    raise _Unavailable from exc
                if not isinstance(response, dict):
                    raise _Unavailable
                if response.get("id") != request_id:
                    # Notifications have no id. A response to another id is
                    # never expected with our sequential requests.
                    if "id" in response:
                        raise _Unavailable
                    continue
                if "error" in response or not isinstance(response.get("result"), dict):
                    raise _Unavailable
                return response["result"]
            if len(self.buffer) > _MAX_MESSAGE_BYTES:
                self._discard_line(deadline)
                raise _Oversized
            self._read_more(deadline)

    def _read_more(self, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _Unavailable
        assert self.process.stdout is not None
        try:
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            chunk = os.read(self.process.stdout.fileno(), 1 << 20) if ready else b""
        except OSError as exc:
            raise _Unavailable from exc
        if not chunk:
            raise _Unavailable
        self.total += len(chunk)
        if self.total > _MAX_TOTAL_BYTES:
            raise _Unavailable
        self.buffer.extend(chunk)

    def _discard_line(self, deadline: float) -> None:
        """Drop the rest of an oversized answer, so the next request reads a clean line."""
        while True:
            newline = self.buffer.find(b"\n")
            if newline >= 0:
                del self.buffer[:newline + 1]
                return
            del self.buffer[:]
            self._read_more(deadline)

    def initialize(self) -> None:
        self.request("initialize", {
            "clientInfo": {"name": "joyride-history-import", "version": "1.0"},
            "capabilities": {},
        })
        self._send({"method": "initialized"})


def _stamp(seconds: Any) -> str | None:
    if type(seconds) is not int:
        return None
    try:
        return datetime.fromtimestamp(seconds, timezone.utc).isoformat(
            timespec="milliseconds").replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError):
        return None


def _safe_path(value: Any, locations: list[Path], skipped: Counter[str]) -> Path | None:
    if not isinstance(value, str) or not value.startswith("/") or "\x00" in value:
        return None
    path = Path(value)
    try:
        safe_root = False
        for location in locations:
            if location.is_symlink() or location.parent.is_symlink():
                continue
            for base in (location.absolute(), location.resolve()):
                try:
                    relative = path.relative_to(base)
                except ValueError:
                    continue
                current = base
                symlink = False
                for part in relative.parts:
                    current /= part
                    if current.is_symlink():
                        symlink = True
                        break
                if symlink:
                    continue
                resolved = path.resolve(strict=True)
                canonical_root = location.resolve()
                if resolved == canonical_root or canonical_root in resolved.parents:
                    safe_root = True
                    break
            if safe_root:
                break
        if not safe_root or not resolved.is_file():
            skipped["unsafe_file"] += 1
            return None
        if resolved.stat().st_size > MAX_FILE_BYTES:
            skipped["oversized_file"] += 1
            return None
    except (OSError, RuntimeError):
        skipped["unreadable_file"] += 1
        return None
    return resolved


def _created_commits(turns: list[Any], repo: Path, roots: list[Path]) -> list[dict[str, str]]:
    candidates: set[str] = set()
    for turn in turns:
        if not isinstance(turn, dict) or not isinstance(turn.get("items"), list):
            continue
        for item in turn["items"]:
            if not isinstance(item, dict) or item.get("type") != "commandExecution":
                continue
            command, cwd, output = item.get("command"), item.get("cwd"), item.get("aggregatedOutput")
            if (item.get("status") != "completed" or type(item.get("exitCode")) is not int or
                    item["exitCode"] != 0 or not isinstance(output, str) or len(output) > 65536 or
                    _FAILED_TOOL_OUTPUT.search(output)):
                continue
            flags = _accepted_commit(command, cwd, roots)
            if flags is None:
                continue
            for short in _commit_summaries(output, flags):
                candidates.add(short)
    if candidates:
        from .history_evidence import _promisor_clone
        if _promisor_clone(repo, time.monotonic() + 3):
            return []
    created: set[str] = set()
    for short in candidates:
        full = _git(repo, "rev-parse", "--verify", f"{short}^{{commit}}")
        if re.fullmatch(r"[0-9a-f]{40}", full):
            created.add(full)
    return [{"sha": sha, "evidence": "created_commit"} for sha in sorted(created)]


def _session(thread: dict[str, Any], repo: Path, roots: list[Path],
             locations: list[Path], skipped: Counter[str]) -> tuple[dict[str, Any], list[Path]] | None:
    native = thread.get("id")
    if not isinstance(native, str) or not _ID.fullmatch(native):
        skipped["invalid_session_id"] += 1
        return None
    if not _in_repo(thread.get("cwd"), roots):
        skipped["outside_repository"] += 1
        return None
    if thread.get("ephemeral") is True:
        return None
    path = _safe_path(thread.get("path"), locations, skipped)
    if path is None:
        return None
    first, last = _stamp(thread.get("createdAt")), _stamp(thread.get("updatedAt"))
    if first is None:
        skipped["missing_timestamp"] += 1
        return None
    turns = thread.get("turns")
    if not isinstance(turns, list):
        raise _Unavailable
    prompt_ids: set[str] = set()
    tool_ids: set[str] = set()
    partial = False
    for turn in turns:
        if not isinstance(turn, dict) or not isinstance(turn.get("items"), list):
            partial = True
            continue
        if turn.get("itemsView", "full") != "full":
            partial = True
        for item in turn["items"]:
            if not isinstance(item, dict):
                partial = True
                continue
            item_cwd = item.get("cwd")
            if isinstance(item_cwd, str) and item_cwd.startswith("/") and not _in_repo(item_cwd, roots):
                skipped["mixed_repository"] += 1
                return None
            ident = item.get("id")
            if not isinstance(ident, str) or not _ID.fullmatch(ident):
                continue
            if item.get("type") == "userMessage":
                prompt_ids.add(ident)
            elif item.get("type") in _TOOL_TYPES:
                tool_ids.add(ident)
    branch = (thread.get("gitInfo") or {}).get("branch") if isinstance(thread.get("gitInfo"), dict) else None
    if not isinstance(branch, str) or not _BRANCH.fullmatch(branch) or ".." in branch:
        branch = None
    model = thread.get("model")
    if not isinstance(model, str) or not _MODEL.fullmatch(model):
        model = None
    result: dict[str, Any] = {
        "provider": "codex", "native_session_id": native,
        "started_at": first, "last_activity_at": last or first,
        "branch": branch, "model": model, "usage": [],
        "prompt_count": len(prompt_ids), "tool_call_count": len(tool_ids),
        "completeness": "partial" if partial else "complete",
        "created_commits": _created_commits(turns, repo, roots),
    }
    parent = thread.get("parentThreadId") or thread.get("forkedFromId")
    if isinstance(parent, str) and _ID.fullmatch(parent) and parent != native:
        result["parent_native_session_id"] = parent
    return result, [path]


def read_sessions(repo: Path, roots: list[Path], locations: list[Path],
                  skipped: Counter[str]) -> list[tuple[dict[str, Any], list[Path]]] | None:
    """Return official Codex metadata, or None for a safe native-log fallback."""

    if not locations:
        return []
    repo = repo.resolve()
    roots = [root.resolve() for root in roots]
    selected_skipped: Counter[str] = Counter()
    homes = {location.expanduser().resolve().parent for location in locations}
    if len(homes) != 1:
        return None
    server: _AppServer | None = None
    try:
        server = _AppServer(homes.pop())
        server.initialize()
        listed: dict[str, dict[str, Any]] = {}
        for archived in (False, True):
            cursor: str | None = None
            seen_cursors: set[str] = set()
            while True:
                params: dict[str, Any] = {
                    "archived": archived, "sourceKinds": _SOURCE_KINDS,
                    "limit": 100, "useStateDbOnly": False,
                }
                if cursor is not None:
                    params["cursor"] = cursor
                page = server.request("thread/list", params)
                rows, next_cursor = page.get("data"), page.get("nextCursor")
                if not isinstance(rows, list) or next_cursor is not None and not isinstance(next_cursor, str):
                    raise _Unavailable
                for row in rows:
                    if not isinstance(row, dict):
                        raise _Unavailable
                    ident = row.get("id")
                    if isinstance(ident, str):
                        listed[ident] = row
                if len(listed) > MAX_FILES:
                    raise _Unavailable
                if not next_cursor:
                    break
                if next_cursor in seen_cursors:
                    raise _Unavailable
                seen_cursors.add(next_cursor)
                cursor = next_cursor
        results: list[tuple[dict[str, Any], list[Path]]] = []
        for native, summary in listed.items():
            if not _in_repo(summary.get("cwd"), roots):
                selected_skipped["outside_repository"] += 1
                continue
            if summary.get("ephemeral") is True:
                continue
            try:
                response = server.request("thread/read", {"threadId": native, "includeTurns": True})
            except _Oversized:
                # The native file of this one thread still supplies the session.
                selected_skipped["oversized_session"] += 1
                continue
            thread = response.get("thread")
            if not isinstance(thread, dict) or thread.get("id") != native:
                raise _Unavailable
            session = _session(thread, repo, roots, locations, selected_skipped)
            if session is not None:
                results.append(session)
        skipped.update(selected_skipped)
        return results
    except _Unavailable:
        return None
    finally:
        if server is not None:
            server.close()
