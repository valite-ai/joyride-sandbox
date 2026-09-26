"""Read local Claude sessions through the Agent SDK's public history API."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any


def _time(milliseconds: Any) -> str | None:
    if type(milliseconds) is not int or milliseconds < 0:
        return None
    try:
        return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError):
        return None


def _native_paths(session_id: str, locations: list[Path], skipped: Counter[str]) -> list[Path]:
    """Find exact-ID files in supplied project buckets without reading other transcripts."""
    from .history_import import MAX_DIRS, MAX_FILE_BYTES

    found: list[Path] = []
    directories = 0
    for root in locations:
        if root.is_symlink() or not root.is_dir():
            continue
        try:
            buckets = [root, *root.iterdir()]
        except OSError:
            skipped["unreadable_file"] += 1
            continue
        for bucket in buckets:
            if bucket.is_symlink() or not bucket.is_dir():
                continue
            directories += 1
            if directories > MAX_DIRS:
                skipped["directory_limit"] += 1
                return []
            path = bucket / f"{session_id}.jsonl"
            try:
                if path.is_symlink():
                    skipped["unsafe_file"] += 1
                    continue
                if not path.is_file():
                    continue
                if path.stat().st_size > MAX_FILE_BYTES:
                    skipped["oversized_file"] += 1
                    continue
            except OSError:
                skipped["unreadable_file"] += 1
                continue
            if path not in found:
                found.append(path)
    return found


def _counts(messages: list[Any], session_id: str, *, subagent: bool = False
            ) -> tuple[int, int, str | None, bool]:
    from .history_import import _ID, _MODEL

    prompts: set[str] = set()
    tools: set[str] = set()
    model = None
    partial = not messages
    for item in messages:
        kind = getattr(item, "type", None)
        ident = getattr(item, "uuid", None)
        source = getattr(item, "session_id", None)
        message = getattr(item, "message", None)
        if kind not in {"user", "assistant"} or not isinstance(message, dict):
            partial = True
            continue
        if not isinstance(ident, str) or not _ID.fullmatch(ident):
            partial = True
            continue
        if source not in {None, "", session_id}:
            partial = True
            continue
        if not source:
            partial = True
        content = message.get("content")
        if kind == "user":
            tool_only = isinstance(content, list) and bool(content) and all(
                isinstance(part, dict) and part.get("type") == "tool_result"
                for part in content
            )
            if not subagent and not tool_only:
                prompts.add(ident)
        else:
            candidate = message.get("model")
            if isinstance(candidate, str) and _MODEL.fullmatch(candidate):
                model = candidate
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "tool_use":
                        tool_id = part.get("id")
                        if isinstance(tool_id, str) and _ID.fullmatch(tool_id):
                            tools.add(tool_id)
                        else:
                            partial = True
    return len(prompts), len(tools), model, partial


def read_sessions(repo: Path, roots: list[Path], locations: list[Path],
                  skipped: Counter[str]) -> list[tuple[dict[str, Any], list[Path]]] | None:
    """Return scoped SDK sessions and safe native files for raw usage enrichment.

    ``None`` means the SDK cannot read these configured local projects, so the
    caller can use the legacy reader. Session text never enters the result.
    """
    from .history_import import MAX_FILE_BYTES, MAX_FILES, _BRANCH, _ID, _in_repo, _metadata, _read

    try:
        from claude_agent_sdk import (get_session_messages, get_subagent_messages,
                                      list_sessions, project_key_for_directory)
    except ImportError:
        return None

    configured = Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude"))).expanduser() / "projects"
    if not locations or not any(path.resolve() == configured.resolve() for path in locations):
        return None
    try:
        listed = list_sessions(directory=str(repo), include_worktrees=True)
    except (AttributeError, TypeError, OSError, ValueError):
        return None
    if not isinstance(listed, list):
        return None
    if len(listed) > MAX_FILES:
        skipped["file_limit"] += 1
        return None

    result: list[tuple[dict[str, Any], list[Path]]] = []
    for info in listed:
        session_id = getattr(info, "session_id", None)
        cwd = getattr(info, "cwd", None)
        if not isinstance(session_id, str) or not _ID.fullmatch(session_id):
            skipped["invalid_session_id"] += 1
            continue
        if not _in_repo(cwd, roots):
            skipped["outside_repository"] += 1
            continue
        paths = _native_paths(session_id, locations, skipped)
        if not paths:
            skipped["unreadable_file"] += 1
            continue
        partial = len(paths) != 1
        scoped: list[Path] = []
        project_key = project_key_for_directory(cwd)
        foreign_file = False
        for path in paths:
            records, _lines, damaged = _read(path, skipped)
            if not records:
                partial = True
                continue
            matched, foreign, *_ = _metadata(records, "claude", roots)
            if foreign:
                skipped["mixed_repository"] += 1
                foreign_file = True
                continue
            if not matched and path.parent.name != project_key:
                skipped["outside_repository"] += 1
                continue
            scoped.append(path)
            partial |= damaged
            partial |= any(
                record.get("type") in {"user", "assistant"} and
                (not isinstance(record.get("uuid"), str) or
                 not _ID.fullmatch(record["uuid"]))
                for record in records
            )
        if foreign_file or not scoped:
            continue
        try:
            messages = get_session_messages(session_id, directory=str(repo))
        except (AttributeError, TypeError, OSError, ValueError):
            return None
        if not isinstance(messages, list):
            return None
        prompts, tools, model, incomplete_messages = _counts(messages, session_id)
        started = _time(getattr(info, "created_at", None))
        last = _time(getattr(info, "last_modified", None))
        if not started or not last:
            skipped["missing_timestamp"] += 1
            continue
        branch = getattr(info, "git_branch", None)
        if not isinstance(branch, str) or not _BRANCH.fullmatch(branch) or ".." in branch:
            branch = None
        session = {
            "provider": "claude", "native_session_id": session_id,
            "started_at": started, "last_activity_at": last,
            "branch": branch, "model": model, "usage": [],
            "prompt_count": prompts, "tool_call_count": tools,
            "completeness": "partial" if partial or incomplete_messages else "complete",
            "created_commits": [],
        }
        result.append((session, scoped))
        seen_agents: set[str] = set()
        for parent_path in scoped:
            agent_dir = parent_path.with_suffix("") / "subagents"
            if agent_dir.is_symlink() or not agent_dir.is_dir():
                continue
            try:
                entries = list(agent_dir.iterdir())
            except OSError:
                skipped["unreadable_file"] += 1
                session["completeness"] = "partial"
                continue
            # The SDK's subagent lookup walks recursively. Only hand it a
            # simple canonical directory whose entries are already checked.
            if len(entries) > MAX_FILES or any(entry.is_symlink() or entry.is_dir() for entry in entries):
                skipped["unsafe_file"] += 1
                session["completeness"] = "partial"
                continue
            for child_path in entries:
                if not child_path.name.startswith("agent-") or child_path.suffix != ".jsonl":
                    continue
                agent_id = child_path.stem[6:]
                native = f"{session_id}:{agent_id}"
                if (not _ID.fullmatch(agent_id) or len(native) > 200 or
                        agent_id in seen_agents):
                    skipped["invalid_session_id"] += 1
                    continue
                seen_agents.add(agent_id)
                try:
                    if not child_path.is_file() or child_path.stat().st_size > MAX_FILE_BYTES:
                        skipped["oversized_file"] += 1
                        continue
                except OSError:
                    skipped["unreadable_file"] += 1
                    continue
                records, _lines, child_damaged = _read(child_path, skipped)
                matched, foreign, first, last_child, child_branch, *_ = _metadata(
                    records, "claude", roots
                )
                if foreign or not matched:
                    skipped["mixed_repository" if foreign else "outside_repository"] += 1
                    continue
                try:
                    child_messages = get_subagent_messages(session_id, agent_id, directory=str(repo))
                except (AttributeError, TypeError, OSError, ValueError):
                    return None
                if not isinstance(child_messages, list):
                    return None
                _, child_tools, child_model, child_incomplete = _counts(
                    child_messages, session_id, subagent=True
                )
                if not first or not last_child:
                    skipped["missing_timestamp"] += 1
                    continue
                result.append(({
                    "provider": "claude", "native_session_id": native,
                    "parent_native_session_id": session_id,
                    "started_at": first, "last_activity_at": last_child,
                    "branch": child_branch or branch, "model": child_model,
                    "usage": [], "prompt_count": 0, "tool_call_count": child_tools,
                    "completeness": "partial" if child_damaged or child_incomplete else "complete",
                    "created_commits": [],
                }, [child_path]))
    return result
