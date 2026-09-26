"""Link a recorded successful edit to a unique later Git commit, locally.

The transcript and blob bytes stay on this computer. Only a verified SHA and
an evidence label leave this module. A bounded or ambiguous search makes no
claim; it never guesses from a branch name or nearby timestamp alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Iterable

from .capture import MAX_TEXT_BYTES
from .edit_evidence import MAX_TOTAL_BYTES, _parse_patch, _relative_path, expected_edit
from .history_import import _ID, _in_repo, _timestamp
from .hook_events import response_success
from .runtime import system_subprocess_environment


MAX_EDIT_CALLS = 16
MAX_CANDIDATES = 128
MAX_INPUT_BYTES = 1024 * 1024
MAX_MATCH_SECONDS = 8
MAX_BLOB_CACHE_BYTES = MAX_TOTAL_BYTES
MAX_IMPORT_MATCH_SECONDS = 30
MAX_IMPORT_CANDIDATES = 1024
_SHA = re.compile(r"[0-9a-f]{40}")
_PATCH_TOOLS = {"apply_patch", "functions.apply_patch"}
_CLAUDE_TOOLS = {"Write", "Edit", "MultiEdit"}


class _NoProof(Exception):
    pass


@dataclass
class MatchBudget:
    """One bounded search budget shared by all sessions in an import."""

    seconds_left: float = MAX_IMPORT_MATCH_SECONDS
    candidates_left: int = MAX_IMPORT_CANDIDATES


@dataclass
class _Call:
    ident: str
    name: str
    arguments: dict[str, Any]
    cwd: Path
    root: Path
    call_time: str | None
    call_type: str = "function_call"
    result_time: str | None = None
    result_order: int = 0


def _run(repo: Path, deadline: float, *args: str) -> bytes:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _NoProof
    try:
        result = subprocess.run(
            ["git", "--no-replace-objects", "--literal-pathspecs", "-C", str(repo), *args],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
            timeout=min(remaining, 3), env=system_subprocess_environment(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise _NoProof from exc
    if result.returncode:
        raise _NoProof
    return result.stdout


def _promisor_clone(repo: Path, deadline: float) -> bool:
    """Refuse proof where Git could fetch missing objects while reading blobs."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return True
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "config", "--includes", "--get-regexp",
             r"^(extensions\.partialclone|remote\..*\.(promisor|partialclonefilter))$"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
            timeout=min(remaining, 3), env=system_subprocess_environment(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return True
    if result.returncode not in {0, 1}:
        return True
    if result.returncode == 1:
        return False
    for line in result.stdout.splitlines():
        key, _, value = line.partition(b" ")
        if key.lower().endswith(b".promisor"):
            if value.strip().lower() not in {b"false", b"no", b"off", b"0"}:
                return True
        else:
            return True
    return False


def _root_for(cwd: Any, roots: list[Path]) -> Path | None:
    if not _in_repo(cwd, roots):
        return None
    path = Path(cwd).resolve()
    return next((root for root in sorted(roots, key=lambda item: len(item.parts), reverse=True)
                 if path == root or root in path.parents), None)


def _targets(call: _Call) -> list[str]:
    if call.name == "MultiEdit":
        path = _relative_path(call.arguments.get("file_path"), call.cwd, call.root)
        edits = call.arguments.get("edits")
        if path is None or not isinstance(edits, list) or not 1 <= len(edits) <= MAX_EDIT_CALLS:
            raise _NoProof
        return [path]
    if call.name in _PATCH_TOOLS:
        operations = _parse_patch(call.arguments.get("command"), call.cwd, call.root)
        if operations is None:
            raise _NoProof
        return list(dict.fromkeys(
            path for operation in operations
            for path in (operation.path, operation.destination) if path is not None
        ))
    path = _relative_path(call.arguments.get("file_path"), call.cwd, call.root)
    if path is None:
        raise _NoProof
    return [path]


def _codex_success(output: Any) -> bool:
    if isinstance(output, str):
        if output.startswith("{"):
            try:
                decoded = json.loads(output)
            except (ValueError, TypeError):
                decoded = None
            if isinstance(decoded, dict) and "exit_code" in decoded:
                return _codex_success(decoded)
        return response_success(output, "patch") is True
    if (not isinstance(output, dict) or
            "exit_code" in output and (type(output["exit_code"]) is not int or output["exit_code"] != 0) or
            response_success(output, "patch") is not True):
        return False
    nested = output.get("output")
    return not isinstance(nested, str) or response_success(nested, "patch") is not False


def _paired_calls(records: list[dict[str, Any]], provider: str,
                  roots: list[Path]) -> list[_Call]:
    calls: dict[str, list[_Call]] = {}
    results: dict[str, list[tuple[bool, str | None, int, str]]] = {}
    other_ids: set[str] = set()
    current_cwd: Any = None
    for order, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        stamp = _timestamp(record.get("timestamp"))
        if provider == "claude":
            cwd = record.get("cwd")
            message = record.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                ident = part.get("id") if part.get("type") == "tool_use" else part.get("tool_use_id")
                if (record.get("type") == "assistant" and part.get("type") == "tool_use" and
                        part.get("name") in _CLAUDE_TOOLS and
                        (not isinstance(ident, str) or not _ID.fullmatch(ident))):
                    raise _NoProof
                if not isinstance(ident, str) or not _ID.fullmatch(ident):
                    continue
                if part.get("type") == "tool_use" and part.get("name") not in _CLAUDE_TOOLS:
                    other_ids.add(ident)
                if record.get("type") == "assistant" and part.get("type") == "tool_use" and part.get("name") in _CLAUDE_TOOLS:
                    root = _root_for(cwd, roots)
                    arguments = part.get("input")
                    if root is None or not isinstance(arguments, dict):
                        raise _NoProof
                    calls.setdefault(ident, []).append(_Call(ident, part["name"], arguments, Path(cwd), root, stamp))
                elif record.get("type") == "user" and part.get("type") == "tool_result":
                    valid = "is_error" not in part or type(part["is_error"]) is bool
                    results.setdefault(ident, []).append((valid and part.get("is_error") is not True,
                                                            stamp, order, "tool_result"))
        elif provider == "codex":
            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue
            if isinstance(payload.get("cwd"), str):
                current_cwd = payload["cwd"]
            if record.get("type") != "response_item":
                continue
            ident = payload.get("call_id")
            call_type = payload.get("type")
            if (call_type in {"function_call", "custom_tool_call"} and payload.get("name") in _PATCH_TOOLS and
                    (not isinstance(ident, str) or not _ID.fullmatch(ident))):
                raise _NoProof
            if not isinstance(ident, str) or not _ID.fullmatch(ident):
                continue
            if call_type in {"function_call", "custom_tool_call"} and payload.get("name") not in _PATCH_TOOLS:
                other_ids.add(ident)
            if call_type in {"function_call", "custom_tool_call"} and payload.get("name") in _PATCH_TOOLS:
                if call_type == "custom_tool_call":
                    patch = payload.get("input")
                    cwd = payload.get("cwd") or current_cwd
                else:
                    arguments = payload.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except (ValueError, TypeError):
                            raise _NoProof from None
                    if not isinstance(arguments, dict):
                        raise _NoProof
                    patch = next((arguments[key] for key in ("command", "patch", "input")
                                  if isinstance(arguments.get(key), str)), None)
                    cwd = arguments.get("workdir") or arguments.get("cwd") or current_cwd
                root = _root_for(cwd, roots)
                if not isinstance(patch, str) or root is None:
                    raise _NoProof
                calls.setdefault(ident, []).append(_Call(
                    ident, "apply_patch", {"command": patch}, Path(cwd), root, stamp, call_type,
                ))
            elif call_type in {"function_call_output", "custom_tool_call_output"}:
                exit_code = payload.get("exit_code")
                success = ("exit_code" not in payload or type(exit_code) is int and exit_code == 0)
                results.setdefault(ident, []).append((success and _codex_success(payload.get("output")),
                                                        stamp, order, call_type))
    if not calls or len(calls) > MAX_EDIT_CALLS or calls.keys() & other_ids:
        raise _NoProof
    paired: list[_Call] = []
    total_bytes = 0
    for ident, copies in calls.items():
        first = copies[0]
        try:
            encoded = json.dumps(first.arguments, sort_keys=True, ensure_ascii=True)
        except (TypeError, ValueError) as exc:
            raise _NoProof from exc
        if any((item.name, item.arguments, item.cwd, item.root, item.call_time, item.call_type) !=
               (first.name, first.arguments, first.cwd, first.root, first.call_time, first.call_type) for item in copies):
            raise _NoProof
        total_bytes += len(encoded.encode("utf-8"))
        if total_bytes > MAX_INPUT_BYTES:
            raise _NoProof
        outcomes = results.get(ident)
        expected_result = ("tool_result" if provider == "claude" else
                           "custom_tool_call_output" if first.call_type == "custom_tool_call" else
                           "function_call_output")
        if (not outcomes or first.call_time is None or
                any(not success or stamp is None or stamp < first.call_time or result_type != expected_result
                    for success, stamp, _order, result_type in outcomes)):
            raise _NoProof
        first.result_time, first.result_order = max(
            (stamp, order) for _success, stamp, order, _result_type in outcomes if stamp is not None
        )
        paired.append(first)
    paired.sort(key=lambda item: (item.result_time or "", item.result_order))
    return paired


def _blob(repo: Path, commit: str, path: str, deadline: float,
          cache: dict[tuple[str, str], bytes | None], cache_bytes: list[int]) -> bytes | None:
    key = (commit, path)
    if key in cache:
        return cache[key]
    parts = Path(path).parts
    for index in range(1, len(parts)):
        ancestor = Path(*parts[:index]).as_posix()
        entry = _run(repo, deadline, "ls-tree", "-z", commit, "--", ancestor)
        if not entry:
            continue
        rows = [row for row in entry.split(b"\0") if row]
        if len(rows) != 1:
            raise _NoProof
        try:
            header, actual = rows[0].split(b"\t", 1)
            mode, kind, _oid = header.decode("ascii").split()
            if actual.decode("utf-8") != ancestor or mode != "040000" or kind != "tree":
                raise _NoProof
        except (ValueError, UnicodeError) as exc:
            raise _NoProof from exc
    listing = _run(repo, deadline, "ls-tree", "-z", commit, "--", path)
    if not listing:
        cache[key] = None
        return None
    rows = [row for row in listing.split(b"\0") if row]
    if len(rows) != 1:
        raise _NoProof
    try:
        header, actual = rows[0].split(b"\t", 1)
        mode, kind, oid = header.decode("ascii").split()
        if actual.decode("utf-8") != path or mode not in {"100644", "100755"} or kind != "blob":
            raise _NoProof
        size = int(_run(repo, deadline, "cat-file", "-s", oid).strip())
    except (UnicodeError, ValueError) as exc:
        raise _NoProof from exc
    if size > MAX_TEXT_BYTES or cache_bytes[0] + size > MAX_BLOB_CACHE_BYTES:
        raise _NoProof
    content = _run(repo, deadline, "cat-file", "blob", oid)
    if len(content) != size or b"\0" in content:
        raise _NoProof
    try:
        content.decode("utf-8")
    except UnicodeError as exc:
        raise _NoProof from exc
    cache[key] = content
    cache_bytes[0] += size
    return content


def _matches(repo: Path, parent: str, candidate: str, calls: list[_Call],
             targets: list[str], deadline: float,
             cache: dict[tuple[str, str], bytes | None], cache_bytes: list[int]) -> bool:
    original = {path: _blob(repo, parent, path, deadline, cache, cache_bytes) for path in targets}
    current = dict(original)
    for call in calls:
        steps = [call.arguments]
        if call.name == "MultiEdit":
            if any(not isinstance(step, dict) for step in call.arguments["edits"]):
                return False
            steps = [{**step, "file_path": call.arguments["file_path"]}
                     for step in call.arguments["edits"]]
        for arguments in steps:
            def reader(_repo: Path, paths: Iterable[str]) -> dict[str, bytes | None] | None:
                return {path: current[path] for path in paths if path in current}

            evidence = expected_edit(
                "codex" if call.name == "apply_patch" else "claude",
                call.name if call.name != "MultiEdit" else "Edit",
                arguments, call.cwd, call.root, read_targets_for_edit=reader,
            )
            if evidence is None:
                return False
            before, after = evidence
            if before == after:
                return False
            current.update(after)
    return all(original[path] != current[path] and
               _blob(repo, candidate, path, deadline, cache, cache_bytes) == current[path]
               for path in targets)


def recorded_edit_commits(records: list[dict[str, Any]], provider: str, repo: Path,
                          roots: list[Path], *, budget: MatchBudget | None = None) -> list[dict[str, str]]:
    """Prove one session's strict edit chain against one unique later commit."""
    if provider not in {"claude", "codex"}:
        return []
    budget = budget or MatchBudget()
    started = time.monotonic()
    deadline = started + min(MAX_MATCH_SECONDS, budget.seconds_left)
    try:
        if budget.candidates_left <= 0 or budget.seconds_left <= 0:
            return []
        roots = [root.resolve() for root in roots]
        calls = _paired_calls(records, provider, roots)
        if len({call.root for call in calls}) != 1:
            return []
        targets = list(dict.fromkeys(path for call in calls for path in _targets(call)))
        if not targets or len(targets) > 64:
            return []
        if _promisor_clone(repo, deadline):
            return []
        final = max(call.result_time or "" for call in calls)
        final_seconds = datetime.fromisoformat(final.replace("Z", "+00:00")).timestamp()
        heads = ["HEAD"]
        for root in roots:
            if root == repo.resolve() or not root.is_dir():
                continue
            try:
                head = _run(root, deadline, "rev-parse", "HEAD").decode("ascii").strip()
            except (_NoProof, UnicodeError):
                continue
            if _SHA.fullmatch(head) and head not in heads:
                heads.append(head)
        limit = min(MAX_CANDIDATES, budget.candidates_left)
        listing = _run(
            repo, deadline, "log", "--full-history", "--branches", "--remotes", *heads,
            "--format=%H %ct", f"--since-as-filter={final}", f"--max-count={limit + 1}",
            "--", *targets,
        ).decode("ascii").splitlines()
        if len(listing) > limit:
            budget.candidates_left = 0
            return []
        budget.candidates_left -= len(listing)
        cache: dict[tuple[str, str], bytes | None] = {}
        cache_bytes = [0]
        matches: list[str] = []
        for line in listing:
            fields = line.split()
            if len(fields) != 2 or not _SHA.fullmatch(fields[0]) or not fields[1].isdigit():
                raise _NoProof
            candidate, committed = fields[0], int(fields[1])
            if committed <= final_seconds:
                continue
            parents = _run(repo, deadline, "rev-list", "--parents", "-n", "1", candidate).decode("ascii").split()
            if len(parents) != 2 or parents[0] != candidate or not _SHA.fullmatch(parents[1]):
                continue
            if _matches(repo, parents[1], candidate, calls, targets, deadline, cache, cache_bytes):
                matches.append(candidate)
                if len(matches) > 1:
                    return []
        return [{"sha": matches[0], "evidence": "recorded_edit"}] if len(matches) == 1 else []
    except (_NoProof, OSError, ValueError, UnicodeError, TypeError, RecursionError,
            IndexError, KeyError, OverflowError):
        return []
    finally:
        budget.seconds_left = max(0.0, budget.seconds_left - (time.monotonic() - started))
