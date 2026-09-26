"""Local, reviewable import of coding-session metadata.

Only the allowlisted fields assembled here can cross the network. Session text,
file locations, and Git remote URLs are read locally and then discarded.
"""

from __future__ import annotations

from collections import Counter
from decimal import Decimal
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .hosted import _remote_identity
from .usage_fallback import MAX_LINE_BYTES, parse_claude_lines, parse_codex_lines, price_row

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_FILES = 5000
MAX_DIRS = 10000
MAX_UPLOAD_BYTES = 512 * 1024
# Commit claims in one import. Each one costs the service a GitHub lookup.
MAX_COMMIT_CLAIMS = 200
_ID = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._:-]{0,199}$")
_BRANCH = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]{0,199}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,99}$")
_COMMIT_OUTPUT = re.compile(r"(?m)^\[[^\]\n]{1,200} ([0-9a-f]{7,40})\] ")
_FAILED_TOOL_OUTPUT = re.compile(
    r"(?im)^\s*(?:Process exited with code|exit_code|Exit code)\s*:?\s*[1-9]\d*\s*$"
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def repository_roots(repo: Path, expected: str) -> list[Path]:
    """Verify GitHub identity and collect known worktree locations."""

    if not _REPOSITORY.fullmatch(expected) or ".." in expected:
        raise ValueError("--repository must be a GitHub owner/repository name")
    root_text = _git(repo, "rev-parse", "--show-toplevel")
    if not root_text:
        raise ValueError("--repo must be a Git checkout")
    root = Path(root_text).resolve()
    remote = _git(root, "remote", "get-url", "origin")
    try:
        parsed_remote = urlsplit(remote)
    except ValueError:
        raise ValueError("the GitHub origin URL is invalid") from None
    if parsed_remote.scheme in {"https", "ssh", "git"} and parsed_remote.hostname:
        try:
            remote_port = parsed_remote.port
        except ValueError:
            raise ValueError("the GitHub origin URL is invalid") from None
        remote = urlunsplit((parsed_remote.scheme, parsed_remote.hostname +
                             (f":{remote_port}" if remote_port else ""),
                             parsed_remote.path, "", ""))
    identity = _remote_identity(remote)
    if identity is None or "/".join(identity).lower() != expected.lower():
        raise ValueError("--repository does not match the checkout's GitHub origin")
    roots = {root}
    for line in _git(root, "worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            roots.add(Path(line[9:]).resolve())
    common = _git(root, "rev-parse", "--git-common-dir")
    if common:
        common_path = Path(common)
        if not common_path.is_absolute():
            common_path = root / common_path
        linked = common_path.resolve() / "worktrees"
        if linked.is_dir():
            for entry in linked.iterdir():
                if entry.is_symlink() or not entry.is_dir():
                    continue
                gitdir = entry / "gitdir"
                try:
                    historical = Path(gitdir.read_text(encoding="utf-8").strip()).parent
                except (OSError, UnicodeError):
                    continue
                if historical.is_absolute():
                    roots.add(historical.resolve())
    return [root, *sorted(roots - {root})]


def _in_repo(cwd: Any, roots: list[Path]) -> bool:
    if not isinstance(cwd, str) or not cwd.startswith("/") or "\x00" in cwd:
        return False
    try:
        path = Path(cwd).resolve()
    except (OSError, RuntimeError):
        # A symlink loop or an unreadable path names no checkout.
        return False
    if not any(path == root or root in path.parents for root in roots):
        return False
    # An existing nested checkout can belong to a different repository.
    if path.exists():
        target_common = _git(path, "rev-parse", "--git-common-dir")
        root_common = _git(roots[0], "rev-parse", "--git-common-dir")
        if target_common and root_common:
            def absolute(base: Path, value: str) -> Path:
                item = Path(value)
                return (item if item.is_absolute() else base / item).resolve()
            return absolute(path, target_common) == absolute(roots[0], root_common)
    return True


def _timestamp(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) > 64:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    try:
        return parsed.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError):
        return None


def _commit_command(command: str, *, _wrapped: bool = False) -> tuple[str | None, bool, bool] | None:
    """Parse a plain commit command whose output cannot come from another program.

    Real transcripts chain the commit with a ``cd`` into the checkout, a
    ``git add``, and a ``git log`` or ``git status`` afterwards. Those print
    nothing that looks like a commit summary before the commit's own line.
    Return the ``cd`` target or ``None``, whether segments follow the commit,
    and whether a ``git log`` with only option arguments and no ``--reverse``
    follows, so that its first line names the new commit. Return ``None``
    for every other shape.
    """

    if "\n" in command or "\r" in command:
        return None
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()`")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None
    # Codex runs each command as ``/bin/zsh -lc '<command>'``. The wrapper
    # prints nothing, so the inner command decides. One level only.
    if (len(tokens) == 3 and os.path.basename(tokens[0]) in {"sh", "bash", "zsh"}
            and tokens[1] in {"-c", "-lc"} and not _wrapped):
        return _commit_command(tokens[2], _wrapped=True)
    operators = set(";&|<>()`")
    if any((token != "&&" and all(char in operators for char in token))
           or "$" in token for token in tokens):
        return None
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token == "&&":
            segments.append([])
        else:
            segments[-1].append(token)
    if any(not segment for segment in segments):
        return None
    target = None
    if segments[0][0] == "cd":
        if len(segments) == 1 or len(segments[0]) != 2 or segments[0][1].startswith("~"):
            return None
        target = segments[0][1]
        segments = segments[1:]
    if segments[0][:2] == ["git", "add"]:
        if len(segments) == 1 or len(segments[0]) <= 2:
            return None
        segments = segments[1:]
    commit, *trailing = segments
    if commit[0] != "git":
        return None
    index = 1
    while index + 1 < len(commit) and commit[index] == "-c" and "=" in commit[index + 1]:
        index += 2
    if index >= len(commit) or commit[index] != "commit":
        return None
    if any(segment[:2] not in (["git", "log"], ["git", "status"]) for segment in trailing):
        return None
    log_head = any(
        segment[:2] == ["git", "log"] and all(argument.startswith("-") for argument in segment[2:])
        and not any(argument.startswith("--reverse") for argument in segment[2:])
        for segment in trailing
    )
    return target, bool(trailing), log_head


def _simple_commit_command(command: str) -> bool:
    """Say whether ``_commit_command`` accepts the command."""

    return _commit_command(command) is not None


def _accepted_commit(command: Any, cwd: Any, roots: list[Path]) -> tuple[bool, bool] | None:
    """Return the trailing flags of an accepted commit that ran inside the checkout.

    The flags say whether segments follow the commit and whether a usable
    ``git log`` follows. ``None`` means the command is not a plain commit,
    or it ran, or changed into, a directory outside the repository.
    """

    if not isinstance(command, str):
        return None
    parsed = _commit_command(command)
    if parsed is None:
        return None
    target, trailing, log_head = parsed
    if target is not None:
        if target.startswith("/"):
            cwd = target
        elif isinstance(cwd, str) and cwd.startswith("/"):
            cwd = os.path.join(cwd, target)
        else:
            return None
    return (trailing, log_head) if _in_repo(cwd, roots) else None


# The first line of a git log: ``commit <id>`` or ``<id> subject``.
_LOG_HEAD = re.compile(r"(?m)^(?:commit )?([0-9a-f]{7,40})(?: |$)")


def _commit_summaries(output: str, flags: tuple[bool, bool]) -> list[str]:
    """Return the short commit IDs that a tool output proves.

    With segments after the commit, only the first summary line is the
    commit's own. A later ``git log`` prints commit messages, which can hold
    a line of the same shape. A quiet commit prints no summary. When a
    ``git log`` follows it in the same ``&&`` chain, the chain reached the
    log only after the commit succeeded, so the log's first commit is the
    new one.
    """

    trailing, log_head = flags
    found = _COMMIT_OUTPUT.findall(output)
    if trailing:
        found = found[:1]
    if not found and log_head:
        match = _LOG_HEAD.search(output)
        if match:
            found = [match.group(1)]
    return found


def _roots() -> tuple[list[Path], list[Path]]:
    claude = [Path.home() / ".claude" / "projects"]
    configured = os.environ.get("CLAUDE_CONFIG_DIR")
    if configured:
        claude = [Path(configured).expanduser() / "projects"]
    codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).expanduser()
    return claude, [codex_home / "sessions", codex_home / "archived_sessions"]


def _files(roots: list[Path], skipped: Counter[str]):
    seen: set[tuple[int, int]] = set()
    count = 0
    directories = 0
    for root in roots:
        if not root.is_dir() or root.is_symlink():
            continue
        for directory, dirs, names in os.walk(root, followlinks=False):
            directories += 1
            if directories > MAX_DIRS:
                skipped["directory_limit"] += 1
                return
            dirs[:] = sorted(name for name in dirs if not (Path(directory) / name).is_symlink())
            for name in sorted(names):
                if not name.endswith(".jsonl"):
                    continue
                path = Path(directory) / name
                try:
                    if path.is_symlink() or not path.is_file():
                        skipped["unsafe_file"] += 1
                        continue
                    status = path.stat()
                except OSError:
                    skipped["unreadable_file"] += 1
                    continue
                key = (status.st_dev, status.st_ino)
                if key in seen:
                    continue
                seen.add(key)
                count += 1
                if count > MAX_FILES:
                    skipped["file_limit"] += 1
                    return
                if status.st_size > MAX_FILE_BYTES:
                    skipped["oversized_file"] += 1
                    continue
                yield path


def _read(path: Path, skipped: Counter[str]) -> tuple[list[dict[str, Any]], list[bytes], bool]:
    records: list[dict[str, Any]] = []
    lines: list[bytes] = []
    partial = False
    total = 0
    try:
        with path.open("rb") as source:
            while True:
                raw = source.readline(MAX_LINE_BYTES + 1)
                if not raw:
                    break
                total += len(raw)
                if total > MAX_FILE_BYTES:
                    skipped["oversized_file"] += 1
                    return [], [], True
                if len(raw) > MAX_LINE_BYTES:
                    partial = True
                    skipped["oversized_line"] += 1
                    while raw and not raw.endswith(b"\n"):
                        raw = source.readline(65536)
                    continue
                has_newline = raw.endswith(b"\n")
                try:
                    record = json.loads(raw)
                except (ValueError, RecursionError):
                    partial = True
                    skipped["malformed_line" if has_newline else "truncated_line"] += 1
                    continue
                if not isinstance(record, dict):
                    partial = True
                    skipped["malformed_line" if has_newline else "truncated_line"] += 1
                    continue
                records.append(record)
                lines.append(raw)
    except OSError:
        skipped["unreadable_file"] += 1
        return [], [], True
    return records, lines, partial


def _metadata(records: list[dict[str, Any]], provider: str, roots: list[Path]) -> tuple[bool, bool, str | None, str | None, str | None, set[str], set[str]]:
    matched = False
    foreign = False
    branch = None
    branch_stamp = None
    times: list[str] = []
    prompts: set[str] = set()
    tools: set[str] = set()
    cwd_scope: dict[str, bool] = {}
    for record in records:
        payload = record.get("payload") if provider == "codex" else record
        if not isinstance(payload, dict):
            continue
        cwd = payload.get("cwd")
        if isinstance(cwd, str) and cwd.startswith("/"):
            if cwd not in cwd_scope:
                cwd_scope[cwd] = _in_repo(cwd, roots)
            if cwd_scope[cwd]:
                matched = True
            else:
                foreign = True
        stamp = _timestamp(record.get("timestamp"))
        candidate = payload.get("gitBranch") or payload.get("branch")
        if provider == "codex" and not candidate:
            git = payload.get("git")
            if isinstance(git, dict):
                candidate = git.get("branch")
        if isinstance(candidate, str) and _BRANCH.fullmatch(candidate) and ".." not in candidate:
            if branch is None or (stamp is not None and (branch_stamp is None or stamp >= branch_stamp)):
                branch = candidate
                branch_stamp = stamp
        if stamp:
            times.append(stamp)
        kind = record.get("type")
        if provider == "claude":
            message = record.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            tool_only = isinstance(content, list) and bool(content) and all(
                isinstance(part, dict) and part.get("type") == "tool_result" for part in content
            )
            if kind == "user" and record.get("isSidechain") is not True and not tool_only:
                ident = record.get("uuid")
                if isinstance(ident, str) and _ID.fullmatch(ident):
                    prompts.add(ident)
            if kind == "assistant":
                message = record.get("message")
                content = message.get("content") if isinstance(message, dict) else None
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "tool_use":
                            ident = part.get("id")
                            if isinstance(ident, str) and _ID.fullmatch(ident):
                                tools.add(ident)
        elif kind == "event_msg" and payload.get("type") == "user_message":
            digest = hashlib.sha256(json.dumps(payload.get("message"), sort_keys=True,
                                               ensure_ascii=True).encode("utf-8")).hexdigest()
            prompts.add(f"{stamp}:{digest}")
        elif (kind == "response_item" and isinstance(payload.get("type"), str)
              and payload.get("type") in {"function_call", "custom_tool_call"}):
            ident = payload.get("call_id")
            if isinstance(ident, str) and _ID.fullmatch(ident):
                tools.add(ident)
    return matched, foreign, min(times) if times else None, max(times) if times else None, branch, prompts, tools


def _native_model(records: list[dict[str, Any]], provider: str) -> tuple[str | None, str | None]:
    model = None
    model_stamp = None
    for record in records:
        payload = record.get("payload") if provider == "codex" else record
        if not isinstance(payload, dict):
            continue
        candidate = None
        if provider == "claude":
            message = record.get("message")
            candidate = message.get("model") if isinstance(message, dict) else None
        elif record.get("type") == "turn_context":
            candidate = payload.get("model")
        elif record.get("type") == "event_msg" and payload.get("type") == "thread_settings_applied":
            settings = payload.get("thread_settings")
            candidate = settings.get("model") if isinstance(settings, dict) else None
        if not isinstance(candidate, str) or not _MODEL.fullmatch(candidate):
            continue
        stamp = _timestamp(record.get("timestamp"))
        if model is None or (stamp is None and model_stamp is None) or (
                stamp is not None and (model_stamp is None or stamp >= model_stamp)):
            model = candidate
            model_stamp = stamp
    return model, model_stamp


def _created_commits(records: list[dict[str, Any]], provider: str, repo: Path,
                     roots: list[Path]) -> list[dict[str, str]]:
    """Accept a commit only from a successful, repo-scoped paired tool call."""

    # Accepted commit calls map to the trailing flags of their command.
    commands: dict[str, tuple[bool, bool]] = {}
    modern_calls: set[str] = set()
    outputs: list[tuple[str, str]] = []
    cwd = None

    def scoped(command: str, command_cwd: Any) -> None:
        flags = _accepted_commit(command, command_cwd, roots)
        if flags is not None:
            commands[ident] = flags

    for record in records:
        payload = record.get("payload") if provider == "codex" else record.get("message")
        if not isinstance(payload, dict):
            continue
        record_cwd = payload.get("cwd") if provider == "codex" else record.get("cwd")
        if isinstance(record_cwd, str):
            cwd = record_cwd
        if provider == "claude":
            content = payload.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                ident = part.get("id") if part.get("type") == "tool_use" else part.get("tool_use_id")
                if not isinstance(ident, str) or not _ID.fullmatch(ident):
                    continue
                if part.get("type") == "tool_use" and part.get("name") == "Bash":
                    arguments = part.get("input")
                    command = arguments.get("command") if isinstance(arguments, dict) else None
                    if isinstance(command, str):
                        scoped(command, record_cwd or cwd)
                elif part.get("type") == "tool_result" and part.get("is_error") is not True:
                    output = part.get("content")
                    if (isinstance(output, str) and len(output) <= 65536
                            and not _FAILED_TOOL_OUTPUT.search(output)):
                        outputs.append((ident, output))
        elif record.get("type") == "response_item":
            ident = payload.get("call_id")
            if not isinstance(ident, str) or not _ID.fullmatch(ident):
                continue
            name = payload.get("name")
            if (payload.get("type") == "function_call" and isinstance(name, str)
                    and name in {"exec_command", "functions.exec_command"}):
                arguments = payload.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (TypeError, ValueError):
                        continue
                if not isinstance(arguments, dict):
                    continue
                command = arguments.get("cmd") or arguments.get("command")
                command_cwd = arguments.get("workdir") or arguments.get("cwd") or record_cwd or cwd
                if isinstance(command, str):
                    scoped(command, command_cwd)
                    if name != "exec_command":
                        modern_calls.add(ident)
            elif payload.get("type") == "function_call_output":
                output = payload.get("output")
                structured = output if isinstance(output, dict) else None
                if isinstance(output, str):
                    try:
                        decoded = json.loads(output)
                    except (TypeError, ValueError):
                        decoded = None
                    if isinstance(decoded, dict) and "exit_code" in decoded:
                        structured = decoded
                if ident in modern_calls:
                    if (not isinstance(structured, dict)
                            or type(structured.get("exit_code")) is not int
                            or structured["exit_code"] != 0):
                        continue
                    output = structured.get("output")
                elif isinstance(structured, dict):
                    if "exit_code" in structured and (
                            type(structured.get("exit_code")) is not int
                            or structured["exit_code"] != 0):
                        continue
                    output = structured.get("output")
                if (isinstance(output, str) and len(output) <= 65536
                        and not _FAILED_TOOL_OUTPUT.search(output)):
                    outputs.append((ident, output))
    created: set[str] = set()
    if any(ident in commands and _commit_summaries(output, commands[ident]) for ident, output in outputs):
        from .history_evidence import _promisor_clone
        if _promisor_clone(repo, time.monotonic() + 3):
            return []
    for ident, output in outputs:
        if ident not in commands:
            continue
        for short in _commit_summaries(output, commands[ident]):
            full = _git(repo, "rev-parse", "--verify", f"{short}^{{commit}}")
            if re.fullmatch(r"[0-9a-f]{40}", full):
                created.add(full)
    return [{"sha": sha, "evidence": "created_commit"} for sha in sorted(created)]


def discover(repo: Path, repository: str, *, since: str | None = None,
             claude_roots: list[Path] | None = None, codex_roots: list[Path] | None = None,
             progress: Callable[[str, int, int], None] | None = None) -> dict[str, Any]:
    """Find the sessions of one checkout.

    ``progress`` receives the scan step, the files read so far, and the file
    total: ``catalog`` while a provider's own interface lists sessions, and
    ``files`` for each native file. A page can show a bar from these.
    """

    roots = repository_roots(repo, repository)
    if since is not None:
        try:
            since_date = datetime.strptime(since, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError("--since must be a valid YYYY-MM-DD date") from exc
    else:
        since_date = None
    default_claude, default_codex = _roots()
    skipped: Counter[str] = Counter()
    sessions: dict[tuple[str, str], dict[str, Any]] = {}
    usage_by_session: dict[tuple[str, str], dict[str, Any]] = {}
    native_models: dict[tuple[str, str], tuple[str, str | None]] = {}
    prompts_by_session: dict[tuple[str, str], set[str]] = {}
    tools_by_session: dict[tuple[str, str], set[str]] = {}
    from .history_evidence import MatchBudget, recorded_edit_commits

    match_budget = MatchBudget()
    official: dict[tuple[str, str], dict[str, Any]] = {}
    official_paths: dict[tuple[str, str], set[Path]] = {}
    session_files: Counter[tuple[str, str]] = Counter()
    provider_locations = (("claude", default_claude if claude_roots is None else claude_roots),
                          ("codex", default_codex if codex_roots is None else codex_roots))
    # The file count comes first, so the scan can report a fraction.
    total_files = sum(sum(1 for _ in _files(locations, Counter())) for _, locations in provider_locations) if progress else 0
    scanned = 0
    for provider, locations in provider_locations:
        # Explicit roots support legacy fixtures and nonstandard exported logs.
        # Normal CLI imports use the provider's public local interface first.
        custom_roots = claude_roots if provider == "claude" else codex_roots
        if custom_roots is None:
            if provider == "claude":
                from .history_claude import read_sessions
            else:
                from .history_codex import read_sessions
            if progress:
                progress("catalog", scanned, total_files)
            # The native pass reports skipped records once, including when a
            # provider validates a file and then falls back after an API error.
            for metadata, paths in read_sessions(repo, roots, locations, Counter()) or []:
                key = (provider, metadata["native_session_id"])
                official[key] = metadata
                official_paths[key] = {path.resolve() for path in paths}
        # Native records supply per-request usage that the public interfaces
        # omit. They also retain old sessions absent from provider catalogs.
        for path in _files(locations, skipped):
            scanned += 1
            if progress:
                progress("files", scanned, total_files)
            records, lines, partial = _read(path, skipped)
            if not records:
                for key, selected_paths in official_paths.items():
                    if key in official and path.resolve() in selected_paths:
                        official[key]["completeness"] = "partial"
                continue
            if provider == "claude":
                grouped: dict[tuple[str, str | None, str, str | None], tuple[list[dict[str, Any]], list[bytes]]] = {}
                for record, raw in zip(records, lines):
                    file_agent = path.stem[6:] if path.stem.startswith("agent-") else None
                    source_id = record.get("sessionId") or (
                        path.parent.parent.name if file_agent and path.parent.name == "subagents"
                        else path.stem
                    )
                    if not isinstance(source_id, str) or not _ID.fullmatch(source_id):
                        skipped["invalid_session_id"] += 1
                        continue
                    record_agent = record.get("agentId") if record.get("isSidechain") is True else None
                    agent = record_agent or (file_agent if record.get("isSidechain") is True else None)
                    if agent is not None and (not isinstance(agent, str) or not _ID.fullmatch(agent)):
                        skipped["invalid_session_id"] += 1
                        continue
                    native = source_id
                    if agent:
                        paired = f"{source_id}:{agent}"
                        native = paired if len(paired) <= 200 else "agent-" + hashlib.sha256(paired.encode()).hexdigest()
                    parent = source_id if agent else None
                    group_records, group_lines = grouped.setdefault((native, parent, source_id, agent), ([], []))
                    group_records.append(record)
                    group_lines.append(raw)
                batches = [(native, group_records, group_lines, parent, source_id, agent)
                           for (native, parent, source_id, agent), (group_records, group_lines) in grouped.items()]
            else:
                meta = next((r.get("payload") for r in records if r.get("type") == "session_meta" and isinstance(r.get("payload"), dict)), {})
                thread = meta.get("id", path.stem.rsplit("-", 1)[-1])
                parent = meta.get("parent_thread_id")
                if not isinstance(thread, str) or not _ID.fullmatch(thread):
                    skipped["invalid_session_id"] += 1
                    continue
                batches = [(thread, records, lines, parent, thread, None)]
            for native, batch_records, batch_lines, parent, source_id, agent in batches:
                key = (provider, native)
                matched, foreign, first, last, branch, prompts, tools = _metadata(batch_records, provider, roots)
                selected_path = path.resolve() in official_paths.get(key, set())
                if foreign and selected_path:
                    official.pop(key, None)
                if selected_path:
                    matched = True
                if not matched:
                    skipped["outside_repository"] += 1
                    continue
                if foreign:
                    skipped["mixed_repository"] += 1
                    continue
                session_files[key] += 1
                native_model, native_model_stamp = _native_model(batch_records, provider)
                if provider == "claude":
                    latest_first = [raw for _record, raw in sorted(
                        zip(batch_records, batch_lines),
                        key=lambda pair: _timestamp(pair[0].get("timestamp")) or "",
                        reverse=True,
                    )]
                    rows = parse_claude_lines(latest_first, session_id=source_id, agent_id=agent)
                else:
                    rows, _state = parse_codex_lines(batch_lines, session_id=native, state={})
                if key not in sessions:
                    sessions[key] = {
                        "provider": provider, "native_session_id": native,
                        "started_at": first, "last_activity_at": last,
                        "branch": branch, "model": None, "usage": [],
                        "prompt_count": 0, "tool_call_count": 0,
                        "completeness": "partial" if partial else "complete",
                        "created_commits": [],
                    }
                    usage_by_session[key] = {}
                    prompts_by_session[key] = set()
                    tools_by_session[key] = set()
                session = sessions[key]
                previous_model = native_models.get(key)
                if native_model and (previous_model is None or
                                     (native_model_stamp is None and previous_model[1] is None) or
                                     (native_model_stamp is not None and
                                      (previous_model[1] is None or
                                       native_model_stamp >= previous_model[1]))):
                    native_models[key] = (native_model, native_model_stamp)
                previous_last = session["last_activity_at"]
                if isinstance(parent, str) and _ID.fullmatch(parent):
                    session["parent_native_session_id"] = parent
                session["started_at"] = min(filter(None, (session["started_at"], first)), default=None)
                session["last_activity_at"] = max(filter(None, (session["last_activity_at"], last)), default=None)
                if branch and (session["branch"] is None or
                               (last is not None and (previous_last is None or last >= previous_last))):
                    session["branch"] = branch
                prompts_by_session[key].update(prompts)
                tools_by_session[key].update(tools)
                session["prompt_count"] = len(prompts_by_session[key])
                session["tool_call_count"] = len(tools_by_session[key])
                if partial:
                    session["completeness"] = "partial"
                existing_commits = {row["sha"]: row for row in session["created_commits"]}
                for claim in _created_commits(batch_records, provider, repo, roots):
                    existing_commits[claim["sha"]] = claim
                if not partial and session_files[key] == 1:
                    for claim in recorded_edit_commits(batch_records, provider, repo, roots, budget=match_budget):
                        existing_commits.setdefault(claim["sha"], claim)
                session["created_commits"] = list(existing_commits.values())
                for row in rows:
                    if row.native_session_id != source_id and provider == "claude":
                        continue
                    existing = usage_by_session[key].get(row.event_key)
                    if existing is not None:
                        new_rank = (row.output_tokens or 0, row.observed_at_unix_nano or -1)
                        old_rank = (existing.output_tokens or 0,
                                    existing.observed_at_unix_nano or -1)
                        if new_rank <= old_rank:
                            continue
                    usage_by_session[key][row.event_key] = row
    if progress:
        progress("files", total_files, total_files)
    for key, metadata in official.items():
        if key not in sessions:
            sessions[key] = dict(metadata, usage=[])
            usage_by_session[key] = {}
            continue
        session = sessions[key]
        # Activity dates in the transcript take precedence over catalog mtime.
        for field in ("started_at", "last_activity_at", "branch", "model",
                      "parent_native_session_id"):
            if session.get(field) is None and metadata.get(field) is not None:
                session[field] = metadata[field]
        for field in ("prompt_count", "tool_call_count"):
            session[field] = max(session[field], metadata[field])
        if metadata["completeness"] == "partial":
            session["completeness"] = "partial"
        claims_by_sha = {row["sha"]: row for row in session["created_commits"]}
        claims_by_sha.update({row["sha"]: row for row in metadata["created_commits"]})
        session["created_commits"] = list(claims_by_sha.values())
    selected = []
    for key, session in sessions.items():
        if session_files[key] > 1 or session.get("completeness") != "complete":
            # Copied, split, or partial transcripts can omit tool outcomes.
            # Keep their usage, but do not infer commits from one fragment.
            session["created_commits"] = [claim for claim in session["created_commits"]
                                          if claim["evidence"] != "recorded_edit"]
        model_rows = [row for row in usage_by_session[key].values()
                      if row.model and _MODEL.fullmatch(row.model)]
        if model_rows:
            latest_model = max(model_rows, key=lambda row: (
                row.observed_at_unix_nano if row.observed_at_unix_nano is not None else -1,
                row.event_key,
            ))
            session["model"] = latest_model.model
        elif key in native_models:
            session["model"] = native_models[key][0]
        first, last = session["started_at"], session["last_activity_at"]
        if first is None or last is None:
            skipped["missing_timestamp"] += 1
            continue
        if since_date and datetime.fromisoformat(last.replace("Z", "+00:00")).date() < since_date:
            skipped["before_since"] += 1
            continue
        for row in usage_by_session[key].values():
            priced = price_row(row)
            item = {field: getattr(priced, field) for field in (
                "event_key", "model", "input_tokens", "cached_input_tokens",
                "cache_creation_input_tokens", "cache_creation_1h_input_tokens",
                "output_tokens", "total_tokens", "cost_amount", "cost_unit",
                "unpriced_reason",
            ) if getattr(priced, field) is not None}
            if "model" in item and not _MODEL.fullmatch(item["model"]):
                item.pop("model")
            if "unpriced_reason" in item:
                reason = item["unpriced_reason"].split(":", 1)[0]
                item["unpriced_reason"] = reason if re.fullmatch(r"[a-z_]{1,60}", reason) else "unpriced"
            if "cost_amount" in item:
                amount = Decimal(item["cost_amount"])
                item["cost_amount"] = float(amount)
                item["cost_unit"] = item["cost_unit"].lower()
                item["cost_source"] = "api_equivalent"
            if session["provider"] == "claude" and "total_tokens" not in item:
                counts = (
                    row.input_tokens, row.cached_input_tokens,
                    row.cache_creation_input_tokens, row.output_tokens,
                )
                if all(value is not None for value in counts):
                    item["total_tokens"] = sum(counts)
            if priced.observed_at_unix_nano is not None:
                item["observed_at"] = datetime.fromtimestamp(
                    priced.observed_at_unix_nano / 1_000_000_000,
                    tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            session["usage"].append(item)
        session["usage"].sort(key=lambda row: (row.get("observed_at", ""), row["event_key"]))
        selected.append(session)
    selected.sort(key=lambda session: (session["started_at"], session["provider"], session["native_session_id"]))
    # Each claim costs one GitHub lookup. The newest sessions keep their
    # claims first, because the page shows recent work first.
    claims = 0
    for session in reversed(selected):
        allowed = max(0, MAX_COMMIT_CLAIMS - claims)
        if len(session["created_commits"]) > allowed:
            skipped["commit_claim_limit"] += len(session["created_commits"]) - allowed
            session["created_commits"] = session["created_commits"][:allowed]
        claims += len(session["created_commits"])
    return {"schema": "joyride.session-import@1", "repository": repository,
            "sessions": selected, "complete": True,
            "skipped": [{"reason": reason, "count": count} for reason, count in sorted(skipped.items())]}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def _origin(value: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise ValueError("--hosted-url must be an HTTPS origin") from None
    local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if (parsed.scheme != "https" and not (local and parsed.scheme == "http")) or not parsed.hostname or parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment or (port is not None and not 1 <= port <= 65535):
        raise ValueError("--hosted-url must be an HTTPS origin (HTTP is allowed for localhost)")
    return value.rstrip("/")


def upload(envelope: dict[str, Any], hosted_url: str, code: str) -> dict[str, int]:
    if not code or len(code) > 4096 or any(ord(c) < 33 or ord(c) > 126 for c in code):
        raise ValueError("--code must be a valid import credential")
    origin = _origin(hosted_url)
    opener = build_opener(_NoRedirect)

    def post(data: bytes) -> dict[str, Any]:
        request = Request(origin + "/v1/history/sessions", data=data, method="POST",
                          headers={"Authorization": "Bearer " + code,
                                   "Content-Type": "application/json"})
        with opener.open(request, timeout=20) as response:
            raw = response.read(65537)
            if len(raw) > 65536:
                raise ValueError("hosted response exceeded the size limit")
            answer = json.loads(raw)
            if not isinstance(answer, dict):
                raise ValueError("hosted response was invalid")
            return answer

    return upload_with(envelope, post)


def upload_with(envelope: dict[str, Any], post: Callable[[bytes], dict[str, Any]],
                on_request: Callable[[dict[str, Any], int, int], None] | None = None) -> dict[str, int]:
    """Send the envelope in bounded requests through ``post``.

    ``post`` sends one request body and returns the parsed answer. A temporary
    service error (HTTP 429, 500, 502, 503, or 504) or a network error is
    retried up to two times. ``on_request`` receives each answer with the
    count of requests sent so far and the total.
    """

    sessions = envelope["sessions"]
    results: Counter[str] = Counter()
    chunks: list[list[dict[str, Any]]] = []
    chunk: list[dict[str, Any]] = []
    claims = 0
    fragments: list[dict[str, Any]] = []
    def payload_size(batch: list[dict[str, Any]]) -> int:
        return len(json.dumps({**envelope, "sessions": batch}, separators=(",", ":")).encode("utf-8"))

    for session in sessions:
        usage = session["usage"]
        commit_rows = session["created_commits"]
        start = commit_start = 0
        while True:
            width = min(1000, len(usage) - start)
            commit_batch = commit_rows[commit_start:commit_start + 3]
            while True:
                final = start + width == len(usage) and commit_start + len(commit_batch) == len(commit_rows)
                fragment = {**session, "usage": usage[start:start + width],
                            "created_commits": commit_batch}
                if not final:
                    fragment["completeness"] = "partial"
                if payload_size([fragment]) <= MAX_UPLOAD_BYTES:
                    break
                if width <= 1:
                    raise ValueError("one session record exceeds the upload size limit")
                width = max(1, width // 2)
            fragments.append(fragment)
            start += width
            commit_start += len(commit_batch)
            if final:
                break
    for fragment in fragments:
        candidate = chunk + [fragment]
        candidate_claims = claims + len(fragment["created_commits"])
        size = payload_size(candidate)
        if chunk and (len(candidate) > 100 or candidate_claims > 3 or
                      sum(len(item["usage"]) for item in candidate) > 2000 or
                      size > MAX_UPLOAD_BYTES):
            chunks.append(chunk)
            chunk = [fragment]
            claims = len(fragment["created_commits"])
            if claims > 3 or payload_size(chunk) > MAX_UPLOAD_BYTES:
                raise ValueError("one session fragment exceeds the upload limit")
        elif size > MAX_UPLOAD_BYTES or candidate_claims > 3:
            raise ValueError("one session fragment exceeds the upload limit")
        else:
            chunk = candidate
            claims = candidate_claims
    if chunk:
        chunks.append(chunk)
    for index, batch in enumerate(chunks):
        payload = {**envelope, "sessions": batch,
                   "complete": index == len(chunks) - 1,
                   "skipped": envelope["skipped"] if index == len(chunks) - 1 else []}
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        for attempt in range(3):
            try:
                answer = post(data)
                for name in ("accepted", "created", "updated", "skipped"):
                    number = answer.get(name, 0)
                    if isinstance(number, int) and not isinstance(number, bool) and number >= 0:
                        results[name] += number
                break
            except HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise ValueError(f"hosted import returned HTTP {exc.code}") from None
            except (URLError, TimeoutError, OSError):
                if attempt == 2:
                    raise ValueError("could not reach the hosted import") from None
            time.sleep(0.5 * (attempt + 1))
        if on_request is not None:
            on_request(answer, index + 1, len(chunks))
    return {**results, "selected_sessions": len(sessions), "request_count": len(chunks)}
