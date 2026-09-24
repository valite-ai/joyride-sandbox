"""Token usage read from a coding tool's own session file.

Cost comes from OpenTelemetry when an exporter reached the local collector.
When one did not, the same numbers still live in the transcript that Claude
Code writes or in the rollout that Codex writes. This module reads only numeric
usage fields and short identifiers from that file, in memory, and stores them
as telemetry events that name their source. It never stores the file's path or
any of its text, and it reads nothing in a repository that opted out.

A request that telemetry also reported counts once: ``costing`` keeps the
telemetry event and drops the matching row from this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


FALLBACK_EVENT_NAME = "joyride.transcript_usage"
# Reports list this beside the price source of a request whose usage came
# from the session file.
TRANSCRIPT_USAGE_SOURCE = "transcript_usage"
# The query source of a request that the transcript marks as the main thread's.
# Costing charges it to the main session, not to a subagent that shares the
# native session.
MAIN_THREAD_SOURCE = "main_thread"
HOOK_EVENTS = frozenset({"Stop", "SessionEnd", "SubagentStop"})
# A usage line is short. A longer one holds a large tool input or result, and
# skipping it loses at most one streamed copy of one request.
MAX_LINE_BYTES = 8 * 1024 * 1024
# One hook reads at most this much new data. The cursor lets the next hook
# event continue where this one stopped.
MAX_READ_BYTES = 64 * 1024 * 1024
MAX_TOKENS = 10**12
_MAX_ITERATIONS = 16
_SESSION_ID = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]{0,199}$")
_CLAUDE_MARKERS = (b'"type":"assistant"', b'"type": "assistant"')
_CODEX_MARKERS = (
    b"token_usage_record",
    b"token_count",
    b"turn_context",
    b"thread_settings_applied",
    b"session_meta",
)


@dataclass(frozen=True)
class UsageRow:
    """One request's usage, with nothing but numbers and short identifiers."""

    event_key: str
    provider: str
    event_kind: str
    native_session_id: str
    agent_id: str | None = None
    query_source: str | None = None
    turn_id: str | None = None
    request_id: str | None = None
    client_request_id: str | None = None
    model: str | None = None
    auth_mode: str | None = None
    service_tier: str | None = None
    speed: str | None = None
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_creation_1h_input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    observed_at_unix_nano: int | None = None
    cost_amount: str | None = None
    cost_unit: str | None = None
    cost_source: str | None = None
    unpriced_reason: str | None = None


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    number = int(value)
    return number if 0 <= number <= MAX_TOKENS else None


def _short(value: Any, maximum: int = 200) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > maximum or any(ord(char) < 32 for char in text):
        return None
    return text


def _timestamp(value: Any) -> int | None:
    text = _short(value, 64)
    if text is None:
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        return None
    return int(moment.timestamp() * 1_000_000_000)


def _key(*parts: Any) -> str:
    encoded = json.dumps(parts, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load(raw: bytes) -> Mapping[str, Any] | None:
    """Decode one line, or return None without keeping any of its text."""

    if len(raw) > MAX_LINE_BYTES:
        return None
    try:
        record = json.loads(raw)
    except (ValueError, RecursionError):
        # A decode error holds the line it failed on, so it is dropped here.
        return None
    return record if isinstance(record, Mapping) else None


def parse_claude_lines(
    lines: Iterable[bytes], *, session_id: str, agent_id: str | None = None
) -> list[UsageRow]:
    """Return one row per Claude request, keeping its highest streamed count.

    Claude Code writes one line per content block, and the earlier copies of a
    request can hold a partial output count. A resumed or forked session copies
    earlier lines with their original session ID, so the key is the request
    alone and the session stays the one that made it.
    """

    rows: dict[str, UsageRow] = {}
    for raw in lines:
        if len(raw) > MAX_LINE_BYTES or not any(marker in raw for marker in _CLAUDE_MARKERS):
            continue
        record = _load(raw)
        if record is None or record.get("type") != "assistant":
            continue
        message = record.get("message")
        if not isinstance(message, Mapping):
            continue
        usage = message.get("usage")
        if not isinstance(usage, Mapping):
            continue
        message_id = _short(message.get("id"))
        request_id = _short(record.get("requestId"))
        session = _short(record.get("sessionId")) or session_id
        if message_id is None and request_id is None:
            continue
        base = (
            ("claude-transcript", message_id, request_id)
            if request_id is not None
            else ("claude-transcript", session, message_id)
        )
        sidechain = record.get("isSidechain") is True
        agent = _short(record.get("agentId")) or (agent_id if sidechain else None)
        query_source = MAIN_THREAD_SOURCE if record.get("isSidechain") is False else None
        model = _short(message.get("model"))
        observed = _timestamp(record.get("timestamp"))
        speed = _short(usage.get("speed"), 32)
        # Claude reports fast mode as a speed. The price table reads it as
        # the request's tier, as it does for a telemetry event.
        tier = "fast" if speed == "fast" else _short(usage.get("service_tier"), 32)

        iterations = usage.get("iterations")
        parts: list[tuple[str, Mapping[str, Any], str | None]] = []
        if isinstance(iterations, list) and len(iterations) > 1:
            # A fallback request reports each attempt beside the last one,
            # and the top-level usage repeats only the last attempt.
            for index, item in enumerate(iterations[:_MAX_ITERATIONS]):
                if isinstance(item, Mapping):
                    parts.append(
                        (_key(*base, index), item, _short(item.get("model")) or model)
                    )
        else:
            # An earlier streamed copy can list no attempts yet. It shares the
            # first attempt's key, so a later copy that lists them replaces it.
            parts.append((_key(*base, 0), usage, model))

        for event_key, values, part_model in parts:
            if part_model is None or part_model == "<synthetic>":
                continue
            split = values.get("cache_creation")
            one_hour = (
                _count(split.get("ephemeral_1h_input_tokens"))
                if isinstance(split, Mapping)
                else None
            )
            row = UsageRow(
                event_key=event_key,
                provider="claude",
                event_kind="claude_transcript",
                native_session_id=session,
                agent_id=agent,
                query_source=query_source,
                request_id=request_id,
                client_request_id=message_id,
                model=part_model,
                service_tier=tier,
                speed=speed,
                input_tokens=_count(values.get("input_tokens")),
                cached_input_tokens=_count(values.get("cache_read_input_tokens")),
                cache_creation_input_tokens=_count(values.get("cache_creation_input_tokens")),
                cache_creation_1h_input_tokens=one_hour,
                output_tokens=_count(values.get("output_tokens")),
                observed_at_unix_nano=observed,
            )
            existing = rows.get(event_key)
            if existing is None or (row.output_tokens or 0) > (existing.output_tokens or 0):
                rows[event_key] = row
    return list(rows.values())


def _codex_identity(
    state: Mapping[str, Any], thread: str
) -> tuple[str, str | None, str | None]:
    """Return the native session, agent, and query source of a thread's rows.

    Codex runs a subagent in a thread of its own and names that thread's ID
    as the hook payload's ``agent_id``, so the subagent's ledger session is
    keyed by its parent and that ID. A child thread's rows carry the same two
    values, so costing charges them to the child and not to the parent.
    """

    parent = state.get("parent")
    if isinstance(parent, str) and parent:
        return parent, thread, None
    agent = state.get("agent")
    if isinstance(agent, str) and agent:
        return thread, agent, None
    main = MAIN_THREAD_SOURCE if isinstance(state.get("thread"), str) else None
    return thread, None, main


def _codex_row(
    event_key: str,
    kind: str,
    thread: str,
    usage: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    request_id: str | None,
    turn_id: str | None,
    observed: int | None,
) -> UsageRow:
    native, agent, query_source = _codex_identity(state, thread)
    return UsageRow(
        event_key=event_key,
        provider="codex",
        event_kind=kind,
        native_session_id=native,
        agent_id=agent,
        query_source=query_source,
        turn_id=turn_id or (state.get("turn") if isinstance(state.get("turn"), str) else None),
        request_id=request_id,
        model=state.get("model") if isinstance(state.get("model"), str) else None,
        auth_mode=state.get("auth_mode") if isinstance(state.get("auth_mode"), str) else None,
        service_tier=(
            state.get("service_tier") if isinstance(state.get("service_tier"), str) else None
        ),
        input_tokens=_count(usage.get("input_tokens")),
        cached_input_tokens=_count(usage.get("cached_input_tokens")),
        cache_creation_input_tokens=_count(usage.get("cache_write_input_tokens")),
        output_tokens=_count(usage.get("output_tokens")),
        total_tokens=_count(usage.get("total_tokens")),
        observed_at_unix_nano=observed,
    )


def parse_codex_lines(
    lines: Iterable[bytes], *, session_id: str, state: Mapping[str, Any]
) -> tuple[list[UsageRow], dict[str, Any]]:
    """Return one row per Codex response and the state for the next read.

    A ``token_usage_record`` covers exactly one response. Older CLIs write
    only ``token_count``, whose ``last_token_usage`` covers the latest response
    and whose running total tells a new response from a repeated event. A
    newer CLI writes each response's record before its count, and a record's
    ``thread_token_usage`` is the running total that count reports. A count at
    or below the highest total that records reached repeats a record. Every
    other count is a response that only a count reports, before the first
    record or after an older CLI resumed the session. A child thread's total
    starts from its parent's, so it is never counted.
    """

    candidates = [
        raw
        for raw in lines
        if len(raw) <= MAX_LINE_BYTES and any(marker in raw for marker in _CODEX_MARKERS)
    ]
    records = [record for record in map(_load, candidates) if record is not None]
    current: dict[str, Any] = dict(state)
    for record in records:
        payload = record.get("payload")
        if (
            record.get("type") == "event_msg"
            and isinstance(payload, Mapping)
            and payload.get("type") == "token_count"
        ):
            limits = payload.get("rate_limits")
            if isinstance(limits, Mapping) and _short(limits.get("plan_type"), 64):
                # Only a ChatGPT sign-in reports a plan.
                current["auth_mode"] = "chatgpt"

    rows: dict[str, UsageRow] = {}
    for record in records:
        kind = record.get("type")
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            continue
        observed = _timestamp(record.get("timestamp"))
        if kind == "session_meta":
            thread = _short(payload.get("id"))
            if thread is not None:
                current["thread"] = thread
            parent = _short(payload.get("parent_thread_id"))
            if parent is not None:
                current["parent"] = parent
            continue
        if kind == "turn_context":
            model = _short(payload.get("model"))
            if model is not None:
                current["model"] = model
            turn = _short(payload.get("turn_id"))
            if turn is not None:
                current["turn"] = turn
            continue
        if kind == "event_msg":
            event = payload.get("type")
            if event == "thread_settings_applied":
                settings = payload.get("thread_settings")
                if isinstance(settings, Mapping):
                    model = _short(settings.get("model"))
                    if model is not None:
                        current["model"] = model
                    if "service_tier" in settings and settings["service_tier"] is None:
                        # No tier means the default one, for example after
                        # Fast mode is turned off.
                        current.pop("service_tier", None)
                    else:
                        tier = _short(settings.get("service_tier"), 32)
                        if tier is not None:
                            current["service_tier"] = tier
                continue
            if event != "token_count":
                continue
            info = payload.get("info")
            if not isinstance(info, Mapping):
                continue
            totals = info.get("total_token_usage")
            last = info.get("last_token_usage")
            total = _count(totals.get("total_tokens")) if isinstance(totals, Mapping) else None
            if total is None or not isinstance(last, Mapping):
                continue
            previous = current.get("last_total")
            current["last_total"] = total
            covered = current.get("covered_total")
            if isinstance(covered, int) and total <= covered:
                continue
            if isinstance(previous, int) and total <= previous:
                # A repeated event, or a total that went back after a reset.
                continue
            thread = current.get("thread") if isinstance(current.get("thread"), str) else session_id
            event_key = _key("codex-token-count", thread, total)
            rows[event_key] = _codex_row(
                event_key,
                "codex_token_count",
                thread,
                last,
                current,
                request_id=None,
                turn_id=None,
                observed=observed,
            )
            continue
        if kind == "token_usage_record":
            usage = payload.get("usage")
            response = _short(payload.get("response_id"))
            if response is None or not isinstance(usage, Mapping):
                continue
            thread = (
                _short(payload.get("thread_id"))
                or (current.get("thread") if isinstance(current.get("thread"), str) else None)
                or session_id
            )
            event_key = _key("codex-response", thread, response)
            thread_usage = payload.get("thread_token_usage")
            reached = (
                _count(thread_usage.get("total_tokens"))
                if isinstance(thread_usage, Mapping)
                else None
            )
            if reached is None and event_key not in rows:
                # A record without the thread's total still covers the count
                # that follows it, which adds this response to the last total.
                known = current.get("covered_total"), current.get("last_total")
                base = max((value for value in known if isinstance(value, int)), default=0)
                reached = base + (_count(usage.get("total_tokens")) or 0)
            if reached is not None:
                covered = current.get("covered_total")
                current["covered_total"] = (
                    max(covered, reached) if isinstance(covered, int) else reached
                )
            rows[event_key] = _codex_row(
                event_key,
                "codex_response",
                thread,
                usage,
                current,
                request_id=response,
                turn_id=_short(payload.get("turn_id")),
                observed=observed,
            )
    return list(rows.values()), current


def price_row(row: UsageRow) -> UsageRow:
    """Return the row with its price at the current rates, or its reason.

    Costing prices a stored request without a cost again when it reads it, so
    a model that gains a price later gains it for these rows too.
    """

    from .pricing import price_event

    price = price_event(
        row.provider,
        {
            "model": row.model,
            "auth_mode": row.auth_mode,
            "service_tier": row.service_tier,
            "input_tokens": row.input_tokens,
            "cached_input_tokens": row.cached_input_tokens,
            "cache_creation_input_tokens": row.cache_creation_input_tokens,
            "cache_creation_1h_input_tokens": row.cache_creation_1h_input_tokens,
            "output_tokens": row.output_tokens,
        },
    )
    if price.amount is None:
        return replace(row, unpriced_reason=price.unpriced_reason)
    return replace(
        row,
        cost_amount=str(price.amount),
        cost_unit=price.unit,
        cost_source=price.source,
        unpriced_reason=None,
    )


def _claude_roots() -> list[Path]:
    roots = [Path.home() / ".claude" / "projects"]
    configured = os.environ.get("CLAUDE_CONFIG_DIR")
    if configured:
        roots.append(Path(configured).expanduser() / "projects")
    return roots


def _codex_roots() -> list[Path]:
    configured = os.environ.get("CODEX_HOME")
    home = Path(configured).expanduser() if configured else Path.home() / ".codex"
    return [home / "sessions"]


def guarded_path(
    raw: Any, harness: str, session_id: str, *, agent_id: str | None = None
) -> Path | None:
    """Return the session file a hook named, or None when it is not one.

    The file must be a regular file under the tool's own session folder, and
    its name must name this session, so a payload cannot point the reader at
    any other file on the machine.
    """

    if not isinstance(session_id, str) or not _SESSION_ID.fullmatch(session_id):
        return None
    if agent_id is not None and not _SESSION_ID.fullmatch(agent_id):
        return None
    text = _short(raw, 4096)
    if text is None:
        return None
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        return None
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    if not resolved.is_file():
        return None
    if harness == "claude-code":
        roots = _claude_roots()
        name_ok = resolved.name == f"{session_id}.jsonl" or (
            agent_id is not None
            and resolved.name == f"agent-{agent_id}.jsonl"
            and resolved.parent.name == "subagents"
            and resolved.parent.parent.name == session_id
        )
    elif harness == "codex":
        roots = _codex_roots()
        name_ok = resolved.name.startswith("rollout-") and (
            resolved.name.endswith(f"-{session_id}.jsonl")
            or (agent_id is not None and resolved.name.endswith(f"-{agent_id}.jsonl"))
        )
    else:
        return None
    if not name_ok:
        return None
    for root in roots:
        try:
            resolved_root = root.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if resolved.is_relative_to(resolved_root):
            return resolved
    return None


def _read_new_lines(
    path: Path, cursor: Mapping[str, Any] | None
) -> tuple[list[bytes], dict[str, Any], bool]:
    """Read the complete lines added since the cursor.

    A file with another identity, or one shorter than the cursor, was replaced,
    so it is read from the start and the stored keys keep that from counting a
    request twice.
    """

    status = path.stat()
    offset = 0
    reset = cursor is not None
    if (
        cursor is not None
        and cursor.get("device") == status.st_dev
        and cursor.get("inode") == status.st_ino
        and isinstance(cursor.get("offset"), int)
        and 0 <= cursor["offset"] <= status.st_size
    ):
        offset = cursor["offset"]
        reset = False
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(MAX_READ_BYTES)
    end = data.rfind(b"\n")
    if end >= 0:
        consumed = end + 1
    elif len(data) >= MAX_READ_BYTES:
        # A single line longer than a whole read cannot hold usage worth
        # keeping, so the reader moves past it instead of stopping forever.
        consumed = len(data)
    else:
        consumed = 0
    lines = data[:consumed].splitlines(keepends=True)
    next_cursor = {
        "device": status.st_dev,
        "inode": status.st_ino,
        "offset": offset + consumed,
    }
    return lines, next_cursor, reset


def fallback_enabled(repo: Path) -> bool:
    """Return whether this checkout lets the fallback read session files."""

    from .store import read_install_state

    try:
        state = read_install_state(repo)
    except (OSError, ValueError):
        return False
    if state is None or state.get("usage_fallback") is False:
        return False
    # Either opt-out wins. A repository install's hooks defer to the machine
    # hooks, so the machine's choice governs a clone with either install.
    from .user_install import load_user_manifest

    try:
        machine = load_user_manifest()
    except (OSError, ValueError):
        machine = None
    return not (isinstance(machine, dict) and machine.get("usage_fallback") is False)


def record_from_hook(
    repo: Path | str,
    payload: Mapping[str, Any],
    harness: str,
    event: str,
    *,
    state_dir: str | os.PathLike[str] | None = None,
) -> int:
    """Store the usage a hook's session file holds, and return the row count.

    Stop can run before the final message reaches the file, so each later hook
    event reads what was added and replaces a streamed copy that grew.
    """

    if event not in HOOK_EVENTS or harness not in {"claude-code", "codex"}:
        return 0
    root = Path(repo)
    if not fallback_enabled(root):
        return 0
    session_id = _short(payload.get("session_id"))
    if session_id is None:
        return 0
    agent_id = _short(payload.get("agent_id")) if event == "SubagentStop" else None
    raw_path = (
        payload.get("agent_transcript_path")
        if event == "SubagentStop"
        else payload.get("transcript_path")
    )
    path = guarded_path(raw_path, harness, session_id, agent_id=agent_id)
    if path is None:
        return 0

    from .store import git_common_dir
    from .telemetry import TelemetryStore

    store = TelemetryStore(state_dir)
    file_key = _key("usage-fallback-file", str(path))
    cursor = store.fallback_cursor(file_key)
    lines, next_cursor, reset = _read_new_lines(path, cursor)
    state: dict[str, Any] = {}
    if cursor is not None and not reset and isinstance(cursor.get("state"), dict):
        state = dict(cursor["state"])
    if harness == "claude-code":
        rows = parse_claude_lines(lines, session_id=session_id, agent_id=agent_id)
    else:
        if agent_id is not None:
            state.setdefault("agent", agent_id)
        known_auth = state.get("auth_mode")
        rows, state = parse_codex_lines(lines, session_id=session_id, state=state)
        auth_mode = state.get("auth_mode")
        if isinstance(auth_mode, str) and auth_mode != known_auth:
            # An earlier read stored this thread's responses before a count
            # named the sign-in. They take it now, under their own keys, so
            # the billing unit never depends on where a read stopped.
            thread = state.get("thread")
            native, agent, _source = _codex_identity(
                state, thread if isinstance(thread, str) else session_id
            )
            earlier = [
                replace(UsageRow(**row), auth_mode=auth_mode)
                for row in store.fallback_rows_without_auth("codex", native, agent)
            ]
            rows = earlier + rows
        next_cursor["state"] = state
    return store.record_fallback_usage(
        [price_row(row) for row in rows],
        repository_id=str(git_common_dir(root)),
        repository_path=str(root),
        file_key=file_key,
        cursor=next_cursor,
    )


__all__ = [
    "FALLBACK_EVENT_NAME",
    "HOOK_EVENTS",
    "MAIN_THREAD_SOURCE",
    "TRANSCRIPT_USAGE_SOURCE",
    "UsageRow",
    "fallback_enabled",
    "guarded_path",
    "parse_claude_lines",
    "parse_codex_lines",
    "price_row",
    "record_from_hook",
]
