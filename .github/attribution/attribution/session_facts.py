"""Reduce one session's ledger rows to the facts a pull request's insights read.

A note records which lines a session wrote. The ledger of the clone that ran
the session also holds how the work went: its prompts, its tool calls and their
outcomes, every edit it made, and when each of them happened. This module
reduces those rows to counts and to times relative to the session start, so a
pre-push snapshot can publish them beside the notes. A record carries no
locator, hash, path, command, or text.
"""

from __future__ import annotations

from collections import Counter
import math
import sqlite3
import time
from typing import Any, Iterable, Mapping

from .costing import _effort
from .notes import _lines
from .pr_report import (
    INSIGHT_REPORTED_COUNTS, INSIGHT_TOOL_CLASSES, MAX_INSIGHT_MODELS,
    MAX_INSIGHT_SECONDS, MAX_INSIGHT_TIMELINE_ITEMS, insight_label,
    insight_reported_count,
)
from .report import _parse_time


_RETRY_LOOP_CALLS = 3
_REPEATED_READ_AFTER = 2
# A session can hold more events than one timeline shows. The rarer kinds are
# kept first, because one commit says more about the work than one more failed
# call does.
_EVENT_PRIORITY = (
    "commit", "retry_loop", "compaction", "interrupt", "prompt", "tool_failure",
)


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _whole(value: Any) -> int | None:
    """Return an allocated count, a share of shared requests, as a whole number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(round(value)) if math.isfinite(value) and value >= 0 else None


def retry_loops(calls: Iterable[Mapping[str, Any]]) -> list[Any]:
    """Return when each retry loop in a session's ordered tool calls began.

    A loop is a run of three or more consecutive calls that all failed and all
    name the same locator hash: the agent tried one thing again and again. Any
    other call between two of them ends the run.
    """
    starts: list[Any] = []
    target, began, length = None, None, 0
    for call in calls:
        locator = call["locator_hash"]
        if call["succeeded"] == 0 and locator is not None and locator == target:
            length += 1
            continue
        if length >= _RETRY_LOOP_CALLS:
            starts.append(began)
        if call["succeeded"] == 0 and locator is not None:
            target, began, length = locator, call["occurred_at"], 1
        else:
            target, began, length = None, None, 0
    if length >= _RETRY_LOOP_CALLS:
        starts.append(began)
    return starts


def repeated_reads(calls: Iterable[Mapping[str, Any]]) -> int:
    """Count the reads of a locator that the session had already read twice."""
    seen: Counter[str] = Counter()
    repeated = 0
    for call in calls:
        locator = call["locator_hash"]
        if call["tool_class"] != "read" or locator is None:
            continue
        if seen[locator] >= _REPEATED_READ_AFTER:
            repeated += 1
        seen[locator] += 1
    return repeated


def generated_lines(
    connection: sqlite3.Connection, session_id: str, deadline: float | None = None,
) -> int | None:
    """Count the lines the session's captured edits added, kept or not.

    Each edit is one before and after content of one file. The lines it added
    are the lines of the after content beyond the copies of each that the
    before content already held, a difference of two line multisets that takes
    linear time: a push runs this for many sessions and must stay fast. A line
    that only moved is not new. Past ``deadline``, a ``time.monotonic`` value,
    the count is unknown rather than short.
    """
    total = 0
    for before, after in connection.execute(
        "SELECT before_content, after_content FROM edits WHERE session_id = ? ORDER BY id",
        (session_id,),
    ):
        if deadline is not None and time.monotonic() > deadline:
            return None
        added = Counter(_lines(None if after is None else bytes(after)))
        added.subtract(_lines(None if before is None else bytes(before)))
        total += sum(count for count in added.values() if count > 0)
    return total


def _edits_complete(connection: sqlite3.Connection, session_id: str) -> bool:
    """Say whether the session's edit rows hold every change it made.

    A capture that ended any way but completed kept its edits unattributed,
    and a file skipped for its size or its readability left no edit row. A
    binary file has no lines to miss, and an ignored file is not in the tree.
    """
    unfinished = connection.execute(
        "SELECT 1 FROM hook_captures WHERE ledger_session_id = ? AND status != 'completed' LIMIT 1",
        (session_id,),
    ).fetchone()
    skipped = connection.execute(
        """
        SELECT skip_reason FROM hook_capture_files
        JOIN hook_captures ON hook_captures.id = hook_capture_files.capture_id
        WHERE hook_captures.ledger_session_id = ? AND skip_reason IS NOT NULL
        UNION ALL
        SELECT skip_reason FROM hook_snapshot_files
        JOIN hook_units ON hook_units.unit_key = hook_snapshot_files.unit_key
        WHERE hook_units.attribution_session_id = ? AND skip_reason IS NOT NULL
        """,
        (session_id, session_id),
    ).fetchall()
    return unfinished is None and all(
        row[0] == "binary" or str(row[0]).startswith("ignored") for row in skipped
    )


def _by_model(entries: Any) -> list[dict[str, Any]] | None:
    """Publish one session's cost and tokens for each model and effort.

    A model name that cannot be published drops the whole list, as it drops
    the usage record it came from, rather than hiding part of the cost.
    """
    if not isinstance(entries, list) or not entries or len(entries) > MAX_INSIGHT_MODELS:
        # More models and efforts than the list holds would split only part of
        # the cost, so no split is published rather than a short one.
        return None
    published = []
    for entry in entries:
        model = entry.get("model")
        if model is not None and not insight_label(model):
            return None
        published.append({
            "model": model,
            "effort": entry.get("effort"),
            "requests": _whole(entry.get("request_count")),
            "estimated_usd": entry.get("estimated_cost_usd"),
            "codex_credits": entry.get("codex_credits"),
            "codex_api_equivalent_usd": entry.get("codex_api_equivalent_usd"),
            "tokens": {
                "input": _whole(entry.get("input_tokens")),
                "cached_input": _whole(entry.get("cached_input_tokens")),
                "cache_creation_input": _whole(entry.get("cache_creation_input_tokens")),
                "output": _whole(entry.get("output_tokens")),
            },
        })
    published.sort(key=lambda item: (
        -(item["estimated_usd"] or item["codex_api_equivalent_usd"] or 0.0),
        -(item["requests"] or 0), item["model"] or "", item["effort"] or "",
    ))
    return published


def _downsample(items: list[Any]) -> list[Any]:
    """Keep the first, the last, and evenly spaced items between them."""
    if len(items) <= MAX_INSIGHT_TIMELINE_ITEMS:
        return items
    last = len(items) - 1
    return [
        items[round(index * last / (MAX_INSIGHT_TIMELINE_ITEMS - 1))]
        for index in range(MAX_INSIGHT_TIMELINE_ITEMS)
    ]


def timeline(
    start: float,
    priced: Iterable[tuple[int | None, float]],
    moments: Iterable[tuple[float | None, str]],
) -> dict[str, Any] | None:
    """Return the cumulative estimated dollars and the events of one session.

    ``priced`` holds the observed time in nanoseconds and the estimated dollars
    of each priced request, and ``moments`` the time in seconds since the epoch
    and the kind of each event. Times are whole seconds from ``start``, the
    session start in the same unit. Something that happened
    before the ledger opened the session, such as the first request of the
    prompt that opened it, is placed at the start, and a request with no
    observed time is left out of the curve.
    """

    def offset(moment: float | None) -> int | None:
        if moment is None:
            return None
        seconds = max(math.floor(moment - start), 0)
        return seconds if seconds <= MAX_INSIGHT_SECONDS else None

    cumulative, by_second = 0.0, {}
    for observed, cost in sorted(item for item in priced if item[0] is not None):
        second = offset(observed / 1e9)
        if second is None:
            continue
        cumulative += cost
        by_second[second] = cumulative
    points = [[second, round(total, 6)] for second, total in sorted(by_second.items())]
    if points and points[0][0] > 0:
        points.insert(0, [0, 0.0])

    events = set()
    for moment, kind in moments:
        second = offset(moment)
        if second is not None:
            events.add((second, kind))
    if len(events) > MAX_INSIGHT_TIMELINE_ITEMS:
        events = set(sorted(
            events, key=lambda item: (_EVENT_PRIORITY.index(item[1]), item[0])
        )[:MAX_INSIGHT_TIMELINE_ITEMS])
    ordered = sorted(events, key=lambda item: (item[0], _EVENT_PRIORITY.index(item[1])))
    if not points and not ordered:
        return None
    return {
        "points": _downsample(points),
        "events": [{"t": second, "kind": kind} for second, kind in ordered],
    }


def _seconds(value: Any) -> float | None:
    parsed = _parse_time(value)
    return None if parsed is None else parsed.timestamp()


def session_facts(
    connection: sqlite3.Connection,
    session_id: str,
    usage: Mapping[str, Any] | None = None,
    commit_times: Iterable[float] = (),
    deadline: float | None = None,
) -> dict[str, Any] | None:
    """Return one session's local facts, or None where this ledger has no such session.

    ``usage`` is the session's allocated telemetry with its detail,
    ``commit_times`` holds the time in seconds since the epoch of each commit
    whose note names the session, and ``deadline`` bounds the time spent on its
    edits. A count the ledger cannot vouch for is None, never a zero: the tool
    facts need every tool call of the session, the failures need every outcome,
    and a count of a kind the capture never reports stays unknown.
    """
    session = connection.execute(
        """
        SELECT started_at, effort_level, prompt_count, interrupt_count,
               compaction_count, activity_truncated, integration_mode,
               COALESCE(harness_id, harness) AS harness
        FROM sessions WHERE id = ?
        """,
        (session_id,),
    ).fetchone()
    if session is None:
        return None
    calls = connection.execute(
        """
        SELECT tool_class, locator_hash, succeeded, occurred_at
        FROM tool_calls WHERE session_id = ? ORDER BY sequence, id
        """,
        (session_id,),
    ).fetchall()
    loops = retry_loops(calls)
    native = session["integration_mode"] == "native_hook"
    harness = session["harness"] if native else None
    record: dict[str, Any] = {
        "effort": _effort(session["effort_level"]),
        **{
            field: insight_reported_count(session[f"{field[:-1]}_count"], field, harness)
            for field in INSIGHT_REPORTED_COUNTS
        },
        "tool_calls": None,
        "failed_tool_calls": None,
        "retry_loops": None,
        "repeated_reads": None,
        # A wrapper keeps no record of the files it skipped.
        "generated_lines": (
            generated_lines(connection, session_id, deadline)
            if native and _edits_complete(connection, session_id) else None
        ),
    }
    if calls and not session["activity_truncated"]:
        classes = Counter(call["tool_class"] for call in calls)
        record["tool_calls"] = {name: classes[name] for name in INSIGHT_TOOL_CLASSES}
        record["repeated_reads"] = repeated_reads(calls)
        # A call whose outcome no event reported may have failed.
        if all(call["succeeded"] is not None for call in calls):
            record["failed_tool_calls"] = sum(call["succeeded"] == 0 for call in calls)
            record["retry_loops"] = len(loops)
    usage = usage if isinstance(usage, Mapping) else {}
    by_model = _by_model(usage.get("by_model"))
    if by_model:
        record["by_model"] = by_model
    start = _seconds(session["started_at"])
    if start is not None:
        moments: list[tuple[float | None, str]] = [
            (_seconds(call["occurred_at"]), "tool_failure")
            for call in calls if call["succeeded"] == 0
        ]
        moments.extend((_seconds(moment), "retry_loop") for moment in loops)
        moments.extend(
            (
                _seconds(row["occurred_at"]),
                "prompt" if row["kind"] == "user_prompt" else "compaction",
            )
            for row in connection.execute(
                """
                SELECT kind, occurred_at FROM context_loads
                WHERE session_id = ? AND kind IN ('user_prompt', 'compaction_summary')
                """,
                (session_id,),
            )
        )
        moments.extend((moment, "commit") for moment in commit_times)
        drawn = timeline(start, usage.get("priced_requests") or (), moments)
        if drawn is not None:
            record["timeline"] = drawn
    return record


__all__ = ["generated_lines", "repeated_reads", "retry_loops", "session_facts", "timeline"]
