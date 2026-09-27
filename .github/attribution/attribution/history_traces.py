"""The full trace and the session facts of one imported session.

``history_import.discover`` groups the records of each Claude Code transcript
and Codex rollout by native session and feeds each group to a ``Session``.
``Session.finish`` returns the session's trace, in the shape that the capture
hooks publish and ``traces.validate_trace`` accepts, and its facts, in the
shape of one entry of a report's ``summary.insights.sessions``.

The trace keeps every prompt, reply, tool call with its input, tool result
with its output and outcome, compaction, and subagent result, within the caps
of ``attribution/traces.py``, so an imported trace is the object a captured
one is. Nothing is redacted. NUL characters and lone surrogates are removed,
because the hosted JSON columns cannot hold them. The parsing follows
``scripts/history_replay.py``, which reads the same files for calibration.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import json
import math
import re
from typing import Any, Iterable, Mapping

from . import activity, traces, usage_fallback
from .hook_events import response_success
from .pr_report import INSIGHT_TOOL_CLASSES
from .session_facts import repeated_reads, retry_loops


# Text that a harness pasted into the conversation, not a prompt the developer wrote.
_INJECTED = re.compile(r"\s*(?:<(?!image\b)[a-z][a-z0-9_-]*[\s>/]|# AGENTS\.md instructions)")
_CONTINUED = "This session is being continued from a previous conversation"
_INTERRUPTED = "[Request interrupted by user"
# Codex tool names that the activity table does not know.
_CODEX_CLASSES = {
    "exec_command": "shell", "shell_command": "shell", "local_shell": "shell", "container.exec": "shell",
    "spawn_agent": "agent", "view_image": "read", "web_search": "web",
}
_EXIT_CODE = re.compile(r'"exit_code"\s*:\s*(-?\d+)')
_FAILED = ("failed", "declined", "error", "cancelled")
_TOOL_ITEMS = frozenset({"CommandExecution", "FileChange", "McpToolCall", "WebSearch", "Extension", "ImageView"})
# The bytes that validate_trace allows each short field of an event.
_FIELD_BYTES = {"tool_use_id": 2048, "tool_name": 512, "summary": 256}
# validate_trace allows a tool input this much over its cap, for the second
# encoding of an input that was cut to a text.
_INPUT_SLACK = 4096


def _iso(value: Any) -> str | None:
    """A time as ``YYYY-MM-DDTHH:MM:SS.mmmZ``, from ISO text or epoch milliseconds."""
    if isinstance(value, str) and len(value) == 24 and value[-1] == "Z" and value[19] == ".":
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            moment = datetime.fromtimestamp(value / 1000, timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    elif isinstance(value, str) and value:
        try:
            moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        moment = moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
    else:
        return None
    try:
        return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    except (OverflowError, ValueError):
        return None


def _moment(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def _clean(value: Any) -> Any:
    """The value without NUL characters, lone surrogates, or numbers that JSON cannot hold."""
    if isinstance(value, str):
        value = value.replace("\x00", "")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            value = value.encode("utf-8", "replace").decode("utf-8")
        return value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {_clean(str(key)): _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def _argv_text(argv: list[Any]) -> str:
    parts = [str(part) for part in argv]
    return parts[-1] if len(parts) >= 3 and parts[-2] in ("-lc", "-c") else " ".join(parts)


def _block_text(content: Any, kinds: tuple[str, ...] = ("text", "input_text", "output_text")) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(block.get("text") or "" for block in content
                         if isinstance(block, dict) and block.get("type") in kinds and isinstance(block.get("text"), str))
    return ""


class Session:
    """One imported session being read: its events in trace shape, its calls, its usage, and its counts."""

    def __init__(self, harness: str, session_id: str, parent: str | None = None) -> None:
        self.harness, self.id, self.parent = harness, session_id, parent
        self.events: list[dict[str, Any]] = []
        self.calls: dict[str, dict[str, Any]] = {}
        self.prompts = self.interrupts = 0
        # event key -> (usage row, effort of the request)
        self.usage: dict[str, tuple[usage_fallback.UsageRow, str | None]] = {}
        self.cwd: str | None = None
        self.effort: str | None = None
        self.model: str | None = None
        self.first: str | None = None
        self.last: str | None = None
        # A resumed or forked Claude Code transcript copies earlier records with their uuids.
        self._uuids: set[str] = set()

    def seen(self, at: str | None) -> None:
        if at:
            self.first = at if self.first is None or at < self.first else self.first
            self.last = at if self.last is None or at > self.last else self.last

    def _add(self, kind: str, at: str | None, **fields: Any) -> dict[str, Any]:
        capped, cut = traces._capped_fields(kind, fields)
        capped = _clean(capped)
        for name, limit in _FIELD_BYTES.items():
            if isinstance(capped.get(name), str):
                capped[name] = traces._head(capped[name], limit)
        if kind == "tool_call":
            # A cut input is one text, and its second encoding escapes every quote again.
            while len(traces.canonical_bytes(capped["input"])) > traces.MAX_INPUT_BYTES + _INPUT_SLACK:
                capped["input"]["text"] = capped["input"]["text"][: len(capped["input"]["text"]) * 3 // 4]
        event = {"at": at or "", "kind": kind, **capped}
        if cut:
            event["truncated"] = True
        self.events.append(event)
        self.seen(at)
        return event

    def prompt(self, at: str | None, text: str) -> None:
        self.prompts += 1
        self._add("user_prompt", at, text=text)

    def reply(self, at: str | None, text: str) -> None:
        if text.strip():
            self._add("assistant", at, text=text)

    def compaction(self, at: str | None, text: str, *, merge: bool = False) -> None:
        """Add a compaction, or with ``merge`` give the latest one its summary text."""
        last = next((event for event in reversed(self.events) if event["kind"] == "compaction"), None)
        if merge and last is not None:
            if text and not last.get("text"):
                capped, cut = traces._capped_fields("compaction", {"text": text})
                last.update(_clean(capped), **({"truncated": True} if cut else {}))
            return
        self._add("compaction", at, text=text)

    def subagent_result(self, at: str | None, text: str) -> None:
        self._add("subagent_result", at, text=text)

    def call(self, at: str | None, tool_use_id: str, name: str, tool_class: str, data: Mapping[str, Any]) -> None:
        locator, locator_hash = activity.extract_locator(self.harness, name, tool_class, data, self.cwd)
        event = self._add("tool_call", at, tool_name=name, tool_class=tool_class, summary=locator, input=data,
                          tool_use_id=tool_use_id)
        self.calls[tool_use_id] = {"event": event, "hash": locator_hash, "result": None}

    def result(self, at: str | None, tool_use_id: str, succeeded: bool | None, output: Any) -> None:
        event = self._add("tool_result", at, succeeded=succeeded, output=traces._stringify(output),
                          tool_use_id=tool_use_id)
        call = self.calls.get(tool_use_id)
        if call is not None:
            call["result"] = event

    def fail(self, tool_use_id: str) -> None:
        """Mark a result failed when a later record says the call failed."""
        result = (self.calls.get(tool_use_id) or {}).get("result")
        if result is not None:
            result["succeeded"] = False

    def usage_row(self, row: usage_fallback.UsageRow, effort: str | None) -> None:
        known = self.usage.get(row.event_key)
        if known is None or (row.output_tokens or 0) > (known[0].output_tokens or 0):
            self.usage[row.event_key] = (row, effort)

    # Claude Code -------------------------------------------------------------

    def read_claude(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Read Claude Code records of this session in file order. A record read before adds only its usage."""
        for record in records:
            kind = record.get("type")
            if kind not in ("user", "assistant", "attachment", "system"):
                continue
            at = _iso(record.get("timestamp"))
            self.seen(at)
            if isinstance(record.get("cwd"), str) and record["cwd"] and self.cwd is None:
                self.cwd = record["cwd"]
            uuid = record.get("uuid") if isinstance(record.get("uuid"), str) else None
            repeat = uuid in self._uuids
            if uuid is not None:
                self._uuids.add(uuid)
            if kind == "assistant":
                self._claude_assistant(record, at, repeat)
            elif repeat:
                continue
            elif kind == "user":
                self._claude_user(record, at)
            elif kind == "attachment":
                attachment = record.get("attachment") if isinstance(record.get("attachment"), dict) else {}
                origin = attachment.get("origin") if isinstance(attachment.get("origin"), dict) else {}
                if (attachment.get("type") == "queued_command" and attachment.get("commandMode") == "prompt"
                        and origin.get("kind") == "human" and isinstance(attachment.get("prompt"), str)):
                    self.prompt(at, attachment["prompt"])
            elif record.get("subtype") == "compact_boundary":
                self.compaction(at, "")

    def _claude_assistant(self, record: Mapping[str, Any], at: str | None, repeat: bool) -> None:
        message = record.get("message") if isinstance(record.get("message"), dict) else {}
        if message.get("model") == "<synthetic>":
            # A reply that Claude Code wrote itself, such as an API error, with no request behind it.
            return
        effort = record.get("effort") if isinstance(record.get("effort"), str) else None
        if isinstance(message.get("usage"), dict):
            line = json.dumps({
                "type": "assistant", "requestId": record.get("requestId"), "sessionId": record.get("sessionId"),
                "isSidechain": record.get("isSidechain"), "agentId": record.get("agentId"),
                "timestamp": record.get("timestamp"),
                "message": {"id": message.get("id"), "model": message.get("model"), "usage": message["usage"]},
            }).encode()
            for row in usage_fallback.parse_claude_lines([line], session_id=self.id):
                self.usage_row(row, effort)
        if repeat:
            return
        for block in message.get("content") if isinstance(message.get("content"), list) else []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                self.reply(at, block["text"])
            elif block.get("type") == "tool_use" and isinstance(block.get("id"), str):
                name = str(block.get("name") or "")
                data = block.get("input") if isinstance(block.get("input"), dict) else {"value": block.get("input")}
                self.call(at, block["id"], name, activity.classify_tool("claude-code", name), data)

    def _claude_user(self, record: Mapping[str, Any], at: str | None) -> None:
        message = record.get("message") if isinstance(record.get("message"), dict) else {}
        content = message.get("content")
        blocks = content if isinstance(content, list) else []
        results = [block for block in blocks if isinstance(block, dict) and block.get("type") == "tool_result"]
        text = content if isinstance(content, str) else _block_text(blocks, ("text",))
        if record.get("isCompactSummary") or record.get("compactMetadata") or _CONTINUED in text[:300]:
            # A compact_boundary record just before it marked the same compaction.
            last = self.events[-1] if self.events else None
            self.compaction(at, text, merge=bool(last and last["kind"] == "compaction" and not last.get("text")))
            return
        if text.lstrip().startswith(_INTERRUPTED):
            self.interrupts += 1
        for block in results:
            self._claude_result(record, block, at, single=len(results) == 1)
        if results or record.get("isMeta") or not text.strip() or text.lstrip().startswith(_INTERRUPTED):
            return
        if not _INJECTED.match(text):
            self.prompt(at, text)

    def _claude_result(self, record: Mapping[str, Any], block: Mapping[str, Any], at: str | None,
                       *, single: bool) -> None:
        tool_use_id = block.get("tool_use_id")
        if not isinstance(tool_use_id, str):
            return
        text = _block_text(block.get("content"), ("text",))
        structured = record.get("toolUseResult")
        if block.get("is_error") is True:
            # Claude's failure hook carries the error text, and the capture keeps it as {"error": ...}.
            output: Any = {"error": text}
            if isinstance(record.get("toolDenialKind"), str):
                output["tool_denial_kind"] = record["toolDenialKind"]
        elif single and isinstance(structured, (dict, list)):
            output = structured
        else:
            output = text
        call = self.calls.get(tool_use_id)
        if call is not None and call["event"]["tool_class"] == "agent":
            self.subagent_result(at, text)
        self.result(at, tool_use_id, block.get("is_error") is not True, output)

    # Codex ---------------------------------------------------------------------

    def read_codex(self, records: list[Mapping[str, Any]], lines: list[bytes]) -> None:
        """Read one rollout: this session's own thread, without the history a fork copied from its parent."""
        started = copying = False
        start_ordinal = copied_at = None
        usage_lines: list[bytes] = []
        turn = None
        turn_effort: dict[str, str] = {}
        function_ids: set[str] = set()
        item_status: dict[str, str] = {}
        wrappers: dict[str, dict[str, Any]] = {}
        sources: dict[str, Counter] = {"event": Counter(), "item": Counter(), "response": Counter()}
        emitted: Counter = Counter()
        last_marker: datetime | None = None

        def prompt(source: str, at: str | None, text: str) -> None:
            # The same prompt appears as a response item, an item_completed event, and a user_message event.
            text = text.strip()
            if not text or _INJECTED.match(text):
                return
            sources[source][text] += 1
            if sources[source][text] > emitted[text]:
                emitted[text] += 1
                self.prompt(at, text)

        def compaction(at: str | None, text: str) -> None:
            # One compaction writes up to four markers within a minute or two.
            nonlocal last_marker
            moment = _moment(at)
            near = last_marker is not None and moment is not None and abs((moment - last_marker).total_seconds()) <= 120
            self.compaction(at, text, merge=near)
            last_marker = moment or last_marker

        for raw, record in zip(lines, records):
            kind = record.get("type")
            payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
            at = _iso(record.get("timestamp"))
            if kind == "session_meta":
                if not started:
                    started = True
                    if isinstance(payload.get("cwd"), str):
                        self.cwd = payload["cwd"]
                    ordinal = payload.get("subagent_history_start_ordinal")
                    start_ordinal = ordinal if isinstance(ordinal, int) else None
                    usage_lines.append(json.dumps({"type": "session_meta", "payload": {
                        "id": self.id, "parent_thread_id": self.parent}}).encode())
                    self.seen(at)
                elif payload.get("id") != self.id:
                    # A fork copies its parent's meta and history, all stamped with the fork time.
                    copying, copied_at = True, at
                continue
            if not started:
                continue
            ordinal = record.get("ordinal")
            if start_ordinal is not None and isinstance(ordinal, int) and ordinal < start_ordinal:
                continue
            if copying:
                if payload.get("thread_id") == self.id or (at is not None and at != copied_at):
                    copying = False
                else:
                    continue
            self.seen(at)
            if kind == "turn_context":
                turn = payload.get("turn_id") if isinstance(payload.get("turn_id"), str) else turn
                if isinstance(payload.get("model"), str):
                    self.model = payload["model"]
                if isinstance(payload.get("effort"), str):
                    self.effort = payload["effort"]
                    if turn:
                        turn_effort[turn] = payload["effort"]
                if isinstance(payload.get("cwd"), str) and self.cwd is None:
                    self.cwd = payload["cwd"]
                usage_lines.append(json.dumps({"type": "turn_context", "payload": {
                    "model": payload.get("model"), "turn_id": payload.get("turn_id")}}).encode())
            elif kind == "token_usage_record":
                usage_lines.append(raw)
            elif kind == "compacted":
                compaction(at, payload.get("message") if isinstance(payload.get("message"), str) else "")
            elif kind == "event_msg":
                event = payload.get("type")
                if event == "token_count":
                    usage_lines.append(raw)
                elif event == "thread_settings_applied":
                    settings = payload.get("thread_settings") if isinstance(payload.get("thread_settings"), dict) else {}
                    kept = {key: settings[key] for key in ("model", "service_tier") if key in settings}
                    usage_lines.append(json.dumps({"type": "event_msg", "payload": {
                        "type": event, "thread_settings": kept}}).encode())
                elif event == "turn_aborted" and payload.get("reason") == "interrupted":
                    self.interrupts += 1
                elif event == "context_compacted":
                    compaction(at, "")
                elif event == "user_message" and isinstance(payload.get("message"), str):
                    prompt("event", at, payload["message"])
                elif event == "item_completed" and isinstance(payload.get("item"), dict):
                    item = payload["item"]
                    item_type, item_id = item.get("type"), str(item.get("id"))
                    if item_type == "UserMessage":
                        prompt("item", at, _block_text(item.get("content")))
                    elif item_type == "ContextCompaction":
                        compaction(at, "")
                    elif item_id in function_ids:
                        item_status[item_id] = str(item.get("status"))
                        if item.get("status") == "failed":
                            self.fail(item_id)
                    elif item_type in _TOOL_ITEMS:
                        if wrappers and item_id.startswith("exec-"):
                            wrappers[next(reversed(wrappers))]["nested"] += 1
                        _codex_item(self, item, _iso(payload.get("started_at_ms")) or at,
                                    _iso(payload.get("completed_at_ms")) or at)
            elif kind == "response_item":
                item_type = payload.get("type")
                if item_type == "message" and payload.get("role") == "user":
                    prompt("response", at, _block_text(payload.get("content")))
                elif item_type == "message" and payload.get("role") == "assistant":
                    self.reply(at, _block_text(payload.get("content")))
                elif item_type == "compaction":
                    compaction(at, "")
                elif item_type in ("function_call", "custom_tool_call", "local_shell_call"):
                    call_id = str(payload.get("call_id") or payload.get("id"))
                    name = str(payload.get("name") or ("local_shell" if item_type == "local_shell_call" else "tool"))
                    if item_type == "custom_tool_call" and name == "exec":
                        # Code mode: the JavaScript's own tool calls arrive as item_completed events.
                        wrappers[call_id] = {"at": at, "input": payload.get("input"), "nested": 0}
                        continue
                    namespace = payload.get("namespace")
                    tool_name = f"{namespace}__{name}" if isinstance(namespace, str) and namespace.startswith("mcp__") else name
                    if item_type == "local_shell_call":
                        action = payload.get("action") if isinstance(payload.get("action"), dict) else {}
                        data = _codex_input(name, {"command": action.get("command")})
                    else:
                        data = _codex_input(name, payload.get("arguments") if item_type == "function_call" else payload.get("input"))
                    self.call(at, call_id, tool_name, _CODEX_CLASSES.get(name) or activity.classify_tool("codex", tool_name), data)
                    function_ids.add(call_id)
                elif item_type in ("function_call_output", "custom_tool_call_output"):
                    call_id = str(payload.get("call_id"))
                    output = payload.get("output")
                    text = _block_text(output) if isinstance(output, list) else traces._stringify(output)
                    wrapper = wrappers.pop(call_id, None)
                    if wrapper is not None:
                        if wrapper["nested"] == 0:
                            self.call(wrapper["at"], call_id, "exec", "other", {"input": wrapper["input"]})
                            self.result(at, call_id, _codex_outcome(text), text)
                    elif call_id in self.calls:
                        self.result(at, call_id, _codex_outcome(text, item_status.get(call_id)), text)
        rows, _ = usage_fallback.parse_codex_lines(usage_lines, session_id=self.id, state={})
        for row in rows:
            if (row.agent_id or row.native_session_id) == self.id:
                self.usage_row(row, turn_effort.get(row.turn_id or "") or self.effort)

    # The result ------------------------------------------------------------------

    def finish(self) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Return the session's trace, or None when it holds no event, and its facts."""
        self.events.sort(key=lambda event: event["at"])
        turn = 0
        for seq, event in enumerate(self.events, 1):
            turn += event["kind"] == "user_prompt"
            event["seq"], event["turn"] = seq, turn
        facts = self._facts()
        return self._trace(facts["model"]), facts

    def _facts(self) -> dict[str, Any]:
        """The session as ``summary.insights.sessions`` carries it, from every event and request."""
        calls = sorted(self.calls.values(), key=lambda call: call["event"]["seq"])
        rows = [{
            "tool_class": call["event"]["tool_class"], "locator_hash": call["hash"], "occurred_at": call["event"]["at"],
            "succeeded": None if call["result"] is None or call["result"]["succeeded"] is None
            else int(call["result"]["succeeded"]),
        } for call in calls]
        classes = Counter(row["tool_class"] for row in rows)
        counts = {name: classes[name] for name in INSIGHT_TOOL_CLASSES}
        called = sum(counts.values()) > 0
        known = all(row["succeeded"] is not None for row in rows)
        codex = self.harness == "codex"
        priced, models, efforts = [], Counter(), Counter()
        for row, effort in self.usage.values():
            price = usage_fallback.price_row(replace(row, auth_mode=None) if row.provider == "codex" else row)
            dollars = (float(Decimal(price.cost_amount))
                       if price.cost_amount is not None and price.cost_unit == "USD" else None)
            priced.append((row, effort, dollars))
            models[row.model] += 1
            efforts[effort] += 1
        cost = math.fsum(item[2] for item in priced) if priced and all(item[2] is not None for item in priced) else None
        groups: dict[tuple[Any, Any], list[tuple[Any, Any, Any]]] = {}
        for item in priced:
            groups.setdefault((item[0].model, item[1]), []).append(item)
        by_model = []
        for (model, effort), members in groups.items():
            dollars = None if any(item[2] is None for item in members) else math.fsum(item[2] for item in members)
            by_model.append({
                "model": model, "effort": effort, "requests": len(members),
                "estimated_usd": None if codex else dollars,
                **({"codex_api_equivalent_usd": dollars} if codex else {}),
                "tokens": _tokens(members),
            })
        return _clean({
            "id": self.id, "actor_kind": "ai", "harness": self.harness,
            "model": (models.most_common(1)[0][0] if models else None) or self.model,
            "effort": (efforts.most_common(1)[0][0] if efforts else None) or self.effort,
            "started_at": self.first, "ended_at": self.last, "parent_session_id": self.parent,
            "cost": {"reported_usd": None, "estimated_usd": None if codex else cost, "codex_credits": None,
                     "codex_api_equivalent_usd": cost if codex else None},
            "tokens": _tokens(priced) if priced else None,
            "requests": len(priced) if priced else None,
            "by_model": by_model,
            "prompts": self.prompts, "interrupts": self.interrupts,
            "compactions": sum(event["kind"] == "compaction" for event in self.events),
            # A session without tool calls reports no tool facts, as session_facts does.
            "tool_calls": counts if called else None,
            "failed_tool_calls": sum(row["succeeded"] == 0 for row in rows) if called and known else None,
            "retry_loops": len(retry_loops(rows)) if called and known else None,
            "repeated_reads": repeated_reads(rows) if called else None,
            "generated_lines": None,
        })

    def _trace(self, model: str | None) -> dict[str, Any] | None:
        """The session's trace as build_trace publishes it, within the ledger's row and byte caps."""
        events, size, truncated = [], 0, False
        for event in self.events:
            payload = {key: value for key, value in event.items()
                       if key not in ("seq", "at", "kind", "turn", "tool_use_id", "truncated")}
            width = len(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
            if len(events) >= traces.MAX_SESSION_ROWS or size + width > traces.MAX_SESSION_BYTES:
                truncated = True
                continue
            size += width
            events.append(dict(event))
        trace = {"version": traces.TRACE_VERSION, "session_id": self.id, "harness": self.harness,
                 "model": traces._head(model or "unknown", 512), "events": events, "edits": [],
                 "truncated": truncated}
        turn = 0
        for seq, event in enumerate(events, 1):
            turn += event["kind"] == "user_prompt"
            event.update(seq=seq, turn=turn)
        # A trace over the blob cap loses its oldest turns whole, as build_trace does.
        if size > traces.MAX_TRACE_BYTES * 7 // 8:
            while len(traces.canonical_bytes(trace)) > traces.MAX_TRACE_BYTES and trace["events"]:
                oldest = trace["events"][0]["turn"]
                trace["events"] = [event for event in trace["events"] if event["turn"] != oldest]
                trace["truncated"] = True
            for seq, event in enumerate(trace["events"], 1):
                event["seq"] = seq
        if not trace["events"]:
            return None
        trace["digest"] = traces.trace_digest(trace)
        return trace


def _tokens(items: list[tuple[Any, Any, Any]]) -> dict[str, int]:
    return {
        "input": sum(item[0].input_tokens or 0 for item in items),
        "cached_input": sum(item[0].cached_input_tokens or 0 for item in items),
        "cache_creation_input": sum(item[0].cache_creation_input_tokens or 0 for item in items),
        "output": sum(item[0].output_tokens or 0 for item in items),
    }


def _codex_input(name: str, raw: Any) -> dict[str, Any]:
    value = raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError:
            value = {"command": raw} if name == "apply_patch" else {"input": raw}
    if not isinstance(value, dict):
        value = {"value": value}
    command, cmd = value.get("command"), value.get("cmd")
    if isinstance(command, list):
        value = {**value, "command": _argv_text(command)}
    elif not isinstance(command, str) and isinstance(cmd, str):
        value = {**value, "command": cmd}
    return value


def _codex_outcome(text: str, status: str | None = None) -> bool:
    """A function call's outcome from its output, else from its item's status, else success."""
    if status in _FAILED:
        return False
    try:
        value: Any = json.loads(text)
    except ValueError:
        value = text
    known = response_success(value, "shell") if isinstance(value, (dict, str)) else None
    if known is None:
        match = _EXIT_CODE.search(text)
        known = int(match.group(1)) == 0 if match else not text.startswith(("Script failed", "Error:"))
    return known


def _codex_item(session: Session, item: Mapping[str, Any], started: str | None, ended: str | None) -> None:
    """One tool that an ``item_completed`` event reports, as its call and its result."""
    kind, item_id = item.get("type"), str(item.get("id"))
    status = item.get("status") if isinstance(item.get("status"), str) else None
    if kind == "CommandExecution":
        command = item.get("command")
        text = _argv_text(command) if isinstance(command, list) else str(command or "")
        cwd = str(item.get("cwd") or "").removeprefix("file://")
        name, tool_class, data = "shell", "shell", {"command": text, **({"cwd": cwd} if cwd else {})}
        stdout = item.get("stdout") or ("" if item.get("stderr") else item.get("aggregated_output") or "")
        output: Any = {"stdout": stdout, "stderr": item.get("stderr") or "", "exit_code": item.get("exit_code")}
        succeeded = response_success({"status": status, "exit_code": item.get("exit_code")}, "shell")
    elif kind == "FileChange":
        changes = item.get("changes") if isinstance(item.get("changes"), dict) else {}
        lines = ["*** Begin Patch"]
        for path, change in changes.items():
            change = change if isinstance(change, dict) else {}
            lines.append(f"*** {({'add': 'Add', 'delete': 'Delete'}).get(change.get('type'), 'Update')} File: {path}")
            if isinstance(change.get("unified_diff"), str):
                lines.append(change["unified_diff"])
        name, tool_class, data = "apply_patch", "edit", {"command": "\n".join([*lines, "*** End Patch"])}
        output = {"stdout": item.get("stdout") or "", "stderr": item.get("stderr") or "", "status": status}
        succeeded = None
    elif kind == "McpToolCall":
        name, tool_class = f"mcp__{item.get('server')}__{item.get('tool')}", "mcp"
        arguments = item.get("arguments")
        data = arguments if isinstance(arguments, dict) else {"value": arguments}
        output = item.get("result")
        succeeded = response_success(output, "other") if isinstance(output, (dict, str)) else None
    elif kind in ("WebSearch", "Extension"):
        extension = str(item.get("kind") or "web.search")
        name = "web_search" if kind == "WebSearch" or extension.startswith("web.") else extension
        tool_class = _CODEX_CLASSES.get(name) or activity.classify_tool("codex", name)
        data = {"query": item.get("query")} if isinstance(item.get("query"), str) else {"action": item.get("action")}
        output = item.get("results") if item.get("results") is not None else item.get("result")
        succeeded = False if item.get("failure") else None
    elif kind == "ImageView":
        name, tool_class = "view_image", "read"
        data, output, succeeded = {"file_path": str(item.get("path") or "").removeprefix("file://")}, "", True
    else:
        return
    if succeeded is None:
        succeeded = status not in _FAILED
    session.call(started, item_id, name, tool_class, data)
    session.result(ended, item_id, succeeded, output)
