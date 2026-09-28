"""Translate the hook events of other coding harnesses into receiver events.

The installed receiver (``automation.handle_hook``) reads the documented hook
payloads of Claude Code and Codex. Cursor, Gemini CLI, GitHub Copilot CLI,
OpenCode, and Hermes Agent report the same lifecycle under their own names and
fields. ``canonical_events`` maps one of their payloads to the receiver's
events, so every harness reaches the same receiver. Each mapping reads only
the fields that the harness documents or that a recorded run showed:

- Cursor: https://cursor.com/docs/hooks
- Gemini CLI: https://geminicli.com/docs/hooks/reference/
- GitHub Copilot CLI: https://docs.github.com/en/copilot/reference/hooks-reference
- OpenCode: https://opencode.ai/docs/plugins/ (sent by Joyride's plugin)
- Hermes Agent: https://hermes-agent.nousresearch.com/docs/user-guide/features/hooks
  (sent by Joyride's plugin)

This module imports only the standard library, because the hook client reads
it before it hands an event to the collector.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


# The harnesses whose native events this module translates.
NATIVE_HARNESSES = ("cursor", "gemini", "github-copilot", "hermes", "opencode")
# A turn capture takes the worktree snapshot pair at the start and the end of
# one turn. Cursor asks a hook for permission before each tool, so a hook that
# only observes takes its snapshots at the turn boundaries instead.
TURN_CAPTURE_TOOL = "joyride:turn"
TURN_CAPTURE_HARNESSES = frozenset({"cursor"})
# Receiver events that no native harness sends under these names.
MODEL_EVENT = "ModelReported"
USAGE_EVENT = "UsageReported"

_SESSION_ID = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]{0,199}$")
# GitHub Copilot reports Claude tool names in its PascalCase payloads. The
# documented table maps each runtime name to the same Claude name, so a call
# keeps one name from its pre-tool event to its post-tool event.
_COPILOT_TOOL_NAMES = {
    "apply_patch": "Edit",
    "ask_user": "AskUserQuestion",
    "bash": "Bash",
    "create": "Write",
    "edit": "Edit",
    "glob": "Glob",
    "grep": "Grep",
    "powershell": "Bash",
    "rg": "Grep",
    "str_replace_editor": "Edit",
    "task": "Agent",
    "update_todo": "TodoWrite",
    "view": "Read",
    "web_fetch": "WebFetch",
    "web_search": "WebSearch",
}
_FIRST_LINE_LIMIT = 64 * 1024


def _text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        return None
    return value.strip()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    return int(value) if value >= 0 else None


def pairing_key(tool_name: str | None, tool_input: Any) -> str | None:
    """Return the key that pairs a pre-tool event with its post-tool event.

    Gemini CLI and GitHub Copilot CLI give a tool call no id. Both of its
    events carry the same tool name and input, so the receiver pairs them by
    this key, oldest open call first.
    """

    if tool_name is None:
        return None
    try:
        encoded = json.dumps(tool_input, sort_keys=True, default=str)
    except (TypeError, ValueError):
        encoded = ""
    digest = hashlib.sha256(encoded.encode("utf-8", "surrogatepass")).hexdigest()
    return f"{tool_name}:{digest[:24]}"


def _event(name: str, base: Mapping[str, Any], **fields: Any) -> dict[str, Any]:
    result = {"hook_event_name": name, **base}
    result.update({key: value for key, value in fields.items() if value is not None})
    return result


def _cursor(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    name = payload.get("hook_event_name")
    roots = payload.get("workspace_roots")
    root = _text(roots[0]) if isinstance(roots, list) and roots else None
    base: dict[str, Any] = {
        # sessionStart and sessionEnd name it session_id, "same as conversation_id".
        "session_id": _text(payload.get("conversation_id")) or _text(payload.get("session_id")),
        # User hooks run from ~/.cursor, so the payload names the workspace.
        "cwd": _text(payload.get("cwd")) or root,
    }
    model = _text(payload.get("model_id")) or _text(payload.get("model"))
    if model is not None:
        base["model"] = model
    turn = _text(payload.get("generation_id"))
    if turn is not None:
        base["turn_id"] = turn
    turn_capture = {"tool_name": TURN_CAPTURE_TOOL, "tool_use_id": f"turn:{turn}"} if turn else None
    if name == "sessionStart":
        return [_event("SessionStart", base)]
    if name == "beforeSubmitPrompt":
        events = [_event("UserPromptSubmit", base, prompt=_text(payload.get("prompt")))]
        if turn_capture is not None:
            events.append(_event("PreToolUse", base, **turn_capture))
        return events
    if name in {"postToolUse", "postToolUseFailure"}:
        fields: dict[str, Any] = {
            "tool_name": _text(payload.get("tool_name")),
            "tool_use_id": _text(payload.get("tool_use_id")),
            "tool_input": dict(_mapping(payload.get("tool_input"))),
            "duration_ms": _count(payload.get("duration")),
        }
        if name == "postToolUse":
            output = payload.get("tool_output")
            try:
                fields["tool_response"] = json.loads(output) if isinstance(output, str) else output
            except ValueError:
                fields["tool_response"] = output
            return [_event("PostToolUse", base, **fields)]
        fields["error"] = _text(payload.get("error_message"))
        fields["is_interrupt"] = payload.get("is_interrupt") is True or None
        return [_event("PostToolUseFailure", base, **fields)]
    if name == "stop":
        events = [_event("PostToolUse", base, **turn_capture)] if turn_capture else []
        if payload.get("status") == "aborted":
            events.append(_event("Interrupt", base))
        return [*events, _event("Stop", base)]
    if name == "sessionEnd":
        return [_event("SessionEnd", base)]
    return []


def _gemini(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    name = payload.get("hook_event_name")
    base = {"session_id": _text(payload.get("session_id")), "cwd": _text(payload.get("cwd"))}
    if name == "SessionStart":
        return [_event("SessionStart", base, source=_text(payload.get("source")))]
    if name == "BeforeAgent":
        return [_event("UserPromptSubmit", base, prompt=_text(payload.get("prompt")))]
    if name == "BeforeModel":
        model = _text(_mapping(payload.get("llm_request")).get("model"))
        return [_event(MODEL_EVENT, base, model=model)] if model else []
    if name in {"BeforeTool", "AfterTool"}:
        tool_name = _text(payload.get("tool_name"))
        tool_input = dict(_mapping(payload.get("tool_input")))
        fields = {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "pairing_key": pairing_key(tool_name, tool_input),
        }
        if name == "BeforeTool":
            return [_event("PreToolUse", base, **fields)]
        response = payload.get("tool_response")
        # "If this property is present, the tool call is considered a failure."
        error = _mapping(response).get("error")
        if error:
            message = _text(_mapping(error).get("message")) if isinstance(error, Mapping) else _text(error)
            return [_event("PostToolUseFailure", base, **fields, error=message)]
        return [_event("PostToolUse", base, **fields, tool_response=response)]
    if name == "AfterAgent":
        return [
            _event("Stop", base, last_assistant_message=_text(payload.get("prompt_response")))
        ]
    if name == "SessionEnd":
        return [_event("SessionEnd", base)]
    return []


def _copilot_home(environ: Mapping[str, str]) -> Path:
    configured = environ.get("COPILOT_HOME")
    if configured:
        return Path(configured).expanduser()
    return Path(environ.get("HOME") or os.path.expanduser("~")) / ".copilot"


def copilot_selected_model(session_id: str | None, environ: Mapping[str, str]) -> str | None:
    """Return the model a Copilot CLI session selected, or None.

    Copilot's hook payloads name no model. The session's event log starts with
    a ``session.start`` record whose ``selectedModel`` names it; only that
    first line is read, and only under the Copilot home folder.
    """

    if session_id is None or not _SESSION_ID.fullmatch(session_id):
        return None
    path = _copilot_home(environ) / "session-state" / session_id / "events.jsonl"
    try:
        with path.open("rb") as handle:
            line = handle.readline(_FIRST_LINE_LIMIT)
        record = json.loads(line)
    except (OSError, ValueError):
        return None
    if not isinstance(record, Mapping) or record.get("type") != "session.start":
        return None
    return _text(_mapping(record.get("data")).get("selectedModel"))


def _copilot(payload: Mapping[str, Any], environ: Mapping[str, str]) -> list[dict[str, Any]]:
    name = payload.get("hook_event_name")
    session = _text(payload.get("session_id"))
    base = {"session_id": session, "cwd": _text(payload.get("cwd"))}
    if name == "SessionStart":
        return [
            _event(
                "SessionStart",
                base,
                source=_text(payload.get("source")),
                model=copilot_selected_model(session, environ),
            )
        ]
    if name == "UserPromptSubmit":
        return [_event("UserPromptSubmit", base, prompt=_text(payload.get("prompt")))]
    if name in {"PreToolUse", "PostToolUse", "PostToolUseFailure"}:
        raw_name = _text(payload.get("tool_name"))
        tool_name = _COPILOT_TOOL_NAMES.get(raw_name or "", raw_name)
        tool_input = payload.get("tool_input")
        if isinstance(tool_input, str):
            try:
                tool_input = json.loads(tool_input)
            except ValueError:
                pass
        tool_input = dict(_mapping(tool_input))
        fields = {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "pairing_key": pairing_key(tool_name, tool_input),
        }
        if name == "PostToolUse":
            fields["tool_response"] = payload.get("tool_result")
        elif name == "PostToolUseFailure":
            fields["error"] = _text(payload.get("error"))
        return [_event(name, base, **fields)]
    if name in {"Stop", "SessionEnd"}:
        return [_event(name, base)]
    return []


def _opencode(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Map what Joyride's OpenCode plugin forwards: a hook's input and output,
    or a bus event's properties, under the OpenCode hook or event name."""

    name = payload.get("hook_event_name")
    source = _mapping(payload.get("input"))
    output = _mapping(payload.get("output"))
    properties = _mapping(payload.get("properties"))
    parent = _text(payload.get("parent_session_id"))
    session = (
        _text(source.get("sessionID"))
        or _text(properties.get("sessionID"))
        or _text(_mapping(properties.get("info")).get("sessionID"))
    )
    if session is None:
        return []
    # The plugin names the parent of a subagent's session, so the child's
    # events join the parent's session as a subagent, as Claude Code's do.
    base: dict[str, Any] = {"session_id": parent or session, "cwd": _text(payload.get("directory"))}
    if parent is not None:
        base["agent_id"] = session
    if name in {"tool.execute.before", "tool.execute.after"}:
        fields = {
            "tool_name": _text(source.get("tool")),
            "tool_use_id": _text(source.get("callID")),
        }
        if name == "tool.execute.before":
            return [_event("PreToolUse", base, **fields, tool_input=dict(_mapping(output.get("args"))))]
        response: dict[str, Any] = {}
        if isinstance(output.get("output"), str):
            response["output"] = output["output"]
        exit_code = _mapping(output.get("metadata")).get("exit")
        if isinstance(exit_code, int) and not isinstance(exit_code, bool):
            response["exit_code"] = exit_code
        return [
            _event(
                "PostToolUse",
                base,
                **fields,
                tool_input=dict(_mapping(source.get("args"))),
                tool_response=response or None,
            )
        ]
    if name == "chat.message":
        message = _mapping(output.get("message"))
        model = _mapping(message.get("model")) or _mapping(source.get("model"))
        model_id = _text(model.get("modelID"))
        parts = output.get("parts") if isinstance(output.get("parts"), list) else []
        prompt = "".join(
            part["text"] for part in parts
            if isinstance(part, Mapping) and part.get("type") == "text" and isinstance(part.get("text"), str)
        )
        events = [_event(MODEL_EVENT, base, model=model_id)] if model_id else []
        events.append(
            _event("UserPromptSubmit", base, prompt=_text(prompt), turn_id=_text(message.get("id")))
        )
        return events
    if name == "session.created":
        return [_event("SubagentStart" if parent else "SessionStart", base)]
    if name == "session.idle":
        return [_event("SubagentStop" if parent else "Stop", base)]
    if name == "message.updated":
        info = _mapping(properties.get("info"))
        if info.get("role") != "assistant" or not _mapping(info.get("time")).get("completed"):
            return []
        tokens = _mapping(info.get("tokens"))
        cache = _mapping(tokens.get("cache"))
        output_tokens = _count(tokens.get("output"))
        reasoning = _count(tokens.get("reasoning"))
        cost = info.get("cost")
        return [
            _event(
                USAGE_EVENT,
                base,
                model=_text(info.get("modelID")),
                provider=_text(info.get("providerID")),
                request_id=_text(info.get("id")),
                # OpenCode counts input without cache and output without reasoning.
                input_tokens=_count(tokens.get("input")),
                cached_input_tokens=_count(cache.get("read")),
                cache_creation_input_tokens=_count(cache.get("write")),
                output_tokens=(
                    output_tokens + (reasoning or 0) if output_tokens is not None else None
                ),
                total_tokens=_count(tokens.get("total")),
                # OpenCode prices each message from its model catalog and
                # reports 0 both for a free model and for a model it has no
                # price for, so only a positive cost is a known one.
                cost_usd=cost if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost > 0 else None,
                cost_source="opencode",
            )
        ]
    return []


def _hermes(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Map what Joyride's Hermes plugin forwards: one hook's keyword arguments."""

    name = payload.get("hook_event_name")
    session = _text(payload.get("session_id"))
    if session is None:
        # Tool calls inside execute_code arrive with an empty session id.
        return []
    base = {"session_id": session, "cwd": _text(payload.get("cwd"))}
    model = _text(payload.get("model"))
    if name == "on_session_start":
        return [_event("SessionStart", base, model=model)]
    if name == "pre_llm_call":
        events = [_event(MODEL_EVENT, base, model=model)] if model else []
        events.append(
            _event(
                "UserPromptSubmit",
                base,
                prompt=_text(payload.get("user_message")),
                turn_id=_text(payload.get("turn_id")),
            )
        )
        return events
    if name in {"pre_tool_call", "post_tool_call"}:
        tool_name = _text(payload.get("tool_name"))
        tool_input = dict(_mapping(payload.get("args")))
        tool_use_id = _text(payload.get("tool_call_id"))
        fields: dict[str, Any] = {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "turn_id": _text(payload.get("turn_id")),
        }
        if tool_use_id is not None:
            fields["tool_use_id"] = tool_use_id
        else:
            fields["pairing_key"] = pairing_key(tool_name, tool_input)
        if name == "pre_tool_call":
            return [_event("PreToolUse", base, **fields)]
        fields["duration_ms"] = _count(payload.get("duration_ms"))
        status = payload.get("status")
        if status in {"error", "blocked"}:
            return [
                _event(
                    "PostToolUseFailure",
                    base,
                    **fields,
                    error=_text(payload.get("error_message")) or _text(payload.get("error_type")),
                )
            ]
        result = payload.get("result")
        return [
            _event(
                "PostToolUse",
                base,
                **fields,
                tool_response={"output": result} if isinstance(result, str) else None,
            )
        ]
    if name == "post_api_request":
        usage = _mapping(payload.get("usage"))
        return [
            _event(
                USAGE_EVENT,
                base,
                model=_text(payload.get("response_model")) or model,
                provider=_text(payload.get("provider")),
                request_id=_text(payload.get("api_request_id")),
                input_tokens=_count(usage.get("input_tokens")),
                cached_input_tokens=_count(usage.get("cache_read_tokens")),
                cache_creation_input_tokens=_count(usage.get("cache_write_tokens")),
                output_tokens=_count(usage.get("output_tokens")),
                total_tokens=_count(usage.get("total_tokens")),
            )
        ]
    if name == "on_session_end":
        # Hermes fires this at the end of every turn, not of the session.
        events = [_event("Interrupt", base)] if payload.get("interrupted") is True else []
        return [*events, _event("Stop", base)]
    if name == "on_session_finalize":
        return [_event("SessionEnd", base)]
    return []


def canonical_events(
    harness: str,
    payload: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return the receiver events that one native hook payload reports.

    Claude Code and Codex payloads pass through unchanged. An empty list means
    that the payload reports nothing the receiver records.
    """

    if not isinstance(payload, Mapping):
        return []
    if harness in {"claude-code", "codex"}:
        # Cursor also runs the Claude Code hooks in ~/.claude/settings.json.
        # Its Cursor hooks record that event, so this copy records nothing.
        if harness == "claude-code" and "cursor_version" in payload:
            return []
        return [dict(payload)]
    if harness == "cursor":
        events = _cursor(payload)
    elif harness == "gemini":
        events = _gemini(payload)
    elif harness == "github-copilot":
        events = _copilot(payload, os.environ if environ is None else environ)
    elif harness == "opencode":
        events = _opencode(payload)
    elif harness == "hermes":
        events = _hermes(payload)
    else:
        return []
    return [event for event in events if event.get("session_id")]


__all__ = [
    "MODEL_EVENT",
    "NATIVE_HARNESSES",
    "TURN_CAPTURE_HARNESSES",
    "TURN_CAPTURE_TOOL",
    "USAGE_EVENT",
    "canonical_events",
    "copilot_selected_model",
    "pairing_key",
]
