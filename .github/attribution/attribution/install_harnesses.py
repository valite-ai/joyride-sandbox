"""Machine hooks for Cursor, Gemini CLI, GitHub Copilot CLI, OpenCode, and Hermes.

``joyride install --user`` sets these up beside the Claude Code and Codex
hooks. Each harness is set up only when it is installed on the computer: its
command is on ``PATH`` or its configuration folder exists. Every hook runs the
same launcher as the Claude Code and Codex hooks, which hands the harness's
native event to the receiver, and ``adapters.native_hooks`` translates it
there.

Cursor and Gemini CLI read hooks from one shared settings file, so the install
merges one managed entry for each event into it and uninstall removes exactly
those entries. GitHub Copilot CLI reads every file in its hooks folder, and
OpenCode and Hermes Agent load plugins, so for them Joyride owns one file or
folder of its own.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import Any, Mapping

from .adapters.native_hooks import NATIVE_HARNESSES
from .install import (
    _MARKER,
    _assert_native_target,
    _atomic_write,
    _file_details,
    _json_bytes,
    _load_json_config,
)


HARNESSES = NATIVE_HARNESSES
# The events each harness's hooks subscribe to. Every name is documented:
# https://cursor.com/docs/hooks, https://geminicli.com/docs/hooks/reference/,
# https://docs.github.com/en/copilot/reference/hooks-reference.
# Cursor asks its permission hooks (such as preToolUse) for a decision, so
# Joyride subscribes only to events that observe and takes its snapshots at
# the turn's boundaries.
_EVENTS = {
    "cursor": (
        "sessionStart",
        "beforeSubmitPrompt",
        "postToolUse",
        "postToolUseFailure",
        "stop",
        "sessionEnd",
    ),
    "gemini": (
        "SessionStart",
        "BeforeAgent",
        "BeforeModel",
        "BeforeTool",
        "AfterTool",
        "AfterAgent",
        "SessionEnd",
    ),
    # PascalCase names select Copilot's snake_case payloads with Claude tool names.
    "github-copilot": (
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "Stop",
        "SessionEnd",
    ),
}
# Gemini runs a BeforeTool hook before every matching tool and waits for it,
# so only the tools that change files take the snapshot that the hook starts.
_GEMINI_SNAPSHOT_MATCHER = "^(write_file|replace|run_shell_command)$"
# Gemini counts a hook timeout in milliseconds and the others in seconds. A
# pre-tool hook waits at most 9 seconds for its snapshot.
_GEMINI_TIMEOUT_MS = 10_000
_CURSOR_TIMEOUT_SECONDS = 10
_COPILOT_TIMEOUT_SECONDS = 30
# Hermes plugins run a callback for these hooks. Only the listed keyword
# arguments leave the Hermes process, so no conversation history, request,
# or response body reaches the receiver.
# https://hermes-agent.nousresearch.com/docs/user-guide/features/hooks
_HERMES_FIELDS = {
    "on_session_start": ("session_id", "model", "platform"),
    "pre_llm_call": ("session_id", "turn_id", "user_message", "model", "platform"),
    "pre_tool_call": ("session_id", "turn_id", "tool_name", "args", "tool_call_id"),
    "post_tool_call": (
        "session_id",
        "turn_id",
        "tool_name",
        "args",
        "tool_call_id",
        "result",
        "status",
        "error_type",
        "error_message",
        "duration_ms",
    ),
    "post_api_request": (
        "session_id",
        "turn_id",
        "api_request_id",
        "model",
        "provider",
        "response_model",
        "usage",
    ),
    "on_session_end": ("session_id", "turn_id", "completed", "failed", "interrupted", "model"),
    "on_session_finalize": ("session_id", "reason"),
}
HERMES_PLUGIN = "joyride"
_ENABLE_SECONDS = 120


def _home_value(environ: Mapping[str, str], name: str, default: Path) -> Path:
    value = environ.get(name)
    return Path(os.path.abspath(os.path.expanduser(value))) if value else default


def config_path(harness: str, home: Path, environ: Mapping[str, str]) -> Path:
    """Return the file or folder that holds Joyride's hooks for one harness."""

    if harness == "cursor":
        return home / ".cursor" / "hooks.json"
    if harness == "gemini":
        return home / ".gemini" / "settings.json"
    if harness == "github-copilot":
        return _home_value(environ, "COPILOT_HOME", home / ".copilot") / "hooks" / "joyride.json"
    if harness == "opencode":
        config = _home_value(environ, "XDG_CONFIG_HOME", home / ".config")
        return config / "opencode" / "plugins" / "joyride.js"
    if harness == "hermes":
        return _home_value(environ, "HERMES_HOME", home / ".hermes") / "plugins" / HERMES_PLUGIN
    raise ValueError(f"unsupported machine hook harness: {harness}")


def _config_dir(harness: str, home: Path, environ: Mapping[str, str]) -> Path:
    """Return the folder whose presence shows that the harness ran here."""

    path = config_path(harness, home, environ)
    return path.parent.parent if harness in {"github-copilot", "opencode", "hermes"} else path.parent


_COMMANDS = {
    "cursor": ("cursor-agent", "cursor"),
    "gemini": ("gemini",),
    "github-copilot": ("copilot",),
    "hermes": ("hermes",),
    "opencode": ("opencode",),
}


def detected(harness: str, home: Path, environ: Mapping[str, str]) -> bool:
    """Return whether the harness is installed: its command or its folder exists."""

    search_path = environ.get("PATH")
    return any(shutil.which(name, path=search_path) for name in _COMMANDS[harness]) or (
        _config_dir(harness, home, environ).is_dir()
    )


# Gemini CLI denies a call when a hook exits with 2 or more and prints
# anything, and Copilot CLI denies a tool when its preToolUse hook fails. Both
# run the command in bash, so a missing launcher or runtime exits 0 in silence.
_SHELL_GUARDED = frozenset({"gemini", "github-copilot"})


def managed_command(launcher: Path, harness: str) -> str:
    guard = " 2>/dev/null || true" if harness in _SHELL_GUARDED else ""
    return f"{shlex.quote(str(launcher))} _hook --harness {harness}{guard} {_MARKER}"


def _opencode_plugin(launcher: Path) -> bytes:
    text = _OPENCODE_PLUGIN.replace("@LAUNCHER@", json.dumps(str(launcher)))
    return text.encode("utf-8")


def _hermes_files(launcher: Path) -> dict[str, bytes]:
    hooks = "".join(f"  - {name}\n" for name in _HERMES_FIELDS)
    manifest = (
        f"{_MARKER}\n"
        f"name: {HERMES_PLUGIN}\n"
        "version: 1.0.0\n"
        "description: Send Hermes Agent sessions to the local Joyride receiver.\n"
        f"provides_hooks:\n{hooks}"
    )
    module = _HERMES_PLUGIN.replace("@LAUNCHER@", repr(str(launcher))).replace(
        "@FIELDS@", repr(_HERMES_FIELDS)
    )
    return {"plugin.yaml": manifest.encode("utf-8"), "__init__.py": module.encode("utf-8")}


def _copilot_file(command: str) -> bytes:
    hooks = {
        event: [{"type": "command", "bash": command, "timeoutSec": _COPILOT_TIMEOUT_SECONDS}]
        for event in _EVENTS["github-copilot"]
    }
    return _json_bytes({"version": 1, "hooks": hooks})


def _owned_files(harness: str, path: Path, launcher: Path) -> dict[Path, bytes]:
    """Return the files that Joyride owns outright for one harness."""

    if harness == "github-copilot":
        return {path: _copilot_file(managed_command(launcher, harness))}
    if harness == "opencode":
        return {path: _opencode_plugin(launcher)}
    if harness == "hermes":
        return {path / name: data for name, data in _hermes_files(launcher).items()}
    return {}


def _cursor_entry(command: str) -> dict[str, Any]:
    return {"command": command, "timeout": _CURSOR_TIMEOUT_SECONDS}


def _gemini_group(event: str, command: str) -> dict[str, Any]:
    group: dict[str, Any] = {
        "hooks": [
            {"type": "command", "command": command, "timeout": _GEMINI_TIMEOUT_MS, "name": "joyride"}
        ]
    }
    if event == "BeforeTool":
        group = {"matcher": _GEMINI_SNAPSHOT_MATCHER, **group}
    return group


def _is_managed(entry: Any) -> bool:
    return isinstance(entry, dict) and _MARKER in str(entry.get("command"))


def _without_managed(harness: str, entries: list[Any]) -> list[Any]:
    """Return one event's entries without the Joyride ones."""

    if harness == "cursor":
        return [entry for entry in entries if not _is_managed(entry)]
    kept: list[Any] = []
    for group in entries:
        if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
            kept.append(group)
            continue
        handlers = [handler for handler in group["hooks"] if not _is_managed(handler)]
        if handlers or set(group) - {"matcher", "hooks"}:
            kept.append({**group, "hooks": handlers})
    return kept


# Gemini CLI skips these names in its hooks object, where earlier releases
# kept its hook settings, so they hold no hook entries.
_GEMINI_HOOK_SETTINGS = frozenset({"enabled", "disabled", "notifications"})


def _events(harness: str, hooks: dict[str, Any]) -> list[str]:
    """Return the names in a hooks object that hold hook entries."""

    return [
        name for name in hooks if not (harness == "gemini" and name in _GEMINI_HOOK_SETTINGS)
    ]


def _hooks_object(harness: str, payload: dict[str, Any], path: Path) -> dict[str, Any]:
    hooks = payload.get("hooks")
    if hooks is None:
        return {}
    if not isinstance(hooks, dict) or any(
        not isinstance(hooks[event], list) for event in _events(harness, hooks)
    ):
        raise ValueError(f"Refusing to change the hooks in {path}: they are not an object of lists.")
    return hooks


def _merged(harness: str, payload: dict[str, Any], command: str, path: Path) -> dict[str, Any]:
    """Return a shared settings payload with exactly one Joyride entry per event."""

    result = copy.deepcopy(payload)
    hooks = _hooks_object(harness, result, path)
    for event in _EVENTS[harness]:
        entries = _without_managed(harness, list(hooks.get(event, [])))
        entries.append(
            _cursor_entry(command) if harness == "cursor" else _gemini_group(event, command)
        )
        hooks[event] = entries
    result["hooks"] = hooks
    if harness == "cursor":
        # Cursor requires the schema version, and 1 is the only one.
        result.setdefault("version", 1)
    return result


def _removed(harness: str, payload: dict[str, Any], path: Path) -> dict[str, Any]:
    """Return a shared settings payload without any Joyride entry."""

    result = copy.deepcopy(payload)
    hooks = _hooks_object(harness, result, path)
    for event in _events(harness, hooks):
        entries = _without_managed(harness, hooks[event])
        if entries:
            hooks[event] = entries
        elif hooks[event]:
            # The install added this event's list, so an empty one goes too.
            del hooks[event]
    if "hooks" in result and not hooks:
        del result["hooks"]
    return result


def _shared_health(harness: str, path: Path, command: str) -> tuple[bool, str | None]:
    try:
        payload, raw, _mode, _atime, _mtime = _load_json_config(path)
        if raw is None:
            return False, f"{path} is missing."
        hooks = _hooks_object(harness, payload, path)
    except (OSError, ValueError) as exc:
        return False, str(exc)
    if harness == "gemini" and _mapping(payload.get("hooksConfig")).get("enabled") is False:
        return False, f"Gemini CLI hooks are turned off by hooksConfig.enabled in {path}."
    for event in _EVENTS[harness]:
        entries = hooks.get(event, [])
        if harness == "cursor":
            handlers = [entry for entry in entries if _is_managed(entry)]
        else:
            handlers = [
                handler
                for group in entries
                if isinstance(group, dict) and isinstance(group.get("hooks"), list)
                for handler in group["hooks"]
                if _is_managed(handler)
            ]
        if [handler.get("command") for handler in handlers] != [command]:
            return False, f"The Joyride {event} hook in {path} is missing or duplicated."
    return True, None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hermes_command(home: Path, environ: Mapping[str, str]) -> str | None:
    found = shutil.which("hermes", path=environ.get("PATH"))
    if found:
        return found
    candidate = home / ".local" / "bin" / "hermes"
    return str(candidate) if os.access(candidate, os.X_OK) else None


def _hermes_plugins(action: str, home: Path, environ: Mapping[str, str]) -> str | None:
    """Run ``hermes plugins ACTION joyride``; return a warning when it fails.

    Hermes loads a user plugin only after its name is in ``plugins.enabled``
    in its config.yaml. Hermes's own command edits that file, and its remove
    command also forgets the plugin's config entries.
    """

    command = _hermes_command(home, environ)
    fix = f"Run: hermes plugins {action} {HERMES_PLUGIN}"
    if command is None:
        return f"Hermes Agent's command was not found. {fix}"
    try:
        result = subprocess.run(
            [command, "plugins", action, HERMES_PLUGIN],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=dict(environ),
            timeout=_ENABLE_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"Hermes Agent could not {action} the Joyride plugin: {exc}. {fix}"
    if result.returncode != 0:
        detail = result.stdout.decode("utf-8", errors="replace").strip().splitlines()
        return (
            f"Hermes Agent could not {action} the Joyride plugin"
            + (f": {detail[-1][:300]}" if detail else "")
            + f". {fix}"
        )
    return None


def _missing_dirs(target: Path, home: Path) -> list[str]:
    """Return the folders above ``target`` that do not exist yet, deepest first."""

    missing: list[str] = []
    cursor = target.parent
    while cursor != home and cursor != cursor.parent and not cursor.exists():
        missing.append(str(cursor))
        cursor = cursor.parent
    return missing


def plan(
    harness: str,
    home: Path,
    launcher: Path,
    previous: Mapping[str, Any] | None,
    environ: Mapping[str, str],
) -> dict[str, Any]:
    """Return the manifest record of one harness's hooks without writing them.

    The machine manifest records it before ``apply`` writes anything, so an
    install that stops part way still leaves an uninstall that removes what it
    wrote.
    """

    path = config_path(harness, home, environ)
    record: dict[str, Any] = {"path": str(path), "managed_command": managed_command(launcher, harness)}
    targets = [path] if harness in {"cursor", "gemini"} else list(_owned_files(harness, path, launcher))
    for target in targets:
        if target.is_relative_to(home):
            _assert_native_target(target, home)
    created = [] if previous is None else list(previous.get("created_dirs") or [])
    for target in targets:
        created.extend(name for name in _missing_dirs(target, home) if name not in created)
    record["created_dirs"] = created
    if harness in {"cursor", "gemini"}:
        record["created_file"] = (
            bool(previous.get("created_file")) if previous is not None else not path.exists()
        )
    else:
        record["files"] = {
            str(target): _sha256(data)
            for target, data in _owned_files(harness, path, launcher).items()
        }
    if harness == "hermes":
        record["enabled"] = previous is not None and previous.get("enabled") is True
    return record


def apply(
    harness: str,
    record: dict[str, Any],
    home: Path,
    launcher: Path,
    environ: Mapping[str, str],
) -> list[str]:
    """Write the hooks that ``plan`` recorded; return warnings."""

    path = Path(record["path"])
    if harness in {"cursor", "gemini"}:
        payload, raw, mode, _atime, _mtime = _load_json_config(path)
        desired = _json_bytes(_merged(harness, payload, record["managed_command"], path))
        if raw != desired:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _atomic_write(path, desired, mode=mode)
        return []
    changed = False
    for target, data in _owned_files(harness, path, launcher).items():
        try:
            current = target.read_bytes()
        except FileNotFoundError:
            current = None
        if current is not None and _MARKER.encode("utf-8") not in current:
            raise ValueError(f"Refusing to replace an unowned file at {target}.")
        if current != data:
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _atomic_write(target, data, mode=0o644)
            changed = True
    if harness == "hermes" and (changed or record.get("enabled") is not True):
        warning = _hermes_plugins("enable", home, environ)
        record["enabled"] = warning is None
        return [warning] if warning else []
    return []


def uninstall(
    harness: str, record: Mapping[str, Any], home: Path, environ: Mapping[str, str]
) -> list[str]:
    """Remove exactly the hooks, files, and folders that the install wrote."""

    path = Path(str(record.get("path")))
    warnings: list[str] = []
    if harness in {"cursor", "gemini"}:
        payload, raw, mode, _atime, _mtime = _load_json_config(path)
        if raw is not None:
            updated = _removed(harness, payload, path)
            if record.get("created_file") is True and set(updated) <= {"version"}:
                path.unlink()
            elif updated != payload:
                _atomic_write(path, _json_bytes(updated), mode=mode)
    else:
        if harness == "hermes" and path.exists():
            # Hermes's remove command deletes the folder and forgets its
            # plugins.enabled entry, which only Hermes edits.
            if _hermes_plugins("remove", home, environ) is not None:
                warnings.append(
                    f"Hermes Agent could not remove the Joyride plugin, so Joyride deleted "
                    f"its folder. Remove {HERMES_PLUGIN} from plugins.enabled in the "
                    "Hermes config.yaml."
                )
        files = record.get("files") if isinstance(record.get("files"), dict) else {}
        for name in files:
            target = Path(str(name))
            try:
                raw_file, _mode, _atime, _mtime = _file_details(target)
            except (OSError, ValueError):
                continue
            if _MARKER.encode("utf-8") in raw_file:
                target.unlink()
    for name in record.get("created_dirs") or []:
        try:
            Path(str(name)).rmdir()
        except OSError:
            # A folder that holds other files stays.
            pass
    return warnings


def health(harness: str, record: Mapping[str, Any] | None) -> dict[str, Any]:
    """Report one harness's machine hook state without changing it."""

    if record is None:
        return {"installed": False, "state": "not-installed", "message": None}
    command = record.get("managed_command")
    path = Path(str(record.get("path")))
    if harness in {"cursor", "gemini"}:
        healthy, message = _shared_health(harness, path, str(command))
    else:
        files = record.get("files") if isinstance(record.get("files"), dict) else {}
        healthy, message = bool(files), None
        for name, expected in files.items():
            try:
                raw, _mode, _atime, _mtime = _file_details(Path(str(name)))
            except (OSError, ValueError):
                healthy, message = False, f"The Joyride hook file {name} is missing."
                break
            if _sha256(raw) != expected:
                healthy, message = False, f"The Joyride hook file {name} changed."
                break
        if healthy and harness == "hermes" and record.get("enabled") is not True:
            healthy, message = False, f"Run: hermes plugins enable {HERMES_PLUGIN}"
    return {
        "installed": healthy,
        "state": "enabled" if healthy else "needs-attention",
        "message": message,
    }


_OPENCODE_PLUGIN = """// # attribution-managed-v1
// Joyride sends OpenCode's session, prompt, tool, and usage events to the
// local receiver that its Claude Code and Codex hooks use. It only observes:
// it forwards each hook's input and never changes it. Joyride rewrites this
// file on each install and removes it on uninstall.
import { spawn } from "node:child_process";

const LAUNCHER = @LAUNCHER@;
const OUTPUT_LIMIT = 16384;
const parents = new Map();
const reported = new Set();
let queue = Promise.resolve();

function deliver(payload) {
  return new Promise((resolve) => {
    let child;
    try {
      child = spawn(LAUNCHER, ["_hook", "--harness", "opencode"], {
        stdio: ["pipe", "ignore", "ignore"],
      });
    } catch {
      resolve();
      return;
    }
    const timer = setTimeout(() => {
      try {
        child.kill();
      } catch {}
      resolve();
    }, 15000);
    const done = () => {
      clearTimeout(timer);
      resolve();
    };
    child.on("error", done);
    child.on("close", done);
    child.stdin.on("error", () => {});
    child.stdin.end(JSON.stringify(payload));
  });
}

// Events reach the receiver one at a time and in order, as a harness's
// command hooks do.
function send(directory, name, fields) {
  const sessionID = fields.input?.sessionID ?? fields.properties?.sessionID;
  const payload = { hook_event_name: name, directory, ...fields };
  if (sessionID && parents.has(sessionID)) {
    payload.parent_session_id = parents.get(sessionID);
  }
  queue = queue.then(() => deliver(payload));
  return queue;
}

export const JoyridePlugin = async ({ directory }) => ({
  "chat.message": async (input, output) => {
    const parts = (output.parts ?? [])
      .filter((part) => part.type === "text")
      .map((part) => ({ type: "text", text: part.text }));
    await send(directory, "chat.message", {
      input,
      output: { message: output.message, parts },
    });
  },
  "tool.execute.before": async (input, output) => {
    await send(directory, "tool.execute.before", { input, output: { args: output.args } });
  },
  "tool.execute.after": async (input, output) => {
    const text = typeof output.output === "string" ? output.output.slice(0, OUTPUT_LIMIT) : undefined;
    const exit = output.metadata?.exit;
    await send(directory, "tool.execute.after", {
      input,
      output: { title: output.title, output: text, metadata: typeof exit === "number" ? { exit } : {} },
    });
  },
  event: async ({ event }) => {
    const properties = event.properties ?? {};
    const info = properties.info ?? {};
    if (event.type === "session.created") {
      if (info.parentID) parents.set(info.id, info.parentID);
      send(directory, event.type, {
        properties: { sessionID: properties.sessionID ?? info.id, info: { id: info.id, parentID: info.parentID } },
      });
    } else if (event.type === "session.idle") {
      send(directory, event.type, { properties: { sessionID: properties.sessionID } });
    } else if (event.type === "message.updated" && info.role === "assistant" && info.time?.completed) {
      if (reported.has(info.id)) return;
      reported.add(info.id);
      send(directory, event.type, {
        properties: {
          sessionID: properties.sessionID ?? info.sessionID,
          info: {
            id: info.id,
            sessionID: info.sessionID,
            role: info.role,
            modelID: info.modelID,
            providerID: info.providerID,
            cost: info.cost,
            tokens: info.tokens,
            time: info.time,
          },
        },
      });
    }
  },
  // OpenCode does not wait for event hooks, so the last events finish here.
  dispose: async () => {
    await queue;
  },
});
"""


_HERMES_PLUGIN = '''# attribution-managed-v1
"""Send Hermes Agent's session, prompt, tool, and usage events to Joyride.

The hooks only observe: each callback returns None, so Hermes runs every
tool as it would without them. Only the listed keyword arguments leave the
process. Joyride rewrites this file on each install and removes it on
uninstall.
"""

import json
import os
import subprocess

_LAUNCHER = @LAUNCHER@
_FIELDS = @FIELDS@
_OUTPUT_LIMIT = 16384


def _send(hook, arguments):
    payload = {"hook_event_name": hook, "cwd": os.environ.get("TERMINAL_CWD") or os.getcwd()}
    for key in _FIELDS[hook]:
        value = arguments.get(key)
        if key == "result" and isinstance(value, str):
            value = value[:_OUTPUT_LIMIT]
        payload[key] = value
    try:
        subprocess.run(
            [_LAUNCHER, "_hook", "--harness", "hermes"],
            input=json.dumps(payload, default=str).encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
    except Exception:
        pass


def _callback(hook):
    def callback(**arguments):
        _send(hook, arguments)

    return callback


def register(ctx):
    for hook in _FIELDS:
        ctx.register_hook(hook, _callback(hook))
'''


__all__ = [
    "HARNESSES",
    "apply",
    "config_path",
    "detected",
    "health",
    "managed_command",
    "plan",
    "uninstall",
]
