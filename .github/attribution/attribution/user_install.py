"""Machine-level installation of the native harness hooks at user scope.

Claude Code applies ``~/.claude/settings.json`` to every project it opens and
Codex reads ``~/.codex/hooks.json`` the same way, so one install per machine
covers every clone. The repository installer in :mod:`attribution.install`
still owns the Git publisher and the per-clone manifest; this module writes
the two user-scope configuration files, the cost collection settings for
them, a fixed hook launcher, and its own small manifest.

Every hook command names the launcher at ``~/.attribution/bin/joyride-hook``
instead of an interpreter. A reinstall after an upgrade rewrites only the
launcher, so the command text that a tool may have reviewed stays the same.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

from .install import (
    _DISABLE_TELEMETRY_ENV,
    _EVENTS,
    _MARKER,
    _assert_native_target,
    _atomic_write,
    _config_health,
    _file_details,
    _has_claude_telemetry_credentials,
    _json_bytes,
    _load_json_config,
    _merge_claude_telemetry_env,
    _merge_native_hooks,
    _remove_claude_telemetry_env,
    _remove_native_hooks,
    _runtime,
)
from .runtime import RuntimeCommand


_VERSION = 1
_NOTICE = (
    "Restart existing coding sessions so they load the machine hooks. "
    "Review the hooks in Codex when prompted."
)
USER_PATHS = {
    "codex": Path(".codex/hooks.json"),
    "claude-code": Path(".claude/settings.json"),
}
_LAUNCHER = Path(".attribution/bin/joyride-hook")
# Process names that identify a coding session. A Node launcher reports the
# script as its second argument, so both positions are read.
_AGENT_COMMANDS = {"claude": "Claude Code", "codex": "Codex", "codex.js": "Codex"}
_NODE_COMMANDS = {"node", "nodejs"}


def _home() -> Path:
    return Path(os.path.abspath(Path.home()))


def user_manifest_path() -> Path:
    """Return the machine-level manifest path without creating it."""

    return _home() / ".attribution" / "user-install.json"


def launcher_path() -> Path:
    """Return the fixed launcher path that every machine hook command names."""

    return _home() / _LAUNCHER


def machine_telemetry_id() -> str:
    """Return the key that holds this machine's share of the Codex OTel block."""

    return f"user-scope:{_home()}"


def _config_path(harness: str) -> Path:
    return _home() / USER_PATHS[harness]


def _managed_command(harness: str) -> str:
    return f"{shlex.quote(str(launcher_path()))} _hook --harness {harness} {_MARKER}"


def _stable_executable(runtime: RuntimeCommand) -> Path:
    """Prefer a tool environment's own interpreter link over its target.

    uv and pipx link ``bin/python`` in a tool environment to a versioned
    interpreter. The link survives a patch upgrade of that interpreter, and
    the resolved path does not.
    """

    if runtime.standalone:
        return runtime.executable
    candidate = Path(os.path.abspath(sys.executable))
    try:
        if candidate != runtime.executable and candidate.resolve() == runtime.executable:
            return candidate
    except OSError:
        pass
    return runtime.executable


def _launcher_bytes(runtime: RuntimeCommand, executable: Path) -> bytes:
    # A Python runtime starts the thin hook client, which hands each hook
    # event to the local collector and serves every other command through the
    # CLI. A standalone build keeps its single entry point.
    invocation = (
        runtime.cli(())
        if runtime.standalone
        else runtime.module_child(
            "attribution.hook_client", (), standalone_action="_hook"
        )
    )
    arguments = (str(executable), *invocation.argv[1:])
    lines = [
        "#!/bin/sh",
        _MARKER,
        "# Joyride rewrites this launcher on each install; hook commands stay the same.",
    ]
    if invocation.pythonpath is not None:
        lines.append(f"PYTHONPATH={shlex.quote(str(invocation.pythonpath))}")
        lines.append("export PYTHONPATH")
    lines.append(
        "exec " + " ".join(shlex.quote(argument) for argument in arguments) + ' "$@"'
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _launcher_healthy(manifest: dict[str, Any]) -> bool:
    """Return whether the launcher and the runtime it starts both exist."""

    try:
        raw, mode, _atime, _mtime = _file_details(launcher_path())
    except (OSError, ValueError):
        return False
    executable, source_root = manifest.get("executable"), manifest.get("source_root")
    if not isinstance(executable, str) or not os.access(executable, os.X_OK):
        return False
    if source_root is not None and not (
        isinstance(source_root, str) and os.path.isdir(source_root)
    ):
        return False
    return bool(mode & 0o111) and manifest.get("launcher_sha256") == _sha256(raw)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes, *, mode: int) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _atomic_write(path, data, mode=mode)


def load_user_manifest() -> dict[str, Any] | None:
    """Return the machine-level manifest, or ``None`` when none exists."""

    path = user_manifest_path()
    if not path.exists():
        return None
    raw, _mode, _atime, _mtime = _file_details(path)
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"The machine install manifest is malformed: {exc}.") from exc
    if not isinstance(manifest, dict) or manifest.get("version") != _VERSION:
        raise ValueError("The machine install manifest has an unsupported version.")
    integrations = manifest.get("integrations")
    if not isinstance(integrations, dict) or any(
        not isinstance(item, dict) for item in integrations.values()
    ):
        raise ValueError("The machine install manifest has invalid entries.")
    return manifest


def _telemetry_owner(previous: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a manifest that owns machine cost collection and no hooks yet."""

    if previous is not None:
        return {**previous, "telemetry_registered": True}
    return {
        "version": _VERSION,
        "telemetry_enabled": False,
        "telemetry_registered": True,
        "integrations": {},
    }


def _record(manifest: dict[str, Any] | None, harness: str) -> dict[str, Any] | None:
    if manifest is None:
        return None
    record = manifest["integrations"].get(harness)
    return record if isinstance(record, dict) else None


def _claude_cost_env(
    payload: dict[str, Any],
    record: dict[str, Any] | None,
    expected: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge the collector settings unless another OTel exporter is configured.

    Any ``OTEL_`` key that Joyride did not write belongs to another exporter.
    Redirecting one of its signals would break it, so nothing is added then.
    """

    prior_keys = (
        {key for key in record.get("telemetry_env_keys", []) if isinstance(key, str)}
        if record is not None and isinstance(record.get("telemetry_env_keys"), list)
        else set()
    )
    prior_expected = (
        record.get("telemetry_env_expected", {})
        if record is not None and isinstance(record.get("telemetry_env_expected"), dict)
        else {}
    )
    object_created = (
        bool(record.get("telemetry_env_object_created"))
        if record is not None and "telemetry_env_object_created" in record
        else "env" not in payload
    )
    env = payload.get("env")
    foreign = (
        sorted(
            key
            for key in env
            if isinstance(key, str) and key.startswith("OTEL_") and key not in prior_keys
        )
        if isinstance(env, dict)
        else []
    )
    if foreign:
        return payload, {
            "telemetry_env_expected": prior_expected,
            "telemetry_env_keys": sorted(prior_keys),
            "telemetry_env_conflicts": foreign,
            "telemetry_env_object_created": object_created,
        }
    merged, owned, conflicts = _merge_claude_telemetry_env(
        payload, expected, prior_keys, prior_expected
    )
    return merged, {
        "telemetry_env_expected": expected,
        "telemetry_env_keys": owned,
        "telemetry_env_conflicts": conflicts,
        "telemetry_env_object_created": object_created,
    }


def _codex_warning(codex: Any) -> str | None:
    if not isinstance(codex, dict) or codex.get("enabled") is True:
        return None
    detail = " ".join(
        value
        for value in (codex.get("message"), codex.get("action"))
        if isinstance(value, str) and value
    )
    return "Codex cost collection needs attention." + (f" {detail}" if detail else "")


def install_user_hooks() -> dict[str, Any]:
    """Write the harness hooks once for this machine and every repository."""

    runtime = _runtime()
    previous = load_user_manifest()
    home = _home()
    executable = _stable_executable(runtime)
    launcher = launcher_path()
    _assert_native_target(launcher, home)
    if launcher.exists() and (previous is None or "launcher_sha256" not in previous):
        raw_launcher, _mode, _atime, _mtime = _file_details(launcher)
        if _MARKER.encode("utf-8") not in raw_launcher:
            raise ValueError(f"Refusing to replace an unowned file at {launcher}.")
    loaded: dict[str, tuple[dict[str, Any], bytes | None, int]] = {}
    for harness in USER_PATHS:
        path = _config_path(harness)
        _assert_native_target(path, home)
        payload, raw, mode, _atime, _mtime = _load_json_config(path)
        command = _managed_command(harness)
        record = _record(previous, harness)
        installed = record.get("managed_command") if record is not None else None
        if isinstance(installed, str) and installed != command:
            # Commands written before the launcher named an interpreter. Retire
            # the command this installer wrote before merging the current one.
            payload, _removed = _remove_native_hooks(
                payload, installed, _EVENTS[harness], created_file=False
            )
        # Merging here rejects an invalid hook structure before any change.
        merged = _merge_native_hooks(payload, command, _EVENTS[harness])
        loaded[harness] = (merged, raw, mode)

    # Start the collector only after both configuration files merged.
    warnings: list[str] = []
    telemetry_requested = os.environ.get(_DISABLE_TELEMETRY_ENV) != "1"
    # An earlier install that registered the machine keeps that ownership
    # until uninstall, even when this run cannot configure cost collection.
    registered = previous is not None and (
        previous.get("telemetry_registered") is True
        or previous.get("telemetry_enabled") is True
    )
    telemetry_enabled = telemetry_requested and registered
    telemetry: dict[str, Any] = {}
    claude_env: dict[str, str] = {}
    if telemetry_requested:
        from . import telemetry_setup

        if not registered:
            # Record ownership before the collector starts or Codex changes,
            # so an install that fails later still leaves an uninstall that
            # can undo them.
            _write(
                user_manifest_path(), _json_bytes(_telemetry_owner(previous)), mode=0o600
            )
            registered = True
        try:
            setup = telemetry_setup.configure_telemetry(machine_telemetry_id())
            claude_env = telemetry_setup.claude_telemetry_env()
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            warnings.append(f"Cost collection could not be configured: {exc}")
        else:
            telemetry_enabled = registered = True
            telemetry = {
                "collector": setup.get("collector", {}),
                "codex": setup.get("codex", {}),
            }
            codex_warning = _codex_warning(setup.get("codex"))
            if codex_warning:
                warnings.append(codex_warning)

    launcher_bytes = _launcher_bytes(runtime, executable)
    manifest: dict[str, Any] = {
        "version": _VERSION,
        "runtime_kind": runtime.kind,
        "executable": str(executable),
        "source_root": (
            str(runtime.source_root) if runtime.source_root is not None else None
        ),
        "launcher": str(launcher),
        "launcher_sha256": _sha256(launcher_bytes),
        "telemetry_enabled": telemetry_enabled,
        "telemetry_registered": registered,
        "integrations": {},
    }
    prepared: list[tuple[Path, bytes, int]] = []
    for harness in USER_PATHS:
        path = _config_path(harness)
        merged, raw, mode = loaded[harness]
        command = _managed_command(harness)
        record = _record(previous, harness)
        entry: dict[str, Any] = {
            "path": str(path),
            "managed_command": command,
            "created_file": (
                bool(record.get("created_file")) if record is not None else raw is None
            ),
        }
        if harness == "claude-code" and claude_env:
            merged, telemetry_record = _claude_cost_env(merged, record, claude_env)
            entry.update(telemetry_record)
            conflicts = telemetry_record["telemetry_env_conflicts"]
            if conflicts:
                warnings.append(
                    "Claude cost collection needs attention. "
                    f"{path} already sets {', '.join(conflicts)}; "
                    "Joyride left the existing exporter unchanged."
                )
        elif harness == "claude-code" and record is not None:
            # Keep the ownership of values that an earlier install wrote, so
            # uninstall can still remove them.
            payload_env = merged.get("env")
            for key in (
                "telemetry_env_expected",
                "telemetry_env_keys",
                "telemetry_env_object_created",
            ):
                if key in record and isinstance(payload_env, dict):
                    entry[key] = record[key]
        desired_mode = (
            0o600
            if harness == "claude-code" and _has_claude_telemetry_credentials(merged)
            else mode
        )
        desired = _json_bytes(merged)
        manifest["integrations"][harness] = entry
        if raw != desired or (raw is not None and mode != desired_mode):
            prepared.append((path, desired, desired_mode))
    # Record ownership before the hooks exist so that an interrupted install
    # still leaves an uninstall that can remove every command it wrote. The
    # launcher comes next, so a hook never names a missing launcher.
    _write(user_manifest_path(), _json_bytes(manifest), mode=0o600)
    _write(launcher, launcher_bytes, mode=0o755)
    for path, desired, mode in prepared:
        _write(path, desired, mode=mode)
    status = user_install_status()
    if telemetry_enabled:
        status["telemetry"] = {**telemetry, "claude": user_claude_cost_status()}
    status["warnings"] = list(dict.fromkeys([*warnings, *status["warnings"]]))
    return status


def uninstall_user_hooks() -> dict[str, Any]:
    """Remove only the machine-level hook commands that this installer wrote."""

    manifest = load_user_manifest()
    if manifest is None:
        return user_install_status()
    warnings: list[str] = []
    for harness in USER_PATHS:
        record = _record(manifest, harness)
        command = record.get("managed_command") if record is not None else None
        if not isinstance(command, str) or _MARKER not in command:
            continue
        path = _config_path(harness)
        payload, raw, mode, _atime, _mtime = _load_json_config(path)
        if raw is None:
            continue
        created_file = bool(record.get("created_file"))
        # Pruning the containers that become empty undoes exactly what the
        # merge created, because the merge is what added each managed event.
        updated, removed = _remove_native_hooks(
            payload, command, _EVENTS[harness], created_file=True
        )
        if removed == 0:
            warnings.append(f"Managed hook entries were already absent from {path}.")
        if harness == "claude-code":
            updated = _remove_claude_telemetry_env(updated, record)
        if created_file and not updated:
            path.unlink()
        elif updated != payload:
            _write(path, _json_bytes(updated), mode=mode)
    telemetry_pending = False
    if (
        manifest.get("telemetry_registered") is True
        or manifest.get("telemetry_enabled") is True
    ):
        from . import telemetry_setup

        try:
            result = telemetry_setup.unregister_telemetry(machine_telemetry_id())
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            warnings.append(f"Cost collection could not be removed: {exc}")
            telemetry_pending = True
        else:
            codex = result.get("codex") if isinstance(result, dict) else None
            if isinstance(codex, dict) and codex.get("state") == "conflict":
                warnings.append(
                    f"Cost collection could not be removed: {codex.get('message')}"
                )
                telemetry_pending = True
    launcher = launcher_path()
    try:
        raw_launcher, _mode, _atime, _mtime = _file_details(launcher)
    except FileNotFoundError:
        pass
    else:
        if _MARKER.encode("utf-8") in raw_launcher:
            launcher.unlink()
    if telemetry_pending:
        # The hooks are gone, but the machine still owns cost collection. Keep
        # only that ownership, so a later uninstall can finish the cleanup.
        _write(user_manifest_path(), _json_bytes(_telemetry_owner()), mode=0o600)
    else:
        user_manifest_path().unlink(missing_ok=True)
    status = user_install_status()
    status["warnings"] = warnings
    return status


def _runtime_problem(manifest: dict[str, Any]) -> str | None:
    """Name the missing runtime that the managed hook command starts, if any."""

    executable, source_root = manifest.get("executable"), manifest.get("source_root")
    if not isinstance(executable, str) or not os.access(executable, os.X_OK):
        return (
            f"The Joyride runtime in the machine hook command is missing: {executable}. "
            "Run joyride install --user again."
        )
    if source_root is not None and not (isinstance(source_root, str) and os.path.isdir(source_root)):
        return (
            f"The Joyride source in the machine hook command is missing: {source_root}. "
            "Run joyride install --user again."
        )
    return None


def user_hook_covers(harness: str, event: str | None) -> bool:
    """Return whether one healthy machine hook owns this event."""

    if event not in _EVENTS.get(harness, ()):
        return False
    try:
        manifest = load_user_manifest()
        record = _record(manifest, harness)
        command = record.get("managed_command") if record is not None else None
        if not isinstance(command, str) or manifest is None:
            return False
        # A machine hook whose launcher or runtime is gone records nothing, so
        # it owns no event.
        if not _launcher_healthy(manifest):
            return False
        healthy, _message = _config_health(
            _config_path(harness), command, _EVENTS[harness], harness, only_event=event
        )
        return healthy
    except (OSError, ValueError):
        return False


def user_hook_health() -> dict[str, dict[str, Any]]:
    """Report each harness's machine-level hook state without changing it."""

    try:
        manifest = load_user_manifest()
    except (OSError, ValueError) as exc:
        message = str(exc)
        return {
            harness: {
                "installed": False,
                "state": "needs-attention",
                "message": message,
            }
            for harness in USER_PATHS
        }
    health: dict[str, dict[str, Any]] = {}
    for harness in USER_PATHS:
        record = _record(manifest, harness)
        command = record.get("managed_command") if record is not None else None
        if not isinstance(command, str):
            health[harness] = {
                "installed": False,
                "state": "not-installed",
                "message": "Machine-level hooks are not installed.",
            }
            continue
        healthy, message = _config_health(
            _config_path(harness), command, _EVENTS[harness], harness
        )
        runtime_problem = _runtime_problem(manifest)
        if runtime_problem is None and not _launcher_healthy(manifest):
            runtime_problem = (
                f"The Joyride hook launcher is missing or changed: {launcher_path()}. "
                "Run joyride install --user again."
            )
        if healthy and runtime_problem is not None:
            healthy, message = False, runtime_problem
        health[harness] = {
            "installed": healthy,
            "state": "enabled" if healthy else "needs-attention",
            "message": message,
        }
    return health


def user_claude_cost_status(manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    """Report whether Claude Code exports cost events to the local collector."""

    try:
        selected = manifest if manifest is not None else load_user_manifest()
        if selected is None or selected.get("telemetry_enabled") is not True:
            return {"enabled": False, "state": "disabled"}
        record = _record(selected, "claude-code")
        if record is None:
            raise ValueError("Claude telemetry integration is missing.")
        conflicts = record.get("telemetry_env_conflicts")
        if isinstance(conflicts, list) and conflicts:
            raise ValueError(
                "Existing OTel settings were kept: " + ", ".join(map(str, conflicts))
            )
        expected = record.get("telemetry_env_expected")
        if not isinstance(expected, dict) or not expected:
            raise ValueError("Claude telemetry settings are missing.")
        payload, raw, _mode, _atime, _mtime = _load_json_config(
            _config_path("claude-code")
        )
        env = payload.get("env") if raw is not None else None
        if not isinstance(env, dict):
            raise ValueError("Claude telemetry environment is missing.")
        mismatches = [key for key, value in expected.items() if env.get(key) != value]
        if mismatches:
            raise ValueError(
                "Claude telemetry conflicts with existing environment keys: "
                + ", ".join(sorted(mismatches))
            )
        return {"enabled": True, "state": "enabled"}
    except (OSError, ValueError) as exc:
        return {"enabled": False, "state": "needs-attention", "message": str(exc)}


def user_install_status() -> dict[str, Any]:
    """Return read-only machine-level installation state."""

    harnesses = user_hook_health()
    return {
        "installed": all(item["installed"] for item in harnesses.values()),
        "scope": "user",
        "manifest_path": str(user_manifest_path()),
        "harnesses": harnesses,
        "warnings": [],
        "notice": _NOTICE,
    }


def running_agent_sessions(listing: str | None = None) -> list[dict[str, Any]]:
    """Name the Claude Code and Codex processes that must restart to load hooks.

    ``listing`` is ``ps -Ao pid=,ppid=,args=`` output. A child of a process
    that already counts, such as a native binary behind its Node launcher, is
    the same session and is not listed again.
    """

    if listing is None:
        try:
            listing = subprocess.run(
                ["ps", "-Ao", "pid=,ppid=,args="],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
                check=False,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return []
    matched: dict[int, tuple[int, str]] = {}
    for line in listing.splitlines():
        fields = line.split(None, 2)
        if len(fields) < 3 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        pid, parent = int(fields[0]), int(fields[1])
        if pid == os.getpid():
            continue
        arguments = fields[2].split()
        command = os.path.basename(arguments[0])
        if command in _NODE_COMMANDS and len(arguments) > 1:
            command = os.path.basename(arguments[1])
        name = _AGENT_COMMANDS.get(command)
        if name is not None:
            matched[pid] = (parent, name)
    return [
        {"name": name, "pid": pid}
        for pid, (parent, name) in sorted(matched.items())
        if parent not in matched
    ]


__all__ = [
    "install_user_hooks",
    "launcher_path",
    "load_user_manifest",
    "machine_telemetry_id",
    "running_agent_sessions",
    "uninstall_user_hooks",
    "user_claude_cost_status",
    "user_hook_covers",
    "user_hook_health",
    "user_install_status",
    "user_manifest_path",
]
