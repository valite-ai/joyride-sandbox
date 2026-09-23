"""Codex hook trust entries for the hook commands that Joyride writes.

Codex runs a hook from a ``hooks.json`` file only after the user reviews it in
``/hooks``. It stores that review in the user ``config.toml`` as one
``[hooks.state."<key>"]`` table with a ``trusted_hash``. Codex 0.155.1
(``codex-rs/hooks/src/engine/discovery.rs``) builds both values like this:

* key: ``<resolved hooks.json path>:<event label>:<group index>:<handler index>``
* hash: ``sha256:`` plus the SHA-256 of the canonical JSON of
  ``{"event_name": label, "matcher": matcher, "hooks": [handler]}``. The
  handler holds ``type``, ``command``, ``timeout``, and ``async``. Events that
  take no matcher drop it, and Interrupt and SessionEnd clamp the timeout to
  one to three seconds.

Joyride writes the entries for its own handlers in one marked block per
``hooks.json`` file and edits nothing else in ``config.toml``. If a later Codex
release computes another hash, Codex reports the hook as modified and asks for
a review, as it does today.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import tomllib
from typing import Any, Mapping


_BLOCK_BEGIN = "# >>> joyride-codex-hook-trust-v1"
_BLOCK_END = "# <<< joyride-codex-hook-trust-v1"
_LABELS = {
    "PreToolUse": "pre_tool_use",
    "PermissionRequest": "permission_request",
    "PostToolUse": "post_tool_use",
    "PreCompact": "pre_compact",
    "PostCompact": "post_compact",
    "SessionStart": "session_start",
    "SessionEnd": "session_end",
    "UserPromptSubmit": "user_prompt_submit",
    "SubagentStart": "subagent_start",
    "SubagentStop": "subagent_stop",
    "Stop": "stop",
    "Interrupt": "interrupt",
}
_WITHOUT_MATCHER = {"UserPromptSubmit", "Stop", "Interrupt"}
_SHORT_TIMEOUT = {"SessionEnd", "Interrupt"}
_MAX_CONFIG_BYTES = 2 * 1024 * 1024


def trust_hash(event: str, matcher: str | None, handler: Mapping[str, Any]) -> str:
    """Return the hash that Codex records when a user trusts one handler."""

    timeout = handler.get("timeout")
    timeout = timeout if isinstance(timeout, int) and not isinstance(timeout, bool) else 600
    if event in _SHORT_TIMEOUT:
        timeout = min(max(timeout, 1), 3)
    else:
        timeout = max(timeout, 1)
    normalized: dict[str, Any] = {
        "type": "command",
        "command": handler["command"],
        "timeout": timeout,
        "async": handler.get("async") is True,
    }
    if isinstance(handler.get("statusMessage"), str):
        normalized["statusMessage"] = handler["statusMessage"]
    identity: dict[str, Any] = {"event_name": _LABELS[event], "hooks": [normalized]}
    if matcher is not None and event not in _WITHOUT_MATCHER:
        identity["matcher"] = matcher
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def trust_entries(
    hooks_path: Path, payload: Mapping[str, Any], command: str
) -> dict[str, str]:
    """Return the trust key and hash of every handler that runs ``command``."""

    source = hooks_path.parent.resolve() / hooks_path.name
    entries: dict[str, str] = {}
    hooks = payload.get("hooks")
    if not isinstance(hooks, Mapping):
        return entries
    for event, groups in hooks.items():
        if event not in _LABELS or not isinstance(groups, list):
            continue
        for group_index, group in enumerate(groups):
            if not isinstance(group, Mapping) or not isinstance(group.get("hooks"), list):
                continue
            matcher = group.get("matcher") if isinstance(group.get("matcher"), str) else None
            for handler_index, handler in enumerate(group["hooks"]):
                if (
                    isinstance(handler, Mapping)
                    and handler.get("type") == "command"
                    and handler.get("command") == command
                ):
                    key = f"{source}:{_LABELS[event]}:{group_index}:{handler_index}"
                    entries[key] = trust_hash(event, matcher, handler)
    return entries


def _begin(hooks_path: Path) -> str:
    return f"{_BLOCK_BEGIN} {hooks_path.parent.resolve() / hooks_path.name}"


def _read(path: Path) -> tuple[str | None, int]:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return None, 0o600
    if not path.is_file() or path.is_symlink() or details.st_size > _MAX_CONFIG_BYTES:
        raise ValueError(f"Refusing to edit {path}.")
    return path.read_bytes().decode("utf-8"), details.st_mode & 0o777


def _write(path: Path, text: str, mode: int) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.joyride-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(text)
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _plan(
    text: str, hooks_path: Path, entries: Mapping[str, str], owned: str | None
) -> tuple[str, str | None, list[str]]:
    """Return the new file text, the block it holds, and keys left untrusted.

    Raises ValueError when no block can be written.
    """

    begin = _begin(hooks_path)
    if begin in text:
        if not owned or text.count(owned) != 1:
            raise ValueError("The Joyride trust block was edited, so it was left unchanged.")
        text = text.replace(owned, "", 1)
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"The Codex configuration is not valid TOML: {exc}") from exc
    hooks = parsed.get("hooks")
    state = hooks.get("state") if isinstance(hooks, dict) else None
    existing = state if isinstance(state, dict) else {}
    # A table that the user or Codex already wrote cannot be defined twice.
    # One that holds another hash leaves that Joyride hook untrusted.
    kept = {key: value for key, value in entries.items() if key not in existing}
    stale = [
        key
        for key, value in entries.items()
        if key in existing
        and not (isinstance(existing[key], dict) and existing[key].get("trusted_hash") == value)
    ]
    if not kept:
        return text, None, stale
    lines = [begin]
    for key, value in kept.items():
        lines += [f"[hooks.state.{json.dumps(key)}]", f"trusted_hash = {json.dumps(value)}"]
    lines.append(_BLOCK_END)
    # The owned text holds the blank line before the block too, so removal
    # restores the file byte for byte.
    if not text or text.endswith("\n\n"):
        separator = ""
    else:
        separator = "\n" if text.endswith("\n") else "\n\n"
    block = separator + "\n".join(lines) + "\n"
    updated = text + block
    tomllib.loads(updated)
    return updated, block, stale


def apply_trust(
    config_path: Path,
    hooks_path: Path,
    entries: Mapping[str, str],
    owned: str | None,
) -> tuple[str | None, str | None]:
    """Write the trust block and return it with an optional warning.

    ``owned`` is the block an earlier call returned. It is replaced only while
    it is still byte-for-byte in the file, so a review decision that Codex
    wrote into it is kept.
    """

    try:
        text, mode = _read(config_path)
        current = text or ""
        updated, block, stale = _plan(current, hooks_path, entries, owned)
    except (OSError, UnicodeDecodeError, ValueError, tomllib.TOMLDecodeError) as exc:
        return owned, f"Codex hook trust was not written to {config_path}: {exc}"
    if updated != current:
        _write(config_path, updated, mode)
    if stale:
        return block, (
            f"{config_path} already holds another trust record for "
            f"{len(stale)} Joyride hook(s), so Codex treats them as changed."
        )
    return block, None


def remove_trust(
    config_path: Path, owned: str | None, *, delete_empty: bool = False
) -> str | None:
    """Remove the block that ``apply_trust`` wrote, and return a warning if kept.

    ``delete_empty`` removes a file that Joyride created once nothing remains.
    """

    if not owned:
        return None
    try:
        text, mode = _read(config_path)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return f"Codex hook trust was not removed from {config_path}: {exc}"
    if text is None:
        return None
    if text.count(owned) != 1:
        return f"The Joyride hook trust in {config_path} changed, so it was kept."
    remaining = text.replace(owned, "", 1)
    if delete_empty and not remaining.strip():
        config_path.unlink()
    else:
        _write(config_path, remaining, mode)
    return None


__all__ = ["apply_trust", "remove_trust", "trust_entries", "trust_hash"]
