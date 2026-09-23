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

Joyride appends the entries for its own handlers after one label comment and
edits nothing else in ``config.toml``. Codex edits the same file: it adds
``[projects."<path>"]`` tables and review decisions, and it can put a new table
anywhere. So Joyride owns a table by its key and the hash that Joyride wrote,
not by the bytes around it. Removal takes out only the ``trusted_hash`` line
that Joyride wrote, and the table header when nothing else is left in it.

If a later Codex release computes another hash, Codex reports the hook as
modified and asks for a review, as it does today.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import tomllib
from typing import Any, Mapping


_LABEL = "# joyride-codex-hook-trust-v2"
# Earlier releases wrapped the tables in these two comments and owned the bytes
# between them. Codex put its own tables before the end comment.
_LEGACY_BEGIN = "# >>> joyride-codex-hook-trust-v1"
_LEGACY_END = "# <<< joyride-codex-hook-trust-v1"
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


def _label(hooks_path: Path) -> str:
    return f"{_LABEL} {hooks_path.parent.resolve() / hooks_path.name}"


def _header(key: str) -> str:
    return f"[hooks.state.{json.dumps(key)}]"


def _value(value: str) -> str:
    return f"trusted_hash = {json.dumps(value)}"


def _owned(owned: Any) -> tuple[str | None, dict[str, str]]:
    """Return the appended text and the entries that an earlier call wrote.

    An earlier release recorded only the block text. Its entries are the
    tables inside it.
    """

    if isinstance(owned, str):
        try:
            state = tomllib.loads(owned).get("hooks", {}).get("state", {})
        except (tomllib.TOMLDecodeError, AttributeError):
            return owned, {}
        entries = {
            key: value["trusted_hash"]
            for key, value in state.items()
            if isinstance(value, dict) and isinstance(value.get("trusted_hash"), str)
        }
        return owned, entries
    if isinstance(owned, Mapping):
        block = owned.get("block") if isinstance(owned.get("block"), str) else None
        entries = owned.get("entries")
        if isinstance(entries, Mapping):
            return block, {
                str(key): value for key, value in entries.items() if isinstance(value, str)
            }
        return block, {}
    return None, {}


def _state(parsed: Mapping[str, Any]) -> dict[str, Any]:
    hooks = parsed.get("hooks")
    state = hooks.get("state") if isinstance(hooks, dict) else None
    return state if isinstance(state, dict) else {}


def _intact(parsed: Mapping[str, Any], entries: Mapping[str, str]) -> dict[str, str]:
    """Return the entries whose table still holds the hash that Joyride wrote."""

    state = _state(parsed)
    return {
        key: value
        for key, value in entries.items()
        if isinstance(state.get(key), dict) and state[key].get("trusted_hash") == value
    }


def _without(parsed: dict[str, Any], keys: Any) -> dict[str, Any]:
    """Return ``parsed`` without the trusted hashes of ``keys``, as TOML sees it."""

    result = json.loads(json.dumps(parsed, default=str))
    hooks = result.get("hooks")
    state = hooks.get("state") if isinstance(hooks, dict) else None
    if isinstance(state, dict):
        for key in keys:
            table = state.get(key)
            if isinstance(table, dict):
                table.pop("trusted_hash", None)
                if not table:
                    del state[key]
    return _pruned(result)


def _pruned(parsed: dict[str, Any]) -> dict[str, Any]:
    """Drop empty ``hooks.state`` and ``hooks`` tables, which carry no meaning."""

    result = json.loads(json.dumps(parsed, default=str))
    hooks = result.get("hooks")
    if isinstance(hooks, dict):
        if hooks.get("state") == {}:
            del hooks["state"]
        if not hooks:
            del result["hooks"]
    return result


def _strip(text: str, entries: Mapping[str, str], labels: set[str]) -> str:
    """Remove Joyride's label lines and the trusted hashes that it still owns.

    A table header goes too when nothing but blank lines and comments remain
    in its table, so a review decision that Codex wrote stays in place.
    """

    owned = {_header(key): _value(value) for key, value in entries.items()}
    lines = text.splitlines(keepends=True)
    kept: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped in labels or stripped == _LEGACY_END:
            index += 1
            continue
        if stripped not in owned:
            kept.append(line)
            index += 1
            continue
        end = index + 1
        while end < len(lines) and not lines[end].lstrip().startswith("["):
            end += 1
        body = [
            item
            for item in lines[index + 1 : end]
            if item.strip() not in labels and item.strip() != _LEGACY_END
        ]
        rest = [item for item in body if item.strip() != owned[stripped]]
        meaningful = [item for item in rest if item.strip() and not item.lstrip().startswith("#")]
        if meaningful or len(rest) == len(body):
            kept.append(line)
        kept.extend(rest)
        index = end
    return "".join(kept)


def _remove(
    text: str, hooks_path: Path | None, owned: Any, keep: Mapping[str, str] | None = None
) -> str:
    """Return ``text`` without the trust that ``owned`` names.

    The hashes in ``keep`` stay where they are. Raises ValueError when the
    result would change anything else.
    """

    block, entries = _owned(owned)
    parsed = tomllib.loads(text)
    intact = {
        key: value for key, value in _intact(parsed, entries).items() if key not in (keep or {})
    }
    labels = {line.strip() for line in (block or "").splitlines() if line.startswith("# ")}
    if hooks_path is not None:
        labels |= {_label(hooks_path), f"{_LEGACY_BEGIN} {hooks_path.parent.resolve() / hooks_path.name}"}
    whole = (
        block is not None
        and text.count(block) == 1
        and _LEGACY_END not in text
        and intact == entries == _owned(block)[1]
    )
    if whole:
        updated = text.replace(block, "", 1)
    else:
        updated = _strip(text, intact, labels)
    if _pruned(tomllib.loads(updated)) != _without(parsed, intact):
        raise ValueError("Joyride could not separate its trust entries from other settings.")
    return updated


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
    text: str, hooks_path: Path, entries: Mapping[str, str], owned: Any
) -> tuple[str, dict[str, Any] | None, list[str]]:
    """Return the new file text, what Joyride owns in it, and untrusted keys.

    Raises ValueError when no trust can be written.
    """

    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"The Codex configuration is not valid TOML: {exc}") from exc
    block, previous = _owned(owned)
    legacy = isinstance(owned, str) or _LEGACY_END in text
    if not legacy and previous and _intact(parsed, previous) == dict(entries):
        # Everything is in place. Tables that Codex added since do not matter.
        return text, {"block": block, "entries": dict(entries)}, []
    # A Joyride table that also holds a review decision that Codex wrote stays
    # in place with its hash. Removing only the hash would revoke the trust.
    state = _state(parsed)
    in_place = {
        key: value
        for key, value in _intact(parsed, previous).items()
        if entries.get(key) == value and set(state[key]) != {"trusted_hash"}
    }
    text = _remove(text, hooks_path, owned, keep=in_place)
    existing = _state(tomllib.loads(text))
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
        return text, ({"block": None, "entries": in_place} if in_place else None), stale
    lines = [_label(hooks_path)]
    for key, value in kept.items():
        lines += [_header(key), _value(value)]
    # The owned text holds the blank line before the tables too, so removal
    # restores an untouched file byte for byte.
    if not text or text.endswith("\n\n"):
        separator = ""
    else:
        separator = "\n" if text.endswith("\n") else "\n\n"
    appended = separator + "\n".join(lines) + "\n"
    updated = text + appended
    tomllib.loads(updated)
    return updated, {"block": appended, "entries": {**in_place, **kept}}, stale


def apply_trust(
    config_path: Path,
    hooks_path: Path,
    entries: Mapping[str, str],
    owned: Any,
) -> tuple[Any, str | None]:
    """Write the trust entries and return what Joyride owns with a warning.

    ``owned`` is what an earlier call returned, or the block text that an
    earlier release recorded. Tables that Codex added, and review decisions
    that it wrote into a Joyride table, are kept.
    """

    try:
        text, mode = _read(config_path)
        current = text or ""
        updated, result, stale = _plan(current, hooks_path, entries, owned)
    except (OSError, UnicodeDecodeError, ValueError, tomllib.TOMLDecodeError) as exc:
        # The caller keeps what it owned, so a later uninstall can remove it.
        return owned, f"Codex hook trust was not written to {config_path}: {exc}"
    if updated != current:
        _write(config_path, updated, mode)
    if stale:
        return result, (
            f"{config_path} already holds another trust record for "
            f"{len(stale)} Joyride hook(s), so Codex treats them as changed."
        )
    return result, None


def remove_trust(
    config_path: Path,
    owned: Any,
    *,
    hooks_path: Path | None = None,
    delete_empty: bool = False,
) -> str | None:
    """Remove the trust that ``apply_trust`` wrote, and return a warning if kept.

    ``delete_empty`` removes a file that Joyride created once nothing remains.
    """

    _block, entries = _owned(owned)
    if not entries:
        return None
    try:
        text, mode = _read(config_path)
        if text is None:
            return None
        remaining = _remove(text, hooks_path, owned)
    except (OSError, UnicodeDecodeError, ValueError, tomllib.TOMLDecodeError) as exc:
        return f"The Joyride hook trust in {config_path} was kept: {exc}"
    if remaining == text:
        return None
    if delete_empty and not remaining.strip():
        config_path.unlink()
    else:
        _write(config_path, remaining, mode)
    return None


__all__ = ["apply_trust", "remove_trust", "trust_entries", "trust_hash"]
