"""Run the installed native hooks end to end in a throwaway repository.

The self-test reads the Joyride hook commands that Claude Code and Codex would
run, from the machine configuration and from the selected repository, and
runs them exactly as written against a temporary Git repository. It edits one
file between the pre-tool and post-tool events, commits, and reads the note
back. Only the temporary directory changes, and it is removed at the end.
"""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any
import uuid

from .install import _INTEGRATION_PATHS, _MARKER, _load_json_config
from .notes import NOTES_REF
from .user_install import USER_PATHS


_HARNESSES = (("claude-code", "Claude Code"), ("codex", "Codex"))
_DEFAULT_TIMEOUT_SECONDS = 30
_MAX_TIMEOUT_SECONDS = 60
_WORKFLOW = Path(".github/workflows/attribution-footer.yml")
_SAMPLE = "self_test.py"
_BEFORE = "value = 1\n"
_AFTER = "value = 2\nresult = value * 2\n"
# Variables that would point Git at another repository than the temporary one.
_GIT_LOCATION_VARIABLES = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
)
# Variables that replace the configuration Git reads, so a value such as
# `core.hooksPath` would override the temporary repository's own hooks.
# `GIT_CONFIG_GLOBAL` and `GIT_CONFIG_SYSTEM` stay: they only choose files.
_GIT_CONFIG_VARIABLES = ("GIT_CONFIG", "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT")
_GIT_CONFIG_PREFIXES = ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")


def _is_git_override(name: str) -> bool:
    return (
        name in _GIT_LOCATION_VARIABLES
        or name in _GIT_CONFIG_VARIABLES
        or name.startswith(_GIT_CONFIG_PREFIXES)
    )


def _handlers(path: Path, event: str, tool: str) -> list[dict[str, Any]]:
    """Return the Joyride handlers a harness runs for one event and tool."""

    payload, _raw, _mode, _atime, _mtime = _load_json_config(path)
    groups = payload.get("hooks", {}).get(event, []) if isinstance(payload.get("hooks"), dict) else []
    selected: list[dict[str, Any]] = []
    for group in groups if isinstance(groups, list) else []:
        if not isinstance(group, dict):
            continue
        matcher = group.get("matcher")
        if isinstance(matcher, str) and matcher not in {"", "*"}:
            try:
                if re.fullmatch(matcher, tool) is None:
                    continue
            except re.error:
                continue
        for handler in group.get("hooks", []):
            if (
                isinstance(handler, dict)
                and isinstance(handler.get("command"), str)
                and _MARKER in handler["command"]
            ):
                selected.append(handler)
    return selected


def _claude_hooks_disabled(repo_root: Path | None) -> Path | None:
    """Return the settings file whose ``disableAllHooks`` stops Claude Code hooks."""

    paths = (
        [repo_root / ".claude" / "settings.local.json", repo_root / ".claude" / "settings.json"]
        if repo_root is not None
        else []
    )
    paths.append(Path.home() / USER_PATHS["claude-code"])
    # The most specific file that sets the key decides, as in Claude Code.
    for path in paths:
        payload, _raw, _mode, _atime, _mtime = _load_json_config(path)
        value = payload.get("disableAllHooks")
        if isinstance(value, bool):
            return path if value else None
    return None


def _repository_root(repo: Path | None) -> Path | None:
    if repo is None:
        return None
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        env=_environment(),
    )
    return Path(result.stdout.strip()) if result.returncode == 0 else None


def _environment() -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not _is_git_override(name)
    }
    # The self-test never starts the cost collector.
    environment["HARNESS_ATTRIBUTION_DISABLE_TELEMETRY"] = "1"
    return environment


@contextlib.contextmanager
def _without_git_location() -> Iterator[None]:
    """Keep in-process Git calls on the temporary repository."""

    saved = {name: os.environ.pop(name) for name in list(os.environ) if _is_git_override(name)}
    try:
        yield
    finally:
        os.environ.update(saved)


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        env=_environment(),
    )


def _payload(harness: str, repository: Path, event: str, session: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "session_id": session,
        "cwd": str(repository),
        "hook_event_name": event,
        "model": "joyride-self-test",
    }
    if harness == "claude-code":
        payload.update(
            {
                "tool_name": "Edit",
                "tool_use_id": "toolu_joyride_self_test",
                "permission_mode": "default",
                "tool_input": {
                    "file_path": str(repository / _SAMPLE),
                    "old_string": _BEFORE,
                    "new_string": _AFTER,
                },
            }
        )
        if event == "PostToolUse":
            payload["tool_response"] = {"filePath": str(repository / _SAMPLE), "success": True}
    else:
        payload.update(
            {
                "turn_id": "joyride-self-test-turn",
                "tool_name": "apply_patch",
                "tool_use_id": "call_joyride_self_test",
                "tool_input": {
                    "command": (
                        "*** Begin Patch\n"
                        f"*** Update File: {_SAMPLE}\n"
                        "@@\n"
                        f"-{_BEFORE.rstrip()}\n"
                        + "".join(f"+{line}\n" for line in _AFTER.splitlines())
                        + "*** End Patch\n"
                    )
                },
            }
        )
        if event == "PostToolUse":
            payload["tool_response"] = "Success. Updated the following files:\nM self_test.py\n"
    return payload


def _run_handlers(
    handlers: list[dict[str, Any]],
    event: str,
    payload: dict[str, Any],
    repository: Path,
) -> str | None:
    """Run each handler as the harness would; return the first failure."""

    for handler in handlers:
        timeout = handler.get("timeout")
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            timeout = _DEFAULT_TIMEOUT_SECONDS
        timeout = min(timeout, _MAX_TIMEOUT_SECONDS)
        try:
            result = subprocess.run(
                handler["command"],
                shell=True,
                executable="/bin/sh",
                cwd=repository,
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_environment(),
            )
        except subprocess.TimeoutExpired:
            return f"The {event} hook did not finish within {timeout:g} seconds."
        if result.returncode != 0:
            detail = next(
                (line.strip() for line in result.stderr.splitlines() if line.strip()), ""
            )
            reason = f"The {event} hook exited with status {result.returncode}"
            return f"{reason}: {detail}" if detail else f"{reason}."
    return None


def _base_repository(workspace: Path) -> Path | str:
    repository = workspace / "repository"
    repository.mkdir()
    empty_hooks = workspace / "no-hooks"
    empty_hooks.mkdir()
    for arguments in (
        ("init", "--quiet", "--initial-branch=main"),
        ("config", "user.name", "Joyride self-test"),
        ("config", "user.email", "self-test@joyride.invalid"),
        ("config", "commit.gpgsign", "false"),
        # The Joyride installer chains to the hooks path in effect when it runs.
        # An empty local path keeps the user's global hooks out of both commits.
        ("config", "core.hooksPath", str(empty_hooks)),
    ):
        result = _git(repository, *arguments)
        if result.returncode:
            return f"Git could not prepare the temporary repository: {result.stderr.strip()}"
    (repository / "README.md").write_text("Joyride self-test\n", encoding="utf-8")
    (repository / _SAMPLE).write_text(_BEFORE, encoding="utf-8")
    # The workflow file enrolls this clone for machine-level hooks.
    workflow = repository / _WORKFLOW
    workflow.parent.mkdir(parents=True)
    workflow.write_text("name: Coding attribution footer\n", encoding="utf-8")
    _git(repository, "add", "-A")
    result = _git(repository, "commit", "--quiet", "-m", "Base")
    if result.returncode:
        return f"Git could not commit in the temporary repository: {result.stderr.strip()}"
    return repository


def _verdict(repository: Path, harness: str, name: str) -> tuple[bool, str]:
    shown = _git(repository, "notes", f"--ref={NOTES_REF}", "show", "HEAD")
    if shown.returncode:
        enabled = (repository / ".git" / "attribution" / "install.json").is_file()
        if not enabled:
            return False, "The hooks ran, but Joyride did not enable capture in the temporary repository."
        return False, "The commit has no Joyride note. The Git post-commit hook did not record it."
    try:
        note = json.loads(shown.stdout)
    except json.JSONDecodeError:
        return False, "The Joyride note of the commit is not valid JSON."
    sessions = {
        session.get("id"): session
        for session in note.get("sessions", [])
        if isinstance(session, dict)
    }
    attributed = 0
    added = 0
    for file_record in note.get("files", []):
        added += int(file_record.get("added_lines", 0))
        for line_range in file_record.get("ranges", []):
            session = sessions.get(line_range.get("session_id"), {})
            if harness in {session.get("harness_id"), session.get("harness")} or name == session.get(
                "harness"
            ):
                attributed += int(line_range["end"]) - int(line_range["start"]) + 1
    if not attributed:
        return False, f"The commit note attributes no lines to {name}."
    return True, f"{attributed} of {added} added lines were attributed to {name}."


def _run_harness(harness: str, name: str, repo_root: Path | None) -> dict[str, Any]:
    tool = "Edit" if harness == "claude-code" else "apply_patch"
    sources: list[tuple[str, Path]] = [("machine", Path.home() / USER_PATHS[harness])]
    if repo_root is not None:
        sources.append(("repository", repo_root / _INTEGRATION_PATHS[harness]))
    handlers: dict[str, list[dict[str, Any]]] = {"PreToolUse": [], "PostToolUse": []}
    scopes: list[str] = []
    for scope, path in sources:
        try:
            found = {event: _handlers(path, event, tool) for event in handlers}
        except (OSError, ValueError) as exc:
            return {
                "harness": harness,
                "name": name,
                "passed": False,
                "scopes": scopes,
                "reason": str(exc),
                "commit_line": None,
            }
        if any(found.values()):
            scopes.append(scope)
            for event, items in found.items():
                handlers[event].extend(items)
    result: dict[str, Any] = {
        "harness": harness,
        "name": name,
        "passed": False,
        "scopes": scopes,
        "reason": "",
        "commit_line": None,
    }
    if not handlers["PreToolUse"] or not handlers["PostToolUse"]:
        result["reason"] = (
            f"No Joyride edit hooks are installed for {name}. Run joyride install --user."
        )
        return result
    if harness == "claude-code":
        try:
            disabled = _claude_hooks_disabled(repo_root)
        except (OSError, ValueError) as exc:
            result["reason"] = str(exc)
            return result
        if disabled is not None:
            result["reason"] = (
                f"Claude Code runs no hooks, because {disabled} sets disableAllHooks to true."
            )
            return result

    workspace = Path(tempfile.mkdtemp(prefix="joyride-self-test-"))
    try:
        repository = _base_repository(workspace)
        if isinstance(repository, str):
            result["reason"] = repository
            return result
        if scopes == ["repository"]:
            # Repository hooks capture only in an enabled worktree. Enable
            # this temporary one the way a machine install enables a clone.
            from .install import _install_worktree

            with _without_git_location():
                _install_worktree(repository, native_hooks=False)
        session = f"joyride-self-test-{uuid.uuid4().hex}"
        failure = _run_handlers(
            handlers["PreToolUse"],
            "PreToolUse",
            _payload(harness, repository, "PreToolUse", session),
            repository,
        )
        if failure is None:
            (repository / _SAMPLE).write_text(_AFTER, encoding="utf-8")
            failure = _run_handlers(
                handlers["PostToolUse"],
                "PostToolUse",
                _payload(harness, repository, "PostToolUse", session),
                repository,
            )
        if failure is not None:
            result["reason"] = failure
            return result
        _git(repository, "add", _SAMPLE)
        committed = _git(repository, "commit", "--quiet", "-m", "Joyride self-test edit")
        if committed.returncode:
            result["reason"] = f"Git could not commit the test edit: {committed.stderr.strip()}"
            return result
        result["commit_line"] = next(
            (
                line.strip()
                for line in committed.stderr.splitlines()
                if line.startswith("Joyride: ")
            ),
            None,
        )
        result["passed"], result["reason"] = _verdict(repository, harness, name)
        return result
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def run_self_test(repo: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Run each harness's installed hooks against a temporary repository."""

    repo_root = _repository_root(Path(repo)) if repo is not None else None
    harnesses = [_run_harness(harness, name, repo_root) for harness, name in _HARNESSES]
    return {
        "passed": all(item["passed"] for item in harnesses),
        "harnesses": harnesses,
    }


__all__ = ["run_self_test"]
