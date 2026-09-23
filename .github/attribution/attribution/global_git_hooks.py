"""Machine-wide Git hooks for the machine install.

The machine install points the global ``core.hooksPath`` at
``~/.attribution/git-hooks``. Git then runs these hooks in every repository
that sets no ``core.hooksPath`` of its own, so an enrolled clone needs no Git
hook setup of its own. Each hook first runs the hook that the repository would
have run without Joyride: the previous global hooks directory if there was
one, otherwise the repository's ``hooks`` directory. It passes the arguments
and the input through and returns that hook's exit status.

Only post-checkout, post-commit, post-merge, post-rewrite, and pre-push then
start Joyride, and only in a clone whose Git directory holds a Joyride
manifest. Every other hook, such as reference-transaction, stays a shell
script that starts no Python and, in most repositories, no Git process.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from typing import Any

from .install import _STANDARD_GIT_HOOKS


_MARKER = "# attribution-managed-v1"
_JOYRIDE_EVENTS = ("post-checkout", "post-commit", "post-merge", "post-rewrite", "pre-push")
_SUMMARY_EVENTS = {"post-commit", "post-merge"}
_METADATA_PUSH_GUARD = "ATTRIBUTION_METADATA_PUSH"
# post-index-change is a Git hook that the per-clone installer does not list.
HOOK_NAMES = tuple(sorted(_STANDARD_GIT_HOOKS | {"post-index-change"}))
# A push-to-checkout hook replaces Git's own updateInstead worktree update, so
# a script that exits 0 changes behavior. For every other name, a script that
# exits 0 acts like no hook. This one is written only to chain a hook that
# the previous global hooks directory holds.
_REPLACES_DEFAULT = {"push-to-checkout"}

_COMMON = r'''#!/bin/sh
# attribution-managed-v1
# Joyride machine Git hook. It runs the hook that this repository would run
# without Joyride, then records the event in a clone that Joyride tracks.
_joyride_common=
_joyride_find_common() {
  if [ -n "${GIT_COMMON_DIR:-}" ]; then
    _joyride_common=$GIT_COMMON_DIR
    return
  fi
  if [ -n "${GIT_DIR:-}" ]; then
    _joyride_git=$GIT_DIR
  elif [ -d .git ]; then
    _joyride_git=.git
  else
    _joyride_git=$(git rev-parse --git-dir 2>/dev/null) || _joyride_git=
  fi
  _joyride_common=$_joyride_git
  if [ -n "$_joyride_git" ] && [ -f "$_joyride_git/commondir" ]; then
    _joyride_link=
    IFS= read -r _joyride_link <"$_joyride_git/commondir" || :
    case "$_joyride_link" in
      /*) _joyride_common=$_joyride_link ;;
      ?*) _joyride_common=$_joyride_git/$_joyride_link ;;
    esac
  fi
}
_joyride_previous=@PREVIOUS@
if [ -n "$_joyride_previous" ]; then
  case "$_joyride_previous" in
    /*) _joyride_dir=$_joyride_previous ;;
    *) _joyride_dir=$PWD/$_joyride_previous ;;
  esac
else
  _joyride_find_common
  _joyride_dir=${_joyride_common:+$_joyride_common/hooks}
fi
_joyride_original=${_joyride_dir:+$_joyride_dir/@NAME@}
'''

_PASS_THROUGH = r'''if [ -n "$_joyride_original" ] && [ -x "$_joyride_original" ]; then
  exec "$_joyride_original" "$@"
fi
exit 0
'''

_REPAIR = r'''_joyride_stamp=@STAMP@
if [ ! -f "$_joyride_stamp" ] || [ ! -f @CLAUDE@ ] || [ ! -f @CODEX@ ] \
  || [ ! -f @CODEX_CONFIG@ ] || [ @CLAUDE@ -nt "$_joyride_stamp" ] \
  || [ @CODEX@ -nt "$_joyride_stamp" ] || [ @CODEX_CONFIG@ -nt "$_joyride_stamp" ]; then
  @LAUNCHER@ _repair-hooks >/dev/null 2>&1 || :
fi
[ -n "$_joyride_common" ] || _joyride_find_common
'''

_EVENT = r'''_joyride_status=0
if [ -n "$_joyride_original" ] && [ -x "$_joyride_original" ]; then
  "$_joyride_original" "$@" || _joyride_status=$?
fi
@REPAIR@if [ -n "$_joyride_common" ] && [ -f "$_joyride_common/attribution/install.json" ]; then
  @ACTION@ || :
fi
exit "$_joyride_status"
'''

_BUFFERED = r'''@GUARD@umask 077
_joyride_input=$(mktemp "${TMPDIR:-/tmp}/joyride-hook-input.XXXXXXXXXX" 2>/dev/null) || _joyride_input=
if [ -z "$_joyride_input" ]; then
  if [ -n "$_joyride_original" ] && [ -x "$_joyride_original" ]; then
    exec "$_joyride_original" "$@"
  fi
  exit 0
fi
trap 'rm -f -- "$_joyride_input"' 0
trap 'exit 129' 1
trap 'exit 130' 2
trap 'exit 143' 15
if ! cat >"$_joyride_input"; then
  if [ -n "$_joyride_original" ] && [ -x "$_joyride_original" ]; then
    printf '%s\n' 'Joyride: hook input could not be buffered; the original hook was not run.' >&2
    exit 1
  fi
  exit 0
fi
_joyride_status=0
if [ -n "$_joyride_original" ] && [ -x "$_joyride_original" ]; then
  "$_joyride_original" "$@" <"$_joyride_input" || _joyride_status=$?
fi
if [ "$_joyride_status" -ne 0 ]; then
  exit "$_joyride_status"
fi
@REPAIR@if [ -n "$_joyride_common" ] && [ -f "$_joyride_common/attribution/install.json" ]; then
  @ACTION@
fi
exit 0
'''

_GUARD = r'''if [ "${@GUARD_NAME@:-}" = "1" ]; then
  if [ -n "$_joyride_original" ] && [ -x "$_joyride_original" ]; then
    exec "$_joyride_original" "$@"
  fi
  exit 0
fi
'''


def hooks_directory(home: Path) -> Path:
    """Return the Joyride-owned directory that the global core.hooksPath names."""

    return home / ".attribution" / "git-hooks"


def stamp_path(home: Path) -> Path:
    """Return the file whose time marks the last check of the agent settings."""

    return home / ".attribution" / "agent-settings.checked"


def _git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "config", "--global", *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def _hooks_path_entries() -> list[tuple[str, str | None, str]]:
    """Return the scope, file, and value of each machine-wide core.hooksPath.

    The entries follow Git's read order, so the last one is the value that a
    repository without its own setting uses. The read covers the system file,
    ``~/.gitconfig``, the XDG file, and files that they include. ``--global``
    alone can miss the XDG file and ignores includes.
    """

    home = Path.home()
    result = subprocess.run(
        [
            "git", "config", "--show-scope", "--show-origin", "--includes",
            "--null", "--get-all", "core.hooksPath",
        ],
        cwd=home if home.is_dir() else "/",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode == 1:
        return []
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(message or "Could not read the global Git hooks path.")
    fields = [item.decode("utf-8", errors="surrogateescape") for item in result.stdout.split(b"\0")]
    entries: list[tuple[str, str | None, str]] = []
    for index in range(0, len(fields) - 2, 3):
        scope, origin, value = fields[index : index + 3]
        if scope in {"system", "global"}:
            entries.append((scope, origin[5:] if origin.startswith("file:") else None, value))
    return entries


def _conditional_hooks_paths() -> list[str]:
    """Return files that set core.hooksPath through an ``includeIf`` condition.

    Git applies such a value only in matching repositories, so one machine
    value cannot record and chain it.
    """

    home = Path.home()
    result = subprocess.run(
        [
            "git", "config", "--show-scope", "--show-origin", "--includes",
            "--null", "--get-regexp", r"^includeif\..*\.path$",
        ],
        cwd=home if home.is_dir() else "/",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode == 1:
        return []
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(message or "Could not read the Git include settings.")
    fields = [item.decode("utf-8", errors="surrogateescape") for item in result.stdout.split(b"\0")]
    found: list[str] = []
    for index in range(0, len(fields) - 2, 3):
        scope, origin, entry = fields[index : index + 3]
        if scope not in {"system", "global"} or "\n" not in entry:
            continue
        value = os.path.expanduser(entry.split("\n", 1)[1])
        base = Path(origin[5:]).parent if origin.startswith("file:") else home
        included = Path(value) if os.path.isabs(value) else base / value
        values = subprocess.run(
            ["git", "config", "--file", str(included), "--includes", "--get-all", "core.hooksPath"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if values.returncode == 0 and values.stdout.strip():
            found.append(str(included))
    return found


def global_hooks_path() -> tuple[bool, str | None]:
    """Return whether a machine-wide core.hooksPath applies, and its raw value."""

    entries = _hooks_path_entries()
    return (True, entries[-1][2]) if entries else (False, None)


def _file_values(target: str) -> list[str]:
    result = subprocess.run(
        ["git", "config", "--file", target, "--null", "--get-all", "core.hooksPath"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode not in {0, 1}:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(message or f"Could not read {target}.")
    return [
        item.decode("utf-8", errors="surrogateescape")
        for item in result.stdout.split(b"\0")
        if item
    ]


def _set_hooks_path(value: str) -> dict[str, Any]:
    """Point Git at ``value`` from the file that holds the effective setting.

    Replacing the value where it lives keeps its place in Git's read order and
    adds no second value to another file. Returns what restore needs.
    """

    entries = _hooks_path_entries()
    if entries and entries[-1][0] == "global" and entries[-1][1]:
        target = entries[-1][1]
        values = _file_values(target)
        subprocess.run(
            ["git", "config", "--file", target, "--replace-all", "core.hooksPath", value],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return {"file": target, "file_values": values}
    _git("--replace-all", "core.hooksPath", value)
    written = _hooks_path_entries()
    target = written[-1][1] if written and written[-1][2] == value else None
    return {"file": target, "file_values": []}


def _restore_hooks_path(previous: dict[str, Any]) -> None:
    """Put back the values that the file held before the install."""

    target = previous.get("file")
    values = previous.get("file_values")
    if not isinstance(target, str) or not isinstance(values, list):
        if previous.get("present") is True and isinstance(previous.get("value"), str):
            _git("--replace-all", "core.hooksPath", previous["value"])
        else:
            _git("--unset-all", "core.hooksPath", check=False)
        return
    command = ["git", "config", "--file", target]
    if not values:
        result = subprocess.run(
            [*command, "--unset-all", "core.hooksPath"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode not in {0, 5}:
            raise subprocess.CalledProcessError(result.returncode, result.args)
        return
    subprocess.run([*command, "--replace-all", "core.hooksPath", str(values[0])], check=True)
    for value in values[1:]:
        subprocess.run([*command, "--add", "core.hooksPath", str(value)], check=True)


def _summary_supported() -> bool:
    """Return whether this runtime's ``_git-hook`` accepts ``--summary-fd``."""

    from .cli import parser

    try:
        _arguments, unknown = parser().parse_known_args(
            ["_git-hook", "post-commit", "--summary-fd", "3"]
        )
    except SystemExit:
        return False
    return not unknown


def _action(name: str, launcher: str, summary: bool) -> str:
    if name == "pre-push":
        return (
            f'{launcher} _share "$@" <"$_joyride_input" || '
            "printf '%s\\n' 'Joyride: metadata hook failed; branch push will continue.' >&2"
        )
    if name == "post-rewrite":
        return (
            f'{launcher} --repo "$PWD" _git-hook post-rewrite <"$_joyride_input" '
            ">/dev/null 2>&1 || :"
        )
    if summary and name in _SUMMARY_EVENTS:
        # Only the one-line commit summary, on descriptor 3, reaches the terminal.
        return f'{launcher} --repo "$PWD" _git-hook {name} --summary-fd 3 3>&2 >/dev/null 2>&1'
    return f'{launcher} --repo "$PWD" _git-hook {name} >/dev/null 2>&1'


def proxy_bytes(
    name: str,
    *,
    previous: str | None,
    launcher: Path,
    watched: tuple[Path, Path, Path],
    stamp: Path,
    summary: bool,
) -> bytes:
    """Return the shell script for one hook name."""

    chained = os.path.expanduser(previous) if previous else ""
    text = _COMMON.replace("@PREVIOUS@", shlex.quote(chained)).replace("@NAME@", name)
    if name not in _JOYRIDE_EVENTS:
        return (text + _PASS_THROUGH).encode("utf-8")
    quoted = shlex.quote(str(launcher))
    claude, codex, codex_config = (shlex.quote(str(path)) for path in watched)
    repair = (
        _REPAIR.replace("@STAMP@", shlex.quote(str(stamp)))
        .replace("@CLAUDE@", claude)
        .replace("@CODEX_CONFIG@", codex_config)
        .replace("@CODEX@", codex)
        .replace("@LAUNCHER@", quoted)
    )
    if name in {"pre-push", "post-rewrite"}:
        guard = (
            _GUARD.replace("@GUARD_NAME@", _METADATA_PUSH_GUARD) if name == "pre-push" else ""
        )
        body = _BUFFERED.replace("@GUARD@", guard)
    else:
        body = _EVENT
    body = body.replace("@REPAIR@", repair).replace("@ACTION@", _action(name, quoted, summary))
    return (text + body).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.joyride-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
        os.chmod(temporary, 0o755)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def install_git_hooks(
    home: Path,
    launcher: Path,
    watched: tuple[Path, Path, Path],
    record: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Write the hook scripts and point the global core.hooksPath at them.

    ``record`` is the result of an earlier install. It keeps the value that
    the first install replaced, so a reinstall never records Joyride's own
    directory as the value to restore.
    """

    directory = hooks_directory(home)
    ours = str(directory)
    conditional = _conditional_hooks_paths()
    if conditional:
        # A per-clone dispatcher reads the value that applies in each clone,
        # so no repository loses its hooks. Remove a machine value that an
        # earlier install wrote, because it would override that value.
        warnings = uninstall_git_hooks(record) if isinstance(record, dict) else []
        return None, [
            *warnings,
            "An includeIf section sets core.hooksPath in "
            + ", ".join(conditional)
            + ", so Joyride left the global Git hooks path unchanged. Each clone "
            "gets its own Git hook dispatcher instead.",
        ]
    current = global_hooks_path()
    if isinstance(record, dict) and isinstance(record.get("previous"), dict):
        previous = record["previous"]
        if current not in {(True, ours), (previous.get("present") is True, previous.get("value"))}:
            return record, [
                "The global core.hooksPath changed after installation, so Joyride "
                "left it unchanged and its Git hooks do not run."
            ]
    elif current == (True, ours):
        previous = {"present": False, "value": None}
    else:
        previous = {"present": current[0], "value": current[1]}
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"Expected a directory at {directory}.")
    summary = _summary_supported()
    chained = (
        Path(os.path.expanduser(previous["value"]))
        if previous.get("present") and isinstance(previous.get("value"), str)
        else None
    )
    names = [
        name
        for name in HOOK_NAMES
        if name not in _REPLACES_DEFAULT
        or (
            chained is not None
            and chained.is_absolute()
            and os.access(chained / name, os.X_OK)
        )
    ]
    hashes: dict[str, str] = {}
    for name in sorted(set(HOOK_NAMES) - set(names)):
        # A script that an earlier install wrote would still replace Git's
        # default for this name.
        path = directory / name
        try:
            existing = path.read_bytes()
        except FileNotFoundError:
            continue
        if _MARKER.encode("utf-8") in existing:
            path.unlink()
    for name in names:
        data = proxy_bytes(
            name,
            previous=previous["value"] if previous.get("present") else None,
            launcher=launcher,
            watched=watched,
            stamp=stamp_path(home),
            summary=summary,
        )
        path = directory / name
        try:
            existing = path.read_bytes()
        except FileNotFoundError:
            existing = None
        if existing is not None and _MARKER.encode("utf-8") not in existing:
            raise ValueError(f"Refusing to replace an unowned Git hook at {path}.")
        if existing != data or not os.access(path, os.X_OK):
            _write(path, data)
        hashes[name] = _sha256(data)
    if current != (True, ours):
        previous = {**previous, **_set_hooks_path(ours)}
    if global_hooks_path() != (True, ours):
        return {"path": ours, "previous": previous, "proxy_hashes": hashes}, [
            "A later Git configuration value overrides the Joyride Git hooks."
        ]
    return {"path": ours, "previous": previous, "proxy_hashes": hashes}, []


def uninstall_git_hooks(record: dict[str, Any] | None) -> list[str]:
    """Restore the previous global core.hooksPath and remove the scripts."""

    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        return []
    warnings: list[str] = []
    directory = Path(record["path"])
    previous = record.get("previous") if isinstance(record.get("previous"), dict) else {}
    if global_hooks_path() == (True, record["path"]):
        # Restore Git before deleting anything, so a failure leaves a record
        # that a later uninstall can still restore from.
        _restore_hooks_path(previous)
    else:
        warnings.append(
            "The global core.hooksPath changed after installation; the newer value was kept."
        )
    hashes = record.get("proxy_hashes") if isinstance(record.get("proxy_hashes"), dict) else {}
    for name, expected in hashes.items():
        path = directory / str(name)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            continue
        if _sha256(data) == expected:
            path.unlink()
        else:
            warnings.append(f"Modified Git hook was kept: {path}.")
    try:
        directory.rmdir()
    except OSError:
        pass
    return warnings


def git_hooks_health(record: dict[str, Any] | None) -> dict[str, Any]:
    """Report whether Git runs the machine hooks, without changing anything."""

    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        return {"installed": False, "state": "not-installed", "message": "Machine Git hooks are not installed."}
    try:
        if global_hooks_path() != (True, record["path"]):
            raise ValueError("The global core.hooksPath no longer names the Joyride Git hooks.")
        hashes = record.get("proxy_hashes")
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError("The machine Git hook record is invalid.")
        for name, expected in hashes.items():
            path = Path(record["path"]) / str(name)
            if _sha256(path.read_bytes()) != expected or not os.access(path, os.X_OK):
                raise ValueError(f"The machine Git hook {name} was changed.")
    except (OSError, ValueError) as exc:
        return {"installed": False, "state": "needs-attention", "message": str(exc)}
    return {"installed": True, "state": "enabled", "message": None}


def runs_machine_hooks(repository_root: Path, record: dict[str, Any] | None) -> bool:
    """Return whether Git in this repository runs the machine hooks."""

    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        return False
    result = subprocess.run(
        ["git", "-C", str(repository_root), "config", "--path", "--get", "core.hooksPath"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    configured = result.stdout.decode("utf-8", errors="surrogateescape").rstrip("\n")
    return result.returncode == 0 and configured == record["path"]


__all__ = [
    "HOOK_NAMES",
    "git_hooks_health",
    "global_hooks_path",
    "hooks_directory",
    "install_git_hooks",
    "proxy_bytes",
    "runs_machine_hooks",
    "stamp_path",
    "uninstall_git_hooks",
]
