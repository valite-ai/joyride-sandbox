"""Count how much of each merged PR's code still stands on the default branch.

The trusted workflow runs this module once a week from the default branch. For
each merged PR with a metadata snapshot, it rebuilds who owned each added line
at the PR head, the way the PR report does, follows those lines into the merge
result, and counts how many of them the default branch still held 7, 30, and
90 days after the merge. Only counts, commit IDs, and session IDs leave the
runner: no path, source line, commit message, or title.

A line is identified by its blame origin at the merge result: the commit that
introduced it, the path there, and its line number there. Only an origin that
the PR itself put on the default branch counts: its commits for a merge
commit, the squash commit, or the rebased commits. That identity is the same
for all three merge methods, and blame keeps it through whitespace-only edits
(-w), moves within a file (-M), and file renames. A later blame of the default
branch that still finds an identity finds a surviving line.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.client import HTTPException
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Iterator
import unicodedata
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, build_opener

from .github_footer import (
    MAX_SNAPSHOT_BYTES, _METADATA_PREFIX, _NoRedirect, _OID, _git_environment, _host_url,
    _ingest_url, _oidc_token, _report_git_environment, _repository_name,
)
from .notes import _BLAME_HEADER, _decode_blame_path
from .report import (
    _Warnings, _decode_git_path, _merge_session, _normalise_session, _parse_note,
)
from .runtime import system_subprocess_environment


SCHEMA = "harness-attribution/pr-survival@1"
CHECKPOINT_DAYS = (7, 30, 90)
WINDOW_DAYS = 120
MAX_PULL_REQUESTS = 40
MAX_BATCH = 100
# The hosted service accepts at most 3 MiB in one upload.
MAX_BATCH_BYTES = 2_500_000
MAX_SESSIONS = 200
MAX_SESSION_BYTES = 512 * 1024
MAX_SKIP_REASONS = 8
# The changed files one PR measures, as the PR report inspects them.
MAX_FILES = 2000
MAX_FILE_BYTES = 1024 * 1024
# Blob bytes one run may read at the head, the merge, and each checkpoint.
MAX_RUN_BYTES = 128 * 1024 * 1024
# The workflow job allows 30 minutes; measuring stops early enough to upload.
MAX_RUN_SECONDS = 20 * 60
MAX_PR_SECONDS = 5 * 60
_MAX_PAGES = 20
_MAX_HTTP_BYTES = 16 * 1024 * 1024
_MAX_NOTE_BYTES = 2 * 1024 * 1024
_GIT_TIMEOUT = 120
_PATH_CHUNK = 500
_RENAME_LIMIT = "-l32767"
_USER_AGENT = "attribution-survival"
_EVENTS = frozenset({"schedule", "workflow_dispatch"})
_REGULAR = frozenset({"100644", "100755"})
_BUCKETS = {"ai": "ai", "manual": "human"}
_DELIVERY = re.compile(r"^[A-Za-z0-9_.:-]{1,180}$")
# The hosted service accepts a skip reason only as a short lowercase word.
_SKIP_REASON = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_REVERTS_COMMIT = re.compile(r"This reverts commit ([0-9a-fA-F]{7,64})")
_REVERTS_PR = re.compile(
    r"^\s*Reverts ([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)#([0-9]{1,10})\s*$", re.MULTILINE
)
_REVERT_BRANCH = re.compile(
    r"^Merge pull request #[0-9]{1,10} from \S+/revert-([0-9]{1,10})-", re.MULTILINE
)
_REVERT_TITLE = re.compile(r'^Revert "(.*)"(?: \(#[0-9]{1,10}\))?$')
_deadline: ContextVar[float | None] = ContextVar("survival_deadline", default=None)


class _OutOfTime(ValueError):
    """A PR ran past its time budget; a later run measures it."""


class _OutOfBudget(Exception):
    """This run cannot afford a PR; a later run measures it."""


@dataclass(frozen=True)
class MergedPullRequest:
    """One same-repository PR merged into the default branch."""

    number: int
    base_sha: str
    head_sha: str
    merge_commit_sha: str
    merged_at: datetime
    title: str


class _Budget:
    def __init__(self, limit: int = MAX_RUN_BYTES, seconds: float = MAX_RUN_SECONDS):
        self.left = limit
        self.deadline = time.monotonic() + seconds

    def take(self, size: int) -> None:
        if size > self.left:
            raise _OutOfBudget
        self.left -= size

    def seconds_left(self) -> float:
        return self.deadline - time.monotonic()


@contextmanager
def _time_limit(seconds: float) -> Iterator[None]:
    """Bound every Git call inside the block by one shared deadline."""

    token = _deadline.set(time.monotonic() + seconds)
    try:
        yield
    finally:
        _deadline.reset(token)


def _git(repo: Path, *args: str, check: bool = True, data: bytes | None = None):
    timeout: float = _GIT_TIMEOUT
    deadline = _deadline.get()
    if deadline is not None:
        timeout = min(timeout, deadline - time.monotonic())
        if timeout <= 0:
            raise _OutOfTime("Git ran past its time budget.")
    try:
        result = subprocess.run(
            ["git", "-c", "core.quotePath=false", "-C", str(repo), *args],
            input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=system_subprocess_environment(), timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise _OutOfTime("Git ran past its time budget.") from exc
    except OSError as exc:
        raise ValueError(f"Git {args[0]} could not run.") from exc
    if check and result.returncode:
        # Remote and path errors can hold repository text; do not echo it.
        raise ValueError(f"Git {args[0]} failed.")
    return result


def _time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _encode(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode("ascii")


def merged_pull_requests(
    items: Iterable[Any], *, repository_id: int, cutoff: datetime
) -> list[MergedPullRequest]:
    """Return the valid same-repository PRs merged at or after ``cutoff``."""

    result: list[MergedPullRequest] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        number = item.get("number")
        merged_at = _time(item.get("merged_at"))
        head = item.get("head") if isinstance(item.get("head"), dict) else {}
        base = item.get("base") if isinstance(item.get("base"), dict) else {}
        head_repo = head.get("repo") if isinstance(head.get("repo"), dict) else {}
        shas = (base.get("sha"), head.get("sha"), item.get("merge_commit_sha"))
        if (
            type(number) is not int or not 0 < number <= 999_999_999 or merged_at is None
            or merged_at < cutoff
            or any(not isinstance(sha, str) or not _OID.fullmatch(sha) for sha in shas)
            # Hosted reports do not accept fork PRs, so neither does survival.
            or head_repo.get("id") != repository_id
        ):
            continue
        title = item.get("title")
        result.append(MergedPullRequest(
            number, shas[0], shas[1], shas[2], merged_at,
            title if isinstance(title, str) and len(title) <= 1024 else "",
        ))
    return result


def due_checkpoints(
    pull_requests: Iterable[MergedPullRequest],
    measured: set[tuple[int, int]],
    now: datetime,
) -> list[tuple[MergedPullRequest, tuple[int, ...]]]:
    """Return each PR's unmeasured due checkpoints, the oldest due first."""

    cutoff = now - timedelta(days=WINDOW_DAYS)
    due: list[tuple[datetime, int, int, MergedPullRequest]] = []
    seen: set[int] = set()
    for pull_request in pull_requests:
        if (
            pull_request.number in seen
            or not cutoff <= pull_request.merged_at <= now
        ):
            continue
        seen.add(pull_request.number)
        for days in CHECKPOINT_DAYS:
            when = pull_request.merged_at + timedelta(days=days)
            if when <= now and (pull_request.number, days) not in measured:
                due.append((when, pull_request.number, days, pull_request))
    due.sort(key=lambda item: item[:3])
    selected: dict[int, tuple[MergedPullRequest, list[int]]] = {}
    for _when, number, days, pull_request in due:
        selected.setdefault(number, (pull_request, []))[1].append(days)
    return [(pull_request, tuple(days)) for pull_request, days in selected.values()]


def _chunks(values: list[Any]) -> Iterator[list[Any]]:
    for start in range(0, len(values), _PATH_CHUNK):
        yield values[start:start + _PATH_CHUNK]


def _entries(repo: Path, commit: str, paths: Iterable[str]) -> dict[str, tuple[str, str, int]]:
    """Return the mode, blob, and size of each named file that is a blob."""

    entries: dict[str, tuple[str, str, int]] = {}
    for names in _chunks(sorted(set(paths))):
        output = _git(
            repo, "--literal-pathspecs", "ls-tree", "-r", "-l", "-z", "--full-tree",
            commit, "--", *names,
        ).stdout
        for record in output.split(b"\0"):
            if not record:
                continue
            try:
                fields, raw_path = record.split(b"\t", 1)
                mode, kind, blob, size = fields.decode("ascii").split()
            except (ValueError, UnicodeError) as exc:
                raise ValueError("Git returned an invalid tree.") from exc
            if kind == "blob" and size.isdigit():
                entries[raw_path.decode("utf-8", "replace")] = (mode, blob, int(size))
    return entries


def _generated(repo: Path, paths: Iterable[str]) -> set[str]:
    """Return the paths the checkout's attributes mark ``linguist-generated``."""

    names = sorted(set(paths))
    if not names:
        return set()
    output = _git(
        repo, "check-attr", "-z", "--stdin", "linguist-generated",
        data=b"".join(name.encode("utf-8", "surrogateescape") + b"\0" for name in names),
    ).stdout.split(b"\0")
    return {
        output[index].decode("utf-8", "replace")
        for index in range(0, len(output) - 2, 3)
        if output[index + 2] in {b"set", b"true"}
    }


def _changes(repo: Path, *args: str) -> list[tuple[str | None, str | None]]:
    """Return the ``(old, new)`` path of each file one diff changes."""

    tokens = _git(
        repo, "diff", "--no-ext-diff", "--no-textconv", "--name-status", "-z", *args, "--",
    ).stdout.split(b"\0")
    changes: list[tuple[str | None, str | None]] = []
    index = 0
    while index < len(tokens) and tokens[index]:
        status = tokens[index][:1]
        paths = 2 if status in {b"R", b"C"} else 1
        if index + paths >= len(tokens):
            raise ValueError("Git returned an invalid change list.")
        names = [tokens[index + offset].decode("utf-8", "replace") for offset in range(1, paths + 1)]
        index += paths + 1
        if status == b"A":
            changes.append((None, names[0]))
        elif status == b"D":
            changes.append((names[0], None))
        else:
            changes.append((names[0], names[-1]))
    return changes


def _changed_text(
    repo: Path, merge_base: str, head: str
) -> tuple[list[tuple[str | None, str]], dict[str, int], dict[str, str]]:
    """Return the PR's changes, the size of each measurable file, and why others are skipped."""

    changes = [
        (old, new) for old, new in _changes(repo, "--find-renames", merge_base, head)
        if new is not None
    ]
    binary: set[str] = set()
    numstat = _git(
        repo, "diff", "--no-ext-diff", "--no-textconv", "--no-renames", "--numstat", "-z",
        merge_base, head, "--",
    ).stdout
    for record in numstat.split(b"\0"):
        fields = record.split(b"\t", 2)
        if len(fields) == 3 and fields[0] == b"-":
            binary.add(fields[2].decode("utf-8", "replace"))
    paths = [new for _old, new in changes]
    entries = _entries(repo, head, paths)
    generated = _generated(repo, paths)
    sizes: dict[str, int] = {}
    skipped: dict[str, str] = {}
    for path in paths:
        entry = entries.get(path)
        if entry is None:
            continue  # A submodule adds no text line.
        if entry[0] not in _REGULAR:
            skipped[path] = "not_regular"
        elif path in binary:
            skipped[path] = "binary"
        elif entry[2] > MAX_FILE_BYTES:
            skipped[path] = "too_large"
        elif path in generated:
            skipped[path] = "generated"
        elif len(sizes) >= MAX_FILES:
            skipped[path] = "limit"
        else:
            sizes[path] = entry[2]
    return changes, sizes, skipped


def _hunks(patch: bytes, *, new: bool = False) -> dict[str, list[tuple[int, int, int, int]]]:
    """Return the zero-context hunks of each file of one patch, by its old or new path."""

    result: dict[str, list[tuple[int, int, int, int]]] = {}
    paths: dict[str, str | None] = {"-": None, "+": None}
    old_left = new_left = 0
    for raw in patch.split(b"\n"):
        if old_left or new_left:
            if raw.startswith(b"-") and old_left:
                old_left -= 1
            elif raw.startswith(b"+") and new_left:
                new_left -= 1
            elif not raw.startswith(b"\\"):
                raise ValueError("Git returned an incomplete diff hunk.")
            continue
        line = raw.decode("utf-8", "replace")
        if line.startswith("diff --git "):
            paths = {"-": None, "+": None}
        elif line.startswith(("--- ", "+++ ")):
            # Git ends a header path that holds a space with a tab.
            value = _decode_git_path(line[4:].rstrip("\t"))
            prefix = "b/" if line[0] == "+" else "a/"
            paths[line[0]] = value[2:] if value.startswith(prefix) else None
        elif match := _HUNK.match(line):
            old_count = int(match[2]) if match[2] is not None else 1
            new_count = int(match[4]) if match[4] is not None else 1
            path = paths["+" if new else "-"]
            if path is not None:
                result.setdefault(path, []).append(
                    (int(match[1]), old_count, int(match[3]), new_count)
                )
            old_left, new_left = old_count, new_count
    if old_left or new_left:
        raise ValueError("Git returned an incomplete diff.")
    return result


def _diff_hunks(
    repo: Path, old: str, new: str, pairs: list[tuple[str | None, str]], *,
    by_new: bool, renames: bool,
) -> dict[str, list[tuple[int, int, int, int]]]:
    """Return the zero-context hunks of the named files, a bounded list of paths at a time."""

    result: dict[str, list[tuple[int, int, int, int]]] = {}
    for chunk in _chunks(pairs):
        names = sorted({path for pair in chunk for path in pair if path is not None})
        patch = _git(
            repo, "--literal-pathspecs", "diff", "--no-ext-diff", "--no-textconv",
            "--no-color", "--find-renames" if renames else "--no-renames", "--unified=0",
            "--src-prefix=a/", "--dst-prefix=b/", old, new, "--", *names,
        ).stdout
        result.update(_hunks(patch, new=by_new))
    return result


def _map_lines(
    lines: Iterable[int], hunks: list[tuple[int, int, int, int]]
) -> dict[int, int]:
    """Map old-side lines through zero-context hunks; a changed line maps to nothing."""

    mapped: dict[int, int] = {}
    ordered = sorted(hunks)
    offset = index = 0
    for line in sorted(lines):
        while index < len(ordered):
            start, count, _new_start, new_count = ordered[index]
            # A pure insertion follows old line ``start``; a change covers its lines.
            if (start if count == 0 else start + count - 1) >= line:
                break
            offset += new_count - count
            index += 1
        if index < len(ordered) and ordered[index][1] and ordered[index][0] <= line:
            continue
        mapped[line] = line + offset
    return mapped


def _origins(
    repo: Path, revision: str, path: str, *options: str
) -> dict[int, tuple[str, str, int]]:
    """Return the blame origin of each line of one file; a range stops at its base."""

    output = _git(
        repo, "--literal-pathspecs", "blame", "--porcelain", "--no-textconv", *options,
        revision, "--", path,
    ).stdout
    result: dict[int, tuple[str, str, int]] = {}
    # Porcelain names a commit's path once, or again when the path changes.
    names: dict[str, str] = {}
    records = output.split(b"\n")
    index = 0
    while index < len(records):
        match = _BLAME_HEADER.match(records[index])
        index += 1
        if match is None:
            continue
        origin = match.group(1).decode("ascii")
        while index < len(records):
            record = records[index]
            index += 1
            if record.startswith(b"filename "):
                names[origin] = _decode_blame_path(record[len(b"filename "):])
            if record.startswith(b"\t"):
                break
        result[int(match.group(3))] = (origin, names.get(origin, path), int(match.group(2)))
    return result


def _range(boundary: str | None, commit: str) -> str:
    return f"{boundary}..{commit}" if boundary else commit


def _owners(
    repo: Path, merge_base: str, head: str, notes: Any, added: dict[str, set[int]]
) -> dict[str, dict[int, tuple[str, str | None]]]:
    """Return the bucket and session of each added head line, as the PR report counts them."""

    if not isinstance(notes, list):
        raise ValueError("Shared attribution notes must be an array.")
    warnings = _Warnings()
    history = set(_git(repo, "rev-list", head).stdout.decode("ascii").split())
    by_commit: dict[str, dict[str, Any]] = {}
    sessions: dict[str, dict[str, Any]] = {}
    for raw in notes:
        if not isinstance(raw, dict) or not isinstance(raw.get("commit"), str):
            raise ValueError("Invalid shared attribution note.")
        encoded = json.dumps(raw)
        if len(encoded.encode()) > _MAX_NOTE_BYTES:
            raise ValueError("Shared attribution note exceeds the size limit.")
        note = _parse_note(encoded, raw["commit"], warnings=warnings)
        if note["commit"] not in history:
            continue
        if note["commit"] in by_commit:
            raise ValueError("Duplicate attribution note for a commit.")
        by_commit[note["commit"]] = note
        for value in note["sessions"]:
            if isinstance(value, dict):
                session = _normalise_session(value, source="Git notes", warnings=warnings)
                if session:
                    _merge_session(sessions, session, warnings)
    ranges = {
        (sha, item["path"]): item["ranges"]
        for sha, note in by_commit.items() for item in note["files"]
    }
    owners: dict[str, dict[int, tuple[str, str | None]]] = {}
    for path, numbers in added.items():
        # An added line never comes from before the merge base, so blame stops there.
        blame = _origins(repo, _range(merge_base, head), path)
        owned = owners.setdefault(path, {})
        for number in numbers:
            origin = blame.get(number)
            owner = None
            if origin:
                sha, original_path, original_line = origin
                owner = next((
                    item["session_id"] for item in ranges.get((sha, original_path), ())
                    if item["start"] <= original_line <= item["end"]
                ), None)
            session = sessions.get(owner) if owner else None
            owned[number] = (
                ("unknown", None) if session is None
                else (_BUCKETS[session["actor_kind"]], owner)
            )
    return owners


def _patch_ids(repo: Path, *log_arguments: str) -> dict[str, str]:
    output = _git(
        repo, "log", "-p", "--no-merges", "--no-color", "--no-ext-diff", "--no-textconv",
        *log_arguments,
    ).stdout
    ids: dict[str, str] = {}
    for line in _git(repo, "patch-id", "--stable", data=output).stdout.decode("ascii").splitlines():
        patch_id, _, commit = line.partition(" ")
        ids[commit] = patch_id
    return ids


def _introduced(
    repo: Path, merge: str, merge_base: str, head: str
) -> tuple[str | None, set[str]]:
    """Return the commit below the PR's commits on the default branch, and those commits.

    A merge commit brings the PR's commits along; a squash is one commit; a
    rebase puts one commit on the default branch for each PR commit, which
    matches it by patch ID.
    """

    parents = _git(repo, "rev-list", "--parents", "--max-count=1", merge).stdout.decode("ascii").split()[1:]
    oldest = merge
    singles = _git(repo, "rev-list", "--no-merges", f"{merge_base}..{head}").stdout.decode("ascii").split()
    if len(parents) == 1 and len(singles) > 1:
        chain = _git(
            repo, "rev-list", "--first-parent", f"--max-count={len(singles)}", merge
        ).stdout.decode("ascii").split()
        pr_ids = set(_patch_ids(repo, f"{merge_base}..{head}").values())
        chain_ids = _patch_ids(repo, "--first-parent", f"--max-count={len(singles)}", merge)
        for commit in chain[1:]:
            if chain_ids.get(commit) not in pr_ids:
                break
            oldest = commit
    below = _git(repo, "rev-list", "--parents", "--max-count=1", oldest).stdout.decode("ascii").split()[1:]
    boundary = below[0] if below else None
    introduced = set(_git(repo, "rev-list", _range(boundary, merge)).stdout.decode("ascii").split())
    return boundary, introduced


def _moves(
    repo: Path, merge: str, checkpoint: str, missing: dict[str, str]
) -> dict[str, str]:
    """Return where each missing PR file went, detecting renames of those files only."""

    raw = _git(
        repo, "diff", "--no-ext-diff", "--no-renames", "--raw", "--no-abbrev", "-z",
        "--diff-filter=A",
        merge, checkpoint, "--",
    ).stdout.split(b"\0")
    added: list[tuple[str, str]] = []
    for index in range(0, len(raw) - 1, 2):
        fields = raw[index].decode("ascii", "replace").split()
        if len(fields) == 5 and fields[1] in _REGULAR and _OID.fullmatch(fields[3]):
            added.append((raw[index + 1].decode("utf-8", "replace"), fields[3]))
    if not added:
        return {}
    sources = sorted(missing.items())

    def tree(entries: list[tuple[str, str]], prefix: str) -> str:
        # Stand-in names keep every path out of the command line.
        listing = "".join(
            f"100644 blob {blob}\t{prefix}{index}\n" for index, (_path, blob) in enumerate(entries)
        )
        return _git(repo, "mktree", data=listing.encode("ascii")).stdout.decode("ascii").strip()

    tokens = _git(
        repo, "diff-tree", "-r", "-z", "--name-status", "--find-renames", _RENAME_LIMIT,
        "--diff-filter=R", tree(sources, "s"), tree(added, "d"),
    ).stdout.split(b"\0")
    moved: dict[str, str] = {}
    for index in range(0, len(tokens) - 2, 3):
        source, target = tokens[index + 1].decode("ascii"), tokens[index + 2].decode("ascii")
        moved[sources[int(source[1:])][0]] = added[int(target[1:])][0]
    return moved


def _unmeasurable(repo: Path, entry: tuple[str, str, int], budget: _Budget) -> str | None:
    """Return why one blob cannot be blamed, charging its bytes to the run."""

    mode, blob, size = entry
    if mode not in _REGULAR:
        return "not_regular"
    if size > MAX_FILE_BYTES:
        return "too_large"
    budget.take(size)
    if b"\0" in _git(repo, "cat-file", "blob", blob).stdout:
        return "binary"
    return None


def _reverts(
    message: str, pull_request: MergedPullRequest, targets: set[str], full_name: str
) -> bool:
    """Say whether one commit message reverts the PR. A named commit decides alone."""

    named = [sha.lower() for sha in _REVERTS_COMMIT.findall(message)]
    if named:
        return any(target.startswith(sha) for sha in named for target in targets)
    if any(
        match[1].lower() == full_name.lower() and int(match[2]) == pull_request.number
        for match in _REVERTS_PR.finditer(message)
    ) or any(
        int(match[1]) == pull_request.number for match in _REVERT_BRANCH.finditer(message)
    ):
        return True
    subject = _REVERT_TITLE.match(message.strip().split("\n", 1)[0].strip())
    return bool(subject and pull_request.title and subject[1] == pull_request.title)


def _revert_commits(
    repo: Path, pull_request: MergedPullRequest, targets: set[str], tips: list[str],
    full_name: str,
) -> set[str]:
    """Return the default-branch commits after the merge, up to ``tips``, that revert the PR."""

    merge = pull_request.merge_commit_sha
    tips = [tip for tip in dict.fromkeys(tips) if tip != merge]
    if not tips:
        return set()
    output = _git(
        repo, "log", "--max-count=2000", "--regexp-ignore-case", "--grep=revert",
        "--format=%x1e%H%x1f%B", *tips, f"^{merge}",
    ).stdout.decode("utf-8", "replace")
    found: set[str] = set()
    for record in output.split("\x1e"):
        commit, _, message = record.partition("\x1f")
        if commit and _reverts(message, pull_request, targets, full_name):
            found.add(commit.strip())
    return found


def _session_id(value: str) -> bool:
    """Say whether the hosted service stores this session ID, by its own rule."""

    return 0 < len(value) <= 2048 and not any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in value
    )


def _measurement(
    pull_request: MergedPullRequest, days: int, checkpoint: str, measured_at: str,
    owned: dict[Any, tuple[str, str | None]], found: set[Any], reverted: bool,
    scanned: int, skipped: int, reasons: Iterable[str],
) -> dict[str, Any]:
    """Build one Contract C measurement; surviving lines are a subset of owned ones."""

    at_merge = {"ai": 0, "human": 0, "unknown": 0}
    remaining = dict(at_merge)
    sessions: dict[str, list[int]] = {}
    for identity, (bucket, session_id) in owned.items():
        alive = identity in found
        at_merge[bucket] += 1
        remaining[bucket] += alive
        if session_id is not None and _session_id(session_id):
            counts = sessions.setdefault(session_id, [0, 0])
            counts[0] += 1
            counts[1] += alive
    kept: list[dict[str, Any]] = []
    size = 2
    for session_id, counts in sorted(sessions.items(), key=lambda item: (-item[1][0], item[0])):
        item = {"session_id": session_id, "at_merge": counts[0], "remaining": counts[1]}
        size += len(_encode(item)) + 1
        if len(kept) >= MAX_SESSIONS or size > MAX_SESSION_BYTES:
            break
        kept.append(item)
    return {
        "pr_number": pull_request.number,
        "merge_commit_sha": pull_request.merge_commit_sha,
        "measured_commit_sha": checkpoint,
        "checkpoint_days": days,
        "measured_at": measured_at,
        "lines_at_merge": at_merge,
        "lines_remaining": remaining,
        "sessions": kept,
        "reverted": reverted,
        "coverage": {
            "files_scanned": scanned,
            "files_skipped": skipped,
            "skip_reasons": sorted(
                reason for reason in set(reasons) if _SKIP_REASON.fullmatch(reason)
            )[:MAX_SKIP_REASONS],
        },
    }


def _scope(repo: Path, base: str, head: str) -> tuple[str, list[str]]:
    bases = _git(repo, "merge-base", "--all", base, head).stdout.decode("ascii").split()
    if len(bases) != 1:
        raise ValueError("The PR must have one unambiguous merge base.")
    return bases[0], _git(repo, "rev-list", f"{bases[0]}..{head}").stdout.decode("ascii").split()


def _is_ancestor(repo: Path, ancestor: str, commit: str) -> bool:
    return not _git(repo, "merge-base", "--is-ancestor", ancestor, commit, check=False).returncode


def measure_pull_request(
    repo: Path,
    pull_request: MergedPullRequest,
    checkpoints: list[tuple[int, str]],
    *,
    notes: Any,
    full_name: str,
    measured_at: str,
    ignore_revs: Path | None = None,
    budget: _Budget | None = None,
) -> list[dict[str, Any]]:
    """Measure one PR at each ``(checkpoint_days, default-branch commit)``.

    A PR the byte budget cannot cover raises ``_OutOfBudget`` and yields
    nothing, so no measurement the budget cut is ever stored.
    """

    budget = budget or _Budget()
    merge, head = pull_request.merge_commit_sha, pull_request.head_sha
    merge_base, commits = _scope(repo, pull_request.base_sha, head)
    changes, sizes, skipped = _changed_text(repo, merge_base, head)
    tips = list(dict.fromkeys(commit for _days, commit in checkpoints))
    if sum(sizes.values()) * (2 + len(tips)) > budget.left:
        raise _OutOfBudget
    budget.take(sum(sizes.values()))
    added: dict[str, set[int]] = {}
    for path, hunks in _diff_hunks(
        repo, merge_base, head, [pair for pair in changes if pair[1] in sizes],
        by_new=True, renames=True,
    ).items():
        lines = {line for _old, _count, start, count in hunks for line in range(start, start + count)}
        if path in sizes and lines:
            added[path] = lines
    owners = _owners(repo, merge_base, head, notes, added)
    boundary, introduced = _introduced(repo, merge, merge_base, head)
    options = ["-w", "-M", *(("--ignore-revs-file", str(ignore_revs)) if ignore_revs else ())]
    reasons = set(skipped.values())

    # Follow each owned head line to the merge result and take its identity there.
    at_merge: dict[str, dict[tuple[str, str, int], tuple[str, str | None]]] = {}
    merge_entries = _entries(repo, merge, owners)
    hunks = _diff_hunks(
        repo, head, merge, [(path, path) for path in sorted(merge_entries)],
        by_new=False, renames=False,
    )
    for path, lines in sorted(owners.items()):
        entry = merge_entries.get(path)
        if entry is None:
            continue  # The merge result does not hold this file.
        reason = _unmeasurable(repo, entry, budget)
        if reason:
            skipped[path] = reason
            reasons.add(reason)
            continue
        mapped = _map_lines(lines, hunks.get(path, []))
        origins = _origins(repo, _range(boundary, merge), path, *options)
        # Two lines with one origin, or an origin the PR did not add, cannot be
        # told apart later, so neither is counted.
        shared = Counter(origins.values())
        identities: dict[tuple[str, str, int], tuple[str, str | None]] = {}
        for line, owner in sorted(lines.items()):
            origin = origins.get(mapped.get(line, 0))
            if origin is None:
                continue
            if origin[0] not in introduced or shared[origin] > 1:
                reasons.add("untraced_lines")
                continue
            identities[origin] = owner
        if identities:
            at_merge[path] = identities

    reverts = _revert_commits(repo, pull_request, introduced | set(commits), tips, full_name)
    measured: dict[str, tuple[Any, ...]] = {}
    for checkpoint in tips:
        entries = _entries(repo, checkpoint, at_merge)
        missing = {path: merge_entries[path][1] for path in at_merge if path not in entries}
        moved = _moves(repo, merge, checkpoint, missing) if missing else {}
        entries.update(_entries(repo, checkpoint, moved.values()))
        targets = {path: path if path in entries else moved.get(path) for path in at_merge}
        generated = _generated(repo, [target for target in targets.values() if target])
        owned: dict[tuple[str, tuple[str, str, int]], tuple[str, str | None]] = {}
        found: set[tuple[str, tuple[str, str, int]]] = set()
        scanned = 0
        late: dict[str, str] = {}
        for path, identities in at_merge.items():
            target = targets[path]
            if target is not None:
                reason = "generated" if target in generated else _unmeasurable(
                    repo, entries[target], budget
                )
                if reason:
                    late[path] = reason
                    continue
                alive = set(_origins(repo, _range(boundary, checkpoint), target, *options).values())
                found.update((path, origin) for origin in identities if origin in alive)
            # A file the default branch deleted is scanned and holds no line.
            scanned += 1
            owned.update(((path, origin), owner) for origin, owner in identities.items())
        reverted = any(
            commit == checkpoint or _is_ancestor(repo, commit, checkpoint) for commit in reverts
        )
        measured[checkpoint] = (
            owned, found, reverted, scanned, len(skipped) + len(late),
            reasons | set(late.values()),
        )
    return [
        _measurement(pull_request, days, checkpoint, measured_at, *measured[checkpoint])
        for days, checkpoint in checkpoints
    ]


def _checkpoint_commit(
    repo: Path, default_head: str, merge: str, when: datetime
) -> str | None:
    """Return the default-branch tip at ``when`` if it holds the merge, else nothing."""

    found = _git(
        repo, "rev-list", "--first-parent", "--max-count=1",
        f"--before={when.astimezone(timezone.utc):%Y-%m-%d %H:%M:%S} +0000", default_head,
    ).stdout.decode("ascii", "replace").strip()
    if _OID.fullmatch(found) and _is_ancestor(repo, merge, found):
        return found
    return None


def _ignore_revs(repo: Path, head: str, directory: Path) -> Path | None:
    """Copy the full commit IDs of the default branch's ``.git-blame-ignore-revs``."""

    entry = _entries(repo, head, [".git-blame-ignore-revs"]).get(".git-blame-ignore-revs")
    if entry is None or entry[0] not in _REGULAR or entry[2] > MAX_FILE_BYTES:
        return None
    text = _git(repo, "cat-file", "blob", entry[1]).stdout.decode("utf-8", "replace")
    revisions = [
        token for token in (
            line.split("#", 1)[0].strip().lower() for line in text.splitlines()
        ) if _OID.fullmatch(token)
    ]
    if not revisions:
        return None
    path = directory / "ignore-revs"
    path.write_text("\n".join(revisions) + "\n", encoding="ascii")
    return path


def _snapshot_heads(repo: Path, remote: str) -> dict[str, str]:
    """Return the metadata commit of each PR head that has a snapshot ref."""

    output = _git(repo, "ls-remote", "--refs", remote, _METADATA_PREFIX + "*").stdout
    heads: dict[str, str] = {}
    for line in output.decode("ascii", "replace").splitlines():
        commit, _, ref = line.partition("\t")
        head = ref[len(_METADATA_PREFIX):] if ref.startswith(_METADATA_PREFIX) else ""
        if _OID.fullmatch(commit) and _OID.fullmatch(head):
            heads[head] = commit
    return heads


def _snapshot_notes(store: Path, remote: str, head: str, expected: str) -> list[Any]:
    """Fetch one snapshot as data and return the notes of its ``attribution.json``."""

    local = f"refs/attribution/metadata/{head}"
    _git(
        store, "fetch", "--quiet", "--no-tags", "--no-recurse-submodules", "--depth=1",
        remote, f"+{_METADATA_PREFIX}{head}:{local}",
    )
    fetched = _git(store, "rev-parse", f"{local}^{{commit}}").stdout.decode("ascii").strip()
    if fetched != expected:
        raise ValueError("Joyride metadata changed while it was being fetched.")
    # Survival needs the notes only; traces beside them are never read.
    found = [
        record.split(b"\t", 1)[0].decode("ascii", "replace").split()
        for record in _git(store, "ls-tree", "-z", fetched).stdout.split(b"\0")
        if record.endswith(b"\tattribution.json")
    ]
    if len(found) != 1 or found[0][:2] != ["100644", "blob"] or not _OID.fullmatch(found[0][2]):
        raise ValueError("Joyride metadata must hold one attribution.json file.")
    blob = found[0][2]
    if int(_git(store, "cat-file", "-s", blob).stdout) > MAX_SNAPSHOT_BYTES:
        raise ValueError("Joyride metadata exceeds the 8 MiB limit.")
    try:
        payload = json.loads(_git(store, "cat-file", "blob", blob).stdout)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Joyride metadata is not valid JSON.") from exc
    if (
        not isinstance(payload, dict)
        or type(payload.get("version")) is not int or payload["version"] != 1
        or payload.get("head_commit") != head
        or not isinstance(payload.get("notes"), list)
        or any(not isinstance(note, dict) for note in payload["notes"])
    ):
        raise ValueError("Joyride metadata does not match this PR head.")
    return payload["notes"]


def _has_commit(repo: Path, sha: str) -> bool:
    return not _git(repo, "cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode


def _ensure_commit(repo: Path, remote: str, sha: str) -> None:
    if not _has_commit(repo, sha):
        _git(repo, "fetch", "--quiet", "--no-tags", "--no-recurse-submodules", remote, sha)
        if not _has_commit(repo, sha):
            raise ValueError("A PR commit could not be fetched.")


def _warn(message: str) -> None:
    print(f"::warning::Joyride survival: {message}", file=sys.stderr)


def scan(
    repo: Path,
    pull_requests: list[MergedPullRequest],
    measured: set[tuple[int, int]],
    *,
    default_head: str,
    remote: str,
    full_name: str,
    now: datetime,
    limit: int = MAX_PULL_REQUESTS,
    budget: _Budget | None = None,
    deliver: Callable[[list[dict[str, Any]]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Measure due checkpoints, oldest first, until ``limit`` PRs are measured.

    Cheap checks run first, and a PR that fails, runs past its time, or does
    not fit the byte budget does not count toward the limit. ``deliver``
    receives each PR's measurements as soon as they are complete.
    """

    budget = budget or _Budget()
    snapshots = _snapshot_heads(repo, remote)
    due = due_checkpoints(pull_requests, measured, now)
    summary = {
        "due_pull_requests": len(due), "without_snapshot": 0, "measured_pull_requests": 0,
        "not_on_default_branch": 0, "skipped_checkpoints": 0, "failed": 0, "timed_out": 0,
        "deferred": 0,
    }
    measured_at = _timestamp(now)
    measurements: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="attribution-survival-") as directory:
        store = Path(directory) / "snapshots"
        _git(Path(directory), "init", "--bare", "--quiet", str(store))
        ignore_revs = _ignore_revs(repo, default_head, Path(directory))
        for pull_request, days in due:
            if pull_request.head_sha not in snapshots:
                summary["without_snapshot"] += 1
                continue
            if summary["measured_pull_requests"] >= limit or budget.seconds_left() <= 0:
                summary["deferred"] += 1
                continue
            merge = pull_request.merge_commit_sha
            try:
                with _time_limit(min(MAX_PR_SECONDS, budget.seconds_left())):
                    if not _has_commit(repo, merge) or not _is_ancestor(repo, merge, default_head):
                        summary["not_on_default_branch"] += 1
                        continue
                    checkpoints = []
                    for day in days:
                        commit = _checkpoint_commit(
                            repo, default_head, merge, pull_request.merged_at + timedelta(days=day)
                        )
                        if commit is None:
                            summary["skipped_checkpoints"] += 1
                        else:
                            checkpoints.append((day, commit))
                    if not checkpoints:
                        continue
                    notes = _snapshot_notes(
                        store, remote, pull_request.head_sha, snapshots[pull_request.head_sha]
                    )
                    _ensure_commit(repo, remote, pull_request.base_sha)
                    _ensure_commit(repo, remote, pull_request.head_sha)
                    result = measure_pull_request(
                        repo, pull_request, checkpoints, notes=notes, full_name=full_name,
                        measured_at=measured_at, ignore_revs=ignore_revs, budget=budget,
                    )
            except _OutOfBudget:
                summary["deferred"] += 1
                continue
            except _OutOfTime:
                summary["timed_out"] += 1
                _warn(f"PR #{pull_request.number} ran past its time budget.")
                continue
            except (KeyError, OSError, RecursionError, TypeError, ValueError):
                summary["failed"] += 1
                _warn(f"PR #{pull_request.number} could not be measured.")
                continue
            summary["measured_pull_requests"] += 1
            measurements.extend(result)
            if deliver is not None:
                deliver(result)
    summary["measurements"] = len(measurements)
    return measurements, summary


def _request(
    method: str, url: str, token: str, headers: dict[str, str], body: bytes | None = None,
    *, accept: str = "application/json",
) -> Any:
    request = Request(url, data=body, method=method, headers={
        "Accept": accept, "Authorization": f"Bearer {token}", "User-Agent": _USER_AGENT,
        **headers,
    })
    label = urlsplit(url).path
    try:
        with build_opener(_NoRedirect()).open(request, timeout=60) as response:
            raw = response.read(_MAX_HTTP_BYTES + 1)
    except HTTPError as exc:
        raise ValueError(f"{method} {label} failed (HTTP {exc.code}).") from exc
    except (HTTPException, OSError) as exc:
        raise ValueError(f"{method} {label} could not be completed.") from exc
    if len(raw) > _MAX_HTTP_BYTES:
        raise ValueError(f"{method} {label} returned too much data.")
    try:
        return json.loads(raw or b"{}")
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError(f"{method} {label} returned invalid JSON.") from exc


def _github(url: str, token: str) -> Any:
    return _request(
        "GET", url, token, {"X-GitHub-Api-Version": "2022-11-28"},
        accept="application/vnd.github+json",
    )


def _repository_info(api_url: str, token: str, repository: str) -> tuple[int, str, str]:
    value = _github(f"{api_url}/repos/{repository}", token)
    identifier = value.get("id") if isinstance(value, dict) else None
    full_name = value.get("full_name") if isinstance(value, dict) else None
    branch = value.get("default_branch") if isinstance(value, dict) else None
    if (
        type(identifier) is not int or identifier < 1
        or not isinstance(full_name, str) or full_name.lower() != repository.lower()
        or not isinstance(branch, str) or not branch or len(branch) > 255
        or any(character < " " or character == "\x7f" for character in branch)
    ):
        raise ValueError("GitHub returned an invalid repository.")
    return identifier, _repository_name(full_name), branch


def _pull_requests(
    api_url: str, token: str, full_name: str, branch: str, repository_id: int,
    cutoff: datetime,
) -> list[MergedPullRequest]:
    """List the PRs merged into the default branch since ``cutoff``, newest update first."""

    found: dict[int, MergedPullRequest] = {}
    for page in range(1, _MAX_PAGES + 1):
        items = _github(
            f"{api_url}/repos/{full_name}/pulls?state=closed&base={quote(branch, safe='')}"
            f"&sort=updated&direction=desc&per_page=100&page={page}",
            token,
        )
        if not isinstance(items, list):
            raise ValueError("GitHub returned an invalid pull request list.")
        for pull_request in merged_pull_requests(
            items, repository_id=repository_id, cutoff=cutoff
        ):
            found.setdefault(pull_request.number, pull_request)
        updated = _time(items[-1].get("updated_at")) if items and isinstance(items[-1], dict) else None
        # A PR merged in the window was updated in it, so older pages hold none.
        if len(items) < 100 or updated is None or updated < cutoff:
            break
    return list(found.values())


def _endpoints(ingest_url: str) -> tuple[str, str, str]:
    """Derive the survival routes beside the configured artifact endpoint."""

    endpoint, audience = _ingest_url(ingest_url)
    parsed = urlsplit(endpoint)
    upload = urlunsplit((parsed.scheme, parsed.netloc, "/v1/artifacts/survival", "", ""))
    return upload + "/measured", upload, audience


def _measured(url: str, token: str, headers: dict[str, str]) -> set[tuple[int, int]]:
    value = _request("GET", url, token, headers)
    items = value.get("measured") if isinstance(value, dict) else None
    if not isinstance(items, list):
        raise ValueError("The measured-survival list is invalid.")
    measured: set[tuple[int, int]] = set()
    for item in items:
        if (
            not isinstance(item, list) or len(item) != 2
            or type(item[0]) is not int or item[0] < 1
            or type(item[1]) is not int or item[1] not in CHECKPOINT_DAYS
        ):
            raise ValueError("The measured-survival list is invalid.")
        measured.add((item[0], item[1]))
    return measured


class _Uploader:
    """Send measurements in batches as each fills; a failed batch is counted, never raised."""

    def __init__(
        self, url: str, audience: str, *, repository_id: int, full_name: str,
        headers: dict[str, str], oidc_request_url: str, oidc_request_token: str,
        delivery_id: str,
    ) -> None:
        self.url, self.audience, self.headers = url, audience, headers
        self.oidc = (oidc_request_url, oidc_request_token)
        self.delivery_id = delivery_id
        self.repository = {"id": str(repository_id), "full_name": full_name}
        self.envelope = len(self._body([]))
        self.pending: list[dict[str, Any]] = []
        self.size = 0
        self.sent = 0
        self.result = {"uploaded": 0, "upload_failed": 0}

    def _body(self, batch: list[dict[str, Any]]) -> bytes:
        return _encode({"schema": SCHEMA, "repository": self.repository, "measurements": batch})

    def add(self, measurements: list[dict[str, Any]]) -> None:
        for measurement in measurements:
            size = len(_encode(measurement)) + 1
            if self.pending and self.envelope + self.size + size > MAX_BATCH_BYTES:
                self.close()
            self.pending.append(measurement)
            self.size += size
            if len(self.pending) >= MAX_BATCH:
                self.close()

    def close(self) -> None:
        """Send what is pending."""

        if not self.pending:
            return
        batch, self.pending, self.size = self.pending, [], 0
        try:
            token = _oidc_token(*self.oidc, self.audience)
            _request("POST", self.url, token, {
                **self.headers, "Content-Type": "application/json",
                "X-Attribution-Delivery": f"{self.delivery_id}:{self.sent}",
            }, self._body(batch))
        except (HTTPException, OSError, ValueError) as exc:
            self.result["upload_failed"] += len(batch)
            _warn(f"{len(batch)} measurements were not uploaded. {exc}")
        else:
            self.result["uploaded"] += len(batch)
        self.sent += 1


def upload(
    url: str, audience: str, measurements: list[dict[str, Any]], **identity: Any
) -> dict[str, int]:
    """Upload measurements in size-bounded batches of at most ``MAX_BATCH``."""

    uploader = _Uploader(url, audience, **identity)
    uploader.add(measurements)
    uploader.close()
    return uploader.result


def run(
    repo: Path, *, repository: str, default_head: str, ref: str, token: str,
    api_url: str, server_url: str, ingest_url: str, oidc_request_url: str,
    oidc_request_token: str, workflow_ref: str, delivery_id: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Measure due checkpoints for one repository and upload the counts."""

    now = now or datetime.now(timezone.utc)
    measured_url, upload_url, audience = _endpoints(ingest_url)
    repository_id, full_name, branch = _repository_info(api_url, token, repository)
    if ref != f"refs/heads/{branch}":
        return {"status": "ignored", "reason": "The scan runs only on the default branch."}
    headers = {
        "X-GitHub-Repository": full_name,
        "X-GitHub-Repository-ID": str(repository_id),
        "X-GitHub-Workflow-Ref": workflow_ref,
    }
    measured = _measured(
        measured_url, _oidc_token(oidc_request_url, oidc_request_token, audience), headers
    )
    pull_requests = _pull_requests(
        api_url, token, full_name, branch, repository_id, now - timedelta(days=WINDOW_DAYS)
    )
    uploader = _Uploader(
        upload_url, audience, repository_id=repository_id, full_name=full_name,
        headers=headers, oidc_request_url=oidc_request_url,
        oidc_request_token=oidc_request_token, delivery_id=delivery_id,
    )
    with _report_git_environment(_git_environment(server_url, token)):
        _measurements, summary = scan(
            repo, pull_requests, measured, default_head=default_head,
            remote=f"{server_url}/{full_name}.git", full_name=full_name, now=now,
            deliver=uploader.add,
        )
    uploader.close()
    return {"status": "measured", **summary, **uploader.result}


def main() -> int:
    environment = os.environ
    try:
        event = environment.get("GITHUB_EVENT_NAME", "")
        if event not in _EVENTS:
            raise ValueError(
                "This module requires a trusted schedule or workflow_dispatch workflow."
            )
        ingest_url = (environment.get("ATTRIBUTION_INGEST_URL") or "").strip()
        if not ingest_url:
            print(
                "Joyride survival: ATTRIBUTION_INGEST_URL is not set, so nothing is measured.",
                file=sys.stderr,
            )
            print(json.dumps({"status": "not-configured"}))
            return 0
        _endpoints(ingest_url)
        repository = _repository_name(environment.get("GITHUB_REPOSITORY"))
        default_head = environment.get("GITHUB_SHA", "")
        if not _OID.fullmatch(default_head):
            raise ValueError("GITHUB_SHA is not a commit ID.")
        token = environment.get("GITHUB_TOKEN", "")
        if not token:
            raise ValueError("GITHUB_TOKEN is required to list merged PRs.")
        delivery_id = (
            f"{environment.get('GITHUB_RUN_ID', '')}:"
            f"{environment.get('GITHUB_RUN_ATTEMPT', '')}:{event}:survival"
        )
        if not _DELIVERY.fullmatch(delivery_id):
            raise ValueError("The workflow run identity is invalid.")
        api_url = _host_url(environment.get("GITHUB_API_URL", "https://api.github.com"), api=True)
        server_url = _host_url(environment.get("GITHUB_SERVER_URL", "https://github.com"))
    except (KeyError, ValueError) as exc:
        print(f"Joyride survival: {exc}", file=sys.stderr)
        return 1
    try:
        result = run(
            Path(environment.get("GITHUB_WORKSPACE") or ".").resolve(),
            repository=repository, default_head=default_head,
            ref=environment.get("GITHUB_REF", ""), token=token, api_url=api_url,
            server_url=server_url, ingest_url=ingest_url,
            oidc_request_url=environment.get("ACTIONS_ID_TOKEN_REQUEST_URL", ""),
            oidc_request_token=environment.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN", ""),
            workflow_ref=environment.get("GITHUB_WORKFLOW_REF", ""),
            delivery_id=delivery_id,
        )
    except (HTTPException, KeyError, OSError, ValueError) as exc:
        # The scan is a weekly measurement: a network or API failure is logged
        # and the next run measures what this one could not.
        _warn(f"the scan stopped. {exc}")
        print(json.dumps({"status": "unavailable"}))
        return 0
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
