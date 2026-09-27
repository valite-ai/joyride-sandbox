"""Replay a merged pull request's task on a cheaper model and score it against what merged.

``joyride replay --pr <number> --model <model>`` finds the commit that merged
the pull request in the checkout's history (GitHub's record through ``gh``
when it answers) and runs the first prompt of the pull request's root session
on ``model``, headless, at the merge's first parent. It then compares the
replay's diff with the merged diff: the Jaccard of the changed paths, the
share of the merged diff's added lines that the replay also added after
whitespace collapse, over all of them and for code, tests, and documentation
apart, whether the repository's test command passes on the replay, and again
with the merged pull request's own test files in place of the replay's (the
held-out tests), the lines the replay added, and the replay's cost against the
pull request's. ``--control`` also replays the model and effort that paid for
the pull request on the same prompt, so a miss that it repeats is the task's,
not the cheaper model's.

The task, the test command, and the pull request's cost come from the hosted
next steps page, whose cards carry a ``replay`` block for the pull request
they suggest replaying, else from the local history cache that
scripts/history_replay.py loads; the test command else comes from the
instructions file at the base commit. The replay runs in a clone of the
checkout under a temporary directory, which has no remote and whose git hooks
point at an empty directory. Claude Code runs with every hook off and no saved
session, and Codex in a Codex home of its own with an API key and its hooks
off, so a replay touches neither the developer's branches nor their hooks,
capture, history, or memory. The harness stops at the budget and at a wall-clock cap,
and the clone is removed afterwards. The record goes to
``~/.joyride/replays/<pr>-<model>.json``, never into the repository, and with
``JOYRIDE_API_TOKEN`` also to ``POST /v1/me/next/replays``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, TextIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from . import __version__, usage_fallback
from .install_code import _NoRedirect
from .next_command import TIMEOUT_SECONDS, TOKEN_ENV, _hosted_origin, checkout_repository, fetch


# A replay takes a merged diff of at most 40 files and 3,000 changed lines unless --force.
MAX_FILES = 40
MAX_LINES = 3_000
BUDGET_USD = 3.0
# The harness stops after 30 minutes, and each run of the test command after 15.
HARNESS_SECONDS = 30 * 60
TEST_SECONDS = 15 * 60
POLL_SECONDS = 2.0
STATE_DIR = Path("~/.joyride")
CACHE = STATE_DIR / "history-replay.sqlite3"
# A record keeps at most this many paths of each diff.
MAX_PATHS = 200
# Nobody answers a headless replay, so the harness hears that once.
NOTE = ("This is a replay of a past task in a scratch clone. Nobody will answer questions, so make reasonable "
        "choices and finish the task. Do not push, open pull requests, or change files outside this clone.")
HARNESSES = {"claude-code": "Claude Code", "codex": "Codex"}
_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_EFFORT = re.compile(r"^[a-z]{1,20}$")
# The test runners of hosted.next_steps._TEST: the first command in the traces that runs one is the test command.
_TEST = re.compile(
    r"\b(?:pytest|unittest|(?:npm|pnpm|yarn|bun|deno)(?:\s+run)?\s+test|node\s+--test|go\s+test|cargo\s+test"
    r"|devstack\s+test|make\s+(?:test|check)|tox|vitest|jest|playwright\s+test|mvn\s+test|gradle\w*\s+test"
    r"|dotnet\s+test|rspec|phpunit)\b"
)
_QUOTED = re.compile(r"""(['"]).*?\1""", re.DOTALL)
_SEGMENT = re.compile(r"&&|\|\||[;|\n]")
# Text that a harness or a script wrote as a prompt, as hosted.next_steps_developer._HARNESS_TEXT reads it.
_HARNESS_TEXT = re.compile(
    r"(?i)^\s*(?:[<\[]|generate a short git branch name|review (?:the changes on this branch|pull request #)"
    r"|work only inside the linked worktree|the following is the codex agent history"
    r"|you are one finder in a multi-angle code review)|<task-notification>|<system-reminder>"
)
_SPAN = re.compile(r"`([^`\n]+)`")
# The engine's path classes: hosted.next_steps._DOC for documentation and hosted.next_steps_models._TEST_PATH for
# tests, where a document among the tests is documentation. Every other path is code.
_DOC = re.compile(r"(?i)(?:^|/)docs/|\.(?:md|rst|txt)$")
_TEST_PATH = re.compile(
    r"(?i)(?:^|/)(?:tests?|__tests__|specs?)/|(?:^|/)test_[^/]+$|_test\.\w+$|\.(?:test|spec)\.[cm]?[jt]sx?$"
    r"|__snapshots__/|\.snap$"
)
CLASSES = ("code", "tests", "documentation")
_OID = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


class ReplayError(ValueError):
    """The replay could not run, or its record could not be kept."""


def test_command(command: Any) -> str | None:
    """The part of a shell command that runs tests, with its whitespace collapsed, as the page finds it."""
    if not isinstance(command, str):
        return None
    for part in _SEGMENT.split(command):
        if _TEST.search(_QUOTED.sub("", part)) and "\n" not in part.strip() and len(part) <= 160:
            return " ".join(part.split())
    return None


def _money(value: Any, digits: int = 4) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return round(float(value), digits)


def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


# Git ----------------------------------------------------------------------------------


def _git(checkout: Path, *args: str, check: bool = True) -> str:
    try:
        result = subprocess.run(["git", "-C", str(checkout), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, errors="replace", timeout=300, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReplayError(f"git {args[0]} failed: {exc}") from exc
    if result.returncode != 0:
        if check:
            raise ReplayError(f"git {args[0]} failed: {result.stderr.strip()[:300]}")
        return ""
    return result.stdout


def _gh_merge_commit(checkout: Path, pr: int) -> str | None:
    """The merge commit that GitHub records for the pull request, or None when gh cannot say."""
    if shutil.which("gh") is None:
        return None
    try:
        result = subprocess.run(["gh", "pr", "view", str(pr), "--json", "mergeCommit"], cwd=checkout,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=30, check=False)
        payload = json.loads(result.stdout) if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    commit = payload.get("mergeCommit") if isinstance(payload, dict) else None
    oid = commit.get("oid") if isinstance(commit, dict) else None
    return oid if isinstance(oid, str) and _OID.fullmatch(oid) else None


def _default_refs(checkout: Path) -> list[str]:
    """The default branch's refs that exist: origin's HEAD, then main and master, remote before local."""
    head = _git(checkout, "symbolic-ref", "--quiet", "refs/remotes/origin/HEAD", check=False).strip()
    refs = [head] if head else []
    refs += [ref for ref in ("refs/remotes/origin/main", "refs/remotes/origin/master", "refs/heads/main",
                             "refs/heads/master") if ref not in refs]
    return [ref for ref in refs if _git(checkout, "rev-parse", "--verify", "--quiet", ref, check=False).strip()]


def merge_commit(checkout: Path, pr: int) -> str:
    """The commit that merged the pull request: GitHub's record, else a merge or squash subject on the default
    branch."""
    oid = _gh_merge_commit(checkout, pr)
    if oid and _git(checkout, "cat-file", "-t", oid, check=False).strip() == "commit":
        return oid
    subject = re.compile(rf"^Merge pull request #{pr}(?!\d)|\(#{pr}\)$")
    for ref in _default_refs(checkout):
        for line in _git(checkout, "log", "--first-parent", "--format=%H %s", ref, check=False).splitlines():
            sha, _, text = line.partition(" ")
            if subject.search(text):
                return sha
    raise ReplayError(f"No merge of #{pr} is in this checkout's history. Run git fetch origin and try again; "
                      "a pull request that did not merge has no merged diff to compare with.")


def path_class(path: str) -> str:
    """documentation, tests, or code, by the engine's path classes."""
    return "documentation" if _DOC.search(path) else "tests" if _TEST_PATH.search(path) else "code"


def added_by_class(patch: str) -> dict[str, Counter[str]]:
    """The added lines of a unified diff by the class of their file, with their whitespace collapsed.

    A file's class comes from its ``+++`` header, and blank lines do not count.
    """
    lines: dict[str, Counter[str]] = {name: Counter() for name in CLASSES}
    current, in_hunk = lines["code"], False
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            in_hunk = False
        elif line.startswith("@@"):
            in_hunk = True
        elif in_hunk and line.startswith("+"):
            text = " ".join(line[1:].split())
            if text:
                current[text] += 1
        elif line.startswith("+++ "):
            header = line[4:].strip().strip('"')
            current = lines[path_class(header[2:] if header.startswith("b/") else header)]
    return lines


def added_lines(patch: str) -> Counter[str]:
    """The added lines of a unified diff with their whitespace collapsed; blank lines do not count."""
    return sum(added_by_class(patch).values(), Counter())


def change(checkout: Path, base: str, head: str | None = None) -> dict[str, Any]:
    """The paths, the added and deleted line counts, and the added lines of the diff from ``base``.

    With ``head`` the diff runs to that commit, else to what the checkout has staged.
    """
    target = [base, head] if head else ["--cached", base]
    options = ["--no-renames", "--no-ext-diff", "--no-textconv"]
    files, added, deleted = [], 0, 0
    for item in _git(checkout, "diff", *options, "--numstat", "-z", *target).split("\0"):
        parts = item.split("\t", 2)
        if len(parts) == 3:
            files.append(parts[2])
            added += int(parts[0]) if parts[0].isdigit() else 0
            deleted += int(parts[1]) if parts[1].isdigit() else 0
    by_class = added_by_class(_git(checkout, "diff", *options, "--no-color", "-U0", *target))
    return {"files": sorted(files), "lines_added": added, "lines_deleted": deleted,
            "added": sum(by_class.values(), Counter()), "by_class": by_class}


def score(merged: Mapping[str, Any], replay: Mapping[str, Any]) -> dict[str, Any]:
    """The Jaccard of the two diffs' paths, and the share of the merged diff's added lines that the replay added.

    Lines count as often as they occur. The share is over all the added
    lines, and for each class apart over the lines that the replay added in
    files of that class. A merged diff, or a class, that added no line has no
    share.
    """
    ours, theirs = set(merged["files"]), set(replay["files"])
    union = ours | theirs
    total = sum(merged["added"].values())
    shared = sum((merged["added"] & replay["added"]).values())
    classes = {}
    for name in CLASSES:
        lines = sum(merged["by_class"][name].values())
        found = sum((merged["by_class"][name] & replay["by_class"][name]).values())
        classes[name] = {"merged_lines": lines, "shared_lines": found,
                         "overlap": round(found / lines, 4) if lines else None}
    return {
        "files_jaccard": round(len(ours & theirs) / len(union), 4) if union else 1.0,
        "line_overlap": round(shared / total, 4) if total else None,
        "shared_files": len(ours & theirs), "shared_lines": shared, "compared_lines": total, "classes": classes,
    }


# The task --------------------------------------------------------------------------------


def hosted_task(payload: Mapping[str, Any], repository: str, pr: int) -> dict[str, Any] | None:
    """The task that a card of the hosted page carries for the pull request, or None."""
    for item in payload.get("suggestions") or []:
        block = item.get("replay") if isinstance(item, Mapping) else None
        if (isinstance(block, Mapping) and str(block.get("repository") or "").lower() == repository.lower()
                and block.get("pr_number") == pr and isinstance(block.get("task"), str) and block["task"].strip()):
            effort = block.get("original_effort")
            return {"source": "hosted", "task": block["task"], "test_command": test_command(block.get("test_command")),
                    "original_cost_usd": _money(block.get("cost_usd")),
                    "original_model": block["original_model"] if isinstance(block.get("original_model"), str) else None,
                    "original_effort": effort if isinstance(effort, str) and _EFFORT.fullmatch(effort) else None}
    return None


def _input_command(raw: Any) -> Any:
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return None
    return value.get("command") if isinstance(value, dict) else None


def history_task(cache: Path, repository: str, pr: int) -> dict[str, Any] | None:
    """The task of the pull request from the local history cache, or None.

    Its sessions are the root sessions linked to it and their subagents, as
    scripts/history_replay.py groups them. The task is the first prompt that
    the developer typed in the earliest of those root sessions, the test
    command the first test command that the sessions ran, the cost their sum
    when every one has a cost, the model the one that cost the most, and the
    effort the one that cost the most on that model.
    """
    if not cache.is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{cache}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        sessions = {row[0]: row for row in conn.execute(
            "SELECT session_id, parent_session_id, pr_number, pr_repository, model, estimated_usd, started_at, effort "
            "FROM sessions")}
        roots = sorted((row[0] for row in sessions.values() if row[2] == pr and row[1] not in sessions
                        and str(row[3] or "").lower() == repository.lower()),
                       key=lambda key: (sessions[key][6] or "", key))
        children: dict[str, list[str]] = {}
        for row in sessions.values():
            if row[1] in sessions:
                children.setdefault(row[1], []).append(row[0])
        members, queue = [], list(roots)
        while queue:
            key = queue.pop(0)
            if key not in members:
                members.append(key)
                queue.extend(children.get(key, []))
        events = conn.execute(
            f"SELECT session_id, kind, input, text FROM events WHERE session_id IN ({','.join('?' * len(members))}) "
            "AND (kind = 'user_prompt' OR (kind = 'tool_call' AND tool_class = 'shell')) ORDER BY seq", members,
        ).fetchall() if members else []
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    order = {key: index for index, key in enumerate(sorted(members, key=lambda key: (sessions[key][6] or "", key)))}
    events.sort(key=lambda event: order[event[0]])
    task = next((text for key in roots for session, kind, _, text in events if session == key and kind == "user_prompt"
                 and isinstance(text, str) and text.strip() and not _HARNESS_TEXT.search(text)), None)
    if task is None:
        return None
    test = next((found for _, kind, raw, _ in events
                 if kind == "tool_call" and (found := test_command(_input_command(raw)))), None)
    costs = [sessions[key][5] for key in members]
    paid: Counter[str] = Counter()
    efforts: dict[str, Counter[str]] = {}
    for key in members:
        model, dollars, effort = sessions[key][4], sessions[key][5], sessions[key][7]
        if model and dollars is not None:
            paid[model] += dollars
            effort = effort if isinstance(effort, str) and _EFFORT.fullmatch(effort) and effort != "default" else ""
            efforts.setdefault(model, Counter())[effort] += dollars
    model = max(paid, key=lambda name: (paid[name], name)) if paid else None
    effort = max(efforts[model], key=lambda name: (efforts[model][name], name)) if model else ""
    return {"source": "history", "task": task, "test_command": test,
            "original_cost_usd": _money(math.fsum(costs)) if None not in costs else None,
            "original_model": model, "original_effort": effort or None}


def instructions_test_command(checkout: Path, base: str) -> str | None:
    """The first test command in a code block or a code span of AGENTS.md, else CLAUDE.md, at ``base``."""
    for name in ("AGENTS.md", "CLAUDE.md"):
        fenced = False
        for line in _git(checkout, "show", f"{base}:{name}", check=False).splitlines():
            if line.lstrip().startswith("```"):
                fenced = not fenced
                continue
            for candidate in [line] if fenced else _SPAN.findall(line):
                found = test_command(candidate)
                if found:
                    return found
    return None


# The harness ------------------------------------------------------------------------------


def _environment() -> dict[str, str]:
    """The developer's environment without the Joyride token, which can record replays."""
    return {key: value for key, value in os.environ.items() if key != TOKEN_ENV}


def _stop(process: subprocess.Popen[Any]) -> None:
    """Stop a process and everything it started, which shares its process group."""
    for number in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, number)
        except (ProcessLookupError, PermissionError):
            return
        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


def _supervise(command: list[str], *, cwd: Path, env: Mapping[str, str], task: Path, scratch: Path,
               budget_usd: float,
               spent: Callable[[], float | None] | None = None) -> tuple[int | None, str | None, float]:
    """Run the harness until it exits, the budget runs out, or the wall-clock cap passes.

    Its output goes to files in the scratch directory, and whatever it
    started in the background stops with it.
    """
    started = time.monotonic()
    try:
        with task.open("rb") as stdin, (scratch / "harness.out").open("wb") as out, \
                (scratch / "harness.err").open("wb") as err:
            process = subprocess.Popen(command, cwd=cwd, env=dict(env), stdin=stdin, stdout=out, stderr=err,
                                       start_new_session=True)
    except FileNotFoundError as exc:
        raise ReplayError(f"{command[0]} is not installed or not on PATH.") from exc
    stopped, code = None, None
    try:
        while code is None:
            try:
                code = process.wait(timeout=POLL_SECONDS)
            except subprocess.TimeoutExpired:
                if time.monotonic() - started > HARNESS_SECONDS:
                    stopped = "time"
                elif spent is not None and (cost := spent()) is not None and cost >= budget_usd:
                    stopped = "budget"
                if stopped:
                    _stop(process)
                    code = process.wait()
    finally:
        _stop(process)
    return code, stopped, time.monotonic() - started


def _last_error(scratch: Path) -> str | None:
    """The last line that the harness wrote to standard error, for the terminal; a record never holds it."""
    try:
        text = (scratch / "harness.err").read_text(errors="replace")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
    except OSError:
        return None
    return lines[-1][:300] if lines else None


def _claude(model: str, effort: str | None, budget_usd: float, task: Path, checkout: Path,
            scratch: Path) -> dict[str, Any]:
    command = [
        "claude", "-p", "--output-format", "json", "--model", model, "--max-budget-usd", f"{budget_usd:g}",
        "--permission-mode", "acceptEdits", "--allowedTools", "Bash", "Edit", "Write", "Read", "Glob", "Grep",
        "--disallowedTools", "Bash(git push:*)", "Bash(gh:*)",
        # Every hook off, the developer's capture hooks too, no telemetry, and no saved session for a history import.
        "--settings", json.dumps({"disableAllHooks": True, "env": {"CLAUDE_CODE_ENABLE_TELEMETRY": "0"}}),
        "--no-session-persistence", "--append-system-prompt", NOTE,
    ]
    if effort:
        command += ["--effort", effort]
    # Auto memory would leave a folder for the clone under ~/.claude/projects.
    env = {**_environment(), "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}
    code, stopped, seconds = _supervise(command, cwd=checkout, env=env, task=task, scratch=scratch,
                                        budget_usd=budget_usd)
    try:
        result = json.loads((scratch / "harness.out").read_text(encoding="utf-8", errors="replace"))
    except ValueError:
        result = None
    result = result if isinstance(result, dict) else {}
    usage = result.get("modelUsage") if isinstance(result.get("modelUsage"), dict) else {}
    tokens: Counter[str] = Counter()
    for entry in usage.values():
        for key, field in (("input", "inputTokens"), ("cached_input", "cacheReadInputTokens"),
                           ("cache_creation_input", "cacheCreationInputTokens"), ("output", "outputTokens")):
            tokens[key] += _count(entry.get(field) if isinstance(entry, dict) else None)
    outcome = str(result["subtype"])[:64] if isinstance(result.get("subtype"), str) else None
    errors = result.get("errors") if isinstance(result.get("errors"), list) else []
    return {"exit_code": code, "stopped": stopped or ("budget" if outcome == "error_max_budget_usd" else None),
            "outcome": outcome, "seconds": round(seconds, 1), "cost_usd": _money(result.get("total_cost_usd")),
            "tokens": dict(tokens), "models": sorted(str(name) for name in usage),
            "error": str(errors[-1])[:300] if errors else _last_error(scratch)}


def _codex_rows(home: Path) -> list[usage_fallback.UsageRow]:
    """Each request of the Codex sessions under ``home``, priced at the API rates that its key pays."""
    rows = []
    for path in sorted(home.glob("sessions/**/rollout-*.jsonl")):
        try:
            lines = path.read_bytes().splitlines()
        except OSError:
            continue
        found, _ = usage_fallback.parse_codex_lines(lines, session_id=path.stem, state={})
        rows += [usage_fallback.price_row(replace(row, auth_mode=None)) for row in found]
    return rows


def _codex_cost(rows: list[usage_fallback.UsageRow]) -> float | None:
    """The sum of the requests' prices, 0 before the first request, and None while one has no price."""
    if any(row.cost_amount is None or row.cost_unit != "USD" for row in rows):
        return None
    return float(sum((Decimal(row.cost_amount) for row in rows), Decimal(0)))


def _codex(model: str, effort: str | None, budget_usd: float, task: Path, checkout: Path,
           scratch: Path) -> dict[str, Any]:
    key = os.environ.get("CODEX_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ReplayError("Set OPENAI_API_KEY for a Codex replay: it runs with an API key in a Codex home of its own, "
                          "so your sign-in, hooks, and history stay as they are.")
    probe = usage_fallback.UsageRow(event_key="price", provider="codex", event_kind="codex_response",
                                    native_session_id="price", model=model, input_tokens=1, output_tokens=1)
    if usage_fallback.price_row(probe).cost_amount is None:
        raise ReplayError(f"Joyride has no API price for {model}, so it cannot hold a Codex replay to a budget.")
    home = scratch / "codex-home"
    home.mkdir()
    command = ["codex", "exec", "--json", "-m", model, "--sandbox", "workspace-write", "--disable", "hooks",
               "-c", f"developer_instructions={json.dumps(NOTE)}", "-C", str(checkout), "-"]
    if effort:
        command[5:5] = ["-c", f'model_reasoning_effort="{effort}"']
    env = {**_environment(), "CODEX_HOME": str(home), "CODEX_API_KEY": key}
    code, stopped, seconds = _supervise(command, cwd=checkout, env=env, task=task, scratch=scratch,
                                        budget_usd=budget_usd, spent=lambda: _codex_cost(_codex_rows(home)))
    rows = _codex_rows(home)
    tokens: Counter[str] = Counter()
    for row in rows:
        for name, value in (("input", row.input_tokens), ("cached_input", row.cached_input_tokens),
                            ("cache_creation_input", row.cache_creation_input_tokens), ("output", row.output_tokens)):
            tokens[name] += _count(value)
    events = [json.loads(line) for line in (scratch / "harness.out").read_text(encoding="utf-8", errors="replace")
              .splitlines() if line.startswith("{")]
    kinds = {event.get("type") for event in events if isinstance(event, dict)}
    outcome = "failed" if kinds & {"turn.failed", "error"} else "completed" if "turn.completed" in kinds else None
    return {"exit_code": code, "stopped": stopped, "outcome": outcome, "seconds": round(seconds, 1),
            "cost_usd": _money(_codex_cost(rows)), "tokens": dict(tokens),
            "models": sorted({row.model for row in rows if row.model}), "error": _last_error(scratch)}


def run_harness(harness: str, *, model: str, effort: str | None, budget_usd: float, prompt: str, checkout: Path,
                scratch: Path) -> dict[str, Any]:
    """Run one headless session of the harness in the clone; tests replace this function.

    It returns the exit code, why it stopped early ("budget" or "time"), the
    harness's own outcome, the seconds, the cost in dollars, the tokens, the
    models, and the harness's last error line: Claude Code's usage from its
    JSON result, Codex's from its session rollout at the API rates.
    """
    task = scratch / "task.txt"
    task.write_text(prompt, encoding="utf-8")
    runner = _claude if harness == "claude-code" else _codex
    return runner(model, effort, budget_usd, task, checkout, scratch)


# The replay ----------------------------------------------------------------------------------


def _clone(checkout: Path, base: str, scratch: Path) -> Path:
    """A clone of the checkout at ``base`` without a remote or git hooks, whose refs are its own."""
    clone, hooks = scratch / "checkout", scratch / "no-hooks"
    hooks.mkdir()
    _git(scratch, "clone", "--quiet", "--shared", "--no-checkout", str(checkout), str(clone))
    _git(clone, "remote", "remove", "origin")
    _git(clone, "config", "core.hooksPath", str(hooks))
    _git(clone, "checkout", "--quiet", "--detach", base)
    return clone


def _run_tests(clone: Path, command: str) -> int | None:
    """The test command's exit status in the clone, or None when it ran past the cap."""
    process = subprocess.Popen(command, shell=True, cwd=clone, env=_environment(), stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        return process.wait(timeout=TEST_SECONDS)
    except subprocess.TimeoutExpired:
        return None
    finally:
        _stop(process)


def _tests(clone: Path, command: str | None, merged: str, held_out: list[str]) -> dict[str, Any]:
    """Whether the test command passes on the replay, and on the replay with the held-out test files.

    The held-out run copies the merged pull request's own versions of
    ``held_out`` over the replay's and runs the command again. A failure
    counts only when the command passes on what merged; when it fails there
    too, the run has no result.
    """
    if command is None:
        return {"passed": None, "held_out_passed": None, "replay_exit": None, "held_out_exit": None,
                "merged_exit": None}
    replay = _run_tests(clone, command)
    held = None
    if held_out:
        _git(clone, "checkout", merged, "--", *held_out)
        held = _run_tests(clone, command)
    merged_exit = None
    if replay != 0 or (held_out and held != 0):
        _git(clone, "reset", "--quiet", "--hard", merged)
        _git(clone, "clean", "--quiet", "-fd")
        merged_exit = _run_tests(clone, command)

    def verdict(code: int | None) -> bool | None:
        return True if code == 0 else False if merged_exit == 0 else None
    return {"passed": verdict(replay), "held_out_passed": verdict(held) if held_out else None,
            "replay_exit": replay, "held_out_exit": held, "merged_exit": merged_exit}


def save(record: Mapping[str, Any], state_dir: Path | None = None) -> Path:
    """Write the record to <state>/replays/<pr>-<model>.json, in place of an earlier replay on the model."""
    directory = (state_dir or STATE_DIR).expanduser() / "replays"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / f"{record['pr_number']}-{record['model']}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def post(origin: str, token: str, record: Mapping[str, Any]) -> None:
    """Send the record to the hosted service, which keeps it for the owner of the token."""
    request = Request(f"{origin}/v1/me/next/replays", data=json.dumps(record).encode(), method="POST", headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json",
        "User-Agent": f"joyride-cli/{__version__}",
    })
    try:
        # A redirect could carry the token to another origin.
        with build_opener(_NoRedirect()).open(request, timeout=TIMEOUT_SECONDS) as response:
            response.read(4096)
    except HTTPError as exc:
        try:
            message = json.loads(exc.read(4096)).get("error")
        except (OSError, ValueError, AttributeError):
            message = None
        detail = message.strip()[:300] if isinstance(message, str) and message.strip() else f"HTTP {exc.code}"
        raise ReplayError(f"{origin} did not keep the replay record: {detail}") from exc
    except (URLError, OSError) as exc:
        raise ReplayError(f"{origin} did not answer: {exc}") from exc


def render(record: Mapping[str, Any], path: Path) -> str:
    """The replay's scores as plain text."""
    details = record["record"]
    merged, replay, held_out = details["merged"], details["replay"], details["held_out"]
    effort = f" at {record['effort']} effort" if record["effort"] else ""
    control = ", the model that paid for it, as a control," if record["control"] else ""
    lines = [f"Replayed #{record['pr_number']} of {record['repository']} on {record['model']}{control} in "
             f"{HARNESSES[record['harness']]}{effort}, from {details['base_sha'][:7]}."]
    if replay["stopped"] == "budget":
        lines.append(f"The replay stopped at its budget of ${details['budget_usd']:,.2f}.")
    elif replay["stopped"] == "time":
        lines.append(f"The replay stopped at the {HARNESS_SECONDS // 60}-minute cap.")
    elif replay["exit_code"] != 0:
        lines.append(f"The harness exited with status {replay['exit_code']} ({replay['outcome'] or 'no result'}).")
    lines.append(f"Files: the replay changed {len(replay['files'])}, {replay['shared_files']} of the "
                 f"{len(merged['files'])} that merged (Jaccard {record['files_jaccard']:.2f}).")
    if record["line_overlap"] is None:
        lines.append(f"Lines: the merged change added no line to compare; the replay added {record['lines_added']}.")
    else:
        split = ", ".join(f"{name} {round(100 * item['overlap'])}%" for name, item in details["classes"].items()
                          if item["overlap"] is not None)
        lines.append(f"Lines: {replay['shared_lines']} of the {merged['compared_lines']} lines that the merged change "
                     f"added appear in the replay ({round(100 * record['line_overlap'])}%: {split}); the replay added "
                     f"{record['lines_added']}.")
    command = f"`{details['test_command']}`" if details["test_command"] else None
    if command is None:
        lines.append("Tests: Joyride found no test command in the traces or the instructions file.")
    elif record["tests_passed"] is not None:
        lines.append(f"Tests: {command} {'passed' if record['tests_passed'] else 'failed'} on the replay.")
    else:
        lines.append(f"Tests: {command} failed on the replay and on what merged, so they tell nothing here.")
    if held_out.get("skipped"):
        lines.append(f"Held-out tests: skipped. {held_out['skipped']}")
    elif record["held_out_tests_passed"] is not None:
        count = len(held_out["files"])
        lines.append(f"Held-out tests: {command} {'passed' if record['held_out_tests_passed'] else 'failed'} with the "
                     f"merged pull request's {count} test file{'s' if count != 1 else ''} in place of the replay's.")
    else:
        lines.append(f"Held-out tests: {command} failed with the merged test files and on what merged, so they tell "
                     "nothing here.")
    cost = lambda value: "an unknown amount" if value is None else f"${value:,.2f}"  # noqa: E731
    lines.append(f"Cost: {cost(record['replay_cost_usd'])} for the replay against "
                 f"{cost(record['original_cost_usd'])} for the pull request.")
    lines.append(f"Record: {path}")
    return "\n".join(lines)


def run(pr: int, *, repo: Path, model: str, harness: str | None = None, effort: str | None = None,
        budget_usd: float = BUDGET_USD, force: bool = False, as_json: bool = False, control: bool = False,
        stdout: TextIO | None = None, state_dir: Path | None = None, cache: Path | None = None) -> int:
    """Replay the pull request's task on ``model``, print its scores, keep its record, and return the exit status.

    With ``control``, the model and effort that paid for the pull request
    replay the same prompt afterwards, within the same budget each, and keep
    a record of their own marked as the control.
    """
    output = stdout or sys.stdout
    if pr < 1:
        raise ReplayError("--pr must be the number of a merged pull request.")
    if not _MODEL.fullmatch(model):
        raise ReplayError("--model must be a model name, for example gpt-6-luna or claude-haiku-4-5.")
    harness = harness or ("claude-code" if model.lower().startswith("claude") else "codex")
    if harness not in HARNESSES:
        raise ReplayError("--harness must be claude-code or codex.")
    if effort is not None and not _EFFORT.fullmatch(effort):
        raise ReplayError("--effort must be an effort level, for example low, medium, or high.")
    if not (math.isfinite(budget_usd) and budget_usd > 0):
        raise ReplayError("--budget-usd must be more than zero dollars.")
    checkout = Path(_git(repo, "rev-parse", "--show-toplevel").strip())
    repository = checkout_repository(str(checkout))
    if repository is None:
        raise ReplayError("The checkout has no GitHub origin, so Joyride cannot tell which repository the pull "
                          "request belongs to.")
    merged_sha = merge_commit(checkout, pr)
    base = _git(checkout, "rev-parse", f"{merged_sha}^1").strip()
    merged = change(checkout, base, merged_sha)
    size = merged["lines_added"] + merged["lines_deleted"]
    if not force and (len(merged["files"]) > MAX_FILES or size > MAX_LINES):
        raise ReplayError(f"#{pr} changed {len(merged['files'])} files and {size:,} lines, more than the {MAX_FILES} "
                          f"files or {MAX_LINES:,} lines that a replay takes. Add --force to replay it anyway.")
    token = os.environ.get(TOKEN_ENV, "").strip()
    origin = _hosted_origin() if token else None
    found = hosted_task(fetch(origin, token, timeout=TIMEOUT_SECONDS, fast=True), repository, pr) if origin else None
    found = found or history_task((cache or CACHE).expanduser(), repository, pr)
    if found is None:
        raise ReplayError(f"Joyride found no first prompt for #{pr}: no card of your next steps page names it, and "
                          "the local history cache holds no session linked to it.")
    command = found["test_command"] or instructions_test_command(checkout, base)
    # The held-out tests: the test files that the merged change added or changed.
    held_out = [path for path in _git(checkout, "diff", "--no-renames", "--name-only", "-z", "--diff-filter=AM",
                                      base, merged_sha).split("\0") if path and path_class(path) == "tests"]
    runs = [(model, harness, effort, False)]
    if control:
        original = found["original_model"]
        if not isinstance(original, str) or not _MODEL.fullmatch(original):
            raise ReplayError(f"Joyride does not know the model that paid for #{pr}, so --control has nothing to "
                              "replay.")
        if original.lower() == model.lower():
            raise ReplayError(f"--control replays {original}, the model that paid for #{pr}, which --model names "
                              "already.")
        runs.append((original, "claude-code" if original.lower().startswith("claude") else "codex",
                     found["original_effort"], True))

    def replay(model: str, harness: str, effort: str | None, control: bool) -> dict[str, Any]:
        """One replay in a clone of its own, scored against what merged."""
        scratch = Path(tempfile.mkdtemp(prefix="joyride-replay-"))
        try:
            clone = _clone(checkout, base, scratch)
            outcome = run_harness(harness, model=model, effort=effort, budget_usd=budget_usd, prompt=found["task"],
                                  checkout=clone, scratch=scratch)
            if not sum(outcome.get("tokens", {}).values()):
                # Without usage there is no result to score, and a record would read as one. Claude Code reports its
                # usage when it ends, so one stopped at the cap has none.
                reason = (f"stopped at the {HARNESS_SECONDS // 60}-minute cap before it reported its usage"
                          if outcome.get("stopped") == "time" else "made no request")
                raise ReplayError(f"{HARNESSES[harness]} {reason}, so there is nothing to score"
                                  f"{': ' + outcome['error'] if outcome.get('error') else '.'}")
            _git(clone, "add", "--all")
            replayed = change(clone, base)
            tests = _tests(clone, command, merged_sha, held_out)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        scores = score(merged, replayed)
        return {
            "repository": repository, "pr_number": pr, "model": model, "harness": harness, "effort": effort,
            "control": control, "replay_cost_usd": outcome.get("cost_usd"),
            "original_cost_usd": found["original_cost_usd"],
            "files_jaccard": scores["files_jaccard"], "line_overlap": scores["line_overlap"],
            "tests_passed": tests["passed"], "held_out_tests_passed": tests["held_out_passed"],
            "lines_added": replayed["lines_added"],
            "record": {
                "version": 1,
                "replayed_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "base_sha": base, "merged_sha": merged_sha, "task_source": found["source"],
                "original_model": found["original_model"], "original_effort": found["original_effort"],
                "budget_usd": budget_usd, "test_command": command,
                "tests": {key: tests[key] for key in ("replay_exit", "held_out_exit", "merged_exit")},
                "held_out": ({"files": held_out[:MAX_PATHS]} if held_out and command else
                             {"files": held_out[:MAX_PATHS], "skipped": "Joyride found no test command to run them."}
                             if held_out else {"skipped": "The merged change added or changed no test files."}),
                "classes": scores["classes"],
                "merged": {"files": merged["files"][:MAX_PATHS], "lines_added": merged["lines_added"],
                           "lines_deleted": merged["lines_deleted"], "compared_lines": scores["compared_lines"]},
                "replay": {"files": replayed["files"][:MAX_PATHS], "lines_added": replayed["lines_added"],
                           "lines_deleted": replayed["lines_deleted"], "shared_files": scores["shared_files"],
                           "shared_lines": scores["shared_lines"],
                           **{key: outcome.get(key)
                              for key in ("exit_code", "stopped", "outcome", "seconds", "tokens", "models")}},
            },
        }

    # Each record prints as it finishes, one line of JSON with --json, so a record that the service refuses
    # still shows.
    for index, item in enumerate(runs):
        record = replay(*item)
        path = save(record, state_dir)
        output.write((json.dumps(record, ensure_ascii=False) if as_json
                      else ("\n" if index else "") + render(record, path)) + "\n")
        if origin:
            post(origin, token, record)
            if not as_json:
                output.write(f"Recorded on {origin}.\n")
    return 0
