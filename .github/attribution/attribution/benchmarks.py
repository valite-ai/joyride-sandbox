"""Public coding benchmarks for the models in the price snapshot: the weakest of three layers of evidence.

Whether a cheaper model would have done the same work has three answers of falling strength: the developer's
own matched history (``hosted.next_steps_models.by_task_type``), a replay of a past pull request on the cheaper
model (``attribution/replay_command.py``), and the gap between the two models on a public benchmark, here. A
public gap says whether two models are in the same class on tasks like the benchmark's. It never says what a
model would have done with one of the developer's tasks, so the engine never makes a card from it alone.

``benchmarks.json`` holds one score per model and benchmark id, each with its unit, the day it was published,
its source, who reported it (``vendor``, ``competitor``, ``leaderboard``, or ``independent``), and the effort it
ran at, for models that ``attribution/model_prices.py`` prices. The scores were chosen on 2026-09-27, one
protocol per benchmark id, and ``scripts/refresh_benchmarks.py`` rewrites them from the same sources. Every
function answers None, or no scores, when the snapshot lacks what it needs: nothing is estimated from a
missing score.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from attribution import model_prices, pricing


SNAPSHOT = Path(__file__).with_name("benchmarks.json")

# The benchmark ids that speak for each task type (hosted.next_steps_models.task_type), in the order gap()
# tries them; it uses the first one on which both models have a score. From the research of 2026-09-27, which
# names the study behind each row:
TASK_BENCHMARKS: dict[str, tuple[str, ...]] = {
    # None: no benchmark grades prose, and a documentation change passes no test.
    "documentation": (),
    # SWT-bench, writing tests that reproduce an issue, in the OpenHands Index: the only test-writing benchmark
    # with snapshot models.
    "tests": ("openhands-index-testing",),
    # Renames, moves, and bumps are the easy end of agentic work, and no benchmark isolates them: LiveBench's
    # fresh agentic coding questions, then Vals.ai's Terminal-Bench 2.1 run. The hard benchmarks would overstate
    # the gap on easy work.
    "mechanical refactor": ("livebench-agentic-coding-2026-06-25", "terminal-bench-2.1-vals"),
    # Cognition's FrontierCode 1.1 judges whether a change would be merged; Cursor's CursorBench 4.0 traces its
    # tasks from committed code back to the agent request; Vals.ai's Vibe Code Bench builds an app and runs its
    # tests; Scale's SWE-Bench Pro V2 dropped its invalid tasks. They measure feature work whether or not the
    # change adds tests.
    "feature": ("frontiercode-v1.1-main", "cursorbench-4.0", "vibe-code-bench-v1.1-vals", "swe-bench-pro-v2-public"),
    # Issues fixed and checked by tests: SWE-rebench's issues from after the models' data, Datacurve's DeepSWE,
    # OpenAI's DeepSWE runs, SWE-Bench Pro V2, then Terminal-Bench 2.1.
    "bug fix": ("swe-rebench-2026-05-15-to-07-01", "deepswe-v1.1", "deepswe-v1.1-openai", "swe-bench-pro-v2-public",
                "terminal-bench-2.1-vals"),
    # Terminal-Bench's builds, services, scripts, and system tasks: Vals.ai's runs first, which use one harness for
    # every model, and 2.1 before 4.0, since routine ops work is nearer its difficulty.
    "shell and ops": ("terminal-bench-2.1-vals", "terminal-bench-4.0-vals", "terminal-bench-2.1", "terminal-bench-4.0"),
    # None: no benchmark measures reading a codebase to plan.
    "research or planning": (),
}
# The bands of a gap in points. Anthropic measured 6 points between the most and least resourced setups of one
# benchmark and found gaps under 3 points to deserve skepticism (2026-02-05), so a near match ends just above
# that noise, at 5; past 15 points the public gap is wide.
NEAR_POINTS = 5.0
TRY_POINTS = 15.0
# A blended price weighs 3 input tokens to 1 output token, as public price comparisons do.
BLEND_INPUT = 3


@dataclass(frozen=True)
class Score:
    """One published score of one model on one benchmark id."""

    model: str
    benchmark: str
    score: float
    unit: str
    date: str
    source: str
    reported_by: str
    effort: str | None


@dataclass(frozen=True)
class Gap:
    """Two models' scores on one benchmark id of a task type; ``points`` is how far ``b`` trails ``a``."""

    task_type: str
    a: Score
    b: Score

    @property
    def benchmark(self) -> str:
        return self.a.benchmark

    @property
    def points(self) -> float:
        return round(self.a.score - self.b.score, 2)

    @property
    def band(self) -> str:
        """``near`` within 5 points, a cheaper model ahead included; ``try`` within 15; else ``wide``."""
        return "near" if self.points <= NEAR_POINTS else "try" if self.points <= TRY_POINTS else "wide"


@lru_cache(maxsize=1)
def _index() -> dict[tuple[str, str], Score]:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    return {(item["model"], item["benchmark"]): Score(**item) for item in snapshot["scores"]}


def _name(model: Any, names: Any) -> str | None:
    """A recorded model's name among ``names``, without a prefix, a date, or a context suffix; an alias is its model."""
    name = pricing.normalize_model(model)
    return pricing._match(pricing.OPENAI_ALIASES.get(name, name), names) if name else None


def scores(model: Any) -> list[Score]:
    """The snapshot's scores of a recorded model, by benchmark id; none for a model it does not hold."""
    name = _name(model, {key for key, _ in _index()})
    return sorted((item for key, item in _index().items() if key[0] == name), key=lambda item: item.benchmark)


def gap(model_a: Any, model_b: Any, task_type: str | None) -> Gap | None:
    """How far ``model_b`` trails ``model_a`` on the first benchmark of the task type that both have a score on.

    None when no benchmark of the task type holds a score of both, or when the
    two are one model.
    """
    names = {key for key, _ in _index()}
    a, b = _name(model_a, names), _name(model_b, names)
    if a is None or b is None or a == b:
        return None
    for benchmark in TASK_BENCHMARKS.get(task_type or "", ()):
        if (a, benchmark) in _index() and (b, benchmark) in _index():
            return Gap(task_type, _index()[a, benchmark], _index()[b, benchmark])
    return None


def blended_price(model: Any) -> float | None:
    """The catalog's dollars per million tokens at 3 input tokens to 1 output token, or None without both prices."""
    entries = {key: entry for models in model_prices.MODELS.values() for key, entry in models.items()}
    name = _name(model, set(entries))
    entry = entries.get(name) if name else None
    if entry is None or entry.get("input") is None or entry.get("output") is None:
        return None
    return (BLEND_INPUT * float(entry["input"]) + float(entry["output"])) / (BLEND_INPUT + 1)


def capability_per_dollar(model: Any, task_type: str | None) -> float | None:
    """The model's score on the first benchmark of the task type that it has, over its blended price, for ranking.

    Two models rank alike only when their scores come from the same benchmark,
    which gap() names.
    """
    name, price = _name(model, {key for key, _ in _index()}), blended_price(model)
    found = next((_index()[name, benchmark] for benchmark in TASK_BENCHMARKS.get(task_type or "", ())
                  if (name, benchmark) in _index()), None)
    return found.score / price if found is not None and price else None
