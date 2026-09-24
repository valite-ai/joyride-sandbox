"""Readable reasons for a cost that stays unknown, and the proxy-price label.

The pricing module stores the reason of each unpriced request as "<kind>" or
"<kind>:<detail>". Reports sum those request counts per session and per agent,
and these helpers give the terminal, the footer, and the pages one wording.
The detail comes from telemetry, so each caller escapes it for its medium.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from .pricing import UNPRICED_REASON_KINDS

MAX_REASONS = 16
MAX_DETAIL_CHARS = 120
# The reasons past the largest ones fold into this one count, so a total over
# many sessions stays within MAX_REASONS without losing a request.
OTHER_REASON = "other"
# An input count above this is not a real request count, so it is discarded,
# and a sum of accepted counts is clamped to it, so every total stays valid.
_MAX_COUNT = 1e9
_PROXY_MARK = "_proxy_"
_WORDS = {
    "missing_model": "no model name",
    "missing_tokens": "no token counts",
    "unknown_model": "no price for {detail}",
    "no_published_price": "no published price for {detail}",
    "no_long_context_price": "no long-context price for {detail}",
    "unknown_service_tier": "unknown service tier {detail}",
    OTHER_REASON: "other reasons",
}


def _parts(reason: Any) -> tuple[str, str | None] | None:
    """Split one stored reason into its kind and detail, or None if malformed."""
    if not isinstance(reason, str):
        return None
    if reason == OTHER_REASON:
        return reason, None
    kind, separator, detail = reason.partition(":")
    if kind not in UNPRICED_REASON_KINDS:
        return None
    if not separator:
        return kind, None
    if (
        not detail.strip()
        or len(detail) > MAX_DETAIL_CHARS
        or any(character < " " or character == "\x7f" for character in detail)
        or not _encodable(detail)
    ):
        return None
    return kind, detail


def _encodable(text: str) -> bool:
    """Return whether text can be published as UTF-8 (no lone surrogate)."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _count(value: Any) -> float | None:
    """Return one request count, or None when it is malformed or oversized."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    # A huge integer cannot become a float, so it is checked as an integer.
    if isinstance(value, int) and not 0 < value <= _MAX_COUNT:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    # A count is kept to three decimals, so one that rounds to zero is dropped.
    number = round(number, 3)
    return number if 0 < number <= _MAX_COUNT else None


def limit_reasons(reasons: Mapping[str, float]) -> dict[str, float]:
    """Keep the largest reasons and fold the rest into one ``other`` count."""
    other = reasons.get(OTHER_REASON, 0.0)
    ranked = sorted(
        ((reason, count) for reason, count in reasons.items() if reason != OTHER_REASON),
        key=lambda item: (-item[1], item[0]),
    )
    if len(ranked) + (1 if other else 0) > MAX_REASONS:
        kept, folded = ranked[:MAX_REASONS - 1], ranked[MAX_REASONS - 1:]
        other = min(other + sum(count for _reason, count in folded), _MAX_COUNT)
        ranked = kept
    limited = dict(ranked)
    if other:
        limited[OTHER_REASON] = round(other, 3)
    return dict(sorted(limited.items()))


def sanitize_reasons(raw: Any) -> dict[str, float]:
    """Keep the well-formed reasons of an untrusted map, within fixed bounds.

    A malformed entry is dropped on its own and never raises, because a reason
    only explains a cost and must not block the report that carries it.
    """
    if not isinstance(raw, Mapping):
        return {}
    kept: dict[str, float] = {}
    for reason, count in raw.items():
        number = _count(count)
        if number is None or _parts(reason) is None:
            continue
        kept[reason] = number
    return limit_reasons(kept)


def add_reasons(total: dict[str, float], reasons: Any) -> None:
    """Add one session's reason counts to a running total.

    The total can pass MAX_REASONS, so a caller applies limit_reasons to it
    before it publishes or shows the total.
    """
    for reason, count in sanitize_reasons(reasons).items():
        total[reason] = min(round(total.get(reason, 0.0) + count, 3), _MAX_COUNT)


def is_proxy_priced(sources: Any) -> bool:
    """Return whether any cost source priced a request at a proxy model's rate."""
    return isinstance(sources, (list, tuple, set, frozenset)) and any(
        isinstance(source, str) and _PROXY_MARK in source for source in sources
    )


def describe(
    reasons: Any,
    sessions_without_usage: int = 0,
    escape: Callable[[str], str] = str,
) -> str | None:
    """Say why a cost is unknown, most common reason first, or None."""
    parts = []
    counted = sanitize_reasons(reasons)
    ordered = sorted(
        counted.items(), key=lambda item: (item[0] == OTHER_REASON, -item[1], item[0])
    )
    for reason, count in ordered:
        kind, detail = _parts(reason)  # type: ignore[misc]
        words = _WORDS[kind].format(
            detail=escape(detail) if detail is not None else "an unnamed model"
        )
        requests = max(1, math.floor(count + 0.5))
        parts.append(f"{words} ({requests} {'request' if requests == 1 else 'requests'})")
    if sessions_without_usage > 0:
        noun = "session" if sessions_without_usage == 1 else "sessions"
        parts.append(f"no usage data for {sessions_without_usage} {noun}")
    return " · ".join(parts) or None


__all__ = [
    "MAX_REASONS", "OTHER_REASON", "add_reasons", "describe", "is_proxy_priced",
    "limit_reasons", "sanitize_reasons",
]
