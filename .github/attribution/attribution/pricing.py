"""Prices for every model Joyride estimates, with the source of each price.

Two kinds of data feed this module, and neither reaches the network:

* ``model_prices.MODELS`` is a generated snapshot of the first-party OpenAI and
  Anthropic entries of https://models.dev/api.json. It holds the API prices
  per 1M tokens, the long-context tier, and the fast-mode prices. Refresh it
  with ``python3 scripts/update_model_prices.py``.
* The tables below were read by hand from official pages. Each table names its
  page and the date it was read. They hold what the snapshot lacks: Codex
  credit rates, credit multipliers for fast mode, models the catalog does not
  list, aliases, and proxy prices.

Every function returns a ``Price``. A request that cannot be priced carries an
``unpriced_reason`` from a fixed set, so a report can say why a cost is unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import re
from typing import Any, Collection, Mapping

from . import model_prices


_MILLION = Decimal(1_000_000)

# Pages read by hand on 2026-09-23.
OPENAI_API_PRICING_URL = "https://developers.openai.com/api/docs/pricing"
OPENAI_MODEL_PAGE_URL = "https://developers.openai.com/api/docs/models/gpt-6-sol"
CODEX_CREDIT_RATES_URL = "https://learn.chatgpt.com/docs/pricing"
CODEX_SPEED_URL = "https://learn.chatgpt.com/docs/agent-configuration/speed"
ANTHROPIC_PRICING_URL = "https://platform.claude.com/docs/en/about-claude/pricing"
HAND_READ_ON = "2026-09-23"

# Credits per 1M input, cached input, and output tokens, from the Codex
# pricing page. "Codex credit billing has no separate cache-write charge."
CODEX_CREDIT_RATES: dict[str, dict[str, Decimal]] = {
    "gpt-6-astra": {"input": Decimal("250"), "cached_input": Decimal("25"), "output": Decimal("1250")},
    "gpt-6-sol": {"input": Decimal("50"), "cached_input": Decimal("5"), "output": Decimal("250")},
    "gpt-6-luna": {"input": Decimal("2.5"), "cached_input": Decimal("0.25"), "output": Decimal("12.5")},
    "gpt-5.6-sol": {"input": Decimal("100"), "cached_input": Decimal("10"), "output": Decimal("500")},
    "gpt-5.6-cyber": {"input": Decimal("312.5"), "cached_input": Decimal("31.25"), "output": Decimal("1875")},
    "gpt-5.6-terra": {"input": Decimal("50"), "cached_input": Decimal("5"), "output": Decimal("300")},
    "gpt-5.6-luna": {"input": Decimal("5"), "cached_input": Decimal("0.5"), "output": Decimal("30")},
    "gpt-5.5": {"input": Decimal("125"), "cached_input": Decimal("12.5"), "output": Decimal("750")},
    "gpt-5.4": {"input": Decimal("62.5"), "cached_input": Decimal("6.25"), "output": Decimal("375")},
    "gpt-5.4-mini": {"input": Decimal("18.75"), "cached_input": Decimal("1.875"), "output": Decimal("113")},
}

# Fast mode credit multipliers, from the Codex speed page: "Fast mode consumes
# credits at 2.5x the Standard rate" for GPT-6, "GPT-5.6 and GPT-5.5 consume
# credits at 2.5x", and "GPT-5.4 consumes credits at 2x".
CODEX_CREDIT_FAST_MULTIPLIERS: dict[str, Decimal] = {
    "gpt-6-astra": Decimal("2.5"),
    "gpt-6-sol": Decimal("2.5"),
    "gpt-6-luna": Decimal("2.5"),
    "gpt-5.6-sol": Decimal("2.5"),
    "gpt-5.6-cyber": Decimal("2.5"),
    "gpt-5.6-terra": Decimal("2.5"),
    "gpt-5.6-luna": Decimal("2.5"),
    "gpt-5.5": Decimal("2.5"),
    "gpt-5.4": Decimal("2"),
    "gpt-5.4-mini": Decimal("2"),
}

# API prices the catalog does not list, from the OpenAI pricing page. The page
# publishes no long-context or fast price for this model, so a request that
# needs one stays unpriced.
OPENAI_API_EXTRA_MODELS: dict[str, dict[str, Any]] = {
    "gpt-5.6-cyber": {
        "input": "12.5",
        "cache_read": "1.25",
        "cache_write": "15.625",
        "output": "75",
        "long_context_unpublished": True,
    },
}

# "gpt-daybreak-blue-latest and gpt-daybreak-red-latest are aliases that
# currently point to gpt-5.6-sol and gpt-5.6-cyber, respectively."
OPENAI_ALIASES: dict[str, str] = {
    "gpt-daybreak-blue-latest": "gpt-5.6-sol",
    "gpt-daybreak-red-latest": "gpt-5.6-cyber",
}

# Models with no published price that still bill work. Codex runs its approval
# reviewer on codex-auto-review under a ChatGPT sign-in and on gpt-5.6-luna
# under an API key, so gpt-5.6-luna prices the same work as a labeled proxy.
OPENAI_PROXY_PRICES: dict[str, str] = {"codex-auto-review": "gpt-5.6-luna"}
OPENAI_NO_PUBLISHED_PRICE: frozenset[str] = frozenset({"gpt-reserve"})

# "Prompts with more than 272K input tokens are priced at 2x input and cache
# rates and 1.5x output for the full request." "Batch and Flex are priced at
# 50% of Standard rates. Fast mode is priced at 2x the applicable rates."
OPENAI_DEFAULT_FAST_MULTIPLIER = Decimal("2")
OPENAI_HALF_PRICE_TIERS = frozenset({"flex", "batch"})
CODEX_LONG_CONTEXT_THRESHOLD = 272_000

# Anthropic cache rules: "5-minute cache write 1.25x base input price, 1-hour
# cache write 2x base input price, cache read (hit) 0.1x base input price."
ANTHROPIC_CACHE_WRITE_5M = Decimal("1.25")
ANTHROPIC_CACHE_WRITE_1H = Decimal("2")
ANTHROPIC_CACHE_READ = Decimal("0.1")

CHATGPT_AUTH_MODES = frozenset({"chatgpt", "swic"})
API_KEY_AUTH_MODES = frozenset({"api", "api_key", "apikey"})
_STANDARD_TIERS = frozenset({"", "default", "standard", "auto"})
_FAST_TIERS = frozenset({"fast", "priority"})

SYNTHETIC_MODEL = "<synthetic>"

UNPRICED_REASON_KINDS = (
    "missing_model",
    "missing_tokens",
    "unknown_model",
    "no_published_price",
    "no_long_context_price",
    "unknown_service_tier",
)
_MAX_REASON_DETAIL = 256

_PREFIXES = ("global.openai.", "openai.", "openai/", "anthropic/")
_ONE_MILLION_CONTEXT_SUFFIX = "[1m]"
_DATE_SUFFIX = re.compile(r"^(?P<base>.+?)-(?:\d{4}-\d{2}-\d{2}|\d{8})$")


@dataclass(frozen=True)
class Price:
    """One priced request, or the reason it has no price."""

    amount: Decimal | None
    unit: str | None = None
    source: str | None = None
    model: str | None = None
    proxy_for: str | None = None
    unpriced_reason: str | None = None


def _reason(kind: str, detail: str | None = None) -> str:
    if detail is None:
        return kind
    return f"{kind}:{detail[:_MAX_REASON_DETAIL]}"


def _unpriced(kind: str, detail: str | None = None) -> Price:
    return Price(amount=None, unpriced_reason=_reason(kind, detail))


def normalize_model(model: Any) -> str | None:
    """Return a model name without provider prefixes, context suffix, or case."""

    if not isinstance(model, str):
        return None
    text = model.strip().casefold()
    if text.endswith(_ONE_MILLION_CONTEXT_SUFFIX):
        text = text[: -len(_ONE_MILLION_CONTEXT_SUFFIX)].rstrip()
    for prefix in _PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text or None


def _match(name: str, names: Collection[str]) -> str | None:
    """Find the catalog name for a model name, or None.

    An exact name wins. A dated snapshot maps onto its base only when the base
    is itself a catalog name. Otherwise the longest catalog name that ends at a
    ``-`` or ``.`` boundary decides. When the rest after it starts with a
    digit, the name is a new version, so it has no price: no shorter catalog
    name may price it instead.
    """

    if name in names:
        return name
    dated = _DATE_SUFFIX.match(name)
    if dated is not None and dated.group("base") in names:
        return dated.group("base")
    candidates = [
        candidate
        for candidate in names
        if len(name) > len(candidate) + 1
        and name.startswith(candidate)
        and name[len(candidate)] in "-."
    ]
    if not candidates:
        return None
    best = max(candidates, key=len)
    return None if name[len(best) + 1].isdigit() else best


@dataclass(frozen=True)
class _Resolved:
    key: str | None
    proxy_for: str | None = None
    reason: str | None = None


def _openai_names() -> set[str]:
    return (
        set(model_prices.MODELS.get("openai", {}))
        | set(OPENAI_API_EXTRA_MODELS)
        | set(OPENAI_ALIASES)
        | set(OPENAI_PROXY_PRICES)
        | set(OPENAI_NO_PUBLISHED_PRICE)
        | set(CODEX_CREDIT_RATES)
    )


def _resolve_openai(model: Any) -> _Resolved:
    name = normalize_model(model)
    if name is None:
        return _Resolved(None, reason=_reason("missing_model"))
    matched = _match(name, _openai_names())
    if matched is None:
        return _Resolved(None, reason=_reason("unknown_model", name))
    if matched in OPENAI_NO_PUBLISHED_PRICE:
        return _Resolved(None, reason=_reason("no_published_price", matched))
    if matched in OPENAI_PROXY_PRICES:
        target = OPENAI_PROXY_PRICES[matched]
        return _Resolved(target, proxy_for=target)
    return _Resolved(OPENAI_ALIASES.get(matched, matched))


def _resolve_anthropic(model: Any) -> _Resolved:
    name = normalize_model(model)
    if name is None:
        return _Resolved(None, reason=_reason("missing_model"))
    matched = _match(name, set(model_prices.MODELS.get("anthropic", {})))
    if matched is None:
        return _Resolved(None, reason=_reason("unknown_model", name))
    return _Resolved(matched)


def _decimal(value: Any) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def _rate_or(value: Any, fallback: Decimal) -> Decimal:
    rate = _decimal(value)
    return fallback if rate is None else rate


def _count(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return max(number, 0)


def _tier(service_tier: Any) -> str:
    return str(service_tier or "").strip().casefold()


def _source(label: str, day: str, proxy_for: str | None) -> str:
    source = f"{label}_{day}"
    return f"{source}_proxy_{proxy_for}" if proxy_for else source


def _openai_entry(key: str) -> tuple[Mapping[str, Any] | None, str]:
    entry = model_prices.MODELS.get("openai", {}).get(key)
    if entry is not None:
        return entry, model_prices.FETCHED_ON
    extra = OPENAI_API_EXTRA_MODELS.get(key)
    return extra, HAND_READ_ON


def price_openai_api(
    model: Any,
    input_tokens: Any,
    cached_input_tokens: Any,
    output_tokens: Any,
    *,
    cache_write_input_tokens: Any = None,
    service_tier: Any = None,
    label: str = "openai_api_standard_rate",
) -> Price:
    """Price one OpenAI request in US dollars at the published API rates.

    OpenAI reports input tokens that include the cached and cache-written
    tokens, so those parts come out of the input count before it is priced.
    """

    resolved = _resolve_openai(model)
    if resolved.key is None:
        return Price(amount=None, unpriced_reason=resolved.reason)
    entry, day = _openai_entry(resolved.key)
    if entry is None:
        return _unpriced("unknown_model", resolved.key)
    input_count = _count(input_tokens)
    output_count = _count(output_tokens)
    if input_count is None or output_count is None:
        return _unpriced("missing_tokens")

    rates: Mapping[str, Any] = entry
    long_context = entry.get("long_context")
    threshold = (
        long_context.get("threshold", CODEX_LONG_CONTEXT_THRESHOLD)
        if isinstance(long_context, Mapping)
        else CODEX_LONG_CONTEXT_THRESHOLD
    )
    if input_count > threshold:
        if isinstance(long_context, Mapping):
            rates = long_context
        elif entry.get("long_context_unpublished"):
            return _unpriced("no_long_context_price", resolved.key)

    tier = _tier(service_tier)
    multiplier = Decimal("1")
    if tier in _FAST_TIERS:
        fast = entry.get("fast")
        if isinstance(fast, Mapping) and fast.get("input") is not None:
            multiplier = Decimal(str(fast["input"])) / Decimal(str(entry["input"]))
        elif entry.get("long_context_unpublished"):
            return _unpriced("unknown_service_tier", tier)
        else:
            multiplier = OPENAI_DEFAULT_FAST_MULTIPLIER
    elif tier in OPENAI_HALF_PRICE_TIERS:
        multiplier = Decimal("0.5")
    elif tier not in _STANDARD_TIERS:
        return _unpriced("unknown_service_tier", tier)

    input_rate = _decimal(rates["input"])
    output_rate = _decimal(rates["output"])
    # An unpublished cache price bills like uncached input.
    cache_read_rate = _rate_or(rates.get("cache_read"), input_rate)
    cache_write_rate = _rate_or(rates.get("cache_write"), input_rate)
    cached = min(_count(cached_input_tokens) or 0, input_count)
    written = min(_count(cache_write_input_tokens) or 0, input_count - cached)
    uncached = input_count - cached - written
    amount = multiplier * (
        Decimal(uncached) * input_rate
        + Decimal(cached) * cache_read_rate
        + Decimal(written) * cache_write_rate
        + Decimal(output_count) * output_rate
    ) / _MILLION
    return Price(
        amount=amount,
        unit="USD",
        source=_source(label, day, resolved.proxy_for),
        model=resolved.key,
        proxy_for=resolved.proxy_for,
    )


def price_codex_credits(
    model: Any,
    input_tokens: Any,
    cached_input_tokens: Any,
    output_tokens: Any,
    *,
    cache_write_input_tokens: Any = None,
    service_tier: Any = None,
    input_includes_cached: bool = True,
) -> Price:
    """Price one Codex request in ChatGPT credits at the Codex credit rates.

    The credit table publishes no long-context, Batch, or Flex rule, so those
    requests have no credit price and a caller falls back to API dollars.
    """

    resolved = _resolve_openai(model)
    if resolved.key is None:
        return Price(amount=None, unpriced_reason=resolved.reason)
    rates = CODEX_CREDIT_RATES.get(resolved.key)
    if rates is None:
        return _unpriced("no_published_price", resolved.key)
    input_count = _count(input_tokens)
    output_count = _count(output_tokens)
    if input_count is None or output_count is None:
        return _unpriced("missing_tokens")
    if input_count > CODEX_LONG_CONTEXT_THRESHOLD:
        return _unpriced("no_long_context_price", resolved.key)
    tier = _tier(service_tier)
    if tier in _STANDARD_TIERS:
        multiplier = Decimal("1")
    elif tier in _FAST_TIERS and resolved.key in CODEX_CREDIT_FAST_MULTIPLIERS:
        multiplier = CODEX_CREDIT_FAST_MULTIPLIERS[resolved.key]
    else:
        return _unpriced("unknown_service_tier", tier)
    cached = _count(cached_input_tokens) or 0
    written = _count(cache_write_input_tokens) or 0
    output = output_count
    if input_includes_cached:
        cached = min(cached, input_count)
        written = min(written, input_count - cached)
        # Codex credit billing has no separate cache-write charge.
        uncached = input_count - cached - written
    else:
        uncached = input_count
    amount = multiplier * (
        Decimal(uncached) * rates["input"]
        + Decimal(cached) * rates["cached_input"]
        + Decimal(output) * rates["output"]
    ) / _MILLION
    return Price(
        amount=amount,
        unit="credits",
        source=_source("chatgpt_credit_rate", HAND_READ_ON, resolved.proxy_for),
        model=resolved.key,
        proxy_for=resolved.proxy_for,
    )


def price_codex_request(
    model: Any,
    auth_mode: Any,
    input_tokens: Any,
    cached_input_tokens: Any,
    output_tokens: Any,
    *,
    cache_write_input_tokens: Any = None,
    service_tier: Any = None,
) -> Price:
    """Price one Codex request on the basis its sign-in bills.

    An API key bills dollars at the API rates. A ChatGPT sign-in bills credits.
    When no credit price applies, or the sign-in is unknown, the request still
    gets a dollar estimate at the API rates, labeled as an API equivalent.
    """

    if normalize_model(model) is None:
        return _unpriced("missing_model")
    auth = str(auth_mode or "").strip().casefold()
    tokens = (input_tokens, cached_input_tokens, output_tokens)
    options = {
        "cache_write_input_tokens": cache_write_input_tokens,
        "service_tier": service_tier,
    }
    if auth in API_KEY_AUTH_MODES:
        return price_openai_api(model, *tokens, **options)
    if auth in CHATGPT_AUTH_MODES:
        credits = price_codex_credits(model, *tokens, **options)
        if credits.amount is not None:
            return credits
    return price_openai_api(model, *tokens, label="openai_api_equivalent_rate", **options)


def price_anthropic(
    model: Any,
    input_tokens: Any,
    cache_read_input_tokens: Any,
    cache_creation_input_tokens: Any,
    output_tokens: Any,
    *,
    cache_creation_1h_input_tokens: Any = None,
    service_tier: Any = None,
) -> Price:
    """Price one Claude request in US dollars at the Anthropic list prices.

    Anthropic reports uncached input, cache reads, and cache writes as separate
    counts. Cache writes bill at the 5-minute price unless the 1-hour part is
    given. Fast mode applies across the full context window, and the cache
    multipliers apply on top of it.
    """

    if isinstance(model, str) and model.strip() == SYNTHETIC_MODEL:
        return Price(amount=Decimal("0"), unit="USD", source="no_request")
    resolved = _resolve_anthropic(model)
    if resolved.key is None:
        return Price(amount=None, unpriced_reason=resolved.reason)
    entry = model_prices.MODELS["anthropic"][resolved.key]
    input_count = _count(input_tokens)
    output_count = _count(output_tokens)
    if input_count is None or output_count is None:
        return _unpriced("missing_tokens")
    tier = _tier(service_tier)
    rates: Mapping[str, Any] = entry
    multiplier = Decimal("1")
    if tier == "fast":
        # A model without fast prices runs a fast request at standard rates.
        fast = entry.get("fast")
        if isinstance(fast, Mapping):
            rates = fast
    elif tier == "batch":
        multiplier = Decimal("0.5")
    elif tier not in _STANDARD_TIERS:
        return _unpriced("unknown_service_tier", tier)
    input_rate = _decimal(rates["input"])
    output_rate = _decimal(rates["output"])
    read_rate = _rate_or(rates.get("cache_read"), input_rate * ANTHROPIC_CACHE_READ)
    write_5m_rate = _rate_or(rates.get("cache_write"), input_rate * ANTHROPIC_CACHE_WRITE_5M)
    write_1h_rate = input_rate * ANTHROPIC_CACHE_WRITE_1H
    creation = _count(cache_creation_input_tokens) or 0
    creation_1h = min(_count(cache_creation_1h_input_tokens) or 0, creation)
    amount = multiplier * (
        Decimal(input_count) * input_rate
        + Decimal(_count(cache_read_input_tokens) or 0) * read_rate
        + Decimal(creation - creation_1h) * write_5m_rate
        + Decimal(creation_1h) * write_1h_rate
        + Decimal(output_count) * output_rate
    ) / _MILLION
    return Price(
        amount=amount,
        unit="USD",
        source=_source("joyride_list_price", model_prices.FETCHED_ON, None),
        model=resolved.key,
    )


def price_event(provider: Any, event: Mapping[str, Any]) -> Price:
    """Price one stored telemetry event from its model and token counts."""

    name = str(provider or "").strip().casefold()
    if name == "claude":
        return price_anthropic(
            event.get("model"),
            event.get("input_tokens"),
            event.get("cached_input_tokens"),
            event.get("cache_creation_input_tokens"),
            event.get("output_tokens"),
            cache_creation_1h_input_tokens=event.get("cache_creation_1h_input_tokens"),
            service_tier=event.get("service_tier"),
        )
    if name in {"codex", "openai"}:
        return price_codex_request(
            event.get("model"),
            event.get("auth_mode"),
            event.get("input_tokens"),
            event.get("cached_input_tokens"),
            event.get("output_tokens"),
            cache_write_input_tokens=event.get("cache_creation_input_tokens"),
            service_tier=event.get("service_tier"),
        )
    return _unpriced("unknown_model", normalize_model(event.get("model")) or name or None)


__all__ = [
    "API_KEY_AUTH_MODES",
    "CHATGPT_AUTH_MODES",
    "CODEX_CREDIT_FAST_MULTIPLIERS",
    "CODEX_CREDIT_RATES",
    "OPENAI_ALIASES",
    "OPENAI_API_EXTRA_MODELS",
    "OPENAI_NO_PUBLISHED_PRICE",
    "OPENAI_PROXY_PRICES",
    "Price",
    "UNPRICED_REASON_KINDS",
    "normalize_model",
    "price_anthropic",
    "price_codex_credits",
    "price_codex_request",
    "price_event",
    "price_openai_api",
]
