"""Retry timing and per-attempt delivery budgets."""

from dataclasses import dataclass
import random


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Calculate exponential retry delays with full jitter."""

    max_attempts: int = 5
    base_delay: float = 1.0
    multiplier: float = 2.0
    max_delay: float = 60.0
    max_age: float = 3600.0

    def next_delay(
        self,
        attempt: int,
        age: float,
        retry_after: float | None = None,
    ) -> float | None:
        if attempt >= self.max_attempts:
            return None

        raw_delay = min(
            self.max_delay,
            self.base_delay * self.multiplier ** (attempt - 1),
        )
        delay = random.uniform(0.0, raw_delay)
        if retry_after is not None:
            delay = max(delay, min(self.max_delay, max(0.0, retry_after)))

        if age + delay > self.max_age:
            return None
        return delay


@dataclass(frozen=True, slots=True)
class DeliveryBudget:
    """Limit one delivery attempt to a number of seconds."""

    seconds: float = 10.0
