"""An in-memory queue for webhook deliveries."""

from dataclasses import dataclass
import heapq
import itertools
import time
from typing import Any, Callable
from uuid import uuid4

from .client import Client, Outcome, DELIVERED, PERMANENT
from .retry import DeliveryBudget, RetryPolicy


@dataclass(slots=True)
class Delivery:
    id: str
    url: str
    payload: Any
    attempts: int
    created_at: float
    due_at: float
    last_outcome: Outcome | None = None


@dataclass(frozen=True, slots=True)
class DeadLetter:
    delivery: Delivery
    last_outcome: Outcome
    reason: str


class DeliveryQueue:
    """Deliver due webhooks one attempt at a time."""

    def __init__(
        self,
        client: Client | None = None,
        policy: RetryPolicy | None = None,
        budget: DeliveryBudget | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.client = client if client is not None else Client()
        self.policy = policy if policy is not None else RetryPolicy()
        self.budget = budget if budget is not None else DeliveryBudget()
        self._clock = clock
        self._pending: list[tuple[float, int, Delivery]] = []
        self._sequence = itertools.count()
        self._dead_letters: list[DeadLetter] = []

    @property
    def dead_letters(self) -> list[DeadLetter]:
        return list(self._dead_letters)

    def submit(self, url: str, payload: Any) -> str:
        now = self._clock()
        delivery = Delivery(str(uuid4()), url, payload, 0, now, now)
        self._push(delivery)
        return delivery.id

    def run_once(self) -> Outcome | None:
        now = self._clock()
        if not self._pending or self._pending[0][0] > now:
            return None

        _, _, delivery = heapq.heappop(self._pending)
        delivery.attempts += 1
        outcome = self.client.post(
            delivery.url,
            delivery.payload,
            headers={"Webhook-Id": delivery.id},
            timeout=self.budget.seconds,
        )
        delivery.last_outcome = outcome

        if outcome.kind == DELIVERED:
            return outcome
        if outcome.kind == PERMANENT:
            self._dead_letters.append(
                DeadLetter(delivery, outcome, "permanent outcome")
            )
            return outcome

        now = self._clock()
        delay = self.policy.next_delay(
            delivery.attempts,
            now - delivery.created_at,
            outcome.retry_after,
        )
        if delay is None:
            self._dead_letters.append(
                DeadLetter(delivery, outcome, "retry policy exhausted")
            )
        else:
            delivery.due_at = now + delay
            self._push(delivery)
        return outcome

    def run_until_empty(self) -> None:
        while self.run_once() is not None:
            pass

    def _push(self, delivery: Delivery) -> None:
        heapq.heappush(
            self._pending,
            (delivery.due_at, next(self._sequence), delivery),
        )
