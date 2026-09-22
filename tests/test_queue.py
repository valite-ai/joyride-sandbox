import unittest
from unittest import mock

from webhook_relay.client import DELIVERED, PERMANENT, RETRYABLE, Outcome
from webhook_relay.queue import DeliveryQueue
from webhook_relay.retry import DeliveryBudget, RetryPolicy


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FakeClient:
    """Return scripted outcomes and record each call."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def post(self, url, payload, headers=None, timeout=10.0):
        self.calls.append((url, payload, headers, timeout))
        return self.outcomes.pop(0)


def make_queue(*outcomes, **policy):
    clock = FakeClock()
    client = FakeClient(*outcomes)
    queue = DeliveryQueue(client, RetryPolicy(**policy), DeliveryBudget(3.0), clock)
    return queue, client, clock


class DeliveryQueueTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("random.uniform", side_effect=lambda a, b: b)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_delivers_with_webhook_id_and_budget_timeout(self):
        queue, client, _ = make_queue(Outcome(DELIVERED, 200))
        delivery_id = queue.submit("http://partner/hook", {"a": 1})
        self.assertEqual(queue.run_once().kind, DELIVERED)
        self.assertEqual(client.calls, [
            ("http://partner/hook", {"a": 1}, {"Webhook-Id": delivery_id}, 3.0)])
        self.assertIsNone(queue.run_once())

    def test_retry_waits_for_due_time_then_succeeds(self):
        queue, client, clock = make_queue(Outcome(RETRYABLE, 503), Outcome(DELIVERED, 200))
        queue.submit("u", {})
        queue.run_until_empty()
        self.assertEqual(len(client.calls), 1)
        clock.now += 0.5
        self.assertIsNone(queue.run_once())
        clock.now += 0.5
        self.assertEqual(queue.run_once().kind, DELIVERED)
        self.assertEqual(queue.dead_letters, [])

    def test_retry_after_sets_due_time(self):
        queue, _, clock = make_queue(Outcome(RETRYABLE, 429, retry_after=20.0))
        queue.submit("u", {})
        queue.run_once()
        clock.now += 19.9
        self.assertIsNone(queue.run_once())

    def test_permanent_outcome_is_dead_lettered(self):
        queue, _, _ = make_queue(Outcome(PERMANENT, 404))
        queue.submit("u", {})
        queue.run_until_empty()
        [dead] = queue.dead_letters
        self.assertEqual((dead.reason, dead.last_outcome.status), ("permanent outcome", 404))
        self.assertEqual(dead.delivery.attempts, 1)

    def test_exhausted_policy_is_dead_lettered(self):
        queue, client, clock = make_queue(*[Outcome(RETRYABLE, 500)] * 3, max_attempts=3)
        queue.submit("u", {})
        for _ in range(3):
            clock.now += 60
            queue.run_once()
        [dead] = queue.dead_letters
        self.assertEqual((dead.reason, dead.delivery.attempts), ("retry policy exhausted", 3))
        self.assertEqual(len(client.calls), 3)

    def test_due_deliveries_run_in_due_order(self):
        queue, client, clock = make_queue(Outcome(RETRYABLE, 500), *[Outcome(DELIVERED, 200)] * 2)
        queue.submit("first", {})
        clock.now += 0.1
        queue.submit("second", {})
        queue.run_once()
        clock.now += 5
        queue.run_until_empty()
        self.assertEqual([call[0] for call in client.calls], ["first", "second", "first"])
