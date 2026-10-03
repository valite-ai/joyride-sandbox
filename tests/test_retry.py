import unittest
from unittest import mock

from webhook_relay.retry import DeliveryBudget, RetryPolicy


class RetryPolicyTest(unittest.TestCase):
    def test_full_jitter_uses_capped_exponential_delay(self):
        policy = RetryPolicy(max_attempts=10, max_delay=5.0)
        with mock.patch("random.uniform", side_effect=lambda a, b: b) as uniform:
            self.assertEqual(policy.next_delay(1, 0.0), 1.0)
            self.assertEqual(policy.next_delay(3, 0.0), 4.0)
            self.assertEqual(policy.next_delay(9, 0.0), 5.0)
        uniform.assert_called_with(0.0, 5.0)

    def test_retry_after_is_a_floor_capped_at_max_retry_after(self):
        policy = RetryPolicy(max_delay=30.0, max_retry_after=200.0)
        with mock.patch("random.uniform", return_value=0.0):
            self.assertEqual(policy.next_delay(1, 0.0, retry_after=7.0), 7.0)
            self.assertEqual(policy.next_delay(1, 0.0, retry_after=120.0), 120.0)
            self.assertEqual(policy.next_delay(1, 0.0, retry_after=900.0), 200.0)

    def test_default_policy_honors_acme_retry_after(self):
        policy = RetryPolicy()
        with mock.patch("random.uniform", return_value=0.0):
            self.assertEqual(policy.next_delay(1, 0.0, retry_after=120.0), 120.0)
            self.assertEqual(policy.next_delay(1, 0.0, retry_after=86400.0), 300.0)
        self.assertEqual(policy.max_delay, 60.0)

    def test_retry_after_still_respects_max_age(self):
        policy = RetryPolicy(max_age=3600.0)
        with mock.patch("random.uniform", return_value=0.0):
            self.assertEqual(policy.next_delay(1, 3480.0, retry_after=120.0), 120.0)
            self.assertIsNone(policy.next_delay(1, 3481.0, retry_after=120.0))

    def test_default_retry_after_cap_fits_every_attempt_in_max_age(self):
        policy = RetryPolicy()
        retries = policy.max_attempts - 1
        self.assertLessEqual(retries * policy.max_retry_after, policy.max_age)

    def test_gives_up_at_max_attempts_or_max_age(self):
        policy = RetryPolicy(max_attempts=3, max_age=10.0)
        with mock.patch("random.uniform", return_value=2.0):
            self.assertIsNotNone(policy.next_delay(2, 0.0))
            self.assertIsNone(policy.next_delay(3, 0.0))
            self.assertIsNone(policy.next_delay(1, 9.0))

    def test_budget_default(self):
        self.assertEqual(DeliveryBudget().seconds, 10.0)
