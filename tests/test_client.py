import json
import socket
import time
import unittest

from webhook_relay import __version__
from webhook_relay.client import DELIVERED, PERMANENT, RETRYABLE, Client
from tests.server import LocalServer, drip, garbage


class ClientTest(unittest.TestCase):
    def test_delivered_sends_json_and_headers(self):
        with LocalServer((204, {}, b"")) as server:
            outcome = Client().post(server.url + "/hook", {"a": 1}, {"Webhook-Id": "d1"})
        self.assertEqual((outcome.kind, outcome.status), (DELIVERED, 204))
        path, headers, body = server.requests[0]
        self.assertEqual((path, json.loads(body)), ("/hook", {"a": 1}))
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["User-Agent"], f"webhook-relay/{__version__}")
        self.assertEqual(headers["Webhook-Id"], "d1")

    def test_classifies_statuses(self):
        cases = [(408, RETRYABLE), (425, RETRYABLE), (429, RETRYABLE),
                 (503, RETRYABLE), (400, PERMANENT), (410, PERMANENT)]
        with LocalServer(*[(s, {}, b"") for s, _ in cases]) as server:
            for status, kind in cases:
                self.assertEqual(Client().post(server.url, {}).kind, kind, status)

    def test_does_not_follow_redirects(self):
        with LocalServer((302, {"Location": "/other"}, b"")) as server:
            outcome = Client().post(server.url, {})
        self.assertEqual((outcome.kind, outcome.status), (PERMANENT, 302))
        self.assertEqual(len(server.requests), 1)

    def test_error_body_is_capped_at_4_kib(self):
        with LocalServer((500, {}, b"x" * 10000)) as server:
            outcome = Client().post(server.url, {})
        self.assertEqual(outcome.error, "x" * 4096)

    def test_retry_after_seconds_and_http_date(self):
        now = 1_000_000_000.0  # Sun, 09 Sep 2001 01:46:40 GMT
        date = "Sun, 09 Sep 2001 01:47:10"
        with LocalServer((429, {"Retry-After": "12"}, b""),
                         (503, {"Retry-After": date + " GMT"}, b""),
                         (503, {"Retry-After": date + " -0000"}, b""),
                         (503, {"Retry-After": "soon"}, b"")) as server:
            client = Client(clock=lambda: now)
            results = [client.post(server.url, {}).retry_after for _ in range(4)]
        self.assertEqual(results, [12.0, 30.0, 30.0, None])

    def test_connection_refused_is_retryable(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        outcome = Client().post(f"http://127.0.0.1:{port}", {})
        self.assertEqual(outcome.kind, RETRYABLE)
        self.assertTrue(outcome.error)

    def test_malformed_response_is_retryable(self):
        with LocalServer(garbage) as server:
            self.assertEqual(Client().post(server.url, {}).kind, RETRYABLE)

    def test_timeout_bounds_the_whole_attempt(self):
        with LocalServer(drip) as server:
            start = time.monotonic()
            outcome = Client().post(server.url, {}, timeout=0.5)
            elapsed = time.monotonic() - start
        self.assertEqual(outcome.kind, RETRYABLE)
        self.assertLess(elapsed, 1.5)

    def test_unencodable_payload_raises(self):
        with self.assertRaises(TypeError):
            Client().post("http://127.0.0.1:9", object())
