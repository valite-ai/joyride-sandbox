# webhook-relay design

webhook-relay delivers webhook events to partner endpoints over HTTP. It uses only the Python standard library and needs Python 3.11 or newer. Delivery is at-least-once, so a partner can get the same event more than one time.

## Modules

The package uses a `src/webhook_relay/` layout with three modules:

- `retry.py` holds `RetryPolicy` and `DeliveryBudget`.
- `client.py` holds `Client` and the `Outcome` result type.
- `queue.py` holds `Delivery`, `DeliveryQueue`, and `DeadLetter`.

All time values are in seconds. Each class takes a `clock` callable (default `time.monotonic`), so tests can control time.

## RetryPolicy

`RetryPolicy` uses exponential backoff. Exponential backoff is a wait that grows by a fixed factor after each failure. The fields and defaults are `max_attempts=5`, `base_delay=1.0`, `multiplier=2.0`, `max_delay=60.0`, and `max_age=3600.0`.

`next_delay(attempt, age, retry_after=None)` returns the wait before the next attempt, or `None` to give up. The raw delay is `min(max_delay, base_delay * multiplier ** (attempt - 1))`. The policy applies full jitter, which is a random wait between zero and the raw delay. Jitter stops many failed deliveries from retrying at the same moment.

If the partner sent a `Retry-After` value, the delay is at least that value, capped at `max_delay`. The policy gives up in two cases. The first case is when `attempt` reaches `max_attempts`. The second case is when `age` plus the delay is more than `max_age`.

## Delivery budget

`DeliveryBudget` limits the time that the worker spends on one delivery in one turn. The default is `seconds=10.0`. The worker makes one attempt per turn and never sleeps between attempts. The request timeout is the budget, so one slow partner blocks the worker for 10 seconds at most.

If an attempt fails and the policy gives a delay, the queue puts the delivery back with a new due time. The worker then moves to the next due delivery. The budget limits one turn, and `max_age` limits the full life of a delivery.

## Client

`Client.post(url, payload, headers=None, timeout)` sends one request and returns an `Outcome`. It encodes `payload` with `json.dumps` and sends it with `urllib.request`. It sets `Content-Type: application/json`, `User-Agent: webhook-relay/<version>`, and `Webhook-Id: <delivery id>`. Partners can use `Webhook-Id` to find and drop duplicate events.

The client does not follow redirects, because a redirected POST can change method or lose the body. `Outcome` has a `kind`, the `status` if one exists, a short `error` text, and a parsed `retry_after`. The client sorts each response into one of three kinds:

| Response | Kind |
| --- | --- |
| 2xx | `DELIVERED` |
| 408, 425, 429, 5xx, timeout, connection or DNS error | `RETRYABLE` |
| Other 3xx and 4xx | `PERMANENT` |

The client reads at most 4 KiB of the response body, for the `error` text. It never raises for HTTP or network errors. It raises only for programming errors, for example a payload that `json.dumps` cannot encode.

## DeliveryQueue

A `Delivery` has an `id`, `url`, `payload`, `attempts`, `created_at`, `due_at`, and `last_outcome`. `DeliveryQueue` keeps due deliveries in a heap ordered by `due_at`. It takes a `Client`, a `RetryPolicy`, and a `DeliveryBudget`.

- `submit(url, payload)` creates a `Delivery` that is due now and returns its `id`.
- `run_once()` sends the first due delivery and returns its `Outcome`, or returns `None` if no delivery is due.
- `run_until_empty()` calls `run_once()` until no delivery is due.
- `dead_letters` returns the list of `DeadLetter` records.

A dead letter is a delivery that the queue stops trying to send. `DeadLetter` holds the `Delivery`, the last `Outcome`, and a `reason`. The queue dead-letters a delivery when the outcome is `PERMANENT`. It also dead-letters a delivery when `RetryPolicy.next_delay` returns `None`.

## Out of scope

This version keeps the queue in memory, so a process restart loses pending deliveries. It does not sign payloads and does not run more than one worker. Each of these items can come later without a change to the public classes above.
