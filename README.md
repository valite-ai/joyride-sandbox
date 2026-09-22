# webhook-relay

webhook-relay sends webhook events to partner endpoints. It requires Python 3.11 or newer and uses only the Python standard library.

## Run the tests

From the repository root, run:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -t .
```

## Delivery behavior

The relay delivers each event at least once. Partners must use the `Webhook-Id` header to discard duplicate events.

The relay retries HTTP 408, 425, 429, and 5xx responses. It also retries timeouts, connection errors, and DNS errors.

Exponential backoff increases each retry delay. Random jitter selects a shorter delay to spread retries across time.

The default policy allows five attempts within one hour and caps each delay at 60 seconds.

A valid `Retry-After` header sets the minimum delay, up to the 60-second cap. Other 3xx and 4xx responses stop delivery.

A delivery budget limits the time for one attempt. The default budget is 10 seconds.

The worker makes one attempt per turn, then moves to the next due delivery.

See [the design](docs/design.md) for details and [partner endpoints](docs/providers.md) for delivery configuration.
