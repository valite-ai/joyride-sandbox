# Partner endpoints

The relay sends webhook events to these partner endpoints.

| Name | Endpoint path | Auth header | Rate limit behavior |
| --- | --- | --- | --- |
| Acme Pay | `/webhooks/payments` | `Authorization: Bearer <token>` | Returns 429 with `Retry-After: 120` after a merchant burst. The relay waits at most 60 seconds because of its delay cap. |
| Beacon CRM | `/api/webhooks/events` | `X-API-Key: <key>` | Returns 503 briefly during deploys. The relay retries with exponential backoff. |
| Northstar Shipping | `/v1/hooks/deliveries` | `Authorization: Bearer <token>` | Returns 503 briefly during deploys. The relay retries with exponential backoff. |

Store all header values outside the repository. Send the correct header for each partner with every request.
