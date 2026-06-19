---
title: "Long-Running Operations: Async Jobs, Polling, and Callbacks"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How to design an API for work that cannot finish inside one request — the 202 Accepted plus operation-resource pattern, polling done right with Retry-After and conditional GETs, webhooks as the push alternative, idempotent kick-off, cancellation, and the Google AIP-151 LRO convention."
tags:
  [
    "api-design",
    "api",
    "rest",
    "http",
    "async",
    "long-running-operations",
    "polling",
    "webhooks",
    "idempotency",
    "rfc-9110",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/long-running-operations-async-jobs-polling-and-callbacks-1.png"
---

At 9:14 on a Monday, the finance team at our fictional commerce platform clicked "Run weekly payouts." Behind that button sat a single endpoint, `POST /payouts/batch`, that took a list of eight thousand sellers, computed each balance, called the bank rail once per seller, and returned a settlement report. In the demo, with twelve test sellers, it returned in under a second. In production, with eight thousand real sellers and a bank API that answered in 150–400 ms each, the handler needed somewhere between twenty and fifty minutes to finish.

The request never got that long to find out. At thirty seconds, the load balancer in front of the service gave up on the upstream, returned a `504 Gateway Timeout` to the caller, and closed the connection. The finance dashboard, which was a thin client that simply retried failed requests, saw the `504`, waited a moment, and fired `POST /payouts/batch` again. And again. By 9:21 the batch job had been kicked off four times. Three of those runs were still grinding through the seller list in the background — the timeout killed the *connection*, not the *work* — and two sellers ended up paid twice before anyone noticed the duplicate bank transfers. The post-mortem had a one-line root cause: **we tried to do slow work inside a request that was designed to be fast.**

This is the single most common mistake in API design once a system grows past toy scale. Some operations simply cannot finish inside the window an HTTP request gives you: generating a 40-page PDF report, transcoding a video, importing a million-row CSV, running a batch payout, training a model, reindexing a search corpus. The fix is not "raise the timeout" — that just moves the cliff and ties up a connection and a worker thread for the whole duration. The fix is a different contract: **accept the work, acknowledge it immediately, and give the caller a handle to track it.** This is the long-running operation (LRO) pattern, and it is the difference between an API that wedges under load and one that stays responsive while the slow work happens out of band.

![A timeline of a batch payout showing POST returning 202 with a Location header, two polls reporting running status, then a poll reporting succeeded and a final fetch of the result](/imgs/blogs/long-running-operations-async-jobs-polling-and-callbacks-1.png)

By the end of this post you will be able to design an async API end to end: decide when an operation must go async, return `202 Accepted` with a well-shaped **operation resource**, let clients **poll** it cheaply with `Retry-After` and conditional requests, offer **webhooks** as the push alternative, make the kick-off **idempotent** so a retry never starts a second job, support **cancellation**, hand back results with sane **expiry**, and follow the **Google AIP-151** convention so your operations look familiar to anyone who has used a big cloud API. We will build it all on the series' running Payments & Orders example — a `POST /payouts/batch` that runs async, is polled through an operation resource, and pushes a webhook on completion. As always, every decision lands on the same question this series keeps asking: *what does the caller get to assume, and can I change this later without breaking them?* If you want the foundation underneath this, the contract framing is in [what an API really is](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems), and everything here is summarized in [the API design playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).

## 1. Why you cannot just block

Let us be precise about *why* blocking fails, because the reasons are not arbitrary — they fall out of how HTTP and the network path between a client and your handler actually work. A synchronous request holds a chain of scarce, time-bounded resources open for its entire duration, and slow work strains every link in that chain.

**Request timeouts exist at every hop, and the shortest one wins.** A request from a browser or SDK to your handler usually passes through several intermediaries, and each imposes its own deadline. A typical chain is: client socket timeout → CDN / edge → load balancer or API gateway → application server → your handler. AWS Application Load Balancer defaults its idle timeout to 60 seconds. Many API gateways and reverse proxies default to 30 or 60 seconds. AWS API Gateway's integration timeout is capped at 29 seconds and cannot be raised. Heroku famously terminates any request that produces no bytes within 30 seconds. The caller's own HTTP library often defaults to a 30- or 60-second read timeout. **The effective ceiling on a synchronous request is the minimum of all these, and you do not control most of them.** A handler that needs twenty minutes is fighting a budget of thirty seconds it can never win.

**Each in-flight request pins a connection and usually a worker.** While your handler blocks on the bank API for twenty minutes, it holds a TCP connection open to the client, and in most server models it occupies a worker thread or process or async task for the whole time. A server with, say, 200 worker slots that hands each slot to a twenty-minute job can serve at most 200 of these per twenty minutes — roughly **one new job every six seconds** before it saturates and starts refusing or queuing everything, including the fast requests. Slow synchronous work does not just risk one timeout; it consumes the concurrency budget the rest of your API needs to stay alive. The fast `GET /orders/{id}` that should take 8 ms now waits behind a wall of stuck payout handlers.

**A timeout kills the connection, not the work — so retries re-trigger the job.** This is the subtle, dangerous one, and it is exactly what bit the payout batch in the opening story. When the gateway returns `504`, your handler is often still running; the database transaction is still open, the bank calls are still going out. The client, seeing a `504` (or worse, a closed socket with no response at all), cannot tell whether the work succeeded, failed, or is still going. A naive client retries. Now you have two — or four — copies of an expensive, *side-effecting* job running concurrently. For a report that is merely wasteful. For a payout it is a double-charge and a real-money incident. Idempotency keys (which we cover in [safe retries and the exactly-once illusion](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions)) can save you here, but the cleaner fix is to never let the request block long enough to time out in the first place.

**Gateways and proxies cap response time deliberately, and you should let them.** It is tempting to view the 30-second cap as the enemy. It is not. That cap protects your infrastructure from slow-loris attacks, from leaked connections, from a single bad upstream consuming all the front-end's sockets. Raising it everywhere to "fix" long operations weakens a defense that exists for good reasons. The right move is to design the long operation so it *fits comfortably inside* the synchronous budget — by making the synchronous part just the acknowledgement, and pushing the work elsewhere.

There is a rough rule of thumb worth internalizing. **If an operation's p99 latency approaches even half your tightest gateway timeout, treat it as a candidate for async.** A handler whose worst case is 14 seconds against a 30-second gateway is one slow dependency away from a timeout storm. The async pattern removes that entire failure mode.

It helps to make the concurrency arithmetic concrete, because it is the part teams underestimate until an incident teaches them. Suppose your service runs on 8 instances, each with a pool of 64 worker slots — 512 slots total. A fast endpoint at 10 ms per request can, in principle, serve $512 / 0.010 = 51{,}200$ requests per second from those slots (network and CPU permitting). Now route a single slow synchronous endpoint through the same pool, each call holding a slot for 20 minutes (1,200 seconds). Each slot can finish $1 / 1200$ of these per second, so 512 slots clear at most $512 / 1200 \approx 0.43$ jobs per second — roughly **one every 2.3 seconds before the entire pool is occupied**. The moment a 26th seller in a minute clicks "run payouts," every slot is pinned, and your fast 10 ms endpoints — the ones that should never have been affected — start queuing behind stuck payout handlers and timing out themselves. One slow endpoint took down the fast ones. This is *head-of-line blocking* at the worker-pool level, and it is why "slow work in a request" is not a local problem you can contain to one endpoint; it is a shared-resource problem that degrades everything sharing the pool. Moving the work to a background queue takes those 20-minute holds off the request pool entirely: the kick-off handler holds a slot for a few milliseconds, the worker pool that drains the queue is sized and scaled independently, and the fast endpoints keep their slots.

There is a second, quieter cost to blocking that the latency math hides: **you have given the client nothing to hold onto.** A synchronous twenty-minute request that finally fails at second 28 leaves the caller with a `504` and zero information — no job id, no partial progress, no way to ask "did any of it happen?" The caller's only options are to retry the whole thing (dangerous) or give up (lossy). The async pattern's deepest benefit is not just avoiding the timeout; it is that from the very first millisecond the client holds a *durable handle* to the work — an operation id it can poll, share, log, and reconcile against. The work becomes a first-class, observable thing in your system rather than an opaque connection that either returns or does not.

## 2. The async pattern: accept now, finish later

The core move is simple to state and surprisingly subtle to get right: **separate the act of accepting work from the act of doing it.** The request that submits the work returns almost instantly with a receipt; the work runs in the background; the client uses the receipt to check on it.

![A before-and-after contrast showing a blocking request that hits a gateway timeout and a retry that runs the job twice, versus an accepting request that returns 202 with a Location header and is polled safely](/imgs/blogs/long-running-operations-async-jobs-polling-and-callbacks-2.png)

In HTTP terms, the receipt is the `202 Accepted` status code, and RFC 9110 defines its meaning with unusual precision. From §15.3.3: *"The 202 (Accepted) status code indicates that the request has been accepted for processing, but the processing has not been completed. The request might or might not eventually be acted upon, as it might be disallowed when processing actually takes place."* That last sentence is the contract you are signing: `202` promises **acknowledgement, not success.** The client must not assume the work is done — only that you have taken responsibility for it. The same section notes that the representation returned with a `202` *"should describe the request's current status and point to (or embed) a status monitor that can provide the user with an estimate of when the request will be fulfilled."* That status monitor is the **operation resource**, and building it well is most of this post.

Here is the kick-off on the wire. The client submits a batch payout; the server validates the request shape (cheaply, synchronously), records the job, and returns immediately:

```http
POST /v1/payouts/batch HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Content-Type: application/json
Idempotency-Key: 8f14e45f-ceea-467d-9b3a-2c1b9a7e0d11

{
  "currency": "USD",
  "source_account": "acct_main_settlement",
  "payouts": [
    { "seller_id": "sel_001", "amount_cents": 124900 },
    { "seller_id": "sel_002", "amount_cents":  38050 }
  ]
}
```

```http
HTTP/1.1 202 Accepted
Content-Type: application/json
Location: https://api.example.com/v1/operations/op_7f3a9c2e
Operation-Id: op_7f3a9c2e
Retry-After: 5

{
  "id": "op_7f3a9c2e",
  "resource_type": "operation",
  "operation_type": "payouts.batch",
  "status": "pending",
  "progress": { "percent": 0, "processed": 0, "total": 8000 },
  "result": null,
  "error": null,
  "created_at": "2026-06-20T09:14:02Z",
  "updated_at": "2026-06-20T09:14:02Z",
  "expires_at": "2026-06-27T09:14:02Z",
  "self": "https://api.example.com/v1/operations/op_7f3a9c2e"
}
```

Notice the headers carrying the contract. The **`Location` header** points at the operation resource the client should track — this is the canonical receipt, and a well-behaved client stores this URL and forgets about the original request body. The **`Retry-After: 5`** header tells the client not to poll for at least five seconds; we will lean on it heavily in the polling section. The body is a full representation of the operation in its initial `pending` state, so a client that wants to render a progress bar immediately has everything it needs without a second request.

### Where does the work actually run?

`202` is the *contract*; it says nothing about *how* you run the work. In practice the handler does three cheap things and returns: validate the request, persist an operation row (status `pending`), and enqueue a message for a background worker. The actual payout loop runs in a separate worker process pulling from a queue. This is exactly the boundary where the API-design layer hands off to messaging infrastructure — how that queue guarantees the message is not lost, how it handles retries and dead letters, and whether you want a queue, a pub/sub topic, or a log is a deep topic in its own right; see [queue vs pub/sub vs log](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models). For this post the key contract is only this: **the synchronous handler's job ends the moment the work is durably enqueued and the operation row is written.** Everything after that is the worker's problem, observed by the client through the operation resource.

A minimal kick-off handler makes the boundary concrete:

```python
@app.post("/v1/payouts/batch")
def create_batch_payout(req):
    body = validate_batch_payout(req.json)        # synchronous, cheap; 422 on bad shape
    idem_key = req.headers.get("Idempotency-Key")

    existing = operations.find_by_idempotency_key(idem_key)
    if existing:                                  # a retried kick-off: do NOT start a second job
        return respond_202(existing)

    op = operations.create(
        operation_type="payouts.batch",
        status="pending",
        progress={"percent": 0, "processed": 0, "total": len(body["payouts"])},
        idempotency_key=idem_key,
        expires_at=now() + timedelta(days=7),
    )
    queue.enqueue("run_batch_payout", operation_id=op.id, payload=body)  # durable handoff
    return respond_202(op, location=f"/v1/operations/{op.id}", retry_after=5)
```

The handler returns in single-digit milliseconds because it does no payout work — it only records intent and hands off. The synchronous budget is now spent on validation and a database write, both of which finish in well under a second, comfortably inside any gateway timeout.

### Choosing the operation's URL: dedicated `/operations` vs the target resource

You have two reasonable shapes for the receipt, and the choice has consequences worth thinking through.

The first is a **dedicated operations namespace** — `POST /payouts/batch` returns a `Location` of `/operations/{id}`, a generic resource that can represent *any* async job in your API. This is the Google Cloud convention (AIP-151), and its virtue is uniformity: clients learn one polling shape and reuse it for every long operation across the whole API, whether it is a payout, an export, or a reindex. The cost is one extra hop — the operation is a *handle*, and on success the client follows a link to the actual result.

The second is to **return the target resource itself in a not-yet-ready state**. `POST /reports` returns `202` with `Location: /reports/{id}` and a body whose `status` is `pending`; the client polls that same `/reports/{id}` URL, and when `status` becomes `succeeded` the resource simply *is* the report. This is more RESTful in the sense that there is one resource, not two, and no extra hop. The cost is that the report resource now has to model the "not done yet" states (`pending`, `failed`) in addition to its real shape, and clients must always check `status` before trusting the body.

| Dimension | Dedicated `/operations/{id}` | Target resource `/reports/{id}` |
| --- | --- | --- |
| Uniformity across jobs | One shape for every async op | Each resource invents its own status field |
| Extra hop to the result | Yes — follow `result` link | No — the resource becomes the result |
| Where status lives | On a generic operation | On the domain resource itself |
| Best when | Many heterogeneous async jobs (the Google AIP-151 model) | One resource type with a clear "ready" state (a report, an export file) |
| Risk | Two resources to reason about | Clients trust a half-built resource if they skip the status check |

A pragmatic rule: if you have many *kinds* of long jobs, use a uniform `/operations` namespace and save your clients from learning N polling protocols. If you have one resource that is occasionally slow to materialize, returning that resource in a `pending` state is cleaner. The Payments API in this post uses `/operations` because a real payments platform has batch payouts, bulk imports, settlement exports, and reconciliation runs — all async, all better served by one consistent handle.

## 3. The operation resource: what it must carry

The operation resource is the heart of the contract. It is the one thing the client polls, renders, and reasons about, so its shape deserves the same care you would give any first-class resource. Get it right and a client can build a progress bar, retry safely, surface a real error, and fetch the result with zero guesswork.

![A vertical stack of the operation resource fields showing id and self link, status, progress, result link, error envelope, and timestamps](/imgs/blogs/long-running-operations-async-jobs-polling-and-callbacks-3.png)

Here is the full shape, with each field earning its place:

```json
{
  "id": "op_7f3a9c2e",
  "resource_type": "operation",
  "operation_type": "payouts.batch",
  "status": "running",
  "progress": {
    "percent": 62,
    "processed": 4960,
    "total": 8000,
    "message": "settling seller 4960 of 8000"
  },
  "result": null,
  "error": null,
  "created_at": "2026-06-20T09:14:02Z",
  "updated_at": "2026-06-20T09:18:41Z",
  "expires_at": "2026-06-27T09:14:02Z",
  "self": "https://api.example.com/v1/operations/op_7f3a9c2e"
}
```

**`status`** is the field everything else hangs off. It is a small, closed enum, and getting the state set right is worth its own section (next). The canonical set is `pending` → `running` → one of `succeeded` / `failed` / `cancelled`. Keep it closed: a client must be able to write an exhaustive `switch` over the statuses, and an unknown value should be treated as "still in a non-terminal state, keep polling" by the tolerant-reader principle this series leans on (covered in [backward and forward compatibility](/blog/software-development/api-design/backward-and-forward-compatibility-the-rules-of-safe-change)).

**`progress`** is an estimate, and you must label it as such. A `percent` plus a `processed` / `total` pair lets a client render a real progress bar; a `message` gives a human something to read. Be honest about precision: if you cannot compute a meaningful percentage (an unbounded stream, an import whose total is unknown until parsed), omit `percent` and report `processed` alone, or use an indeterminate flag. Never fabricate a percentage that lurches from 10% to 100%; clients build trust on monotonic, believable progress.

**`result`** is null until the operation succeeds, then it carries the answer — and you have a choice between *embedding* and *linking*. Embed when the result is small and bounded (a summary object, a count, an ID). Link when it is large or a different media type (a generated PDF, a CSV export, a multi-megabyte report). For the batch payout, a link is right: the result is a settlement report that lives at its own URL with its own expiry.

```json
{
  "status": "succeeded",
  "progress": { "percent": 100, "processed": 8000, "total": 8000 },
  "result": {
    "type": "payouts.batch.report",
    "href": "https://api.example.com/v1/payouts/batch/pob_91c4/report",
    "summary": { "succeeded": 7991, "failed": 9, "total_cents": 982_400_550 }
  },
  "error": null
}
```

**`error`** is null unless the operation fails, in which case it carries a `problem+json` envelope (RFC 9457 — the error contract this series uses everywhere; see [error design](/blog/software-development/api-design/error-design-a-machine-readable-human-friendly-contract)). Crucially, **the operation itself succeeds even when the work fails.** The `GET /operations/{id}` returns `200 OK` — the *poll* worked — and the failure is reported *inside* the body via `status: "failed"` and a populated `error`. Do not return a 4xx/5xx from the poll just because the underlying job failed; that conflates "I could not tell you the status" with "the job failed," and it breaks polling clients that key off the HTTP status to decide whether to keep trying.

```json
{
  "status": "failed",
  "progress": { "percent": 41, "processed": 3280, "total": 8000 },
  "result": null,
  "error": {
    "type": "https://api.example.com/errors/insufficient-funds",
    "title": "Source account has insufficient funds",
    "status": 422,
    "detail": "Settlement account acct_main_settlement balance 410,000 cents is below required 982,400,550 cents.",
    "instance": "/operations/op_7f3a9c2e"
  }
}
```

**The timestamps** — `created_at`, `updated_at`, `expires_at` — let clients reason about staleness and lifetime. `updated_at` advances each time progress changes (useful for detecting a stuck job). `expires_at` is the contract for how long the operation and its result will remain retrievable, which we return to under result expiry.

**`self`** (and `result.href`) are HATEOAS links — the operation tells the client where to find itself and its result, so the client never hand-builds URLs. This is hypermedia earning its keep: the result lives wherever the server decides, and the client just follows the link. (When links pay off and when they are over-engineering is its own debate; see [HATEOAS in the real world](/blog/software-development/api-design/hateoas-in-the-real-world-hypermedia-links-and-when-to-skip).)

#### Worked example: a 202 plus a full polling loop to completion

Let us walk the entire happy path on the wire, because the choreography is the contract. A client kicks off the batch payout and follows it to the settlement report.

Step 1 — kick off. The client sends the `POST` (shown in §2) and receives `202` with `Location: /v1/operations/op_7f3a9c2e` and `Retry-After: 5`. The client stores that URL.

Step 2 — first poll, after waiting the five seconds the server requested:

```http
GET /v1/operations/op_7f3a9c2e HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
```

```http
HTTP/1.1 200 OK
Content-Type: application/json
ETag: "op_7f3a9c2e-rev-3"
Retry-After: 5
Cache-Control: no-cache

{
  "id": "op_7f3a9c2e",
  "status": "running",
  "progress": { "percent": 40, "processed": 3200, "total": 8000 },
  "result": null, "error": null,
  "updated_at": "2026-06-20T09:16:10Z",
  "self": "https://api.example.com/v1/operations/op_7f3a9c2e"
}
```

Step 3 — the client waits another five seconds (honoring `Retry-After`) and polls again, this time sending the `ETag` it received so an unchanged operation costs almost nothing:

```http
GET /v1/operations/op_7f3a9c2e HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
If-None-Match: "op_7f3a9c2e-rev-3"
```

```http
HTTP/1.1 304 Not Modified
ETag: "op_7f3a9c2e-rev-3"
Retry-After: 5
```

No body, no progress change since the last poll — a cheap `304`. The client waits and polls again.

Step 4 — the operation completes. A later poll returns `200` with `status: "succeeded"` and a `result.href`. The client stops polling and follows the link:

```http
GET /v1/payouts/batch/pob_91c4/report HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{ "succeeded": 7991, "failed": 9, "rows": [ /* … */ ] }
```

The whole dance — one `202`, a handful of paced polls (most of them `304`), and a single result fetch — replaces the broken twenty-minute blocking request, and not one of those calls is at risk of a gateway timeout. That is the contract you are buying.

## 4. The operation state machine

A client polls because it wants to know one thing: *is it done yet, and how did it end?* Your status enum is the answer, and a clean, closed state machine is what makes polling safe to reason about. The cardinal rule: **every operation reaches exactly one terminal state, and once terminal it never changes again.**

![A directed graph of the operation state machine showing pending transitioning to running, and running branching to succeeded, failed, or cancelled](/imgs/blogs/long-running-operations-async-jobs-polling-and-callbacks-4.png)

The states and their transitions:

- **`pending`** — accepted and enqueued, but no worker has picked it up. Progress is 0. This is the state the `202` returns.
- **`running`** — a worker is actively processing. Progress advances. `updated_at` moves.
- **`succeeded`** — terminal. `result` is populated, `error` is null, progress is 100.
- **`failed`** — terminal. `error` is populated with a `problem+json` body, `result` is null, progress is wherever it stopped.
- **`cancelled`** — terminal. The client requested cancellation (see §7) and the worker honored it.

| Status | Terminal? | `result` | `error` | Client should |
| --- | --- | --- | --- | --- |
| `pending` | no | null | null | keep polling (honor `Retry-After`) |
| `running` | no | null | null | keep polling, render progress |
| `succeeded` | yes | set | null | stop polling, fetch/read result |
| `failed` | yes | null | set | stop polling, surface error |
| `cancelled` | yes | null | maybe | stop polling, treat as aborted |

The "terminal means terminal" rule has real teeth. A client that sees `succeeded` is entitled to stop polling forever, cache the result, and never call you again about this operation. If you ever let a `succeeded` operation flip back to `running` (say, because a retry re-queued the work), you have broken every client that trusted the terminal state. The state machine must be a **DAG with a single absorbing layer**: `pending` → `running` → one terminal state, no loops back. If a job needs to retry internally, that retry happens *inside* `running`; it does not surface as a state regression to the client.

A subtle design choice: do you expose `pending` as distinct from `running`, or collapse them? Exposing both lets a client distinguish "queued behind other work" from "actively processing," which is useful for surfacing accurate UI ("Your export is #3 in the queue"). Collapsing them into a single non-terminal state is simpler and is what Google's AIP-151 effectively does with its boolean `done` field — an operation is either `done: false` (running) or `done: true` (terminal). Both are defensible; the richer enum costs you nothing if clients are told to treat any non-terminal status as "keep polling."

This is also where the tolerant-reader principle pays off. **Tell your clients in the docs: treat any status you do not recognize as non-terminal and keep polling.** Then, if you ever add a state — say `paused` for an operation a human suspended — old clients degrade gracefully (they keep polling, eventually see a terminal state) instead of crashing on an unknown enum value. Adding a non-terminal status becomes a non-breaking change. This is the same reasoning that makes adding an optional response field safe, applied to an enum.

## 5. Polling done right

Polling is the default way clients track an operation, and done naively it is wasteful and rude — a client hammering `GET /operations/{id}` every 200 ms turns a single job into thousands of pointless requests, burns your rate limit, and loads your database for nothing. Done right, polling is cheap, paced, and almost free for the long stretches when nothing has changed.

![A before-and-after contrast of tight polling that ignores Retry-After and burns quota until throttled, versus paced conditional polling that honors Retry-After and returns mostly 304 responses](/imgs/blogs/long-running-operations-async-jobs-polling-and-callbacks-5.png)

There are three levers, and a good client and a good server cooperate on all three.

**Lever 1: `Retry-After` to pace the client.** Every poll response — the `202`, every `200`, every `304` — should carry a `Retry-After` header telling the client the minimum seconds to wait before polling again. The *server* sets the pace because the server knows how fast the work moves: a payout that processes ~3 sellers per second has nothing new to say for several seconds, so `Retry-After: 5` is honest. As the operation nears completion the server can shorten it (`Retry-After: 1`) so the client catches the terminal state promptly. RFC 9110 §10.2.3 defines `Retry-After` for exactly this — it accepts either a number of seconds or an HTTP date. A client that honors `Retry-After` is a well-behaved client; a client that ignores it and polls in a tight loop deserves the `429` it will eventually get.

**Lever 2: exponential backoff for resilience.** `Retry-After` paces the *normal* case. When polls start failing — the operation service returns `503`, or the network blips — the client should back off exponentially with jitter rather than retry instantly. A common schedule: start at the server's `Retry-After`, and on errors double the wait up to a cap (say 60 s), with random jitter to avoid a thundering herd of clients all retrying in lockstep. The combination — honor `Retry-After` when the server is healthy, back off exponentially when it is not — keeps polling cheap and keeps your service from being hammered during an incident. (Backoff math and jitter strategies for the broker side are covered in [retries and exponential backoff](/blog/software-development/message-queue/dead-letter-queues-retries-exponential-backoff).)

**Lever 3: `ETag` and `304` to make unchanged polls nearly free.** This is the big win. Most polls of a long operation return *the same thing as the last poll* — the job is still running, progress has not moved much. There is no reason to send the full body each time. Attach an `ETag` to the operation that changes only when the operation meaningfully changes (a revision counter bumped on each status or progress update works well). The client stores the `ETag` and sends it back as `If-None-Match` on the next poll. If nothing changed, the server replies `304 Not Modified` with no body — a conditional request, the same mechanism HTTP caching uses (covered in depth in [caching, ETags, and conditional requests](/blog/software-development/api-design/caching-etags-cache-control-conditional-requests-invalidation)).

```http
GET /v1/operations/op_7f3a9c2e HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
If-None-Match: "op_7f3a9c2e-rev-7"
```

```http
HTTP/1.1 304 Not Modified
ETag: "op_7f3a9c2e-rev-7"
Retry-After: 5
```

The arithmetic is worth seeing. Suppose a payout runs for twenty minutes and a client polls every five seconds — that is 240 polls. With full bodies at ~400 bytes each that is ~96 KB of response payload, plus 240 database reads and 240 JSON serializations. With conditional requests, perhaps a dozen of those polls return a changed body (each time progress crosses a threshold worth reporting) and the other ~228 return an empty `304`. You have cut the response bytes by roughly 95% and turned most of the polls into a cheap revision-number comparison. Across thousands of concurrent operations that is the difference between a polling fleet that is a rounding error on your infrastructure and one that needs its own capacity plan.

A reference polling client that combines all three levers:

```python
def await_operation(client, op_url, max_wait=1800):
    etag = None
    deadline = time.monotonic() + max_wait
    backoff = 1.0
    while time.monotonic() < deadline:
        headers = {"If-None-Match": etag} if etag else {}
        resp = client.get(op_url, headers=headers)

        if resp.status_code == 304:
            backoff = 1.0                          # healthy: reset backoff
        elif resp.status_code == 200:
            etag = resp.headers.get("ETag")
            op = resp.json()
            if op["status"] == "succeeded":
                return op["result"]
            if op["status"] in ("failed", "cancelled"):
                raise OperationError(op["status"], op.get("error"))
            backoff = 1.0
        elif resp.status_code in (429, 503, 502):  # transient: back off
            backoff = min(backoff * 2, 60)
        else:
            resp.raise_for_status()

        wait = float(resp.headers.get("Retry-After", backoff))
        time.sleep(wait + random.uniform(0, 0.5 * wait))   # jitter
    raise TimeoutError("operation did not complete within max_wait")
```

The client honors `Retry-After`, sends `If-None-Match`, treats `304` and a `running` `200` identically (keep waiting), backs off on transient errors, and has an overall deadline so it never polls forever. This is what "polling done right" looks like in code.

One more rule: **do not put the operation poll behind aggressive caching at a CDN or shared proxy.** Send `Cache-Control: no-cache` (revalidate every time) or `private` so a shared cache does not serve one client's stale operation state to another. The `ETag`/`304` mechanism gives you the efficiency; a shared cache would give you correctness bugs.

## 6. Webhooks: the push alternative

Polling has an irreducible cost: latency between completion and the client noticing, plus the steady drip of poll requests. For a twenty-minute payout polled every five seconds, the client learns of completion within five seconds of it happening — usually fine. But for operations that finish at unpredictable times, or where you want to wake a client that is not actively waiting (a server-to-server integration, a partner's backend), **push beats pull.** Instead of the client asking "are you done yet?" repeatedly, the server tells the client "I'm done" exactly once, when it happens. That is a webhook.

![A directed graph of a completion webhook showing the worker finishing and signing an event, delivering it with retries, the client verifying the signature or rejecting a bad one, and fetching the result on success](/imgs/blogs/long-running-operations-async-jobs-polling-and-callbacks-6.png)

When the batch payout reaches a terminal state, the server posts a completion event to a URL the client registered in advance:

```http
POST /webhooks/payments HTTP/1.1
Host: client.example.com
Content-Type: application/json
Webhook-Id: evt_5d2a1f
Webhook-Timestamp: 1781945921
Webhook-Signature: v1,k3y8mF2pQ9rN7tX1wL4aZ6cV0bH5jD8sR3eU2iO1nM=

{
  "type": "operation.succeeded",
  "operation_id": "op_7f3a9c2e",
  "operation_type": "payouts.batch",
  "occurred_at": "2026-06-20T09:34:11Z",
  "data": {
    "result": {
      "href": "https://api.example.com/v1/payouts/batch/pob_91c4/report",
      "summary": { "succeeded": 7991, "failed": 9 }
    }
  }
}
```

Three properties make this safe, and each is a contract the webhook design must honor:

**It must be signed.** The client's webhook endpoint is public — anyone can `POST` to it. So the server signs the payload (typically an HMAC over the timestamp and raw body, using a shared secret) and puts the signature in a header. The client recomputes the HMAC and rejects any request whose signature does not match, returning `401`. The timestamp guards against replay — reject anything older than a few minutes. Without signing, your "payout succeeded" webhook is a forgery waiting to happen.

**It must retry with backoff, and deliveries are at-least-once.** Networks fail; the client's endpoint might be briefly down. The server retries delivery on any non-2xx response (or timeout) with exponential backoff over a window of hours, often with a dead-letter destination and a replay UI for ones that never land. The consequence the client must design for: **a webhook may arrive more than once, and out of order.** Two `operation.succeeded` events for the same operation, or a `succeeded` arriving before a stale `running` event you also sent — the client must dedupe on `operation_id` + `type` and treat the delivery as idempotent. This is the same at-least-once reality that governs all event delivery; the guarantees and ordering trade-offs are detailed in [delivery semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once).

**It should be thin — notify, then let the client fetch.** A robust pattern is the "thin webhook": the event says *what happened and which operation*, and carries a *link* to the result rather than the full result inline. The client receives the event, verifies the signature, and then does an authenticated `GET` on the result link to pull the actual data. This avoids stuffing megabytes of report into the webhook body, sidesteps the "is the webhook payload itself trustworthy?" question (the client fetches from your authenticated API, not from the webhook body), and means a missed or duplicated webhook is harmless because the source of truth is always the operation resource. The deep design of webhooks — signing schemes, retry windows, replay, AsyncAPI documentation — is its own post; see [event-driven and async APIs](/blog/software-development/api-design/event-driven-and-async-apis-webhooks-pubsub-and-asyncapi).

### Polling or webhooks? Usually both.

Step back and the whole design space for "how does the client learn the work is done" has exactly three points, and it is worth seeing them side by side because most arguments about async APIs are really arguments about which of these three you picked.

![A matrix comparing blocking, polling, and webhook delivery models across completion latency, client complexity, and reliability](/imgs/blogs/long-running-operations-async-jobs-polling-and-callbacks-8.png)

**Blocking** is the model we ruled out in §1: zero extra complexity for the client — one request, one response — but it fails outright past the gateway timeout, and the timeout-then-retry path silently doubles the work. It is the simplest to *call* and the most fragile to *operate*, which is exactly the wrong trade for long jobs. **Polling** moves the cost to the client (a loop, backoff, an overall deadline) but is rock-solid: the client always reads current truth, works behind any firewall, and never depends on the server reaching it. **Webhooks** invert the direction: near-instant notification and zero steady-state cost, at the price of a public endpoint, signature verification, and at-least-once delivery the client must dedupe. No row in that matrix is strictly best; the right choice depends on whether the client is interactive or server-to-server, whether it has a reachable endpoint, and how much completion latency it can tolerate.

This is not an either/or. The mature pattern is to **offer both and let the client choose**, because they fail in opposite directions and complement each other.

| Aspect | Polling | Webhooks |
| --- | --- | --- |
| Direction | Client pulls | Server pushes |
| Completion latency | Up to one poll interval | Near-instant |
| Steady-state cost | A drip of poll requests | Zero until completion |
| Client needs | Nothing — just makes requests | A public, reachable HTTPS endpoint |
| Behind a firewall / no public URL | Works fine | Cannot receive the push |
| Failure mode | Wasteful if not paced | Missed delivery if endpoint is down |
| Ordering / duplicates | Always reads current truth | At-least-once; may dupe / reorder |
| Best for | Interactive clients, scripts, no public endpoint | Server-to-server, long unpredictable jobs |

The decisive guidance: **webhooks are the optimization, the operation resource is the source of truth.** A client should be able to function with polling alone — that path always works, needs no public endpoint, and always reads current truth. Webhooks make the common case faster and cheaper but must never be the *only* way to learn an outcome, because a dropped webhook would otherwise strand the client forever. Stripe's model embodies this exactly: you can poll the resource, and you can subscribe to events — and Stripe's own docs tell you to reconcile webhooks against the API because delivery is best-effort. Design so that a client that ignores webhooks entirely still works correctly, just with more latency.

## 7. Idempotent kick-off: never start two jobs

We opened with the double-payout. Here is the fix, and it is the same fix this series uses for any unsafe retry: an **idempotency key.** The problem is specific to the async kick-off but no less dangerous than the synchronous case — arguably more, because the operations are expensive and side-effecting by nature.

![A directed graph of an idempotent kick-off showing a POST with an Idempotency-Key, a lookup that either creates a new operation, replays the cached one, or returns a 409 conflict when the key is reused with a different body](/imgs/blogs/long-running-operations-async-jobs-polling-and-callbacks-7.png)

The scenario: a client sends `POST /payouts/batch`, but the response is lost — a `502` from a flaky proxy, a dropped connection, a client-side timeout. The client cannot tell whether the operation was created. If it retries blindly, and the original `POST` *did* create and enqueue the operation, you now have two batch payouts running. For a payout, that is real money paid twice.

The fix: the client generates a unique `Idempotency-Key` (a UUID) and sends it as a header on the `POST`. The server treats the key as a fingerprint of the request:

- **First time seeing the key:** create the operation, enqueue the work, store the key → operation mapping, return `202`.
- **Key seen before, same request:** do *not* create a second operation. Return `202` with the *existing* operation — the same `Location`, the same `id`. The retry is a no-op that just re-hands the client the receipt it lost.
- **Key seen before, different request body:** the client is misusing the key (reusing it for a genuinely different request). Return `409 Conflict` — refuse to overload one key onto two meanings.

```python
@app.post("/v1/payouts/batch")
def create_batch_payout(req):
    idem_key = req.headers.get("Idempotency-Key")
    if not idem_key:
        raise problem(428, "Precondition Required",
                      "Idempotency-Key header is required for batch payouts.")
    body = validate_batch_payout(req.json)
    fingerprint = sha256(canonical_json(body))

    record = idempotency_store.get(idem_key)
    if record:
        if record.fingerprint != fingerprint:
            raise problem(409, "Idempotency-Key reused",
                          "This key was used with a different request body.")
        return respond_202(operations.get(record.operation_id))   # replay the receipt

    op = operations.create(operation_type="payouts.batch", status="pending", ...)
    idempotency_store.put(idem_key, fingerprint=fingerprint,
                          operation_id=op.id, ttl=timedelta(hours=24))
    queue.enqueue("run_batch_payout", operation_id=op.id, payload=body)
    return respond_202(op, location=f"/v1/operations/{op.id}", retry_after=5)
```

There is a race to close: two retries arriving simultaneously could both pass the `idempotency_store.get` check before either writes. Guard the create with a unique constraint on the idempotency key in the database (or an atomic "insert if not exists") so the second writer loses cleanly and reads back the first writer's operation. The full treatment of races, key TTLs, and storing the cached *response* (not just the mapping) is in [idempotency keys and safe retries](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions); here the point is narrow and vital: **the async kick-off is exactly the kind of expensive, side-effecting `POST` that must be idempotent, because the whole reason it is async is that it does a lot of irreversible work.**

#### Worked example: the same batch payout kicked off twice, safely

A client fires the batch payout. The `POST` reaches the server, which creates `op_7f3a9c2e`, enqueues the work, and starts streaming back a `202` — but a proxy between them dies mid-response and the client gets a connection reset. The client never saw the `Location`.

The client retries with **the same `Idempotency-Key`** and **the same body**:

```http
POST /v1/payouts/batch HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
Idempotency-Key: 8f14e45f-ceea-467d-9b3a-2c1b9a7e0d11
Content-Type: application/json

{ "currency": "USD", "source_account": "acct_main_settlement", "payouts": [ /* same 8000 */ ] }
```

The server looks up the key, finds it (created moments ago), confirms the body fingerprint matches, and returns the **existing** operation — not a new one:

```http
HTTP/1.1 202 Accepted
Location: https://api.example.com/v1/operations/op_7f3a9c2e
Idempotency-Replayed: true

{ "id": "op_7f3a9c2e", "status": "running", "progress": { "percent": 8, "processed": 640, "total": 8000 }, ... }
```

The optional `Idempotency-Replayed: true` header tells the client this was a replay, not a fresh start. The batch ran exactly once; the eight thousand sellers were each paid one time; the retry simply recovered the receipt the network ate. Compare to the opening incident, where four kick-offs ran three concurrent batches and double-paid two sellers — the difference is one header and a uniqueness constraint.

Note the `428 Precondition Required` in the handler: for a money-moving operation, you can *require* the `Idempotency-Key` rather than merely accept it. RFC 9110's `428` exists precisely to say "you must send a precondition" — here, "you must send an idempotency key before I will move money for you." That refusal protects clients who would otherwise forget.

## 8. Cancellation: a job you can stop

Long operations sometimes need to be stopped — the user changed their mind, the wrong file was uploaded, a payout was triggered against the wrong account and someone caught it ten seconds in. A good async API lets a client request cancellation, and the operation resource models the outcome.

There are two reasonable shapes. The first is a **`DELETE` on the operation**:

```http
DELETE /v1/operations/op_7f3a9c2e HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
```

```http
HTTP/1.1 202 Accepted

{ "id": "op_7f3a9c2e", "status": "running", "progress": { "percent": 23 }, ... }
```

Notice that even cancellation is itself asynchronous — `DELETE` returns `202`, not `204`, because *requesting* a stop is instant but *honoring* it is not. The worker is mid-flight; it must reach a safe checkpoint, stop pulling new work, and transition the operation to `cancelled`. The client keeps polling and watches `status` move from `running` to `cancelled`. The second shape is an explicit **cancel action** — `POST /v1/operations/op_7f3a9c2e:cancel` (the AIP-151 custom-method style) — which reads more clearly as "cancel this" than `DELETE` and leaves `DELETE` free to mean "delete the operation record." Either is fine; pick one and be consistent.

The hard part is not the API; it is the semantics. **Cancellation of a side-effecting operation is rarely a clean undo.** If the batch payout has already paid 4,960 of 8,000 sellers when the cancel lands, you cannot un-pay them — the money has moved. So you must define what `cancelled` *means* for each operation:

- **Stop-and-keep:** the worker stops issuing new payouts but the 4,960 already sent stay sent. The `cancelled` operation's `result` reports the partial work done. This is honest for irreversible work.
- **Stop-and-compensate:** the worker stops and *reverses* completed work (issues refunds for the 4,960 payouts). Only possible if the work is reversible, and itself a long operation — you may end up with a cancellation that spawns a compensating operation.
- **Best-effort, no guarantee:** cancellation is a hint; if the worker is past the point of no return, the operation completes normally and the cancel is ignored. Document this clearly.

For a payout, stop-and-keep is usually right and must be stated in the docs: *"Cancelling a batch payout stops further payouts; payouts already sent are final."* The contract is the documentation here as much as the wire shape. A client must never assume `cancelled` means "as if it never happened" for a money-moving job.

One more state subtlety: a cancel request that arrives *after* the operation already reached a terminal state must be a no-op that reports the terminal state — you cannot cancel a `succeeded` operation. Return `200`/`202` with the existing terminal status (or `409 Conflict` if you want to be loud about it), never flip a terminal operation back to `cancelled`.

## 9. Result retrieval and expiry

The operation succeeded; now the client wants the answer. Two questions: *where does the result live*, and *how long does it stay there?*

**Where:** as covered in §3, embed small results in the operation body and link large ones. For the batch payout the result is a settlement report — large, paginated, a distinct media type possibility (JSON or CSV) — so it lives at its own URL, `/payouts/batch/{id}/report`, linked from `result.href`. This keeps the operation resource small and lets the report be fetched, paginated, and content-negotiated on its own terms. The report URL is a normal resource: it can be cached with an `ETag`, paginated (the failure rows might number in the thousands — pagination strategy in [offset, cursor, and keyset](/blog/software-development/api-design/pagination-offset-cursor-and-keyset-tradeoffs-at-scale)), and offered as CSV via content negotiation (`Accept: text/csv`).

**How long:** results do not live forever, and the `expires_at` field is the contract for their lifetime. A generated report, an export file, a transcoded video — each costs storage, and you will garbage-collect them. Be explicit: *"Operation results are retrievable for 7 days, after which `GET` returns `410 Gone`."* The `410 Gone` is the correct, honest status for a result that *existed and was deliberately removed* — it tells the client "this is permanently gone, do not retry," distinct from `404 Not Found` ("never heard of it"). RFC 9110 reserves `410` for exactly this: a resource that is intentionally and permanently unavailable.

```http
GET /v1/payouts/batch/pob_91c4/report HTTP/1.1
Host: api.example.com
Authorization: Bearer <token>
```

```http
HTTP/1.1 410 Gone
Content-Type: application/problem+json

{
  "type": "https://api.example.com/errors/result-expired",
  "title": "Result expired",
  "status": 410,
  "detail": "This batch payout report expired on 2026-06-27. Re-run the export to regenerate it.",
  "instance": "/payouts/batch/pob_91c4/report"
}
```

Should the *operation* resource outlive its result? Often yes — keep the lightweight operation record (status, timestamps, the summary) longer than the heavyweight result blob. A client polling an old operation can then still learn "this succeeded on June 20 and the result has since expired," which is far better than a bare `404`. The lifecycle is two-tiered: the operation record (cheap, kept for, say, 30 days) and the result blob (expensive, kept for 7 days). State both `expires_at` values and you have given the client everything it needs to reason about staleness without surprises.

For large results, a final refinement: hand back a **pre-signed, time-limited download URL** rather than streaming a multi-gigabyte file through your API. The `result.href` can be a signed object-store URL that expires in an hour; the client follows it to download directly. This keeps big payloads off your API's hot path entirely. (The transfer cost of big payloads — compression, connection reuse, the tail latency they add — is its own subject; see [API performance and payload size](/blog/software-development/api-design/api-performance-payload-size-compression-and-tail-latency).)

## 10. The Google AIP-151 LRO convention

You do not have to invent the operation-resource shape from scratch — Google formalized it as **AIP-151** (API Improvement Proposal 151, "Long-running operations"), and it is the closest thing the industry has to a standard for this pattern. If your API looks like AIP-151, anyone who has used Google Cloud, and a growing set of other APIs that adopted it, will recognize your operations instantly. That familiarity is a developer-experience win worth taking seriously — consistency is a feature.

The AIP-151 `Operation` message, in protobuf terms (because Google's APIs are gRPC-first and surface as REST via transcoding), looks like this:

```protobuf
message Operation {
  // Server-assigned, unique within the service. e.g. "operations/op_7f3a9c2e".
  string name = 1;

  // Service-specific metadata: progress, partial results, creation time.
  google.protobuf.Any metadata = 2;

  // False while running; true when the operation completes (success OR failure).
  bool done = 3;

  // Exactly one of these is set once done == true.
  oneof result {
    google.rpc.Status error = 4;        // failure: code, message, details
    google.protobuf.Any response = 5;   // success: the actual result message
  }
}
```

Several design decisions in AIP-151 are worth absorbing into any LRO design, REST or gRPC:

- **`done` is a boolean, not a multi-valued enum.** AIP-151 collapses pending/running into `done: false` and both terminal outcomes into `done: true`, then uses the `oneof` to say which terminal outcome it was. This is the minimal honest state model — a client only ever needs to ask "are we done?" and, if so, "did it succeed or fail?"
- **Success and failure are mutually exclusive via `oneof`.** Exactly one of `error` and `response` is set when `done` is true. This makes the "operation succeeded but the work failed" distinction structural: the *operation* is `done: true`, and `error` (a `google.rpc.Status`, Google's standard error shape) carries the work's failure. There is no way to accidentally set both.
- **Progress lives in `metadata`.** The generic `Operation` does not bake in a `percent` field; service-specific progress goes in the typed `metadata` (e.g. a `BatchPayoutMetadata` with `processed` and `total`). This keeps the operation envelope universal while letting each operation type report progress in its own terms.
- **Standard methods.** AIP-151 defines `GetOperation` (poll), `ListOperations` (enumerate), `DeleteOperation` (forget the record), `CancelOperation` (request a stop), and even `WaitOperation` (a long-poll that blocks server-side up to a bounded timeout, returning early if the operation finishes — a middle ground between polling and webhooks). You do not need all of these, but they map cleanly onto the REST shapes we built: `GET /operations/{id}`, `GET /operations`, `DELETE /operations/{id}`, and `POST /operations/{id}:cancel`.

The REST projection of AIP-151 is exactly the operation resource this post has been building — `GET /v1/operations/{id}` returning `{ "name": "...", "done": false, "metadata": {...} }`. The mapping between the protobuf model and the JSON-over-HTTP model is mechanical, which is the whole point: design the operation once, expose it consistently. The gRPC contract details — `.proto` design, codegen, streaming — are covered in [gRPC and Protocol Buffers](/blog/software-development/api-design/grpc-and-protocol-buffers-contracts-codegen-and-streaming); here the takeaway is to **steal AIP-151's shape rather than invent your own**, and your async API will feel familiar from the first request.

## 11. Partial results and streaming progress

Polling tells the client "62% done." Sometimes the client wants more — the *partial results so far*, or a live stream of progress without polling at all. Two refinements handle these.

**Partial results.** For some operations the partial output is itself useful before the whole job finishes. A bulk import that has validated and accepted 4,000 of 5,000 rows can expose those 4,000 as they land; a search reindex can serve already-indexed documents. The operation's `metadata` or a sub-resource can carry the partial result, and the client can start consuming it before `status` hits `succeeded`. This only works when the work is *incrementally meaningful* — a batch payout's partial result (4,960 of 8,000 paid) is meaningful and worth exposing; a single atomic transaction's partial result is not (it either committed or it did not). Be clear in the contract about whether partial results are *committed* (those 4,960 payouts are final) or *provisional* (these 4,000 rows are validated but not yet committed and could still be rejected).

**Streaming progress with Server-Sent Events.** If a client is interactive and wants live progress without a poll loop, Server-Sent Events (SSE) is a clean fit: the client opens a long-lived `GET` with `Accept: text/event-stream`, and the server streams progress events as they happen, closing the stream when the operation terminates.

```http
GET /v1/operations/op_7f3a9c2e/events HTTP/1.1
Host: api.example.com
Accept: text/event-stream
Authorization: Bearer <token>
```

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache

event: progress
data: {"percent": 40, "processed": 3200, "total": 8000}

event: progress
data: {"percent": 78, "processed": 6240, "total": 8000}

event: done
data: {"status": "succeeded", "result": {"href": "https://api.example.com/v1/payouts/batch/pob_91c4/report"}}
```

SSE gives the client real-time progress with one connection instead of N polls, degrades to a normal HTTP response (no special protocol like WebSocket), and is trivially consumable in a browser via `EventSource`. The trade-off is that it holds a connection open — which brings back some of the connection-pinning cost that motivated async in the first place — so it is best as an *optional* progress channel layered on top of the pollable operation resource, not a replacement for it. The choice between SSE, WebSocket, and server-streaming gRPC, and how to handle backpressure when the producer outruns the consumer, is its own deep topic; see [streaming APIs](/blog/software-development/api-design/streaming-apis-sse-websockets-and-server-streaming). The contract rule that survives: **streaming is an enhancement; the operation resource remains the source of truth a client can always fall back to.**

## 12. Stress-testing the design

A design is only as good as the failure modes it survives. Let us put the async payout API through the same interrogation this series applies to every contract: what happens when reality is hostile?

**What if the worker crashes mid-job?** The operation is stuck in `running` and never reaches a terminal state, so a polling client waits forever. This is the most important failure to design for, and the fix is a **timeout on the operation itself**, enforced by the server, not the client. Each operation carries a deadline (say, "a batch payout must terminate within 2 hours"); a reaper job scans for operations past their deadline that are still non-terminal and transitions them to `failed` with an error like `operation timed out — worker did not report completion`. The worker, when it picks up a job, should also be able to detect "this operation is already terminal (timed out or cancelled), abort" so a zombie worker that comes back from a pause does not resurrect a dead job. Without a server-side deadline, a single crashed worker becomes a permanently hung operation and a confused client.

**What if two workers grab the same job?** At-least-once queues can deliver the same message twice (covered in [delivery semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once)). If two workers both start the payout, you are back to double-paying. The worker must therefore claim the operation atomically — a conditional update like "set `status = running, worker_id = me` WHERE `id = op AND status = pending`" — and only the worker that wins the update proceeds; the loser sees zero rows affected and drops the message. The operation row is the lock. This is the same single-terminal-state discipline from §4 applied at the start of the work, not just the end.

**What if the client polls an operation it does not own?** Operation ids are guessable enough that authorization matters. A `GET /operations/{id}` must check that the caller is allowed to see *this* operation — tenant isolation, the same authorization you would apply to any resource (see [authorization, scopes, and roles](/blog/software-development/api-design/authorization-scopes-roles-and-resource-level-permissions) for the model). Returning another tenant's payout progress, or worse, their result link, is a data leak dressed up as a status check. Treat the operation and its result as protected resources, not public job trackers.

**What if a million operations pile up?** A busy platform accumulates operations fast, and unbounded retention turns your operations table into a swamp. Two defenses: the two-tier expiry from §9 (drop heavy result blobs early, keep light operation records longer, then garbage-collect those too), and a paginated `GET /operations?status=running&operation_type=payouts.batch` list endpoint so operators can find and triage stuck or failed jobs. List operations is part of AIP-151 for exactly this reason — operations are resources you will want to enumerate, filter, and clean up like any collection (filtering and pagination patterns are in the sibling posts on [pagination](/blog/software-development/api-design/pagination-offset-cursor-and-keyset-tradeoffs-at-scale) and filtering).

**What if the same operation is observed three different ways at once?** A real client might poll, *and* receive a webhook, *and* watch an SSE stream — and all three must agree. This is why the operation resource is the single source of truth and the others are projections of it: the webhook payload carries the same `operation_id` and terminal status, the SSE `done` event carries the same `result.href`, and a reconciling client that sees any of them can confirm by reading the operation. Designate one authority (the operation resource) and make every other channel a cache or notification of it, and the three views can never contradict in a way that corrupts the client's state.

The throughline of all five stress tests is the same principle: **the operation resource is the durable, authoritative record of the work, and every other mechanism — the queue, the worker, the webhook, the stream, the poll — is a way of producing or observing it.** Anchor the design on that record and the failure modes become tractable: a crashed worker is a timed-out operation, a duplicate delivery is a lost atomic claim, a leaked status is a missing authorization check. Each has a clean answer because there is one thing that is true.

## 13. Case studies: how the big platforms do it

The async pattern is not theoretical — every large API has converged on a variant of it. Three accurate, named examples, each illustrating a different emphasis.

**Google Cloud — AIP-151 Operations (the formal model).** Google's long-running APIs (Compute Engine instance creation, BigQuery jobs, Cloud Storage transfers, Vertex AI training) all return an `Operation` resource with the `name` / `done` / `metadata` / `oneof{error,response}` shape described in §10. A client calls a method like `instances.insert`, gets back an `Operation`, and polls `operations.get` (or uses `operations.wait` to long-poll) until `done` is true. BigQuery's query jobs are a textbook case: you insert a job, get a job resource, and poll `jobs.get` for its `status.state` (`PENDING` → `RUNNING` → `DONE`), reading results from a separate `getQueryResults` endpoint once done. The lesson: a single, uniform operation shape across an enormous, heterogeneous API surface is a developer-experience superpower — learn it once, use it everywhere.

**Stripe — async via webhooks plus pollable resources.** Stripe's model leans on the "both, with the resource as truth" principle. Many Stripe objects move through async state transitions — a `PaymentIntent` goes `requires_action` → `processing` → `succeeded`, a payout settles over days, a `Checkout Session` completes asynchronously. Stripe delivers these state changes as **events** (e.g. `payment_intent.succeeded`, `payout.paid`) to your registered webhook endpoint, signed with a secret you verify against the `Stripe-Signature` header. Critically, Stripe's documentation tells you to treat webhook delivery as best-effort and to **reconcile against the API** — you can always `GET` the object to read its current truth. Stripe also pioneered the `Idempotency-Key` header for safe retries of the kick-off, exactly the pattern in §7. The lesson: push for low latency, but make the resource the authority and the kick-off idempotent.

**AWS — job resources and async APIs.** AWS exposes long work as explicit job resources you create and then poll. Amazon Transcribe's `StartTranscriptionJob` returns immediately with a job whose `TranscriptionJobStatus` you poll via `GetTranscriptionJob` (`IN_PROGRESS` → `COMPLETED` / `FAILED`). AWS Batch, MediaConvert transcode jobs, Textract's async document analysis (`StartDocumentAnalysis` → `GetDocumentAnalysis`), and S3 Batch Operations all follow the same start-then-poll shape, and many can also publish completion to an SNS topic or EventBridge — AWS's webhook-equivalent push channel. The lesson: the "create a job resource, poll its status, optionally subscribe to a completion event" pattern is so universal that AWS reuses it across dozens of services with consistent naming (`Start*` / `Get*` / a status enum).

The convergence across three very different companies is itself the evidence: **accept-now-track-later, with an operation/job resource as the durable handle and an optional push channel, is the settled design for work that cannot finish in one request.** You are not inventing; you are adopting a battle-tested contract.

## 14. When to reach for this (and when not to)

Async is powerful and it is not free — it adds an operation resource, a polling protocol, a background worker, and more states for clients to handle. Reach for it deliberately.

**Reach for the LRO pattern when:**

- The operation's worst-case latency approaches or exceeds your tightest gateway timeout (the half-the-timeout rule of thumb from §1). A batch payout, a video transcode, a million-row import, a report generation, a model training run — anything that is *inherently* slow.
- The work is expensive and side-effecting, so a retry that re-triggers it is dangerous. The async kick-off plus idempotency key is the safe shape.
- Latency is unpredictable and depends on input size — a request that is fast for 10 items and slow for 10,000 should be async for *all* sizes, so the contract is uniform and the client does not have to guess whether this particular call will time out.
- The client benefits from progress (a UI progress bar) or from being notified later (a server-to-server integration that should not babysit a connection).

**Do NOT make an operation async when:**

- **It is fast.** Do not wrap a 50 ms `POST /orders` in an operation resource. You would force every client into a poll loop for work that finished before the first poll, double your round-trips, and add states for nothing. If the synchronous response fits comfortably inside your timeout budget, return it synchronously. Async is a cost you pay only when blocking would fail.
- **The latency is borderline but bounded and safe.** A 3-second `POST` against a 30-second gateway, with an idempotent handler, can stay synchronous — the async machinery would be over-engineering. Reserve async for work that genuinely cannot fit.
- **The result is needed atomically in the same logical step and the client truly cannot proceed without it** — though even here, prefer making the *step* async over holding a connection for minutes. (In practice this case is rarer than it feels; most "I need it now" turns out to tolerate a few seconds of polling.)
- **You would use it to paper over a slow query you could fix.** If the `POST /reports` is slow only because of an unindexed query, fixing the index (see [how database indexes work](/blog/software-development/database/b-trees-how-database-indexes-work)) may keep it synchronous and simpler. Async is for *inherently* long work, not for performance bugs.

The honest framing: **async trades client simplicity for server resilience.** A synchronous API is simpler to call — one request, one response. You accept the extra complexity of operations, polling, and webhooks precisely when the alternative (blocking) would fail outright. Do not pay that price for work that does not need it.

## 15. Key takeaways

- **Never block on work that can exceed your tightest gateway timeout.** A `504` kills the connection, not the work, and a naive retry re-triggers an expensive job. If p99 latency nears half your timeout budget, go async.
- **`202 Accepted` means acknowledged, not done.** Return it immediately with a `Location` pointing at an operation resource, and the synchronous handler's only job is to validate, persist, and enqueue.
- **Design the operation resource as a first-class contract:** a closed `status` enum, honest `progress`, a `result` (linked when large), an `error` as `problem+json` on failure, and `created_at` / `updated_at` / `expires_at`. The poll returns `200` even when the job failed — the failure lives in the body.
- **Every operation reaches exactly one terminal state and never leaves it.** `pending` → `running` → `succeeded` / `failed` / `cancelled`. Tell clients to treat unknown statuses as non-terminal so adding a state stays non-breaking.
- **Poll cheaply: honor `Retry-After`, back off exponentially on errors, and use `ETag` / `304` so unchanged polls send no body.** A twenty-minute job polled every five seconds should cost a dozen real responses, not 240.
- **Make the kick-off idempotent with an `Idempotency-Key`.** A retried `POST` must land on the existing operation, never start a second job — especially for money-moving work, where you may even require the key with `428`.
- **Offer webhooks as the push optimization, but keep the operation resource as the source of truth.** Webhook delivery is at-least-once and may drop, dupe, or reorder; a client that ignores webhooks and only polls must still be correct.
- **Model cancellation honestly.** `DELETE` (or a `:cancel` action) returns `202` because stopping is itself async, and `cancelled` rarely means "undone" for side-effecting work — say what partial work survives.
- **Expire results deliberately and return `410 Gone`** when a result is intentionally removed; keep the lightweight operation record longer than the heavyweight result blob.
- **Adopt AIP-151's shape rather than invent your own.** Familiarity is a feature; `done` plus a `oneof` of error/response is the minimal honest model.

## 16. Further reading

- **RFC 9110, HTTP Semantics** — §15.3.3 defines `202 Accepted` (the status monitor language), §10.2.3 defines `Retry-After`, and §15.5.11 defines `410 Gone`. The canonical source for every status and header used here: [https://www.rfc-editor.org/rfc/rfc9110](https://www.rfc-editor.org/rfc/rfc9110).
- **Google AIP-151, Long-running operations** — the formal `Operation` resource model (`name` / `done` / `metadata` / `oneof{error,response}`) and the standard `Get` / `List` / `Cancel` / `Wait` methods: [https://google.aip.dev/151](https://google.aip.dev/151).
- **RFC 9457, Problem Details for HTTP APIs** — the `problem+json` error envelope used in the operation's `error` field and the `410 Gone` response: [https://www.rfc-editor.org/rfc/rfc9457](https://www.rfc-editor.org/rfc/rfc9457).
- **Stripe API docs — idempotent requests and webhooks** — the production reference for an `Idempotency-Key` on the kick-off and signed, reconcilable webhook events as the push channel.
- **Within this series:** start at [what an API really is](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems); pair this post with [idempotency keys and safe retries](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions), [event-driven and async APIs](/blog/software-development/api-design/event-driven-and-async-apis-webhooks-pubsub-and-asyncapi), and [streaming APIs](/blog/software-development/api-design/streaming-apis-sse-websockets-and-server-streaming); and review everything in [the API design playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).
- **Broker internals:** for the queue behind the kick-off and the delivery guarantees behind webhooks, see [queue vs pub/sub vs log](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models) and [delivery semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once).

The async API is the moment your contract stops being "call this function and wait" and becomes "hand me this work and I will keep you posted." Designed well — `202` and a clean operation resource, paced polling, idempotent kick-off, optional webhooks, honest cancellation and expiry — it is the difference between an API that buckles the first time someone uploads ten thousand rows and one that absorbs the load and tells the caller exactly where things stand. That is the whole game this series plays: a contract the caller can trust, that you can run for years, that does not break when the work gets big.
