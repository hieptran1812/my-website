---
title: "Logging at Scale Without Going Broke: Keep the Signal, Cut the Bill"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Make logs queryable with structured fields and trace ids, then keep the bill sane with levels, sampling, and retention tiers — without ever dropping an error."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "logging",
    "observability",
    "structured-logging",
    "trace-id",
    "log-sampling",
    "retention",
    "loki",
    "cost-optimization",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/logging-at-scale-without-going-broke-1.png"
---

A finance team forwarded me an invoice with a sticky note that just said "is this right?" The log management line item read forty-five thousand dollars for the month. Six months earlier it had been eight thousand. Nobody had shipped a new product. Traffic was up maybe fifteen percent. So where did the other thirty-seven thousand dollars a month go?

It went to a single line of code. Someone, debugging a flaky checkout, had added a `log.debug("processing request", payload=request.body)` to the order service's hot path — the function that runs ten thousand times a second at peak. They never took it out. That one line logged the full request payload, several kilobytes each, on every request, at DEBUG, and the log pipeline was configured to keep and index everything for thirty days. Volume times retention times indexing. One chatty line, three multipliers, and a thirty-seven-thousand-dollar surprise.

Logs are the most useful and the most expensive of the three observability pillars. Metrics tell you *that* something is wrong — error rate is up, p99 latency spiked. Traces tell you *where* the time went across services. But logs tell you *why*: the exact decision the code made, with the exact inputs, for the exact request that failed. When you're staring at a Sev1 at 3am, the log line you wrote three weeks ago — the one that recorded which branch the code took and what the values were — is often the difference between a twenty-minute incident and a two-hour one. That is also exactly why logs are so easy to overspend on: the temptation is to log *everything* so the line you need is always there. Do that naively and you bankrupt the budget keeping ninety-nine percent of lines you will never read.

This post is about resolving that tension. By the end you will be able to: write **structured logs** that are queryable instead of greppable; propagate a **trace id** so you can pull one request's story across a whole fleet; use **log levels** that survive an incident, including bumping DEBUG dynamically without a redeploy; **sample** so the boring successes don't cost you a fortune while you keep one hundred percent of failures; tier **retention** from hot to cold so old logs cost cents instead of dollars; and reason about the **cost drivers** so you can spot the next forty-thousand-dollar line before it ships. This is the logging chapter of the broader SRE loop — define reliability, *measure* it, spend the budget, respond, learn, engineer the fix — and logging is squarely the *measure it* and *respond to it* part. For where logging sits next to metrics and traces, see the sibling post on [metrics, logs, and traces and when to use which](/blog/software-development/site-reliability-engineering/metrics-logs-and-traces-when-to-use-which); for the mindset that frames all of it, the series [intro on the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset).

![Diagram showing the four cost drivers of a logging bill, volume times retention times indexing amplified by a hot path, leading to a large monthly bill versus a small set of useful logs](/imgs/blogs/logging-at-scale-without-going-broke-1.png)

## 1. The cost model: volume times retention times indexing

Before we touch a single config, you need the one equation that explains every logging bill you will ever see. It is not complicated, and once it's in your head you will catch overspend before it ships.

The monthly cost of a logging system is approximately:

$$\text{cost} \approx (\text{volume} \times \text{retention} \times \text{index factor}) + \text{query cost}$$

- **Volume** is bytes ingested per day. This is driven by how many lines you emit times the average size of each line. A line is `timestamp + level + message + fields`. Ten thousand requests a second, each logging a five-kilobyte payload at INFO, is fifty gigabytes a day from *one* endpoint.
- **Retention** is how many days you keep each byte, and — crucially — *in what tier*. A byte kept hot and searchable for thirty days costs far more than the same byte archived to cold object storage.
- **Index factor** is how much extra storage and compute you pay to make logs searchable. An "index everything" store like Elasticsearch builds an inverted index over every field, which can cost as much as or more than the raw data itself. An "index on read" store like Loki indexes only a small set of labels and scans the rest at query time, so the index factor is close to one but you pay more per query.
- **Query cost** is what you pay to actually search. In an index-everything model queries are cheap because the index does the work. In an index-on-read model the storage is cheap but a wide query that scans terabytes can be slow and, in a managed service, billed.

The trap is that these multiply. Halving volume halves the bill. But cutting retention from thirty days to seven *and* moving the rest to cold *and* sampling the boring ninety-nine percent doesn't just add up — it compounds, because you're shrinking volume and retention and the index all at once. That is why the same incident-fixing playbook — sample successes, drop payloads, tier retention — routinely cuts a bill by eighty percent rather than thirty.

Here is the mental shift that matters most: **a single chatty line on a hot path is the single most expensive object in your entire system.** A line that runs once a day is free no matter how big it is. A line that runs ten thousand times a second is a budget line item with a price tag, and you should treat adding one with the same seriousness as adding a database. The cost of a log statement is its size times its call frequency times your retention times your index factor — and on a hot path the frequency term dominates everything.

#### Worked example: the forty-thousand-dollar line

Let's price the opening story honestly. The order service runs at 10,000 requests per second at peak, averaging maybe 4,000 over the day, so call it 350 million requests a day. The added line logged the full request payload — about 5 KB serialized as JSON with the surrounding fields.

- Volume from this one line: 350M requests × 5 KB ≈ **1.75 TB per day**.
- Over a 30-day hot retention window, that's ~52 TB resident, indexed.
- At a representative managed-ingest price of roughly \$0.50 per GB ingested plus index and hot storage, 1.75 TB/day is about \$875/day in ingest alone, ≈ **\$26k/month** — and that's before index storage and query.

The numbers above are illustrative — exact vendor pricing varies a lot — but the *shape* is real and defensible: one DEBUG line dumping a payload on a hot path is a five-figure-per-month object. The fix, which we'll build piece by piece, was: drop the payload (keep the ids), sample the successes, and the line that survived cost a rounding error. Same debuggability for the cases that matter, eighty percent off the bill.

### A back-of-envelope cost calculator

It's worth being able to price a log line *before* you ship it, the way you'd estimate a query's cost before adding an index. The arithmetic is simple enough to keep in your head or in a five-line script you run in code review:

```python
def monthly_log_cost(lines_per_sec, bytes_per_line, hot_days,
                     price_per_gb_ingest=0.50, hot_storage_per_gb_day=0.02):
    """Rough monthly cost of one log statement on a hot, indexed tier."""
    bytes_per_day = lines_per_sec * bytes_per_line * 86_400
    gb_per_day = bytes_per_day / 1e9
    ingest = gb_per_day * 30 * price_per_gb_ingest
    # Resident hot storage: ~hot_days of data sitting indexed, billed per day.
    storage = gb_per_day * hot_days * hot_storage_per_gb_day * 30
    return round(ingest + storage)

# The $40k line: 4000 lines/s average, 5 KB each, 30 days hot.
print(monthly_log_cost(4000, 5000, 30))   # ~$28,000+ for ONE statement

# The same line after the fix: drop payload (200 B), sample 1% (40 lines/s), 7 days hot.
print(monthly_log_cost(40, 200, 7))        # a rounding error
```

Run those two numbers and the entire thesis of this post is right there: the first call prices a single chatty statement in the tens of thousands of dollars a month; the second, after dropping the payload, sampling, and shortening the hot window, prices the *same debuggable line* at near-zero. The point of the calculator isn't precision — your real prices differ — it's *ratios*. It tells you which line is the expensive one, and it tells you that the three levers (size, frequency, hot retention) multiply. Put a check like this in your code-review checklist for any log statement on a path that runs more than a few hundred times a second and you will never ship a five-figure surprise again.

The rest of this post is the toolbox for pulling each of those levers without losing the log line you'll wish you had.

## 2. Structured logging: log the decision and the inputs, not "done"

The first and highest-leverage move costs you nothing on the bill and makes every other technique possible: stop writing logs as English sentences and start writing them as **structured events** — JSON or key-value pairs with named fields.

![Before and after comparison contrasting a string-concatenated log line that can only be grepped against a structured JSON event with named fields that can be filtered and aggregated](/imgs/blogs/logging-at-scale-without-going-broke-2.png)

Here's the difference. A string-concatenated log line looks like this:

```python
log.info("User " + user_id + " checkout failed after " + str(elapsed) + "ms, code " + str(code))
```

It produces `User 88213 checkout failed after 1430ms, code 503`. To a human reading one line, fine. But you don't read one line — you read a million. To find all checkout failures for one user you `grep` for a substring and pray the format never changed. To compute the p99 of `elapsed` you write a regex to pull the number out of the middle of a sentence, and that regex breaks the day someone reorders the words. You cannot aggregate. You cannot filter cleanly. You cannot build a chart. The log is a story, and stories don't sum.

A **structured** log of the same event looks like this:

```python
log.info("checkout_failed",
         user_id=user_id,
         elapsed_ms=elapsed,
         status_code=code,
         payment_provider="stripe",
         trace_id=trace_id)
```

It produces `{"ts":"2026-06-20T11:04:31Z","level":"info","msg":"checkout_failed","user_id":88213,"elapsed_ms":1430,"status_code":503,"payment_provider":"stripe","trace_id":"abc123","service":"orders","version":"2.14.1"}`. Now every field is a named, typed column. You can filter `status_code=503`, group by `payment_provider`, compute `quantile(elapsed_ms)`, and pull one `user_id` — all without a regex, all aggregatable.

The principle behind *what* to log is just as important as the format: **log the decision and the inputs, not the fact that you finished.** A log line that says `"done"` or `"processing complete"` tells you nothing when you're debugging — of course it's done, you're holding the response. The line you'll wish you had records *why the code did what it did*: which branch it took, what the inputs were, what it decided. Don't log `"validated order"`; log `order_validation` with `result=rejected`, `reason=insufficient_inventory`, `requested_qty=12`, `available_qty=3`. That line answers the question you'll actually ask at 3am — *why did this specific order get rejected?* — and the bare `"validated order"` does not.

### The fields that make a log debuggable

Every structured log line should carry a baseline set of fields, attached automatically by the logger, not by hand. These are the fields that let you slice the log corpus across an incident:

| Field | Why it's load-bearing |
|---|---|
| `timestamp` | RFC3339 / ISO8601 with timezone. The spine of every query and correlation. |
| `level` | error / warn / info / debug — drives sampling, alerting, and retention. |
| `service` | which service emitted it — you'll filter by this first in a fleet. |
| `version` | the deployed build/commit. Lets you say "only the new version logs this." |
| `trace_id` | the correlation id (next section) — pull one request across all services. |
| `host` / `pod` | which instance — isolates a bad node from a bad release. |
| `message` | a *stable, low-cardinality* event name, not an interpolated sentence. |
| (the actual values) | the inputs to the decision: ids, counts, codes, durations. |

That last row is the whole point. **Log the actual values.** The line that just says `payment_failed` is a metric in disguise — it tells you the count went up but not why. The line that says `payment_failed` with `provider=stripe`, `decline_code=insufficient_funds`, `amount_cents=4999`, `retry_count=2` lets you answer the real question. The cost of these fields is small; the debuggability they buy is enormous. This is the asymmetry that makes structured logging the best deal in observability: a tiny, fixed per-line overhead in exchange for queryability you literally cannot get any other way.

One discipline that follows from structure: **keep the `message` field low-cardinality and stable.** `message` should be the *name of the event* (`checkout_failed`, `cache_miss`, `db_query_slow`), not a sentence with values baked in. Why? Because in an index-on-read store, the message stays scannable and groupable; and because a stable event name is the thing you alert on and dashboard, while the high-cardinality detail lives in the fields. Mixing the user id into the message string defeats both.

### Structured loggers in three languages

The good news is that every mainstream runtime now ships or has a de facto standard for structured logging, and configuring it correctly is a one-time setup that every subsequent log line inherits. The pattern is identical across languages: a JSON handler, a fixed set of baseline fields bound once, and a context mechanism that carries the trace id so you never have to pass it by hand. Here is the same disciplined setup in Go, Python, and Node.

Go's standard-library `slog` (Go 1.21+) gives you JSON output and bound fields for free:

```go
// One-time setup: JSON output, service + version on every line.
logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
    Level: logLevel, // the atomic LevelVar from the dynamic-bump section
}))
logger = logger.With(
    "service", "orders",
    "version", os.Getenv("GIT_SHA"),
)
slog.SetDefault(logger)

// At the call site: log the decision and the inputs, never just "done".
slog.Info("order_validation",
    "result", "rejected",
    "reason", "insufficient_inventory",
    "requested_qty", 12,
    "available_qty", 3,
    "trace_id", traceID)
```

Python's `structlog` does the same, with processors that inject the timestamp, level, and any context-bound fields:

```python
import structlog

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,   # pulls trace_id from context
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),        # emit JSON, not a sentence
    ],
)
log = structlog.get_logger().bind(service="orders", version=os.environ["GIT_SHA"])

# Bind the trace id once per request; every later line carries it.
structlog.contextvars.bind_contextvars(trace_id=trace_id)
log.info("order_validation", result="rejected",
         reason="insufficient_inventory", requested_qty=12, available_qty=3)
```

And Node's `pino`, which is fast precisely because it serializes straight to JSON and does the work off the hot path:

```javascript
const pino = require('pino')
const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  base: { service: 'orders', version: process.env.GIT_SHA }, // every line
})

// A per-request child logger binds the trace id once.
function handler(req, res) {
  const log = logger.child({ trace_id: req.headers['traceparent'] })
  log.info({ result: 'rejected', reason: 'insufficient_inventory',
             requested_qty: 12, available_qty: 3 }, 'order_validation')
}
```

Three languages, one shape: configure JSON once, bind the baseline fields once, carry the trace id through context, and log the decision with its inputs. Notice what is *not* in any of these — string concatenation, interpolated sentences, or a bare `"done"`. The event name is stable and low-cardinality; the variable detail lives in fields. Get this setup right at the start of a service's life and every line for the rest of that service's life is queryable by default. Retrofitting it onto a mature codebase is painful precisely because the value compounds — which is the argument for doing it on day one.

## 3. Trace ids: pull one request's logs across the whole fleet

If structured logging is the highest-leverage cheap move, propagating a **correlation id** — usually called a `trace_id` — is the single most valuable thing you can do in a distributed system. It is the difference between debugging one service and debugging a request.

![Diagram of a trace id minted at the edge proxy and propagated unchanged through the API gateway, auth service, order service, and database, so a single LogQL filter reconstructs the whole request](/imgs/blogs/logging-at-scale-without-going-broke-3.png)

Here's the problem it solves. A single user click — "place order" — fans out into a dozen service calls: the edge proxy, the API gateway, auth, the order service, inventory, payments, the database, a cache, a message queue. Each of those runs on a fleet of instances. When that one order fails, its log lines are scattered across a dozen services and hundreds of pods, interleaved with millions of other requests' lines. Without a shared id, finding "the logs for *this* request" is hopeless — you'd be correlating by timestamp and guessing.

A **trace id** fixes this. You mint a unique id at the very edge of the system — the load balancer or the first service to see the request — and you propagate it, unchanged, through every downstream call, usually as an HTTP header (`traceparent` per the W3C Trace Context standard, or a custom `X-Request-ID`). Every service stamps that id onto every log line it emits for that request. Now pulling one request's entire story across the whole fleet is a single filter: `trace_id="abc123"`. One query, all hops, in order.

The reasoning for *why this is the highest-value distributed move*: it converts an O(fleet-size) search problem into an O(1) lookup. Before the trace id, finding a request's logs is a needle-in-a-haystack scan across every service. After, it's a key lookup. During an incident, that's the difference between "I have the request's full path in thirty seconds" and "I'm grepping six services by timestamp and hoping." This is also the bridge between your logs and your traces: the same `trace_id` ties a log line to a span in your tracing system, which is why the sibling post on [distributed tracing in practice](/blog/software-development/site-reliability-engineering/distributed-tracing-in-practice) and this one are two halves of one idea — the trace id is the join key.

### Propagating the id in practice

The id has to be *propagated*, which means two things: read it from the inbound request (or mint it if absent), and forward it on every outbound call. In Go with `context.Context`, the idiom is to pull the id into context at the inbound middleware and have your logger read from context automatically:

```go
// Inbound middleware: read or mint the trace id, stash it in context.
func TraceMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        traceID := r.Header.Get("traceparent")
        if traceID == "" {
            traceID = newTraceID() // generate a fresh 128-bit id
        }
        ctx := context.WithValue(r.Context(), traceIDKey, traceID)
        // Make every log line in this request carry the id automatically.
        logger := slog.With("trace_id", traceID)
        ctx = context.WithValue(ctx, loggerKey, logger)
        next.ServeHTTP(w, r.WithContext(ctx))
    })
}
```

The two mistakes that break trace ids in practice are worth naming. First, **forgetting to forward the header on outbound calls** — your HTTP client and your message-queue producer both have to copy the id onto the next hop, or the chain snaps at that service. Wrap your outbound client once so it's automatic; don't rely on every call site to remember. Second, **minting a new id per service instead of propagating** — if each service generates its own id, you have a dozen ids for one request and you've gained nothing. Mint once at the edge, propagate everywhere.

Once the id flows, the payoff is the query in §8: `{service="orders"} | json | trace_id="abc123"` and you have that request's life story. We'll come back to that. For now, internalize the rule: **a log line without a trace id is a log line you can't correlate.** Make the trace id non-optional in your logger's baseline fields.

## 4. Log levels that survive an incident

Levels are the throttle on the whole system. They decide what gets emitted, what gets kept, what pages you, and — done right — what you can turn *up* during an incident without a deploy. Most teams treat levels as vibes ("this feels important, log it at INFO"). Treat them as a contract instead.

![Matrix mapping each log level from ERROR down to TRACE to when to use it, whether to keep it in production, and its cost, showing errors always kept and DEBUG off by default](/imgs/blogs/logging-at-scale-without-going-broke-4.png)

Here is the discipline, level by level:

- **ERROR** — something failed that the user or an operator needs to know about: a request that returned 5xx, a payment that couldn't be charged, a write that was lost. ERROR means *page-worthy or at least dashboard-worthy*. You keep one hundred percent of ERROR forever-ish (within retention). If you find yourself logging recoverable, expected things at ERROR, you've broken the contract and you'll train people to ignore it.
- **WARN** — something degraded but the system handled it: a retry succeeded on the second try, a fallback kicked in, a cache missed and you went to the database, you shed load. WARN is the early-warning band. You keep it, and a rising WARN rate is often your first signal before the ERRORs start.
- **INFO** — a notable, normal state change: a request was served, an order was placed, a job started. INFO is the bulk of your volume and the bulk of your bill. The discipline here is *ruthless*: INFO must stay readable, which means you sample it (next section) and you do not log every loop iteration at INFO. If INFO is noise, nobody reads it and you're paying to store noise.
- **DEBUG** — the decision-and-inputs detail you want *when you're debugging*: which branch, what the values were, the intermediate state. DEBUG is **off in production by default** because it's expensive, and you turn it on dynamically and temporarily during an incident (the key move, below).
- **TRACE** — the most granular, per-line, "I am here" detail. Almost never on in production; it bankrupts you if it is. Useful in local dev and the occasional surgical, time-boxed prod enablement on a single instance.

The principle that ties these together: **levels are a cost and attention budget, not a description of how much you care.** A line's level determines its frequency-weighted cost (DEBUG on a hot path is the \$40k line) and its claim on a human's attention (every ERROR should be something a human would want to know). Keep the contract and INFO stays useful, ERRORs stay trustworthy, and DEBUG stays affordable.

### DEBUG in prod, behind a dynamic flag

Now the move that separates teams that survive incidents from teams that flail: **you must be able to bump a service's log level to DEBUG, in production, for a few minutes, without redeploying.**

Why is this so valuable? Because the bug you're chasing at 3am is happening *right now*, in production, and you cannot reproduce it locally. The DEBUG lines that would tell you exactly what's happening are in the code — but they're suppressed because DEBUG-always-on would bankrupt you. So you need a runtime control: a flag, a config endpoint, an environment value the process re-reads, that flips one service (or one pod) to DEBUG for a bounded window. You get the rich detail for exactly the slice of time and traffic you need, then it flips back, and the bill never notices.

Concretely, most logging libraries expose a way to set the level at runtime via an admin endpoint. With Go's `slog`, you hold the level in an atomic `LevelVar`:

```go
var logLevel = new(slog.LevelVar) // defaults to Info

func init() {
    logLevel.Set(slog.LevelInfo)
    h := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: logLevel})
    slog.SetDefault(slog.New(h))
}

// Admin endpoint, authenticated, to bump the level at runtime.
// curl -XPOST localhost:9000/loglevel -d 'level=debug'
func handleLogLevel(w http.ResponseWriter, r *http.Request) {
    switch r.FormValue("level") {
    case "debug":
        logLevel.Set(slog.LevelDebug)
    case "info":
        logLevel.Set(slog.LevelInfo)
    }
    // ALWAYS schedule a revert so you can't leave DEBUG on by accident.
    time.AfterFunc(10*time.Minute, func() { logLevel.Set(slog.LevelInfo) })
}
```

Two non-negotiables: **authenticate the endpoint** (an unauthenticated "make me log everything" is a denial-of-service and a data-exfiltration risk), and **auto-revert** on a timer so a forgotten DEBUG flag doesn't become next month's \$40k surprise. Kubernetes shops often do this with a `ConfigMap` the process watches, or a feature-flag service (LaunchDarkly-style) so you can target one service or even one user. We'll walk a full incident driven by this move in §10.

## 5. Sampling: keep 100% of failures, a slice of the successes

Here is the central insight of cost-controlled logging: **your successes are boring and your failures are not.** The millionth identical "request served, 200, 12ms" tells you nothing new. The one "request failed, 503, timeout" tells you everything. So the strategy writes itself — keep a *sample* of the boring successes and *one hundred percent* of the failures.

![Tree diagram contrasting head sampling that drops boring successes at ingress with tail sampling that rescues failed and slow traces at egress, both keeping all errors](/imgs/blogs/logging-at-scale-without-going-broke-5.png)

This is sampling, and the discipline that makes it safe is the rule **always keep errors.** A sampling policy that drops a fraction of *everything* will eventually drop the one error line you needed, and you'll never know it existed. A policy that keeps every ERROR and WARN and samples only the successful INFO is the one that lets you cut volume by ninety-plus percent while never losing a signal that matters.

There are two places you can decide to drop, and they catch different things:

- **Head sampling** decides *at ingress*, before you know how the request turned out. You flip a weighted coin: "keep one in a hundred of these." It's cheap (you decide and drop immediately, you never pay to store the dropped lines) but blind — you might sample-out a request that was about to fail, because at head time you don't know yet. Head sampling is right for high-volume, mostly-uniform success traffic where you just want a representative slice.
- **Tail sampling** decides *at egress*, after the request finishes and you know its outcome. You buffer the request's lines and then decide: "this one errored — keep it; this one was slow, over the p99 — keep it; this one was a boring 200 in 8ms — drop it." Tail sampling is smarter (it rescues exactly the interesting requests) but costs more (you buffer everything until you can decide, and you need the whole request's lines in one place). Tail sampling is right when you specifically want every error and every latency outlier and can afford the buffering.

The honest trade-off: head is cheap and simple but can miss; tail catches the interesting ones but costs buffer and complexity. Many real pipelines do *both* — head-sample the firehose of clearly-boring successes to cut raw volume, then tail-sample the rest to guarantee every error and slow trace survives.

There's one more subtlety that separates a naive sampler from a good one: **sample consistently per request, not per line.** If you flip an independent coin on *each* log line, a kept request might have its first three lines dropped and its last two kept — and you've turned one coherent request into useless fragments. Instead, make the keep/drop decision once per `trace_id` (hash the id, keep if the hash falls under your rate) so that *either* a request's whole story is kept *or* it's entirely dropped. Deterministic, hash-based sampling on the trace id has a second nice property: every service in the fleet makes the *same* decision for the same request without coordinating, because they all hash the same id. So a sampled-in request is sampled in *everywhere* — you get its complete cross-service story, not a service's worth of orphaned fragments. That coherence is what makes a 1% sample actually usable for debugging rather than just for counting.

#### Worked example: the math of a 1% sample

It's worth seeing why 1% is usually plenty for the boring path and why "always keep errors" barely costs anything. Suppose the order service does 4,000 requests/sec on average, and 99.8% succeed (a healthy 0.2% error rate). That's ~8 errors/sec and ~3,992 successes/sec.

- Keep **100% of errors**: ~8 lines/sec. Even with a few lines per failed request, that's a trickle — a few dozen lines a second, trivially affordable, and you have *every* failure in full.
- Keep **1% of successes**: ~40 lines/sec instead of ~3,992. You've dropped ~99% of the volume that was almost entirely identical 200s.
- Total kept: ~48 lines/sec versus ~4,000 — about **1.2% of the original volume**, and yet you retained *every single error* and a statistically solid 1-in-100 sample of the successes (more than enough to characterize normal behavior, latency distributions, and traffic mix).

The asymmetry is the whole point: errors are rare, so keeping all of them is cheap; successes are common and repetitive, so keeping a sample of them loses almost nothing. You cut volume ~80× while keeping 100% of the signal that matters. That is why "sample successes, keep errors" is the highest-return single policy in log cost control.

### Sampling on the hot path: log-once and rate-limit

Sampling has a second job beyond cost: **protecting you from log-driven self-harm during an incident.** Here's the failure mode. A dependency goes down. Suddenly every request hits the same error and logs it. At ten thousand requests a second, you're now emitting ten thousand identical ERROR lines a second — and the act of logging (serializing, writing, shipping) consumes CPU and I/O and network, which makes the incident *worse*, and floods your log store so badly that searching for the *cause* becomes impossible because the symptom drowns it. The log meant to help you debug the outage becomes part of the outage.

The defenses are **log-once** and **rate-limiting** on hot paths:

```go
// Rate-limit a hot-path log to at most one line per second per key,
// so a dependency outage doesn't emit 10k identical errors a second.
var logLimiter = rate.NewLimiter(rate.Every(time.Second), 1)

func logDependencyError(err error) {
    if logLimiter.Allow() {
        slog.Error("payment_provider_down",
            "err", err.Error(),
            "suppressed_note", "rate-limited to 1/s")
    }
    // Always still increment a metric — counts belong in metrics, not 10k logs.
    paymentErrors.Inc()
}
```

The principle underneath: **counts belong in metrics, not in a log line per occurrence.** If you want to know "how many payment errors," that's a Prometheus counter — one cheap time series — not ten thousand log lines a second. Log *one* representative line per second with the detail (the error, the trace id, the provider), increment a metric for the count, and you have both the "what's happening" detail and the "how much" number, for the price of one line per second instead of ten thousand. This is one of those places where logging and metrics divide the labor cleanly; the sibling on [metrics, logs, and traces and when to use which](/blog/software-development/site-reliability-engineering/metrics-logs-and-traces-when-to-use-which) draws the full boundary.

#### Worked example: cutting the bill 80% without losing a signal

Back to the \$45k bill. Here's the surgery, lever by lever, with the arithmetic:

- **Starting point:** the order service logs the full payload at DEBUG on the hot path, plus a verbose INFO line per request, all kept hot and indexed for 30 days. Roughly 50 GB/day of *useful-looking* volume sitting in the hot tier.
- **Lever 1 — drop the payload, keep the ids.** The payload was 5 KB; the fields that actually help (trace_id, user_id, route, status, latency) are ~200 bytes. Dropping the payload alone cuts per-line size ~25×. The DEBUG line, which was the bulk, falls off entirely once it's no longer dumping the body.
- **Lever 2 — sample INFO at 1%, keep 100% of ERROR/WARN.** The success-path INFO was ~90% of the remaining lines and almost entirely identical 200s. Sampling those at 1% removes ~89% of the INFO volume. Every error and every warning still goes through untouched.
- **Lever 3 — tier retention (next section).** Keep 7 days hot, age the rest to cold S3 archive. The hot tier — the expensive part — shrinks from 30 days to 7.

Net effect on the *hot, indexed* volume: from ~50 GB/day to roughly **9 GB/day**, an ~82% cut. The bill fell from **\$45k to about \$9k per month**. And here's the part that matters: **debuggability for the cases that count went up, not down.** Every error is still 100% retained, every error and every sampled request still carries its trace id, and you can still pull one failed request's whole story. You stopped paying to store a million identical successes; you kept every failure. That's the whole game.

## 6. Retention tiers: hot, warm, and cold

The second multiplier in the cost equation is retention, and the lever here is that **not all logs need to be equally fast to search.** A log from five minutes ago, during an active incident, needs to be searchable in under a second. A log from five months ago, needed for a compliance audit or a rare forensic dig, can take minutes to retrieve and nobody minds. Charging hot-tier prices for both is how you overpay.

![Stacked diagram of three retention tiers, a hot tier indexed for seven days, a warm tier for the rest of the month, and a cold S3 Glacier archive for a year, with cost falling by an order of magnitude per tier](/imgs/blogs/logging-at-scale-without-going-broke-6.png)

The standard structure is three tiers, and the cost falls by roughly an order of magnitude at each step:

| Tier | Retention | Search speed | Cost (illustrative) | What lives here |
|---|---|---|---|---|
| **Hot** | 3–7 days | sub-second, fully indexed | ~\$0.50/GB/month | active debugging, live incidents, recent dashboards |
| **Warm** | 8–30 days | seconds, lighter index | ~\$0.10/GB/month | last-month trend digs, slower investigations |
| **Cold / archive** | 90 days – 7 years | minutes (restore from S3/Glacier) | ~\$0.004/GB/month | compliance, audits, rare forensics |

Those per-GB numbers are illustrative and depend heavily on vendor and region, but the *ratio* is real: cold object storage is something like a hundred times cheaper per byte than hot, indexed log storage. The discipline is simply to **keep recent logs hot and age everything else down.** Most queries hit the last few days; serve those fast and expensive, and let the long tail of old logs live cheaply where it belongs.

The reasoning for *why this works so well*: query frequency decays sharply with log age. The overwhelming majority of your log searches are for the last twenty-four to seventy-two hours — the incident you're in, the deploy you just shipped, the bug reported this morning. Queries against logs older than a week are rare and almost never time-critical. So you're paying premium hot-tier prices for data that, after about a week, is queried approximately never with any urgency. Tiering aligns the cost of storage with the actual value-over-time of the data. You're not deleting anything you might need; you're just not paying sub-second-search prices for data you'll search at most once, at your leisure.

A concrete tier policy for Loki, expressed as retention rules, looks like this:

```yaml
# Loki retention: keep most logs short, audit logs long, all aged to cold.
limits_config:
  retention_period: 168h  # default 7 days hot for everything
schema_config:
  configs:
    - from: 2026-01-01
      store: tsdb
      object_store: s3   # chunks live in S3; cheap cold storage by default
      schema: v13
compactor:
  retention_enabled: true
  delete_request_store: s3
overrides:
  # Compliance/audit streams kept far longer than ordinary app logs.
  "tenant-audit":
    retention_period: 2160h   # 90 days
  # Chatty debug stream aged out fast — it's only useful while fresh.
  "tenant-debug":
    retention_period: 24h
```

Two things make this policy good. First, **retention is per-stream, not global** — your audit logs (which a regulator may require for years) and your DEBUG firehose (useful for a day) have wildly different value-over-time, so they get wildly different retention. Second, **the object store is S3 from the start** — Loki keeps its chunks in cheap object storage, so "cold" isn't a separate manual archival step, it's the default substrate; you tune how long the *index* and hot path keep things searchable. The same idea applies to Elasticsearch via index lifecycle management (ILM) rolling indices from hot to warm to frozen nodes, and to CloudWatch via exporting old log groups to S3 with a lifecycle rule to Glacier.

### Stress-testing the retention policy

Let's pressure-test it, because a retention policy you haven't stress-tested is a policy that fails at the worst moment.

*What if you need a log from 100 days ago during an incident?* It's in cold archive. Restoring from S3 Glacier can take minutes to hours depending on the retrieval tier. That's fine for forensics but *not* fine mid-incident — so the rule is: anything you might need *during* an incident must be in a tier that's fast enough for an incident. If you discover you're regularly cold-restoring during incidents, your hot window is too short; lengthen it. The tiers are a dial, not a religion.

*What if a regulator requires seven years of audit logs?* Then audit logs get a seven-year retention on the cold tier — but *only the audit logs*, via per-stream overrides, not your entire log volume. The mistake teams make is applying the strictest retention requirement to *all* logs, which is how you end up keeping seven years of DEBUG. Separate the streams; retain each for what *it* actually needs.

*What if you delete something you needed?* This is the real risk of aggressive retention. The mitigation is to be generous with cold (it's nearly free) and aggressive only with hot. Cheap cold storage means "keep it just in case for a year" costs cents, so default to *aging down* rather than *deleting*. Delete is for when storage genuinely isn't free or when privacy law *requires* deletion — which brings us to the one thing that must never be logged at all.

## 7. The pipeline, the index models, and the one rule about PII

Let's assemble the moving parts into a pipeline, because *where* in the pipeline you apply each lever determines how much it saves.

![Diagram of the log pipeline from an application emitting JSON events to a Vector or Fluent Bit agent that buffers and samples, dropping boring successes, then routing to a Loki hot store and an S3 archive, with Grafana querying by trace id](/imgs/blogs/logging-at-scale-without-going-broke-7.png)

The standard shape is: **app → collector agent → store(s) → query**.

- **The app** emits structured JSON to stdout (in containers) or a local socket. Crucially, the app should do as little as possible — just emit the event. It should *not* be the place you do heavy filtering or shipping, because that work steals CPU from serving requests.
- **The collector agent** — Fluent Bit, Vector, or the OpenTelemetry Collector — runs as a sidecar or a node-level daemon. This is where the magic happens: it buffers (so a downstream outage doesn't block the app), parses, enriches (adds pod/node/region labels), **samples and rate-limits**, drops fields you don't want (like that payload), redacts PII, and routes. *The collector is the cheapest place to drop a log line, because you drop it before it's ingested, indexed, and billed.* Every byte you drop at the agent is a byte you never pay to store.
- **The store(s)** — your hot store (Loki, Elasticsearch, CloudWatch) plus your cold archive (S3). The agent can fan out: send the sampled, enriched stream to the hot store and the full firehose (or a fuller sample) straight to cheap S3.
- **The query layer** — Grafana over Loki (LogQL), Kibana over Elasticsearch, CloudWatch Logs Insights — where you actually search, ideally starting from a trace id.

A minimal Vector config that samples successes, keeps all errors, drops the payload, and fans out to hot and cold:

```yaml
# vector.toml — keep all errors, sample the boring 200s, archive everything cheap.
[sources.app_logs]
type = "stdin"

[transforms.parse]
type = "remap"
inputs = ["app_logs"]
source = '. = parse_json!(.message)'

[transforms.redact]
type = "remap"
inputs = ["parse"]
# NEVER ship raw PII/secrets. Drop or hash sensitive fields at the agent.
source = '''
del(.password)
del(.authorization)
del(.request_payload)        # the $40k field — gone before it costs a cent
if exists(.email) { .email = "redacted" }
'''

[transforms.sample]
type = "sample"
inputs = ["redact"]
rate = 100                    # keep 1 in 100 by default ...
exclude.type = "vrl"
exclude.source = '.level == "error" || .level == "warn"'  # ... but never drop errors

[sinks.loki_hot]
type = "loki"
inputs = ["sample"]
endpoint = "http://loki:3100"
labels = { service = "{{ service }}", level = "{{ level }}" }

[sinks.s3_cold]
type = "aws_s3"
inputs = ["redact"]           # full stream (pre-sample) to cheap cold storage
bucket = "logs-archive"
```

Read that `exclude` carefully — it's the "always keep errors" rule encoded as config. The `sample` transform drops ninety-nine of every hundred lines *except* those where `level` is error or warn, which pass through untouched. That one rule is what lets you turn the sampling knob hard without fear.

### Index-everything versus index-on-read

The biggest architectural cost decision is your store's indexing model, and it's a genuine trade-off, not a free lunch.

| Model | Example | Index cost | Query speed | Best when |
|---|---|---|---|---|
| **Index everything** | Elasticsearch / OpenSearch | high (inverted index over all fields) | fast arbitrary queries | you search lots of fields, need full-text, can pay for it |
| **Index on read** | Loki | low (index only labels) | fast on labels, scans the rest | you mostly filter by a few labels + grep, want cheap storage |

The reasoning: Elasticsearch builds an inverted index over every field at ingest time, so any query is fast — but you pay for that index in storage and compute, often doubling your effective cost, and high-cardinality fields (like a unique trace id per request) can blow up the index. Loki takes the opposite bet: it indexes only a small set of *labels* (service, level, namespace) and stores the rest of the line as a compressed, *unindexed* blob; queries filter cheaply by label to narrow the data, then brute-force scan the matching chunks. Storage is dramatically cheaper, but a query that scans terabytes is slower and, if you're not careful with labels, can be expensive in compute.

The practical guidance: if your access pattern is "filter by service and level, then look for a trace id" — which is *most* SRE work — index-on-read (Loki) is far cheaper and fits perfectly, because the labels you filter on are low-cardinality and the trace id you grep for is a fast scan once you've narrowed to one service. If you genuinely need fast, arbitrary, full-text queries across many high-cardinality fields all the time, index-everything (Elasticsearch) earns its cost. A critical Loki gotcha: **never put a high-cardinality field like `trace_id` in a label** — it explodes the index and recreates the cost you switched to Loki to avoid. Keep `trace_id` in the log body and grep it; keep labels low-cardinality.

### Cardinality is the other cost driver nobody mentions

The cost equation in §1 has a hidden fourth term that only bites in index-everything stores and in the *label* set of index-on-read stores: **cardinality** — the number of distinct values a field takes. A field like `level` has cardinality 5. A field like `service` might have cardinality 50. A field like `trace_id` has cardinality equal to your request count — millions per day. When a high-cardinality field becomes an index key (an Elasticsearch indexed field, or worse a Loki *label*), the index has to track every distinct value, and the index size and memory pressure explode super-linearly. A single team that adds `user_id` as a Loki label can take down a Loki cluster, because each unique label-value combination creates a separate *stream*, and millions of streams overwhelm the index.

The rule that follows is simple and worth posting on the wall: **low-cardinality fields are labels and index keys; high-cardinality fields are body content you scan.** `service`, `level`, `region`, `env`, `namespace` — low cardinality, fine as labels. `trace_id`, `user_id`, `order_id`, `session_id`, full URLs with query strings — high cardinality, keep them in the log *body* where they cost nothing to store and are found by a fast scan after you've narrowed by labels. This single discipline is the difference between a Loki bill that scales with your log *volume* (fine) and one that scales with your *request count* (ruinous). When someone proposes a new label, ask "how many distinct values?" If the answer is "one per request" or "one per user," it's body content, not a label.

### Govern the log schema so it stays cheap

At a certain scale, logging stops being a per-developer decision and becomes a shared contract that needs light governance — not bureaucracy, but a few agreed conventions that keep the corpus queryable and the bill predictable. The conventions that matter:

- **A shared baseline field set.** Every service emits `timestamp`, `level`, `service`, `version`, `trace_id` with the *same field names*. Nothing is more frustrating mid-incident than discovering one service calls it `trace_id`, another `traceId`, and a third `request_id` — your cross-service query now needs three OR clauses. Standardize the names once, ship them in a shared logging library, and every service inherits them.
- **A stable, enumerated set of event names.** The `message`/event-name field should come from a known vocabulary (`checkout_failed`, `cache_miss`, `db_query_slow`), not a free-text sentence. This keeps the field low-cardinality (so it can be a label or a cheap group-by) and makes alerts and dashboards stable across refactors.
- **A field deny-list, enforced at the agent.** The names that must never carry values — `password`, `token`, `secret`, `authorization`, `card_number`, `ssn` — are stripped centrally so no individual developer's mistake leaks PII. Defense in depth: redact at the source *and* at the agent.
- **A budget per service, surfaced as a metric.** Track each service's log volume as a Prometheus metric and alert when it jumps. The forty-thousand-dollar line would have been caught in a day if "order-service log bytes/sec" had had an alert on a 10× week-over-week jump. You monitor your error budget; monitor your *log* budget the same way.

None of this is heavy process. It's a shared library, a deny-list config, and one alert. The payoff is that your logs stay queryable as a single coherent corpus and your bill stays a number you chose rather than a number that surprised you.

### The one rule: PII and secrets must never be logged

There is exactly one absolute rule in logging, and it overrides cost, debuggability, and convenience: **personally identifiable information and secrets must never be logged.** No passwords, no auth tokens, no API keys, no full credit-card numbers, no government IDs, no raw request bodies that might contain any of those. Once a secret is in your log store, it's in your hot tier, your cold archive, your backups, your replicas, and your screen-shares — and rotating it becomes the only fix. Once PII is in your logs, you've created a compliance liability (GDPR, CCPA, PCI-DSS) and an attractive target.

The defense is redaction at the agent (as in the Vector config above) *plus* discipline at the source: never log a raw request payload (log the specific fields you need), and maintain a deny-list of field names (`password`, `token`, `secret`, `ssn`, `card_number`, `authorization`) that the logger or agent strips automatically. The opening story's \$40k line was *also* a PII incident waiting to happen — it logged full request payloads, which on a checkout endpoint means it was logging payment details. Dropping that payload fixed both the bill and a latent breach. That's not a coincidence: **the chatty payload-dumping line is usually both your biggest cost and your biggest privacy risk, and the same fix kills both.**

## 8. Querying: start from the trace id

All of this structure pays off at the moment you query. The single most powerful query you can run during an incident is "show me everything for this one request," and structured logs plus a trace id make it trivial.

In Loki's query language, LogQL, pulling one request's whole story across a service is:

```yaml
# Everything the orders service logged for one request, in order.
{service="orders"} | json | trace_id="abc123"
```

The `{service="orders"}` narrows by label (cheap, index-backed). The `| json` parses each line's structured fields. The `| trace_id="abc123"` filters to one request. Drop the service label and you get *every* service's view of that request:

```yaml
# Every service's logs for one request, fleet-wide.
{namespace="prod"} | json | trace_id="abc123"
```

That second query is the one that turns a multi-service mystery into a readable timeline. From there, LogQL lets you aggregate structured fields directly — for example, the error rate by route over five minutes, computed from logs:

```yaml
# Per-route error count over 5 minutes, straight from structured logs.
sum by (route) (
  count_over_time({service="orders"} | json | level="error" [5m])
)
```

Notice what structured logging bought you: that query groups by `route` and filters by `level` because those are *fields*, not substrings you had to regex out of a sentence. With string-concat logs, none of this is possible — you'd be back to grepping. The structure you added in §2, the trace id from §3, and the levels from §4 all converge here, in the query, which is where logs actually earn their keep.

A discipline worth adopting: **log the line you'll wish you had, then go run the query you'll wish you could.** When you write a log line, ask "when this fails at 3am, what will I filter by?" If the answer isn't a field on the line, add the field. The best test of a logging setup is whether, given a single trace id from a user complaint, you can reconstruct exactly what happened to that request — every decision, every input, every hop — in one query. If you can, your logs are doing their job. If you can't, no amount of volume will save you.

### Correlating logs with traces via OpenTelemetry

The trace id you've been propagating is not just a logging convenience — it's the same id your *tracing* system uses, and that's by design. Under the W3C Trace Context standard that OpenTelemetry implements, every request carries a `traceparent` header containing a 128-bit `trace_id` and a 64-bit `span_id`. If your logger stamps the *same* `trace_id` (and ideally `span_id`) onto every log line, then your logs and your distributed traces share a join key, and your observability tool can jump from a slow span in a trace straight to the log lines emitted *during* that span. That round-trip — "this span took 250ms, what was the code doing?" → the DEBUG logs from exactly that window — is the single most powerful debugging motion in a distributed system, and it's free once the ids line up.

The OpenTelemetry idiom is to pull the active span's context and inject its ids into the log record:

```python
from opentelemetry import trace

span = trace.get_current_span()
ctx = span.get_span_context()
log.info("db_query_slow",
         table="orders",
         duration_ms=312,
         # 032x / 016x render the ids in the hex form traces use.
         trace_id=format(ctx.trace_id, "032x"),
         span_id=format(ctx.span_id, "016x"))
```

Now a query like `{service="orders"} | json | trace_id="<id from the slow span>"` returns the exact log lines for the exact request the trace flagged. This is why the [distributed tracing sibling post](/blog/software-development/site-reliability-engineering/distributed-tracing-in-practice) and this one are two views of one id: the trace shows you *where* the time went across services; the logs, joined on the same id, show you *why* the code did what it did at each hop. Logs without trace ids are islands; logs with them are a searchable, cross-pillar graph of every request.

## 9. War story: the cascading log flood

Let me tell you about a class of incident that has bitten enough teams to be a genre: **the log flood that became the outage.**

The setup is always similar. A service depends on a downstream — a database, a payment provider, an auth service. The downstream gets slow or starts erroring. The dependent service, doing the "right" thing, logs every failure at ERROR with full context. Under normal load that's a trickle. But the downstream is failing *every* request now, so the trickle becomes a firehose: at ten thousand requests a second, ten thousand identical ERROR lines a second, each serialized to JSON, each written, each shipped over the network to the log store.

Now three things happen at once, and they compound. First, the *act of logging* — serialization, disk I/O, network — consumes CPU and I/O that the service needs to serve requests, so the service itself slows down, which makes the incident worse. Second, the log pipeline gets overwhelmed: the collector agent's buffers fill, back-pressure builds, and either the agent starts dropping lines (you lose data exactly when you need it) or it blocks the app (worse). Third, and most insidious: the log *store* is now so flooded with the symptom — the millions of identical "downstream failed" lines — that searching for the *cause* is impossible. The signal is buried under its own echo. The thing meant to help you debug the outage has become a second outage on top of the first.

This is a real, recurring pattern. Versions of it have shown up in public postmortems across the industry — a dependency degrades, the retry-and-log behavior amplifies the load, and the logging subsystem itself becomes a bottleneck or a cost spike. The reliability literature, including the Google SRE Book's treatment of cascading failures and overload, describes exactly this dynamic where a well-intentioned behavior under stress amplifies rather than dampens the problem. (The cross-asset architectural version of this — circuit breakers and bulkheads to stop the cascade at the source — is covered in the system-design series on [cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads); here we're covering the logging half of the blast.)

The fixes are the ones we've already built, now seen as *load-shedding for logs*:

- **Rate-limit hot-path error logs** to one representative line per second per error key (§5). You lose nothing — the lines were identical — and you cap the self-inflicted load.
- **Move counts to metrics.** "How many downstream failures" is a counter, not ten thousand log lines. A Prometheus counter incrementing ten thousand times a second costs nothing; ten thousand log lines a second costs everything.
- **Give the collector agent bounded buffers with a drop-oldest or drop-newest policy** so it sheds load gracefully instead of blocking the app or filling memory. Logging should *never* be able to take down the service it's observing.
- **Sample even errors when they're identical and high-volume.** "Always keep errors" means keep every *distinct* error; ten thousand byte-identical copies of one error is not ten thousand signals, it's one signal logged ten thousand times. A "log-once per error-key per window" policy keeps the signal and kills the flood.

The meta-lesson, and it's the one to carry: **your logging system must fail safe.** It must never be able to amplify an incident or take down the service it observes. Design it so that when everything is on fire, the logs help you and don't pour fuel on the blaze. For the broader discipline of debugging production from the artifacts logs leave behind, the debugging series goes deep in [logging as a debugging instrument](/blog/software-development/debugging/logging-as-a-debugging-instrument).

There's a darker variant of this story worth naming because it surprises people: **the log flood that didn't take down the service but did blow the budget and trigger a second incident a week later — the invoice.** A noisy deploy ships on a Friday, logs ten times its usual volume from a new over-eager INFO line, nobody notices because the service is *fine*, and the only symptom is a bill that arrives two weeks later showing a spike that already happened and can't be undone. This is why a *log-volume alert per service* (the last bullet of the schema-governance section) is not optional at scale. You alert on error budget burn because you can't afford to discover an SLO breach after the fact; treat log spend the same way. A 10× week-over-week jump in a service's log bytes is almost always either a bug, a misplaced DEBUG, or a payload being dumped — all three are worth a page-free, business-hours ticket the day they start, not a finance escalation a fortnight later.

One more defensive pattern from the field: **separate your logging pipeline's failure domain from your application's.** If your app writes logs to stdout and a node-level agent ships them, then a log-store outage means the agent buffers (or drops oldest) and the app keeps serving — the blast radius of "log store is down" is "we temporarily can't search recent logs," not "the site is down." Contrast that with an app that writes synchronously to a remote log endpoint on the request path: now a slow log store *adds latency to every request*, and a down log store can hang the app. The rule is that **logging must be asynchronous and best-effort from the application's point of view.** Emit to a local buffer, let the agent deal with shipping, and never let the act of logging block the act of serving. This is the same fail-safe principle as load-shedding, applied to your own observability plumbing.

## 10. Worked incident: DEBUG for ten minutes, no redeploy

![Before and after comparison of a logging bill, the before column showing DEBUG on the hot path with full payloads at fifty gigabytes a day costing forty five thousand a month, the after column showing sampled info, dropped payloads, and nine gigabytes a day costing nine thousand a month](/imgs/blogs/logging-at-scale-without-going-broke-8.png)

Let me close the practice with the second worked example the whole post has been building toward: an incident solved *because* you could bump the log level dynamically.

#### Worked example: the intermittent 503

The symptom: about one in five hundred checkout requests returns a 503, intermittently, on the order service. Your metrics see it — there's a small but real bump in the 5xx rate, p99 latency is occasionally spiking — but metrics can't tell you *why*. The error rate is too low to reproduce locally and too intermittent to catch by staring. The INFO logs show the failed requests but not the decision path that led to the 503, because the detail that would explain it is at DEBUG, which is off in prod (correctly — leaving it on is the \$40k line).

Here's the playbook, and it takes about twelve minutes:

1. **Grab a failing trace id.** From the metrics-linked logs or a user report, you get one trace id of a request that 503'd: `abc123`. (This is why every line carries a trace id — even your INFO lines let you anchor on one failed request.)

2. **Bump that one service to DEBUG, with auto-revert.** You hit the authenticated admin endpoint from §4:
   ```bash
   curl -XPOST -H "Authorization: Bearer $OPS_TOKEN" \
     https://orders.internal:9000/loglevel -d 'level=debug'
   ```
   Now the order service is logging the full decision-and-inputs detail — but *only* the order service, and *only* for the next ten minutes, after which it auto-reverts to INFO. The bill barely notices ten minutes of DEBUG on one service. There was no redeploy, no rollout, no config-push-and-wait. You flipped a flag and the detail appeared.

3. **Wait for the bug to recur and grep the trace id.** Within a few minutes another request 503s. You pull its full DEBUG story:
   ```yaml
   {service="orders"} | json | level=~"debug|error" | trace_id="def456"
   ```
   And there it is, in the DEBUG lines: the service was calling the inventory service with a 200ms timeout, and under a specific cache-cold condition inventory was taking 250ms, so the call timed out, the order service retried once, the retry *also* timed out, and it gave up with a 503. The INFO logs showed "checkout failed, 503." The DEBUG logs showed *the timeout, the retry, and the exact latency* — the why.

4. **Let it auto-revert.** Ten minutes pass, the level flips back to INFO on its own, the bill is unaffected, and you have your root cause: the inventory timeout is too tight for the cache-cold path. The fix is a longer timeout plus a cache warm-up, which is a code change you make calmly, not under fire.

Tally the proof: **MTTR for this class of intermittent bug dropped from "we never figured it out, it just kept happening" to about twelve minutes** — and the cost of getting there was ten minutes of DEBUG on one service, effectively zero on the bill. *That* is the payoff of the whole design. Structured logs gave you queryable fields. The trace id let you pull one request's story. Levels kept DEBUG affordable by keeping it off. The dynamic flag let you turn it on for exactly the slice you needed. And the auto-revert kept it from becoming next month's surprise. Every piece earned its place.

Contrast this with the alternative world: no dynamic level, so to get DEBUG you'd add a log line and *redeploy* — minutes to build, minutes to roll out, and you'd have to leave it on (because you can't predict when the bug recurs), which means days of DEBUG on a hot path and a five-figure bill spike to debug a bug. The dynamic flag turns a "redeploy and pray and overpay" into a "flip, grep, revert." That's the difference between a logging system designed for incidents and one that just accumulates lines.

## 11. How to reach for this (and when not to)

Every technique here has a cost, and the senior move is knowing when *not* to apply one. Here's the honest guidance.

**Always do these — they're nearly free and pay off immediately:**

- **Structured logging.** There is essentially no reason to log string-concat in a new service. The per-line overhead is tiny and the queryability is the whole value of logs. If you do one thing from this post, structure your logs.
- **Trace id propagation.** In any system with more than one service, this is non-negotiable. It's the join key for everything. The cost is a header and a baseline field; the value is the difference between debugging a service and debugging a request.
- **Always keep errors.** Whatever else you sample, never sample out an ERROR. The cost of keeping all errors is small (errors should be rare if your levels are disciplined); the cost of dropping the one you needed is an unsolvable incident.

**Do these when volume justifies the complexity:**

- **Sampling.** If your log bill is small and your volume is modest, don't add sampling complexity — you'll spend more engineering time than you save. Sampling earns its keep when you have genuinely high-volume, repetitive success traffic. Below a few terabytes a month, the simpler win is usually just *not logging the payload* and *tiering retention*; reach for sampling when the boring-success volume is the dominant line item.
- **Tail sampling.** It's more complex and needs buffering. Use it when you specifically need every error and every latency outlier from a high-volume service. For most teams, head-sampling successes plus always-keeping errors is enough; add tail sampling when you've outgrown that.
- **A second hot store / multi-tier indexing.** Only when your access patterns genuinely diverge. Don't build a three-tier pipeline for a service that logs a gigabyte a month — you'll pay more in operational complexity than you'll ever save. Tiering pays off at scale.

**Don't do these:**

- **Don't log at DEBUG in production by default.** This is the single most common way to bankrupt a log bill. DEBUG is for dynamic, time-boxed enablement, not steady state.
- **Don't log on a hot path without rate-limiting.** A hot-path log without a limiter is a latent self-DoS waiting for a dependency outage to trigger it.
- **Don't log PII or secrets, ever — no exceptions, regardless of cost or convenience.** This one overrides everything.
- **Don't put high-cardinality fields in your index labels** (the Loki `trace_id`-in-label trap). It recreates the cost you were avoiding.
- **Don't keep everything hot for thirty days "to be safe."** That's how the bill balloons. Hot for a week, cold for as long as you need; cold is nearly free.

The unifying principle: **logging is a budget you spend on debuggability, and like any budget you spend it where it pays.** Spend lavishly on errors and decision-context (they're rare and invaluable). Spend frugally on repetitive successes (they're common and boring). Spend almost nothing on old data (it's queried almost never). Get that allocation right and you get *better* debuggability for *less* money — which is the only kind of cost-cutting worth doing.

## 12. Key takeaways

- **Cost = volume × retention × index factor.** Memorize it. Every overspend traces to one of those three multipliers, and a chatty line on a hot path inflates all three at once.
- **Structure your logs.** Log the *decision and the inputs* (which branch, what values), not "done." Named fields are queryable and aggregatable; sentences are not. This is the cheapest, highest-leverage move.
- **Propagate a trace id through every hop.** It's the single most valuable distributed move — it turns an O(fleet) search into an O(1) lookup and joins your logs to your traces.
- **Levels are a cost-and-attention contract.** ERROR is page-worthy and 100% kept; INFO is sampled; DEBUG is off in prod by default and bumped dynamically during incidents; TRACE is almost never on. Keep the contract and INFO stays useful.
- **Bump DEBUG dynamically, with auth and auto-revert.** Get incident-grade detail for ten minutes on one service without a redeploy — then let it flip back so it never becomes a bill surprise.
- **Sample the boring, keep every failure.** Head-sample successes, tail-sample to rescue errors and outliers, rate-limit hot paths, and *always keep 100% of errors*. Counts belong in metrics, not in a log line per occurrence.
- **Tier retention hot → warm → cold.** Recent logs hot and fast, old logs cheap and slow. Cold object storage is ~100× cheaper; default to aging down, not deleting.
- **Your logging system must fail safe.** It must never amplify an incident or take down the service it observes. Rate-limit, bound the buffers, shed load.
- **Never log PII or secrets.** Redact at the agent and at the source. The payload-dumping line is usually both your biggest cost and your biggest privacy risk — and one fix kills both.

## 13. Further reading

- [Reliability is a feature: the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — the series intro and the define→measure→budget→respond→learn→engineer loop this post fits into.
- [Metrics, logs, and traces: when to use which](/blog/software-development/site-reliability-engineering/metrics-logs-and-traces-when-to-use-which) — the sibling that draws the boundary between the three pillars (and tells you when a count belongs in a metric, not a log).
- [Distributed tracing in practice](/blog/software-development/site-reliability-engineering/distributed-tracing-in-practice) — the other half of the trace-id idea; the same id that joins your log lines joins your spans.
- [Logging as a debugging instrument](/blog/software-development/debugging/logging-as-a-debugging-instrument) — the debugging series' deep dive on reading production from the artifacts logs leave behind.
- [Cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads) — the architecture-time defense against the log-flood-as-outage pattern from §9.
- **The Google SRE Book and SRE Workbook** — the canonical treatment of observability, overload, and cascading failures; the source for the "logging must fail safe" discipline.
- **Grafana Loki documentation** — the index-on-read model, LogQL, retention, and per-stream overrides used throughout §6–§8.
- **OpenTelemetry documentation** — context propagation (W3C Trace Context / `traceparent`), the Collector, and the spec that makes the trace id portable across services and vendors.
