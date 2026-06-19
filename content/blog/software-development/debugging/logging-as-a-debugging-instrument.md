---
title: "Logging as a Debugging Instrument: The Log Line You Wish Past-You Had Written"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Learn to instrument code so that when a bug strikes in prod and you cannot attach a debugger, the structured log line you wrote months ago lets you replay one request across a fleet and find the root cause in five minutes instead of a week."
tags:
  [
    "debugging",
    "software-engineering",
    "logging",
    "structured-logging",
    "correlation-id",
    "observability",
    "distributed-systems",
    "sampling",
    "troubleshooting",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/logging-as-a-debugging-instrument-1.png"
---

At 02:14 on a Tuesday, a pager goes off. One of your customers, a payments tenant in another timezone, got a 500 on a checkout that should have gone through. By the time you are awake and at your laptop, the request is long over. The process that served it has handled forty thousand requests since. There is no broken state sitting in memory waiting for you to inspect it, no stack frame paused at a breakpoint, no live thread you can step through. The bug already happened. You cannot reproduce it, because you do not know what made it happen — that customer's cart, that customer's account flags, the exact concurrency on that node at that instant. You cannot attach `gdb` or `pdb`, because there is nothing to attach to. The only thing you have, the *only* thing, is whatever your past self decided to write to a log file before the incident.

This is the single most common debugging situation in production software, and it is the one where the techniques from the rest of this series — breakpoints, watchpoints, sanitizers, record-replay — simply do not apply. You cannot set a watchpoint on a request that finished. You cannot single-step a bug in a service running in three availability zones behind a load balancer. The interactive debugger is a microscope, and it is a wonderful microscope, but you can only point it at a slide that is in front of you *right now*. Production bugs that already happened, that are intermittent, that live in the interaction between six services, that show up once a day under a load you cannot recreate on your laptop — for all of those, logs are not a poor substitute for a debugger. They are the *only* instrument that works, because they are a recording. They are a time machine. The figure below is the whole thesis: a bug that already happened is unreachable by any live tool, but if you instrumented well, you can grep your way back to the exact state of one past request and read off the root cause.

![A vertical stack showing how a bug that already happened in production is unreachable by a live debugger but recoverable by grepping structured logs for one correlation id and replaying the request path](/imgs/blogs/logging-as-a-debugging-instrument-1.png)

The catch — and it is the whole game — is that a time machine only takes you to moments you recorded. You cannot go back and read a field you never logged. The log line that would have cracked this bug in thirty seconds is the one past-you didn't write, because past-you didn't think this code path could fail, or logged `"processing complete"` instead of `"charged tenant=t9 amount=4200 idempotency_key=abc result=ok latency_ms=812"`. So this post is really about a discipline you practice *before* the incident: instrumenting your code so that future-you, paged at 2am with no repro, has the recording they need. By the end you will be able to choose what belongs at each log level and why DEBUG should ship to prod behind a flag; write structured logs whose fields make a line queryable instead of a sentence you can only grep blindly; propagate a correlation id across threads, async boundaries, and service hops so you can reconstruct one request's path through a fleet; use logs to binary-search where a value goes wrong; and sample so your logs neither bankrupt you nor lie to you. This is the **observe** stage of the series loop — observe, reproduce, hypothesize, bisect, fix, prevent — done in the one regime where you observe through a recording rather than a live process. If you have not read it, [the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) frames that loop, and this post is its production-grade observation layer.

## 1. Why a debugger does not help here: the mechanism of "the bug already happened"

To understand why logging is the load-bearing instrument in production, you have to understand precisely why the interactive tools fail, because the failure is not a matter of taste or convenience. It is structural. A debugger works by stopping a *running process* and reading its memory. Its entire power comes from the fact that the state you care about — the value of a variable, the contents of the heap, the call stack — is physically present in the address space of a process that exists right now and that you have permission to halt. Every technique a debugger gives you is a way to observe or perturb that live state: a breakpoint pauses the thread when execution reaches a line, a watchpoint pauses it when a memory location changes, single-stepping advances one instruction and lets you re-read memory in between.

Now consider the prod 500 from the intro. The state you need — the cart contents, the account flags, the value of `amount` after the rounding step, the response from the payment gateway — existed for a few hundred milliseconds inside one request handler, in the memory of one process, on one node, at 02:14. Then the request returned, the handler's stack frame unwound, the local variables went out of scope, the garbage collector or the allocator reclaimed them, and the process moved on to the next request. By 02:15 that memory has been overwritten thousands of times. There is no address you can point a debugger at, because the bytes are gone. The information has been *destroyed by the normal forward progress of the program*. This is the fundamental asymmetry of production debugging: the bug is an event in the past, and the past is not in memory.

It gets worse in three independent ways, each of which the rest of this post addresses. First, **distribution**: in a microservice system, "the state of the request" is not even in one process. It is spread across the gateway, the auth service, the order service, the payment service, and two databases, each on a different host. No single debugger can see all of that; the request's true state never existed in one place to be inspected. Second, **intermittency**: a bug that happens once a day, or once per million requests, or only when two requests interleave on the same row, cannot be caught by sitting at a breakpoint, because you would have to sit there for a day, or a million requests, and the act of sitting at the breakpoint changes the timing that produced the bug. Third, **you often cannot attach at all**: attaching a debugger to a process freezes it. Freezing the payments process in prod for thirty seconds while you poke around is, itself, an outage. The on-call runbook quite reasonably forbids it.

So the live tools are out, and what is left is the recording. A log is an append-only record that the program writes *as it runs*, capturing chosen facts at chosen moments and persisting them outside the process, in a file or a log pipeline, where they survive the death of the request and the death of the process. The log is the program telling you, in its own words, what it was doing — but only at the moments, and about the facts, that you told it to. This reframes the entire activity. You are not debugging the running program; you are an archaeologist reading the records a past program left behind. And archaeology is only as good as the records. The discipline of logging is the discipline of being a good record-keeper for a future self who will have nothing else.

There is one more mechanism worth naming because it bites people who try to "just add more logging" reactively: **observer effect on timing**. Logging is not free. Writing a log line does work — formatting the message, acquiring a lock on the output stream, a system call to write bytes. If you add verbose logging inside a tight loop or a hot concurrency path *to chase a bug*, the logging itself slows that path down, and a bug that depended on a narrow timing window can vanish — the classic **heisenbug**, a bug that changes or disappears when you observe it (named after Heisenberg's uncertainty principle, where measuring a system disturbs it). A race condition that fired when two operations completed within microseconds of each other will stop firing if you drop a 50-microsecond log write into one of them. This is why logging volume and placement are debugging decisions, not just hygiene, and it is why the sampling section later is not an optimization afterthought but a correctness concern: log the wrong amount in the wrong place and your instrument lies to you about the very timing you are trying to measure.

## 2. Levels, done right: a contract about audience and volume

Log levels are the oldest idea in logging and the most consistently misused. Most codebases treat them as a vague intensity dial — ERROR is "really bad," WARN is "kinda bad," INFO is "normal," DEBUG is "chatty" — and the result is that ERROR is full of things that are not errors, INFO is so noisy nobody reads it, and DEBUG is either off everywhere or drowning everything. The fix is to stop thinking of levels as intensity and start thinking of them as a **contract about two things: who the audience is, and how much volume you are committing to.** Each level is a promise. Get the promises right and your logs become a layered instrument you can dial up and down at exactly the resolution you need.

![A matrix mapping the five log levels to what content belongs at each one and the production default for whether it is on or off](/imgs/blogs/logging-as-a-debugging-instrument-4.png)

Here is the contract I hold each level to, from the top down. **ERROR** means: something broke, the consequence is real, and a human will eventually need to act — a request failed and the user got an error, a write was lost, a downstream dependency is unreachable in a way you cannot recover from. ERROR is the level your alerting watches. The audience is on-call. The volume commitment is that ERROR should be *rare enough that every ERROR line is worth a human glance*; if you are logging ten thousand ERRORs a minute under normal operation, ERROR has lost its meaning and your alerts are noise. The most common abuse is logging an ERROR for something the code then *successfully retries* — that is not an error, that is a transient blip the system handled, and it belongs at WARN.

**WARN** means: something was not right, but the system coped — a retry succeeded on the second attempt, a request fell back to a degraded path, a deprecated code path was hit, a quota is at 90%, a cache miss forced a slow path. The audience is a human reviewing health, not a pager. The signal of WARN is *trend*: one WARN is fine, but a WARN rate that climbs is the early warning that an ERROR is coming. WARN is where you put the things that are fine in ones but alarming in thousands.

**INFO** means: the program did its normal job, here is the audit trail. The discipline that pays off enormously is **one INFO line per unit of work, written at completion, that records the result and the key facts**: this request finished, here is its id, its outcome, its latency, the tenant, the few inputs that determine behavior. INFO is the level that should *always be on in production* and should be *cheap enough to always be on* — roughly one line per request, not one per function call. When INFO is disciplined like this, it becomes the spine of every investigation: you grep INFO for the request, see its summary, and decide whether you need to go deeper.

**DEBUG** means: the decision and its inputs, for when INFO's one-line summary is not enough. DEBUG is where you log *why* the code chose what it chose — "selected pricing tier B because tenant flag premium=true and region=EU," "skipped cache because key version mismatch 3 vs 4," "computed discount=0 because coupon expired_at < now." DEBUG is verbose, often several lines per request, and **it should be off in prod by default but shippable to prod behind a flag** (more on that in the next section). The reason DEBUG belongs in prod-behind-a-flag and not just on your laptop is the entire premise of this post: the bug only reproduces in prod, so the DEBUG lines you need are the ones that fire *in prod*, on the real request, with the real data — which you can only get if the DEBUG instrumentation is already deployed and you can turn it on for that code path without a redeploy.

**TRACE** means: every step, the firehose — each iteration of a loop, each item processed, each intermediate value. TRACE is for narrow, deliberate, short-lived investigation: you turn it on for one endpoint, on one node, for two minutes, capture what you need, and turn it off. It is almost never on in prod broadly because the volume would be ruinous and, per the heisenbug mechanism above, the volume can change timing. The distinction between DEBUG and TRACE is volume and granularity: DEBUG is "the decisions," TRACE is "every step that led to each decision."

The level you choose is a routing decision. In an incident you say "show me ERROR and WARN across the fleet for the last hour" to triage, then "show me INFO for request a91" to find the request, then "give me DEBUG for that one request" to see the decisions. Levels let you control the resolution of your recording without changing code. That only works if each level keeps its promise.

| Level | Audience | Volume budget | Belongs here | Common abuse |
|-------|----------|---------------|--------------|--------------|
| ERROR | On-call / alerting | Rare; every line worth a glance | Real failure a human must act on | Logging retried/recovered events |
| WARN | Health review | Low; watch the trend | Degraded-but-coped; early warning | Treating it as a softer ERROR |
| INFO | Auditing / triage | ~1 line per unit of work | Result + key facts of each request | One line per function call |
| DEBUG | The investigator | High; off by default | Decisions and their inputs | Leaving it on globally in prod |
| TRACE | Deep dive | Firehose; short bursts only | Every step and intermediate value | Ever leaving it on broadly |

#### Worked example: the ERROR that was not an error

A team I worked with had ERROR alerts that fired roughly two thousand times an hour, every hour, forever. On-call had long since muted them, which meant that when a *real* ERROR storm started — a downstream dependency went hard-down — nobody noticed for forty minutes, because the alert channel was already a wall of noise they had trained themselves to ignore. We grepped a day of ERROR lines and bucketed them by message. Ninety-four percent were a single message: `ERROR: failed to fetch user prefs, using defaults`. That is not an error. The code asked a flaky prefs service for a user's preferences, the call timed out, and the code did exactly the right thing — it used sensible defaults and served the request successfully. The user never saw a problem. We moved that line from ERROR to WARN, added a counter so we could still watch the *rate*, and the ERROR channel dropped from two thousand an hour to about thirty — thirty real failures that a human should look at. The next time a dependency went down, the ERROR rate jumped from thirty to nine hundred in two minutes and the page fired immediately. Nothing about the system's behavior changed; we just made the instrument tell the truth, and the measured result was a real incident's detection time going from forty minutes to under two.

## 3. Turning DEBUG on in prod without a redeploy: dynamic level bumping

The premise that DEBUG should ship to prod but stay off has an obvious implication: you need a way to turn it *on* for a running process, or better, for a specific code path or even a specific request, without shipping new code. A redeploy to flip a log level is unacceptable for two reasons. It is slow — by the time the deploy rolls out, the intermittent bug may not recur for hours. And it is dangerous — restarting processes to pick up a config change can itself clear the very condition you are chasing (a leak resets, a stuck connection drops, the heisenbug evaporates). So mature systems make log level a piece of *runtime configuration* you can change live.

There are several mechanisms, from crude to surgical. The crudest is a signal or admin endpoint that bumps the global level: send the process a `SIGUSR2` or hit an internal `/admin/loglevel?level=DEBUG` endpoint, and the logger's threshold drops. This works but it is a blunt instrument — now *every* request on that node is at DEBUG, which in prod can be a volume flood and, again, a timing perturbation. Better is **per-logger level control**: most logging frameworks let you set the level on a named logger (e.g. `com.acme.payments`) independently, so you can raise just the payments module to DEBUG and leave everything else at INFO. Better still is **dynamic, per-request enabling**: a feature-flag or config service (or a special request header like `X-Debug: 1` that only internal callers may set) flips DEBUG on for the requests you care about and leaves the rest at INFO. This is the surgical version — you reproduce the bug by hitting the endpoint yourself with the debug header, get full DEBUG output for *just your* request, and the other ten thousand requests per second are unaffected in volume and in timing.

Here is the runtime-level-change pattern in Python's standard `logging`, exposed through a tiny admin handler. The key idea is that the level lives in a mutable place you can poke at runtime, not baked into code:

```python
import logging
import signal

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("payments")

def _bump_to_debug(signum, frame):
    # On SIGUSR2, drop the payments logger to DEBUG for live diagnosis.
    log.setLevel(logging.DEBUG)
    log.warning("log level bumped to DEBUG via signal")

def _reset_to_info(signum, frame):
    log.setLevel(logging.INFO)
    log.warning("log level reset to INFO")

signal.signal(signal.SIGUSR2, _bump_to_debug)
signal.signal(signal.SIGHUP, _reset_to_info)
```

Now `kill -USR2 <pid>` raises the payments logger to DEBUG on a live process and `kill -HUP <pid>` puts it back, no redeploy, no restart. In a real system you would gate the per-request version behind your config service. In Go's `slog`, the same idea uses a `LevelVar`, which is explicitly designed to be changed at runtime and is safe for concurrent use:

```go
package main

import (
	"log/slog"
	"net/http"
	"os"
)

var lvl = new(slog.LevelVar) // starts at LevelInfo

func main() {
	lvl.Set(slog.LevelInfo)
	h := slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: lvl})
	slog.SetDefault(slog.New(h))

	// Internal-only endpoint to change level live, e.g. /admin/level?v=debug
	http.HandleFunc("/admin/level", func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Query().Get("v") {
		case "debug":
			lvl.Set(slog.LevelDebug)
		case "info":
			lvl.Set(slog.LevelInfo)
		}
		w.WriteHeader(http.StatusNoContent)
	})
	http.ListenAndServe(":9090", nil)
}
```

The discipline this buys you is enormous. You ship DEBUG instrumentation everywhere, paying nothing in prod volume because it is off, and when an intermittent bug shows its face you turn DEBUG on for exactly the slice you need and let the next occurrence record itself in full detail. The instrument was already installed; you just powered it up. This is the difference between "we'll add some logging and wait for it to happen again after the next deploy" (days) and "turn on DEBUG for that tenant, reproduce, read the answer" (minutes).

## 4. Structured logging: log the decision and the inputs, not a sentence

Everything above assumes you can *query* your logs — "show me ERROR for tenant t9," "give me the request where amount went to zero," "count the WARN rate over time." You cannot do any of that reliably with logs that are English sentences. A line like `log.info("Processed order for user " + user + " in " + ms + "ms")` looks fine to a human and is nearly useless to a machine. To find all orders for one user you grep for a substring and pray no other line happens to contain it. To filter by latency you cannot, because the number is embedded in prose. To group by outcome you cannot, because the outcome is not a field, it is a phrase. String-concatenated logs are write-only: easy to emit, impossible to interrogate at scale.

**Structured logging** fixes this by making every log line a set of typed key-value pairs (rendered as `key=value` pairs or, more commonly in modern stacks, as a JSON object) instead of a sentence. The same line becomes `log.info("order processed", user_id="u123", order_id="o456", latency_ms=812, result="ok", tenant="t9")`. Now every fact is a named, typed field. Your log backend indexes those fields, and at incident time you write queries: `result:"fail" AND tenant:"t9" AND latency_ms:>1000`. The line stopped being prose you read and became a row you query. The figure below is the contrast that matters most in this whole post.

![A two column before and after contrasting a string concatenated log sentence that cannot be filtered against a structured key value log line whose fields can be queried directly](/imgs/blogs/logging-as-a-debugging-instrument-2.png)

But structured logging is not just about format. The deeper discipline is *what you put in the fields*, and the rule is: **log the decision and the inputs that drove it, not the fact that something happened.** A line that says `"done"` tells you nothing when `"done"` was wrong. A line that says `tier=B because premium=true region=EU coupon=expired` tells you, six weeks later when tier B turns out to be the bug, exactly which inputs produced it — and lets you grep for *other* requests with the same inputs to see how widespread it is. When you write a log line, ask: if this line is the only thing I have at 2am, can I reconstruct what the code did and why? That means logging the *actual values* — the computed amount, the chosen branch, the key that missed the cache — not a summary that has already thrown the values away.

There is a standard set of fields that turn a log line from "an event happened" into "a debuggable record," and you should put them on essentially every line (the contextual-logger pattern in section 6 makes this automatic):

| Field | Why it is load-bearing |
|-------|------------------------|
| `request_id` / `trace_id` | Stitch every line of one request together; the single most valuable field |
| `user_id` / `tenant_id` | Filter to the affected customer; spot whether a bug is one tenant or all |
| `service` / `host` / `version` | Which build and which box; isolate a bad deploy or a bad node |
| `level`, `timestamp` | Routing and ordering; timestamps in UTC, ideally with monotonic source |
| The actual values | The amount, the branch taken, the input that drove the decision |
| `latency_ms` / `duration` | Turn logs into a latency dataset; find the slow path without a profiler |
| `error.type` / `error.stack` | The exception class and the full stack, never just a message |

Here is the same structured INFO line in three languages, so the shape is concrete. Python with `structlog`, which renders either key-value or JSON and lets you bind context:

```python
import structlog

log = structlog.get_logger()

def process_order(order, ctx):
    result = charge(order)
    log.info(
        "order_processed",
        order_id=order.id,
        tenant_id=ctx.tenant,
        amount_cents=order.amount_cents,
        result=result.status,        # "ok" | "declined" | "error"
        gateway_latency_ms=result.latency_ms,
        idempotency_key=order.idem_key,
    )
    return result
```

Go with the standard library's `slog`, emitting JSON with typed attributes:

```go
slog.Info("order_processed",
	"order_id", order.ID,
	"tenant_id", ctx.Tenant,
	"amount_cents", order.AmountCents,
	"result", result.Status,
	"gateway_latency_ms", result.LatencyMs,
	"idempotency_key", order.IdemKey,
)
// => {"time":"...","level":"INFO","msg":"order_processed",
//     "order_id":"o456","tenant_id":"t9","amount_cents":4200,
//     "result":"ok","gateway_latency_ms":812,"idempotency_key":"abc"}
```

Node.js with `pino`, which is JSON-first and very fast:

```js
const logger = require('pino')();

function processOrder(order, ctx) {
  const result = charge(order);
  logger.info({
    order_id: order.id,
    tenant_id: ctx.tenant,
    amount_cents: order.amountCents,
    result: result.status,
    gateway_latency_ms: result.latencyMs,
    idempotency_key: order.idemKey,
  }, 'order_processed');
  return result;
}
```

Notice three things these share. The message (`"order_processed"`) is a short, stable, *constant* event name you can group by — not an interpolated sentence, so all order-processed lines share one `msg` and you can count them. Every variable fact is a separate field. And the result is JSON a log backend ingests and indexes without parsing English. That last property is what makes the worked examples later possible: you can ask the haystack questions, and it answers.

## 5. Correlation IDs: reconstructing one request across a fleet

If you adopt exactly one idea from this post, make it this one, because it is the single highest-leverage move in distributed debugging. In a system of many services, the central problem at incident time is that *one user action becomes dozens of log lines scattered across dozens of processes on dozens of hosts*, and by default there is nothing tying them together. The user clicked "pay." That produced a line in the gateway, three in the order service, two in the inventory service, four in the payment service, and a slow-query log in a database — eleven lines on five machines, interleaved with millions of other requests' lines, with no thread of connection between them. You cannot debug what you cannot reassemble.

A **correlation id** (also called a request id, trace id, or in the OpenTelemetry standard a trace context) is the thread that reassembles them. The idea is simple: when a request enters the system at the edge — the gateway, the load balancer, the first service — you mint a unique id for it (a UUID, or accept one from an inbound header if the caller already set it). Then you **propagate that id through every hop**: every internal HTTP call carries it in a header (`traceparent` per the W3C Trace Context standard, or a custom `X-Request-Id`), every message you put on a queue carries it in metadata, every log line in every service includes it as a field. Now, at incident time, you take the one id and run a single query across *all* services' logs: `trace_id:"a91f..."`. Out comes every line from every service for that one request, in time order, and you can read the request's entire journey through the fleet as one story.

![A branching graph of a single request propagating one correlation id through six services where the payment service is the one hop that logged a thirty second timeout](/imgs/blogs/logging-as-a-debugging-instrument-3.png)

The figure shows the payoff. One request, id `a91`, minted at the gateway. Auth logged it, ok in 4ms. The order service logged it and fanned out to inventory (ok, 7ms) and payment. The payment service logged it — and logged a 30-second timeout waiting on the payments database, which was in a lock wait. Without the correlation id, you would see, somewhere in the payment service's millions of lines, *a* timeout, but you would have no way to know it belonged to *this* user's 500, and no way to see that auth and inventory were both fine so the problem is isolated to the payment hop. With the id, the query returns five lines, they line up in order, and the one that says `timeout` next to `payments DB lock wait` is your answer. You went from "a 500 happened somewhere in a fleet" to "the payment service's DB call timed out for this request" in one query.

The mechanism that makes correlation ids *hard* — and the reason this is a discipline and not a one-liner — is **propagation across boundaries where context is not automatic**. There are three boundaries that drop context if you are not careful:

**Across services.** This is the easy one: pass the id in a header on every outbound call and read it from the header on every inbound call. The discipline is doing it on *every* call, including the ones you forget — the background job, the retry, the call to the third-party API (so its support team can find the request too). A single service that fails to forward the id breaks the chain; the request's story has a gap exactly where that service is.

**Across async boundaries within a process.** When a request handler kicks off async work — an `await`, a callback, a task posted to an executor, a goroutine — the "current request" context can be lost, because the new execution unit does not automatically inherit the variables of the one that spawned it. The async stack unwound at the `await`; the goroutine started fresh. Languages provide context-propagation mechanisms exactly for this: Go's `context.Context`, which you pass explicitly down every call so the id rides along; Python's `contextvars`, which propagate across `await` boundaries within a task; Node's `AsyncLocalStorage`, which carries a store across the event loop's callback boundaries. Use them, or your async work logs lines with no id and the chain breaks at every concurrency hop.

**Across threads.** Classic thread-per-request servers stash the id in thread-local storage (the MDC, "Mapped Diagnostic Context," in the Java/SLF4J world). The trap is a thread *pool*: when you hand work to a pool, the worker thread has whatever thread-local the *previous* task left behind, which is either empty or — worse — *another request's id*, so your line gets stamped with the wrong correlation id and you stitch two requests together by accident. The fix is to capture the id at submission time and re-establish it inside the pooled task, and to clear thread-locals when the task finishes.

Here is the async-propagation pattern in Node with `AsyncLocalStorage`, which is the cleanest illustration because Node's whole model is callbacks across the event loop:

```js
const { AsyncLocalStorage } = require('async_hooks');
const als = new AsyncLocalStorage();
const pino = require('pino')();

// At the edge: mint or accept the id, then run the whole request inside it.
function middleware(req, res, next) {
  const traceId = req.headers['x-request-id'] || crypto.randomUUID();
  als.run({ traceId }, () => next());
}

// Anywhere, even across awaits, the id is recoverable from the store.
function log(obj, msg) {
  const store = als.getStore() || {};
  pino.info({ ...obj, trace_id: store.traceId }, msg);
}

async function handler() {
  await db.query('...');     // trace_id survives this await
  log({ step: 'charged' }, 'order_processed'); // line carries trace_id
}
```

And the Go version, where propagation is explicit through `context.Context` — verbose, but it never loses the id because you pass it by hand:

```go
type ctxKey string

const traceKey ctxKey = "trace_id"

// Edge: put the id into the context for this request.
func withTrace(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, traceKey, id)
}

// Anywhere downstream, including new goroutines you pass ctx to:
func logInfo(ctx context.Context, msg string, args ...any) {
	id, _ := ctx.Value(traceKey).(string)
	slog.InfoContext(ctx, msg, append([]any{"trace_id", id}, args...)...)
}

// Outbound call: forward the id as a header so the next service continues it.
func callPayment(ctx context.Context, req *http.Request) {
	if id, ok := ctx.Value(traceKey).(string); ok {
		req.Header.Set("X-Request-Id", id)
	}
	// ... do the call ...
}
```

In practice you should adopt **OpenTelemetry** and the W3C `traceparent` header rather than rolling your own, because the standard gives you not just an id but a parent-child span structure (which hop called which), and the propagation is handled by instrumentation libraries for common frameworks. For the full treatment of how traces, metrics, and logs fit together as a designed system, see the system-design post on [observability: metrics, logs, and traces by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design); here the point is narrower and sharper — the correlation id is the field that turns a fleet's worth of disconnected lines back into one debuggable request. A sibling post on debugging across service boundaries (planned in this series) goes deeper on the cross-service reconstruction workflow.

## 6. The contextual logger: bind the request's fields once

Putting `request_id`, `tenant_id`, `version`, and `user_id` on *every* log line sounds like a lot of typing, and if you pass those fields by hand into every single `log.info(...)` call, two things happen: the code gets unbearably noisy, and — more dangerously — people forget, so the lines that matter most (the ones near the bug, written hastily) are exactly the ones missing the context that would let you find them. The solution is the **contextual logger** (also called a bound logger or a child logger): you capture the request's context *once*, at the boundary, into a logger object, and then every line written through that logger automatically carries those fields. You bind once; every line inherits.

![A vertical stack showing a request binding its correlation id tenant and version into a logger once at the boundary so every downstream log line inherits the context automatically](/imgs/blogs/logging-as-a-debugging-instrument-6.png)

The figure is the pattern in one glance: at the request boundary you read the id and tenant, create a child logger bound with those fields, pass that logger down the call chain (or stash it in the request/async context), and from then on every `log.info("...")` you write — even fifteen function calls deep, even hastily, even by a colleague who forgot the convention exists — comes out stamped with the full context. The cost, which the figure marks as the trade-off, is plumbing: you have to get the bound logger to the code that logs, which means either passing it as a parameter or putting it in the request context (Go's `context`, Python's `contextvars`, Node's `AsyncLocalStorage`). That plumbing is the same plumbing the correlation id needs, so you do it once and get both.

In `structlog` (Python), binding returns a new logger with the fields attached:

```python
import structlog

base = structlog.get_logger()

def handle_request(req):
    # Bind the request's context ONCE, at the boundary.
    log = base.bind(
        request_id=req.id,
        tenant_id=req.tenant,
        user_id=req.user_id,
        version=BUILD_VERSION,
    )
    log.info("request_received", path=req.path)
    result = do_work(req, log)          # pass the bound logger down
    log.info("request_completed", status=result.status, latency_ms=result.ms)

def do_work(req, log):
    # Fifteen calls deep, this line STILL carries request_id, tenant, etc.
    log.debug("pricing_decision", tier="B", premium=True, region="EU")
    ...
```

`pino` (Node) calls it a child logger; `slog` (Go) uses `logger.With(...)` to return a logger carrying base attributes:

```js
// pino child logger: bind once, inherit everywhere.
const reqLog = pino.child({
  request_id: req.id,
  tenant_id: req.tenant,
  version: BUILD_VERSION,
});
reqLog.info({ path: req.path }, 'request_received');
// every reqLog.* line below carries request_id, tenant_id, version
```

```go
// slog: a logger that carries base attributes on every line.
reqLog := slog.With(
	"request_id", req.ID,
	"tenant_id", req.Tenant,
	"version", buildVersion,
)
reqLog.Info("request_received", "path", req.Path)
```

The payoff at incident time is direct: because *every* line carries the request id and tenant, the cross-service query from section 5 actually returns everything — there are no orphan lines missing the id, because no human had to remember to add it. The contextual logger is what makes "grep one id, get the whole request" reliable rather than aspirational. It is the difference between a correlation id that works in the demo and one that works at 2am when the lines you need were written by someone in a hurry.

## 7. Log-driven bisection: binary-search where a value goes wrong

The series' master technique is binary search — turn the bug into a question you can answer at a midpoint, and halve the suspect space (we covered it in depth in [binary-search your bug with bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection)). Logging is how you run that binary search *across the path of a single request* in production, where you cannot step through with a debugger. The setup: a value is correct when the request starts and wrong by the time it produces output. Somewhere along the request's path — through a dozen functions, several services, a few transformations — a step corrupted it. You do not know which step. With a debugger you would set a watchpoint on the value and see exactly when it changed; in prod you cannot. But you can do the logging equivalent: **log the value at each decision point, then read the logs to find the first point where it is wrong.**

![A timeline of one request logging an amount value at each decision point showing it is correct through ingress FX and discount then wrong after the rounding step](/imgs/blogs/logging-as-a-debugging-instrument-7.png)

The figure walks the canonical case. A checkout's `amount` should be 42 (in whatever unit), but the customer was charged 0. You have, or you add, a log line at each transformation that touches the amount: at ingress (42, correct), after the FX conversion (42, correct), after the discount step (42, correct — the coupon was expired so discount was zero), and after the rounding step (0 — wrong). The bug is in the rounding step, full stop. You did not need to reproduce anything; you read four log lines for the one request, found the first one where the value was wrong, and the transition from the line before it to that line *is* the buggy step. That is binary search collapsed onto the log timeline: each decision-point log is a probe, and the first bad probe brackets the fault to a single transformation.

This is why logging *the decision and its inputs* (section 4) matters so much: a log that just says "rounding done" would not have told you the amount was now 0. The line had to carry the value. In the steady state you keep these decision-point logs at DEBUG so they cost nothing in prod, and when a value-corruption bug appears you turn DEBUG on for the affected path (section 3), let the next occurrence record its values at every step, and read off the bad transition. If the path crosses services, the correlation id stitches the probes across the service boundary so the binary search continues through the fleet. The conceptual move is identical to setting watchpoints, but it works on a request that already finished, in a process you cannot attach to, distributed across machines.

#### Worked example: the once-a-day 500, found by grepping one id across six services

Here is the full investigation that motivated the figures, with the numbers. The symptom: roughly one checkout 500 per day, no pattern anyone could see, never reproducible in staging. Six services were in the path. The first thing we had — because the system had a correlation id and structured logs, which is the entire point — was the failing request's id, captured by the gateway and returned to the client in an `X-Request-Id` response header that the customer-support tool surfaced. Call it `a91f3c`.

Step one: one query across all services' logs, `trace_id:"a91f3c"`, time-ordered. It returned eleven lines across five services. Reading them top to bottom: gateway received it, auth approved it in 4 ms, order service started, inventory reserved stock in 7 ms, and then the payment service logged `payment_attempt` followed *30 seconds later* by `gateway_timeout error.type=DeadlineExceeded`, and then the order service logged the 500 it returned to the user. Eleven lines, one of which — the 30-second gap inside the payment service — was obviously the fault. Without the correlation id this would have been hopeless: the payment service logs millions of lines a day, and a single timeout among them carries no sign that it belonged to *this* user's 500.

Step two: why did *that* payment call time out when the same call succeeds millions of times a day? We had the payment service's structured fields. We queried `error.type:"DeadlineExceeded" AND service:"payment"` over the last week and got the distribution: it happened about once a day, and *every single occurrence* had the same `db_query:"SELECT ... FOR UPDATE"` field and a `lock_wait_ms` over 30,000. The timeout was always a row lock wait on the payments table. We then queried for what *held* the lock at those timestamps — the slow-query log, joined by time — and found a nightly reconciliation job that, once a day, took a long lock on the same rows for about 35 seconds. The 500 happened precisely when a customer checkout collided with the reconciliation job's lock window. One a day, because the job ran once a day; unreproducible in staging, because staging had no reconciliation job running against live-shaped data.

The root cause was found entirely from logs — no debugger, no repro — because three things were already in place before the incident: a correlation id propagated across all six services, structured fields (`error.type`, `db_query`, `lock_wait_ms`) we could query and aggregate, and one-line-per-request INFO that gave us the request to anchor on. The fix was to make the reconciliation job take shorter locks in batches; the measured result was the once-a-day 500 going to zero over the following month, and a `lock_wait_ms` p99 on that table dropping from 31,000 ms to under 200 ms. The investigation, start to finish, took about twenty minutes of querying. Past-us, who had instrumented those services months earlier, did the hard part.

## 8. Sampling: so your logs neither lie nor bankrupt you

Logging everything, everywhere, at full volume, is not virtuous; it is a way to go broke and, worse, to make your instrument lie. A high-traffic service can emit hundreds of thousands of log lines a second. Storing, shipping, and indexing those lines costs real money — the dreaded 2am realization that the log bill is \$40k this month because someone left DEBUG on in a hot path. And the volume itself can change behavior: per the heisenbug mechanism in section 1, logging inside a tight loop slows it down and can mask a timing bug. So you sample — you keep a fraction of the lines — but sampling done carelessly makes your logs *lie*, because if you keep 1% of lines uniformly and then ask "what's our error rate," you are estimating it from a 1% sample, and rare events get dropped. The art is sampling that preserves the signal you debug with while shedding the volume you cannot afford.

![A decision tree for sampling that always keeps errors and then chooses head sampling tail sampling or rate limiting for normal success lines](/imgs/blogs/logging-as-a-debugging-instrument-5.png)

The figure is the decision tree, and its first branch is the rule that makes sampling safe: **always log errors, never sample them away.** Errors are rare and precious; you keep 100% of them no matter what your sampling rate is for everything else. You only sample the high-volume *normal* lines — the successful requests, the routine INFO. For those, you have three tools:

**Head sampling** decides at the start of the request, before you know how it turns out: roll a die at ingress, and with probability *p* keep *all* of this request's lines, otherwise drop them. It is cheap (one decision per request, made early, so you never pay to generate the dropped lines) and it keeps requests *whole* (you keep all of a sampled request's lines, so the kept ones are still debuggable end to end). Its weakness is that it is blind to outcome — it decides to keep or drop before it knows whether the request was slow or failed, so it will, by bad luck, drop interesting requests and keep boring ones. **Tail sampling** decides at the *end*, after you know the outcome: keep this request's lines if it was slow, or errored, or hit some condition; drop it if it was a fast clean success. Tail sampling is far better for debugging because it keeps exactly the interesting requests — but it costs more, because you have to buffer a request's lines until you know its fate, then decide, so you pay to generate lines you might discard. **Rate limiting / log-once** handles the pathological case of a *single* line that fires in a hot loop a million times a second: you cap it ("log this message at most once per second per host") or log it once and then suppress duplicates with a count ("...and 14,231 more like this in the last minute"). This is what saves you from the bug where one misplaced line in a retry storm generates the \$40k bill.

Here is a rate-limiting / log-once helper, because it is the one most people have to write themselves and the one that most often prevents a runaway bill. This Python version logs a given key at most once per interval and tells you how many it suppressed:

```python
import time, threading

class RateLimitedLogger:
    def __init__(self, log, interval_s=1.0):
        self._log = log
        self._interval = interval_s
        self._last = {}        # key -> last emit time
        self._suppressed = {}  # key -> count suppressed since last emit
        self._lock = threading.Lock()

    def warn_once(self, key, msg, **fields):
        now = time.monotonic()
        with self._lock:
            last = self._last.get(key, 0.0)
            if now - last >= self._interval:
                dropped = self._suppressed.pop(key, 0)
                self._last[key] = now
                self._log.warning(msg, suppressed=dropped, **fields)
            else:
                self._suppressed[key] = self._suppressed.get(key, 0) + 1
```

A retry storm that would have logged a million identical WARN lines a second now logs roughly one per second, each carrying a `suppressed=` count so you do not lose the *magnitude* of the problem — you still see it is happening a million times a second, you just do not pay to store it a million times. The crucial property is that you kept the signal (something is firing at huge volume) while shedding the volume.

| Strategy | Decides when | Cost | Keeps the interesting requests? |
|----------|--------------|------|---------------------------------|
| Head sampling | At ingress, before outcome | Cheap; never generates dropped lines | No — blind to outcome, drops some interesting |
| Tail sampling | At end, after outcome | Higher; must buffer then decide | Yes — keeps slow/failed requests by design |
| Always-log errors | Per error line | Negligible | Yes — never sampled away |
| Rate-limit / log-once | Per hot repeated line | Negligible | Keeps signal, sheds duplicate volume |

The cost/signal tradeoff also has a second axis worth naming: **cardinality and retention tiers.** High-cardinality fields (a field with millions of distinct values, like a raw URL with embedded ids, or a per-user field) blow up the index in many log backends and are the other common source of a surprise bill — so you template them (`/users/:id` not `/users/8412`) and put truly unbounded values in the message, not the indexed fields. And not all logs deserve the same retention: keep ERROR and a sampled INFO for ninety days where you can query them, but ship full DEBUG/TRACE to cheap object storage with a seven-day life, because you almost never need DEBUG older than a week and the cost of keeping it queryable for ninety is enormous. Retention tiers and sampling together are how you make logs *affordable enough to keep on*, which matters because a log you turned off to save money is a log that will not be there at 2am.

#### Worked example: one log line that turned "unreproducible" into a five-minute fix

A reporting feature intermittently produced a CSV with a blank column for some customers, maybe one report in a few hundred, and nobody could reproduce it — it never happened in dev, never in staging, only sometimes in prod for some accounts. The code that built the column was a switch over an account's plan type with a `default` branch that, fatefully, did nothing. There was no log line in that function at all; it silently produced a blank when it hit a plan type it did not recognize.

We added *one* structured DEBUG line in the default branch: `log.debug("unhandled_plan_type", plan=account.plan, account_id=account.id)`. We shipped it (the function was not hot, so even at INFO it would have been fine, but we put it at DEBUG to be safe), turned DEBUG on for the reporting service for an hour using the runtime level bump from section 3, and waited for the next blank report. It came within forty minutes, and the log line was right there: `plan="enterprise_trial"` — a plan type that had been added by the billing team three weeks earlier and that the reporting switch had never been updated to handle. The `default` branch was eating it. The fix was a one-line addition of the `enterprise_trial` case; the *diagnosis* was the one log line that turned an invisible silent failure into a queryable, named event with the exact unrecognized value attached.

The sampling rule that kept it affordable: that DEBUG line, once we understood it, became a permanent WARN — `unhandled_plan_type` is genuinely a degraded path worth a low-volume warning — but rate-limited with `warn_once` keyed on `plan`, so if billing adds another unhandled plan type and it starts happening thousands of times a day, we get one WARN per second per plan with a `suppressed=` count, not a flood. We kept the signal (a new plan type is unhandled, here it is), shed the volume, and never paid a runaway bill for a log line that was, before, a blank column nobody could explain. One well-placed structured line, costed correctly, is worth more than a thousand verbose ones.

## 9. Anti-patterns: the logs that fail you at 2am

Everything above is about logs that *help*. Just as important is recognizing the logs that actively *hurt* — the ones that look like diligence but leave you blind, or worse, get you breached or fined. Each of these is common, each has a clean fix, and each is worth eliminating from your codebase as a class. The figure collects the five worst and their fixes.

![A matrix listing five logging anti-patterns including error occurred and log-and-rethrow and swallowed stack traces and secrets in logs and missing correlation ids each paired with its fix](/imgs/blogs/logging-as-a-debugging-instrument-8.png)

**The "error occurred" log.** A line that says `log.error("An error occurred")` or `log.error("Something went wrong: " + e.getMessage())` and nothing else is the canonical useless log. At 2am it tells you a thing happened, gives you no id to find the request, no inputs to understand what the code was doing, and often not even the stack trace — just the exception's *message*, which for many exceptions is itself useless ("null"). The fix is the structured-logging discipline: log the event name, the correlation id (free, if you use a contextual logger), the inputs that drove the failing code, and the full exception with its stack. `log.error("charge_failed", order_id=..., amount=..., gateway=..., exc_info=True)` is a line you can act on; `"An error occurred"` is a line you can only sigh at.

**Log-and-rethrow (double logging).** A pattern that feels responsible and is corrosive: every layer catches an exception, logs it, and rethrows it to the layer above, which catches it, logs it, and rethrows. One real failure becomes five identical-ish ERROR lines at five different stack depths, all for the *same* event, and now your ERROR rate is inflated five-fold, your alerts are noisier, and when you grep for the failure you find five lines and waste time wondering if they are five failures or one. The rule is **log an exception exactly once, at the boundary where it is actually handled** — the request handler that turns it into a 500, the top-level catch that decides the request's fate. Inner layers should let it propagate (or wrap it with context and rethrow *without logging*). Log once, at the place that owns the decision about what to do.

**Swallowing the stack trace.** Closely related and even worse: a `catch` block that logs only `e.getMessage()` (or `str(e)`), discarding the stack trace that tells you *where* the exception came from. You are left knowing *what* went wrong ("connection refused") but not *where* in your tens of thousands of lines of code the offending call lives. Always log the exception object in a way your framework renders the full trace — `exc_info=True` in Python's `logging`, `%+v` or an error-with-stack wrapper in Go, the `Error` object (not `error.message`) in Node, the `Throwable` (not `.getMessage()`) in Java. And when you wrap an exception to add context, *chain the cause* (Python's `raise ... from e`, Go's `fmt.Errorf("...: %w", err)`, Java's `new XException(msg, cause)`) so the original stack survives.

**Secrets and PII in logs.** This is the anti-pattern that is not just annoying but a security and compliance incident. Logging a password, an API key, a credit-card number, a session token, a government id, or even a full email or home address dumps sensitive data into a system — log storage, the log pipeline, third-party log vendors — that is typically *less* secured than your database and visible to *more* people (every engineer who can read logs). A leaked secret in logs is a breach surface; leaked PII is a regulatory fine (GDPR, CCPA, PCI-DSS all have teeth here). The classic way it happens is the well-meaning `log.debug("request body", body=request.body)` that one day carries a password field, or logging an entire object whose `toString` includes a token. The fix is defense in depth: an **allow-list** of fields you log (log the fields you named, never "the whole object"), a **redaction layer** in the logger itself that masks known-sensitive keys (`password`, `authorization`, `ssn`, `card_number`) before they ever hit the output — `pino` has `redact` paths, `structlog` supports processors that scrub, and most enterprise loggers have masking filters — and a CI check or log scanner that fails the build if a forbidden field name reaches a log call. Never log a raw request body, an `Authorization` header, or a full credentials object. When in doubt, log a *reference* (the last four digits, a hash, the id) not the secret itself.

**Missing correlation id.** Already covered as the inverse of section 5: a log line with no `request_id`/`trace_id` cannot be stitched into the request it belongs to, so even if it contains the perfect diagnostic value, you cannot find it amid millions of lines, and you cannot connect it to the user's failure. The fix is the contextual logger that binds the id once so it is impossible to forget. A diagnostic field with no correlation id is a clue with no case number — technically present, practically unfindable.

| Anti-pattern | Why it fails you | The fix |
|--------------|------------------|---------|
| `"An error occurred"` | No id, no inputs, no stack — unactionable | Structured fields: event, id, inputs, full exception |
| Log-and-rethrow | One failure inflated to N ERROR lines | Log once, at the handled boundary |
| Swallowed stack | Know *what*, not *where* | Log the exception object; chain the cause |
| Secrets / PII | Breach surface; regulatory fine | Redact at the logger; allow-list fields |
| Missing correlation id | Clue you cannot find or connect | Bind the id once via contextual logger |

There is a sixth, quieter anti-pattern worth a sentence: **logging that lies about timing.** A log timestamp is when the *log call ran*, not when the *event happened*, and if you log asynchronously through a buffer the gap can be seconds; worse, lines from one request can interleave with another's so that reading the file top-to-bottom does not give you true causal order. For ordering within a request, trust the explicit `latency_ms`/`duration` fields and the trace's span structure, not the wall-clock timestamps alone — and always log timestamps in UTC, because a fleet whose nodes log in local time produces a timeline you cannot reassemble across regions.

## 10. Where the logs land, and how you query them

A log line is useless if it dies in a file on a box you cannot reach. The other half of logging-as-instrument is the *pipeline*: how lines get off the host, where they are stored and indexed, and how you query them at 2am. You do not need to build this — there are mature stacks — but you do need to understand the shape so your structured fields actually become queryable.

The common pattern is: your app writes structured JSON to stdout (the 12-factor approach — the app does not manage files; the platform captures stdout), a collector on the host (Fluent Bit, Vector, the cloud agent) ships those lines to a central store, and a backend indexes them so you can query by field. The three you will most often meet are the **ELK / OpenSearch stack** (Elasticsearch/OpenSearch for storage and indexing, Logstash or Beats for shipping, Kibana for querying — full-text and field search, powerful, operationally heavy), **Grafana Loki** (indexes only a small set of *labels* like service and level, and stores the rest cheaply — much cheaper at scale, you grep within a label-narrowed stream rather than indexing every field), and the cloud-native services (**AWS CloudWatch Logs** with Logs Insights queries, **GCP Cloud Logging**, **Azure Monitor**) which are zero-ops and integrate with the rest of the cloud but can get expensive and have their own query dialects.

The query is where structured logging pays off concretely. In CloudWatch Logs Insights, the once-a-day-500 investigation's first query is literally:

```sql
fields @timestamp, service, msg, error.type, lock_wait_ms
| filter trace_id = "a91f3c"
| sort @timestamp asc
```

In Loki's LogQL, narrow by label then filter by field, then aggregate to get the distribution that revealed the once-a-day pattern:

```bash
# All lines for one request across services (label = correlated stream):
{env="prod"} | json | trace_id="a91f3c"

# Rate of the timeout over the last day, to see it is ~once/day:
sum(count_over_time(
  {service="payment"} | json | error_type="DeadlineExceeded" [1d]
))
```

In Kibana / OpenSearch the same thing is a field query (`trace_id:"a91f3c"`) and a date-histogram aggregation on `error.type`. The dialect differs; the capability is identical and it is the capability the whole post has been building toward: because the lines are structured, you ask the haystack precise questions — "this request, in order," "this error, how often, with what fields" — and it answers in seconds. Lines that were English sentences would force you to grep substrings and eyeball, which does not scale past a few thousand lines, let alone the billions a fleet produces.

One operational note that bites teams: **make sure the structured fields survive the pipeline.** A common failure is the app emitting clean JSON, but a collector or a wrapping layer re-wrapping it so the fields end up as an escaped JSON *string* inside another field, un-queryable. Test the end-to-end path — emit a known line, then query it by field in the real backend — as part of onboarding a service, because discovering at 2am that your `trace_id` is buried in a stringified blob is the worst time to learn your pipeline mangles structure.

## 11. War story: when the logs were the only witness

Real incidents are won and lost on whether the right log line existed. A few realistic and historical cases, told for what they teach about instrumentation.

**The thundering-herd retry storm.** A backend dependency had a brief 5-second blip. Every client that got an error retried — immediately, all at once — and the retries hit the recovering dependency in a synchronized wave that knocked it down again, whose errors triggered another synchronized wave, and so on: a self-sustaining storm that turned a 5-second blip into a 40-minute outage. The thing that made it diagnosable was a single structured WARN, rate-limited, that logged `retry_attempt` with a `backoff_ms` field and a `suppressed=` count. Reading those lines showed `backoff_ms=0` on every retry across thousands of clients firing in lockstep — the retries had no jitter, so they synchronized. The fix (add randomized exponential backoff with jitter) is textbook; the *diagnosis* came from one rate-limited line that, crucially, logged the backoff value. Had the line said only "retrying," the lockstep would have been invisible. For the mechanism of retry storms and idempotency in distributed systems, this is where you cross-link out rather than re-derive — the queue and retry semantics live in the message-queue and system-design material.

**The leap-second cascade (2012, and again 2015).** When a leap second was inserted, some Linux/NTP combinations and JVMs mishandled the 23:59:60 second, and certain applications — notably some running on the affected kernel — spun threads to 100% CPU, causing widespread slowdowns and outages at large sites. The teams that recovered fastest were the ones whose logs recorded *per-thread CPU and timing* such that the correlation with the exact UTC second of the leap was visible; the ones that flailed had logs that said "high CPU" with no timestamp precision or thread attribution to connect it to the clock event. The lesson is dull and vital: **log timestamps precisely and in UTC**, and when something is timing-related, the value of a log is only as good as the precision and timezone-correctness of its time fields.

**Heartbleed and the limits of logging (2014).** The Heartbleed bug in OpenSSL let an attacker read up to 64 KB of server memory per request via a malformed TLS heartbeat, potentially leaking keys and other secrets. Part of why it was so dangerous is that, by default, the malicious heartbeat requests were *not logged at all* — the vulnerable code path produced no record, so exploitation left little trace, and operators could not tell from logs whether they had been hit. The lesson here is the inverse of "don't log too much": there are security-relevant code paths where the *absence* of a log line means an attack is invisible. Instrumenting the boundaries where untrusted input is parsed — logging anomalies, oversized requests, malformed inputs at WARN — is part of making your system debuggable *and* auditable. A log you didn't write is a question you can't answer later, and for security events that question may be "were we breached?"

**The Knight Capital deploy (2012).** Knight Capital lost about \$440 million in 45 minutes when a deploy left old, repurposed code (a flag that used to mean one thing now meaning another) running on one of eight servers, which began sending a flood of erroneous orders. Logs and alerts existed but the signal was buried and the meaning of the flag's repurposing was not surfaced in any line a human read in time. The debugging lesson that generalizes: **log the version and the config a process is running with, on startup and on every request**, so that "one of eight servers is on old code" is a queryable fact (`version:` field, count by version) and not a needle you find after the money is gone. A `version` field on every line — which the contextual logger gives you for free — turns "is some node running a stale build?" from a frantic manual check into a one-line query.

The thread through all four: the incident was survivable in proportion to whether past-engineers had instrumented the relevant boundary — the retry's backoff, the timestamp's precision, the parser's anomalies, the process's version. None of these is exotic. Each is a field on a structured line that someone either did or did not write before the day it mattered.

## 12. How to reach for logging (and when not to)

Logging is the right instrument for a specific and very large class of situations, and the wrong one for others. Being decisive about which is which saves you from both blindness and bankruptcy.

**Reach for logging — instrument before the incident — when:** the bug happens in production and you cannot attach a debugger; the bug is intermittent and you cannot sit at a breakpoint waiting for it; the request crosses process or service boundaries so no single debugger sees the whole thing; you need a historical record because the bug already happened and the state is gone; or you need to *aggregate* across many occurrences ("how often, which tenants, what inputs") to see a pattern. In all of these, a debugger is structurally unable to help, and a well-instrumented log is the only instrument that works. This is most production debugging.

**Do not reach for logging — or reach for a different tool — when:** you can reproduce the bug locally and attach a debugger, in which case [the debugger is a microscope](/blog/software-development/debugging/the-debugger-is-a-microscope-use-it) and you should use it, because stepping through live state is faster and richer than reading logs you have to add and redeploy. Do not add a permanent log line to answer a one-time local question — that is what a breakpoint or a one-off `print` is for (and print-debugging done right is its own sibling craft in this series). Do not crank logging volume up in a hot concurrency path to chase a race, because you will perturb the timing and create or hide a heisenbug — reach for a thread sanitizer or a deterministic record-replay instead. Do not log your way to a CPU-hotspot answer; that is what a profiler and a flame graph are for, and logs will mislead you about where time goes. And do not treat logging as free: a log line in a million-times-per-second path, un-sampled, is a performance regression and a budget hole, so cost it like any other code.

The honest framing is that logging is your *production observation* instrument and the debugger is your *local investigation* instrument, and the great teams are good at both and clear about the handoff. You instrument with logs so that when production hands you a mystery, you can usually solve it from the recording; and on the rare occasions you can pull the bug down to a reproducible local case, you switch to the microscope. Logging's unique, irreplaceable value is the one no live tool has: it works on the past, across the fleet, for the bug you cannot reproduce — which is exactly the bug that pages you at 2am.

A practical adoption order, if you are starting from string-concat logs and want the biggest wins first: (1) adopt structured logging in one format across services — this alone makes everything queryable; (2) add a correlation id and propagate it everywhere, because cross-request reconstruction is the highest-leverage single capability; (3) adopt the contextual logger so the id and tenant and version ride on every line for free; (4) get levels right so ERROR means error and DEBUG ships off-by-default-but-flippable; (5) add sampling and redaction so it is affordable and safe to keep on. Each step compounds the previous one, and after step two you will already solve incidents you used to escalate.

## Key takeaways

- **When the bug already happened, logs are the only instrument that works.** A debugger reads live memory; a production bug's state was destroyed by the program's forward progress. You can only go back to moments you recorded.
- **Levels are a contract about audience and volume, not an intensity dial.** ERROR = a human must act; WARN = degraded but coped; INFO = one line per unit of work; DEBUG = decisions and inputs, off-in-prod-behind-a-flag; TRACE = the firehose, short bursts only.
- **Ship DEBUG to prod but keep it off, and bump the level at runtime** — per-logger or per-request, with no redeploy and no restart, so the bug records itself in full the next time it appears.
- **Structure every line as key-value or JSON, and log the decision and its inputs**, not "done." The fields that make a line debuggable: request id, tenant/user, version/host, the actual values, latency, and the full exception with its stack.
- **Propagate one correlation id across services, async boundaries, and threads.** It is the single highest-leverage move in distributed debugging: one query reassembles a fleet's worth of scattered lines into one request's story.
- **Bind context once with a contextual logger** so every line carries the id, tenant, and version automatically — which is what makes "grep one id, get the whole request" reliable instead of aspirational.
- **Use logs to binary-search a value across a request's path:** log the value at each decision point and find the first line where it is wrong; the step before it is the fault.
- **Sample so logs neither lie nor bankrupt you:** always keep errors, head- or tail-sample the rest (tail is better for debugging), rate-limit hot repeated lines, template high-cardinality fields, and tier retention.
- **Eliminate the anti-patterns as a class:** "an error occurred," log-and-rethrow double logging, swallowed stack traces, secrets/PII in logs, and lines with no correlation id. Each has a clean, known fix.
- **The best log line is the one past-you wrote before the incident.** Logging is a discipline you practice when nothing is broken, for a future self who will have nothing else.

## Further reading

- [Stop guessing: the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) — the series intro and the observe → reproduce → hypothesize → bisect → fix → prevent loop this post's observation layer plugs into.
- [The debugger is a microscope, use it](/blog/software-development/debugging/the-debugger-is-a-microscope-use-it) — the live-investigation counterpart; reach for it when you *can* reproduce locally, and reach for logs when you cannot.
- [Binary-search your bug with bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection) — the master technique that section 7's log-driven value bisection applies to a single request's path.
- [Observability: metrics, logs, and traces by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) — how logs fit alongside metrics and distributed traces as a designed system, including OpenTelemetry and trace context.
- Sibling posts in this series (planned): print-debugging done right, observability for debugging prod, and debugging across service boundaries — the cross-service reconstruction workflow that the correlation id makes possible.
- The OpenTelemetry specification and the W3C Trace Context standard — the canonical, vendor-neutral definitions of trace/span propagation and the `traceparent` header.
- *Debugging* by David J. Agans (especially "Quit Thinking and Look" and "Keep an Audit Trail") and *Why Programs Fail* by Andreas Zeller — the timeless principles behind treating logs as evidence rather than noise.
- The official docs for the loggers named here: Python `logging` and `structlog`, Go `log/slog`, Node `pino`, plus your log backend's query reference (Kibana/OpenSearch query DSL, Grafana Loki LogQL, or CloudWatch Logs Insights).
