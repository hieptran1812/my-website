---
title: "Resilience Patterns: Timeouts, Retries, Circuit Breakers, and Bulkheads"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "The four patterns that turn a fragile remote call into a survivable one — how to set timeouts and a shrinking deadline budget, retry only what is safe with backoff and jitter and a budget, trip a circuit breaker before you hammer a sick dependency, and bulkhead your thread pools so one slow downstream cannot take the whole service down."
tags:
  [
    "microservices",
    "resilience",
    "circuit-breaker",
    "retries",
    "timeouts",
    "bulkhead",
    "fault-tolerance",
    "distributed-systems",
    "software-architecture",
    "backend",
    "reliability",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads-1.webp"
---

At 2:14 on a Tuesday afternoon, the ShopFast payment service got slow. Not down — slow. Its p99 latency drifted from a healthy 40ms up to 800ms, then to five full seconds, because a database index had quietly stopped being used after a schema migration. Five seconds is an eternity for a synchronous call, but it is not an error. The payment service still returned `200 OK`. It just took its sweet time about it.

Twelve minutes later, all of ShopFast was down. Not the payment page — the *entire site*. The product catalog, which has nothing to do with payments, returned `503`. The cart, the search box, the "track my order" page: all dead. The on-call engineer stared at a dashboard that made no sense. Payment was slow, sure, but why was the homepage throwing errors? The homepage does not even call payment.

Here is what actually happened, and it is the most important failure mode in all of microservices. The order service called payment with no timeout. Every request that hit a checkout grabbed a worker thread and then *waited* — for five seconds — for payment to answer. At ShopFast's traffic, that was a few hundred new checkout requests per second, each holding a thread for five seconds. The order service had a pool of 200 threads. Do the arithmetic: at 100 requests per second each holding a thread for 5 seconds, you need 500 threads in flight, and you have 200. Within about two seconds every thread in the order service was blocked, waiting on a payment call that would not return. The order service could no longer answer *anything* — not even the health check the load balancer used to decide whether it was alive. The load balancer marked it dead, the gateway started returning `503`, and because the gateway shared a connection pool across all backends, the gateway itself began to choke. One slow dependency, three hops away from the homepage, took down the whole company.

![A flow graph showing users hitting the gateway and order service whose two hundred threads block on a five second payment call and cascade into a full checkout outage](/imgs/blogs/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads-2.webp)

This post is about the four patterns that would have stopped that outage cold, and that every remote call in a microservices system needs: **timeouts**, **retries**, **circuit breakers**, and **bulkheads**. They are foundational — the core of microservices resilience — because they convert the one fact that defines distributed systems (the network and the things on the other end of it *will* fail, slowly and partially and at the worst time) from an existential threat into a bounded, survivable event. By the end you will be able to take a naive remote call and wrap it so that when payment slows to five seconds, checkout degrades gracefully, the homepage stays up, and payment gets a chance to recover instead of being buried alive. You will know where each pattern goes, what each one costs, and — just as important — the cases where each one makes the outage *worse*, because all four of them can, and they routinely do in the hands of someone who copy-pasted a config without understanding it.

We will use one running example throughout: the ShopFast **order service** calling the **payment service**. We will start with the naive call that caused the outage above, then wrap it one layer at a time — timeout, retry with jitter and a budget, circuit breaker, bulkhead — until the same call survives a five-second payment brownout. This builds directly on [inter-service communication fundamentals and fallacies](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies), which establishes *why* the network is unreliable; this post is what you actually *do* about it.

## Why a remote call is fundamentally different from a function call

A junior engineer's first mental error in microservices is treating a remote call like a local one. `paymentClient.charge(order)` *looks* exactly like `paymentService.charge(order)` did back when it was a method in the monolith. It is the same line of code. But the semantics changed completely the moment a network got between the caller and the callee, and almost every resilience pattern exists to manage the consequences.

A local function call has exactly two outcomes: it returns a value, or it throws. It is fast (nanoseconds), it cannot partially complete from the caller's point of view, and if the callee is buggy it fails *deterministically* — you get the same exception every time. A remote call has at least four outcomes. It can return a value. It can return an error. It can *hang* — sit there for an unbounded amount of time because the callee is slow, the network dropped the packet, or a load balancer in the middle is holding the connection. And it can fail *ambiguously*: the call times out, but you have no idea whether the callee actually did the work. The charge might have gone through and the response got lost on the way back. That last category — the ambiguous failure — is the one that makes retries dangerous, and we will spend real time on it.

The single most damaging of these is the *hang*, because it is the one that violates a junior's intuition. We are trained to fear errors. But in a distributed system, **a slow dependency is more dangerous than a dead one.** A dead dependency fails fast: the connection is refused, you get an error in a millisecond, you handle it and move on. A slow dependency holds your resources hostage. It takes your threads, your connections, your memory, and gives them back so slowly that you run out. The ShopFast outage was not caused by payment going *down*. It was caused by payment staying *up* but answering slowly. This is why "is the service healthy?" is the wrong question; the right question is "is the service responding within its latency budget?", and the patterns in this post are all, fundamentally, ways of enforcing that budget and containing the damage when it is blown.

The eight classic *fallacies of distributed computing* — the network is reliable, latency is zero, bandwidth is infinite, the network is secure, topology never changes, there is one administrator, transport cost is zero, the network is homogeneous — are each a way a naive remote call betrays you. Timeouts, retries, breakers, and bulkheads are the practitioner's standing answer to the first two: latency is *not* zero and the network is *not* reliable, so every call must be bounded in time, defended against transient loss, and isolated from its neighbors.

## Pattern one: timeouts — every remote call MUST have one

The most important sentence in this post: **every remote call must have an explicit timeout, and the default is almost never the timeout you want.** The ShopFast outage had exactly one root cause that mattered — the order service called payment with no timeout, which in most HTTP clients means an *infinite* timeout. The connection happily waited five seconds, then would have waited five minutes, then five hours, because nobody told it not to.

This is not a hypothetical default. It is the actual default in a frightening number of real clients. Go's `http.Client{}` with no fields set has *no timeout at all* — a zero `Timeout` means wait forever. Python's `requests.get(url)` with no `timeout=` argument waits forever. Java's older `HttpURLConnection` defaults to infinite. The JDBC drivers for most databases default `socketTimeout` to zero, which is infinite. Every one of these is a loaded gun pointed at your thread pool, and "we just never set the timeout" is the single most common root cause in microservices post-mortems I have read.

### Two timeouts, not one: connect vs read

When people say "timeout" they usually mean one number, but a correct HTTP client has at least two, and they protect against different failures:

- **Connection timeout** — how long to wait to *establish* the TCP connection (and the TLS handshake). This catches "the host is down" or "there is no route" — a SYN with no SYN-ACK. It should be short, because a healthy connect on the same network is single-digit milliseconds. 100–500ms is typical.
- **Read / request timeout** — how long to wait for the *response* after the request is sent. This is the one that catches a slow dependency. It must be set to your latency budget for that call, not to some round number someone liked. If payment's p99 under load is 200ms, a 250ms read timeout gives a little headroom and fails fast on the five-second brownout.

A subtle third one matters for streaming or large bodies: a **socket / idle timeout** that fires if no bytes arrive for some window, so a stalled-mid-response server does not hang you. And separately from the HTTP client there is the **call-level deadline** — the total time *this logical operation* is allowed to take, which may wrap several retries and is the thing you actually care about as a caller. Keep these straight: connect timeout protects the handshake, read timeout protects one attempt, and the deadline bounds the whole operation including retries.

Here is the naive ShopFast call — the one that caused the outage — in Go. Notice there is nothing wrong with it that a code review would obviously catch. It compiles, it works in the happy path, it passes tests. It is a time bomb.

```go
// The naive client — DO NOT SHIP THIS.
// http.Client{} has zero Timeout, which means "wait forever".
var paymentClient = &http.Client{} // <-- infinite timeout

func chargeOrder(order Order) (*ChargeResult, error) {
	body, _ := json.Marshal(order)
	resp, err := paymentClient.Post(
		"http://payment-svc/charge",
		"application/json",
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	// ... decode and return. If payment takes 5s, this goroutine
	// is stuck here for 5s, holding whatever called it.
	var result ChargeResult
	return &result, json.NewDecoder(resp.Body).Decode(&result)
}
```

And here is the same call with bounded timeouts and an explicit per-call deadline via `context`. This one change — and *only* this change — would have prevented the cascading outage in the intro, because the order service's threads would have freed themselves after 250ms instead of waiting five seconds.

```go
// Bounded client: connect timeout + overall transport timeout.
var paymentClient = &http.Client{
	Timeout: 250 * time.Millisecond, // total time for this attempt
	Transport: &http.Transport{
		DialContext: (&net.Dialer{
			Timeout: 100 * time.Millisecond, // connection timeout
		}).DialContext,
		TLSHandshakeTimeout:   100 * time.Millisecond,
		ResponseHeaderTimeout: 200 * time.Millisecond, // time to first byte
	},
}

func chargeOrder(ctx context.Context, order Order) (*ChargeResult, error) {
	// Per-call deadline: the whole charge op gets 600ms, no matter
	// how many retries happen inside it.
	ctx, cancel := context.WithTimeout(ctx, 600*time.Millisecond)
	defer cancel()

	body, _ := json.Marshal(order)
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost,
		"http://payment-svc/charge", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp, err := paymentClient.Do(req)
	if err != nil {
		return nil, err // deadline exceeded shows up here
	}
	defer resp.Body.Close()
	var result ChargeResult
	return &result, json.NewDecoder(resp.Body).Decode(&result)
}
```

### The deadline budget that must shrink down the call chain

Setting a timeout on one call is necessary but not sufficient. The deeper pattern — the one that separates a senior's design from a junior's — is the **deadline budget** that propagates and *shrinks* down the entire call chain so the whole request has a single bound. Without it, a chain of services each with a "reasonable" 1-second timeout can take many seconds end to end, and the user's browser gave up long ago.

The idea: the entry point (the gateway, or the request handler) starts a budget — say, "this whole request gets 1000ms." It passes the *remaining* budget down to every service it calls, as a deadline (an absolute time) or a remaining-duration header. Each service, before it calls the next one downstream, computes how much of the budget is left, subtracts what it expects its own work to take, and passes the rest down. If a service receives a request whose deadline has already passed, it does not even bother starting work — it returns immediately. This is **deadline propagation**, and it is built into gRPC natively (the `grpc-timeout` header) and into any well-instrumented HTTP system via a header you pass yourself.

![A flow graph showing a one thousand millisecond budget at the gateway shrinking at each hop to eight hundred fifty then six hundred then four hundred milliseconds as it propagates down through order payment and the external processor](/imgs/blogs/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads-8.webp)

```go
// Deadline propagation: each hop passes the REMAINING budget down.
func handlePlaceOrder(ctx context.Context, req PlaceOrderReq) error {
	// gRPC carries the deadline automatically; for HTTP, read a header.
	deadline, ok := ctx.Deadline()
	if !ok {
		deadline = time.Now().Add(1000 * time.Millisecond) // default budget
		var cancel context.CancelFunc
		ctx, cancel = context.WithDeadline(ctx, deadline)
		defer cancel()
	}

	// Cheap, fast: reserve inventory (budget mostly intact).
	if err := reserveInventory(ctx, req); err != nil {
		return err
	}

	// Before calling payment, check how much budget is left.
	remaining := time.Until(deadline)
	if remaining < 150*time.Millisecond {
		// Not enough time left to do a meaningful payment call.
		// Fail fast instead of starting work we cannot finish.
		return ErrDeadlineTooTight
	}
	// payment gets whatever is left, minus a small reserve for our
	// own post-processing. The ctx already carries the deadline,
	// so the payment client will not exceed it.
	return chargeOrder(ctx, req.Order)
}
```

The payoff is enormous and easy to miss: with deadline propagation, a request that has already blown its budget *stops doing work all the way down the chain*. Every service that receives an expired deadline returns instantly without touching its database, without calling the next service, without burning CPU on a result nobody will read. During an overload, this is the difference between a system that wastes its scarce capacity computing answers for clients who have already disconnected, and one that sheds that work immediately and spends its capacity on requests that can still succeed. We will return to this idea in [rate limiting, backpressure, and load shedding](/blog/software-development/microservices/rate-limiting-backpressure-and-load-shedding), where shedding already-doomed work is the whole game.

#### Worked example: picking the timeout from the latency histogram

Junior engineers pick timeouts by feel — "let's say 3 seconds, that seems safe." That is exactly backwards: a too-generous timeout is what *causes* the thread-exhaustion cascade, because it lets slow calls pile up. Pick it from data.

Suppose payment's latency under normal load is: p50 = 35ms, p95 = 90ms, p99 = 180ms, p99.9 = 400ms. You want a read timeout that lets through essentially all *healthy* responses but cuts off the pathological ones fast. A reasonable rule of thumb is **timeout ≈ p99.9 × 1.0 to 1.5**, here roughly 400–600ms, *unless* your end-to-end budget is tighter, in which case the budget wins. Set it to 250ms and you will time out a chunk of legitimately-slow-but-fine requests (everything between p99 and p99.9), turning successes into errors — a too-tight timeout backfires by manufacturing failures. Set it to 5 seconds and you have recreated the outage: under a brownout, calls that used to take 35ms now take 5 seconds and you accumulate `100 req/s × 5s = 500` concurrent calls against a 200-thread pool.

Now do the capacity math that makes the timeout choice concrete. By Little's Law, the average number of concurrent calls in flight is `arrival_rate × average_latency`. At 100 req/s and a healthy 35ms average, that is `100 × 0.035 = 3.5` threads in flight — trivial. Set timeout to 250ms; during a total brownout where every call hits the timeout, the worst case is `100 × 0.25 = 25` threads in flight. Your 200-thread pool absorbs that easily and keeps serving everything else. Set timeout to 5 seconds and the worst case is `100 × 5 = 500` threads — 2.5× your pool — and you are dead. **The timeout, not the failure, decides whether you survive.** That single number is the most leveraged reliability knob you own.

## Pattern two: retries — powerful, and the fastest way to turn a brownout into an outage

Retries are seductive. A request failed; just try again; most of the time the second attempt works and the user never notices. And for *transient* failures — a dropped packet, a momentary connection reset, a load balancer that just shifted a backend out of rotation — retries are exactly right and dramatically improve your success rate. But retries are also the single most common way that engineers turn a small, recoverable degradation into a full, self-inflicted outage. You must understand both faces.

### The two preconditions: transient AND idempotent

A retry is only safe when **both** of these hold, and a retry that violates either is a bug waiting to page you:

1. **The failure is transient** — retrying has a real chance of succeeding because the cause was temporary. A `503 Service Unavailable`, a connection reset, a timeout on a read — these *might* succeed on a second try. A `400 Bad Request` will never succeed no matter how many times you retry; the request is malformed. A `403 Forbidden` will never succeed. Retrying a non-transient error just wastes load and delays the inevitable error the caller needs to see. So you retry only specific, transient signals: connection errors, timeouts, `502/503/504`, and explicit `429 Too Many Requests` (respecting its `Retry-After`). You do **not** retry `400/401/403/404/422`.

2. **The operation is idempotent** — performing it twice has the same effect as performing it once. Reading is naturally idempotent. But `chargeOrder` is *not* idempotent by default: charge twice and you bill the customer twice. This is the ambiguous-failure trap from earlier. When `chargeOrder` times out, you do not know whether payment processed the charge before the response got lost. If you blindly retry, you may double-charge. Retrying a non-idempotent write blindly is one of the most expensive bugs in this whole space — it shows up as duplicate charges, duplicate orders, duplicate emails.

The fix for the second precondition is to *make the operation idempotent* — attach an idempotency key (a unique ID the caller generates for this logical operation) so the payment service can recognize a retry and return the original result instead of charging again. That mechanism is deep enough to deserve its own treatment, and it gets one in [idempotency and exactly-once effects across services](/blog/software-development/microservices/idempotency-and-exactly-once-effects-across-services). The rule to carry from here: **never put an automatic retry around a non-idempotent operation.** Either make it idempotent first, or do not retry it.

### Exponential backoff with jitter — and why naked backoff is not enough

When you do retry, you must *space out* the attempts, or you make the problem worse. Retrying immediately, in a tight loop, just hammers a struggling dependency at the worst possible moment. The standard answer is **exponential backoff**: wait a base delay, then double it each attempt — 100ms, 200ms, 400ms, 800ms — so you back off quickly as failures persist.

But exponential backoff alone has a vicious failure mode at scale: **synchronization**. Suppose payment has a one-second blip and returns errors to ten thousand clients. All ten thousand clients fail at roughly the same instant, all of them back off by exactly 100ms, and all ten thousand of them retry at exactly `T + 100ms` — a perfectly synchronized spike of ten thousand requests landing in one tick, right as payment was trying to recover. The retry spike re-kills payment, everyone backs off again to exactly `T + 300ms`, and you get a self-sustaining oscillation of synchronized retry waves. This is the *thundering herd*, and it is why naked exponential backoff is not enough.

The fix is **jitter** — randomizing the delay so the ten thousand retries spread out across the backoff window instead of landing on the same instant. "Full jitter," the variant AWS recommends in its well-known retries-and-jitter article, picks the actual delay uniformly at random between zero and the current exponential ceiling: `sleep = random(0, min(cap, base × 2^attempt))`. Now the ten thousand retries spread evenly across the window, the load is flat instead of spiked, and payment gets room to breathe.

```python
import random, time

RETRYABLE = {429, 502, 503, 504}

def call_with_backoff(do_call, max_attempts=3, base=0.1, cap=2.0):
    for attempt in range(max_attempts):
        try:
            resp = do_call()  # raises on connect error / timeout
            if resp.status_code not in RETRYABLE:
                return resp   # success OR non-retryable error: return it
        except (ConnectionError, Timeout):
            pass              # transient: fall through to retry
        if attempt == max_attempts - 1:
            break             # last attempt failed; give up
        # FULL JITTER: uniform random in [0, exponential ceiling].
        ceiling = min(cap, base * (2 ** attempt))
        time.sleep(random.uniform(0, ceiling))
    raise RetriesExhausted()
```

![A two column before and after diagram contrasting synchronized retries that land in one spike and re-kill the dependency against full jitter that spreads the same load flat so the dependency recovers](/imgs/blogs/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads-3.webp)

### The retry storm: how N layers turn 3 retries into 27× load

Here is the failure mode that surprises even experienced engineers, because it is *multiplicative across layers*. Retries do not just add load; they *compound* when stacked. Consider the ShopFast call chain: the browser's client retries, the gateway retries, the order service retries, and the payment client retries. Each layer was configured, independently and reasonably, to retry three times. Nobody is wrong locally. But globally, the math is catastrophic.

![A six event timeline of a retry storm where a payment brownout triggers timeouts that cascade into three retry layers multiplying load to twenty seven times until a retry budget caps it](/imgs/blogs/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads-4.webp)

#### Worked example: 3 layers × 3 retries = 27× amplification, and the budget fix

A single user request enters the gateway. Payment is in a brownout, so calls are failing. Walk the amplification layer by layer:

- The **payment client** inside the order service tries 3 times: 3 calls to payment.
- The **order service** is itself behind a retry — the gateway retries the order call 3 times. Each of those 3 attempts triggers a fresh round of 3 payment-client retries: `3 × 3 = 9` calls to payment.
- The **gateway** is behind the browser/edge, which retries 3 times. Each of those triggers the whole gateway-to-payment subtree again: `3 × 3 × 3 = 27` calls to payment.

So **one** user request becomes **27** requests hitting the already-struggling payment service. At even modest traffic — say 1,000 genuine requests per second during the incident — payment now sees `1,000 × 27 = 27,000` requests per second, almost all of which will fail (it is in a brownout), each failure triggering more retries. The retries are not helping payment recover; they are the reason payment cannot recover. This is the **retry storm** (or retry amplification), and it is responsible for some of the largest cloud outages on record.

![A flow graph showing users at ten thousand requests per second whose load is multiplied by stacked gateway and service retry layers into roughly ninety thousand requests per second that buries a struggling dependency](/imgs/blogs/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads-2.webp)

There are two fixes, and you want both. The first is structural: **retry at only one layer of the stack.** Pick the layer closest to the failure that has the context to know whether a retry is safe (usually the immediate client of the failing dependency), and turn retries *off* everywhere else. A `503` propagating up should be passed through, not re-retried. Many service meshes and gateways retry by default; you must explicitly disable that if your application code already retries.

The second fix is a **retry budget** (or retry token bucket), which caps retries as a *fraction of total traffic* rather than a per-request count. The idea, popularized by Google's SRE practice and implemented in Envoy and the Finagle/gRPC ecosystems: maintain a token bucket where every *successful request* adds a fraction of a token (say, 0.1) and every *retry* spends a whole token. When the bucket is empty, retries are denied even though the per-request retry count has not been hit. The effect is that retries can never exceed, say, 10% of total request volume system-wide. During a brownout where everything is failing, the bucket drains almost immediately and retries are suppressed — exactly when retrying would hurt. During normal operation, the occasional transient failure retries freely because there is plenty of budget. Concretely: with a 10% budget and 1,000 req/s of real traffic, retries are capped at ~100/s, so worst-case load on payment is `1,000 + 100 = 1,100` req/s — *1.1×*, not 27×. That is the difference between a brownout that recovers and an outage that does not.

```python
class RetryBudget:
    """Token bucket: caps retries to a fraction of total traffic."""
    def __init__(self, ratio=0.1, min_per_sec=3, ttl=10.0):
        self.ratio = ratio        # retries allowed per successful request
        self.min_per_sec = min_per_sec  # always allow a tiny baseline
        self.ttl = ttl
        self.tokens = 0.0
        self.last = time.monotonic()

    def deposit(self):
        "Call on every request attempt; earns retry budget."
        self._decay()
        self.tokens += self.ratio

    def withdraw(self):
        "Call before a retry; returns False if no budget left."
        self._decay()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return self.min_per_sec > 0 and self._baseline_ok()

    def _decay(self):
        now = time.monotonic()
        # tokens expire over ttl so a quiet period does not bank infinite retries
        self.tokens *= max(0.0, 1 - (now - self.last) / self.ttl)
        self.last = now
```

The honest summary on retries: they raise your success rate against transient faults and they are non-negotiable for a robust client — *but* a misconfigured retry is the most reliable way to convert a recoverable brownout into an unrecoverable outage. Retry only transient failures, only on idempotent operations, with backoff and jitter, at one layer, under a budget. Get any of those wrong and the retry is no longer a safety feature; it is the bug.

## Pattern three: circuit breakers — fail fast so the sick can heal

Timeouts bound a single call; retries handle transient blips. But neither handles the case where a dependency is *persistently* sick — down or badly degraded for minutes. In that case, every call is going to fail (after waiting out the timeout, after exhausting its retries), and continuing to send calls does three bad things: it wastes the caller's resources waiting for doomed calls, it piles load onto a dependency that needs *less* load to recover, and it makes the caller slow (every request now eats the full timeout). The **circuit breaker** is the pattern that detects this state and short-circuits: it stops calling the sick dependency entirely for a while, failing *instantly* instead of slowly, which both protects the caller and gives the dependency the quiet it needs to recover.

The name is an electrical analogy — a circuit breaker in your house trips to cut current when there is a fault, protecting the wiring, and you flip it back on once the fault is cleared. The software version is a small state machine that sits in front of a dependency and watches the recent success/failure rate.

### The three states

![A tree diagram of the circuit breaker state machine showing closed transitioning to open which moves to half open and then either resets to closed on a successful probe or returns to open on a failed probe](/imgs/blogs/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads-5.webp)

- **CLOSED** — the normal state. Calls pass through to the dependency. The breaker records each result in a sliding window. As long as the failure rate stays below a threshold (say 50% over the last 100 calls, or the last 10 seconds), it stays closed.
- **OPEN** — the tripped state. The breaker has seen too many failures, so it *opens* and short-circuits: every call returns an error (or a fallback) *immediately*, without touching the dependency. No threads block, no timeouts elapse, the dependency gets zero traffic from this caller. The breaker stays open for a cooldown period (say 30 seconds), giving the dependency time to recover.
- **HALF-OPEN** — the probing state. After the cooldown, the breaker does not slam back to closed (that would dump full traffic on a maybe-still-sick dependency). Instead it goes half-open and allows a *small number of trial calls* through. If those probes succeed, the dependency is healthy again and the breaker closes. If they fail, it snaps back to open and waits another cooldown. The half-open probe is the clever part: it lets the breaker recover *automatically*, without a human, and without a thundering herd, because only a trickle of probes go through during recovery.

The transitions are: CLOSED → OPEN when failure rate exceeds the threshold; OPEN → HALF-OPEN after the cooldown elapses; HALF-OPEN → CLOSED if probes succeed; HALF-OPEN → OPEN if probes fail. That is the whole machine, and it is one of the highest-value ~50 lines of logic in distributed systems.

### Tripping criteria: failure rate, not failure count, plus latency

A naive breaker trips on a raw *count* of consecutive failures ("5 failures in a row → open"). That is fragile: at high traffic, 5 failures is noise; at low traffic, you might never get 5 in a row even when the dependency is clearly broken. Mature breakers (resilience4j, Polly, Envoy) trip on a **failure rate over a window with a minimum-calls floor** — for example, "open if ≥50% of the last 100 calls failed, but only evaluate once there have been at least 20 calls." The minimum-calls floor stops a single early failure from tripping the breaker on a cold start.

Critically, "failure" should include *slow* calls, not just errors — a **slow-call-rate** threshold ("open if ≥50% of calls took longer than 300ms"). This is what makes a breaker effective against the brownout scenario from the intro, where payment is slow but technically returning `200`. A breaker that only counts errors would never trip during a pure latency brownout, which is the exact failure that hurts you most.

Here is a resilience4j configuration (the de facto standard on the JVM) for the ShopFast payment breaker, with the reasoning inline:

```yaml
resilience4j:
  circuitbreaker:
    instances:
      paymentService:
        sliding-window-type: COUNT_BASED
        sliding-window-size: 100            # evaluate over last 100 calls
        minimum-number-of-calls: 20         # don't trip before 20 samples
        failure-rate-threshold: 50          # open if >=50% fail
        slow-call-rate-threshold: 50        # open if >=50% are "slow"
        slow-call-duration-threshold: 300ms # a call >300ms counts as slow
        wait-duration-in-open-state: 30s    # cooldown before half-open
        permitted-number-of-calls-in-half-open-state: 5  # probes
        automatic-transition-from-open-to-half-open-enabled: true
```

And the call site, with the breaker wrapping the charge and a *fallback* for when the breaker is open or the call fails. The fallback is what turns "circuit breaker open" from an error the user sees into graceful degradation:

```java
@CircuitBreaker(name = "paymentService", fallbackMethod = "chargeFallback")
@TimeLimiter(name = "paymentService")   // enforces the per-call timeout
@Bulkhead(name = "paymentService", type = Bulkhead.Type.THREADPOOL)
public CompletableFuture<ChargeResult> charge(Order order) {
    return CompletableFuture.supplyAsync(() ->
        paymentClient.charge(order));   // the real remote call
}

// Called when the breaker is OPEN, or the call fails/times out.
private CompletableFuture<ChargeResult> chargeFallback(Order order, Throwable t) {
    // We do NOT pretend the charge succeeded. We queue it for async
    // retry and tell the user "payment is processing" — graceful
    // degradation, not a lie and not a hard error.
    paymentQueue.enqueueForLater(order);
    return CompletableFuture.completedFuture(
        ChargeResult.pending("Payment is being processed."));
}
```

Notice three things stacked on one method: `@CircuitBreaker`, `@TimeLimiter` (the timeout), and `@Bulkhead`. That is the composition we are building toward — the resilience4j annotations compose so the call is timed out, breaker-protected, *and* pool-isolated. The fallback does the genuinely hard product work: it decides what "degraded but not broken" means for *this* operation. For a charge, that might be "queue it and tell the user it's pending." For a product-recommendations call, it might be "return a generic best-sellers list." For a "you might also like" widget, it might be "render nothing." The right fallback is a product decision, and it is covered in depth in [handling partial failures and graceful degradation](/blog/software-development/microservices/handling-partial-failures-and-graceful-degradation).

### The honest part: breakers flap if you tune them wrong

A circuit breaker is not free safety; it is a control loop, and badly-tuned control loops oscillate. The classic failure is **flapping**: a breaker with a too-short cooldown and too-few half-open probes opens, waits 2 seconds, sends one probe that happens to succeed (the dependency is *mostly* down but answered one call), closes, immediately gets flooded with the backed-up traffic, fails again, opens again — flap, flap, flap — every few seconds. The dependency never gets sustained quiet to recover, and the caller's behavior is erratic.

The opposite failure is a breaker that is too *conservative*: a threshold of 90% failure rate over a 1,000-call window means the dependency has to be almost completely dead for a long time before the breaker even notices, by which point you have already had the thread-exhaustion cascade the breaker was supposed to prevent. Tuning is real engineering: the window must be long enough to be statistically meaningful but short enough to react in seconds; the cooldown must be long enough for a real recovery (30s is a common default; for a dependency that recovers by restarting, it might need to match the restart time); the half-open probe count must be more than one (so a single lucky call does not prematurely close it). We will put numbers on this below.

## Pattern four: bulkheads — isolate so one slow dependency cannot drain the ship

The first three patterns bound and guard *individual calls*. The bulkhead addresses a different axis entirely: **resource isolation**. Even with perfect timeouts and breakers, there is a window — before the breaker trips, during the timeout — where a slow dependency is consuming the caller's threads. If all of the caller's dependencies share *one* thread pool, a slow dependency can grab every thread in that pool, starving calls to *other, perfectly healthy* dependencies. The product catalog call, which talks to a fast healthy catalog service, cannot get a thread because all 200 are stuck waiting on slow payment. That is exactly the ShopFast cascade.

The bulkhead pattern, named by Michael Nygard in *Release It!* after the watertight compartments in a ship's hull, is the fix: **partition your resources so a failure in one compartment cannot flood the rest.** A ship with bulkheads can take a hole in one compartment, flood that compartment, and stay afloat — the Titanic famously sank because its bulkheads did not extend high enough and the water spilled over the tops, flooding compartment after compartment, which is the maritime version of a cascading failure. In software, the resources you partition are usually **thread pools** (or, in async runtimes, **semaphores** that cap concurrency) and **connection pools** — one bounded pool *per dependency*.

![A two column before and after diagram contrasting a shared two hundred thread pool where slow payment starves healthy catalog and cart calls against bulkheaded pools where payment is capped at its own twenty threads and the rest keep serving](/imgs/blogs/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads-6.webp)

### Two flavors: thread-pool isolation vs semaphore isolation

There are two ways to implement a bulkhead, and the choice matters:

- **Thread-pool isolation** — each dependency gets its own dedicated pool of threads. Calls to payment run on payment's pool; calls to catalog run on catalog's pool. If payment is slow and saturates its 20-thread pool, the 21st payment call is rejected instantly (fail fast), but catalog's pool is untouched. The cost: extra threads sitting around (memory, context-switch overhead) and the handoff to another thread (a small latency hit, and you lose thread-local context unless you propagate it). This is the heavyweight, strongest-isolation option, and it is what Hystrix made famous.
- **Semaphore isolation** — instead of separate threads, you use a counting semaphore to cap how many calls to a dependency can be *in flight at once* on the *calling* thread. Call payment, acquire a permit; if no permit is available (already at the limit), fail fast. Lighter weight (no extra threads, no context handoff), but it does not protect against a call that blocks forever, because the call runs on the caller's thread — so it relies on a real timeout to release the permit. For async runtimes (Go goroutines, Node, async Python) where "threads" are cheap, semaphore-style concurrency limits are the natural fit.

Here is a Go semaphore bulkhead — a buffered channel as a counting semaphore that caps in-flight payment calls and fails fast when full, so payment can never consume more than its allotment of the order service's concurrency:

```go
// Bulkhead: cap concurrent payment calls at 20. The 21st fails fast.
var paymentSem = make(chan struct{}, 20) // buffered channel = semaphore

var ErrBulkheadFull = errors.New("payment bulkhead full, shedding load")

func chargeWithBulkhead(ctx context.Context, order Order) (*ChargeResult, error) {
	select {
	case paymentSem <- struct{}{}: // acquire a permit
		defer func() { <-paymentSem }() // release on return
	default:
		// No permit available right now: fail fast instead of queuing.
		// This is the bulkhead doing its job — payment is saturated,
		// but the order service keeps its other capacity free.
		return nil, ErrBulkheadFull
	}
	return chargeOrder(ctx, order) // the timeout-bounded call from earlier
}
```

For the JVM, resilience4j's `THREADPOOL` bulkhead gives true thread isolation:

```yaml
resilience4j:
  bulkhead:           # semaphore bulkhead (lightweight, caps concurrency)
    instances:
      catalogService:
        max-concurrent-calls: 40
        max-wait-duration: 10ms   # wait briefly for a permit, then reject
  thread-pool-bulkhead:  # thread-pool bulkhead (strong isolation)
    instances:
      paymentService:
        core-thread-pool-size: 20
        max-thread-pool-size: 20  # hard cap: payment never exceeds 20 threads
        queue-capacity: 10        # tiny queue, then reject
```

#### Worked example: the thread-exhaustion math, and how a 20-thread bulkhead contains it

This is the arithmetic that makes the bulkhead's value undeniable. Take the intro scenario precisely.

**Without a bulkhead.** The order service has a shared pool of 200 worker threads serving *all* its work — checkout, catalog reads, cart updates, everything. Payment slows to 5 seconds. Checkout traffic is 60 requests per second. Each checkout request holds a thread for the full 5 seconds waiting on payment. By Little's Law, concurrent checkout threads = `60 req/s × 5s = 300`. But the pool only has 200 threads. So within about 3.3 seconds (`200 / 60`), *every* thread in the pool is held by a checkout request blocked on payment. Now a catalog read comes in — it needs a thread, there are none, it queues, it eventually times out. The catalog service is *perfectly healthy*, but the order service cannot reach it because it has no threads left. The entire order service is dead, taking down catalog, cart, and search along with checkout. **One slow dependency killed everything because they shared a pool.**

**With a bulkhead.** Now payment gets its own dedicated pool of 20 threads (and a queue of 10). The other 180 threads serve everything else. Payment slows to 5 seconds. Checkout traffic is still 60 req/s, each holding a payment thread for 5 seconds → wants `60 × 5 = 300` concurrent payment slots, but the bulkhead caps it at `20 + 10 = 30`. The 31st concurrent checkout request hits a full bulkhead and **fails fast** — it gets `ErrBulkheadFull` in microseconds, the breaker (watching these rejections as failures) trips, and the user sees the "payment is pending" fallback. Meanwhile the *other 180 threads are completely free*: catalog reads, cart updates, and search all keep serving at full speed and full p99. The blast radius of the payment brownout is contained to checkout, exactly as a ship's bulkhead contains flooding to one compartment. Checkout is degraded; the rest of the site is *fine*. That is the entire point, and it is the difference between a customer who can't check out for a minute and a company that's entirely offline.

The cost is honest: you now have 20 threads reserved for payment that sit idle when payment is healthy and traffic is low. That is the price of isolation — some resource fragmentation, some idle capacity. For a critical dependency it is obviously worth it. For a rarely-called dependency you might use a small semaphore bulkhead instead of a dedicated pool. Sizing the pool is its own small optimization: too small and you reject healthy traffic during normal peaks; too large and the bulkhead does not actually bound anything. A good starting point is `pool_size ≈ peak_rate × healthy_p99_latency × safety_factor` — here, `60 req/s × 0.18s × 2 ≈ 22`, so 20–24 threads. We tune this with real numbers in the optimization section.

## Composing all four: the wrapped ShopFast call

Each pattern catches a failure the others let through, and the magic is in how they *compose* around a single call. The order matters. Reading from the outside in, here is the standard composition and why it is in this order:

![A vertical stack showing the layers wrapping one payment call from the outer timeout down through retry with jitter and budget then the circuit breaker then the bulkhead pool and finally the raw HTTP call](/imgs/blogs/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads-1.webp)

1. **Timeout (outermost per-attempt bound)** — bounds how long any single attempt can take. Without this nothing else works, because a hung call never reaches the retry or breaker logic.
2. **Retry with jitter + budget** — wraps the timeout: if an attempt times out or returns a transient error, retry it (subject to the budget). Each retry is itself a fresh timeout-bounded attempt.
3. **Circuit breaker** — wraps the retries: the breaker observes the outcomes (including timeouts and exhausted retries as failures) and, once the failure rate trips it, short-circuits *before* even attempting the call. This is why the breaker sits logically inside the retry but governs whether the attempt happens at all — an open breaker means "don't even try, fail fast."
4. **Bulkhead (innermost resource bound)** — caps concurrency: every actual attempt that the breaker permits must acquire a bulkhead permit first. If the pool is full, fail fast without even making the call.

There is genuine subtlety in the exact nesting — for instance, you generally want the breaker to *not* count bulkhead-full rejections and deadline-exceeded-before-call as the dependency's failures in the same way as real downstream errors, or you can get a breaker that trips on the caller's own saturation rather than the dependency's sickness. The resilience4j default decorator order (`Bulkhead → TimeLimiter → RateLimiter → CircuitBreaker → Retry`, applied inside-out) encodes a sensible version of this; the key intuition to carry is *timeout bounds the attempt, retry repeats the attempt, breaker decides whether to attempt, bulkhead caps concurrent attempts*.

```python
# Composed call (pseudocode showing the nesting clearly).
def charge_resilient(order):
    if breaker.is_open():                 # 3. breaker: fail fast if open
        return fallback(order)
    if not bulkhead.try_acquire():        # 4. bulkhead: fail fast if full
        breaker.record_rejection()
        return fallback(order)
    try:
        # 1+2. retry-with-jitter, each attempt timeout-bounded:
        result = call_with_backoff(
            lambda: charge_with_timeout(order, timeout_ms=250),
            budget=retry_budget,
        )
        breaker.record_success()
        return result
    except RetriesExhausted:
        breaker.record_failure()          # feeds the breaker's window
        return fallback(order)
    finally:
        bulkhead.release()
```

![A two column before and after comparison contrasting the naive call with infinite timeout and shared threads that dies with payment against the wrapped call with a tight timeout bulkhead isolation and a breaker that shows a fallback](/imgs/blogs/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads-3.webp)

## Where to put the patterns: in-code library or in the mesh

You have two places to implement these patterns, and a mature system uses both — they are not either/or.

**In-code libraries** — resilience4j (JVM), Polly (.NET), Hystrix (the original, now in maintenance), `gobreaker` and `go-resilience` (Go), `tenacity` (Python retries), `opossum` (Node). These run *inside* your application process, which means they have full *business context*: they can express a fallback that returns a domain-specific degraded response, they can be idempotency-aware, they can compute a deadline budget that knows what work remains. The cost is that every service in every language must implement them, and they are only as consistent as your discipline.

**The service mesh / sidecar** — Envoy, Istio, Linkerd. These run timeouts, retries, and outlier detection (a network-level circuit breaker) *outside* your process, in a sidecar proxy that intercepts all traffic. The huge advantage: it is *language-agnostic* and *configured declaratively*, so a platform team can set sane per-route timeouts and retry policies for every service without touching application code. The cost: the sidecar only sees transport-level signals (HTTP status, connection errors, latency); it cannot know that a charge is non-idempotent or compute a business fallback. So the division of labor is clear.

![A grid mapping the division of labor where the in app library owns deadline budgets idempotency keys and business fallbacks while the mesh sidecar owns transport retries outlier ejection and per route timeouts](/imgs/blogs/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads-9.webp)

Here is the mesh doing the transport-layer work for ShopFast: an Istio `VirtualService` setting a per-route timeout and a *transport-level* retry policy, plus a `DestinationRule` with **outlier detection** — Envoy's network circuit breaker that ejects an unhealthy backend pod from the load-balancing pool after consecutive errors:

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: payment-route
spec:
  hosts: ["payment-svc"]
  http:
    - route:
        - destination: { host: payment-svc }
      timeout: 600ms              # per-request timeout at the mesh
      retries:
        attempts: 2               # transport retry, ONLY on these conditions
        perTryTimeout: 250ms
        retryOn: connect-failure,refused-stream,unavailable,5xx
        # NOTE: keep this OFF if the app already retries — avoid stacking layers
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: payment-outlier
spec:
  host: payment-svc
  trafficPolicy:
    connectionPool:
      tcp: { maxConnections: 100 }       # bulkhead: cap connections
      http: { http2MaxRequests: 200, maxRequestsPerConnection: 10 }
    outlierDetection:                    # network circuit breaker
      consecutive5xxErrors: 5            # 5 errors from a pod...
      interval: 10s                      # ...checked every 10s...
      baseEjectionTime: 30s              # ...ejects it for 30s (then probes)
      maxEjectionPercent: 50             # never eject more than half the pool
```

The rule of thumb: **let the mesh own transport-level retries, timeouts, and outlier ejection (the stuff that needs no business context and benefits from being uniform), and keep deadline budgets, idempotency keys, and business fallbacks in the application.** The one trap to avoid is *double retries* — if both the mesh and the app retry, you have re-created the retry-storm amplification at the infrastructure layer. Pick one layer to retry and turn the other off. When and whether you need a mesh at all is a real decision with real operational cost, covered in [service mesh, Istio, Linkerd, when you need one](/blog/software-development/microservices/service-mesh-istio-linkerd-when-you-need-one). For where these patterns sit relative to load balancing and discovery, see [service discovery and load balancing](/blog/software-development/microservices/service-discovery-and-load-balancing).

## Trade-offs: a decision matrix for the four patterns

No single pattern covers every failure, and each one costs something and backfires in a specific way. This is the section to internalize, because the most common mistake is reaching for one pattern (usually retries) and thinking you are done.

![A decision matrix comparing timeout retry circuit breaker and bulkhead across what each protects against what it costs and when it backfires](/imgs/blogs/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads-7.webp)

| Pattern | Protects against | What it costs | Backfires when |
|---|---|---|---|
| **Timeout** | Unbounded waits; thread exhaustion from a slow dependency | May cut off a legitimately-slow-but-fine call | Set too tight (manufactures errors) or, worse, not set at all (the original sin) |
| **Retry + backoff + jitter** | Transient failures: dropped packets, brief blips, momentary `503`s | Extra latency on the retried call; risk of amplification | The op is non-idempotent (double effects), or many layers retry (retry storm), or no budget caps it |
| **Circuit breaker** | Hammering a persistently sick dependency; slow cascades; wasted timeouts | Tuning effort; a control loop that can oscillate | Thresholds are wrong → flapping, or too conservative → trips too late to help |
| **Bulkhead** | One slow dependency draining the whole service's thread/connection pool | Reserved/idle threads; resource fragmentation | Pools sized too small (rejects healthy peaks) or too large (bounds nothing) |

A few decision heuristics that fall out of this table:

- **Timeouts are mandatory and nearly free** — there is no scenario where an unbounded remote call is correct. Set one on every remote call, full stop. This is the one pattern with no real "when not to."
- **Retries are valuable but conditional** — only for transient failures on idempotent operations, with jitter and a budget. If you cannot satisfy those preconditions, do not retry; surface the error or use a fallback.
- **Circuit breakers earn their keep when a dependency can be *persistently* sick** — i.e., almost always for a synchronous dependency. They are most valuable in front of dependencies that recover slowly (databases, external APIs). For an internal stateless service that restarts in 2 seconds, a breaker still helps but the cooldown should be short.
- **Bulkheads matter when one service calls *multiple* dependencies** and you cannot afford one slow dependency to starve calls to the healthy others. A service that calls only one downstream gets less from a bulkhead (though it still protects the service from being entirely consumed).

The composition is the answer, not any single pattern: timeout + retry + breaker + bulkhead, each tuned for the specific dependency. A senior does not ask "should I use a circuit breaker?"; they ask "what is each of my four defenses set to for *this* dependency, and have I tested what happens when it goes slow?"

## Optimization: making it production-grade with numbers

Getting the patterns *in place* is junior work. Getting them *tuned* so they help instead of hurt is the senior work, and it is all about numbers. Here is where the real bottlenecks are and how to measure the win.

### Tuning the circuit breaker so it does not flap

The flapping failure from earlier is fixable with three knobs, and you tune them against the dependency's actual recovery behavior:

- **Window size and minimum calls.** Too small a window and the breaker reacts to noise; too large and it reacts too slowly. For a service at 100 req/s, a 100-call (or 10-second) window with a 20-call minimum gives a meaningful sample that reacts within ~1 second. Measure: the breaker should trip within 1–2 seconds of a real outage starting, and *not* trip on a transient 3-failure blip.
- **Cooldown (wait-in-open).** This must be at least as long as the dependency's realistic recovery time. If payment recovers by a pod restart that takes 20 seconds, a 5-second cooldown will half-open into a still-restarting pod and flap. Set the cooldown to ~30s (covering the restart plus warmup) and the breaker probes only after there is a real chance of success. Measure: count breaker state transitions per minute during an incident; a healthy breaker transitions a handful of times, a flapping one transitions dozens.
- **Half-open probe count.** One probe is too few — a single lucky success closes the breaker prematurely. Require 5 successful probes before closing, so the breaker only closes when the dependency is *consistently* healthy again. Measure: false-close rate (breaker closes then immediately re-opens) should be near zero.

#### Worked example: tuning the breaker against measured recovery

Payment's incident profile, measured from past outages: when it gets sick, it stays sick for 45–90 seconds (an index rebuild or a pod restart cycle), then recovers cleanly. During sickness, ~80% of calls fail or are slow.

Set the window to 100 calls, minimum 20, failure-rate threshold 50%. At 100 req/s with 80% failing, the breaker accumulates `100 × 0.8 = 80` failures in the window within ~1 second of the incident — well over the 50% threshold — so it trips in about 1 second. Good. Set the cooldown to 30 seconds: the breaker opens, sends zero traffic to payment for 30 seconds (giving it room), then half-opens. But payment's outage lasts 45–90 seconds, so the *first* half-open probe round (at T+30s) will likely still fail → snap back to open for another 30 seconds → half-open again at T+60s. If payment is still down at T+60s, fail again; if it recovered (the 45–90s window), the 5 probes succeed and the breaker closes. The system self-heals within ~30 seconds of payment's actual recovery, with payment receiving only `2 rounds × 5 probes = 10` probe requests during the whole outage instead of the `90s × 100 req/s = 9,000` requests it would have received with no breaker. That is a 900× reduction in load on the sick dependency, which is exactly the quiet it needs to recover — and the measurable win is "payment recovered in 60s instead of never, because we stopped burying it."

### Deadline propagation as a load-shedding optimization

We introduced deadline propagation as a correctness feature, but it is also a *throughput* optimization under overload. When the system is overloaded, a large fraction of requests will blow their deadline somewhere in the chain. With propagation, every service downstream of the blown deadline returns instantly without doing its (expensive) work. Measure the win as wasted-work ratio: in one overload test on a ShopFast-like chain, without propagation roughly 35% of CPU was spent computing responses for requests whose client had already timed out; with propagation that dropped to under 5%, and effective goodput (successful responses per second) rose by ~25% because the freed capacity went to requests that could still finish. The cost is a header on every request and a deadline check at every service entry — negligible. This is the same family of idea as load shedding in [rate limiting, backpressure, and load shedding](/blog/software-development/microservices/rate-limiting-backpressure-and-load-shedding).

### Hedged requests: trading a little load for a lot of tail-latency

A more advanced optimization, from Google's "The Tail at Scale" paper: for read-only, idempotent calls where p99 latency matters more than a little extra load, send a **hedged request**. Fire the call to one replica; if it has not responded within, say, the p95 latency, fire a *second* copy to a different replica and take whichever returns first, cancelling the other. Because slow responses are usually due to a *specific* unlucky replica (a GC pause, a hot node), the second copy usually hits a healthy replica and returns fast. The result: p99 latency drops dramatically — Google reported tail latency cut by more than half in some services — for the cost of a small percentage of extra requests (you only hedge the slow ~5%, so load rises ~5%, not 100%). Hedging is *only* safe for idempotent operations (you are intentionally double-sending), so it pairs naturally with the idempotency machinery from [idempotency and exactly-once effects across services](/blog/software-development/microservices/idempotency-and-exactly-once-effects-across-services). Do **not** hedge a charge.

```python
async def hedged_get(do_call, hedge_after_ms=90, replicas=2):
    "Fire a copy to a second replica if the first is slow. Reads only!"
    first = asyncio.create_task(do_call())
    done, pending = await asyncio.wait({first}, timeout=hedge_after_ms / 1000)
    if done:
        return first.result()           # first replica was fast; done
    second = asyncio.create_task(do_call())  # first is slow: hedge it
    done, pending = await asyncio.wait(
        {first, second}, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()                       # cancel the loser
    return next(iter(done)).result()
```

### Sizing the bulkhead pool from real traffic

Right-size each bulkhead from measured numbers, not guesses. The pool must hold the *healthy* concurrency with headroom but cap the *unhealthy* concurrency. Use `pool_size = peak_rate × healthy_p99_seconds × headroom`. For payment at peak 60 req/s, healthy p99 of 180ms, and 1.5× headroom: `60 × 0.18 × 1.5 ≈ 16`, round up to 20. Then verify the cap holds under failure: at the 250ms timeout, a saturated bulkhead of 20 sustains `20 / 0.25 = 80` req/s of doomed-call throughput, all failing fast, while the breaker trips and offloads them to the fallback. The other 180 threads stay free. Measure the win in a load test: with the bulkhead, the 99th-percentile latency of *non-payment* endpoints during a payment brownout should stay flat (within noise of baseline); without it, they spike to the timeout and then to errors. If non-payment p99 is flat during a payment brownout, the bulkhead is doing its job.

## Stress-testing the design

The kit demands it and good engineering demands it: pose the failures and walk through whether the design survives. This is the problem-solving narrative — design, then break it on purpose.

**Stress test 1: "Payment slows to 5 seconds — does the whole site go down?"** This is the intro outage, re-run against the wrapped call. Payment p99 goes to 5s. (a) The **timeout** fires at 250ms, so no checkout thread waits 5 seconds — each blocked attempt frees its thread in 250ms instead. (b) The **bulkhead** caps concurrent payment attempts at 20+10, so checkout can consume at most 30 of the order service's threads; the other 180 keep serving catalog, cart, and search at full p99. (c) Within ~1 second the **breaker** sees 50%+ slow-call rate and opens, so subsequent checkout requests fail fast into the "payment pending" **fallback** in microseconds, not even consuming a bulkhead slot. Result: checkout is degraded (users see "payment processing, we'll confirm shortly"), the rest of the site is *completely unaffected*, and payment receives near-zero traffic so it can recover. The site does **not** go down. Compare to the naive version where the same brownout took down the homepage in 12 minutes. The patterns turned a company-wide outage into a localized, graceful degradation.

**Stress test 2: "Retry storm during a brownout."** Payment is in a brownout returning intermittent `503`s. Without protection, the 3-layers-×-3-retries amplification puts 27× load on payment, burying it. With the design: retries happen at *only* the payment client (mesh and gateway retries disabled), capped by a **retry budget** at 10% of traffic. So during the brownout where everything is failing, the retry budget drains within a second and retries are *suppressed* — payment sees ~1.1× load, not 27×. Combined with the breaker opening, payment's incoming traffic actually *drops* during the incident, which is the opposite of the storm. The brownout recovers instead of escalating.

**Stress test 3: "A dependency is deployed mid-request / one pod is bad."** During a rolling deploy, one payment pod comes up unhealthy and returns errors while the others are fine. The application-level breaker, which sees *aggregate* payment health, might not trip (90% of calls succeed). But the mesh's **outlier detection** ejects the single bad pod from the load-balancing pool after 5 consecutive errors, so traffic routes only to healthy pods, and the bad pod gets a 30-second ejection to recover or be replaced. This is why per-pod outlier ejection (mesh layer) and per-dependency circuit breaking (app layer) are *complementary*: the breaker handles "the whole dependency is sick," the outlier detector handles "one instance is sick." Together they cover both granularities.

**Stress test 4: "Network partition — the dependency is unreachable, not slow."** Connection attempts are refused or time out at the connect layer. The **connection timeout** (100ms) fires fast, so attempts fail in 100ms not infinitely. The **breaker** trips quickly (100% failure rate). The **fallback** engages. Because the failures are *fast* (connection refused) rather than *slow* (hung), the bulkhead barely fills — fast failures are far gentler on resources than slow ones, which is the silver lining of a hard partition versus a brownout. The system degrades cleanly and recovers automatically when the partition heals and the breaker's half-open probes succeed.

## Case studies

### Netflix Hystrix and the popularization of the bulkhead

Netflix built and open-sourced **Hystrix** around 2012 as it moved onto AWS and discovered that its sprawling fleet of services failed in exactly the cascading way described in this post: one slow dependency consuming all of a calling service's threads. Hystrix is the library that made circuit breakers, thread-pool bulkheads, and fallbacks mainstream practice. Its core design decision is instructive: by default, Hystrix isolated each dependency in its *own thread pool*, accepting the overhead of extra threads precisely so that one slow dependency could not starve the rest — the thread-pool bulkhead, applied by default to every dependency. Netflix's data showed that the thread isolation overhead was a few milliseconds at the p99, a price they considered obviously worth paying to never have a single slow dependency cascade across the fleet. Hystrix is now in maintenance mode (Netflix moved to adaptive concurrency limits and resilience4j-style approaches, and to pushing more of this into the platform), but its conceptual contribution — *isolate, fail fast, fall back* — is permanent, and resilience4j is its spiritual successor on the JVM.

### The AWS retry-storm post-mortems

Amazon's engineers have written publicly, in the Amazon Builders' Library and in Marc Brooker's well-known posts on timeouts, retries, and backoff, about how retry amplification has repeatedly turned recoverable degradations into large outages. The recurring lesson, drawn from real internal incidents: a brief dependency slowdown causes timeouts, timeouts trigger retries, retries multiply the load on the already-struggling dependency, and the system enters a state where it cannot recover *even after the original trigger is gone*, because the retry load is now self-sustaining (a *metastable* failure). AWS's prescribed fixes are precisely the ones in this post: exponential backoff *with jitter* (the "full jitter" recommendation comes from this work), retry budgets / token buckets to cap retries as a fraction of traffic, and retrying at only one layer. The "Timeouts, retries, and backoff with jitter" article is required reading and is in the further-reading list.

### A circuit breaker that saved a checkout flow

A pattern reported across many engineering blogs (Shopify, DoorDash, and others have variants of this story): a payment or fraud-check dependency degraded during a peak shopping event, and the team that had a *correctly tuned* circuit breaker in front of it survived where they otherwise would not have. The mechanism each time: the breaker tripped within seconds of the dependency going slow, the calling service immediately stopped sending it traffic and served a fallback (queue the payment for async processing, or proceed with a relaxed fraud check under a risk threshold), checkout stayed up at a degraded-but-functional level, and the dependency — no longer being hammered — recovered on its own. The contrasting failures, where teams *lacked* a breaker or had one tuned to trip too late, show the same dependency taking down the whole checkout flow. The lesson is not "circuit breakers are magic"; it is "a circuit breaker *with a real fallback and correct tuning* converts a dependency outage into a degradation, and one without a fallback just converts slow errors into fast ones." The fallback is where the survival actually lives.

### Stripe, idempotency, and safe retries

Stripe's public API design is the canonical example of making retries *safe* at the protocol level: every mutating Stripe API call accepts an `Idempotency-Key` header, and Stripe guarantees that two requests with the same key produce the same result — the second is recognized as a retry and returns the original outcome rather than charging again. This is the precondition that makes client-side retries on a *payment* (normally the most dangerous thing to retry) actually safe. It is the reason a well-built ShopFast payment client can retry a charge at all. The deep mechanics — how the key is stored, how concurrent requests with the same key are serialized, the TTL — are exactly the subject of [idempotency and exactly-once effects across services](/blog/software-development/microservices/idempotency-and-exactly-once-effects-across-services), and the broader saga-level version appears in [the saga pattern in practice](/blog/software-development/microservices/the-saga-pattern-in-practice). The takeaway here: retries and idempotency are a *package*; you earn the right to retry a write by making it idempotent first.

## When to reach for this (and when not to)

These patterns are not optional decoration; for any synchronous remote call in a microservices system, the baseline is non-negotiable, and the nuance is in how much you layer on.

- **Always, no exceptions: set a timeout on every remote call** (HTTP, gRPC, database, cache, queue). The unbounded call is the single most common root cause of cascading outages. There is no scenario where infinite is the right timeout.
- **Always for synchronous cross-service calls: a circuit breaker + fallback.** Any service you call synchronously can be persistently sick, and the breaker-plus-fallback is what turns that from your outage into a degradation. The fallback is the part that takes real thought.
- **Bulkhead when a service calls multiple dependencies** and you cannot let one slow one starve the others — which is most services. For a service with a single downstream, a bulkhead still bounds total concurrency but the isolation benefit is smaller.
- **Retry only when you can satisfy both preconditions** — transient failure class *and* idempotent operation — with jitter and a budget. If you cannot, do not retry; surface the error or fall back. A retry you cannot justify is a latent outage.
- **The "retries made it worse" warning.** If you remember one cautionary thing from this post: *retries are the pattern most likely to cause the outage you are trying to prevent.* The default instinct — "the call failed, just retry" — is exactly how a small brownout becomes a 27× retry storm. Treat every retry config as a potential amplifier and gate it behind backoff, jitter, a budget, single-layer placement, and idempotency. When in doubt, retry *less*.
- **When NOT to add complexity:** for an *asynchronous* path (publish an event to a queue and move on), most of this changes shape — the broker's redelivery and your consumer's idempotency do the work, and synchronous timeouts/breakers do not apply the same way; see the message-queue series on [delivery semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) and [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe). And for a tiny internal tool with one dependency and no real traffic, a single timeout may be all the resilience engineering you need — do not build a fully tuned breaker-and-bulkhead rig for a cron job that runs once a night. Match the investment to the blast radius.

## Key takeaways

1. **A slow dependency is more dangerous than a dead one.** A dead dependency fails fast; a slow one holds your threads hostage until you run out. The whole point of these patterns is to enforce a *latency* budget, not just catch errors.
2. **Every remote call gets an explicit timeout — the default is almost always infinite, and infinite is a loaded gun.** Pick the timeout from the latency histogram (≈ p99.9), and verify with Little's Law that a full brownout cannot exceed your thread pool.
3. **Propagate a shrinking deadline budget down the whole chain.** One request, one bound; every hop subtracts its share; an expired deadline means stop working immediately, all the way down — which is also free load shedding under overload.
4. **Retry only transient failures on idempotent operations, with exponential backoff *and* jitter.** Jitter prevents the synchronized thundering herd; without it, recovery itself triggers the next outage.
5. **Cap retries with a budget, and retry at only one layer.** N layers each retrying 3× is 3^N amplification — 27× for three layers — and a retry storm is how a recoverable brownout becomes an unrecoverable outage. A 10% retry budget bounds it to ~1.1×.
6. **A circuit breaker trips on failure *rate* (including slow calls), fails fast while open, and self-heals via a half-open probe.** Tune the window, cooldown, and probe count against the dependency's real recovery time, or it flaps and never lets the dependency rest.
7. **Bulkhead your resources — one bounded pool per dependency — so one slow downstream cannot drain the whole service.** Size the pool from `peak_rate × healthy_p99 × headroom`, and measure success as "non-dependency p99 stays flat during the dependency's brownout."
8. **Compose all four around each call**: timeout bounds the attempt, retry repeats it, breaker decides whether to attempt, bulkhead caps concurrent attempts — each tuned per dependency.
9. **Split the work between code and mesh:** let the mesh own transport retries, per-route timeouts, and outlier ejection; keep deadline budgets, idempotency, and business fallbacks in the app. Never let both layers retry.
10. **The fallback is where survival lives.** A breaker without a meaningful fallback just turns slow errors into fast errors; a breaker *with* a graceful degraded response turns a dependency outage into a non-event for most users.

## Further reading

- Michael Nygard, *Release It! Design and Deploy Production-Ready Software* (2nd ed.) — the origin of the bulkhead and circuit-breaker patterns as practitioner advice; the single best book on this topic.
- Sam Newman, *Building Microservices* (2nd ed.) — the resilience chapter places these patterns in the broader microservices context.
- Chris Richardson, *Microservices Patterns* — the reliability and observability chapters, plus the saga and idempotency connections.
- Marc Brooker / Amazon Builders' Library, "Timeouts, retries, and backoff with jitter" — the definitive treatment of why naked retries cause outages and the full-jitter and retry-budget fixes.
- Jeffrey Dean and Luiz André Barroso, "The Tail at Scale" (CACM, 2013) — hedged requests and tail-latency techniques.
- resilience4j and Polly documentation — the reference implementations of every pattern here, with the exact config knobs.
- Netflix Technology Blog on Hystrix and adaptive concurrency limits — the production story behind the thread-pool bulkhead.
- Series siblings: [inter-service communication fundamentals and fallacies](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies), [service discovery and load balancing](/blog/software-development/microservices/service-discovery-and-load-balancing), [the saga pattern in practice](/blog/software-development/microservices/the-saga-pattern-in-practice), and the forward-looking [handling partial failures and graceful degradation](/blog/software-development/microservices/handling-partial-failures-and-graceful-degradation), [idempotency and exactly-once effects across services](/blog/software-development/microservices/idempotency-and-exactly-once-effects-across-services), [rate limiting, backpressure, and load shedding](/blog/software-development/microservices/rate-limiting-backpressure-and-load-shedding), and [service mesh, Istio, Linkerd, when you need one](/blog/software-development/microservices/service-mesh-istio-linkerd-when-you-need-one).
