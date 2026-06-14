---
title: "Observability for Async Systems: Tracing Across the Queue Boundary"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Async messaging silently breaks the end-to-end trace you get for free with request/response — learn to propagate W3C trace context through message headers, link the consumer span to the producer span instead of nesting it, and wire up the metrics, logs, and traces that let you answer where did my message go."
tags:
  [
    "message-queue",
    "observability",
    "distributed-tracing",
    "opentelemetry",
    "kafka",
    "rabbitmq",
    "distributed-systems",
    "event-driven",
    "trace-context",
    "correlation-id",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/observability-tracing-across-the-queue-boundary-1.webp"
---

The first time it happens to you it feels like a betrayal. You have spent years building the muscle memory of distributed tracing: a request comes in, you open the trace, and there it is — one waterfall, top to bottom, every database call and downstream service nested neatly under the root span, p99 latencies attached to each hop, a flame graph that tells you exactly where the 800ms went. Then your team adopts a message queue to decouple the checkout service from the fulfillment service, and the very next incident the trace just... stops. The web request returns `202 Accepted` in 4 milliseconds, the trace ends right there at the enqueue, and the actual work — the part that failed — happened three seconds later in a worker that, as far as your tracing backend is concerned, lives in a completely different universe. There is a brand-new trace for the consumer, with a brand-new trace id, utterly disconnected from the request that caused it. You cannot follow your unit of work across the boundary. The single most useful tool you had for debugging distributed systems just evaporated the moment you introduced asynchrony.

This is not a tooling bug. It is the natural and correct default behavior of every tracing system when you cross a queue, because the queue deliberately severs the synchronous call relationship that tracing relies on to stitch spans together. Synchronous RPC propagates trace context for you almost invisibly — the HTTP client library or gRPC interceptor injects a `traceparent` header on the outbound call, the server extracts it on the way in, and the parent-child relationship is established before your code ever runs. A message queue has no such convention out of the box. You call `producer.send()`, the message lands in a topic, and minutes or hours later a consumer in a different process, on a different host, in a different deployment, picks it up. Nothing carried the trace context across, so nothing can reconnect the two halves. Figure 1 shows the two worlds side by side: the broken trace with a gap at the queue, and the repaired trace where context rode through the message and the spans are linked.

![Side-by-side comparison of a broken trace with a gap at the queue versus a context-propagated trace whose spans are linked across the boundary](/imgs/blogs/observability-tracing-across-the-queue-boundary-1.webp)

The good news is that this is a solved problem, and the solution is precise and unglamorous: you propagate trace context *through the message itself*, you create the consumer span as a *link* to the producer span rather than a child of it, and you build per-role metrics and correlated logs around the traces so that when something goes wrong you can answer the three questions every async incident reduces to — is it broken, where did it stop, and why. By the end of this post you will be able to instrument a producer and consumer to carry W3C trace context across a Kafka or RabbitMQ boundary, decide correctly between span links and parent-child relationships, design the metrics dashboard that catches an async failure before your users do, correlate logs by trace id and message id, and run the "where did my message go" investigation methodically instead of by panic and grep. We will use the OpenTelemetry messaging semantic conventions as our north star, because they are the closest thing the industry has to a shared contract, and because building on them means your traces interoperate with every modern backend.

## 1. Why async breaks observability

To understand why the trace breaks, you have to understand what a trace actually *is* underneath the pretty waterfall UI. A trace is a tree of spans that all share a single `trace_id`. Each span has its own `span_id`, and each non-root span records the `span_id` of its parent. The tracing backend reconstructs the waterfall by grouping every span with the same `trace_id` and then arranging them by the parent pointers. That is the entire model. There is no magic — the only thing that connects a span in service A to a span in service B is that B's span carries A's `trace_id` and points its parent at A's `span_id`. How does B get those two values? Somebody had to *transmit* them from A to B. In synchronous HTTP, the transmission channel is the request headers: A's tracing instrumentation writes `traceparent: 00-<trace_id>-<span_id>-01` into the outbound request, and B's instrumentation reads it back out. The trace is unbroken because the context traveled with the call.

Now look at what an async hop does to that chain. The producer calls `send()` and gets back a future that resolves when the broker acknowledges the write — but that acknowledgement carries no information about who will eventually consume the message, when, or where. The producer's span ends. From the tracing system's point of view, the request is *complete*: it returned `202`, the span closed, the trace is done. The message sits in the broker. Some unknown time later — could be 5 milliseconds, could be 5 hours if the consumer was down — a consumer polls, gets the message, and starts processing. Its tracing instrumentation, having no `traceparent` to extract (because nothing put one in the message), does the only sensible thing it can: it starts a fresh root span with a *new* `trace_id`. Two traces now exist for one logical unit of work, sharing nothing, and no query on earth will join them because there is no shared key.

### The request/response observability story you took for granted

Request/response gives you four things almost for free, and async takes all four away unless you actively rebuild them. First, *causal linkage*: the call stack literally encodes who called whom, so the trace tree falls out of the wire protocol. Async severs the call stack — the producer does not "call" the consumer, it drops a message and walks away. Second, *temporal locality*: in synchronous RPC the whole operation happens within one bounded window, usually milliseconds, so a span's start and end bracket the real work. Async smears the work across an unbounded window — the message can wait in the queue far longer than it takes to process, and that wait is invisible unless you measure it. Third, *backpressure as a signal*: when a synchronous downstream is slow, the caller blocks and you see the latency immediately in the caller's span. Async absorbs the slowness into queue depth — the producer never feels it, so the symptom hides in a metric (lag) that you only see if you are watching for it. Fourth, *error propagation*: a synchronous failure throws back up the stack to the caller, who logs it in context. An async failure happens in a worker the original caller has long forgotten about, so the error lands in a log file with no connection to the request that triggered it.

These are not minor inconveniences. Each one is a load-bearing assumption in how engineers reason about systems, and decoupling kicks all four out from under you simultaneously. That is *why* teams that adopt messaging for the very real benefits — load leveling, decoupling, elasticity, the things covered in [Message queues: async decoupling and load leveling](/blog/software-development/message-queue/message-queues-async-decoupling-and-load-leveling) — so often find their on-call experience gets dramatically worse before it gets better. The benefits are architectural and the cost is observational, and the cost is invisible until 2am.

### The shape of the rebuild

The fix has three layers, and the rest of this post is essentially a tour of them. Layer one is *trace context propagation*: put the `traceparent` into the message so the consumer can reconnect to the producer. Layer two is *the right span relationship*: link rather than nest, because the temporal smear means a parent-child relationship would lie about latency. Layer three is *the three pillars working together*: metrics to know something is wrong and how bad, logs correlated by ids to know why, and traces with span links to know where. Nail all three and you get back the followable unit of work — not identical to the synchronous waterfall, because the physics genuinely are different, but good enough that an incident is once again a methodical investigation instead of a séance.

## 2. The broken trace at the queue boundary

Let us make the breakage painfully concrete, because the precise mechanism matters for the fix. Here is a producer that does the naive thing — sends a message with no context propagation — and a consumer that processes it. Both are instrumented with OpenTelemetry, both create spans, and yet the spans will never join.

```python
# producer.py — the BROKEN version (no context propagation)
from opentelemetry import trace
from kafka import KafkaProducer
import json

tracer = trace.get_tracer("checkout")
producer = KafkaProducer(
    bootstrap_servers="broker:9092",
    value_serializer=lambda v: json.dumps(v).encode(),
)

def place_order(order):
    # This span belongs to the inbound HTTP request's trace.
    with tracer.start_as_current_span("publish order.created") as span:
        span.set_attribute("order.id", order["id"])
        # We send the value, but NOTHING about the active trace
        # context goes into the message. The trace dies here.
        producer.send("orders", value=order)
    # HTTP handler returns 202 here; the request trace ends.
```

```python
# consumer.py — the BROKEN version (starts a fresh, disconnected trace)
from opentelemetry import trace
from kafka import KafkaConsumer
import json

tracer = trace.get_tracer("fulfillment")
consumer = KafkaConsumer(
    "orders",
    bootstrap_servers="broker:9092",
    value_deserializer=lambda b: json.loads(b.decode()),
    group_id="fulfillment",
)

for msg in consumer:
    # No traceparent to extract, so this starts a NEW root span
    # with a NEW trace_id. It will never join the producer's trace.
    with tracer.start_as_current_span("process order.created") as span:
        span.set_attribute("order.id", msg.value["id"])
        fulfill(msg.value)
```

Trace this order through your backend and you will find two traces. The first, rooted at the HTTP request, contains the `publish order.created` span and nothing downstream of it — it looks like a complete, fast, successful request, because from the web tier's perspective it was. The second, rooted at `process order.created`, floats free with its own trace id; if the worker threw an exception, the error is recorded *on that orphan trace*, with no breadcrumb pointing back to the order, the customer, or the request. When a customer emails support saying "I placed an order and it never shipped," you have the order id, and you can grep for it, but the trace id you would normally pivot on connects to only half the story.

### Why "just use the same trace id" is harder than it sounds

The instinct is right — you want both halves under one trace id — but the implementation has a subtlety that trips people up. You cannot simply read the producer's trace id and stuff it into the message body as a field, because a trace id alone is not enough to *reconstruct the span relationship*. The consumer needs the producer's `span_id` too, plus the trace flags (was this trace sampled?), and it needs them in the standardized W3C format so that any backend understands them. Hand-rolling this by copying a UUID into a JSON field gives you a correlation key — better than nothing, and we will see correlation ids do real work later — but it does not give you a *trace*. A trace needs the full context, formatted per the W3C Trace Context spec, transmitted through a channel the consumer's instrumentation knows to look at. That channel, for messaging, is the message *headers* — the per-message metadata fields that brokers carry alongside the payload but separate from it. Kafka has record headers, AMQP has message headers, SQS has message attributes, NATS has headers. Every serious broker has somewhere to put metadata that is not the body, and that is exactly where trace context belongs.

### The boundary is a serialization boundary

There is a deeper reason headers are the correct home for context. The boundary between producer and consumer is fundamentally a *serialization* boundary: everything that crosses it has to be turned into bytes and back. In-process context — the thread-local or async-local that OpenTelemetry uses to track "the current span" — does not survive serialization. The producer's current-span context lives in process memory and is gone the instant `send()` returns. So the act of crossing the queue is exactly the act of *losing* in-memory context, which is why you must explicitly serialize the context into the message and deserialize it on the other side. This is the same reason the [transactional outbox pattern](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing) exists for reliability: crossing an async boundary forces you to make explicit what synchronous code keeps implicit. Tracing across a queue is the observability analog of the same discipline — make the implicit context explicit, write it into the bytes, read it back on the far side.

There is one more reason headers, specifically, rather than the message body. A great deal of async tooling treats the message body as an opaque, schema-governed contract — your Avro or Protobuf schema (see [Schema management and evolution](/blog/software-development/message-queue/schema-management-evolution-avro-protobuf-registry)) defines exactly what fields the payload carries, and a schema registry enforces it. If you smuggle trace context into the body, you have polluted the business schema with infrastructure concerns, you have to evolve the schema every time the propagation format changes, and any consumer that validates strictly will reject your messages. Headers are explicitly the *out-of-band metadata channel* — they are where infrastructure concerns like content-type, encoding, retry count, and trace context belong, precisely so they can travel alongside the payload without contaminating it. Keeping trace context in headers means the body schema stays clean, the propagation format can evolve independently, and an old consumer that ignores headers still parses the body perfectly. This separation of concerns is not pedantry; it is what lets you turn tracing on for an existing topic without a coordinated schema migration across every producer and consumer.

## 3. Propagating trace context through message headers

Here is the repaired version, and the entire fix lives in two operations: `inject` on the producer side and `extract` on the consumer side. OpenTelemetry ships a *propagator* whose only job is to serialize the current trace context into a carrier (a key-value map) and to deserialize it back. For messaging, the carrier is the message headers. Figure 2 traces the path the context takes: the producer injects the `traceparent` into the headers, the broker carries those headers as opaque bytes without ever inspecting them, and the consumer extracts the context to link its span.

![Pipeline showing trace context injected into message headers, carried opaquely by the broker, then extracted at the consumer to link the span](/imgs/blogs/observability-tracing-across-the-queue-boundary-2.webp)

```python
# producer.py — FIXED: inject trace context into Kafka record headers
from opentelemetry import trace, propagate
from kafka import KafkaProducer
import json

tracer = trace.get_tracer("checkout")
producer = KafkaProducer(bootstrap_servers="broker:9092",
                         value_serializer=lambda v: json.dumps(v).encode())

def place_order(order):
    with tracer.start_as_current_span("orders publish") as span:
        span.set_attribute("messaging.system", "kafka")
        span.set_attribute("messaging.destination.name", "orders")
        span.set_attribute("order.id", order["id"])

        # Serialize the CURRENT trace context into a carrier dict.
        carrier = {}
        propagate.inject(carrier)   # writes "traceparent" (and "tracestate")

        # Kafka headers are list of (key, value-bytes) tuples.
        headers = [(k, v.encode()) for k, v in carrier.items()]
        producer.send("orders", value=order, headers=headers)
```

```python
# consumer.py — FIXED: extract context and create a LINKED span
from opentelemetry import trace, propagate
from opentelemetry.trace import SpanKind, Link
from kafka import KafkaConsumer
import json

tracer = trace.get_tracer("fulfillment")
consumer = KafkaConsumer("orders", bootstrap_servers="broker:9092",
                         value_deserializer=lambda b: json.loads(b.decode()),
                         group_id="fulfillment")

for msg in consumer:
    # Rebuild a carrier dict from Kafka headers, then extract context.
    carrier = {k: v.decode() for k, v in (msg.headers or [])}
    producer_ctx = propagate.extract(carrier)

    # Get the producer span context to LINK (not parent) to it.
    link = Link(trace.get_current_span(producer_ctx).get_span_context())

    with tracer.start_as_current_span(
        "orders process",
        kind=SpanKind.CONSUMER,
        links=[link],            # <-- the key line: link, do not nest
    ) as span:
        span.set_attribute("messaging.system", "kafka")
        span.set_attribute("messaging.destination.name", "orders")
        span.set_attribute("messaging.kafka.message.offset", msg.offset)
        span.set_attribute("order.id", msg.value["id"])
        fulfill(msg.value)
```

The `traceparent` header that `inject` writes looks like `00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01`. Reading it left to right: `00` is the version, the long hex string is the 16-byte `trace_id`, the shorter hex string is the 8-byte parent `span_id` (the producer's span), and `01` is the trace-flags byte whose low bit means "this trace was sampled, please record it." That single header is the whole W3C Trace Context core. There is a companion `tracestate` header that carries vendor-specific key-value pairs, which `inject` also populates if present; you should propagate it too, untouched, because dropping it loses sampling decisions and vendor metadata that downstream systems rely on.

### What the broker does with the headers: nothing, on purpose

The most important property of this scheme is that the broker is a *dumb pipe* for the headers. Kafka stores record headers as opaque byte arrays in the log alongside the value; it never parses `traceparent`, never validates it, never acts on it. RabbitMQ carries the `headers` table in the AMQP basic-properties; SQS carries message attributes; all of them treat your trace context as inert metadata to be faithfully reproduced on the consuming side. This is exactly what you want. It means trace propagation works uniformly across brokers, it means a broker upgrade cannot break your tracing, and it means the overhead is just the bytes — roughly 55 bytes for a `traceparent`, a rounding error against any real payload. The broker carries the context the same way a postal service carries the address on an envelope: it reads enough to route, and the trace context is not even that — it is a note to the recipient that the postal service never opens.

### Doing it the easy way: auto-instrumentation

You will rarely write the inject/extract by hand in production, because the OpenTelemetry instrumentation libraries for Kafka, RabbitMQ (via the AMQP client), and the major cloud queues do it automatically. In Java, adding the OpenTelemetry agent plus `opentelemetry-kafka-clients` makes the producer interceptor inject and the consumer interceptor extract, and you get linked spans for free. In Python, `opentelemetry-instrumentation-kafka-python` wraps the client similarly. The reason to understand the manual version anyway is twofold: first, auto-instrumentation routinely makes the *wrong* span-relationship choice by default — many instrumentations create the consumer span as a child rather than a link, which we will see distorts your latency numbers — and you need to know how to override it. Second, when you have a custom transport, a homegrown message envelope, or you batch-publish, the auto-instrumentation does not cover you and you fall back to inject/extract by hand. Know the primitive and the convenience layers stop being magic.

```java
// Java: manual inject using the TextMapPropagator and a Kafka Headers setter
TextMapSetter<Headers> setter = (carrier, key, value) ->
    carrier.add(key, value.getBytes(StandardCharsets.UTF_8));

try (Scope scope = span.makeCurrent()) {
    GlobalOpenTelemetry.getPropagators()
        .getTextMapPropagator()
        .inject(Context.current(), record.headers(), setter);
    producer.send(record);
}
```

### The same pattern across RabbitMQ, SQS, and NATS

The injection mechanism is identical across brokers because it is the *carrier* that differs, not the algorithm. On RabbitMQ over AMQP, the carrier is the `headers` table inside the message's basic-properties: you set `properties.headers["traceparent"]` on publish and read it on delivery. On Amazon SQS the carrier is the message *attributes* map, with the small wrinkle that SQS caps you at ten message attributes, so a chatty propagator that emits many keys can collide with the limit — keep the propagated set tight. On NATS, headers landed as a first-class feature and the carrier is the `Msg.Header` map, used exactly like HTTP headers. Google Pub/Sub uses the message `attributes` map. The point is that you write the propagator's carrier adapter once per broker — a setter for inject and a getter for extract — and from then on the W3C context rides every message regardless of transport. Below is the RabbitMQ extract side, which mirrors the Kafka one almost line for line.

```python
# RabbitMQ consumer: extract trace context from AMQP message headers
def on_message(channel, method, properties, body):
    carrier = properties.headers or {}        # AMQP headers table is the carrier
    producer_ctx = propagate.extract(carrier)
    link = Link(trace.get_current_span(producer_ctx).get_span_context())
    with tracer.start_as_current_span(
        f"{method.routing_key} process",
        kind=SpanKind.CONSUMER, links=[link],
    ) as span:
        span.set_attribute("messaging.system", "rabbitmq")
        span.set_attribute("messaging.destination.name", method.exchange)
        span.set_attribute("messaging.rabbitmq.destination.routing_key",
                           method.routing_key)
        span.set_attribute("messaging.message.id", properties.message_id)
        handle(body)
        channel.basic_ack(method.delivery_tag)
```

One broker-specific subtlety worth flagging: RabbitMQ messages carry a built-in `message_id` property, so you get a real message identifier for free and should populate it on publish; Kafka has no global message id, so you synthesize the coordinate from `(topic, partition, offset)`. That asymmetry shows up in your logs and your investigation queries, and it is the reason the RabbitMQ extract above can set `messaging.message.id` directly while the Kafka version sets `messaging.kafka.message.offset` instead. Whichever broker you run, the discipline is the same: make sure *some* stable per-message identifier is set, propagated, and logged on both sides, because that id is the anchor of every "where did it go" investigation.

## 4. Span links vs parent-child for async work

Here is the decision that separates a tracing setup that *technically works* from one that tells you the truth. Once the context is in the headers, you have two ways to wire the consumer span to the producer span: make it a *child* (parent-child, the normal RPC relationship) or make it a *link* (a typed cross-reference between spans that may belong to different traces or the same trace). Both keep the spans connected. Only one of them reports honest latencies. Figure 5 shows the structure: the HTTP request span is the parent of the producer span, and the producer span *links* to the consumer spans in the workers, which in turn parent their own database-write spans.

![Graph showing the producer span linking to asynchronous consumer spans in two workers, each parenting a database write span, forming an acyclic structure](/imgs/blogs/observability-tracing-across-the-queue-boundary-5.webp)

The problem with parent-child across a queue is *time*. A parent span's duration is supposed to *contain* its children — the parent starts, children happen inside that window, the parent ends. Tracing UIs and latency calculations bake this assumption in deeply: a child that starts after its parent has ended, or that runs for three seconds while the parent ran for three milliseconds, breaks the waterfall geometry. And that is exactly the async case. The producer span lasts a few milliseconds and ends when `202` is returned. The consumer span starts whenever the consumer gets around to it — 900 milliseconds later in a healthy system, possibly minutes later under lag — and runs for as long as processing takes. If you parent the consumer under the producer, you are claiming the producer span "contains" work that happened long after the producer finished. The waterfall renders nonsense: either the parent's duration gets stretched to swallow the queue wait (so your "publish" span now appears to take 940ms, which is a lie — publishing took 3ms), or the child dangles outside its parent's bar and the UI does something undefined.

### What a link actually means

A link says: "this span was *caused by* or *relates to* that span, but it is not nested under it." It is a sideways reference, not a vertical one. Crucially, a link does *not* make the linked span part of the consumer span's duration, and it does not force the two into the same trace timeline. The consumer span gets to be the root of its own trace (or part of the consumer's batch trace), with its own honest start and end times, and it carries a link pointing back at the producer span's context. When you open the consumer trace, the UI shows "this work was linked from trace X, span Y" and lets you jump to the producer's trace. When you open the producer trace, you can follow the link forward to every consumer that processed the message. You get the causal connection — the followable unit of work — *without* the lie that the producer span contains 900ms of queue wait it never experienced.

#### Worked example: why parent-child distorts the numbers

Suppose your orders pipeline runs at steady state with these real timings: HTTP handler 4ms, of which publishing to Kafka is 3ms; the message then waits in the partition for 896ms because your single consumer is keeping up but not instantly; the consumer processes in 35ms including a 12ms database write. The honest picture is two spans: a 4ms request and a 35ms consumer task, separated by ~900ms of queue time you can measure as lag. Now wire it parent-child. The consumer span (35ms, starting at T+900ms) becomes a child of the producer span (3ms, starting at T+3ms). A naive latency calculation that takes "root span duration" as the operation latency now reports either 3ms (ignoring everything downstream, dangerously optimistic) or, if your backend stretches parents to contain children, `940ms` for the publish — a number that will send an engineer hunting for a slow Kafka producer that does not exist. The actual bottleneck, the 896ms of queue wait, is a *consumer lag* problem, and no amount of staring at a distorted producer span will reveal it. Link the spans instead and each reports its true duration; the 896ms gap shows up where it belongs, as the delta between the producer span's end and the consumer span's start, and as a spike on your lag metric. The relationship you choose literally determines whether your latency numbers are true.

### When parent-child is actually fine

There is a legitimate exception, and it is worth stating so you do not over-apply the rule. If your messaging is being used as a *synchronous-style request/reply* — the caller sends a request and blocks waiting for a reply on a reply queue, as covered in [Request-reply over messaging](/blog/software-development/message-queue/request-reply-over-messaging-correlation-ids) — then the whole exchange genuinely is one bounded synchronous operation from the caller's perspective, and parenting the server-side span under the caller can be defensible because the temporal containment holds. But the moment the work is *fire-and-forget* or *fan-out* or *delayed*, which is the whole reason most teams reach for a queue, links are correct. The default mental rule: if the producer waits for the consumer's result, consider parent-child; if the producer drops the message and moves on, use a link. Most async messaging is the latter, so links should be your default and parent-child the exception you justify.

| Relationship | Same trace? | Effect on parent duration | Use when |
|---|---|---|---|
| Parent-child | Yes (child inherits trace id) | Child time counts inside parent | Synchronous request/reply, bounded window |
| Span link | No (each keeps own trace id) | None — durations independent | Fire-and-forget, fan-out, delayed work |
| Nothing (broken) | No | N/A — spans unrelated | Never, if you can help it |

## 5. The three pillars: metrics, logs, traces

Tracing is necessary but not sufficient. A trace tells you the path and the timing of *one* request you already decided to look at, but it cannot tell you that your error rate just tripled across all requests, and it is an expensive way to ask "is the system healthy right now." Real async observability is three pillars working in concert — metrics, logs, and traces — each answering a different question, and the art is knowing which pillar to reach for. Figure 4 lays out the mapping: which pillar answers "is it broken," "how bad," "where did it stop," and "why."

![Matrix mapping the three pillars metrics, logs, and traces against the operational questions each one answers best](/imgs/blogs/observability-tracing-across-the-queue-boundary-4.webp)

Think of it as a triage sequence. *Metrics* are your always-on, cheap, aggregated signal: they tell you *something* is wrong and roughly *how bad* — error rate up, consumer lag climbing, throughput dropped to zero. Metrics fire the alert. They are pre-aggregated time series, so they are cheap to store and fast to query, but they are also dimensionally limited: you cannot ask a counter "which specific message failed." *Traces* are your *where*: given that metrics say the consumer is slow, a trace of one slow message shows you that 896ms went to queue wait and 12ms to the database, so the problem is consumer throughput, not database performance. Traces are richly dimensional but expensive, so you sample them. *Logs* are your *why*: the trace points you at the span that failed, and the log line emitted within that span — correlated by trace id — carries the stack trace, the exception message, the variable values, the human-readable cause. Logs are the highest-cardinality, most detailed, most expensive-at-scale pillar.

### Why async needs all three more than synchronous systems do

In a synchronous system you can sometimes limp along with just logs and metrics, because the call stack gives you implicit tracing for free. In an async system the three pillars are not optional — they are how you reconstruct the unit of work that the architecture deliberately fragmented. Metrics per role catch the failure modes that hide in queue depth (a slow consumer never throws an error the producer can see). Traces with span links re-stitch the fragmented request. Logs correlated by trace id and message id let you join the breadcrumbs the workers scattered. Remove any one pillar and an async incident becomes guesswork: metrics alone tell you lag is high but not which messages or why; traces alone are too sparse to catch a rare failure and too expensive to keep all of; logs alone, without a correlation key, are a million disconnected lines you cannot join across services. The pillars are interlocking, and in async systems the interlock is load-bearing.

### The stack underneath the pillars

All three pillars share a delivery path, and it helps to see it as one stack so you provision it as one stack. Figure 7 shows the four layers every piece of telemetry flows through: instrumentation in your process emits the data, a collector receives and processes it, storage holds it, and a query layer lets you ask questions. The OpenTelemetry Collector sits at the collection layer and is doing more work than people realize — batching to reduce egress, tail-sampling to keep the interesting traces and drop the boring ones, enriching spans with resource attributes (which pod, which region, which deployment), and fanning out to multiple backends. Getting the collection layer right is where a lot of cost and signal is won or lost, because a trace you sampled away is a trace you cannot consult during the incident.

![Four-layer stack showing telemetry flowing from instrumentation up through collection, storage, and query](/imgs/blogs/observability-tracing-across-the-queue-boundary-7.webp)

```yaml
# otel-collector.yaml — tail-sampling so you KEEP the interesting async traces
processors:
  tail_sampling:
    decision_wait: 10s            # wait for late consumer spans to arrive
    policies:
      - name: keep-errors
        type: status_code
        status_code: { status_codes: [ERROR] }
      - name: keep-high-latency
        type: latency
        latency: { threshold_ms: 1000 }   # any unit of work over 1s end-to-end
      - name: baseline-sample
        type: probabilistic
        probabilistic: { sampling_percentage: 5 }
```

That `decision_wait: 10s` is doing something specific to async: because the consumer span arrives much later than the producer span, the collector must hold the sampling decision open long enough for the linked consumer spans to show up, or it will sample on the producer alone and you lose the half of the trace you actually wanted. Tail sampling and async tracing have a real tension — the whole "trace" trickles in over an unbounded window — and `decision_wait` is the knob that trades completeness against memory in the collector.

There is a sharp edge here worth naming, because it bites teams that adopt span links and tail sampling together without thinking it through. Span links break the usual tail-sampling grouping assumption. Tail sampling normally buffers all spans *of one trace id* and decides once they have all arrived. But with span links, the producer span and the consumer span are in *different traces with different trace ids* — that was the whole point of linking rather than nesting. So a tail sampler keyed purely on trace id will make *independent* sampling decisions for the producer trace and the consumer trace, and you can end up keeping the producer half and dropping the consumer half, or vice versa, leaving you a stub. The mitigations are real but require thought: sample at the *head* on the propagated `traceparent` flag (so the producer's sampling decision rides the header into the consumer and both halves agree), or use a backend that understands links well enough to co-sample linked traces, or accept that each half is sampled independently and lean harder on logs and metrics to bridge the gap. Whichever you pick, decide it deliberately — the interaction between linking and sampling is exactly the kind of subtlety that silently degrades your trace coverage until an incident reveals you only kept half the spans.

## 6. The metrics that matter per role

Metrics are the cheapest pillar and the one most teams under-build, so let us be specific about *which* metrics, organized by *role*, because the role is the unit of ownership and the unit of alerting. A messaging metric only means something in the context of who emits it — a send-error counter is a producer concern, under-replicated partitions is a broker concern, lag is a consumer concern — and conflating them produces dashboards no one can act on. Figure 6 organizes the full metric taxonomy as a tree by role; Figure 3 shows how each pillar, including metrics, manifests at producer, broker, and consumer.

![Tree taxonomy of message-queue metrics grouped under producer, broker, and consumer roles](/imgs/blogs/observability-tracing-across-the-queue-boundary-6.webp)

### Producer metrics: are we successfully publishing?

The producer's job is to get messages durably into the broker, so its metrics measure *that* and only that. **Send rate** (messages per second, bytes per second) is your baseline volume — a sudden drop to zero is often the first sign of an outage, and a producer that silently stops publishing is one of the nastier failures because nothing downstream errors, the work just never arrives. **Send errors and retry rate** tell you whether publishes are failing or struggling: a climbing retry rate means the broker is under pressure or a partition leader is unavailable, and retries inflate latency and can reorder or duplicate. **Producer-side latency** (time from `send()` to broker ack) tells you whether the broker is acknowledging promptly; spikes here usually mean the broker is slow to replicate (with `acks=all`) or is GC-pausing. **In-flight/buffer metrics** — Kafka's `buffer.available.bytes`, the size of the unsent record accumulator — warn you that the producer is about to block or drop because the broker is not draining fast enough. If that buffer fills, the producer either blocks the calling thread or throws, and a blocked producer thread can stall your whole web tier, turning a downstream slowdown into a frontend outage.

### Broker metrics: is the pipe healthy?

The broker is usually managed (or a separate team owns it), but you still alert on a handful of its signals because broker health is your delivery guarantee. **Throughput** (bytes in/out per second per topic) is the broker's pulse. **Under-replicated partitions** is the single most important Kafka broker alert: a partition with fewer in-sync replicas than configured has lost redundancy, and if it drops to one replica you are one disk failure from data loss — this maps directly to the durability mechanics in [Kafka replication, ISR, acks, and durability](/blog/software-development/message-queue/kafka-replication-isr-acks-durability). **Request queue depth and handler idle ratio** tell you the broker is saturated and about to add latency to everyone. **Disk usage and retention pressure** matter because if the broker fills its disk it stops accepting writes, and if retention is shorter than your consumer lag, messages get deleted before they are consumed — the dreaded retention cliff. For RabbitMQ the analogs are queue depth, memory and disk alarms (which trigger flow control), and unacked message counts.

### Consumer metrics: are we keeping up?

This is the role where async failures actually live, and **consumer lag** is the headline metric — the offset gap between the latest message produced and the latest message the consumer group has committed, per partition. Lag is the async system's blood pressure: zero or low and flat means you are keeping up; steadily climbing means consumers are falling behind and the gap will eventually become latency your users feel, or worse, data loss if lag exceeds retention. Lag deserves its own deep treatment, which is exactly the subject of [Consumer lag monitoring and autoscaling](/blog/software-development/message-queue/consumer-lag-monitoring-and-autoscaling); here the point is that lag belongs on every async dashboard, alerted on both *absolute* value and *rate of change*, because a slowly climbing lag is a warning and a fast-climbing lag is an incident. Alongside lag: **consumer throughput** (messages processed per second), **per-message processing latency** (p50/p99 of the consumer span — this is where your linked traces feed your metrics), **commit rate and commit failures** (a consumer that processes but cannot commit will reprocess on restart, which interacts with the [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) you hopefully built), and **rebalance frequency** (frequent rebalances mean churn that pauses consumption and can thrash partitions between members).

#### Worked example: catching a slow consumer before users notice

Walk through how these metrics catch a real degradation. At 14:00 you deploy a change to the fulfillment consumer that adds a synchronous call to a third-party shipping API, adding 80ms per message. Before, your single consumer processed 200 msg/s comfortably against a 150 msg/s arrival rate, lag pinned at ~0. After the deploy, per-message latency jumps from 35ms to 115ms, so max throughput drops from ~285 msg/s to ~85 msg/s — *below* the 150 msg/s arrival rate. Now do the arithmetic: messages arrive 65 faster per second than they leave, so lag grows by 65/s. After 5 minutes lag is ~19,500 messages; after an hour, ~234,000. Your *processing-latency* metric spiked at 14:00 (the leading indicator), your *throughput* metric dropped at 14:00, and your *lag* metric began its steady climb at 14:00 — three independent signals all pointing at the same deploy. If retention is 6 hours and arrival continues, you can compute the cliff: lag grows at 65/s = 234k/hour, so you have until lag reaches the retention horizon before messages start being deleted unconsumed. That is the alert that pages someone, and the fix (scale consumers out, or make the shipping call async) is obvious *because the metrics localized it to the consumer role and the deploy time*. Without per-role metrics you would have seen "orders are slow" and burned an hour guessing.

| Role | Headline metric | Alert on | Failure it catches |
|---|---|---|---|
| Producer | Send error / retry rate | Errors > 0.1%, or send rate drops to 0 | Silent publish failure, broker unreachable |
| Producer | Buffer available bytes | Buffer < 10% free | Producer about to block the web tier |
| Broker | Under-replicated partitions | Any value > 0 sustained | Lost redundancy, near data loss |
| Broker | Disk / retention pressure | Disk > 80%, retention < lag | Retention cliff, write rejection |
| Consumer | Consumer lag | Absolute high OR rate climbing | Falling behind, latency, data loss |
| Consumer | Processing latency (p99) | p99 above SLO | Slow downstream, code regression |
| Consumer | Commit failures / rebalances | Any sustained | Reprocessing, consumption pauses |

## 7. Correlating logs with trace and message IDs

Traces tell you *where* and metrics tell you *how bad*, but when you finally narrow it to one failed message, you want the *why*, and the why lives in the logs. The catch is that in an async system the logs are scattered across the producer process, the consumer process, and possibly a dozen worker replicas, and a raw log search by timestamp drowns you. The fix is to make every log line carry the keys that let you join across the boundary: the **trace id** and the **message id** (and ideally a business **correlation id**). Figure 9 contrasts the two worlds — without a propagated correlation id the logs are islands you cannot join, and with one a single query stitches every line of a unit of work together.

![Side-by-side comparison of logs with no correlation id that cannot be joined versus a propagated correlation id that joins all logs end to end](/imgs/blogs/observability-tracing-across-the-queue-boundary-9.webp)

### Three ids, three jobs

These ids are not redundant; they answer different questions and you want all three. The **trace id** joins logs to the *trace* and to each other within one technical unit of work — it is generated by your tracing instrumentation, it is the W3C trace id from the `traceparent`, and because you propagated context through the headers it is the *same* trace id on both sides of the queue. Log it on every line and your log backend can pivot from a trace straight to its logs and back. The **message id** is the broker's (or your envelope's) unique identifier for *this specific message* — Kafka does not assign a global message id but the `(topic, partition, offset)` triple is a unique coordinate, while RabbitMQ, SQS, and most envelopes carry an explicit `message_id`. The message id answers "what happened to *this exact message*," which is precisely the question in a "where did my message go" investigation, and it survives even when tracing was sampled away. The **correlation id** is a *business* identifier for the unit of work that spans potentially many messages and many traces — an order id, a saga id, a user-facing request id — and it is what support hands you when a customer complains. You propagate the correlation id the same way as trace context: in a header, injected at produce, logged on both sides.

### Structured logging that actually correlates

The mechanism is mundane and the discipline is everything: emit structured (JSON) logs, and put the ids in fields, not in the free-text message. Here is a consumer that establishes the correlation context once and lets every subsequent log line inherit it.

```python
import logging, json
from opentelemetry import trace

logger = logging.getLogger("fulfillment")

def process_message(msg, ctx):
    span = trace.get_current_span()
    sctx = span.get_span_context()
    # Build a correlation bundle once, attach to every log line.
    log_ctx = {
        "trace_id": format(sctx.trace_id, "032x"),
        "span_id": format(sctx.span_id, "016x"),
        "message_id": f"{msg.topic}-{msg.partition}-{msg.offset}",
        "correlation_id": dict(
            (k, v.decode()) for k, v in (msg.headers or [])
        ).get("correlation-id", "none"),
        "order_id": msg.value.get("id"),
    }
    try:
        fulfill(msg.value)
        logger.info(json.dumps({"event": "fulfilled", **log_ctx}))
    except Exception as e:
        # This error line is now joinable to the trace AND the message.
        logger.error(json.dumps({"event": "fulfill_failed",
                                 "error": str(e), **log_ctx}))
        raise
```

Now the join queries you actually run during an incident become trivial. "Show me every log line for this trace" is `trace_id = "4bf9..."`. "What happened to this specific message" is `message_id = "orders-3-88214"`. "Trace this customer's whole order journey across every service and every retry" is `correlation_id = "order-91father..."`. The last one is the killer feature, because it crosses not just the queue boundary but every service the order touches and every redelivery — it is the thread that ties together what async tore apart. Modern log backends and the OpenTelemetry logging bridge will even auto-inject `trace_id` and `span_id` into log records when a span is active, so you often get the trace correlation for free and only have to add the message id and correlation id by hand.

### A note on log volume and the async multiplier

One honest warning. At-least-once delivery, which most queues default to, means a message can be redelivered, so the *same* message id can appear in your logs multiple times across retries. This is a feature for the investigation — you can see every attempt — but it is a cost driver, because a poison message stuck in a retry loop (the subject of [Poison messages and retry storms](/blog/software-development/message-queue/poison-messages-and-retry-storms-containment)) can emit thousands of log lines for one message id, blowing up your log bill and drowning the signal. The correlation discipline that makes investigation possible is the same discipline that makes retry storms visible: a sudden spike of log lines sharing one message id *is* the alarm. Build the dashboard that counts log lines per message id, and a flat line means health while a spike means a message is stuck looping.

## 8. The "where did my message go" investigation

This is the question that defines async on-call, and it deserves a *method*, not a panic. A customer says their order never shipped. You have the order id (a correlation id). The message either was never produced, is sitting in the queue unconsumed, was consumed but failed, was sent to a dead-letter queue, or was processed successfully and the problem is elsewhere. Each possibility has a distinct signature across your three pillars, and the investigation is a decision tree that uses the ids to walk from "I have an order id" to "here is exactly where it stopped." Figure 8 shows the timeline of a single healthy message's journey with timestamps, which is the baseline you compare the broken case against.

![Timeline of one message journey across services with timestamps showing queue wait dominating end-to-end latency](/imgs/blogs/observability-tracing-across-the-queue-boundary-8.webp)

### The decision tree, step by step

Start at the correlation id and walk forward. **Step one: was it produced at all?** Query logs for `correlation_id = "order-91..."` on the *producer* side. If there is no producer log line and no producer span, the message was never published — the bug is upstream of the queue, in the web tier, and you have your answer in thirty seconds without ever touching the broker. This is the most common surprise: half of "lost message" tickets are *never-sent* messages, and the correlation id finds them instantly. **Step two: did it reach the broker?** If the producer logged a successful send, you have a message id (the `(topic, partition, offset)`). Now you know the message is in the log; the question becomes whether it was consumed. **Step three: was it consumed?** Compare the message's offset against the consumer group's committed offset for that partition (this is literally what lag measures). If the committed offset is *behind* the message's offset, the message is still in the queue, unconsumed — the problem is consumer lag, and your lag metric will confirm it is high for that partition. The message is not lost, it is *waiting*, and the fix is to scale consumers or unblock the stuck one. **Step four: was it consumed but failed?** If the committed offset is *past* the message's offset, a consumer processed it. Search consumer logs for the message id. A `fulfill_failed` line tells you it was consumed and errored, and the trace id on that line takes you to the trace and the stack. **Step five: did it dead-letter?** A message that failed repeatedly gets routed to a dead-letter queue (see [Dead-letter queues, retries, and exponential backoff](/blog/software-development/message-queue/dead-letter-queues-retries-exponential-backoff)). Check the DLQ for the message id; if it is there, you have found exactly where it stopped and why, with the original headers (including your trace context and correlation id) preserved.

#### Worked example: locating a stopped message with id, lag, and DLQ

Make it concrete with real numbers. Customer reports order `ORD-77310` never shipped, around 09:15. **Step one**, query producer logs: `correlation_id = "ORD-77310"` returns one line at 09:14:58, `event: published, message_id: orders-2-150847, trace_id: a1b2...`. Good — it was produced, to partition 2 at offset 150847. **Step three**, check partition 2's consumer-group committed offset: it is at 150,201, which is *behind* 150,847, and the lag dashboard for partition 2 shows 646 messages of lag climbing since 09:05. So the message is *still in the queue*, unconsumed — it never failed, it is waiting behind 646 others. **Why is partition 2 lagging when the others are flat?** Pull the consumer trace for the message currently being processed on partition 2 (trace id from its log line): its consumer span shows 4.2 seconds of duration, almost all in a single child span calling the address-validation service. One partition's consumer is wedged on a slow dependency, so that partition's lag balloons while the others stay healthy — a classic *partition skew* in processing time. The message is not lost; it is queued behind a slow consumer. The fix is immediate (the slow dependency) and the investigation took four queries, each keyed on an id you propagated on purpose. Compare that to the alternative: grepping a million log lines by timestamp across forty worker pods, with no key to join on, hoping to stumble onto the order. The ids turned an archaeological dig into a four-step lookup.

### Build the runbook, not the hero

The deeper lesson is that this investigation should be a *runbook*, not a feat of individual brilliance. Write it down as the decision tree above, with the exact queries for your stack, so the most junior on-call engineer can execute it at 3am. The whole reason you invested in propagating trace context, message ids, and correlation ids is to make this investigation *mechanical*. If finding a lost message still requires your most senior engineer's intuition, your observability is not done — the ids exist precisely so that following a message across the queue boundary is a procedure, not an art.

### The investigation as a query, not a hunt

It is worth being explicit about *how cheap* each step becomes once the ids are in place, because the contrast with the un-instrumented world is the entire justification for the effort. In a system with no propagated ids, "where did my message go" is an open-ended hunt: you start with a timestamp window and a business value (the order id, which appears only in the payload), you grep producer logs, broker logs, and forty consumer pods' logs, you try to mentally join lines by timing because nothing else connects them, and you frequently give up and declare the message "lost in the queue" because you literally cannot prove otherwise. Every step is a guess. In a system with propagated ids, every step is an *exact-match query* on an indexed field: `correlation_id = X` returns the producer line in milliseconds, the message id from that line is an exact coordinate into the broker, the offset comparison is a single metrics lookup, and the trace id pivots you straight to the span tree. The investigation went from O(all logs in the window) to O(a handful of point lookups). That is not a marginal improvement; it is the difference between an on-call task with a bounded, predictable cost and one whose cost is unbounded and depends on luck. When you justify the instrumentation budget to a skeptical team, this is the argument: you are converting unbounded archaeology into bounded lookups, and the conversion pays off on the very first incident.

There is also a quieter benefit that compounds over time. Because the runbook is mechanical and keyed on ids, you can *automate* parts of it. A small tool that takes a correlation id and runs the five queries — producer log, broker offset, committed offset, consumer log, DLQ check — and prints a one-line verdict ("message ORD-77310 is queued on partition 2, lag 646, oldest unprocessed") turns the most common ticket into a self-serve lookup that support can run without paging engineering at all. You cannot build that tool without the propagated ids; with them it is an afternoon. The ids are not just for humans reading dashboards — they are the API surface that lets you build the tooling that makes async on-call sustainable.

## 9. OpenTelemetry messaging conventions

Everything above works better if everyone agrees on the *names*. If your producer tags the destination `topic` and your consumer tags it `queue_name` and your dashboards expect `kafka.topic`, you have three vocabularies for one concept and nothing joins. OpenTelemetry's *messaging semantic conventions* are the shared vocabulary — a standardized set of span names, span kinds, and attribute keys for messaging operations — and adopting them is what makes your traces portable across services, teams, and backends. Figure 3's stack of metrics, logs, and traces across the three roles is exactly the structure the conventions formalize.

![Grid of the observability stack showing metrics, logs, and traces across producer, broker, and consumer roles with every cell filled](/imgs/blogs/observability-tracing-across-the-queue-boundary-3.webp)

### The attributes you should always set

The conventions define attributes that your dashboards and your future self will thank you for. The core ones, with their standard keys:

- `messaging.system` — the broker family: `kafka`, `rabbitmq`, `aws_sqs`, `nats`. This is the dimension that lets a single dashboard work across brokers.
- `messaging.operation.type` — what the span did: `publish` (send), `receive` (poll/get), or `process` (the actual handling). Distinguishing `receive` from `process` is what lets you separate "time spent fetching" from "time spent doing the work."
- `messaging.destination.name` — the topic or queue name. The single most useful filter dimension; almost every messaging query starts here.
- `messaging.message.id` — the message identifier where one exists (RabbitMQ, SQS), so your traces carry the same id your logs do.
- `messaging.kafka.message.offset` and `messaging.destination.partition.id` — for Kafka, the offset and partition that together form the message coordinate.
- `messaging.consumer.group.name` — the consumer group, so you can slice lag and throughput per group.
- `messaging.batch.message_count` — when a consumer polls a batch, how many messages this span covers.

### Span naming and span kind

The conventions also standardize the span *name* and *kind*. The recommended span name is `{destination} {operation}` — for example `orders publish` and `orders process` — which sorts and groups beautifully in any trace UI. The span *kind* is `PRODUCER` on the send side and `CONSUMER` on the receive/process side; these kinds are what tell the backend to render the messaging hop specially and what some backends use to know that a link, not a parent-child edge, is the expected relationship across the gap. Set the kind correctly and a good backend will *expect* the span-link pattern and render it as a clean cross-trace reference rather than a broken waterfall.

### Batching: one receive span, linked process spans

The conventions handle a case the naive model misses: batch consumption. When a consumer polls and gets 500 messages in one fetch, you do not want 500 separate `receive` spans, but you also do not want to lose the per-message context. The convention is a single `receive` span for the batch (with `messaging.batch.message_count = 500`), and then a separate `process` span *per message*, each linking back to its own producer span via the context extracted from that message's headers. So one batch fetch produces one receive span plus N linked process spans, and each of the N reconnects to a different upstream producer trace. This is genuinely elegant: the batch is one operation, but causality is preserved per message, and you can still answer "where did *this one* message go" even though it arrived in a batch of 500.

```python
# Batch consumption following OTel conventions: 1 receive span, N linked process spans
records = consumer.poll(timeout_ms=500, max_records=500)
for tp, batch in records.items():
    with tracer.start_as_current_span(
        f"{tp.topic} receive", kind=SpanKind.CONSUMER
    ) as recv_span:
        recv_span.set_attribute("messaging.system", "kafka")
        recv_span.set_attribute("messaging.operation.type", "receive")
        recv_span.set_attribute("messaging.batch.message_count", len(batch))
        for msg in batch:
            carrier = {k: v.decode() for k, v in (msg.headers or [])}
            link = Link(trace.get_current_span(
                propagate.extract(carrier)).get_span_context())
            with tracer.start_as_current_span(
                f"{tp.topic} process", kind=SpanKind.CONSUMER, links=[link]
            ) as proc_span:
                proc_span.set_attribute("messaging.operation.type", "process")
                proc_span.set_attribute(
                    "messaging.kafka.message.offset", msg.offset)
                fulfill(msg.value)
```

### Why conventions beat cleverness

You could invent your own attribute names and they would work *for you*. The reason to use the standard ones is that they work for *everyone* — the OpenTelemetry-native dashboards in your backend already know `messaging.destination.name`, the auto-instrumentation already emits these keys, the next team that integrates with your topics already speaks the vocabulary, and when you switch backends or add a new service nothing has to be re-mapped. Conventions are how a fragmented async system regains a *shared language*, which is the same problem at the human layer that trace propagation solves at the machine layer. The whole theme of this post is re-stitching what async tore apart, and the semantic conventions stitch the *vocabulary* the same way span links stitch the *spans*.

## Case studies and war stories

Theory survives contact with production differently than it reads, so here are four scenarios — composites of common, real failures — and the specific lesson each one burns in.

### The 202 that lied for three weeks

A payments team migrated their refund flow to a queue: the API returned `202 Accepted` and a worker did the actual refund asynchronously. Refunds quietly failed for a subset of currencies because of a serialization bug in the worker, but the API's trace looked *perfect* — fast, green, `202`, no error — because the trace ended at the enqueue and the failure lived in an orphan consumer trace nobody connected to the request. The dashboards were green for three weeks while customers waited on refunds. The fix was not just propagating trace context; it was the realization that *the producer span succeeding tells you nothing about whether the work succeeded*. They added a consumer-side success metric (`refunds_completed` vs `refunds_requested`) and alerted on the *gap*. The lesson: in async, a successful enqueue is the beginning of the story, not the end, and your dashboards must measure *completion*, not *acceptance*. A green `202` is the most dangerous shade of green there is.

### The latency that wasn't where everyone looked

An e-commerce platform paged on "checkout is slow" — p99 end-to-end had crept from 1.2s to 4s. Three engineers spent two hours staring at the database, because the consumer's trace showed the database write taking a long time. The database was fine. What they had missed: their auto-instrumentation had wired the consumer span as a *child* of the producer span, so the tracing backend stretched the producer span to contain the 2.8s of *queue wait*, and that queue wait got mis-attributed in their latency breakdown to "processing," which they reflexively assumed meant the database. Once they switched to span *links*, the queue wait popped out as its own measurable gap, the lag metric confirmed consumers were behind, and the real fix — scaling consumers — was obvious. The lesson is the one from section four, lived: parent-child across a queue does not just look wrong, it actively *misdirects* the investigation by attributing queue time to the wrong span. The relationship is not cosmetic.

### The message that was never sent

A logistics company had a recurring "lost shipment label" complaint and a recurring four-hour investigation, because the on-call would start at the consumer ("why didn't the worker print the label?"), search worker logs, find nothing, and conclude the message had been "lost in the queue." It had never been *sent*. A race condition in the producer skipped the publish under a specific retry path. The investigation kept starting in the wrong place. After they added a propagated correlation id and a runbook that *started at the producer* ("step one: did we publish it?"), the same investigation took five minutes and immediately revealed the missing producer log line. The lesson: "where did my message go" investigations must start at the *first* possible failure point (was it produced?) and walk forward, because human intuition tends to start where the symptom appeared (the consumer) and the message is often missing far upstream. The correlation id lets you start at the beginning.

### The retry storm that buried the signal

A streaming team shipped a poison message — one record that threw on every process attempt — into a consumer with naive infinite retries. The same message id reprocessed thousands of times per minute, each attempt emitting a full structured log line, and the log volume for that single message id spiked the log bill and buried every other signal in noise. Ironically, the correlation discipline that *enabled* the investigation also *was* the early warning: a dashboard counting log lines per message id would have shown one id with thousands of lines while all others had one or two. They added that panel and a dead-letter route after N attempts. The lesson, which connects to [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) and to poison-message containment: at-least-once means the same id recurs, so *cardinality of a single id* is itself a signal — a flat one-line-per-id distribution is health, a spike is a stuck message, and you should alert on the shape, not just the volume.

## When to reach for this (and when not to)

Trace-context propagation, span links, and per-role metrics are not free — they are instrumentation you write, maintain, and pay to store — so here is the decisive guidance on when the investment pays off and when it is overkill.

**Reach for the full setup when** your async path crosses a service boundary that you will have to debug in production, which is almost always. The moment a unit of work spans a producer and a consumer in different processes, you need at minimum trace-context propagation and a correlation id, because without them the very first production incident becomes the séance described in the intro. If you run any non-trivial event-driven architecture — orders, payments, fulfillment, anything where a stuck message has business consequences — the per-role metrics (especially consumer lag) and the correlation ids are not optional; they are the difference between a four-minute and a four-hour incident. If you have multiple consumers, fan-out, or chained queues (the output of one consumer is the input to another), span links are essential, because the unit of work now crosses *several* boundaries and only linked spans let you follow it through all of them.

**You can skip some of it when** the async hop is trivial and internal — a single in-process background task queue with no cross-service boundary, where the work completes in the same logical operation and a simple log with the request id suffices. You can also defer the full tail-sampling collector setup for low-volume systems where you can afford to keep 100% of traces; sampling is a cost optimization, and at low volume the cost is not worth the complexity. And if your "messaging" is genuinely synchronous request/reply with a blocking caller, you are back in RPC-land where parent-child is correct and the standard HTTP-style propagation mostly just works — do not over-engineer span links for a synchronous exchange.

**Do not** skip correlation ids to save effort — they are the cheapest, highest-leverage piece, a single header and a single log field, and they pay for themselves the first time a customer asks where their order went. And do not let auto-instrumentation make the span-relationship decision for you without checking: verify whether it links or nests across your queue, because the default is frequently wrong and the wrong default silently corrupts your latency numbers, as two of the war stories above show. The one thing that is never optional is *some* way to follow a unit of work across the boundary; everything else is a question of how much fidelity you are buying.

| Situation | Propagate context | Span links | Per-role metrics | Tail sampling |
|---|---|---|---|---|
| Cross-service async, business-critical | Yes | Yes | Yes | Yes (high volume) |
| In-process background tasks | Optional | No | Lag only | No |
| Synchronous request/reply over MQ | Yes (RPC-style) | No (parent-child) | Yes | Optional |
| Fan-out / chained queues | Yes | Yes (essential) | Yes | Yes |
| Low-volume internal events | Yes (cheap) | Yes | Yes | No (keep all) |

## Key takeaways

- **Async deliberately breaks the trace.** A queue severs the synchronous call relationship tracing depends on, so by default the producer trace ends at enqueue and a brand-new, disconnected trace starts at the consumer. This is correct default behavior, not a bug — you have to actively rebuild the connection.
- **Propagate W3C trace context through the message headers.** Inject the `traceparent` (and `tracestate`) into the message headers at produce time and extract them at consume time. Headers are the right home because crossing a queue is a serialization boundary that destroys in-memory context. The broker carries the headers as opaque bytes and never inspects them.
- **Link the consumer span, do not nest it.** A consumer runs asynchronously and possibly much later, so making it a child of the producer span distorts latency — it either hides the work or mis-attributes queue wait to the producer. A span link preserves causality without forcing temporal containment, and each span reports its honest duration.
- **Run all three pillars, because async needs them interlocked.** Metrics tell you *something* is wrong and how bad, traces tell you *where* it stopped, logs tell you *why* it failed. In async systems the three pillars are how you reconstruct the unit of work the architecture fragmented; remove one and investigation becomes guesswork.
- **Build metrics per role.** Producer: send rate, error/retry rate, buffer pressure. Broker: throughput, under-replicated partitions, retention pressure. Consumer: lag (the headline metric, alerted on value *and* rate of change), processing latency, commit failures, rebalances. The role is the unit of ownership and alerting.
- **Correlate logs with three ids.** Trace id joins logs to the trace, message id (or topic-partition-offset) identifies the exact message and survives sampling, and a business correlation id follows the unit of work across every service, message, and retry. Emit them as structured fields, not free text.
- **Make "where did my message go" a runbook.** Walk the decision tree from the correlation id: was it produced, did it reach the broker, was it consumed (lag), did it fail (consumer logs), did it dead-letter. Start at the *first* failure point and walk forward — many "lost" messages were never sent.
- **Adopt the OpenTelemetry messaging conventions.** Standard attribute keys (`messaging.system`, `messaging.destination.name`, `messaging.operation.type`), span names (`{destination} {operation}`), and span kinds (`PRODUCER`/`CONSUMER`) make your traces portable and your dashboards reusable. Batch consumption is one receive span plus N linked process spans.
- **A green 202 is not a green outcome.** The producer succeeding tells you nothing about whether the async work succeeded — measure *completion*, not *acceptance*, and alert on the gap between requested and completed.

## Further reading

- [Message queues: async decoupling and load leveling](/blog/software-development/message-queue/message-queues-async-decoupling-and-load-leveling) — why teams adopt the asynchrony that breaks the trace in the first place.
- [Request-reply over messaging: correlation IDs and reply queues](/blog/software-development/message-queue/request-reply-over-messaging-correlation-ids) — the synchronous-style exchange where parent-child relationships are legitimately correct.
- [Consumer lag monitoring and autoscaling](/blog/software-development/message-queue/consumer-lag-monitoring-and-autoscaling) — the deep dive on the single most important consumer metric for async health.
- [Idempotency and deduplication: making at-least-once safe](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) — why the same message id recurs in your logs and how to handle the duplicates safely.
- [Dead-letter queues, retries, and exponential backoff](/blog/software-development/message-queue/dead-letter-queues-retries-exponential-backoff) — where messages land when they fail repeatedly, and the last stop in the investigation tree.
- [Poison messages and retry storms: containment](/blog/software-development/message-queue/poison-messages-and-retry-storms-containment) — how one bad message floods your logs with a single recurring id and how to catch the cardinality spike.
- [Kafka replication, ISR, acks, and durability](/blog/software-development/message-queue/kafka-replication-isr-acks-durability) — the broker-side mechanics behind under-replicated partitions and the retention cliff.
- The OpenTelemetry messaging semantic conventions and the W3C Trace Context specification — the two standards this entire approach builds on; read them once and your instrumentation stops being guesswork.
