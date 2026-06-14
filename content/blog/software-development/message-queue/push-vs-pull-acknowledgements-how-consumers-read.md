---
title: "Push vs Pull, Acknowledgements, and How Consumers Actually Read"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Trace a message from broker to consumer and back: the push model and why it needs prefetch, the pull model and its free backpressure, and the full acknowledgement machinery — ack, nack, requeue, redelivery, and the visibility timeout — that decides whether your work is durable or lost."
tags:
  [
    "message-queue",
    "consumers",
    "acknowledgements",
    "flow-control",
    "kafka",
    "rabbitmq",
    "sqs",
    "distributed-systems",
    "event-driven",
    "backpressure",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/push-vs-pull-acknowledgements-how-consumers-read-1.webp"
---

There are only two ways a message can get from a broker to a consumer, and almost every operational surprise you will ever have with a queue traces back to which one you picked. Either the broker decides when to hand you a message and shoves it down the connection the instant one is ready — that is **push** — or you decide when you are ready and go ask for a batch — that is **pull**. It sounds like a trivial implementation detail. It is not. It determines who is responsible for not melting your consumers when they fall behind, how much latency you pay on an idle queue, whether batching is natural or a fight, and how complicated your consumer code has to be. RabbitMQ is fundamentally a push broker. Kafka is fundamentally a pull broker. The fact that those two systems feel so different to operate is, more than anything else, a consequence of that one choice.

And then there is the other half of the round trip, the half that beginners forget exists until it bites them: how the consumer tells the broker *I am done with that one, you can stop tracking it*. That is the **acknowledgement**, and it is the single most important contract in the whole pipeline. Get acknowledgements wrong and you will either lose messages silently on a crash or reprocess the same message a hundred times in a redelivery storm. Acks are where at-most-once and at-least-once delivery actually live; they are not abstract semantics, they are a method call you either make at the right moment or you don't. If you have read the companion post on [delivery semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once), this post is where those semantics stop being a taxonomy and become code with a timer attached.

![A two-panel comparison of push delivery where the broker drives the rate against pull delivery where the consumer sets its own pace and gets backpressure for free](/imgs/blogs/push-vs-pull-acknowledgements-how-consumers-read-1.webp)

By the end of this post you will be able to reason about delivery from both sides of the wire. You will know why a push broker with no prefetch limit will happily hand a slow consumer ten thousand messages it cannot process and then watch it die. You will know why a Kafka consumer that does its work *between* `poll()` calls is fine but one that does its work in a background thread can get itself kicked out of the group. You will be able to size a prefetch count from a consumer's processing time and round-trip latency. You will know exactly what `basic.ack`, `basic.nack`, and `basic.reject` do, when a requeue is a bug, and how SQS's visibility timeout is really just a distributed ack with a clock. And you will have a clear, defensible answer to the question that opens every broker-selection meeting: do we want push or pull, and why.

## 1. Two ways to move a message: push vs pull

Let us pin down the two models precisely, because the words "push" and "pull" get used loosely and the distinction that actually matters is *who controls the delivery rate*.

In a **push** model, the broker is the active party. The consumer opens a connection, registers interest in a queue (in RabbitMQ this is `basic.consume`), and then waits. When a message becomes available, the broker initiates delivery: it sends a `basic.deliver` frame down the open channel without the consumer asking for anything. The consumer's client library invokes your callback. The consumer is reactive. It does not say "give me the next message" — messages simply arrive. The broker decides the timing and the rate.

In a **pull** model, the consumer is the active party. Nothing arrives unasked. The consumer calls a fetch method — Kafka's `poll()`, SQS's `ReceiveMessage` — and the broker responds with whatever is available up to the limits the consumer specified. If the consumer never calls fetch, no message ever moves, no matter how full the queue is. The consumer decides the timing and the rate.

That single difference — who initiates the transfer — cascades into everything else. Look at the figure above. On the push side the arrows of control point from broker to consumer: the broker is calling `basic.deliver`, it is pushing up to some prefetch limit, and a slow consumer can be flooded if that limit is set wrong or not at all. On the pull side the arrows point the other way: the consumer is calling `poll()`, fetching a bounded batch, and because nothing arrives that the consumer did not ask for, backpressure is automatic. A consumer that is busy simply does not call `poll()` as often, and the broker, which never pushes, never overwhelms it.

### Why the active party matters

The party that controls the rate is the party responsible for not overwhelming the slow party. This is the deepest consequence of the push/pull choice and it is worth saying slowly.

In push, the broker controls the rate, so the *broker* must be told how fast it is allowed to go. If you give it no guidance, its instinct is to deliver as fast as the TCP connection allows, which for a fast network and a slow consumer means it will pile messages into the consumer's socket buffer and then into its in-memory unacked set faster than the consumer can drain them. The consumer's heap fills with messages it has accepted delivery of but not yet processed. Eventually it pauses for garbage collection, the broker keeps pushing, and you have a crash. To prevent this, push systems bolt on a **flow-control knob** — RabbitMQ calls it the prefetch count, set with `basic.qos` — that caps how many messages the broker may have outstanding to a consumer before it must stop and wait for acks. Flow control in push is the broker's job, and it is *opt-in*: forget to set it and the default is dangerously permissive.

In pull, the consumer controls the rate, so flow control is free. There is no knob to forget. A consumer that takes a long time to process a batch simply takes a long time to come back and call `poll()` again. During that time, no data moves. The broker holds everything on disk, the consumer's lag grows, and that is *fine* — lag is a number on a dashboard, not a crashed process. The slow party paces the fast party by construction. This is the single biggest reason Kafka can survive consumers that are orders of magnitude slower than the producers: the consumer's own slowness is the throttle.

### A first table

| Property | Push (RabbitMQ-style) | Pull (Kafka-style) |
| --- | --- | --- |
| Who initiates delivery | Broker (`basic.deliver`) | Consumer (`poll()`) |
| Who controls rate | Broker | Consumer |
| Flow control | Opt-in (prefetch / QoS) | Automatic (don't poll if busy) |
| Idle-queue latency | Near zero (delivered instantly) | Up to one poll interval |
| Batching | Per-message by default | Batch-native |
| Risk if misconfigured | Flood a slow consumer | Higher tail latency |

Keep this table in your head; the rest of the post is mostly an elaboration of its rows. Notice that neither column is strictly better. Push trades a flow-control obligation for lower latency; pull trades some latency for automatic backpressure. That tradeoff is the spine of section 9.

### A mental model from everyday life

If you want a non-technical handle on the distinction, the cleanest analogy is the difference between a restaurant where the kitchen sends out plates as they are ready versus a buffet where you go to the counter and take what you can carry. In the kitchen-sends-plates model — push — the kitchen controls the pace, and if it cooks faster than you can eat, plates pile up at your table until they fall on the floor. The kitchen must be told to slow down: "send me at most three plates at a time, and wait for me to finish one before sending the next." That instruction is prefetch. In the buffet model — pull — *you* control the pace: you walk to the counter exactly when you are ready for more, take what fits on your plate, and the food never overwhelms you because you only ever fetch what you can carry. The cost of the buffet is the walk to the counter — that is your poll latency — and if the buffet is empty you either keep walking back to check (busy-polling, exhausting) or you ask the staff to tap you on the shoulder when food appears (long-polling, efficient). Every property in the table above falls out of this one difference, and it is worth holding onto as we go deeper.

## 2. Push: the broker delivers (and must not overwhelm)

Let us go deep on the push model using RabbitMQ as the canonical example, because it is the system most engineers meet first and the one whose failure modes are most instructive.

A RabbitMQ consumer connects, opens a channel, and issues `basic.consume` against a queue. This registers the channel as a *consumer* of that queue and, critically, flips the channel into a mode where the broker will proactively deliver. From that moment, whenever a message is ready in the queue and the channel has capacity (more on capacity in a second), the broker sends a `basic.deliver` method frame followed by the message header and body. Your client library — Pika in Python, the Java client, whatever — receives these frames and dispatches them to the callback you registered. You did not ask for the message. It arrived.

Here is the minimal push consumer in Python with Pika:

```python
import pika

connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
channel = connection.channel()
channel.queue_declare(queue="work", durable=True)

# THIS is the flow-control knob. Without it the broker pushes unboundedly.
channel.basic_qos(prefetch_count=10)

def on_message(ch, method, properties, body):
    process(body)                     # do the actual work
    ch.basic_ack(delivery_tag=method.delivery_tag)  # tell broker we're done

# basic_consume registers us for PUSH delivery — no poll loop anywhere
channel.basic_consume(queue="work", on_message_callback=on_message)
channel.start_consuming()   # block forever, react to deliveries
```

Notice what is *not* in that code: there is no loop that fetches messages. `start_consuming()` blocks and the library calls `on_message` whenever the broker pushes something. The control flow is inverted compared to pull. You write a callback and surrender the timing to the broker.

### The flood: push without flow control

Now delete the `basic_qos` line and suppose the queue has a backlog of fifty thousand messages because a producer ran a batch job overnight. The consumer connects. The broker sees a consumer with no prefetch limit — which means *unlimited* prefetch — and starts delivering. It does not wait for acks. It does not wait for the callback to return. It pushes `basic.deliver` frames as fast as the channel will accept them, which on a local network is tens of thousands per second.

Your callback processes one message in fifty milliseconds. The broker delivers thousands per second. The arithmetic is brutal: the broker is shoving messages into your process roughly a thousand times faster than you can handle them. Pika buffers the undelivered ones. Your process's memory climbs. The unacked messages — delivered to you but not yet acked — accumulate both on your side and in the broker's per-consumer tracking. A few seconds later your consumer is holding tens of thousands of message bodies in memory, the JVM or the Python heap is thrashing, and the OOM killer ends the story. The broker, having delivered all those messages, now has to redeliver every single one of them because none were acked. You have turned a backlog into an outage.

This is not a hypothetical. It is the single most common RabbitMQ production incident, and it has a one-line fix: set a prefetch count. We will size it properly in section 4. For now the lesson is the headline of this section — in a push system, *not overwhelming the consumer is the broker's responsibility, and you have to tell the broker how.*

### Latency: the upside of push

Push is not all peril. Its great virtue is latency. When a message is published to an otherwise-empty queue and a consumer is waiting, the broker delivers it *immediately* — there is no poll interval to wait through, no fetch to initiate. The message lands in the consumer's callback in roughly the time it takes a frame to cross the network and the broker to do its routing, which for a co-located consumer is sub-millisecond. For workloads where latency matters — a request-reply RPC over a queue, a real-time notification fan-out, a chat backend — this immediacy is exactly what you want. The broker's eagerness, which is a liability under backlog, is an asset under low load. That tension is the whole story of push.

### What the broker actually tracks per consumer

There is a hidden cost to push that does not show up until you have thousands of consumers, and it is worth naming because it shapes why push systems scale differently from pull systems. To push, the broker must hold open, stateful machinery for every consumer: a TCP connection, a channel, the registration from `basic.consume`, and — critically — the per-consumer unacked set. Every message the broker has delivered but not yet had acked is a row in an in-memory table keyed by that consumer's delivery tags. The broker is responsible for remembering, for every connected consumer, exactly which messages are outstanding so that if the channel closes it can requeue them.

This is fine for hundreds or low thousands of consumers. It becomes a problem at very high consumer counts, because the broker's memory and bookkeeping grow with the number of consumers *and* the prefetch depth of each. A push broker with ten thousand consumers each holding a prefetch of a hundred is tracking a million in-flight messages in memory, every one of which it may have to requeue on a disconnect. Compare this to a pull broker like Kafka, which tracks almost nothing per consumer — just a committed offset, a single integer per partition per group, persisted in a compacted topic. The broker does not remember which records a consumer is "working on," because in pull there is no such concept: the consumer either committed an offset or it did not. This asymmetry — push brokers carry per-consumer in-flight state, pull brokers carry a single offset — is one of the deepest structural reasons Kafka scales to enormous consumer-group counts while RabbitMQ prefers more modest fan-out with richer routing. It is the same tradeoff viewed from the broker's side: push pays in broker state for its low latency; pull pays in consumer-side loop complexity for its statelessness.

### Channel-level versus consumer-level flow control

RabbitMQ actually layers flow control at more than one level, and operators who only know about prefetch get surprised by the others. At the *consumer* level, prefetch caps unacked messages, as we have discussed. At the *channel* level, a single channel can carry multiple consumers, and the prefetch can be applied per-consumer or shared across the channel depending on the `global` flag in `basic.qos`. And at the *connection* level, RabbitMQ has an internal credit-based flow-control mechanism: if a publisher is sending faster than the broker can accept and persist, the broker stops reading from that connection's socket, which propagates TCP back-pressure all the way back to the publisher. That last mechanism is why, when a RabbitMQ node is under memory pressure, you sometimes see *publishers* block — the broker has deliberately stopped reading their bytes to protect itself. Understanding that there are three flow-control surfaces, not one, is the difference between diagnosing a stall in minutes versus hours.

## 3. Pull: the consumer polls at its own pace

Now flip to pull, using Kafka. A Kafka consumer does not register a callback and wait. It runs a loop. At the top of the loop it calls `poll()`, which goes to the broker, fetches whatever records are available for the partitions this consumer owns (up to configured limits), and returns them as a batch. The consumer processes the batch. Then it commits its offset — the bookmark that says "I have consumed up to here" — and loops back to `poll()` again. Nothing arrives that the consumer did not ask for by calling `poll()`.

![A pipeline showing the pull loop where the consumer calls poll then processes the batch then commits the offset before looping back to the next poll](/imgs/blogs/push-vs-pull-acknowledgements-how-consumers-read-2.webp)

The figure above is the canonical pull loop, and it is worth tracing the way it actually works in production. `poll()` is called with a timeout — say one hundred milliseconds. It returns a batch of up to `max.poll.records` records (default 500). The consumer iterates that batch, does the work — fifty milliseconds per message in our running example — and once the batch is fully handled, commits the offset. Then it loops to the next `poll()`. The consumer is in complete control of its pace: it calls `poll()` exactly when it is ready, never sooner. If processing the batch takes a long time, the next `poll()` is simply delayed, and during that delay no data moves and the broker is undisturbed.

Here is the loop in Java, the way it is almost always written:

```java
Properties props = new Properties();
props.put("bootstrap.servers", "broker:9092");
props.put("group.id", "billing-workers");
props.put("enable.auto.commit", "false");   // we commit manually
props.put("max.poll.records", "500");        // batch ceiling per poll
props.put("max.poll.interval.ms", "300000"); // 5 min to finish a batch

KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
consumer.subscribe(List.of("payments"));

while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    for (ConsumerRecord<String, String> record : records) {
        process(record);   // 50 ms of work
    }
    consumer.commitSync(); // commit AFTER the batch is fully processed
}
```

Read that loop and notice the structure that makes pull safe. The work happens *inside the loop, between `poll()` calls*. The commit happens *after* the work. The next `poll()` does not fire until the previous batch is fully done. There is no way for the broker to hand you a second batch before you finish the first, because you do not ask for the second batch until you are ready. Backpressure is not a feature you configured — it is a property of the loop's shape.

### The poll-interval contract and the rebalance trap

Pull has its own footgun, and it is the mirror image of push's flood. In push, the danger is the broker pushing too fast. In pull, the danger is the consumer pulling too *slowly* — specifically, taking so long between `poll()` calls that the group coordinator decides the consumer is dead.

Kafka's consumer group protocol uses `poll()` as a liveness signal. If you do not call `poll()` within `max.poll.interval.ms` (default five minutes), the coordinator concludes your consumer has hung, kicks it out of the group, and **rebalances** — reassigning its partitions to other members. Now two consumers may briefly process the same partition, your offsets get messy, and you have a duplicate-processing incident. The cause is almost always the same: a batch of 500 records where each takes too long, so the total batch processing time blows past the interval before the loop can call `poll()` again.

The fixes are exactly what you would expect once you understand the cause. Lower `max.poll.records` so each batch is smaller and finishes faster. Raise `max.poll.interval.ms` if your work genuinely is slow. Or — the cleanest fix — decouple processing from polling: hand each batch to a worker pool, pause the partitions while they work, and resume when capacity frees up. But the naive version, "spin up a background thread per message and let `poll()` keep firing," breaks the liveness contract in the other direction: you keep polling, you keep getting batches, the background threads pile up, and you have reinvented the push flood inside a pull system. The discipline that keeps pull safe is keeping the work *between* the polls, not beside them.

### Idle latency: the cost of pull

Pull's price is latency on a quiet queue. Suppose a single message arrives just after your consumer finished a `poll()` that returned nothing. The consumer is now processing (nothing) and will not call `poll()` again until the loop comes back around. With a naive busy-poll using a zero timeout, the loop spins and the latency is tiny but the CPU burns. With a longer poll timeout, the message waits until the next fetch. Either way, pull cannot match push's instant delivery on an idle queue without burning CPU. The cure for that — long-polling — is section 8, and it is one of the more elegant ideas in the whole space.

### Fetch limits: the knobs that shape a pull batch

A `poll()` is not a single dial — it is governed by a small cluster of fetch settings that together determine how much data comes back and how long the consumer is willing to wait for it. Knowing them turns the poll loop from a black box into something you can tune deliberately.

`max.poll.records` (default 500) caps how many records a single `poll()` returns to your application, regardless of how much data the broker sent over the wire. This is the knob you reach for first when batches take too long to process, because it directly bounds the work between polls. `fetch.min.bytes` (default 1) tells the broker the minimum amount of data it should accumulate before responding to a fetch — set it to, say, 64 KB and the broker will wait (up to `fetch.max.wait.ms`, default 500 ms) for that much data to pile up before answering, which trades a little latency for much better batching and far fewer fetch round-trips. `fetch.max.bytes` and `max.partition.fetch.bytes` cap the byte size of a fetch response so a single huge batch cannot blow your consumer's memory. Together these settings let you dial the pull loop anywhere from "low-latency, return whatever you have immediately" (`fetch.min.bytes=1`) to "high-throughput, wait and batch aggressively" (`fetch.min.bytes` large, `fetch.max.wait.ms` generous). The default is tuned toward latency; throughput-heavy pipelines almost always raise `fetch.min.bytes` to cut the per-fetch overhead.

The subtle interaction to remember: `max.poll.records` bounds the *application* batch, but the broker may have sent more records than that in the fetch response, which the client buffers and hands out across subsequent `poll()` calls without another network round-trip. So `poll()` is often *not* a network call at all — it frequently just drains the client's internal buffer. This is why a well-tuned Kafka consumer can sustain enormous throughput with very few actual broker round-trips: one fetch brings back a megabyte of records, and dozens of `poll()` calls serve them out locally before the next fetch fires.

## 4. Prefetch and the in-flight window (flow control in push systems)

Prefetch is the single most important knob in any push-based broker, so it deserves its own section with real numbers. Prefetch (RabbitMQ's `basic.qos(prefetch_count=N)`) tells the broker: *you may have at most N messages delivered-but-unacknowledged to this consumer at any time.* When the consumer has N unacked messages outstanding, the broker stops delivering. As soon as the consumer acks one, a slot frees and the broker delivers the next. Prefetch is the dam between the broker's eagerness and the consumer's capacity.

![A branching graph showing a queue feeding a channel with prefetch ten that delivers ten in-flight unacked messages while the eleventh waits until an ack frees a slot](/imgs/blogs/push-vs-pull-acknowledgements-how-consumers-read-3.webp)

The figure above shows the mechanism with a prefetch of ten. The queue has many ready messages. The channel has `prefetch=10`. The broker delivers ten messages — they are now in-flight, delivered but unacked. The eleventh message does not move. It waits in the queue until the consumer acks one of the ten, at which point exactly one slot frees and the broker delivers the eleventh. The prefetch count is, precisely, the size of the in-flight window: the maximum number of messages the broker will let a consumer hold without acknowledgement.

### The three layers of the flow-control window

It helps to see prefetch as the outermost of three nested limits, because operators conflate them and then can't explain a stall.

![A stack of three nested flow-control layers showing the prefetch window bounding the in-flight limit bounding the unacked set with an ack freeing a slot](/imgs/blogs/push-vs-pull-acknowledgements-how-consumers-read-6.webp)

The stack above reads top to bottom. The **prefetch window** is the configured cap — N=10 per channel. The **in-flight limit** is what that cap means operationally: at most N messages are delivered-but-not-yet-acked at once. The **unacked set** is the broker's per-consumer bookkeeping — the actual data structure on the broker tracking which delivery tags are outstanding. And the **ack** is the event that frees a slot: every `basic.ack` removes one delivery tag from the unacked set and lets the broker push the next. Set prefetch too low and the window is too small to keep the pipeline full; set it too high and you are back toward the unbounded flood. The right size is a calculation, not a guess.

#### Worked example: sizing a prefetch count

Here is the calculation every RabbitMQ operator should be able to do on a whiteboard. A consumer processes one message in **50 ms**. The round-trip time between consumer and broker — the time for an ack to travel to the broker and for the next delivery to come back — is **2 ms**. You want the consumer never to be idle waiting for the next message, but you also do not want it hoarding a huge backlog in memory.

The danger of prefetch=1 is this: the consumer processes a message (50 ms), sends the ack, and then must *wait the full round-trip* (2 ms) for the broker to deliver the next one before it can start working again. With prefetch=1, every message costs 50 ms of work plus 2 ms of idle waiting — a 4% throughput penalty, and worse, a guaranteed stall after every single message. The consumer's effective rate drops to 1000 / 52 ≈ 19.2 messages per second instead of the 20 it is capable of.

The fix is to make the window deep enough that there is always a next message already sitting in the consumer's buffer when it finishes the current one. The minimum prefetch that hides the round-trip is:

```
prefetch_min = ceil( (processing_time + RTT) / processing_time )
             = ceil( (50 ms + 2 ms) / 50 ms )
             = ceil(1.04)
             = 2
```

So a prefetch of **2** is the theoretical minimum to keep this consumer busy — while it works on message one, message two is already in its buffer, so there is no round-trip stall. In practice you want a little headroom for jitter (a slow ack, a GC pause on the broker), so a prefetch of **3 to 5** is a sane choice for this workload. Crucially, you do *not* want prefetch of 500 here: that would put up to 25 seconds of work (500 × 50 ms) of in-flight messages in the consumer's memory, hoard them away from other idle consumers on the same queue, and — if this consumer crashes — force the redelivery of 500 messages instead of a handful. The rule of thumb that falls out of this: **prefetch should be just large enough to cover the round-trip, plus a small margin — not a large buffer.** Fast consumers with tiny per-message work and high RTT want a larger prefetch; slow consumers with heavy per-message work want a small one.

### The fairness consequence

Prefetch is also a *fairness* knob, and this is the part people miss. RabbitMQ delivers to consumers round-robin, but a consumer that has prefetched a big batch is committed to those messages even if it is slow and another consumer is idle. With a high prefetch and uneven message processing times, one unlucky consumer can grab a long run of slow messages while its peers sit empty — head-of-line blocking inside a single queue. The default in modern RabbitMQ deployments is a low prefetch precisely so that work spreads evenly. If you see one worker pinned at 100% CPU while three others idle on the same queue, your prefetch is too high. Lower it and the broker re-spreads the load.

## 5. Acknowledgements: auto vs manual

Now the other half of the round trip. Delivery gets a message to the consumer. The acknowledgement tells the broker the consumer is *done* with it and the broker can stop tracking it. Everything about durability across the consume boundary lives in *when* you send that ack.

There are two fundamental modes, and the choice between them is the choice between at-most-once and at-least-once delivery on the consume side.

**Auto-ack** (RabbitMQ's `auto_ack=True`, Kafka's `enable.auto.commit=true`) means the broker considers the message acknowledged the moment it is delivered — before your code has even looked at it. The broker hands you the message and immediately removes it from its tracking. It will never redeliver it. This is fast and simple: no ack call, no unacked set, no redelivery machinery. But it is also **at-most-once**: if your consumer crashes after the message is delivered but before the work is done, that message is gone. The broker thinks you handled it. You did not. No one will ever see it again.

**Manual ack** (RabbitMQ's `basic_ack`, Kafka's `commitSync` with auto-commit disabled) means the broker keeps the message in its unacked set until *you* explicitly acknowledge it, after the work is done. If you crash before acking, the broker eventually notices (the channel closes, or a timer fires) and redelivers the message to another consumer. This is **at-least-once**: the message is processed at least once, possibly more than once if a redelivery happens after the work was actually done but before the ack landed.

![A before-and-after comparison of auto-ack which is fast but loses work on a crash against manual ack which is safe and redelivers unfinished work](/imgs/blogs/push-vs-pull-acknowledgements-how-consumers-read-8.webp)

The figure above makes the tradeoff concrete. On the auto-ack side: the message is acked on delivery, the consumer crashes mid-work, and the message is lost with no redelivery — at-most-once. On the manual-ack side: the consumer acks only after the work is done, a crash leaves the message unacked, and the broker redelivers it to another consumer — at-least-once. There is no third option that is both fast-and-simple *and* crash-safe; you choose which property you cannot live without.

### The cardinal rule: ack after the work, never before

The most common acknowledgement bug in the world is acking too early. It usually looks innocent:

```python
def on_message(ch, method, properties, body):
    ch.basic_ack(delivery_tag=method.delivery_tag)  # WRONG: acked before work
    process(body)   # if this throws or the process dies here, message is lost
```

The instant you ack, you have told the broker "I have this handled, forget about it." If `process(body)` then throws an exception, or the pod gets OOM-killed, or the node loses power, that message is gone and the broker will never redeliver it because you already acked it. You have manually recreated the at-most-once loss of auto-ack while believing you have manual-ack safety. The correct order is always:

```python
def on_message(ch, method, properties, body):
    try:
        process(body)   # do the work FIRST
        ch.basic_ack(delivery_tag=method.delivery_tag)  # ack only on success
    except Exception:
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)  # retry
```

Do the work, *then* ack. The ack is the consumer's promise that the side effects are durably done. Make that promise only when it is true. This same rule governs Kafka offset commits: commit the offset only after the batch is processed, never before, or a crash between commit and processing silently skips records.

### Auto-commit in Kafka: the subtle data-loss window

Kafka's auto-commit deserves a specific warning because it looks safe and isn't quite. With `enable.auto.commit=true`, the consumer commits offsets on a timer — every `auto.commit.interval.ms` (default 5 seconds), the offsets returned by the *most recent* `poll()` are committed in the background. The trap: those offsets are committed for records that `poll()` *returned*, which may not be records you have finished *processing*. If you poll a batch of 500, start processing, and the auto-commit timer fires after you have processed 100 of them, the offset for all 500 may be committed. Crash now, and the 400 unprocessed records are skipped on restart because their offset is already committed. Auto-commit gives you at-most-once-ish behavior with a confusing window of potential loss. For anything that matters, disable auto-commit, process the batch, and `commitSync()` afterward — exactly the loop in section 3.

### commitSync versus commitAsync

Once you disable auto-commit, you face a second choice that trips up a lot of teams: `commitSync()` or `commitAsync()`. They differ in exactly the way their names suggest, and the difference matters under failure.

`commitSync()` blocks until the broker confirms the offset commit, retrying automatically on retriable errors. It is the safe default: when it returns, you *know* the offset is durably committed. The cost is latency — your loop pauses on every commit waiting for the broker round-trip. For a loop committing once per batch of 500, that pause is negligible relative to the work, so `commitSync()` after each batch is almost always the right call.

`commitAsync()` fires the commit and returns immediately without waiting for confirmation, taking a callback for the result. It is faster — no pause in the loop — but it does *not* retry on failure, because by the time a retry would fire, a later commit may already have superseded it, and retrying a stale offset could move the committed position *backwards* and cause reprocessing. The common production pattern is a hybrid: `commitAsync()` inside the loop for speed, and a single `commitSync()` in a `finally` block when the consumer shuts down, to guarantee the final position is durably recorded. Get this wrong — `commitAsync()` with no final sync — and a clean shutdown can lose the last few seconds of commits, replaying them on the next start. Most teams over-think this; the honest default for the vast majority of consumers is `commitSync()` once per batch, and only reach for async if profiling shows commit latency actually matters.

### Where the ack lives in each broker

It is worth a table to keep the vocabulary straight across the three systems, because the *concept* of an acknowledgement is universal but every broker names and implements it differently.

| System | Positive ack | Negative ack | Redelivery trigger | State held by broker |
| --- | --- | --- | --- | --- |
| RabbitMQ | `basic.ack` | `basic.nack` / `basic.reject` | channel close or nack-requeue | per-consumer unacked set |
| Kafka | offset commit | (none — you re-seek) | offset not advanced + restart | committed offset per group |
| SQS | `DeleteMessage` | (none — let it time out) | visibility timeout expiry | per-message invisibility timer |

Notice the deep difference in the "negative ack" column. RabbitMQ has an explicit negative acknowledgement — you tell the broker "I failed, requeue or drop this one." Kafka has *no* negative ack at all: there is no way to say "redeliver record 4,815" because the consumer's position is a single monotonic offset, not a set of individual messages. To "retry" a record in Kafka you either don't advance past it (and reprocess the whole batch from there) or you publish it to a separate retry topic. SQS also has no explicit negative ack — you simply *don't* call `DeleteMessage`, and the visibility timeout does the redelivery for you. This is why per-message retry is natural in RabbitMQ, awkward in Kafka, and timer-driven in SQS, and it flows directly from whether the broker tracks individual messages or just a position.

## 6. nack, reject, requeue, and redelivery

A positive ack says "done, forget it." But what about a message you *cannot* process — a malformed payload, a downstream that is temporarily down, a poison message that crashes your parser? You need a vocabulary for negative outcomes, and brokers provide one. The taxonomy of what a consumer can do with a delivered message is small but each branch has consequences.

![A tree of acknowledgement outcomes branching from a delivered message into auto-ack and manual ack which further splits into positive ack requeue and drop](/imgs/blogs/push-vs-pull-acknowledgements-how-consumers-read-7.webp)

The tree above is the full set of outcomes. A delivered message either auto-acks (and is gone) or is in manual-ack mode, where it resolves to exactly one of three things: a positive **ack** that removes it permanently; a **nack or reject with requeue=true** that puts it back on the queue for redelivery; or a **reject with requeue=false** that drops it (typically to a dead-letter queue). Every delivered message ends at one of these leaves. Let us walk the three manual outcomes.

### Positive ack

`basic.ack` removes the message from the unacked set permanently. The broker considers it done and frees the prefetch slot. This is the success path you call after the work succeeds, covered in section 5.

### Negative ack with requeue (nack)

`basic.nack(requeue=True)` (or `basic.reject(requeue=True)`) tells the broker: *I could not process this, put it back so someone can try again.* The broker returns the message to the queue. By default it goes back near the front, so it will be redelivered soon — possibly to the same consumer, possibly to another. This is the right response to a **transient** failure: the downstream database had a brief blip, a lock timed out, a rate limit was hit. Retry will probably succeed.

The danger of requeue is the **poison-message loop**. If the failure is *not* transient — the message body is malformed and your parser will throw on it every single time — then requeue=true creates an infinite loop. The broker delivers it, you nack-requeue it, the broker delivers it again, you nack it again, forever, at the speed of your loop, burning CPU and starving good messages behind it. A poison message with naive requeue can pin a consumer fleet at 100% doing nothing but failing on the same bad message thousands of times per second. The fix is a **redelivery count and a dead-letter route**, which we get to next.

### Reject without requeue (drop / dead-letter)

`basic.reject(requeue=False)` or `basic.nack(requeue=False)` tells the broker: *I cannot process this and it should not come back.* The message is removed from the queue. If the queue is configured with a dead-letter exchange (a `x-dead-letter-exchange` argument), the rejected message is routed there instead of being discarded, landing in a dead-letter queue where it can be inspected, alerted on, and reprocessed by hand after a fix. This is the correct terminal outcome for a poison message: drop it from the main flow so it stops blocking good work, but keep it somewhere safe so you don't lose it.

### Redelivery and the redelivered flag

When a message is requeued and delivered again, the broker sets a **redelivered** flag on it (`method.redelivered` in RabbitMQ; in Kafka you infer it from a delivery-count header or your own tracking). Your consumer should check this flag and implement a redelivery budget: count how many times this message has been redelivered and, past a threshold (say five attempts), stop requeuing and dead-letter it instead. This is the single most important piece of defensive consumer code, and it is the difference between a transient blip that self-heals and a poison message that takes down the fleet:

```python
MAX_ATTEMPTS = 5

def on_message(ch, method, properties, body):
    attempt = (properties.headers or {}).get("x-attempt", 0) + 1
    try:
        process(body)
        ch.basic_ack(method.delivery_tag)
    except TransientError:
        if attempt < MAX_ATTEMPTS:
            republish_with_header(body, attempt)          # requeue with count
            ch.basic_ack(method.delivery_tag)             # ack the old copy
        else:
            ch.basic_reject(method.delivery_tag, requeue=False)  # dead-letter
    except PoisonError:
        ch.basic_reject(method.delivery_tag, requeue=False)      # straight to DLQ
```

The distinction between a transient error (retry with a budget) and a poison error (dead-letter immediately) is a judgment your consumer code must make explicitly. A 503 from a downstream is transient. A `JSONDecodeError` is poison. Treating them the same is how you get either a loop or a silent loss.

### Requeue ordering and the retry-with-delay problem

There is a subtlety in `basic.nack(requeue=True)` that surprises people: by default the requeued message goes back to the *front* of the queue, so it is redelivered almost immediately. For a transient failure that is exactly wrong — if the downstream database is down, retrying in the next millisecond will just fail again, and again, as fast as the loop runs, until you exhaust the retry budget in a fraction of a second and dead-letter a message that would have succeeded fine thirty seconds later when the database came back. Immediate requeue is a *retry without backoff*, which is to say a retry that mostly defeats the purpose of retrying.

The fix is a **delayed retry**, and because RabbitMQ has no native "redeliver this in 30 seconds" primitive, the idiomatic pattern uses a dead-letter queue with a message TTL as a delay mechanism. You publish the failed message to a "retry" queue that has no consumers and a message TTL of, say, 30 seconds, and a dead-letter exchange pointing back at the main queue. The message sits in the retry queue doing nothing for 30 seconds, then its TTL expires, it is dead-lettered back to the main queue, and a consumer picks it up again — 30 seconds later, with the downstream likely recovered. Chain several retry queues with increasing TTLs (30 s, 2 min, 10 min) and you have exponential backoff built entirely out of TTLs and dead-letter routing. It is a clever abuse of the primitives, and it is the standard way to do delayed retry in RabbitMQ. (Kafka's equivalent is the retry-topic pattern: failed records go to `topic-retry-30s`, `topic-retry-2m`, etc., each consumed by a worker that waits and republishes to the main topic.)

#### Worked example: a poison message without a redelivery budget

Walk the arithmetic of a poison-message loop to feel why the redelivery budget is not optional. A single malformed message lands in a queue. The consumer's parser throws a `JSONDecodeError` on it. The naive handler catches the exception and calls `basic.nack(requeue=True)`. The broker returns the message to the front of the queue. The consumer's loop comes back around in, say, 1 millisecond, fetches the same message, parses it, throws again, and nacks it again. This repeats at the speed of the loop — roughly **1,000 redeliveries per second** for this one message. The consumer is now pinned at 100% CPU doing nothing but failing on one bad message, and every *good* message behind it in the queue is starved because the poison message keeps jumping back to the front. Across a fleet of ten consumers all sharing the queue, you have ten cores burning, zero useful work, and a growing backlog of legitimate messages that will never be processed until someone notices and manually purges the poison. With a redelivery budget of 5, the same poison message fails 5 times in about 5 milliseconds, is dead-lettered on the 6th delivery, and the queue resumes processing good messages — total damage: 5 wasted parse attempts and one message safely parked in the DLQ for a human to inspect. The redelivery budget converts an unbounded outage into a bounded, logged, recoverable event. That is the entire reason it exists.

## 7. Visibility timeout and the at-least-once contract (SQS-style)

So far we have assumed an open, stateful connection between broker and consumer — RabbitMQ's channel, Kafka's session. But the most widely deployed queue in the world, Amazon SQS, has no such persistent connection. SQS is HTTP-based and pull-only: the consumer makes a `ReceiveMessage` request, gets messages, and makes a separate `DeleteMessage` request when done. There is no channel to hold an unacked set against. So how does SQS know when to redeliver a message whose consumer crashed? The answer is the **visibility timeout**, and it is one of the cleverest ideas in the space: it is an acknowledgement implemented as a clock instead of a connection.

![A pipeline of the SQS visibility timeout where a received message goes hidden for the timeout then either is deleted permanently or reappears for redelivery](/imgs/blogs/push-vs-pull-acknowledgements-how-consumers-read-9.webp)

The figure above traces the lifecycle. When a consumer calls `ReceiveMessage` and gets a message, SQS does not delete it. Instead it makes the message **invisible** for a configurable window — the visibility timeout, 30 seconds by default. During that window the message still exists in the queue but no other consumer can receive it. The consumer that got it now has until the timeout expires to do two things: process the message *and* call `DeleteMessage`. If it calls `DeleteMessage` in time, the message is gone for good — that is the positive ack. If the timeout expires *without* a delete — because the consumer crashed, or hung, or was simply too slow — the message becomes visible again and the next `ReceiveMessage` call can return it. That reappearance is the redelivery. The visibility timeout is exactly a manual ack with a deadline: delete-before-timeout means ack, timeout-without-delete means redeliver.

This gives SQS its **at-least-once** guarantee with no persistent connection and no broker-side per-consumer state beyond a per-message invisibility timer. It is elegant because it is stateless on the connection: any consumer can pick up any visible message, and a dead consumer needs no detection mechanism — its messages simply reappear when their timers expire. The cost is the same at-least-once duplicate risk as any manual-ack system: if your consumer finishes the work and crashes *before* the delete call lands, the message reappears and gets processed twice.

### The visibility timeout is a contract you must honor

The visibility timeout is a *promise to the consumer*: "you have this many seconds of exclusive access; finish within it." If you do not honor it — if your processing routinely takes longer than the timeout — SQS will redeliver the message to a *second* consumer while the *first* is still working on it, and now two consumers are processing the same message concurrently. That is the classic SQS duplicate-processing bug, and it is entirely a tuning failure.

#### Worked example: tuning a visibility timeout

A worker processes an SQS message in, on average, **20 seconds**, but the 99th percentile is **70 seconds** because some messages trigger a slow external API. The visibility timeout is left at the default **30 seconds**. What happens?

For the typical message (20 s), all is well: the worker finishes and deletes within the 30-second window. But for the slow tail — every message that takes longer than 30 seconds — the timeout expires *while the worker is still processing*. At t=30 s the message becomes visible again. Another worker calls `ReceiveMessage`, gets the same message, and starts processing it from scratch. Now two workers are doing the same 70-second job. When the first finishes at t=70 s and calls `DeleteMessage`, the message is deleted — but the second worker is still mid-flight and will eventually try to delete an already-deleted message (harmless) after having *re-run all the side effects*. If those side effects are not idempotent — say, charging a card or sending an email — you have just charged the customer twice and sent two emails. A too-short visibility timeout is a duplicate-processing factory.

The fix has two parts. First, set the visibility timeout above your **p99 (or p99.9) processing time**, not your average — here, at least 90 seconds, giving headroom above the 70-second tail. Second, for the genuinely long-tail messages that might still exceed even that, use SQS's `ChangeMessageVisibility` API to **extend the timeout from inside the worker** as a heartbeat: every 25 seconds of work, push the deadline out another 60 seconds. This way a worker that is making progress keeps the message invisible for exactly as long as it needs, and only a worker that has truly hung lets the timeout lapse. The rule: **set the visibility timeout to cover the tail, and heartbeat-extend for the outliers.** And because at-least-once always allows *some* duplicate, make the side effects idempotent regardless — which the [delivery semantics post](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) covers in depth.

### A note on the in-flight limit

SQS standard queues have a hard cap of **120,000 in-flight messages** (messages received but not yet deleted) per queue. This is the SQS analogue of the prefetch in-flight window from section 4 — a ceiling on how many messages can be simultaneously checked-out. If your consumers are slow and your visibility timeouts are long, you can pile up against this limit, at which point `ReceiveMessage` returns nothing even though the queue has a backlog, because everything is in-flight. The symptom — "the queue has millions of messages but my consumers get empty receives" — is almost always a too-long visibility timeout combined with stuck consumers holding messages invisible.

## 8. Long-polling vs busy-polling

Pull's one real weakness is idle latency: if nothing is in the queue, a consumer that calls fetch gets an empty response, and the question is what it does next. The naive answer — call fetch again immediately — is **busy-polling**, and it is a quiet disaster.

![A timeline of one message's acknowledgement lifecycle from delivered to in-flight to either acked on success or requeued when the timer fires](/imgs/blogs/push-vs-pull-acknowledgements-how-consumers-read-5.webp)

Before the busy-poll discussion, the figure above zooms in on the lifecycle of a single in-flight message that section 5 and 6 described in prose, so the timer relationship is concrete. The message is delivered, it enters the in-flight unacked set, and from there exactly one of two things happens: the work finishes and an ack is sent at t=40 ms, removing it; or no ack arrives, the timer fires at t=30 s, and the message is requeued for redelivery. The timer is the safety net under at-least-once — the thing that turns a crashed consumer's silence into a redelivery rather than a permanent loss. Now back to polling.

### Busy-polling burns money for nothing

A busy-polling consumer that gets an empty response and immediately re-fetches will, on an idle queue, hammer the broker with requests as fast as the network round-trip allows — thousands of empty `ReceiveMessage` or fetch calls per second per consumer. Each one costs a network round trip, broker CPU to check for messages, and — on SQS — *actual money*, because SQS bills per request. A fleet of fifty consumers busy-polling an idle queue can generate millions of empty requests per hour, costing real dollars to receive *zero* messages. The CPU and the request bill both scale with how empty the queue is, which is exactly backwards: you pay the most when there is the least to do.

### Long-polling: block until there is something

The fix is **long-polling**. Instead of returning immediately when the queue is empty, the consumer asks the broker to *hold the request open* for a while — up to 20 seconds in SQS (`WaitTimeSeconds`), or for Kafka, the `poll()` timeout combined with `fetch.max.wait.ms` — and only respond when either a message arrives or the wait time elapses. A long-poll consumer on an idle queue makes one request, the broker holds it for 20 seconds waiting for a message, and if one arrives at second 3 the broker returns it immediately. So you get near-push latency (the message comes back the moment it lands, not at the next poll boundary) while making one request every 20 seconds instead of thousands per second. Long-polling collapses the empty-request storm into a trickle and recovers most of the latency that pull otherwise costs.

```python
# SQS long-polling: hold the request up to 20s; return early if a message lands
response = sqs.receive_message(
    QueueUrl=queue_url,
    MaxNumberOfMessages=10,     # batch up to 10
    WaitTimeSeconds=20,         # LONG poll — the magic line
    VisibilityTimeout=90,       # cover the p99 from section 7
)
for msg in response.get("Messages", []):
    process(msg["Body"])
    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=msg["ReceiptHandle"])
```

The single line `WaitTimeSeconds=20` is the difference between a consumer that costs a fortune and gets high latency, and one that costs almost nothing and gets low latency. There is essentially never a reason to busy-poll SQS; long-polling is strictly better for any consumer that does not need sub-100ms reaction on a queue that is usually empty. The same idea lives in Kafka's `fetch.max.wait.ms`: a `poll()` with a reasonable timeout will block at the broker waiting for `fetch.min.bytes` to accumulate or the wait to elapse, batching small messages and avoiding a tight empty-fetch spin.

### Long-polling makes pull feel like push

This is the punchline that resolves much of the push-versus-pull tension. The latency gap between push and pull is largest with busy-polling and *smallest* with long-polling. A long-polling pull consumer on an idle queue gets a message back almost as fast as a push consumer would, because the broker returns the held request the instant a message arrives. So the real tradeoff is not "push is low latency, pull is high latency" in the abstract — it is "push is low latency, *naive* pull is high latency, and *long-polling* pull is low latency at the cost of holding a connection open." Once you long-poll, pull keeps its free backpressure and recovers most of push's latency. That is why so many modern systems are pull-with-long-polling.

### The empty-receive accounting

It helps to put numbers on just how much long-polling saves, because the magnitude is genuinely startling. Take a single consumer polling an idle queue. With busy-polling at a zero wait, each empty `ReceiveMessage` completes in roughly the network round-trip — call it 20 milliseconds. So one consumer issues about 50 empty receives per second, or 4.32 million per day. SQS bills receives in batches of 64 KB as one request and charges on the order of \$0.40 per million requests after the free tier, so one busy-polling consumer costs about \$1.73 per day in pure empty-receive requests — to receive *nothing*. A fleet of 200 such consumers costs roughly \$345 per day, or over \$125,000 per year, to poll empty queues. Switch every one of them to `WaitTimeSeconds=20` and each consumer now issues one receive every 20 seconds on an idle queue — about 4,320 receives per day instead of 4.32 million, a thousand-fold reduction. The fleet's empty-receive bill drops from \$345 per day to about \$0.35 per day. And — this is the part that feels like cheating — latency *improves*, because a long-poll returns a message the instant it lands rather than waiting an average of half a busy-poll cycle. You pay less and you wait less. There is genuinely no downside for a queue that is ever idle, which is why I treat `WaitTimeSeconds=20` as a non-negotiable default rather than an optimization.

### What long-polling does and does not fix

Long-polling fixes the *idle* case beautifully. It does not change anything about a busy queue: when there is always data waiting, a poll returns immediately whether or not long-polling is enabled, because the broker has something to give and does not need to hold the request. So long-polling is a pure win precisely when the queue oscillates between idle and busy — which is almost every real queue. The one thing to watch is that a held long-poll consumes a connection slot for its whole wait, so a very large fleet of long-pollers holds many open connections; for SQS this is invisible (it is a managed service designed for it), but for a self-hosted broker you size connection limits accordingly. The held connection is the cost; the thousand-fold request reduction and the lower latency are the benefit. It is one of the most lopsided trades in all of systems engineering.

## 9. The push-vs-pull tradeoff and how real brokers choose

We can now state the central tradeoff cleanly and see why each major broker landed where it did.

![A matrix comparing push and pull across tail latency backpressure batching and consumer complexity showing each model wins different rows](/imgs/blogs/push-vs-pull-acknowledgements-how-consumers-read-4.webp)

The matrix above is the decision boiled down to four rows. On **tail latency**, push wins: it delivers immediately, while pull pays up to a poll interval (mitigated, but not eliminated, by long-polling). On **backpressure**, pull wins decisively: it is automatic, while push requires you to configure prefetch correctly or risk a flood. On **batching**, pull wins: it is batch-native — a single `poll()` returns hundreds of records — while push delivers per-message by default and batching is awkward. On **consumer complexity**, push is simpler to write (a callback, no loop) but pull's loop buys you control. There is no universal winner; you weigh the rows by your workload.

### Why RabbitMQ chose push

RabbitMQ's design center is *flexible routing and low-latency task distribution*. Its exchanges and bindings let you route a message to exactly the right queue based on patterns, and its typical workload is jobs and RPC where a message should reach a waiting worker as fast as possible. For that, push is the natural fit: when a job lands and a worker is free, deliver it now. The cost — having to manage flow control with prefetch — is acceptable because RabbitMQ workloads usually have bounded backlogs and modest per-consumer fan-in. RabbitMQ then spent years adding flow-control sophistication (per-channel and per-consumer prefetch, connection-level back-pressure, credit-based flow in the AMQP 1.0 sense) precisely because push's flood risk had to be tamed. The [RabbitMQ production architecture post](/blog/software-development/system-design/rabbitmq-production-architecture-scaling) goes deep on how those controls interact at scale.

### Why Kafka chose pull

Kafka's design center is *high-throughput, replayable streams* — see the companion on [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log). The data is a durable, ordered log on disk, retained for days. Consumers read at their own offsets and can be wildly slower than producers without endangering the broker, because the broker never has to hold anything in memory for a slow consumer — it is all already on disk, and the consumer just reads from wherever its offset points. For this, pull is the only sane choice: the broker cannot and should not try to push at a rate matched to thousands of independent consumers each reading at a different speed. Let each consumer pull at its own pace, and the broker stays simple, stateless about consumer speed, and fast. Pull's automatic backpressure is not a nice-to-have for Kafka — it is the property that makes a slow consumer a non-event. The cost, idle-queue latency, barely matters for streaming workloads that are rarely idle, and `fetch.max.wait.ms` handles the rest.

### Why SQS chose pull-with-visibility-timeout

SQS optimizes for *operational simplicity at massive scale and zero per-consumer state*. A push model would require SQS to maintain connections to millions of consumers and track flow control per connection — operationally brutal at AWS scale. Pull over HTTP with a visibility timeout needs none of that: consumers come and go, make stateless `ReceiveMessage` calls, and the only per-message state is an invisibility timer. Long-polling recovers the latency. The visibility timeout provides at-least-once. The result is a queue you cannot overwhelm and barely have to operate, at the cost of the duplicate-processing and tuning subtleties of sections 6 and 7. The choice is coherent: SQS sells "you never operate it," and pull-with-timeout is what makes that promise keepable.

## Case studies and war stories

### The prefetch flood that ate the night

A payments team ran a RabbitMQ consumer fleet processing webhook events, each taking about 80 ms. One night a partner sent a backfill of two million events. The consumers had been written before anyone read the QoS docs, so they ran with default (unlimited) prefetch. When they reconnected after a routine deploy, the broker, seeing a two-million-message backlog and consumers with no prefetch cap, delivered as fast as the channels would take. Each consumer's heap filled with hundreds of thousands of unacked event bodies within seconds. They OOM-crashed. On restart, the broker redelivered everything (nothing had been acked), and they crashed again. The fleet entered a crash-loop, processing almost nothing while pegged at 100% memory churn. The fix took one line — `basic_qos(prefetch_count=20)` — deployed at 3 a.m. With a bounded in-flight window, each consumer held at most 20 events, processed them, acked, and got 20 more. The backlog drained in forty minutes. The lesson: **unlimited prefetch is a loaded gun pointed at your consumers; set a count before you ever go to production.**

### The Kafka rebalance storm

A team enriching clickstream events ran a Kafka consumer with `max.poll.records=500` and the default `max.poll.interval.ms` of five minutes. Most events enriched in 10 ms, so a 500-record batch took 5 seconds — well within the interval. Then a feature shipped that, for one event type, made a synchronous call to a slow ML scoring service taking 800 ms each. A batch with 400 of those events now took over five minutes to process. The consumer blew past `max.poll.interval.ms`, the coordinator declared it dead and rebalanced, the partitions moved to another consumer that *also* hit a slow batch and *also* got kicked, and the group thrashed — every consumer cycling in and out, partitions never staying put long enough to make progress, and the same batches reprocessed over and over because offsets never committed. Lag climbed into the millions. The fix combined three things: drop `max.poll.records` to 50 so batches stayed small and fast; move the ML scoring to a bounded async worker pool with `pause()`/`resume()` so a slow downstream stopped blocking the poll loop; and raise the interval to ten minutes for headroom. The lesson: **in pull, the silent killer is polling too slowly — keep the work between the polls small and bounded, or the group decides you are dead.**

### The SQS double-charge

A subscription billing system used SQS with the default 30-second visibility timeout. Charging a card normally took 4 seconds, comfortably inside 30. But during a payment-processor incident, charges started taking 45 to 90 seconds as the processor retried internally. With the timeout at 30 seconds, every slow charge became visible again mid-processing, a second worker picked it up, and customers were charged twice — and then, when the slow first attempt finally succeeded, a third time. Support was flooded with double- and triple-charge complaints. Two fixes shipped: the visibility timeout went to 180 seconds (above the worst observed processing time), and the charge operation was made idempotent with the message ID as an idempotency key passed to the payment processor, so a duplicate `Charge` with the same key was a no-op. The lesson: **the visibility timeout is a contract — set it above your tail processing time and make side effects idempotent, because at-least-once means duplicates are not an edge case, they are inevitable.**

### The busy-poll bill

A startup ran a fleet of 200 small SQS-polling workers across several mostly-idle queues. The workers had been written with `WaitTimeSeconds=0` — busy-polling — because the author copied an example that did not set it. Each worker, on an empty queue, re-polled the instant it got an empty response: roughly 50 requests per second per worker. Two hundred workers across mostly-empty queues generated on the order of ten thousand requests per second, almost all empty, around the clock. The SQS bill for *requests* — billed per million `ReceiveMessage` calls — dwarfed the cost of the actual messages by two orders of magnitude. The fix was, again, one line: `WaitTimeSeconds=20`. Request volume dropped by more than 99%, latency actually *improved* (long-poll returns a message the instant it lands, faster than the next busy-poll cycle on average), and the bill fell off a cliff. The lesson: **never busy-poll a queue that can be empty; long-polling is cheaper and lower-latency at the same time.**

## When to reach for push vs pull (and when not to)

Choose **push** (RabbitMQ-style, broker delivers) when:

- **Latency is paramount and queues are usually shallow.** RPC-over-queue, real-time notifications, task distribution to waiting workers — push delivers the instant a message lands, with no poll interval.
- **You want simple, callback-style consumers** without managing a poll loop, and your team is disciplined about setting prefetch.
- **Your routing is complex** — exchanges, bindings, topic patterns — and the broker should decide which consumer gets what. Push pairs naturally with rich routing.
- But **only if you set a prefetch count.** Push without flow control is the single most reliable way to crash a consumer fleet under backlog.

Choose **pull** (Kafka- or SQS-style, consumer fetches) when:

- **Backlogs can be large and consumers can be slow.** Stream processing, batch enrichment, anything where the consumer may fall hours behind. Pull's automatic backpressure makes a slow consumer a dashboard number, not an outage.
- **Throughput and batching dominate.** A single `poll()` returning 500 records amortizes per-message overhead; pull is batch-native.
- **You need replay or independent consumer speeds.** Many consumer groups reading the same log at different offsets is pull's home turf.
- **You want minimal operational surface** (SQS): stateless HTTP pulls with a visibility timeout, no flow-control tuning, no connections to manage.
- But **use long-polling** (`WaitTimeSeconds`, `fetch.max.wait.ms`) so you don't pay the idle-latency and empty-request tax, and **keep the work between the polls** so you don't trip the liveness timeout.

The honest meta-rule: **you usually don't pick push or pull directly — you pick a broker for other reasons (routing, replay, ops burden), and the delivery model comes with it.** What you *do* control is configuring it correctly: prefetch on push, poll-interval discipline and long-polling on pull, and the acknowledgement timing and visibility timeout on both. Those configurations, not the model itself, are where the incidents come from. This sits directly upstream of [backpressure and flow control](/blog/software-development/message-queue/backpressure-and-flow-control) and [consumer optimization](/blog/software-development/message-queue/consumer-optimization-and-scaling), which take the flow-control knobs introduced here and push them to their operational limits.

## Key takeaways

- **Push vs pull is about who controls the rate.** Push: the broker initiates delivery (`basic.deliver`) and therefore must be told how fast to go via prefetch. Pull: the consumer initiates (`poll()`) and paces itself, so backpressure is automatic.
- **Push's flood is the broker's flow-control failure.** Push without a prefetch limit will hand a slow consumer everything and crash it. Set a prefetch count before production — it is non-optional.
- **Size prefetch to cover the round-trip, not to buffer.** `prefetch ≈ ceil((processing + RTT) / processing)` plus a small margin. Big prefetch hoards work, hurts fairness, and worsens redelivery on crash.
- **Pull's danger is polling too slowly.** Keep the work *between* `poll()` calls and the batch small enough to finish inside `max.poll.interval.ms`, or the group declares you dead and rebalances into a storm.
- **Ack after the work, never before.** Acking early recreates at-most-once loss while you believe you have at-least-once safety. The ack is your promise that the side effects are durably done.
- **Auto-ack is at-most-once; manual ack is at-least-once.** Choose the property you cannot live without. Kafka's auto-commit has a subtle loss window — disable it and `commitSync()` after processing for anything that matters.
- **nack/requeue is for transient failures; reject-to-DLQ is for poison.** Always carry a redelivery count and dead-letter past a threshold, or one bad message loops the fleet to death.
- **The visibility timeout is a deadline-based ack.** Set it above your p99 processing time, heartbeat-extend for outliers, and make side effects idempotent — at-least-once means duplicates are inevitable, not rare.
- **Long-poll, never busy-poll.** One line (`WaitTimeSeconds=20`) cuts empty-request cost by 99% and *lowers* latency. It makes pull feel almost like push.
- **You usually inherit the delivery model from your broker choice.** What you control is the configuration — and that is where every incident in this post came from.

## Further reading

- [Delivery semantics: at-most-once, at-least-once, exactly-once](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) — the formal contracts that acks and visibility timeouts implement.
- [Backpressure and flow control](/blog/software-development/message-queue/backpressure-and-flow-control) — the next post: prefetch and the in-flight window taken to their operational limits.
- [Consumer optimization and scaling](/blog/software-development/message-queue/consumer-optimization-and-scaling) — batch sizing, parallelism, and the pause/resume pattern.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — why pull is the only sane model for a replayable on-disk log.
- [RabbitMQ production architecture and scaling](/blog/software-development/system-design/rabbitmq-production-architecture-scaling) — how push flow control behaves at scale.
- [RabbitMQ Consumer Acknowledgements and Publisher Confirms](https://www.rabbitmq.com/docs/confirms) — the official reference for ack, nack, reject, and prefetch.
- [Kafka Consumer configuration](https://kafka.apache.org/documentation/#consumerconfigs) — `max.poll.records`, `max.poll.interval.ms`, `fetch.max.wait.ms`, and auto-commit.
- [Amazon SQS visibility timeout](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-visibility-timeout.html) — the canonical description of the deadline-based ack and long-polling.
