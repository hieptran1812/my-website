---
title: "Backpressure and Flow Control, End to End"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "The fast-producer slow-consumer problem and the mechanisms that keep it from melting your system: why the unbounded queue is a deferred out-of-memory crash, how bounded queues force the block-or-drop choice, and how pull, credit, reactive demand, TCP windows, and load shedding propagate the slow-down all the way back to the true source."
tags:
  [
    "message-queue",
    "backpressure",
    "flow-control",
    "load-shedding",
    "kafka",
    "rabbitmq",
    "reactive-streams",
    "distributed-systems",
    "event-driven",
    "tcp",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/backpressure-and-flow-control-1.webp"
---

Every queue you have ever deployed is lying to you about one thing, and the lie is comforting right up until the moment it isn't. The lie is that the queue is absorbing the spike. You watch a load test hit your service, the request rate doubles, and the queue depth climbs from a few hundred to a few hundred thousand without a single error in the logs. The dashboard is green. Latency on the producer side is flat. It feels like the queue did its job: it soaked up the burst, the consumers are chewing through it, everything is fine. And then twenty minutes later a process gets killed by the kernel out-of-memory reaper, or a broker node falls over, or the consumer's heap grinds into a garbage-collection death spiral, and the green dashboard turns into a 3 a.m. page. What actually happened is that the queue was never absorbing anything. It was *deferring*. It took a problem that should have shown up immediately — your consumer cannot keep up with your producer — and hid it inside a growing buffer until that buffer ran out of room, at which point the failure arrived all at once, with interest.

This post is about the single most important property a queue can have, which is the property almost nobody thinks about until it bites them: a **bound**, and the machinery that fires when you hit it. That machinery is **backpressure** — a signal that propagates *upstream*, from the thing that is overwhelmed back toward the thing producing the load, that says, in effect, *slow down, I cannot keep up*. Backpressure is what turns a queue from an infinite buffer that pretends to work into a finite shock absorber that tells the truth. When you cannot slow the source — because you do not control it, or because slowing it would itself be a failure — backpressure's sibling takes over: **load shedding**, the deliberate choice to drop, sample, or degrade work so that the work you *do* keep flows at a sustainable rate. The figure below sets up the whole argument in one picture: the same fast producer feeding two queues, one unbounded that quietly marches toward an out-of-memory crash, one bounded that hits its cap and forces a decision.

![Side-by-side comparison of an unbounded queue whose backlog grows until an out-of-memory crash versus a bounded queue that fills its cap in seconds and then blocks or drops](/imgs/blogs/backpressure-and-flow-control-1.webp)

By the end of this post you will be able to look at any pipeline — a Kafka topic, a RabbitMQ work queue, an in-process actor mailbox, a reactive stream, even a raw TCP socket — and answer four questions that decide whether it survives a spike: *Is the buffer bounded?* If so, *what happens when it fills — block or drop?* *Does the slow-down signal reach the true source, or does it just move the bottleneck?* And *can I slow the source at all, or do I have to shed?* These four questions are the same questions whether you are tuning a JVM `BlockingQueue` or a fleet of a thousand consumers, and the mechanisms that answer them — pull-based pacing, credit and prefetch, reactive `request(n)`, the TCP receive window, and load shedding — are variations on one idea. Get the idea and the specific broker config falls out of it. This builds directly on the [push vs pull and acknowledgements](/blog/software-development/message-queue/push-vs-pull-acknowledgements-how-consumers-read) post, where pull's "free backpressure" first showed up; here we take that thread and follow it all the way to its end.

## The fast-producer, slow-consumer problem

Strip away the brokers and the protocols and you are left with the most fundamental imbalance in all of computing: one component produces work faster than another consumes it. That is it. That is the entire problem. A web server accepts requests faster than the database can serve them. A log producer emits events faster than the indexer can index them. A camera streams frames faster than the model can run inference. An upstream microservice publishes orders faster than the fulfillment service can fulfill them. Every one of these is the same shape — a producer rate **P** and a consumer rate **C**, and the whole story turns on the sign of **P − C**.

When **P ≤ C**, you have no problem at all. The consumer keeps up, the queue between them stays near empty, and any buffer you put in the middle is just there to smooth out jitter — small bursts that the consumer drains before they accumulate. This is the regime everyone designs for and tests in, because it is the regime where everything is easy. The queue depth oscillates around some small number, latency is dominated by processing time, and the buffer is doing exactly what a buffer is supposed to do: decoupling the producer and consumer so a momentary stall in one does not immediately stall the other.

The trouble is that **P** and **C** are not constants. **P** spikes — a marketing email goes out, a retry storm kicks off, a batch job dumps a million records, a viral moment triples your traffic. **C** sags — a consumer node dies and its partitions get reassigned, a downstream dependency slows from 10ms to 100ms, a deploy ships a regression, the garbage collector pauses, a database index gets dropped. The dangerous regime is whenever **P > C**, even briefly, and especially when it persists. In that regime the queue depth grows at a rate of exactly **P − C** messages per second, and it will keep growing for as long as the imbalance lasts, because a queue has no mechanism of its own to make the producer slow down or the consumer speed up. It is a passive buffer. It accumulates.

### Why your instinct to "just add a bigger queue" is wrong

The first reflex of almost every engineer who hits this is to make the queue bigger. The consumer fell behind during a spike and the queue overflowed, so obviously the queue was too small, so let's give it more room. This reflex is not just wrong, it is *exactly backwards*, and understanding why is the whole point of this post.

A bigger queue does not fix a sustained **P > C** imbalance. It cannot. If your producer is doing 10,000 messages a second and your consumer is doing 6,000, you are accumulating 4,000 messages every second, forever, until something changes. A queue twice as large just takes twice as long to fill. You have not solved anything; you have bought time, and the amount of time you bought is exactly (extra capacity) ÷ (P − C). Worse, you have made the eventual failure *larger* and *later* — larger because there is more in-flight work to lose or replay when you crash, and later because the bigger buffer hides the problem for longer, so by the time it surfaces the imbalance may have been running for an hour and the backlog is enormous.

A bigger queue is only the right answer to a *transient* spike — a burst where **P > C** for a bounded window and then **P** drops back below **C** so the consumer can catch up. For that case, the buffer is a genuine shock absorber: it stores the temporary excess and releases it over the next few minutes. Sizing that buffer is a real engineering problem (how big a spike, how long, how fast can the consumer drain). But for a *sustained* imbalance, no finite buffer is large enough, and an infinite buffer is a crash waiting for a calendar. The only real fixes for a sustained imbalance are to make the consumer faster (scale out, optimize, parallelize — see [consumer optimization and scaling](/blog/software-development/message-queue/consumer-optimization-and-scaling)), to make the producer slower (backpressure), or to throw work away (load shedding). The buffer is never the fix. The buffer is the place the problem hides.

## The unbounded queue is a deferred crash

Let me make the danger concrete, because "the queue grows unbounded" sounds abstract until you put memory numbers on it, and then it becomes terrifying in a very specific, schedulable way.

An **unbounded queue** is any buffer with no maximum size: a Java `LinkedBlockingQueue` constructed with no capacity argument (its default cap is `Integer.MAX_VALUE`, which is effectively unbounded), a Go channel you keep writing to with a goroutine that drains slowly, a Python `asyncio.Queue(maxsize=0)`, a RabbitMQ queue with no `x-max-length`, an in-memory list you `append` to in a callback. They all share one property: when the producer outpaces the consumer, the only thing limiting the backlog is the amount of memory in the machine. And memory is a hard wall. When you hit it, you do not get a graceful slowdown. You get an `OutOfMemoryError`, or the Linux OOM killer picks a victim process and sends it `SIGKILL`, or your allocator starts thrashing and the whole process spends all its time in garbage collection doing no useful work. The failure is sudden, total, and usually takes the in-flight data with it.

#### Worked example: how long until the OOM

Let me do the arithmetic that every engineer should do before they deploy an unbounded queue, because it turns a vague fear into a number on the clock.

Suppose your producer is publishing at **P = 10,000 messages per second**, your consumer is draining at **C = 6,000 messages per second**, and each message is **2 KB** once you account for the payload plus the per-message object overhead the queue carries (headers, references, framing — in a JVM a "2 KB message" is often 3 to 4 KB of live heap once you count object headers and the entries that hold it, but let us be generous and say 2 KB). The queue backlog grows at:

```
growth = P - C = 10,000 - 6,000 = 4,000 messages/second
bytes/second = 4,000 msg/s x 2 KB = 8 MB/second
```

Now suppose this process has **16 GB** of heap available to the queue. Realistically the queue cannot use all 16 GB — the rest of the process needs memory, and the JVM will start GC-thrashing well before the heap is full — but let us take the optimistic ceiling. Time to fill:

```
16 GB / 8 MB/s = 16,384 MB / 8 MB/s = 2,048 seconds = ~34 minutes
```

So you have **about half an hour** from the moment the imbalance starts until the process dies. In practice you have *less*, because GC pressure makes the consumer slower as the heap fills, which makes **C** drop, which makes **P − C** grow, which fills the heap faster — a positive feedback loop that pulls the crash earlier, often into the 20-to-25-minute range. The point is not the exact number. The point is that an unbounded queue under a sustained imbalance has a **deterministic time of death**, and that time is computable from three numbers you already know. The queue is not absorbing the load. It is a countdown timer that nobody set deliberately.

And here is the cruel part: for those 34 minutes, *everything looks fine on the producer side*. The producer's `send()` returns instantly. Its latency is flat. Its error rate is zero. The only signal that anything is wrong is the queue-depth metric climbing — and if you are not alerting on queue depth and its rate of change, you will not see it until the page fires. The unbounded queue does not just defer the crash; it actively hides the warning signs by keeping the producer happy. This is why "the dashboard was green" is the most common opening line of an unbounded-queue post-mortem.

### The retry storm makes it worse

There is a vicious special case worth calling out, because it turns a manageable imbalance into a runaway one. Suppose your consumer is slow because a downstream dependency is failing, and your consumer retries on failure. Now every message gets processed multiple times before it succeeds or gives up, which *multiplies* the effective load, which makes **C** (the rate of messages actually leaving the queue) plummet while the producer keeps pushing. The backlog explodes far faster than the simple **P − C** math suggests. This is the retry-storm failure mode, and it deserves its own treatment, which it gets in [poison messages and retry storms](/blog/software-development/message-queue/poison-messages-and-retry-storms-containment). For now, just note that an unbounded queue is the accelerant that turns a retry storm from a bad afternoon into a full outage: the retries amplify the imbalance, and the unbounded buffer gives the amplified imbalance unlimited room to accumulate before it kills you.

## Bounded queues force the choice: block or drop

The fix for the deferred crash is almost insultingly simple to state: **put a maximum size on the queue.** Bound the buffer. Once the buffer has a cap, the imbalance can no longer hide inside unlimited memory — it has to surface, and it surfaces at the moment the queue hits its cap. At that moment, the system is *forced to make a decision it was avoiding*, and there are exactly two choices: **block the producer** (refuse to accept the new message until there is room — this is backpressure), or **drop something** (refuse to store the new message, or evict an old one — this is load shedding). There is no third option. A bounded queue under sustained overload *must* either block or drop. The whole discipline of flow control is about choosing which, choosing it deliberately, and propagating the consequence to the right place.

This is the most important sentence in the post, so let me say it plainly: **a bound does not make the overload problem go away — it makes the overload problem visible and forces you to handle it.** That is the entire value. An unbounded queue lets you pretend the problem does not exist until it kills you. A bounded queue rubs your face in the problem the instant it appears, which is exactly what you want, because now you can respond — slow the producer, shed load, scale the consumer, alert a human — instead of marching silently toward an OOM.

### Block: the producer waits

When a bounded queue is full and you choose to **block**, the producer's attempt to enqueue does not return — it parks, waiting until a slot opens (which happens when the consumer drains one message out the other end). In Java, `BlockingQueue.put()` does exactly this: it blocks the calling thread until space is available. In Go, writing to a full unbuffered or full buffered channel blocks the goroutine. This is backpressure in its rawest form: the slow consumer's slowness has propagated, through the full queue, all the way back into the producer's thread, which is now sitting idle instead of producing more work. The producer's effective rate has been *throttled down to the consumer's rate*. **P** has been forced to equal **C** by physics, not by politeness.

Blocking is the right default when **losing a message is unacceptable** and the producer is something you control and can afford to slow down — an internal job feeding an internal processor, an ETL stage, a thread pool. The cost of blocking is latency and, if you are not careful, propagation: a blocked producer is itself now slow, so whatever feeds *it* may start backing up, and the backpressure travels another hop upstream. That propagation is a feature (it eventually reaches and slows the true source) but it is also a hazard (if the true source is a user-facing request thread, blocking it means request timeouts and a thread-pool exhaustion that looks like an outage). We will come back to the end-to-end propagation question, because it is where most real systems get this subtly wrong.

### Drop: the queue refuses

When a bounded queue is full and you choose to **drop**, the new message (or an old one) is thrown away and the producer is *not* slowed — it keeps going at full speed, blissfully unaware, except that some fraction of its messages now vanish. This is load shedding. It is the right choice when **the producer cannot be slowed** (it is a firehose you do not control — a stream of sensor readings, market-data ticks, user clicks from millions of browsers) and when **a missing message is tolerable** — metrics you can sample, telemetry where you only need a statistical picture, frames where dropping one is invisible. Dropping keeps latency bounded and the system alive at the cost of completeness.

Different systems drop differently, and the policy matters. **Drop-newest** (reject the incoming message) is simple and keeps the oldest, most-aged work. **Drop-oldest** (evict the head to make room for the tail) keeps the freshest data, which is usually what you want for telemetry where stale readings are worthless — RabbitMQ's `x-overflow: drop-head` does exactly this. **Sample** (keep one in N) reduces the rate uniformly. **Degrade** (switch to a cheaper code path that produces a lower-fidelity result faster) raises **C** instead of lowering **P**. The taxonomy figure later in the post lays these out; for now the key insight is that a bounded queue *forces* you to pick a drop policy, whereas an unbounded queue lets you avoid the question until the OOM picks the most violent policy of all — drop everything, by crashing.

#### Worked example: sizing a bounded queue for a transient spike

The flip side of the deferred-OOM math is the legitimate use of a buffer: absorbing a *transient* spike. The question is how big to make the bound, and the answer comes from the spike's shape, not from "as big as memory allows." Suppose your steady state is **P = C = 6,000 messages/second** (balanced), and you want to survive a spike where the producer jumps to **P = 18,000 messages/second** (3x) for a burst lasting **10 seconds**, after which it returns to 6,000 and the consumer drains the accumulated backlog. During the spike the queue fills at **P − C = 18,000 − 6,000 = 12,000 messages/second** for 10 seconds, accumulating:

```
peak backlog = 12,000 msg/s x 10 s = 120,000 messages
```

So a bounded queue with a cap of **120,000** (plus a margin, say **150,000**) absorbs this specific spike with zero drops and zero blocking — the buffer does exactly the shock-absorber job it exists for. At 2 KB per message that is 150,000 x 2 KB = **300 MB**, a perfectly reasonable bound. After the spike ends, the consumer drains the 120,000-message backlog at its surplus rate of (C − P) = 6,000 − 0... no — once the spike ends P returns to 6,000 and C stays 6,000, so there is no surplus to drain the backlog. That is the catch: a buffer sized for a spike only recovers if **C exceeds P after the spike**, even briefly. If steady-state P equals C exactly, the backlog never drains — it sits there forever, one bad spike away from the cap. So sizing a buffer for a spike implicitly requires headroom: the consumer must be provisioned to run *faster* than the steady-state producer rate, so that after a spike it can claw back the backlog. A buffer is a loan against future consumer surplus; if there is no surplus, the loan is never repaid. This is the number most spike-sizing exercises forget, and it is why "the queue absorbed the spike but never recovered" is a real and common post-mortem. The correct provisioning is C strictly greater than steady-state P, with the buffer cap set to the spike's accumulated excess plus margin.

The pipeline figure below shows the block path concretely: the slow consumer drains the bounded queue slowly, the queue fills to its cap, and that "full" condition becomes a backpressure signal that travels back to throttle the producer. This is the propagation that an unbounded queue can never produce, because an unbounded queue is never full.

![Pipeline showing a slow consumer draining a bounded queue until it fills to cap, which emits a backpressure signal that travels upstream to throttle the producer](/imgs/blogs/backpressure-and-flow-control-2.webp)

## Pull-based natural backpressure

Now we get to the mechanisms, and the first one is the most elegant because it requires no extra machinery at all — backpressure falls out of the architecture for free. It is the **pull** model, and it is why Kafka, fundamentally, cannot be overwhelmed by its producers the way a naive push system can.

In a pull-based consumer, *the consumer sets the pace.* The broker does not push messages at the consumer whenever it feels like it; instead the consumer calls `poll()` (or `fetch`, or `receive`) when it is ready for more work, the broker hands back a batch, the consumer processes that batch, and only then does it call `poll()` again. The consumer's loop is fetch → process → fetch → process, and the rate of that loop is governed entirely by how fast the consumer can process. If the consumer is slow, it calls `poll()` less often, and it simply *asks for less*. There is no buffer accumulating on the consumer side, because the consumer never receives anything it did not explicitly request. The slow-down is automatic and requires no signal to be sent anywhere — the absence of a `poll()` call *is* the backpressure.

This is why a Kafka consumer that falls behind does not crash and does not melt the broker. It just **lags**. The messages it has not gotten to yet sit in the broker's log, on disk, where they were going to sit anyway — Kafka stores every message durably regardless of whether anyone has consumed it. A slow Kafka consumer has a growing *consumer lag* (the gap between the latest offset in the log and the offset the consumer has committed), but that lag is not memory pressure on the broker; it is just a number that says "this consumer is N messages behind." The broker is completely insulated from the consumer's slowness because the broker only ever does work when the consumer asks. The backpressure propagated from the consumer to itself: the consumer simply did not pull. The relationship between push, pull, and this free backpressure is the heart of the [push vs pull](/blog/software-development/message-queue/push-vs-pull-acknowledgements-how-consumers-read) post; here the takeaway is that pull is the cleanest possible flow-control mechanism because the entity that knows its own capacity — the consumer — is the entity that controls the rate.

### The catch: pull moves the bound, it does not remove it

Pull-based consumption gives you free backpressure between the broker and the consumer, but it does not magically make the imbalance disappear — it relocates the buffer to the broker's durable log, which is bounded by *disk* and *retention* rather than memory. That is a much friendlier bound (disk is cheap and large, and the broker is engineered to hold terabytes of log), but it is still a bound, and it still has a failure mode. If a Kafka consumer lags so far behind that the messages it has not consumed yet *age out past the retention window*, those messages get deleted by the broker's retention policy before the consumer ever sees them. The consumer pulls, asks for offset X, and the broker says "offset X no longer exists, the oldest I have is X+5,000,000" — the consumer has fallen off the retention cliff and silently lost five million messages. So pull does not eliminate the fast-producer slow-consumer problem; it converts "consumer OOM in 34 minutes" into "consumer data loss when lag exceeds retention," which buys you vastly more time (retention is usually measured in days, not minutes) and a clearer signal (consumer lag is a first-class, easily-alerted metric) but still demands that you eventually fix the imbalance.

```python
# Kafka pull-based consumer: the poll() loop IS the backpressure.
# The consumer fetches only when it is ready; a slow process loop
# simply polls less often and the broker is never overwhelmed.
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    "orders",
    bootstrap_servers="broker:9092",
    group_id="fulfillment",
    enable_auto_commit=False,
    # max_poll_records bounds how much one poll() can hand back, so
    # the in-memory batch is capped no matter how far behind we are.
    max_poll_records=500,
    # If processing a batch takes longer than this, the broker assumes
    # we are dead and rebalances. This is the safety valve that stops
    # a stuck-slow consumer from holding partitions forever.
    max_poll_interval_ms=300000,
)

for batch in iter(lambda: consumer.poll(timeout_ms=1000), None):
    for tp, records in batch.items():
        for record in records:
            process(record)          # slow? we just poll() less often
    consumer.commit()                # advance offset only after work is done
```

Notice `max_poll_records=500`. Even in a pull model you want to bound the batch size, because `poll()` returning an unbounded batch would reintroduce the unbounded-buffer problem inside the consumer's process: you would pull a million records into a list and OOM while processing them. The pull model gives you natural pacing *between* fetches, but you still bound the size of each fetch so a single fetch cannot blow your heap. Bounds all the way down.

## Push with credit and prefetch

Pull is elegant but it is not always available or appropriate. Push systems — where the broker actively delivers messages to a connected consumer as soon as they arrive — give you lower latency on idle queues (no polling delay) and a simpler consumer programming model (you register a callback and messages arrive). RabbitMQ is fundamentally a push broker: it dispatches messages to consumers over a long-lived channel. The problem is obvious from everything we have said so far — if the broker pushes messages at the consumer the instant they arrive, and the producer is fast, the broker will flood the consumer faster than it can process, and the consumer's in-memory buffer of received-but-not-yet-processed messages grows unbounded. Push, naively, is the unbounded-queue problem with extra steps.

The fix is **credit-based flow control**, which RabbitMQ implements as **prefetch** (the `basic.qos` setting, also called the prefetch count). The idea is to bolt a bound onto the push model: the consumer tells the broker "you may have at most **N** unacknowledged messages outstanding to me at any time." The broker pushes messages eagerly, but it counts how many it has sent that the consumer has not yet acknowledged, and the moment that count hits **N**, the broker *stops pushing* and holds the rest. As the consumer processes messages and sends acks back, each ack frees up one credit, and the broker pushes one more. The unacknowledged-message count is a bounded credit budget, and the consumer's in-flight buffer can never exceed it. Push regains the bound that pull had for free.

![Side-by-side comparison of unlimited push that overwhelms a consumer's unbounded buffer versus pull or bounded prefetch that paces delivery to the consumer's processing speed](/imgs/blogs/backpressure-and-flow-control-4.webp)

The before-and-after above is the whole story of why prefetch exists: on the left, the broker pushes everything ready and the consumer's buffer grows without limit toward an OOM or a GC stall; on the right, prefetch (or pull) bounds the buffer to a small number of messages paced to processing speed. Credit-based flow control is the general name for this pattern, and you will find it not just in RabbitMQ's prefetch but in AMQP's channel-level flow control, in gRPC's HTTP/2 flow-control windows, in the Disruptor's claimed-sequence gating, and — as we will see — in TCP itself. The graph below shows the credit loop explicitly: the consumer grants a credit budget, the producer (or broker) sends only up to that budget, and each ack replenishes one credit, so in-flight work is capped at the budget size.

![Graph of credit-based flow control where a consumer grants a prefetch budget of twenty, the producer sends only up to the budget, in-flight messages stay at or below twenty unacked, and each ack replenishes one credit while a zero budget makes the producer wait](/imgs/blogs/backpressure-and-flow-control-3.webp)

#### Worked example: sizing prefetch to keep a consumer busy

Here is the question that actually matters when you configure prefetch: what number do you set it to? Too low and your consumer starves — it finishes a message, sends the ack, and then sits idle waiting for the network round trip to deliver the next one. Too high and you have reintroduced an unbounded-ish buffer and lost the point. The right value comes from a little queueing arithmetic.

Suppose each message takes **50 ms** to process (this is your **C** per message: one consumer does 1 ÷ 0.050 = **20 messages/second**). Suppose the network round trip between broker and consumer — the time for an ack to travel to the broker and a new message to come back — is **10 ms** (a `latency` of 10 ms). If you set **prefetch = 1**, then the cycle for each message is: process (50 ms) + send ack + wait for the next message to arrive (10 ms round trip) = 60 ms per message. Your consumer is busy 50 of every 60 ms — it spends **17% of its time idle**, waiting on the network. Throughput drops from 20 msg/s to 1000 ÷ 60 ≈ **16.7 msg/s**. You left 17% of your consumer's capacity on the floor.

To eliminate that idle time you want enough messages buffered locally that the consumer never has to wait for the network: while it processes message K, message K+1 should already be sitting in its local buffer. The minimum prefetch to keep the consumer continuously busy is roughly:

```
prefetch >= ceil( (processing_time + round_trip) / processing_time )
         =  ceil( (50ms + 10ms) / 50ms )
         =  ceil( 1.2 )
         =  2
```

So **prefetch = 2** is the minimum to hide the round trip — one message being processed, one already buffered. In practice you set it a bit higher (say **prefetch = 10** or **20**) to absorb jitter in processing time and network latency, but you keep it *bounded and small*. The classic mistake is the opposite of starvation: someone reads that prefetch improves throughput, sets `prefetch = 10000` or leaves it unlimited, and now a single consumer hoards ten thousand messages in its local buffer. With **2 KB** messages that is 20 MB of buffer per consumer; with a hundred consumers it is 2 GB of messages sitting in client buffers instead of in the broker, and worse, if that consumer dies, all ten thousand unacked messages have to be redelivered, and your "fair" round-robin dispatch has become wildly unfair because one consumer grabbed a huge share. The sweet spot is small: large enough to hide the round trip and absorb jitter, small enough that the buffer is trivial and redelivery on crash is cheap. For most workloads that is somewhere between **5 and 50**.

```python
# RabbitMQ prefetch (basic.qos) = credit-based flow control.
# prefetch_count caps unacknowledged messages outstanding to THIS
# consumer. The broker stops pushing once it hits the cap and resumes
# as acks come back. This bounds the consumer's in-flight buffer.
import pika

connection = pika.BlockingConnection(pika.ConnectionParameters("rabbit"))
channel = connection.channel()

# Grant a credit budget of 20 unacked messages. Tune via the
# processing_time + round_trip formula above; keep it small.
channel.basic_qos(prefetch_count=20)

def on_message(ch, method, props, body):
    process(body)
    # The ack returns one credit; the broker may now push one more.
    ch.basic_ack(delivery_tag=method.delivery_tag)

# auto_ack=False is mandatory — auto-ack disables flow control entirely
# because the broker considers a message "done" the instant it is sent,
# so prefetch never throttles and you are back to unbounded push.
channel.basic_consume(queue="tasks", on_message_callback=on_message, auto_ack=False)
channel.start_consuming()
```

The comment on the last line is load-bearing and catches people constantly: **`auto_ack=True` disables flow control.** With auto-ack, RabbitMQ considers a message acknowledged the moment it puts it on the wire, so the unacked count never rises, so prefetch never throttles, so you have unbounded push again — and you have *also* lost at-least-once delivery, because a message the broker thinks is acked but your consumer hasn't processed is gone if the consumer crashes. Auto-ack turns off both the safety net and the flow control in one go. If you take one operational rule from this section: manual acks plus a small prefetch is the RabbitMQ flow-control contract, and breaking either half breaks both.

There is a second subtlety with prefetch that bites teams running competing consumers, and it is worth a beat because it shows how flow control and fairness are the same mechanism viewed from two angles. RabbitMQ dispatches messages round-robin across the consumers on a queue, but the prefetch count interacts with that dispatch in a way that determines *fairness under uneven processing time*. With prefetch = 1, each consumer holds exactly one unacked message at a time, so when a consumer finishes, it gets the next available message — meaning a fast consumer naturally pulls more work and a slow consumer pulls less, and the load self-balances. With a large prefetch, the broker pre-assigns a big chunk of messages to each consumer up front; if one consumer then hits a slow message (or stalls), the chunk it is holding sits idle in its buffer while other consumers go hungry, and throughput drops even though work is available. So the small-prefetch rule is not only a memory-safety rule — it is *also* a fairness rule. The bound on in-flight messages is simultaneously the bound on how much a single slow consumer can hoard, which is why "keep prefetch small" shows up independently in both the flow-control literature and the work-distribution literature. They are describing the same knob.

### Credit windows versus prefetch counts

A point of vocabulary that trips people moving between systems: "credit" and "prefetch" describe the same mechanism but count different things in different brokers, and conflating them leads to mis-sizing. RabbitMQ's prefetch counts *messages* — N unacknowledged messages, regardless of their size — which is simple but means a prefetch of 20 could be 20 KB or 20 MB depending on message size, so for highly variable message sizes you may also want `prefetch_size` (a byte limit) to bound memory rather than count. AMQP 1.0 and protocols like it use a *link credit* that is also message-count-based but flows in an explicit `flow` performative the receiver sends. gRPC over HTTP/2 uses a *byte-based* flow-control window per stream, much closer to TCP. The practical consequence: when you port a flow-control configuration from one system to another, do not copy the number — recompute it, because a "credit of 100" might mean 100 messages in one system and 100 KB in another, and those size very differently. The mechanism is universal; the unit is not, and the unit is where the bugs live.

## Reactive streams and demand signaling

The credit idea generalizes beautifully, and the cleanest place to see it generalized is the **Reactive Streams** specification — the standard behind RxJava's `Flowable`, Project Reactor's `Flux` and `Mono`, Akka Streams, and the `java.util.concurrent.Flow` API that shipped in Java 9. Reactive Streams exists to solve exactly the problem of this post — asynchronous stream processing with non-blocking backpressure — and it solves it by making the credit *the protocol*. Instead of a producer pushing items at a subscriber, the subscriber explicitly signals **demand**: it calls `request(n)`, which means "I am ready for at most **n** more items," and the publisher is contractually forbidden from emitting more than the outstanding demand. The subscriber drives the rate by deciding when, and how much, to request.

This is credit-based flow control elevated to a first-class programming model. The `Subscription.request(n)` call *is* the grant of **n** credits. The publisher emits at most **n** `onNext` calls, then waits for more demand. A slow subscriber simply requests slowly — or requests one at a time, processing each before requesting the next — and the publisher, however fast its underlying source, cannot get ahead of the demand. The backpressure is woven into the four-method protocol (`onSubscribe`, `onNext`, `onError`, `onComplete`) so tightly that a correctly-implemented reactive publisher *cannot* overwhelm a subscriber. The bound is not a config knob bolted on the side; it is the contract.

```java
// Reactive Streams: the Subscriber drives the rate via request(n).
// This is credit-based flow control as a protocol — the publisher
// may never emit more items than the outstanding demand.
class BackpressuredSubscriber implements Flow.Subscriber<Order> {
    private Flow.Subscription subscription;
    private static final int BATCH = 16;   // request 16 credits at a time
    private int processed = 0;

    @Override
    public void onSubscribe(Flow.Subscription s) {
        this.subscription = s;
        s.request(BATCH);                  // initial demand: send me 16
    }

    @Override
    public void onNext(Order order) {
        process(order);                    // slow work paces the stream
        if (++processed % BATCH == 0) {
            subscription.request(BATCH);   // replenish demand only when ready
        }
    }

    @Override public void onError(Throwable t) { /* ... */ }
    @Override public void onComplete() { /* ... */ }
}
```

The subtle and important property here is that this works **across asynchronous boundaries without blocking a thread.** The blocking-queue version of backpressure parks a thread on a full `put()`. The reactive version never blocks: a slow subscriber just *doesn't call `request`*, and the publisher, having no outstanding demand, simply does nothing — no thread is parked, no buffer grows. This is what makes reactive backpressure suitable for high-concurrency, low-thread-count systems (a Netty event loop, a single-threaded reactor) where blocking a thread would be catastrophic. The demand signal carries the backpressure instead of a blocked thread carrying it.

### Where reactive backpressure leaks

Reactive Streams is not magic, and it has a well-known escape hatch that re-creates the unbounded-queue problem if you reach for it carelessly. The spec governs backpressure *within* a reactive pipeline, but the moment your source is something that does not itself respect demand — a callback from a non-reactive library, a UI event stream, a sensor that emits whether or not you asked — you have a backpressure boundary, and you must choose an operator to bridge it. Reactor and RxJava give you `onBackpressureBuffer` (buffer the excess — *unbounded by default*, which is the OOM trap dressed in reactive clothing), `onBackpressureDrop` (load-shed the excess), `onBackpressureLatest` (keep only the newest, drop-oldest shedding), and bounded variants of buffer. If you use `onBackpressureBuffer()` with no size argument on a fast non-demand source, you have built an unbounded queue inside your "backpressure-safe" reactive pipeline, and it will OOM exactly as the very first figure warned. The lesson recurs: backpressure only works if it propagates all the way to a source that can actually slow down; where it can't, you are back to bounding and shedding.

## The TCP flow-control analogy

If all of this credit-and-demand machinery feels like something the networking people must have figured out decades ago, that is because they did, and the analogy is exact enough to be worth making explicit, because once you see that TCP flow control *is* credit-based backpressure, the whole pattern clicks into place as a fundamental, not a broker quirk.

TCP has the same problem every queue has: the sender can put bytes on the wire faster than the receiver can process them off the wire and hand them to the application. If the sender blasts away with no regard for the receiver's pace, the receiver's kernel buffer (the socket receive buffer) fills up, and bytes have nowhere to go — they would have to be dropped, forcing retransmission, wasting the network. TCP solves this with the **receive window**: in every acknowledgement it sends back, the receiver advertises a number — the window size — that says "I have this many bytes of free buffer space; do not send me more than this without waiting for further acks." The sender is contractually bound to keep its unacknowledged in-flight bytes at or below the advertised window. As the receiving application reads bytes out of the socket buffer and frees space, the receiver advertises a larger window, granting the sender permission to send more. If the application stops reading entirely, the buffer fills, the advertised window shrinks to **zero**, and the sender *stops* — it is blocked at the transport layer until the receiver opens the window again.

Read that paragraph again and map it onto the prefetch paragraph from earlier. The advertised window is the credit budget. The receiver granting a larger window as it drains the buffer is the consumer's ack replenishing a credit. A zero window is prefetch = 0, the producer parked waiting. TCP flow control is credit-based backpressure operating on bytes instead of messages, built into the most-used protocol on Earth, and it has worked at planetary scale for forty years. When you configure a RabbitMQ prefetch or call `request(n)` in a reactive stream, you are re-implementing the TCP receive window at the application layer. The receiver always sets the pace by advertising how much it can take; the sender always respects it. Same idea, different unit.

The reason this analogy is operationally useful and not just a cute observation is that **TCP backpressure can save you even when your application has none** — and it can also surprise you. Consider a RabbitMQ producer publishing to a broker over a TCP connection. If the broker is overwhelmed and stops reading from that connection's socket fast enough, the TCP receive window on the broker side shrinks, which eventually blocks the *producer's* `write()` syscall at the OS level, which (if the producer is doing synchronous publishes) blocks the producer. Backpressure propagated all the way from an overwhelmed broker, through the kernel's TCP stack, into the producer's thread, *without any application-level flow control at all* — just because TCP is doing its job. This is genuinely useful: it is a last-line-of-defense backpressure that fires even when you forgot to configure anything. But it is also a trap for the unwary, because it means a slow broker can silently block your producer threads via TCP, and if those threads are serving user requests, you have a TCP-window-induced outage that does not show up in any application metric. The flow control was there; you just didn't know it was the thing throttling you.

## Load shedding when you cannot slow the source

Everything so far has assumed you *can* slow the source — block the producer, withhold demand, shrink a window. But there is a large and important class of systems where you simply cannot, and pretending otherwise is how you get an outage. You cannot slow the source when **you do not control it**: the source is millions of users' browsers firing click events, a fleet of IoT devices emitting telemetry on their own schedule, a financial market data feed that ticks at whatever rate the market ticks, an upstream you have no authority over. You also should not slow the source when **slowing it is itself a failure**: if the "producer" is a user-facing request thread, blocking it to apply backpressure means the user's request hangs and eventually times out — you have converted a capacity problem into a latency-and-availability problem, which is usually worse. When backpressure cannot reach a source that can slow down, the only sustainable response is to **shed load**: deliberately drop, sample, or degrade work so that the work you keep flows at a rate the system can sustain.

Load shedding is not failure — it is *controlled* failure, which is the only kind worth having. The choice is never "shed load" versus "serve everything," because serving everything is not on the menu when **P > C** and you cannot lower **P**. The choice is "shed load deliberately, picking what to drop and keeping the system healthy" versus "shed load chaotically, by collapsing — OOM, cascading timeouts, total brownout — and dropping *everything* including the requests you most wanted to serve." Deliberate shedding strictly dominates collapse. The hard part is doing it well, which means three things: shedding *early* (before the system is already melting), shedding the *right* work (cheap and low-value first, expensive and high-value last), and shedding *visibly* (so the drops are a metric you alert on, not a silent loss).

The mechanisms, roughly in order of bluntness:

- **Drop.** Refuse work when a bound is hit. The bounded queue's drop policy (drop-newest, drop-oldest) is the simplest form. At the service layer, this is returning `503 Service Unavailable` (or `429 Too Many Requests`) when a concurrency or queue limit is exceeded, instead of accepting the request into an overflowing buffer.
- **Sample.** Keep one in **N**. The standard move for high-volume telemetry, traces, and metrics: you do not need every span to understand a system's behavior, you need a statistically representative fraction, so you keep 1% or 10% and drop the rest, recovering the true rate by scaling up.
- **Degrade.** Switch to a cheaper code path that produces a lower-fidelity result faster — serve a cached or approximate response, skip an optional enrichment, return a smaller result set. Degradation raises **C** (the system processes each item faster) instead of lowering **P**, which is often more palatable because every request still gets *an* answer, just a cheaper one.
- **Prioritize and shed by class.** Not all work is equal. Shed the low-priority traffic first (background re-indexing, analytics events, non-paying-tier requests) and protect the high-priority traffic (checkout, authentication, paying customers). This requires the system to *know* the priority, which means carrying it in the message or request, which is design work you do before the incident, not during it.

The matrix below lines up the five flow-control mechanisms we have covered against two questions — who paces the source, and does the mechanism actually slow the source — and makes the central point of this whole section visible: every mechanism except shedding works by slowing the source, and shedding is the only one that handles a source you cannot slow.

![Matrix comparing pull, credit prefetch, TCP window, rate limit, and load shed across who paces the source and whether each one actually slows the source, with shedding the only mechanism that drops load instead of slowing it](/imgs/blogs/backpressure-and-flow-control-5.webp)

### Rate limiting is shedding at the door

A close cousin of load shedding that deserves its own mention is **rate limiting** — a token bucket or leaky bucket placed at the *entrance* to your system that caps the rate at which work is admitted in the first place. Where load shedding drops work that has already entered when an internal bound is hit, rate limiting refuses work at the door before it consumes any internal resources, which is cheaper and cleaner. A token-bucket limiter holds tokens that refill at the sustainable rate **C**; each admitted request consumes a token; when the bucket is empty, new requests are rejected (or queued briefly, in a small bounded buffer that smooths bursts). The bucket size sets how big a burst you tolerate; the refill rate sets the long-run admitted rate. Rate limiting is how you protect a system from a source you cannot slow by *making* the admitted load slow, dropping the overflow at the boundary where it is cheapest to drop.

```python
# Token-bucket rate limiter: shed load at the door so the internal
# pipeline never sees more than the sustainable rate C.
import time

class TokenBucket:
    def __init__(self, rate_per_sec, burst):
        self.rate = rate_per_sec      # refill rate = sustainable C
        self.capacity = burst         # max burst we absorb
        self.tokens = burst
        self.last = time.monotonic()

    def allow(self):
        now = time.monotonic()
        # Refill tokens for the elapsed time, capped at capacity.
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True               # admit: a token was available
        return False                  # shed: bucket empty, drop at the door

limiter = TokenBucket(rate_per_sec=6000, burst=10000)  # admit 6k/s, tolerate 10k burst

def handle(request):
    if not limiter.allow():
        return reject(503)            # deliberate, visible, early shed
    enqueue(request)                  # only admitted work enters the pipeline
```

The timeline below ties the spike story together: a burst arrives, the bounded queue absorbs it first (the buffer is doing its real job — smoothing a transient), then as the buffer fills, backpressure engages and throttles whatever source it can reach, and finally, for the portion of the load coming from sources that *cannot* be slowed, load shedding kicks in to drop the excess and hold the line at the cap — until the spike ends and the backlog drains. Absorb, then throttle, then shed: that is the correct order of defenses, each taking over where the previous one runs out.

![Timeline of a traffic spike where the bounded queue first absorbs the burst, then fills and engages backpressure to throttle the source, then triggers load shedding to drop excess at the cap before the spike ends and the backlog drains](/imgs/blogs/backpressure-and-flow-control-6.webp)

## End-to-end backpressure propagation

Here is the failure mode that catches even experienced engineers, the one that turns a system *with* backpressure into a system that still falls over: **backpressure that does not propagate all the way to the true source just moves the bottleneck — it does not relieve it.** You can put a beautifully bounded queue with perfect block-or-drop semantics in the middle of your pipeline, and if the slow-down it generates is absorbed by an *unbounded* buffer one hop upstream, you have accomplished nothing except relocating the OOM to a different process.

Picture a three-stage pipeline: an HTTP ingest service receives requests and writes them to a bounded internal queue; a worker pool reads from that queue and writes results to a second bounded queue; a publisher reads the second queue and ships to Kafka. Suppose the publisher slows down (Kafka is having a bad day). The second queue fills, applies backpressure, and blocks the worker pool — good. The worker pool, now blocked, stops reading from the first queue, which fills and applies backpressure to the HTTP ingest service — good. But now the HTTP ingest service's request-handling threads are blocked trying to enqueue. What happens next depends *entirely* on what is upstream of those threads, and this is where systems quietly fail. If the HTTP server has a bounded thread pool and a bounded accept queue, the backpressure propagates correctly: blocked threads mean the accept queue fills, the server stops accepting connections, and clients get connection refusals or timeouts — the backpressure reached the true source (the clients) and they are now slowed. But if the HTTP server has an *unbounded* request queue or spawns an unbounded number of threads (one per request), then the backpressure hits that unbounded buffer and *vanishes* — requests pile up in threads or in the accept backlog, memory grows, and the ingest service OOMs. You moved the crash from the publisher to the ingest service. The bottleneck moved; the danger did not.

The rule is stark: **backpressure must propagate, unbroken, all the way to a source that can actually slow down — a source with a bound of its own or the ability to reject.** A single unbounded buffer anywhere in the chain is a backpressure sink: it absorbs the slow-down signal and converts it back into accumulating memory, defeating every bounded stage downstream of it. The true source is wherever the load *originates*: real users (who slow down when you return 503s or when their requests hang), an upstream service (which slows down when its own publish blocks or errors), a scheduled job (which slows down when its writes block). Your job is to trace the chain from the bottleneck back to that origin and verify there is no unbounded buffer anywhere along the way. The stack figure shows the four layers where the bound must hold simultaneously — application demand, the queue's own bound, the transport window, and the OS limits — because a leak at any one layer un-bounds the layer above it.

![Stack of four flow-control layers from application demand through the queue bound and transport window down to OS limits, showing that a leak at any layer un-bounds the layer above it](/imgs/blogs/backpressure-and-flow-control-7.webp)

### The taxonomy: three families of response

Step back from the specific mechanisms and they sort into exactly three families, and naming the families is what lets you reason about a new system quickly. When **P > C**, you can **slow the source** (pull pacing, credit/prefetch, reactive demand, TCP windows — all the backpressure mechanisms, which require a source you control and can slow), **bound the buffer** (cap the queue and block at the cap — the prerequisite that forces the block-or-drop choice and is what makes backpressure even *possible*), or **shed the load** (drop, sample, degrade, rate-limit — the only family that works on a source you cannot slow). Every real system uses some combination: it bounds its buffers (always — this is non-negotiable), it slows the sources it can reach (backpressure), and it sheds the load from sources it cannot (shedding). The tree below lays out the taxonomy so you can place any mechanism you encounter into one of the three families and immediately know what assumption it depends on.

![Tree taxonomy of overload responses branching into slowing the source, bounding the buffer, and shedding the load, each with its concrete mechanisms](/imgs/blogs/backpressure-and-flow-control-8.webp)

The final decision — the one a bounded queue forces and you cannot dodge — is the block-versus-drop choice when the buffer is full and the source is something you must handle right now. Block preserves every message at the cost of stalling the source and propagating latency upstream; drop protects latency and keeps the source running at the cost of losing data. The before-and-after below frames the trade cleanly. There is no universally right answer: a payments pipeline blocks (never lose a charge, accept the latency), a metrics pipeline drops (never stall the app to record a counter, accept the gap). Knowing which one your data is — *is a lost message a bug or a shrug?* — is the question that decides the entire flow-control design, and it is a product question disguised as an engineering one.

![Side-by-side comparison of blocking the producer to apply backpressure with zero loss but rising latency versus dropping messages to load-shed with bounded latency but data loss](/imgs/blogs/backpressure-and-flow-control-9.webp)

## Case studies and war stories

Theory is cheap. Here are four scenarios — some named public incidents, some composite patterns drawn from common production failures — and the specific flow-control lesson each one carved into someone's memory.

### The unbounded `LinkedBlockingQueue` and the thread pool that wasn't

This is the single most common Java backpressure bug, and it lives inside `ThreadPoolExecutor`. The standard way people build a bounded thread pool is `new ThreadPoolExecutor(core, max, ...)` with a work queue, and the trap is the queue. If you pass `new LinkedBlockingQueue<>()` with no capacity argument, the work queue is *unbounded*, which has two consequences nobody intends. First, the `maximumPoolSize` parameter becomes a dead letter — the pool will never grow past `corePoolSize`, because new threads are only created when the queue is *full*, and an unbounded queue is never full. So your "pool of up to 200 threads" is permanently a pool of `core` threads. Second, and worse, every task that arrives when the core threads are busy goes into the unbounded queue, which grows with exactly the deferred-OOM dynamics from earlier in this post. A team I know shipped a service with `Executors.newFixedThreadPool(50)` — which is internally a `ThreadPoolExecutor` with an unbounded `LinkedBlockingQueue` — feeding it from a fast upstream. Under a traffic spike the 50 threads couldn't keep up, the queue grew to several million `Runnable` objects over about 25 minutes, and the JVM died. The fix was a bounded queue (`new ArrayBlockingQueue<>(1000)`) plus a `RejectedExecutionHandler` — specifically `CallerRunsPolicy`, which makes the *submitting thread* run the rejected task itself, which slows the submitter, which is backpressure propagating to the source. The lesson: `Executors.newFixedThreadPool` and `newCachedThreadPool` are unbounded-queue traps; always build the executor by hand with an explicit bounded queue and a rejection policy.

### Kafka consumer lag and the retention cliff

A streaming team ran a Kafka consumer group that was comfortably keeping up — until a code change in the processing logic tripled per-message processing time. **C** dropped below **P**, and because Kafka is pull-based, nothing crashed; the consumer just started lagging. Consumer lag is a benign-looking metric when it's small, and theirs climbed slowly enough that the gradual increase didn't trip a static threshold alert. Over about six hours the lag grew to the point where the oldest un-consumed messages aged past the topic's 24-hour... no — past the topic's much shorter retention (they had set retention to a few hours to save disk, a decision made by a different team for a different reason), and the broker started *deleting* messages the consumer had not yet read. The consumer, pulling along, suddenly found its committed offset pointing at a deleted segment, reset to the earliest available offset per its `auto.offset.reset` policy, and silently skipped the gap — permanent data loss, discovered days later in a reconciliation report. The lessons stack up: pull-based backpressure protects the broker but *converts* the imbalance into a lag-vs-retention race, you must alert on lag *rate of change* not just absolute lag, retention is a hidden bound that the consumer's flow control depends on, and a too-short retention turns "consumer is slow" into "consumer loses data." Backpressure didn't fail; it relocated the failure to a boundary nobody was watching.

### The reactive pipeline that bought an OOM with `onBackpressureBuffer`

A team migrated a data pipeline to Project Reactor specifically *for* its backpressure guarantees, then hit a source that didn't honor demand — a third-party client library that invoked a callback for every event whether or not anything was ready. To bridge the non-demand source into their reactive flow, they reached for `Flux.create(...)` and, finding that it dropped or errored under load, wrapped it in `.onBackpressureBuffer()` — the parameterless version, which buffers *without bound*. They had, inside a framework chosen for backpressure safety, rebuilt the exact unbounded queue from figure one. Under a burst the buffer grew until the JVM OOMed, and the post-mortem's most painful line was "we adopted reactive streams to prevent this." The fix was `onBackpressureBuffer(10_000, dropped -> meter.increment(), BufferOverflowStrategy.DROP_OLDEST)` — a *bounded* buffer with a drop-oldest shedding policy and a metric on the drops. The lesson: a framework's backpressure guarantees end exactly where a non-demand source begins, and the operator you choose at that boundary is a real block-or-drop decision; the unbounded default is the trap, and "reactive" on the label does not save you from it.

### TCP backpressure as the silent thread-pool killer

A service published synchronously to a downstream broker over a single shared connection. The broker had a bad GC pause and stopped reading its socket fast enough; its TCP receive window shrank toward zero; the publishing service's `write()` calls blocked at the OS level on a full send buffer. Because the publishes were synchronous and happening on the service's request-handling threads, those threads blocked inside `write()`, the request thread pool exhausted, and the service started returning 503s to *its own* upstream callers — for requests that had nothing to do with the broker. The operators stared at dashboards showing the broker recovered minutes ago while their service was still wedged, with no application-level error explaining why, because the throttle was happening three layers down in the TCP stack. TCP flow control had propagated backpressure from the broker all the way into their request threads, exactly as designed, and the design surprised them because they'd never thought of TCP as part of their flow-control story. The lessons: TCP gives you backpressure you didn't ask for and can't see in app metrics; synchronous publishing on request threads couples your availability to your slowest downstream via the kernel; the fix is to publish asynchronously with a bounded in-memory queue and a shed/block policy you *chose*, so the backpressure surfaces as a metric you own rather than as mysteriously hung threads.

## When to reach for each mechanism (and when not to)

The mechanisms are not interchangeable; each assumes something about your source and your data, and picking the wrong one for your assumptions is how you get the war stories above. Here is the decision framework.

| Situation | Reach for | Why | Avoid |
|---|---|---|---|
| You control the source and cannot lose data | **Block (backpressure)**: bounded queue + blocking put, or pull, or credit | Preserves every message; slows the source you own | Dropping (loses data you must keep) |
| Source is a firehose you do not control | **Load shed**: drop / sample / rate-limit at the door | You cannot slow it, so admit only the sustainable rate | Blocking (nothing to block; buffer just OOMs) |
| Source is a user-facing request thread | **Shed via 503/429**, not blocking | Blocking a request thread becomes a latency/availability failure | Blocking (hangs users, exhausts thread pool) |
| Pull broker (Kafka) consumer falls behind | **Let it lag**, alert on lag rate, scale consumers | Pull gives free backpressure; lag is the signal | Ignoring lag until it hits the retention cliff |
| Push broker (RabbitMQ) consumer | **Small prefetch (5–50) + manual ack** | Bounds in-flight buffer to processing speed | `auto_ack` or huge prefetch (disables flow control) |
| Async stream across non-blocking boundary | **Reactive `request(n)`** with bounded bridge operators | Non-blocking demand signaling; no parked threads | `onBackpressureBuffer()` unbounded at a non-demand source |
| Telemetry / metrics / traces | **Sample (1-in-N) and drop-oldest** | Completeness is unnecessary; freshness and survival matter | Blocking the app to record a counter |
| Multi-tier internal pipeline | **Bound every stage; verify end-to-end propagation** | One unbounded stage is a backpressure sink that defeats the rest | Any single unbounded buffer in the chain |

The meta-rule that sits above the table: **always bound your buffers, no exceptions.** Bounding is not a mechanism you choose among; it is the prerequisite for choosing at all. An unbounded buffer takes the decision out of your hands and gives it to the OOM killer. Once everything is bounded, you choose block-versus-drop per stage based on whether that stage's data is loss-tolerant, and you choose backpressure-versus-shed based on whether the relevant source can be slowed. Those two binary questions, answered per stage, *are* your flow-control design.

A few honest "do nots." Do not apply blocking backpressure to a user-facing synchronous request path expecting it to be graceful — it converts a capacity problem into hung requests and thread exhaustion, which is worse; shed with a fast 503 instead. Do not set prefetch or buffer sizes large "to be safe" — large buffers are unbounded buffers in slow motion and they make redelivery-on-crash expensive and dispatch unfair; small and bounded is safe, large is the danger. Do not assume a framework labeled "reactive" or "backpressure-aware" has saved you — it has saved you exactly up to the first non-demand source, and no further. And do not deploy any queue without a metric on its depth *and the rate of change of its depth*, because depth alone tells you where you are and the rate tells you when you will hit the wall — and the rate is the number that gives you time to react.

## Key takeaways

- **The fast-producer slow-consumer imbalance (P > C) is the root problem**, and no buffer is the fix — a buffer only defers the consequence. The fixes are a faster consumer, a slower producer (backpressure), or less work (shedding).
- **An unbounded queue is a deferred out-of-memory crash with a computable time of death.** At 10k in, 6k out, 2 KB each, a 16 GB heap dies in roughly half an hour — sooner once GC pressure drags the consumer down. The producer side stays green the whole time, which is what makes it dangerous.
- **A bound does not solve overload; it makes overload visible and forces the block-or-drop choice.** That is its entire value. A bounded queue under sustained overload *must* either block the producer or drop messages — there is no third option.
- **Pull-based consumption (Kafka) gives backpressure for free** because the consumer sets the pace by deciding when to poll. It does not erase the imbalance; it relocates the buffer to the durable log, converting "consumer OOM in minutes" into "consumer data loss when lag exceeds retention."
- **Push (RabbitMQ) needs credit-based flow control** — prefetch caps unacknowledged in-flight messages. Size it just above `(processing + round_trip) / processing` to hide the network without hoarding; keep it small (5–50). Manual acks are mandatory, because auto-ack disables flow control entirely.
- **Reactive Streams makes the credit *the protocol*** via `request(n)` demand signaling, giving non-blocking backpressure across async boundaries — until you hit a non-demand source, where the bridge operator you pick is a real block-or-drop decision and the unbounded buffer default is the trap.
- **TCP flow control is credit-based backpressure on bytes**, via the receive window, and it will throttle you whether or not you configured anything — which is a free last line of defense and a silent thread-pool killer if you publish synchronously on request threads.
- **When you cannot slow the source, shed load deliberately** — drop, sample, degrade, or rate-limit at the door — because the alternative is not "serve everything," it is "collapse and drop everything." Shed early, shed the right work, shed visibly.
- **Backpressure must propagate unbroken to a source that can actually slow down.** A single unbounded buffer anywhere in the chain is a backpressure sink that absorbs the slow-down and turns it back into accumulating memory, defeating every bounded stage downstream of it.
- **Alert on queue depth *and its rate of change*.** Depth tells you where you are; the rate tells you when you hit the wall and how long you have to react.

## Further reading

- [Push vs pull, acknowledgements, and how consumers read](/blog/software-development/message-queue/push-vs-pull-acknowledgements-how-consumers-read) — where pull's free backpressure and the prefetch/ack contract are developed in full.
- [Poison messages and retry storms](/blog/software-development/message-queue/poison-messages-and-retry-storms-containment) — how retries amplify an imbalance and turn a slow consumer into a runaway backlog.
- [Consumer optimization and scaling](/blog/software-development/message-queue/consumer-optimization-and-scaling) — raising C (the consumer rate) so the imbalance never starts: batching, parallelism, and scale-out.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — why a pull-based, disk-backed log relocates the buffer bound from memory to retention.
- [RabbitMQ production architecture and scaling](/blog/software-development/system-design/rabbitmq-production-architecture-scaling) — prefetch, channel flow control, and memory alarms in a push broker at scale.
- *Reactive Streams Specification* (reactive-streams.org) — the `request(n)` demand protocol and the `Publisher`/`Subscriber`/`Subscription` contract that bakes backpressure into the type system.
- *RFC 9293 (TCP)* — the receive window and flow-control machinery that this whole pattern descends from.
- Marc Brooker, "Fairness in multi-tenant systems" and the AWS Builders' Library articles on *load shedding* and *timeouts, retries, and backoff* — production-grade guidance on shedding the right work early and visibly.
