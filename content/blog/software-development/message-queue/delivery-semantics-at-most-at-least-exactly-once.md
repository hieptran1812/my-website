---
title: "Delivery Semantics: At-Most-Once, At-Least-Once, and the Exactly-Once Myth"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Learn why one line of code — where you put the acknowledgement relative to processing — decides whether your queue loses messages or duplicates them, and why the exactly-once delivery you were sold is really effectively-once built from idempotency and transactions."
tags:
  [
    "message-queue",
    "delivery-semantics",
    "exactly-once",
    "idempotency",
    "kafka",
    "rabbitmq",
    "distributed-systems",
    "event-driven",
    "reliability",
    "two-generals",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/delivery-semantics-at-most-at-least-exactly-once-1.webp"
---

Somewhere in your system right now there is a payment that was charged twice, an email that was sent three times, or an order that silently vanished — and every one of those bugs traces back to a single design decision that almost nobody writes down. Not a framework choice, not a broker choice, not a config flag you can grep for. It is the relative position of two operations in your consumer loop: do you acknowledge the message *before* you process it, or *after*? That one ordering decision is the whole of delivery semantics. Put the ack first and a crash loses messages — at-most-once. Put it after and a crash redelivers messages — at-least-once. There is no third position that gives you both safety and uniqueness for free, no matter what the marketing slide says, because the network underneath you cannot tell the difference between a lost message and a lost acknowledgement.

This is the post in the series where we stop hand-waving about "reliable delivery" and pin down exactly what your queue does and does not promise. By the end you will be able to look at any consumer and tell me its delivery guarantee just by finding where the ack happens. You will know why "exactly-once delivery" is, at the level of the wire, a category error — and what the systems that advertise it actually do instead. You will be able to compose a producer's semantics with a consumer's semantics to get the real end-to-end guarantee, which is always the weaker of the two. And you will leave with one decision rule sharp enough to tattoo on the inside of your eyelids: **design for at-least-once plus idempotency, by default, always.** The figure below is the entire argument compressed into one table — three guarantees, defined by whether they lose, whether they duplicate, and when the ack fires.

![A matrix table showing the three delivery semantics as rows and three properties as columns, marking which can lose messages, which can duplicate, and when the acknowledgement fires](/imgs/blogs/delivery-semantics-at-most-at-least-exactly-once-1.webp)

We are going to build this from the bottom. First we define the three guarantees with surgical precision, because the loose definitions you have heard cause real bugs. Then we establish the central insight — ack timing *is* the guarantee — and walk each of the three semantics in turn with code and crash traces. We take a hard look at why exactly-once at the delivery layer collides with a sixty-year-old impossibility result, the two-generals problem, and what "effectively-once" means instead. We cover the producer side, where retries quietly duplicate publishes and the idempotent producer fixes it. We compose producer and consumer into an end-to-end guarantee. And we finish with a decision procedure for choosing a target guarantee per workload, two fully worked numerical examples, war stories from systems that learned this the hard way, and a crisp set of takeaways. Two sibling posts go deeper on the pieces this one only opens up: [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) for the consumer-side machinery, and [exactly-once in Kafka](/blog/software-development/message-queue/exactly-once-in-kafka-idempotent-producer-transactions) for how transactions actually wire it together.

## 1. The three guarantees, defined precisely

Delivery semantics answer one question: when a producer hands a message to the system, how many times does the consumer's processing run against it? There are exactly three answers anyone offers, and the sloppiness with which they are usually described is the source of an astonishing amount of production pain. Let us define them so tightly that there is no room left to be confused.

**At-most-once.** Every message is processed zero or one times. It is never processed twice. But it *may* be processed zero times — that is, it may be lost. The guarantee is purely an upper bound: no duplicates, but no promise the message survives. If something goes wrong at the wrong instant, the message is simply gone, and nobody will ever retry it. You accept loss in exchange for never duplicating.

**At-least-once.** Every message is processed one or more times. It is never lost — the system will keep trying until it sees a successful acknowledgement. But it *may* be processed more than once, because if the system is unsure whether processing succeeded, it retries, and "unsure" happens far more often than beginners expect. The guarantee is a lower bound: at least one successful processing, possibly several. You accept duplicates in exchange for never losing.

**Exactly-once.** Every message is processed exactly one time — never zero, never two. No loss, no duplication. This is what everybody wants and what a depressing amount of documentation claims to provide. We will spend a whole section dismantling the version of this claim that lives at the delivery layer, and rebuilding the version that is actually achievable, which has a different and more honest name: *effectively-once*.

The three are not three points on a smooth dial; they are defined by which of two bad outcomes you are willing to tolerate. At-most-once tolerates loss to forbid duplicates. At-least-once tolerates duplicates to forbid loss. Exactly-once claims to forbid both — and the entire interesting content of this post is the question of what that claim really costs and where it really holds. The figure above lays the three side by side: notice that at-most-once and at-least-once are exact mirror images, and exactly-once is the row everyone points at and nobody fully delivers at the layer they think they do.

It helps to anchor these to count outcomes. For any single message, the number of times your processing logic runs against it falls into one of three sets. At-most-once: the count is in `{0, 1}`. At-least-once: the count is in `{1, 2, 3, ...}` — one or more, unbounded in principle, though in practice bounded by your retry policy and your dead-letter threshold. Exactly-once: the count is `{1}` exactly. The reason exactly-once is special is that it is the *intersection* of the other two — it is the only outcome that is both at-most-once (never two) and at-least-once (never zero). To deliver it you must simultaneously satisfy both an upper bound and a lower bound on the processing count, using mechanisms that are in tension with each other: forbidding duplicates pushes you toward acking early (risking loss), and forbidding loss pushes you toward acking late (risking duplicates). Squaring that circle is the entire problem, and the resolution — when it exists — is never "deliver once" but "deliver many, apply once."

### Why the loose definitions cause bugs

The reason I am being this pedantic is that the loose version of these definitions — "at-least-once means you might get the message twice" — drops the part that actually bites: *why* you get it twice, and therefore *when*. Engineers who hold the loose definition treat duplicates as a rare anomaly to be patched after the fact. Engineers who hold the precise definition understand that duplicates are the *guaranteed consequence* of a lost acknowledgement, that lost acknowledgements happen on every healthy network constantly, and therefore that duplicates are a steady-state condition to be designed for up front, not an incident to be cleaned up after. The precise definition turns "handle duplicates eventually" into "duplicates are load-bearing; build for them on day one." That shift in mindset is worth more than any framework feature.

### A subtlety worth fixing now: delivery versus processing

People say "exactly-once delivery." The word *delivery* is the trap. The network can deliver a message to your process; what it cannot do is guarantee that your process *acts on it* exactly once, because the moment your processing produces a side effect — a database write, an HTTP call, an email — the question is no longer about the message arriving but about the effect happening. The honest framing is not about how many times a message is delivered but about how many times its *effect* lands. Once you reframe the goal from "deliver once" to "ensure the effect lands once," the whole problem becomes tractable, because effects can be made idempotent even when deliveries cannot be made unique. Hold onto that reframing; it is the hinge the rest of the post turns on.

## 2. Ack timing IS the guarantee (the central insight)

Here is the single most important sentence in this entire post: **the delivery guarantee is determined by when you send the acknowledgement relative to when you finish processing.** Everything else — broker features, client libraries, config flags — is machinery in service of, or in spite of, that one ordering decision. If you internalize nothing else, internalize this.

Walk through it concretely. A consumer pulls a message off the queue. It has two jobs to do: *process* the message (run the business logic, write to the database, charge the card) and *acknowledge* it (tell the broker "I'm done with this one, you can stop tracking it"). The acknowledgement is the broker's signal to drop the message from the queue or advance the committed offset. Until the broker sees that ack, it considers the message in-flight and will redeliver it if the consumer's lease expires or its session dies. So the consumer must choose an order:

- **Ack, then process.** The broker is told "done" before any work happens. If the consumer crashes between the ack and the end of processing, the broker has already forgotten the message — it will never redeliver. The work was never completed, and nobody will ever do it. **This is at-most-once.** You can lose.
- **Process, then ack.** The work completes, *then* the broker is told "done." If the consumer crashes after processing but before the ack lands, the broker never heard "done," its lease eventually expires, and it redelivers the message to some consumer — who processes it *again*. The work runs twice. **This is at-least-once.** You can duplicate.

That is the whole mechanism. The figure below shows the two orderings as a before-and-after: the same poll, the same crash, but the ack on the wrong side of processing produces loss while the ack on the right side produces a duplicate. There is no ordering of these two operations that avoids both failure modes, because the crash can land in the window between them regardless of which way you order them — and the window can never be closed to zero.

![A before-and-after diagram contrasting an ack-before-process path that loses a message on crash with an ack-after-process path that redelivers and duplicates on crash](/imgs/blogs/delivery-semantics-at-most-at-least-exactly-once-2.webp)

### Why the window can never be zero

A natural reaction is: "Fine, but if I make the ack and the processing *atomic* — both or neither — then there is no window, and I get exactly-once." This is exactly the right instinct, and it is the seed of everything that actually works. But notice what it requires: the ack and the side effect must commit together, atomically, as one indivisible operation. That is trivial when the side effect and the ack live in the same transactional resource — for example, both are writes to the same database. It is *impossible* when they live in different resources that cannot share a transaction — for example, the ack lives in Kafka and the side effect is an email sent through a third-party API that has no notion of your transaction. You cannot wrap "send email" and "commit Kafka offset" in one atomic step, because the email provider will not enroll in your two-phase commit. So the window between processing and acknowledging can be shrunk, hidden, or moved — but across a boundary where no shared transaction exists, it cannot be eliminated. That boundary is where exactly-once goes to die, and we will return to it with a vengeance in section 5.

### The same insight, stated as a rule

If you want to know a system's delivery guarantee, do not read its documentation. Find the line where it acknowledges, and find the lines where it does its work, and look at the order. Ack above the work: at-most-once. Ack below the work: at-least-once. Ack inside an atomic transaction with the work: effectively-once, but only within the boundary of that transaction. This rule has never once failed me in a code review, and it has caught more delivery bugs than any amount of staring at broker configs.

### The ack has different names in different systems

Part of what makes this hard to see is that the "acknowledgement" wears different clothes depending on the broker, and people fail to recognize it. It is worth cataloguing the disguises so you can spot the ack-versus-process ordering anywhere:

- **In RabbitMQ and classic AMQP brokers**, the ack is literal: `basic.ack` (or `basic.nack`/`basic.reject` for negative acks). With `autoAck=true` the broker acks on *delivery* — before your handler runs — which is at-most-once. With manual acks, you ack after processing, which is at-least-once. The ordering is right there in your handler.
- **In Kafka**, there is no per-message ack; the equivalent is the *offset commit*. Committing offset N+1 means "I have processed everything through offset N." Auto-commit on a timer commits independently of your processing — at-most-once if the timer fires mid-batch. Manual commit after processing is at-least-once. The "ack" is the commit, and its position relative to processing is the whole game.
- **In SQS**, the ack is `DeleteMessage`. A message you receive is *invisible* for the visibility timeout but not deleted; you must call `DeleteMessage` to acknowledge it. Delete before processing is at-most-once; delete after is at-least-once. If you never delete, the visibility timeout expires and the message reappears — automatic redelivery.
- **In Google Pub/Sub**, the ack is `acknowledge()` against an ack ID, with the same timing logic.

Different verbs, identical physics. Whenever you adopt a new broker, the first question to answer is: what is the ack here, and where is it relative to my processing? Everything else follows.

## 3. At-most-once: ack first, accept loss

At-most-once is the guarantee you get when you optimize for throughput and latency and decide that losing the occasional message is acceptable. It is not a mistake — it is a legitimate, sometimes optimal choice — but it is shockingly often chosen *by accident*, by people who did not realize that "auto-commit" or "fire and forget" meant "I am okay with loss."

The defining move is acknowledging before processing. In Kafka this is what you get when you enable auto-commit with a short interval and the offset advances on a timer regardless of whether your processing of those records finished. In a classic message broker it is what you get with `autoAck=true`, where the broker considers the message delivered the instant it is handed to the socket, before your handler has even run. In a UDP-style fire-and-forget producer, it is what you get because there is no ack at all.

Here is an at-most-once consumer in Python against a generic broker, written to make the failure explicit:

```python
# AT-MOST-ONCE: acknowledge before processing.
# A crash inside process() loses the message forever.
def consume_at_most_once(channel):
    while True:
        msg = channel.poll(timeout=1.0)
        if msg is None:
            continue
        # Ack FIRST. The broker now forgets this message.
        channel.ack(msg.delivery_tag)
        # If the process crashes on the next line, the work
        # is never done and the message is never redelivered.
        process(msg)  # write to DB, call a service, etc.
```

The danger is the line ordering. Between `channel.ack(...)` and the completion of `process(msg)`, the message exists nowhere that can recover it. The broker has dropped it. If the process is killed — OOM, deploy, kernel panic, a `kill -9` from a frazzled on-call engineer — the message is gone with no trace and no retry. There is no error, no dead-letter, no alert. It simply did not happen.

### When at-most-once is the right call

At-most-once is correct when the cost of processing a message twice exceeds the cost of occasionally not processing it at all, *and* when the data stream is naturally lossy or self-correcting. The canonical examples:

- **High-frequency metrics and telemetry.** You are shipping a million CPU-utilization samples per minute. If you drop a handful during a broker hiccup, your dashboards do not care — the next sample arrives in a second and the trend is unbroken. Duplicating a metric, on the other hand, can skew a counter or a rate. Here, loss is cheap and duplication is expensive, so at-most-once is the *better* guarantee.
- **Live sensor or video frames.** A dropped frame is invisible; a duplicated or out-of-order frame causes a visible glitch. Drop and move on.
- **Cache-warming and best-effort notifications** where a missed event is harmless because something else will refresh the state soon.

The pattern across all of these: the value of an individual message decays to near zero almost immediately, and the stream as a whole is what matters. When that is your data, paying the cost of at-least-once — retries, dedup tables, idempotency keys — is pure overhead for a guarantee you do not need.

There is also a *latency* argument for at-most-once that is sometimes decisive. When you ack first, the broker can release the message and reclaim its tracking state immediately, and the producer with `acks=0` does not wait for a round trip at all — it writes to the socket and moves on. For a telemetry agent shipping a million samples a minute from an edge device on a constrained link, the difference between waiting for an ack and not waiting is the difference between keeping up and falling behind. At-most-once is the lowest-latency, highest-throughput guarantee precisely *because* it does the least: no waiting for confirmation, no retry bookkeeping, no dedup state. You are buying speed with the currency of occasional loss.

### When it bites you by accident

The tragedy of at-most-once is how often teams ship it without meaning to. The Kafka consumer default historically *was* auto-commit enabled. A developer writes a poll loop, processing happens, and meanwhile a background timer commits offsets every five seconds. Then a record takes longer than expected, a rebalance happens, the offset was already committed past records that were never fully processed, and those records are silently skipped on the next assignment. No error. The team discovers it weeks later when a customer reports a missing order. The lesson: **if you did not explicitly choose where your offset commits relative to your processing, you do not know your delivery guarantee, and "I don't know" almost always resolves to "I'm losing messages."**

## 4. At-least-once: process first, accept duplicates

At-least-once is the workhorse of reliable messaging, the default that almost every production system should start from, and the guarantee that every serious broker can provide cheaply. The move is the mirror of at-most-once: process *first*, acknowledge *after*. You only tell the broker "done" once the work has actually, verifiably completed. If anything goes wrong before the ack lands, the broker redelivers, and the work runs again.

The figure below traces the canonical at-least-once consumer loop and shows the one branch that defines the guarantee: when the ack is lost or the consumer crashes after processing, the message is requeued and redelivered, so it is never lost — at the cost of running again.

![A pipeline diagram of an at-least-once consumer loop where processing precedes the ack and a lost ack or crash sends the message back through requeue and redelivery](/imgs/blogs/delivery-semantics-at-most-at-least-exactly-once-4.webp)

Here is the at-least-once version of the same consumer, with the two lines swapped:

```python
# AT-LEAST-ONCE: process before acknowledging.
# A crash before ack() redelivers the message (a duplicate).
def consume_at_least_once(channel):
    while True:
        msg = channel.poll(timeout=1.0)
        if msg is None:
            continue
        process(msg)  # do the work FIRST
        # If we crash here, the broker never saw the ack and
        # will redeliver this message. We never lose it.
        channel.ack(msg.delivery_tag)  # then acknowledge
```

The same in the Kafka idiom — disable auto-commit and commit the offset only after the batch is fully processed:

```python
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    "orders",
    bootstrap_servers="broker:9092",
    enable_auto_commit=False,        # we control commit timing
    group_id="order-processor",
    auto_offset_reset="earliest",
)

for batch in consumer.poll(timeout_ms=500).values():
    for record in batch:
        process(record)              # work first
    # Commit only after the whole batch processed.
    # Crash before this line -> redelivery of the batch.
    consumer.commit()                # ack after
```

The crucial property is that **at-least-once never loses a message that was successfully published**, provided the broker itself is durable (replicated, fsynced — see [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) for the durability mechanics). The cost is duplicates, and the size of that cost is the whole reason the next several sections exist.

The SQS idiom makes the same shape explicit because the acknowledgement is a separate API call you must remember to make:

```python
import boto3

sqs = boto3.client("sqs")
QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/123/orders"

while True:
    resp = sqs.receive_message(
        QueueUrl=QUEUE_URL,
        MaxNumberOfMessages=10,
        WaitTimeSeconds=20,          # long polling
        VisibilityTimeout=60,        # 60s to process before redelivery
    )
    for m in resp.get("Messages", []):
        process(m)                   # work FIRST
        # DeleteMessage is the ack. If we crash before this,
        # the message becomes visible again after 60s and is
        # redelivered. At-least-once.
        sqs.delete_message(
            QueueUrl=QUEUE_URL,
            ReceiptHandle=m["ReceiptHandle"],
        )
```

Note the `VisibilityTimeout=60`: it is the SQS analogue of a lease. While a message is being processed it is invisible to other consumers for 60 seconds. If your processing reliably takes 50 seconds you are fine; if it sometimes takes 70, the message reappears and a second consumer picks it up *while you are still working* — a duplicate with no crash involved. Tuning that timeout is the same tradeoff as tuning a Kafka session timeout, and it has the same trap: set it too low and slow processing duplicates; set it too high and a genuinely dead consumer holds messages hostage for minutes before anyone else can retry them.

### Why duplicates happen more than you think

Newcomers assume duplicates are rare — they picture a crash, which feels like an exotic event. In reality, redelivery is triggered by a whole family of mundane occurrences, most of which involve no crash at all:

- **A lost ack.** Processing succeeded, the consumer sent the ack, but the ack packet was dropped or the connection reset before the broker recorded it. The broker, never having heard "done," redelivers. The work was complete; the network just lost the receipt.
- **A slow consumer exceeding its lease.** The broker grants you a visibility timeout or a session deadline. If your processing takes longer than that — a slow downstream call, a GC pause, a noisy neighbor — the broker assumes you died and redelivers to someone else, even though you are still working. Now two consumers process the same message.
- **A consumer-group rebalance.** In Kafka, when consumers join or leave, partitions are reassigned. Any records processed but not yet committed by the old owner are re-fetched by the new owner. Every rebalance is a duplicate-generating event for in-flight work.
- **A genuine crash** between processing and commit, the case everyone pictures.

The first two are the killers because they require nothing to actually go wrong with your code. A 200-millisecond network blip or a one-second GC pause is enough. In a healthy system processing millions of messages a day, you will see duplicates *every single day*. They are not an edge case; they are a constant background rate. Designing as though duplicates are rare is the bug.

### The poison-message corollary

At-least-once has a failure mode that at-most-once does not: the **poison message**. Suppose a message cannot be processed — it is malformed, it references a row that was deleted, it trips a bug in your handler. The handler throws, the consumer never acks, the broker redelivers, the handler throws again, forever. Because at-least-once *will not give up*, a single bad message can spin a consumer in an infinite redelivery loop, blocking the partition behind it and burning CPU. The cure is a **retry limit plus a dead-letter queue (DLQ)**: after N failed deliveries, the broker (or your consumer) routes the message to a side queue for human inspection instead of redelivering it forever. The DLQ is the safety valve that makes at-least-once operable — without it, "never give up" becomes "wedge the pipeline on the first bad message." This is the price of the no-loss guarantee: you must decide, explicitly, when to stop retrying and quarantine, because the system itself will retry until the heat death of the universe otherwise.

#### Worked example: counting duplicates from one crash

Let us make the duplicate count concrete, because "you might get duplicates" is uselessly vague. Suppose a consumer reads from a partition, processing records one at a time, and committing the offset after every 100 records as a batch — a common throughput optimization. It has processed records at offsets 0 through 149: the first 100 were committed (offset committed = 100), and records 100 through 149 have been processed but *not yet* committed. Now the consumer crashes.

On restart, the consumer group reassigns the partition and the new owner resumes from the last committed offset, which is 100. It re-fetches and re-processes records 100 through 149 — fifty records that were *already* fully processed, side effects and all. So this single crash produces **50 duplicate processings**. If each of those records was an order, fifty orders just got processed twice. If processing meant charging a card, fifty customers were charged twice.

Crank the batch size up to 1,000 for throughput, and the same crash now redelivers up to 999 already-processed records. The duplicate exposure of a crash is bounded by your commit interval: **the larger your commit batch, the more duplicates a single crash produces.** This is the fundamental tension of at-least-once tuning — commit often and you pay throughput overhead; commit rarely and a crash duplicates a larger window. There is no setting that makes the duplicate count zero, which is precisely why the answer is never "tune the batch size to avoid duplicates" and always "make processing idempotent so duplicates are harmless." We will see exactly how in sections 6 and 7.

The figure below puts that crash on a timeline so you can see the window where the duplicate is born: processing completes, the side effect has already happened, and *then* the crash strikes before the ack — so the broker, never having heard "done," requeues and the effect runs a second time.

![A timeline showing a consumer crash occurring after processing completes but before the ack is sent, leading to lease expiry, requeue, redelivery and the effect running twice](/imgs/blogs/delivery-semantics-at-most-at-least-exactly-once-3.webp)

## 5. Why exactly-once is (mostly) a myth — the two-generals problem

Now we confront the claim head-on. "Exactly-once delivery" — meaning the message is delivered to your processing logic precisely one time, never zero, never two — is, at the level of two machines talking over an unreliable network, *impossible*. Not hard. Not expensive. Impossible, in the same way that sorting in less than n-log-n comparisons is impossible. The impossibility has a name and a sixty-year pedigree: the **two-generals problem**.

The setup: two generals must coordinate an attack by sending messengers across a valley patrolled by the enemy. Any messenger may be captured, so any message may be lost. General A sends "attack at dawn." Did it arrive? A cannot know unless B sends back an acknowledgement. B sends "acknowledged." Did *that* arrive? B cannot know unless A acknowledges the acknowledgement. And A cannot be sure *that* arrived without an ack of the ack of the ack. The recursion never terminates. There is no finite exchange of messages over a lossy channel that lets both sides become *certain* they agree. It is provably unsolvable.

Map that onto your queue and the consequence is stark. The consumer processes a message and sends an ack. The broker either receives it or does not. **The sender of an ack can never know whether its ack arrived.** If the ack is lost, the broker redelivers — a duplicate. If, to avoid duplicates, the broker assumed acks always arrive and dropped the message on send, then a genuinely lost ack would cause a *loss* — and we are back to at-most-once. The network's inability to confirm delivery of the ack is exactly the window from section 2, and the two-generals result says that window is *fundamental*, not an implementation gap somebody will close in the next release.

So there are only two stable points: assume the ack might be lost and redeliver (at-least-once, duplicates), or assume it arrived and forget (at-most-once, loss). Exactly-once delivery is the unstable point in between, and the network will not let you stand on it.

### Why "just retry until you get an ack of the ack" does not save you

The tempting escape is to add more acknowledgements: the consumer acks, the broker acks the ack, the consumer acks the broker's ack-of-ack. But each additional message has the same lossy channel beneath it and the same uncertainty at its far end. You can add a thousand rounds of acknowledgement; the thousand-and-first message can still be the one that is lost, and whoever sent it still cannot know. The recursion does not converge to certainty — it just moves the uncertainty up one level each time. This is not a quantitative problem you can drive arbitrarily small with enough retries; it is a *qualitative* impossibility. You can make the *probability* of an undetected discrepancy tiny, but you can never make it zero, and "tiny but nonzero" over millions of messages a day means it happens. The two-generals result is the formal statement that no protocol over an unreliable channel achieves *common knowledge* of delivery in finite time. Exactly-once delivery would require exactly that common knowledge, so it cannot exist.

### Crash failures make it strictly worse

The two-generals problem assumes only message loss. Real systems add a second source of uncertainty: processes *crash*, and a crashed process is indistinguishable from a slow one or an unreachable one. When the broker stops hearing from a consumer, it cannot tell whether the consumer (a) crashed before processing, (b) crashed after processing but before acking, (c) is alive but slow, or (d) is alive and the network between them is partitioned. Cases (a) and (b) demand *opposite* responses — redeliver in (a) to avoid loss, do *not* redeliver in (b) to avoid a duplicate — and the broker cannot distinguish them. Forced to choose one policy, it picks redeliver (favoring no-loss), and that choice *is* the decision to be at-least-once. The impossibility of distinguishing "crashed before" from "crashed after" is the same window as the lost ack, viewed from the broker's side. Every road leads back to the same fork: tolerate loss or tolerate duplicates. There is no third road.

### So what do the "exactly-once" systems actually sell?

Kafka ships a feature literally called exactly-once semantics. AWS offers exactly-once SQS FIFO queues. Are they lying? No — they are doing something real and valuable, but it is *not* exactly-once *delivery*. It is **effectively-once processing within a closed boundary**, achieved by combining at-least-once delivery (which still produces duplicate *deliveries*) with deduplication and atomic commits (which make those duplicate deliveries produce no duplicate *effects*). The message may still be delivered twice. What is guaranteed is that the *effect* — the offset commit, the output write — lands once, because the duplicate is detected and discarded *inside the system*.

The figure below shows the taxonomy: the three guarantees and the mechanism each one is built from. Notice that exactly-once is not a peer of the other two; it is a *composite*, assembled from at-least-once delivery plus an idempotency or transaction mechanism layered on top. It is a property of your processing pipeline, not a property of the wire.

![A tree diagram classifying the three delivery guarantees and the concrete mechanism each requires, with exactly-once shown as a composite of idempotency and transactions over at-least-once delivery](/imgs/blogs/delivery-semantics-at-most-at-least-exactly-once-6.webp)

### The boundary is everything

The reason Kafka's exactly-once works *inside* Kafka is that the entire read-process-write cycle lives in *one transactional resource*: the consumer reads from a Kafka topic, processes, and writes results back to a Kafka topic, and commits the source offset — all in a single Kafka transaction. The offset commit and the output write are the *same* atomic operation, so the window from section 2 is genuinely closed. Both happen or neither does. That is real, and it is exactly the atomic-commit idea from section 2's "why the window can never be zero" applied where it *can* hold.

The myth is the belief that this property survives crossing the boundary. The instant your processing produces a side effect outside the transactional resource — an email through SendGrid, a charge through Stripe, a row in a database Kafka cannot enroll in its transaction — the atomicity breaks. The email API cannot participate in a Kafka transaction. So the offset commit and the email send are once again two separate operations with a window between them, and a crash in that window either re-sends the email (duplicate) or loses the offset (re-process). Exactly-once is a boundary-local property. Outside its boundary, you are back to at-least-once, and you had better have made the side effect idempotent. We will dedicate section 6 to exactly that.

## 6. Effectively-once: idempotency and transactions

If exactly-once delivery is impossible and exactly-once processing only holds inside a transactional boundary, what is the achievable goal for the general case — a consumer whose work touches an external system? The answer is **effectively-once**: the message may be *delivered* any number of times, but its *effect* is applied exactly once, because the effect is engineered to be idempotent or transactional. The deliveries are still at-least-once. The effects are once. That gap — between deliveries and effects — is where all the real engineering lives.

An operation is **idempotent** when applying it twice has the same result as applying it once. `SET balance = 100` is idempotent: run it ten times, the balance is 100. `balance = balance + 50` is *not* idempotent: run it ten times and you have added 500. The entire game of effectively-once is converting your non-idempotent side effects into idempotent ones, usually by attaching a **deduplication key** — a unique identifier carried by the message — and recording which keys you have already applied, so a redelivery of an already-applied key becomes a no-op.

The single most important property of a good dedup key is that it is **stable across retries** — the *same* logical operation must always carry the *same* key, no matter how many times it is published or redelivered. This rules out generating the key at send time with something like a fresh UUID per `send()` call, because a producer retry would generate a *different* UUID and defeat the whole mechanism. The key must be derived from the *business identity* of the operation: the order ID, the payment intent ID, a hash of the request payload, an idempotency token the client minted once when the user first clicked the button. Get this wrong and your dedup table fills with distinct keys for what is logically the same operation, and the duplicates sail straight through. Key design is subtle enough that it gets its own [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) post; here it is enough to know that the key is the linchpin and it must be stable.

The figure below contrasts the two worlds for a payment: the naive consumer charges the card again on redelivery, while the effectively-once consumer checks the dedup key, sees it has already applied that key, and skips — turning a duplicate delivery into a harmless no-op and a single charge.

![A before-and-after diagram contrasting a naive consumer that double-charges a card on redelivery with an effectively-once consumer that checks a dedup key, skips the seen key, and charges once](/imgs/blogs/delivery-semantics-at-most-at-least-exactly-once-7.webp)

### Two mechanisms: dedup keys and atomic transactions

There are two ways to get effectively-once, and good systems use both.

**Deduplication keys (idempotent consumer).** The producer stamps each message with a stable, unique key — an order ID, a payment ID, a UUID generated once at the source. The consumer keeps a record of applied keys (a database table, a Redis set with a TTL, a unique constraint). Before applying the effect, it checks: have I seen this key? If yes, skip. If no, apply and record the key. The check-and-record must itself be atomic with the effect, or you reintroduce a window — which leads to the second mechanism.

```python
# Effectively-once via a dedup key and an atomic upsert.
# The unique constraint on dedup_key makes the second
# delivery a no-op at the database level.
def process_payment(conn, msg):
    dedup_key = msg.headers["idempotency-key"]   # stable per payment
    with conn.transaction():
        # INSERT ... ON CONFLICT DO NOTHING: if the key already
        # exists, this charges nothing and returns 0 rows.
        rows = conn.execute(
            """
            INSERT INTO applied_payments (dedup_key, amount, charged_at)
            VALUES (%s, %s, now())
            ON CONFLICT (dedup_key) DO NOTHING
            RETURNING id
            """,
            (dedup_key, msg.amount),
        )
        if not rows:
            return  # duplicate delivery; effect already applied
        # First time we have seen this key: do the real work
        # in the SAME transaction as recording the key.
        debit_balance(conn, msg.account_id, msg.amount)
    # ack only after the transaction commits (at-least-once on the queue)
```

**Atomic transactions (transactional outbox / read-process-write).** When the side effect and the offset commit live in resources that *can* share a transaction, you wrap them together so they commit atomically — closing the window entirely. The classic pattern is the [transactional outbox and change data capture](/blog/software-development/database/change-data-capture-and-the-outbox-pattern), where you write the business change and an outbox event in one database transaction, and a separate process relays the outbox to the broker. Kafka's exactly-once is the read-process-write version: read offset, process, write output, and commit the consumed offset *as part of producing the output*, all in one Kafka transaction.

The deep reason the dedup-key approach works without a distributed transaction is that it folds the "have I done this?" check and the effect itself into *one* atomic operation against *one* resource — your database. The `INSERT ... ON CONFLICT DO NOTHING` and the `debit_balance` in the snippet above share a single database transaction, so either both happen or neither does. There is no window between recording the key and applying the effect, because they are the same commit. This is the crucial discipline that beginners miss: it is not enough to *check* a dedup table and then *apply* the effect, because a crash between the check and the apply, or a race between two concurrent consumers, slips a duplicate through the gap. The check and the apply must be **atomic** — which usually means leaning on a database unique constraint or a single conditional write, not a read followed by a write. If you find yourself writing "if not seen, then do," stop and make it one atomic statement instead.

The figure below stacks the layers an effectively-once guarantee must hold across. Every layer from the idempotent producer down through the broker's dedup and atomic commit to the idempotent consumer must cooperate; a gap in any one layer and the guarantee leaks. There is no single switch — it is a property of the whole stack, which is why it is so often gotten wrong.

![A stack diagram showing the layers effectively-once must hold across, from idempotent producer through broker dedup and atomic transaction down to the idempotent consumer](/imgs/blogs/delivery-semantics-at-most-at-least-exactly-once-5.webp)

#### Worked example: a double charge, and how a dedup key fixes it

Let us run the canonical disaster and then fix it. A payments consumer reads charge requests off a queue at at-least-once. Each message says "charge account 1234 the amount of \$50." The naive consumer does exactly that: it calls Stripe to charge \$50, then acks.

Now a network blip delays the ack. The broker's visibility timeout is 30 seconds; the ack is delayed 31 seconds because of a TCP retransmit storm. At second 30, the broker concludes the consumer died and redelivers the message to a second consumer. Both consumers call Stripe. **The customer is charged \$50 twice — a total of \$100 for a single intended \$50 charge.** No code threw an error. Both consumers did exactly what they were told. The duplicate came entirely from the redelivery, which came entirely from the lost-ack window the two-generals problem guarantees exists.

Now the fix. The producer stamps each charge message with an idempotency key generated once when the charge was first requested — say `idem-9f3a`. The consumer passes that key to Stripe as Stripe's own `Idempotency-Key` header (a real Stripe feature built for exactly this), *and* records it locally in an `applied_payments` table with a unique constraint. On the first delivery, the insert succeeds and Stripe charges \$50. On the redelivery, *either* the local unique-constraint insert fails (so the consumer skips before ever calling Stripe) *or*, if both consumers raced past the local check simultaneously, Stripe itself sees the repeated `Idempotency-Key` and returns the *original* charge result without charging again. The customer is charged **exactly \$50, once**, even though the message was delivered and processed twice. The delivery was at-least-once; the *effect* was effectively-once. That is the whole pattern, and it is the single most important defensive technique in event-driven systems. The deep mechanics — key design, storage, TTLs, race conditions — are the subject of the [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) post.

## 7. Producer-side duplicates and the idempotent producer

So far we have looked at the *consumer* side, where redelivery creates duplicates. But duplicates are born on the *producer* side too, and the cause is the same impossibility — the producer cannot know whether its publish succeeded.

Picture a producer sending a message to the broker. The broker receives it, writes it to the log, and sends back an ack. But the ack is lost in transit. The producer, having waited for an ack that never came, does the only safe thing: it *retries*, re-sending the message. Now the broker has the message **twice** in its log. The producer was trying to guarantee the message was not *lost*, and in doing so it duplicated it. This is the exact same lost-ack window from the consumer side, just one hop upstream. Producer retries are a duplicate-generating machine, and you cannot turn them off without accepting publish loss.

```python
# Naive producer: retries on a lost ack cause duplicate publishes.
def publish_naive(producer, topic, payload):
    for attempt in range(3):
        try:
            ack = producer.send(topic, payload).get(timeout=5)
            return ack
        except TimeoutError:
            # The broker may have ALREADY written the message and
            # only the ack was lost. Retrying duplicates it.
            continue
    raise PublishFailed()
```

### The idempotent producer

The fix on the broker side is the **idempotent producer**: the producer is assigned a unique producer ID (a PID), and every message it sends carries a monotonically increasing sequence number per partition. The broker tracks the highest sequence number it has committed for each producer-partition pair. When a retry arrives with a sequence number the broker has *already* committed, the broker recognizes the duplicate and *discards it*, returning a success ack as if it had written it — because, from the producer's point of view, it did. The producer retries freely; the broker dedupes by sequence number. Publishes become effectively-once *into the broker's log*.

In Kafka this is a single config flag, and as of recent versions it is the default:

```properties
# Kafka producer: idempotent publishing.
# enable.idempotence=true gives the producer a PID and
# per-partition sequence numbers so the broker dedupes retries.
enable.idempotence=true
acks=all
retries=2147483647
max.in.flight.requests.per.connection=5
```

Turning on `enable.idempotence` automatically sets `acks=all` (the message must be acknowledged by all in-sync replicas before the producer considers it durable — see [distributed replication](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless)), bumps retries effectively to infinite, and caps in-flight requests at five so the broker can still detect and reorder retried batches by sequence number. The result: even though the producer retries aggressively to avoid loss, the broker guarantees each message appears in the log exactly once. This is genuine, and it is the *one* place "exactly-once" earns the name without an asterisk — because the producer and the broker log share the sequence-number dedup mechanism, closing the window. But note the scope carefully: it dedupes a producer's *retries of the same send*. It does **not** dedupe two *different* `send()` calls of the same logical message from application code — if your application calls `send()` twice for the same order because *it* crashed and retried, the broker sees two different sequence numbers and writes both. Application-level duplicates still need application-level dedup keys.

This scope boundary is exactly where most teams get burned, so it is worth drawing the line sharply. The idempotent producer protects the segment between a *single* producer client instance and the broker, across that client's own internal retries, for the lifetime of its producer ID. Three things fall *outside* that protection, and each is a real duplicate source the flag does nothing about:

- **A new producer session.** If the producer process restarts, it gets a *new* PID, and the broker's sequence-number tracking for the old PID does not carry over. A message the old session was retrying when it died, re-sent by the new session, is a duplicate the idempotent producer cannot catch. (Transactional producers with a stable `transactional.id` close part of this gap by fencing the old session, which is one reason Kafka transactions are more than just "idempotence plus output dedup.")
- **Application-level resends.** Your service receives an HTTP request, publishes an event, but crashes before responding. The client retries the HTTP request, your service publishes the event *again* from a fresh code path — two distinct `send()` calls, two sequence numbers, two log entries. The broker faithfully stores both.
- **At-most-once-per-key needs.** The idempotent producer guarantees *per-partition* ordering and dedup, but says nothing about logical uniqueness across partitions or across producers.

So the honest summary is: `enable.idempotence=true` is necessary and cheap and you should turn it on everywhere, but it is a *narrow* guarantee — no duplicate writes from one client's retries — not the broad "my events are unique" property its name suggests. Logical uniqueness is still your job, enforced with a dedup key at the consumer, which brings us full circle to section 6.

The figure below shows where this fits in the larger picture: producer semantics on one axis, consumer semantics on the other, composing into an end-to-end guarantee — which is the subject of the next section.

![A grid diagram showing how producer semantics and consumer semantics compose into an end-to-end delivery guarantee, with the weaker leg determining the result](/imgs/blogs/delivery-semantics-at-most-at-least-exactly-once-8.webp)

## 8. End-to-end: composing producer × consumer semantics

A message's journey has two legs: producer-to-broker, and broker-to-consumer. Each leg has its own delivery semantics, and the **end-to-end guarantee is the weaker of the two.** A chain is as strong as its weakest link, and delivery semantics chain exactly this way. You cannot bolt a perfectly idempotent consumer onto a producer that drops messages and get a system that does not drop messages; the loss already happened upstream before the consumer ever saw anything.

Let us reason through the combinations, which is what the grid figure above lays out:

- **At-most-once producer × anything.** If the producer can lose a publish (fire-and-forget, no retries, `acks=0`), then some messages never reach the broker at all. No consumer, however careful, can process a message it never received. End-to-end: **at-most-once.** The lossy leg dominates.
- **At-least-once producer × at-most-once consumer.** The producer guarantees the message reaches the broker (with possible duplicates), but the consumer acks before processing and loses on crash. End-to-end: **at-most-once** — the consumer's loss dominates, *and* you carry the producer's duplicates too, getting the worst of both unless you dedupe.
- **At-least-once producer × at-least-once consumer.** Neither leg loses; both may duplicate. End-to-end: **at-least-once.** This is the realistic baseline for most production systems, and it is exactly the case where you must layer idempotency on top to reach effectively-once.
- **Idempotent producer × idempotent consumer (with dedup at both hops).** The producer dedupes its retries into the broker log; the consumer dedupes redeliveries by key. No loss, no duplicate *effects*. End-to-end: **effectively-once.** This is the best generally achievable result, and it requires deliberate work at *both* legs.

The non-obvious lesson is the asymmetry of effort. Getting to effectively-once requires *every* leg to be at-least-once-or-better *and* idempotent. Getting to at-most-once requires only *one* leg to be lossy. Reliability is a property you can only lose, never partially gain — one weak link and the whole chain degrades to the weakest semantics in it. That is why "we use Kafka, so we have exactly-once" is such a dangerous sentence: Kafka's exactly-once covers the read-process-write loop inside Kafka, but the moment your producer is a non-Kafka service publishing over a flaky network without idempotence, or your consumer writes to a database without a dedup key, the end-to-end guarantee silently collapses to whatever your weakest leg provides.

### Chains of consumers compound the legs

It gets worse when a message flows through multiple stages — consumer A reads a topic, processes, and *publishes* to a second topic that consumer B reads. Now you have *four* legs: A's input, A's output (A is a producer here), B's input, B's output. The end-to-end guarantee is the weakest of all four, and each producer-consumer boundary introduces a fresh lost-ack window. A five-stage pipeline has ten legs, and if any one of them is at-most-once the whole pipeline can lose data; if any one is at-least-once without idempotency the whole pipeline can corrupt with duplicates. This is why streaming frameworks that promise exactly-once go to such lengths to keep the *entire* multi-stage flow inside one transactional fabric (Kafka topics throughout, committed in coordinated transactions): the instant a stage steps outside that fabric — to call a REST API, to write to a non-Kafka store — that stage's output leg loses the transactional guarantee and the chain degrades. The practical upshot: keep your effectively-once-critical pipeline inside a single transactional resource if you possibly can, and where it must step outside, do so at exactly one well-marked stage that you make idempotent by hand. Draw the boundary deliberately; do not let it be wherever the code happened to land.

### A quick way to audit a real system

To find the true end-to-end guarantee of a system you did not write, trace one message from origin to final effect and mark each hop: at each producer, is there idempotence or a stable dedup key? At each consumer, where is the ack relative to processing, and is the effect idempotent? Take the weakest mark you found. That is your real guarantee, regardless of what any config or README claims. I have done this audit on systems that were *certain* they had exactly-once and found an at-most-once auto-commit two hops in, quietly dropping data nobody had noticed because the loss was rare and silent. The audit takes an hour and is worth a quarter of incident response.

### The exactly-once illusion, assembled

Now we can show the full machine that the "exactly-once" systems actually run, and exactly where it breaks. The figure below is the assembled illusion: an idempotent producer dedupes retries into the broker; an atomic read-process-write transaction commits the offset and the output together inside the broker boundary; an idempotent consumer dedupes by key. Inside that boundary, the effect is genuinely once. But follow the arrow to the rightmost node — the external effect, the email or the card charge that leaves the boundary. The moment the effect crosses out of the transactional resource, the atomicity that held everything together is gone, and you are back to at-least-once with whatever idempotency you remembered to build into that external call.

![A pipeline diagram of the exactly-once illusion showing an idempotent producer, an atomic commit and an idempotent consumer holding inside the broker boundary, breaking the moment an external effect like an email or charge leaves that boundary](/imgs/blogs/delivery-semantics-at-most-at-least-exactly-once-9.webp)

This figure is the single most important mental model to leave with. "Exactly-once" is a property that holds *inside a drawn boundary* and shatters the instant an effect crosses it. The engineering question is never "do I have exactly-once?" — you do not, in general. The question is "where is my boundary, and what makes the effects *outside* it idempotent?" If you can answer that, you have a correct system. If you cannot, you have a system that double-charges customers and you do not yet know it.

## 9. Choosing a target guarantee for real workloads

Enough theory — here is the decision procedure. For any given message flow, you choose a target guarantee by answering two questions about the *consumer's effect*: what does loss cost, and what does duplication cost?

**Step 1 — What does losing this message cost?** If the answer is "nothing, the next message corrects it" (metrics, telemetry, cache warming, live frames), you are allowed to consider at-most-once. If the answer is anything with business consequence (an order, a payment, an email the customer expects, a state transition), loss is unacceptable and at-most-once is off the table. You need at-least-once or better.

**Step 2 — What does processing this message twice cost?** If duplicate processing is naturally harmless — because the operation is already idempotent (a `SET`, an upsert, a state machine that ignores repeated transitions) — then plain at-least-once *is* effectively-once for free, and you are done; you need no dedup machinery at all. If duplicate processing is harmful (a second charge, a second email, a double-incremented counter), you must add idempotency: a dedup key plus an atomic check-and-apply, getting you to effectively-once.

The most underrated move in this whole space is **designing the effect to be naturally idempotent so that step 2 answers itself.** A surprising number of operations can be rewritten from non-idempotent to idempotent with no dedup table at all:

- Replace `balance = balance + 50` (not idempotent) with `INSERT INTO ledger (txn_id, delta) VALUES (...) ON CONFLICT DO NOTHING` and compute the balance as a sum (idempotent, because the ledger row is keyed by `txn_id`).
- Replace "send email" (not idempotent) with "set `confirmation_sent_at` if null, and let a separate idempotent mailer act on that flag" — moving the non-idempotent boundary to a single place you control.
- Replace "create resource" with "create resource with this client-supplied ID" so a retried create against an existing ID is a no-op or returns the existing resource (this is why so many REST APIs accept a client-generated ID on `PUT`).
- Replace "increment a counter" with "record an event and count distinct events," trading a non-idempotent write for an idempotent insert plus a query.

Every one of these turns the message-twice problem into a non-problem *structurally*, so you never need to reason about delivery semantics for that effect at all. The discipline of asking "can I make this effect idempotent by construction?" before reaching for a dedup table is the mark of someone who has operated these systems. The dedup table is the fallback for effects you genuinely cannot make idempotent — most often calls to external systems whose API is not idempotent and that you cannot change. For those, you wrap with a dedup key as in section 6; for everything else, you redesign the effect and the problem evaporates.

That is the entire decision tree, and it collapses to one default for the overwhelming majority of real workloads: **at-least-once delivery plus idempotent processing.** Start there. It never loses, and the idempotency makes duplicates harmless, so you get the *behavior* of exactly-once without the impossible promise. Reach for at-most-once only when you have explicitly decided loss is cheaper than duplication and the data stream is self-correcting. Reach for in-system transactional exactly-once (Kafka EOS) only when your entire effect stays inside one transactional resource and you have measured that idempotent at-least-once is not enough.

| Workload | Loss cost | Dup cost | Target guarantee | Mechanism |
| --- | --- | --- | --- | --- |
| Metrics / telemetry | Negligible | High (skews counters) | At-most-once | Fire-and-forget, auto-commit |
| Order processing | Catastrophic | High (double order) | Effectively-once | At-least-once + dedup key |
| Payment / charge | Catastrophic | Catastrophic | Effectively-once | At-least-once + idempotency key |
| Cache invalidation | Low (self-heals) | Negligible (idempotent) | At-least-once | At-least-once, no dedup needed |
| Kafka stream join (in-system) | Catastrophic | High | Exactly-once (EOS) | Read-process-write transaction |
| Email confirmation | High (customer waits) | High (spammy) | Effectively-once | At-least-once + dedup on message ID |

The far-right column is the punchline of the whole post: **almost every row says "at-least-once plus idempotency."** The exotic guarantees are exceptions, justified case by case. The default is boring and correct.

#### Worked example: sizing the duplicate rate you must tolerate

One more number, to make "design for duplicates" quantitative rather than a vibe. Suppose your consumer fleet processes 50,000 messages per second across a partition set, and you observe — by instrumenting redeliveries — that 0.05% of messages get redelivered at least once during normal operation (lost acks, lease expiries, the occasional rebalance). That is `50,000 × 0.0005 = 25` duplicate deliveries *per second*, or **2.16 million duplicates per day**. Not per outage — per day, on a healthy system. If even one in a thousand of those duplicates hits a non-idempotent payment path, that is `2,160` double-charges a day. The math forces the conclusion: at this scale, idempotency is not a nice-to-have you add after an incident. It is a load-bearing structural requirement, because the duplicate rate is a *constant*, not an *event*. Now add a deployment that triggers a consumer-group rebalance, and for the rebalance window the redelivery rate spikes by an order of magnitude. Your idempotency layer is what makes a routine deploy a non-event instead of a billing incident.

## Case studies and war stories

### The double-charged checkout (lost ack, no idempotency)

A mid-size e-commerce company ran its payment consumer at at-least-once with no dedup key — the classic "we'll add idempotency later." For months it was fine, because their traffic was low and lost acks were rare. Then a Black Friday traffic surge pushed their consumer processing time past the broker's 30-second visibility timeout. Messages that were *still being processed* got redelivered to other consumers. Hundreds of customers were charged two and three times in a single afternoon. The fix was not a bigger timeout — that only moves the cliff. The fix was an idempotency key passed to the payment gateway and recorded in a unique-constrained table, exactly the pattern in section 6. The lesson the team wrote on the wall: **a delivery guarantee you have not load-tested is a delivery guarantee you do not have.** The lost-ack window is invisible at low load and lethal at high load.

### The vanished orders (accidental at-most-once via auto-commit)

A logistics startup used Kafka with the consumer defaults, including auto-commit. Their processing occasionally took longer than the auto-commit interval, and during rebalances the committed offset would sometimes advance past records that had been fetched but never fully processed — silently skipping them. Orders disappeared with no error, no dead-letter, no alert. They discovered it only when a customer's shipment never arrived and the order was nowhere in the database. The root cause was that **nobody had chosen where the offset committed relative to processing** — they inherited at-most-once by default and called it "reliable Kafka." The fix was `enable.auto.commit=false` and an explicit commit after processing, converting them to at-least-once, plus idempotent upserts to make the now-possible duplicates harmless.

### The exactly-once that wasn't (boundary crossing)

A data platform team built a Kafka Streams pipeline with exactly-once semantics enabled and proudly told stakeholders that duplicates were impossible. The pipeline read events, enriched them, and — here was the bug — called an external notification service to send a push notification, *then* wrote the enriched event to an output topic. Kafka's exactly-once covered the read and the output-topic write atomically. It did *not* and could not cover the external push notification, which lived outside the transaction. When a task failed and Kafka correctly retried the transaction, the output write was deduped by Kafka — but the push notification had *already* fired on the first attempt and fired *again* on the retry. Users got duplicate notifications despite "exactly-once" being enabled. The lesson is section 5's whole thesis: **exactly-once is a boundary-local property, and the team had drawn their effect outside the boundary.** The fix was to move the notification *out* of the stream task and into a downstream idempotent consumer keyed on the event ID.

### The metrics pipeline that should have been lossy

A counter-example, because at-most-once is sometimes right. An observability team built their metrics ingestion at at-least-once with full dedup, treating every sample as precious. The dedup table became a bottleneck — billions of keys, constant lookups — and the pipeline fell behind, ironically making the monitoring system the least reliable thing in the company. The realization: metrics are *self-correcting*. A dropped CPU sample is invisible; the next one arrives in a second. They tore out the dedup layer, switched to at-most-once fire-and-forget, and the pipeline got an order of magnitude faster and *more* reliable. The lesson: **the right guarantee is workload-specific, and over-guaranteeing is its own failure mode.** Paying for at-least-once plus dedup on data that does not need it is not safety; it is overhead that can take down the system it was meant to protect.

## When to reach for each guarantee (and when not to)

**Reach for at-most-once when** the per-message value decays to near-zero immediately and the stream is self-correcting: metrics, telemetry, live frames, best-effort cache warming, high-volume logs where sampling is already acceptable. Choose it *deliberately* — by setting `acks=0` or fire-and-forget explicitly — never by accident through an unexamined auto-commit default. The danger is inheriting it without knowing.

**Reach for at-least-once (the default) when** loss is unacceptable, which is almost always. Make it your starting point for every business-meaningful flow: orders, payments, state transitions, user-facing notifications, anything you would be embarrassed to lose. Then immediately ask the second question — is the effect idempotent? — and add a dedup key wherever it is not.

**Reach for effectively-once (at-least-once + idempotency) when** duplicates would cause visible harm: double charges, double emails, double-incremented counters, duplicate orders. This is the destination for the majority of serious workloads. It is not a broker feature you turn on; it is a discipline you build — stable keys at the producer, atomic check-and-apply at the consumer.

**Reach for transactional exactly-once (Kafka EOS) only when** your entire effect stays inside one transactional resource — typically a Kafka-to-Kafka stream processing job — *and* you have measured that idempotent at-least-once is genuinely insufficient. It adds latency (transaction coordination, the transaction marker overhead) and operational complexity, and it gives you nothing the moment an effect crosses the boundary. Do not enable it as a reflex because it sounds safest. The deep mechanics and the real tradeoffs are the subject of [exactly-once in Kafka](/blog/software-development/message-queue/exactly-once-in-kafka-idempotent-producer-transactions).

**Do not** believe any system that tells you it gives exactly-once delivery across an arbitrary side effect. Do not skip idempotency "for now" on a payment path. Do not assume your guarantee without finding the ack and reading the order. And do not over-guarantee — dedup on self-correcting data is overhead masquerading as safety.

## Key takeaways

- **The delivery guarantee is decided by where you acknowledge relative to processing.** Ack before processing is at-most-once (can lose). Ack after processing is at-least-once (can duplicate). There is no third position that gives both safety and uniqueness for free.
- **At-most-once forbids duplicates and tolerates loss; at-least-once forbids loss and tolerates duplicates.** They are exact mirror images, and you choose between them by deciding which bad outcome is cheaper for *this* workload.
- **Exactly-once delivery is impossible over an unreliable network.** The two-generals problem proves the sender of an ack can never know whether the ack arrived, so the redelivery-or-loss window is fundamental, not an implementation gap.
- **What real systems call "exactly-once" is effectively-once within a boundary:** at-least-once delivery plus deduplication plus atomic commits, so duplicate *deliveries* produce no duplicate *effects*. It holds inside a transactional resource and breaks the moment an effect crosses out of it.
- **Producers duplicate too.** A retry after a lost publish-ack writes the message twice. The idempotent producer (PID + per-partition sequence numbers) dedupes retries into the broker log — but does not dedupe distinct application-level `send()` calls.
- **The end-to-end guarantee is the weaker of the producer and consumer legs.** One lossy or non-idempotent link degrades the whole chain. Reliability is a property you can only lose, never partially gain.
- **The default for almost every real workload is at-least-once plus idempotency.** Stamp a stable dedup key at the source, apply it atomically with the effect at the consumer, and you get the *behavior* of exactly-once without the impossible promise.
- **Duplicates are a constant, not an event.** At scale you see them every day from lost acks, lease expiries, and rebalances — not just from crashes. Design as though every message will arrive twice.
- **Over-guaranteeing is a failure mode.** Dedup on self-correcting data (metrics) is overhead that can take down the system it was meant to protect. Match the guarantee to the workload.
- **To audit any system's guarantee, find the ack and read the order.** Documentation lies; line ordering does not.

## Further reading

- [Idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) — the consumer-side machinery: key design, storage, TTLs, and race conditions that make effectively-once actually work.
- [Exactly-once in Kafka](/blog/software-development/message-queue/exactly-once-in-kafka-idempotent-producer-transactions) — how transactions, the transactional producer, and read-process-write wire together the in-system exactly-once this post describes.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — the durability and replication mechanics that make at-least-once on the broker side trustworthy in the first place.
- [Change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — the transactional-outbox approach to atomically committing a business change and its event.
- [RabbitMQ production architecture and scaling](/blog/software-development/system-design/rabbitmq-production-architecture-scaling) — acknowledgement modes, redelivery, and dead-lettering in a classic broker.
- [Distributed replication: leader, multi-leader, leaderless](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless) — why `acks=all` and in-sync replicas are what durability actually rests on.
- ["Exactly-Once Semantics Are Possible: Here's How Kafka Does It"](https://www.confluent.io/blog/exactly-once-semantics-are-possible-heres-how-apache-kafka-does-it/) — Confluent's own account, read critically with this post's boundary caveat in mind.
- The two-generals problem — the original impossibility result that underpins everything here; any distributed-systems text or the original 1975 framing by Akkoyunlu, Ekanadham, and Huber.
