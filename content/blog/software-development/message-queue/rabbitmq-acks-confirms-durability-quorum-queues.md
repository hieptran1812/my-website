---
title: "RabbitMQ Deep Dive, Part 2: Acks, Publisher Confirms, Durability, and Quorum Queues"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Trace a message end to end through RabbitMQ and learn exactly where it gets lost or saved: consumer acks and the unacked set, publisher confirms, why durability needs both a durable queue and a persistent message, dead-letter exchanges, and why quorum queues replaced mirrored queues for HA."
tags:
  [
    "message-queue",
    "rabbitmq",
    "acknowledgements",
    "publisher-confirms",
    "durability",
    "quorum-queues",
    "kafka",
    "distributed-systems",
    "event-driven",
    "reliability",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/rabbitmq-acks-confirms-durability-quorum-queues-1.webp"
---

Here is a question that separates the people who have operated RabbitMQ in production from the people who have only read the tutorial: if your broker process is killed by the OOM killer right now, in the middle of a busy Tuesday, how many messages do you lose? The honest answer for most teams the first time they ask is "I have no idea," and the second honest answer, after they go look, is usually "more than I would like." Losing messages on a broker restart is almost never a RabbitMQ bug. It is a configuration that did exactly what it was told to do, which was to keep everything in RAM for speed and throw it away on shutdown. RabbitMQ will happily run that way. It will run fast that way. It will also lose your data that way, and it will not warn you, because you asked for it.

This post is about the machinery that decides the answer to that question. It is the second part of a deep dive on RabbitMQ. [Part 1 covered AMQP itself](/blog/software-development/message-queue/rabbitmq-amqp-exchanges-bindings-routing) — exchanges, bindings, routing keys, and how a publish finds its way to a queue. This part picks up where the message lands in the queue and asks the harder question: once it is there, how do we make sure it does not vanish? We will trace a single message all the way from a publisher that wants a guarantee, through the broker's persistence and replication layers, out to a consumer that has to tell the broker it is done, and we will stop at every single point where a crash, a network blip, or a misconfiguration could silently drop it on the floor.

![The full reliability path from a publisher in confirm mode through broker persistence and a durable queue to consumer acknowledgement, showing the five hops a message must survive](/imgs/blogs/rabbitmq-acks-confirms-durability-quorum-queues-1.webp)

Look at the path in the figure above, because it is the spine of everything that follows. A message is not "safe" at any single point — it is safe only when every hop on that chain is configured to hold it. The publisher has to ask for a confirmation. The broker has to mark the message persistent and the queue durable so both end up on disk. The disk write has to actually reach the platter, not just the OS page cache. If you want to survive a whole node dying, the message has to be replicated to a majority of nodes before it counts. And the consumer has to acknowledge it only after the work is genuinely done, not the instant it arrives. Break any one of those links and the chain does not hold. By the end of this post you will be able to look at any RabbitMQ setup and say, precisely, which links are present and which are missing — and therefore exactly how much data a crash will cost you.

## 1. The end-to-end reliability chain

Let me set the frame before we descend into mechanics, because the single most common mistake with RabbitMQ reliability is to fix one link and assume the chain is now strong. People turn on publisher confirms and feel safe, not realizing their queue is transient and a restart wipes it anyway. People mark messages persistent and feel safe, not realizing their consumers ack on receipt and lose in-flight work on every crash. Reliability is a property of the *whole path*, and the path has exactly three owners.

The **publisher** owns the question "did the broker actually take responsibility for my message?" The only honest answer to that comes from a publisher confirm — an asynchronous acknowledgement from the broker that says "I have it, it is safely handled, you can let go." Without that, a `basic.publish` is fire-and-forget. The publish call returns successfully the instant the bytes leave the socket buffer, which tells you nothing about whether the broker received them, routed them, or wrote them down. A network partition between you and the broker can swallow a publish whole and your code will never know.

The **broker** owns the question "if I crash and restart, do I still have this message?" That answer comes from durability: the queue definition must be durable so the queue itself survives a restart, and the message must be persistent so its bytes were written to disk rather than held only in memory. If you want to survive not just a restart but the permanent loss of an entire node, the broker also owns replication — and that is where quorum queues come in, replacing the old mirrored-queue mechanism that we will spend real time dismantling later.

The **consumer** owns the question "did this message's work actually complete, or did I crash halfway through?" That answer comes from acknowledgements. In manual-ack mode the broker hands you a message but keeps a copy in an *unacked set* and considers it un-finished until you send `basic.ack`. If your channel or connection drops before that ack, the broker automatically requeues the message and redelivers it. The consumer's ack is the contract that converts "the broker delivered it" into "the work got done."

The reason it helps to think of these as three separate owners is that they fail independently and you configure them independently. You can have a bulletproof broker with a sloppy consumer and lose messages on every consumer crash. You can have careful consumers and a transient queue and lose everything on a broker restart. The reliability chain figure is your checklist: five hops, three owners, and you have to satisfy all of them or you do not have the guarantee you think you have. This is also why it pairs so naturally with the broader [delivery semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) discussion — at-most-once, at-least-once, and the fiction of exactly-once are not abstract categories, they are emergent properties of which links in this chain you turned on.

### What "lost" actually means

Before we go further it is worth being precise about the failure we are defending against, because "lost a message" covers several very different events:

- **Lost on publish**: the publisher thought it sent the message but the broker never got it (network drop, broker overload rejecting the connection, an exchange with no matching binding silently discarding it).
- **Lost on broker restart**: the broker had the message in memory but it was never written to disk, so a restart wipes it.
- **Lost on node failure**: the message was on disk on one node, but that node's disk or hardware is gone for good and the message was never replicated elsewhere.
- **Lost on consumer crash**: the broker delivered the message and removed it from the queue, the consumer crashed mid-processing, and because it had already acked (or was in auto-ack mode), the broker has no copy to redeliver.

Each of these is defended by a different link in the chain, and confusing them is how people build systems that are robust against the failure they expected and helpless against the one that actually happens. Publisher confirms defend against lost-on-publish. Durability defends against lost-on-restart. Replication defends against lost-on-node-failure. Manual acks defend against lost-on-consumer-crash. Four distinct failures, four distinct defenses, and you need all four for a true no-loss setup. We will build exactly that setup in section 9.

## 2. Consumer acknowledgements and the unacked set

Start at the consumer end, because it is the link people understand worst and the one that bites earliest. When a consumer registers with `basic.consume`, it chooses an acknowledgement mode. The two that matter are *automatic* acknowledgement and *manual* acknowledgement, and the difference between them is the difference between at-most-once and at-least-once delivery on the consumer side of the wire.

In **automatic ack mode** (`auto_ack=True`, sometimes called "no-ack"), the broker considers a message acknowledged the moment it is written into the delivery to the consumer's socket. The broker hands the message off and immediately forgets about it. This is fast — there is no second round trip for the ack — but it is also reckless. The instant the broker sends the message, it deletes its own copy. If the message is sitting in the consumer's TCP buffer when the consumer process crashes, or if your handler throws an exception three lines into processing, that message is gone. The broker already considers it delivered and done. Auto-ack is at-most-once delivery, and it is appropriate only when losing a message costs you nothing — a metrics sample, a non-critical cache invalidation, a log line you can afford to drop.

In **manual ack mode** (the default and correct choice for anything that matters), the broker delivers the message but keeps a copy and considers it *unacknowledged*. It enters what RabbitMQ calls the unacked set: the collection of messages that have been delivered to a consumer but not yet acked. The broker tracks this set per channel. The message stays in the unacked set until one of three things happens: the consumer sends `basic.ack` for it (the broker drops its copy, the message is finally done), the consumer sends `basic.nack` or `basic.reject` for it (we will get to those), or the channel or connection drops (the broker requeues every unacked message on that channel and redelivers it). That last behavior is the whole point: manual acks give you at-least-once delivery because an unacked message is never lost — it is held until you confirm completion or until the broker has reason to believe you failed.

### The unacked set and prefetch

Here is the part that surprises people: the unacked set is not just a correctness mechanism, it is a *flow-control* mechanism, and it interacts directly with prefetch. In [the push-versus-pull post](/blog/software-development/message-queue/push-vs-pull-acknowledgements-how-consumers-read) I argued that RabbitMQ is a push broker and therefore needs an explicit knob to stop the broker from flooding a slow consumer. That knob is `basic.qos` with a prefetch count, and what it actually limits is the *size of the unacked set per consumer*.

If you set prefetch to 1, the broker will deliver exactly one message and then refuse to deliver another until that one is acked. The unacked set never exceeds one. This gives you the fairest possible distribution across consumers and the smallest possible blast radius on a crash, at the cost of a round trip of latency per message. If you set prefetch to 100, the broker will keep up to 100 unacked messages outstanding to that consumer, which lets it pipeline delivery and amortize network latency, at the cost of having 100 messages requeued and redelivered if that consumer dies. Prefetch is a direct dial between throughput and crash blast radius, and it is denominated in unacked-set size. Forget to set it and the default in modern RabbitMQ is an unlimited unacked set, which means a single greedy consumer can pull the entire queue into its own memory.

Here is the consumer side in Python with the `pika` client, doing it correctly:

```python
import pika

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host="rabbit-1")
)
channel = connection.channel()

# Declare the queue as durable so it survives a broker restart.
channel.queue_declare(queue="orders", durable=True)

# Limit the unacked set: at most 20 messages outstanding to this consumer.
channel.basic_qos(prefetch_count=20)

def handle(ch, method, properties, body):
    try:
        process_order(body)          # the real work
        ch.basic_ack(delivery_tag=method.delivery_tag)   # ack ONLY after success
    except Exception:
        # Don't requeue a poison message forever; route it to a DLX instead.
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

# auto_ack defaults to False here, which is what we want.
channel.basic_consume(queue="orders", on_message_callback=handle)
channel.start_consuming()
```

The single most important line in that snippet is `ch.basic_ack(...)` sitting *after* `process_order(body)`. If you move it before the work, you have silently converted a safe at-least-once consumer into a lossy at-most-once one. The ordering of those two lines is the entire guarantee.

### Multiple-ack: acking a batch at once

`basic.ack` carries a `delivery_tag`, which is a per-channel monotonically increasing integer identifying the delivery. By default an ack confirms exactly that one tag. But the ack frame also has a `multiple` flag, and when you set it, the ack confirms *every* unacknowledged delivery up to and including that tag. This is a real throughput optimization: if you have processed deliveries 1 through 50 and you send a single ack for tag 50 with `multiple=True`, the broker drops all 50 from the unacked set in one operation instead of fifty. The catch is that you must actually be done with all of them — multiple-ack confirms a contiguous range, so if delivery 37 failed, you cannot multiple-ack up to 50, because that would falsely confirm 37. Multiple-ack is a batching tool for the happy path where you process strictly in order and want to amortize the ack round trips.

### Channels, connections, and the heartbeat that triggers requeue

The unacked set is tracked *per channel*, and to understand when it gets requeued you have to be precise about the difference between a channel and a connection, because the requeue is triggered by the loss of one of them and beginners routinely conflate the two. A **connection** is a single TCP socket from your client to the broker. A **channel** is a lightweight logical session multiplexed *inside* that connection — you can open dozens of channels over one TCP connection, and AMQP frames carry a channel number so the broker can demultiplex them. Channels exist so you do not have to pay for a TCP connection per concurrent stream of work; a connection is heavyweight (a real socket, a real file descriptor on the broker), a channel is cheap.

The reason this matters for reliability is that the unacked set, the consumer registration, and the delivery tags are all *channel-scoped*. When a channel closes — because you closed it, because the broker closed it after a protocol error, or because the underlying connection died — every message in that channel's unacked set is requeued. So there are two granularities of failure that trigger redelivery: a single channel dying (other channels on the same connection are unaffected, only that channel's unacked messages requeue) and the whole connection dying (every channel on it dies, and all of their unacked sets requeue at once). The blast radius of a crash is therefore "all unacked messages on all channels of the dead connection," which is one more reason the prefetch-times-channel-count product is the number you actually care about when sizing.

How does the broker *notice* a connection has died when the client process is killed without cleanly closing the socket? Through **heartbeats**. AMQP connections negotiate a heartbeat interval (commonly 60 seconds, often tuned lower); if the broker sees no frame and no heartbeat from a client for two intervals, it concludes the connection is dead and tears it down — requeuing all the unacked messages on it. This is why a hung consumer (one that is alive but stuck, not sending heartbeats because its event loop is blocked) can get its connection forcibly closed and its in-flight messages redelivered to a healthy consumer. It is also why a too-short heartbeat plus a long, CPU-bound, single-threaded message handler is a classic foot-gun: the handler blocks the I/O loop, heartbeats stop, the broker kills the connection mid-processing, and the message gets redelivered even though the consumer was making progress. The fix is to do heavy work off the I/O thread or to set a heartbeat interval comfortably longer than your worst-case processing time. The heartbeat is the broker's liveness detector, and the requeue is what it does when liveness fails.

## 3. nack, reject, requeue, and redelivery

Acking says "done, drop it." But what do you say when processing *fails*? That is the job of `basic.reject` and `basic.nack`, and the difference between requeuing and not requeuing is where a lot of production incidents are born.

`basic.reject` is the original AMQP negative acknowledgement. It rejects a single delivery and carries one flag: `requeue`. If `requeue=True`, the broker puts the message back at (approximately) the head of the queue to be redelivered. If `requeue=False`, the broker discards the message — or, if the queue has a dead-letter exchange configured, routes it there instead (more on that in section 6). `basic.reject` handles exactly one message at a time.

`basic.nack` is RabbitMQ's extension to AMQP, and it is `basic.reject` plus the `multiple` flag. It can negatively acknowledge a whole range of deliveries at once, just like multiple-ack does for the positive case. In practice you reach for `basic.nack` whenever you would reach for `basic.reject` but want the batch capability; functionally for a single message they are identical.

![Timeline of a consumer acknowledgement lifecycle showing a message entering the unacked set, the channel dropping before the ack, and the broker automatically requeuing and redelivering with the redelivered flag set](/imgs/blogs/rabbitmq-acks-confirms-durability-quorum-queues-4.webp)

The timeline above traces the most important failure path in this whole section: a channel that drops before the ack. Walk it left to right. The broker delivers a message; it immediately enters the unacked set. The consumer starts doing work. Then — at T+80ms in the figure — the channel drops. Maybe the consumer crashed, maybe a network blip killed the TCP connection, maybe the consumer's heartbeat timed out and the broker tore the connection down. The broker does not know *why*; it only knows the channel carrying that unacked message is gone. So it does the safe thing: it requeues every unacked message on that channel and makes it available for redelivery to another consumer (or the same one when it reconnects). The redelivered message arrives with its `redelivered` flag set to `true`, which is the broker's honest admission: "I am giving you this again because I am not sure it was fully processed the first time."

### The redelivery storm, and why requeue=true is a trap

That automatic requeue is a feature, but `requeue=True` on an *explicit* nack is one of the most dangerous defaults in the system. Picture a poison message: a message whose content reliably crashes your handler — a malformed payload, a reference to a row that was deleted, anything that throws every single time. Now picture a handler that catches the exception and nacks with `requeue=True`. The broker requeues the message. Your consumer picks it up again. It crashes again. It nacks with requeue again. The broker requeues it again. You have built an infinite loop that pins a CPU, fills your logs, and — because the poison message keeps jumping to the front of the queue — can starve every legitimate message behind it. This is the **redelivery storm**, and it has taken down more RabbitMQ-backed services than any broker bug.

The fix is a discipline: never blindly `requeue=True` a message that failed because of its *content*. Requeue is appropriate for *transient* failures — the downstream database was briefly unreachable, you got a 503 from an API you call, the message is fine and will probably succeed on retry. For *permanent* failures — the message is malformed, the work is impossible — you must `requeue=False` and let the message go to a dead-letter exchange where a human or a separate process can inspect it. Distinguishing transient from permanent failures inside your handler is the single most important piece of error-handling logic in a RabbitMQ consumer. Get it wrong in the requeue-everything direction and you get redelivery storms; get it wrong in the discard-everything direction and you throw away messages that would have succeeded on retry.

A more robust pattern caps the retries. RabbitMQ does not natively count redeliveries, so the common approach is a dead-letter loop with a TTL: reject to a DLX, the DLX routes to a delay queue with a message TTL, and on expiry the message dead-letters *back* to the original queue, with a header counting the attempts. After N attempts you route it to a final parking queue. We will build the DLX half of that in section 6. The point for now is that the `requeue` flag is binary and dumb, and any production system needs a retry-counting layer on top of it.

### Negative acks and ordering

One subtlety worth flagging: when the broker requeues a message — whether from a dropped channel or an explicit `requeue=True` — it tries to put it back near the head of the queue, but it makes no strict guarantee about exact position relative to messages that arrived in the meantime. If your application depends on strict FIFO ordering, a requeue can reorder a message relative to its neighbors. For most workloads this is fine. For workloads that genuinely require ordering, you need the ordering guarantees discussed in [message ordering and partitioning](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees) — and you should know that requeue is one of the operations that can quietly violate the order you assumed.

## 4. Publisher confirms vs transactions

Now flip to the other end of the wire. Everything so far has been about the consumer's relationship with the broker. But there is a symmetric question on the publisher side: when you call `basic.publish` and it returns, what exactly has happened? The uncomfortable answer is *almost nothing you can rely on*. AMQP's `basic.publish` is asynchronous and unacknowledged by default. Your client serializes the message, writes it to the socket, and returns. The broker may not have received it yet. It may receive it and find no queue to route it to and discard it. It may receive it, route it, and then crash before writing it to disk. In none of those cases does your publish call report a problem. Fire-and-forget is the default, and fire-and-forget loses messages.

There are two mechanisms to fix this, and one of them is the wrong answer that everyone tries first.

### AMQP transactions: the slow, deprecated answer

AMQP has a transaction mechanism: `tx.select` puts the channel into transactional mode, you publish a batch of messages, then you call `tx.commit` and the broker confirms that all of them are handled, or `tx.rollback` to discard them. This *works* — a successful `tx.commit` does genuinely mean the broker has accepted and persisted the batch. The problem is that it is brutally slow. Each `tx.commit` is synchronous and blocks the publisher until the broker has fully processed and fsynced the entire transaction. You publish, you wait, you publish, you wait. On a real workload, transactions can cut publish throughput by an order of magnitude or worse, because you have serialized your entire publish stream behind synchronous round trips. The RabbitMQ team's own guidance is blunt: do not use transactions for throughput-sensitive publishing. They exist mostly for the rare case where you need all-or-nothing atomicity across a small batch and can afford the cost.

### Publisher confirms: the right answer

The modern, correct mechanism is **publisher confirms**. You call `confirm.select` once to put the channel into confirm mode. From then on, the broker assigns every message you publish a monotonically increasing sequence number, and it asynchronously sends you a `basic.ack` (a confirm) once the message is *safely handled* — which for a persistent message routed to a durable queue means it has been written to disk, and for a quorum queue means it has been committed to a majority of replicas. Crucially, this is *asynchronous*: you keep publishing without blocking, the confirms stream back on their own schedule, and you correlate them to your published messages by sequence number. You get the safety of knowing the broker took responsibility, without the throughput collapse of synchronous transactions.

![Publisher confirms flow showing a message routed to two durable queues, each fsynced, with the broker returning a basic.ack carrying the sequence number once both are persisted, or a basic.nack on internal failure](/imgs/blogs/rabbitmq-acks-confirms-durability-quorum-queues-3.webp)

The flow in the figure is the heart of it. The publisher, in confirm mode, publishes a message. The exchange routes it to whatever durable queues match — two in the figure. Each queue persists the message to disk. Only once *all* of the queues the message was routed to have safely persisted it does the broker send back a `basic.ack` carrying that message's sequence number. That confirm is the broker saying "I now own this; it is on disk; if I crash and restart, it will still be here." If the broker hits an internal error that prevents it from taking responsibility, it sends a `basic.nack` instead — which on the publisher side means "I could not safely handle this, you should republish it." The asymmetry with consumer acks is worth noting: a consumer's `basic.nack` means "I, the consumer, failed"; a broker's `basic.nack` to a publisher means "I, the broker, failed." Same frame, opposite direction, opposite owner of the failure.

Here is the publisher side, again with `pika`, using confirms correctly:

```python
import pika

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host="rabbit-1")
)
channel = connection.channel()
channel.queue_declare(queue="orders", durable=True)

# Put the channel into confirm mode. From here on the broker confirms publishes.
channel.confirm_delivery()

def publish_order(payload: bytes) -> bool:
    try:
        channel.basic_publish(
            exchange="",
            routing_key="orders",
            body=payload,
            properties=pika.BasicProperties(
                delivery_mode=2,   # 2 = persistent; the message is written to disk
            ),
            mandatory=True,        # fail loudly if it can't be routed (section 6)
        )
        # With BlockingConnection, confirm_delivery makes basic_publish
        # raise on nack / return, so reaching here means a positive confirm.
        return True
    except pika.exceptions.UnroutableError:
        log.error("order was unroutable; no queue matched")
        return False
    except pika.exceptions.NackError:
        log.error("broker nacked the publish; message NOT safely stored")
        return False
```

The `delivery_mode=2` property is doing essential work there — it marks the message persistent, which we are about to see is one of the two halves of durability. A publisher confirm on a *non*-persistent message into a *non*-durable queue confirms only that the broker has it in memory; the confirm is honest but the guarantee is weak. Confirms and durability are separate knobs, and you almost always want both.

### The high-throughput confirm pattern

The blocking style above is simple but it serializes one confirm per publish, which throttles you. The high-throughput pattern is fully asynchronous: register a confirm callback, keep a map of outstanding sequence numbers to messages, publish as fast as you can, and let confirms arrive out of band. When a confirm for sequence N arrives with `multiple=True`, you can clear every outstanding message up to N (the same multiple semantics as consumer acks). When a nack arrives, you republish just those messages. This decouples your publish rate from your confirm latency and is how you get both safety and throughput. The bookkeeping is real work, which is why most mature client libraries and frameworks ship a "publisher confirms with retry" helper rather than making you write the sequence-number map by hand.

| Mechanism | Throughput | Guarantee | When to use |
| --- | --- | --- | --- |
| Fire-and-forget | Highest | None (lossy) | Metrics, non-critical events |
| AMQP transactions | Lowest | Atomic batch, persisted | Rare all-or-nothing batches |
| Publisher confirms (sync) | Medium | Each publish persisted | Simple, correctness over speed |
| Publisher confirms (async) | High | Each publish persisted | Production default for safety |

## 5. Durability: durable queues + persistent messages (you need both)

Now the link that catches the most people. There are two completely separate flags involved in surviving a broker restart, they are named confusingly, and you need *both*. Set one without the other and you have a setup that looks durable, passes a casual code review, and loses everything on restart.

The first flag is **queue durability**, set when you declare the queue: `queue_declare(queue="orders", durable=True)`. This flag is about the queue *definition*, the metadata: its name, its bindings, its arguments. A durable queue's definition is written to disk, so when the broker restarts, the queue still exists. A *non*-durable (transient) queue's definition lives only in memory, so on restart the queue itself is simply gone — and any messages it held vanish with it, persistent or not.

The second flag is **message persistence**, set per message via the `delivery_mode` property: `delivery_mode=2` means persistent, `delivery_mode=1` (the default) means transient. A persistent message's *bytes* are written to disk. A transient message lives only in memory even if the queue it sits in is durable.

Now the trap, stated as plainly as I can: **a message survives a broker restart only if the queue is durable AND the message is persistent.** Both. Here is why each combination fails or succeeds:

- **Transient queue + persistent message**: lost. The queue definition is gone on restart, so there is no queue to hold the message; the persistent flag is irrelevant because the container itself evaporated.
- **Durable queue + transient message**: lost. The queue comes back on restart, but it comes back *empty*, because the messages were never written to disk — they were memory-only.
- **Transient queue + transient message**: lost, obviously, on both counts.
- **Durable queue + persistent message**: survives. The queue definition is on disk, the message bytes are on disk, the broker reconstructs the queue and replays its contents on restart.

![A before-and-after comparison contrasting a transient queue and message that are gone after a broker restart against a durable queue holding a persistent fsynced message that survives the restart](/imgs/blogs/rabbitmq-acks-confirms-durability-quorum-queues-2.webp)

The figure makes the two paths concrete. On the left, the queue lives in RAM only; the broker restarts; the queue and its messages are gone. On the right, the queue definition is on disk and the message bytes were fsynced with `delivery_mode=2`; the restart finds everything where it left it. The reason both flags exist as separate knobs is that durability costs performance, and RabbitMQ wants to let you pay for it only where you need it. A durable queue full of transient messages is a legitimate, useful thing — you keep the queue's existence and bindings stable across restarts but accept that the in-flight messages are best-effort. The system does not assume; it makes you ask for each guarantee explicitly.

### The fsync and batching reality

"Written to disk" is doing a lot of work in that paragraph, and here is where the honest, operations-level truth lives. When the broker "persists" a message, the first thing that happens is a write to the OS, which lands in the operating system's page cache — RAM that the OS will *eventually* flush to the physical disk. A write that is only in the page cache is **not** safe against a power loss or a kernel panic, because the bytes have not reached the platter. To make the write truly durable, the broker must call `fsync`, which forces the OS to flush the page cache to the physical device and does not return until the device confirms the bytes are down.

`fsync` is expensive — it is one of the slowest operations in the whole stack, easily milliseconds, because it waits on physical hardware. If RabbitMQ called `fsync` after every single persistent message, persistent-message throughput would crater. So it doesn't. The broker **batches** writes and fsyncs them as a group — historically on a timer (on the order of a couple hundred milliseconds) or when the batch reaches a size threshold, whichever comes first. This is the crucial and under-appreciated detail: there is a window, typically up to a few hundred milliseconds wide, during which a "persistent" message that has been written but not yet fsynced lives only in the page cache. If the machine loses power in that window, that message is gone *even though it was marked persistent and the queue was durable.*

This is not a bug; it is the fundamental durability-versus-throughput tradeoff, and every disk-backed system makes some version of it. The point is to understand what your durability actually buys you. Durable queue plus persistent message protects you fully against a clean broker *restart* (a graceful shutdown flushes, a process crash leaves the OS page cache intact so the data is still there). It protects you against a process crash. It does *not* fully protect you against a sudden hardware power loss within the fsync window — for that you need replication so that the message exists on more than one machine's disk, which is exactly what quorum queues give you. The layers stack, and that is the next idea.

![A stack of durability layers from durable queue at the top through persistent message and batched fsync down to replication, illustrating that a message is only as safe as the lowest layer it reaches](/imgs/blogs/rabbitmq-acks-confirms-durability-quorum-queues-5.webp)

The stack in the figure is the mental model to carry around. Each layer protects against a strictly larger class of failure than the one above it. A durable queue protects the queue's existence. A persistent message protects the message's bytes against a restart. An fsync protects those bytes against a power loss — but only once the batch flushes. Replication protects against the permanent loss of an entire machine. A message is exactly as safe as the lowest layer it has actually reached, no safer. When someone tells you their RabbitMQ is "durable," the right follow-up question is *down to which layer* — and the honest answer for most single-node setups is "down to a clean restart, not down to a power loss, and definitely not down to a dead disk."

## 6. Mandatory, returns, and dead-letter exchanges

We have covered surviving crashes. Now cover the quieter loss: a message that is published successfully, persisted faithfully, and *still* disappears because it had nowhere to go. This is the routing-loss problem, and RabbitMQ gives you three tools to handle it: the `mandatory` flag with `basic.return`, dead-letter exchanges, and TTL.

### Mandatory and basic.return

When you publish to an exchange, the exchange tries to route the message to one or more queues based on bindings (the full mechanics are in [Part 1](/blog/software-development/message-queue/rabbitmq-amqp-exchanges-bindings-routing)). But what if *no* binding matches? By default, the exchange silently discards the message. It is gone, and — here is the cruel part — if you are using publisher confirms, you still get a *positive* confirm, because from the broker's point of view it handled the message correctly: it routed it according to the rules, and the rules said "nowhere." The message was unroutable, and unroutable is not, by default, an error.

The `mandatory` flag fixes this. When you publish with `mandatory=True`, you are telling the broker: this message *must* be routed to at least one queue; if it cannot be, do not discard it silently — send it back to me. An unroutable mandatory message comes back to the publisher as a `basic.return` frame, which your client surfaces as a return callback (or, in the blocking `pika` style above, as an `UnroutableError`). This is your early warning that a binding is missing or a routing key is wrong — a class of bug that otherwise manifests as messages vanishing with no trace and a green confirm count. Always set `mandatory=True` on publishes you care about, and always handle the return. A returned message is a configuration bug shouting at you; ignoring it is how you ship a service that loses a fraction of its traffic to a typo in a routing key.

### Dead-letter exchanges

A dead-letter exchange (DLX) is a second exchange that a queue forwards messages to when they meet a "dead" condition. There are exactly three conditions that dead-letter a message:

1. The message is **rejected** with `basic.reject` or `basic.nack` with `requeue=False`.
2. The message's **TTL expires** while it sits in the queue (per-message TTL or a queue-wide message-TTL argument).
3. The queue exceeds its **length limit** and the message is dropped from the head to make room.

You configure a DLX by setting the `x-dead-letter-exchange` argument on a queue. When any of those three conditions fires, instead of discarding the message, the broker republishes it to the named dead-letter exchange, which routes it like any other exchange to whatever queues are bound to it — typically a dedicated dead-letter queue where failed messages accumulate for inspection. The dead-lettered message arrives with an `x-death` header recording why it died, which queue it came from, and how many times it has been dead-lettered. That header is what you use to build the retry-counting layer I mentioned in section 3.

```python
# Declare a main queue that dead-letters to a DLX on reject/expiry/overflow.
channel.exchange_declare(exchange="orders.dlx", exchange_type="direct", durable=True)
channel.queue_declare(queue="orders.dead", durable=True)
channel.queue_bind(queue="orders.dead", exchange="orders.dlx", routing_key="orders")

channel.queue_declare(
    queue="orders",
    durable=True,
    arguments={
        "x-dead-letter-exchange": "orders.dlx",     # where rejected msgs go
        "x-dead-letter-routing-key": "orders",      # routing key used on dead-letter
        "x-message-ttl": 600000,                    # 10 min TTL; expiry dead-letters too
    },
)
```

This is the standard shape of a production RabbitMQ topology: a main queue, a DLX, and a dead-letter queue, with TTL providing both a freshness bound and a mechanism for building delayed retries. The DLX is what turns "a message failed" from "a message was lost" into "a message is sitting in a queue where I can find it, retry it, alert on it, or hand it to a human." The forward-looking siblings on [the dead-letter queue pattern](/blog/software-development/message-queue/dead-letter-queues-and-poison-message-handling) and [idempotency](/blog/software-development/message-queue/idempotent-consumers-and-deduplication) go deeper on the operational side — how to drain a DLQ safely, how to make redelivery harmless. For our purposes here, the DLX is the link in the reliability chain that catches everything the other links reject, so that "rejected" never means "vanished."

### A taxonomy of the mechanisms so far

We have now introduced enough machinery that it is worth stepping back and organizing it by which side of the wire owns it, because that organization is exactly how you should think about configuring a new service.

![A tree taxonomy of RabbitMQ reliability mechanisms grouped under publisher side, broker side, and consumer side, with confirms and mandatory under the publisher, durability and quorum queues and dead-letter exchanges under the broker, and manual acks with requeue under the consumer](/imgs/blogs/rabbitmq-acks-confirms-durability-quorum-queues-9.webp)

The tree splits the entire reliability surface into the three owners from section 1. The publisher owns confirms and the mandatory flag — the mechanisms that tell the producer whether the broker accepted and could route its message. The broker owns durability (durable queues plus persistent messages), quorum queues (replication), and dead-letter exchanges — the mechanisms that keep a message alive across restarts, node failures, and rejections. The consumer owns manual acks and the requeue discipline — the mechanisms that ensure work is not marked done until it is done. When you onboard a new service onto RabbitMQ, walk this tree and ask, for each branch, "is this turned on, and is it turned on correctly?" Most reliability incidents are a missing leaf on this tree, and naming the owner is the fastest way to find which one.

## 7. Classic mirrored queues and why they fell short

Everything to this point keeps a message safe on a *single* broker. But a single broker is a single point of failure. To survive a node dying — its disk failing, its hardware catching fire, its rack losing power — the message has to exist on more than one machine. That means replication, and RabbitMQ has had two very different answers to replication over its history. The old answer was classic mirrored queues. The new answer is quorum queues. Understanding why the old answer failed is the best possible motivation for the new one, so we will dwell on it.

A **classic mirrored queue** (sometimes "HA queue") worked like this. One node held the **queue master** — the authoritative copy that handled all the work: every publish to the queue went to the master, every consume came from the master, the master did all the routing and ordering. The other nodes in the cluster held **mirrors** — passive copies that received a stream of updates from the master and kept their own replica of the queue's contents. If the master node failed, RabbitMQ promoted one of the mirrors to be the new master, and consumers and publishers reconnected to it. On paper this is straightforward leader-follower replication, the same shape covered in [distributed replication](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless). In practice it had a set of problems severe enough that the RabbitMQ team eventually deprecated it entirely.

### Problem one: the master is a bottleneck

Because *all* traffic for a mirrored queue funneled through its single master node, that one node's capacity was the queue's capacity. Mirrors did no useful read or write work; they only absorbed the replication stream. You could not scale a single mirrored queue's throughput by adding nodes — you could only add more *copies* of the same bottleneck. A hot queue pinned one node while the rest of the cluster sat underused. The replication itself added overhead to the master without buying it any more capacity, so turning on mirroring made a queue *slower*, not just safer. This is the "queue master bottleneck" and it is why mirroring scaled badly precisely where you most wanted it: on your busiest queues.

### Problem two: synchronization was painful

When a mirror fell behind, or a new mirror was added, or a failed node rejoined, it had to **synchronize** — copy the queue's current contents from the master to catch up. For a queue holding a lot of messages, this synchronization could transfer gigabytes, and critically, the default behavior blocked the queue during sync: the master could not make progress while a mirror was catching up, so publishers and consumers stalled. Operators learned to dread `rabbitmqctl sync_queue` on a deep queue because it could freeze the queue for minutes. The alternatives — letting unsynchronized mirrors be promotion candidates — were worse, because promoting an out-of-date mirror meant *losing every message the old master had that the mirror had not yet received*. You were forced to choose between availability (promote a stale mirror, lose messages) and consistency (refuse to promote, the queue is down until sync completes). That is a miserable choice to make at 3 a.m.

### Problem three: split-brain

The deepest problem was **split-brain**. Mirrored queues used a coordination model that, under a network partition, could end up with two halves of the cluster each believing they held the legitimate master for a queue. Both halves accept publishes. Both halves run consumers. When the partition heals, RabbitMQ has to reconcile two divergent histories, and its reconciliation strategies all involved *throwing one side's data away* — the `autoheal` and `pause_minority` partition-handling modes were attempts to manage this, and none of them could conjure back the messages that were sacrificed. Mirroring's replication had no notion of a *quorum* — no requirement that a majority of nodes agree before a write counts — so it had no principled way to prevent two minorities from each making progress independently. Split-brain plus mirrored queues meant silent, unrecoverable message loss on partition heal, and that is the failure mode that finally made the mechanism untenable for serious use.

The lesson of mirrored queues is the lesson of a generation of distributed systems: ad-hoc leader-follower replication without a consensus protocol underneath cannot give you both safety and availability under partitions, because it has no rigorous definition of which side is allowed to make progress. The industry's answer to that problem has a name, and it is Raft.

## 8. Quorum queues: Raft replication for safety

A **quorum queue** is RabbitMQ's replacement for mirrored queues, and it is built on the **Raft consensus algorithm**. Raft is a well-understood, formally specified protocol for getting a cluster of machines to agree on an ordered log of operations even as individual machines fail and recover. Kafka's replication, etcd, Consul, and CockroachDB all rest on Raft or its close cousin; if you have read [the Kafka-as-a-log post](/blog/software-development/database/kafka-as-a-distributed-log), the replicated-log idea is the same family. Quorum queues bring that rigor to RabbitMQ, and the difference in operational character is night and day.

Here is the model. A quorum queue is replicated across an odd number of nodes — typically three or five. One replica is the **leader**; the others are **followers**. Every operation that changes the queue — a publish enqueuing a message, an ack dequeuing one — is an entry that the leader appends to its Raft log and replicates to the followers. The decisive rule is the **majority commit**: an operation is *committed*, and only then acknowledged back to the publisher, once a **majority** of the replicas have written it to their own logs. For a three-node quorum queue, a majority is two. For a five-node queue, a majority is three.

![A grid showing a quorum queue's Raft commit cycle: a publisher awaits confirmation while the leader appends an entry, two followers replicate it, the write commits once a majority of two of three replicas persist it, and the ack flows back to the leader and then the publisher](/imgs/blogs/rabbitmq-acks-confirms-durability-quorum-queues-7.webp)

Trace the commit cycle in the figure. The publisher sends a message and awaits its confirm. The leader appends the entry to its Raft log. The two followers replicate it to their logs. As soon as a *majority* of the three replicas — the leader plus at least one follower — have persisted the entry, it is committed: `majority = 2 of 3`. The ack flows back through the leader to the publisher's confirm. The beauty of the majority rule is what it guarantees about failure. Because a write is committed only when a majority has it, and because any *new* leader must also be elected by a majority, any two majorities necessarily overlap in at least one node — and that overlapping node carries every committed entry. This is the mathematical heart of why Raft cannot lose a committed write or split-brain: there is no way for two non-overlapping groups to each form a majority, so there is never a second leader making independent progress. The problems that killed mirrored queues are *structurally impossible* here, not merely handled.

### Failover with Raft

When the leader of a quorum queue fails, the followers detect the missing heartbeats and hold an election. A follower whose log is at least as up to date as the majority becomes the new leader — and because committed entries are by definition on a majority, the new leader is guaranteed to have every committed message. Failover is a leader election that completes in seconds, and it loses *no committed data*. Compare that to the mirrored-queue choice between a slow safe promotion and a fast lossy one: Raft makes the safe promotion the fast one, because the protocol already knows which replicas are eligible. There is no agonizing synchronization decision because the log-matching property of Raft means followers are continuously kept consistent, not bulk-synced on demand.

### Why the log-matching property removes synchronization pain

The single most operationally pleasant thing about quorum queues compared to mirroring is that the dreaded on-demand synchronization simply does not exist in the same form, and it is worth understanding why. Raft maintains an invariant called the **log-matching property**: if two replicas' logs contain an entry with the same index and the same term (Raft's logical clock for leadership epochs), then their logs are *identical* up to that point. The leader enforces this by including, with each new entry it replicates, the index and term of the entry immediately preceding it; a follower refuses an entry whose predecessor does not match its own log, which forces the leader to walk backward and resend until it finds the point where the logs agree, then bring the follower forward from there. The consequence is that followers are *continuously* converging on the leader's log as a normal part of operation — there is no separate "sync this queue now" command that blocks the world. A follower that fell behind, or a node that just rejoined the cluster, catches up by the ordinary replication path, incrementally, without freezing the queue. The agonizing mirrored-queue choice between a blocking full sync and a lossy stale promotion is gone because Raft never lets a follower's log silently diverge in the first place; keeping replicas consistent is the steady-state behavior, not an exceptional operation you have to trigger and dread.

There is a real cost hidden in this, which is that a quorum queue is not built for very deep backlogs. Because the entire queue is a replicated Raft log and entries are retained until acknowledged, a quorum queue holding millions of unacknowledged messages consumes proportional memory and replication overhead across every replica. Quorum queues are tuned for queues that stay relatively shallow — work that is produced and consumed at comparable rates — not for using the queue as a long-term buffer of tens of millions of messages. If your design treats the queue as a deep durable store that consumers drain slowly over hours, that is again a sign you want a log like Kafka rather than a quorum queue; the replicated-log-as-a-queue model is excellent for in-flight work and poor as a bulk reservoir.

### The cost: throughput and node count

Quorum queues are not free. The majority-commit requirement means every write waits for a network round trip to at least one follower plus that follower's disk write before it is confirmed — you have traded the single-node fsync latency for a fsync-plus-network-majority latency. A quorum queue's per-message confirm latency is therefore *higher* than a classic non-replicated queue's, and its peak throughput is *lower*, because each message does more work across more machines. You also pay in storage: three replicas means three copies of every message on disk. And quorum queues require a properly sized cluster — you need at least three nodes to tolerate one failure, and the cluster should be an odd number to avoid the ambiguity of an even split.

![A before-and-after comparison contrasting a mirrored queue suffering sync lag and split-brain message loss on a mid-sync master failure against a quorum queue using a Raft log with majority commit that elects a new leader with no loss and no split-brain](/imgs/blogs/rabbitmq-acks-confirms-durability-quorum-queues-8.webp)

The before-and-after figure is the whole argument in one frame. On the left, the mirrored queue: asynchronous mirror sync that lags, a master that fails mid-sync, and the split-brain that loses messages when the partition heals. On the right, the quorum queue: a Raft log with majority commit, a leader failure that triggers a clean follower election, and the outcome that matters — no loss, no split-brain. You pay for that with throughput and node count, and for the overwhelming majority of workloads that is a trade worth making, which is exactly why the RabbitMQ team made quorum queues the default recommendation for any queue that needs high availability and deprecated mirroring outright. The detailed cluster-sizing and capacity-planning side of running quorum queues at scale is covered in the companion [RabbitMQ production architecture and scaling](/blog/software-development/system-design/rabbitmq-production-architecture-scaling) post; here the point is narrower and sharper — quorum queues replaced mirrored queues because Raft makes safety a theorem instead of a hope.

### The replication trade in one table

| Property | Classic mirrored queue | Quorum queue (Raft) |
| --- | --- | --- |
| Replication model | Async master-to-mirror | Raft majority commit |
| Commit rule | Master decides, mirrors follow | Majority of replicas must persist |
| Failover | Slow safe sync, or fast lossy promote | Fast leader election, no committed loss |
| Split-brain | Possible; loses data on heal | Structurally impossible |
| Throughput | Higher peak (one busy master) | Lower, steadier |
| Status | Deprecated | Default for HA |

![A decision matrix comparing classic mirrored queues and quorum queues across replication model, failover behavior, data safety, and throughput, showing quorum queues winning on safety and failover while mirroring wins only on raw peak throughput](/imgs/blogs/rabbitmq-acks-confirms-durability-quorum-queues-6.webp)

The matrix in the figure summarizes the comparison as a decision table: across replication, failover, and data safety, quorum queues win decisively; the *only* column where classic mirroring comes out ahead is raw peak throughput, because a single unreplicated-style master with no majority round trip can push more messages per second. For a system whose entire reason to use replication is to *not lose data*, winning on safety and losing a bit on peak throughput is exactly the trade you want. If you genuinely need maximum throughput and can tolerate single-node durability, you would not use a replicated queue at all; you would use a plain durable classic queue on one node and accept its failure domain. The choice between mirrored and quorum is, in 2026, not really a choice — it is quorum, and mirrored exists only to be migrated away from.

## 9. Putting it together: a no-loss configuration

Now assemble every link into one concrete configuration and then prove it by tracing a message through it and asking, at each hop, "what happens if the system crashes right here?"

A no-loss RabbitMQ setup needs, on the publisher side, **publisher confirms** (`confirm.select`) so the producer knows the broker took responsibility, and **`mandatory=True`** so an unroutable message comes back instead of vanishing. On the broker side it needs **durable queues** so queue definitions survive restarts, **persistent messages** (`delivery_mode=2`) so message bytes are written to disk, **quorum queues** so a node failure does not lose committed messages, and a **dead-letter exchange** so rejected messages are parked rather than discarded. On the consumer side it needs **manual acks** (`auto_ack=False`) with the ack placed *after* the work completes, a sane **prefetch** to bound the unacked set, and a **requeue discipline** that distinguishes transient from permanent failures. That is the full chain, every link present.

```python
# Publisher side: confirms + mandatory + persistent + quorum durable queue.
channel.confirm_delivery()
channel.queue_declare(
    queue="payments",
    durable=True,
    arguments={
        "x-queue-type": "quorum",                 # Raft-replicated quorum queue
        "x-dead-letter-exchange": "payments.dlx", # rejected msgs are parked, not lost
    },
)
channel.basic_publish(
    exchange="",
    routing_key="payments",
    body=payload,
    properties=pika.BasicProperties(delivery_mode=2),  # persistent
    mandatory=True,                                      # unroutable -> returned
)

# Consumer side: manual ack AFTER the work, bounded prefetch, requeue discipline.
channel.basic_qos(prefetch_count=30)

def handle(ch, method, properties, body):
    try:
        do_work(body)
        ch.basic_ack(method.delivery_tag)              # ack only after success
    except TransientError:
        ch.basic_nack(method.delivery_tag, requeue=True)   # retry transient failures
    except PermanentError:
        ch.basic_nack(method.delivery_tag, requeue=False)  # park poison in the DLX
```

That single block is the entire reliability chain expressed as code: confirm mode, a durable quorum queue with a DLX, persistent and mandatory publishes, bounded-prefetch manual-ack consumption, and a requeue policy that tells transient and permanent failures apart. Everything in this post exists to justify each line of it.

#### Worked example: tracing a publish and finding every loss point

Let me trace one payment message through that setup and, at each hop, name the crash that *would* lose it without the corresponding setting — so you can see what each knob is buying.

The publisher serializes a `\$50` payment and calls `basic_publish` with `delivery_mode=2`, `mandatory=True`, on a confirm-mode channel.

- **Crash point A — before the broker receives it.** The network drops the publish in flight. Because the channel is in confirm mode, the publisher never receives a confirm, times out, and republishes. *Without confirms*, the publish call would have returned successfully and the message would be silently gone. Confirms saved it.
- **Crash point B — broker has it but cannot route it.** Suppose a binding is missing. With `mandatory=True`, the broker returns the message and the publisher logs an unroutable error and alerts. *Without mandatory*, the broker discards it and — cruelly — still sends a positive confirm, so the producer believes it succeeded. Mandatory saved it.
- **Crash point C — message routed, broker crashes before writing to disk.** The message is `delivery_mode=2` going into a durable quorum queue. On a quorum queue the confirm is withheld until a majority of replicas have the entry committed to their Raft logs. If the leader crashes after routing but before committing, the publisher gets no confirm and republishes. *Without persistence or durability*, even a committed-looking message would be memory-only and lost on restart. Persistence plus durability plus quorum commit saved it.
- **Crash point D — one whole node dies after commit.** The message is committed to two of three replicas. The dead node is the leader. A follower with the committed entry is elected leader in seconds; the message is intact. *With a classic mirrored queue*, a mid-sync failure here could have lost it or split-brained; *with a single non-replicated queue*, the dead node's disk takes the message with it. Quorum replication saved it.
- **Crash point E — delivered to a consumer that crashes mid-processing.** The consumer is in manual-ack mode and crashes after charging the card but before acking. The broker sees the channel drop, requeues the unacked message, and redelivers it with `redelivered=true`. The card gets charged *twice* unless the consumer is idempotent — which is why the chain does not end at delivery and why [idempotent consumers](/blog/software-development/message-queue/idempotent-consumers-and-deduplication) are the necessary companion to at-least-once delivery. *With auto-ack*, the message would have been dropped from the queue on delivery and the crash would have lost the work entirely. Manual acks saved it from loss — and handed you the duplicate problem to solve separately.

Five crash points, five settings, and every one of them is a link you can individually forget. The `\$50` survives precisely because none of them were forgotten. This trace is the whole post in one example: reliability is not a feature you turn on, it is a chain you keep unbroken.

#### Worked example: quorum-queue sizing for a 3-node cluster

Now the capacity side. Suppose a three-node cluster running a single hot quorum queue, and let us reason about both survival and cost with concrete numbers.

**Survival.** Majority of three is two. The queue tolerates the loss of **one** node and stays fully available: with one node down, two remain, two is a majority, writes still commit and a leader can still be elected. Lose a **second** node and you are down to one of three — no majority, so the queue goes read-unavailable for writes; it will not accept publishes until a second node returns, *and crucially it has not lost any committed data*, it has merely paused. That pause-rather-than-lose behavior is the whole point of quorum: it would rather stop than risk split-brain. If you need to tolerate two simultaneous node failures, you must size up to a **five**-node quorum queue, where the majority is three and you can lose two and keep going.

**Throughput cost.** Put numbers on it. Say a plain single-node durable classic queue confirms a persistent publish in roughly **2 ms** at the median — one local fsync-batched write. The quorum queue must, before confirming, replicate to a majority: the leader's local write plus at least one follower's network round trip and that follower's own write. If the intra-cluster round trip is **1 ms** and the follower's batched fsync adds another **1 ms**, the quorum confirm lands around **4 ms** — call it roughly *double* the single-node latency. Because confirm latency bounds how many in-flight publishes you need to sustain a given rate (by Little's law, in-flight = rate × latency), to hold the same publish throughput you must keep about *twice* as many publishes outstanding, which means a more aggressive asynchronous-confirm pipeline. If you have, say, 100 MB of memory budget for outstanding unconfirmed publishes and messages average 2 KB, you can hold ~50,000 in flight; at 4 ms confirm latency that supports on the order of 50,000 / 0.004 = **12.5 million publishes per second** of *pipeline* headroom — far above what a single queue actually does, which tells you the confirm latency is rarely the real ceiling; the leader node's CPU and the replication bandwidth are. The honest summary: a quorum queue costs you roughly 2x per-message confirm latency and 3x storage versus a single durable classic queue, in exchange for surviving a node death with zero committed loss. For anything carrying payments, that is not a close call.

## Case studies and war stories

**The silent restart loss.** A team ran a classic RabbitMQ setup for a year with what they believed was durable messaging. They had marked all their messages `delivery_mode=2`. What they had *not* done was declare their queues `durable=True` — the queues were transient. Everything worked perfectly because the broker never restarted. Then a routine kernel upgrade rebooted the broker node, RabbitMQ restarted, and every queue came back empty: the transient queue definitions had evaporated and taken their persistent messages with them. Roughly forty thousand in-flight messages were lost. The lesson is the exact trap from section 5: persistence without durability is not durability. Both flags or neither.

**The redelivery storm.** A payments consumer caught all exceptions and nacked with `requeue=True` "to be safe." A single message referenced an account that had been deleted, so every attempt to process it threw a not-found error, and every throw triggered a requeue. The poison message looped through the consumer thousands of times per second, pinned a CPU core, flooded the logs at gigabytes per hour, and because it kept returning to the front of the queue, starved legitimate payments behind it. The on-call engineer's dashboards showed a queue that was simultaneously "not empty" and "not draining," which is the signature of a poison loop. The fix was two lines: distinguish permanent failures and nack them with `requeue=False` into a dead-letter exchange. The deeper lesson is that `requeue=True` is not the safe default it looks like — it is the default that turns one bad message into an outage.

**The mirrored-queue split-brain.** A team running mirrored queues across two data centers hit a network partition between the sites. Each side's `pause_minority` configuration was meant to prevent split-brain, but their cluster was split exactly in half — no minority to pause — so both halves kept accepting traffic against their own copy of the queue master. When the link healed, RabbitMQ reconciled by discarding one side's divergent messages, silently losing every order that side had accepted during the partition. This is the textbook mirrored-queue failure and the textbook motivation for quorum queues: with a quorum queue across an *odd* number of nodes split between sites, one side always lacks a majority and *correctly refuses to make progress* rather than diverging. They migrated to quorum queues with a three-node layout, accepting that a full-site partition makes the minority side read-only — which is precisely the safe behavior they wanted all along.

**The fsync window power loss.** A single-node broker carrying "durable persistent" messages lost power from a failed PSU. On reboot, the team discovered a few hundred messages that had been confirmed to publishers in the last fraction of a second before the outage were gone. They had been written to the page cache but were inside the fsync batching window when the power died — never flushed to the platter. Nothing was misconfigured; this is the inherent limit of single-node durability from section 5. The fix was not a flag but an architecture change: move the queue to a quorum queue so a committed message exists on a majority of nodes' disks, making a single machine's power loss survivable. Durability protects against a restart; only replication protects against a hardware death.

## When to reach for this (and when not to)

Turn on the **full reliability chain** — confirms, durable quorum queues, persistent messages, manual acks, a DLX — for anything where losing a message has a business cost you would have to explain to someone: payments, orders, account changes, anything that triggers an irreversible side effect or that a customer is waiting on. The latency and throughput cost of the full chain is real but modest, and it is dwarfed by the cost of explaining lost data.

Turn the chain *down* deliberately where loss is genuinely cheap. For high-volume telemetry, metrics samples, cache-warming hints, or any stream where an individual missing message is invisible, auto-ack and transient non-persistent messages are not sloppiness — they are the right engineering trade, because you are buying throughput with a guarantee you do not need. The mistake is not using weak guarantees; the mistake is using weak guarantees *by accident* on data that needed strong ones.

Reach for **quorum queues** whenever a queue needs to survive a node failure, which in a clustered deployment is essentially every queue that matters. Reach for **classic durable queues** (single-node, no replication) only when you have a genuinely throughput-bound queue whose data you can afford to lose with its node — a niche, but a real one. **Never** reach for mirrored queues in a new system; they are deprecated, and quorum queues do their job strictly better for almost every workload.

And reach for a different broker entirely when RabbitMQ's per-message model is the wrong shape for your problem. If you need to replay a stream from an arbitrary point in the past, or you need millions of messages per second through a single logical topic, or you need consumers to independently track their own position in a retained log, you want a log, not a queue — that is [Kafka's distributed log](/blog/software-development/database/kafka-as-a-distributed-log) territory, and the [outbox pattern with change data capture](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) is how you bridge a database's truth into that log reliably. RabbitMQ's reliability machinery is excellent at making *individual messages* durable and acknowledged; it is not trying to be a replayable event store, and stretching it into that role is how teams end up unhappy.

## Key takeaways

- **Reliability is a chain, not a feature.** A message is safe only when the publisher confirms, the broker persists and replicates, and the consumer acks — break any link and the guarantee is gone, no matter how strong the others are.
- **Manual acks after the work, always.** Auto-ack is at-most-once and loses in-flight work on every crash. Put `basic.ack` *after* processing, and size prefetch to bound the unacked set and the crash blast radius.
- **`requeue=True` is a trap for content failures.** Requeue transient failures; dead-letter permanent ones with `requeue=False`. Blindly requeuing poison messages causes redelivery storms that starve healthy traffic.
- **Publisher confirms, not transactions.** Confirms give you per-publish broker acknowledgement asynchronously without the order-of-magnitude throughput collapse of AMQP transactions. Transactions are a rarely-needed atomic-batch tool.
- **Durability needs BOTH flags.** A message survives a restart only if the queue is `durable` AND the message is `persistent` (`delivery_mode=2`). Either one alone loses everything on restart.
- **Persistent is not the same as fsynced.** Writes batch into the OS page cache and fsync on a timer; a power loss inside that window loses confirmed messages on a single node. Only replication closes that gap.
- **Set `mandatory=True` and handle returns.** An unroutable message is discarded silently and still gets a positive confirm. Mandatory returns turn a vanishing-message routing bug into a loud error.
- **Quorum queues replaced mirrored queues for a reason.** Raft majority commit makes split-brain structurally impossible and failover lossless, at the cost of ~2x latency and 3x storage. Mirroring is deprecated; do not start new systems on it.
- **Size quorum clusters for the failures you must survive.** Three nodes tolerate one loss; five tolerate two. A quorum queue pauses rather than diverges when it lacks a majority — that is the safety, not a bug.
- **At-least-once hands you duplicates.** Manual acks plus requeue mean a message can be delivered twice; pair the reliability chain with idempotent consumers so redelivery is harmless.

## Further reading

- [RabbitMQ Deep Dive, Part 1: AMQP, exchanges, bindings, and routing](/blog/software-development/message-queue/rabbitmq-amqp-exchanges-bindings-routing) — where the message gets to the queue in the first place.
- [RabbitMQ production architecture and scaling](/blog/software-development/system-design/rabbitmq-production-architecture-scaling) — cluster sizing, capacity planning, and operating quorum queues at scale.
- [Delivery semantics: at-most-once, at-least-once, exactly-once](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) — the taxonomy these mechanisms implement.
- [Push vs pull, acknowledgements, and how consumers read](/blog/software-development/message-queue/push-vs-pull-acknowledgements-how-consumers-read) — prefetch and the unacked set from the flow-control angle.
- [Dead-letter queues and poison-message handling](/blog/software-development/message-queue/dead-letter-queues-and-poison-message-handling) — draining a DLQ and building retry-counting layers.
- [Idempotent consumers and deduplication](/blog/software-development/message-queue/idempotent-consumers-and-deduplication) — making at-least-once redelivery harmless.
- [Distributed replication: leader, multi-leader, leaderless](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless) — the replication theory underneath quorum queues.
- The official RabbitMQ documentation on Publisher Confirms, Consumer Acknowledgements, and Quorum Queues — the authoritative reference for every flag named here.
