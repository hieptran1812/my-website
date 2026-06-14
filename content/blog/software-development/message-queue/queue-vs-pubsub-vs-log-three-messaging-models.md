---
title: "Queue vs Pub/Sub vs Log: Three Messaging Models and When to Use Each"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Learn the single most clarifying distinction in messaging: the three fundamental models — point-to-point queue, pub/sub broadcast, and append-only log — how each handles retention, fan-out, replay, ordering, and coupling, why a log can emulate the other two, and a hard decision rule for picking one."
tags:
  [
    "message-queue",
    "pub-sub",
    "event-log",
    "kafka",
    "rabbitmq",
    "distributed-systems",
    "event-driven",
    "stream-processing",
    "system-design",
    "messaging-patterns",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/queue-vs-pubsub-vs-log-three-messaging-models-1.webp"
---

Here is a question I have asked dozens of engineers in design reviews, and almost none of them answer it cleanly: "When you publish a message, who reads it, and is it still there afterward?" The reason the question lands awkwardly is that "message queue" is a single phrase covering three genuinely different machines. People say "let's put it on a queue" and mean a work distributor. Other people say "let's publish an event" and mean a broadcast. Other people say "let's stream it through Kafka" and mean an append-only ledger that nobody can delete out from under them. These are not three flavors of the same thing. They are three distinct contracts about who reads a message, how many copies get delivered, and whether the message survives being read.

Get this distinction wrong and you will build the wrong system. I have watched a team try to use a classic competing-consumers queue as an event bus, then discover that adding a second consumer to "also listen" silently stole half the messages from the first consumer, because in a queue each message goes to exactly one reader. I have watched another team use a fanout broadcast where they needed durable work distribution, then lose a day of jobs because nobody was subscribed at the instant the messages were published. And I have watched a third team reach for a heavyweight log when all they needed was a simple task buffer, paying for partitions and offset management and a Kafka cluster to run a nightly email job. Each of those failures is the same root error: confusing the three messaging models.

This post is about that one distinction, because it is the most clarifying idea in the entire field. The three models are the point-to-point **queue**, the **pub/sub** broadcast, and the append-only **log**. By the end of this post you will be able to look at any messaging requirement and immediately know which model it wants; you will understand why a log can *emulate* both a queue and pub/sub, which is exactly why log-based systems came to dominate modern data infrastructure; and you will have a blunt decision rule you can apply in a design review without hand-waving. The figure below is the whole post compressed into one table: the three models down the columns, and the four properties that actually distinguish them — retention, fan-out, replay, and ordering — down the rows.

![A comparison matrix with rows for retention, fan-out, replay, and ordering and columns for queue, pub/sub, and log, showing the log as the only model that retains and replays](/imgs/blogs/queue-vs-pubsub-vs-log-three-messaging-models-1.webp)

We will build the argument from the bottom up. We will take each model in turn — its delivery contract, its retention behavior, its failure modes, the real brokers that implement it — and then arrive at the central realization: the log is not a third sibling of the other two. It is a more general substrate, and the queue and pub/sub are both *views* you can take on a log. Once you internalize that, a large chunk of modern architecture (event sourcing, CQRS, stream processing, change data capture) stops looking like a pile of separate techniques and starts looking like obvious consequences of "keep the log, read it many ways."

This post is part of a forty-part message-queue series. It sits right after the series opener on [why asynchronous messaging exists at all](/blog/software-development/message-queue/message-queues-async-decoupling-and-load-leveling) and the [anatomy of a message system](/blog/software-development/message-queue/anatomy-of-a-message-system-producers-brokers-consumers). It complements the existing deep-dive on [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) and the [RabbitMQ production architecture](/blog/software-development/system-design/rabbitmq-production-architecture-scaling) post — this one is the conceptual map those two live inside. Where they go deep on one broker, this one draws the territory.

## 1. Three models, one diagram

Strip a messaging system down to its essence and you are left with a producer that emits messages and one or more consumers that read them. The only interesting questions are about the contract in the middle. There are exactly three answers that matter, and they correspond to the three models.

In a **point-to-point queue**, a message is delivered to *exactly one* consumer, and once that consumer acknowledges it, the message is *gone*. Multiple consumers attached to the same queue compete for messages; the broker hands each message to whichever consumer is free. This is work distribution. Think of a printer queue or a job queue: you have a backlog of tasks and a pool of workers, and you want each task done once, by somebody, soon.

In **pub/sub**, a message is delivered to *every* subscriber. One publish, many copies. Subscribers are usually independent of each other, and in the classic form they only receive messages published *while they are subscribed* — a subscription is an ephemeral interest, not a durable mailbox. This is broadcast. A stock-price tick goes to every dashboard currently watching it; a cache-invalidation event goes to every server holding the cache.

In a **log**, messages are *appended* to an ordered, immutable sequence and *retained* for a configured period regardless of who has read them. Reading does not remove anything; it merely advances a cursor — an *offset* — that the reader controls. Many independent consumer groups can each read the whole log at their own pace, each tracking their own offset, and a brand-new reader can start from the beginning of history and replay everything. This is a durable, replayable ledger.

The single sentence that captures all three: **a queue deletes on read and delivers to one; pub/sub copies to all but keeps nothing; a log keeps everything and lets anyone read it any number of times.** Everything else — the brokers, the configs, the war stories — is detail hung on that frame. The four properties in the figure above are the dimensions along which these three differ, and they are the four questions to ask of any messaging requirement: how long is the message kept, how many readers see each message, can history be replayed, and what ordering is guaranteed.

### Why this is the most important distinction

Most messaging confusion in the wild traces back to applying one model's intuitions to another model's system. Someone who learned messaging on RabbitMQ queues expects "consume" to mean "remove," and is baffled by Kafka, where messages stay put and "consuming" just moves an offset. Someone who learned on Kafka expects to add a new reader freely without disturbing existing ones, and is baffled when adding a second consumer to a RabbitMQ queue *steals* messages from the first instead of duplicating them. Neither person is wrong about their system; they are wrong about which model they are standing in.

So before any broker comparison, before any throughput benchmark, before any discussion of acks and retries, you ask: queue, pub/sub, or log? That question alone resolves most architecture arguments, because the three models have fundamentally different answers to "what happens to a message after it is read" and "how many readers does each message reach." Let us take them one at a time.

## 2. The point-to-point queue: competing consumers, consume-and-delete

The queue is the oldest and most intuitive of the three. A producer puts a message in; a consumer takes it out; the message is then gone. If you have ever used a to-do list where you cross off and erase each item as you finish it, you already understand the queue. Its defining property is **consume-and-delete**: a successfully processed message is removed from the queue, so it can never be processed again.

The reason the queue is so powerful for work distribution is the **competing consumers** pattern. You attach several consumers to the same queue, and the broker load-balances messages across them. Each message goes to exactly one consumer. If you have a thousand jobs and ten workers, each worker pulls jobs as fast as it can finish the previous one, and the queue naturally balances the load — a fast worker that finishes quickly simply pulls more, a slow or busy worker pulls fewer. You scale throughput by adding workers, with no coordination between them and no central scheduler.

![A queue diagram showing one producer feeding a single FIFO buffer that dispatches each message to exactly one of three competing workers, with the message deleted after the worker acknowledges](/imgs/blogs/queue-vs-pubsub-vs-log-three-messaging-models-2.webp)

The figure above shows the shape. One producer emits at roughly five thousand messages per second into a queue. Three workers are attached. A given message goes to exactly one of them — in the diagram, worker 2 picks it up while workers 1 and 3 are idle and available for the next message. When worker 2 finishes and acknowledges, the message is deleted from the queue. This is the way the queue works: dispatch to one, process, acknowledge, delete.

### The acknowledgment dance

The "delete on ack" part is more subtle than it sounds, and it is where queue semantics get interesting. A naive queue might delete a message the instant it hands it to a consumer. But what if that consumer crashes mid-processing? The message would be lost. So real queues use an acknowledgment protocol: the broker delivers the message but keeps a copy (marked "in flight" or "unacknowledged"), and only deletes it after the consumer explicitly acks. If the consumer crashes before acking, the broker eventually times out the in-flight message and redelivers it to another consumer.

This is the foundation of **at-least-once delivery**: a message is held until somebody confirms they processed it, which means under failure it may be delivered more than once (the original consumer might have finished the work but crashed before the ack reached the broker). The full treatment of delivery semantics lives in the sibling post on [delivery semantics: at-most, at-least, and exactly once](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once); the key point here is that the queue's consume-and-delete contract is built on this ack protocol, and the protocol is what makes the queue durable against worker crashes.

Here is a minimal RabbitMQ consumer in Python showing the ack dance explicitly:

```python
import pika

connection = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
channel = connection.channel()

# Durable queue survives a broker restart; messages persist to disk.
channel.queue_declare(queue="jobs", durable=True)

# Fair dispatch: do not give a worker a new message until it acks the
# previous one. Without this, a fast-but-busy worker gets starved while
# a slow worker hoards a backlog of prefetched messages.
channel.basic_qos(prefetch_count=1)

def handle(ch, method, properties, body):
    try:
        process(body)                       # do the actual work
        ch.basic_ack(method.delivery_tag)   # delete the message
    except Exception:
        # Negative-ack and requeue so another worker retries it.
        ch.basic_nack(method.delivery_tag, requeue=True)

channel.basic_consume(queue="jobs", on_message_callback=handle)
channel.start_consuming()
```

Two configuration choices in that snippet carry the whole weight of queue semantics. The `prefetch_count=1` (called QoS or prefetch) controls how many unacknowledged messages the broker will push to one consumer before waiting — it is the knob that makes competing consumers actually balance work rather than letting one greedy consumer buffer the whole backlog. And the `basic_ack` / `basic_nack` pair is the consume-and-delete contract made literal: ack deletes, nack-with-requeue puts it back for someone else to try.

### What a queue cannot do

The queue's strength — each message to exactly one consumer — is also its hard limitation. **You cannot broadcast with a plain queue.** If you attach a second, logically-different consumer to the same queue hoping it will *also* see every message, it will instead *compete* with the first and steal half the messages. This is the single most common queue mistake. If you need two different services to each react to every order — say, an email service and an analytics service — a single queue will not do it. You need either two queues (and the producer publishes to both, or a router copies into both), or you need the pub/sub model.

The second thing a queue cannot do is **replay**. Once a message is acked and deleted, it is gone. There is no rewind, no "let me reprocess yesterday's orders because I found a bug in my consumer." The history simply does not exist anymore. For a job queue this is usually fine — you do not want to reprint yesterday's print jobs — but for an event stream it is often catastrophic, and it is the property that pushes teams toward the log.

The third limitation is **ordering under concurrency**. A single-consumer queue preserves FIFO order. But the moment you add competing consumers for throughput, strict order across the whole queue is lost: worker 1 might finish message 5 before worker 2 finishes message 3. You get rough order, not strict order. If you need strict ordering *and* parallelism, the queue alone cannot give it to you — you need per-key partitioning, which is a log concept. We will return to this.

## 3. Pub/sub: fan-out to every subscriber

Pub/sub flips the queue's delivery contract. Instead of one message to one consumer, pub/sub delivers **one message to every subscriber**. The producer (here called a *publisher*) emits a message; the broker copies it to each registered subscriber. There is no competition; everybody gets their own copy. This is broadcast, and it is the natural model whenever an event is *interesting to many independent parties*.

The canonical example is a domain event. An "order placed" event is interesting to the email service (send a confirmation), the analytics service (update the dashboard), the fraud service (score the transaction), the inventory service (decrement stock), and probably three more teams you have not met yet. With pub/sub, the order service publishes one event and walks away; the broker fans it out to all five subscribers. The publisher does not know or care who is subscribed — that is the decoupling pub/sub buys you. New subscribers can appear without the publisher changing a line of code.

![A pub/sub diagram showing one publisher emitting an order-placed event into a fanout exchange that copies it to email, analytics, and fraud services, with a fourth subscriber receiving nothing because it was offline](/imgs/blogs/queue-vs-pubsub-vs-log-three-messaging-models-3.webp)

The figure above shows the fan-out. One publisher emits `order.placed` into a fanout exchange. The exchange copies the message — copy A to email, copy B to analytics, copy C to fraud. Crucially, look at the fourth path: a subscriber that is *offline at publish time* gets nothing. In classic, ephemeral pub/sub, the message is delivered only to subscribers present at the moment of publication. If your fraud service is restarting when the event fires, it simply misses that event. There is no mailbox holding it.

### Ephemeral subscriptions are the catch

The defining and most dangerous property of classic pub/sub is that **subscriptions are often ephemeral**. The subscriber expresses an interest *now*, and receives messages *from now on, while connected*. Disconnect and you stop receiving; reconnect and you resume from the present, having missed everything in between. Redis pub/sub is the purest example: it does no buffering at all. If you `PUBLISH` to a channel with no subscribers, the message evaporates instantly — Redis does not even know it was supposed to go anywhere. If a subscriber's network blips for two seconds, those two seconds of messages are gone forever.

Here is Redis pub/sub showing exactly that fire-and-forget behavior:

```python
import redis

r = redis.Redis()

# Publisher side: if zero subscribers are listening RIGHT NOW,
# the return value is 0 and the message is simply dropped.
delivered = r.publish("prices", "AAPL 187.42")
print(f"delivered to {delivered} subscribers")   # 0 means: gone

# Subscriber side, in another process:
pubsub = r.pubsub()
pubsub.subscribe("prices")
for message in pubsub.listen():
    # Only sees messages published while this loop is running and
    # the connection is healthy. A reconnect resumes from "now",
    # not from where it left off. There is no offset, no replay.
    if message["type"] == "message":
        update_dashboard(message["data"])
```

This ephemerality is a feature for the right use case and a footgun for the wrong one. For live dashboards, presence systems, and cache invalidation, ephemeral pub/sub is perfect — you only care about *current* state, and a missed message just means a slightly stale view for a moment until the next update arrives. But for anything where *every* event must be processed — orders, payments, audit records — ephemeral pub/sub will silently drop messages and you will not even get an error. It is the messaging equivalent of shouting into a room and assuming everyone who needed to hear you was present and listening.

### Durable pub/sub: adding a mailbox per subscriber

The fix for ephemerality is **durable subscriptions**: each subscriber gets its own persistent queue, and the broker copies every published message into every subscriber's queue, where it waits until that subscriber consumes it. This is how RabbitMQ does pub/sub — a fanout or topic exchange copies each message into every *bound queue*, and each of those queues is then a durable, competing-consumers queue for that one subscriber. It is also how AWS does it: SNS (the pub/sub fan-out) typically delivers into SQS queues (the durable per-subscriber mailboxes), the famous "SNS-to-SQS fan-out" pattern.

Notice what durable pub/sub really is: **pub/sub composed with queues.** The fan-out copies the message N ways; each copy lands in a durable queue; each queue is then consumed with normal competing-consumers semantics. This composition is your first hint of the deeper truth: these models are not isolated. You can build one out of the others. Pub/sub-with-durable-subscriptions is fan-out plus N queues. Hold that thought, because the log takes it much further.

### Routing: topics and filters

Most pub/sub systems add a routing layer so subscribers can express interest in *some* messages, not all. A topic exchange in RabbitMQ routes by pattern matching on a routing key (`orders.us.*` matches `orders.us.california` but not `orders.eu.berlin`). SNS message filtering lets a subscription accept only messages whose attributes match a filter policy. This routing turns pub/sub from a blunt broadcast into a content-based delivery network: the publisher emits richly-keyed messages, and each subscriber declares which slice it wants. The fan-out still happens — every matching subscriber gets a copy — but the matching can be narrow.

The mental model stays the same: **one publish, a copy to every interested subscriber, nothing retained for absent ones** (unless you added durable per-subscriber queues). Routing just narrows "every subscriber" to "every matching subscriber." The retention and replay story is unchanged: classic pub/sub does not let a new subscriber go back and see what it missed before it subscribed. For that, you need the log.

## 4. The log: append-only, retained, replayable

The log is the youngest of the three models as a *first-class product* (Kafka popularized it around 2011) but the oldest as an *idea* — it is just a write-ahead log, the same structure databases have used internally for decades, exposed as the primary interface instead of hidden inside the storage engine. The log's contract is radically different from the other two, and understanding that difference is the heart of this post.

A log is an **append-only, ordered, immutable sequence of records**. Producers only ever *append* to the end; they never modify or delete existing records. Each record gets a monotonically increasing **offset** — record 0, record 1, record 2, and so on. The log is **retained** for a configured time or size (seven days is a common default; some keep months or forever), and — this is the crucial part — **reading does not remove anything**. A consumer reads record N, then record N+1, and all the broker does is remember "this consumer has read up to offset N." The records themselves stay exactly where they are. Reading is just a cursor advancing over an immutable tape.

![A pipeline showing a producer appending to an offset-numbered log while two consumer groups read the same records at different offsets, one near the live tail and one far behind, with records retained for seven days](/imgs/blogs/queue-vs-pubsub-vs-log-three-messaging-models-4.webp)

The figure above shows the two properties that make the log special. A producer appends to a log holding offsets 0 through 9999. Two *independent consumer groups* read the same log. Group A is caught up, reading near the live tail at offset 9990. Group B is far behind, processing at offset 4200 — maybe it is a slower batch job, maybe it was down for an hour and is catching up. **Neither group affects the other**, because each tracks its own offset, and the records are retained regardless. The same physical records serve both readers. This is the log's superpower: many independent readers, each at their own position, over one retained history.

### Offsets are owned by the consumer, not the broker

In a queue, the broker owns the message state — it knows what is in flight, what is acked, what is deleted. In a log, the broker owns only the records (which it never deletes until retention expires); the *read position* is owned by the consumer. This inversion is everything. Because the consumer owns its offset, the consumer can:

- **Rewind**: seek back to an earlier offset and reprocess. Found a bug in yesterday's logic? Set your offset back to where yesterday started and replay.
- **Fast-forward**: seek to the latest offset and skip a backlog. Recovering from a long outage and you only care about *now*? Jump to the tail.
- **Restart from anywhere**: a brand-new consumer group can start at offset 0 (the beginning of retained history) or at the latest offset (only new messages) or at a specific timestamp.

Here is the Kafka consumer config that exposes these choices. Compare it to the RabbitMQ consumer from earlier — notice there is no "ack-to-delete," only offset management:

```python
from kafka import KafkaConsumer, TopicPartition

consumer = KafkaConsumer(
    bootstrap_servers="localhost:9092",
    group_id="analytics-v2",          # the consumer GROUP, not the consumer
    enable_auto_commit=False,         # we commit offsets explicitly
    # Where to start if this group has NO committed offset yet:
    #   "earliest" = offset 0, replay all retained history
    #   "latest"   = the tail, only brand-new messages
    auto_offset_reset="earliest",
)

tp = TopicPartition("orders", 0)
consumer.assign([tp])

# Replay: explicitly rewind this group to the very beginning.
consumer.seek_to_beginning(tp)

for record in consumer:
    process(record.value)             # reading does NOT delete the record
    # Commit our offset so a restart resumes here. The RECORD stays in
    # the log either way; we are only saving our cursor position.
    consumer.commit()
```

The line `auto_offset_reset="earliest"` plus `seek_to_beginning` is the replay button. There is no equivalent in a queue, because in a queue the records you would replay no longer exist. This is not a minor convenience — it is a categorically different capability that reshapes how you build systems, which is why the next section is devoted entirely to retention and replay.

### Partitions: how a log scales and orders

A single append-only log is inherently serial — one writer appending to one tail. To scale beyond one machine's write throughput, logs are split into **partitions**: the topic is divided into P independent logs, each with its own offset sequence, each living on a (possibly different) broker. A producer assigns each message to a partition, usually by hashing a key (all events for `order-12345` go to the same partition). This gives you two things at once: **parallelism** (P partitions can be written and read concurrently) and **per-key ordering** (all messages with the same key land in the same partition, and within a partition order is strict). You lose *global* order across partitions but keep *per-key* order, which is almost always what you actually need. Partitioning is deep enough to deserve its own treatment, and it gets one in the sibling post on partitioning and ordering — here it is enough to know that partitions are how a log turns a serial structure into a scalable, ordered one. The mechanics overlap heavily with [database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding), which is no accident: a partitioned log *is* a sharded write-ahead log.

### Time-based retention vs log compaction

There is a second flavor of retention worth knowing, because it changes what the log *means*. The default is **time-based (or size-based) retention**: keep every record for seven days (or until the partition reaches some size), then delete the oldest. This treats the log as a *window onto recent history* — a rolling buffer of events. It is what you want for an event stream where old events stop mattering: a `page.viewed` event from three weeks ago has no bearing on anything today.

The alternative is **log compaction**, where instead of deleting by age the broker keeps, for each *key*, at least the most recent record, and garbage-collects superseded older records for that key. A compacted log is no longer a window onto recent history — it is a **complete, latest-value snapshot keyed by message key**, retained indefinitely. If you publish `user-123 → {plan: pro}` and later `user-123 → {plan: enterprise}`, compaction eventually discards the first and keeps the second, so a consumer replaying from offset 0 reconstructs the *current* state of every user, not the full change history. This is exactly what you want for a changelog or a materialized-state topic: a new consumer can bootstrap the entire current state by reading the compacted log once, and that state never falls off a time cliff. Kafka exposes this as `cleanup.policy=compact`, and many of its own internal topics (consumer-group offsets, for instance) are compacted.

The distinction matters for the rest of this post: time-retained logs give you *replay of recent events*; compacted logs give you *replay of complete current state*. Both are impossible on a queue or ephemeral pub/sub, and both flow from the same root property — the log retains, and reading does not delete. A queue cannot be compacted because there is no "key with a latest value" concept once messages are consumed and gone; the data simply is not there to compact.

### Reading is sequential, which is why the log is fast

One more property worth pausing on, because it explains the log's performance reputation. Because the log is append-only and consumers read in offset order, *both writes and reads are sequential disk access*. Producers append to the tail (sequential write); consumers stream forward from their offset (sequential read), and recently-written records are usually still in the OS page cache, so the read often never touches disk at all. Sequential I/O on modern hardware is wildly faster than the random I/O a queue's per-message acknowledge-and-delete bookkeeping tends to incur, and it is why a single log partition can sustain hundreds of megabytes per second. A queue, by contrast, must track per-message in-flight state and delete individual messages out of the middle of its structure, which is inherently more random and more expensive per message. The log trades the queue's fine-grained per-message control for coarse-grained sequential throughput — a trade that pays off enormously at high volume and matters not at all at low volume. This performance asymmetry is another reason high-throughput event pipelines gravitated to the log model even when they only needed queue semantics.

## 5. The big idea: a log emulates both queue and pub/sub

Now we reach the realization that this whole post has been building toward, the one that explains why log-based systems like Kafka and Pulsar took over so much of the data infrastructure world. It is this: **a log can emulate both a queue and pub/sub.** The log is not a third option sitting beside the other two — it is a more general substrate, and the queue and pub/sub are both *views* you can take on top of a log.

![A layered stack showing an append-only retained log at the base, partitions and consumer groups in the middle, and a queue view and a pub/sub view expressed on top of the same log](/imgs/blogs/queue-vs-pubsub-vs-log-three-messaging-models-6.webp)

The figure above is the architecture of the idea. At the base sits the append-only, retained, replayable log. On top of it sit partitions (for parallelism and ordering) and consumer groups (the unit that tracks offsets). And from those two primitives, you get *both* a queue view and a pub/sub view. Here is the mechanism for each:

**Emulating a queue (competing consumers):** Put all the consumers in *one* consumer group. The log assigns each partition to exactly one member of the group. Now each message — living in some partition — is read by exactly one consumer (the one that owns that partition). Add more consumers to the group and the broker rebalances, spreading partitions across them. This is precisely competing-consumers work distribution: each message processed once, by one of a pool of workers, with load balanced by partition assignment. The only difference from a classic queue is that "delete on ack" becomes "advance the offset" — but functionally, for work distribution, it behaves like a queue.

**Emulating pub/sub (fan-out):** Put each independent subscriber in its *own* consumer group. Now every group reads *every* message, because each group has its own independent offset over the full log. Five subscribers, five consumer groups, each sees all messages — that is fan-out, that is pub/sub. And because the log is retained, a new group can join later and replay history, which classic pub/sub could never do.

So the two knobs are: **how many consumers per group** (controls queue-style load balancing) and **how many groups** (controls pub/sub-style fan-out). One group with N members is a queue. N groups is pub/sub. N groups each with M members is *both at once* — fan-out across groups, load-balanced within each. The log subsumes the other two models as special cases.

### Why this made the log dominant

This is why, over the last decade, so many systems that "just needed a queue" or "just needed an event bus" ended up on Kafka. The log gives you the queue's work distribution *and* pub/sub's fan-out *and* retention *and* replay, all from one substrate, with one operational footprint. You do not have to choose up front. You can start with one consumer group doing work distribution, and later add a second group for analytics without touching the producer or the first consumer — because adding a group is non-disruptive on a log, where adding a "listener" to a queue would have stolen messages.

There is a real cost to this generality, and we should be honest about it: a log is heavier to operate than a simple queue. You manage partitions, consumer groups, offset commits, retention policies, and rebalancing. For a nightly batch job, that is absurd overkill — a simple SQS queue or a Redis list is the right tool. The log wins when you need *more than one* of {durable work distribution, multi-consumer fan-out, replay, ordered history}, because then a single log replaces what would otherwise be several specialized systems. But if you genuinely need only one of those — just work distribution, nothing else — a plain queue is simpler and cheaper, and reaching for a log is the overkill mistake I warned about in the intro.

#### Worked example: the same "order placed" event, served three ways

Let us make the emulation concrete with one event served by all three models, and tally what each costs. Suppose your order service emits an `order.placed` event at a steady **2,000 events per second**, and three downstream services must each react: an email service, an analytics service, and a fraud-scoring service. Each event is about **1 KB**.

**Served as a queue (one queue per service):** You create three queues. The producer must publish each event *three times*, once into each queue — or you put a small router in front that copies the event into all three. Each service runs its own pool of competing consumers draining its queue. Throughput is fine. But: there is no shared history, no replay. If the fraud service ships a bug and you want to re-score yesterday's orders, you cannot — those messages were acked and deleted. And the producer (or router) is now coupled to the *set* of consumers: adding a fourth service means adding a fourth queue and a fourth publish. Producer write amplification is 3x (three publishes per event).

**Served as pub/sub (fanout exchange, durable per-subscriber queues):** The producer publishes *once* into a fanout exchange. The broker copies the event into three bound durable queues, one per service. Producer write amplification drops to 1x at the producer (the broker does the copying). Adding a fourth service means binding a fourth queue — *no producer change*. This is strictly better than the three-queue approach for fan-out. But still: no replay. A new subscriber bound today sees only events from today forward; it cannot reconstruct what it missed.

**Served as a log (one topic, three consumer groups):** The producer appends each event *once* to the `orders` topic, partitioned by `order_id` across, say, 12 partitions. Each of the three services is a consumer group; each reads every event at its own offset. Producer write amplification is 1x; the storage holds *one* copy of each event (not three), retained seven days. Adding a fourth service is a fourth consumer group — no producer change, and the new group can start at offset 0 and *replay the last seven days* to backfill itself. At 2,000 events/s × 1 KB, that is 2 MB/s of ingest, about 173 GB/day, roughly 1.2 TB for a 7-day retention window — a very manageable footprint on commodity disk.

Tally the costs: the queue approach has 3x producer write amplification and triple storage (three copies) and no replay. Pub/sub fixes the producer amplification and keeps three copies in three queues, still no replay. The log keeps *one* copy, 1x producer write, fan-out to all three groups, *and* full replay, at the price of running a partitioned log and managing offsets. For three-plus consumers that care about history, the log is the clear winner. For a single consumer that just needs the work done once and forgotten, the log is overkill and a plain queue is right. That tradeoff is the entire decision, and we will formalize it into a rule shortly.

## 6. Retention and replay: the property that changes everything

If I had to name the *single* property that most distinguishes the three models and most changes how you architect a system, it would be **retention** — and its consequence, **replay**. Retention is the answer to "after a message is read, is it still there?" The queue says no (deleted on ack). Classic pub/sub says no (never stored in the first place, for absent subscribers). The log says yes (retained for days regardless of who read it). And from that one "yes" flows a capability the other two simply cannot offer: you can go back and read history again.

![A before-and-after comparison contrasting a queue that deletes each message on acknowledgment and loses its history with a log that appends at an offset, advances only the read cursor, and can rewind to replay seven days of records](/imgs/blogs/queue-vs-pubsub-vs-log-three-messaging-models-5.webp)

The figure above contrasts the two contracts directly. On the left, the queue: deliver to one consumer, ack removes the message, history is gone — there is no replay because there is nothing left to replay. On the right, the log: append at offset N, a read advances only the cursor (the record stays put), and any reader can rewind to replay seven days of history. The left side is a tape that erases itself as you read; the right side is a tape you can rewind. That difference is not a feature you can bolt onto a queue later — it is architectural. A queue that retained everything *would be* a log.

### What replay actually buys you

Replay sounds like a niche convenience until you realize how many real problems it solves. Here are the big ones, each of which is *impossible* without retention:

**Rebuilding a derived store.** You maintain a read-optimized projection — a search index, a denormalized cache, a materialized view — derived from a stream of events. The projection gets corrupted, or you change its schema, or you find a bug in how it was built. With a log, you blow away the projection, rewind a fresh consumer group to offset 0, and rebuild the whole thing from history. This is the foundation of [event sourcing and CQRS](/blog/software-development/message-queue/event-sourcing-and-cqrs-with-an-event-log): the log is the source of truth, and every derived store is a replayable function of the log. With a queue, the events that built the projection are gone, so there is nothing to rebuild from — your projection *is* your only copy, and if it is corrupt you are stuck.

**Fixing a consumer bug retroactively.** Your fraud model had a bug last Tuesday and scored a day of transactions wrong. With a log, you fix the code and replay Tuesday's offsets through the corrected consumer. With a queue, Tuesday's transactions were consumed and deleted — you have no way to re-run them.

**Onboarding a new consumer with history.** A new team wants to build a feature off your events, but they need the last week of events to bootstrap, not just events from now on. On a log, they create a consumer group starting at offset 0 and replay the week. On pub/sub, they only see events from the moment they subscribe — the past is unreachable.

**Debugging and audit.** When something goes wrong, you can read exactly what messages flowed through the system, in order, after the fact. The log *is* an audit trail. A queue's history evaporates as it is consumed, so post-hoc forensics on "what messages did we actually receive" is impossible.

#### Worked example: replaying 7 days to rebuild a derived store

Make it concrete. You run a product-search service whose index is a derived store built from a `product.updated` event stream on a Kafka topic. The topic carries about **40 million records** over a 7-day retention window (roughly 66 events/second average). One Monday you discover your indexing consumer has been dropping the `brand` field for two weeks due to a schema mismatch, so search-by-brand is broken for everything updated recently.

On a **log**, the recovery is mechanical. You fix the indexing code to read `brand`. You create a *new* consumer group, `search-indexer-v2`, configured with `auto_offset_reset="earliest"` and a `seek_to_beginning`. It starts at offset 0 and reprocesses all 40 million records into a freshly-built index. If your consumer processes, say, 20,000 records/second on the rebuild (no rate limit, just churning history), 40 million records take about **33 minutes** to replay. When `v2` catches up to the live tail, you flip search traffic to the new index and retire the old one. Total downtime for users: zero — you built `v2` alongside the live `v1`. The figure below traces this exact sequence as a timeline.

![A timeline showing a new consumer group that starts, seeks to offset zero, reprocesses forty million records over ten minutes, rebuilds the derived store, and catches up to the live tail](/imgs/blogs/queue-vs-pubsub-vs-log-three-messaging-models-9.webp)

Now run the *same* scenario, but with `product.updated` as a **queue** instead of a log. The events that updated those products were consumed and deleted as they arrived. Your search index is the *only* place that information ever landed, and it is the very thing that is corrupt. There is no source to rebuild from. Your options collapse to: scan the *primary* product database and re-derive the index from current state (which only works if the primary still holds everything and you have a separate batch path to read it — you are no longer using the message system at all), or accept the data loss. The replay that took 33 minutes on a log is simply *impossible* on a queue. This is the property that changes everything: retention turns "we lost data and cannot recover" into "we replayed history and fixed it in half an hour."

### Retention is not free

Honesty demands the counterweight: retention costs storage, and replay has operational hazards. Retaining 7 days at 2 MB/s is 1.2 TB; retaining 90 days is 15 TB. That is cheap on disk but not nothing, and it grows with throughput. More subtly, **retention sets a hard deadline on slow consumers**. If a consumer falls behind by more than the retention window, the oldest unread messages get deleted out from under it — the "retention cliff" — and it permanently loses data. A queue does not have this failure mode (it keeps unacked messages indefinitely), so the log trades the queue's unbounded backlog for bounded retention plus replay. And replay itself can be dangerous: re-running a week of events through a consumer that has *side effects* (sends emails, charges cards) will re-trigger every one of those side effects unless the consumer is idempotent. Replay is safe for rebuilding derived state; it is a loaded gun for anything with external effects. Plan for both.

## 7. Ordering and consumer coupling across the three

Two more properties separate the models in ways that bite you in production: **ordering guarantees** and **consumer coupling** (how much adding or removing a consumer disturbs the system). These are less famous than retention but just as decisive in a design review.

### Ordering

**Queue:** A single-consumer queue is strict FIFO. Add competing consumers and you keep rough order but lose strict order — message 5 can finish before message 3 because different workers process at different speeds. So a queue gives you *either* strict order (one consumer, no parallelism) *or* parallelism (many consumers, no strict order), but not both. Some brokers add message-group ordering (SQS FIFO queues with a `MessageGroupId`) to thread the needle, but that is essentially per-key ordering bolted on.

**Pub/sub:** Classic pub/sub guarantees little about ordering, especially across subscribers — each subscriber gets its own copy and may process at its own rate. Within one subscriber's durable queue you get that queue's ordering, but the broadcast itself makes no cross-subscriber promise. For broadcast use cases (each subscriber independent), this rarely matters.

**Log:** A log gives **strict per-partition ordering**. Within a partition, records are read in exactly the order they were appended, by offset. Across partitions there is no global order. Since you choose the partition key, you choose your ordering domain: partition by `order_id` and every event for a given order is strictly ordered, while different orders proceed in parallel. This is the best of both worlds — parallelism *and* strict order, scoped per key — and it is a major reason logs won for event streams where order matters (a `payment.refunded` must not be processed before the `payment.captured` it refunds).

### Consumer coupling

This is the property nobody puts on the comparison chart and everybody trips over. **How much does adding or removing a consumer disturb the rest of the system?**

**Queue:** High coupling in a sneaky way. Adding a consumer to a queue *changes message distribution* — the new consumer competes for messages, so existing consumers get *fewer*. If you add a consumer expecting it to *also* see all messages, you have just broken the existing consumers by stealing half their workload. Consumers on a queue are not independent; they share one stream and divide it.

**Pub/sub:** Low coupling for fan-out — adding a subscriber gives it its own copy without touching existing subscribers. This is pub/sub's whole point. But removing a subscriber (in ephemeral pub/sub) means messages destined for it vanish, and an absent subscriber silently misses everything, which is a different kind of coupling: the *timing* of subscription matters.

**Log:** Lowest coupling of all, and this is the log's quiet killer feature. Consumer groups are fully independent — each has its own offset over the same retained records. Add a group: it reads everything, disturbing no one, and can even start in the past via replay. Remove a group: nothing else changes. A slow group falls behind on its own offset without slowing any other group. This independence — many consumers, zero cross-consumer interference — is what makes the log so composable. You can keep bolting on new consumers (new teams, new features, new analytics) indefinitely, and none of them affects the producer or each other. That is the property that lets one log become the backbone of an entire organization's data flow.

The cross-cutting lesson: **queue consumers share one stream and divide it (high coupling), pub/sub subscribers each get a copy but only while present (medium coupling), log groups each get an independent replayable view (lowest coupling).** Match the coupling profile to your organizational reality — if many independent teams will read your events over time, you want the log's independence; if you just have a pool of identical workers draining a backlog, the queue's sharing is exactly right.

There is an organizational dimension to consumer coupling that is easy to miss when you are only thinking about a single team. In a large company, the *producer* of an important event (say, the order service) often has no idea, years out, who will want to consume its events. New teams form, new products launch, new analytics questions arise. With a queue, every new consumer requires the producer (or a router) to change — a new queue, a new binding, a new copy of the publish. The producer becomes a coordination bottleneck: it has to know about and accommodate every consumer. With a log, the producer publishes once to a topic and never thinks about consumers again; any team can spin up a consumer group, read the full retained history to bootstrap, and proceed without ever talking to the producing team. This is why log-based event platforms tend to *accrete* consumers over time: each new use case is additive and zero-coordination, so the log becomes a kind of company-wide nervous system. A queue cannot become that, because each new consumer is a change to the shared stream that risks disturbing the existing ones. The coupling property, in other words, is not just a technical nicety — it determines whether your messaging system can scale across an *organization*, not just across machines.

## 8. Mapping real brokers to models (RabbitMQ, SQS/SNS, Kafka, Redis, Pulsar)

Theory is clean; real brokers are messier, because most of them support *more than one* model through different features. Let us map the popular brokers onto the three models so you know what you are actually getting when you reach for each.

![A taxonomy tree branching from messaging models into point-to-point queue, pub/sub broadcast, and append-only log, with example brokers under each including SQS, RabbitMQ, SNS, Redis, Kafka, Pulsar, and Kinesis](/imgs/blogs/queue-vs-pubsub-vs-log-three-messaging-models-7.webp)

The tree above is the map. Below is the broker-by-broker reality, with the important caveat that several brokers straddle multiple models.

**RabbitMQ** is fundamentally a **queue** broker with a flexible routing layer that lets it *also* do pub/sub. The core building block is the queue (durable, competing consumers, consume-and-delete). Exchanges sit in front of queues and decide routing: a `direct` exchange routes to one queue (queue model), a `fanout` exchange copies to every bound queue (pub/sub via durable subscriptions), and a `topic` exchange routes by key pattern (content-based pub/sub). So RabbitMQ does queue and pub/sub well, but it does *not* do the log model — once a message is acked it is gone; there is no retention or replay. Its strength is rich routing and per-message control; its weakness is no replayable history. The full operational picture is in the [RabbitMQ production architecture and scaling](/blog/software-development/system-design/rabbitmq-production-architecture-scaling) post.

**SQS** is a pure **queue**: durable, competing consumers, at-least-once delivery, consume-and-delete (a message becomes invisible while in flight and is deleted on `DeleteMessage`). No fan-out, no replay. SQS FIFO queues add per-`MessageGroupId` ordering and exactly-once-ish dedup. It is the simplest possible durable queue, fully managed, and ideal for work distribution where you never need history.

**SNS** is pure **pub/sub**: publish a message to a topic, and it fans out to every subscription (HTTP endpoints, Lambda, email, or — most usefully — SQS queues). On its own SNS is *ephemeral* fan-out (an endpoint that is down misses the message), which is why the durable pattern is **SNS-to-SQS**: SNS fans out, each SQS queue durably holds one subscriber's copy. That composition — pub/sub for fan-out, queue for durability — is the AWS-native way to get durable pub/sub, and it is exactly the "pub/sub composed with queues" idea from section 3.

**Kafka** is the canonical **log**: partitioned, append-only, retained, replayable, with consumer groups and offsets. It emulates a queue (one consumer group with competing partition consumers) and pub/sub (many consumer groups) as described in section 5, which is why it so often replaces both. Its weakness is operational weight and weaker per-message routing (no topic exchange equivalent; routing is by partition key and topic). Deep dive: [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log).

**Redis** offers *all three* in primitive form, which is both convenient and a trap. Redis **pub/sub** (`PUBLISH`/`SUBSCRIBE`) is pure ephemeral pub/sub — zero retention, fire-and-forget, perfect for cache invalidation and presence, disastrous for anything that must not be missed. Redis **lists** (`LPUSH`/`BRPOP`) make a simple **queue**. Redis **Streams** (`XADD`/`XREAD`) are a genuine **log** — append-only, retained, with consumer groups and offset-like IDs — a lightweight Kafka-alike for moderate scale. So "Redis" tells you nothing about the model; you have to ask *which* Redis feature, and they have wildly different contracts.

**Pulsar** is a **log** like Kafka (append-only, retained, replayable) but with a unification trick: its *subscription types* let one system behave as queue or pub/sub explicitly. An `Exclusive` or `Failover` subscription behaves like ordered single-consumer; a `Shared` subscription behaves like a competing-consumers queue (round-robin across consumers, even within a partition); and multiple subscriptions on the same topic give pub/sub fan-out. Pulsar essentially exposes the section-5 emulation as first-class subscription modes, which some teams find cleaner than Kafka's group conventions.

Here is the mapping as a table you can keep:

| Broker | Native model | Queue? | Pub/sub? | Log (retain + replay)? |
| --- | --- | --- | --- | --- |
| SQS | Queue | Yes | No | No |
| RabbitMQ | Queue + routing | Yes | Yes (fanout/topic) | No |
| SNS | Pub/sub | No | Yes | No (ephemeral) |
| SNS to SQS | Pub/sub + queue | Yes | Yes (durable) | No |
| Redis pub/sub | Pub/sub | No | Yes | No (ephemeral) |
| Redis Streams | Log | Yes (groups) | Yes (groups) | Yes |
| Kafka | Log | Yes (1 group) | Yes (N groups) | Yes |
| Pulsar | Log | Yes (Shared sub) | Yes (N subs) | Yes |
| Kinesis | Log | Yes | Yes | Yes (24h–365d) |

The single most useful row to internalize: every broker in the **log** rows (Kafka, Pulsar, Kinesis, Redis Streams) can express *both* queue and pub/sub, while no broker in the queue or pub/sub rows can express the log. Generality flows one direction only — the log is the superset.

## 9. Choosing a model for a given problem

Enough taxonomy. Here is the decision procedure I actually use, reduced to questions you answer in order. Each answer prunes the space.

**Question 1 — How many consumers must see each message: one, or all?**
If each message should be processed by *exactly one* consumer (work distribution, jobs, tasks), you want **queue semantics**. If each message should reach *every* interested consumer (events, broadcasts, notifications), you want **pub/sub or log semantics**. This single question separates work distribution from broadcast and is the first fork in the road.

**Question 2 — Do you need replay or durable history?**
If yes — if you will ever need to reprocess past messages, rebuild a derived store, onboard a consumer with history, or audit what flowed through — you want the **log**, full stop. Replay is impossible to retrofit; you either retained the messages or you did not. If you genuinely never need history (a message processed once is forever irrelevant), a queue or pub/sub is lighter.

**Question 3 — How independent are the consumers, and how many will there be over time?**
If you have a fixed pool of identical workers draining a backlog, a **queue** fits perfectly — they share and divide one stream. If you have *many independent consumers* that will grow over time (new teams, new features, new analytics, each reacting to your events differently), you want the **log**'s consumer-group independence, where adding a consumer disturbs nothing.

**Question 4 — What ordering do you need?**
If you need strict per-key ordering *with* parallelism (refunds after captures, state-machine transitions in order), the **log** with key-based partitioning is the only clean answer. If you need no ordering or only rough order, a queue is fine. If ordering is irrelevant (independent broadcasts), pub/sub is fine.

**Question 5 — How much operational weight can you afford?**
A log (Kafka/Pulsar) is heavier to run than a queue (SQS/RabbitMQ) or a fan-out (SNS/Redis). If you need only *one* of {one-of-N delivery, fan-out, replay, ordered history}, take the lightest tool that gives it. If you need *two or more*, the log's generality earns its weight by replacing several systems with one.

Collapsing those into a one-line rule: **Need replay or durable ordered history, or many independent growing consumers? Use a log. Need pure one-of-N work distribution with no history? Use a queue. Need ephemeral fan-out where missing a message is fine? Use pub/sub.** When two or more of those pull at once, the log is almost always the consolidation that pays off, which is exactly why logs became the default backbone of modern event-driven systems.

![A grid showing one system using all three models at once: an API request feeds a command queue, which emits events to a pub/sub fan-out across four services, which feed an audit log retained ninety days, replayed to rebuild derived stores read by ops and analytics](/imgs/blogs/queue-vs-pubsub-vs-log-three-messaging-models-8.webp)

The figure above is the punchline of the decision section: a mature system rarely picks *one* model — it uses **all three for different jobs**. An incoming API request ("place order") goes onto a **command queue** so exactly one worker handles each job (one-of-N work distribution). That worker emits domain **events** via **pub/sub** fan-out to four services that each need to react (broadcast). And every event is also appended to an **audit log** retained 90 days, which can be replayed to rebuild derived stores that ops and analytics teams read at their own offsets (durable, replayable history). Commands want a queue; events want fan-out; history wants a log. The skill is not picking one model for your whole architecture — it is recognizing which of the three each *individual* flow needs. And once you see that a log can express the other two, you can often serve all three flows from one log-based system, which is the consolidation that simplifies the operational footprint.

## Case studies and war stories

Abstract rules stick better when attached to scars. Here are four real-shaped situations and the lesson each one teaches about the three models.

### The "second consumer" that stole half the orders

A team had an order-processing service draining a RabbitMQ queue with a single consumer. A new requirement arrived: also emit analytics for each order. An engineer, reasoning that "it is already on the queue, I will just add another consumer," attached a second consumer to the *same queue*. Analytics started flowing — and order processing immediately started dropping half its throughput, because the two consumers were now *competing* on the same queue, each grabbing roughly half the messages. Orders were not lost (the analytics consumer "processed" them too), but the order-processing consumer now only saw half the orders, and half the orders skipped order processing entirely while the analytics consumer handled them as no-ops.

The lesson is the most fundamental one in this post: **a queue divides; it does not duplicate.** Adding a consumer to a queue does not give you fan-out — it gives you competition. The fix was to introduce a fanout exchange (pub/sub): the order producer publishes to the exchange, which copies into *two* queues, one for order processing and one for analytics, each with its own consumer. Two independent streams, no competition. Had this been a Kafka topic, the fix would have been even simpler — put analytics in its own consumer group and it would have read every message independently, with replay available for free.

### The midnight cache-invalidation that wasn't

A platform used Redis pub/sub to broadcast cache-invalidation events: when a record changed, a `cache.invalidate` message went out on a Redis channel, and every application server subscribed and dropped the stale entry. It worked beautifully for months. Then one night a brief network partition disconnected several app servers from Redis for about four seconds. Those four seconds of invalidation messages — published while the servers were disconnected — simply vanished, because Redis pub/sub retains nothing. The affected servers reconnected and resumed listening, blissfully unaware they had missed anything, and served stale data for those records until the next, unrelated change happened to re-invalidate them. Some records served stale data for *hours*.

The lesson: **ephemeral pub/sub silently drops messages to absent subscribers, and "absent" includes a four-second network blip.** Ephemeral pub/sub is correct only when missing a message is *self-healing* — and cache invalidation is borderline (a missed invalidation persists until the next change). The team moved cache invalidation to a Redis *Stream* (a log) with a short retention, so a reconnecting server could read the few messages it missed via its consumer-group offset and catch up. The model changed from ephemeral pub/sub to a retained log, and the silent-drop failure mode disappeared.

### The retention cliff that ate a day of events

A data team ran a Kafka pipeline with 24-hour retention (chosen to keep storage low) feeding a derived analytics store. The derived-store consumer normally kept up easily. Then a bad deploy slowed that consumer to a crawl, and over a long weekend nobody noticed the lag climbing. By the time it was caught, the consumer was *more than 24 hours* behind — which meant the oldest events it had not yet read had already been *deleted* by retention. The consumer's offset pointed at records that no longer existed; Kafka reset it forward to the earliest *available* offset, and roughly a day of events were skipped entirely. The derived store had a permanent hole.

The lesson: **a log's retention window is a hard deadline on your slowest consumer.** Unlike a queue, which holds unacked messages indefinitely, a log deletes on a clock regardless of whether anyone read the messages. Replay is only possible *within* the retention window — fall behind by more than retention and the data is gone, exactly as if it had been a queue. The fixes were threefold: alert on consumer lag (not just consumer liveness), set retention generously larger than the worst plausible recovery time (they moved to 7 days), and treat "lag approaching retention" as a page-worthy incident. Retention gives you replay, but only if you stay inside the window.

### Replaying a payment stream — into a wall of duplicate emails

A team correctly used a Kafka log for payment events and correctly used replay to rebuild a derived ledger after a schema change. They replayed three days of `payment.captured` events through the rebuild consumer. Unfortunately, the rebuild consumer shared code with the *live* consumer, and that code sent a "payment received" email as a side effect. Replaying three days of events sent three days' worth of duplicate emails to customers — tens of thousands of them — for payments that had completed days earlier.

The lesson: **replay re-triggers side effects.** A log makes replay easy, which makes it easy to forget that re-reading history *re-executes* whatever the consumer does. Replay is safe for rebuilding *derived state* (idempotent writes to a store) and dangerous for *external effects* (emails, charges, webhooks). The discipline is to separate the projection logic (safe to replay) from the side-effect logic (not safe to replay), and to run replays through a code path that has side effects disabled, or to make every side effect idempotent with a dedup key keyed on the event's offset. Replay is a superpower with a sharp edge; respect it.

### The Kafka cluster that ran a nightly email

Not every war story is about losing data; some are about paying too much for a non-problem. A small team needed to send a batch of marketing emails once a night — a few thousand emails, a single producer, a single consumer, no fan-out, no replay, no ordering concern. A well-meaning engineer, having just read about event-driven architecture, set up a three-broker Kafka cluster with a partitioned topic, consumer groups, offset management, and a monitoring stack to watch consumer lag. It worked, but it was now a standing operational burden: brokers to patch, ZooKeeper (at the time) to babysit, partition rebalances to understand, retention to tune, disk to provision — all to move a few thousand messages once a day, a workload a single SQS queue or even a database table would have handled with zero standing infrastructure.

The lesson is the mirror image of the others: **the log's generality is real weight, and weight you do not need is pure cost.** When you need exactly one of {work distribution, fan-out} and nothing else — no replay, no history, no growing set of independent consumers — a log is overkill, and the simplest durable queue or fan-out is the right call. The team moved the job to an SQS queue (a few lines of config, nothing to operate) and decommissioned the cluster. The skill in choosing a model is not always "pick the most capable tool." Often it is "pick the *least* capable tool that still meets the requirement," because every capability you do not use is operational surface area you pay to maintain. Reach for the log when its retention, replay, ordering, or consumer-independence actually earn their keep — and not a moment sooner.

## When to reach for each model (and when not to)

Here is the decisive recommendation, model by model, with the anti-patterns called out.

**Reach for a queue when:** you have work to distribute across a pool of workers, each task should be done once, and you do not need history. Job processing, task queues, background work, request buffering, load leveling in front of a slow downstream. SQS and RabbitMQ queues shine here. **Do not reach for a queue when** you need multiple different consumers to each see every message (you will get competition, not fan-out) or when you will ever need to replay (the messages are gone on ack).

**Reach for pub/sub when:** you have events many independent parties care about, missing one is tolerable or self-healing, and you do not need replay. Live dashboards, presence, cache invalidation, real-time notifications, ephemeral broadcasts. SNS, Redis pub/sub, RabbitMQ fanout. **Do not reach for ephemeral pub/sub when** every message must be processed (absent subscribers silently lose data) — in that case use *durable* pub/sub (SNS-to-SQS) or a log instead.

**Reach for a log when:** you need any of replay, durable ordered history, per-key ordering with parallelism, or many independent consumer groups that will grow over time — and especially when you need *more than one* of those. Event sourcing, stream processing, change data capture, event-driven backbones, derived-store rebuilds, audit trails. Kafka, Pulsar, Kinesis, Redis Streams. **Do not reach for a log when** you need exactly one of {work distribution, fan-out} and nothing else — the partitions, consumer groups, offset management, and retention tuning are real operational weight that a simple queue or fan-out avoids. Running a Kafka cluster to power a nightly email job is the overkill anti-pattern.

![A decision matrix with rows for retention, fan-out, replay, and ordering and columns for queue, pub/sub, and log, summarizing that the log dominates on retention and replay while the queue and pub/sub are lighter for their specialties](/imgs/blogs/queue-vs-pubsub-vs-log-three-messaging-models-1.webp)

The matrix above (the same one that opened the post) is the summary you should be able to reconstruct from memory after reading this: the queue deletes on read and serves one consumer with no replay; pub/sub copies to all present subscribers with no retention and no replay; the log retains everything, serves every group, and replays freely, at the cost of ordering and operational complexity you must manage with partitions. Pin that table to the wall of your mind, and most messaging design questions answer themselves.

The meta-recommendation: do not pick one model for your whole system. Pick the right model *per flow*. Commands want queues, events want fan-out, history wants a log — and a single log-based platform can often serve all three by varying consumer-group count, which is why so many teams consolidate onto Kafka or Pulsar. But consolidation is a choice, not an obligation; a small system with one work queue and one cache-invalidation channel has no business running a log. Match the model to the flow, and the tool to the model.

## Key takeaways

- **There are exactly three messaging models, and they differ in what happens to a message after it is read.** A queue delivers to one consumer and deletes on ack. Pub/sub copies to every present subscriber and keeps nothing. A log retains everything and lets any reader replay it. This single distinction resolves most messaging design arguments.
- **A queue divides a stream; it does not duplicate it.** Adding a consumer to a queue makes consumers *compete* for messages, not each receive a copy. If you need fan-out, you need pub/sub or a log, never a second consumer on the same queue.
- **Ephemeral pub/sub silently drops messages to absent subscribers.** It is correct only when missing a message is self-healing (dashboards, presence). For must-process events, use durable pub/sub (SNS-to-SQS) or a log.
- **The log's defining property is retention, and its superpower is replay.** Because reading only advances a consumer-owned offset, history survives, and any consumer can rewind to rebuild a derived store, fix a bug retroactively, or onboard with history — all impossible on a queue.
- **A log can emulate both a queue and pub/sub.** One consumer group with competing partition consumers is a queue; many consumer groups is pub/sub. This generality — the log as a superset — is why log-based systems became the default backbone of event-driven architecture.
- **Consumer coupling is the underrated property.** Queue consumers share and divide one stream (high coupling); pub/sub subscribers each get a copy but only while present (medium coupling); log consumer groups each get an independent, replayable view (lowest coupling). For many growing independent consumers, the log's independence wins.
- **Retention is a hard deadline, not a safety net.** A log deletes on a clock regardless of who has read; a consumer that falls behind by more than the retention window loses data permanently, just like a queue. Alert on lag and size retention against worst-case recovery time.
- **Replay re-triggers side effects.** Re-reading history re-executes the consumer. Replay is safe for rebuilding derived state and dangerous for external effects (emails, charges). Separate projection logic from side-effect logic, or make every effect idempotent.
- **Most mature systems use all three models for different flows.** Commands want a queue, events want fan-out, history wants a log. The skill is matching the model to each flow — and a single log-based platform can often serve all three.

## Further reading

- [What is a message queue? Async, decoupling, and load leveling](/blog/software-development/message-queue/message-queues-async-decoupling-and-load-leveling) — the series opener on why asynchronous messaging exists at all.
- [Anatomy of a message system: producers, brokers, consumers](/blog/software-development/message-queue/anatomy-of-a-message-system-producers-brokers-consumers) — the structural building blocks underneath all three models.
- [Delivery semantics: at-most, at-least, and exactly once](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) — the acknowledgment and offset protocols that make these models durable.
- [Event sourcing and CQRS with an event log](/blog/software-development/message-queue/event-sourcing-and-cqrs-with-an-event-log) — building systems where the log is the source of truth and every store is a replayable projection.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — the deep dive on the canonical log implementation.
- [RabbitMQ production architecture and scaling](/blog/software-development/system-design/rabbitmq-production-architecture-scaling) — the deep dive on the canonical queue-plus-routing broker.
- [Change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — turning a database's write-ahead log into an event stream, a log-model technique in disguise.
- [Database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding) — the partitioning mechanics a log reuses to scale and order at once.
