---
title: "Queues and Event Streaming for Architects: Decoupling, Delivery, and the Outbox"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Learn when a senior reaches for async messaging, how to choose a queue versus a log, why exactly-once delivery is a myth, and how the transactional outbox makes publishing as reliable as your database commit."
tags:
  [
    "system-design",
    "message-queue",
    "event-streaming",
    "kafka",
    "outbox-pattern",
    "delivery-semantics",
    "saga",
    "architecture",
    "distributed-systems",
    "scalability",
    "optimization",
  ]
category: "software-development"
subcategory: "System Design"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/queues-and-event-streaming-for-architects-1.webp"
---

There is a moment in every system's life when a single synchronous call becomes a liability you can no longer afford. The checkout endpoint that used to just write an order row now also has to charge a card, decrement inventory, send a confirmation email, update the search index, notify the warehouse, and ping the analytics pipeline — and it does all of it inline, in the request, while the customer's browser spins. Each of those downstreams has its own latency and its own bad day, and because the call is synchronous, your checkout's p99 is the *sum* of everyone's p99 and your checkout's availability is the *product* of everyone's availability. The day the email provider has a slow incident, your checkout times out, and you lose orders not because you could not save them but because you insisted on telling six other systems first.

The senior move here is not "make the email service faster." It is to notice that almost none of that work needs to happen *before* you tell the customer their order is placed. The order is real the instant the row commits; everything else is a *consequence* that can happen a few seconds later. That observation — that you can separate "the thing happened" from "everyone who cares has reacted" — is the entire reason async messaging exists, and figure 1 shows the shape of the win: collapse a chain of synchronous failures into one fast write plus a backlog that drains on its own schedule.

![A two-column comparison showing synchronous coupling where the API calls four services inline against async decoupling where the API writes one event and a broker absorbs the backlog](/imgs/blogs/queues-and-event-streaming-for-architects-1.webp)

This post is the architect's decision layer on async messaging. The [message-queue folder on this blog](/blog/software-development/message-queue/anatomy-of-a-message-system-producers-brokers-consumers) already deep-dives the mechanisms — how brokers store messages, how Kafka lays out a partitioned log on disk, how an idempotent producer works — and I will cross-link to those rather than re-derive them. My job is the layer above: *when* a senior introduces a queue, *which* kind, what reliability you are actually buying versus what you think you are buying, and the one pattern (the transactional outbox) that separates messaging systems that lose data from ones that do not. By the end you should be able to decide queue versus log versus a plain synchronous call for a concrete feature, reason about delivery semantics without hand-waving "exactly-once," design an outbox flow and state the latency and consistency it costs you, and stress-test the whole thing against a slow consumer, a poison message, and a downstream outage.

## 1. What async messaging actually buys you (and the bill it sends)

Before choosing any technology, get crisp on *why* you are adding a broker, because every reason maps to a property you are buying and a cost you are signing up for. There are four reasons a senior reaches for async messaging, and "it's more scalable" is not one of them — that is a slogan, not a reason.

The first is **decoupling in time**. The producer and the consumer no longer have to be alive at the same instant. The order service can publish `OrderPlaced` while the recommendation service is mid-deploy and down for ninety seconds; the event waits in the broker and gets processed when the consumer comes back. In a synchronous world that ninety-second deploy is ninety seconds of failed checkouts. With a broker it is ninety seconds of harmless backlog. This is the single most underrated benefit, because it changes what "downstream is down" means from "I fail" to "I am slightly behind."

The second is **load leveling and peak-shaving**. Real traffic is spiky — a marketing email goes out, a product trends, a batch job kicks off at midnight — and the spike is often 10x the baseline for a few minutes. If every request synchronously consumes downstream capacity, you must provision downstream for the *peak*, which sits idle 99% of the time and costs you accordingly. A queue lets the producer accept the spike at memory speed and the consumer process at its own steady rate, so you provision the consumer for the *average* and let the queue depth absorb the difference. The buffer converts a capacity problem into a latency problem, and latency on a background task is usually free.

The third is **fan-out**. One event, many independent reactions. `OrderPlaced` needs to reach email, search indexing, the warehouse, fraud scoring, and the data warehouse. Synchronously, the order service would need to know about and call all five, and every new consumer means a code change in the producer. With a broker — especially a log — the producer emits the event once and is done; consumers subscribe on their own, and adding a sixth consumer next quarter touches zero producer code. This is the property that turns a tangle of point-to-point calls into an event-driven architecture.

The fourth is **resilience and absorption**. A queue in front of a fragile dependency acts as a shock absorber. If the downstream slows down or briefly fails, the work piles up safely instead of erroring back to the user, and a retry from the queue is cheap and automatic. The queue *contains* the blast radius of a downstream incident.

Now the bill, because none of this is free and a senior names the cost before recommending the pattern. Async messaging buys you those four properties and charges you in **eventual consistency** (the email is not sent the instant the order commits; there is a window where the order exists and the side effects have not happened), **duplicate delivery** (at-least-once means your consumer *will* see some messages twice and must be built to tolerate it), **ordering complexity** (a single queue across many consumers does not preserve global order, and even a log only orders within a partition), and a real **operational burden** (a broker is now a tier-0 dependency you must run, monitor, capacity-plan, and reason about during incidents — consumer lag dashboards, partition rebalancing, dead-letter queues, the works). The trade is almost always worth it for the right workload, but if you add a broker without a clear answer to "which of the four am I buying, and can I tolerate the four costs," you have added complexity for a slogan.

## 2. The fundamental fork: a queue or a log

Almost every confused messaging design I have reviewed traces back to one unmade decision: is this a **queue** or a **log**? They look similar — producers put messages in, consumers take messages out — but they have opposite data models, and choosing the wrong one means fighting the tool forever. This is the most important conceptual distinction in the whole domain, more important than RabbitMQ-versus-Kafka, because the product choice falls out of it.

A **queue** (RabbitMQ, Amazon SQS, classic JMS brokers) is a *work-distribution* primitive. A message is a unit of work that needs to be done *once*, by *one* of several interchangeable workers. The defining behavior is **competing consumers**: you run N workers on the same queue, the broker hands each message to exactly one of them, and when that worker acknowledges, the message is *deleted*. The queue's purpose is to spread work across a pool and shrink as work gets done — an empty queue is a healthy queue. Messages are typically consumed in roughly FIFO order, but because multiple consumers pull concurrently and any one can fail and redeliver, you do not get a strict global order at the consumer. The picture to hold is a to-do list that many hands work through and that you cross items off of.

A **log** (Apache Kafka, AWS Kinesis, Apache Pulsar in its streaming mode, Redpanda) is a *replayable, ordered, multi-reader* primitive. Messages are *appended* to an ordered, partitioned log and **retained** — for a fixed window (seven days) or until a size cap or, with compaction, forever for the latest value per key. Consumers do not delete anything; each consumer (or consumer group) tracks its own **offset**, a cursor into the log, and reads forward from wherever it left off. The same message can be read by ten different consumer groups, each at its own pace, and any of them can rewind its offset to reprocess history. The log's purpose is to be a durable, ordered, shared source of truth that many independent systems read. The picture to hold is a ledger or an append-only journal that many readers scan, each keeping their own bookmark — and that bookmark is the whole game.

The consequences of this difference are large and worth internalizing:

- **Deletion vs retention.** A queue forgets a message the moment it is acked; a log keeps it for the retention window regardless of who has read it. So a queue cannot replay — once consumed, the work is gone. A log can replay the last seven days into a brand-new consumer at full speed.
- **Single consumption vs multi-consumption.** A queue gives each message to one consumer in the group; that is the point. A log gives the *whole stream* to each consumer group; adding a new group does not steal messages from existing groups.
- **Ordering.** A queue does not promise order across concurrent consumers. A log promises strict order *within a partition* and nothing across partitions. (The mechanisms behind this are covered in [message ordering and partitioning guarantees](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees).)
- **Scaling unit.** A queue scales by adding consumers to the pool. A log scales by adding partitions, because a partition is the unit of parallelism and ordering at once — more on that in section 7.

When does each fit? Reach for a **queue** when the message is a *command* — "send this email," "resize this image," "charge this card" — that one worker should execute once and then it is done forever, and when you want the operational simplicity of a shrinking backlog and per-message acks. Reach for a **log** when the message is an *event* — "an order was placed," "a user changed their email" — that multiple independent systems care about, when you need replay (to rebuild a cache, seed a new service, recover from a bug that corrupted a downstream), or when you need strict per-key ordering. A useful heuristic: if you find yourself wishing you could "add another consumer that also sees these," or "replay last week into the new service," you wanted a log and built a queue. If you find yourself running a single-consumer Kafka topic just to send emails, you wanted a queue and built a log.

For the deep mechanics of the log model — segments, the offset index, log compaction, how Kafka turns a partition into an ordered append-only file — see [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log). I will stay at the decision layer.

## 3. The decision matrix: queue, log, or just call it

Most features do not need a broker at all. The third option, and the right one more often than architects admit, is a plain **synchronous call** — an in-process function, a database transaction, or an HTTP/gRPC request — because it gives you an answer *now*, with the caller's exact ordering, no eventual-consistency window, and no broker to operate. The cost of sync is the coupling we opened with: you wait for the callee, and you fail when it does. So the real architectural decision is a three-way one, and figure 2 lays the three transports against the properties that matter.

![A decision matrix scoring queue, log, and synchronous RPC across decoupling, ordering, replay, fan-out, latency to answer, and operational cost](/imgs/blogs/queues-and-event-streaming-for-architects-2.webp)

The matrix earns its keep when you read it as "which column can I not compromise on for *this* feature." If the feature needs an answer *in the request* — the user is waiting on the result, like a balance check or an auth decision — the "latency to answer" row decides it and you call synchronously, full stop. Async is for work the user does not need to wait for. If the feature needs *replay* — you know you will want to rebuild a downstream from history, or you are feeding multiple analytics systems — the "replay" row eliminates the queue and you reach for a log. If the feature is fire-once work for a pool of identical workers with no replay need — send the email, transcode the video — the queue's lower operational cost wins and a log would be over-engineering.

The operational-cost row deserves a word, because it is the one architects discount and operators do not. A log like Kafka is genuinely more to run than a queue: you are managing partition counts (which you cannot easily decrease), consumer-group rebalances, broker replication and ISR, retention sizing, and a ZooKeeper or KRaft control plane. A managed queue like SQS is close to zero operational surface — you create it and it scales itself. That difference is real money and real on-call load. "We might want replay someday" is not sufficient justification to take on a log's operational weight; "we have two named consumers today and a clear third coming, and we have already needed to replay once" is.

Here is the table form, which I keep in design docs because it forces the trade-off into the open:

| Property | Queue (SQS, RabbitMQ) | Log (Kafka, Kinesis) | Sync RPC (HTTP, gRPC) |
| --- | --- | --- | --- |
| Decoupling | Strong (time + space) | Strong (time + space) | None — caller waits |
| Ordering | Weak across consumers | Strict per partition | Caller's own order |
| Replay history | Gone on ack | Days of replay | None |
| Fan-out to readers | Competing, one wins | N groups each read all | One caller |
| Latency to an answer | Async, eventual | Async, eventual | Now, p99 inline |
| Throughput ceiling | Very high | Very high | Bounded by callee |
| Operational cost | Low (managed) | High (partitions, groups) | Low |
| Best when | Fire-once commands | Multi-reader events, replay | Need an answer in-band |

A senior does not pick the most powerful option; they pick the *least* powerful option that satisfies the requirement, because every unused capability is operational cost you pay forever. A synchronous call you can delete in an afternoon. A Kafka cluster you live with for years.

## 4. Delivery semantics, and why "exactly-once delivery" is a lie

This is the section where most messaging discussions go wrong, so let me be blunt. There are three delivery semantics, and figure 3 turns them into a decision you can actually make.

![A decision tree for choosing a delivery guarantee, branching on whether loss is tolerable and whether the consumer can deduplicate, leading to at-most-once, at-least-once, or effectively-once](/imgs/blogs/queues-and-event-streaming-for-architects-3.webp)

**At-most-once** means the broker delivers each message zero or one times — you might lose it, you will never see it twice. You get this by acknowledging the message *before* you process it (or by using fire-and-forget with no ack at all): if the consumer crashes after ack but before doing the work, the message is gone and nobody retries. The win is speed and simplicity; the loss is, well, loss. At-most-once is the right choice for high-volume, low-value, individually-disposable data: metrics samples, debug logs, a clickstream where one dropped click does not change the aggregate. Never use it for anything you would be paged about losing.

**At-least-once** means the broker delivers each message one or more times — you will never lose it, but you *will* sometimes see it twice. You get this by acknowledging *after* you finish processing: if the consumer crashes after doing the work but before the ack lands, the broker has not seen the ack, assumes the work failed, and redelivers. So the work runs twice. This is the default and the correct default for almost everything that matters, because losing a message is usually catastrophic and seeing it twice is usually merely annoying — *if* your consumer is built to tolerate it. The cost is duplicates, and the duplicates are not rare or exotic; they happen on every consumer crash, every rebalance, every network blip that drops an ack. You will get duplicates. Plan for them.

**Exactly-once** is where the lie lives. Exactly-once *delivery* across a network is impossible in the general case — it runs straight into the Two Generals problem. The sender cannot know whether a message it sent was received unless it gets an ack, and it cannot know whether an ack it failed to receive was actually sent, so to guarantee the message arrives it must be willing to resend, and resending means the receiver might get it twice. No protocol escapes this; "exactly-once delivery" is marketing. What *is* achievable, and what people actually want, is **effectively-once processing**: the message may be *delivered* more than once, but its *effect* on your system happens exactly once. You build that on top of at-least-once delivery with two ingredients — **idempotency** (processing the same message twice produces the same result as processing it once) and **deduplication** (recognizing a message you have already processed and skipping its side effects). The broker delivers at-least-once; your consumer makes it effectively-once. Kafka's transactions and idempotent producer give you effectively-once *within Kafka* (read-process-write where input offsets and output records commit atomically), which is real and useful but is not magic across your whole system — the moment your consumer writes to an external database or calls a third-party API, you are back to needing idempotency on your side. The mechanism details are in [exactly-once in Kafka](/blog/software-development/message-queue/exactly-once-in-kafka-idempotent-producer-transactions); the deeper treatment of the semantics is in [delivery semantics: at-most, at-least, exactly-once](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once).

The senior takeaway is a posture, not a feature flag: **assume at-least-once delivery, and make every consumer idempotent.** If you internalize that one rule, duplicates stop being incidents and become a non-event. I cover the design techniques — natural idempotency keys, dedup tables, upserts, conditional writes — in the companion post on [idempotency and exactly-once by design](/blog/software-development/system-design/idempotency-and-exactly-once-by-design), and the queue-specific dedup tactics in [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe). For now, hold the line: there is no exactly-once delivery, there is effectively-once processing, and you earn it with idempotency.

#### Worked example: choosing the semantic for two features

Take two concrete features and decide.

*Feature A — emitting page-view events for the analytics dashboard.* Volume is roughly 50,000 events per second at peak; each event is a tiny JSON blob; the dashboard shows hourly aggregates with millions of points per bucket. What does losing one event cost? Effectively nothing — one missing page view does not move a bar chart aggregating millions. What does double-counting one cost? Slightly more than losing one, because a duplicate inflates a number, but still negligible at this scale. The cheapest semantic that satisfies the requirement is **at-most-once**: fire-and-forget, no consumer acks, no dedup table, maximum throughput. Building idempotency here would be paying for a guarantee the data does not need. Decision: at-most-once, accept rare loss, save the engineering.

*Feature B — processing payment-capture commands.* Volume is roughly 200 per second; each command captures a previously-authorized charge for a real dollar amount. What does losing one cost? A customer's order ships and you never charged them — direct revenue loss and a reconciliation nightmare. What does processing one twice cost? You double-charge a customer — a chargeback, a furious support ticket, and a trust hit. So you can tolerate *neither* loss nor duplication, which means you need **effectively-once**: at-least-once delivery (so you never drop a capture) plus strict idempotency keyed on the authorization ID (so a redelivered capture command is a no-op the second time). The consumer checks "have I already captured authorization `auth_8f3a`?" against a dedup table inside the same transaction as the capture, and if so, returns success without calling the payment processor again. Decision: at-least-once transport, idempotent consumer keyed on the auth ID, effectively-once outcome.

The same broker can carry both features; the *semantic* is a property of the consumer's design, not the broker's brand.

## 5. The dual-write problem and the transactional outbox

If you remember one pattern from this entire post, make it this one. The dual-write problem is the single most common source of silent data corruption in event-driven systems, and the transactional outbox is the fix. I have seen multi-week debugging sessions that all traced back to a team not knowing this pattern.

Here is the trap. Your order service needs to do two things when an order is placed: write the order row to its database, and publish an `OrderPlaced` event to the broker so everyone else reacts. The naive code does them as two separate operations:

```python
def place_order(order):
    db.insert(order)                    # write 1: the database
    broker.publish("OrderPlaced", order)  # write 2: the broker
```

This is a **dual write** — two independent writes to two independent systems with no shared transaction — and it is broken in both directions. If the process crashes *between* the two lines, the order is in the database but the event was never published: downstream systems never learn the order exists, the customer never gets a confirmation, the warehouse never ships, and nothing errors, so you do not even know it happened until a customer complains weeks later. If you flip the order — publish first, then write the row — and the database write fails, you have published an event for an order that does not exist, and downstreams act on a phantom. There is no ordering of the two writes that is safe, because there is no transaction spanning a database and a message broker. You cannot make them atomic by trying harder. Figure 7 shows the race concretely.

![A two-column comparison contrasting a dual write where the database commit succeeds but the broker publish crashes and the event is lost, against an outbox where the row and outbox entry commit in one transaction and a relay retries publishing](/imgs/blogs/queues-and-event-streaming-for-architects-7.webp)

The **transactional outbox** closes the window by making the event part of the *same database transaction* as the business write. Instead of publishing to the broker inside your request, you insert a row into an `outbox` table in the same database, in the same transaction as the order. Either both the order row and the outbox row commit, or neither does — that is what a database transaction guarantees, and it is the one atomicity you actually have. Then a separate process — a **relay** (also called a message relay or publisher) — reads unpublished rows from the outbox table and ships them to the broker, marking each as published once the broker acks. Figure 4 shows the flow end to end.

![A pipeline showing a database transaction writing both business state and an outbox row, an outbox table of unpublished rows, a relay that polls or tails the write-ahead log, and a broker delivering at-least-once](/imgs/blogs/queues-and-event-streaming-for-architects-4.webp)

The code becomes one transaction plus a background relay:

```python
def place_order(order):
    with db.transaction() as txn:
        txn.insert("orders", order)
        txn.insert("outbox", {
            "id": uuid4(),
            "aggregate": "order",
            "event_type": "OrderPlaced",
            "payload": json.dumps(order),
            "created_at": now(),
            "published": False,
        })
    # both committed atomically, or neither did — no dual write

# separate relay process, runs continuously
def relay():
    while True:
        rows = db.query(
            "SELECT * FROM outbox WHERE published = false "
            "ORDER BY created_at LIMIT 100 FOR UPDATE SKIP LOCKED")
        for row in rows:
            broker.publish(row["event_type"], row["payload"],
                           key=row["id"])            # id as dedup key
            db.execute("UPDATE outbox SET published = true WHERE id = %s",
                       row["id"])
        if not rows:
            sleep(0.2)
```

Note what the outbox guarantees and what it does not. It guarantees that *if the transaction committed, the event will eventually be published* — the relay retries until the broker acks, so a broker outage just means the outbox grows and drains later. It does **not** guarantee exactly-once publishing: if the relay publishes a row to the broker and then crashes before marking it `published = true`, it will publish that row again on restart. The outbox is fundamentally an **at-least-once** publisher. That is fine, because we already decided every consumer must be idempotent — the relay stamps each event with a stable ID (the outbox row's UUID), and consumers dedup on it. At-least-once publishing plus idempotent consumers equals effectively-once processing, end to end.

There are two ways to build the relay, and the choice matters. The **polling** relay (the code above) periodically `SELECT`s unpublished rows. It is dead simple, works on any database, and is the right starting point. Its costs are polling latency (the gap between commit and the next poll — tunable down to tens of milliseconds, up at the price of constant query load) and the write amplification of marking rows published and eventually deleting them. The **change-data-capture (CDC)** relay tails the database's write-ahead log directly — Debezium reading the Postgres WAL or the MySQL binlog — and emits a broker event for every committed change, with no polling and no separate outbox-marking write. CDC gives you lower latency and zero query load, at the cost of operating a CDC pipeline (a Kafka Connect cluster, schema handling, WAL retention tuning) and a less obvious failure model. The full treatment of both, including the WAL-tailing details, is in [change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern), and the publishing-reliability angle is in [the transactional outbox pattern](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing). Start with polling; graduate to CDC when polling latency or load becomes the bottleneck.

#### Worked example: the latency and consistency the outbox buys

Let me put numbers on what the outbox costs you, because "eventual consistency" is too vague to design against.

Suppose the order service commits at time T. With the *naive dual write*, the event would publish at roughly T + 5ms (the broker round-trip) — fast, but with a real probability of total loss on crash, which at, say, 2,000 orders per second and a deploy-or-crash event a few times a day works out to a handful of silently-lost orders per incident. That is the status quo we are fixing.

With a *polling outbox* at a 200ms poll interval, the event publishes at roughly T + 100ms on average (half the poll interval) and T + 200ms worst case, plus the 5ms broker round-trip. So you have added about 100ms of median latency to the *event* — not to the user's response, which still returns at T the instant the transaction commits. The user sees no slowdown; the confirmation email arrives ~100ms later than it theoretically could have, which no human notices. In exchange, the probability of losing the event drops to effectively zero — the only way to lose it now is to lose the committed database transaction itself, which is your database's durability problem, not a messaging problem. You traded ~100ms of event latency for the elimination of a silent-loss class of bug. That is one of the best trades in distributed systems.

If that 100ms matters — say a fraud system needs to react within tens of milliseconds — you move to a *CDC outbox*, which publishes at roughly T + 10–20ms (WAL flush plus CDC processing) with no polling, recovering most of the latency at the cost of running Debezium. And if you need to tighten the consistency window further on the *read* side, that is a different lever (read-your-writes, covered in the [consistency models guide](/blog/software-development/system-design/consistency-models-a-practical-guide-for-architects)), not a messaging one. The outbox's contract is clean: the event is *guaranteed* and *eventually* published, with "eventually" tunable from ~200ms (cheap polling) down to ~15ms (CDC), and the guarantee never weakens as you tune the latency.

## 6. The broker is now tier-0: durability config you must get right

The outbox guarantees the event leaves your database safely; the broker has to not lose it after that, and this is where defaults betray you. The moment a broker carries events that matter, it becomes a **tier-0 dependency** — if it loses a message or goes unavailable, the loss or outage propagates everywhere downstream. Yet the out-of-the-box settings on most brokers favor throughput over durability, so a senior reviews the durability configuration explicitly rather than trusting defaults. There are three knobs that decide whether your broker actually keeps what you give it, and they exist in some form on every broker.

The first is **producer acknowledgement level**. On Kafka this is `acks`, and the three settings are a real durability dial. `acks=0` means the producer does not wait for any acknowledgement — it fires and assumes success, so a broker that drops the message on the floor is invisible to the producer. That is at-most-once at the producer level and is appropriate only for the disposable-data case. `acks=1` means the producer waits for the *leader* replica to write the message, but not the followers — so if the leader crashes after acking but before the followers replicate, the message is gone even though the producer thinks it succeeded. This is the dangerous middle setting that looks safe and is not. `acks=all` (with `min.insync.replicas` set appropriately) means the producer waits until the message is replicated to a quorum of in-sync replicas before considering it durable — so it survives any single broker loss. For anything you ran an outbox to protect, you want `acks=all`; using `acks=1` on a financial event stream is the kind of thing that surfaces as data loss only during a broker failover, which is to say at the worst possible time.

The second is **replication factor and in-sync replicas**. A topic with replication factor 1 has no redundancy — one broker dies and that partition's data is gone. Production topics that matter run replication factor 3 across 3 availability zones, with `min.insync.replicas=2`, which means a write is acknowledged only when at least 2 of the 3 replicas have it, and the system can lose one broker and keep accepting writes (and lose two and reject writes rather than accept un-redundant ones). This is the same quorum logic as a database — covered in [quorums, anti-entropy, and read repair](/blog/software-development/database/quorums-anti-entropy-and-read-repair) — applied to a log. The architect's call is the same trade as any replication decision: more replicas means more durability and more cross-AZ write latency and bandwidth cost; the standard answer for important data is RF=3, ISR=2, and you deviate only with a reason.

The third is the **producer's idempotence and retry behavior**. A producer that retries a failed send (which it must, for at-least-once) can create *duplicate* messages if the original actually succeeded but the ack was lost — the same Two Generals problem, now on the produce side. Kafka's `enable.idempotence=true` makes the producer attach a sequence number so the broker deduplicates retries within a session, giving you exactly-once *produce* semantics into the log (no duplicates from producer retries). This does not extend to your consumers' external side effects — those still need application-level idempotency — but it does mean the log itself does not accumulate produce-retry duplicates. Turn it on; it is nearly free and removes one source of duplication.

Here is the durable-producer configuration a senior ships for an important topic, with the reasoning inline:

```python
producer = KafkaProducer(
    bootstrap_servers=brokers,
    acks="all",                 # wait for the in-sync quorum, not just leader
    enable_idempotence=True,    # no duplicates from producer retries
    retries=10,                 # retry transient failures
    max_in_flight_requests_per_connection=5,  # ok with idempotence on
    linger_ms=5,                # batch for throughput (section 12)
    compression_type="lz4",     # cheaper bandwidth and storage
)
# topic created with: replication.factor=3, min.insync.replicas=2
```

The cost of all this durability is real and measurable: `acks=all` adds the replication round-trip to produce latency (single-digit milliseconds within a region, more cross-region), RF=3 triples storage and replication bandwidth, and `min.insync.replicas=2` means you *reject writes* when you are down to one replica rather than accept un-redundant ones — a deliberate availability-for-durability trade, the [CAP/PACELC choice](/blog/software-development/system-design/articulating-tradeoffs-cap-pacelc-and-beyond) made concrete at the messaging layer. You are choosing consistency and durability over raw availability and latency, which for events that matter is the right choice — but it is a choice, and a senior states it rather than inheriting whatever the defaults happened to be. The failure mode of *not* making this choice is the worst kind: everything works in testing, throughput looks great, and then a routine broker failover six months later silently drops the messages that `acks=1` never really protected.

## 7. Ordering, partitioning, and the guarantees you actually get

Ordering is where async architectures quietly break, because the guarantee people *assume* and the guarantee they *have* are different, and the gap surfaces as a bug six months in. Let me state the real guarantees plainly.

In a **queue** with competing consumers, there is **no meaningful ordering guarantee** at the consumer. The broker may hand messages out in FIFO order, but the moment you have two consumers pulling concurrently, consumer A might finish message 2 before consumer B finishes message 1, so the *effects* land out of order. And on redelivery (the at-least-once retry), a failed message goes to the back of the line and gets processed long after its successors. If you need ordering from a queue, you must either run a single consumer (throwing away parallelism) or use a feature like SQS FIFO queues or RabbitMQ consistent-hash routing that pins related messages to one consumer — which is really the log's partition idea in disguise.

In a **log**, the guarantee is precise and worth memorizing: **strict order within a partition, no order across partitions.** Messages in partition 3 are delivered to consumers in exactly the order they were appended; messages spread across partitions 0–11 have no defined relative order. This is the most important sentence in log-based design. It means ordering is a property you *engineer* by choosing the **partition key** — the field the broker hashes to pick a partition. If you key by `user_id`, then *all* events for a given user land in the same partition and are processed in order, while different users spread across partitions for parallelism. You get per-user ordering and full parallelism at the same time, which is exactly what most systems actually need — you rarely need *global* order, you need order *within an entity*.

This is the same insight as choosing a database partition key (see [partitioning and sharding without downtime](/blog/software-development/system-design/partitioning-and-sharding-without-downtime)): the key co-locates the things that must be ordered together and spreads everything else. Get it wrong and you either lose the ordering you needed (keyed too coarsely, related events scattered) or you create a hot partition (keyed too finely, one celebrity user's events all hammer one partition). The partition is simultaneously the unit of ordering *and* the unit of parallelism, and those two pull against each other — more partitions means more parallelism but smaller ordering scopes, fewer partitions means stronger ordering scopes but a parallelism ceiling. Choosing the partition key and count is the core log-design decision, and the mechanics are in [message ordering and partitioning guarantees](/blog/software-development/message-queue/message-ordering-and-partitioning-guarantees).

One more honest caveat: even with per-partition ordering, *retries* can reorder effects unless you are careful. If message 5 fails and you send it to a dead-letter queue and keep going with message 6, then message 5's effect lands after 6's — or never. Strict per-partition ordering with retries forces a choice: stop-the-partition on a failure (preserves order, but one poison message blocks the whole partition — we will stress-test exactly this in section 9), or skip-and-DLQ (keeps the partition moving, but breaks order for the skipped message). There is no free lunch; you pick which property you need per topic.

## 8. Scaling consumers: partitions, lag, and the parallelism ceiling

Now the operational heart of running a log in production: how you scale consumers, and the one number that tells you whether you are winning or losing — **consumer lag**.

The unit of consumer parallelism in a log is the **partition**, and the rule is hard: **within a consumer group, a partition is consumed by at most one consumer.** So if a topic has 12 partitions, you can run up to 12 consumers in a group working in parallel; a 13th consumer sits idle because there is no partition to give it. This makes partition count your *parallelism ceiling*, fixed at topic-creation time and painful to increase (increasing partitions changes the hashing and breaks per-key ordering for in-flight keys). The senior move is to **over-partition deliberately at creation** — pick a partition count comfortably above your *peak* expected consumer count, often 2–4x, so you have headroom to scale consumers out during an incident without re-partitioning. Partitions are cheap; re-partitioning a live topic is not. Figure 5 shows the multi-group fan-out that makes a log worth its operational cost: each consumer group reads the whole stream independently at its own offset.

![A graph showing an order service publishing to a twelve-partition log that fans out to an email group, a search-index group, and an analytics group, each tracking its own offset with the analytics group running behind](/imgs/blogs/queues-and-event-streaming-for-architects-5.webp)

**Consumer lag** is the difference between the latest offset produced and the offset a consumer group has reached — literally "how many messages behind is this consumer." It is *the* health metric for a streaming consumer, more telling than CPU or memory, because it directly measures whether you are keeping up. Lag of zero means you are processing in real time. Lag that is flat-but-nonzero means you are keeping up with a fixed buffer. Lag that is *growing* means your consume rate is below your produce rate and you are falling behind — and if it keeps growing it will eventually cross your retention window, at which point unread messages are *deleted before you read them* and you have silent data loss. So the alert that matters is not "lag is high," it is "lag is *growing* and its slope, extrapolated, hits the retention window before we can react."

The math is simple and you should do it. If you produce 10,000 messages/second and your consumers process 8,000/second, lag grows at 2,000/second = 120,000/minute. With a 24-hour retention and currently-empty lag, you have `(10,000 × 86,400) / 2,000` ≈ 432,000 seconds — but that is not the right number, because retention is about message age, not count. The real question is: at this growth rate, when does the oldest unread message age past 24 hours? If lag grows at 2,000/sec and your throughput is 10,000/sec, the oldest unread message ages out when the backlog represents 24 hours of production, i.e. `10,000 × 86,400` = 864M messages of backlog. You will alert long before that. The practical rule: **alert on lag growing for more than a few minutes, page when projected time-to-retention-window drops below your time-to-mitigate.**

How do you mitigate growing lag? Three levers, in order of preference: (1) **scale out consumers** up to the partition count — if you have spare partitions, add consumer instances and the group rebalances work onto them; (2) **make the consumer faster** — batch writes to the downstream, increase fetch sizes, remove a synchronous call inside the consumer loop; (3) if you are already at the partition ceiling and still behind, you have a capacity problem that requires re-partitioning or a fundamentally faster consumer. The first lever is why over-partitioning at creation pays off: it is the difference between "add three pods and recover in twenty minutes" and "we are pinned at the ceiling and there is nothing to do but wait."

## 9. Backpressure: what a full queue is telling you

A growing queue or growing lag is not just an operational annoyance; it is *information*. It is the system telling you that, over some window, demand exceeds capacity. The dangerous instinct is to treat the queue as infinite — "it's fine, the broker will hold it" — because that hides the imbalance until it becomes a crisis. The senior instinct is to treat queue depth as a **backpressure signal** and decide, deliberately, what to do when it builds.

There are only a few honest responses to a queue that is filling faster than it drains, and you must choose one *in advance*:

- **Scale the consumer** to drain faster — the right answer when the spike is real demand you want to serve and you have the capacity headroom.
- **Shed load at the producer** — reject or sample incoming work when the queue is over a threshold, so you protect the consumer from a backlog it can never clear. This is the correct answer when the work is droppable (metrics, low-value events): better to drop 5% at the door than to fall hours behind on 100%.
- **Block the producer** (bounded queue, producer waits when full) — propagates the pressure upstream to the true source, which is correct when the producer is itself elastic or when slowing the source is acceptable. An unbounded queue cannot do this and so cannot protect downstream; that is its danger.
- **Let it buffer** — the legitimate use of a queue's depth, *for a bounded spike*. A queue absorbing a 5-minute 10x spike and draining over the next 20 minutes is the system working as designed. A queue absorbing a *sustained* overload is just deferring an outage.

The distinction that separates a buffer from a time bomb is **bounded versus unbounded** and **transient versus sustained**. A bounded queue forces the backpressure decision (it gets full, and now you must shed, block, or scale); an unbounded queue lets you pretend there is no limit until you hit the real one — memory, retention, or a downstream that falls so far behind it is useless. Always run with explicit limits and explicit policies for crossing them. The full treatment of flow control — token buckets, bounded channels, credit-based schemes, TCP-style windows — is in [backpressure and flow control](/blog/software-development/message-queue/backpressure-and-flow-control) and the rate-limiting companion in [rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure). The architect's job is upstream of the mechanism: decide, per queue, what a full queue *means* and what the system should do about it, and encode that as a policy rather than discovering it during an incident.

Figure 8 names the three kinds of decoupling a broker stack buys, because each one is also a place backpressure can be applied: time (the consumer can be offline), space (readers come and go), and rate (the buffer absorbs the mismatch). The rate layer is exactly where backpressure lives.

![A layered stack showing a producer that fires and forgets, then time decoupling so the consumer can be offline, space decoupling so readers are added freely, rate decoupling that buffers peaks, and a consumer draining the backlog](/imgs/blogs/queues-and-event-streaming-for-architects-8.webp)

## 10. Stress-testing the design: three incidents that find your bugs

A design is only as good as how it behaves on its worst day. Let me run the canonical async architecture — producers writing to a partitioned log, consumer groups with at-least-once processing and idempotent consumers, an outbox on the producer side — through the three failures that find the bugs. This is the stress-test discipline a senior applies to *every* messaging design before it ships.

**Incident 1: a slow consumer (lag grows).** A bad deploy doubles your consumer's per-message processing time — a new synchronous call to an external service crept into the consume loop. Now consume rate drops below produce rate and lag starts climbing. Figure 6 walks the timeline: at T+0 the consumer's p99 doubles, by T+5m lag is growing at 8,000/minute, by T+40m there is a 320,000-message backlog representing about an hour of unprocessed events, and the on-call decision is whether to scale out, speed up, or roll back. If you over-partitioned (section 7), you scale consumers out and the backlog drains; if you are at the ceiling, you roll back the slow deploy and let it catch up. The architecture *survives* this gracefully precisely because the backlog is absorbed rather than erroring back to users — but only if your lag alert fired while there was still time to act. The lesson the timeline teaches: monitor lag *slope*, not just level, and keep partition headroom so "scale out" is an available move.

![A timeline of a consumer lag incident where the consumer p99 doubles, lag grows at eight thousand per minute, reaches a backlog of one hour, then drains to zero after adding partitions and consumers](/imgs/blogs/queues-and-event-streaming-for-architects-6.webp)

**Incident 2: a poison message.** One message in the stream cannot be processed — a malformed payload, a schema the consumer does not understand, a referenced entity that no longer exists. The consumer tries it, throws, does not ack, and the broker redelivers it. The consumer tries it again, throws again, redelivers again. If you are preserving per-partition order with stop-on-failure, that one message now **blocks its entire partition forever** — every message behind it on that partition is stuck, and lag on that partition climbs to the moon while the other partitions are fine. This is the classic "one bad message took down a third of our throughput" incident. The fix is a **dead-letter queue (DLQ)** with a retry budget: try the message a bounded number of times (say 5, with backoff), and if it still fails, move it to a separate DLQ topic and *advance past it* so the partition keeps flowing. The poison message is now quarantined for a human to inspect, the partition is unblocked, and you have traded strict ordering (the poison message is skipped) for liveness (the partition keeps moving). You must decide per topic which you need; for most topics, liveness wins and the DLQ is mandatory. A messaging design without a DLQ strategy is a design with a single point of total stall, and a senior will flag its absence in review.

**Incident 3: a downstream outage (does the queue absorb it?).** The search-indexing service — one of several consumer groups on the `orders` log — goes fully down for 30 minutes during a deploy gone wrong. What happens? In the *log* model, beautifully little: the search consumer group simply stops advancing its offset, its lag grows for 30 minutes, and the *other* consumer groups (email, warehouse, analytics) are completely unaffected because they read independently at their own offsets. When search comes back, it resumes from its last committed offset and processes the 30-minute backlog — every order it missed is still in the log, in order, waiting. Nothing was lost, no other consumer noticed, and the producer never knew. *This is the resilience we paid for.* Contrast it with the synchronous design we opened with, where the search service being down for 30 minutes would have meant 30 minutes of failed checkouts. The queue *absorbed* the outage. The only thing you must verify is that your retention window comfortably exceeds your worst-case downstream-outage duration — if search had been down for *eight days* against a seven-day retention, the oldest unread orders would have aged out and been lost. Retention sizing is therefore a *reliability* decision, not just a cost one: it is the maximum downstream outage you can survive without data loss.

Run every messaging design through these three. If you cannot answer "what happens to lag, to the partition, and to the other consumers" for each, the design is not done.

## 11. Event-driven vs request-driven, and choreography vs orchestration

Stepping up from individual queues to system shape: introducing async messaging is often a step toward an **event-driven architecture**, and there is a real architectural fork inside it that you should make consciously rather than drift into.

**Request-driven** (also command-driven) systems work by one service *telling* another to do something: "charge this card," "reserve this inventory." The caller knows who it is calling and what it wants done, and usually waits for a result. It is direct, easy to trace, and easy to reason about — the control flow is explicit in the code. Its cost is coupling: the caller must know the callee exists, be deployed compatibly with it, and tolerate its latency and failures.

**Event-driven** systems work by services *announcing* that something happened — "an order was placed" — without knowing or caring who reacts. The producer emits a fact; consumers decide independently what to do with it. This is maximally decoupled (the producer has zero knowledge of consumers) and maximally extensible (new consumers attach without touching the producer), which is the whole appeal. Its cost is that the control flow is now *implicit* — distributed across the event subscriptions — so understanding "what happens when an order is placed" means tracing every consumer of `OrderPlaced`, and there is no single place that holds the business process. Debugging an event-driven flow is genuinely harder; you trade explicit-and-coupled for flexible-and-diffuse.

When a business process spans *multiple* services — place order, take payment, reserve stock, ship — you face the **choreography vs orchestration** decision, and it maps directly onto event-driven vs request-driven.

**Choreography**: each service listens for the prior event and emits the next, with no central brain. The order service emits `OrderPlaced`; the payment service hears it and emits `PaymentTaken`; the stock service hears *that* and reserves inventory and emits `StockReserved`; shipping hears that and ships. The process *emerges* from the chain of events. Figure 9 shows it, including the failure paths: if payment is declined or stock runs out, the failing service emits a *compensation* event that unwinds the prior steps (refund the payment, release the hold). This is a **saga** implemented by choreography. It is beautifully decoupled and has no single point of failure, but the process logic is smeared across five services and there is no one place to look to understand or change it.

![A choreography graph where the order service emits an event the payment service reacts to, then the stock service, branching to shipping when in stock or to a compensation step when payment is declined or stock is unavailable](/imgs/blogs/queues-and-event-streaming-for-architects-9.webp)

**Orchestration**: a central coordinator (an orchestrator or a workflow engine like Temporal, AWS Step Functions, or Netflix Conductor) explicitly drives the steps — it calls payment, waits, calls stock, waits, calls shipping, and on any failure it runs the compensations in order. The process logic lives in *one* place you can read, version, and debug, at the cost of a central component that every saga flows through and that you must keep highly available.

The senior heuristic: **choreography for simple, stable, 2–3-step flows where decoupling matters most; orchestration for complex, evolving, many-step business processes where you need visibility, explicit compensation, and a single place to change the logic.** Choreography that grows past three or four steps becomes an undebuggable web of events where no one can answer "why didn't this order ship," and that is the signal to introduce an orchestrator. The full saga treatment — compensations, isolation anomalies, the semantics of distributed rollback — is in [the saga pattern for distributed transactions](/blog/software-development/database/saga-pattern-distributed-transactions); here the architectural call is just *which* coordination style fits the flow's complexity.

## 12. Migrating from sync to async without a rewrite

You rarely get to design async from scratch; usually you are *extracting* it from a synchronous monolith that has hit a wall, and doing that without a big-bang rewrite is its own skill. The pattern that works is **strangler-style incremental extraction**, one side effect at a time.

Start by identifying a single synchronous side effect that does not need to happen in-band — say, the confirmation email currently sent inline in the checkout handler. Step one: introduce the broker and a consumer for that one event, and have the handler *both* send the email synchronously (as today) *and* publish the event — a temporary dual path, with the consumer's email-sending disabled or sending to a test address. This lets you validate that events flow and the consumer works under real traffic with zero user impact, because the real email still goes out the old way. Step two: flip the consumer to actually send the email and remove the synchronous send from the handler, ideally behind a feature flag and ramped 1% → 10% → 100% so you can watch error rates and roll back instantly. Step three: the handler no longer sends email inline; its p99 drops by the email provider's latency, and a slow email provider can no longer fail checkout. Repeat for the next side effect. Each extraction is small, independently shippable, and reversible.

Two failure modes to watch during migration. First, **double side effects** during the dual-path phase — if both the old synchronous path and the new consumer send the email, customers get two emails. Mitigate by keeping the consumer's effect disabled or idempotent (dedup on order ID) until you cut over, so even an accidental double-fire is a no-op. Second, **the dual-write trap reappears** the moment the handler does "commit the order, then publish the event" as two operations — which is exactly the bug from section 5. So the migration should introduce the outbox *at the same time* it introduces the first event, not bolt it on later. Extract async and add the outbox together; they are the same project.

A note on cost and timeline, because migrations get sold as quick and never are: extracting a handful of side effects from a critical path, with the broker stood up, the outbox added, monitoring and DLQs in place, and each cutover ramped safely, is a multi-week effort for one service even when it goes well. Budget for the operational learning curve — your first lag incident, your first poison message, your first rebalance storm — because the team will hit all three in the first quarter of running a broker, and those are tuition, not failures.

## 13. The optimization lens: throughput, parallelism, and peak-shaving

Once the architecture is right, the optimization questions are about getting more throughput per dollar and lower latency per message, and they cluster into three levers with concrete, measurable wins.

**Batch for throughput.** The single biggest throughput lever in messaging is batching — both on the producer (accumulate messages for a few milliseconds and send them as one batched request) and on the consumer (fetch and process many messages per poll, and *write to the downstream in batches*). The reason is fixed per-operation overhead: a network round-trip, a syscall, a transaction commit each cost roughly the same whether they carry one message or a thousand, so amortizing that fixed cost across a batch can lift throughput 10–100x. A consumer that does one database `INSERT` per message might sustain 2,000 messages/second; the same consumer batching 500 messages into one multi-row `INSERT` per transaction can sustain 50,000+ — a 25x win from a one-line change to the write. Kafka producers expose `linger.ms` (how long to wait accumulating a batch) and `batch.size` (max batch bytes) for exactly this trade; raising `linger.ms` from 0 to 5–10ms adds a few milliseconds of latency and can multiply throughput. The measurement that proves the win is messages/second per consumer instance and the downstream's write QPS; the cost you are spending is a few milliseconds of per-message latency, which is free for background work.

**Partition for parallelism.** As established, partitions are the parallelism unit, so throughput scales (up to a point) with partition count × per-consumer rate. The optimization is to *measure* your per-consumer sustainable rate, divide your peak produce rate by it to get the consumer count you need, and provision partitions at 2–4x that for headroom. If one consumer sustains 8,000 messages/second and your peak is 60,000/second, you need 8 consumers, so create the topic with ~24–32 partitions. Under-partition and you hit the ceiling under load with no recourse; over-partition wildly and you pay in per-partition overhead (each partition is open files, replication traffic, and a slice of broker memory) and tiny batches. The sweet spot is "enough for peak plus incident headroom, not more."

**Size for peak-shaving.** The queue's depth is a capacity-amortization tool: by sizing the buffer to absorb your peak-to-average ratio, you provision consumers for the *average* load instead of the *peak*, which is often a 5–10x cost reduction on the consumer fleet. If your peak is 10x your average but lasts only minutes, a consumer fleet sized for 1.5x average plus a queue that buffers the spike serves the same traffic as a fleet sized for 10x peak — at roughly a sixth of the consumer cost. The measurable win is consumer-fleet \$ cost at equal end-to-end SLA; the cost you accept is added end-to-end latency during the spike (the buffer drains over minutes), which is the whole point of choosing async for that work. This is the load-leveling benefit from section 1, quantified: you are trading a bounded latency increase during spikes for a large, permanent reduction in steady-state capacity.

The meta-point: in messaging, the bottleneck is almost never the broker — Kafka and friends push millions of messages per second per cluster. The bottleneck is the **consumer's per-message work**, usually a downstream write. So optimization effort belongs on the consumer's write path (batch it), the parallelism (partition for it), and the capacity model (peak-shave with the buffer), in that order. Measure messages/second per consumer, lag slope, and consumer-fleet cost; those three numbers tell you whether your optimizations are working.

## 14. Case studies: what production taught

Concrete systems make the trade-offs real. Here are three drawn from how large engineering organizations have publicly described their architectures, with the lesson each teaches — stated at the level of pattern, not at a level of specific internal numbers I cannot verify.

**An outbox rollout at a payments company.** Payments companies live and die by not losing or duplicating financial events, and the outbox is essentially table stakes there. The pattern publicly associated with companies like Stripe and the broader fintech world is to treat the database transaction as the source of truth and derive all downstream events from it — the event is *never* published as a separate write that could diverge from the ledger. The lesson is the one from section 5, sharpened by stakes: in any system where the database and the event stream must never disagree, the event must be a *consequence* of the committed transaction (via outbox or CDC), never an independent second write. Teams that learn this the hard way learn it through a reconciliation discrepancy that takes weeks to root-cause back to a dual-write race. The outbox is cheap insurance against an expensive class of bug, and the senior move is to adopt it *before* the first incident, not after.

**A consumer-lag incident at scale.** The lag-incident shape in section 9 is universal, and large streaming shops — LinkedIn (which created Kafka), Uber, and others that run Kafka at enormous scale — have all built their operational practice around lag as the primary health signal. The publicly discussed lesson is consistent: lag monitoring with slope-based alerting and generous partition headroom is what turns a "slow consumer" from an outage into a shrug. Organizations that alert only on absolute lag level get paged too late (lag is already huge) or too often (a benign buffer trips the threshold); the ones that alert on *growth that projects to retention loss* catch the real incidents with time to act. The architectural takeaway is that over-partitioning at topic creation is not premature optimization — it is the difference between having a mitigation lever during an incident and not having one.

**A sync-to-async migration in a monolith.** Many companies that started as synchronous monoliths — the archetypal mid-2010s Rails or Django shop — eventually extracted async messaging exactly the way section 11 describes: one side effect at a time, behind flags, with the outbox introduced alongside the first event. Shopify and similar high-traffic commerce platforms have publicly described moving heavy, non-critical work (emails, webhooks, search indexing, analytics) off the synchronous request path onto job queues and event streams, dropping checkout latency and decoupling checkout availability from the availability of a dozen downstreams. The lesson is the migration discipline: incremental extraction with reversible cutovers beats a big-bang rewrite every time, and the win shows up immediately as a lower, more stable p99 on the critical path because it no longer waits for the slowest downstream. The trap they warn about is the same dual-write race — extract the async *and* add the outbox in the same change, never separately.

The common thread across all three: the architecture decisions (outbox, log-with-headroom, incremental extraction) matter more than the technology choices (which broker), and the failures that hurt are the ones where a team skipped a known pattern — the outbox, lag-slope alerting, the DLQ — and rediscovered why it exists during an incident.

## 15. Trade-offs, restated as decisions

Pulling the threads together into the decisions a senior actually defends in a design review. Every one of these is a place to name the cost, not just the benefit.

| Decision | Choose this | When it wins | What you pay |
| --- | --- | --- | --- |
| Sync vs async | Synchronous | You need an answer in the request | Coupled latency + availability |
| | Async (broker) | Work can happen seconds later | Eventual consistency + ops burden |
| Queue vs log | Queue | Fire-once commands, one worker | No replay, weak ordering |
| | Log | Multi-reader events, replay, per-key order | High ops cost, fixed partition ceiling |
| Delivery semantic | At-most-once | High-volume disposable data | Silent loss on crash |
| | Effectively-once | Anything you'd be paged about | Idempotency engineering everywhere |
| Publishing | Dual write | Never | Silent event loss on crash |
| | Outbox (polling) | Default reliable publishing | ~100ms event latency |
| | Outbox (CDC) | Need low-latency events | Run a CDC pipeline |
| Coordination | Choreography | Simple 2–3 step flows | Diffuse, hard-to-trace logic |
| | Orchestration | Complex evolving processes | A central component to keep up |
| Poison handling | Stop-on-failure | Strict ordering required | One bad message blocks a partition |
| | DLQ + skip | Liveness over strict order | Skipped message needs manual handling |

The discipline this table encodes: never adopt a capability without naming what it costs, and always prefer the least-powerful option that meets the requirement. A synchronous call beats a queue when you need an answer now; a queue beats a log when you have one consumer and no replay need; at-most-once beats effectively-once when the data is disposable. Power you do not use is operational cost you pay forever.

## 16. When to reach for async messaging (and when not to)

A decisive recommendation, because "it depends" is not an answer a senior gives in a review.

**Reach for a queue when** you have work that one worker should do once and that does not need to happen in the request — sending emails, transcoding media, generating reports, processing webhooks — and you want the operational simplicity of a managed, shrinking backlog. Start with the simplest managed option (SQS, or your cloud's equivalent); you can always graduate.

**Reach for a log when** an event has *multiple* independent consumers, when you need *replay* (to rebuild downstreams, seed new services, or recover from a bug), or when you need *strict per-key ordering* at scale. The presence of a second real consumer, or a concrete replay need you have already hit, is the signal — not a hypothetical "we might."

**Reach for the outbox whenever** you publish an event as a consequence of a database write, which is almost always. There is no good reason to do a raw dual write in new code; the outbox is the default, and the only decision is polling versus CDC.

**Do NOT reach for async when** the caller needs the result to proceed (an auth check, a balance lookup, a synchronous validation) — async there just adds a broker and an eventual-consistency window to a problem that wanted a function call. Do not reach for a *log* when you have a single fire-once consumer and no replay need — you are paying a log's operational tax for a queue's job. Do not introduce a broker at all to "decouple" two components that are deployed together, change together, and have no independent failure or scaling need — that is complexity cosplaying as architecture. And do not adopt async without a plan for its four costs: idempotent consumers (for duplicates), partition-key choice (for ordering), lag alerting and DLQs (for the operational burden), and an explicit consistency story (for the eventual-consistency window). If you cannot staff the operational burden, a synchronous design you can actually run beats an async design you cannot.

The honest meta-rule: async messaging is a powerful tool that solves real problems — decoupling, load-leveling, fan-out, resilience — and charges a real, ongoing tax in complexity and operations. Introduce it when the problems it solves are problems you actually have and the tax is one you can afford to pay. Reaching for it reflexively because event-driven sounds modern is how teams end up with a distributed monolith that has all the coupling of synchronous calls *and* all the complexity of async.

## Key takeaways

- **Add a broker for a reason, not a slogan.** The four real reasons are decoupling in time, load-leveling, fan-out, and resilience. The four real costs are eventual consistency, duplicates, ordering complexity, and operational burden. Name both before you recommend it.
- **Queue or log is the first decision.** A queue distributes fire-once work and forgets it; a log replays an ordered stream to many independent readers. Running a single-consumer log, or wishing your queue could replay, means you chose wrong.
- **There is no exactly-once delivery.** Assume at-least-once and make every consumer idempotent; that turns duplicates from incidents into non-events and gets you effectively-once processing.
- **The transactional outbox is non-negotiable for reliable publishing.** Never do a raw dual write. Write the event into the same transaction as the state, and let a relay (polling or CDC) ship it at-least-once. The cost is ~100ms of event latency; the benefit is eliminating a whole class of silent data loss.
- **Ordering is engineered, not free.** A log gives strict order within a partition and none across; you get per-entity ordering and parallelism together by choosing the partition key well.
- **Consumer lag slope is the health metric.** Alert on lag *growing*, page when projected time-to-retention-window drops below time-to-mitigate, and over-partition at creation so "scale out" is always an available move.
- **A full queue is a backpressure signal.** Decide in advance — scale, shed, block, or buffer — and run with bounded queues so the system forces the decision instead of hiding the imbalance.
- **Stress-test every design against three failures**: a slow consumer (does lag drain?), a poison message (does a DLQ unblock the partition?), and a downstream outage (does the queue absorb it within retention?). If you cannot answer all three, the design is not done.
- **Choose coordination by complexity**: choreography for simple stable flows, orchestration for complex evolving processes that need a single place to read and change the logic.
- **Prefer the least-powerful option that meets the requirement.** Sync beats a queue when you need an answer now; a queue beats a log when you have one consumer and no replay; power you do not use is operational cost forever.

## Further reading

- [Anatomy of a message system: producers, brokers, consumers](/blog/software-development/message-queue/anatomy-of-a-message-system-producers-brokers-consumers) — the mechanism layer under this whole post.
- [Delivery semantics: at-most, at-least, exactly-once](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) — the deep treatment of the guarantees from section 4.
- [The transactional outbox pattern for reliable publishing](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing) — the outbox in full mechanical detail.
- [Change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — the CDC relay, WAL-tailing, and Debezium specifics.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — how a log is actually stored and served.
- [The saga pattern for distributed transactions](/blog/software-development/database/saga-pattern-distributed-transactions) — compensations and distributed rollback in full mechanical detail.
- [Idempotency and exactly-once by design](/blog/software-development/system-design/idempotency-and-exactly-once-by-design) — how to build the idempotent consumers this post depends on.
- [Rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure) — the flow-control mechanisms behind section 8.
- [Designing Data-Intensive Applications](https://dataintensive.net/) by Martin Kleppmann — chapters 11 (stream processing) and the dual-write/outbox discussion are the canonical references.
- The official [Apache Kafka documentation](https://kafka.apache.org/documentation/) on delivery semantics, consumer groups, and exactly-once.
