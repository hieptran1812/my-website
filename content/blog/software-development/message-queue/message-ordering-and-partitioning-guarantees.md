---
title: "Message Ordering and Partitioning: The Guarantees You Actually Get"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Ordering is the most misunderstood guarantee in messaging. Learn why total order needs a single serial point that caps throughput, why partitioning only buys you per-partition and per-key order, and the exact mechanisms — producer retries, in-flight requests, multiple writers, consumer concurrency, rebalances — that silently break the order you thought you had."
tags:
  [
    "message-queue",
    "ordering",
    "partitioning",
    "kafka",
    "rabbitmq",
    "distributed-systems",
    "event-driven",
    "idempotency",
    "stream-processing",
    "consistency",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/message-ordering-and-partitioning-guarantees-1.webp"
---

Here is a bug I have watched ruin three different on-call weeks at three different companies, and it always wears the same costume. A user updates their email address, then a half-second later updates it again to fix a typo. Two events go onto the bus: `email -> alice@new.com` and then `email -> alice@correct.com`. Downstream, the profile service applies them. The user's email ends up set to `alice@new.com` — the *old*, wrong value. The events were produced in the right order. They were even consumed by the right service. And yet the later event lost to the earlier one, the customer's mailbox is wrong, and the engineer staring at the logs swears the system is haunted, because in the broker the two records are sitting right there in the correct sequence.

The system is not haunted. The engineer made an assumption that no message broker on earth actually promised: that "the order I produced messages in" equals "the order they get processed in," globally, for free. That assumption is wrong in a way that is both subtle and absolutely fundamental, and almost every production ordering bug I have ever debugged is a variation on someone believing it. Ordering is the single most misunderstood guarantee in all of messaging, and the gap between what people assume and what brokers promise is where the bugs live.

This post is going to make the gap explicit and then teach you to design inside it. The core truth, the one sentence to tattoo on your forearm, is this: **total ordering requires a single serial point, and a single serial point caps your throughput.** The moment you parallelize — the moment you split a topic into partitions or spin up a second consumer — you give up global order and get something weaker in exchange for scale. What you get is *per-partition* order, and, if you partition by a key, *per-key* order: all the messages for one user, one account, one order arrive at one consumer in the sequence they were produced, while different keys flow in parallel. The figure below is the whole argument in one image: a single serial log that orders everything but cannot scale, beside a keyed-partition layout that orders each entity perfectly while running many consumers at once.

![A two-panel comparison showing a single partition that totally orders every message but runs one consumer with no parallelism on the left, and a partitioned-by-key layout that preserves per-key order across many parallel consumers on the right](/imgs/blogs/message-ordering-and-partitioning-guarantees-1.webp)

By the end you will be able to do four things. You will be able to state precisely what "ordered" means and what nobody actually promises you. You will be able to choose a partition key that matches your ordering domain, so the entities that must stay in order do, while everything else scales out. You will know the exact mechanisms that silently break ordering even after you have done the partitioning right — producer retries that race past each other, in-flight request windows, multiple writers to one key, consumer-side thread pools, and rebalances — and you will know the specific config knobs that close each hole. And you will be able to test for out-of-order processing instead of discovering it from an angry customer. This is the foundational post on ordering in the series; it forward-links to [partitioning capacity planning](/blog/software-development/message-queue/partitioning-capacity-planning), the Kafka deep-dives, the idempotency post, and the existing [database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding) write-up, all of which build on the ideas here.

## 1. What "ordered" even means (and what nobody promises you)

Before we can talk about guarantees we have to be brutally precise about the word "order," because it is doing too much work in casual conversation. There are at least three distinct orderings in any message-passing system, and they routinely disagree.

There is **production order**: the sequence in which a producer called `send()`. There is **storage order**: the sequence in which records physically sit in the log or queue. And there is **processing order**: the sequence in which a consumer actually applies the side effects. The naive mental model assumes all three are the same. The whole craft of ordering is managing the places where they diverge.

Consider production order first. If a single thread calls `send(msg1)` then `send(msg2)`, you might assume msg1 is "before" msg2. But `send()` in every high-throughput client is asynchronous: it enqueues the record into a buffer and returns immediately. The two records then race through batching, the network, retries, and broker acceptance. Production order — the order of the `send()` calls — does **not** by itself determine storage order. We will spend a whole section on exactly how those two come apart.

Storage order is the one brokers are actually built to give you, but only within a unit. In Kafka that unit is the partition: within a single partition, records have monotonically increasing offsets and are stored, replicated, and read back in that exact sequence, forever. Across partitions there is no order at all — partition 3's offset 100 and partition 7's offset 100 have no defined relationship. In RabbitMQ the unit is a single queue with a single active consumer: messages are dequeued in roughly FIFO order, but the instant you add a second consumer or a competing channel, "roughly" stops being "strictly." There is no broker that gives you free, global, total order across a parallel system, because — and this is not an implementation limitation, it is a theorem about distributed systems — total order *requires* funneling everything through one serial point.

Processing order is where the customer actually feels the consequence, and it is the loosest of the three. Even if storage order is perfect, a consumer that hands records to a thread pool, or commits offsets out of band, or retries a failed handler, can apply effects in a different order than it read them. Many teams get the storage layer perfectly ordered and then scramble the order in their own consumer code.

So the honest statement of what a broker promises is narrow. Kafka promises: **within one partition, records are delivered to a consumer in offset order.** That is it. It does not promise cross-partition order. It does not promise that production order equals storage order unless you configure the producer correctly. And it certainly does not promise that your consumer processes in order if your consumer is concurrent. RabbitMQ promises even less by default: a single queue is FIFO-ish, but redelivery, multiple consumers, and priorities all perturb it. Every stronger guarantee you want, you build on top of these narrow primitives — and the first and most important tool you build with is the partition.

### The three orderings rarely line up

A useful discipline is to label, for any feature, which of the three orderings actually matters for correctness. For an audit log, storage order is the contract — you replay the log and it must read back as it was written. For a state machine like "account balance," processing order is the contract — the debits and credits must apply in sequence or the balance is wrong. For analytics counting "events per minute," none of the three may matter at all, because addition is commutative and you can process events in any order and get the same count. Knowing which ordering your feature actually requires is the single most important design decision, because it tells you how much ordering you need to pay for — and ordering, as we are about to see, is expensive.

### "Ordered" is also relative to a scope, not absolute

There is one more dimension of imprecision to drain out of the word, and it is the one this whole post pivots on: *ordered with respect to what?* "These messages are ordered" is meaningless until you say *which* messages are ordered relative to *which others*. Total order claims every message is ordered relative to every other message. Per-partition order claims a message is ordered only relative to other messages *in the same partition*. Per-key order claims a message is ordered only relative to other messages *with the same key*. None of these is more "correct" than the others — they are different *scopes*, and the right scope is the one that matches your correctness requirement. When someone says "the messages came out of order," the first diagnostic question is always "out of order relative to what guarantee did you actually have?" Nine times out of ten the answer is that they assumed total order from a system that only ever promised per-partition order, and the messages were perfectly in order with respect to the guarantee that actually existed. The bug is in the expectation, not the broker. Internalizing that ordering is *scoped* — never absolute — is the conceptual leap that the rest of this post cashes out into concrete design rules.

### Causal order, the middle ground you rarely get for free

Between total order and no order sits *causal* order: the guarantee that if event A *caused* event B, then everyone sees A before B, but unrelated events may appear in any order. Causal order is often what you *actually* want — you do not care whether Alice's and Bob's edits are globally ordered, only that the edits *within* a single document or thread respect their cause-and-effect chain. The catch is that message brokers do not track causality for you; they track *position in a log*. So you approximate causal order by *co-locating causally related events on the same key* — make the document ID the key, and all edits to that document share a partition and thus an order, capturing the causal chain you care about. When causality crosses key boundaries, the broker cannot help and you fall back to application-level versioning. The mental model to hold is that keyed partitioning is a *tool for encoding causal scope into the partition assignment*: you decide what is causally related (the ordering domain) and the key makes the broker preserve order exactly over that scope. The [consistency models](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) post places causal order precisely on the spectrum between linearizable and eventual.

## 2. Total order needs a single serial point (and it costs throughput)

Let us derive the central constraint from scratch, because once you feel it in your bones you will stop expecting brokers to do the impossible.

What does it mean for messages to be *totally* ordered? It means there is a single, agreed-upon sequence: message A is before B, B is before C, and every participant in the system would agree on that exact line-up. For that to be true, something somewhere has to *assign* the sequence numbers, and it has to assign them one at a time, in a single line, with no two messages ever getting the same number or an ambiguous relationship. That "something" is a serial point: one log, one queue, one leader, one counter that increments under a lock. Everything that needs to be totally ordered has to pass through it, single-file.

This is not a Kafka fact or a RabbitMQ fact; it is a property of ordering itself. To have a total order you need a total function from messages to positions, and to build that function without contradictions across a distributed system you need consensus on the sequence — which, in practice, means routing through one authority. You can make that authority highly available with replication, but you cannot make it *parallel* without breaking the very thing that made it a total order. The serial point is the order.

And a serial point caps throughput. One partition in Kafka is one append-only log owned by one leader broker; all writes for that partition land on that one broker's disk, are replicated from that one leader, and are consumed by at most one consumer per group. However fast that single leader's disk and network are — and modern Kafka can push a single partition to tens of megabytes per second and tens of thousands of small messages per second — that is your ceiling. You cannot add a second machine to make *that partition* faster, because a second machine would mean two serial points, which means no total order. The same logic governs a single RabbitMQ queue: one queue with one consumer is your serial point, and its throughput is bounded by that one consumer's rate.

So total order and horizontal scalability are in direct tension, by definition, not by accident. If you genuinely need every message in the system globally ordered, you accept a single-partition ceiling and you size that one partition carefully. The good news, which the rest of this post develops, is that you almost never need *global* total order. You need order *within an entity*, and that requirement is dramatically cheaper to satisfy.

#### Worked example: the cost of insisting on total order

Suppose you run a payments ledger that emits 200,000 events per second at peak, each event about 500 bytes. That is 100 MB/s of raw event throughput. If you insist on total order across the entire ledger, you must funnel all 200,000 events per second through one partition. A single well-tuned Kafka partition on good hardware might sustain, say, 30–40 MB/s of replicated throughput with `acks=all` before the single leader's disk or the single in-sync replica's network becomes the bottleneck. You are asking for 100 MB/s through a 35 MB/s pipe. It does not fit. You will either drop the durability (lower acks), accept enormous producer-side buffering and latency, or — the only real answer — relax the ordering requirement.

Now ask the honest question: does the *ledger* need total order, or does each *account* need its own order? Almost always it is the latter. Account 12345's debits and credits must apply in sequence; account 12345 has no ordering relationship with account 67890 at all — they are independent state machines. The moment you accept that, you can split the 100 MB/s across, say, 12 partitions keyed by account, each carrying roughly 8 MB/s, every account perfectly ordered within its partition, and the aggregate throughput is now a non-issue. The cost of total order was a 65 MB/s shortfall; the cost of per-account order is a one-line change to your partition key. That asymmetry — global order is brutally expensive, per-entity order is nearly free — is the most important economic fact in this entire domain.

### Why you cannot "cheat" the serial point

Engineers who first meet this constraint reach for clever escapes, and it is worth walking through why each one fails, because the failures are instructive. "Can I have two partitions but a sequencer that merges them in order?" — yes, but the sequencer is now your serial point, and it has the same throughput ceiling; you have just moved the bottleneck and added a hop. "Can I assign global sequence numbers from a fast atomic counter and let consumers reorder?" — the counter is the serial point (every message must touch it to get a number), and now consumers must *buffer and wait* for missing sequence numbers, which means a single slow message stalls the whole stream and you have reinvented head-of-line blocking. "Can I use timestamps for global order?" — wall-clock timestamps across machines are not totally ordered (clocks skew, and two events can share a millisecond), so timestamp order is only an *approximation* of total order, and approximations are exactly what cause the email-revert bug. There is no free lunch. Any mechanism that produces a genuine total order contains a serial point somewhere, and that serial point is your throughput ceiling. The engineering question is never "how do I get total order cheaply" — it is "how do I narrow the scope so I need total order over a *small* domain." That narrowing is what partitioning by key accomplishes.

### Total order has a latency cost too, not just throughput

The throughput ceiling is the famous cost, but a serial point also imposes a *latency* cost that bites under bursty load. Because every message must pass through one log, a momentary burst that exceeds the single partition's drain rate forms a queue, and that queue is strictly FIFO — so a message that arrives during the burst waits behind *every* message ahead of it, with no possibility of a faster lane. In a partitioned system, a burst on one key does not delay another key's messages at all, because they are on different partitions draining in parallel. So total order does not just cap your sustained throughput; it couples the latency of every message to the backlog of every other message, turning any localized hotspot into a global latency event. This is one more reason the per-entity scope is so valuable: it isolates not just throughput but tail latency between unrelated entities.

## 3. Per-partition order: the real guarantee

Since total order is too expensive and usually unnecessary, the actual workhorse guarantee is **per-partition order**, and it is worth understanding exactly what it does and does not say.

A partition is a single append-only log. Records get an offset — 0, 1, 2, 3 — assigned by the partition leader at the moment of append, monotonically, with no gaps from the consumer's point of view. A consumer reading that partition reads offset 0, then 1, then 2, in that order, always. This is the bedrock. It holds across consumer restarts (you resume from your committed offset), across leader failover (the new leader has the same log, replicated), and across time (the log is immutable; offset 5 is always the same record). Within the partition, storage order is a hard, durable contract.

What per-partition order does *not* say is anything at all about other partitions. If your topic has 6 partitions, you have 6 independent ordered logs. A record in partition 0 at offset 1000 and a record in partition 4 at offset 1000 have no defined temporal relationship; one may have been produced an hour before the other. If you consume all 6 partitions and interleave their records by reading round-robin, the resulting stream is **not** in production order and never was promised to be. People build a consumer that subscribes to a topic, reads from all partitions, and then are shocked that records appear "out of order" — but they were never in order across partitions in the first place. The order exists only inside each partition.

This is why the single most consequential decision in a Kafka design is **how records map to partitions**, because that mapping decides which records share an ordering guarantee. Two records in the same partition are ordered relative to each other. Two records in different partitions are not. Therefore: whatever must stay ordered must land in the same partition. That sentence is the whole game, and the mechanism for making it happen is keyed partitioning.

### How Kafka partitions and RabbitMQ queues differ on ordering

It is worth pausing on the fact that "the unit of ordering" is broker-specific, because the differences shape your design. In Kafka the unit is the *partition*, and it is a persistent, replayable log: order is a property of stored offsets that survives consumption (the record stays in the log after you read it, up to retention). In RabbitMQ the unit is a *queue*, and a classic queue is a transient buffer: a message is delivered and then, on ack, *removed*. Order in RabbitMQ is therefore a property of the delivery sequence, and it holds only under specific conditions — one queue, one consumer, no redelivery, no priorities. Add a second competing consumer and RabbitMQ load-balances messages across them, so two messages that were FIFO in the queue can be processed concurrently and finish in either order. Turn on consumer prefetch (multiple unacked messages per consumer) and even a single consumer can have several messages in flight that its own concurrency may reorder.

The practical upshot: in Kafka, ordering scales by *adding partitions* and is preserved by *one consumer per partition*; in RabbitMQ, strict order requires *one queue per ordering domain with a single consumer*, which is why RabbitMQ shops that need per-entity order often run a queue-per-shard topology (a consistent-hash exchange routing each key to its own queue) — structurally the same as Kafka's keyed partitions, just expressed with exchanges and queues. Both brokers obey the same underlying law: order lives inside one serial unit, and parallelism comes from having many such units, one per ordering domain. The vocabulary differs; the physics does not. The [RabbitMQ production architecture and scaling](/blog/software-development/system-design/rabbitmq-production-architecture-scaling) post details the consistent-hash-exchange pattern.

### Per-partition order survives the things you fear, within limits

Per-partition order is robust against a surprising number of failures, which is why it is the guarantee worth building on. A broker can crash; the partition's leader fails over to an in-sync replica that holds the identical ordered log, and consumers resume at their committed offset with order intact. A consumer can crash; it restarts and re-reads from its last commit in offset order. A whole rack can go down; if your replication factor and in-sync-replica settings are sane, the surviving replicas preserve the log and its order. The guarantee is genuinely durable. What it does *not* survive is the two things this post is really about: a producer that lets records into the partition out of order, and a consumer that processes records out of the order it read them. The broker's part of the contract is solid. The endpoints are where order leaks, and we will spend sections 6 and 7 on exactly how.

## 4. Partitioning by key: per-entity order with parallelism

Here is where the design becomes elegant. We want two things that seem opposed: order (for the entities that need it) and parallelism (for throughput). Keyed partitioning gives us both, by making the *key* — not the broker, not luck — decide which serial log a record joins.

When a producer sends a record with a key, the client runs the key through a partitioner. The default Kafka partitioner computes `hash(key) % numPartitions` (using murmur2 over the key bytes) and that integer is the partition. The crucial property is determinism: the *same key always hashes to the same partition* (as long as the partition count does not change). So every record for account `12345` lands in, say, partition 7, every time, forever. Within partition 7 those records are ordered by offset. Therefore every record for account 12345 is ordered relative to every other record for account 12345 — that is per-key order, and you got it for free just by setting the key. Meanwhile account 67890 hashes to partition 2, account 99999 to partition 5, and they all process in parallel on different consumers. The figure below shows the routing: the producer hands a keyed record to the partitioner, which hashes the key and pins it to exactly one of the partitions.

![A routing diagram showing a producer sending a keyed record into a partitioner that computes hash of key modulo three and deterministically routes account A to partition zero, account B to partition one, and account C to partition two](/imgs/blogs/message-ordering-and-partitioning-guarantees-2.webp)

This is the move that resolves the tension from section 2. You do not get total order, and you do not need it. You get **per-key order** — a guarantee scoped exactly to the entity that requires it — *and* full horizontal parallelism across keys. If you have a million accounts spread over 12 partitions, each partition carries roughly 83,000 accounts' streams, every individual account perfectly ordered, and 12 consumers chew through them concurrently. The ordering domain (one account) and the parallelism domain (different accounts) are cleanly separated, and the partition key is the knob that sets the boundary between them.

The mental model that makes this click: **the partition key defines your ordering scope.** Pick a key equal to your ordering domain, and the system orders exactly what you need and nothing more. Pick `user_id` and you get per-user order. Pick `account_id` and you get per-account order. Pick `order_id` and you get per-order order. Pick nothing (null key, round-robin) and you get no per-entity order at all, just per-partition order over an arbitrary mix. The key is a one-line decision that determines the entire ordering behavior of your system, which is why section 8 is devoted to choosing it well.

```python
# Kafka producer: keyed send pins each account's events to one partition,
# so per-account order is preserved while different accounts run in parallel.
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers="broker1:9092,broker2:9092",
    # idempotent producer (see section 6) — order-safe even across retries
    enable_idempotence=True,
    acks="all",
    retries=2_147_483_647,        # retry forever; idempotence keeps order
    max_in_flight_requests_per_connection=5,  # safe BECAUSE idempotence is on
)

def emit_balance_change(account_id: str, delta_cents: int):
    # The KEY is the ordering domain. Same account -> same partition -> ordered.
    producer.send(
        topic="ledger.balance-changes",
        key=account_id.encode("utf-8"),   # <-- this single line sets ordering scope
        value=encode_event(delta_cents),
    )
```

### Sticky partitioning, null keys, and the batching wrinkle

A subtlety worth knowing: when you send records with a *null* key, the producer does not just round-robin every record one at a time — modern Kafka uses *sticky partitioning*, which sends a batch of null-keyed records to one partition, then "sticks" to a new partition for the next batch. This improves batching efficiency (fewer, larger requests) but it means null-keyed records are not evenly sprayed record-by-record; they are sprayed batch-by-batch. For ordering purposes this does not matter — null-keyed records have no ordering domain and no per-entity guarantee regardless — but it matters for *distribution*, and it surprises people who expected perfect round-robin. The lesson reinforces the main point: if you want per-entity order, you must set a key; if you do not set a key, you get neither per-entity order nor a guarantee about exactly how records spread, only that they spread reasonably across partitions over time.

You can also supply a *custom partitioner* when the default `hash(key) % N` is not what you want — for example, to route a known hot key to a dedicated partition, or to implement a locality-aware scheme. But beware: a custom partitioner is one more place to accidentally break ordering. If your custom partitioner ever maps the same key to two different partitions (say, it factors in a timestamp or a round-robin counter), you have shattered per-key order without realizing it, because the key's records now split across partitions. The iron rule for any partitioner, default or custom, is **same key must always map to the same partition**, deterministically, for the life of the topic. If your partitioner violates that, no amount of producer or consumer configuration will give you per-key order.

```java
// A custom partitioner that pins one whale key to a dedicated partition
// while hashing everyone else. The iron rule still holds: same key -> same partition.
public class WhaleAwarePartitioner implements Partitioner {
    private static final String WHALE = "tenant-corp-001";
    private static final int WHALE_PARTITION = 0;   // dedicated lane for the whale

    @Override
    public int partition(String topic, Object key, byte[] keyBytes,
                         Object value, byte[] valueBytes, Cluster cluster) {
        int n = cluster.partitionCountForTopic(topic);
        if (WHALE.equals(key)) return WHALE_PARTITION;        // deterministic
        // Hash everyone else across the remaining partitions (1..n-1).
        int h = Utils.murmur2(keyBytes) & 0x7fffffff;
        return 1 + (h % (n - 1));                             // also deterministic
    }
    // ...configure(), close() omitted
}
```

### What the key does NOT buy you

It is worth being precise about the limits, because over-claiming is how the email bug from the intro happens. Keyed partitioning orders records *with the same key*. It says nothing about records with *different* keys, even if they are causally related. If a "user created account" event keys on `user_id` and a "payment received" event keys on `payment_id`, those two events are in different partitions and have no order relationship, even though business logic might assume the account exists before the payment. If two events must be ordered relative to each other, they must share a key. Causal relationships that cross key boundaries are not ordered by the broker, full stop — you either co-key them or handle the out-of-order case explicitly downstream.

## 5. The throughput vs ordering tradeoff

We can now lay out the whole tradeoff space crisply, because it is governed by one inverse relationship: **the narrower your ordering scope, the more parallelism you get; the wider the scope, the less.** Total order (widest scope — the whole stream) buys zero parallelism. No order (narrowest scope — nothing is ordered) buys maximum parallelism. Per-partition and per-key sit in between, and per-key is the sweet spot for most real systems because it scopes ordering to the entity while leaving the rest free to scale. The matrix below maps each ordering scope to the parallelism, throughput, and cost it implies.

![A decision matrix with rows for total order, per-partition, per-key, and no order, and columns for parallelism, throughput, and cost, showing that narrowing the ordering scope trades a single serial point for parallel throughput at the price of partition and key design](/imgs/blogs/message-ordering-and-partitioning-guarantees-3.webp)

Read the matrix as a menu, not a ranking — there is no universally "best" row, only the row that matches what your feature actually requires. If you are emitting an immutable audit stream that must replay exactly, you may genuinely want total order and accept the single-partition ceiling; size that partition and move on. If you are processing per-account state transitions, per-key is correct and you scale partitions to throughput. If you are counting commutative metrics, no order is correct and you maximize fan-out. The mistake is not picking a row; the mistake is picking per-key parallelism while *believing* you have total order, or picking no key while *needing* per-entity order. Match the row to the requirement and the tradeoff resolves itself.

There is a second, subtler cost dimension hiding in the "cost" column: **partition skew.** If you key by `account_id` and one account — a whale, a bot, a misconfigured test harness — produces 40% of your traffic, that account's partition becomes a hot spot. All of that key's records must stay on one partition to preserve order, so you cannot spread the hot key out without breaking its order. The throughput of a keyed topic is therefore bounded not by the average partition but by the *hottest* partition. We will return to this in section 8, because choosing a key with good cardinality and even distribution is half the art, but note it here: per-key order is cheap only when keys are reasonably balanced. A skewed key reintroduces a serial-point bottleneck for the hot key, sneaking the section-2 problem back in through the side door.

### Picking partition count is a throughput question, not an ordering one

A common confusion: people think more partitions means "more ordering" or "better ordering." It does not. Partition *count* is purely a throughput and parallelism knob — it sets how many consumers can work in parallel and how much aggregate write bandwidth you have. It does not change the *kind* of ordering you get; with keyed partitioning you get per-key order whether you have 3 partitions or 300. What partition count *does* affect is the blast radius of a hot key (more partitions, finer spread of cold keys) and, critically, the stability of the key-to-partition mapping, which brings us to a sharp edge: repartitioning. We will hit it in section 7. For now, internalize that ordering scope (set by the key) and parallelism (set by the partition count) are *orthogonal* dials, and conflating them is a frequent source of muddled designs. The deeper sizing math lives in the [partitioning capacity planning](/blog/software-development/message-queue/partitioning-capacity-planning) post.

## 6. How ordering silently breaks: retries and in-flight requests

You have partitioned by key. Same key, same partition, ordered by offset — done, right? Not yet. There is a hole between your `send()` calls and the partition's offsets, and it is the most insidious ordering bug in all of messaging because it happens on the *producer* side, before the broker's per-partition guarantee even applies. The culprit is the combination of **retries** and **multiple in-flight requests**.

Modern producers do not send one record and wait. To get throughput, they pipeline: they allow several un-acknowledged requests to be in flight to the broker at once. In Kafka this is `max.in.flight.requests.per.connection`, default **5**. So the producer can have batches 1, 2, 3, 4, 5 all on the wire simultaneously. Now suppose batch 2 fails — a transient network blip, a leader election, a momentary timeout — but batches 3, 4, 5 succeed. The producer's retry logic kicks in and re-sends batch 2. But batches 3, 4, and 5 *already landed* in the partition at earlier offsets. Retried batch 2 now appends *after* them. The log order is now 1, 3, 4, 5, 2. You produced in order; the partition stored them out of order; per-partition order faithfully preserved the *wrong* order all the way to your consumer. The timeline below shows exactly this race: message 2 times out, message 3 lands first, then the retried message 2 appends behind it, leaving the log reading 1, 3, 2.

![A timeline showing message one sent and in-flight, message two sent then timing out, message three landing at offset one, the retried message two landing at offset two, and the resulting log reading one three two out of order](/imgs/blogs/message-ordering-and-partitioning-guarantees-4.webp)

This bug is vicious because it is rare and load-dependent. It only fires when a send fails *and* a later send to the same partition succeeds before the retry. In a healthy cluster that might be one in a million sends — which means it sails through every test, ships to production, and then one day a leader election reorders a handful of a customer's records and you spend a day convinced your code is fine because it *is* fine; the producer config is the problem.

It helps to understand *why* the default is 5 and not 1. The in-flight window exists for throughput: with a 2 ms round trip, a producer that waits for each batch's ack before sending the next is capped at one batch per 2 ms, or 500 batches/second; allowing five in-flight requests lets the producer keep the pipe full and reach roughly five times that rate. So the default trades a tiny, easily-fixed ordering risk for a large, always-present throughput gain — a reasonable default *if* you also turn on idempotence, which removes the risk entirely. Before Kafka 0.11 there was no idempotent producer, and the only order-safe option was `max.in.flight=1`, which is why a lot of old code and old advice still says "set in-flight to 1 for ordering." That advice is obsolete. On any modern broker you turn on idempotence and keep the pipeline full. The one situation where you still cannot use idempotence — talking to a very old broker, or a non-Kafka system without sequence numbers — is the one situation where `max.in.flight=1` remains the correct, if costly, answer.

A related trap: `retries=0` does *not* make you order-safe by avoiding the retry race; it makes you *lose messages*, because a transient failure now drops the record entirely. People sometimes set `retries=0` reasoning "no retries, no reordering," which trades a rare reordering bug for a guaranteed data-loss bug. The correct posture is `retries` high (effectively infinite) *with* idempotence on, so failures are retried *and* the retries cannot reorder. Never disable retries to dodge the ordering problem; fix the ordering problem with idempotence and keep retrying.

There are two correct fixes. The blunt one: set `max.in.flight.requests.per.connection=1`. With only one request on the wire at a time, a retry of batch 2 cannot land after batch 3, because batch 3 was never sent until batch 2 was acknowledged. Order is guaranteed, but you have serialized your producer pipeline and thrown away throughput — every batch waits for the previous one's round trip, so your producer rate is bounded by one batch per RTT. The surgical fix: enable the **idempotent producer** (`enable.idempotence=true`). The idempotent producer attaches a producer ID and a per-partition monotonic sequence number to every batch. The broker tracks the last sequence number it accepted per partition and *rejects out-of-order or duplicate sequence numbers*, forcing retries to be re-ordered correctly even with up to 5 in-flight requests. You keep the pipeline throughput *and* you keep order. This is why the idempotent producer is non-negotiable for any ordered topic, and why modern Kafka (3.0+) turns it on by default.

```properties
# THE order-safe producer config for any keyed/ordered topic.
# Idempotence lets you keep 5 in-flight requests AND preserve order on retry.
enable.idempotence=true
acks=all
max.in.flight.requests.per.connection=5   # safe ONLY with idempotence=true
retries=2147483647                          # retry "forever"; sequence numbers keep order
# Without idempotence, you would be forced to:
#   max.in.flight.requests.per.connection=1   # order-safe but ~1 batch / RTT
```

#### Worked example: why msg2 lands after msg3, and the two fixes

Walk it slowly with numbers. Your producer has `max.in.flight.requests.per.connection=5` and idempotence *off*. It sends, to the same partition, five batches in quick succession: B1, B2, B3, B4, B5. Round-trip time to the broker is 2 ms. All five go on the wire at t=0. At t=2 ms, the broker acknowledges B1, B3, B4, B5 — but B2 hit a momentary leader hiccup and timed out. The broker appended B1 at offset 0, B3 at offset 1, B4 at offset 2, B5 at offset 3 (it appends them as they arrive and pass; B2 never passed). At t=200 ms the producer's retry timer fires and re-sends B2. This time it succeeds and appends at offset 4. The partition now reads: offset 0 = B1, offset 1 = B3, offset 2 = B4, offset 3 = B5, offset 4 = B2. Production order was 1,2,3,4,5; storage order is 1,3,4,5,2. A consumer reading in offset order processes B2 *last*, after B5, even though you sent it second. If these were the two email updates from the intro, the consumer applies the corrected email and *then* re-applies the wrong one. Bug shipped.

Fix one, `max.in.flight=1`: B2 is the only request allowed on the wire after B1's ack. B3 is not sent until B2 is acknowledged. So even if B2 needs three retries, B3 cannot overtake it — order is mechanically guaranteed. Cost: your producer now does one batch per round trip. At 2 ms RTT that is at most 500 batches/second per connection, which for many high-volume producers is a 5–10x throughput cut. Fix two, idempotence on with `max.in.flight=5`: B1..B5 carry sequence numbers 0..4. When retried-B2 (sequence 1) arrives after the broker has already accepted sequence 0 (B1) but is expecting sequence 1 next, the broker accepts it in the right slot and the de-duplication and sequencing logic ensures the final committed order is 1,2,3,4,5. You keep five in-flight requests, keep the pipeline full, and keep order. This is strictly better, which is why "turn on the idempotent producer" is the first thing I check when anyone reports record reordering.

## 7. More ways it breaks: multiple producers, consumer concurrency, rebalances

The retry hole is the famous one, but it is not the only one. Once you start looking, out-of-order processing has a whole taxonomy of causes, and they live at the producer, the partition layer, and the consumer. The tree below organizes them so you can localize a reordering bug to the layer that caused it.

![A taxonomy tree of out-of-order causes branching into producer side with retry-and-in-flight and multiple writers per key, partition side with no key or round-robin routing, and consumer side with thread-pool concurrency](/imgs/blogs/message-ordering-and-partitioning-guarantees-7.webp)

### Multiple producers to one key

Keyed partitioning guarantees that two records with the same key land in the same partition. It does **not** guarantee anything about the order in which two *different producer processes* writing the same key land their records. Suppose two application instances both handle requests for account 12345 — instance A processes the email-change-1 event, instance B processes email-change-2, and both produce to the ledger topic. Both records go to the same partition (same key). But which one appends first is a race between two independent network paths, two producer buffers, two sets of in-flight windows. The broker appends them in the order they *arrive*, which is not necessarily the order the business events occurred. The idempotent producer does not help here — it sequences within *one* producer's stream, not across producers; each producer has its own producer ID and its own sequence space. The before-and-after below contrasts the racy two-writer case with the single-writer-per-key discipline that fixes it.

![A two-panel comparison showing two producers writing the same key racing on the network and producing nondeterministic log order on the left, versus a single writer per key with bounded in-flight requests producing stable per-key order on the right](/imgs/blogs/message-ordering-and-partitioning-guarantees-9.webp)

The fix is structural: **ensure a single writer per key.** Route all work for a given key through one producer process, so there is one serial point per key on the producer side too. This is often done with a consistent-hash routing layer in front of your producers, or by having an upstream partitioned input (a partitioned topic, a sharded queue) that already pins each key to one worker, so the worker that owns key K is the only one that ever produces key K. This connects directly to [database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding): the discipline of "one owner per shard key" is the same idea on the write path. If you cannot enforce single-writer-per-key, you must make your downstream logic tolerate cross-writer reordering — usually with a version number or timestamp on the event and last-writer-wins or conditional-update semantics, which is the [idempotency](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) story.

### Consumer-side concurrency

Here is the one that bites teams who did *everything* on the broker side correctly. You have keyed partitioning, the idempotent producer, single-writer-per-key — the storage order is pristine. Then your consumer, for throughput, reads a batch of records from the partition and hands them to a thread pool to process in parallel. The instant you do that, you have thrown away order *inside your own consumer*, because thread 3 might finish record 5 before thread 1 finishes record 2. The broker delivered them in order; your consumer un-ordered them. This is heartbreakingly common because the consumer-side concurrency looks like a local optimization — "I'll just parallelize the handler" — and it silently destroys the guarantee you paid for all the way up the stack.

The rule is: **per-partition processing must be single-threaded, or parallelized only by key, never blindly.** If you want consumer parallelism, the right pattern is one worker per partition (a consumer group with as many consumers as partitions), or, inside one consumer, a *key-aware* executor that routes all records for a given key to the same worker thread, so order is preserved per key while different keys run on different threads. Reactor frameworks and Kafka Streams do this for you; a hand-rolled `executor.submit(record)` over a partition's batch does not, and will reorder. The pipeline below shows the order-preserving path — record routed by key, appended to the partition tail at a monotonic offset, then consumed strictly in offset order — which only holds if the consumer respects offset order all the way to the side effect.

![A pipeline showing a keyed record routed by the partitioner to one partition, appended at the partition tail with a monotonic offset assigned, then consumed in offset order so that storage order becomes processing order](/imgs/blogs/message-ordering-and-partitioning-guarantees-5.webp)

```java
// WRONG: parallelizes a partition's batch, destroying per-partition order.
for (ConsumerRecord<String, byte[]> rec : records) {
    executor.submit(() -> handle(rec));   // thread 3 may finish before thread 1
}

// RIGHT: key-aware routing — same key always to the same single-threaded lane,
// so per-key order is preserved while different keys run in parallel.
for (ConsumerRecord<String, byte[]> rec : records) {
    int lane = Math.floorMod(rec.key().hashCode(), NUM_LANES);
    laneExecutors[lane].submit(() -> handle(rec));  // one thread per lane
}
// Each laneExecutor is a single-threaded executor. Records with the same key
// hash to the same lane and are processed strictly in the order they were read.
```

### Rebalances and the brief reorder window

The last producer-and-partition-layer hazard is the **consumer group rebalance**. When a consumer joins or leaves the group — a deploy, a crash, an autoscale event — Kafka reassigns partitions across the surviving consumers. During and right after a rebalance there is a window where ordering can be perturbed *if you are not careful with offset commits*. The classic failure: consumer C1 owns partition 7, reads records up to offset 1000, processes through offset 990 but has not yet committed past 985, then gets the partition revoked in a rebalance. Consumer C2 picks up partition 7 and starts from the last *committed* offset, 985. Records 986–990 get processed *again* — that is at-least-once redelivery — and if C1's in-flight processing of 986–990 had side effects that C2 now re-applies, you can get effects applied in a tangled order. The order on the *partition* is fine; the *processing* order across the handoff is what gets perturbed.

The defenses are: commit offsets *after* processing (so you never skip), make processing idempotent (so reprocessing 986–990 is harmless), and use cooperative rebalancing (`CooperativeStickyAssignor`) so partitions are not needlessly shuffled. RabbitMQ has an analogous hazard: a message delivered to a consumer that then dies before acking is *requeued* and redelivered, potentially to a different consumer and potentially *behind* messages that were delivered after it — so a single queue's FIFO order is broken by any redelivery. The general lesson across both brokers is that **redelivery is a reordering event**, and any system that relies on order must treat redelivery as a first-class case, not an edge case. Repartitioning — changing the partition *count* — is the most violent version of this: when `numPartitions` changes, `hash(key) % numPartitions` changes for most keys, so a key that lived on partition 7 now maps to partition 3, and its old records (still on partition 7) and new records (now on partition 3) are in *different* logs with no order between them. **Never repartition a keyed, ordered topic in place.** If you must grow partitions, you create a new topic and migrate, or you accept an ordering discontinuity at the cutover. This is the sharpest edge in the whole domain and the reason partition count should be chosen generously up front; the [partitioning capacity planning](/blog/software-development/message-queue/partitioning-capacity-planning) post covers the migration playbook.

## 8. Designing for the ordering you need (choosing a partition key)

Everything so far converges on one design decision: the partition key. Choose it well and the system orders exactly what must be ordered, scales everything else, and stays balanced. Choose it poorly and you either lose order you needed or create a hot partition that bottlenecks the topic. Here is the decision procedure I actually use.

**Step one: identify your ordering domain.** Ask, "for which entity must events apply in sequence?" For a ledger it is the account. For a chat app it is the conversation (or the room). For an order-fulfillment pipeline it is the order. For a user-profile service it is the user. The ordering domain is the boundary inside which order matters and outside which it does not. This is a *business* question, not a technical one, and getting it right is 80% of the work.

**Step two: make the partition key equal the ordering domain's identity.** Key on `account_id`, `conversation_id`, `order_id`, `user_id` — whatever uniquely names the entity whose events must stay ordered. Now `hash(key) % N` pins each entity to one partition, per-entity order holds, and different entities parallelize. The key *is* the ordering contract.

**Step three: check cardinality and distribution.** The key must have enough distinct values to spread across your partitions (cardinality ≥ partition count, ideally ≫), and those values must be reasonably uniform so no single key dominates. If 1% of your accounts generate 50% of traffic, keying on account creates hot partitions. Mitigations: if the hot entity does not *truly* need single-stream order (e.g., a shared "system" account whose events are commutative), give it a composite key that spreads it; if it does need order, you are stuck with a serial point for that key and you size for it. Never sacrifice correctness for balance — if account order matters, you keep the hot account on one partition even if it is hot, because a balanced-but-wrong system is worse than an unbalanced-but-correct one.

The stack below ranks the guarantees from strongest at the top to weakest at the bottom, so you can read off exactly what each key choice gives you: a single partition for total order, keyed partitions for per-key order, keyed partitions with bounded in-flight for solid per-partition order, round-robin keys for no per-entity order, and concurrent consumers for no order at all.

![A layered stack ranking ordering strength from a single partition with total order at the top, down through keyed partitions with per-key order, partitions with in-flight one for per-partition order, round-robin keys with no per-entity order, and concurrent consumers with no order at all at the bottom](/imgs/blogs/message-ordering-and-partitioning-guarantees-6.webp)

#### Worked example: a partition key for a bank-transfer stream

Concretely: you operate a transfer-processing stream that emits balance-change events, peaking at 200,000 events/second, and you need each account's balance changes to apply in strict sequence (a debit before the credit that depends on it). You have decided — correctly — that the ordering domain is the account, not the whole stream. So you key on `account_id`.

Size the partitions for throughput. Two hundred thousand events/second at 500 bytes is 100 MB/s. You target roughly 8–10 MB/s per partition to stay well under a single leader's safe ceiling with `acks=all`, so you provision 12 partitions, giving headroom at about 8.3 MB/s per partition. Cardinality check: you have, say, 5 million active accounts, vastly more than 12 partitions, so each partition carries roughly 417,000 accounts' streams — plenty of spread. Distribution check: you sample a day of traffic and find your hottest account (a corporate settlement account) generates 3% of events. Three percent of 100 MB/s is 3 MB/s — that account's partition runs hotter than average but still under the 8.3 MB/s budget, so it is fine; you note it and move on. Every account's events are ordered within its partition; 12 consumers in the group process the 12 partitions in parallel; aggregate throughput is 100 MB/s with comfortable headroom; per-account order is guaranteed end to end. And you wire the producer with `enable.idempotence=true`, `acks=all`, `max.in.flight=5` so retries cannot reorder, and you make each consumer single-threaded per partition (or key-aware) so consumer concurrency cannot reorder. That is the complete, correct design: ordering domain = account, key = account_id, 12 partitions for throughput, idempotent producer, single-writer-per-key enforced by an upstream sharded input, single-threaded-per-partition consumers. The grid below shows the consumer-side shape — three partitions, three keyed streams A, B, C, each owned by exactly one consumer so per-key order survives all the way to the side effect.

![A grid showing partition zero carrying keys A, partition one carrying keys B, and partition two carrying keys C across the top row, with consumer one owning partition zero, consumer two owning partition one, and consumer three owning partition two across the bottom row, illustrating one consumer per partition](/imgs/blogs/message-ordering-and-partitioning-guarantees-8.webp)

### Composite keys and the order-vs-balance escape hatch

When a single natural key is both your ordering domain *and* badly skewed, a composite key is the escape hatch — but only when ordering allows it. Suppose you key by `user_id` and one power user is a hot spot, but that user's events are *independent* (say, telemetry samples where order does not matter for that user). You can key those events by `user_id + random_bucket` to spread them, accepting no per-user order for that user specifically while keeping per-user order for everyone whose events still need it. The composite key is a deliberate, surgical relaxation: you are saying "for *this* slice of traffic, balance matters more than order, and I have checked that order is not required here." Done carelessly, though, a composite key silently destroys an order you needed — if you append a bucket to a key whose events *do* require sequence, you have split that entity across buckets and shattered its order. So the composite-key move is safe only after you have answered, per entity, the section-8 question: does this entity truly need cross-event order? If yes, the hot key stays whole and you size for it; if no, spread it freely. There is no automatic way to have both order and balance for a genuinely hot, genuinely order-requiring key — that is the section-2 serial point reasserting itself, and the only honest options are to size for it or to discover the requirement was softer than you thought.

### When you have no natural ordering domain

Sometimes events are genuinely independent — a stream of "page viewed" analytics events with no per-entity sequence requirement. Then you have no ordering domain, and the right move is to *not* key (or key on something high-cardinality and uniform just for balance, like a request ID) and process with maximum parallelism. Forcing an ordering key onto an order-free workload only costs you balance and throughput for a guarantee you do not need. Conversely, if you find yourself wanting "global order" across independent entities, stop and ask whether you actually need it or whether you have just not identified the real, narrower ordering domain. Nineteen times out of twenty the narrower domain exists and per-key order is the answer.

## 9. Testing and detecting out-of-order processing

Ordering bugs are silent and load-dependent, which means you will not catch them by accident — you have to test and monitor for them deliberately. Here is how.

**Sequence-number every event at the source.** Have the producer stamp a monotonically increasing per-key sequence number (or a high-resolution logical timestamp) onto every event, *before* it goes to the broker. Then the consumer can assert that, per key, the sequence numbers it processes are strictly increasing. The instant it sees sequence 7 after sequence 9 for the same key, you have detected a reorder — at the exact point it happened, with the key and offsets to debug it. This is the single most valuable ordering safeguard you can add, and it costs a few bytes per event.

```python
# Consumer-side ordering assertion. Detects ANY per-key reorder immediately.
last_seq = {}  # key -> last sequence number processed

def process(record):
    key = record.key
    seq = record.headers["seq"]   # producer-stamped per-key monotonic sequence
    prev = last_seq.get(key)
    if prev is not None and seq <= prev:
        # Reorder or duplicate detected — alert with full context, do not silently apply.
        log.error(
            "OUT-OF-ORDER for key=%s: got seq=%s after seq=%s at offset=%s partition=%s",
            key, seq, prev, record.offset, record.partition,
        )
        metrics.increment("ordering.violation", tags={"partition": record.partition})
        # Optionally: route to a quarantine topic instead of applying.
        return
    last_seq[key] = seq
    apply(record)
```

**Inject chaos to provoke the rare race.** Ordering bugs hide because the triggering conditions (a failed send with a later success, a rebalance mid-flight) are rare in healthy clusters. So manufacture them in a staging environment: kill the partition leader during a load test to force retries; trigger rebalances by bouncing consumers under load; inject network delays on the producer so in-flight requests stack up and a retry can overtake. With a no-idempotence, `max.in.flight=5` producer, leader-kill chaos will reliably produce reorders, and your sequence-number assertion will catch them. This is how you *prove* your config is order-safe rather than hoping.

**Monitor for the symptoms in production.** Even with everything configured right, instrument the ordering-violation counter above and alert on any non-zero value — it should be flat at zero, and any blip is a real defect. Watch consumer lag *per partition*, not just in aggregate, because a single hot partition (skewed key) shows up as one partition's lag climbing while the rest are flat — the signature of partition skew that throttles your hottest key. And in RabbitMQ, watch redelivery rates, because every redelivery is a potential reorder and a rising redelivery rate means rising reorder risk.

```bash
# Per-partition lag — a single climbing partition is the skewed-hot-key signature.
kafka-consumer-groups.sh --bootstrap-server broker:9092 \
  --describe --group ledger-consumers
# TOPIC                 PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG
# ledger.balance-changes 0          1048120         1048140         20
# ledger.balance-changes 7          1048120         1240315        192195   <-- hot partition
# ledger.balance-changes 11         1051002         1051040         38
```

**Write a property test for the producer path specifically.** Because the retry race is a producer-config bug, the highest-value automated test spins up a real broker (Testcontainers makes this a few lines), produces a few thousand keyed records while a background thread repeatedly kills and restarts the partition leader, then reads the partition back and asserts that, per key, the stamped sequence numbers are strictly increasing. Run this test twice: once with idempotence *off* and `max.in.flight=5` (it should *fail*, proving the test can actually detect reordering — a test that never fails is worthless), and once with idempotence *on* (it must *pass*). That pair of runs is a permanent regression guard: if someone later flips idempotence off in a config refactor, the failing test catches it before production does. Most ordering bugs ship precisely because no test ever provoked a retry, so the config was never exercised under the conditions that break it.

There is a deeper principle here worth stating plainly: **ordering is a property you can only verify under failure, because it only breaks under failure.** A happy-path integration test where every send succeeds will *always* show perfect order, no matter how broken your config is, because the reordering mechanisms (retries, redelivery, rebalances) never fire on the happy path. This is why ordering bugs are so durable — the conditions that expose them are exactly the conditions normal tests avoid. The only way to gain real confidence is to inject the failures deliberately. A green test suite that never kills a leader, never bounces a consumer, and never delays a send tells you nothing about your ordering guarantees. Treat "we tested ordering" as meaning "we provoked retries and rebalances and asserted order survived," and treat any weaker claim as untested.

The discipline is the same as any subtle-correctness property: make the invariant *checkable* (sequence numbers), *provoke* the failure (chaos), and *alert* on the symptom (violation counter, per-partition lag). Teams that do this find ordering bugs in staging; teams that do not find them in a customer's mailbox.

## Case studies and war stories

### The double email update (the bug from the intro)

A profile service consumed a `user.profile-changed` topic. Events were keyed by `user_id`, so per-user order *should* have held. But the producer — a legacy service predating idempotent producers — ran with `acks=all`, `retries=10`, and `max.in.flight.requests.per.connection=5`, idempotence off. Ninety-nine-point-nine percent of the time order was fine. Then a routine broker upgrade triggered a leader election on the partition holding a particular user's stream, a send failed, the retry landed behind two later updates, and that user's email reverted to a stale value. The fix was one line: `enable.idempotence=true`. The lesson: keyed partitioning is necessary but not sufficient for per-key order — the producer must also be configured so retries cannot reorder. Most teams set the key and stop, leaving the retry hole wide open.

### The parallel consumer that scrambled a saga

An order-fulfillment system keyed events by `order_id` and had a flawless broker setup. To hit a throughput target, an engineer changed the consumer to hand each polled batch to a 16-thread pool. Throughput tripled; correctness collapsed. Orders went through their state machine out of sequence — "shipped" applied before "payment confirmed," "cancelled" applied before "created" — because threads finished in random order. The storage order was perfect; the consumer un-ordered it. The fix was a key-aware executor: route each record to a lane by `hash(order_id) % NUM_LANES`, single-threaded per lane. Throughput stayed high (different orders parallelized) and per-order sequence was restored. The lesson: consumer-side concurrency is the most common place perfectly-ordered storage gets scrambled, and it looks like an innocent local optimization.

### The repartition that split a customer in two

A team running near a partition's throughput ceiling decided to grow a keyed topic from 6 to 12 partitions to add capacity. They did it in place. Overnight, `hash(key) % 6` became `hash(key) % 12`, so most keys remapped to different partitions. A customer whose events had all lived on partition 4 now produced to partition 10 — while their *historical* events still sat on partition 4. Consumers reading partition 10 saw the new events with no relationship to the old ones on partition 4; the per-customer timeline split across two unrelated logs and the ordering guarantee evaporated at the cutover. They had to freeze writes, drain, and migrate to a fresh topic to restore order. The lesson: **`hash(key) % N` is only stable while `N` is fixed.** Repartitioning a keyed ordered topic in place is one of the few genuinely irreversible mistakes in this space. Choose partition count generously up front and migrate via a new topic when you must grow.

### The hot key that throttled the whole topic

A real-time analytics pipeline keyed events by `tenant_id`. It worked beautifully until one enterprise tenant onboarded and generated 60% of all traffic. That tenant's partition saturated its single leader while the other partitions sat nearly idle; aggregate topic throughput was capped not by the cluster but by one partition, and that tenant's consumer lag climbed into the millions while everyone else was real-time. Per-tenant order was correct — but the design had recreated a single serial point for the hot tenant. They could not spread the hot tenant without breaking its order, so they negotiated: that tenant's events did *not* actually require cross-event order (they were independent metric samples), so they switched that tenant to a composite key (`tenant_id + bucket`) that spread its load across all partitions, accepting no order for that tenant specifically. The lesson: per-key order is cheap only when keys are balanced; a skewed key smuggles the section-2 serial-point bottleneck back in, and the fix requires knowing which keys genuinely need order and which only seemed to.

## When to reach for strong ordering (and when not to)

Reach for **total order (single partition / single queue)** only when the entire stream is genuinely one sequence that must replay exactly — a single-aggregate event-sourced log, a small audit trail, a leader-election or metadata stream. Accept the single-partition throughput ceiling consciously and size for it. If you find yourself wanting total order for a high-volume stream, you have almost certainly mis-identified the ordering domain; look for the narrower one.

Reach for **per-key order (keyed partitions)** for the vast majority of stateful event streams: per-account, per-user, per-order, per-device, per-conversation. This is the default and correct choice when events update per-entity state and must apply in sequence. Pay the cost of choosing a good key (right ordering domain, decent cardinality, even distribution) and configuring the producer and consumer to not reorder, and you get order exactly where you need it with full parallelism everywhere else.

Reach for **no ordering** when events are commutative or independent — metrics, page views, telemetry, idempotent upserts where last-writer-wins is fine. Forcing ordering onto an order-free workload only costs you throughput and balance for a guarantee you do not use. Be honest about this: a great many streams that teams agonize over ordering for do not actually need it, because the downstream operation is commutative or idempotent. The cheapest ordering bug is the one you designed away by realizing you never needed order in the first place.

And when you have cross-entity causal relationships that *cannot* be co-keyed, do not try to force them into broker order — you will lose. Instead push the ordering responsibility into the data: stamp events with versions or logical timestamps and use conditional/last-writer-wins updates downstream, the [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) pattern. Broker ordering is a powerful tool with a hard boundary at the key; past that boundary, application-level ordering takes over.

| Requirement | Ordering scope | Mechanism | Parallelism | Watch out for |
| --- | --- | --- | --- | --- |
| Replayable single sequence | Total | One partition / one queue | None (1 worker) | Throughput ceiling |
| Per-entity state machine | Per-key | Key = entity id, idempotent producer | N partitions | Hot-key skew, consumer concurrency |
| Per-shard FIFO | Per-partition | Any key, in-flight=1 or idempotence | N partitions | Cross-partition has no order |
| Commutative / independent | None | No key, round-robin | Maximum | Do not over-engineer order |
| Cross-key causal | App-level | Version / timestamp + LWW | Maximum | Broker won't order it for you |

## Key takeaways

- **Total order requires a single serial point, and a single serial point caps throughput.** This is a theorem about ordering, not a limitation of any broker. Global order and horizontal scale are in direct, irreconcilable tension.
- **The real guarantee is per-partition order**, and you almost never need more. Within one partition, records are read in offset order, durably, across failover and restart. Across partitions there is no order at all.
- **The partition key sets your ordering scope.** Key on your ordering domain (account, user, order) and you get per-entity order with full parallelism across entities. The key is a one-line decision that defines the entire ordering behavior of the system.
- **Partition count is a throughput dial, not an ordering dial.** More partitions buy parallelism; they do not change the *kind* of ordering you get. Ordering scope and parallelism are orthogonal.
- **Producer retries with in-flight requests > 1 silently reorder.** A retried batch can land behind a later one. Fix it with the **idempotent producer** (`enable.idempotence=true`), which keeps order *and* pipeline throughput; `max.in.flight=1` also fixes it but throws away throughput.
- **Multiple producers to one key race**, and the idempotent producer does not save you because it sequences within one producer. Enforce a **single writer per key**, or push ordering into the data with versions.
- **Consumer-side concurrency un-orders perfectly-ordered storage.** Process each partition single-threaded, or use a key-aware executor — never blindly `submit()` a partition's batch to a thread pool.
- **Redelivery and rebalances are reordering events.** Commit after processing, make handlers idempotent, and use cooperative rebalancing. Treat redelivery as a first-class case.
- **Never repartition a keyed ordered topic in place** — `hash(key) % N` only holds while `N` is fixed. Migrate to a new topic instead. Choose partition count generously up front.
- **Make ordering checkable**: stamp per-key sequence numbers, assert monotonicity in the consumer, provoke reorders with chaos in staging, and alert on the violation counter and per-partition lag. Ordering bugs are silent; you must hunt them deliberately.

## Further reading

- [Database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding) — the same key-to-shard mapping and single-owner-per-shard discipline on the storage side.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — how the partitioned, replicated log under all of this actually works.
- [Idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) — how to make consumers safe against the redelivery and cross-key reordering you cannot prevent.
- [Partitioning and capacity planning](/blog/software-development/message-queue/partitioning-capacity-planning) — sizing partition count for throughput, hot-key mitigation, and the repartition migration playbook.
- [Consistency models: from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) — where total order, causal order, and per-key order sit on the consistency spectrum.
- [RabbitMQ production architecture and scaling](/blog/software-development/system-design/rabbitmq-production-architecture-scaling) — how single-queue FIFO, redelivery, and competing consumers interact with ordering.
- [Apache Kafka documentation: producer idempotence and ordering guarantees](https://kafka.apache.org/documentation/#semantics) — the canonical reference for `enable.idempotence`, `max.in.flight.requests.per.connection`, and the exactly-once sequencing model.
