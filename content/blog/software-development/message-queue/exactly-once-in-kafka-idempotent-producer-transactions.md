---
title: "Exactly-Once in Kafka: Idempotent Producers and Transactions"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Learn exactly how Kafka delivers exactly-once semantics inside its own boundary — producer IDs and sequence numbers that kill retry duplicates, transactions and control markers that make multi-partition writes atomic, and the precise point where the guarantee stops and your own idempotency must take over."
tags:
  [
    "message-queue",
    "kafka",
    "exactly-once",
    "transactions",
    "idempotency",
    "kafka-streams",
    "distributed-systems",
    "event-driven",
    "reliability",
    "stream-processing",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/exactly-once-in-kafka-idempotent-producer-transactions-1.webp"
---

There is a phrase that has caused more confusion in distributed systems than almost any other, and Kafka put it on a feature page: *exactly-once*. The moment a junior engineer reads that Kafka supports exactly-once semantics, they tend to conclude that they can stop worrying about duplicates forever — that they can fire off a database write, send an email, debit a card, and trust the broker to make sure none of it ever happens twice. That conclusion is wrong, and the gap between what Kafka actually delivers and what people assume it delivers is responsible for a long tail of double-charged customers and twice-sent notifications. Kafka's exactly-once is real, it is genuinely impressive engineering, and it is bounded by a wall that is rarely drawn on the marketing slide. This post draws that wall.

Here is the honest one-sentence version, and it is worth tattooing somewhere: **Kafka gives you exactly-once for Kafka-to-Kafka data flows, and the guarantee evaporates the instant a side effect leaves Kafka.** Inside that boundary — consume from a topic, transform, produce to another topic, advance your offsets — Kafka can make the whole step atomic and duplicate-free even across crashes, retries, and rebalances. Outside that boundary — the moment your processing writes to Postgres, calls Stripe, or hits an SMTP server — you are back in at-least-once land and you need your own idempotency. The skill this post teaches is seeing exactly where that line falls in your own architecture, and which of Kafka's two distinct mechanisms is doing the work on each side of it.

Those two mechanisms are the spine of everything that follows, and they are constantly conflated. The first is the **idempotent producer**, a single config flag that solves exactly one narrow problem: a producer that retries a send after a network hiccup must not write the same record twice. The second is **transactions**, a much heavier machine that lets a producer write to many partitions — including the internal offsets topic — as a single atomic unit that either all commits or all aborts. The idempotent producer is the foundation; transactions are built on top of it. Most teams need the first and never the second, and telling them apart is half the battle. The figure below shows the idempotent producer's whole trick in one line: a producer ID, a per-partition sequence number, and a broker that drops the duplicate.

![A pipeline diagram showing an idempotent producer stamping each record with a producer ID and a per-partition sequence number, a retry on timeout, and the broker dropping the duplicate after checking the sequence number](/imgs/blogs/exactly-once-in-kafka-idempotent-producer-transactions-1.webp)

We will build this from the bottom. First we pin down what "exactly-once" can and cannot mean, because the loose definition is the root of the confusion. Then we take apart the idempotent producer — the producer ID, the sequence number, and the exact broker-side check that dedups a retry — with a crash trace you can follow record by record. From there we climb to transactions: the `transactional.id`, the transaction coordinator, the internal `__transaction_state` topic, and the full lifecycle from `initTransactions` to `commitTransaction`, including the commit and abort control markers that get written into the log itself. We cover what `read_committed` consumers actually see (and the latency cost they pay), the read-process-write pattern that ties it all together, exactly-once in Kafka Streams where one config flag wires the whole thing for you, and finally the boundary — the precise place where Kafka's guarantee stops and your idempotency must begin. This deepens the exactly-once figure from [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log); if you have not read that or [delivery semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once), skim them first — this post assumes you know what at-least-once means and why it is the default.

## 1. What "exactly-once" can and cannot mean

The single most important thing to understand before any config flag is that "exactly-once" is not one guarantee — it is a family of guarantees that differ by *what* is happening exactly once. Conflate them and you will design the wrong thing. Let me separate the three meanings that people cram into the one phrase.

The first meaning is **exactly-once delivery**: the network delivers each message to the consumer process exactly one time, never zero, never two. This is the meaning most people assume, and it is *impossible* at the wire level. The reason is the two-generals problem, covered in depth in [delivery semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once): a sender cannot distinguish a lost message from a lost acknowledgement, so it must either resend (risking a duplicate) or not resend (risking a loss). No protocol closes that gap. Kafka does not deliver exactly-once in this sense and never claims to. Packets still get retried; duplicates still cross the wire.

The second meaning is **exactly-once processing**: regardless of how many times a message is *delivered*, the consumer's processing logic produces the effect of having run against it exactly once. This is achievable, but only for effects Kafka controls — namely, writes to other Kafka topics and the advancement of consumer offsets. If your "effect" is appending records to an output topic and committing your read position, Kafka can make that look like it happened exactly once even though the underlying deliveries happened at-least-once. This is the meaning Kafka's exactly-once semantics (EOS) actually delivers, and the word everyone should use for it is *effectively-once*: deliver many, apply once.

The third meaning is **end-to-end exactly-once across systems**: the message arrives once, your business logic runs once, your database write happens once, your email goes out once, all atomically. This is what teams actually want and it is *not* what Kafka provides, because Kafka cannot reach inside Postgres or Stripe to make those writes part of its transaction. The instant the effect is external, you are composing Kafka's effectively-once with a different system's guarantee, and the composite is only as strong as the weakest link — which is your own idempotency code.

So when someone says "Kafka does exactly-once," the precise translation is: *Kafka does effectively-once processing for state that lives in Kafka — output records and consumer offsets — and nothing more.* That is a genuinely powerful guarantee. A stream-processing topology that reads from topic A, joins, aggregates, and writes to topic B can be made exactly-once with that and only that. But the second your topology's sink is a REST call, Kafka's guarantee ends at the producer's `send()` and your idempotency begins at the API call. Hold that distinction; the entire post is an elaboration of it.

### Why the count, not the delivery, is what matters

The reframing that makes EOS tractable is to stop counting deliveries and start counting *effects*. Deliveries cannot be made unique — the network forbids it. But effects can be made *idempotent*: a write that produces the same final state no matter how many times it runs. Kafka's idempotent producer does this for the effect "append this record to this partition": no matter how many times the producer retries, the partition ends up with one copy. Kafka transactions do this for the compound effect "append these records to these partitions and advance these offsets": no matter how the producer crashes and restarts, the log ends up either with all of them or none of them. In both cases the trick is not to deliver once but to make the *append* idempotent through a deduplication key. For the producer that key is the producer ID plus a sequence number. For transactions it is the transaction's epoch plus a commit marker. Everything mechanical in this post is a variation on "attach a key, dedup on the key."

### The three duplicate sources, named

It helps to name the three distinct places duplicates come from, because each of Kafka's mechanisms targets a different one, and a team that does not know which source is biting them reaches for the wrong fix. The first source is the **producer retry**: a send whose ack was lost, resent, and appended twice. This is the narrowest and most common source, and the idempotent producer alone closes it. The second source is the **torn multi-write**: a producer that writes one logical unit across several partitions and crashes partway, leaving some partitions written and others not — a partial, inconsistent state that no single-record dedup can fix because the problem is the *set*, not any one record. Only transactions close this. The third source is the **reprocess-on-restart**: a consume-process-produce loop that produced output but crashed before recording that it consumed the input, so on restart it reprocesses and re-produces. This is the source that bites stream-processing jobs hardest, and only the *atomic offset commit inside a transaction* closes it. Three sources, three mechanisms, one escalating ladder — and the rest of this post climbs it rung by rung. Keep the three sources in mind as labels; every section below is closing one of them.

The honest accounting is that most teams only ever face the first source. A service that publishes events to Kafka and occasionally retries on a timeout needs the idempotent producer and nothing more. The second and third sources only appear when you are doing multi-partition atomic writes or running a consume-transform-produce loop, which is a narrower population than the number of teams who turn on transactions "to be safe." Knowing which source you actually have is the difference between a free fix and a 20% throughput tax you did not need to pay.

## 2. The idempotent producer: PID and sequence numbers

Start with the narrowest problem, because solving it is the foundation for everything heavier. A producer sends a batch of records to a partition leader. The leader appends them and sends back an acknowledgement. But the acknowledgement is lost in the network — the records are safely on disk, but the producer never hears so. The producer's retry logic kicks in and resends the identical batch. Without protection, the leader appends it *again*, and now the partition has two copies of every record. This is a producer-side duplicate, and it has nothing to do with the consumer; it happens before any consumer reads anything.

You turn on the fix with one line:

```java
Properties props = new Properties();
props.put("bootstrap.servers", "broker1:9092,broker2:9092");
props.put("enable.idempotence", "true");   // the whole fix
props.put("acks", "all");                  // implied by idempotence
props.put("retries", Integer.MAX_VALUE);   // implied; retry freely
props.put("max.in.flight.requests.per.connection", "5"); // <=5 required
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
KafkaProducer<String,String> producer = new KafkaProducer<>(props);
```

In modern Kafka (3.0+) `enable.idempotence=true` is the default, so you may already have it without knowing. When idempotence is on, Kafka silently forces three companion settings: `acks=all` (you cannot dedup against a write that was never durable on the leader), `retries` greater than zero (the whole point is to retry safely), and `max.in.flight.requests.per.connection` at most 5 (more than that and the broker cannot keep the per-partition ordering it needs to dedup correctly). If you explicitly set conflicting values, the producer throws a `ConfigException` at startup rather than silently weakening the guarantee — which is the right call.

So what does turning that flag on actually do on the wire? Two new pieces of state appear. First, when the producer initializes, it asks the broker to assign it a **producer ID** (PID) — a unique 64-bit integer that identifies this producer instance. Second, for every partition the producer writes to, it maintains a **sequence number** that starts at zero and increments by one for every record (actually every record within a batch; the batch carries the base sequence and the count). The producer stamps every record batch with its PID and the base sequence number of the first record in the batch. The broker, for each (PID, partition) pair, remembers the highest sequence number it has successfully appended. The figure below contrasts the two worlds: without idempotence a retried batch lands twice; with it, the second copy carries a sequence the broker has already seen, so the broker drops it.

![A before-and-after diagram contrasting a non-idempotent producer that lands two copies of a retried batch in the log against an idempotent producer whose retried batch carries a sequence number the broker has already seen and is deduplicated to one copy](/imgs/blogs/exactly-once-in-kafka-idempotent-producer-transactions-2.webp)

The PID is the producer's identity for the broker's bookkeeping, and the sequence number is the per-record counter the broker checks against. Together they form the deduplication key: the broker can ask, for any incoming batch, "have I already appended a record from this PID, on this partition, with this sequence number?" If yes, the batch is a duplicate retry and gets dropped. If the sequence is exactly one past the last one it saw, the batch is new and gets appended. If the sequence is further ahead than that — a gap — something is wrong and the broker rejects it with an `OutOfOrderSequenceException`, because a gap means a batch was lost and the per-partition ordering guarantee is broken.

### The PID is per-session, not durable forever

A crucial limitation that trips people up: the idempotent producer's guarantee holds only *within a single producer session*. The PID is assigned when the producer starts and is forgotten when the producer closes or crashes. If your producer process dies and a new one starts, it gets a *new* PID, and the broker's dedup state for the old PID is now useless against records the new producer sends. So the idempotent producer protects you against *retries within one producer's lifetime* — exactly the network-hiccup-and-resend case — but it does *not* protect you against application-level duplicates where your code calls `send()` twice for the same logical event, and it does *not* protect you across a producer restart. For cross-session and cross-restart guarantees you need transactions with a stable `transactional.id`, which we get to in section 4. The idempotent producer is necessary but not sufficient; it is the floor, not the ceiling.

### What the idempotent producer costs

Almost nothing. The PID assignment is one extra round trip at producer startup. The sequence number is a few bytes per batch. The broker's dedup state is a small amount of memory per (PID, partition) pair, bounded because the broker only needs to remember the last few batches per producer (controlled by how many in-flight requests it allows). There is no throughput penalty worth measuring — in benchmarks the idempotent producer runs within a percent or two of the non-idempotent one at the same `acks=all` setting, and since `acks=all` is what you should be running for durability anyway (see [Kafka replication](/blog/software-development/message-queue/kafka-replication-isr-acks-durability)), the marginal cost of idempotence over plain `acks=all` is essentially free. This is why Kafka made it the default. There is genuinely no reason to run a producer without it.

### The max.in.flight cap, and why it is five

The one config that surprises people is `max.in.flight.requests.per.connection` being capped at 5 under idempotence. Before idempotence, you could run with this much higher to keep the pipe full, and a higher value genuinely improves throughput on high-latency links because more batches are in flight at once. So why the cap? Because the broker's dedup logic must reason about sequence ordering, and to detect a gap correctly it can only tolerate a bounded amount of reordering among in-flight requests. With at most 5 in-flight batches, the broker keeps enough recent batch metadata to recognize a retry of any of them and to detect a true gap. Push past 5 and a retried batch could fall outside the window the broker retains, making it indistinguishable from a genuine out-of-order arrival — at which point the dedup guarantee breaks. Five is the number the Kafka authors picked as the safe ceiling where the broker can still maintain exactly-once ordering per partition without unbounded memory. In practice 5 in-flight requests is plenty to saturate most links; if you were relying on a higher value for throughput, batch larger instead.

There is a related historical footnote worth knowing. In Kafka versions before 1.0, idempotence required `max.in.flight=1`, which serialized sends and hurt throughput badly — a real reason teams avoided it. The improvement to allow up to 5 (KAFKA-5494, shipped in 1.0) is what made idempotence cheap enough to eventually become the default. If you ever read old advice that "the idempotent producer kills throughput because it forces in-flight to one," that advice is a decade stale; on any modern Kafka the producer keeps 5 batches in flight and the throughput cost is in the noise.

## 3. How the broker dedups producer retries

Let me make the dedup mechanism completely concrete, because "the broker checks the sequence number" hides the exact bookkeeping, and the exactness is where the guarantees live. The broker maintains, in memory and persisted in the producer-state snapshot for each partition, a small map keyed by PID. For each PID it stores the producer's epoch (more on epochs in section 4) and the sequence numbers of the last few batches it appended — specifically enough to cover the maximum number of in-flight requests, which is why that config is capped at 5.

When a batch arrives for partition P from producer PID with base sequence `S` and record count `N` (so it covers sequences `S` through `S+N-1`), the broker runs this check:

- If `S` equals `lastSeq + 1` (where `lastSeq` is the last sequence the broker appended for this PID on this partition), the batch is **in order and new**. Append it, set `lastSeq = S + N - 1`, ack.
- If the batch's sequence range exactly matches one the broker has already appended (a retry of a batch it has seen), the batch is a **duplicate**. Do *not* append. Return the same ack the broker would have returned originally — including the original offset and timestamp — so the producer believes its send succeeded and stops retrying. The duplicate vanishes silently; the application never knows.
- If `S` is greater than `lastSeq + 1`, there is a **gap** — a batch in between was never appended (lost, or arrived out of order beyond what the broker buffers). The broker rejects with `OutOfOrderSequenceException`. This is fatal for the producer's idempotent guarantee on that partition; the producer must be re-initialized.
- If `S` is less than `lastSeq + 1` but does not match a known batch, it is a stale or already-superseded retry and is also treated as a duplicate or rejected depending on whether it falls within the retained window.

The middle case — duplicate detected, original ack replayed — is the heart of it. The broker does not just drop the duplicate; it *lies convincingly* to the producer by returning the metadata of the original successful append. That is what lets the producer's retry loop terminate cleanly without the application ever seeing an error or a duplicate. This is the entire mechanism by which `enable.idempotence=true` "kills producer-retry duplicates."

#### Worked example: tracing a retry through the dedup check

Let me trace one concretely so the bytes are visible. Suppose a producer with PID `4217` is writing to partition `orders-3`. It has already successfully sent records with sequences 0 through 4; the broker's `lastSeq` for (PID 4217, orders-3) is 4. Now the producer sends a batch of three records with base sequence 5, covering sequences 5, 6, 7.

1. The batch travels to the leader. The leader checks: base sequence 5 equals `lastSeq + 1` = 5. In order and new. It appends records at log offsets, say, 1050, 1051, 1052, sets `lastSeq = 7`, and sends back an ack reporting base offset 1050.
2. **The ack is lost in the network.** The TCP connection hiccups; the producer's request times out. The records are safely on disk at offsets 1050–1052, replicated to the ISR, but the producer does not know that.
3. The producer's retry logic resends the *identical* batch — same PID 4217, same base sequence 5, same three records. It cannot do otherwise; from its point of view the send never completed.
4. The retried batch arrives at the leader. The leader checks: base sequence 5. Its `lastSeq` is now 7, so 5 is *not* `lastSeq + 1`. The leader looks at its retained batch metadata and finds that it already appended a batch with base sequence 5, count 3, at base offset 1050. **This is a duplicate.**
5. The leader does *not* append the records again. It returns an ack reporting base offset 1050 — the *original* offset. The producer receives this ack, concludes its send finally succeeded (at offset 1050), and stops retrying.

The net result: three records in the log, exactly once, at offsets 1050–1052. The producer believes it sent them once. The consumer will read them once. The duplicate retry was absorbed completely by the (PID, sequence) check and never touched the log. Run the same trace *without* idempotence and step 4 changes: the leader would append the three records again at offsets 1053–1055, and the consumer would see each order twice. That is the entire difference one flag makes, and the dedup is invisible to everyone — which is exactly what you want.

There is one subtlety worth stating: this works *per partition*. The sequence numbers are per (PID, partition), so a producer writing to twelve partitions maintains twelve independent sequence counters and the broker maintains twelve independent `lastSeq` values. A retry to partition 3 is deduped against partition 3's state and has nothing to do with partition 7. This is why the idempotent producer scales fine across high-fan-out producers; the bookkeeping is partitioned along with the data.

One more failure mode is worth naming because it surprises people: the `OutOfOrderSequenceException`. If the broker ever sees a gap — a base sequence higher than `lastSeq + 1` with no matching retained batch — it concludes that a batch in between was lost and the partition's per-producer ordering is broken. This is fatal for that producer session, and the client surfaces it as `OutOfOrderSequenceException`. It can happen if the broker's producer state was lost (for instance, an unclean leader election that rolled back the log past records the producer already sent, or a retention deletion that expired the producer's state entry before the producer was done). The correct response is to close and recreate the producer, which gets a fresh PID and resets the sequence bookkeeping. In a transactional producer this is handled for you by re-initializing; in a bare idempotent producer you must catch it and recreate. Seeing this exception in your logs is usually a signal of something upstream — an unclean election you should investigate, or a producer idle longer than `transactional.id.expiration.ms` — not a bug in your send loop.

## 4. Transactions: the transactional.id and the coordinator

The idempotent producer solves retry duplicates within one session for one record at a time. It does not solve two larger problems. First, **atomicity across partitions**: a producer that writes one logical event as records on three different partitions cannot, with idempotence alone, guarantee that either all three or none of them become visible — a crash between the first and second write leaves a torn partial state. Second, **atomicity across a restart**: if your producer crashes and a new instance starts, the new instance gets a fresh PID and cannot dedup against work the old instance did. Transactions solve both, and they do it by adding a stable identity and a coordinator that durably tracks the fate of each transaction.

The stable identity is the **`transactional.id`** — a string *you* choose and that survives restarts. Unlike the PID, which is broker-assigned and ephemeral, the `transactional.id` is application-assigned and durable. When a producer initializes transactions with the same `transactional.id` after a restart, the broker recognizes it as the *same logical producer* and can fence out the old instance. Here is the config:

```java
Properties props = new Properties();
props.put("bootstrap.servers", "broker1:9092,broker2:9092");
props.put("transactional.id", "order-enricher-task-3"); // stable, unique per task
props.put("enable.idempotence", "true");                 // required, implied
props.put("acks", "all");
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
KafkaProducer<String,String> producer = new KafkaProducer<>(props);
producer.initTransactions(); // one-time handshake with the coordinator
```

Setting `transactional.id` automatically turns on `enable.idempotence` — transactions are built on top of the idempotent producer, not beside it. The `transactional.id` must be **unique per producer instance** in a way that maps stably to the work that instance does. If two live producers share a `transactional.id`, the coordinator fences one of them off (the one with the older epoch), so picking the right granularity matters: in a partitioned read-process-write job, the standard pattern is one `transactional.id` per input-partition assignment, so that when the partition moves to another consumer the new owner reuses the same id and the old owner is fenced.

### The transaction coordinator and __transaction_state

Behind transactions sits a broker-side component called the **transaction coordinator**, and a special internal topic called **`__transaction_state`**. The coordinator is to transactions what the group coordinator is to consumer groups: a designated broker that owns the durable state for a set of `transactional.id`s. Which broker is the coordinator for a given `transactional.id` is determined by hashing the id to a partition of `__transaction_state` — the leader of that partition is the coordinator. This means the transaction state is itself stored as a Kafka log, replicated like any other topic, so the coordinator can fail over: if the broker hosting the coordinator dies, another broker takes over leadership of that `__transaction_state` partition and reconstructs the in-flight transaction state from the log.

When the producer calls `initTransactions()`, three things happen. The producer finds its coordinator (by asking any broker which broker owns its `transactional.id`'s partition). The coordinator assigns or recovers the producer's PID and, critically, **bumps the producer epoch**. The epoch is a monotonically increasing counter attached to the (transactional.id, PID) pair. Bumping it on every `initTransactions()` is the fencing mechanism: any older producer instance still running with the previous epoch will have its writes rejected by the broker as `ProducerFenced`, because the broker only accepts writes from the highest epoch it has seen for that transactional.id. This is how a restarted producer cleanly evicts a zombie predecessor that the system thought was dead but was actually just slow. The figure below shows the commit flow: the producer talks to the coordinator, the coordinator durably logs the decision into `__transaction_state`, and then commit markers fan out into every partition the transaction touched, including the offsets topic.

![A graph diagram showing a transactional producer sending commitTransaction to the transaction coordinator, the coordinator logging PREPARE_COMMIT to the internal transaction-state topic, and commit markers being written into two data partitions and the consumer-offsets partition](/imgs/blogs/exactly-once-in-kafka-idempotent-producer-transactions-3.webp)

The coordinator's job during a commit is genuinely a two-phase commit, but a special one where the participants are Kafka partitions rather than independent databases, and the coordinator's own log is the durable decision record. It first writes a `PREPARE_COMMIT` entry to `__transaction_state` — that is the durable point of no return, the moment after which the transaction *will* commit even if everything else crashes. Then it writes commit markers into every partition that participated, including the `__consumer_offsets` partition if offsets were part of the transaction. Then it writes a `COMPLETE_COMMIT` entry to close out the transaction. If the coordinator crashes after `PREPARE_COMMIT` but before finishing, the new coordinator reads the log, sees the prepared state, and *completes* the commit by writing the remaining markers. The decision is never lost once prepared.

### Why epochs and fencing matter more than they look

The fencing-by-epoch mechanism is the unsung hero of Kafka transactions and the reason the guarantee survives the nastiest failure mode in distributed systems: the zombie. Picture a producer that suffers a long stop-the-world garbage-collection pause or a network partition. The rest of the system gives up on it, a new instance starts with the same `transactional.id`, calls `initTransactions()`, and gets epoch N+1. The coordinator now refuses any write tagged with epoch N. When the zombie wakes up from its pause and tries to finish its transaction with epoch N, every one of its writes is rejected as `ProducerFenced`, and it shuts down. Without this, the zombie could commit a transaction the system already abandoned, producing exactly the duplicate or torn write that transactions are meant to prevent. The epoch is a generation number, and the broker's rule — only the latest generation may write — is what makes the guarantee hold under the partial failures that real systems actually experience.

### Choosing the transactional.id granularity

The most consequential design decision in hand-rolled transactional code is what string you put in `transactional.id`, because the fencing semantics flow directly from it. The rule is that the id must map *stably* to a unit of work, so that whoever currently owns that unit of work uses the same id and fences whoever owned it before. In a read-process-write job partitioned across N consumers, the right granularity is one `transactional.id` per *input partition* (or per stable assignment of input partitions). When a rebalance moves input partition 7 from consumer A to consumer B, consumer B initializes a producer with the same `transactional.id` that A used for partition 7, bumps the epoch, and fences A's producer. If instead you keyed the id by *consumer instance* (say, the hostname), then after a rebalance B would use a different id than A did, A's zombie producer would *not* be fenced, and it could commit work for partition 7 that B is now also processing — a duplicate. Getting this wrong is subtle because it only manifests during a rebalance plus a slow predecessor, which is rare enough to pass testing and common enough to bite in production. This is precisely the bookkeeping Kafka Streams handles for you, and the strongest argument for using Streams rather than rolling your own.

A common mistake at the other extreme is using a *single* shared `transactional.id` across all instances of a service for simplicity. That guarantees that only one instance can ever have a live transaction at a time — the others are all fenced — which serializes your entire job through one producer and destroys parallelism. The id must be unique per concurrently-active unit of work and stable across that unit's migrations: not too coarse (serializes), not too instance-specific (fails to fence). One id per input partition threads that needle.

## 5. The transaction lifecycle and control markers

Now the full lifecycle, the sequence of API calls your code makes and what each one does on the wire. The five methods are `initTransactions`, `beginTransaction`, `send` / `sendOffsetsToTransaction`, and `commitTransaction` (or `abortTransaction`). Here is the canonical loop:

```java
producer.initTransactions(); // ONCE at startup: fence predecessors, get epoch

while (running) {
    ConsumerRecords<String,String> records = consumer.poll(Duration.ofMillis(200));
    if (records.isEmpty()) continue;

    producer.beginTransaction();            // open a new transaction
    try {
        for (ConsumerRecord<String,String> rec : records) {
            ProducerRecord<String,String> out = transform(rec);
            producer.send(out);             // produce result(s) into the txn
        }
        // commit the SOURCE offsets as part of the SAME transaction
        Map<TopicPartition, OffsetAndMetadata> offsets = currentOffsets(records);
        producer.sendOffsetsToTransaction(
            offsets, consumer.groupMetadata());
        producer.commitTransaction();       // atomically commit results + offsets
    } catch (ProducerFencedException | OutOfOrderSequenceException e) {
        producer.close();                   // fatal: a zombie was fenced; bail out
        break;
    } catch (KafkaException e) {
        producer.abortTransaction();        // recoverable: discard partial work
    }
}
```

Walk the calls. `initTransactions()` runs once and does the coordinator handshake, epoch bump, and recovery of any pending transaction from a prior instance (it will abort or complete a transaction the previous instance left hanging). `beginTransaction()` is purely client-side bookkeeping — it does not talk to the broker; it just marks the start of a transactional scope locally. The first `send()` to a new partition triggers an `AddPartitionsToTxn` request to the coordinator, registering that partition as a participant so the coordinator knows where to write markers later. `sendOffsetsToTransaction()` does the same for the `__consumer_offsets` partition, making the offset commit part of the transaction. `commitTransaction()` triggers the two-phase commit described in section 4.

### Control markers: the records that aren't your records

Here is the mechanism that makes the whole thing visible to consumers correctly: **control markers** (also called control batches). When a transaction commits or aborts, the coordinator writes a special record — a commit marker or an abort marker — into each participating partition's log, at a real offset, interleaved with the data records. These markers are not your data; they are control records that the broker filters out before handing records to consumers, but they occupy offsets and they carry the transaction's outcome. A consumer reading the partition sees the data records of the transaction followed (eventually) by a commit or abort marker, and the marker is the signal that tells a `read_committed` consumer whether those preceding records are now visible or must be skipped.

This is why a transactional partition's offsets are not perfectly contiguous with your record count: some offsets are consumed by control markers. If you have ever computed end-offset-minus-committed-offset to estimate lag and gotten a number that did not match your record count on a transactional topic, control markers are why — they advance the offset without being data. The figure below shows the two-phase commit as a timeline: begin locally, send records and offsets, the coordinator logs PREPARE_COMMIT, commit markers get written into the partitions, and finally COMPLETE_COMMIT closes the transaction.

![A timeline diagram showing the two-phase commit of a Kafka transaction from beginTransaction through sending records and offsets, logging PREPARE_COMMIT, writing COMMIT markers into the partitions, and logging COMPLETE_COMMIT to close it out](/imgs/blogs/exactly-once-in-kafka-idempotent-producer-transactions-6.webp)

### Abort: the path that makes a crash safe

The abort path is what makes a mid-transaction crash harmless, and it is worth dwelling on because it is the failure case people fear. If `commitTransaction()` is never reached — the producer crashes, throws a recoverable exception, or the transaction times out (controlled by `transaction.timeout.ms`, default 60 seconds) — the coordinator eventually writes an **abort marker** into every participating partition instead of a commit marker. The data records the producer already wrote are physically still in the log (you cannot un-append to a log), but the abort marker tells every `read_committed` consumer to *skip* them. They become invisible ghosts: present on disk, occupying offsets, but never delivered to a committed reader. From the application's point of view, the transaction never happened. The crash left no torn write, no half-processed batch, no duplicate — just some skippable records that compaction or retention will eventually clean up. We will trace this exact scenario in the read-process-write worked example.

The transaction timeout matters operationally: if a producer hangs holding an open transaction, it blocks `read_committed` consumers from advancing past it (they cannot decide whether those records are committed until the marker arrives). The coordinator's timeout bounds this — after `transaction.timeout.ms` the coordinator force-aborts the hung transaction, writes abort markers, and unblocks consumers. Set it long enough to cover your real processing time but short enough that a hung producer does not stall your readers for an eternity.

## 6. read_committed: what consumers see

Transactions on the producer side are only half the system. They are useless unless consumers can be told to *respect* them — to skip records from aborted transactions and to not read records from transactions still in flight. That is the job of one consumer config: **`isolation.level`**. It has two values:

```java
Properties props = new Properties();
props.put("bootstrap.servers", "broker1:9092");
props.put("group.id", "downstream-readers");
props.put("isolation.level", "read_committed"); // the EOS-aware setting
props.put("enable.auto.commit", "false");
props.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
props.put("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
KafkaConsumer<String,String> consumer = new KafkaConsumer<>(props);
```

The default is `read_uncommitted`, and it means exactly what it says: the consumer reads every record in the log regardless of transaction state, including records from transactions that later abort and records from transactions still open. A `read_uncommitted` consumer of a transactional topic gets *ghost records* — data that was written inside a transaction that aborted, which from the application's perspective never happened, but which a `read_uncommitted` reader sees and processes anyway. If your downstream depends on transactional atomicity, `read_uncommitted` silently breaks it.

`read_committed` is the EOS-aware setting. A `read_committed` consumer sees only records from transactions that have *committed*, plus all non-transactional records (which are treated as always-committed). The mechanism is subtle and worth getting right. The consumer fetches records from the broker as usual, but the broker tells it the **last stable offset** (LSO) — the offset below which all transactions are decided (committed or aborted). The consumer will not deliver any record at or above the LSO, because those records belong to transactions whose fate is not yet sealed. Within the stable region, the consumer's client filters out records belonging to aborted transactions (the broker sends it the list of aborted transaction ranges) and the control markers, delivering only the committed data. The figure below contrasts the two: `read_uncommitted` leaks aborted records, while `read_committed` buffers past open transactions and skips aborted ones.

![A before-and-after diagram contrasting a read_uncommitted consumer that processes records from a transaction that later aborts against a read_committed consumer that buffers past open transactions and skips aborted records so only committed data is seen](/imgs/blogs/exactly-once-in-kafka-idempotent-producer-transactions-8.webp)

### The latency cost of read_committed

`read_committed` is not free, and the cost is *latency*, not throughput. Because a `read_committed` consumer cannot deliver records at or above the last stable offset, an open transaction *blocks* the consumer from reading past its first record until that transaction commits or aborts. If a producer holds a transaction open for 500 milliseconds while it batches work, every `read_committed` consumer downstream sees an extra ~500 milliseconds of latency on records that fall after the open transaction's start, because they cannot be delivered until the transaction's fate is sealed. This is the fundamental tradeoff of transactional reads: you trade head-of-line latency for atomicity. Keep your transactions short — batch a few hundred records, not a few hundred thousand — precisely because long transactions stall your readers. The figure layers this in: the consumer's view is clean, but it lags the producer by the transaction's open duration.

There is a second-order effect worth knowing: `read_committed` consumers can experience a *burst* when a long transaction finally commits. All the records that were buffered behind the open transaction become deliverable at once. So a topology with long transactions shows a sawtooth latency profile — quiet, then a burst on commit — rather than smooth flow. If you see periodic latency spikes on a `read_committed` consumer that line up with a producer's transaction boundaries, that is the cause, and the fix is shorter transactions, not a bigger consumer.

#### Worked example: measuring the read_committed latency tax

Put numbers on it so the tradeoff is concrete. Suppose a producer commits a transaction every 500 milliseconds, batching all the records it produced in that window into one transaction. A downstream `read_committed` consumer fetches from the partition, but the broker reports a last stable offset that sits at the start of the currently-open transaction. So the moment a transaction opens at time T, every record produced between T and T+500ms is *invisible* to the consumer until the commit marker lands at T+500ms. A record produced at T+50ms — near the start of the transaction — waits roughly 450ms before it becomes visible. A record produced at T+490ms waits only 10ms. Averaged across the window, records pick up about 250ms of extra latency they would not have under `read_uncommitted`, with a worst case approaching the full 500ms transaction duration.

Now shorten the commit interval to 100ms. The same arithmetic gives an average added latency of about 50ms and a worst case near 100ms — a fivefold improvement in the latency tax, paid for with five times as many transactions (five times the coordinator round trips and marker writes per second). If your producer was committing 2 transactions per second, it now commits 10. At, say, 50,000 records per second through the partition, the 100ms interval means transactions of ~5,000 records each instead of ~25,000 — still healthy batches, far better latency. The lesson the numbers teach: the commit interval is a direct latency dial for every `read_committed` consumer downstream, and the right setting is the shortest interval whose per-transaction overhead you can still afford. Defaulting to long transactions for "throughput" silently taxes everyone reading committed data.

## 7. Read-process-write atomicity

Now we assemble the pieces into the pattern that is the entire reason transactions exist for most teams: **read-process-write**. The shape is a consume-transform-produce loop, and the magic is that the offset commit of the *source* topic is done *inside the same transaction* as the produce to the *destination* topic. That single design move makes the whole step atomic: either the output records are produced *and* the source offsets advance, or neither happens. There is no in-between where you produced the output but failed to record that you consumed the input (which would reprocess and duplicate), and no in-between where you advanced the offset but failed to produce the output (which would skip and lose).

This is the crucial insight that separates transactional EOS from the naive consume-process-produce loop. In the naive loop you produce to the output topic, then *separately* commit your consumer offsets. Those are two independent operations, and a crash between them produces exactly the duplicate-or-loss you were trying to avoid: if you produced and crashed before committing the offset, you reprocess and re-produce on restart (duplicate); if you committed the offset and crashed before producing, you lose the output. The transactional loop collapses both into one atomic unit by routing the offset commit *through the producer* via `sendOffsetsToTransaction()` so it shares the transaction's fate. The figure below shows the atomic loop: poll, begin, process, produce, send offsets, commit — all one transaction.

![A grid diagram of the read-process-write loop showing poll, beginTransaction, process, produce results, sendOffsetsToTransaction, and a single atomic commitTransaction that ties the output records and the source offsets together](/imgs/blogs/exactly-once-in-kafka-idempotent-producer-transactions-4.webp)

Notice what changed structurally: the consumer no longer commits its own offsets. `enable.auto.commit` is `false`, and you never call `consumer.commitSync()`. Instead the *producer* commits the offsets via `sendOffsetsToTransaction()`, passing the consumer's group metadata so the offsets land in the right group's `__consumer_offsets` partition as part of the transaction. The consumer's only job is to `poll()` and to be `read_committed` if it reads transactional input. This inversion — producer owns the offset commit — is the single most surprising thing about transactional EOS for people seeing it the first time, and it is the linchpin of the whole pattern.

#### Worked example: a crash mid-transaction in a read-process-write loop

Let me trace a crash and show precisely what it leaves behind, because this is where the guarantee earns its keep. We have an enrichment job: it reads orders from topic `orders` partition 2, enriches each with customer data, and writes to topic `enriched-orders` partition 2. It uses a transactional producer with `transactional.id = order-enricher-p2`.

1. The consumer polls and gets 50 order records from `orders-2`, at offsets 8000 through 8049.
2. The producer calls `beginTransaction()`. It processes all 50, producing 50 enriched records, sending them to `enriched-orders-2`. The first `send` to that partition triggers `AddPartitionsToTxn`, registering `enriched-orders-2` as a participant. The 50 enriched records are now *physically in the log* of `enriched-orders-2`, at some offsets, but no commit marker has been written.
3. The producer calls `sendOffsetsToTransaction()` with offset 8050 (one past the last consumed) for `orders-2`, registering `__consumer_offsets` as a participant. This too is now pending, not committed.
4. **The process crashes** — `kill -9`, OOM, whatever — *before* `commitTransaction()` is reached.

What is the state of the world? The 50 enriched records sit in `enriched-orders-2`'s log with *no commit marker*. The offset 8050 sits in `__consumer_offsets` with *no commit marker*. The transaction is open and orphaned. Now the recovery:

5. The coordinator's `transaction.timeout.ms` (say 60 seconds) elapses with no commit. The coordinator force-aborts: it writes an **abort marker** into `enriched-orders-2` and into the `__consumer_offsets` partition.
6. A new enrichment instance starts with the same `transactional.id = order-enricher-p2` and calls `initTransactions()`. This bumps the epoch (fencing any zombie) and the coordinator confirms the prior transaction is aborted.
7. The new instance's consumer reads `__consumer_offsets` and finds the last *committed* offset for `orders-2` is still 8000 (the offset 8050 was aborted, never committed). So it resumes from 8000 and re-polls the same 50 records.

What does a downstream `read_committed` consumer of `enriched-orders-2` see across this whole episode? **Nothing from the aborted transaction.** The 50 enriched records the crashed instance wrote are in the log, but the abort marker tells `read_committed` consumers to skip them — they are ghosts. The downstream sees the 50 records exactly once, produced by the *retry* in step 7, which commits successfully. No duplicate, no loss, no torn state. The crash that would have produced 50 duplicates (or 50 lost outputs) in the naive loop produced *nothing visible* under transactions. That is exactly-once processing, and the abort marker plus the atomic offset commit are what delivered it.

Contrast the naive non-transactional version of step 4: if the job had produced the 50 records, then crashed before committing offset 8050, the restart would resume from 8000, reprocess the same 50, and produce 50 *more* enriched records — duplicates, fully visible to any consumer. The transaction is the difference between 50 ghosts that nobody sees and 50 duplicates that everybody does.

## 8. Exactly-once in Kafka Streams

Everything in sections 4 through 7 is real and you can wire it by hand, but it is fiddly: getting the `transactional.id` granularity right, inverting offset ownership to the producer, handling fencing exceptions, sizing transaction timeouts. Kafka Streams hides all of it behind one config flag:

```java
Properties props = new Properties();
props.put(StreamsConfig.APPLICATION_ID_CONFIG, "order-enrichment-app");
props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "broker1:9092");
props.put(StreamsConfig.PROCESSING_GUARANTEE_CONFIG, "exactly_once_v2"); // the flag
// that is it — Streams wires transactional producers, read_committed
// consumers, per-task transactional.ids, and state-store changelogs into one
// atomic unit for you.
```

Setting `processing.guarantee=exactly_once_v2` makes Kafka Streams do all the transactional plumbing automatically. It assigns a `transactional.id` per task (derived from the application id and the task's partition), so that when a task migrates during a rebalance the new owner reuses the id and fences the old owner. It sets consumers to `read_committed`. It commits input offsets through the producer inside the transaction. And — this is the part you cannot easily do by hand — it makes the **state store changelog** part of the same transaction. A Streams application that maintains state (a count, a join table, a windowed aggregate) writes that state to a local store backed by a changelog topic; under `exactly_once_v2` the changelog writes are *also* in the transaction, so the persisted state is consistent with the output and the offsets. A crash cannot leave the state store ahead of or behind the output.

The `_v2` suffix matters. The original `exactly_once` (now deprecated) used one producer per task, which meant a topology with hundreds of tasks opened hundreds of producers and hundreds of transactions — expensive in connections, memory, and broker-side coordinator load. `exactly_once_v2` (introduced in Kafka 2.5, the only one you should use today) uses a single producer per *instance* that multiplexes all its tasks' partitions into shared transactions, dramatically reducing the overhead. If you are on a modern Kafka and see `exactly_once` without the `_v2`, that is a config to upgrade. The figure below stacks the EOS layers Streams assembles for you: idempotent producer, transactions, read_committed, and the atomic read-process-write step on top.

![A stack diagram showing the four exactly-once layers Kafka assembles: the idempotent producer that kills retry duplicates, transactions for atomic multi-partition writes, read_committed that hides aborted records, and the atomic read-process-write step on top](/imgs/blogs/exactly-once-in-kafka-idempotent-producer-transactions-5.webp)

### What Streams EOS costs in throughput and latency

There is a real cost, and you should size it. Under `exactly_once_v2`, Streams commits a transaction every `commit.interval.ms` (default 100ms under EOS, versus 30 seconds without). Each commit is a round trip to the coordinator plus marker writes, so a shorter commit interval means more transactional overhead but lower latency; a longer one means fewer transactions but more head-of-line blocking for downstream `read_committed` consumers (since records are not visible until the transaction commits). The default 100ms is a reasonable balance. Benchmarks typically show EOS costing on the order of 10–25% throughput versus at-least-once for the same topology, driven mostly by the commit frequency and the `read_committed` latency, not by the idempotent producer (which is nearly free). That is a meaningful but usually acceptable cost for the correctness it buys in a stateful streaming job — and it is a cost you pay only on Kafka-to-Kafka flows, which brings us to the wall.

### What Streams EOS does to the state store on crash

The part of Streams EOS that earns its keep, and that you genuinely cannot replicate by hand without enormous effort, is the handling of the **state store** on a crash. A stateful Streams operation — a `count`, an `aggregate`, a windowed join — keeps its running state in a local RocksDB store, backed by a changelog topic in Kafka for recovery. Under at-least-once, a crash can leave the local store and the changelog out of sync: the store advanced in memory but the changelog write was lost, or vice versa, and on restart the recovered state can double-count records processed just before the crash. This is exactly the silent-corruption-of-aggregates bug that makes "my counts are slightly off and I cannot reproduce it" such a nightmare.

Under `exactly_once_v2`, the changelog writes are part of the same transaction as the output records and the input offsets. So a crash mid-transaction aborts *all three together*: the output is ghosted, the offsets do not advance, and the changelog write is aborted. On restart, Streams discards the local store contents that were not committed and rebuilds from the committed changelog up to the committed offset — landing in a state perfectly consistent with the input it has acknowledged. Your aggregate is exactly right, not "right except for the records in flight during the last crash." For any job that maintains state, this is the entire reason to pay the EOS tax: at-least-once stateful processing produces aggregates that are subtly, unreproducibly wrong, and EOS makes them exactly right. The figure earlier in the post stacks these layers; the state-store-in-the-transaction is the piece that only Streams adds.

## 9. The boundary: where EOS stops and idempotency begins

Here is the wall, drawn explicitly, because everything above can lull you into thinking Kafka has solved exactly-once for your whole system. It has not. **Kafka's exactly-once guarantee holds for state that lives in Kafka — output topic records and consumer offsets — and ends the instant a side effect leaves Kafka.** The transaction can atomically write to `enriched-orders` and commit offsets, but it cannot reach into your Postgres, your Redis, your Stripe API, or your SMTP server and make those writes part of the transaction. The moment your processing does anything external, you have stepped outside the boundary, and the external effect is governed by at-least-once delivery and needs its own idempotency.

Why can't Kafka extend the transaction to your database? Because a Kafka transaction is a two-phase commit *among Kafka partitions*, coordinated by the transaction coordinator using `__transaction_state` as its durable decision log. To include your database, the database would have to participate in that two-phase commit as a resource manager — speaking Kafka's commit protocol, holding prepared state, honoring the coordinator's commit-or-abort decision. Databases do not do this; there is no XA-style distributed transaction spanning Kafka and arbitrary external systems. So the transaction's reach is exactly the set of Kafka partitions it touched, and not one byte further.

This means a very common architecture is **not** exactly-once even with `exactly_once_v2` turned on: a Kafka Streams job (or any consumer) that reads a topic and writes to a database. The Kafka side is exactly-once — offsets advance transactionally — but the database write is a side effect outside the transaction, so on a crash-and-retry the same record can be processed again and the database write can happen twice. The flag bought you nothing for the database write. To make *that* exactly-once you need an idempotent database write: a deduplication key (the source topic-partition-offset is a natural one), a unique constraint, or an upsert — exactly the machinery covered in [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe). Kafka's transaction gets the message to the edge of your system atomically; your idempotency carries it the last mile.

The cleanest way to handle the Kafka-to-external boundary is the **outbox pattern**, which inverts the problem: instead of trying to make Kafka and your database commit atomically, you make your database write and an "outbox" record commit atomically *in your database's own transaction*, then a separate process reads the outbox and publishes to Kafka idempotently. The atomicity lives in the database; Kafka publishing is at-least-once and made safe by the idempotent producer plus a dedup key. This is the subject of [change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern), and it is the right tool whenever your source of truth is a database rather than Kafka. The taxonomy figure below maps the Kafka EOS mechanisms so you can see that all three — idempotent producer, transactions, Streams EOS — live entirely inside the Kafka boundary.

![A tree diagram showing the taxonomy of Kafka exactly-once mechanisms with the idempotent producer providing PID and sequence numbers, transactions providing the coordinator and commit markers and read_committed, and Streams EOS providing the built-in read-process-write guarantee](/imgs/blogs/exactly-once-in-kafka-idempotent-producer-transactions-9.webp)

### A rule for finding the boundary in your own design

Trace every effect your processing produces and ask one question of each: *does this effect land in a Kafka topic or in the offsets, or somewhere else?* If it lands in Kafka, a transaction can make it exactly-once. If it lands anywhere else — a database, a cache, an API, a file, an email — it is outside the boundary and needs its own idempotency. Draw the line through your topology at exactly the points where effects leave Kafka, and put a deduplication key at each of those points. Inside the line, lean on transactions; outside it, lean on idempotency. The most dangerous bugs come from teams that turned on `exactly_once_v2`, saw "exactly-once" in the config, and assumed the line was around their *whole system* when it was only ever around the Kafka part.

### Why the natural dedup key is the source offset

When you do reach the external boundary and need an idempotency key, the best one is usually free: the **source topic-partition-offset** of the record that triggered the effect. It is unique (no two records share a (topic, partition, offset) triple), it is stable across retries (a reprocessed record carries the same offset), and it is already in your hands — every `ConsumerRecord` exposes its partition and offset. So an external write made idempotent on `(topic, partition, offset)` is exactly-once-safe against reprocessing: the first time you see offset 8042 you perform the write and record the key; if a crash makes you reprocess offset 8042, the recorded key short-circuits the second write. This is the bridge that carries Kafka's in-boundary exactly-once across the wall to your external system. Kafka transactions advance the offset atomically inside the boundary; the source offset as a dedup key carries that same atomicity to the database or API outside it. The two halves compose into end-to-end exactly-once *if and only if* you build the second half — and that second half is your code, not Kafka's. The [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) post is the full treatment of how to store and expire those keys without unbounded growth.

There is one trap with the offset-as-key approach: it only holds if the same record always maps to the same external write. If your processing is non-deterministic — it reads the current time, a random number, or a value from a third system that changes between runs — then reprocessing the same offset can produce a *different* effect, and dedup on the offset will skip the second, different write, which may not be what you want. For deterministic transformations the source offset is the perfect key; for non-deterministic ones you need a key derived from the effect's intended outcome, not the input that produced it. Most stream processing is deterministic, so the source offset works the vast majority of the time, but the non-determinism caveat is the kind of thing that produces a once-a-quarter mystery if you do not account for it.

## Case studies and war stories

These are the patterns and incidents that teach the lesson better than any spec.

**The double-charge that survived `exactly_once_v2`.** A payments team built a Kafka Streams job that consumed a `payments-requested` topic and, for each request, called the Stripe API to charge the card, then produced a `payment-completed` event. They turned on `processing.guarantee=exactly_once_v2`, saw "exactly-once" in the config, and shipped. A broker rebalance caused a task to migrate; the new task reprocessed a handful of records the old task had already charged but not yet committed the offset for — because the Stripe call was *outside* the transaction. Customers were charged twice. The Streams EOS flag made the *Kafka* side exactly-once (the `payment-completed` events were not duplicated) but did nothing for the Stripe call, which was the side effect that actually mattered. The fix was an idempotency key on the Stripe request (Stripe supports `Idempotency-Key` headers natively) derived from the source offset, turning the external charge into an at-least-once-safe operation. Lesson: the EOS flag protects the Kafka boundary, never the external effect — and the external effect is usually the one that hurts.

**The `read_uncommitted` reader that processed ghosts.** A data platform team had a transactional producer writing curated events and a downstream analytics consumer aggregating them. The analytics numbers were subtly inflated, and nobody could explain why until someone noticed the consumer was running the default `isolation.level=read_uncommitted`. It was reading records from transactions that *aborted* — ghost records that should never have counted — and folding them into aggregates. Because aborts were rare (only on the occasional producer crash), the inflation was small and intermittent, which made it maddening to diagnose. Setting `isolation.level=read_committed` fixed it instantly. Lesson: transactions on the producer are only half the system; a consumer that does not opt into `read_committed` sees through the transaction and processes records that atomically never happened.

**The long transaction that stalled the readers.** A streaming team batched aggressively to maximize throughput, holding a transaction open while accumulating tens of thousands of records before committing — sometimes for tens of seconds. Throughput on the producer looked great. But downstream `read_committed` consumers showed a brutal sawtooth latency: quiet for tens of seconds, then a flood when the transaction finally committed and the buffered records all became visible at once. End-to-end p99 latency was terrible despite excellent producer throughput. The fix was to commit far more frequently — every few hundred milliseconds — accepting slightly more transactional overhead in exchange for smooth, low-latency delivery to readers. Lesson: a `read_committed` consumer cannot read past an open transaction; long transactions are head-of-line blocking for everyone downstream, and producer throughput is the wrong metric to optimize in isolation.

**The zombie that tried to commit.** A processing job experienced a multi-second stop-the-world GC pause. The cluster's session timeout elapsed, the consumer group rebalanced, and the partition moved to a fresh task that started a new transaction. When the original task woke up from its pause and tried to commit the transaction it had been holding, every write was rejected as `ProducerFenced` because the new task had bumped the epoch on its `initTransactions()`. The job logged the fence exception, closed the producer, and exited cleanly; the new task had already correctly processed the records. Lesson: this is the system working *as designed*. The epoch-fencing mechanism is precisely what prevents a zombie from committing work the cluster already reassigned, and a `ProducerFencedException` is not a bug to suppress — it is the guarantee doing its job. The only mistake is catching it and retrying instead of shutting down.

## When to reach for transactions (and when not to)

Be decisive about which mechanism a given workload needs, because reaching for the heavy one when the light one suffices is a common and costly mistake.

**Always turn on the idempotent producer.** `enable.idempotence=true` is the default in modern Kafka, costs essentially nothing over `acks=all`, and kills retry duplicates. There is no workload where you want it off. If you are on an older client where it is not the default, turn it on. This is not a decision; it is hygiene.

**Reach for transactions when, and only when, your atomic unit is entirely inside Kafka.** The canonical fit is the read-process-write topology: consume from Kafka, transform, produce to Kafka, commit offsets — all Kafka-to-Kafka. If that describes your job and you need each input processed exactly once into the output, transactions (or `exactly_once_v2` if you are using Streams) are exactly right. The classic examples are stream enrichment, aggregation, repartitioning, and joins — pure Kafka-to-Kafka data plumbing.

**Do not reach for transactions when your effect is external.** If your processing's real output is a database write, an API call, or an email, transactions buy you nothing for that effect — it is outside the boundary. Turning on `exactly_once_v2` here is worse than useless: it pays the throughput cost and gives a false sense of safety while the external effect remains at-least-once. Use idempotency at the external effect instead — a dedup key, a unique constraint, an upsert, or a provider-native idempotency key like Stripe's. For database-sourced events, use the [outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) rather than trying to span Kafka and the database transactionally.

**Do not reach for transactions for a fire-and-forget producer.** If you are just publishing events and not consuming-and-producing in a loop, you do not need transactions — the idempotent producer alone handles your retry duplicates. Transactions add the coordinator round trips, the commit markers, and the `read_committed` latency tax for no benefit if you are not making a multi-partition or offset-coupled write atomic.

Here is a compact decision table:

| Your situation | Mechanism | Why |
| --- | --- | --- |
| Any producer at all | `enable.idempotence=true` | Kills retry duplicates, ~free |
| Fire-and-forget publishing | Idempotent producer only | No atomic unit to protect |
| Read-process-write, Kafka-to-Kafka | Transactions (or `exactly_once_v2`) | Atomic output + offset commit |
| Stateful stream processing | `exactly_once_v2` in Streams | Wires changelog into the txn too |
| Output is a database/API/email | Idempotency key at the effect | Outside Kafka's boundary |
| Source of truth is a database | Outbox pattern + idempotent producer | Atomicity lives in the DB |

The matrix figure earlier in the post climbs this same ladder from possible-loss up to in-Kafka exactly-once; the table above is the operational version of it.

![A matrix table mapping producer config tiers from acks=1 through transactions against whether each tier permits loss, permits producer duplicates, or supports atomic multi-partition writes](/imgs/blogs/exactly-once-in-kafka-idempotent-producer-transactions-7.webp)

## Key takeaways

- **Kafka's exactly-once is effectively-once processing for Kafka state only** — output records and consumer offsets. It is not exactly-once delivery (impossible) and not end-to-end exactly-once across external systems (Kafka cannot reach them).
- **The idempotent producer (`enable.idempotence=true`) kills retry duplicates** by stamping each record with a producer ID and a per-partition sequence number; the broker drops any batch whose sequence it has already seen and replays the original ack. It is the default, costs almost nothing, and you should never run without it.
- **The idempotent producer's guarantee is per-session and per-partition.** A new PID after a restart cannot dedup against the old one's work; for cross-restart and cross-partition atomicity you need transactions with a stable `transactional.id`.
- **Transactions add a stable `transactional.id`, a transaction coordinator, and the internal `__transaction_state` topic.** Commit is a two-phase write: the coordinator logs `PREPARE_COMMIT` durably, then writes commit markers into every participating partition, including `__consumer_offsets`.
- **Epoch fencing is the quiet hero.** Each `initTransactions()` bumps the producer epoch; the broker accepts writes only from the latest epoch, so a zombie producer that wakes from a GC pause is fenced off with `ProducerFencedException` and cannot commit abandoned work.
- **Control markers (commit/abort) are real records in the log.** An aborted transaction's data records physically remain on disk but are skipped by `read_committed` consumers — invisible ghosts that retention eventually cleans up.
- **`isolation.level=read_committed` is mandatory for consumers of transactional topics.** The default `read_uncommitted` reads through transactions and processes aborted ghost records. The cost of `read_committed` is head-of-line latency, not throughput — keep transactions short.
- **Read-process-write atomicity comes from committing source offsets inside the transaction** via `sendOffsetsToTransaction()`. The producer, not the consumer, commits the offsets, so output records and read position share one fate.
- **`exactly_once_v2` in Kafka Streams wires all of this — plus the state-store changelog — into one atomic unit per task.** Use `_v2`, never the deprecated per-task `exactly_once`. Budget roughly 10–25% throughput cost.
- **The boundary is everything.** EOS ends where a side effect leaves Kafka. Database writes, API calls, and emails are at-least-once and need their own idempotency key. Trace where effects exit Kafka and put a dedup key at each exit.

## Further reading

- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — the log, offsets, and the exactly-once figure this post deepens.
- [Delivery semantics: at-most, at-least, exactly once](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) — why exactly-once delivery is impossible at the wire and what effectively-once means.
- [Idempotency and deduplication: making at-least-once safe](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) — the dedup-key machinery you need at every external boundary.
- [Kafka replication: ISR, acks, and durability](/blog/software-development/message-queue/kafka-replication-isr-acks-durability) — why `acks=all` is the durability floor that idempotence builds on.
- [Change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — the right tool when your source of truth is a database, not Kafka.
- Apache Kafka documentation: the producer configs (`enable.idempotence`, `transactional.id`), the consumer `isolation.level`, and the Kafka Streams `processing.guarantee` settings.
- KIP-98 (Exactly-Once Delivery and Transactional Messaging) and KIP-447 (Producer Scalability for Exactly-Once Semantics) — the design documents behind the idempotent producer, transactions, and `exactly_once_v2`.
