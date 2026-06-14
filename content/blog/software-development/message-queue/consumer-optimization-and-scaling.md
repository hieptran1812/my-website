---
title: "Consumer Optimization and Scaling: Prefetch, Concurrency, and Poll Tuning"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Make your consumers keep up: batch the fetch so you stop paying a round trip per record, size max.poll.records against the max.poll.interval trap, scale out only to the partition ceiling, scale up with parallel workers without breaking ordering, and decouple fetch from process so a slow handler never gets you evicted."
tags:
  [
    "message-queue",
    "consumers",
    "scaling",
    "throughput",
    "kafka",
    "rabbitmq",
    "distributed-systems",
    "event-driven",
    "backpressure",
    "concurrency",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/consumer-optimization-and-scaling-1.webp"
---

Producers almost never become the bottleneck. You can fire-and-forget a few hundred thousand messages a second into a single Kafka partition from one process without breaking a sweat, and a RabbitMQ publisher with confirms batched is similarly cheap. The reason your pipeline backs up at three in the morning, the reason lag climbs into the millions, the reason the on-call pager goes off — it is almost always the consumer side. The consumer is where the actual work happens: the database write, the HTTP call to a flaky third party, the JSON parse, the model inference. Producing is a memcpy and a network send. Consuming is your whole business logic. So when people say "the queue is slow," what they almost always mean is "my consumers cannot keep up with the producers," and the fix lives entirely on the read side of the pipeline.

This post is about that read side, and specifically about the gap between a consumer that is configured out of the box and a consumer that is tuned to saturate the hardware you are paying for. The default Kafka consumer fetches conservatively, polls in a tight loop, and processes one record at a time on the poll thread. That configuration will get you maybe a few thousand records a second on real work, and you will conclude Kafka is slow. It is not slow. You are doing one network round trip per record, leaving seven of your eight cores idle, and committing offsets in a way that throttles you further. Fix the fetch batching, move the work off the poll thread, fan it across a worker pool, and the same consumer on the same hardware does fifty times the throughput. None of that is exotic. It is four or five config values and one structural decision about where your processing runs.

![A two-panel comparison showing record-by-record fetch where every record costs one network round trip against batched fetch where one round trip pulls hundreds of records at once](/imgs/blogs/consumer-optimization-and-scaling-1.webp)

There is also a trap waiting in the middle of all this, and it catches even experienced engineers: `max.poll.interval.ms`. The instant your batch processing takes longer than that interval, the Kafka group coordinator decides your consumer is dead, kicks it out, and triggers a rebalance — even though your consumer is alive and busily working. Now the batch you were processing gets reassigned to another consumer and reprocessed, your offsets are in a confusing state, and if you misread the symptom you will "fix" it by adding more consumers, which makes the rebalancing worse. Understanding that trap, and the relationship between batch size, per-record processing time, and the poll interval, is the single highest-leverage thing in this entire post. Get it wrong and no amount of scaling saves you.

By the end you will be able to size `fetch.min.bytes`, `fetch.max.wait.ms`, `max.partition.fetch.bytes`, and `max.poll.records` from first principles; you will know exactly when adding consumers helps and when they just idle against the partition ceiling; you will know how to parallelize processing inside a single consumer without losing ordering; and you will be able to decouple fetching from processing with pause and resume so a slow handler never gets you evicted. We will work the numbers twice — once for the partition ceiling and once for the poll-interval trap — so the math is concrete, not hand-wavy. This is the companion to [Push vs Pull](/blog/software-development/message-queue/push-vs-pull-acknowledgements-how-consumers-read), which covers how consumers read at all, and to [Kafka consumer groups, offsets, and rebalancing](/blog/software-development/message-queue/kafka-consumer-groups-offsets-rebalancing), which covers the group machinery this post tunes against.

## 1. Why the consumer is usually the bottleneck

Start with the asymmetry, because it explains everything downstream. A producer does a fixed, small amount of work per message: serialize it, append it to an in-memory batch, and let the client flush the batch over the network when it fills or the linger timer fires. That work is bounded and cheap — microseconds of CPU and an amortized fraction of a network round trip. A consumer does an unbounded, application-defined amount of work per message: it runs your handler. Your handler might write to Postgres, call a payment API, run a fraud model, or render a PDF. The producer's per-message cost is a constant of the messaging library. The consumer's per-message cost is your code, and your code is almost always orders of magnitude more expensive than a memcpy.

That is why, in steady state, a pipeline's throughput ceiling is set by the consumers, not the producers or the broker. Brokers are deliberately built to be cheap: Kafka appends to a log and serves reads from the page cache, doing close to zero per-message CPU; RabbitMQ routes a message through an exchange with a hash or a trie lookup. The broker is a conveyor belt. The producers drop boxes on one end as fast as they can, the belt moves them along essentially for free, and the consumers at the far end have to actually open each box and do something with the contents. Whoever is opening boxes is the slow station, and that is the consumer.

### The two ways a consumer falls behind

A consumer falls behind in one of two ways, and the distinction matters because the fixes are different. The first is a **per-message efficiency problem**: each message individually costs more than it should because of overhead you can eliminate — a network round trip per record, a synchronous offset commit per record, a single CPU core doing work that could be spread across eight. This is the optimization half of the post, sections two through five and eight. You fix it by batching the fetch, batching the commit, and using more of the machine.

The second is a **raw capacity problem**: even with every message processed as efficiently as physically possible, the total volume exceeds what one consumer instance can do. A single consumer maxes out its cores or its downstream database connection pool, and you simply need more hands. This is the scaling half, sections six and seven, plus the autoscaling section nine. You fix it by adding consumer instances (scale out) up to the partition ceiling, by parallelizing within each instance (scale up), or both.

Most real incidents are a blend. A team launches a consumer with defaults, gets a few thousand records a second, sees lag climbing, and reaches straight for scale-out — they add ten more consumer pods. If the topic has only six partitions, four of those pods get no partitions and sit idle burning money, and the six that do get partitions are each still doing one round trip per record and using one core. They have scaled the wrong axis. The right first move was almost always to fix per-message efficiency on the consumers they already had. Scaling out an inefficient consumer just multiplies the inefficiency.

### The latency-versus-throughput frame

Everything on the consumer side is a negotiation between throughput and latency, the same negotiation covered in [Throughput vs latency](/blog/software-development/message-queue/throughput-vs-latency-tuning-tradeoff) for the pipeline as a whole. Batching the fetch raises throughput but adds up to `fetch.max.wait.ms` of latency on a near-empty queue. A bigger `max.poll.records` raises throughput by amortizing per-poll overhead but risks the poll-interval trap. More worker threads raise throughput but complicate ordering and can reorder side effects. There is no free lunch knob that improves both axes at once; every consumer setting buys one at the expense of the other or trades safety for speed. The art is knowing which axis your workload actually cares about and tuning hard toward it, rather than leaving everything at a default that splits the difference badly for your case.

### The arithmetic of keeping up

It helps to write down the one inequality that governs whether a consumer keeps up, because every decision in this post is an attempt to satisfy it. Let the producers write to a topic at a rate of *P* messages per second. Let each consumer instance process at *C* messages per second. Let there be *N* consumer instances in the group, capped at the partition count. The group keeps up if and only if *N* times *C* is greater than or equal to *P*. That is the entire game. If the left side falls below the right, lag grows without bound; if it stays above, lag drains to zero and stays there.

Every lever in this post moves one of those three variables. Fetch tuning and downstream batching raise *C* by eliminating per-message overhead. Scaling out raises *N* up to the partition ceiling. Scaling up raises *C* by using more cores per instance. Autoscaling adjusts *N* dynamically as *P* rises and falls. And the partition ceiling is the hard constraint that *N* can never exceed the partition count, which is why, once you hit that ceiling, the only remaining move is to raise *C* — you have run out of room to raise *N*. Keeping this inequality in your head turns vague worry about whether a pipeline will keep up into a concrete capacity calculation you can do on a napkin: measure *P*, measure *C*, count your partitions, and check whether the arithmetic closes.

The subtlety is that *C* is not a constant. It sags exactly when you need it most, because the downstream your consumer depends on — a database, an API, a cache — degrades under the same load spike that raised *P*. A consumer that did 7,500 records a second at noon might do 3,000 during an incident when its database is slow. So you do not size for the average *C*; you size for the *degraded* *C*, and you leave headroom. A group that is exactly at *N* times *C* equals *P* in steady state has zero margin and will fall behind the first time the downstream hiccups. The rule of thumb is to provision for roughly 1.5 to 2 times your steady-state need, so that when *C* sags you are still above *P* and lag drains rather than grows.

## 2. Fetch tuning: fetch.min.bytes and fetch.max.wait

The first and cheapest win is to stop doing a network round trip per record. By default a Kafka consumer is configured to be responsive: `fetch.min.bytes` defaults to 1, meaning "return as soon as you have even a single byte for me." On a busy topic that is not catastrophic because data is always available, but it sets you up for a pathological pattern where the broker returns tiny responses constantly, and on a topic that trickles you pay a full round trip to receive a handful of bytes. The cure is to tell the broker: do not bother waking me up until you have accumulated a worthwhile amount of data, or until a timeout fires, whichever comes first.

That is exactly what the pair `fetch.min.bytes` and `fetch.max.wait.ms` does. `fetch.min.bytes` is the minimum amount of data the broker will accumulate before responding to a fetch. Set it to, say, one megabyte and the broker will hold your fetch request open, accumulating records across the partitions you are subscribed to, until it has a megabyte to hand back. `fetch.max.wait.ms` (default 500ms) is the safety valve: even if the megabyte never accumulates, the broker will not make you wait longer than this — once the timer fires, it returns whatever it has, even if that is less than `fetch.min.bytes`. So the broker returns on whichever trigger fires first: enough bytes, or enough time.

![A two-panel before-and-after diagram contrasting a per-record fetch that does one network round trip per record against a batched fetch that pulls roughly five hundred records per round trip](/imgs/blogs/consumer-optimization-and-scaling-1.webp)

Look at figure 1 again with that framing. On the left, every record drags its own round trip behind it: the consumer asks, the broker answers with one record, the consumer asks again. With a network round trip time of around a millisecond, you have capped yourself at roughly a thousand records a second no matter how fast your processing is, because the network handshake alone eats the whole budget — you are network-bound, not CPU-bound, and adding cores does nothing. On the right, the consumer raised `fetch.min.bytes` to a megabyte with a fifty-millisecond wait. Now one round trip brings back hundreds of records, the per-record network cost collapses by two or three orders of magnitude, and the consumer becomes bound by how fast it can process the batch — which is the bound you actually want, because that is the one you can scale.

### The cost: latency on a quiet queue

The price of fetch batching is latency when the queue is nearly empty. If only one record arrives per second and you have set `fetch.min.bytes` to a megabyte, the broker will hold each record for the full `fetch.max.wait.ms` before giving up and handing it over, because a megabyte never accumulates. So a record that could have been delivered in a millisecond now waits up to your `fetch.max.wait.ms`. For a high-throughput pipeline this is irrelevant — the queue is never empty, the bytes accumulate in milliseconds, and you never hit the wait timer. For a low-latency request-reply path it can be unacceptable. The right move is to match the setting to the workload: large `fetch.min.bytes` with a modest wait for bulk pipelines, small `fetch.min.bytes` for latency-sensitive ones.

```python
from kafka import KafkaConsumer

# Throughput-tuned consumer: batch the fetch hard.
consumer = KafkaConsumer(
    "orders",
    bootstrap_servers="broker:9092",
    group_id="order-processors",
    # Do not return until 1 MB has accumulated...
    fetch_min_bytes=1_048_576,
    # ...but never make me wait longer than 50 ms for it.
    fetch_max_wait_ms=50,
    # Per-partition cap on a single fetch response (default 1 MB).
    max_partition_fetch_bytes=2_097_152,
    # Hand at most 500 records to one poll() call.
    max_poll_records=500,
    enable_auto_commit=False,
)
```

### max.partition.fetch.bytes and the big-message gotcha

There is a sibling knob that bites people: `max.partition.fetch.bytes` (default 1MB). This is the maximum amount of data the broker will return *per partition* in a single fetch. It exists to bound the memory a single fetch can consume, but it has a sharp edge. If a single message is larger than `max.partition.fetch.bytes`, older Kafka behavior could stall the consumer entirely — it could not fetch the message because the message exceeded the per-partition limit, and it could not skip it. Modern brokers will return an oversized message anyway to avoid the stall, but you still want this value comfortably above your largest expected message, and you want to size it against your total fetch buffer: a consumer subscribed to fifty partitions with `max.partition.fetch.bytes` of 2MB can theoretically pull 100MB in a single fetch, which has to fit in memory. The aggregate `fetch.max.bytes` (default ~50MB) caps the total across all partitions in one fetch, so it is the real memory ceiling for a single poll.

The mental model is a hierarchy of caps. `fetch.min.bytes` is the floor that triggers a response. `max.partition.fetch.bytes` is the per-partition ceiling. `fetch.max.bytes` is the all-partitions ceiling for one fetch. And `max.poll.records` — the next section — is the cap on how many of those fetched records get handed to a single `poll()` call. The fetch can pull a megabyte of records over the network and buffer them client-side; `max.poll.records` controls how many of those buffered records each `poll()` returns to your code. Fetch size is about network efficiency; poll size is about how much work you bite off per loop iteration.

## 3. max.poll.records and the poll loop

A Kafka consumer is a loop. You call `poll()`, you get a batch of records, you process them, you commit, and you call `poll()` again. That loop is the heartbeat of the consumer in more than a metaphorical sense: calling `poll()` is what tells the group coordinator you are alive. The whole design assumes you come back to `poll()` regularly, and `max.poll.records` is the knob that controls how much work you take on between two consecutive calls.

![A linear pipeline of the consumer loop showing fetch batch, then a poll buffer, then process, then commit, then poll again](/imgs/blogs/consumer-optimization-and-scaling-4.webp)

Figure 4 lays out the loop. A fetch pulls a megabyte of records over the network into a client-side buffer. Then each `poll()` call drains up to `max.poll.records` (default 500) from that buffer and returns them to your code. You process the batch, commit the offset, and loop. Critically, the fetch and the poll are decoupled inside the client: one network fetch can feed many `poll()` calls if the fetch brought back more records than `max.poll.records`. This is good — it means a big efficient fetch does not force you to process a thousand records in one gulp. You fetch big for network efficiency and poll in digestible chunks for processing control.

### Why max.poll.records exists at all

The reason this knob is separate from the fetch knobs is the poll-interval contract, which the next section covers in full. In short: the time between two `poll()` calls must stay under `max.poll.interval.ms`, and the time between two `poll()` calls is dominated by how long it takes you to process the batch the previous `poll()` returned. So `max.poll.records × your_per_record_processing_time` must stay under `max.poll.interval.ms` with margin. `max.poll.records` is the knob you turn to keep that product safe. If your handler is slow, you lower `max.poll.records` so each batch is smaller and processes faster, getting you back to `poll()` sooner. If your handler is fast, you can raise it to amortize per-poll overhead across more records.

```java
// Java consumer loop with explicit commit per batch.
while (running) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    for (ConsumerRecord<String, String> record : records) {
        process(record);            // your business logic
    }
    consumer.commitSync();          // commit after the whole batch
    // loop back to poll() promptly — this is the heartbeat
}
```

### Per-poll overhead and the case for bigger batches

Every `poll()` call has fixed overhead independent of how many records it returns: deserialization setup, iterator construction, the heartbeat bookkeeping, the offset accounting, and in many frameworks a layer of instrumentation and tracing. If you process one record per `poll()`, that fixed overhead is paid once per record and can easily dominate. If you process five hundred records per `poll()`, the overhead is amortized across five hundred records and becomes negligible. This is why processing in batches is faster even when the per-record work is identical: you are spreading the loop overhead thinner.

There is a second, larger win available when your downstream supports batch operations. If your handler writes each record to a database with a single-row INSERT, you do one round trip to the database per record, and that database round trip is usually far more expensive than the Kafka fetch round trip you already eliminated. But if you batch — collect the five hundred records from one `poll()` and write them with a single multi-row INSERT or a COPY — you collapse five hundred database round trips into one. That is frequently a ten-to-fifty-times throughput improvement on the part of the pipeline that actually matters, the downstream write. Batch the fetch, then batch the processing against the downstream, and you have attacked both round-trip taxes.

### Commit per batch, not per record

The third round-trip tax most consumers pay without noticing is the offset commit. A naive consumer commits the offset after every single record, calling `commitSync()` in the inner loop so that "if I crash I lose at most one record." That intention is reasonable but the implementation is expensive: `commitSync()` is a synchronous round trip to the group coordinator, and doing one per record means you pay a coordinator round trip per record on top of everything else — frequently the most expensive thing the loop does once you have batched the fetch and the downstream write. A consumer committing per record can be coordinator-round-trip-bound, capped at a few thousand commits a second no matter how fast everything else is.

The fix is to commit once per batch, not once per record, as the Java loop above already does. You process all five hundred records from a `poll()`, then call `commitSync()` once for the whole batch. The crash window grows from one record to one batch — on a crash you reprocess up to five hundred records instead of one — but since your processing is idempotent (it must be, because at-least-once delivery already forces that, as covered in [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe)), reprocessing five hundred records on the rare crash is harmless, and you have eliminated 499 coordinator round trips out of every 500. For even more throughput you can use `commitAsync()`, which fires the commit without waiting for the coordinator's acknowledgement, removing the round trip from the critical path entirely — at the cost that a failed async commit is not retried, so you pair it with a final `commitSync()` on shutdown to guarantee the last offset lands. The general rule mirrors fetch and downstream batching: amortize every per-record round trip — network, downstream, and coordinator — across a whole batch, and the only per-record cost left is the processing itself, which is the cost you actually wanted to pay.

## 4. The max.poll.interval trap and slow processing

Here is the trap that catches everyone at least once. Kafka's group coordinator needs to know whether a consumer is alive so it can reassign that consumer's partitions if it dies. It uses two mechanisms. A background heartbeat thread sends a heartbeat every `heartbeat.interval.ms` (default 3s), and if no heartbeat arrives within `session.timeout.ms` (default 45s in modern Kafka), the consumer is declared dead. That handles process crashes and network partitions. But there is a second, sneakier liveness check: `max.poll.interval.ms` (default 300000ms, five minutes). If the consumer does not call `poll()` again within this interval, the coordinator concludes the consumer is alive at the network level — heartbeats are still coming from the background thread — but **stuck**, unable to make progress, and it evicts it from the group and rebalances its partitions to other members.

![A timeline showing a slow batch that processes past the max.poll.interval, gets the consumer evicted by the broker, triggers a rebalance and reprocessing, then is fixed with a smaller batch and off-thread work](/imgs/blogs/consumer-optimization-and-scaling-7.webp)

Figure 7 walks the failure. A `poll()` returns five hundred records. The consumer starts processing. At two minutes it is still going. At five minutes — `max.poll.interval.ms` — it still has not come back to `poll()`, so the coordinator evicts it mid-batch. The partitions it was working get reassigned to another consumer, which starts reprocessing the same records from the last committed offset, because nothing in this batch was committed. Meanwhile the evicted consumer, when it finally finishes the batch and tries to commit, gets a `CommitFailedException` because it no longer owns those partitions. If your handler has side effects, those records were just processed twice, and possibly are being processed a third time by yet another consumer if the rebalance churns. This is how a single slow batch turns into a reprocessing storm and a cascade of rebalances.

#### Worked example: a 500-record batch at 12ms each versus a 300s interval

Let us put numbers on it. Suppose your handler does a database write and an external API call, and on average each record takes 12 milliseconds end to end — perfectly reasonable for real work. You left `max.poll.records` at the default of 500 and `max.poll.interval.ms` at the default of 300000ms (five minutes).

Time to process one batch: 500 records times 12 milliseconds equals 6000 milliseconds, or 6 seconds. Your poll interval budget is 300 seconds. So 6 seconds is comfortably under 300 seconds, and you will *not* rebalance. Good — defaults work here, with a 50-times safety margin. This is the happy case, and it is worth seeing that the defaults are not insane; they assume your per-record work is fast.

Now change one thing: the external API the handler calls degrades, and each call now takes 700 milliseconds instead of 12 because of retries against a struggling dependency. Time to process one batch: 500 records times 700 milliseconds equals 350000 milliseconds, or 350 seconds. Your budget is still 300 seconds. Now 350 exceeds 300, the coordinator evicts you mid-batch at the 300-second mark, and you rebalance. Worse, every consumer in the group is calling the same degraded API, so they all blow the interval, and you get a continuous rebalance storm where consumers are evicted, reassigned, evicted again — making zero forward progress while burning CPU on rebalances. The pipeline grinds to a halt precisely when it is under stress, which is the worst possible time.

How do you fix it? Two levers, and you usually pull both. First, lower `max.poll.records` so the batch is small enough to process within the interval even when the downstream is slow. If you set `max.poll.records` to 50, then even at the degraded 700ms per record a batch takes 50 times 700 equals 35000 milliseconds, or 35 seconds — comfortably under 300. Second, raise `max.poll.interval.ms` to give genuinely slow work more room, say to 600000ms (ten minutes), if your processing legitimately needs it. The combination — smaller batches and a longer interval — buys margin from both directions. But the cleaner structural fix, covered in section eight, is to stop processing on the poll thread at all: fetch on the poll thread and process on a worker pool, calling `poll()` frequently regardless of how slow the work is. Then the poll interval is decoupled from processing time entirely, and the trap simply cannot fire.

### The deeper rule: bound the work between polls

The general principle behind the trap is this: **the work you do between two `poll()` calls must be bounded, and that bound must stay under the poll interval with margin for the worst case, not the average case.** Your batch took 6 seconds on average, but the moment a dependency degrades it can take 350 seconds, and the interval check does not care about your average — it fires on the single slow batch. So size `max.poll.records` against your *worst-case* per-record time, not your typical time, or move the work off the poll thread so per-record time stops mattering for liveness at all. Engineers who size against the average get away with it for months and then get paged during exactly the incident — a degraded dependency — when they can least afford a rebalance storm on top.

## 5. RabbitMQ prefetch: the push-side equivalent

Everything above is about Kafka's pull model, where the consumer controls the rate by deciding when to fetch and how much. RabbitMQ is a push broker: it shoves messages down the channel to the consumer as fast as it can, without the consumer asking. That changes the shape of the tuning knob but not its purpose. In the pull world, you control flow by controlling your fetch. In the push world, you control flow with **prefetch**, set via the QoS (quality of service) call `basic.qos`, and it is the single most important consumer setting in RabbitMQ — the direct analog of all the Kafka fetch and poll tuning combined.

Prefetch (the `prefetch_count`) is the maximum number of unacknowledged messages the broker is allowed to have outstanding to a single consumer at any moment. The broker will push messages until the consumer has this many in flight and unacked, then it stops pushing to that consumer until some are acknowledged. It is a sliding window of permitted in-flight work. With a prefetch of 100, the broker keeps the consumer's pipeline full with up to 100 messages; as the consumer acks one, the broker pushes one more, keeping the window topped up. This is what makes a push broker safe: without it, the broker would push the entire queue at a slow consumer and bury it.

```python
import pika

connection = pika.BlockingConnection(pika.ConnectionParameters("rabbitmq"))
channel = connection.channel()

# The single most important RabbitMQ consumer knob.
# Allow at most 50 unacked messages outstanding to this consumer.
channel.basic_qos(prefetch_count=50)

def handle(ch, method, properties, body):
    process(body)
    ch.basic_ack(delivery_tag=method.delivery_tag)  # frees a prefetch slot

channel.basic_consume(queue="orders", on_message_callback=handle)
channel.start_consuming()
```

### The prefetch sizing tradeoff

Prefetch sizing is a Goldilocks problem with the same throughput-versus-fairness tension we keep meeting. Set prefetch to 1 and you get perfect fairness — the broker hands one message, waits for the ack, then considers who is least busy and hands the next — but you pay a full round trip of latency between every message, because the consumer is idle waiting for the next push while it acks. That caps you at roughly one message per round trip time, the same network-bound ceiling we eliminated on the Kafka side. Set prefetch to 10000 and you get maximum pipelining — the consumer always has work queued locally and never waits on the network — but you have destroyed fairness, because one consumer grabs ten thousand messages and a second consumer that joins finds the queue already drained into the first one's buffer, sitting idle while the first consumer slowly works through its hoard.

The standard guidance is to size prefetch to roughly the number of messages a consumer can process during one network round trip, plus a small buffer. If processing a message takes 10 milliseconds and your round trip to the broker is 1 millisecond, then in the time it takes to refill the pipeline you can process about ten messages, so a prefetch around 10 to 20 keeps the consumer busy without overcommitting. If your processing is slow — say 200 milliseconds per message — a prefetch of 1 or 2 is fine, because the consumer is the bottleneck anyway and a small window keeps the load balanced across consumers. The rule of thumb: fast processing wants a larger prefetch to hide round-trip latency; slow processing wants a small prefetch to keep work fairly distributed. This is the same logic as the Kafka fetch tuning, dressed in push-model clothes, and it is covered from the protocol angle in [Push vs Pull](/blog/software-development/message-queue/push-vs-pull-acknowledgements-how-consumers-read).

### A side-by-side of the knobs

| Concern | Kafka (pull) | RabbitMQ (push) |
| --- | --- | --- |
| Batch the network | `fetch.min.bytes` + `fetch.max.wait.ms` | `prefetch_count` (window of in-flight) |
| Cap work per unit | `max.poll.records` | `prefetch_count` |
| Liveness on slow work | `max.poll.interval.ms` trap | no equivalent (ack timeout / connection) |
| Per-partition cap | `max.partition.fetch.bytes` | n/a (per-queue, per-channel) |
| Flow control | consumer doesn't poll if busy | broker stops at prefetch window |

![A decision matrix mapping consumer knobs of fetch min bytes, max poll records, prefetch, and worker threads against throughput, latency, and rebalance risk](/imgs/blogs/consumer-optimization-and-scaling-2.webp)

Figure 2 collects the four main knobs into one decision matrix. Read it as a quick reference for the tradeoff each one makes. Raising `fetch.min.bytes` is almost pure upside for throughput and adds latency only on a quiet queue. Raising `max.poll.records` raises throughput by amortizing overhead but turns dangerous if your processing is slow, because of the poll-interval trap. RabbitMQ prefetch raises throughput and can even lower latency by pipelining, but set too high it creates unfair load across consumers. More worker threads raise throughput up to your core count and lower per-message latency, but they introduce ordering risk that you have to manage deliberately. No row improves every column; each is a deliberate trade you make knowing which axis your workload values.

## 6. Scaling out: consumers up to the partition count

Now the scaling half. The first and most natural way to scale a consumer is to run more of them — add instances to the consumer group and let the broker spread the partitions across them. This is **scaling out**, and it is the right first move when one consumer instance is genuinely saturated and you have already fixed per-message efficiency. But it has a hard ceiling that surprises people, and the ceiling is the partition count.

In a Kafka consumer group, each partition is assigned to exactly one consumer in the group at a time. That is the ordering guarantee — within a partition, messages are processed in order by a single consumer, which is the whole point of partitioning (covered in the capacity sibling, [partitioning and capacity planning](/blog/software-development/message-queue/partitioning-capacity-planning)). The direct consequence is that the maximum number of *useful* consumers in a group equals the number of partitions. If you have six partitions and six consumers, each consumer owns one partition — perfect. If you have six partitions and three consumers, each owns two. If you have six partitions and ten consumers, six of them own one partition each and the other four own *nothing*: they are members of the group, they poll, they receive nothing, and they sit idle burning CPU and money. This is the partition ceiling, and it is the most common scaling mistake in the entire ecosystem.

![A branching graph showing a topic with six partitions feeding several consumers up to the partition count with one extra consumer left idle holding zero partitions](/imgs/blogs/consumer-optimization-and-scaling-3.webp)

Figure 3 shows the ceiling directly. A topic with six partitions feeds a group. The first consumers each get partitions and do useful work. But the seventh consumer gets zero partitions — there is nothing left to assign — and it idles. Adding it did not increase throughput by one record per second; it only added a member to the group, which makes every future rebalance slightly slower because there is one more member to coordinate. The lesson is blunt: **you cannot scale a consumer group past its partition count.** If you need more parallelism than you have partitions, you must add partitions first (which has its own consequences for ordering and is not free), or you must scale up within each consumer (the next section), or both.

#### Worked example: 12 partitions with 8, 12, and 16 consumers

Make the ceiling concrete. You have a topic with 12 partitions. Each partition delivers a steady 5,000 records per second, so the topic produces 60,000 records per second total. A single consumer instance, well tuned, can process 7,500 records per second before it saturates its cores. You need enough consumers to handle 60,000 records per second, which is 60,000 divided by 7,500 equals 8 consumers' worth of processing capacity. Let us see what happens at three group sizes.

With **8 consumers** and 12 partitions, the assignment is uneven: 12 partitions across 8 consumers means four consumers get 2 partitions each (8 partitions) and four consumers get 1 partition each (4 partitions), totaling 12. A consumer with 2 partitions must handle 10,000 records per second, but it can only do 7,500 — so those four consumers are overloaded and lag builds on their partitions, while the four single-partition consumers cruise at 5,000 with headroom. Aggregate capacity is technically 8 times 7,500 equals 60,000, exactly matching the load, but the *uneven distribution* means some consumers are over and some under, and the over ones build lag. Eight consumers is right on the edge and unbalanced because 12 does not divide evenly by 8.

With **12 consumers** and 12 partitions, every consumer owns exactly one partition and handles exactly 5,000 records per second, which is under its 7,500 ceiling — a clean 33% headroom on every consumer, evenly distributed, no lag. This is the sweet spot: one consumer per partition, balanced load, room to absorb bursts. Twelve is the number you want.

With **16 consumers** and 12 partitions, the first 12 consumers each own one partition exactly as before, and the remaining **4 consumers own zero partitions and sit completely idle.** You are paying for 16 instances and getting the throughput of 12. The four extra add nothing but cost and rebalance overhead. The headroom per active consumer is identical to the 12-consumer case — adding consumers past the partition count does not add headroom, it adds waste. The only way to use those four extra instances would be to repartition the topic up to 16 partitions, at which point each of the 16 consumers would own one partition handling 3,750 records per second. The takeaway: size your consumer group to your partition count, and size your partition count to your *target* maximum parallelism, with the capacity-planning math living in the partitioning sibling post.

### The repartitioning escape hatch and its cost

When you hit the partition ceiling and genuinely need more parallelism, the escape hatch is to increase the partition count. Kafka lets you add partitions to a topic online. But it is not free and it is not always safe. Adding partitions changes the key-to-partition mapping for the default hash partitioner, because the partition is chosen by `hash(key) % partition_count` and you just changed the divisor. Keys that used to land on partition 3 now land on partition 7. If your application relies on per-key ordering — all events for user 42 on the same partition, processed in order — adding partitions breaks that guarantee for the transition period, because in-flight events for user 42 are spread across the old and new partition for that key. For ordered workloads you cannot casually add partitions; you over-provision partitions up front instead, choosing a partition count high enough for your maximum anticipated scale at topic-creation time. This is why partition count is one of the most consequential early decisions, and why it lives in its own capacity-planning post.

### Why scaling out is not free even below the ceiling

Even when you stay under the partition ceiling, adding and removing consumers is not instantaneous and not cost-free, and the cost is the rebalance. Every time a consumer joins or leaves the group, the coordinator must reassign partitions across the new set of members, and during a classic eager rebalance *all* consumers stop consuming, give up their partitions, and wait for the new assignment before resuming. This is the stop-the-world rebalance, and on a group with a long `max.poll.interval.ms` and slow handlers it can take seconds to tens of seconds, during which your entire group processes nothing and lag climbs. So scaling out from 6 to 12 consumers in response to a load spike triggers a rebalance that briefly *stops* all consumption right when you need it most — a counterintuitive and real cost.

Two features blunt this. **Cooperative rebalancing** (the `CooperativeStickyAssignor`, the modern default) changes the rebalance from stop-the-world to incremental: only the partitions that actually need to move are revoked, and consumers that keep their partitions never pause. A consumer joining a group of six no longer freezes all six; it just takes a partition or two from the consumers that have a surplus, and everyone else keeps working. **Static membership** (`group.instance.id`) lets a consumer that restarts — a routine deploy, a pod reschedule — rejoin within the session timeout *without* triggering a rebalance at all, because the coordinator recognizes the same instance ID coming back and hands it its old partitions. Together these turn scaling and deploys from disruptive stop-the-world events into smooth incremental ones, and they are close to mandatory for any group large enough or busy enough that rebalances hurt. The mechanics live in [Kafka consumer groups, offsets, and rebalancing](/blog/software-development/message-queue/kafka-consumer-groups-offsets-rebalancing); the takeaway here is that scaling out has a rebalance cost you should minimize before you lean on it heavily.

## 7. Scaling up: parallel processing within a consumer

If scaling out is capped at the partition count, and you need more throughput than that ceiling allows — or you simply want each consumer instance to use the eight cores you are paying for instead of one — you scale *up*: you parallelize processing inside a single consumer instance. The poll thread fetches a batch, then instead of processing the batch sequentially on that same thread, it hands the records to a worker pool that processes many at once across many cores. One consumer, eight workers, eight times the per-instance throughput.

![A before-and-after diagram contrasting single-threaded processing where one core is busy and seven idle against a parallel worker pool where eight workers process records across all cores](/imgs/blogs/consumer-optimization-and-scaling-5.webp)

Figure 5 shows the win. On the left, the poll thread does everything: it fetches and it processes, sequentially, on one core. Seven of your eight cores are idle. You are paying for an eight-core machine and using an eighth of it, and your throughput is capped at what one core can do — maybe five thousand records a second. On the right, the poll thread does nothing but fetch and dispatch; it hands records to a pool of eight workers, each on its own core, and now eight records process concurrently. The per-instance throughput jumps roughly eight-fold to thirty-eight thousand records a second, limited now by cores or downstream capacity rather than by the artificial single-thread bottleneck. This is often a bigger and cheaper win than scaling out, because you are using hardware you already rent.

### The ordering caveat, which is the entire catch

Parallelism inside a consumer breaks ordering, and that is the catch that makes this dangerous if you do it naively. Kafka gives you per-partition ordering: records within a partition are delivered in offset order. The instant you hand a partition's records to eight workers, those eight workers finish in nondeterministic order — worker 3 might finish record 105 before worker 1 finishes record 100 — and your side effects land out of order. If the records are independent (each is a self-contained event with no ordering relationship to its neighbors), this is fine, and you can fan out freely. If the records have ordering dependencies — two events for the same bank account, where the order of debit and credit matters — naive fan-out corrupts your state.

The fix is **key-based parallelism**: you parallelize *across* keys but preserve order *within* each key. You route all records with the same key to the same worker (or the same per-key queue), so each key is processed sequentially in order, but different keys run concurrently on different workers. With enough distinct keys — and there usually are far more keys than workers — you get near-full parallelism while preserving the per-key ordering you actually care about. The Confluent Parallel Consumer library and similar tools implement exactly this: they fan out a partition's records to a worker pool keyed by the record key, giving you concurrency without losing order. It is the right default for scaling up an ordered workload, and figure 9 later shows where it sits in the architecture.

```java
// Key-based parallelism: same key -> same single-thread executor,
// so order within a key is preserved while keys run concurrently.
ExecutorService[] workers = new ExecutorService[8];
for (int i = 0; i < 8; i++) workers[i] = Executors.newSingleThreadExecutor();

void dispatch(ConsumerRecord<String, String> r) {
    int w = Math.floorMod(r.key().hashCode(), workers.length);
    workers[w].submit(() -> process(r));   // ordered per key
}
```

### The offset-commit complication

Scaling up creates a real problem for offset commits that you must handle. The poll thread normally commits the offset after it finishes the batch, signaling "everything up to here is done." But once processing is offloaded to a worker pool, the poll thread finishes dispatching long before the workers finish processing. If the poll thread commits the offset immediately after dispatch, and then the process crashes while workers are still mid-flight, you have committed offsets for records that were never actually processed — and on restart those records are skipped. That is silent message loss, and it is exactly the failure mode covered in [consumer offset commit strategies](/blog/software-development/message-queue/consumer-offset-commit-strategies-failure-modes).

The correct pattern is to commit the offset only after the workers confirm completion, and to commit conservatively — you can only safely advance the committed offset to the lowest offset that has *not yet* completed across all in-flight work, because Kafka offsets are a single high-water mark per partition, not a set of individual acknowledgements. If record 100 is still processing but records 101 through 110 have finished, you cannot commit past 100, because committing 110 would skip 100 if you crash. So parallel processing forces you to track per-record completion and commit only the contiguous-completed prefix. Libraries like the Parallel Consumer handle this bookkeeping for you; if you roll your own, getting the commit logic right is the hard part, not the fan-out.

## 8. Decoupling fetch from process (pause/resume)

The cleanest structural answer to almost every problem above — the poll-interval trap, slow processing, parallel workers, backpressure — is to fully decouple fetching from processing. Stop doing the work on the poll thread. Let the poll thread do nothing but fetch records and hand them to an internal buffer or queue, and let a separate processing stage drain that buffer at its own pace. The poll thread returns to `poll()` immediately after fetching, so the poll interval is never threatened no matter how slow the processing is. The processing happens elsewhere, on its own threads, decoupled from the liveness machinery entirely.

![A vertical stack showing consumer throughput layers from fetch at the bottom through poll buffer, process, and commit, with the slowest layer setting the rate](/imgs/blogs/consumer-optimization-and-scaling-6.webp)

Figure 6 stacks the layers this decoupling exposes. Fetch pulls records over the network and is cheap. The poll buffer holds fetched-but-not-yet-processed records in memory, governed by `max.poll.records`. Process runs your business logic and is the expensive layer — twelve milliseconds a record in the running example. Commit writes the offset back, cheap again. The total throughput is set by the slowest layer, which is almost always process, and the value of decoupling is that you can scale the process layer (more workers) without touching the fetch layer, and you can let the fetch layer run ahead and buffer while the process layer catches up — as long as you do not let the buffer grow unbounded and run out of memory.

### Pause and resume: the backpressure handle

The problem with letting the poll thread run ahead unbounded is exactly that: unbounded. If you keep calling `poll()` and the records pile up in your internal buffer faster than the workers drain it, your heap fills and the process dies with an out-of-memory error. You need backpressure: when the internal buffer is full, stop fetching until the workers catch up. Kafka gives you precisely this with `consumer.pause()` and `consumer.resume()`. You pause the partitions whose buffer is full, keep calling `poll()` (which now returns nothing for the paused partitions but still sends heartbeats and resets the poll-interval clock), and resume the partitions once the buffer drains below a threshold.

```java
// Decouple fetch from process with bounded buffering + pause/resume.
BlockingQueue<ConsumerRecord<String, String>> buffer = new ArrayBlockingQueue<>(10_000);

while (running) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    for (ConsumerRecord<String, String> r : records) {
        if (!buffer.offer(r)) {                       // buffer full
            consumer.pause(consumer.assignment());    // stop fetching
            buffer.put(r);                            // block until space
        }
    }
    if (buffer.remainingCapacity() > 2_000) {
        consumer.resume(consumer.assignment());       // backpressure released
    }
    // poll() is called every loop regardless of worker speed,
    // so max.poll.interval.ms is never threatened.
}
```

This is the structural fix that makes the poll-interval trap simply impossible to hit, because the poll thread calls `poll()` every hundred milliseconds no matter how slow the processing is. It is also the foundation for clean parallel processing and clean backpressure: the buffer is the boundary where you apply backpressure, the worker pool is where you apply parallelism, and the pause/resume calls are the valve that keeps memory bounded. RabbitMQ achieves the equivalent with prefetch — the broker stops pushing when the prefetch window is full — so on the push side this decoupling is partly built into the protocol, whereas on Kafka's pull side you build it yourself with pause and resume.

### When decoupling is overkill

Decoupling is not free in complexity. You now own a buffer, a worker pool, the pause/resume logic, and the careful offset-commit bookkeeping from section seven. For a consumer doing fast, simple, independent work — process a record in two milliseconds, no ordering constraints, comfortably under the poll interval — decoupling is over-engineering, and the straightforward poll-process-commit loop is correct and far easier to operate. Reach for decoupling when your processing is slow enough to threaten the poll interval, when you need parallelism with ordering, or when you need explicit backpressure against a downstream that can stall. For everything else, keep it simple; the simplest correct consumer is the one you can debug at 3am.

## 9. Autoscaling on lag

All the tuning above is about making a *fixed* number of consumers go faster. But load is not fixed — it has daily peaks, weekly cycles, and unpredictable spikes — and a consumer group sized for the peak is wasteful at the trough, while one sized for the average drowns at the peak. The answer is to scale the consumer group automatically in response to the one metric that actually measures whether consumers are keeping up: **consumer lag**, the gap between the latest produced offset and the consumer's committed offset, measured in messages or in time.

Lag is the right autoscaling signal because it directly measures the supply-demand imbalance on the consumer side. CPU utilization is a poor signal — a consumer can be at 40% CPU and still falling catastrophically behind because it is blocked on a slow database, while another at 95% CPU is keeping up perfectly. Lag does not care why the consumer is slow; it measures the only thing that matters, which is whether the consumed rate is keeping up with the produced rate. Rising lag means consumers are falling behind and you should add capacity; falling lag means you have headroom and can scale down. The whole mechanism, the metrics, and the autoscaler wiring are the subject of the lag sibling post, [consumer lag monitoring and autoscaling](/blog/software-development/message-queue/consumer-lag-monitoring-and-autoscaling); here we cover just enough to connect it to the scaling math above.

### The partition ceiling constrains the autoscaler

The crucial interaction is that autoscaling on lag runs straight into the partition ceiling from section six. An autoscaler that scales the consumer group from 4 to 8 to 16 pods in response to lag will help — until it tries to scale past the partition count, at which point the extra pods get zero partitions and add zero throughput, and lag keeps climbing while the autoscaler keeps adding useless idle pods. A naive autoscaler in this situation can spin up dozens of idle consumers, each one making rebalances slower, while lag never improves, because the bottleneck was never the consumer count — it was the partition count, which the autoscaler cannot change. The autoscaler must be capped at the partition count, and beyond that ceiling the only remedies are scaling up (more workers per consumer) or repartitioning, neither of which a pod autoscaler does on its own.

In practice the clean architecture is KEDA (the Kubernetes Event-Driven Autoscaler) with its Kafka scaler, which reads lag per consumer group and scales the deployment between a floor and a ceiling — and you set that ceiling to the partition count. KEDA also handles the trough: when lag is zero and stays zero, it can scale the consumers down, even to zero for a fully idle topic, and scale back up the instant messages arrive. That gives you a consumer fleet that tracks the actual workload, paying for capacity only when there is lag to work off, capped intelligently at the partition ceiling so it never wastes money on idle pods. The full configuration and the time-based versus message-count lag debate live in the lag sibling.

```yaml
# KEDA ScaledObject: scale consumers on Kafka lag, capped at partition count.
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: order-processor
spec:
  scaleTargetRef:
    name: order-processor       # the consumer Deployment
  minReplicaCount: 1
  maxReplicaCount: 12           # MUST cap at the partition count
  triggers:
    - type: kafka
      metadata:
        bootstrapServers: broker:9092
        consumerGroup: order-processors
        topic: orders
        lagThreshold: "1000"    # add a pod per ~1000 messages of lag
```

### Message-count lag versus time lag

There is a subtlety in *what* you measure as lag, and it changes the autoscaling behavior. Message-count lag is the number of messages between the latest offset and the committed offset — easy to read, but its meaning depends on message size and processing cost. A lag of one hundred thousand tiny telemetry pings that process in a microsecond each is a non-event; a lag of one hundred thousand records that each trigger a multi-second ML inference is an emergency. The same number means wildly different things depending on the work behind each message. Time lag — how many seconds or minutes behind real time the consumer is, derived from the timestamp of the oldest unprocessed message — is often the more honest signal, because it directly answers the question a human actually cares about: how stale is my data right now. A consumer that is "thirty seconds behind" is intuitive in a way that "four hundred thousand messages behind" is not. Many teams alert and autoscale on time lag for exactly this reason, and reserve message-count lag for the raw KEDA trigger. The tradeoff between the two, and how to compute time lag without an expensive timestamp scan, is the heart of the lag sibling post.

### Scaling down without losing your place

Scaling up on rising lag is the easy direction; scaling down on falling lag is where teams get burned, because removing a consumer triggers a rebalance, and a rebalance during active processing can cause reprocessing if offsets were not committed cleanly before the consumer was terminated. A pod that is killed mid-batch by the autoscaler, before it commits, hands its partition to another consumer that reprocesses the in-flight records — harmless if you are idempotent, a duplicate-side-effect bug if you are not. The safe pattern is a graceful shutdown: when the pod receives its termination signal, it stops fetching, finishes the in-flight batch, commits the offset, and *then* exits, so the next owner starts cleanly from a committed position. Combined with cooperative rebalancing and static membership from section six, graceful shutdown makes scale-down a smooth, lossless operation rather than a source of duplicate processing. Without it, an aggressive autoscaler that scales down hard during a lull can generate a steady background hum of reprocessing every time it removes a pod, which is exactly the kind of subtle, intermittent bug that takes weeks to diagnose.

## Case studies and war stories

Patterns are easier to remember when they are attached to an incident. Here are four, each isolating one failure mode from the sections above.

### The idle-fleet money pit

A payments team saw lag climbing on their settlement topic and did the obvious thing: they scaled the consumer deployment from 6 pods to 30 in their Kubernetes config, reasoning that five times the consumers would mean five times the throughput. Lag did not improve at all. The topic had 6 partitions. Six of the thirty pods owned one partition each and did all the work; the other twenty-four owned zero partitions and sat idle, polling and receiving nothing, while the cloud bill for the deployment quintupled. Worse, the rebalances now took noticeably longer because the coordinator had to coordinate thirty members instead of six every time a pod restarted, and pod restarts during a deploy now caused multi-second stalls across the whole group. The fix was two-part: scale the deployment back down to 6, and address the actual bottleneck, which was that each consumer was doing a single-row database INSERT per record. They batched the writes — collecting each poll's records and writing them with one multi-row INSERT — and the existing 6 consumers, now efficient, cleared the lag in twenty minutes. The lesson is the partition ceiling: scaling out past the partition count is pure waste, and the real win was per-message efficiency, not more pods.

### The rebalance storm from a degraded dependency

An e-commerce order-enrichment service consumed orders, called an inventory API for each, and wrote the enriched order back. It ran fine for months at around 10 milliseconds per record, with `max.poll.records` at the default 500 and `max.poll.interval.ms` at the default 5 minutes — a batch took 5 seconds, far under the interval. Then the inventory service had a bad deploy and its latency went from 10 milliseconds to 1.2 seconds per call. Now each batch of 500 took 600 seconds — ten minutes — and every consumer in the group blew the 5-minute poll interval. The coordinator evicted them all, rebalanced, the reassigned consumers immediately blew the interval again on their new partitions, and the group entered a continuous rebalance storm: no forward progress, all CPU spent rebalancing, lag exploding. The team's first instinct — add more consumers — made it worse, because more members meant slower rebalances. The real fix was to drop `max.poll.records` to 25 (so a batch took 25 times 1.2 equals 30 seconds, safely under the interval even at the degraded latency) and to add a circuit breaker around the inventory call so a degraded dependency fast-failed instead of hanging. The deeper lesson: size `max.poll.records` against worst-case per-record latency, not average, because the interval check fires on the single slow batch, not the average.

### The silent loss from committing before processing

A data team built a high-throughput consumer that offloaded processing to a thread pool for speed — the poll thread fetched and dispatched to 16 workers, then committed the offset and looped. Throughput was excellent. But they had committed the offset immediately after *dispatch*, not after *completion*. During a routine deploy, pods were terminated while workers were mid-flight on records whose offsets had already been committed. On restart, those records were past the committed offset and so were never re-fetched. The result was a slow trickle of silently dropped records — a fraction of a percent — that nobody noticed until a downstream reconciliation job found the gap weeks later. The fix was to commit only the contiguous-completed offset prefix: track which offsets the workers had actually finished, and advance the committed offset only up to the lowest still-in-flight offset. Throughput dropped slightly because commits became more conservative, but the loss stopped. The lesson: when you decouple processing from polling, the offset commit must follow *completion*, never dispatch — exactly the trap the offset-commit post warns about.

### The prefetch-of-one latency cliff

A RabbitMQ-based image-processing service used `prefetch_count=1` for fairness — they wanted work distributed evenly across a fleet of GPU workers, and prefetch 1 guarantees the broker hands the next message to whoever is least busy. That reasoning was sound for the *fairness* goal. But each image took about 50 milliseconds to process and the round trip to the broker was about 2 milliseconds, and with prefetch 1 the worker sat idle for that 2-millisecond round trip after every single image, waiting for the broker to push the next one. At 50 milliseconds of work plus 2 milliseconds of idle per image, they were leaving about 4% of capacity on the table — small per image, but across a large fleet it was several wasted GPU instances. Raising prefetch to 5 let each worker keep a few images queued locally, eliminating the round-trip idle, recovering the 4%, and still keeping load reasonably fair because the window was small relative to the queue depth. The lesson: prefetch 1 maximizes fairness at the cost of throughput, and unless you genuinely need strict fairness, a small prefetch that covers one round-trip's worth of processing is the better default.

![A tree taxonomy of consumer scaling strategies branching into scale out, scale up, and decouple, each with its specific technique and ceiling](/imgs/blogs/consumer-optimization-and-scaling-8.webp)

Figure 8 organizes the strategies these stories illustrate into one taxonomy. Scaling out adds consumers up to the partition count and lifts the ceiling by adding partitions. Scaling up adds worker threads keyed per record key to preserve ordering. Decoupling separates fetch from process with a buffer so the poll interval is never threatened. Each branch has a distinct ceiling and a distinct failure mode, and the right answer for a given pipeline is usually a deliberate combination — scale out to the partition count, scale up within each consumer to the core count, and decouple if processing is slow enough to threaten liveness. The stories above are each what happens when you reach for one strategy and forget its ceiling.

## When to reach for this (and when not to)

The honest recommendation depends on where your bottleneck actually is, so diagnose before you tune. Measure first: is the consumer network-bound (doing a round trip per record), CPU-bound (one core pegged, others idle), or downstream-bound (blocked on a database or API)? Each points to a different fix, and applying the wrong one wastes effort.

**Reach for fetch tuning first, always.** Raising `fetch.min.bytes` with a sensible `fetch.max.wait.ms` is nearly pure upside for any throughput-oriented pipeline, costs nothing but a few milliseconds of latency on a quiet queue, and is the single cheapest improvement available. There is almost no high-throughput Kafka consumer that should be running with the default `fetch.min.bytes` of 1. Do this before anything else.

**Reach for scaling out when one well-tuned consumer is saturated and you are below the partition ceiling.** If your consumers are already efficient — batched fetch, batched downstream writes — and a single instance still cannot keep up, and you have spare partitions, add instances. But only up to the partition count. Past that, more instances are waste. Size the group to the partition count and stop.

**Reach for scaling up when you have hit the partition ceiling, or when each consumer is wasting cores.** If you are at one consumer per partition and still behind, or if each consumer is pinning one core while seven idle, parallelize within the consumer using key-based worker pools. This is often cheaper than scaling out because it uses hardware you already pay for, but it costs you ordering complexity and careful offset commits, so only take it on when you need it.

**Reach for decoupling with pause and resume when processing is slow enough to threaten the poll interval,** or when you need backpressure against a stalling downstream, or when you want clean parallelism with bounded memory. It is the most robust pattern but also the most code to own. For fast, simple, independent processing comfortably under the poll interval, *do not* decouple — the plain poll-process-commit loop is correct and far easier to operate.

**Do not reach for autoscaling as a substitute for fixing efficiency.** Autoscaling on lag is the right way to track variable load, but it cannot scale past the partition ceiling and it will not fix a consumer that is inefficient or blocked on a degraded dependency. An autoscaler in front of an inefficient consumer just spins up more inefficient consumers, or worse, more idle ones. Fix per-message efficiency first, set the autoscaler's ceiling to the partition count, and let it handle the variable component of load on top of a baseline that is already efficient.

A blunt decision rule: tune the fetch, batch the downstream writes, size the group to the partition count, parallelize within each consumer to the core count if still behind, decouple if processing threatens the poll interval, and autoscale on lag capped at the partition count to handle the peaks. In that order. Each step is cheaper or simpler than the one after it, and most pipelines never need to go past the first three.

## Key takeaways

- **The consumer is almost always the bottleneck**, because producing is a cheap fixed cost and consuming runs your unbounded business logic — so optimization and scaling effort belongs on the read side of the pipeline.
- **Batch the fetch before anything else.** `fetch.min.bytes` plus `fetch.max.wait.ms` turns one network round trip per record into one per batch of hundreds, often a 100-times throughput jump, at the cost of a little latency only when the queue is nearly empty.
- **`max.poll.records` controls the work you take per loop iteration, and `max.poll.records` times worst-case per-record time must stay well under `max.poll.interval.ms`** — size against the worst case, not the average, or a degraded dependency triggers a rebalance storm exactly when you can least afford it.
- **The max.poll.interval trap evicts a busy consumer that takes too long on a batch**, reassigning and reprocessing its records; fix it by lowering `max.poll.records`, raising the interval, or — best — moving processing off the poll thread.
- **RabbitMQ prefetch is the push-side equivalent of all the Kafka fetch tuning**: it caps in-flight unacked messages, and you size it to about one round-trip's worth of processing — small for slow work to keep load fair, larger for fast work to hide round-trip latency.
- **Scaling out is capped at the partition count.** Consumers beyond the partition count get zero partitions and idle; size the consumer group to the partition count, and size the partition count to your target maximum parallelism up front.
- **Scaling up parallelizes within a consumer to use all your cores, but breaks ordering** unless you use key-based parallelism — same key to the same worker — and commit only the contiguous-completed offset prefix to avoid silent loss.
- **Decoupling fetch from process with pause and resume** makes the poll-interval trap impossible to hit and is the clean foundation for parallelism and backpressure, but it is real complexity you only take on when slow processing demands it.
- **Autoscale on consumer lag, not CPU**, because lag directly measures whether consumers keep up — but cap the autoscaler at the partition count, or it spins up useless idle pods while lag keeps climbing.

![A grid architecture mapping twelve partitions to a four-member consumer group to per-consumer worker thread pools with per-key queues preserving order](/imgs/blogs/consumer-optimization-and-scaling-9.webp)

Figure 9 ties the whole architecture together. Twelve partitions carry ordered keys. A four-member consumer group takes three partitions each. Inside each consumer, an eight-thread worker pool fans the work out, with per-key queues preserving order within each key, reaching roughly thirty-eight thousand records a second per consumer. That single picture contains every lever in this post: partitions feeding a group up to the partition ceiling (scale out), worker threads using every core (scale up), per-key queues keeping order (the ordering caveat), and the whole thing tunable with fetch and prefetch settings and scalable on lag. Build your consumer like this and it will keep up with almost anything a producer can throw at it.

## Further reading

- [Push vs Pull, acknowledgements, and how consumers read](/blog/software-development/message-queue/push-vs-pull-acknowledgements-how-consumers-read) — the protocol layer beneath this tuning: how consumers read at all and why push needs prefetch.
- [Kafka consumer groups, offsets, and rebalancing](/blog/software-development/message-queue/kafka-consumer-groups-offsets-rebalancing) — the group machinery the poll-interval trap and partition ceiling sit on top of.
- [Throughput vs latency tuning](/blog/software-development/message-queue/throughput-vs-latency-tuning-tradeoff) — the pipeline-wide version of the consumer-side throughput-versus-latency negotiation.
- [Partitioning and capacity planning](/blog/software-development/message-queue/partitioning-capacity-planning) — how to choose the partition count that sets your scale-out ceiling.
- [Consumer lag monitoring and autoscaling](/blog/software-development/message-queue/consumer-lag-monitoring-and-autoscaling) — the full lag-based autoscaling story this post forward-links.
- [Consumer offset commit strategies and failure modes](/blog/software-development/message-queue/consumer-offset-commit-strategies-failure-modes) — the commit subtleties that parallel and decoupled processing force you to get right.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — the storage model that makes pull-based consumption and replay cheap.
- [Apache Kafka consumer configuration reference](https://kafka.apache.org/documentation/#consumerconfigs) — the authoritative list of every knob discussed here.
- [RabbitMQ consumer prefetch documentation](https://www.rabbitmq.com/consumer-prefetch.html) — the official guidance on sizing QoS prefetch.
