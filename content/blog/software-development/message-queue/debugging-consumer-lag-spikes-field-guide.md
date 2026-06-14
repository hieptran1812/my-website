---
title: "Debugging Consumer Lag Spikes: A Field Guide"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "The lag alert just fired and the backlog is climbing. This is the on-call field guide: a differential diagnosis that starts with one discriminating question, walks the five investigation signals in order, names the specific fix for each cause, and hands you a runbook you can paste into the incident channel."
tags:
  [
    "message-queue",
    "consumer-lag",
    "incident-response",
    "debugging",
    "kafka",
    "rabbitmq",
    "distributed-systems",
    "event-driven",
    "observability",
    "runbook",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/debugging-consumer-lag-spikes-field-guide-1.webp"
---

It is 02:14 and your phone is buzzing. The alert says consumer lag on `orders-events` is climbing and has crossed two million records. You are half awake, the dashboard is a wall of red and green sparklines, and there is a strong, primal urge to do the one thing that feels like action: restart the consumers. Resist it. Restarting consumers before you understand the spike is the on-call equivalent of slapping the side of a machine you do not understand — sometimes the noise stops, and you learn nothing, and it comes back at 04:00 worse than before because you triggered a rebalance on top of whatever was already wrong.

A lag spike is not one problem. It is a *symptom* with at least four distinct underlying diseases, and the entire job of the first five minutes is differential diagnosis: ruling causes in and out with cheap observations until exactly one suspect remains, then applying the fix that matches *that* cause and not some other cause's fix. A producer traffic surge and a stuck consumer both show up as "lag is high," but the fix for one (scale consumers or wait it out) actively harms the other (where scaling does nothing and the real move is to evict a poison message). Treating every lag spike the same way is why so many lag incidents take an hour instead of ten minutes.

![A decision tree that starts at a lag spike alert, splits on whether all partitions or one partition are growing, and branches into producer faster, consumer slower, hot key skew, and stuck consumer causes](/imgs/blogs/debugging-consumer-lag-spikes-field-guide-1.webp)

This post is the field guide I wish every on-call engineer had taped to the wall. It is structured as a differential diagnosis (figure 1). We start with the single most discriminating question — is lag growing on *all* partitions or on *one*? — because the answer immediately halves the search space. Then we work the cause tree: producer got faster, consumer got slower, consumer stalled completely, or one partition is skewed. For each branch we cover the telltale signals, two fully worked numeric examples, and the specific fix. We finish with the investigation workflow in the order you should actually run it, a section on confirming recovery and draining the backlog without overshooting, and a copy-pasteable runbook. This is the incident-response companion to [Consumer lag monitoring and autoscaling](/blog/software-development/message-queue/consumer-lag-monitoring-and-autoscaling) — that post tells you how to build the alerts and autoscalers; this one tells you what to do the moment one of them pages you. By the end you will be able to take a cold lag alert and reach the right root cause in minutes, not by intuition but by elimination.

## 1. The first question: all partitions or one?

Before you look at anything else, answer one question: is lag growing on every partition of the topic, or on a small subset — often exactly one? This is the most discriminating single observation available to you, and it is the hinge of the entire diagnosis. Almost every command you run after this depends on which branch you are in, so spend your first thirty seconds here and nowhere else.

The reason this question is so powerful is structural. Consumer lag is, per partition, the log-end-offset minus the committed-offset — the count of records written but not yet processed *on that partition*. A consumer group divides the topic's partitions among its members; each partition is owned by exactly one consumer at a time. So the *pattern* of lag across partitions is a fingerprint of where the imbalance lives. If lag is climbing uniformly across all partitions, the cause has to be something that affects all of them at once: either every consumer slowed down together (a global capacity problem — a shared downstream got slow, a deploy regressed every instance, GC is stalling the whole fleet) or the producer started writing faster than the group can drain across the board (a traffic spike, a backfill, a retry storm). If lag is climbing on one partition while the others stay flat or drain normally, the cause is local to that partition: either the data on it is skewed (a hot key is hashing everything onto it) or the single consumer that owns it is stuck (a deadlock, a poison message it keeps retrying, a slow path only that partition's data hits).

![A before-and-after comparison contrasting all twelve partitions climbing at eight thousand per second against a single partition at two million lag while the others sit near zero](/imgs/blogs/debugging-consumer-lag-spikes-field-guide-4.webp)

Figure 4 makes the split concrete. On the left, all twelve partitions climb together at roughly eight thousand records per second each — this is a *group-level* fault, and the fixes are group-level: add consumers, roll back a slow deploy, or wait out a transient producer burst. On the right, partition seven sits at two million records of lag while the other eleven hover near zero — this is a *partition-local* fault, and adding consumers will not help at all, because you cannot put two consumers on one partition. The fix has to target that partition: find the hot key, or unstick that one consumer. Reading the wrong branch sends you to the wrong toolbox, which is exactly why the all-or-one question comes first.

### How to actually read the per-partition pattern

You read the pattern with a per-partition lag view, not the topic-total lag your alert probably fired on. The topic total is a sum, and a sum hides the shape. Two million records of total lag could be two hundred thousand on each of ten partitions (a group problem) or two million on one partition and nothing on the rest (a partition problem), and those are completely different incidents. Your first command is always: break the total down by partition.

```bash
# Kafka: per-partition lag for a consumer group.
# The LAG column per row is what you read — NOT the sum.
kafka-consumer-groups.sh \
  --bootstrap-server broker:9092 \
  --describe --group orders-processor

# GROUP            TOPIC          PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG
# orders-processor orders-events  0          4821002         4823110         2108
# orders-processor orders-events  1          4819774         4821990         2216
# ...
# orders-processor orders-events  7          2630551         4630551         2000000   <-- the one
# ...
```

If every row's `LAG` is similar and rising, you are in the all-partitions branch. If one row dwarfs the others, you are in the one-partition branch. That single `--describe` call resolves the most important fork in the whole diagnosis. Keep its output; you will refer back to it. For RabbitMQ there is no partition concept, but the analogous split is per-queue: is depth rising across many queues (a broker or producer-wide problem) or on one queue (one consumer or one binding gone wrong)? The mental model transfers cleanly — the unit of locality is the partition in Kafka and the queue in RabbitMQ.

One more refinement before we descend the tree. There is a third pattern that is neither "all rising" nor "one rising," and it is the most alarming: the committed offset is *completely flat* — not advancing at all — on one or more partitions, while the log-end-offset keeps climbing. Flat offsets mean the consumer is not making progress, which is categorically different from making slow progress. A slow consumer's committed offset still creeps up; a stalled consumer's does not move. We will treat the flat-offset case as its own branch in section 4, because the causes (rebalance loop, deadlock, poison message) and the fixes are distinct from the merely-slow case. So the real first question has three answers: all rising, one rising, or flat. Figure 2 is the lookup table that maps each of these symptoms to its prime suspect.

![A symptom-to-cause matrix mapping all partitions climbing, one partition climbing, a totally flat offset, and one partition far ahead against producer surge, consumer slowdown, stalled or poison, and key skew](/imgs/blogs/debugging-consumer-lag-spikes-field-guide-2.webp)

The matrix in figure 2 is your at-a-glance prior. Read the row that matches your symptom and the highlighted column is where to look first. All partitions climbing points at a producer surge or a fleet-wide consumer slowdown. One partition climbing points hardest at key skew. A totally flat offset points at a stall or a poison message. One partition far ahead of the rest points at skew or a single stuck consumer. None of these is a certainty — they are priors that tell you where to spend your next two minutes confirming. The rest of this guide is about doing that confirmation efficiently.

## 2. Producer got faster: spikes, backfills, upstream retries

Start with the branch that is least your fault and easiest to confirm: the producer started writing faster. Lag is the running integral of the capacity deficit — its derivative is exactly the produce rate *P* minus the aggregate consume rate *C*. If lag is climbing, either *P* went up or *C* went down, and *P* going up is the cleanest thing to rule in or out because the produce rate is a single, well-defined, easily-graphed number. Pull up produce rate (records per second written to the topic) over the last hour and look at it. If it stepped up right when lag started climbing, and your consume rate is flat at its normal level, you have your answer: the consumers did not get slower, the firehose got bigger.

There are three flavors of "producer got faster," and they call for different responses, so name which one you have.

The first is an **organic traffic spike**: a marketing email went out, a flash sale started, it is the first of the month and every subscription renews. Produce rate steps up and stays up for a bounded, predictable window. The consume rate is unchanged — your consumers are fine, there are just more messages. This is the most benign lag spike there is, because it is self-limiting: when the spike ends, *P* drops back below *C*, and lag drains on its own. The right response is often to *do nothing but watch*, or to scale consumers temporarily if the spike is large enough that the backlog would breach your latency budget before it drains. The cardinal sin here is panic-restarting consumers, which triggers a rebalance and makes lag worse during the exact window you most need throughput.

The second is a **backfill or replay**: someone (possibly you, possibly a data team) kicked off a job that republishes a large historical dataset, or a downstream service requested a reprocessing of last week's events. Produce rate steps up *enormously* — often ten or a hundred times normal — and the step is usually rectangular: it turns on hard, runs flat, and turns off hard. This is not an organic pattern; it is a batch job. The fix is rarely on the consumer side at all. The fix is to find who started the backfill and ask whether it can be throttled, paused, or run on a separate topic so it does not contend with live traffic. A backfill running on your production topic is a self-inflicted lag spike, and the cleanest resolution is operational, not technical: rate-limit the backfill producer.

The third, and the nastiest, is an **upstream retry storm**: a service upstream of your producer is failing and retrying, and each retry republishes a message. Now produce rate climbs not because there is more real work but because the same work is being duplicated by a feedback loop. This is the most dangerous of the three because it can be *unbounded* — a retry storm with no backoff and no circuit breaker will produce faster and faster until something melts. The tell is that produce rate is not just elevated but *accelerating*, and that the duplicated messages often share keys or correlation IDs. The fix is upstream: get the failing dependency healthy, or get the retry policy to back off, or shed the duplicate load with deduplication. This is exactly the dynamic covered in [Poison messages and retry storms](/blog/software-development/message-queue/poison-messages-and-retry-storms-containment) — a retry storm is a poison-message problem that has escaped one consumer and become a producer-side flood.

#### Worked example: telling a spike from a regression

You get paged: lag on `orders-events` is climbing at roughly eight thousand records per second, summed across twelve partitions, and it crossed two million. Is this a producer spike or a consumer regression? Pull the two rates. Normal steady state for this topic is a produce rate of thirty-six thousand records per second and a consume rate of thirty-six thousand — balanced, lag flat. You graph the last hour and see this: produce rate is *still* thirty-six thousand, dead flat, no step. Consume rate, however, dropped from thirty-six thousand to twelve thousand at 02:10, four minutes before the page. The deficit is thirty-six thousand minus twelve thousand, which is twenty-four thousand records per second of group-wide shortfall — but wait, the alert said eight thousand per second per partition, and twenty-four thousand across twelve partitions is two thousand per partition, not eight thousand. The numbers do not match, which means the per-partition climb is uneven: three or four partitions are climbing hard and the rest are flat. That single arithmetic inconsistency just told you something — this is not a clean all-partitions slowdown; some partitions are worse than others, which nudges you toward skew layered on top of a slowdown. Always reconcile the per-partition numbers against the topic total; when they disagree, the disagreement is data.

Now suppose instead the graphs had shown produce rate stepping from thirty-six thousand to sixty thousand at 02:10 while consume rate held at thirty-six thousand. The deficit is sixty thousand minus thirty-six thousand, which is twenty-four thousand per second across twelve partitions, two thousand per second per partition, uniform. That uniformity plus the produce-rate step is the signature of a clean organic spike. If you know the spike is a forty-minute flash sale, you can even compute whether to act: at two thousand per second per partition for forty minutes, each partition accumulates four point eight million records of lag, and at the normal drain rate of three thousand per second per partition once the spike ends (consume thirty-six thousand minus produce zero, over twelve partitions), each partition drains in twenty-seven minutes. If that twenty-seven-minute drain plus the forty-minute spike keeps you inside your end-to-end latency SLO, do nothing. If it does not, scale consumers up to the partition count for the duration. The math, not the adrenaline, makes the call.

The lesson of this branch: the produce-rate graph is the cheapest, highest-information observation in the entire diagnosis. Look at it second, right after the per-partition split. If produce rate stepped up and consume rate is flat, you are done diagnosing — the only remaining question is whether the spike is organic (wait or temporarily scale), a backfill (throttle the source), or a retry storm (fix upstream). If produce rate is flat and lag is still climbing, the producer is innocent and you move to the consumer.

## 3. Consumer got slower: downstream, GC, deploy regressions

If produce rate is flat and lag is climbing on all partitions, the consume rate *C* dropped, and the consumers got slower. This is the most common cause of a real lag incident, and it has a small number of usual suspects. The unifying signature is: aggregate consume rate is below produce rate, the drop is roughly uniform across partitions (because it affects every consumer instance), and the consumers are still *making progress* — committed offsets are advancing, just too slowly. That last point distinguishes a slow consumer (this section) from a stalled one (section 4). A slow consumer's offset creeps up; a stalled consumer's offset is flat.

![A causal directed graph showing a deploy that adds a synchronous database lookup and a heavier deserialization path, both slowing per-message processing from three to nine milliseconds, dropping group throughput and growing lag](/imgs/blogs/debugging-consumer-lag-spikes-field-guide-6.webp)

Figure 6 is the causal chain for the single most frequent cause of a sudden consumer slowdown: a deploy. The way this works is mechanical. A consumer's throughput is governed by its per-message processing time — if it processes a message in three milliseconds, one thread does roughly three hundred and thirty messages per second, and a deploy that pushes that to nine milliseconds cuts the same thread to one hundred and ten per second, a 3x throughput collapse across the whole fleet at once. The new code path in figure 6 added a synchronous database lookup per message and a heavier deserialization path, and either alone might be tolerable, but together they tripled per-message latency. Because the deploy rolled out to every instance, every partition's consumer slowed by the same factor, so lag climbs uniformly — the all-partitions signature. The fix is almost always to roll back, and the catch-up math (worked below) tells you how long recovery takes.

The three usual suspects for a consumer slowdown, in rough order of frequency:

A **slow downstream dependency** is the most common and the most insidious, because your consumer code did not change at all — the database, the cache, the third-party API, or the internal service your consumer calls per message got slow, and your consumer's per-message time inflated to match. The tell is that your consumer's CPU is *not* pegged (it is blocked waiting on I/O, not burning cycles) while per-message latency climbs. Pull the downstream's p99 latency over the same window; if it stepped up when your consume rate dropped, you have found it. The fix is to fix the downstream, or to make your consumer resilient to it (batch the downstream calls, add a cache, increase concurrency so more messages are in flight while each waits). This is covered in depth in [Consumer optimization and scaling](/blog/software-development/message-queue/consumer-optimization-and-scaling) — the per-message time is the lever, and a slow downstream is the most common thing that moves it the wrong way.

A **GC pause or memory pressure** slows the JVM-based consumer (Kafka clients are heavily JVM) by stealing CPU and stopping the application threads. The tell is a sawtooth in consume rate that correlates with GC pause-time metrics, and often a recent change that increased allocation — a bigger batch size, a bigger payload, a memory leak. The fix is to tune the heap and GC, reduce per-message allocation, or reduce `max.poll.records` so each poll processes a smaller batch and allocates less. A GC death spiral can even tip a slow consumer into a stalled one if pauses grow long enough to blow the session timeout, which links this section to the next.

A **deploy regression** is the cleanest to confirm because it has a timestamp. If consume rate dropped at the exact minute of a deploy, the deploy is the cause until proven otherwise. A bigger payload (someone added a field that doubled message size, so deserialization and network both got slower), a synchronous call added to the hot path, a logging line that flushes to disk per message, a new validation that walks a large structure — any of these can regress per-message latency by a multiple. The fix is to roll back, and rollback is usually faster and safer than rolling forward a fix at 02:00. Roll back first, diagnose the regression in daylight.

#### Worked example: a 3x per-message latency regression and the catch-up time

This is the canonical incident. At 02:10 you shipped `orders-processor` v2.4.1. At 02:14 the page fires: lag climbing eight thousand records per second on all twelve partitions, now at two million total. Produce rate graph: flat at thirty-six thousand per second — producer innocent. Consume rate graph: dropped from thirty-six thousand to twelve thousand at 02:10, exactly the deploy time. Per-message latency on the consumer: was three milliseconds, now nine milliseconds — a 3x regression. The cause is the deploy. The new version added a synchronous lookup to a user-profile service on every message, and that service's call takes six milliseconds, so three plus six is nine, and throughput fell by the same factor: thirty-six thousand divided by three is twelve thousand. The arithmetic is consistent end to end, which is how you know you have the real cause and not a coincidence.

Now compute the catch-up time after rollback. At 02:18 you roll back to v2.4.0; per-message latency returns to three milliseconds and consume rate returns to thirty-six thousand. But lag did not stop accumulating until the rollback took effect. From 02:10 to 02:18 is eight minutes at a deficit of twenty-four thousand per second (produce thirty-six thousand minus consume twelve thousand), which is eight times sixty times twenty-four thousand, or eleven point five million records of accumulated lag. After rollback, consume rate is thirty-six thousand but produce rate is still thirty-six thousand, so at the normal rate the deficit is *zero* and lag would never drain — you would hold steady at eleven point five million forever. To drain, you need *spare* capacity: either the produce rate must dip below thirty-six thousand, or you must add consumers above the steady-state need. Suppose you scale from the steady consume capacity to fifty percent more, forty-eight thousand per second, by adding consumers (you have twelve partitions and were running six consumers at two partitions each, so you scale to twelve consumers at one partition each, doubling per-partition throughput — but you cannot exceed thirty-six thousand of useful drain unless produce also allows it). The drain rate is the *surplus* consume capacity over produce: forty-eight thousand minus thirty-six thousand is twelve thousand per second. Draining eleven point five million records at twelve thousand per second takes nine hundred and sixty seconds, sixteen minutes. So the full incident timeline is: eight minutes of climb, an instant rollback, and sixteen minutes of drain at fifty-percent-over capacity — about twenty-four minutes from deploy to clear, most of it drain. If you had restarted consumers instead of rolling back, you would have added a rebalance to the eight-minute climb and changed nothing about the regression. Diagnose first; the math rewards it.

![A horizontal incident timeline running from a deploy that ships a three-times latency regression through the lag climb, the page firing, the diagnosis, the rollback, and the fully drained backlog](/imgs/blogs/debugging-consumer-lag-spikes-field-guide-3.webp)

Figure 3 is that incident drawn to scale. Notice that the climb (T+0 to T+9) is short and the drain (T+18 to T+52) is long — this is the universal shape of a lag incident. Lag accumulates fast because the deficit during the fault is large, and it drains slowly because your spare capacity above the steady produce rate is small. The asymmetry is the single most important intuition for managing the back half of an incident: fixing the cause stops lag from *growing*, but it does not make lag *shrink* unless you have headroom, and the drain always takes longer than you expect. We return to draining in section 8.

## 4. Consumer stalled: rebalance loops, deadlocks, poison messages

There is a categorical difference between a consumer that is slow and a consumer that is stalled. A slow consumer's committed offset advances — just too slowly to keep up. A stalled consumer's committed offset is *flat*: it is making zero progress on one or more partitions. When you run the `--describe` command and a partition's `CURRENT-OFFSET` is identical to what it was thirty seconds ago while `LOG-END-OFFSET` climbs, you are not slow, you are stuck, and the fixes from the previous section (roll back, scale, fix the downstream) will do nothing because the consumer is not processing at all. A stall has three usual causes, and you tell them apart by what the consumer's logs and rebalance metrics say.

The first is a **rebalance loop**. A consumer group rebalances whenever membership changes — a consumer joins, leaves, or is declared dead by the broker for missing its heartbeat. Each rebalance pauses *all* consumption in the group while partitions are reassigned (under the classic eager protocol; cooperative rebalancing reduces but does not eliminate this). If something is causing consumers to repeatedly join and leave — a session timeout set too low for the actual processing time, a `max.poll.interval.ms` that the consumer keeps blowing past because a batch takes too long, a deploy that is rolling instances in a flapping loop, or an OOM-kill cycle — the group spends so much time rebalancing that it barely consumes. The tell is the rebalance-rate metric: a healthy group rebalances approximately never; a group rebalancing several times a minute is in a loop. Committed offsets are flat or barely moving because every time a consumer starts to make progress, another rebalance yanks the partition away. This failure mode is severe and common enough that it gets its own deep dive in [Kafka rebalance storms and how to tame them](/blog/software-development/message-queue/kafka-rebalance-storms-and-how-to-tame-them) — for the purposes of this guide, the on-call signature is: lag climbing, offsets flat, rebalance rate high. The immediate mitigation is to stop the churn: if a rolling deploy is flapping, freeze it; if the session timeout is too tight for the workload, raise `max.poll.interval.ms` so a slow batch does not get the consumer evicted.

The second is a **deadlock or hang**. The consumer thread is alive but not making progress — it is blocked on a lock it will never get, an external call with no timeout that will never return, or an infinite loop. The tell is that offsets are flat, the rebalance rate is *not* elevated (the consumer is still heartbeating, so it has not been kicked out — its background heartbeat thread is fine even though its processing thread is wedged), and a thread dump shows the processing thread parked or blocked. This is the case where you do, eventually, restart — but only after you have a thread dump, because the thread dump is the only evidence of *why* it hung, and restarting destroys it. Capture first, restart second.

The third, and the one that masquerades as the other two, is a **poison message**. A single message that the consumer cannot process — it throws on deserialization, it violates an invariant, it triggers a bug — and the consumer's error handling retries it forever without advancing past it. Because the consumer never commits past the poison offset, that partition's offset is flat, and lag on *that one partition* climbs without bound while the others are fine. This is the intersection of the stalled branch and the one-partition branch: a poison message produces a flat offset on a single partition. The tell is the dead-letter-queue rate (if you have a DLQ, are messages flowing to it, or is the consumer stuck before it can even DLQ?) and the consumer error logs showing the same offset retrying over and over. The fix is to get past the poison message: either fix the consumer so it can process the message, or — the standard production move — skip it by committing past it (seek the consumer forward one offset on that partition) and route it to a DLQ for offline analysis. The full machinery of detecting and containing poison messages is in [Poison messages and retry storms](/blog/software-development/message-queue/poison-messages-and-retry-storms-containment); on call, the move is: confirm the offset is stuck on a poison record, skip past it, and let the partition drain.

```python
# Skip a confirmed poison message on one partition by committing past it.
# Use only when you have confirmed the offset is stuck and have the
# bad record captured (logged or sent to a DLQ) for later analysis.
from kafka import KafkaConsumer, TopicPartition

consumer = KafkaConsumer(
    bootstrap_servers="broker:9092",
    group_id="orders-processor",
    enable_auto_commit=False,
)
tp = TopicPartition("orders-events", 7)
consumer.assign([tp])

stuck_offset = 2630551          # the offset that keeps failing
consumer.seek(tp, stuck_offset + 1)   # move PAST the poison record
consumer.commit({tp: stuck_offset + 1})  # persist the skip
# Now normal processing resumes from stuck_offset + 1 on partition 7.
```

Be careful with the skip: you are choosing to *drop* a message (or defer it to a DLQ), which is a data decision, not just an availability decision. Capture the record first. The reason a poison message can hold up two million records of lag is that everything behind it on the partition is blocked — partitions deliver in order, so one undeliverable message at the head dams the whole partition. That is why the one-partition flat-offset signature is so often a poison message, and why the fix is so disproportionately powerful: skipping one record unblocks two million.

## 5. Skew: the one hot partition

Now the pure one-partition case where the offset is *not* flat — it is advancing, the consumer is healthy, but lag on that partition still climbs while the others are fine. This is skew: more work is landing on one partition than the others, and the single consumer that owns it cannot keep up even though it is doing everything right. Skew is a partitioning problem masquerading as a consumer problem, and no amount of scaling the consumer group will fix it, because a partition is owned by exactly one consumer — adding consumers just gives the *other* partitions more owners and leaves the hot one exactly as overloaded as before. This is the single most counterintuitive lag incident, because the instinct (scale out) is precisely the move that does nothing.

Skew comes from the partition key. Kafka assigns a record to a partition by hashing its key (by default, murmur2 of the key bytes, modulo partition count). If keys are uniformly distributed, partitions get roughly equal load. But real keys are rarely uniform. If you partition orders by `customer_id` and one customer is a thousand times bigger than the rest — a marketplace where one seller does half the volume, a multi-tenant SaaS where one tenant is enormous — then that customer's partition gets a thousand times the traffic and falls behind while the others idle. The same happens with any naturally skewed key: a country code where one country dominates, a `null` key that all hashes to one partition, a timestamp bucket during a burst. The tell is unmistakable once you look for it: one partition's lag is high and climbing, its consumer's CPU or per-message latency is normal (it is not broken, just overwhelmed by volume), and the produce rate *into that one partition* is far above the others. You confirm by graphing per-partition produce rate, not just per-partition lag — the lag tells you which partition is behind, the produce rate tells you *why*.

#### Worked example: one partition at 2M lag while others are near zero

The page fires: total lag on `orders-events` is two million and climbing slowly. You run `--describe` and the per-partition breakdown is stark: partitions zero through six and eight through eleven all show lag under three thousand and stable, while partition seven shows two million and climbing at four thousand records per second. This is the one-partition signature, so you are in the skew-or-stall branch. Next discriminator: is partition seven's offset flat (stall) or advancing (skew)? You watch the `CURRENT-OFFSET` for partition seven over thirty seconds: it climbs from 2,630,551 to 2,750,551 — it advanced by one hundred and twenty thousand in thirty seconds, four thousand per second. The offset is moving, so this is *not* a stall and not a poison message. The consumer on partition seven is processing as fast as it can; it is just receiving more than it can drain.

Confirm with per-partition produce rate. Partitions zero through eleven each receive about three thousand records per second normally. Partition seven is receiving seven thousand per second — more than double — because a single high-volume customer's `customer_id` hashes to partition seven, and that customer is in the middle of a bulk import. The consumer on partition seven drains at four thousand per second (its max for this workload), so the deficit on partition seven alone is seven thousand minus four thousand, three thousand per second, which matches the observed climb. The arithmetic confirms skew: produce-into-seven exceeds the single consumer's drain rate, and no other partition is affected.

Now the fix, because this is where most people go wrong. Scaling the consumer group does *nothing* — you already have twelve consumers for twelve partitions, and partition seven still has exactly one owner. The real fixes, in order of how fast you can apply them on call:

The immediate mitigation is to **split the hot key's load across more partitions** by changing the partitioning for that key. If your producer can use a composite key — `customer_id` plus a small random salt or a sub-key like `order_region` — the hot customer's messages spread across several partitions instead of one, and several consumers share the load. This requires a producer change and only helps new messages, so it stops the bleeding going forward but does not drain the existing two million on partition seven.

To drain the existing backlog on partition seven faster, you can **temporarily increase that one consumer's parallelism** if your consumer framework supports intra-partition concurrency — process messages from partition seven on a thread pool rather than one thread, accepting the ordering relaxation that implies. If strict ordering matters (and for orders it often does within a customer), you cannot parallelize blindly; you parallelize across *different* keys within the partition while preserving order *per key*, which is a more careful change.

The structural fix, applied later in daylight, is to **rethink the partition key** so that no single value can dominate, or to **increase the partition count** so the hash spreads the hot key onto a partition it shares with less. Repartitioning is a heavy operation covered in the broader scaling material; on call, the composite-key mitigation plus the per-partition concurrency drain is what gets you through the night.

The deep lesson of skew is that lag is sometimes a *data distribution* problem wearing a *capacity* costume. When one partition is hot, the system has plenty of aggregate capacity — eleven consumers are nearly idle — but it is the wrong capacity in the wrong place. This is why the all-or-one question matters so much: it is the difference between "we need more consumers" and "we need to spread the keys," and those have almost nothing to do with each other.

## 6. The investigation workflow, in order

We have walked the four branches of the cause tree. Now here is the order to actually run the diagnosis, because doing the checks in the right sequence means you rule out whole branches with each step and never waste time. The workflow is five checks, and you run them in this order every single time, because each one's result tells you whether to continue down the list or stop and fix.

![A five-stage investigation pipeline running from lag-per-partition through throughput versus produce rate, rebalance activity, downstream latency, and the dead-letter queue or poison rate](/imgs/blogs/debugging-consumer-lag-spikes-field-guide-5.webp)

Figure 5 is the workflow as a pipeline. You enter at the left and proceed right only as far as you need to.

**Check 1 — lag per partition.** This is the all-or-one question from section 1, and it is always first because it halves the search space. Run `--describe`, read the per-partition `LAG`, and classify: all rising (continue to check 2 thinking producer-or-fleet), one rising (jump toward skew or stall), or flat offset (jump to check 3 for rebalance, then poison). Thirty seconds, and you know which half of the tree you are in.

**Check 2 — consumer throughput vs produce rate.** Graph consume rate *C* and produce rate *P* over the last hour. If *P* stepped up and *C* is flat, the producer got faster (section 2) — go decide whether it is a spike, backfill, or retry storm, and you are nearly done. If *P* is flat and *C* dropped, the consumer got slower (section 3) — continue to check 3 to rule out a stall versus a slowdown. This check is where you confirm the *direction* of the imbalance, and it is cheap because both rates are standard broker metrics.

**Check 3 — rebalance activity.** Pull the rebalance rate and the consumer group's member count over time. A healthy group is calm: rebalances approximately never, member count stable. If the rebalance rate is elevated — multiple rebalances per minute — you are in a rebalance loop (section 4), and the fix is to stop the churn before anything else, because nothing else can make progress while the group is thrashing. If rebalances are calm, the consumer is stably assigned, so a flat offset is a deadlock or a poison message, not a rebalance — continue.

**Check 4 — downstream latency.** If the consumer is stably assigned and slow (not stalled), the most common cause is a downstream dependency. Pull the p99 latency of every service the consumer calls per message — the database, the cache, the internal API. If one stepped up when your consume rate dropped, that is your slowdown (section 3). If all downstreams are healthy and the consumer is still slow, suspect a deploy regression or GC and check the deploy timeline and GC metrics.

**Check 5 — dead-letter queue and poison rate.** If offsets are flat on a partition, rebalances are calm, and the consumer is not deadlocked on an external call, the remaining suspect is a poison message. Check the DLQ rate (is anything flowing to it?) and the consumer error logs for the same offset retrying repeatedly. If you find a stuck offset, you have a poison message (section 4 and 5), and the fix is to skip past it.

![A vertical stack of diagnostic signals to read in order, from lag shape at the top down through throughput, rebalance rate, offset advance, downstream latency, and the dead-letter queue rate at the bottom](/imgs/blogs/debugging-consumer-lag-spikes-field-guide-7.webp)

Figure 7 stacks the same signals as a reading order. The discipline this enforces is important: read top to bottom, and the *first* signal that looks wrong is usually the cause, because the signals are ordered roughly from "broadest, cheapest, most discriminating" to "narrowest, most specific." You almost never need to read all the way to the bottom — if the lag shape and throughput already point at a producer spike, you stop at check 2. The ordering exists so that the common cases (spike, deploy regression) resolve in two checks and only the rare cases (poison message behind a rebalance) require the full descent.

The reason a fixed order matters more than the individual checks is that it protects you from the worst on-call failure mode: confirmation bias. At 02:00 your brain wants to grab the first plausible cause and act. The workflow forces you to run check 1 before you can convince yourself it is a downstream problem, and check 1 might immediately tell you it is one partition, not all — which rules out the downstream you were about to go restart. The order is a checklist precisely because checklists beat intuition when you are tired and the pressure is high.

### Why the order is broad-to-narrow

There is a reason the five checks run from broad to narrow rather than in some other sequence, and it is worth making explicit because it changes how you read each result. Check 1 (lag per partition) is the broadest possible cut: it partitions the entire space of causes into all-rising, one-rising, and flat-offset, three regions that share almost no fixes. Check 2 (throughput vs produce rate) cuts the remaining space by *direction* — producer-faster versus consumer-slower — which is again a coarse, high-information split. Only when those two broad cuts have localized you to a small region do the narrow checks (rebalance rate, downstream latency, DLQ) come into play, and by then each of them is testing a specific hypothesis rather than scanning blindly. Running a narrow check first — say, opening the consumer logs before you have classified the lag shape — is like searching a single drawer before you have decided which room the thing is in. You might get lucky, but on average you waste the most precious minutes of the incident looking in a place the broad checks would have ruled out for free.

This broad-to-narrow ordering also means each check has a natural *exit*: the moment a check resolves the cause, you stop and fix, and you do not run the remaining checks. A producer spike resolves at check 2; you never look at rebalances or DLQ. A poison message survives all the way to check 5 precisely because it is the cause that the broad checks cannot see — it hides as a flat offset on one partition, which checks 1 and 4 surface but cannot fully explain, so the descent continues to the narrow check that names it. The length of the descent is itself diagnostic: common, fleet-wide causes resolve shallow; rare, localized causes resolve deep. If you find yourself at check 5 frequently, that is a signal your system has a recurring poison-message or skew problem worth fixing structurally rather than re-diagnosing each time.

One practical note on running the workflow under pressure: narrate it in the incident channel as you go. Type "check 1: lag is on all 12 partitions, not one" and "check 2: produce rate flat at 36k, consume dropped to 12k at 02:10" as you make each observation. This costs a few seconds per check and buys two things — a written record that the next responder can pick up without re-deriving, and a forcing function that keeps *you* honest about running the checks in order instead of jumping to a conclusion. The act of writing "check 1" before you have an answer is what stops you from skipping straight to "I bet it's the database." The runbook in section 9 is built to be narrated this way.

## 7. The fix for each cause

Diagnosis without the matching fix is just anxiety. Here is the specific remediation for each branch of the tree, including the ones where the right move is to do nothing. The single most important meta-rule: the fix must match the *cause*, and several of these fixes actively harm a different cause, so applying the wrong fix is worse than waiting.

For a **producer organic spike**, the fix is usually to wait, because the spike is self-limiting and lag will drain when it ends — provided the backlog will not breach your latency SLO or the retention cliff before it drains. Do the drain math (section 2's worked example). If the drain fits inside your budget, watch and let it resolve. If it does not, temporarily scale consumers up to (but not beyond) the partition count, because that is the only scaling that helps, and scale back down after the spike. Do *not* restart consumers — a rebalance during a spike is throughput you cannot spare.

For a **backfill or replay**, the fix is operational: find who started it and throttle or pause it. The cleanest long-term answer is to run backfills on a separate topic or a separate consumer group with isolated capacity so they never contend with live traffic. On call, a message to the data team asking them to rate-limit the backfill producer often resolves the incident faster than any consumer-side change.

For an **upstream retry storm**, the fix is upstream: get the failing dependency healthy so the retries stop, or get the retry policy to back off exponentially, or deduplicate the storm at your consumer if the duplicates share an idempotency key. Critically, do not just scale your consumers to absorb a retry storm — an unbounded storm will outrun any amount of consumer capacity you can add, and you will burn money chasing a feedback loop. Break the loop.

For a **slow downstream dependency**, the fix is to fix the downstream or insulate the consumer from it: add a cache so most messages skip the slow call, batch the downstream calls so one round trip serves many messages, or increase consumer concurrency so more messages are in flight while each waits on I/O. The right lever depends on whether the consumer is I/O-bound (more concurrency helps) or the downstream is genuinely saturated (only fixing the downstream or shedding load helps).

For a **GC or memory problem**, the fix is to reduce allocation or tune the collector: lower `max.poll.records` so each poll allocates less, fix the leak, increase heap if the workload genuinely grew, or switch to a lower-pause collector. A GC problem can escalate to a stall if pauses blow the session timeout, so treat a worsening GC sawtooth with urgency.

For a **deploy regression**, the fix is to roll back. Rollback is faster, safer, and more reliable than diagnosing and fixing forward at 02:00. Roll back first; understand the regression in daylight. The catch-up math (section 3) tells you the drain time after rollback so you can set expectations in the incident channel.

For a **rebalance loop**, the fix is to stop the churn: freeze a flapping deploy, raise `max.poll.interval.ms` so a slow batch does not get the consumer evicted, raise the session timeout if heartbeats are being missed, or fix the OOM-kill cycle that keeps killing instances. Until the churn stops, nothing else can make progress, so this fix comes before any throughput tuning.

For a **deadlock or hang**, the fix is to capture a thread dump and then restart the wedged instance. The thread dump is the only evidence of the root cause; capture it first or you will be back here tomorrow with no more information than you have now.

For a **poison message**, the fix is to skip past it — seek the consumer forward one offset on the affected partition, route the bad record to a DLQ, and let the partition drain. Capture the record first, because skipping is a data decision.

For **key skew**, the fix is to spread the hot key: change the producer to use a composite key so the hot value hashes across several partitions, parallelize the hot partition's consumer across distinct keys if ordering allows, and later repartition or rekey structurally. Scaling the consumer group does nothing for skew, so do not reach for it.

The following table compresses this into a lookup you can scan mid-incident.

| Symptom signature | Most likely cause | The fix | What does NOT help |
| --- | --- | --- | --- |
| All partitions rising, produce rate stepped up, bounded window | Organic traffic spike | Wait, or temporarily scale to partition count | Restarting consumers |
| All partitions rising, produce rate 10–100x, rectangular | Backfill / replay | Throttle or isolate the backfill source | Scaling live consumers |
| Produce rate accelerating, duplicate keys | Upstream retry storm | Fix upstream, add backoff, dedup | Scaling to absorb the storm |
| All partitions rising, produce flat, consume dropped, offsets advancing | Slow downstream / deploy / GC | Fix downstream, roll back, or tune GC | Adding partitions |
| Offsets flat, rebalance rate high | Rebalance loop | Stop churn, raise poll interval / session timeout | Restarting (adds rebalances) |
| Offsets flat, rebalances calm, thread blocked | Deadlock / hang | Thread dump, then restart | Scaling |
| One partition flat, same offset retrying | Poison message | Skip past offset, DLQ the record | Restarting (re-reads poison) |
| One partition rising, offset advancing, hot produce rate | Key skew | Composite key, per-key concurrency | Scaling the consumer group |

The column that earns its keep is the last one — "what does not help" — because the wrong fix is the most common reason a lag incident drags on. Restarting consumers helps almost none of these and harms several. Scaling helps only a producer spike (and skew not at all). Internalize which fixes are inert or harmful for which cause, and you will stop making the incident worse while you work.

## 8. Confirming recovery and draining the backlog

You found the cause and applied the fix. You are not done. There is a dangerous moment in every lag incident, right after the fix, where the cause is resolved but the *backlog* is not, and an under-caffeinated engineer declares victory, closes the incident, and goes back to bed while two million records of lag sit undrained and the next producer burst tips it over again. Recovery has two phases, and you must confirm both: first that lag has stopped *growing*, and second that lag is *shrinking* and will reach baseline.

Confirming lag stopped growing is the easy half. After the fix, watch the lag derivative — the rate of change of lag — and confirm it has gone from positive (climbing) to zero or negative (flat or draining). If you rolled back a deploy, consume rate should jump back to its normal value within a minute, and the lag curve should visibly bend from climbing to flat. If lag is still climbing after the fix, the fix did not address the cause, and you are back to diagnosis — do not assume the fix worked because you applied it; confirm it on the curve. The derivative going to zero is the proof that the cause is resolved.

Confirming lag will drain is the half people skip, and it is governed by the asymmetry from figure 3: lag climbed fast and drains slow. The reason is arithmetic. During the fault, the deficit (P minus C) was large — maybe twenty-four thousand records per second. After the fix, the *surplus* (C minus P) is small, because in steady state C barely exceeds P — that is the whole point of a well-sized consumer group. If your steady consume rate is thirty-six thousand and produce is thirty-six thousand, your surplus after the fix is *zero*, and lag will sit at whatever it climbed to *forever*. To drain, you need surplus capacity, which means either the produce rate dips (off-peak) or you add consumers above the steady need. The drain time is the accumulated lag divided by the surplus rate, and the surplus is almost always much smaller than the deficit was, so the drain takes much longer than the climb.

#### Worked example: computing the drain and not overshooting

Take the section-3 incident: eleven point five million records of accumulated lag after rollback, steady produce and consume both thirty-six thousand per second. To drain in a reasonable time you scale up. Suppose you scale to forty-eight thousand per second of consume capacity (you doubled the consumers on the busiest partitions, capped by the partition count). Surplus is forty-eight thousand minus thirty-six thousand, twelve thousand per second. Drain time is eleven point five million divided by twelve thousand, which is nine hundred and fifty-eight seconds, just under sixteen minutes. You announce in the incident channel: "Cause fixed at 02:18, draining at 12k/s surplus, ETA to baseline 02:34." That ETA is a *computed* number, not a guess, and it sets the right expectation — the incident is not over at 02:18, it is over at 02:34.

The overshoot trap is the other half. If you scaled up to drain and you *leave* the extra consumers running after the backlog clears, you are now over-provisioned, paying for capacity you do not need, and — more subtly — if your consumer count now exceeds the partition count, the extra consumers sit idle (a partition has exactly one active owner) and on the next rebalance you may get a surprising assignment. So the drain plan has a *scale-down* step: when lag reaches baseline at 02:34, scale consumers back to the steady-state count. Confirm baseline first (lag is flat and low, derivative zero, for a few minutes), then scale down, then close the incident. The full lifecycle is fix the cause, confirm the derivative went to zero, scale up to drain, compute and announce the ETA, confirm baseline, scale back down, close. Skipping the back half is how a "resolved" incident reopens an hour later.

There is one more thing to confirm before you fully stand down: that you did not lose data. If lag ever crossed the retention cliff — if the oldest unprocessed records aged out of the topic's retention window before the consumer reached them — those records are gone, and "lag returned to zero" is then a lie, because lag dropped partly by the broker *deleting* unread records, not by the consumer processing them. Check whether your minimum offset advanced because you consumed or because retention deleted; if retention deleted unread records, you have a data-loss incident layered on the lag incident, and that needs its own follow-up. This is why the retention cliff is the true deadline of every lag incident and why time-lag (how many seconds of retention you have left) matters more than record-lag during the climb.

## 9. A reusable lag-spike runbook

Here is the runbook. It is deliberately terse — it is meant to be pasted into the incident channel and followed step by step at 02:00 when no one wants to think. Each step is a check or an action, in order, with the decision it gates.

```
LAG-SPIKE RUNBOOK  (paste into incident, follow in order)

0. DO NOT restart consumers yet. Restarting destroys evidence and
   triggers a rebalance. Diagnose first.

1. PER-PARTITION LAG  (the all-or-one question)
   kafka-consumer-groups.sh --describe --group <group>
   -> ALL partitions rising?   go to 2 (producer or fleet)
   -> ONE partition rising?     go to 5 (skew or stall)
   -> OFFSET FLAT (no advance)? go to 4 (stall/poison)

2. THROUGHPUT vs PRODUCE RATE
   Graph produce rate P and consume rate C, last 60 min.
   -> P stepped UP, C flat?  producer got faster -> step 3
   -> P flat, C dropped?     consumer got slower -> step 4 then 6

3. CLASSIFY THE PRODUCER SURGE
   -> Organic spike (bounded)? do drain math; wait or temp-scale.
   -> Backfill (rectangular, 10-100x)? throttle/isolate the source.
   -> Retry storm (accelerating, dup keys)? FIX UPSTREAM, add backoff.

4. REBALANCE + OFFSET ADVANCE
   Check rebalance rate and member count.
   -> Rebalances high?  rebalance loop -> stop churn, raise
                        max.poll.interval.ms / session.timeout.ms.
   -> Rebalances calm, offset flat, thread blocked? deadlock ->
      THREAD DUMP, then restart that instance.
   -> Same offset retrying in logs? poison message -> step 7.

5. SKEW vs STALL on the hot partition
   Is that partition's CURRENT-OFFSET advancing?
   -> Advancing + high produce-into-partition? SKEW ->
      composite key + per-key concurrency. (scaling group: NO-OP)
   -> Flat? -> step 4 (stall/poison path).

6. DOWNSTREAM + DEPLOY
   p99 of each per-message dependency (DB, cache, API).
   -> One stepped up w/ C drop? slow downstream -> cache/batch/concurrency.
   -> All healthy? check deploy timeline + GC -> ROLL BACK if a deploy
      lines up with the C drop.

7. POISON MESSAGE
   Capture the record (log or DLQ). Seek past it:
     consumer.seek(tp, stuck+1); commit(tp, stuck+1)
   Let the partition drain.

8. CONFIRM RECOVERY
   -> Lag derivative back to <= 0 (stopped growing)? if not, fix didn't
      address cause; re-diagnose.
   -> Compute drain ETA = accumulated_lag / surplus_rate. Announce it.
   -> Confirm baseline (lag flat + low for a few min).
   -> Scale consumers BACK to steady-state count.
   -> Check retention cliff: did min offset advance by consuming or by
      retention DELETING unread records? If deleted -> data-loss follow-up.
   -> Close.
```

The runbook encodes everything in this post, but the value of having it written down is that it removes judgment from the tired-engineer path. You do not have to *remember* that restarting destroys evidence, or that scaling does nothing for skew, or that the incident is not over until lag reaches baseline and you have scaled back down — the runbook holds those for you. Print it, pin it in the channel topic, link it from the alert. The best runbook is the one that is one click away from the page that fires.

![A grid dashboard with six panels showing lag per partition, group throughput, produce rate, error rate, rebalance rate, and downstream p99 latency that together localize the fault on sight](/imgs/blogs/debugging-consumer-lag-spikes-field-guide-8.webp)

Figure 8 is the dashboard the runbook assumes you have. Six panels — lag per partition, throughput, produce rate, error rate, rebalance rate, downstream latency — arranged so a single glance localizes the fault. In the snapshot drawn, partition seven's lag is at two million (red), group throughput is depressed at twelve thousand against a produce rate of thirty-six thousand, error rate is a steady low, rebalances are calm, and downstream p99 is elevated at nine milliseconds — that combination reads instantly as a consumer slowdown driven by a downstream, with a skewed partition on top. The point of the dashboard is that the runbook's checks are *fast* when the signals are all in one place; if you have to go hunting for the rebalance metric in a different tool, you lose the minutes that matter. Build the dashboard before the incident, not during.

## Case studies and war stories

These are composite incidents drawn from the patterns above. Each one teaches a lesson that the runbook now encodes, but the story is what makes the lesson stick.

### The deploy that tripled per-message latency

A payments team shipped a routine change to their `transactions` consumer at 14:30 on a Tuesday. The change added a fraud-check call to an internal service on every message — reasonable, reviewed, tested. Within four minutes, lag on all sixteen partitions began climbing at roughly six thousand records per second. The on-call engineer, seeing high lag, did the instinctive thing: restarted the consumers. This triggered a rebalance, paused consumption entirely for forty seconds, and lag jumped another quarter million — and then resumed climbing at the same rate, because the restart did nothing about the regression. It was fifteen minutes before someone looked at the deploy timeline, saw the 14:30 push line up exactly with the consume-rate drop, and rolled back. The per-message time had gone from two milliseconds to seven (the fraud call added five), a 3.5x regression, and the catch-up after rollback took twenty-two minutes of draining at a scaled-up rate. The lesson: the deploy timeline is the cheapest correlation you have, and the restart instinct cost them a rebalance and fifteen minutes of looking in the wrong place. Check the deploy timeline before you touch the consumers. This is the all-partitions, produce-flat, consume-dropped signature, and the workflow would have pointed at the deploy in two checks.

### The one customer who broke a marketplace

A marketplace partitioned its order stream by `seller_id`. For two years this was fine — load spread evenly enough across two hundred partitions. Then one seller ran a flash sale and produced forty times their normal volume for ninety minutes. All of it hashed to one partition. That partition's lag climbed to three million while the other one hundred and ninety-nine sat near zero. The on-call engineer saw three million lag and scaled the consumer group from one hundred to two hundred consumers — and nothing happened, because the hot partition still had exactly one owner, and the other ninety-nine new consumers had no partitions to take. It took a senior engineer to recognize the one-partition signature and explain that scaling is inert for skew. The mitigation was to ship a producer change that salted the hot seller's key across eight partitions, and to drain the existing backlog by running the hot partition's consumer with per-order concurrency. The lesson: skew is a data-distribution problem, and scaling the consumer group is the single most common wrong move. The all-or-one question would have caught it in thirty seconds: one partition, not all, means do not scale.

### The poison message that dammed a partition

A logistics platform's consumer hit a record with a malformed timestamp that threw on deserialization. The consumer's error handler caught the exception, logged it, and — fatally — did not advance the offset, so the next poll re-fetched the same record and threw again, forever. The committed offset on that one partition froze. Lag on that partition climbed past two million over an hour while the other partitions drained normally. The error logs were screaming, but they were screaming the *same* error with the *same* offset thousands of times, which the on-call engineer initially read as "lots of errors" rather than "one error, infinitely." The fix, once recognized, took ten seconds: seek past the bad offset and route it to the DLQ. The lesson: a single poison message at the head of a partition dams everything behind it, the signature is a flat offset on one partition with a repeating error, and the fix is disproportionately powerful — skipping one record unblocked two million. Read the error logs for *repetition*, not just volume.

### The rebalance storm hiding as a slowdown

A team set `max.poll.interval.ms` to thirty seconds, but a downstream batch job occasionally made a single `poll()` cycle take thirty-five seconds. Each time it did, the broker declared that consumer dead, kicked it out, and rebalanced the group — which paused everyone, which made the next poll cycle even longer because of the backlog, which made another consumer blow the interval, which rebalanced again. The group fell into a rebalance loop and lag climbed on all partitions while offsets barely advanced. It looked like a slowdown, but it was a stall driven by churn. The fix was to raise `max.poll.interval.ms` to five minutes so a slow batch never got the consumer evicted, which broke the loop instantly. The lesson: a too-tight poll interval turns an occasional slow batch into a self-sustaining rebalance storm, and the tell is high rebalance rate with flat offsets. The deeper treatment of this failure mode is in [Kafka rebalance storms and how to tame them](/blog/software-development/message-queue/kafka-rebalance-storms-and-how-to-tame-them); the on-call signature is in check 3 of the workflow.

## When to reach for systematic diagnosis (and when restarting is actually fine)

The whole thesis of this guide is that you should diagnose before you act. But let me be honest about the boundary, because there is a narrow band of cases where the restart-and-pray instinct is defensible, and pretending otherwise makes the advice less credible.

![A before-and-after comparison contrasting bouncing all consumers blindly which triggers a rebalance and worsens lag against reading the lag shape and five signals to isolate the cause in minutes](/imgs/blogs/debugging-consumer-lag-spikes-field-guide-9.webp)

Figure 9 draws the contrast. Restarting consumers blindly (left) triggers a full rebalance, pauses consumption, often deepens the lag, and — worst of all — destroys the evidence you need to find the actual cause, so the problem recurs. Systematic diagnosis (right) reads the lag shape and the five signals, isolates the cause in minutes, and applies a targeted fix that stays fixed. For the overwhelming majority of lag incidents, the right column is correct, because the cost of two minutes of diagnosis is far less than the cost of a wrong fix plus a recurrence.

Reach for systematic diagnosis whenever lag is *growing* and you do not already know why. A growing-lag incident with an unknown cause is exactly the situation the differential diagnosis is built for, and restarting before you understand it is how incidents triple in length. Reach for it especially when the lag pattern is one-partition or flat-offset, because those signatures have causes (skew, poison, deadlock) where the common fixes (scale, restart) are inert or harmful — getting the cause right is the entire game.

When is a restart actually fine? When you have *already* diagnosed a deadlock or a hung instance and captured the thread dump — restarting is then the correct fix, not a guess. When a single consumer instance is provably wedged (its partitions' offsets are flat, the rest of the group is healthy) and you have the evidence you need, a targeted restart of *that instance* is surgical, not pray. The thing to avoid is the *blind, group-wide* restart of healthy consumers in the hope that it shakes something loose — that one triggers a rebalance, destroys evidence, and usually makes lag worse. Restart with a diagnosis behind it; never restart as a substitute for one. And never restart during a producer spike, when the rebalance costs you the exact throughput you cannot spare.

There is also a case for *not* acting at all: a small, transient lag bump that is already draining. If the derivative is already negative — lag peaked and is coming down on its own — the incident may be self-resolving, and the correct action is to watch it for a few minutes and confirm it reaches baseline rather than to intervene and risk making it worse. Not every lag alert is an incident; some are the system absorbing a burst exactly as designed. Backpressure and burst absorption are features, not bugs, and the topic of [Consumer lag monitoring and autoscaling](/blog/software-development/message-queue/consumer-lag-monitoring-and-autoscaling) is in part about alerting on the derivative and the retention cliff precisely so that self-draining bumps do not page a human at all.

## Key takeaways

- The first and most discriminating question is always: is lag growing on **all** partitions or on **one** (or is the offset flat)? This single observation halves the search space and points you at completely different cause trees — capacity and producer faults on one side, skew and stuck consumers on the other.
- Lag's derivative is **produce rate minus consume rate**. If lag is climbing, either the producer got faster or the consumer got slower (or stalled). Graphing P and C over the last hour is the cheapest, highest-information observation after the per-partition split.
- A **slow** consumer's offset advances too slowly; a **stalled** consumer's offset is **flat**. They are different incidents with different fixes — never apply the slow-consumer fixes (scale, roll back, fix downstream) to a stalled consumer, where the cause is a rebalance loop, a deadlock, or a poison message.
- **Skew is inert to scaling.** One partition far ahead of the rest, with its offset advancing, means a hot key — and adding consumers does nothing because a partition has exactly one owner. The fix is to spread the key (composite key, salt) and add per-key concurrency, not to scale the group.
- A **poison message** at the head of a partition dams everything behind it: one undeliverable record can hold up millions. The signature is a flat offset on one partition with the same error and offset repeating in the logs. The fix — skip past the offset and DLQ the record — is disproportionately powerful.
- **Restarting consumers is rarely the fix and often harms.** It triggers a rebalance, destroys evidence, and helps almost no cause except a confirmed deadlock or a single wedged instance. Diagnose first; restart only with evidence behind it.
- Run the **five checks in order**: lag per partition, throughput vs produce rate, rebalance activity, downstream latency, DLQ/poison. The order rules out whole branches with each step and protects you from confirmation bias at 02:00.
- The incident is **not over when the cause is fixed**. Lag climbs fast and drains slow because the post-fix surplus is much smaller than the fault deficit. Compute the drain ETA from accumulated lag divided by surplus rate, announce it, confirm baseline, scale back down, and check the retention cliff for data loss.
- Keep a **runbook** pinned to the alert and a **dashboard** with lag-per-partition, throughput, rebalance rate, and downstream latency in one place. The runbook removes judgment from the tired-engineer path; the dashboard makes the five checks fast.

## Further reading

- [Consumer lag monitoring and autoscaling](/blog/software-development/message-queue/consumer-lag-monitoring-and-autoscaling) — the companion to this guide: how to define lag precisely, alert on the derivative and the retention cliff instead of an absolute threshold, and autoscale on lag up to the partition ceiling.
- [Consumer optimization and scaling](/blog/software-development/message-queue/consumer-optimization-and-scaling) — how to make a single consumer faster (the per-message-time lever) before you reach for more of them, and the capacity inequality that governs whether lag grows.
- [Poison messages and retry storms](/blog/software-development/message-queue/poison-messages-and-retry-storms-containment) — detecting and containing the undeliverable message that dams a partition, and the upstream retry storm it can escalate into.
- [Kafka rebalance storms and how to tame them](/blog/software-development/message-queue/kafka-rebalance-storms-and-how-to-tame-them) — the deep dive on the rebalance loop: why a too-tight poll interval turns a slow batch into a self-sustaining storm, and how to configure the group to stay calm.
- [Kafka consumer groups, offsets, and rebalancing](/blog/software-development/message-queue/kafka-consumer-groups-offsets-rebalancing) — the mechanics of the offsets you subtract to compute lag and the rebalance protocol that reassigns partitions.
- [Apache Kafka documentation: consumer configs](https://kafka.apache.org/documentation/#consumerconfigs) — the authoritative reference for `max.poll.interval.ms`, `session.timeout.ms`, `max.poll.records`, and the other knobs that govern stalls and rebalances.
- [Burrow: Kafka consumer lag checking](https://github.com/linkedin/Burrow) — LinkedIn's lag-monitoring tool that classifies consumer status (OK, WARN, ERR) from offset history rather than a static threshold.
