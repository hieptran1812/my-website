---
title: "Rebalance Storms: Causes, Diagnosis, and How to Tame Them"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Diagnose and stop a Kafka consumer group that rebalances over and over while lag climbs: understand the vicious slow-processing cycle, read the coordinator logs to name the trigger, then apply cooperative-sticky assignment, static membership, and poll-loop tuning to turn a self-inflicted outage back into a steady stream."
tags:
  [
    "message-queue",
    "kafka",
    "rebalancing",
    "consumer-groups",
    "rabbitmq",
    "distributed-systems",
    "event-driven",
    "reliability",
    "incident-response",
    "observability",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/kafka-rebalance-storms-and-how-to-tame-them-1.webp"
---

There is a particular kind of three-in-the-morning incident that breaks people's mental model of Kafka. The dashboards say the consumers are up. The pods are green. The processes are not crash-looping in any way the orchestrator notices. CPU is moderate, memory is fine, the network is healthy. And yet consumer lag is climbing at a steady tens of thousands of messages a minute, and it will not stop. You scale the consumers out, hoping more workers will chew through the backlog, and the lag climbs *faster*. You restart the group, and for a few seconds it looks like it might recover, and then it goes right back to climbing. Nothing is on fire. Everything is broken. This is a rebalance storm, and it is one of the most baffling failure modes in the whole streaming stack precisely because every individual component looks healthy while the group as a whole makes essentially zero forward progress.

A rebalance storm is a consumer group that rebalances over and over, spending so much of its wall-clock time in the stop-the-world coordination dance that it never gets to do meaningful work between rounds. The cruel part is the feedback loop. The thing that triggers the rebalance — usually a batch of messages that takes longer to process than the consumer is allowed to go silent — is *exactly* the thing that will happen again on the very next attempt. So the group rebalances, reassigns partitions, a consumer picks up where the slow batch left off, starts processing it, takes too long again, gets kicked out again, and the whole group pauses again to redo the assignment. Figure 1 traces this loop as a timeline: a slow batch, a missed poll deadline, an eviction, a group-wide pause, a rejoin, and then the same slow batch starting over. The offsets never commit. The lag never falls.

![Timeline of a Kafka rebalance storm showing a slow batch leading to a missed poll deadline, member eviction, a group-wide pause, a rejoin, and the same slow batch repeating with lag still rising and zero commits](/imgs/blogs/kafka-rebalance-storms-and-how-to-tame-them-1.webp)

By the end of this post you will be able to look at a group that is "up but not progressing," read the coordinator logs to find which member is triggering the churn and why, distinguish a slow-processing storm from a GC-driven session-timeout storm from a crash-loop storm from autoscaling churn, and apply the right fix for each: the cooperative-sticky assignor for incremental rebalances, static membership so routine restarts stop triggering reassignment at all, and poll-loop tuning that decouples how much you fetch from how long you take to process it. This post operationalizes the mechanics laid out in [Kafka consumer groups, offsets, and rebalancing](/blog/software-development/message-queue/kafka-consumer-groups-offsets-rebalancing) — if the words "coordinator," "generation," and "assignor" are unfamiliar, read that first and come back. Here we are concerned with what happens when that machinery turns against you.

## What a rebalance storm looks like

Before we get into causes, it helps to fix in your mind what the symptom actually is, because "rebalancing" is normal and healthy in small doses. Every time a consumer joins or leaves a group, Kafka has to redistribute partitions across the surviving members so that each partition is owned by exactly one consumer. That is the entire point of a consumer group: elastic, fault-tolerant parallelism over a partitioned log. A deploy that rolls ten pods will cause a handful of rebalances, each lasting a few seconds, and then the group settles. That is not a storm. That is the system working.

A storm is when the group cannot settle. The defining characteristic is *frequency*: the group rebalances every few seconds to every couple of minutes, continuously, for as long as you let it run. If you graph the rebalance rate — and you should be graphing it, more on that later — a healthy group shows occasional spikes around deploys and is flat at zero the rest of the time. A storming group shows a solid band: rebalances back to back, never quiet. The second characteristic is *no progress*: committed offsets barely move, or do not move at all, while the log-end offset keeps advancing because producers keep producing. Lag is the gap between those two, so lag grows without bound.

### The deceptive health signals

The reason this incident is so confusing is that all the signals you normally trust to mean "the consumer is healthy" are still green. The process is running — it has not crashed, it is not OOM-killed, the liveness probe passes because the HTTP health endpoint responds. CPU is often *low*, counterintuitively, because the consumer spends most of its time blocked in the rebalance protocol waiting for the coordinator and the other members, not doing CPU-bound work. From the outside, the consumer looks like a perfectly healthy process that has simply stopped getting anything done.

The give-away is in three places. First, the consumer logs, which will be full of lines about revoking partitions, rejoining, and being assigned partitions — over and over, with the generation number ticking up each round. Second, the lag metric, which climbs monotonically. Third, the `RebalanceInProgress` exceptions that the client throws when you try to commit an offset during a rebalance: if your commit calls are constantly failing with that error, the group is constantly rebalancing. None of these three show up on a default "is the service up" dashboard, which is exactly why teams stare at green tiles for twenty minutes before someone thinks to open the consumer logs.

### Why "scale out" makes it worse

The instinct, when lag climbs, is to add consumers. In a normal slow-consumer situation that is correct: more consumers means more parallelism means faster drain, up to the partition count. In a storm it is poison. Every consumer you add is another join event, which triggers another rebalance, which pauses the *entire* group again under the eager protocol. You have taken a group that was rebalancing every thirty seconds and given it more reasons to rebalance. Worse, the new consumers inherit the same slow processing that caused the storm in the first place, so they too will be evicted, adding leave events on top of the join events. Scaling out during a storm is like adding cars to a traffic jam: each one individually seems like it should help throughput, and collectively they grind everything to a halt. The fix is never "more consumers." The fix is "stop the churn."

### A storm versus a one-off rebalance

It is worth being precise about the boundary, because the same machinery produces both a healthy rebalance and a pathological one, and an on-call engineer who cannot tell them apart will either panic over a normal deploy or shrug off a real storm. A *healthy* rebalance is bounded and self-terminating: a trigger fires (a deploy, a scale event, a single transient failure), the group reassigns once or a small handful of times, and then the rebalance count goes flat again because the trigger is gone. The group converges to a stable assignment and stays there. The total time spent rebalancing is a few seconds to a few tens of seconds, and then it is over for hours.

A *storm* is unbounded and self-sustaining: the trigger does not go away when the rebalance finishes, because the rebalance itself recreates the conditions that fire the next trigger. A slow batch causes an eviction; the eviction causes a rebalance; the rebalance reassigns the same slow batch to someone who will also be evicted by it. The feedback closes. That is the single most useful diagnostic question to ask in the first thirty seconds of an incident: *is the rebalance count converging or diverging?* If it converges, you have a transient and you mostly need to wait. If it diverges — if it has been rebalancing continuously for minutes with no sign of settling — you have a storm, and waiting will only let lag climb toward the retention cliff. Everything in this post is about the diverging case.

The other clean separator is whether *committed offsets are moving*. A group can rebalance frequently and still make progress if each consumer manages to finish and commit some batches between rounds — that is unpleasant but survivable, and it is really a tuning problem rather than a storm. A true storm is defined by committed offsets that are *frozen*: between every pair of rebalances, no consumer manages to complete a batch and commit, so the committed position never advances. Watch the committed-offset metric, not just the rebalance rate. Rebalances with progress is a degraded group; rebalances without progress is an outage.

## The vicious cycle: slow processing and max.poll.interval

The single most common cause of a rebalance storm — and the one whose mechanics you must internalize — is processing that takes longer than `max.poll.interval.ms`. To understand why, you have to understand what `poll()` actually does and what promise the consumer makes to the group coordinator by calling it.

A Kafka consumer is a single-threaded loop at its core. You call `poll(timeout)`, which returns a batch of records (up to `max.poll.records`, default 500). You process those records — write to a database, call an API, run inference, whatever your business logic is. Then you loop back and call `poll()` again. That next `poll()` call does two critical things beyond fetching more records: it sends the heartbeats and, more importantly, it is the signal to the coordinator that *this consumer is alive and making progress*. The contract is: you must call `poll()` again within `max.poll.interval.ms` (default 300 seconds, five minutes). If you do not — if processing the previous batch takes longer than that interval — the coordinator concludes the consumer has stalled, removes it from the group, and triggers a rebalance to reassign its partitions to someone else.

### The clock you keep missing

Here is the trap, drawn as a feedback DAG in Figure 5. Suppose your per-record processing is heavy: each record makes a synchronous call to a downstream service that takes, on average, 640 milliseconds. With `max.poll.records=500`, a single poll batch takes 500 × 0.64 = 320 seconds to process. Your `max.poll.interval.ms` is the default 300 seconds. So every single batch you process blows past the deadline by 20 seconds. The coordinator evicts you mid-batch. It reassigns your partitions — possibly back to you after the rebalance, possibly to a peer — and whoever gets them starts reading from the last *committed* offset, which is before this batch (because you never finished it, so you never committed it). They fetch the same 500 records. They start the same 320-second slog. They get evicted at 300 seconds. Forever.

![Directed acyclic graph of the storm feedback showing a slow batch exceeding the interval leading to eviction, revocation, rejoin, and reassignment that resumes the same slow batch, all routing to a no-progress outcome where lag climbs](/imgs/blogs/kafka-rebalance-storms-and-how-to-tame-them-5.webp)

Notice what is conserved across every iteration of this loop: the work. The same 500 records are fetched, partially processed, and abandoned every single round. The system is doing real CPU and real downstream calls — it is *busy* — but it is busy redoing the same prefix of the same batch over and over. That is why CPU is not pinned at zero and why "scale out" does not help: you do not have a capacity problem in the sense of "not enough total compute," you have a structural problem where no unit of work ever completes because the deadline guillotine falls before the batch is done.

#### Worked example: a 20-consumer group hitting the interval

Let me make this concrete with numbers. You run a consumer group of 20 consumers reading from a topic with 60 partitions, so each consumer owns 3 partitions. Each consumer is configured with the defaults: `max.poll.records=500` and `max.poll.interval.ms=300000` (300 seconds). Your processing does an enrichment lookup per record against a service that, under the current load, responds in about 640 ms at p50. Single-threaded processing on the poll thread means each batch of 500 records takes 500 × 0.64 = 320 seconds.

Every consumer, on every poll, exceeds the 300-second interval by 20 seconds. The coordinator evicts each one as it crosses the deadline. Because evictions are staggered (the consumers did not all start their batches at the same instant), you get a near-continuous stream of leave-and-rejoin events: roughly one eviction every 320/20 = 16 seconds across the group, each triggering a stop-the-world rebalance that pauses all 20 consumers for the duration of the assignment round (call it 4 seconds of pause per round on a 20-member group). So the group spends 4 seconds out of every 16 — a full 25% of wall-clock time — frozen in rebalance, and the other 12 seconds redoing the same partial batches. Committed offsets: zero. Lag at, say, 30,000 messages/second of producer rate: climbing by 30,000 × 60 = 1.8 million messages per minute. This group is, for all practical purposes, down.

The fix here is purely arithmetic, and you have two dials. Dial one: lower `max.poll.records` so the batch fits inside the interval. If you set `max.poll.records=400`, a batch takes 400 × 0.64 = 256 seconds, comfortably under 300, and processing completes, offsets commit, the storm stops. Dial two: raise `max.poll.interval.ms` so the deadline accommodates the batch. Setting it to 600000 (10 minutes) makes the 320-second batch legal with margin. Either one breaks the loop. In practice you do both — shrink the batch *and* widen the interval — to leave headroom for the inevitable day when the downstream service slows to 900 ms and your batch math changes underneath you. We will return to the structural fix (get the processing off the poll thread entirely) in the poll-tuning section, because tuning the batch is a tourniquet, not a cure.

### Why the offset never advances

It is worth dwelling on *why* committed offsets stay frozen, because it explains the "lag climbs forever" part. Kafka consumers commit offsets to mark how far they have processed, and the standard pattern (whether auto-commit or manual) commits *after* a batch is processed, at the top of the next poll loop. If processing never completes a batch — because the consumer is evicted mid-batch every time — the commit at the top of the next loop never runs for that batch. After eviction, when partitions are reassigned, the new owner reads from the *last successfully committed* offset, which predates the doomed batch. So the consumer group is permanently stuck reading and re-reading the same window of the log, never advancing the committed position, while the producer-side log-end offset marches on. Lag, being log-end minus committed, grows by exactly the producer rate, unbounded, until either the storm is fixed or the unprocessed data ages out of retention and is lost — which converts a performance incident into a data-loss incident.

There is a sharper version of this that catches people who try to be clever with commits. Suppose you commit *incrementally* mid-batch — say, every 50 records — hoping to preserve some progress across evictions. That helps a little: the re-read window shrinks from 500 records to the last 50 you had not yet committed. But it does not break the storm, because the binding constraint is not how much you re-read; it is that *the full batch never fits inside the interval*. Even with incremental commits, you still fetch a 500-record batch, you still spend more than 300 seconds on it, and you are still evicted before you finish — you just resume 50 records further along next time. The storm continues; the lag still climbs, just slightly slower. Incremental commits are good practice for *crash* recovery (less reprocessing on a clean restart), but they are not a fix for a poll-interval storm. The only fixes for a poll-interval storm are the ones that make the batch fit the interval or remove the interval as the constraint: smaller batches, a wider interval, or processing moved off the poll thread.

### The poll loop is single-threaded by design

One thing that surprises engineers coming from thread-pool web frameworks is that the Kafka consumer is fundamentally single-threaded for processing. There is a background heartbeat thread, but `poll()`, your record handling, and offset commits all run on *one* thread, in sequence, by design. That design is deliberate and good — it gives you a simple, deterministic processing order per partition without locks — but it means the consumer makes Kafka a promise it can only keep by returning to `poll()` promptly. The entire `max.poll.interval.ms` mechanism exists because the broker has no other way to tell a *slow* consumer apart from a *dead* one. A consumer stuck in an infinite loop and a consumer doing legitimately slow work both look identical from the coordinator's side: a member that stopped polling. The interval is the broker's only signal, and it is necessarily a blunt one. That is why the storm is so common: any time your real processing time approaches the interval, a perfectly alive consumer gets treated as dead, and the cure (reassign its work) recreates the disease (the work is still slow). Understanding that the broker cannot see *inside* your poll loop — it sees only the gaps between calls — is the key intuition for why every fix in this post is ultimately about controlling those gaps.

## Other triggers: session timeout, crash loops, autoscaling churn

Slow processing is the headline cause, but it is not the only way a group gets stuck in a rebalance loop. Figure 2 lays out the four root causes as a tree; each branch has a different signature in the logs and a different fix, so misdiagnosing which one you have will send you tuning the wrong knob for an hour while lag climbs.

![Tree of rebalance storm causes branching into slow processing, session timeout too low, crash loop, and autoscaling churn, each with a concrete leaf such as a batch exceeding the poll interval, a heartbeat missed in six seconds, a join-then-die loop, and pods flapping on a CPU spike](/imgs/blogs/kafka-rebalance-storms-and-how-to-tame-them-2.webp)

### Session timeout and the heartbeat clock

There are actually *two* independent liveness clocks in a Kafka consumer, and conflating them is a classic source of confusion. The first is the poll-interval clock we just covered: are you calling `poll()` often enough to prove you are making progress? The second is the *heartbeat* clock: `session.timeout.ms` (commonly 45 seconds in recent defaults, historically 10 seconds) bounds how long the coordinator will wait without receiving a heartbeat before declaring the consumer dead. Heartbeats are sent by a background thread every `heartbeat.interval.ms` (default 3 seconds), independent of your processing. Figure 7 stacks these knobs.

![Stack of the three timeout knobs showing heartbeat interval at three seconds in the background, session timeout at forty-five seconds for liveness, max poll interval at three hundred seconds for progress, max poll records at five hundred for batch size, and the warning that exceeding either liveness or progress leads to eviction](/imgs/blogs/kafka-rebalance-storms-and-how-to-tame-them-7.webp)

So why would heartbeats fail if they run on a background thread? Two reasons. First, a stop-the-world garbage-collection pause. If your JVM does a 12-second full GC and your `session.timeout.ms` is 10 seconds, the heartbeat thread is frozen along with everything else for longer than the session allows, the coordinator misses heartbeats, and you get evicted — even though your processing was perfectly fast and your poll loop was tight. The fix here is *not* to touch `max.poll.interval.ms`; it is to raise `session.timeout.ms` above your worst-case GC pause (or, better, fix the GC), and to keep `heartbeat.interval.ms` at roughly one-third of the session timeout so you get three heartbeat attempts before the session expires. Setting `session.timeout.ms` too low relative to your GC behavior is a self-inflicted storm: every long GC evicts a member, every eviction triggers a rebalance, and on a busy heap the GCs keep coming.

The second reason heartbeats fail is when the heartbeat thread itself is blocked — for example, in old client versions where certain operations ran on the heartbeat thread, or when the process is swapping and even background threads stall. These are rarer, but the symptom (evictions with no slow processing, no crash) points at the session/heartbeat clock rather than the poll-interval clock, and the log message tells you which: a session-timeout eviction reads differently from a poll-interval eviction.

### Crash loops that look like a storm

A consumer that joins the group, processes for thirty seconds, crashes, gets restarted by the orchestrator, rejoins, and crashes again is a different beast that *presents* as a rebalance storm because every join and every leave triggers a rebalance. If you have a pod stuck in a crash loop — say it hits a poison message that throws an unhandled exception, or it has a memory leak that OOM-kills it every two minutes — that single bad pod will rebalance the whole group every time it cycles. Ten restarts in ten minutes is twenty rebalance events (a leave on crash, a join on restart), each pausing the group.

The tell here is that the rebalance frequency correlates with a pod's restart count, and the same member id (or, with static membership, the same `group.instance.id`) keeps appearing in the eviction logs. The fix is not a Kafka config at all — it is to stop the crash. Often that means containing the poison message with a [dead-letter queue and bounded retries](/blog/software-development/message-queue/kafka-consumer-groups-offsets-rebalancing) so one bad record cannot take down the consumer, or fixing the memory leak. But static membership *also* helps here as a damage limiter, because under static membership a member that restarts within the session timeout does not trigger a rebalance at all — the group waits for it to come back. We will get to that.

### Autoscaling churn

The fourth cause is self-inflicted by your own infrastructure. If you run consumers on a horizontal pod autoscaler keyed on CPU, and your processing is bursty, the autoscaler can flap: a CPU spike adds three pods, the spike passes, the autoscaler removes three pods, another spike adds them back. Every scale-up is N join events; every scale-down is N leave events. A flapping autoscaler can keep a group in continuous rebalance purely through membership churn, even when each individual consumer is perfectly healthy and fast. This is especially nasty because the autoscaler is *trying* to help — it sees high CPU (often caused by the rebalances themselves) and adds capacity, which causes more rebalances, which raises CPU, which adds more capacity. The fix is to make autoscaling decisions on a stable signal (consumer lag, not instantaneous CPU), add generous stabilization windows so the autoscaler does not react to transient spikes, and cap the scaling velocity. We cover lag-based scaling in depth in [consumer optimization and scaling](/blog/software-development/message-queue/consumer-optimization-and-scaling); the point for storms is that membership should change slowly and deliberately, never reflexively.

## Why a storm is effectively an outage

It is tempting to file a rebalance storm under "performance degradation" rather than "outage," because the consumers are technically running. That framing will cost you. Under the default *eager* rebalance protocol, a rebalance is stop-the-world for the entire group: when any single member joins or leaves, *every* member revokes *all* of its partitions, the assignment is recomputed from scratch, and only then does processing resume. During that window — typically a few seconds, but longer on large groups or under contention — not one partition in the group is being consumed. Figure 3 contrasts this with the cooperative-plus-static end state.

![Before-and-after comparison of eager stop-the-world rebalancing where any trigger revokes all partitions and twenty members idle while a restart causes N rebalances, versus cooperative-sticky with static membership where only the moved partitions are revoked, the others keep processing, and a restart causes zero rebalances](/imgs/blogs/kafka-rebalance-storms-and-how-to-tame-them-3.webp)

### The duty-cycle math of an eager storm

Quantify it. Say a single rebalance round on your 20-member group takes 4 seconds end to end — gather all members, compute assignment, distribute, members re-fetch and seek. If the group rebalances once every 30 seconds (a mild storm), it is frozen for 4 of every 30 seconds, a 13% duty-cycle loss. That alone would knock your effective throughput down by an eighth. But in a real storm the frequency is much higher — every 10 seconds, every 5 seconds — and at one rebalance every 8 seconds with a 4-second round, you are frozen *half* the time, and the other half is spent redoing abandoned work. Effective throughput is not 50%; it is near zero, because the work done in the un-frozen half never completes a batch and never commits.

So the duty-cycle loss is the *optimistic* framing. The pessimistic and accurate framing is that a group in a storm produces no committed offsets at all. From the perspective of everything downstream of that consumer group — the database it writes to, the cache it warms, the search index it updates, the alerts it fires — the data has simply stopped flowing. If this group is the one that updates your fraud-detection features, fraud detection is now running on stale data. If it is the one that materializes order events into the fulfillment system, orders are not being fulfilled. The fact that a `ps` shows the process running does not change the business reality: the pipeline is down. Treat it like a page, not a ticket.

The table below makes the contrast between a healthy group, a degraded group, and a storming group explicit, so you can place an incident on the spectrum quickly. The key column is the last one — committed-offset progress — because that is what separates a survivable degradation from an outage.

| Group state | Rebalance rate | Per-round pause (eager) | Committed offsets | Verdict |
| --- | --- | --- | --- | --- |
| Healthy | ~0 outside deploys | n/a | advance steadily | normal |
| Deploy / scale event | a few, then quiet | 2–4 s, converging | brief pause, then resume | transient |
| Degraded | several per minute | 2–4 s each | advance, but slowly | tuning problem |
| Storm | continuous, every few s | 3–5 s each, never quiet | frozen, no advance | outage |

The reason "rebalance rate" alone is not a sufficient alarm is visible in this table: a deploy and a storm can momentarily look similar on a rebalance-rate graph. The discriminator is *duration and convergence* — does the rate come back to zero (deploy) or stay high (storm) — combined with whether committed offsets are moving. An alert that fires on a high *sustained* rebalance rate (say, more than N rebalances over a 5-minute window) plus *flat* committed offsets catches storms while ignoring deploys. That two-signal alert is worth building before you ever need it.

### The retention cliff turns it into data loss

There is a hard deadline lurking underneath every storm: topic retention. Kafka keeps messages for a configured time or size (say seven days, or some terabytes), after which the oldest segments are deleted regardless of whether anyone has consumed them. While your group is stuck reading and re-reading the same window, the producer keeps appending, and the *un*-consumed frontier keeps growing. If the storm runs long enough that lag exceeds retention, the oldest unprocessed messages are deleted before your group ever processes them. Now you have lost data, permanently, and when you finally fix the storm the group will either throw an offset-out-of-range error or silently skip forward to the earliest still-retained offset, depending on `auto.offset.reset`. A storm that runs for hours on a high-throughput topic with tight retention is a data-loss incident with a fuse on it. This is the single best argument for treating storms as P1 outages and for alerting on rebalance rate directly rather than waiting to notice the lag.

## Diagnosing the trigger

You cannot fix a storm until you know which of the four causes you are dealing with, and the good news is that Kafka tells you — in the coordinator and consumer logs — exactly which member triggered each rebalance and, usually, why. The diagnosis follows a fixed order, shown as a pipeline in Figure 6: measure the rebalance rate to confirm you actually have a storm, identify the member the logs blame, classify the trigger, apply the matching fix, and verify the rate drops to zero.

![Pipeline of the diagnosis workflow moving from measuring the rebalance rate above one per minute, to finding the triggering member in the logs, to classifying the trigger, to applying the matching fix, to verifying the rate drops toward zero per hour](/imgs/blogs/kafka-rebalance-storms-and-how-to-tame-them-6.webp)

### Step one: measure the rebalance rate

First, confirm the rate. The broker-side coordinator and the client both expose rebalance metrics. On the consumer client, the JMX metric `rebalance-rate-per-hour` (and `rebalance-total`) under the `consumer-coordinator-metrics` group tells you how often this consumer has been rebalancing. A healthy group sits near zero outside deploys. Anything sustained above a few per hour is suspicious; sustained above a few per *minute* is a storm. You should be scraping these into your metrics system continuously, not discovering them during an incident — an alert on `rate(kafka_consumer_coordinator_rebalance_total[5m]) > threshold` is the single most valuable storm detector you can deploy, because it fires on the *cause* (churn) rather than the lagging *symptom* (lag), buying you minutes before the retention cliff.

If you do not have the metric wired up, you can read it from the logs directly:

```bash
# Count rebalance-trigger log lines per minute from a consumer's log
# to confirm a storm and see its cadence.
grep -E "Revoke previously assigned partitions|Request joining group|\
Successfully joined group" consumer.log \
  | awk '{print $1, $2}' \
  | cut -d: -f1-2 \
  | sort | uniq -c

# Each cluster of "join group" lines is one rebalance.
# A storm shows tens of them per minute, back to back.
```

### Step two: find the member the logs name

Kafka's group coordinator logs the reason for each rebalance and names the member that triggered it. On the broker side, look in the controller/coordinator logs for lines like:

```
[GroupCoordinator] Member consumer-7-a1b2c3 in group orders-enrich
  has failed, removing it from the group
[GroupCoordinator] Preparing to rebalance group orders-enrich in state
  PreparingRebalance with old generation 412 (reason: removing member
  consumer-7-a1b2c3 on heartbeat expiration)
```

That `reason` field is gold. `heartbeat expiration` points at the session-timeout/GC clock. `removing member ... on LeaveGroup request` with a poll-interval message points at slow processing (the client sends an explicit LeaveGroup when it detects it has blown the poll interval, in newer clients). A member id that changes every time points at restarts (no static membership); a member id that *stays the same* and keeps getting evicted points at one specific consumer with a problem. The generation number ticking up rapidly (412, 413, 414…) confirms the cadence. On the client side, the consumer logs the complementary view:

```
[Consumer clientId=consumer-7, groupId=orders-enrich] Member
  consumer-7-a1b2c3 sending LeaveGroup request to coordinator due to
  consumer poll timeout has expired. This means the time between
  subsequent calls to poll() was longer than the configured
  max.poll.interval.ms, which typically implies that the poll loop is
  spending too much time processing messages.
```

That message is the diagnosis written out in full: this is a slow-processing storm. Different message, different cause. Read the actual log line; do not guess.

### Step three: classify and confirm

With the member named and the reason in hand, classify against the four causes. If the reason is poll-timeout and the named member rotates, it is slow processing across the whole group — tune the poll loop. If the reason is heartbeat expiration and you can correlate it with GC logs showing long pauses, it is the session-timeout/GC clock — raise the session timeout and fix GC. If one specific member id keeps dying and its restart count is climbing in the orchestrator, it is a crash loop — fix the crash, contain the poison message. If the rebalances correlate with autoscaler scale events, it is membership churn — stabilize the autoscaler. The classification dictates the fix; getting it wrong wastes the window before the retention cliff. The `RebalanceInProgress` exception flooding your commit path is a *confirmation* that you are storming but does not by itself tell you which cause — it is the symptom, the logs carry the cause.

#### Worked example: reading the logs to pick the fix

Suppose during an incident you grep the coordinator log and see, over a two-minute window, 14 rebalances, all with `reason: removing member ... on heartbeat expiration`, and the member ids rotate across all 20 consumers rather than fixating on one. You pull the GC logs from three of the named consumers and find full-GC pauses of 11 to 14 seconds occurring every 30 to 60 seconds. Your `session.timeout.ms` is set to 10000 (10 seconds). The diagnosis writes itself: every long GC pause freezes the heartbeat thread past the 10-second session timeout, the coordinator evicts the member, and because the heap pressure is group-wide (they are all doing the same work), the GCs and therefore the evictions hit every member in rotation. This is *not* a slow-processing storm — `max.poll.interval.ms` is irrelevant here. The fix is to raise `session.timeout.ms` to, say, 45000 so it comfortably exceeds the worst-case GC pause, set `heartbeat.interval.ms` to 15000 (one-third), and then go fix the actual GC problem (right-size the heap, switch to G1 or ZGC, reduce allocation in the hot path). Raising `max.poll.interval.ms` instead would have done nothing, and you would have watched lag climb for another hour. Read the reason field.

## Fix 1: cooperative-sticky incremental rebalancing

Now the fixes. The first and most broadly effective is to stop using the eager rebalance protocol and switch to the *cooperative* one, via the cooperative-sticky assignor. This does not eliminate rebalances — triggers still cause reassignment — but it changes a stop-the-world reassignment of *everything* into an incremental reassignment of *only what needs to move*, which dramatically shrinks the blast radius and the pause time of each rebalance.

### Eager vs cooperative, mechanically

Under the eager protocol (the historical default, with the `RangeAssignor` or `RoundRobinAssignor`), every rebalance is a clean slate. The moment a trigger fires, every member revokes *all* of its partitions — even the ones it is going to keep — and stops processing entirely. The coordinator then runs the assignor, hands out the new assignment, and members re-acquire their partitions and resume. The revoke-everything step is the killer: a member that owns partitions 0, 1, 2 and will own 0, 1, 2 again after the rebalance *still* gives them up and re-takes them, pausing processing on all three for the whole round.

The cooperative protocol (the `CooperativeStickyAssignor`) changes this in two ways. First, it is *sticky*: it tries hard to keep each partition with its current owner across rebalances, minimizing movement. Second, and crucially, the revocation is *incremental* and happens in two phases. In the first rebalance round, members keep processing all their current partitions and report what they hold; the assignor computes the desired end state and identifies only the partitions that need to *move* from one member to another. In a second, brief round, only those specific partitions are revoked from their old owners and assigned to their new owners. Partitions that are not moving are never revoked and never stop processing. So during a cooperative rebalance, the vast majority of the group's partitions keep flowing the entire time. Only the handful that actually change hands experience a brief pause.

### Why this matters for storms

The connection to storms is direct. In an eager storm, every rebalance pauses 100% of the group, so a high rebalance frequency means a high fraction of total time frozen, as the duty-cycle math showed. Under cooperative rebalancing, even a fairly high rebalance frequency only pauses the small slice of partitions that move each round — often zero, if the membership is stable and the trigger was transient. Cooperative rebalancing does not fix the *trigger* (a slow batch still gets you evicted), but it removes the amplification: one member's problem no longer freezes the whole group. It converts a group-wide outage into a localized hiccup, which is exactly the difference between a P1 and a non-event.

Switching is a config change on the consumers, but it requires care because you cannot mix eager and cooperative members in the same group safely — they disagree about the protocol. The supported path is a *rolling* upgrade through an intermediate step:

```properties
# Step 1: deploy with BOTH assignors listed, cooperative-sticky first.
# During the roll, the group negotiates down to the common protocol
# (eager) until every member supports cooperative, then upgrades.
partition.assignment.strategy=org.apache.kafka.clients.consumer.\
CooperativeStickyAssignor,org.apache.kafka.clients.consumer.RangeAssignor

# Step 2: after EVERY member is on step-1 config, deploy again with
# only the cooperative assignor. Now the whole group is cooperative.
partition.assignment.strategy=org.apache.kafka.clients.consumer.\
CooperativeStickyAssignor
```

```java
// The same upgrade in the Java consumer config, step 2 (final state).
Properties props = new Properties();
props.put(ConsumerConfig.GROUP_ID_CONFIG, "orders-enrich");
props.put(
    ConsumerConfig.PARTITION_ASSIGNMENT_STRATEGY_CONFIG,
    "org.apache.kafka.clients.consumer.CooperativeStickyAssignor");
// Cooperative revocation means onPartitionsRevoked is called with only
// the partitions actually being taken away, not the full set. If your
// rebalance listener commits offsets on revoke, it now commits only for
// the moved partitions, which is exactly what you want.
KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
```

The one behavioral change to watch: your `ConsumerRebalanceListener.onPartitionsRevoked` callback now receives only the partitions being revoked (the moving ones), not your entire assignment. If you wrote that callback assuming it always gets the full set — for example to flush all in-flight state — adjust it to handle a partial set. This is a feature, not a bug: you only flush what is actually moving.

### What cooperative-sticky does not fix

It is important to be honest about the limits, because cooperative-sticky is sometimes oversold as *the* fix for storms when it is really a fix for the *cost* of storms. It does not change *why* a member gets evicted. If your processing overruns `max.poll.interval.ms`, you still get evicted under the cooperative protocol exactly as under the eager one — the slow batch is still slow, the deadline still fires, the member still leaves. What cooperative-sticky changes is what happens *to the rest of the group* when that one member leaves: under the eager protocol, the whole group revokes everything and freezes; under cooperative, the survivors keep processing their partitions and only the departed member's partitions get reassigned. So cooperative-sticky converts a single member's slow-batch eviction from a group-wide outage into a localized handoff. That is enormously valuable — it is the difference between one bad partition and a dead pipeline — but if *every* member is overrunning the interval (as in the 20-consumer worked example, where all of them have the same slow downstream), cooperative-sticky alone will not save you, because then it is not one member moving, it is all of them churning. For the all-members case you still need the trigger fix: smaller batches or decoupled processing. Cooperative-sticky is necessary but rarely sufficient on its own; it is the partner of a trigger fix, not a replacement for one.

There is also a sticky-assignment benefit that compounds with stateful consumers. If your consumer maintains per-partition local state — a windowed aggregation, a cache keyed by partition, an open file handle, a database connection scoped to a partition's keys — then *moving* a partition is expensive beyond the rebalance pause itself: the new owner has to rebuild that state from scratch. The eager `RangeAssignor` shuffles partitions around freely on every rebalance, so even a member that survives the rebalance can lose and regain different partitions, throwing away warm state. The sticky assignor's stickiness explicitly minimizes such moves, so warm state stays warm. For a stateful streaming consumer, that state-rebuild cost can dwarf the rebalance-protocol cost, and stickiness attacks it directly. This is another reason the cooperative-*sticky* assignor is the right default rather than a plain cooperative round-robin: you want both the incremental revocation *and* the minimal movement.

## Fix 2: static membership

Cooperative rebalancing shrinks the cost of each rebalance. Static membership attacks a different lever: it *eliminates* a whole class of rebalances entirely, specifically the ones caused by routine, expected restarts — deploys, rolling upgrades, pod reschedules, node drains. For a service that deploys several times a day across dozens of pods, this is often the single biggest reduction in rebalance count.

### What static membership changes

Normally, when a consumer leaves the group (process exits, deploy, crash) and then a new consumer joins (the replacement starts), the coordinator sees a leave followed by a join — two membership changes, two rebalances — and the new consumer gets a fresh, randomly generated member id. The coordinator has no way to know that "the consumer that just left" and "the consumer that just joined" are *the same logical worker* coming back after a five-second restart. So it dutifully rebalances on the leave (reassigning the departed member's partitions to survivors) and again on the join (taking some partitions back to give to the newcomer).

Static membership tells the coordinator that identity. You assign each consumer a stable `group.instance.id` — a string that is *the same across restarts* for a given logical worker (e.g. derived from the pod's stable ordinal in a StatefulSet, or the worker slot number). When a static member leaves, the coordinator does *not* immediately rebalance; instead it remembers the member's `group.instance.id` and its partition assignment, and waits for up to `session.timeout.ms` for a consumer with that same instance id to rejoin. If the replacement comes back within that window — which a routine restart does, in seconds — it is recognized as the *same* member, handed back its *exact same* partitions, and *no rebalance happens at all*. The group never even notices the worker bounced.

```java
// Static membership: a stable group.instance.id per logical worker,
// plus a session timeout long enough to cover your restart time.
Properties props = new Properties();
props.put(ConsumerConfig.GROUP_ID_CONFIG, "orders-enrich");
// The critical line. Must be STABLE across restarts and UNIQUE per
// worker. In Kubernetes a StatefulSet ordinal works well:
//   group.instance.id = orders-enrich-${HOSTNAME##*-}
props.put(
    ConsumerConfig.GROUP_INSTANCE_ID_CONFIG,
    System.getenv("POD_ORDINAL_INSTANCE_ID"));
// Session timeout must exceed your worst-case restart-to-rejoin time so
// the coordinator waits instead of rebalancing. 45s covers most rolls.
props.put(ConsumerConfig.SESSION_TIMEOUT_MS_CONFIG, "45000");
props.put(ConsumerConfig.HEARTBEAT_INTERVAL_MS_CONFIG, "15000");
KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
```

The key tuning relationship: `session.timeout.ms` must be *longer than your restart time*. If a pod takes 8 seconds to terminate, get rescheduled, and rejoin, a 45-second session timeout gives you ample margin to come back as the same static member before the coordinator gives up and rebalances. If your restart takes 60 seconds (slow image pull, long JVM warmup) and your session timeout is 45 seconds, the coordinator gives up at 45 seconds, rebalances, and then rebalances *again* when you finally rejoin — worse than not having static membership at all. Size the session timeout to your real restart distribution, p99, with headroom.

#### Worked example: rolling restart of 10 consumers, with and without static membership

Consider a group of 10 consumers over 40 partitions (4 partitions each), and you do a rolling restart for a deploy — one pod at a time, each taking about 12 seconds to terminate, reschedule, and rejoin. *Without* static membership: each pod's termination is a leave (1 rebalance, the survivors absorb its 4 partitions), and each pod's restart is a join (1 rebalance, partitions redistributed to include the newcomer). Ten pods, two rebalances each, is 20 rebalances for one deploy. Under the eager protocol, that is 20 stop-the-world pauses of the entire group; even at a modest 3 seconds each, that is a full minute of cumulative group-wide freeze, plus all the offset-commit churn and re-fetch overhead, every single deploy. Deploy ten times a day and you have ten minutes of self-inflicted daily outage from deploys alone — and if any pod is slow to come back and trips the storm threshold, the deploy itself can *kick off* a storm.

*With* static membership and a session timeout (45s) comfortably larger than the 12-second restart: each pod leaves, the coordinator notes its `group.instance.id` and holds its 4 partitions in reserve, the pod comes back within 12 seconds as the same static member, and is handed its exact 4 partitions back. Zero rebalances. The entire rolling deploy of all 10 pods triggers *no* reassignment whatsoever, because at no point did the coordinator believe the membership had actually changed. The group keeps processing throughout the deploy. The difference is 20 rebalances versus 0, per deploy. Combine static membership (eliminate the routine-restart rebalances) with cooperative-sticky (shrink the cost of the rebalances you cannot avoid) and you have removed both the frequency and the amplitude of the storm's two biggest contributors.

One caveat worth stating: static membership means a *genuinely* dead member (one that will not come back) is not noticed until `session.timeout.ms` elapses. So there is a deliberate tradeoff — you trade faster failure detection for fewer spurious rebalances. For a deploy-heavy workload, that trade is almost always correct, because real deaths are rare and deploys are constant. Just keep the session timeout sane (tens of seconds, not minutes) so a truly crashed worker's partitions are not stranded for too long.

## Fix 3: tune the poll loop and decouple processing

The first two fixes change how rebalances *behave*. This one attacks the most common *trigger* head-on: processing that overruns `max.poll.interval.ms`. There are two levels to it. The tactical level is tuning the poll-loop config so a batch fits inside the interval. The structural level is getting the heavy processing off the poll thread entirely so the interval is never the binding constraint again. Figure 8 contrasts the two structures.

![Before-and-after comparison of doing heavy work on the poll thread, where poll is followed by three hundred twenty seconds of work and no poll for over five minutes leading to eviction, versus offloading to a worker pool, where poll hands off in under a second to eight async workers with pause and resume for backpressure](/imgs/blogs/kafka-rebalance-storms-and-how-to-tame-them-8.webp)

### Tactical: size the batch against the interval

The arithmetic is the one from the first worked example. Your batch processing time is roughly `max.poll.records × per-record-time`, and it must be comfortably less than `max.poll.interval.ms`. You have two knobs:

- **Lower `max.poll.records`** so fewer records come back per poll and each batch finishes faster. If each record takes 640 ms and you need batches under 250 seconds with margin, cap at `max.poll.records=350` (350 × 0.64 = 224 s). This is the cleanest tactical fix because it bounds the worst case directly.
- **Raise `max.poll.interval.ms`** so the deadline accommodates your batch with headroom. Push it to 600000 (10 minutes) if your processing legitimately needs that long. The cost: a *genuinely* stuck consumer now takes up to 10 minutes to be detected and reassigned, so do not set it absurdly high.

```java
// Tactical poll-loop tuning to keep a batch inside the interval.
Properties props = new Properties();
// Fetch fewer records per poll so the batch finishes well under the
// interval. 350 records x 640ms ~= 224s, comfortably under 300s.
props.put(ConsumerConfig.MAX_POLL_RECORDS_CONFIG, "350");
// Give processing more headroom; 10 minutes covers occasional slow
// downstreams without making a truly stuck consumer invisible for long.
props.put(ConsumerConfig.MAX_POLL_INTERVAL_MS_CONFIG, "600000");
// Keep the heartbeat clock independent and generous for GC.
props.put(ConsumerConfig.SESSION_TIMEOUT_MS_CONFIG, "45000");
props.put(ConsumerConfig.HEARTBEAT_INTERVAL_MS_CONFIG, "15000");
```

The trap is treating these as a permanent fix. They buy headroom, but they are fragile: the day the downstream service slows from 640 ms to 1.2 s, your 350-record batch jumps from 224 s to 420 s and blows the 300-second interval again — or even the 600-second one. Tuning the batch tracks a moving target. The durable fix decouples the batch size from the processing time entirely.

### Structural: decouple fetch from process

The real cure is to stop doing heavy processing on the poll thread. The poll thread's job becomes narrow: call `poll()`, hand the records off to a worker pool, and immediately loop back to `poll()` again. Because the hand-off is fast (it is just enqueuing references), `poll()` is called frequently regardless of how long the actual processing takes, so the `max.poll.interval.ms` clock is never the binding constraint. The processing runs asynchronously on N worker threads, giving you real parallelism within a single consumer and decoupling your throughput from the single-threaded poll loop.

The subtlety is flow control and offset commits. If you let the poll thread race ahead, fetching faster than the workers can drain, you will accumulate an unbounded backlog in memory and OOM. So you apply backpressure: when the in-flight work exceeds a threshold, you call `consumer.pause(partitions)` to stop fetching, and `consumer.resume(partitions)` when the workers catch up. Critically, you keep calling `poll()` even while paused — a paused `poll()` returns no records but still sends heartbeats and resets the interval clock, keeping you in the group. And you only commit an offset once all records up to that offset have *actually finished processing* on the workers, tracked carefully so you never commit ahead of completed work (which would lose messages on a crash). This pattern is involved enough that you should lean on a library — the Confluent Parallel Consumer is the canonical implementation — rather than hand-rolling the offset-tracking and pause/resume logic, which is easy to get subtly wrong.

```java
// Decoupled structure: poll thread enqueues, worker pool processes,
// poll thread pauses fetching under backpressure but KEEPS polling
// so the max.poll.interval clock never expires.
ExecutorService workers = Executors.newFixedThreadPool(8);
Semaphore inFlight = new Semaphore(2000); // cap outstanding work

while (running) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(200));
    for (ConsumerRecord<String, String> r : records) {
        inFlight.acquire();                  // backpressure: block fetch loop
        workers.submit(() -> {
            try {
                process(r);                  // the slow 640ms work, off poll thread
                offsetTracker.markDone(r);    // record completion for safe commit
            } finally {
                inFlight.release();
            }
        });
    }
    // Pause partitions whose in-flight backlog is too deep; keep polling
    // (a paused poll still heartbeats and resets the interval clock).
    backpressureController.adjustPauseResume(consumer);
    consumer.commitSync(offsetTracker.committableOffsets()); // only completed work
}
```

With this structure, `max.poll.interval.ms` stops being a storm trigger because the poll thread *always* returns to `poll()` within a couple hundred milliseconds, no matter how slow the downstream is. You have converted a fragile timing relationship (batch time must fit inside the interval) into a robust one (poll is always fast; processing is bounded by a separate concurrency limit). This is the same decoupling described in [consumer optimization and scaling](/blog/software-development/message-queue/consumer-optimization-and-scaling), here applied specifically as the structural cure for slow-processing storms. The matrix in Figure 4 summarizes which fix addresses which trigger, so you can see why a durable solution usually combines two or three of them rather than relying on any single one.

![Matrix mapping the four fixes of cooperative-sticky, static membership, poll-loop tuning, and decoupled processing against the four triggers of slow processing, restart churn, GC or timeout eviction, and autoscaling flap, showing that each fix is strong for a different trigger so combinations are usually required](/imgs/blogs/kafka-rebalance-storms-and-how-to-tame-them-4.webp)

## A rebalance-storm runbook

When the pager goes off and lag is climbing while the consumers look up, you do not want to be reasoning from first principles. You want a checklist. Here is the runbook, distilled from everything above into the order you should execute it during an incident. Figure 9 shows the group-state transition you are driving: from a churning group with zero commits to a stable group with steadily falling lag.

![Grid contrasting a churning group with members joining, all partitions revoked every thirty seconds, and zero offsets committed, against a stable group with the assignment held, commits every five seconds, and lag falling steadily](/imgs/blogs/kafka-rebalance-storms-and-how-to-tame-them-9.webp)

### Triage (first 5 minutes)

1. **Confirm it is a storm, not just slow consumers.** Check the `rebalance-rate-per-hour` metric or grep the consumer log for join/revoke cadence. If the group is rebalancing every few seconds, it is a storm. If it is rebalancing rarely but lag still climbs, it is a plain capacity or slow-consumer problem — different incident, see [debugging consumer lag spikes](/blog/software-development/message-queue/debugging-consumer-lag-spikes-field-guide).
2. **Do NOT scale out.** Resist the instinct. Adding consumers adds join events and amplifies the storm. Freeze the autoscaler if one is attached (`kubectl scale` to a fixed replica count, or pause the HPA).
3. **Open the coordinator and consumer logs and read the `reason` field** on the rebalance lines. This single field — `heartbeat expiration` vs `poll timeout` vs a specific member crash-looping — classifies the cause and dictates the fix.

### Stabilize (next 15 minutes)

4. **If the reason is poll timeout (slow processing):** immediately lower `max.poll.records` and/or raise `max.poll.interval.ms` and deploy. Cutting `max.poll.records` in half is the fastest tourniquet — it halves the batch time and usually drops it back under the interval. This is a config-only change; you can often push it via a rolling config update without a full rebuild.
5. **If the reason is heartbeat expiration (GC/session):** raise `session.timeout.ms` above your worst-case GC pause and set `heartbeat.interval.ms` to one-third of it. Then look at the GC logs — you have a heap problem to fix after the fire is out.
6. **If one member is crash-looping:** identify the bad pod (its member id keeps appearing, its restart count climbs), and either cordon it or fix the crash. If it is a poison message, that consumer needs a dead-letter path so one bad record cannot take it down.
7. **If it correlates with autoscaler events:** pin the replica count to a stable number and stop the flapping. Re-key the autoscaler on lag rather than CPU before re-enabling it.

### Harden (after the incident)

8. **Adopt static membership** so routine restarts and deploys stop triggering rebalances. Assign a stable `group.instance.id` per worker and size `session.timeout.ms` above your p99 restart time.
9. **Adopt the cooperative-sticky assignor** via the two-step rolling upgrade so the rebalances you cannot avoid stop being stop-the-world.
10. **Decouple processing from the poll thread** with a worker pool and pause/resume backpressure (or a parallel-consumer library) so `max.poll.interval.ms` is never the binding constraint again.
11. **Alert on rebalance rate directly**, not just on lag, so the next storm pages you on its cause minutes before it threatens the retention cliff.

This sequence — triage to find the cause, stabilize with the fastest matching tourniquet, then harden so it cannot recur — is the difference between a storm that costs you four hours of climbing lag and one that costs you four minutes.

## Case studies

### The enrichment service that scaled into the ground

A team ran a 24-consumer group enriching order events, each event making a synchronous lookup against an inventory service. Under normal load the inventory service answered in 180 ms, so a 500-record batch processed in 90 seconds, well under the 300-second `max.poll.interval.ms`. One afternoon a marketing push tripled order volume, the inventory service's p50 latency climbed to 700 ms under the load, and suddenly each 500-record batch took 350 seconds — over the interval. The group started storming. The on-call engineer, watching lag climb, did the natural thing: scaled the group from 24 to 48 consumers. Lag climbed faster. The extra consumers each triggered a join rebalance, doubled the eager stop-the-world pauses, and inherited the same 700-ms-per-record slowness, so they too overran the interval and got evicted. The group spent the next forty minutes in continuous rebalance, committing nothing, while lag passed eight million messages. The fix, once they read the consumer log and saw the `poll timeout has expired` message, was embarrassingly small: drop `max.poll.records` from 500 to 250 (batch time back to 175 seconds, under the interval) and stop scaling. The group recovered in minutes. The lesson: a downstream slowdown can push a previously-healthy batch over the interval, and the storm that follows looks like a capacity problem but is actually a timing problem — and scaling out makes a timing problem strictly worse.

### The 10-second session timeout and the garbage collector

A streaming-analytics group inherited a config from an old tutorial: `session.timeout.ms=10000`. The consumers ran a heavy aggregation that allocated aggressively, and on the default parallel collector the JVM did periodic full-GC pauses of 8 to 13 seconds. Most of the time the pauses were just under 10 seconds and nothing happened. But under load, pauses crept past 10 seconds, the heartbeat thread froze through the pause, the coordinator missed heartbeats, and members got evicted on `heartbeat expiration` — triggering rebalances that added even more pressure as members re-fetched and re-warmed their aggregation state, which caused *more* allocation and *more* GC. A self-reinforcing storm with no slow-processing component at all; `max.poll.interval.ms` was never the issue. They chased it for two hours looking at poll tuning before someone correlated the eviction timestamps with the GC log and saw the pauses lining up exactly with the evictions. The fix was two changes: raise `session.timeout.ms` to 45 seconds (above the worst GC pause) and switch the collector to G1 with a smaller pause target. Rebalances dropped to zero. The lesson: a session timeout shorter than your GC pauses turns the garbage collector into a storm generator, and the cause is invisible unless you cross-reference eviction times against GC logs.

### The deploy that became a storm

A payments team deployed their consumer group — 16 pods, no static membership, eager assignor — multiple times a day via a rolling restart. Each deploy caused roughly 32 rebalances (a leave and a join per pod), each a stop-the-world pause. Usually the group absorbed it: 32 pauses of 2 to 3 seconds each was annoying but survivable, finishing in a couple of minutes. Then one day a slow container image pull made several pods take 50 seconds to come back, longer than the rebalance machinery expected. The group rebalanced on each leave, sat partially-assigned waiting for slow rejoins, rebalanced again on each delayed join, and the overlapping reassignments cascaded into a sustained storm that outlasted the deploy by twenty minutes. The fix that ended this class of incident permanently was static membership: assign each pod a stable `group.instance.id` from its StatefulSet ordinal and set `session.timeout.ms` to 60 seconds (above the worst-case 50-second slow pull). After that, a rolling deploy triggered *zero* rebalances — the coordinator recognized each restarting pod as the same static member and handed back its partitions. Deploys went from "32 rebalances and a chance of a storm" to "no rebalances at all." The lesson: deploy-time rebalances are pure self-inflicted churn, and static membership removes them entirely, which also removes the risk that a slow deploy tips into a storm.

### The flapping autoscaler

An events-ingestion group ran on a CPU-keyed horizontal pod autoscaler. The workload was bursty — quiet for minutes, then a flood. When a flood hit, CPU spiked, the autoscaler added 6 pods; the flood passed, CPU dropped, the autoscaler removed 6 pods; another flood, 6 pods back. Each scale event was 6 join or 6 leave rebalances. The autoscaler flapped every few minutes, and the group never settled — a continuous low-grade storm driven entirely by membership churn, with every individual consumer perfectly healthy and fast. Worse, the rebalances themselves consumed CPU (re-fetching, re-warming), which the autoscaler read as "still need more capacity," feeding the loop. The team broke it by re-keying the autoscaler on consumer lag instead of CPU (lag is a stable, slowly-changing signal that reflects actual backlog), adding a 5-minute stabilization window so the autoscaler ignored transient bursts, and capping scale velocity to one pod per minute. Membership churn dropped from constant to rare, and the storm stopped. The lesson: scaling on instantaneous CPU makes membership reflexive, and reflexive membership is a storm; scale on lag, slowly.

## When to reach for each fix (and when not to)

These fixes are not mutually exclusive — the durable answer usually layers several — but they address different triggers, and reaching for the wrong one wastes the window before the retention cliff. Here is the decision logic.

| Symptom in the logs | Most likely cause | First fix | Durable fix |
| --- | --- | --- | --- |
| `poll timeout has expired` | Slow processing overruns interval | Lower `max.poll.records` | Decouple processing to a worker pool |
| `heartbeat expiration` + long GC | Session timeout below GC pause | Raise `session.timeout.ms` | Fix GC; right-size heap |
| One member id evicted repeatedly | A consumer crash-looping | Cordon / fix that pod | DLQ for poison messages |
| Rebalances track deploy events | Routine restart churn | Static membership | Static membership (it is the cure) |
| Rebalances track autoscaler events | Membership flapping | Pin replica count | Lag-based, stabilized autoscaling |

**Always adopt cooperative-sticky and static membership** unless you have a specific reason not to — they are close to pure wins for any group that deploys regularly, and together they remove both the frequency (static membership eliminates deploy rebalances) and the amplitude (cooperative-sticky de-amplifies the rest). The main reason *not* to use static membership is if your restart time is genuinely long and unpredictable and you cannot size a session timeout to cover it without making real failures slow to detect — but that is rare, and usually a sign you should fix the slow restart.

**Reach for poll-loop tuning as a tourniquet, not a cure.** Lowering `max.poll.records` and raising `max.poll.interval.ms` stops a slow-processing storm fast, and you should absolutely do it during an incident. But it tracks a moving target — the day your downstream slows, the math breaks again. Treat it as buying time to do the structural fix (decoupling).

**Decouple processing when slow processing is your recurring storm cause.** If you keep getting poll-timeout storms despite tuning, the processing genuinely needs to come off the poll thread. This is the most invasive change (it restructures your consumer) so it is the right call when the simpler fixes keep failing, not the first thing you reach for at 3 a.m. — though a parallel-consumer library makes it far less invasive than hand-rolling.

**Do not reach for "more consumers" during a storm, ever.** Scaling out is the correct fix for a *capacity* problem (lag climbing with rare rebalances), and the wrong fix for a *storm* (lag climbing with constant rebalances). The first triage step exists precisely to tell these apart, because the fixes are opposite. If you are not sure which you have, check the rebalance rate before you touch the replica count.

## Key takeaways

- **A rebalance storm is a group that rebalances continuously and makes no progress** while every individual consumer looks healthy. The defining signals are a high rebalance rate, climbing lag, and `RebalanceInProgress` errors on commit — none of which appear on a default "is the service up" dashboard.
- **The classic cause is processing that overruns `max.poll.interval.ms`.** Each slow batch gets the consumer evicted mid-batch; the reassigned owner re-reads the same uncommitted batch and overruns again. The same work is redone every round, so offsets never advance and lag grows unbounded.
- **There are two independent liveness clocks.** `max.poll.interval.ms` checks that you call `poll()` often enough (progress); `session.timeout.ms` checks that heartbeats arrive (liveness, can be broken by GC). They are tuned separately and have different eviction reasons in the logs — read the `reason` field.
- **Other triggers are crash loops** (one pod cycling rebalances the group on every restart), **and autoscaling churn** (a flapping autoscaler turns every scale event into a rebalance). Both present as storms but are fixed outside Kafka config.
- **A storm is effectively an outage**, not a slowdown: eager rebalancing pauses the whole group, committed offsets stop entirely, and if the storm outlasts retention it becomes permanent data loss. Page on it; do not ticket it.
- **Diagnose by reading the coordinator and consumer logs**, which name the triggering member and the reason for each rebalance. The reason field — poll timeout vs heartbeat expiration vs a crash-looping member — classifies the cause and selects the fix.
- **The cooperative-sticky assignor** replaces stop-the-world reassignment with incremental reassignment, so only the partitions that actually move pause — shrinking each rebalance's blast radius from the whole group to a handful of partitions. Roll it out in two steps.
- **Static membership eliminates deploy and routine-restart rebalances** by giving each worker a stable `group.instance.id` and letting the coordinator wait for it to rejoin instead of rebalancing. Size `session.timeout.ms` above your p99 restart time. A rolling restart goes from N rebalances to zero.
- **Tuning the poll loop is a tourniquet; decoupling is the cure.** Lowering `max.poll.records` or raising the interval stops a slow-processing storm fast, but the durable fix moves heavy processing off the poll thread to a worker pool with pause/resume backpressure, so the interval is never the binding constraint again.
- **Never scale out during a storm.** Adding consumers adds membership churn and amplifies the storm; it is the correct fix for a capacity problem and the wrong fix for a storm. Tell the two apart by the rebalance rate before touching the replica count.

## Further reading

- [Kafka consumer groups, offsets, and rebalancing](/blog/software-development/message-queue/kafka-consumer-groups-offsets-rebalancing) — the mechanics of the coordinator, generations, and assignors that this post operationalizes; read it first if the rebalance protocol is unfamiliar.
- [Consumer optimization and scaling](/blog/software-development/message-queue/consumer-optimization-and-scaling) — prefetch, concurrency, and the decouple-fetch-from-process pattern that is the structural cure for slow-processing storms.
- [Debugging consumer lag spikes](/blog/software-development/message-queue/debugging-consumer-lag-spikes-field-guide) — the differential diagnosis for lag that climbs *without* a storm, the sibling incident to this one.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — the partitioned-log foundation that makes consumer groups and rebalancing necessary in the first place.
- [Apache Kafka documentation: consumer configs](https://kafka.apache.org/documentation/#consumerconfigs) — the authoritative reference for `max.poll.interval.ms`, `session.timeout.ms`, `group.instance.id`, and the assignment strategies.
- [KIP-429: Incremental Cooperative Rebalancing](https://cwiki.apache.org/confluence/display/KAFKA/KIP-429%3A+Kafka+Consumer+Incremental+Rebalance+Protocol) — the design doc for the cooperative-sticky assignor and why incremental rebalancing exists.
- [KIP-345: Static Membership](https://cwiki.apache.org/confluence/display/KAFKA/KIP-345%3A+Introduce+static+membership+protocol+to+reduce+consumer+rebalances) — the design doc for `group.instance.id` and why static membership avoids restart rebalances.
- [Confluent Parallel Consumer](https://github.com/confluentinc/parallel-consumer) — a production library that decouples processing from the poll thread with safe offset tracking, so you do not hand-roll the pause/resume and commit logic.
