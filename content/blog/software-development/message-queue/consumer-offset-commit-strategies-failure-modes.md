---
title: "Consumer Offset Management: Commit Strategies and Their Failure Modes"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "The offset commit is where at-least-once and at-most-once are actually decided. This is a deep dive into auto-commit's twin hazards, commitSync versus commitAsync, why commit ordering picks your failure mode, committing on rebalance, batch cadence, storing offsets in the same transaction as the result for effectively-once, and the commit failures that surface in production."
tags:
  [
    "message-queue",
    "kafka",
    "offsets",
    "consumers",
    "delivery-semantics",
    "distributed-systems",
    "event-driven",
    "idempotency",
    "reliability",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/consumer-offset-commit-strategies-failure-modes-1.webp"
---

Almost every duplicate-message bug and almost every silently-lost-message bug I have ever chased in a Kafka consumer traced back to the same three lines of code: where, when, and how the consumer committed its offset. Not the broker. Not replication. Not the network. The commit. It is the least glamorous part of the whole pipeline — a single monotonically increasing integer written to a topic you never look at — and it is precisely where the delivery guarantee of your entire system is decided. You can run `acks=all` with three replicas and a perfectly durable log, and still lose messages or process them twice, because durability on the broker and correctness on the consumer are two different problems and the commit is the seam between them.

Here is the sentence that holds the whole post together: **a committed offset is a promise that says "I have processed everything up to here, and you never need to send it to me again."** Everything that goes wrong with offsets goes wrong because that promise was made at the wrong moment — either too early, before the processing actually happened (and now a crash skips the work, which is loss), or the promise was *true* but never durably recorded (and now a crash replays work that was already done, which is duplication). At-least-once versus at-most-once is not a broker setting you flip. It is an emergent consequence of when your code calls commit relative to when your code finishes processing. You pick your failure mode by ordering two statements.

![A matrix comparing auto-commit, commitSync, commitAsync, and external transactional storage across duplicate risk, loss risk, and latency](/imgs/blogs/consumer-offset-commit-strategies-failure-modes-1.webp)

By the end of this post you will be able to look at any consumer loop and tell me, before you run it, whether a crash at any given instant loses a message, duplicates one, or neither — and exactly how many. You will know why the convenient `enable.auto.commit=true` default opens *two* opposite hazards at once, why `commitSync` and `commitAsync` are not interchangeable, why the safest production loop uses both, where to commit so a rebalance does not replay a whole batch, and how to store the offset in the same transaction as your result so that "effectively-once" stops being a slogan and becomes a property you can prove. This post goes much deeper than the auto-versus-manual sketch in [Kafka consumer groups, offsets, and rebalancing](/blog/software-development/message-queue/kafka-consumer-groups-offsets-rebalancing); think of that post as introducing the offset and this one as dissecting the commit. It is also the consumer-side companion to [delivery semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once), which establishes *what* the guarantees are; here we establish *how the commit produces them*.

## 1. What a committed offset actually means

Before any of the strategies make sense, you have to be ruthlessly precise about what a commit is and — just as importantly — what it is not. The vagueness here is the source of more confusion than any other single thing in Kafka consumption.

A partition is an ordered log of records, each addressed by an **offset**: 0, 1, 2, 3, and so on, increasing forever. A consumer reading that partition has a **position** — the offset of the *next* record it will fetch. That position lives only in the consumer's memory and advances every time `poll()` hands back records. If the consumer crashed right now and restarted, that in-memory position would be gone. It is not durable. It is a guess about where to resume, and it is wrong the moment the process dies.

The **committed offset** is the durable answer to the question "where should a *new* consumer resume?" It is a number the consumer explicitly writes to durable storage — by default, an internal Kafka topic called `__consumer_offsets`, keyed by the tuple of group, topic, and partition. When a consumer starts, or when a partition is reassigned during a rebalance, the new owner reads the committed offset for that partition and resumes from it. If there is no committed offset at all, the consumer falls back to `auto.offset.reset` (typically `earliest` or `latest`). So the committed offset is the *only* thing that survives a crash or a rebalance. The in-memory position does not.

### The position is optimistic; the committed offset is conservative

There is a useful framing for the gap between these two numbers. The in-memory position is **optimistic** — it advances the instant `poll()` hands you records, on the assumption you will process them, before you actually have. The committed offset, when you manage it correctly, is **conservative** — it advances only after work is genuinely done and durably recorded. Every offset bug is, at bottom, a confusion between these two: code that commits the optimistic position (auto-commit, or a manual commit of the client's position) claims work is done that has only been *fetched*. The discipline that makes a consumer correct is refusing to ever commit the optimistic number — always committing your own tracked, conservative "I finished through here" number instead. Hold that distinction and most of this post is a corollary.

### The off-by-one that bites everyone

Here is the single most important detail about a committed offset, and it is the one people get wrong constantly: **you commit the offset of the next record you want, not the last record you processed.** If you have fully processed records at offsets 100 through 149, the offset you commit is **150**, not 149. The committed offset is a "resume here" pointer, and you want to resume at 150 because 149 is done. Commit 149 and on restart you will reprocess record 149 — a guaranteed duplicate every single restart.

The Kafka Java client hides this for you when you call the no-argument `commitSync()`, which commits "the last offset returned by poll, plus one" for every partition. But the moment you commit *specific* offsets — which you must do for any non-trivial strategy — you own the plus-one. The API even names it for you: the `OffsetAndMetadata` you pass holds the offset to commit, and the convention in every Kafka tutorial is `new OffsetAndMetadata(record.offset() + 1)`. Forget the `+ 1` and you have built a duplicate generator that fires on every restart and every rebalance. I have seen this exact bug ship to production and manifest as "we double-process the first record after every deploy" — which is precisely what omitting the plus-one does, because every deploy restarts the consumer.

### A commit is a claim about processing, not about delivery

The deepest point in this whole section: **a commit is a statement about your processing, but Kafka has no idea what your processing is.** The broker cannot see whether you wrote to a database, called a payment gateway, or did nothing at all. It only sees the number you sent. So when you commit offset 150, you are *asserting* to the world that records 100–149 are handled and must never be redelivered. Kafka takes you at your word. If that assertion is false — if you committed 150 before actually processing 140–149 — Kafka will not catch the lie. It will simply never send you 140–149 again, and those records are gone. This is why the commit is where loss happens: a commit is a promise only your code can keep, made to a broker that cannot verify it.

That is the entire foundation. The strategies below are all just different answers to one question — *at what moment, and how durably, do I make that promise?* — and the failure modes are all just the consequences of making it at the wrong moment.

### Where `__consumer_offsets` actually stores the number

It is worth knowing what the default storage actually is, because its properties explain several behaviors that otherwise look like magic. `__consumer_offsets` is a normal Kafka topic — fifty partitions by default — that the brokers create automatically the first time a group commits. It is **log-compacted**, which means Kafka keeps only the *latest* value for each key and garbage-collects older ones. The key is `(group.id, topic, partition)` and the value is the committed offset plus optional metadata. Because it is compacted, the committed-offset topic does not grow without bound even though every group commits constantly; old offset values for a key are compacted away, leaving only the current position. When a consumer needs to find its committed offset, the **group coordinator** — a specific broker chosen by hashing the `group.id` into one of those fifty partitions — reads the latest value for the key and hands it back.

Two consequences fall out of this design. First, commits are *cheap* — they are just a produce to a Kafka topic, the same fast append path everything else in Kafka uses, which is why even a commit-per-record is survivable (slow, but not catastrophic). Second, the committed offset is exactly as durable as any other Kafka write: replicated according to the topic's replication factor, acknowledged by the in-sync replicas. So when `commitSync()` returns, your offset has the same durability guarantee as a produced message with `acks=all` — which is precisely the guarantee you want for a "resume here" pointer. The offset storage is not a database with transactions across your data and your offset; it is a separate replicated log, and that *separateness* is exactly the two-store problem section 7 exists to solve.

## 2. Auto-commit and its two hazards

The path of least resistance is `enable.auto.commit=true`, which is the Kafka consumer default. With it on, the client commits offsets for you automatically, on a timer governed by `auto.commit.interval.ms` (default 5000 ms — five seconds). You write a `poll()` loop, process records, and never think about offsets. It feels like a feature. It is a trap, and the trap has two jaws that close from opposite directions.

The crucial mechanical detail is *when* the auto-commit actually fires. It does not fire continuously. The client commits the current position during a `poll()` call, but only if at least `auto.commit.interval.ms` has elapsed since the last commit. So the committed offset is the position **as of the most recent poll that crossed a five-second boundary.** It commits whatever has been *fetched and handed to you*, on a timer, with zero knowledge of whether you have *finished processing* what was handed to you. That gap — between "handed to you by poll" and "finished processing" — is where both hazards live.

![A before-and-after figure showing commit-before-process leading to at-most-once loss versus commit-after-process leading to at-least-once duplication](/imgs/blogs/consumer-offset-commit-strategies-failure-modes-2.webp)

### Hazard one: the duplicate window (commit lags processing)

Suppose your processing is fast and your batches are small, so by the time the five-second auto-commit fires you have genuinely finished everything it is about to commit. Now you crash one second after processing a batch but four seconds before the next auto-commit. On restart, the committed offset reflects the *previous* commit, five seconds stale. Everything you processed in those last four seconds gets redelivered, because the commit that would have recorded it never fired. This is the **duplicate hazard**: the commit lags behind your processing, so a crash replays the work done since the last timer tick. It is bounded — you can lose at most `auto.commit.interval.ms` worth of progress — but at a few thousand messages per second, five seconds is thousands of duplicates per crash.

### Hazard two: the loss window (commit races ahead of processing)

Now flip it. Suppose your processing is *slow* — each record takes real work, a database write, an external call. You `poll()`, get 500 records, and start processing them one by one. Five seconds in, you are only on record 200 of the 500. The next `poll()` call (which you must make to stay in the group — more on that) triggers an auto-commit, and the auto-commit commits the position of the **last poll**, which already advanced past all 500 records the moment poll returned them. Kafka now believes you have processed all 500. You have processed 200. You crash. On restart, the committed offset is past record 500, and records 201–500 are **never redelivered**. They are gone. This is the **loss hazard**: the auto-commit committed the *fetched* position, not the *processed* position, and the two diverged because processing fell behind the fetch.

This second hazard is the genuinely dangerous one, because it is silent. There is no error, no dead-letter, no exception. The records simply vanish, and you discover it weeks later when a customer's order is nowhere in the database. The [delivery semantics post](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) calls this "accidental at-most-once," and it is the single most common way teams ship at-most-once while believing they shipped reliable Kafka. They never *chose* to tolerate loss; they inherited it from a default.

### Why you cannot "just not poll" to stop the loss

A natural reaction to the loss hazard is "fine, I will not call `poll()` again until I have finished the current batch, so the timer cannot commit ahead of me." This does not work, and understanding why is instructive. The auto-commit fires *during* a `poll()` call, committing the position established by the *previous* poll. So if you process slowly and delay your next poll, you are not avoiding the bad commit — you are delaying it. And worse, delaying your poll has its own lethal consequence: `max.poll.interval.ms` (the five-minute default we will meet in section 8) bounds how long you may go between polls before the broker evicts you from the group as presumed-dead. So the loss hazard is structural to auto-commit: the timer commits the fetched position, and the only way to delay that commit is to delay polling, which gets you evicted. You cannot tune your way out of it; you can only turn auto-commit off. That is the honest conclusion — auto-commit's loss window is not a misconfiguration, it is the design, and the design is wrong for any data you must not lose.

### The interaction with `auto.offset.reset`

One more auto-commit trap that produces silent skips, even without a crash. If a *brand-new* group subscribes and has no committed offset at all, the consumer falls back to `auto.offset.reset`, whose **default is `latest`** — meaning "start from the end, ignore everything already in the topic." A team spins up a new consumer group to process a topic, expecting to handle the backlog, and instead the consumer jumps to the tail and processes only records produced *after* it started. Every record already in the topic is silently skipped — not by the commit logic, but by the reset policy that fires when there is no committed offset to honor. The fix is to set `auto.offset.reset=earliest` for any group that must process history. It is not strictly a commit issue, but it is in the same family — a default that silently skips data — and it bites the same teams for the same reason: they trusted a default to do the safe thing, and the default optimizes for "don't replay the world," not "don't lose anything."

### The honest summary of auto-commit

So auto-commit is not "a little bit of duplicates" — it is *both* failure modes at once, and which one you get depends on whether your processing is faster or slower than the commit timer, a property that changes with load. Under light load you get the duplicate window; under heavy load, when processing falls behind, you slide into the loss window — exactly when you can least afford it. That is what makes auto-commit treacherous: its failure mode is load-dependent and flips to the worse one precisely under stress. It is fine for a strictly self-correcting stream where neither duplicates nor a few lost samples matter — metrics, telemetry, best-effort cache warming. For anything you would be embarrassed to lose or double, turn it off. Set `enable.auto.commit=false` and own the commit yourself. The rest of this post is what owning it looks like.

```python
# The default you should treat as a loaded gun for business data.
consumer = KafkaConsumer(
    "orders",
    group_id="order-processor",
    enable_auto_commit=True,        # commits on a 5s timer, regardless of processing
    auto_commit_interval_ms=5000,   # the size of BOTH your duplicate and loss windows
)
# Under light load: a crash replays up to 5s of work (duplicates).
# Under heavy load (processing slower than poll): poll advances past records the
# timer then commits as "done" -> a crash skips them entirely (silent loss).
```

## 3. commitSync vs commitAsync

Once you set `enable.auto.commit=false`, you are committing manually, and the Kafka client gives you two ways to do it. They look almost identical and they behave completely differently under failure. Getting them confused is the second-most-common offset bug after the missing plus-one.

`commitSync()` is **blocking and retried**. When you call it, the consumer sends the commit to the group coordinator and *waits* for the broker to acknowledge it. If the commit fails for a retriable reason (a coordinator that is momentarily unavailable, a leader election in flight), the client retries it automatically, up to a configured timeout. When `commitSync()` returns normally, the offset is durably committed — you have a guarantee. If it cannot succeed, it throws, and you know the commit failed. The cost is latency: the call blocks your poll loop for a full round trip to the broker, and under coordinator hiccups it can block for the whole retry window. At high commit frequency, that blocking is real throughput lost.

`commitAsync()` is **non-blocking and not retried**. You call it, it fires the commit request off, and returns immediately — your loop keeps going. You can pass a callback that runs when the broker responds (success or failure), but the client does **not** retry a failed async commit. And here is the subtle, dangerous part: it does not retry *on purpose*, because a retry would be wrong. Suppose you call `commitAsync()` to commit offset 200, it is slow, and meanwhile your loop calls `commitAsync()` again to commit offset 300, which succeeds. If the client then retried the failed offset-200 commit, it would overwrite the committed offset *backward* from 300 to 200 — and on a crash you would reprocess 200–299 needlessly. So async commits are **last-write-wins with no retry**, and that is the only correct behavior for an out-of-order async stream. The price is that any single async commit can silently fail and you will never recover it.

### The standard production pattern uses both

The right loop does not choose between them — it uses each where it shines. Call `commitAsync()` inside the hot loop for speed (a transient failure here is harmless because the *next* successful async commit, a few hundred milliseconds later, carries a higher offset and supersedes it). Then call `commitSync()` exactly once, in a `finally` block, when the consumer is shutting down or losing partitions — because that is the *last* commit, there is no "next one" to supersede a failure, so you want the blocking, retried, guaranteed version. This async-in-the-loop, sync-on-exit pattern is the canonical Kafka consumer shape, and it is worth memorizing.

```java
try {
    while (running) {
        ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(200));
        for (ConsumerRecord<String, String> record : records) {
            process(record);                       // your work
        }
        consumer.commitAsync();                    // fast, fire-and-forget, last-write-wins
    }
} catch (WakeupException e) {
    // expected on shutdown signal
} finally {
    try {
        consumer.commitSync();                     // blocking + retried: the LAST commit must stick
    } finally {
        consumer.close();                          // close() also triggers a final auto-commit if enabled
    }
}
```

| Property | `commitSync()` | `commitAsync()` |
| --- | --- | --- |
| Blocks the loop | Yes, until broker acks | No, returns immediately |
| Retries on failure | Yes, up to timeout | No, never (avoids backward overwrite) |
| Guarantee on return | Offset is committed | Only that the request was sent |
| Failure visibility | Throws synchronously | Callback only, easy to ignore |
| Throughput cost | High at high commit rate | Low |
| Right place to use | Shutdown, rebalance, final commit | Hot loop, frequent commits |

The mental model: `commitSync` is "I need this to be true before I move on." `commitAsync` is "probably fine, and if not the next one fixes it." Use the first when there is no next one.

### The async callback, and what you can and cannot do in it

`commitAsync()` takes an optional callback that fires when the broker responds. People reach for it to "retry on failure," and that instinct is exactly the bug. You must *not* blindly retry inside the callback, for the same backward-overwrite reason: by the time the failure callback runs, a later async commit with a higher offset may already have succeeded, and re-sending the old lower offset would roll your committed position backward. If you genuinely want retry semantics on async commits, the correct pattern is to track a monotonically increasing commit sequence number, and in the callback retry *only if* no newer commit has been issued since — which is fiddly enough that most teams correctly conclude the callback is for *logging and metrics*, not retry. Log the failure, increment a `commit_failures` counter, and rely on the next successful async commit plus the sync-on-shutdown to make progress durable. The callback's honest job is observability, not recovery.

```java
consumer.commitAsync((offsets, exception) -> {
    if (exception != null) {
        // Do NOT blindly retry here — a newer commit may have superseded this one.
        log.warn("async commit failed for {}, relying on next commit", offsets, exception);
        commitFailures.increment();   // a metric you alert on, not a retry trigger
    }
});
```

### Why the final commit must be sync

The asymmetry is worth stating plainly because it is the whole reason the dual pattern exists. In a steady loop, an async commit that fails is *self-healing*: a few hundred milliseconds later the next async commit carries a higher offset and supersedes it, so the failure leaves no trace. The only commit that is *not* self-healing is the **last** one — the commit before you leave the group, because there is no "next commit" coming to fix it. A failed final async commit is a failed commit, full stop, and on restart you replay everything since the previous successful one. That is why the last commit, and only the last commit, must be the blocking-and-retried `commitSync`. The pattern is not "sync is safer so use it everywhere" — sync everywhere wrecks throughput — it is "use sync exactly where a failure cannot be superseded," which is the final commit and the rebalance-revoke commit of the next section.

## 4. Commit timing: before vs after processing (you pick the failure mode)

This is the heart of the entire post, and it is shockingly simple once you see it. Forget auto versus manual, sync versus async, for a moment. There is a more fundamental choice underneath all of them, and it is just the *order of two statements in your code*: do you commit the offset before you process the record, or after?

The figure under section 2 already showed the two columns; now we make them exact. Commit-before-process is **at-most-once**: you tell Kafka "I've handled this" and *then* try to handle it, so if you crash between the commit and the completion of processing, the record is committed-as-done but never actually done — it is lost, and never redelivered. Commit-after-process is **at-least-once**: you handle the record fully and *then* commit, so if you crash between completing the processing and the commit landing, the record is done but not marked done — on restart it is redelivered and reprocessed, a duplicate. There is no third ordering that gives you both safety and uniqueness for free; the impossibility is the whole subject of the [delivery semantics post](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once). You are choosing which bad outcome you can tolerate.

```python
# AT-MOST-ONCE: commit first, then process. A crash in process() loses the record.
for record in consumer:
    consumer.commit()        # "done" — but we haven't done it yet
    process(record)          # crash here => record lost forever, never redelivered

# AT-LEAST-ONCE: process first, then commit. A crash before commit replays the record.
for record in consumer:
    process(record)          # do the real work
    consumer.commit()        # crash here => record reprocessed on restart (duplicate)
```

### Why at-least-once is almost always the right default

For nearly every business-meaningful workload, loss is the worse outcome, so you commit *after* processing and accept that crashes produce duplicates. Then you make the duplicates harmless. That second step is not optional — at-least-once without idempotency is a double-charge generator — and it is the entire subject of [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe). The combination, at-least-once delivery plus an idempotent consumer, is what real systems mean when they say "exactly-once": duplicate *deliveries* that produce no duplicate *effects*. You do not get there by committing more cleverly; you get there by committing after processing and deduping the inevitable replays.

### When at-most-once is the correct, deliberate choice

At-most-once is not always wrong — it is wrong only when chosen by accident. For a stream where each message decays to worthless almost instantly and the source is self-correcting — a live metrics feed, sensor samples, frames from a video stream — losing one is invisible because the next arrives in milliseconds, and *reprocessing* one might actually be worse (a double-counted metric is a lie; a missing one is just noise). For those, commit-before-process is correct, and you should choose it on purpose, in a comment, so the next engineer knows you meant it. The sin is not at-most-once; the sin is at-most-once you did not know you had.

#### Worked example: commit-before vs commit-after on the same crash

Make it concrete with numbers. A single partition delivers records with offsets 100 through 109 — ten records — in one `poll()`. Your consumer processes them one at a time, and the process crashes **right after fully processing record 104 and before touching record 105.** Count duplicates and losses under each ordering.

**Commit-after-process (at-least-once).** You process record N, then commit N+1. So you processed and committed 100→101, 101→102, ... up through processing 104. The crash hits after processing 104 but — critically — the question is whether the commit for 104 (committing offset 105) landed before the crash. Say it did *not*: you finished `process(104)` and the crash struck before `commit(105)` returned. The last durably committed offset is 104 (committing 104 means "resume at 104"). On restart, the consumer resumes at 104 and reprocesses record 104. **Result: 0 records lost, 1 record duplicated (104).** Every record is processed at least once; record 104 is processed twice. If `process` is idempotent, the duplicate is a harmless no-op and the net effect is exactly-once.

**Commit-before-process (at-most-once).** You commit N+1, then process N. So before touching record 104 you committed offset 105 ("resume at 105, I'm about to do 104"). You start `process(104)`, and the crash strikes mid-processing — say it wrote half a database row and died. The committed offset is 105. On restart, the consumer resumes at 105 and **never sees record 104 again.** Record 104 was committed-as-done but only half-done. **Result: 1 record lost (104), 0 duplicated.** And worse, it was lost in a partial state — half a row written — with no signal that anything went wrong.

Same crash, same instant, opposite outcomes, decided entirely by which of two lines ran first. Commit-after gives you a duplicate you can dedupe away; commit-before gives you a silent, possibly-partial loss you cannot recover. That is the trade in its purest form, and it is why "process, then commit" is the default discipline for everything that matters.

![A pipeline showing the safe manual commit loop: poll, process the batch, commitSync the last offset, then poll again](/imgs/blogs/consumer-offset-commit-strategies-failure-modes-4.webp)

## 5. Committing on rebalance

A crash is not the only way you lose a partition. The more frequent and more insidious way is a **rebalance**: the group coordinator decides to reassign partitions — because a member joined, left, or timed out — and your consumer is told to give up partitions it was actively processing. If you are not careful here, every rebalance becomes a duplicate storm, and rebalances happen far more often than crashes — every deploy, every scale event, every brief network hiccup.

The mechanics of the rebalance handshake itself — heartbeats, the session timeout, eager versus cooperative protocols — are covered in [Kafka consumer groups, offsets, and rebalancing](/blog/software-development/message-queue/kafka-consumer-groups-offsets-rebalancing). What matters *here* is one callback: `onPartitionsRevoked`. When you `subscribe()`, you can register a `ConsumerRebalanceListener`, and Kafka calls its `onPartitionsRevoked` method **just before** it takes partitions away from you, while you still own them. That callback is the last safe moment to commit. Commit there, synchronously, and the new owner resumes from a clean offset reflecting everything you actually finished. Skip it, and the new owner resumes from your last *periodic* commit — replaying every record you processed since.

![A graph showing a rebalance: trigger leads to onPartitionsRevoked, which either commits the current offset before losing the partition or skips it and forces reprocessing, before the coordinator reassigns to a new owner](/imgs/blogs/consumer-offset-commit-strategies-failure-modes-5.webp)

### Why this is `commitSync`, never `commitAsync`

Inside `onPartitionsRevoked`, you have a hard deadline: you must commit *before* you return from the callback, because the instant you return, Kafka reassigns the partition and any commit you fire afterward will be **rejected** — you no longer own the partition, so the coordinator refuses the offset. So this is the textbook case for `commitSync`: you need the commit to actually complete before you proceed, there is no "next commit" to fix a failure, and blocking is exactly what you want. An async commit here is a bug — it might still be in flight when you return, get rejected after reassignment, and the new owner reprocesses a batch.

```java
consumer.subscribe(Collections.singletonList("orders"), new ConsumerRebalanceListener() {
    @Override
    public void onPartitionsRevoked(Collection<TopicPartition> partitions) {
        // LAST chance to commit before we lose these partitions. Must be sync.
        consumer.commitSync(currentOffsets);   // our tracked next-offset per partition
    }
    @Override
    public void onPartitionsAssigned(Collection<TopicPartition> partitions) {
        // optionally seek to externally-stored offsets here (section 7)
    }
});
```

### Tracking offsets so you have something to commit

To commit a meaningful offset in `onPartitionsRevoked`, you have to *track* your progress as you go — you cannot rely on the client's notion of position, because the position includes records you fetched but have not finished. The standard pattern is a `Map<TopicPartition, OffsetAndMetadata>` you update after processing each record (or each batch), storing `record.offset() + 1`. When the revoke callback fires, you commit that map. This is the same map you commit in the normal loop. The point is that *you* own the definition of "processed," and the revoke callback commits *your* definition, not the client's optimistic position. With cooperative-sticky rebalancing, `onPartitionsRevoked` is called only for the partitions you are actually losing — so the commit is scoped and cheap — but the discipline is identical: commit what you finished before you let go.

### The cost of getting it wrong

Without the revoke commit, here is the failure: a deploy rolls a consumer, which triggers a rebalance. Your consumer had processed 4,000 records since its last periodic (say, async) commit. It loses the partition without committing those 4,000. The new owner reads the stale committed offset and reprocesses all 4,000. Multiply by every partition moving in the rebalance, and a single deploy reprocesses tens of thousands of records. If your consumer is idempotent, this is wasted work and a latency blip. If it is not, this is the deploy that double-sent ten thousand emails. The revoke commit turns a rebalance from a duplicate storm into a clean handoff, and it is the single most impactful thing you can add to a manual-commit consumer after getting the basic ordering right.

## 6. Batch processing and commit cadence

So far the examples committed after every record. In production you almost never do that — committing once per record means a coordinator round trip per record, which destroys throughput. Real consumers process in **batches** and commit per batch. The choice of batch granularity and commit cadence is a direct throughput-versus-duplicate-window tradeoff, and it is worth tuning deliberately rather than inheriting.

The natural batch unit is one `poll()` call. You configure `max.poll.records` (default 500) to bound how many records each poll returns, process the whole batch, then commit once. This is the loop the pipeline figure under section 4 shows: poll, process the batch in order, `commitSync` the last offset, poll again. Committing once per batch instead of once per record cuts your commit rate by up to 500×, and the duplicate window on a crash is bounded by *one batch* — you reprocess at most the current batch, not the last five seconds.

### The cadence tradeoff is the size of your duplicate window

The commit cadence directly sizes your replay-on-crash. Commit after every batch of 500 and a crash mid-batch replays up to 500 records. Commit after every 10 batches (5,000 records) to save more commit overhead, and a crash replays up to 5,000. Commit after every record and a crash replays at most 1 record — but you pay a coordinator round trip per record. So:

- **Smaller batches / more frequent commits:** smaller duplicate window, higher commit overhead, lower throughput.
- **Larger batches / less frequent commits:** larger duplicate window, lower commit overhead, higher throughput.

There is no universally right answer; there is the right answer *for your duplicate tolerance and your throughput target.* If your consumer is idempotent, lean toward larger batches and fewer commits — duplicates are cheap, throughput is precious. If your consumer is expensive-but-not-idempotent (and you cannot fix that), lean toward smaller batches to bound the blast radius. One commit per `poll()` is the sane default that most consumers should start from.

#### Worked example: auto-commit crash at 3 seconds into the 5-second interval

Now the auto-commit replay made fully concrete, because this is the number people guess wrong. A consumer reads a single partition with `enable.auto.commit=true` and the default `auto.commit.interval.ms=5000`. It processes records at a steady **200 records per second**. Timeline:

- **T+0.0s:** an auto-commit fires during a poll, committing offset **1000** (everything through 999 is recorded as done).
- **T+0.0s to T+3.0s:** the consumer polls and processes records 1000 through 1599 — that is 200 rec/s × 3 s = **600 records** processed since the last commit.
- **T+3.0s:** the process crashes. The next auto-commit would have fired at T+5.0s but never does.
- **T+8.0s:** a new consumer (or the restarted one) takes the partition, reads the committed offset **1000**, and resumes there.

How many records reprocess? Every record from offset 1000 through 1599 — **600 records replayed.** All 600 were already fully processed before the crash; all 600 run again. At 200 rec/s that is three seconds of duplicate work injected by a crash that happened to land three-fifths of the way through the commit interval. Crash at T+4.9s instead and you would replay ~980 records — almost the whole interval. Crash at T+0.1s and you replay ~20. The duplicate count is *linear in how long you've gone since the last timer commit*, and the worst case is the full `auto.commit.interval.ms` worth of throughput. This is exactly why shrinking `auto.commit.interval.ms` shrinks the duplicate window — and why it does nothing for the *loss* window, which depends on processing falling behind poll, not on the timer.

![A timeline showing an auto-commit crash three seconds into a five-second interval forcing reprocessing of every record handled since the last commit](/imgs/blogs/consumer-offset-commit-strategies-failure-modes-3.webp)

#### Worked example: choosing batch size against a duplicate-cost budget

Put a price on duplicates and the batch-size decision becomes arithmetic instead of intuition. Suppose a consumer processes **1,000 records per second** on one partition, each record does a database upsert that costs roughly 1 ms, and the consumer crashes (or rebalances without a revoke commit) on average **twice per hour** across the fleet. You commit once per batch. The question: how big should the batch be?

The duplicate cost of one crash is "one batch worth of reprocessing." With a batch of 1,000 records (one second of work), a crash replays up to 1,000 upserts — about 1 second of extra DB load — and at two crashes per hour that is `2 × 1,000 = 2,000` duplicate upserts per hour, or roughly 0.0006% of the `3,600,000` records processed in that hour. Negligible. Now go the other direction: batch 50,000 records (fifty seconds of work) to slash commit overhead. A crash now replays up to 50,000 upserts, and two crashes per hour means `2 × 50,000 = 100,000` duplicate upserts per hour — fifty times more reprocessing, plus a fifty-second replay spike that briefly doubles your DB write load right after every crash. The throughput you saved on commits — going from one commit per second to one per fifty seconds saves 49 coordinator round trips per partition per fifty seconds, maybe a millisecond of latency each — is dwarfed by the duplicate cost you took on. The lesson the numbers force: **commit overhead is cheap and duplicate-replay is the real cost, so when in doubt commit more often, not less.** A batch around one `poll()` (a few hundred records) sits in the sweet spot for almost everyone — small enough that a crash replays a fraction of a second of work, large enough that commit overhead is a rounding error. You only widen the batch when you have *measured* commit overhead to be a real bottleneck and confirmed the consumer is idempotent enough that the bigger replay is harmless.

### Partial-batch failures

One sharp edge of batch commits: what if record 250 of a 500-record batch throws? You cannot commit 500 (records 250–499 may not be done) and you do not want to commit 0 (records 0–249 *are* done — recommitting from 0 replays them). The disciplined answer is to commit the offset of the *last successfully processed record plus one* — here, offset 250 — and then decide what to do with the poison record 250 (retry it, dead-letter it, skip it with a log). This requires tracking per-record progress inside the batch, not just per-batch. Many teams skip this and just let the whole batch replay on any failure, accepting that the already-done prefix gets reprocessed — which is fine *if the consumer is idempotent* and intolerable if it is not. Once again the batch-commit blast radius is only safe because idempotency makes the replay a no-op.

## 7. Storing offsets externally for effectively-once

Everything above lives with an unavoidable gap: the offset commit and the processing result are written to *two different places* — the result to your database, the offset to `__consumer_offsets` — and there is no way to make those two writes atomic. Process the record, write the result to your DB, then commit the offset to Kafka: crash between the two and you have a result with no committed offset, so you replay (duplicate). Commit the offset to Kafka first, then write the result: crash between and you have a committed offset with no result (loss). The two-store split *is* the duplicate-or-loss window. You cannot close it as long as the offset and the result live in different systems.

The way to close it is to stop using `__consumer_offsets` for offsets that must be atomic with your result, and **store the offset in the same store as the result, written in the same transaction.** If your result goes into PostgreSQL, you also write the offset into a small `consumer_offsets` table in PostgreSQL, and you write both inside one `BEGIN ... COMMIT`. Now there is exactly one durable commit point. Either the result row and the offset row both land, or neither does — the transaction guarantees it. On restart, you read the offset from *your* table (not from Kafka) and `seek()` the consumer there. There is no longer a window where the result exists without its offset or vice versa, because they are the same atomic write. This is **effectively-once**, and it is the strongest consumer-side guarantee you can build without Kafka transactions.

![A grid showing effectively-once: poll a record, begin a transaction, write the result row, upsert the offset, commit atomically, and on restart seek to the stored offset](/imgs/blogs/consumer-offset-commit-strategies-failure-modes-9.webp)

### The pattern in code

The shape is: disable Kafka's offset commit entirely, store offsets in your result store, and seek from there on assignment.

```python
# enable_auto_commit=False, and we NEVER call consumer.commit() — Kafka offsets are unused.
for record in consumer:
    with db.transaction() as txn:                  # ONE atomic transaction
        txn.execute(
            "INSERT INTO results (id, payload) VALUES (%s, %s) "
            "ON CONFLICT (id) DO NOTHING",          # idempotent result write
            (record.key, record.value),
        )
        txn.execute(
            "INSERT INTO consumer_offsets (topic, partition, next_offset) "
            "VALUES (%s, %s, %s) "
            "ON CONFLICT (topic, partition) DO UPDATE SET next_offset = EXCLUDED.next_offset",
            (record.topic, record.partition, record.offset + 1),
        )
    # If we crash before COMMIT, BOTH the result and the offset roll back -> clean replay.
    # If COMMIT succeeds, BOTH are durable together -> no divergence ever.
```

And on partition assignment, you must `seek()` to your stored offset instead of letting Kafka resume from `__consumer_offsets`:

```python
def on_assign(consumer, partitions):
    for tp in partitions:
        stored = db.query(
            "SELECT next_offset FROM consumer_offsets WHERE topic=%s AND partition=%s",
            (tp.topic, tp.partition),
        )
        consumer.seek(tp, stored if stored is not None else 0)
```

![A stack showing the offset storage layers: in-memory position, pending async commit, the __consumer_offsets topic, and an external store written in the same transaction as the result](/imgs/blogs/consumer-offset-commit-strategies-failure-modes-6.webp)

### Why this beats committing to Kafka, and what it costs

The reason this is stronger than any Kafka-side commit ordering is that it eliminates the two-store problem entirely — there is one store, one commit, one atomic point. No ordering of two separate writes can match a single transaction. This is the same insight behind the transactional outbox pattern, and it is closely related to change data capture; the [change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) post explores the producer-side mirror image, where you write a business change and its outgoing event in one transaction.

### The offset belongs to the partition, not the record

A subtlety that trips up first implementations: the offset you store is *per partition*, and your `consumer_offsets` table is keyed by `(topic, partition)`, not by record key. When you batch the result writes and commit once per batch, you store **one** offset row per partition — the highest processed offset plus one — not one per record. And on assignment you must `seek` each *partition* to its stored offset. If you key the offset table wrong — say by message key instead of partition — you get a table that grows without bound and a `seek` that has no single answer per partition. Keep the offset table keyed exactly the way `__consumer_offsets` is keyed, `(topic, partition)`, and treat it as a drop-in replacement for that topic, because that is precisely what it is.

### Idempotency still earns its keep here

Notice the result insert in the code above uses `ON CONFLICT DO NOTHING` — an idempotent write — even though the transaction already guarantees offset-and-result atomicity. Why belt *and* suspenders? Because the transaction protects you against the *commit-point* gap, but it does not protect you against a record that gets delivered twice for reasons *outside* this consumer — a producer retry that wrote the same logical event twice, or a replay you triggered manually. The transactional offset store closes the consumer-side window; the idempotent write closes the everything-else window. Together they are robust; either alone leaves a gap. This is the same layered argument the [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) post makes at length — atomic commit and idempotent effect are complements, not substitutes.

### The cost, stated honestly

The cost is real. You take on a transaction per result (or per batch — you can batch the inserts and commit once), you give up Kafka's lag-monitoring tooling that reads `__consumer_offsets` (your lag now lives in your DB and you must monitor it yourself), and you must handle the `seek`-on-assign dance. For a Kafka-to-Kafka pipeline, this external-store pattern is usually the wrong tool — that is exactly what Kafka's own transactions are for, covered in [exactly-once in Kafka](/blog/software-development/message-queue/exactly-once-in-kafka-idempotent-producer-transactions), which makes the read, the process, and the output-topic write atomic *within Kafka*. But for the extremely common Kafka-to-database consumer — the workhorse that drains a topic into a relational store — storing the offset in the same DB transaction as the result is the cleanest effectively-once you can get, and it does not require Kafka transactions or an idempotent producer at all.

## 8. Commit failure modes in production

The strategies above are the happy path. Production is where commits fail in ways that are easy to miss in a code review and obvious only in an incident. Here are the ones that actually show up, with what they look like and how to defend against them.

### The lost async commit

`commitAsync()` does not retry. If the last async commit before a crash fails — the broker was momentarily unreachable, the request timed out — that commit is simply lost, and nothing recovers it. On restart you replay from the previous successful commit. In a steady loop this is usually harmless because the *next* async commit would have superseded it, but if the crash follows the failed commit closely, you replay everything since the last *successful* async commit, which could be several batches. The defense is the standard pattern from section 3: async in the loop, **`commitSync` in the shutdown path**, so the final commit is the retried, guaranteed one. The lost-async-commit window only hurts when there is no successful commit after it, and a sync commit on shutdown guarantees there is.

![A before-and-after figure contrasting an async-only commit loop that loses its last commit on crash with a loop that adds a synchronous commit on shutdown to make progress durable](/imgs/blogs/consumer-offset-commit-strategies-failure-modes-7.webp)

### The commit rejected after a long pause (max.poll.interval.ms)

This is the nastiest one and it surprises people who think they are doing everything right. There is a config called `max.poll.interval.ms` (default 300000 ms — five minutes) that bounds how long you may go *between* `poll()` calls. If your processing of one batch takes longer than that — a slow downstream, a giant batch, a GC pause — Kafka assumes your consumer is dead, **kicks it out of the group, and revokes its partitions.** You, meanwhile, are still happily processing, oblivious. You finish, you call `commitSync()` to commit your hard-won progress, and the broker **rejects it with a `CommitFailedException`** — because you no longer own those partitions. They were reassigned to someone else minutes ago, who has been reprocessing your batch in parallel the whole time. Your commit is refused, your work is duplicated, and the error message ("Commit cannot be completed since the group has already rebalanced and assigned the partitions to another member") is the telltale.

The fix is to make sure you never blow `max.poll.interval.ms`: shrink `max.poll.records` so each batch is processable within the interval, raise `max.poll.interval.ms` if your processing is legitimately slow, or move slow processing off the poll thread entirely (a worker pool, with careful pause/resume). The deepest fix is to *not do long blocking work between polls* — Kafka's group membership is a liveness heartbeat measured in part by poll frequency, and a consumer that disappears for ten minutes to process one giant batch is, from the coordinator's view, indistinguishable from a dead one.

### Committing an uncommitted-but-fetched offset (the auto-commit loss again)

Worth restating as a failure mode because it is the most common loss in the wild: with auto-commit on and processing slower than poll, the timer commits the *fetched* position while records sit unprocessed in your loop. A crash then skips them. There is no exception, no log line, no dead-letter — the records are simply never redelivered. The only way you find it is by reconciliation: counting records produced versus records landed in your sink and noticing a gap. The defense is structural, not operational: turn off auto-commit and commit after processing. You cannot monitor your way out of silent loss; you have to design it out.

### The rebalance-during-shutdown double commit race

When you call `consumer.close()`, the client triggers a final commit (if auto-commit is on) and leaves the group, which causes a rebalance. If your shutdown also has its own `commitSync` in a `finally`, and a rebalance listener that *also* commits, you can have two commit paths racing. This is usually benign (last-write-wins on the same or higher offset), but it bites if your rebalance-revoke commit and your shutdown commit disagree on the tracked offset — for instance if one uses the client position and the other uses your tracked map. The defense is to have exactly one source of truth for "the offset to commit" (your tracked map) and commit *that* from every path. Never mix the client's optimistic position with your tracked processed-offset in the same consumer.

### The offset that goes backward (a manual seek or a bad reset)

A rarer but spectacular failure: the committed offset moves *backward*, and a chunk of the topic gets reprocessed wholesale. The usual causes are a manual `seek` to an earlier offset that someone ran to "replay a few messages" and forgot to bound, an offset-reset tool pointed at the wrong group, or — the subtle one — an async commit retry that overwrote a higher offset with a lower one (the exact bug the no-retry design of `commitAsync` exists to prevent, which is why you must never hand-roll async retry). When the offset jumps back, every record between the new lower offset and the old position is redelivered, often thousands or millions of records, and if the consumer is not idempotent every one of those is a duplicate effect. The defense is twofold: never run an offset-reset against a live group without understanding the blast radius, and make the consumer idempotent so even a catastrophic backward seek produces no duplicate *effects*, only wasted work. An idempotent consumer turns "we accidentally reset the offset to zero" from a billing incident into a slow afternoon of harmless reprocessing — which is the single best argument for idempotency there is.

### The poison record that blocks the commit forever

If one record reliably throws during processing and your loop retries it forever before committing, the committed offset never advances past it. The partition stalls; lag climbs; eventually the lag crosses retention and you lose *unrelated* downstream data. A single bad record has taken down a partition. The defense is a bounded retry with a dead-letter escape hatch: try the record N times, and if it still fails, send it to a dead-letter topic, **commit past it**, and move on. Committing past a poison record is the deliberate choice to skip one record to keep the partition flowing — and it is the right one, because a stalled partition is a far larger loss than a single dead-lettered record you can replay later from the dead-letter topic.

### The same problem in RabbitMQ: per-message acks instead of offsets

It is worth a paragraph to see that this is not a Kafka quirk — it is the universal shape of consumer-side delivery guarantees, just with different machinery. RabbitMQ has no offset; it tracks delivery per message with an **acknowledgement**. A consumer reads a message, and the broker holds it as "unacknowledged" until the consumer sends a `basic.ack`. Ack before processing is at-most-once (the broker drops it from the unacked set, and a crash mid-process loses it); ack after processing is at-least-once (a crash before the ack causes the broker to redeliver on the next consumer). The *exact same ordering choice*, with `basic.ack` playing the role of the offset commit. The differences are real — RabbitMQ acks one message at a time (or a range with the `multiple` flag) rather than a single advancing pointer, and it redelivers from its own queue rather than from a replayable log — but the failure-mode logic is identical: ack-before is loss, ack-after is duplicates, and you choose. The mechanics of acks, publisher confirms, and quorum queues are covered in [RabbitMQ production architecture and scaling](/blog/software-development/system-design/rabbitmq-production-architecture-scaling); the point here is that "where do I commit relative to processing" is the question for *every* consumer in *every* broker, and the answer always picks your failure mode. Learn it once and it transfers.

The one genuine asymmetry: Kafka's single advancing offset means you cannot ack message 105 while leaving 103 unacked — the offset is a high-water mark, so committing 106 implicitly claims 103, 104, and 105 are all done. RabbitMQ's per-message acks let you ack out of order, acknowledging 105 while 103 is still in flight. This makes RabbitMQ more flexible for unordered concurrent processing within a partition and makes Kafka strictly ordered and simpler to reason about. Neither is better; they are different points on the ordering-versus-flexibility trade. But the commit-timing decision — and its duplicate-or-loss consequence — is the same in both.

## 9. Choosing a commit strategy for your guarantee

Pull it together into a decision. The commit strategy is downstream of one question: **what guarantee does this workload need, and what is the cost of getting it wrong?** Answer that and the strategy falls out.

![A tree taxonomy of commit strategies, splitting into automatic and manual, with manual branching into commitSync, commitAsync, external storage, and on-rebalance commits](/imgs/blogs/consumer-offset-commit-strategies-failure-modes-8.webp)

- **Self-correcting stream, loss and duplicates both cheap** (metrics, telemetry, best-effort): leave `enable.auto.commit=true`. The convenience is worth it and the hazards do not matter for this data. Choose this *explicitly* and comment why, so nobody "fixes" it into a slower manual loop.
- **Loss unacceptable, duplicates harmless** (the consumer is idempotent — upserts, dedup keys): `enable.auto.commit=false`, commit *after* processing, `commitAsync` in the loop, `commitSync` on rebalance and shutdown, commit per batch. This is at-least-once, and idempotency makes the duplicates no-ops. This is the right answer for the majority of serious Kafka-to-anything consumers.
- **Loss unacceptable, duplicates also visibly harmful, sink is a database** (payments, orders, anything that double-applies badly): store the offset in the same DB transaction as the result. Effectively-once, no Kafka transactions needed, and the strongest guarantee you can build for a Kafka-to-DB consumer. Disable Kafka offset commits, `seek` from your store on assign.
- **Loss unacceptable, duplicates harmful, sink is another Kafka topic** (stream processing, read-process-write): use Kafka transactions / exactly-once semantics, the subject of [exactly-once in Kafka](/blog/software-development/message-queue/exactly-once-in-kafka-idempotent-producer-transactions). This is the one case where the offset commit is *itself* part of a Kafka transaction, atomic with the output write. Do not reach for it outside Kafka-to-Kafka; it gives you nothing the moment an effect crosses the boundary out of Kafka.
- **At-most-once on purpose** (a record is worthless if reprocessed, loss is fine): commit *before* processing, and write a comment that says so. Rare, but legitimate.

### The configs that actually matter

A short list of the consumer configs that govern all of this, with the values I reach for:

| Config | Default | What it controls | Reach-for value |
| --- | --- | --- | --- |
| `enable.auto.commit` | `true` | Timer-based auto-commit on/off | `false` for any business data |
| `auto.commit.interval.ms` | `5000` | Size of the auto-commit duplicate window | only relevant if auto-commit on |
| `max.poll.records` | `500` | Batch size per poll; bounds duplicate window | tune so a batch fits in `max.poll.interval.ms` |
| `max.poll.interval.ms` | `300000` | Max gap between polls before eviction | raise if processing is legitimately slow |
| `auto.offset.reset` | `latest` | Where to start with no committed offset | `earliest` to avoid silent skip on fresh group |

The single highest-leverage line is `enable.auto.commit=false`. Everything correct flows from owning the commit; everything subtle-and-broken flows from leaving it on a timer.

### A four-question decision procedure

If you want a procedure rather than a taxonomy, ask four questions in order and stop at the first that decides it.

1. **Can this data tolerate both loss and duplicates?** (Self-correcting metrics, telemetry, sampled logs.) If yes, leave auto-commit on, comment that you mean it, and you are done. The convenience is correct here.
2. **Is the sink another Kafka topic and nothing else?** (A read-process-write stream job.) If yes, use Kafka transactions / exactly-once semantics, where the offset commit is atomic with the output write. The boundary caveat from [exactly-once in Kafka](/blog/software-development/message-queue/exactly-once-in-kafka-idempotent-producer-transactions) applies — it covers only effects inside Kafka.
3. **Is the sink a database, and do duplicates cause visible harm?** (Payments, ledgers, anything that double-applies badly.) If yes, store the offset in the same DB transaction as the result. One atomic commit point, effectively-once, no Kafka transactions required.
4. **Otherwise** — loss is unacceptable, duplicates are tolerable-if-deduped — use manual at-least-once: `enable.auto.commit=false`, commit after processing, async in the loop, sync on rebalance and shutdown, one commit per batch, and an idempotent consumer. This is where the majority of consumers land, and it is the right default to assume until one of the three questions above redirects you.

Notice that every branch except the first turns auto-commit *off*. That is the through-line: the moment correctness matters, you own the commit. Auto-commit is the answer to exactly one question — "does anything here matter?" — and only when the answer is no.

### Monitoring the commit, not just the lag

One operational note that closes the loop. Most teams monitor consumer *lag* — the gap between the log-end offset and the committed offset — because Kafka's tooling reads `__consumer_offsets` and surfaces it. That is necessary but not sufficient. Lag tells you the *committed* offset is falling behind production; it tells you nothing about the gap between *committed* and *actually processed*, which is exactly where auto-commit loss hides. If you have moved offsets to an external store (section 7), lag tooling that reads `__consumer_offsets` is blind entirely, because you are not committing there. So monitor two things the default dashboards miss: the rate of commit failures (the metric the async callback should increment) and, for external-store consumers, a hand-rolled lag computed from your offset table against the partition's log-end offset. A consumer that looks healthy on the default lag dashboard can still be silently losing data through an auto-commit window or silently failing every async commit — the default dashboard cannot see either, and only commit-level metrics can.

```yaml
# A sane production consumer config for at-least-once + idempotency.
enable.auto.commit: false        # own the commit, always
auto.offset.reset: earliest      # don't silently skip on a brand-new group
max.poll.records: 200            # small enough to process within max.poll.interval.ms
max.poll.interval.ms: 300000     # raise only if a batch legitimately needs longer
# Then in code: process the batch, commitAsync in the loop,
# commitSync on rebalance-revoke and on shutdown.
```

## Case studies and war stories

### The vanished orders (auto-commit loss under load)

A logistics startup ran its order-ingest consumer on Kafka defaults, auto-commit included. For months it worked, because at low traffic each poll's records were fully processed well before the next five-second commit. Then a marketing campaign tripled volume, processing time per batch crept past five seconds, and the auto-commit started firing on the *fetched* position while records still sat unprocessed in the loop. Consumers crashed and restarted normally during a routine deploy — and on restart, the committed offset was past records that had never been written. Orders disappeared. No error, no dead-letter, no alert, because committed-past-unprocessed is silent by construction. They found it only when a customer's shipment never came and the order was in no database. The root cause was that nobody had ever *chosen* where the offset committed relative to processing — they inherited at-most-once from a default and called it reliable Kafka. The fix was `enable.auto.commit=false`, commit after processing, and idempotent upserts so the now-possible duplicates were harmless. The lesson on the wall: **auto-commit's failure mode flips to loss exactly when load rises, which is exactly when you can least afford it.**

### The deploy that double-sent ten thousand emails (no revoke commit)

A notifications team ran an at-least-once consumer with idempotent-*looking* code, committing offsets asynchronously every few hundred milliseconds — but they had no `ConsumerRebalanceListener`, so they never committed on partition revoke. Every deploy rolled the consumer fleet, triggering a rebalance, and each revoked partition handed off a stale offset reflecting only the last successful async commit. The new owner reprocessed every record since that commit — several thousand per partition. The emails were *supposed* to be deduped by a downstream key, but the dedup table had a bug for one message type, and that type went out twice to every affected user during the deploy: roughly ten thousand duplicate emails in fifteen minutes. The fix was a two-liner — a rebalance listener whose `onPartitionsRevoked` did a `commitSync` of the tracked offsets — which turned the rebalance from a duplicate storm into a clean handoff. The lesson: **a rebalance is a partition loss you can prepare for, and the revoke commit is the preparation; without it every deploy replays a batch per partition.**

### The CommitFailedException nobody could explain (max.poll.interval.ms blown)

A data team's enrichment consumer called a slow external API per record. Occasionally the downstream slowed and a batch of 500 took over five minutes to process. The consumer would finish, call `commitSync`, and get a `CommitFailedException` saying the group had rebalanced and reassigned the partitions. They could not reproduce it on demand and burned a week thinking it was a coordinator bug. The truth: blowing `max.poll.interval.ms` got them evicted from the group mid-batch; the partition was reassigned to another member that reprocessed the whole batch in parallel; their post-facto commit was rejected because they no longer owned the partition. The fix was to cut `max.poll.records` from 500 to 50 so a batch always finished inside the interval, plus moving the slow API calls to a bounded worker pool. The lesson: **poll frequency is a liveness signal, and a consumer that disappears for minutes to process one giant batch is, to the coordinator, a dead consumer — so make every batch fit inside the poll interval.**

### The off-by-one that double-processed every deploy

A team migrated a consumer from auto-commit to manual commit to fix exactly the silent-loss problem above, and in the process introduced a textbook off-by-one. Their manual commit code committed `record.offset()` — the offset of the record just processed — instead of `record.offset() + 1`. The consumer worked perfectly under steady state, because the in-memory position kept advancing and they never re-read from the committed offset while running. But every restart, and every rebalance, resumed from the committed offset, which pointed at the *last processed* record rather than the *next* one. So on every single deploy, every partition reprocessed exactly one record — the last one it had committed. With dozens of partitions and a deploy cadence of several times a day, that was hundreds of duplicate processings a week, all clustered at deploy time, which is exactly when an on-call engineer is watching and most likely to misattribute it. They chased it as a "rebalance bug" for a month before someone noticed the commit was missing its plus-one. The fix was a single character. The lesson: **the committed offset is a resume-here pointer, so it must point at the next record, not the last one; the plus-one is not a detail, it is the definition.**

### The Kafka-to-Postgres pipeline that got effectively-once for free

A payments team needed a topic of settlement events drained into Postgres with no double-applies and no losses. They reached first for Kafka exactly-once semantics, then realized their sink was a database, not another topic — so Kafka transactions could cover the read but not the Postgres write. They switched to storing the offset in a `consumer_offsets` table in the *same* Postgres database, written in the same transaction as the settlement row, with the offset commit to Kafka disabled entirely and a `seek` from the table on assignment. One atomic commit point per record; a crash rolled back both the settlement and the offset together; a restart resumed exactly where the last committed transaction left off. They got effectively-once with no Kafka transactions, no idempotent producer, and tooling they already understood. The lesson: **for a Kafka-to-database consumer, the offset belongs in the database, in the same transaction as the result — that single move closes the duplicate-or-loss window that no Kafka-side commit ordering can close.**

## When to reach for each strategy (and when not to)

**Reach for auto-commit when** the data is genuinely self-correcting and both failure modes are cheap — metrics, telemetry, best-effort cache fills, sampled logs. Choose it deliberately and comment that you mean it, because its silent loss mode under load is the most common way teams ship accidental at-most-once. Never leave it on for business data because changing it later "felt like a big change."

**Reach for manual commit-after-process plus idempotency (at-least-once) when** loss is unacceptable and duplicates can be made harmless — which is most serious workloads. `enable.auto.commit=false`, commit after processing, `commitAsync` in the loop, `commitSync` on rebalance and shutdown, one commit per batch. Pair it with a dedup key or an upsert so the inevitable replays are no-ops. This is the default destination for the majority of consumers and the one you should assume you want until proven otherwise.

**Reach for external offset storage in one transaction when** the sink is a database and duplicates are visibly harmful. It is the strongest consumer-side guarantee that does not require Kafka transactions, and it is the natural fit for the ubiquitous Kafka-to-DB consumer. The cost is a transaction per result and losing Kafka's native lag tooling — pay it when correctness matters more than the convenience of `__consumer_offsets`.

**Reach for Kafka transactions / exactly-once semantics only when** the entire effect stays inside Kafka — a read-process-write stream job whose output is another topic. Then the offset commit is part of the transaction, atomic with the output. The mechanics and the boundary caveat are in [exactly-once in Kafka](/blog/software-development/message-queue/exactly-once-in-kafka-idempotent-producer-transactions). Do not enable it as a reflex; it adds coordination latency and gives you nothing once an effect leaves Kafka.

**Do not** leave auto-commit on for anything you would be embarrassed to lose. Do not commit before processing unless you have written down that at-most-once is intended. Do not use `commitAsync` for the final or rebalance commit. Do not forget the `+ 1`. Do not block the poll thread long enough to blow `max.poll.interval.ms`. And do not assume your guarantee — find the commit, read its position relative to processing, and you will know exactly which messages a crash loses or duplicates.

## Key takeaways

- **A committed offset is a promise that everything below it is processed and need never be redelivered.** Kafka cannot verify the promise — it trusts the number you send — so a commit made before processing finished is a silent lie that turns into loss.
- **You commit the next offset you want, not the last one you processed.** Commit `record.offset() + 1`. Forget the plus-one and you reprocess one record on every restart and every rebalance.
- **Auto-commit opens two opposite hazards at once.** When commit lags processing you get duplicates on crash; when processing lags poll the timer commits unprocessed records and you get silent loss — and rising load slides you from the first into the second.
- **commitSync is blocking and retried; commitAsync is fast and never retried.** Use async in the hot loop (the next commit supersedes a failure) and sync where there is no next commit — shutdown and rebalance-revoke.
- **Commit ordering picks your failure mode.** Commit before processing is at-most-once (lose on crash); commit after processing is at-least-once (duplicate on crash). There is no ordering that gives both safety and uniqueness for free.
- **Commit on `onPartitionsRevoked`, synchronously, before you lose the partition.** Skip it and every rebalance — every deploy — replays a batch per partition to the new owner.
- **Commit cadence is the size of your duplicate window.** One commit per `poll()` is the sane default; larger batches mean fewer commits and a bigger replay-on-crash, which is fine only if the consumer is idempotent.
- **Store the offset in the same transaction as the result for effectively-once.** One atomic commit point eliminates the two-store duplicate-or-loss window that no Kafka-side commit ordering can close — the right pattern for Kafka-to-database consumers.
- **The classic production failures are knowable.** Lost async commit (fix with sync on shutdown), `CommitFailedException` after blowing `max.poll.interval.ms` (fix with smaller batches), silent auto-commit loss (fix by disabling auto-commit), and the poison record that stalls a partition (fix with a dead-letter escape and committing past it).
- **To audit any consumer's guarantee, find the commit and read its position relative to processing.** The config docs describe intent; the line ordering describes reality, and only one of them is binding.

## Further reading

- [Kafka consumer groups, offsets, and rebalancing](/blog/software-development/message-queue/kafka-consumer-groups-offsets-rebalancing) — where offsets and rebalances come from, the heartbeat and timeout machinery, and the auto-versus-manual distinction this post dissects in depth.
- [Delivery semantics: at-most-once, at-least-once, exactly-once](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) — the guarantees themselves and why exactly-once delivery is impossible, the conceptual frame this post grounds in the commit.
- [Idempotency and deduplication: making at-least-once safe](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe) — the consumer-side machinery that turns the duplicates from commit-after-process into harmless no-ops.
- [Exactly-once in Kafka: the idempotent producer and transactions](/blog/software-development/message-queue/exactly-once-in-kafka-idempotent-producer-transactions) — how Kafka transactions make the offset commit atomic with an output-topic write for Kafka-to-Kafka pipelines.
- [Change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — the producer-side mirror of storing-offset-with-result: writing a business change and its event in one transaction.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — the durability and replication that make a committed offset trustworthy on the broker side in the first place.
- Apache Kafka consumer documentation — the authoritative reference for `commitSync`, `commitAsync`, `ConsumerRebalanceListener`, and every config in the table above.
