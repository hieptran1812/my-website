---
title: "Cluster Operations: Scaling, Rolling Upgrades, and Partition Reassignment Without Downtime"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "The unglamorous day-2 work of keeping a broker cluster online while you scale it, upgrade it, and move terabytes of partitions across machines — with throttles, controlled shutdowns, rack awareness, and the one metric that tells you whether any of it is going sideways."
tags:
  [
    "message-queue",
    "kafka",
    "cluster-operations",
    "partition-reassignment",
    "rolling-upgrade",
    "rack-awareness",
    "distributed-systems",
    "event-driven",
    "reliability",
    "operations",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/cluster-operations-scaling-upgrades-partition-reassignment-1.webp"
---

Nobody puts "moved 2 terabytes of partition replicas across a live cluster on a Tuesday afternoon without a single producer noticing" on their promotion packet, and that is exactly the problem. The work that keeps a message-queue cluster healthy is the least visible work you do. When it goes right, the graphs are flat, the on-call phone is silent, and it looks like nothing happened — because nothing did, which was the entire point. When it goes wrong, you take down a payments pipeline at peak, your incident channel fills up, and someone asks why a routine broker upgrade caused a forty-minute outage. The difference between those two outcomes is almost never heroics. It is a small set of boring, learnable disciplines: throttle the data movement, shut brokers down gracefully, never take two replicas of one partition offline at the same time, spread replicas across failure domains, and watch one number — under-replicated partitions — like it owes you money.

This post is about day-2 operations: everything that happens *after* the cluster is built and serving traffic. Adding and removing brokers. Moving partitions around so a new machine actually earns its keep. Rolling a new broker version through without losing availability. Placing replicas so a single availability-zone outage cannot wipe out all copies of a partition. Throttling a runaway client so it cannot starve everyone else. Nudging leadership back into balance after a restart. And underneath all of it, the monitoring that tells you whether the operation you just kicked off is healing or hemorrhaging. By the end you will be able to plan and execute each of these on a live cluster, reason about how long they take and what they cost in live-traffic impact, and know the exact metric to watch so you abort before you cause an incident instead of after.

![A linear pipeline of a rolling upgrade for one broker moving through controlled shutdown, leadership migration to peers, binary swap, restart, and a verification that under-replicated partitions return to zero before the next broker](/imgs/blogs/cluster-operations-scaling-upgrades-partition-reassignment-1.webp)

I will use Kafka's vocabulary because it is the most widely operated of the log-style brokers and because its mechanisms — the in-sync replica set, the high watermark, the controller, the preferred leader — are the ones most worth understanding deeply. The ideas transfer. RabbitMQ quorum queues, Pulsar bookies, and NATS JetStream all face the same physics: data has to be copied before a node is useful, copying competes with live traffic for disk and network, and a node leaving must hand off its responsibilities cleanly or clients feel it. This is the third in a small arc of operational posts. It leans on [Kafka replication, the ISR, acks, and durability](/blog/software-development/message-queue/kafka-replication-isr-acks-durability) for what a replica and the ISR actually are, on [Partition count, sizing, and capacity planning](/blog/software-development/message-queue/partitioning-capacity-planning) for why you are adding brokers in the first place, and on [Securing message queues](/blog/software-development/message-queue/securing-message-queues-tls-authz-acls) for the authorization story that quotas and client identity build on. If you have not internalized what "under-replicated" means, read the replication post first; everything here is downstream of it.

## 1. Day-2 operations and the no-downtime constraint

There is a useful split in any system's lifecycle. Day 0 is design — you pick replication factor, partition counts, rack topology. Day 1 is provisioning — you stand the cluster up and point producers at it. Day 2 is *everything after that*, for the next five years, while the thing is serving real money-moving traffic and can never go fully dark. Day-2 operations are where the cluster spends 99.9% of its life, and they are governed by a single hard constraint that does not apply on days 0 or 1: **you may not stop serving traffic to perform them.**

That constraint sounds obvious and is deceptively expensive. It means every operation has to be incremental and reversible. You cannot "take the cluster down for maintenance" the way you might restart a single database in a quiet window, because a healthy message bus has no quiet window — it is the load-bearing wall between dozens of services, and stopping it stops all of them. So the entire craft of cluster operations is built around making big changes look, to clients, like nothing changed. A broker upgrade has to happen one machine at a time so the other machines keep serving. A 2 TB data move has to happen in the background at a rate slow enough that live producers never notice their bandwidth shrink. A failing node has to hand off its leaderships before it stops, not after, so clients re-route in milliseconds instead of timing out.

### The two things that actually break

In practice, day-2 operations cause production incidents in exactly two ways, and naming them up front gives you a checklist.

The first is **starving live traffic.** Every operation that moves bytes — adding a broker and reassigning data to it, healing a dead replica, rebalancing — competes for the same finite disk and network bandwidth that producers and consumers need. A reassignment with no throttle will happily consume every byte of a broker's 1 Gbps NIC copying old data, and your producers, which were comfortably using 600 Mbps of it, suddenly cannot get their writes through. Latency spikes, producer buffers fill, and `send()` starts timing out. The fix is a throttle, and we will spend a whole section on it.

The second is **taking out a quorum.** Every operation that takes a broker offline — an upgrade, a restart, a decommission — removes one replica of every partition that broker hosts. Do that to one broker and the partitions go from three replicas to two; annoying but safe. Do it to two brokers that happen to share replicas of the same partition, *at the same time*, and that partition drops below `min.insync.replicas` and stops accepting writes — or worse, if both of those were the only in-sync replicas, you are one failure away from data loss. The fix is sequencing: one broker at a time, and never start the next until the previous one is fully healed.

Figure 1 above is the whole no-downtime discipline compressed into one picture, applied to an upgrade: drain leadership, swap the binary, restart, *verify the cluster healed*, and only then touch the next machine. Hold that shape in your head. Almost every operation in this post is a variation on it.

### A map of the operations and their risk classes

Before we dive in, it helps to see all the day-2 operations side by side with the two things that actually distinguish them: whether they move bulk data (and therefore need a throttle) and whether they take replicas offline (and therefore risk availability). Figure 5 lays the four core operations against exactly those axes plus the downtime each one exposes when run correctly.

![A decision matrix scoring adding a broker, reassigning data, rolling upgrades, and leader election against their main risk, whether they need a throttle, and their downtime exposure](/imgs/blogs/cluster-operations-scaling-upgrades-partition-reassignment-5.webp)

Read the matrix and a pattern jumps out. Adding a broker is the safest operation in the catalog — done by itself it moves no data and carries no risk, which is exactly why it also relieves no load until you follow it with a reassignment. Reassigning data is the one operation that moves bulk bytes, so it is the only one that *needs* a throttle, and its main risk is starving live traffic. Rolling upgrades move no bulk data but take replicas offline, so their risk is availability — managed not by a throttle but by the one-at-a-time URP gate. Leader election moves only metadata; its only cost is a few milliseconds of re-route per partition. Internalize this table and you already know, for any operation you are about to run, which guardrail it needs. The rest of the post is just filling in the mechanics of each row.

### Why "without downtime" is a spectrum, not a binary

It is worth being honest that "no downtime" rarely means *zero* client-visible effect. When a leader migrates, the clients that were producing to it get one `NOT_LEADER_FOR_PARTITION` error and retry to the new leader — a sub-second blip, invisible if your producer is configured with retries (which it must be), but not literally zero. A throttled reassignment does slightly raise tail latency because the disk is busier. The goal is not metaphysical perfection; it is keeping every client-visible effect *inside the error budget your retry configuration already absorbs.* A well-run operation lives entirely under the noise floor of normal retries. A badly-run one pokes above it and pages someone.

This reframes what your job actually is during a maintenance operation. You are not trying to achieve the impossible — a change to a distributed system that no participant can detect. You are trying to keep the magnitude of every transient effect below the threshold at which a *client's own resilience* — its retries, its connection pool, its metadata refresh — silently absorbs it. That threshold is real and measurable. A producer with `retries=MAX_INT` and a sane `delivery.timeout.ms` will retry a `NOT_LEADER_FOR_PARTITION` so fast that the application above it never sees an error; the leadership migration that caused it is, from the application's vantage point, a non-event. The same migration done abruptly, with no controlled shutdown and no warning, blows past that threshold — the client waits out a full metadata refresh interval, the retry budget is consumed by timeouts rather than fast re-routes, and the blip becomes a stall the application *does* see. Same physical event, different magnitude, completely different outcome. Every technique in this post is, at bottom, a way of keeping the magnitude small.

There is a corollary worth stating plainly: **your clients must be configured for resilience for any of this to work.** A producer with `retries=0` will turn a routine leader migration into a hard error. A consumer with an aggressively short session timeout will declare itself dead during a brief rebalance. The cluster operator and the client owner share responsibility for "no downtime" — the operator keeps the blips small, and the clients are configured to swallow blips of that size. If your clients are not resilient, no amount of careful operation will hide a leadership change from them, and you will be blamed for an outage that is really a client-configuration bug. Audit client configs before you trust your own runbook.

## 2. Adding and removing brokers

Here is the single most common surprise for engineers new to operating a log cluster: **you add a broker, and nothing happens.** The new machine joins the cluster, registers itself, shows up in the broker list, and then sits there, completely idle, holding zero partitions and zero bytes. Your dashboards still show the original brokers at the same load. You did not scale anything. You added capacity that is doing nothing.

![A two-panel before-and-after diagram showing a newly added fourth broker sitting completely idle with zero partitions while the original three stay fully loaded, versus the same broker carrying a fair share after partition reassignment spreads replicas onto it](/imgs/blogs/cluster-operations-scaling-upgrades-partition-reassignment-2.webp)

This is by design, and once you understand why, it stops being a surprise and becomes a feature. A partition's replicas live on a specific, fixed set of brokers, recorded in the cluster metadata. That assignment does not change just because the broker population grew. New *topics* and new *partitions* created after the broker joins may be placed on it by the auto-placement logic, but existing partitions — which is to say, all your actual data — stay exactly where they were. The new broker is a blank disk waiting for an explicit instruction. Figure 2 shows both states: the idle broker on the left, holding nothing, and on the right the same broker carrying a fair share after you reassign replicas onto it.

### Scaling out is a two-step act

So scaling out is always two steps, and people who forget the second step file confused tickets about how "adding brokers didn't help."

1. **Provision and join the broker.** Give it a unique broker ID, point it at the cluster (the controller quorum in KRaft, or ZooKeeper on older clusters), start it. It joins in seconds.
2. **Reassign partitions onto it.** This is the part that moves data, takes time, and needs a throttle. It is the entire subject of section 3.

The reason for this separation is that data movement is the dangerous part, and the cluster refuses to do it implicitly. An automatic rebalance triggered by a broker joining would mean that booting a replacement node — something that happens during routine failure recovery — could kick off a terabyte-scale data shuffle at the worst possible moment. By making reassignment explicit and operator-driven, the system guarantees that bulk data movement only happens when a human (or an automated operator like Cruise Control) decided it should, with a throttle attached.

### Removing a broker is reassignment in reverse

Decommissioning is the mirror image. You cannot just turn a broker off and walk away — if you do, every partition it hosted drops a replica and goes under-replicated, and any partition for which it was the *only* remaining in-sync replica is now in serious trouble. To remove a broker cleanly you **reassign its partitions away first**, moving every replica it holds onto the brokers that will remain, and only once it holds zero partitions do you shut it down. At that point its departure is a non-event, because nothing depends on it.

There is a subtlety worth flagging: removing a broker reduces total capacity, so you must confirm the *remaining* brokers can absorb both the data and the leadership load before you start. Pulling a broker out of a cluster that was running at 80% disk utilization can push the survivors past their limits. Capacity-plan the shrink the same way you would plan the growth — the [capacity planning post](/blog/software-development/message-queue/partitioning-capacity-planning) covers the arithmetic.

A concrete way to sanity-check a shrink: take the broker's current disk usage, divide by the number of brokers that will remain, and add that to each survivor's current usage. If any survivor crosses your high-water mark (commonly 70% to 75% disk, leaving room for compaction, retention spikes, and the reassignment's own temporary double-storage), you do not have room to remove the broker — full stop. The same check applies to leadership: the leaderships the departing broker held must redistribute across the survivors, and if that pushes any survivor's leader count well above the cluster average, you will create a hot broker. The arithmetic is unglamorous but it is the difference between a clean decommission and turning a four-broker cluster into a three-broker cluster that immediately tips over.

One more failure mode specific to removal: if you shut a broker down *without* draining it first, the cluster will eventually re-replicate its partitions onto the survivors anyway — but as an *emergency* recovery, not a throttled, planned move, and at the worst time, because it is reacting to a broker that vanished rather than executing a plan you reviewed. The whole reason to drain explicitly is to convert an uncontrolled, unthrottled, reactive re-replication into a controlled, throttled, proactive one. Never decommission by just turning the machine off.

```bash
# Generate a reassignment plan that REMOVES broker 4 by listing only
# the brokers that will remain. The tool computes a new assignment that
# uses brokers 1,2,3 and drains broker 4.
cat > topics-to-move.json <<'EOF'
{"topics": [{"topic": "payments"}, {"topic": "clickstream"}], "version": 1}
EOF

kafka-reassign-partitions.sh \
  --bootstrap-server kafka1:9092 \
  --topics-to-move-json-file topics-to-move.json \
  --broker-list "1,2,3" \
  --generate
```

That `--generate` step does not move anything. It prints a *proposed* assignment (and the current one, which you save so you can roll back). You review it, save it to a file, and only then execute it — throttled. Generate, review, execute is the cadence for every reassignment, and skipping the review step is how people accidentally pile everything onto one broker.

## 3. Partition reassignment and throttling

Partition reassignment is the load-bearing operation of this entire post. Scaling out, scaling in, healing a permanently-dead broker, rebalancing a hot broker — all of them are partition reassignment underneath. And reassignment is, at its core, a *copy*: to move a replica of a partition from broker A to broker B, broker B must fetch the partition's entire log — every segment, possibly hundreds of gigabytes — from the current leader, append it to its own disk, catch up to the live tail, join the in-sync replica set, and only then can the old replica be dropped. There is no shortcut. The bytes have to travel.

![A branching dataflow diagram showing a reassignment plan dispatching two source-broker leaders through a fifty-megabyte-per-second throttle to a target broker, which then catches up and joins the in-sync replica set so under-replicated partitions clear](/imgs/blogs/cluster-operations-scaling-upgrades-partition-reassignment-6.webp)

Figure 6 shows the shape of a throttled move: a plan dispatches the partitions to relocate, the source brokers stream their logs through a rate cap, and the target broker ingests, catches up, and joins the ISR. The whole reason a throttle sits in the middle is the starvation problem from section 1. Without it, the replication fetchers doing the copy will use all available bandwidth, because they have no reason not to — and the live produce and fetch traffic, which shares that bandwidth, gets squeezed.

The end state of a reassignment is a rewritten map of which partition replicas live on which broker. Figure 3 makes that concrete: three brokers carrying nine partitions while a fourth sits empty, and then the same nine partitions redistributed so the fourth broker carries its fair share and every broker is lighter. The reassignment is precisely the act of getting from the top row to the bottom row, one partition's replica at a time, copying its log onto its new home before dropping the old copy.

![A grid showing nine partitions packed onto three brokers with a fourth broker empty, then the same partitions redistributed across all four brokers so each one carries a balanced share after reassignment](/imgs/blogs/cluster-operations-scaling-upgrades-partition-reassignment-3.webp)

Notice in Figure 3 that the redistribution is not a wholesale shuffle — most partitions stay exactly where they are, and only a few move. A good reassignment plan minimizes movement: it relocates just enough replicas to balance the load and leaves everything else untouched, because every moved replica is a full log copy that costs bandwidth and time. This is why you *review* the generated plan before executing it. A plan that needlessly relocates partitions that were already well-placed turns a one-hour move into an overnight one, and the only way to catch that is to read the proposed assignment and ask whether each move earns its bandwidth.

### How the throttle works

Kafka's reassignment throttle is a per-broker rate limit, expressed in bytes per second, applied specifically to replication traffic that is part of a reassignment. It has two halves: a `leader.replication.throttled.rate` (how fast a source broker may *serve* the data being moved off it) and a `follower.replication.throttled.rate` (how fast a target broker may *ingest* it). You set both, usually to the same value, when you execute the reassignment. Critically, the throttle applies *only* to the partitions being moved — normal replication of live data, the kind that keeps your existing replicas in sync, is never throttled, because throttling that would directly cause under-replication.

```bash
# Execute the reassignment with a 50 MB/s throttle (50000000 bytes/s).
# The throttle caps BOTH the source serving rate and the target ingest rate.
kafka-reassign-partitions.sh \
  --bootstrap-server kafka1:9092 \
  --reassignment-json-file reassignment.json \
  --throttle 50000000 \
  --execute

# Watch progress. "is complete" per partition means it joined the ISR.
kafka-reassign-partitions.sh \
  --bootstrap-server kafka1:9092 \
  --reassignment-json-file reassignment.json \
  --verify
```

That `--verify` step does two jobs: it reports which partitions have finished moving, and — importantly — it *removes the throttle from partitions that are done.* If you forget to run `--verify` after the reassignment completes, the throttle configs are left behind on the topics, silently rate-limiting future replication and setting up a baffling incident weeks later when a replica cannot catch up after a failure. Always verify to completion; it is the cleanup step, not just a status check.

### Picking a throttle: the headroom calculation

The throttle is not a number you guess. It is the *spare* bandwidth your cluster has after live traffic. Suppose each broker has a 10 Gbps NIC, which is about 1,250 MB/s of theoretical bandwidth, and in practice you plan to about 70% of that for safety, call it 875 MB/s usable. Your live produce-plus-replicate-plus-consume traffic at peak uses, say, 600 MB/s of that. Your headroom is 275 MB/s. You do *not* set the throttle to 275 — you leave a margin for traffic spikes and pick something like 50 to 100 MB/s, accepting a slower move in exchange for never touching live traffic's headroom. The slower you go, the safer; the only cost is wall-clock time, and wall-clock time on a background operation is cheap compared to a latency incident. When in doubt, throttle lower and let it run overnight.

A subtlety that trips people up: bandwidth is not the only resource a reassignment consumes — *disk I/O* is the other, and on spinning disks or saturated SSDs it is often the real bottleneck. Reading old log segments off the source disk and writing them to the target disk competes with the page-cache-missing reads that serve lagging consumers and with the sequential writes of live produce traffic. A throttle expressed in network bytes per second does bound the disk work indirectly (you cannot write to disk faster than you pull over the network), but on a disk-bound broker you may need to throttle *lower* than the network headroom alone would suggest, because the disk gives out before the NIC does. The signal to watch is the broker's disk utilization and I/O wait during the move; if they climb toward saturation, lower the throttle regardless of what the network graph says.

Another consideration is the *direction* of the headroom check. The throttle's leader half consumes bandwidth and disk on the *source* brokers; the follower half consumes them on the *target*. A brand-new broker you just added is empty and idle, so its ingest side has plenty of headroom — but the source brokers serving the data are your busy, fully-loaded existing machines, and *they* are where starvation happens. So when you size a throttle for a scale-out, the binding constraint is almost always the source brokers' spare capacity, not the empty target's. Check the headroom on the machines you are copying *from*, because they are the ones already carrying live traffic.

#### Worked example: reassigning 2 TB at a 50 MB/s throttle

Let us put real numbers on a move. You are adding a fourth broker to a three-broker cluster and need to relocate enough replicas onto it that it carries a fair share. The data to relocate totals 2 TB — that is 2,000,000 MB.

At a 50 MB/s throttle, the raw copy time is straightforward division:

```
2,000,000 MB / 50 MB/s = 40,000 seconds = 11.1 hours
```

So a single-stream view says about eleven hours. But reassignment moves many partitions, and the throttle is *per broker*, not per partition — if the 2 TB is spread across replicas being served by two source brokers, both serve in parallel, each capped at 50 MB/s, so the aggregate is 100 MB/s and the move finishes in roughly 5.5 hours. The arithmetic depends on how many distinct source brokers feed the copy; more sources, more parallelism, faster completion, but also more total bandwidth consumed, so the per-broker cap is what keeps any one machine safe.

Now the live-traffic impact. The whole point of the 50 MB/s cap is that it is *small* relative to the broker's capacity. If each source broker has 875 MB/s usable and is using 600 MB/s for live traffic, adding a 50 MB/s throttled stream brings it to 650 MB/s — well under the 875 ceiling, with 225 MB/s of headroom still untouched. Producers and consumers see essentially nothing: maybe a barely-perceptible bump in p99 fetch latency from the disk being slightly busier. That is the trade you wanted — eleven hours of patience in exchange for zero customer impact. Had you run unthrottled, the copy might have finished in 40 minutes, but during those 40 minutes the source brokers would have been pinned at 875 MB/s, live producers would have been starved of 275 MB/s they needed, `send()` calls would time out, and you would have caused exactly the incident the throttle exists to prevent.

The lesson in one line: **reassignment time is a dial you turn with the throttle, and you should almost always turn it toward "slow and invisible."**

| Throttle | 2 TB via 1 source | 2 TB via 2 sources | Live-traffic impact |
| --- | --- | --- | --- |
| Unthrottled | ~40 min | ~20 min | Severe — starves producers |
| 200 MB/s | ~2.8 hr | ~1.4 hr | Noticeable p99 bump |
| 50 MB/s | ~11 hr | ~5.5 hr | Essentially invisible |
| 10 MB/s | ~55 hr | ~28 hr | None, but very slow |

### Reassignment moves leaders, too — be careful

One trap: a reassignment plan can change not just *where* replicas live but *which replica is the leader.* If your new assignment lists broker 4 first for a partition, broker 4 becomes the preferred leader, and once the cluster runs a preferred-leader election (section 8), leadership migrates there. That is usually fine, but if you reassign hundreds of partitions and they all suddenly want to lead from the brand-new broker, you can dump a leadership avalanche onto a machine that has not warmed its page cache yet, and its first few minutes of serving are slow. Stage leadership changes separately from data moves when the cluster is large.

## 4. Rolling upgrades without losing availability

A rolling upgrade is how you move a running cluster from one broker version to the next — say 3.7 to 3.8 — without ever having the cluster fully down. The principle is simple and the discipline is everything: **upgrade one broker at a time, and never have two brokers down simultaneously**, because two-down can drop a shared partition below its in-sync minimum.

![A timeline of a six-broker rolling upgrade advancing from broker one to broker six, each step gated on under-replicated partitions returning to zero before the next broker, finishing with a protocol version bump after every broker is on the new release](/imgs/blogs/cluster-operations-scaling-upgrades-partition-reassignment-4.webp)

Figure 4 lays out the sequence across six brokers. Notice it is strictly serial, and notice the gate between each step: *under-replicated partitions back to zero.* That gate is the entire safety mechanism. You do not advance on a timer; you advance on a metric.

### The two compatibility knobs that make rolling upgrades possible

Rolling upgrades work because brokers of different versions can talk to each other, *temporarily*, during the upgrade window. Two settings govern this and you must understand them or you will get it wrong:

- **`inter.broker.protocol.version`** (IBP) — the version of the protocol brokers use to talk to *each other*. During the upgrade you leave this pinned to the *old* version so that a new-version broker still speaks the old broker-to-broker protocol that the not-yet-upgraded brokers understand.
- **`log.message.format.version`** (on older clusters) — the on-disk message format. Bumping this too early means old brokers cannot read messages the new broker wrote.

The procedure, therefore, is:

1. Roll the new *binary* through all brokers one at a time, with both compatibility versions still pinned to the old value. Every broker is now running new code but speaking the old protocol — fully interoperable.
2. Once *every* broker is on the new binary and healthy, do a *second* rolling restart that bumps `inter.broker.protocol.version` to the new value.
3. Finally, if relevant, bump the message format version.

This two-pass structure is the part people skip and regret. If you bump the protocol version on broker 1 before broker 2 is upgraded, broker 1 starts speaking a dialect broker 2 cannot follow, replication between them breaks, and you have created under-replication during the very operation that was supposed to be invisible. **New binary first, protocol version second, and only after the binary is everywhere.**

The reason this works at all is a deliberate compatibility guarantee in the broker's design: a newer broker binary can always speak an older inter-broker protocol. That backward compatibility is what lets the cluster run in a *mixed-version* state for the duration of the binary roll — for a window of minutes to an hour, you genuinely have brokers on 3.7 and 3.8 in the same cluster, replicating to each other, electing leaders, serving clients, all because every 3.8 broker is politely speaking 3.7's broker-to-broker protocol. The protocol version is the explicit switch that says "everyone is now new enough; start using the new dialect." You flip it only when that statement is true for *every* broker, which is why it is a second pass and not a per-broker setting you bump as you go.

A practical note on *clients* during an upgrade: clients (producers and consumers) talk to brokers over a separate, independently-versioned client protocol that is also backward and forward compatible across a wide range. You generally do not need to upgrade clients in lockstep with brokers — a 3.5-era client talks fine to a 3.8 broker and vice versa, within the documented compatibility window. This decoupling is what makes broker upgrades a *cluster-operator* concern that does not require coordinating a fleet-wide client redeploy. Upgrade the brokers on the operator's schedule; let client teams upgrade their libraries on theirs. The two only need to meet when you want a brand-new client feature that requires a newer broker, and even then the old clients keep working.

```properties
# server.properties DURING the binary roll (every broker), upgrading 3.7 -> 3.8.
# Keep the protocol pinned to the OLD version while binaries are mixed.
inter.broker.protocol.version=3.7
# log.message.format.version is deprecated on modern Kafka; on older clusters
# you would also pin it here, then bump it last.

# AFTER every broker is on the 3.8 binary and URP=0, do a second rolling
# restart with this line changed to 3.8, which lights up new wire features.
# inter.broker.protocol.version=3.8
```

### The downgrade question

Why keep the protocol pinned to old for a whole extra pass? Because it preserves your **downgrade path.** While the protocol version is still old, any broker that misbehaves on the new binary can be rolled back to the old binary safely — the on-disk and on-wire formats are unchanged. The moment you bump the protocol version, you have crossed a one-way door; downgrading now risks brokers being unable to read data written under the new protocol. So the discipline is: get fully onto the new binary, *soak it* under real traffic for a while (hours to a day, depending on your risk tolerance), confirm nothing regressed, and only then bump the protocol version and burn the downgrade bridge. Treat that bump as a separate, deliberate change, not part of the same maintenance window.

## 5. Controlled shutdown and leadership migration

Now zoom into a single broker in that rolling upgrade. When it is broker 3's turn, you do not `kill -9` it. You trigger a **controlled shutdown**, and the difference between a controlled and an uncontrolled stop is the difference between a sub-second blip and a multi-second stall for every client touching that broker's partitions.

A broker, remember, is the leader for some partitions and a follower for others. The partitions it *leads* are the problem during a stop, because all client traffic for a partition goes to its leader. If you yank the broker away abruptly, here is what clients experience: they keep sending to the dead leader, get connection errors or timeouts, and only after their metadata refresh interval do they discover a new leader has been elected and re-route. That window — the time between "leader vanished" and "client found the new leader" — can be several seconds of failed requests, multiplied across every partition the broker led.

### What controlled shutdown actually does

Controlled shutdown front-loads all of that. When a broker receives a controlled-shutdown request, before it stops it asks the controller to **migrate leadership of every partition it leads to another in-sync replica**, proactively. The controller picks healthy followers, makes them leaders, and propagates the new metadata to clients — all while the old leader is still alive and able to serve. By the time the broker actually stops, it leads *nothing*; every partition has already failed over cleanly to a replica that was fully caught up. Clients see the leadership change as a normal metadata update, retry once, and carry on. The result is a leadership migration measured in milliseconds per partition instead of multi-second timeouts.

This is the `migrate leaders to peers` step in Figure 1, and it is why that step comes *before* the binary swap, not after.

```properties
# server.properties — make controlled shutdown robust.
controlled.shutdown.enable=true
# How many times the broker retries the controlled-shutdown handshake with
# the controller before giving up and shutting down hard. Bump this so a
# transient hiccup doesn't fall back to an ungraceful stop.
controlled.shutdown.max.retries=5
controlled.shutdown.retry.backoff.ms=5000
```

### Why "never two replicas at once" lives here

Controlled shutdown migrates leadership, but it does not magically create new replicas — when broker 3 stops, every partition it hosted (whether as leader or follower) is down one replica until it returns. That is the precise reason the rolling upgrade is serial. If broker 3 and broker 5 share replicas of partition P and you stop both at once, P loses two replicas. With replication factor 3 and `min.insync.replicas=2`, P now has one in-sync replica, which equals the minimum, which means a single further hiccup on that last replica stops writes to P entirely. Stop a third overlapping broker and you may have lost the partition. The URP-zero gate between brokers exists to guarantee the previous broker's replicas are *fully restored* — back in the ISR, no longer under-replicated — before you remove the next replica from the pool. One down at a time, healed before the next.

#### Worked example: rolling upgrade of a six-broker cluster

Let us walk the full window for a real six-broker cluster, replication factor 3, `min.insync.replicas=2`, carrying live payments traffic. The plan in Figure 4 made it look tidy; here is the operator's actual minute-by-minute.

**Pre-flight.** Confirm under-replicated partitions is already 0. If it is not, the cluster is unhealthy *before* you start, and you fix that first — never begin a rolling upgrade on a cluster that is already under-replicated, because you cannot tell your operation's URP from the pre-existing URP. Confirm `min.insync.replicas` is 2, not 3 — with RF=3 and min-ISR=3, taking even one broker down halts writes, and a rolling upgrade is impossible. (This is a classic foot-gun; min-ISR must be at most RF minus one to survive a single broker being down.)

**Per-broker loop, brokers 1 through 6:**

1. Trigger controlled shutdown on the broker. Leadership of its ~partitions migrates to peers over a few seconds. Clients retry once and re-route.
2. Swap the binary from 3.7 to 3.8, keeping `inter.broker.protocol.version=3.7`.
3. Restart the broker. It rejoins, and its replicas — which fell behind while it was down — start catching up. **Under-replicated partitions climbs**, because those replicas are temporarily out of the ISR.
4. **Wait for URP to return to 0.** This is the gate. The broker was down maybe 90 seconds; catching up the data it missed during that window takes another minute or two depending on traffic. Call it ~8 minutes per broker end to end, generously.
5. Only now, advance to the next broker.

Six brokers at roughly 8 minutes each is about 48 minutes for the binary pass — call it the ~50 minutes shown at the end of Figure 4's timeline. **Upgrade the controller (or the active KRaft controller) last**, so the metadata plane stays stable while the data brokers churn.

**Second pass.** After the cluster has soaked on the 3.8 binary — you might wait a day, not minutes, for a payments cluster — you do a second rolling restart bumping `inter.broker.protocol.version=3.8`. Same one-at-a-time discipline, same URP gate, another ~50 minutes. Total *active* operator time across both passes is under two hours, spread over whatever soak period your risk tolerance demands.

The number that matters is not the 50 minutes. It is that under-replicated partitions touched a non-zero value six times — once per broker — and returned to zero six times, and at no point did any partition drop below two in-sync replicas. That is what "without losing availability" looks like as a measurement, and the URP graph is the proof you did it right.

## 6. Rack awareness and failure domains

Everything so far assumed replicas are spread across machines. But machines share fate. Two brokers in the same rack share a top-of-rack switch and a power strip; two brokers in the same cloud availability zone share data-center-level infrastructure. If all three replicas of a partition happen to land in the same failure domain, then a single rack power failure or a single AZ outage takes down all three at once — and no amount of replication factor saved you, because replication factor only helps if the replicas fail *independently.*

![A layered stack diagram showing a single partition with replication factor three placing its leader and two followers across three distinct availability zones, so the in-sync replica set survives the loss of any one zone](/imgs/blogs/cluster-operations-scaling-upgrades-partition-reassignment-8.webp)

Figure 8 shows the goal: the three replicas of one partition placed in three different availability zones, so the loss of any one zone leaves two replicas serving. **Rack awareness** is the feature that makes the cluster place replicas this way automatically. You tag each broker with its failure domain via `broker.rack`, and the replica-placement logic then spreads each partition's replicas across as many distinct racks as it can before doubling up.

```properties
# server.properties — tell this broker which failure domain it lives in.
# Set this to the cloud AZ for cloud deployments, or the physical rack
# for on-prem. The placement logic spreads replicas across distinct values.
broker.rack=us-east-1a
```

### What rack awareness guarantees, and what it does not

When `broker.rack` is set on every broker, both initial topic creation *and* reassignment honor it: a partition with replication factor 3 across three racks gets exactly one replica per rack. With replication factor 3 across only two racks, it gets two in one and one in the other — still better than three in one. The guarantee is "spread as evenly as the rack count allows," not "magically more racks than you have."

There is one more guarantee that rack awareness quietly provides and that pays off during the very operations this post is about: it makes the cluster's *placement* rack-balanced, which in turn makes a *whole-rack* maintenance event survivable. If you ever need to take an entire rack or availability zone down for maintenance — a data-center power test, a kernel upgrade across a zone's hosts — a rack-aware cluster guarantees that doing so removes at most one replica of every partition, so the partitions stay above their in-sync minimum. Without rack awareness, taking a rack down could remove two or three replicas of some partitions at once, the exact two-down condition the entire rolling-upgrade discipline exists to avoid, except now at rack granularity. Rack awareness is therefore not only an outage-survival feature; it is a *maintainability* feature that lets you operate on whole failure domains at a time.

The payoff is concrete and large.

![A two-panel before-and-after diagram contrasting a partition whose three replicas all landed in one availability zone being lost entirely when that zone fails, against a rack-aware placement where the same zone outage leaves two replicas still serving](/imgs/blogs/cluster-operations-scaling-upgrades-partition-reassignment-9.webp)

Figure 9 makes the stakes unambiguous. On the left, without rack awareness, an unlucky placement put all three replicas of partition P7 in availability zone 1a; when 1a has an outage — and zones do have outages, this is a *when* — P7 is simply gone, offline, until the zone recovers, and if the outage destroyed disks, gone for good. On the right, rack-aware placement spread P7 across three zones, so the same 1a outage leaves two replicas in the surviving zones still serving reads and writes without interruption. Same replication factor, same hardware, wildly different outcome — the only difference is whether `broker.rack` was set before the partitions were placed.

### The cross-AZ bandwidth bill

Rack awareness is not free, and pretending otherwise gets you a surprise invoice. Replication is cross-rack by design now, which in a cloud means cross-AZ, which means you pay inter-AZ data-transfer charges on every replicated byte. For a high-throughput cluster this is a real line item — replication traffic is typically (RF − 1) times your produce traffic, all of it now crossing AZ boundaries. The mitigation is follower fetching: configure consumers to read from the *nearest* replica (`rack.id` on the consumer plus `replica.selector.class` on the broker) so at least the *consume* side stays in-zone. But the *replication* side fundamentally must cross zones for the durability you wanted. This is the price of surviving an AZ outage, and for most systems carrying important data it is unambiguously worth paying. The relationship between replicas, failure domains, and independence is the same one explored generally in [distributed replication](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless) — rack awareness is that theory made concrete in a placement constraint.

## 7. Quotas to protect the cluster

Every operation so far protected the cluster from things *you* do. Quotas protect it from things your *clients* do — specifically, from a single misbehaving client consuming so much of a broker's capacity that everyone else suffers. A multi-tenant cluster is a commons, and commons get destroyed by one greedy participant unless there is a mechanism to bound each one.

The classic incident: a batch job is deployed with a bug, or a consumer falls catastrophically behind and then tries to catch up by fetching as fast as the broker will serve, and suddenly one client is pulling 800 MB/s off a broker, saturating its NIC, and every other producer and consumer on that broker sees their latency triple. The cluster is technically up, but it is effectively down for everyone sharing that broker with the runaway. Quotas prevent this by capping how much any one client can do.

### The three quota types

Kafka enforces quotas per *client* (identified by client ID, or by authenticated principal when security is on — which ties directly into the identity story in [securing message queues](/blog/software-development/message-queue/securing-message-queues-tls-authz-acls)), and there are three kinds:

- **Produce quota** — bytes per second a client may write. Exceed it and the broker delays the client's produce responses, applying backpressure.
- **Fetch quota** — bytes per second a client may read. This is the one that contains the runaway-catch-up consumer.
- **Request quota** — a percentage of broker request-handler and network-thread time a client may consume, which catches clients that hammer the broker with many small, cheap-looking requests that are individually under the byte quotas but collectively pin the CPU.

The enforcement mechanism is elegant: the broker does not reject over-quota requests, it *delays the response*. A producer over its quota gets its acknowledgement back a little later, which naturally slows it down without erroring — the client's in-flight limit fills, and it stops sending faster. It is backpressure, applied at the broker, and it composes cleanly with the producer-side flow control covered in the broader queueing literature.

```bash
# Cap client 'batch-reporter' at 10 MB/s produce and 10 MB/s fetch.
# Over-quota requests are DELAYED (throttled), not rejected.
kafka-configs.sh --bootstrap-server kafka1:9092 --alter \
  --add-config 'producer_byte_rate=10485760,consumer_byte_rate=10485760' \
  --entity-type clients --entity-name batch-reporter

# A default quota for ALL clients without a specific one — the safety net
# that bounds a tenant you forgot to configure explicitly.
kafka-configs.sh --bootstrap-server kafka1:9092 --alter \
  --add-config 'producer_byte_rate=52428800,consumer_byte_rate=52428800' \
  --entity-type clients --entity-default
```

That second command is the one that saves you. **Set a default quota** so that a tenant nobody configured cannot exceed it. The most common quota mistake is having quotas only on the clients you remembered, leaving an unbounded hole for the one you forgot — which, by Murphy's law, is the one that ships the bug. A sane default with explicit overrides for trusted high-volume clients is the pattern that survives contact with reality.

Quotas interact with the operations in this post in a way worth calling out. During a reassignment, the *throttle* protects the cluster from the data-movement traffic, but *quotas* protect it from the clients — and the two are independent mechanisms aimed at different sources of load. It is entirely possible to have a perfectly throttled reassignment running smoothly while an unquoted client simultaneously saturates a broker; the throttle did its job and the client still caused an incident. This is why quotas are not a "nice to have you set up someday" but a standing guardrail that should already be in place before you ever run a maintenance operation, so that the cluster's headroom calculation — the one you used to size the throttle — is *enforced* rather than merely assumed. If your headroom math assumed clients use 600 MB/s but nothing stops a client from using 900, your throttle was sized against a fiction.

#### Worked example: containing a runaway catch-up consumer

Walk a concrete quota save. A consumer group falls four hours behind because of a deploy bug, accumulating tens of millions of unprocessed messages. The bug is fixed and the group restarts, and now it wants to catch up — so it fetches as fast as the brokers will serve. With no fetch quota, that group pulls 850 MB/s off a single broker that is hosting the partitions it reads, saturating that broker's NIC. Every other consumer reading from the same broker sees fetch latency climb from 5 ms to over a second, and producers writing to that broker's lead partitions see their acknowledgements delayed because the broker's network threads are pinned serving the runaway. One client's recovery becomes everyone's incident.

Now the same scenario *with* a 50 MB/s fetch quota on that consumer group. The group requests data as fast as ever, but the broker delays each over-quota fetch response so the group's effective throughput is capped at 50 MB/s. Catching up four hours of backlog at 50 MB/s takes longer — if the backlog is, say, 500 GB, that is 500,000 MB divided by 50 MB/s, about 10,000 seconds or just under three hours to drain. Slower for the lagging group, yes. But the broker's other 800 MB/s of capacity stays available to every other client, who notice nothing. The quota traded one tenant's recovery speed for everyone else's stability, which is exactly the trade a shared cluster should make. The runaway is contained, not by rejecting it, but by pacing it.

## 8. Preferred-leader election and rebalancing leadership

After a rolling upgrade, after a broker restart, after any operation that moved leadership around, you will find your leadership *imbalanced*. Recall from section 5 that a controlled shutdown migrates a broker's leaderships to its peers. When that broker comes back, those leaderships do *not* automatically come home — the followers that got promoted stay leaders. Roll all six brokers through an upgrade and you can end up with leadership piled onto whichever brokers happened to be up at the wrong moments, while the freshly-restarted ones lead almost nothing. Now your load is lopsided: a few brokers handle most of the client traffic and run hot, while others idle.

The fix is **preferred-leader election.** Every partition has a *preferred* leader — the first broker in its replica list, chosen at assignment time precisely to balance leadership evenly across the cluster. After leadership has drifted, you tell the cluster to run an election that restores each partition's leadership to its preferred replica (when that replica is in the ISR and healthy). Leadership snaps back to the balanced distribution it was assigned.

```bash
# Restore every partition's leadership to its preferred (balanced) replica.
# Each individual leader move is a few-millisecond metadata change; clients
# retry once and re-route, just like during a controlled shutdown.
kafka-leader-election.sh \
  --bootstrap-server kafka1:9092 \
  --election-type PREFERRED \
  --all-topic-partitions
```

### Automatic vs manual rebalancing

There is a setting, `auto.leader.rebalance.enable`, that makes the controller run preferred-leader elections automatically when leadership imbalance crosses a threshold (`leader.imbalance.per.broker.percentage`). On steady-state clusters this is convenient — leadership self-heals after restarts. But many seasoned operators *disable* it for a sharp reason: an automatic election firing in the middle of *another* operation, like a reassignment, can cause leadership to migrate onto a broker that is busy catching up or has a cold page cache, producing a latency spike at the worst time. The conservative posture is to disable automatic rebalancing and run preferred-leader election *manually*, as a deliberate final step after upgrades and reassignments, when you can watch its effect. Each leader move is a sub-millisecond metadata change and clients absorb it with a single retry — the same cheap re-route that happens during controlled shutdown — so doing it on purpose at a controlled moment is nearly free and far safer than letting it fire unpredictably.

Figure 7 places preferred-leader election in the broader taxonomy: it is the *leadership* family of operations, distinct from scaling, upgrades, and data rebalance, and its safety mechanism is that leader moves are cheap metadata changes rather than data copies.

![A hierarchy tree classifying cluster operations into scaling, upgrade, data rebalance, and leadership families, each branch ending in its concrete operation and the safety control that governs it](/imgs/blogs/cluster-operations-scaling-upgrades-partition-reassignment-7.webp)

That taxonomy is worth internalizing because it tells you, for any operation, which risk class it belongs to. Scaling and data rebalance *move bytes* and therefore need throttles. Upgrades *take replicas offline* and therefore need the one-at-a-time URP gate. Leadership operations move only metadata and are cheap — their only risk is timing, not bandwidth. Knowing which family an operation belongs to tells you immediately which safety mechanism it needs.

## 9. Monitoring during operations

You have noticed a refrain: after every operation, *watch under-replicated partitions.* This is not incidental. **Under-replicated partitions (URP) is the single most important health metric for a broker cluster, and it is the number-one signal during any operation.** It counts, across the whole cluster, how many partition replicas are *not* currently caught up with their leader. In a healthy steady state it is exactly 0. Any non-zero value means at least one partition is running with fewer in-sync replicas than it should, which means reduced durability and reduced fault tolerance for that partition.

During operations, URP is your gating signal. You *expect* it to go non-zero — a restarted broker's replicas fall behind and must catch up, briefly under-replicating. What you watch for is the *shape* of the curve: it should spike and then **decline back to zero** within a few minutes as the replica catches up. A URP that climbs and *keeps climbing*, or plateaus and refuses to come down, means the operation is not healing — replicas cannot catch up, perhaps because the throttle starved them or because a second broker is also struggling — and that is your signal to *stop and investigate* before advancing. URP returning to zero is the green light for the next step; URP refusing to return is the abort signal.

### The supporting metrics

URP is the headline, but a few others complete the picture during operations:

| Metric | Healthy | What a bad value means during ops |
| --- | --- | --- |
| Under-replicated partitions | 0 | Replicas behind; do not advance until it's 0 again |
| Offline partitions | 0 | A partition has *no* leader — actively unavailable, page now |
| ISR shrink/expand rate | low, stable | Replicas thrashing in and out of the ISR; instability |
| Active controller count | exactly 1 | 0 means no controller (cluster brain dead); 2 means split brain |
| Request handler idle % | > 20% | Brokers CPU-saturated; an operation may be overloading them |
| Leader imbalance % | low | Leadership lopsided; run preferred-leader election |

**Offline partitions** deserves a special call-out because it is strictly worse than under-replicated. An under-replicated partition is *up* but fragile — it still has a leader, still serves reads and writes, just with thinner redundancy. An *offline* partition has no leader at all and is *unavailable* — producers and consumers for it are getting errors right now. Offline partitions during an operation means you have taken out too many replicas of something and lost its leader entirely; it is a stop-everything, page-immediately condition. The distinction matters: URP is a "watch and gate" signal, offline-partitions is a "you already broke it" signal.

**Active controller count** is the one that catches the rare but catastrophic failures. Exactly one broker should be the active controller. Zero means the cluster has no brain and no metadata changes can happen — no leader elections, no reassignment progress. Two means split brain, which should be impossible in a correctly-running cluster and indicates something has gone badly wrong with the controller quorum. Always have an alert on `active controller count != 1`.

## 10. A note on KRaft versus ZooKeeper

For a decade, Kafka's cluster metadata — which broker leads which partition, the ISR membership, topic configs — lived in an external ZooKeeper ensemble. Modern Kafka replaces ZooKeeper with **KRaft**, where the metadata lives in an internal Raft-replicated log managed by a quorum of *controller* brokers. ZooKeeper is deprecated and removed in current releases, so new clusters are KRaft, but plenty of running clusters still use ZooKeeper, and the operational differences matter for the operations in this post.

The biggest practical difference is **failover speed and scale.** Under ZooKeeper, the controller kept cluster metadata in memory and propagated changes to brokers individually; when the controller failed, a new one had to *reload all metadata from ZooKeeper*, which on a large cluster with hundreds of thousands of partitions could take many seconds to minutes — a real availability gap during the very controller failover that is supposed to be transparent. Under KRaft, metadata *is* a replicated log that the controller quorum already holds; a controller failover is a fast Raft leader election with no metadata reload, so it completes in well under a second even on huge clusters. For the operations here, that means leadership migrations and reassignment metadata updates propagate faster and more reliably, and a controller dying mid-operation is far less disruptive.

The operational *procedures* are mostly unchanged: you still add brokers and reassign, still roll upgrades one at a time with the URP gate, still set rack awareness and quotas, still run preferred-leader election. The commands (`kafka-reassign-partitions.sh`, `kafka-leader-election.sh`, `kafka-configs.sh`) work the same. What changes is the substrate beneath them. On KRaft you operate a *controller quorum* — typically three or five dedicated controller nodes — and during a rolling upgrade you treat the controllers as their own one-at-a-time upgrade group, just like the data brokers, and you upgrade the *active* controller last. There is no separate ZooKeeper ensemble to operate, patch, and reason about, which removes an entire class of operational toil and an entire failure mode (ZooKeeper session expiry causing spurious controller churn). If you are starting fresh, start on KRaft; if you are operating ZooKeeper, the most valuable upgrade on your roadmap is migrating off it.

## Case studies and war stories

These are composites drawn from the kind of incidents that recur across teams operating log clusters. The numbers are representative; the failure modes are exact.

### The unthrottled reassignment that took down checkout

A team adds two brokers to an eight-broker cluster ahead of a sales event and kicks off a reassignment to spread load onto them. They run it the obvious way — generate a plan, execute it — and forget the `--throttle` flag. With no throttle, the replication fetchers on the source brokers immediately saturate their 10 Gbps NICs copying historical segments to the new brokers. The cluster's live produce traffic, which had been comfortably using about 65% of available bandwidth, is suddenly squeezed into the scraps. Producer `send()` latency p99 jumps from 8 ms to 4 seconds, producer buffers fill, and the checkout service — which produces an event per order — starts dropping requests because its producer cannot flush. The "capacity-adding" operation caused a partial checkout outage *during the exact event it was meant to prepare for.* The fix in the moment was to abort the reassignment, let the cluster recover, and re-run it that night with a 30 MB/s throttle, which finished in six hours with zero customer impact. **The lesson: an unthrottled reassignment is not a faster reassignment, it is an incident.** Make the throttle flag non-optional in your runbook — a reassignment script with no `--throttle` should refuse to run.

### Two brokers, one rack, one power failure

A cluster runs replication factor 3 and the team is proud of it — three copies of everything, surely safe. Nobody set `broker.rack`. The placement logic, having no rack information, distributed replicas to balance disk and leadership but had no reason to avoid co-locating replicas in the same physical rack. For a meaningful fraction of partitions, two of the three replicas happened to land on brokers sharing one top-of-rack switch. A switch firmware bug takes that rack offline. Every partition with two replicas in that rack instantly drops to one in-sync replica; with `min.insync.replicas=2`, those partitions stop accepting writes. A "replication factor 3" cluster experiences a write outage from a *single rack* failure, because the three copies were not failure-independent. The post-incident fix was to set `broker.rack` on every broker and run a full reassignment to spread replicas across racks — a multi-day throttled operation, done belatedly under the pressure of a near-miss. **The lesson: replication factor is a count of copies, not a guarantee of independence. Rack awareness is what converts copies into fault tolerance, and it has to be set before placement, not after the outage.**

### The min-ISR foot-gun that blocked an upgrade

An operator starts a routine rolling upgrade. They trigger controlled shutdown on the first broker — and immediately, producers across a swath of topics start getting `NOT_ENOUGH_REPLICAS` errors and writes stall. The cause: someone had set `min.insync.replicas=3` on a replication-factor-3 cluster, reasoning that "more in-sync replicas is more durable." It is, until you need to take *one* broker down — at which point every affected partition drops from 3 in-sync replicas to 2, which is below the minimum of 3, so writes halt. With min-ISR equal to RF, the cluster cannot tolerate *any* broker being offline, which makes rolling upgrades and even routine restarts impossible without a write outage. The fix was to lower `min.insync.replicas` to 2 (giving the standard RF=3, min-ISR=2 configuration that survives one broker down while still requiring a majority for durable writes), then proceed with the upgrade normally. **The lesson: `min.insync.replicas` must be at most replication-factor-minus-one, or you have built a cluster that cannot be operated without downtime.** Durability and operability are both real constraints; min-ISR is where you balance them, and RF minus one is almost always the right value.

### The forgotten throttle config that broke recovery weeks later

A team runs a textbook throttled reassignment. It completes. They move on — but they never run the `--verify` step that removes the throttle configuration from the affected topics. Weeks later, an unrelated broker dies and its replicas need to be re-replicated onto a replacement during normal recovery. The recovery crawls. Replicas that should catch up in minutes take hours, because the leftover throttle config from the old reassignment is silently rate-limiting the recovery replication to 30 MB/s. The cluster sits under-replicated and fragile far longer than it should during an actual failure, all because of a cleanup step skipped weeks earlier. **The lesson: a reassignment is not done when the data finishes moving; it is done when you run `--verify` and confirm the throttle configs are removed.** Better still, audit for stray `*.replication.throttled.*` configs as a periodic hygiene check.

## When to reach for this (and when not to)

Day-2 cluster operations are not optional — every running cluster needs them — but *how* you approach them depends on scale and risk.

**Do the operations in this post manually, with the CLI tools and the URP gate, when** your cluster is small enough to reason about (single digits to low tens of brokers), when an operation is rare (a few times a year), or when the change is high-stakes enough that you want a human watching every step. A six-broker payments cluster getting a quarterly upgrade is exactly this case: the operator's attention is the safety mechanism, and the manual cadence — one broker, watch URP, next broker — is both teachable and reliable. Manual operation is also how you *learn* these mechanisms; automate only what you already understand by hand.

**Reach for an automated operator (Cruise Control, Strimzi on Kubernetes, a managed service) when** the cluster is large (dozens to hundreds of brokers), when rebalancing is frequent enough that doing it by hand is a full-time job, or when you want continuous, goal-driven balancing rather than occasional big reassignments. Cruise Control in particular turns "move these partitions" into "keep the cluster balanced against this resource model," generating throttled reassignment plans automatically and respecting rack awareness. At scale, the *throttle and the URP gate are still the safety mechanisms* — the operator just chooses the moves for you and applies them continuously.

**Do not** treat any of these operations as fire-and-forget, at any scale. Even with an automated operator, the throttle, the rack constraints, the min-ISR setting, and the URP alert are the guardrails that keep automation from causing the incident a human would have caught. Automation changes *who decides the moves*; it does not change the physics that data movement competes with traffic and that taking replicas offline reduces fault tolerance. And do not perform two byte-moving operations at once — a reassignment and an upgrade overlapping multiply each other's risk and make the URP signal ambiguous. Sequence them. One thing at a time is not a beginner's crutch; it is how senior operators stay un-paged.

## Key takeaways

- **A new broker is idle until you reassign.** Adding a broker adds zero load relief on its own; partition reassignment is the second, mandatory step that actually spreads data onto it. Removing a broker is the same in reverse — drain its partitions first, then stop it.
- **Throttle every reassignment.** Data movement competes with live traffic for disk and network. A throttle (e.g. 50 MB/s) trades wall-clock time for invisibility; an unthrottled move is an incident, not a fast move. Always run `--verify` afterward to remove the throttle config.
- **Roll upgrades one broker at a time, gated on URP returning to zero.** Never take two brokers down at once — overlapping replicas can drop a partition below min-ISR. New binary first across all brokers, protocol-version bump second, only after the binary is everywhere, to preserve the downgrade path.
- **Use controlled shutdown so leadership migrates cleanly.** It moves a broker's leaderships to healthy peers *before* the broker stops, turning multi-second client timeouts into a single-retry, sub-second blip.
- **Set `broker.rack` before placement.** Replication factor counts copies; rack awareness makes those copies failure-independent so a single rack or AZ outage cannot take all replicas of a partition. It costs cross-AZ replication bandwidth, and it is worth it.
- **Set `min.insync.replicas` to at most RF minus one** (RF=3, min-ISR=2 is the standard), or the cluster cannot survive a single broker being down and rolling upgrades become impossible without a write outage.
- **Quota your clients, with a sane default.** A produce, fetch, and request quota bounds any one tenant from starving the cluster. The default quota is the safety net for the tenant you forgot to configure.
- **Run preferred-leader election as a deliberate final step** after upgrades and reassignments to rebalance lopsided leadership. Consider disabling automatic rebalancing so an election cannot fire mid-operation onto a cold broker.
- **Under-replicated partitions is the number-one signal.** It should spike and return to zero during operations; a URP that climbs or plateaus is your abort signal. Offline partitions and active-controller-count anomalies are stop-everything conditions.
- **Prefer KRaft.** Controller failover is sub-second versus seconds-to-minutes under ZooKeeper, and you lose an entire failure mode and operational burden by retiring the external ensemble.

## Further reading

- [Kafka replication, the ISR, acks, and durability](/blog/software-development/message-queue/kafka-replication-isr-acks-durability) — what a replica, the ISR, and the high watermark actually are; everything in this post is downstream of it.
- [Partition count, sizing, and capacity planning](/blog/software-development/message-queue/partitioning-capacity-planning) — why and when you add brokers, and the bandwidth arithmetic behind throttle and shrink decisions.
- [Securing message queues with TLS, authz, and ACLs](/blog/software-development/message-queue/securing-message-queues-tls-authz-acls) — the client-identity and authorization model that quotas build on.
- [Distributed replication: leader, multi-leader, leaderless](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless) — the general theory of single-leader replication and why replica independence is the thing that buys fault tolerance.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — the foundational model of the partitioned, replicated log these operations keep healthy.
- Apache Kafka documentation, "Operations" and "Upgrading" sections — the authoritative reference for `inter.broker.protocol.version`, reassignment, quotas, and the KRaft migration path.
- LinkedIn Cruise Control — an open-source operator that automates throttled, rack-aware rebalancing at scale, and a good model for what "goal-driven balancing" looks like in practice.
