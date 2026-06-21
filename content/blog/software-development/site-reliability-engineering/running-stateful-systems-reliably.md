---
title: "Running Stateful Systems Reliably: Why You Can't Treat a Database Like a Pod"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Stateless services are easy to operate; databases, queues, and caches are where reliability gets hard. Learn the stateful SRE playbook — replication lag as an SLI, safe failover with quorum and fencing, backpressure before the queue becomes the outage, and the 3am calculus where the safe move is to stop, not act."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "stateful-systems",
    "databases",
    "replication-lag",
    "failover",
    "backpressure",
    "split-brain",
    "rpo-rto",
    "on-call",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/running-stateful-systems-reliably-1.png"
---

There is a moment every on-call engineer remembers. A primary database flickers — a health check times out twice, an alert fires, and your automation, trying to be helpful, promotes a replica to take over. Within seconds the new primary is serving traffic and the page clears. You go back to sleep feeling like the system worked.

Then, at 9am, the support tickets start. Twelve hundred orders from a four-minute window are simply gone. Customers were charged; no order exists. And worse: when the old primary came back online a minute after the blip, it still thought *it* was the primary and kept accepting writes for ninety seconds before anyone noticed. Now you have two databases that disagree about reality, and untangling which writes are real is a forensic exercise that will eat the rest of your week.

This is the kind of outage that does not happen to a stateless web server. If you kill a stateless pod, the scheduler starts another one and traffic shifts in seconds; nothing is lost because there was nothing *in* the pod to lose. But a database, a message broker, a cache holding the only copy of a session — these carry **state**, and state has gravity, identity, and a single source of truth you cannot just restart away. The reflexes that make you a good stateless on-call engineer — *restart it, kill the bad replica, scale out, let the orchestrator heal it* — are exactly the reflexes that corrupt data at 3am. Figure 1 shows the asymmetry at the heart of this post: the same operation, "restart the node," is a cheap shrug for a stateless pod and a potential data-loss event for a stateful one.

![A side by side comparison showing that restarting a stateless pod loses no state and recovers in seconds while restarting a stateful database node can lose the un-replicated tail and take 20 to 60 minutes to recover](/imgs/blogs/running-stateful-systems-reliably-1.png)

This post applies the SRE lens — the define, measure, budget, respond, learn, engineer loop from the [intro to the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — to the systems where it is hardest: the data tier. I am not going to re-derive how replication or consensus *work*; the [database series on leader, multi-leader, and leaderless replication](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless) and the [message-queue series on backpressure and flow control](/blog/software-development/message-queue/backpressure-and-flow-control) already do that, and I will point you there. This post is about how to *run* these systems reliably: what you measure, what you alert on, how you fail over without losing data, how you keep a queue from becoming the outage, and what changes about your on-call calculus when a wrong move is permanent instead of transient. By the end you will have a replication-lag SLI and alert, a stateful-failover runbook, a queue-depth alert, and a clear decision rule for the 3am moment that decides whether you lose data or not.

## 1. Why "stateful" is a different operational category

Let me define the terms so a mid-level engineer who has only ever restarted servers can follow along. A **stateless** service holds no durable data of its own between requests — it reads everything it needs from somewhere else (a database, a cache) and writes everything back. Any instance can serve any request; instances are interchangeable, like the cashiers at a supermarket who all share one inventory system. A **stateful** system *is* the inventory system. It is the single place where the truth lives. Databases (Postgres, MySQL, Cassandra), message brokers (Kafka, RabbitMQ), and caches that are the only copy of something (Redis used as a primary store, not a cache) are all stateful.

Here is why that distinction changes everything you do operationally. Figure 2 lays out the four properties of data that turn routine operations into careful procedures.

![A vertical stack of the four properties that make stateful systems harder to operate, covering data gravity, replica identity, single source of truth, and monotonic storage growth, leading to a different SRE playbook](/imgs/blogs/running-stateful-systems-reliably-2.png)

**Data has gravity.** You cannot add a database replica the way you add a web server. A new web server is ready the instant it boots. A new database replica is useless until it has *copied the data* — which for a multi-terabyte database can take hours of streaming a base backup plus catching up on the write-ahead log. Scaling a stateful system is a **data migration**, not a deploy. The same gravity makes you reluctant to move data around, which is why hot shards stay hot.

**Replicas have identity.** A stateless instance is fungible: load-balance across them and you are done. A database replica has a *role* — primary or secondary — and a *position* in the replication stream. You cannot promote a replica to primary unless it is caught up; promote a stale one and you lose the writes it never received. The replica's identity (how far behind it is) determines whether it is safe to use.

**There is a single source of truth.** For a stateless service, losing an instance costs you capacity — one fewer worker. For a stateful system, losing the node that holds the only copy of recent writes costs you *data*. Capacity comes back when you launch another pod; data does not come back unless you replicated it first. This is the difference between an incident that is annoying and an incident that ends up in a regulatory filing.

**Storage grows monotonically.** A stateless service uses roughly constant resources per request. A database's disk usage only ever goes up (until you delete data, and even then compaction or vacuum has to reclaim the space, often needing *more* space temporarily to do it). There is a cliff at 100% disk where the database stops accepting writes, refuses to start, or corrupts its files. You do not "scale up CPU" your way out of disk-full at 3am.

Put these together and you get a system where you cannot freely add nodes (the data must be replicated and rebalanced first), where failover is not instant (a new primary must be promoted *with the data intact*), where losing a node can lose data and not just capacity, and where scaling is a migration and not a rollout. That is a fundamentally different operational category, and it deserves its own SRE playbook. The rest of this post builds that playbook.

> The one-sentence thesis: **operating stateful systems demands a different SRE playbook, because the data has a single source of truth you cannot restart away.** Every section below is a corollary of that.

## 2. Replication lag is a first-class SLI

If you take one number away from this post, make it **replication lag** — how far behind your replica is from the primary, measured in seconds (or in bytes, or in number of operations). For a stateless service the analogous "is this instance healthy" question is binary: up or down. For a stateful system the question is continuous: *how far behind?* And that continuous number is the single most important reliability signal you have for a replicated data store.

Why does it matter so much? Because almost every stateful failure mode reduces to "we trusted a replica that was further behind than we thought." Figure 3 shows the three consequences of lag and how they gate your decisions.

![A branching diagram showing replication lag measured as an SLI on the primary, splitting into a fresh replica safe to read from and a stale replica that returns stale reads, both feeding read routing, with the stale path also setting the failover recovery point objective](/imgs/blogs/running-stateful-systems-reliably-3.png)

**Stale reads break read-your-writes.** If a user updates their profile (a write to the primary) and then immediately reloads the page (a read routed to a lagging replica), they see their *old* profile and conclude your app is broken. This is the "read-your-writes" consistency guarantee, and async replication breaks it by construction: the replica simply does not have the write yet. The deeper the lag, the wider the window where users see the past. This is the same ordering problem the [debugging series covers under distributed race conditions](/blog/software-development/debugging/distributed-race-conditions-and-ordering) — a write and a read that race across two nodes, and the read wins by reaching a node that has not caught up.

**Stale reads poison derived state.** Worse than a confused user is a *system* that reads stale data and acts on it: a billing job that reads a lagging replica, sees a charge has not been applied, and applies it again. A fraud check that reads a stale balance and approves an overdraft. Stale reads in the wrong place are not a UX bug; they are a correctness bug that writes wrong data back.

**Lag is your failover RPO.** **RPO** — recovery point objective — is the amount of data you are willing to lose in a disaster, measured in time ("we can lose at most 5 seconds of writes"). If you fail over to a replica that is 8 seconds behind, you have just chosen, whether you meant to or not, an RPO of 8 seconds: those 8 seconds of writes the replica never received are *gone*. Your replication lag at the moment of promotion **is** your realized RPO. This is the single most under-appreciated fact in stateful operations, and it is why lag is not a vanity metric — it is a data-loss budget you are spending continuously.

So we treat lag as an SLI — a Service Level Indicator, the measured quantity that tells you whether the service is meeting its objective. And like any SLI, we want it scraped, graphed, and alerted.

### Measuring lag honestly

The naive measurement — `master_position - replica_position` — is fine but has a trap: if the primary is *idle*, the replica catches up and lag reads zero even if the replication link is broken (no new writes means nothing to fall behind on). The robust approach is a **heartbeat**: the primary writes a row with the current timestamp every second into a dedicated table; the replica reads that row and computes `now() - heartbeat_timestamp`. This measures wall-clock lag and stays honest even when traffic is bursty. Most managed databases expose a lag metric directly (`pg_replication` views in Postgres, `Seconds_Behind_Master` in MySQL, exporter metrics in Prometheus form).

Here is the Prometheus recording rule that turns the raw exporter metric into a clean lag SLI, plus an alerting rule. (Inside the code fence the `$` characters in PromQL are literal — leave them as-is.)

```yaml
# recording rule: a clean, per-replica lag SLI in seconds
groups:
  - name: stateful_replication
    interval: 15s
    rules:
      - record: db:replication_lag_seconds
        expr: |
          max by (instance, cluster) (
            pg_stat_replication_replay_lag_seconds
          )

      # alerting rule: page only when lag threatens the failover RPO
      - alert: ReplicationLagHigh
        expr: db:replication_lag_seconds > 30
        for: 2m
        labels:
          severity: page
          tier: data
        annotations:
          summary: "Replica {{ $labels.instance }} is {{ $value | humanizeDuration }} behind"
          description: >
            Replication lag exceeds the 30s failover-RPO budget for 2m.
            Reads routed to this replica may be stale; a failover now would
            lose up to this many seconds of writes. Runbook: /runbooks/replication-lag
          runbook_url: "https://runbooks.example.com/replication-lag"
```

Two design choices in that alert matter. First, the threshold (30s) is tied to a *decision*, not a round number: it is the point past which the failover RPO becomes unacceptable, so the alert means "your data-loss budget is at risk," which is actionable. This is the symptom-based-alerting principle from the [alerting that doesn't cry wolf](/blog/software-development/site-reliability-engineering/alerting-that-doesnt-cry-wolf) sibling post applied to data: alert on the consequence (data at risk), not the cause. Second, the `for: 2m` clause suppresses the transient spikes that happen during a large batch write — you only page when lag is *sustained*.

### Read routing: the operational consequence

Once you measure lag, you can act on it. The single most valuable thing lag-awareness buys you is **smart read routing**: send consistency-sensitive reads to the primary (or to a replica you have confirmed is caught up), and send tolerant reads (analytics, dashboards, "recent activity" feeds where 30 seconds of staleness is invisible) to the replicas. Many connection poolers and ORMs support a "read-your-writes" mode that routes a read to the primary if the same session recently wrote, falling back to a replica otherwise.

The anti-pattern to kill: a connection pooler that round-robins *all* reads across replicas regardless of lag. The first time one replica lags 40 seconds during a batch import, a fraction of your users get 40-second-old data and you get a confusing, intermittent "sometimes my changes don't show up" bug report that you will spend a day chasing. Bake the lag check into routing:

```python
# route a read based on consistency need and measured lag
MAX_SAFE_LAG_SECONDS = 1.0

def pick_read_target(needs_strong_consistency: bool, replicas: list) -> str:
    if needs_strong_consistency:
        return "primary"
    # only route to replicas inside the freshness budget
    fresh = [r for r in replicas if r.lag_seconds <= MAX_SAFE_LAG_SECONDS]
    if not fresh:
        # all replicas are lagging; fall back to primary rather than serve stale
        return "primary"
    return least_loaded(fresh).endpoint
```

#### Worked example: how much data does lag cost you?

Suppose your service handles 150 writes per second at peak, and your failover automation promotes whatever replica is "most caught up." During a routine `VACUUM`-heavy window, your best replica drifts to 8 seconds of lag. A primary blip triggers a promotion at exactly that moment.

The math is brutally simple: $\text{lost writes} = \text{write rate} \times \text{lag} = 150 \times 8 = 1{,}200$ writes. Twelve hundred writes — orders, in our opening story — silently gone, because the replica never received them and the primary that held them is now considered dead. Your *intended* RPO might have been "near zero." Your *realized* RPO was 8 seconds, and you only discovered the gap during the incident. That is the entire argument for treating lag as a budgeted SLI: the number you tolerate on a normal Tuesday is the number you lose on a bad one.

Now flip it. If you had run **semi-synchronous replication** — where the primary waits for at least one replica to acknowledge each commit before telling the client it succeeded — your lag-at-promotion would be bounded to the in-flight commits only, typically a handful of milliseconds. The cost is a little write latency (you wait for one network round-trip to a replica). The benefit is your RPO is bounded by design instead of by luck. That trade — a few milliseconds of latency for a bounded data-loss budget — is one every stateful operator should make consciously, and we return to it in the failover section.

### The replication-mode trade-off, made explicit

It helps to lay the three replication modes side by side, because the choice is a direct purchase of RPO with write latency, and most teams make it by accident rather than on purpose. Here is the comparison every stateful operator should be able to recite.

| Mode | Primary waits for | Realized RPO on failure | Write latency cost | When to use it |
|---|---|---|---|---|
| Asynchronous | Nothing — local commit only | Up to the current lag (seconds, unbounded) | None | Tolerant data; analytics replicas; cross-region where latency forbids sync |
| Semi-synchronous | ≥1 replica acknowledges | Bounded to in-flight commits (≈ ms) | One replica round-trip | The system of record; orders/payments/ledgers |
| Synchronous (quorum) | A majority acknowledges | Zero committed-write loss | Slowest acked replica round-trip | Highest-stakes data; consensus stores |

The reasoning behind the table: asynchronous replication never blocks the writer, so it is the fastest, but the price is that on a primary loss you forfeit everything the replicas had not yet pulled — and as the worked example showed, that is your write-rate times your lag, which can be thousands of records. Synchronous replication forfeits nothing committed, because a commit is not acknowledged to the client until durably on a majority — but now every write pays the round-trip to the slowest replica in the quorum, and a slow or partitioned replica can stall *all* writes. Semi-sync is the pragmatic middle: it caps the data-loss window to the tiny set of in-flight, not-yet-acked commits, at the cost of one replica round-trip per write.

The operational subtlety that bites people: a semi-sync primary that cannot reach *any* replica will, depending on configuration, either *block all writes* (it refuses to acknowledge a commit it cannot replicate — availability sacrificed for durability) or *fall back to async* (durability sacrificed for availability). You must know which your system does, because it determines what happens when the replicas are unreachable but the primary is fine — a partition. A primary configured to block will look like a total write outage during a partition even though the data is safe; a primary configured to fall back will keep serving but quietly resume async-style data-loss risk. Neither is wrong, but the choice is a deliberate CAP-theorem trade-off, and discovering it for the first time during an incident is the wrong time to learn which one you picked.

## 3. Failover for stateful systems: the dangerous part

Failover is where stateful operations earn their reputation. For a stateless service, "failover" is trivial: the load balancer notices an instance is unhealthy and stops sending it traffic; the remaining instances absorb the load; done. There is no promotion, no data, no risk of two of anything. For a stateful system, failover means **promoting a replica to be the new primary**, and that is a genuinely dangerous operation with three distinct ways to lose or corrupt data.

### Danger one: promoting a replica that is not caught up

We covered this in the lag section, but it bears repeating as a failover principle: **you cannot promote a replica that is behind without losing the tail.** The correct sequence is *catch up, then promote* — wait (within a bounded timeout) for the candidate replica to apply every write the old primary committed, then promote it. If the old primary is truly dead and the writes never made it off it, those writes are unrecoverable, which is exactly why bounded-RPO replication (semi-sync) matters: it ensures a caught-up replica *exists* before you ever need to promote one.

### Danger two: split-brain

This is the worst one, and it is unique to stateful systems. **Split-brain** is when two nodes both believe they are the primary and both accept writes. Now you have two divergent copies of the truth, and merging them is often impossible — which of two conflicting writes to the same row is "correct"? Split-brain does not lose data; it *corrupts* it, which is harder to recover from than loss. Figure 7 shows how it happens and how to prevent it.

![A branching diagram contrasting a safe promotion path through quorum agreement and fencing of the old primary against an unsafe path that skips fencing and ends in two primaries accepting divergent writes](/imgs/blogs/running-stateful-systems-reliably-7.png)

The classic split-brain scenario: a network partition isolates the primary from the rest of the cluster. The primary is *fine* — it is still running, still has the data, still accepting writes from clients on its side of the partition. But the cluster, unable to reach it, declares it dead and promotes a replica. Now both are taking writes. When the partition heals, you have two histories.

Two mechanisms prevent this, and you need both:

**Quorum.** Require that a *majority* of nodes agree before promoting a new primary. With three nodes, you need two to agree. A node isolated by a partition is in the minority (one out of three) and so cannot be a primary — it lacks quorum. This is the consensus principle (Raft, Paxos) that the [database replication post](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless) derives in detail; operationally, what you need to know is that quorum-based promotion makes it *impossible* to have two primaries with majority backing, because there can only be one majority.

**Fencing (STONITH — "shoot the other node in the head").** Quorum stops the minority node from becoming a *new* primary, but it does not stop the *old* primary from continuing to think it is the primary on its side of the partition. Fencing is the act of forcibly stopping the old primary before promoting the new one — revoking its lease, blocking it at the network or storage layer, or killing it outright. The rule is: **never promote until the old primary is provably fenced.** The fencing token (a monotonically increasing number issued at promotion) is the mechanism that lets storage and clients reject writes from a stale primary even if it does not know it has been deposed. This is the same fencing discipline the [redundancy and failover sibling post](/blog/software-development/site-reliability-engineering/redundancy-and-failover-that-actually-works) covers for the general HA case; for data systems it is non-negotiable.

### Danger three: a wrong automatic promotion

Now the policy question every stateful operator must answer: **automatic or manual failover?**

For a stateless service the answer is obvious — automatic, always; speed wins and there is no downside. For a stateful system the calculus inverts. Automatic failover is fast (seconds, not minutes), which protects availability. But a *wrong* automatic failover — promoting on a transient blip when the primary was actually fine, or promoting a lagging replica — *corrupts or loses data*, and that is permanent. The opening story is exactly this: the automation promoted on a two-second blip, and the cost was 1,200 orders plus a split-brain.

The principled position is a **split policy**:

| Failover decision | Recommended for stateful | Why |
|---|---|---|
| Auto-promote on confirmed quorum-backed primary death | Yes | Genuine death, majority agrees, fenced — fast and safe |
| Auto-promote on a single failed health check | No | A blip or a slow query looks like death; you risk a wrong promote |
| Auto-promote a replica that is behind your RPO budget | No | Guarantees data loss; require a caught-up candidate first |
| Manual-confirm promote during a partial/ambiguous failure | Yes | A human can tell "slow" from "dead"; the cost of a wrong call is permanent |

The middle ground most mature shops land on: **automate the safe, unambiguous case** (quorum says the primary is gone, a fenced caught-up replica is available) and **require a human confirm for the ambiguous case** (is the primary dead or just slow? is the candidate caught up enough?). The human is not there to be fast; the human is there because the cost of a wrong promote is forever and a machine cannot tell "the primary is overloaded and will recover in 20 seconds" from "the primary is dead." This is the [mitigate-first, diagnose-later](/blog/software-development/site-reliability-engineering/mitigate-first-diagnose-later) principle with a crucial stateful caveat: *mitigate-first does not mean act-recklessly when the action is irreversible.*

### Danger four (operational): the reconnection storm

When a primary fails over, every client that was connected to it suddenly has a dead connection. They all reconnect to the new primary *at the same instant* — a thundering herd of reconnections that can overwhelm the freshly-promoted node before it has warmed its caches, refilled its connection pool, or stabilized. The new primary, hit with a reconnection storm plus the full write load it has never seen before, can fall over, triggering *another* failover, and you have an oscillation.

Two mitigations: **connection draining** on the old node when failover is planned (let in-flight queries finish, stop accepting new ones, give clients a clean signal to reconnect gradually), and **reconnection backoff with jitter** on clients (the same [timeouts, retries, and backoff](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) discipline that prevents retry storms — every client retrying instantly and in lockstep is the storm; randomized backoff spreads the herd). Without jitter, all clients reconnect, fail because the node is saturated, all back off the same fixed interval, and all retry again in lockstep — a synchronized hammer.

There is a worse version: the **failover oscillation** (or "flapping"). The freshly-promoted primary, hit by the reconnection storm *and* the full write load it has never carried, exceeds its health-check thresholds — so the controller, watching health checks, decides *this* primary is unhealthy too and promotes *another* replica. Now you are failing over in a loop, each promotion making the next worse, and possibly each one losing a little more of the un-replicated tail. The guardrails are a **promotion cooldown** (refuse to promote again within N minutes of the last promotion — give the new primary time to warm up) and a **flap detector** that escalates to a human instead of promoting a third time. Auto-failover without a cooldown is a machine that, under load, will fail your database over and over until someone pulls the plug — exactly when you can least afford the churn. A connection pooler in front of the database that absorbs the reconnection storm (the clients reconnect to the pooler, which holds a small stable set of connections to the actual primary) is the cleanest structural fix: it decouples the thousands of client reconnections from the few connections the new primary actually has to warm.

#### Worked example: the failover that lost data (and the fix)

Let me walk the opening incident end to end, because it ties every failover danger together. Figure 4 is the timeline.

![A left to right timeline of a data losing failover, starting with a primary blip and a failed health check, then an automatic promotion of a replica eight seconds behind, the loss of twelve hundred orders, the old primary rejoining to create two primaries, the resulting split brain with divergent writes, and finally the fix of fencing plus semi synchronous replication](/imgs/blogs/running-stateful-systems-reliably-4.png)

The chain of events:

1. **00:00** — The primary has a 2-second pause (a long checkpoint flush stalled the health-check query). Two consecutive health checks time out.
2. **00:02** — The failover controller, configured for fast auto-promotion on health-check failure, picks the most-caught-up replica — which is 8 seconds behind due to a concurrent `VACUUM` — and promotes it. No fencing of the old primary; the controller assumed it was dead.
3. **00:02** — The 8 seconds of un-replicated writes (1,200 orders at 150 wps) are now orphaned on the old primary, which is *not* dead.
4. **00:40** — The old primary's checkpoint finishes; it resumes serving the clients still pointed at it and accepts ~90 seconds of new writes, believing it is still primary.
5. **02:10** — Monitoring notices two primaries; an engineer stops writes to the old one. Now there are two divergent histories to reconcile.

The blast radius: 1,200 lost orders plus a multi-day reconciliation of the divergent writes. Every danger fired: a too-eager auto-promote (danger three), a lagging candidate (danger one), and no fencing (danger two).

The fix, layered:

- **Semi-synchronous replication** so at least one replica is always within a few milliseconds of the primary — bounds the RPO to near-zero and guarantees a caught-up promotion candidate exists.
- **Quorum-based death detection** — promote only when a majority agrees the primary is gone, not on two failed health checks from one observer.
- **Fencing before promotion** — issue a fencing token, revoke the old primary's lease, and have it self-demote (or get killed) the instant it can no longer renew its lease.
- **Manual-confirm for ambiguous failures** — a single observer's health-check failure pages a human who confirms death vs. slowness before promotion; quorum-confirmed total death can still auto-promote.

After this change, the next two real primary failures failed over in under 30 seconds with **zero data loss and zero split-brain**, and the one ambiguous "is it slow or dead?" event was correctly diagnosed by the on-call as a slow checkpoint that recovered on its own — no failover needed, no data risk taken. That is the measured proof: the fix did not just prevent the rare catastrophe, it also *avoided an unnecessary failover*, which is its own reliability win.

## 4. Backpressure and overload: when the queue becomes the outage

The second major stateful failure family is **overload**, and it has a signature that surprises people: in a stateful system, *the buffer that is supposed to protect you becomes the thing that kills you.* A message queue exists precisely to decouple a fast producer from a slow consumer — to absorb bursts. But a queue is stateful: the messages pile up *in storage*. If the consumer is slower than the producer for long enough, the queue does not gracefully shed load — it *grows*, monotonically, until it runs out of memory or disk. At that point the broker stops accepting writes, and now *every producer* is blocked. The queue you added for resilience has become the single point of failure for everything upstream of it.

This is the inverse of the stateless overload story. An overloaded stateless service sheds requests (returns 503s, the load balancer routes around it) and recovers when load drops. An overloaded queue *accumulates*, and the accumulation has inertia: even after the producer slows down, the consumer has to grind through a multi-million-message backlog before the queue drains. The queue's depth is state, and state has gravity.

### The principle: bounded buffers and Little's Law

Why does an unbounded queue inevitably blow up? Because of a basic law of queueing systems. **Little's Law** says the average number of items in a system equals the arrival rate times the average time each item spends in the system: $L = \lambda W$. If the consumer's *service rate* $\mu$ is even slightly below the producer's *arrival rate* $\lambda$, the queue length grows without bound — the time $W$ an item waits goes to infinity as the backlog grows. There is no steady state when $\lambda > \mu$; the only question is how fast you hit the wall.

The mature design accepts this and *bounds the buffer*. A bounded queue has a maximum depth; when full, it pushes back — it either blocks the producer (backpressure: the producer slows down because it cannot enqueue), or it rejects/sheds the message (load shedding), or it diverts the message to overflow storage. Backpressure is the principled choice: it propagates the slowness *upstream* so the system slows down as a whole instead of one component silently accumulating until it dies. The [message-queue series goes deep on backpressure and flow control](/blog/software-development/message-queue/backpressure-and-flow-control); the SRE point is that **an unbounded buffer is a deferred outage**, and your job is to make the slowness visible and bounded before it becomes a disk-full broker death.

### Consumer lag as an SLI (and the connection-pool sibling)

Just as replication lag is the SLI for a replicated database, **consumer lag** — the number of messages produced but not yet consumed, or the time-equivalent — is the SLI for a queue. Kafka exposes it directly as consumer-group lag (the offset gap between the latest produced offset and the consumer's committed offset). You scrape it, graph it, and alert on it:

```yaml
groups:
  - name: stateful_queue
    interval: 15s
    rules:
      # SLI: total unconsumed backlog per consumer group
      - record: queue:consumer_lag_messages
        expr: |
          sum by (consumer_group, topic) (
            kafka_consumergroup_lag
          )

      # is the backlog growing? positive derivative over 5m = falling behind
      - record: queue:consumer_lag_growth_per_s
        expr: |
          deriv(queue:consumer_lag_messages[5m])

      - alert: ConsumerFallingBehind
        # lag is high AND still growing - the dangerous combination
        expr: |
          queue:consumer_lag_messages > 100000
          and
          queue:consumer_lag_growth_per_s > 0
        for: 5m
        labels:
          severity: page
          tier: data
        annotations:
          summary: "{{ $labels.consumer_group }} backlog {{ $value }} and growing"
          description: >
            Consumer lag is high and the derivative is positive, so the
            consumer is not keeping up. Left unchecked the broker disk fills.
            Mitigations: scale consumers, shed/DLQ poison messages, throttle producer.
            Runbook: /runbooks/queue-backlog

      # the cliff: alert on the resource itself, well before 100%
      - alert: BrokerDiskFilling
        expr: |
          (1 - node_filesystem_avail_bytes{mountpoint="/var/lib/kafka"}
             / node_filesystem_size_bytes{mountpoint="/var/lib/kafka"}) > 0.80
        for: 5m
        labels:
          severity: page
          tier: data
        annotations:
          summary: "Broker disk at {{ $value | humanizePercentage }}"
```

The key insight in that ruleset: **alert on lag *and* its growth together.** A high-but-stable backlog after a recovery is draining and fine; a high-*and-growing* backlog is heading for the cliff. The derivative turns "the queue is deep" (often benign) into "the queue is deep and we are losing the race" (always actionable). And the separate disk alert at 80% is the backstop — because the real outage is disk-full, and you want to be paged with 20% of headroom left to act, not at 99%.

The database analog of queue overload is **connection-pool exhaustion**. A database accepts a bounded number of connections (memory and per-connection overhead make them finite). When a slow query holds connections, or a leak fails to return them, or a traffic spike opens more than the pool allows, new requests *queue waiting for a connection* — and that queue, too, can grow unbounded until requests time out and the application appears down even though the database is "up." This is the same shape as the message-queue overload (a bounded resource, an unbounded wait queue in front of it) and it ties directly to the [database locks and deadlocks in production](/blog/software-development/database/database-locks-and-deadlocks-in-production) post: a long-held lock holds a connection, which exhausts the pool, which queues every request behind it. The mitigations rhyme: bound the pool, set a connection-acquire timeout (fail fast rather than queue forever), and shed load when the pool is saturated.

### The hot partition / hot shard

One more stateful overload mode worth naming: the **hot partition**. When you shard data (or partition a Kafka topic) by a key, an uneven key distribution sends a disproportionate share of traffic to one shard. That shard saturates while the others sit idle — you have plenty of *aggregate* capacity but one node is on fire. The classic trigger is a "celebrity" key (one user, one product, one tenant with 100× the activity) or a poorly chosen partition key (timestamp-based keys send all *current* writes to one partition). The fix is at the design layer (better key, salting, splitting the hot key) — cross-link to the data-modeling docs — but the *operational* signal is per-shard load skew, which you must graph per-partition, not in aggregate, or the average hides the fire.

### The cache that is secretly a database

Caches deserve their own warning because they straddle the line and people misjudge which side they are on. A cache used *as a cache* — every value can be regenerated from a durable source of truth — is operationally stateless: lose it, repopulate it, no data lost. But a cache used *as a store* — sessions that exist nowhere else, a rate-limiter's counters, a queue of jobs held only in Redis, the only copy of a shopping cart — is stateful, and losing the node loses data. Teams routinely treat the second case like the first ("it's just Redis, restart it") and are shocked when a restart drops every user's session and logs the whole site out.

The diagnostic is again the Figure 8 question: *if this node dies, do I lose data?* If yes, it is a stateful store and needs persistence (Redis AOF/RDB durability tuned to your RPO), replication, and the same failover care as a database — Redis Sentinel or Cluster does quorum-based promotion and fencing, and the same split-brain risk applies if you misconfigure it. If no, it is a cache and you can be cavalier. The danger is the *gradual drift*: a system that started as a pure cache slowly accreted "just this one thing we only store here," and now it is a database wearing a cache's operational reputation. Audit periodically what is *only* in the cache; that is your real RPO surface.

The cache also has a unique overload mode worth its own line: **memory eviction under pressure.** When a cache fills, it evicts (LRU, LFU, or TTL-based). For a true cache that is fine — evicted items just regenerate. For a cache-as-store with `noeviction` configured, a full cache *rejects writes* — and now the rate-limiter cannot increment, sessions cannot be created, and the "cache" is an outage. Match the eviction policy to the role: evict freely for a cache, but a store must be capacity-planned like a database (it has the same monotonic-growth disk cliff, just in RAM, where the cliff is closer and harder).

### Poison messages and the dead-letter queue

A subtle queue killer: the **poison message** — a single message the consumer cannot process (malformed, references deleted data, triggers a bug). The naive consumer retries it, fails, retries, fails — forever. It never advances past the poison message, so the backlog behind it grows without bound even though the *producer* is behaving. One bad message becomes a total stall. The fix is a **dead-letter queue (DLQ)**: after N failed attempts, move the message to a separate queue for later inspection and *advance past it*. This contains the blast radius of one bad message to one bad message. The [poison messages and retry-storms containment](/blog/software-development/message-queue/poison-messages-and-retry-storms-containment) post covers the patterns; operationally, a DLQ with an alert on its depth ("we are dead-lettering, something is systematically wrong") is the difference between losing one message and stalling the entire pipeline.

#### Worked example: the queue that became the outage (and the fix)

Here is the second incident in full. Figure 6 shows the before and after.

![A side by side comparison of an unbounded queue where a slow consumer lets depth grow to two million and fill the disk taking the broker down, versus a bounded queue with a lag SLI that alerts at five minutes behind, autoscales consumers, sheds to a dead letter queue, and keeps depth flat with the broker healthy](/imgs/blogs/running-stateful-systems-reliably-6.png)

**Before.** A consumer service deployed a change that doubled its per-message processing time (an N+1 query crept in). The producer kept sending at a steady 5,000 messages/second; the consumer's throughput dropped from 6,000/s to 3,000/s. By Little's Law, with $\lambda = 5{,}000$ and $\mu = 3{,}000$, the queue grew at 2,000 messages/second. There was no consumer-lag alert and no bound on the queue. Over six hours the backlog reached ~43 million messages, the broker's disk hit 100%, the broker crashed, and now *every producer* across the company that used that broker was blocked — a single slow consumer took down a shared bus. Recovery was slow precisely because it is stateful: even after rolling back the consumer change, the system had to grind through 43 million backlogged messages before it was current, which took hours.

**After.** Four changes:

1. **A consumer-lag SLI and alert** (the ruleset above) — paged the on-call at a 1.5-million backlog *while it was still growing*, hours before the disk cliff, with 90%+ disk headroom remaining.
2. **A bounded queue / retention cap** — the topic retention and a producer-side backpressure signal mean the broker sheds or applies backpressure rather than accumulating to disk-full; the slowness propagates upstream visibly instead of silently filling disk.
3. **Consumer autoscaling on lag** — a horizontal-pod-autoscaler keyed on consumer lag spins up more consumer replicas when the backlog grows, restoring $\mu > \lambda$ automatically for the common "needs more workers" case.
4. **A DLQ for poison messages** — so a single unprocessable message cannot stall the whole partition and masquerade as "the consumer is slow."

The measured result: in the next two consumer slowdowns, the lag alert fired within 5 minutes, the autoscaler added capacity, and the peak backlog topped out around 400,000 messages and drained in under 20 minutes — **the broker never came close to disk-full and no producer was ever blocked.** The queue went back to being a shock absorber instead of a bomb. The before→after on the lag curve is the proof: an unbounded ramp to the disk cliff became a small, self-correcting bump.

## 5. The stateful on-call: what is different at 3am

Everything above changes how you carry the pager for a stateful system. The stateless on-call playbook — *restart it, kill the bad instance, scale out, roll back, let the orchestrator heal it* — is built on one assumption: **mistakes are transient.** If you restart the wrong pod, you lose a few seconds of capacity and it comes back. That assumption is *false* for stateful systems, and that single fact reshapes the entire on-call discipline. Figure 5 makes the contrast concrete.

![A comparison matrix with rows for restart, failover, wrong action, and recovery, and columns for a stateless service versus a stateful system, showing that each operation is cheap and safe for stateless but slow, risky, or permanent for stateful](/imgs/blogs/running-stateful-systems-reliably-5.png)

Walk the four rows:

**You cannot just restart it.** Restarting a stateless pod is the first thing you try and it costs nothing. Restarting a database might lose un-flushed writes, trigger crash recovery (replaying the WAL, which can take minutes to an hour on a large database), or — if the data is corrupted — fail to come up at all and force a restore. "Have you tried turning it off and on again?" is dangerous advice for a database. The first instinct must be *suppressed*.

**Recovery is slow.** A stateless recovery is a reschedule — seconds. A stateful recovery is a *process*: replay the write-ahead log, rebuild an index, re-replicate from a peer, restore from backup, run a consistency check. These take tens of minutes to hours, and during that time the system is degraded or down. Your **RTO** — recovery time objective, the target time to restore service — is dominated by these slow data operations, not by how fast you can launch a node. You must *know* your real RTO (have you actually timed a restore?) because at 3am you need to decide "wait for recovery or fail over?" and that decision depends on which is faster.

**The blast radius of a wrong action is permanent.** This is the heart of it. A wrong stateless action self-corrects. A wrong stateful action — a `DROP TABLE` on the wrong database, a `DELETE` without a `WHERE`, a bad failover that loses the tail, a restore that overwrites good data with stale backup data — is *forever*. There is no undo. This is why the most senior stateful operators are the *slowest to act* under pressure: they have learned that the cost asymmetry (a wrong action is permanent, waiting 60 seconds to confirm is usually free) means **confirm before you commit.**

### The shifted mitigate-first calculus

The SRE mantra is [mitigate first, diagnose later](/blog/software-development/site-reliability-engineering/mitigate-first-diagnose-later) — stop the user pain before you root-cause it. That is correct, and it still applies. But for stateful systems it carries a critical amendment: **the safest mitigation is sometimes to STOP, not to act.**

Consider the decision tree in Figure 8. It is the single most important habit to build for stateful on-call: before any action, ask *can this action lose or corrupt data?*

![A decision tree starting from a stateful page at 3am, branching on whether the action can lose data, with the no branch leading to safe actions like restarting a read replica and the yes branch leading to either stopping and escalating or a careful manual promotion after catch up and fencing](/imgs/blogs/running-stateful-systems-reliably-8.png)

If the answer is *no* (restart a read-only replica, fail over the read path to another replica, scale up CPU on a node that is just busy) — then act, mitigate-first applies normally. These actions cannot corrupt the source of truth.

If the answer is *yes* (promote a replica, restart the primary, restore from backup, run a destructive `DELETE` to clear a poison row) — then **slow down.** The cost of pausing 60 seconds to get a second pair of eyes, confirm the candidate is caught up, confirm the old primary is fenced, double-check the `WHERE` clause, is almost always smaller than the cost of an irreversible mistake. In stateful on-call, *stopping* is a legitimate, often correct mitigation. A primary that is slow but recovering does not need a failover; forcing one anyway is how you turn a 5-minute slowdown into a data-loss incident. "First, do no harm" is not a cliché for data systems — it is the operating principle.

This is why stateful runbooks are *more careful* than stateless ones. A stateless runbook can say "restart the pod" as step one. A stateful runbook front-loads the *checks* — confirm the failure mode, confirm the candidate state, confirm fencing, confirm you have a recent backup — before any irreversible step, and it explicitly calls out the points of no return.

### A stateful-failover runbook

Here is a runbook for the most dangerous routine stateful operation — a primary database failover. Notice how much of it is *verification* before the irreversible step, and how it names the point of no return. This is the kind of runbook that survives 3am, in the spirit of the [runbooks that survive 3am](/blog/software-development/site-reliability-engineering/runbooks-that-survive-3am) sibling.

```yaml
# RUNBOOK: Primary database failover (stateful — read fully before acting)
title: db-primary-failover
severity_to_invoke: Sev1 or Sev2 (primary unavailable)
golden_rule: >
  A wrong promotion is PERMANENT. If the primary is SLOW (not dead),
  do NOT fail over — wait or mitigate the slowness. Only fail over a
  DEAD primary, and only to a CAUGHT-UP, FENCED candidate.

steps:
  - id: 1-classify
    do: "Is the primary DEAD or SLOW? Check: connections accepted? disk full?
          quorum reachable? Is it a long checkpoint/lock that will recover?"
    decision: >
      SLOW and recovering -> STOP. Mitigate slowness (kill long query,
      add disk, clear lock). Do NOT promote. DEAD/quorum-confirmed -> continue.

  - id: 2-pick-candidate
    do: "List replicas with db:replication_lag_seconds. Pick the most
          caught-up. Confirm its lag is within the RPO budget."
    decision: "No candidate within RPO -> escalate to data-owner before promoting.
                Promoting a lagging replica WILL lose the tail."

  - id: 3-fence-old-primary
    do: "FENCE the old primary BEFORE promoting: revoke its lease / block
          its writes / STONITH. Confirm it can no longer accept writes."
    point_of_no_return: false
    note: "Skipping this risks split-brain. NEVER skip."

  - id: 4-confirm
    do: "If failure is ambiguous, get a second on-call to confirm the
          promotion target and that fencing succeeded."
    note: "Two minutes of confirmation is cheaper than permanent corruption."

  - id: 5-promote
    do: "Promote the chosen, caught-up, confirmed candidate."
    point_of_no_return: true
    note: "After this the new primary diverges from the old. Irreversible."

  - id: 6-redirect-and-drain
    do: "Update the service endpoint / DNS / pooler to the new primary.
          Expect a reconnection storm — clients reconnect with backoff+jitter."

  - id: 7-verify
    do: "Confirm writes succeed, replication to remaining replicas resumes,
          replication-lag SLI returns to baseline, error budget burn stops."

  - id: 8-rebuild
    do: "Rebuild the old primary as a fresh replica from the new primary
          (do NOT rejoin it as-is — it has divergent un-replicated writes)."
```

The structure is the lesson: steps 1–4 are all *verification and confirmation*; step 5 is the single irreversible action, explicitly flagged; step 8 prevents the split-brain by refusing to rejoin the old primary as-is. A stateless runbook would never need most of this.

### Stress-testing the stateful on-call

Good runbooks survive contact with the ugly cases. Let me stress-test:

- **What if the dependency (the database) is down for 2 hours?** Then your RTO is blown and you need the *application-layer* mitigation: degrade gracefully (serve from cache, queue writes for later replay, show a read-only mode) rather than hard-failing. This is where the [graceful degradation and fallbacks](/blog/software-development/site-reliability-engineering/graceful-degradation-and-fallbacks) sibling earns its keep — the stateful tier being down should not be a total outage if you designed fallbacks.
- **What if the on-call is asleep and the auto-failover already fired wrongly?** Then you are doing damage control on a split-brain. The fix is upstream: auto-failover should be restricted to the unambiguous quorum-confirmed case so it *cannot* fire wrongly on a blip while you sleep.
- **What if two incidents overlap — a primary failover during a queue backlog?** Triage by blast radius and reversibility: the failover (irreversible, data at risk) gets the careful human; the queue backlog (recoverable, just needs capacity) gets the autoscaler or a second responder. Never let the urgent-but-recoverable distract you from the irreversible.
- **What if the budget is already spent?** A blown error budget means you freeze risky changes — and a manual stateful failover *is* a risky change. If the budget is gone, that is a stronger argument for the careful, confirmed path, not a reason to YOLO a fast auto-promote.
- **What if the backup has never been restored?** Then you do not have a backup; you have a *hope*. An untested backup is the single most common cause of "we had backups and still lost the data." This is the whole argument of the `backups-that-actually-restore` sibling: schedule a restore drill, time it (that is your real RTO), and verify the restored data. A backup you have never restored is Schrödinger's backup — both valid and invalid until you actually try it, usually during the disaster.
- **What if the region fails?** Now you need cross-region replication and a regional failover plan, with the same fencing and quorum discipline at a larger blast radius, plus the higher RPO that cross-region async replication implies. The principles scale; the latencies and the data-loss windows grow.

#### Worked example: the restore drill that found your real RTO

A team believed their RTO was "about 30 minutes" — that was the number in the incident plan, written by someone who once watched a small-database restore. They had never restored the *production* database from backup. During a quarterly game day, they ran the drill against a staging copy of production: restore the latest backup, replay write-ahead log to the failure point, run a consistency check, and bring it up.

It took **4 hours and 40 minutes.** The base backup of a 3 TB database streamed for over 2 hours; replaying a day's worth of WAL added another 90 minutes; the consistency check and reindex added the rest. Their "30-minute RTO" was off by nearly an order of magnitude — and they only knew because they *measured it*, not at 3am during a real disaster, but on a calm Tuesday afternoon with coffee.

Two things came out of that single drill. First, the documented RTO was corrected to a number they could actually meet, which changed the failover-vs-restore decision (failing over to a warm replica, ~5 minutes, is now obviously preferred over restore for any recoverable-by-failover scenario; restore is the last resort). Second, they invested in *faster* restore — incremental backups and a continuously-restored warm standby — to drag the worst-case RTO down from nearly 5 hours to under 45 minutes. The measured proof is the drill time itself: an unknown, untested "30 minutes" became a known, tested, and then *improved* 45 minutes. You cannot improve a number you have never measured, and a backup you have never restored has no RTO — only a hope. This is the heart of the `backups-that-actually-restore` discipline: the restore drill is not paperwork, it is the only thing that turns a backup from Schrödinger's snapshot into a known-good recovery path with a known cost.

## 6. Capacity for stateful systems: the disk-full cliff

Capacity planning for a stateless service is mostly about CPU and request rate — and it is *elastic*: scale out when busy, scale in when quiet, and the resource returns. The [capacity planning and forecasting](/blog/software-development/site-reliability-engineering/capacity-planning-and-forecasting) sibling covers the general method. Stateful capacity has a different shape, dominated by one ugly property: **storage grows monotonically and the cliff is hard.**

A CPU spike is a soft limit — you slow down, you queue, you shed, and when it passes you recover. **Disk-full is a hard cliff.** At 100% disk a database stops accepting writes (best case), refuses to start, or corrupts its files (worst case). A broker dies. There is no "graceful slowdown" at the disk wall; you go from working to broken. And because storage only goes up, you *will* hit it eventually unless you actively manage it. So stateful capacity is less about elastic autoscaling and more about *forecasting the cliff and acting before it.*

Three stateful-specific capacity concerns:

**The growth forecast and the disk alert.** Trend your disk usage and project when it hits the cliff. Alert with enough lead time to *act* — and acting on storage is slow (provisioning a bigger volume, migrating data, archiving old partitions all take time), so you need *days* of warning, not minutes. A linear-fit predictive alert is worth its weight here:

```yaml
- alert: DiskWillFillIn7Days
  # predict_linear extrapolates the 6h trend; alert if it crosses 0 in 7 days
  expr: |
    predict_linear(node_filesystem_avail_bytes{mountpoint="/var/lib/postgresql"}[6h], 7 * 24 * 3600) < 0
  for: 1h
  labels:
    severity: ticket
  annotations:
    summary: "Postgres data disk projected to fill within 7 days"
```

Note the severity is `ticket`, not `page` — a 7-day forecast is not a 3am wake-up, it is a "handle it this week" ticket. (The 80%-now disk alert from earlier *is* a page.) Matching alert urgency to time-to-act is the [alerting that doesn't cry wolf](/blog/software-development/site-reliability-engineering/alerting-that-doesnt-cry-wolf) discipline applied to capacity.

**Compaction and vacuum pressure.** Stateful stores reclaim space asynchronously — Postgres `VACUUM`, Cassandra/LSM compaction, log-segment cleanup. These processes need *headroom* (compaction can temporarily need extra space to rewrite data) and *I/O budget* (they compete with your live traffic). Two failure modes: compaction falls behind and space is never reclaimed (slow disk creep toward the cliff), or compaction runs aggressively and *steals I/O* from live queries (a latency spike that looks like an outage). Monitor both the reclaim backlog and the I/O impact; a vacuum that cannot keep up is a capacity alarm even if disk looks fine today.

**The migration that needs 2× storage.** The nastiest stateful capacity trap: many data operations temporarily need roughly double the space. Rebuilding an index, doing an online schema change that copies a table, a major-version upgrade, restoring alongside the live data — each can need the dataset twice over, transiently. The on-call who tries to free space at 95% disk by *running a migration* discovers the migration needs more space than exists and makes it worse. Plan storage with this headroom in mind, and know which operations need it before you reach for them under pressure.

## 7. Schema and migration safety as a reliability concern

One more place stateful operations cause outages that stateless ones never do: **schema migrations.** A code deploy on a stateless service is reversible — roll back the binary and you are done. A schema migration mutates the *data layer*, the shared source of truth, and a bad one is an outage that a rollback may not even fix (you cannot un-drop a column's data).

The classic migration outage: someone runs `ALTER TABLE orders ADD COLUMN ...` or, worse, adds an index on a huge table, and the migration takes a **table-level lock** for the duration. Every query that touches that table now blocks behind the lock. On a busy table the migration takes minutes, every request piles up waiting, the connection pool exhausts (section 4's failure mode), and the application is down — not because the database crashed, but because one DDL statement froze the hottest table in the system. This is the same locking dynamic the [database locks and deadlocks in production](/blog/software-development/database/database-locks-and-deadlocks-in-production) post covers, surfacing here as a reliability incident.

The safe pattern is **expand/contract** (also called parallel-change), and it is the data-tier version of the progressive-delivery discipline from [deploying safely with progressive delivery](/blog/software-development/site-reliability-engineering/deploying-safely-progressive-delivery):

1. **Expand** — make an *additive, backward-compatible* schema change (add the new column as nullable, add the new table). Old code ignores it; new code can use it. No destructive change, no long lock if done with online DDL.
2. **Migrate** — backfill data into the new structure *in small batches* (not one giant locking `UPDATE`), and dual-write from the application so both old and new shapes stay current.
3. **Contract** — only after the new shape is fully populated and all code reads from it, remove the old column/table. This is the irreversible step, done last, when you are confident.

Rules that keep migrations from being outages: use **online/non-blocking DDL** (Postgres `CREATE INDEX CONCURRENTLY`, `ADD COLUMN` without a volatile default; tools like `gh-ost` or `pt-online-schema-change` for MySQL that build the new table and swap it without a long lock); **batch every backfill** with a sleep between batches so you do not lock-storm the table or saturate replication; set a **lock timeout** so a migration that *would* block fails fast instead of freezing the table (`SET lock_timeout = '2s'`); and watch **replication lag during the migration** — a heavy backfill is exactly the workload that spikes lag and breaks read routing (section 2), so a migration can *cause* the stale-read incident if you are not watching the lag SLI.

```sql
-- safe: concurrent index build does not take a blocking lock
-- (cannot run inside a transaction block)
CREATE INDEX CONCURRENTLY idx_orders_customer ON orders (customer_id);

-- safe: fail fast instead of freezing the table if a lock can't be had
SET lock_timeout = '2s';
ALTER TABLE orders ADD COLUMN promo_code text;  -- nullable, no rewrite

-- batched backfill: never one giant locking UPDATE
-- (loop in app code, sleeping between batches, watching replication lag)
UPDATE orders SET promo_code = legacy_promo
WHERE id BETWEEN :lo AND :hi AND promo_code IS NULL;
```

The reliability framing: a migration is a *change*, and changes spend error budget. Treat a schema migration with the same care as a risky deploy — small, reversible-until-the-contract-step, observed via the lag and lock SLIs, with a clear back-out plan for each phase. The expand/contract sequence is what makes it reversible at every step except the final contract.

## 8. War story: real cascading failures from the stateful tier

The stateful tier shows up in a surprising share of the famous outages, because its failures *cascade* — the data layer is the thing everything else depends on, so when it stalls, everything upstream stalls with it.

**The thundering herd on a cache.** A widely-cited pattern (popularized in writeups from Facebook and many others) is the **cache stampede**: a hot cache key expires, and the thousands of requests that were being served from cache all *miss simultaneously* and stampede the database to regenerate the value. The database, sized for the cache-hit-rate it normally sees, is suddenly hit with the full unprotected load and falls over — and now *nothing* is cached, so every request hits the database, a self-sustaining overload. The stateful lesson: the cache is load-bearing capacity, and its expiry is a correlated event. The fixes are operational (request coalescing / single-flight so only one request regenerates the key; staggered/jittered TTLs so keys do not all expire together; serving stale-while-revalidate). This is the same thundering-herd shape as the reconnection storm in section 3 and the retry storm in [timeouts, retries, and backoff](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) — a synchronized event overwhelming a stateful resource.

**The replication-lag outage at scale.** Numerous large operators have public postmortems where a *read replica fell far behind* during a heavy write event (a big migration, a mass backfill, a bulk import), and read traffic routed to the lagging replica returned stale or empty data — manifesting as "data disappeared" to users even though no data was actually lost. The systems were *up*; they were *behind*. The lesson that recurs: lag is invisible until you measure it as an SLI and route reads around it. A replica that is "up" but 10 minutes behind is, for consistency-sensitive reads, *worse* than down — at least a down replica fails loudly instead of quietly serving the past.

**GitLab's 2017 database incident** is the canonical untested-backup disaster and a public, blamelessly-documented one (they livestreamed the recovery). During an incident response, an engineer ran a destructive command against the wrong database — the *irreversible stateful action* this whole post warns about — and then discovered that several backup mechanisms were not actually working: the backups were not being taken correctly, the replication had its own issues, and the most recent usable snapshot was hours old. The result was several hours of data loss. The lessons GitLab themselves drew are exactly this post's thesis: destructive stateful actions need guardrails and confirmation (the wrong-database `DROP`), and **a backup is not a backup until you have restored it** (the untested-backup trap). I cite it because they documented it openly and the lessons are durable; the specifics are theirs, honestly reported, not invented here.

**The shared-bus single point of failure.** A recurring pattern (Kafka/Kinesis/RabbitMQ outages across many companies) is the section-4 story at organizational scale: one team's runaway producer or stuck consumer fills a *shared* message bus, and because the bus is stateful and shared, *every* team that depends on it is taken down at once. The operational lesson is isolation — quotas per producer, separate clusters for unrelated criticality tiers, and bounded retention — so one tenant's overload cannot consume the shared stateful resource and cascade across the whole company. This is the [circuit-breaker, bulkhead, and load-shedding](/blog/software-development/site-reliability-engineering/circuit-breakers-bulkheads-and-load-shedding) discipline applied to a shared data tier: bulkhead the tenants so a flood in one compartment does not sink the ship.

The common thread across all four: stateful failures *propagate* and *persist* in a way stateless ones do not. A stateless instance that dies takes its own requests down and nothing more. A stateful tier that stalls, lags, corrupts, or fills takes everything that depends on it — which is everything.

## 9. How to reach for this playbook (and when not to)

Every practice in this post has a cost, and applying the heavy version everywhere is its own failure mode. Here is the decisive guidance.

**Use the full stateful playbook — semi-sync replication, quorum + fencing, manual-confirm failover, lag SLIs, careful runbooks — when the data is the product and loss/corruption is unacceptable**: the primary transactional database for orders/payments/users, the system of record, the financial ledger. Here the cost asymmetry (a permanent data-loss event versus a few milliseconds of write latency and a slower failover) overwhelmingly favors caution. Pay the latency, require the human confirm, run the restore drills.

**Do not apply it where the state is disposable.** A pure read-through cache (Redis as a cache, not a store) is *operationally stateless* — if you lose it, you repopulate it from the source of truth, and you can absolutely "just restart it." Treating a disposable cache with the full database ceremony is wasted toil. The test is the question from Figure 8: *if I lose this node, do I lose data?* If the answer is no (it can be rebuilt from elsewhere), treat it like a stateless service and move fast.

**Do not auto-promote on ambiguous signals.** Fast auto-failover is the right default for stateless. For a stateful primary, restrict auto-promotion to the quorum-confirmed, fenced, caught-up case, and route the ambiguous "slow vs. dead" case to a human. The speed you gain from auto-promoting on a single health check is not worth the split-brain you eventually cause.

**Do not over-engineer RPO/RTO for low-stakes data.** A 99.999% availability target and zero-RPO synchronous replication for an internal analytics warehouse that can tolerate a day of staleness is wasted money and complexity. Match the RPO/RTO to the *business* cost of loss and downtime — the [error budget](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) framing applies: do not buy a nine the users (or the business) cannot feel.

**Do not run a migration as a panic fix at 95% disk.** The migration that needs 2× storage will make a near-full disk worse. Forecast the cliff (section 6) and act with days of headroom, not minutes.

**Do skip the heavy failover machinery for systems that can simply re-derive their state.** A stateless stream processor that checkpoints to durable storage can be restarted freely — its "state" lives in the durable store, not the process. Know which of your "stateful-looking" systems are actually stateless-with-external-state, and operate those the easy way.

The meta-rule: **the right amount of stateful caution is proportional to the permanence and value of the data.** More permanence and more value buys more ceremony. Disposable data buys none.

## 10. Key takeaways

- **You cannot treat a database like a pod.** Data has gravity, identity, a single source of truth, and monotonic growth, so adding, killing, and scaling are migrations and promotions, not restarts and reschedules.
- **Replication lag is a first-class SLI.** It governs stale reads, read-your-writes, and — most importantly — your realized failover RPO. Your lag at the moment of promotion *is* the data you lose. Measure it with a heartbeat, alert on a threshold tied to your RPO budget, and route consistency-sensitive reads around lagging replicas.
- **Stateful failover has four dangers**: promoting a lagging replica (lose the tail), split-brain (two primaries corrupt the truth — prevent with quorum *and* fencing), a wrong auto-promote on a blip (permanent), and the reconnection storm (mitigate with draining and backoff+jitter). The runbook is mostly verification before one irreversible step.
- **Automate the unambiguous failover, confirm the ambiguous one.** A machine cannot tell "slow" from "dead," and the cost of a wrong promote is forever.
- **The buffer becomes the outage.** An unbounded queue or connection pool grows until the resource is exhausted and the broker/database dies, blocking everything upstream. Bound buffers, treat consumer lag as an SLI (alert on lag *and* its growth), autoscale consumers, and DLQ poison messages.
- **At 3am, the safe move is sometimes to STOP.** Mitigate-first still applies for reversible actions, but for irreversible ones (promote, restore, destructive delete) the cheapest mitigation is to pause and confirm. A wrong stateful action is permanent.
- **Storage is a hard cliff, not a soft limit.** Forecast disk with `predict_linear`, alert with days of lead time, account for compaction/vacuum headroom and the migration that needs 2× space.
- **Schema migrations are changes that spend error budget.** Use expand/contract and online DDL, batch backfills, set lock timeouts, and watch replication lag — a migration can cause the stale-read incident.
- **A backup is not a backup until you have restored it.** The untested backup is the most common reason "we had backups and still lost the data." Drill the restore; the time it takes is your real RTO.
- **Match the ceremony to the data.** Full caution for the system of record; stateless-style speed for disposable caches and re-derivable state.

## Further reading

- [Reliability is a feature: the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — the series intro and the define, measure, budget, respond, learn, engineer loop this post lives inside.
- [Redundancy and failover that actually works](/blog/software-development/site-reliability-engineering/redundancy-and-failover-that-actually-works) — the general HA case: active-active vs active-passive, split-brain, fencing, and the spare you never tested.
- [Capacity planning and forecasting](/blog/software-development/site-reliability-engineering/capacity-planning-and-forecasting) — the forecasting method this post specializes for the monotonic-storage disk cliff.
- [Mitigate first, diagnose later](/blog/software-development/site-reliability-engineering/mitigate-first-diagnose-later) — the incident-response principle, with the stateful amendment that the safe mitigation is sometimes to stop.
- `backups-that-actually-restore` (sibling, planned) — the restore drill that turns a hope into a backup, and why an untested backup is no backup.
- [Distributed replication: leader, multi-leader, leaderless](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless) — the database mechanics of replication and consensus this post runs on top of.
- [Database locks and deadlocks in production](/blog/software-development/database/database-locks-and-deadlocks-in-production) — the locking dynamics behind connection-pool exhaustion and migration freezes.
- [Backpressure and flow control](/blog/software-development/message-queue/backpressure-and-flow-control) and [poison messages and retry-storms containment](/blog/software-development/message-queue/poison-messages-and-retry-storms-containment) — the queue mechanics behind the "queue became the outage" story.
- [Distributed race conditions and ordering](/blog/software-development/debugging/distributed-race-conditions-and-ordering) — the read-versus-write race that replication lag turns into a stale-read bug.
- The Google SRE Book and SRE Workbook (chapters on data integrity, managing critical state, and addressing cascading failures); the Prometheus and Alertmanager docs on recording/alerting rules and `predict_linear`; Kafka's consumer-group lag documentation; and the GitLab 2017 database incident postmortem for an honest, public account of the untested-backup trap.
