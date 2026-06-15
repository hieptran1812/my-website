---
title: "Multi-Region Microservices and Data Locality: A Data Problem Wearing a Deployment Costume"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Why going multi-region is never the deployment task it looks like: the speed of light sets your latency floor, a cross-region partition is guaranteed, and the only winning move is to partition data by region and never make a synchronous cross-ocean call on the hot path."
tags:
  [
    "microservices",
    "multi-region",
    "data-locality",
    "distributed-systems",
    "active-active",
    "disaster-recovery",
    "software-architecture",
    "backend",
    "gdpr",
    "replication",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/multi-region-microservices-and-data-locality-1.webp"
---

The ShopFast team had been running happily in a single US region for three years. Then the company signed its first big European retail customer, and within a week the support inbox filled with the same complaint phrased a dozen ways: the site is slow. Not down — slow. Pages took three seconds to feel responsive from Berlin when they took half a second from Boston. A junior engineer pulled up the dashboards, saw green across the board — p99 latency inside the region was a healthy 45ms — and concluded, reasonably, that the customer was imagining it or had bad WiFi. The dashboards were not lying. They were just measuring the wrong thing. They measured how long the servers took to answer once a request *arrived*. They did not measure the 90 milliseconds each request spent crossing the Atlantic in each direction, and they did not measure the dozen round trips a single page load made across that ocean.

The fix, everyone agreed in the design review, was "just deploy to a European region too." That phrase — *just deploy to another region* — is where the trouble starts. It sounds like an infrastructure ticket. Spin up the same Kubernetes manifests in `eu-west-1`, point a load balancer at it, done. Two weeks later the team had EU servers running, and the system was *worse*: the EU checkout flow now called the US fraud service synchronously on every order, so European users paid the 90ms Atlantic crossing twice on the critical path, and when a US deploy briefly hiccuped, checkout failed *in Europe too* — a region that was supposed to make them more reliable now had a brand-new way to fail. They had copied the compute and left the data and the dependencies stranded on the other side of the world. They had treated a data-architecture problem as a deployment problem, and the architecture took its revenge.

This post is about why that revenge is inevitable for the naive approach, and how to do multi-region properly. The thesis, which I will repeat until it is annoying, is this: **multi-region is a data-architecture problem wearing a deployment costume.** Deploying your stateless services to a second region is the easy 20% that everyone sees. The hard 80% is the data: where it lives, who is allowed to write it, how it gets to the other region, what happens when the link between regions breaks, and which law decides whether a row may cross a border at all. Get the data model right and the deployment is mechanical. Get it wrong and no amount of clever routing will save you.

![A branching topology diagram showing US and EU users routed by a geo-router into separate home regions that write region-local data while only the catalog replicates asynchronously across the ocean](/imgs/blogs/multi-region-microservices-and-data-locality-1.webp)

By the end you will be able to do the things the ShopFast team could not. You will be able to say *why* you are going multi-region — and whether you actually need to yet, because most teams do not. You will be able to pick a topology — single-region, active-passive, active-active, or region-partitioned — and name exactly what each one costs. You will know the one rule that, if you break it, undoes every benefit: never make a synchronous cross-region call on the request path. You will know how to partition data by region so the hardest problem in active-active — write conflicts — simply does not arise. And you will be able to reason about a region failure, a cross-region network partition, and a GDPR audit without panicking, because you designed for all three on purpose. We will keep ShopFast as our running example throughout: a US company expanding to the EU, going active-active with users pinned to their home region, with a real region failover and a real data-residency constraint to work through.

## Why go multi-region at all (and the honest answer: maybe don't yet)

Before any topology talk, get the *why* straight, because the why determines the *how*. There are exactly three legitimate reasons to run across regions, and they pull your design in different directions.

**Reason one: user latency.** This is physics, not engineering, and you cannot optimize your way out of it. Light travels through fiber at roughly two-thirds the speed of light in a vacuum — about 200,000 km/s. New York to London is about 5,600 km of great-circle distance, but the actual fiber path is longer, and once you add router hops and the fact that TCP and TLS each need round trips before any data flows, a single request from a European user to a US server costs something like **80–150ms of round-trip time before your server does any work at all**. A page that makes ten serial round trips across that ocean — and many do, between DNS, TLS handshake, the HTML, then the API calls the HTML triggers — has spent a full second on the wire before rendering. No CPU upgrade fixes this. The only fix is to put a server *near* the user, so the round trip is 5–20ms instead of 100ms+. That is the latency case for multi-region, and it is the one most teams hit first.

**Reason two: disaster recovery.** A single region — even a multi-zone single region — can fail entirely. Whole-region outages are rare but they are not theoretical; every major cloud provider has had one, and "rare" over a long enough horizon means "will happen to you." If your entire business lives in one region and that region goes dark, you are down until the provider recovers, which has historically meant hours, not minutes. A second region is an insurance policy: if one dies, the other carries the load. This is the *availability* case, and it is measured in two numbers you must learn cold — **RTO** (recovery time objective: how long until you are serving again) and **RPO** (recovery point objective: how much recent data you are willing to lose). We will do the math on both later.

**Reason three: data residency and sovereignty.** Some data is not allowed to leave a jurisdiction. The EU's GDPR does not literally forbid all transfers of personal data outside the EU, but it makes them legally fraught, and many enterprise customers — and some national laws, like various data-localization rules — simply require that their citizens' personal data is *stored and processed* within their borders. If you signed a contract promising a German bank that its customers' data stays in the EU, then "EU customer PII lives only in the EU region" is not a performance optimization, it is a contractual and legal hard constraint that overrides everything else in your architecture. This is the *sovereignty* case, and it is the one that turns "we'll replicate everything everywhere for resilience" into "we are now legally exposed."

Now the honest part, because this series refuses to sell you complexity you do not need. **You probably do not need multi-region yet.** Multi-region roughly doubles your infrastructure cost, multiplies your operational surface area, and introduces the single hardest class of bug in distributed systems — cross-region data divergence — that you will spend years learning to debug. If your users are concentrated in one geography, if a few hours of downtime during a once-in-three-years regional outage is survivable for your business, and if no law forces your hand, then a single well-run region with good multi-zone redundancy and solid backups is the correct, boring, cheap answer. The teams that go multi-region "to be safe" before they have a latency, availability, or legal *requirement* usually end up with all the cost and complexity and none of the benefit, plus a fragile distributed monolith straddling an ocean. Reach for multi-region when you have a concrete driver, and let that driver — latency, DR, or sovereignty — dictate the topology. We will come back to the decisive recommendation at the end; for now, assume ShopFast has a real one (a paying EU customer who needs both low latency and GDPR compliance) and earn the complexity.

## The cardinal rule: never make a synchronous cross-region call on the request path

If you remember one sentence from this entire post, make it this one: **a synchronous cross-region call on the request path is a bug, even when it works.** Everything else in multi-region architecture is a consequence of taking this rule seriously.

Here is why it is the rule and not merely a guideline. The previous post on [inter-service communication fundamentals and fallacies](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies) established two laws that become brutal at continental scale. The first is that **availability multiplies down a synchronous call chain**: if service A synchronously calls service B to serve a request, A's effective availability is at most A's own availability times B's. The second is that **a synchronous call couples the caller's fate to the callee's** — if B is slow or down, A is slow or down. In a single region those laws are manageable because round trips are sub-millisecond and you control both ends in the same failure domain. Across regions they are catastrophic, for two compounding reasons.

The first reason is latency, and it is the obvious one. A cross-region round trip is not 0.5ms, it is 80–150ms. Put one of those on your hot path and you have added more latency than your entire in-region request budget, all at once, to *every single request*. The dashboards that showed 45ms p99 will show 185ms p99, and that is the *good* case — the case where nothing is wrong.

The second reason is the one that gets you paged, and it is about coupling availability across a link you do not control. When ShopFast's EU checkout synchronously calls the US fraud service, the EU region's checkout availability is now bounded by the US region's availability *times the availability of the network link between the continents*. That trans-oceanic link is the least reliable component in your entire system: it is long, it crosses many administrative boundaries, and a cross-region network partition — where the regions cannot talk to each other for a while — is not an "if," it is a "when." We will dwell on that "when" in the CAP section. The point here is that a synchronous cross-region call means a problem in *one* region, or a problem with the *link*, takes down *both* regions. You built a second region to be more available and instead invented a way to be *less* available, because you doubled your dependencies without decoupling them.

![A before and after comparison showing a synchronous cross-region checkout call that adds 140 milliseconds and couples region availability versus a region-local call that adds 4 milliseconds and lets regions fail independently](/imgs/blogs/multi-region-microservices-and-data-locality-2.webp)

#### Worked example: the 140ms checkout tax

Let us make this concrete with numbers, because the senior move is always to attach a price tag. ShopFast's EU checkout flow, in the naive design, does this on the request path: validate cart (EU, 8ms), reserve inventory (EU, 12ms), and then — the mistake — call the fraud-scoring service, which lives only in the US (cross-Atlantic round trip ~70ms each way, plus 15ms of compute = ~155ms), then charge payment (EU, 30ms). The cross-region fraud call alone dominates: of a ~205ms checkout, **about 140ms is the Atlantic crossing and the remote work behind it.** Remove that one cross-region hop — by running a fraud-scoring replica in the EU that reads a model and rules replicated asynchronously from the US — and the same call becomes ~15ms in-region. Checkout drops from ~205ms to ~65ms, a 68% reduction, *and* EU checkout no longer fails when the US region or the Atlantic link has a bad day. One architectural change bought a latency win and an availability win simultaneously. That is what getting the data architecture right looks like: the optimizations stop fighting each other.

How do you obey the cardinal rule in practice? You make every dependency a service needs to answer a request *present in the same region.* If checkout needs fraud scoring, fraud scoring runs in every region. If it needs the catalog, the catalog is replicated read-only into every region. If it needs something that genuinely only exists elsewhere — a single global ledger, say — then you do not call it synchronously; you write a local record and reconcile asynchronously, or you redesign so the request does not need it. Anything that must cross a region boundary moves off the hot path and onto an asynchronous channel where 100ms of latency and an occasional partition are tolerable. The hot path stays region-local, end to end, all the way down to the datastore. That last clause — all the way down to the datastore — is the whole game, and it is why this is a data problem.

## The four topologies, and what each one actually costs

There are four ways to lay a microservices system across geography. They form a ladder of increasing capability and increasing data difficulty. Walk up it deliberately; do not skip rungs you do not need.

**Single-region** is the bottom rung: everything in one region, multiple availability zones inside it for redundancy. This is correct for most systems. Latency is great for nearby users and bad for far ones; availability is "as good as one region," which is very good but not perfect; data is trivial because there is exactly one copy of the truth and one place that writes it. If the region dies you are down and, depending on your backup strategy, you may lose recent data. Cost is the baseline 1x. Do not leave this rung without a reason.

**Active-passive** (also called primary-secondary or warm standby) is the disaster-recovery rung. One region serves all traffic; a second region runs the same services but takes no live traffic — it is a standby. Data replicates *asynchronously* from the active region to the passive one. When the active region fails, you *fail over*: promote the passive region to active, repoint DNS, and start serving from it. This buys you DR — you can recover from a whole-region outage in minutes instead of hours — at the cost of running a mostly-idle second region (so cost is roughly 1.4x, not 2x, since the standby can be smaller until it is promoted) and the acceptance of an **RPO window**: any writes that had not yet replicated when the active region died are lost. Latency is *not* improved for far users, because they are all still hitting the one active region. Active-passive is the right answer when your driver is *availability/DR* but not latency. It is conceptually simple and the failover is the only hard part.

**Active-active** is the rung where all regions serve live traffic simultaneously. This is the best topology for both latency (every user hits a nearby region) and availability (losing one region just sheds its share of traffic to the others, no failover dance required). It is also, by a wide margin, the hardest data problem in this entire post, because now you have *multiple regions writing data at the same time*, and the moment two regions write to the same logical record without coordinating, you have a conflict — and resolving conflicts correctly is genuinely hard, as the data-consistency literature will tell you. Active-active costs roughly 2x or more (you run full capacity in every region) and demands that you solve the multi-writer problem. Naive active-active — every region writes everything, replicate bidirectionally, hope for the best — is how you get silent data corruption. Which leads to the fourth, and best, rung.

**Region-partitioned active-active** is active-active with one crucial discipline: **each piece of data has exactly one home region that owns its writes.** You shard your users (or accounts, or tenants) by region — a German user's home region is EU, an American user's home is US — and you route each user's *writes* to their home region. Every region serves traffic, so you keep the latency and availability of active-active, but because any given user's data is only ever *written* in one place, the dreaded write-conflict problem *evaporates*. Two regions never write the same user's row, so there is nothing to conflict. Other regions may hold *read-only* copies (for failover or for the rare cross-region read), but the authoritative write path is single-homed per partition. This is the topology most large global systems actually run, and it is the one ShopFast will adopt. Cost is ~2x like active-active, but the data complexity drops from "hardest" to "medium" because you have engineered the conflicts out of existence rather than resolving them after the fact.

![A matrix comparing single-region, active-passive, active-active, and region-partitioned topologies across global latency, availability, data complexity, monthly cost, and recovery objectives](/imgs/blogs/multi-region-microservices-and-data-locality-3.webp)

The matrix above is the decision tool, and here it is again as a table you can copy into a design doc, because trade-offs belong in writing where reviewers can argue with them.

| Property | Single-region | Active-passive | Active-active (naive) | Region-partitioned active-active |
|---|---|---|---|---|
| Latency for far users | Poor (full RTT) | Poor (still one active) | Best (local region) | Best (local region) |
| Availability | One region's | DR via failover | Highest (no failover) | Highest (no failover) |
| Whole-region outage | Down until recovery | Failover, RTO minutes | Shed load, no downtime | Shed load, no downtime |
| Write conflicts | None (one writer) | None (one active) | Hard — must resolve | None — engineered out |
| Data complexity | Trivial | Low | Hardest | Medium |
| Cost (relative) | 1x | ~1.4x | ~2x+ | ~2x |
| RPO / RTO | Loss / hours | Seconds / minutes | ~0 / seconds | ~0 / seconds |
| When it wins | Default; one geography | DR driver, not latency | Avoid — use partitioned | Latency + DR + global users |

Read the table as a ladder, not a menu. You climb to active-passive when you need DR. You climb to region-partitioned active-active when you also need latency for a global user base. You almost never deliberately choose *naive* active-active — it is in the table only so you recognize it as the trap it is. ShopFast, with EU users who need both low latency and GDPR-resident data, lands squarely on region-partitioned active-active, which conveniently also satisfies sovereignty: if EU users' writes go only to the EU region, then EU personal data simply never has a reason to be written in the US.

## Data locality: the whole game in one idea

Strip away the topologies and the routing and the failover, and multi-region reduces to a single principle: **keep data near the users who use it, and let the region that owns the data also serve the requests for it.** That is data locality. Everything we have discussed is in service of it. The cardinal rule (no synchronous cross-region calls) is just data locality applied to dependencies. Region partitioning is data locality applied to writes. Read replicas per region are data locality applied to reads.

Why does locality solve so much at once? Because the cross-region link is simultaneously your slowest component and your least reliable one, and locality means you only touch it *asynchronously and off the request path.* If the data a request needs is in the same region as the service handling the request, then the request never crosses the ocean, which means it is fast (no 100ms tax) and it is resilient (a partition between regions does not affect it). The cross-region link becomes a background channel for replication — and replication can lag, can pause during a partition, and can catch up afterward, all without any user noticing, *as long as no user request is synchronously waiting on it.*

The discipline this demands is to classify every piece of data your system holds by its locality pattern. In ShopFast there are three patterns, and naming them is most of the design:

- **Region-local, written here** — orders, payments, a user's cart and session. These are written by the user's home region and read by their home region. They never need to be written elsewhere. This is the easy, dominant case once you partition by region.
- **Globally-read, written-once-somewhere** — the product catalog, pricing rules, the fraud model. There is one source of truth (say, the US region's catalog service), and every region holds a *read-only asynchronous replica* of it. EU services read the local replica at 2ms; the replica lags the US source by maybe a few seconds, which is totally fine for a product description. This is data locality for reads: replicate the read-heavy, rarely-written global data into every region.
- **Truly global, multi-writer** — the genuinely hard case, like a global inventory count that any region might decrement, or a username-uniqueness registry. This is the data you want to *minimize* and, where it exists, handle with deliberate conflict resolution or by routing all writes for a given key to a single owner. We will spend real time on this.

The senior insight is that the third category is the expensive one, and a good design pushes as much data as possible out of it and into the first two. ShopFast's "global inventory" feels like it must be multi-writer — surely both regions sell the same product? — but if you partition the *inventory* by warehouse, and warehouses are region-local, then EU sales decrement EU warehouse stock and US sales decrement US warehouse stock, and the multi-writer problem disappears again. Most "truly global" data, examined closely, can be re-partitioned into region-local data with a little domain modeling. That is the work, and it is domain work, not infrastructure work — which is exactly why "just deploy to another region" fails.

There is a useful heuristic for the classification: ask, for each dataset, "who writes this, and from where?" If the answer is "exactly one party, always from the same region" it is category one and you are done. If the answer is "many readers everywhere, one writer somewhere" it is category two and you replicate read-only. Only if the honest answer is "anyone, from anywhere, to the same key" is it category three — and that answer is rarer than it first appears, because most data that *feels* global is actually owned by some entity (a user, an account, a tenant, a warehouse) that has a natural home. The trap juniors fall into is treating the *physical* table ("the orders table") as the unit of analysis rather than the *logical* ownership ("this order belongs to this EU user"). Partition by the owner, not the table, and the table follows. When you genuinely cannot find an owner — a true global counter, a uniqueness constraint that spans all users — that is the small residue you handle with the conflict-resolution tools in the next section, and you want that residue to be as small as you can make it, because every byte of it is a future incident.

## The cross-region data problem: replication, conflicts, and CAP

Now we descend into the data layer itself, because this is where multi-region architectures live or die. Three sub-problems: how data gets between regions (replication), what happens when two regions write the same thing (conflict resolution), and what happens when the regions cannot talk (CAP). Each has a dedicated mechanism deep-dive elsewhere on this blog; my job here is the practitioner's framing of how they bite you across regions.

### Replication: async by default, sync almost never across regions

Replication is how a write in one region becomes visible in another. The fundamental choice — covered in depth in [database replication: sync vs async, logical vs physical](/blog/software-development/database/database-replication-sync-async-logical-physical) — is whether the write waits for the replica to confirm (synchronous) or returns immediately and ships the change in the background (asynchronous). Within a single region, synchronous replication to a nearby replica is reasonable: the latency cost is a millisecond or two, and it buys you zero data loss on failover.

**Across regions, synchronous replication is almost always the wrong choice**, and for the now-familiar reason: it puts a cross-region round trip on your write's critical path. If your EU write must wait for the US replica to acknowledge before it returns to the user, every write just inherited the 100ms+ Atlantic tax *and* coupled the EU write path to US availability — the cardinal rule, violated at the storage layer. So across regions you replicate **asynchronously**: the write commits locally, returns to the user fast, and a background process ships the change to the other region, where it applies a few hundred milliseconds to a few seconds later. The cost of async replication is precisely the **RPO window** — the gap between "committed locally" and "replicated remotely" is exactly the data you lose if the source region dies in that gap. You trade a small, bounded risk of recent-data loss for fast, decoupled writes, and across regions that is almost always the right trade.

There is a deeper taxonomy of replication topologies — leader/single-leader, multi-leader, and leaderless — explored in [distributed replication: leader, multi-leader, and leaderless](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless). The mapping to our topologies is clean: active-passive is **single-leader** (one region leads, the other follows asynchronously); naive active-active is **multi-leader** (both regions lead, hence the conflicts); and region-partitioned active-active is *also* effectively single-leader *per partition* (each region leads the writes for its own users and follows for everyone else's read-only copies). That last framing is worth pausing on: **region partitioning turns the hard multi-leader problem into many easy single-leader problems**, one per region-partition. That is precisely why it is the topology to reach for.

![A vertical stack showing the geo-routing layers descending from anycast DNS through a global load balancer and regional load balancer into the service mesh and region-local datastore with async replication to the peer region](/imgs/blogs/multi-region-microservices-and-data-locality-5.webp)

### Conflict resolution: the price of multi-writer, and how to avoid paying it

Suppose you ignore my advice and run genuine multi-writer active-active: both regions accept writes to the same logical record. A user updates their profile from the EU at the same instant a support agent updates it from the US. Both writes commit locally and replicate. Now each region receives a conflicting update for the same record. Which one wins? This is the conflict-resolution problem, and there are three honest answers, each covered in the [data consistency and eventual consistency in practice](/blog/software-development/microservices/data-consistency-and-eventual-consistency-in-practice) sibling post and grounded in the [consistency models from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) deep-dive.

**Last-writer-wins (LWW)** picks the write with the latest timestamp and discards the other. It is simple and it is *lossy*: the discarded write's data is gone, silently. It also depends on clocks being synchronized across regions, which they are not perfectly, so "latest" is fuzzy. LWW is acceptable for data where losing one of two concurrent updates is tolerable (a "last seen" timestamp, a cache entry) and dangerous for anything where both updates carry real information (a shopping cart, an account balance).

**CRDTs (conflict-free replicated data types)** are data structures designed so that concurrent updates *merge* deterministically without losing information — a counter that both regions can increment and that converges to the correct sum, a set that both can add to. CRDTs are the principled answer to multi-writer data, and they are real (Riak, Redis, and others ship them; collaborative editors like the ones behind Google Docs use the closely related idea). The catch is that not all data fits a CRDT cleanly, and a CRDT has overhead — it carries metadata to track causality. Use them where the data model genuinely is a counter, a set, or a register and you can tolerate eventual convergence.

**Partition by region (the one I keep recommending)** is the answer that avoids the question. If each record has exactly one home region that owns its writes, there are no concurrent writes to the same record, so there is nothing to resolve. This is not a conflict-resolution *strategy*; it is a conflict-*avoidance* architecture, and it is strictly easier to reason about than either LWW or CRDTs because the conflict never exists. The cost is that a user's writes must be routed to their home region — which is a routing problem, and routing is much easier to get right than distributed conflict resolution.

![A branching diagram showing requests routed by home region so US-home users write the US partition and EU-home users write the EU partition with read-only copies flowing to the other region](/imgs/blogs/multi-region-microservices-and-data-locality-6.webp)

### CAP across regions: the partition is not optional

Here is the truth that makes everyone uncomfortable, and it is non-negotiable physics-meets-networking: **a cross-region network partition will happen.** The link between your regions — undersea cables, transit providers, BGP, all of it — *will* at some point fail or degrade to the point where your regions cannot reliably talk to each other. Not "might." Will. Over a multi-year horizon it is a certainty, and you must have decided in advance what your system does during those minutes or hours.

This is the CAP theorem applied to geography, and CAP — explored fully in [the CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) — says that when a partition (P) happens, you must choose between consistency (C) and availability (A). You cannot have both during the partition. For a cross-region system this is not an abstract trilemma; it is a concrete operational decision: *when the EU region cannot reach the US region, does the EU region keep serving EU users (choosing availability, accepting that its data will diverge from the US until they reconnect) or does it refuse to serve (choosing consistency, accepting downtime)?*

For most user-facing systems the answer is **choose availability**: keep serving. A user in Berlin should still be able to browse and buy even if the trans-Atlantic link is down, because their data lives in the EU region anyway and the partition does not affect it. This is exactly why region partitioning is so powerful under CAP: **if each user's data is local to their region, a cross-region partition barely matters to them**, because their requests never needed the other region in the first place. The partition only affects the things that genuinely span regions — cross-region reads of replicated data (which simply serve slightly staler data until the link returns) and the replication stream itself (which pauses and catches up). The blast radius of the guaranteed partition shrinks to almost nothing, *because you localized the data.* A naive active-active system, by contrast, faces an ugly choice during every partition: keep accepting conflicting writes in both regions (and deal with the mess later) or stop accepting writes (and go down). Region partitioning makes the partition a non-event for the common path. PACELC adds the "else" clause: even when there is no partition (E), you still trade latency (L) against consistency (C) — and across regions the latency cost of consistency is so high that you almost always pick latency, i.e., async replication and local reads.

It is worth being precise about what "the partition heals" actually involves, because teams underestimate it. While the link is down, each region has been accumulating writes its peer has not seen and replication has been queueing up un-shipped changes. When connectivity returns, the replication stream must drain that backlog — potentially many minutes of accumulated changes — and during that drain the cross-region replicas are *still* stale, just catching up. If your backlog grew large (a long partition under heavy write load), the catch-up itself can take a while and can saturate the link, so a 20-minute partition might mean 25 minutes before the replicas are fully consistent again. None of this hurts the home-local request path — EU users were never waiting on US data — but it does mean your *cross-region read staleness* metric spikes during and after a partition, and you should alarm on it so an operator knows the replicas are behind. The system that survives the partition gracefully is the one where staleness is a monitored, bounded, *expected* condition rather than a surprise. That mindset — treating staleness as a normal operating state with an upper bound rather than an anomaly — is the difference between a team that sleeps through a cross-region partition and one that gets paged into a panic.

## Routing users to the right region: geo-DNS, anycast, and global load balancing

We have decided *where* data lives. Now we need to get each user *to* the right region — both to the nearest healthy region for latency, and to their *home* region for writes. This is the routing layer, and it has two jobs that are easy to conflate but are different.

The first job is **proximity routing for reads and stateless work**: send a user to the *geographically nearest healthy region* so their latency is low. This is what geo-DNS and anycast do.

- **Geo-DNS** resolves your domain to different IP addresses based on the resolver's location. A DNS query from a German ISP gets the EU region's IP; one from a US ISP gets the US region's IP. It is simple and widely supported, but it has a coarse granularity (it routes by resolver location, not user location, and it is subject to DNS caching, so changes propagate at the speed of your TTL).
- **Anycast** announces the *same* IP address from multiple regions via BGP, and the internet's routing fabric delivers each packet to the topologically nearest announcement. It is the mechanism global load balancers and CDNs rely on internally, and it fails over fast — if a region stops announcing, traffic reroutes to the next-nearest within seconds, no DNS TTL to wait on.

A modern global load balancer (a cloud provider's global LB, or a CDN with origin routing) combines these: anycast entry, health-aware routing that pulls a region out of rotation when its health checks fail, and the ability to route by geography. Here is a representative geo-routing configuration — the shape is the same across providers:

```yaml
# Global load balancer with health-aware geo routing.
# Sends each user to the nearest HEALTHY region; on region failure,
# anycast reroutes within seconds without waiting on DNS TTL.
globalLoadBalancer:
  name: shopfast-global
  anycastIP: 203.0.113.10        # same IP announced from both regions
  defaultRegion: us-east-1        # fallback if geo lookup is ambiguous
  healthCheck:
    path: /healthz/ready          # readiness, not just liveness
    intervalSeconds: 5
    unhealthyThreshold: 3         # 3 fails => pull region from rotation (~15s)
    healthyThreshold: 2
  regions:
    - id: us-east-1
      backend: us-east-1-region-lb
      geoMatch: [NA, SA]          # North & South America -> US
    - id: eu-west-1
      backend: eu-west-1-region-lb
      geoMatch: [EU, AF, ME]      # Europe, Africa, Middle East -> EU
  failover:
    mode: nearest-healthy         # if a region is unhealthy, route to the other
    drainSeconds: 30              # graceful drain on planned removal
```

The second job is the subtle one: **home-region routing for writes.** Proximity routing sends a *traveling* German user who happens to be in New York to the US region — which is correct for latency, but their *data* lives in the EU. If their write lands in the US region, you are back to a cross-region write. So for write operations you route by the user's *home region* (a property of their account, decided when they signed up), not their current location. The two routings coexist: proximity for reads and stateless work, home-region for writes. In practice you implement home-region routing at the edge or in the gateway by reading a claim from the user's auth token (their JWT carries `home_region: eu`) and routing write requests accordingly — which is one more reason the [auth and token-propagation](/blog/software-development/microservices/authentication-and-authorization-oauth2-jwt-token-propagation) work matters here.

```python
# Home-region write routing at the API gateway.
# Reads home_region from the validated JWT claim and forwards writes
# to that region. Reads stay local (handled by proximity routing).
from fastapi import Request, HTTPException
import httpx

REGION_ENDPOINTS = {
    "us": "https://us-east-1.internal.shopfast.com",
    "eu": "https://eu-west-1.internal.shopfast.com",
}
LOCAL_REGION = "eu"  # this gateway instance runs in the EU region

async def route_request(request: Request, claims: dict):
    is_write = request.method in ("POST", "PUT", "PATCH", "DELETE")
    home = claims.get("home_region")
    if is_write:
        if home is None:
            raise HTTPException(400, "write requires a home region")
        target = REGION_ENDPOINTS[home]
        # If the user's home region is THIS region, stay local (fast path).
        # If not, this is a rare traveling-user write: forward, do not
        # write locally, so the user's data is never split across regions.
        if home == LOCAL_REGION:
            return await forward_local(request)
        return await forward_cross_region(request, target)  # rare, async-friendly
    # Reads serve from the local region's replica (already proximity-routed here).
    return await forward_local(request)
```

That `forward_cross_region` path for a traveling user's *write* is the one exception where a request touches another region — and notice it is rare (most users write from home), and you should design these writes to tolerate the latency (a profile update can take 150ms; a checkout for a traveling user is the genuinely awkward case, handled by either accepting the latency for that minority or by having the home region's data also readable locally). The vast majority of writes are home-local and fast.

## Region-local data: read-local, write-home in code

The pattern that ties data locality together at the application layer is **read-local, write-home**: a service reads from its region-local datastore (fast, always) and writes to the data's home region (local for home users, the rare forwarded write otherwise). Let me show it concretely for ShopFast's order service.

```go
// Order service: read-local, write-home.
// Reads always hit the region-local replica. Writes for the user's
// home region commit locally; the order is then replicated async.
package orders

type Store struct {
    localDB     *sql.DB // region-local primary for this region's home users
    localReplica *sql.DB // read replica of cross-region data (e.g. catalog)
    region      string  // "eu"
}

// Read is ALWAYS local. Catalog/product data is an async replica;
// the user's own orders are in the local primary (they are home here).
func (s *Store) GetOrder(ctx context.Context, orderID string) (*Order, error) {
    return queryOrder(ctx, s.localReplica, orderID) // 2ms, never crosses ocean
}

// Write is home-region. The gateway has already ensured this request
// only reaches a region where the user is home, so a local commit is correct.
func (s *Store) CreateOrder(ctx context.Context, o *Order) error {
    o.HomeRegion = s.region            // stamp the owning region
    if err := insertOrder(ctx, s.localDB, o); err != nil {
        return err                     // local commit: fast, no cross-region wait
    }
    // The async replicator (CDC/outbox) ships this to the peer region
    // as a READ-ONLY copy for failover. No synchronous cross-region call.
    return nil
}
```

The replication itself runs out-of-band. The robust way to get a write reliably from one region's database into the replication stream is the **transactional outbox** plus **change data capture** — the same mechanism the [transactional outbox and reliable event publishing](/blog/software-development/microservices/the-transactional-outbox-and-reliable-event-publishing) post covers in depth — so that you never lose a change and never publish a change that did not actually commit. Across regions, the outbox events flow over an async pipeline (a cross-region Kafka mirror, or the database's native async replication) into the peer region. Here is the replication channel configured as async, region-aware, and compressed (compression matters for the egress bill, as we will see):

```yaml
# Cross-region async replication of the order stream (read-only copy in peer).
# Async => no cross-region wait on the write path. Compression => lower egress $.
crossRegionReplication:
  source:
    region: eu-west-1
    stream: orders.outbox          # transactional outbox -> CDC -> stream
  destination:
    region: us-east-1
    table: orders_eu_readonly      # READ-ONLY copy; US never writes EU orders
  mode: async                       # the cardinal rule, at the storage layer
  compression: zstd                 # ~3-5x smaller payloads => lower egress bill
  batchMs: 200                      # batch up to 200ms => fewer, fatter transfers
  maxLagAlertSeconds: 10            # page if replication lag exceeds RPO target
  conflictPolicy: none              # region-partitioned => conflicts impossible
```

Notice `conflictPolicy: none`. That is not laziness — it is the payoff of region partitioning. Because EU orders are only ever written in the EU and only ever *replicated* (never written) to the US, there is no conflict to resolve, ever. The replication is a one-way, read-only feed. If we had chosen naive active-active, this config would need an LWW or CRDT conflict policy and a whole reconciliation pipeline behind it. We engineered that complexity away by partitioning.

## Failover: promoting a region, the RPO window, and failback

Active-passive lives or dies on the failover, and even active-active needs a failover story for the read-only copies. Let us walk the full lifecycle of a region failure, because the hand-wavy version ("just promote the other region") hides the two numbers that actually matter and the one decision that loses data.

When a region dies, six things happen in sequence, and the timeline below shows them with realistic clocks. Health checks detect the failure (a few seconds to a few tens of seconds, depending on your thresholds — too aggressive and you flap, too lax and you extend the outage). The router pulls the dead region out of rotation. You *promote* the surviving region's replica to be a writable primary for the dead region's data. DNS or anycast reroutes the dead region's traffic to the survivor (anycast in seconds, DNS bounded by TTL). The survivor begins serving the combined load. And then comes the number that hurts: the **RPO window** — the writes that committed in the dead region but had not yet replicated to the survivor when it died — are *lost*, or at best stuck in the dead region's storage until it recovers.

![A six-event timeline of an active-passive failover from region death through health-check detection, replica promotion, DNS reroute, and full traffic to the resulting RPO data-loss window](/imgs/blogs/multi-region-microservices-and-data-locality-4.webp)

#### Worked example: the failover RTO/RPO math

ShopFast runs active-passive for its (hypothetical) DR-only US service. The US region dies at 02:00. Health checks are set to 3 failures at 5-second intervals, so detection takes ~15 seconds. Promoting the EU replica to writable primary is automated and takes ~75 seconds (it must replay the last bit of replication log and flip to read-write). DNS TTL is 60 seconds, so the slowest clients reroute within a minute; anycast clients reroute in seconds. Add a safety drain and verification step of ~90 seconds. Total **RTO ≈ 15 + 75 + 60 + 90 ≈ 4 minutes** until full service is restored — versus the *hours* a single-region outage would cost while waiting for the provider. Now the **RPO**: async replication was running with a typical lag of ~2 seconds. So the writes lost are those committed in the final ~2 seconds before death — at ShopFast's ~500 writes/sec, that is roughly **1,000 orders in limbo.** Those orders are not "gone forever" if the dead region's storage survived (you reconcile them when it returns), but they are *unavailable and possibly invisible* during the outage, and any that depended on volatile state may be unrecoverable. The lesson: **your RPO is exactly your replication lag at the moment of failure.** Want a smaller RPO? Reduce replication lag (smaller batches, more bandwidth) — but you can never reach zero with async replication, and reaching zero requires synchronous cross-region replication, which violates the cardinal rule on every write. This is the irreducible trade-off: fast writes *xor* zero data loss across regions. Choose your RPO budget consciously.

After the dead region recovers, you must **fail back** — and failback is more dangerous than failover, which surprises people. The recovered region's storage is now *stale* (it missed all the writes that happened on the survivor during the outage) and may *also* hold those ~1,000 limbo writes the survivor never saw. Naively repointing traffic back to the recovered region would resurrect stale data and lose everything written during the outage. The correct failback re-replicates the survivor's current state *into* the recovered region first, reconciles the limbo writes (often by hand, or via an idempotent replay — see [idempotency and exactly-once effects across services](/blog/software-development/microservices/idempotency-and-exactly-once-effects-across-services)), verifies the recovered region is caught up, and *only then* routes traffic back, usually gradually. Here is a condensed failover runbook of the kind you want written *before* the incident, not improvised at 2am:

```bash
#!/usr/bin/env bash
# failover-to-eu.sh — promote EU region after US region failure.
# RUN ORDER MATTERS. This is rehearsed quarterly, not improvised at 2am.
set -euo pipefail

# 1. Confirm US is actually down (avoid split-brain from a flaky health check).
if ! confirm_region_down us-east-1 --quorum 3; then
  echo "US not confirmed down by 3 independent probes; ABORTING failover."
  exit 1
fi

# 2. Stop replication INTO the dead region's copy (prevent zombie writes later).
replication-ctl pause --source eu-west-1 --dest us-east-1

# 3. Promote EU replica of US-home data to writable primary.
db-ctl promote --region eu-west-1 --dataset us_home --to-primary
echo "RPO note: writes in the last $(replication-ctl lag --region us-east-1)s are at risk."

# 4. Reroute US traffic to EU at the global LB (anycast handles most clients).
glb-ctl set-region-weight us-east-1 0
glb-ctl set-region-weight eu-west-1 100

# 5. Verify EU is serving combined load within SLO before declaring success.
verify-slo --region eu-west-1 --p99-ms 250 --error-rate 0.01 --window 120s

echo "Failover complete. DO NOT fail back until US is re-synced (see failback runbook)."
```

The deeper lesson is that **failover must be rehearsed.** A runbook nobody has run is fiction. Practice it on a schedule (this is where chaos engineering and game-days from [testing microservices from unit to chaos](/blog/software-development/microservices/testing-microservices-from-unit-to-chaos) earn their keep) so that when the real region dies, the team executes a drill instead of inventing a procedure under pressure.

## The cost of multi-region: egress, duplicated capacity, and operational tax

Multi-region is expensive in ways that do not show up until the bill arrives, and a senior engineer surfaces these costs in the design review, not the finance review. There are three buckets.

**Inter-region data egress** is the one that surprises teams. Cloud providers charge little or nothing for data *into* a region and for traffic *within* a region, but they charge real money for data *leaving* a region across the internet or their backbone — typically on the order of \$0.02 per GB between regions, sometimes more. That sounds tiny until you replicate a high-volume event stream across the ocean continuously. If ShopFast naively replicates *every event* — every page view, every click, every internal message — bidirectionally, the volume balloons. The fix is to replicate only what the other region actually needs, and to compress it.

![A before and after comparison showing replicating every event producing 50 terabytes a month of egress versus replicating only compressed state deltas producing 5 terabytes a month at the same rate](/imgs/blogs/multi-region-microservices-and-data-locality-7.webp)

#### Worked example: the egress bill, naive vs lean

Naive design: replicate the full event firehose both ways. Say ShopFast generates 50 TB/month of events that get shipped cross-region. At \$0.02/GB, that is 50,000 GB × \$0.02 = **\$1,000/month** in egress, and that is *just* the replication — before any user-facing cross-region traffic. Now the lean design: replicate only *authoritative state changes* (orders, profile updates — the data the peer region needs for failover and the rare cross-region read), not the full event stream, and compress with zstd at ~4x. The authoritative changes are a fraction of the firehose — say 5 TB/month before compression, ~1.25 TB after — but let us be conservative and say the *billed* cross-region volume drops to ~5 TB/month including overhead: 5,000 GB × \$0.02 = **\$100/month.** A 10x reduction from one design decision — *what* you replicate — with no loss of correctness, because the events you stopped shipping were never needed in the other region. Egress is a tax on chattiness; data locality is the deduction. The same principle from [performance and cost optimization in microservices](/blog/software-development/microservices/performance-and-cost-optimization-in-microservices) — measure the bottleneck, attack the biggest line item — applies directly: profile your egress, find the chattiest stream, and ask whether the other region truly needs it.

**Duplicated capacity** is the second bucket. Active-active runs full (or near-full) capacity in every region, so two regions cost roughly 2x one region for compute and storage. This is unavoidable for active-active — it is the price of having every region able to serve. Active-passive softens it: the passive region can run smaller (it only needs enough to take over, and can scale up on promotion), so it is closer to 1.4x. The optimization here is to right-size the passive region and use autoscaling so the standby is cheap until it is needed — but never so small that it cannot absorb the failover load fast enough, which would blow your RTO.

**Operational tax** is the third and least visible bucket: every region is another thing to deploy to, monitor, patch, secure, and debug. Your CI/CD must roll out to multiple regions safely (canary in one region before the other — see [deployment strategies: blue-green, canary, feature flags](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags)). Your observability must let you slice metrics *by region* so you can tell "the EU region is slow" from "everything is slow." Your on-call must understand region failover. None of this is a line item on the cloud bill, but it is real cost in engineering time, and it is the cost that makes "you probably don't need it yet" the honest default.

## Data sovereignty: when the law is your architecture

For some data, the constraint is not latency or cost but *law*, and law does not negotiate. ShopFast's EU customer requires that EU residents' personal data stays in the EU. Under GDPR, transferring EU personal data to the US is legally possible only under specific mechanisms and remains a compliance burden; the simplest, most defensible posture is to *not transfer it at all* — to store and process EU personal data only in the EU region. That requirement reaches deep into your architecture, and the region-partitioned design happens to satisfy it almost for free: if EU users' writes go only to the EU region, then EU personal data is *born* in the EU and never has a reason to leave.

But "almost for free" is not "free." You must be able to *prove* it, which means classifying every piece of data by its residency rules and enforcing those rules in the replication layer, not in a wiki page that says "please don't replicate PII." The mechanism is a **data-residency tag** on every dataset and a policy that gates replication on the tag.

![A matrix classifying ShopFast data categories by whether they may leave the EU, whether they replicate to the US, and which region stores them, keeping personal data EU-only](/imgs/blogs/multi-region-microservices-and-data-locality-8.webp)

The matrix above is the classification ShopFast lands on: EU customer PII and EU order history are tagged EU-only and the replication policy *blocks* them from crossing to the US; the global catalog (no personal data) replicates freely; aggregated metrics replicate only after anonymization; auth tokens never replicate at all. Here is that policy as enforceable configuration — the kind a compliance auditor can read and a pipeline can enforce:

```yaml
# Data residency policy. The replicator REFUSES to ship any dataset
# whose residency tag forbids the destination region. Enforced in code,
# auditable, not a guideline.
dataResidencyPolicy:
  version: 2
  rules:
    - dataset: customer_pii_eu
      residency: EU_ONLY
      mayReplicateTo: []              # NOTHING. Never leaves the EU.
      classification: personal-data
    - dataset: orders_eu
      residency: EU_ONLY
      mayReplicateTo: []              # read-only copy stays in EU only
      classification: personal-data
    - dataset: product_catalog
      residency: GLOBAL
      mayReplicateTo: [us-east-1, eu-west-1]
      classification: non-personal
    - dataset: metrics_aggregated
      residency: GLOBAL
      mayReplicateTo: [us-east-1, eu-west-1]
      classification: anonymized       # must pass anonymization gate first
      requires: [anonymization-verified]
    - dataset: auth_tokens
      residency: HOME_ONLY
      mayReplicateTo: []               # tokens never cross regions
      classification: secret
  enforcement: hard                    # replicator FAILS CLOSED on violation
  onViolation: block-and-alert         # never silently ship regulated data
```

The crucial property is `enforcement: hard` with `failClosed` semantics: if the replicator encounters a dataset whose residency it cannot verify as allowed, it *refuses to replicate it* and alerts, rather than shipping it and asking forgiveness. Compliance failures are not eventually-consistent; you want them to fail loudly and immediately. And the residency tag must travel *with the data through the whole pipeline* — through the outbox, the CDC stream, the replicator — so that a tagged EU-PII record cannot be laundered into an untagged stream and accidentally cross the ocean. This is where the [shared-data anti-patterns](/blog/software-development/microservices/shared-data-anti-patterns-and-the-distributed-monolith) discipline matters: a shared, untagged data lake that pulls from every service is precisely how regulated data leaks across a border by accident.

## Stress-testing the design: three scenarios that break the naive version

A design is only trustworthy after you have tried to break it. Here are the three failure scenarios every multi-region system must survive, posed against ShopFast's region-partitioned active-active design, with the naive version's failure shown for contrast.

**Stress test 1: an entire region goes down — what's the RPO?** The EU region dies at peak. In the *naive* design where the US synchronously calls EU services, this is catastrophic: US requests that touched EU data now fail or hang. In ShopFast's region-partitioned design, the blast radius is bounded to EU-home users: their writes were single-homed in the EU, so until failover completes, EU users cannot write (their home is down) and the US region serves only its own US-home users, unaffected. EU users fail over to the read-only EU copy that lives... wait — under strict GDPR, that copy is *also* in the EU (it cannot be in the US), so DR for EU data means a *second EU availability zone or a second EU region*, not the US region. This is the subtle interaction: **sovereignty constrains where your DR copy can live.** ShopFast's answer is a second EU region (`eu-central-1`) holding the read-only EU replica, so EU data has DR *within the EU*. The RPO is the replication lag to that EU DR copy — a few seconds — and the lost writes are the few seconds of EU orders in flight at the moment of death. The US region is completely unaffected. That is the design working: the failure is *contained by region*, exactly as intended.

**Stress test 2: a cross-region partition — the active-active write conflict.** The trans-Atlantic link drops for 20 minutes. In a *naive* multi-leader active-active system, both regions keep accepting writes to the same records, and when the link returns, you have a pile of conflicting writes to reconcile — the LWW-loses-data or CRDT-merge problem, at scale, under time pressure. In ShopFast's region-partitioned design, *there is no conflict to have*, because no record is written in two regions. During the partition, EU users write EU data (fine, it is local), US users write US data (fine, it is local), and the only casualty is that the read-only cross-region replicas go stale — a US user reading the EU-owned catalog sees data up to 20 minutes old, which for a product catalog is invisible. When the link returns, replication catches up; no reconciliation, no conflict, no data loss. The partition that would have been a multi-hour reconciliation incident is a non-event. *This is the single biggest argument for partitioning over multi-leader.*

**Stress test 3: a synchronous cross-region call sneaks onto the hot path.** Six months after launch, a well-meaning engineer adds a feature: show EU users their "global loyalty points," which are tracked in a single US ledger, and they call it synchronously on the EU profile page. Latency for EU users jumps 90ms+ on that page, and worse, when the US region has a blip, the EU profile page *fails* — a region that was supposed to be independent now has a US dependency on a hot path. This is the cardinal rule being violated in a code review nobody flagged. The fix is the same as always: do not call the US ledger synchronously from the EU hot path. Either replicate the loyalty balance read-only to the EU (accept it is a few seconds stale — fine for a points display), or fetch it asynchronously and render the page without it, filling it in when it arrives ([graceful degradation](/blog/software-development/microservices/handling-partial-failures-and-graceful-degradation)). The architectural defense is to make cross-region calls *visible and hard*: instrument them, alert on any synchronous cross-region call on a user-facing path, and treat one appearing in a trace as a bug to be filed. The discipline does not maintain itself; you have to enforce the cardinal rule with tooling, because the next ShopFast engineer will not have read this post.

## Optimization: making multi-region fast and cheap on purpose

Beyond the cardinal rule, there is a toolkit for squeezing latency and cost out of a multi-region system. Each has a measurable win.

**Read replicas per region** are the workhorse. Most workloads are read-heavy, so replicating read-only copies of global data (catalog, pricing, config) into every region means the common read path never crosses a region. Measure the win as the fraction of reads that now serve locally: if 95% of EU reads hit the local replica at 2ms instead of crossing to the US at 90ms, your EU read p99 collapses from ~95ms to ~5ms. The cost is the replication egress (which you minimize as above) and the staleness window (a few seconds, usually fine for read-mostly data).

**Region partitioning of writes** — the thing we keep returning to — is itself the biggest write-latency optimization, because it makes every write home-local. The win is the elimination of the cross-region write tax: writes go from ~100ms+ (if they had crossed) to in-region single-digit milliseconds, and the conflict-resolution machinery (and its bugs) disappears entirely. This connects to [database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding): region is just another sharding key, with the special property that the shard boundary aligns with a network/geography boundary.

**Edge caching** pushes read latency below even a regional round trip. A CDN or edge cache in front of your regions serves static and cacheable dynamic content from points of presence *within tens of milliseconds of the user* — closer than your region. For ShopFast, caching product images, catalog pages, and even personalized-but-cacheable fragments at the edge means many requests never reach a region at all. This is the bridge to the forthcoming [caching strategies across services](/blog/software-development/microservices/caching-strategies-across-services) post, which goes deep on cache placement, invalidation, and consistency; the multi-region angle is simply that the edge is the outermost, fastest layer of your locality story, and the cheapest place to serve a read is the one nearest the user.

**Avoiding cross-region chatter** is the cost optimization with the highest leverage, as the egress worked example showed. Audit every byte crossing a region boundary and ask whether it must. Batch and compress replication. Replicate state, not events. Co-locate chatty service pairs in the same region (if service A calls service B a hundred times per request, they must be in the same region — this is also a [Conway's-law and team-topologies](/blog/software-development/microservices/conways-law-and-team-topologies-for-microservices) signal that A and B might even belong in the same team or service). The measurement is your monthly inter-region egress GB and your count of cross-region calls per user request — drive both toward zero.

```yaml
# Per-region read replica + edge cache config: serve reads locally,
# cache at the edge, and alert if cross-region read ratio creeps up.
readPath:
  edgeCache:
    provider: cdn
    cacheable: [/catalog/*, /products/*, /static/*]
    ttlSeconds: 300                 # 5-min freshness; catalog tolerates it
    staleWhileRevalidate: 60
  regionReplica:
    dataset: product_catalog
    mode: async-read-only
    maxStalenessSeconds: 10
  guards:
    crossRegionReadRatioMax: 0.02   # alert if >2% of reads cross a region
    crossRegionSyncCallsOnHotPath: 0 # alert on ANY; cardinal-rule tripwire
```

## Case studies

Real systems have walked this exact path. The lessons are accurate at the order-of-magnitude level; where a specific number is uncertain I have framed it as such.

**Global active-active for payments and streaming.** Large payment networks and global streaming platforms run active-active across regions precisely because they cannot tolerate a regional outage taking down a continent's worth of users, and they need low latency everywhere. The recurring lesson from their public engineering writeups is the one this post is built on: they do *not* do free-for-all multi-writer. They partition. A payments platform routes a given account's or merchant's writes to a home region or cell, so that the authoritative write for any entity has a single owner, and cross-region traffic is asynchronous replication for resilience plus the rare cross-region read. Streaming platforms similarly keep a user's playback state and account writes region-homed and replicate the (largely read-only, globally-shared) content catalog everywhere. The shared takeaway: **active-active at global scale is almost always region-partitioned active-active under the hood, not naive multi-leader.** The companies that learned this the hard way wrote the postmortems that taught everyone else.

**A region-failover postmortem.** The canonical lesson from real regional-outage postmortems — across multiple cloud providers and the companies riding on them — is that **the failover is harder than the failure.** Teams discover during a real outage that their "standby" region had drifted out of config parity, or that their failover automation had never been tested at full load, or that DNS TTLs were set to hours so reroute took far longer than the RTO they had promised, or — the nastiest one — that failing back after recovery resurrected stale data and lost everything written during the outage. The consistent remediation is the same set of practices: keep the standby in continuous config parity (deploy to it on every release), rehearse failover regularly as a game-day, set DNS TTLs low (or use anycast to sidestep DNS entirely), and treat failback as a careful re-sync, never a flip of the switch. The number that recurs in these stories is the RPO surprise: teams assumed "near-zero data loss" and discovered their async replication lag at peak was tens of seconds, meaning a real region death would have lost a meaningful chunk of recent transactions. Measure your replication lag at peak; that *is* your RPO.

**GDPR data-locality compliance.** Companies serving EU customers have repeatedly had to retrofit data residency, and the lesson is uniformly that **residency is an architecture concern, not a checkbox.** Teams that tried to bolt on "keep EU data in the EU" after building a globally-replicated system found EU personal data scattered through logs, analytics pipelines, backups, and shared data lakes — every one of which had to be audited and re-architected to fail closed at the border. The teams that built region-partitioning in from the start, with residency tags enforced in the replication layer, had a far easier compliance story because the data was born in the right place and *structurally* could not leave. The order-of-magnitude lesson: retrofitting residency onto a globally-replicated system can be a multi-quarter program touching dozens of services; designing for it up front is mostly free if you were going to partition by region anyway. Sovereignty and data locality want the same architecture, which is a happy accident worth exploiting.

## When to reach for multi-region (and when the single region wins)

Let me be decisive, because this series promises decisive recommendations and refuses to sell complexity.

**Stay single-region** if: your users are concentrated in one geography; you can tolerate a rare (once-in-a-few-years) multi-hour regional outage; and no law or contract forces residency. This is most companies, including most companies that *think* they need multi-region. A single region with good multi-zone redundancy, automated backups, and tested restore procedures is dramatically simpler, roughly half the cost, and free of the entire class of cross-region data bugs. Spend your complexity budget elsewhere until you have a concrete driver.

**Go active-passive** if your driver is purely *disaster recovery* — you need to survive a regional outage with minutes of RTO and seconds of RPO — but your users are still mostly in one geography so latency does not demand a second active region. It is the cheapest way to buy DR (~1.4x cost), and the failover is the only hard part, so invest in rehearsing it.

**Go region-partitioned active-active** if you have a *global* user base that needs low latency *and* you want the best availability *and/or* you have data-residency requirements. This is the topology for genuinely global products. It costs ~2x and demands the discipline of this whole post — partition by region, never call synchronously across regions, replicate read-only and async, enforce residency at the border — but it gives you low latency everywhere, no-failover regional fault tolerance, and sovereignty compliance in one coherent design.

**Never go naive multi-leader active-active** — every region writing everything with bidirectional replication and hope. It looks like region-partitioned active-active on a deployment diagram but it is a data-corruption generator. If you find yourself reaching for LWW or CRDTs to "resolve conflicts" across regions, stop and ask whether you can re-partition the data so the conflicts cannot occur. Usually you can.

![A six-event timeline of ShopFast expanding from the US to the EU by standing up a region, async-replicating the catalog read-only, adding home-region routing, removing the sync fraud call, geo-routing EU users, and passing a GDPR audit](/imgs/blogs/multi-region-microservices-and-data-locality-9.webp)

The timeline above is the order ShopFast actually executed the expansion, and the *order* is the lesson: stand up the region, replicate the read-only global data first, add home-region routing, *remove* the synchronous cross-region dependency, *then* send users to the new region, and finally verify compliance. Notice that "deploy services to EU" was step one and the least important — everything after it was the data and dependency work that the naive team skipped, and that work is what made the expansion succeed instead of regress.

## Key takeaways

- **Multi-region is a data-architecture problem, not a deployment problem.** Copying your stateless services to a second region is the easy 20%. Where data lives, who writes it, how it replicates, and what the law allows is the hard 80% and the part that determines success.
- **Never make a synchronous cross-region call on the request path.** It adds 80–150ms to every request and couples your regions' availability through the least reliable link in your system. This one rule generates most of the others.
- **Keep data local to the users who use it.** Region-local writes, per-region read replicas of global data, and edge caching turn the slow, unreliable cross-region link into a background concern instead of a hot-path dependency.
- **Partition by region to make write conflicts impossible.** Region-partitioned active-active gives you the latency and availability of active-active while turning the hardest multi-leader problem into many easy single-leader problems. Avoid naive multi-leader; you do not want to resolve cross-region conflicts if you can engineer them out.
- **Your RPO is your replication lag at the moment of failure, and you cannot make it zero with async replication.** Across regions you choose async (fast writes, small bounded data-loss window) over sync (zero loss, but a cross-region wait on every write). Measure your lag at peak — that number *is* your RPO.
- **A cross-region partition is guaranteed; design for it.** Under CAP, decide in advance whether a region keeps serving (availability) or refuses (consistency) when it cannot reach its peer. Region partitioning makes most partitions a non-event because local requests never needed the peer.
- **Egress and duplicated capacity are real costs; replicate state, not chatter.** Inter-region egress at ~\$0.02/GB punishes chattiness; replicating only authoritative state changes with compression can cut the bill 10x with no loss of correctness.
- **Sovereignty is architecture, not a checkbox.** Tag data by residency, enforce it fail-closed in the replication layer, and let region partitioning satisfy GDPR almost for free by making regulated data born — and kept — in the right region.
- **Rehearse failover; failback is harder than failover.** A runbook nobody has run is fiction. The standby must be in config parity, DNS TTLs low or anycast, and failback a careful re-sync, never a flip of a switch.
- **You probably don't need multi-region yet.** It roughly doubles cost and complexity. Reach for it only when you have a concrete latency, DR, or sovereignty driver — and then let that driver pick the topology.

## Further reading

- [Inter-service communication fundamentals and fallacies](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies) — the fallacies and the availability-multiplication math that the cardinal rule rests on.
- [Performance and cost optimization in microservices](/blog/software-development/microservices/performance-and-cost-optimization-in-microservices) — the measure-the-bottleneck discipline applied to the egress bill and latency budget.
- [Caching strategies across services](/blog/software-development/microservices/caching-strategies-across-services) — edge and regional caching, the outermost layer of your data-locality story.
- [Data consistency and eventual consistency in practice](/blog/software-development/microservices/data-consistency-and-eventual-consistency-in-practice) — conflict resolution, LWW, and what "eventual" really costs.
- [The CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) — the consistency-vs-availability choice you make during the guaranteed cross-region partition.
- [Database replication: sync vs async, logical vs physical](/blog/software-development/database/database-replication-sync-async-logical-physical) — why async is the only sane cross-region default and what it costs.
- [Distributed replication: leader, multi-leader, and leaderless](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless) — single-leader-per-partition is what region partitioning actually is.
- [Database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding) — region is just a sharding key aligned to a geography boundary.
- [Consistency models from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) — the spectrum your read replicas and conflict policies sit on.
- Sam Newman, *Building Microservices* (2nd ed.) — the chapters on resilience and scaling cover the multi-region trade-offs from the practitioner's seat.
- Chris Richardson, *Microservices Patterns* — patterns for data management across services that underpin the locality discipline here.
