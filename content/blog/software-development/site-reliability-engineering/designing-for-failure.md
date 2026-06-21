---
title: "Designing for Failure: Failure Domains, Blast Radius, and the SPOF You Didn't See"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Stop trying to make systems that never fail. Learn to map your failure domains, hunt the hidden single points of failure, shrink your blast radius with cells, and design your degradation modes on purpose."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "failure-domains",
    "blast-radius",
    "single-point-of-failure",
    "graceful-degradation",
    "cellular-architecture",
    "resilience",
    "fault-tolerance",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/designing-for-failure-1.png"
---

At 14:05 on a Tuesday, a service that the architecture diagram swore was "fully redundant" went completely dark. Two load balancers, two app tiers, a primary and a replica database, all in different racks. On paper there was no single point of failure anywhere in the request path. And yet every user got an error, for forty-one straight minutes, while six engineers stared at a topology that had no business being down.

The cause turned out to be embarrassingly small. Both "independent" load balancers resolved the same internal hostname to reach the app tier. That hostname was served by one DNS resolver box. The resolver box's disk filled, it stopped answering, and both redundant front doors lost their ability to find the back end at the exact same instant. The redundancy was real everywhere except the one place that mattered. The two paths looked independent and were not, because they shared a dependency nobody had drawn on the diagram.

That outage is the entire subject of this post in miniature. At scale, everything fails all the time: disks die, kernels panic, NICs flap, a zone loses power, a network partitions, a dependency times out, a config push poisons every replica at once. You do not get to live in a world where components do not fail. So the senior move is not to chase a system that never breaks. It is to *assume every component will break*, and then engineer so that when it does, the damage is contained to a small region and the system degrades instead of collapsing. That is what "designing for failure" means, and the picture below is the spine of it: the nested boundaries inside which a single failure is supposed to stay.

![A vertical stack of failure-domain layers from a single process up through host, rack, availability zone, and region, with a note that spreading replicas across domains keeps a failure contained](/imgs/blogs/designing-for-failure-1.png)

This is the *operate-it* layer of reliability, not the architecture-it layer. The deep architectural treatment of redundancy, partitioning, and cascading failure lives in our system-design series, and I will cross-link it where it earns the space rather than re-deriving it. What this post gives you is the operator's working set: how to enumerate your failure domains, how to hunt the single point of failure hiding behind your "redundant" components, how to shrink your blast radius with cells, and how to design your degradation modes on purpose instead of inheriting whatever the exception handler happens to throw. It plugs directly into the series' loop — [define reliability, measure it, spend the error budget, respond, learn, engineer the fix](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — at the "engineer the fix" end. By the time you finish, you should be able to draw your own failure-domain map, run a SPOF audit on a service you own, fill in a blast-radius table, and write down a fail-open-versus-fail-closed decision for every dependency you call.

## 1. The mindset shift: from "make it not fail" to "assume it fails, contain the damage"

There is a sentence that quietly runs through the culture of every team that has not yet internalized SRE: *"we need to make sure this never happens again."* It is well-intentioned and it is the wrong goal. Stated literally, "never fails" is unachievable, and chasing it produces brittle systems — because every dollar and every hour you spend trying to drive the probability of a single failure from 0.1% to 0.01% is a dollar and an hour you did not spend on making the failure survivable when it inevitably arrives anyway.

The mindset that actually scales is the one Netflix popularized with Chaos Monkey and that Amazon bakes into how it talks about its own infrastructure: *everything fails, all the time*. Werner Vogels has said it as a design principle — you build assuming components will fail, because at fleet scale they provably do. If you run ten thousand disks with a 2% annual failure rate, you are replacing roughly four disks a week, every week, forever. A disk failing is not an incident; it is Tuesday. The incident is when a disk failing takes down something it should not have.

So the reframing is this: stop optimizing the probability that any one thing breaks, and start optimizing what happens *given* that it breaks. You move your engineering energy from prevention to *containment* and *graceful degradation*. The questions change. Instead of "how do we keep this node from crashing?" you ask "when this node crashes — and it will — how many users notice, for how long, and what do they see?" Instead of "how do we make the dependency never go down?" you ask "when the dependency is gone for two hours, what does my service do, and did I choose that behavior or did the timeout choose it for me?"

This is not fatalism. It is the opposite of fatalism, because it is the only mindset that gives you control. A failure you have designed for is a failure you can survive. A failure you have refused to anticipate is a failure that surprises you at 3am with a customer-facing outage and a postmortem that starts with "we always assumed that couldn't happen." The whole practice rests on one idea: treat the system as a set of nested boundaries, accept that something inside will break, and make sure the break stays inside its boundary. Everything else in this post is mechanics for that one idea.

### Why this pays: the arithmetic of containment

Reliability is a number, not a vibe — that is the founding claim of this whole series. If your service has a 99.9% availability objective, you are allowed 43.2 minutes of failure per 30-day month. (We derive that nines-to-minutes table in the [SLI, SLO, SLA post](/blog/software-development/site-reliability-engineering/sli-slo-sla-the-three-numbers-that-matter); the short version is that 0.1% of a 30-day month is $0.001 \times 43{,}200 = 43.2$ minutes.) Now compare two ways to spend an outage. A failure that takes down 100% of users for 30 minutes burns the entire month's budget in one event. The *same underlying failure*, contained so it only hits 10% of users, burns one-tenth of the budget. The bug was identical. The blast radius was the variable you controlled. Designing for failure is, mechanically, the act of turning failures from full-budget events into fractional-budget events — and the error budget is exactly the currency that lets you measure whether you succeeded.

There is a deeper point hiding in that arithmetic, and it is worth dwelling on because it reshapes how you prioritize work. Suppose you have two engineering tasks competing for a sprint. Task A reduces the probability of a particular database failover bug from 1% per quarter to 0.5% per quarter — a genuine 2x improvement in failure *probability*. Task B leaves the probability untouched but cuts the failover's user impact from 100% of users for 8 minutes down to 20% of users for 8 minutes — a 5x improvement in *blast radius*. Which buys you more reliability? Run it through the budget. Task A's expected monthly budget burn from that bug is roughly (failure rate) times (impact); halving the rate halves the burn. Task B leaves the rate alone but cuts the impact by 5x, cutting the burn by 5x. Task B wins, and it wins by more than Task A, and it wins for a reason that generalizes: **probability is bounded below by zero and you fight diminishing returns to get there, but blast radius can often be cut by large integer factors with a single architectural change.** Going from one fleet to eight cells is a one-time 8x improvement. Going from a 1% failure rate to a 0.125% failure rate is a research project that may never land. This is why senior SREs spend so much energy on containment and comparatively little on the last marginal nine of prevention — the math says containment is where the leverage lives.

The mindset shift also changes what you celebrate. On a prevention-first team, the win is a long streak with no incidents, and the streak breeds complacency — the longer it runs, the more the team forgets how to handle failure, so the eventual failure is handled worse. On a containment-first team, the win is *boring failures*: a node died, traffic shifted, nobody noticed, the graph has a tiny dip nobody paged on. You *want* small failures to happen and be absorbed, because each one is evidence your containment works and practice keeping it working. Chaos engineering, which we get to in section ten, is the formalization of this instinct — you deliberately cause small failures so that the large ones never get a chance to surprise you. The cultural tell of a mature team is that they are slightly suspicious of a system that has had zero failures for a long time, because zero observed failures usually means either you are not measuring or you are accumulating risk you will pay for all at once.

## 2. Failure domains: the boundary a failure is supposed to stay inside

A **failure domain** is the boundary within which a single failure is contained. It is the answer to "if *this* breaks, what is the largest set of things that go down with it?" Failure domains nest, from small and frequent at the bottom to large and rare at the top, and the picture at the top of the post lays them out:

- A **process**: a single instance of your service. It can crash, OOM, deadlock, or wedge. Frequency: high. Blast radius if contained: one instance's share of traffic.
- A **host / node**: the machine the process runs on. Its disk dies, its kernel panics, its NIC starts dropping packets. Everything on that host goes with it.
- A **rack / cell**: a set of hosts sharing a top-of-rack switch and a power distribution unit. The switch fails, the PDU trips, and the whole rack disappears at once.
- An **availability zone**: a data-center-scale failure domain that the cloud providers expose explicitly. It has its own power, cooling, and network. A zone can lose power or get cut off by a network event, taking everything in it down together.
- A **region**: a geographic collection of zones. Region-wide failures are rare but spectacular — a control-plane bug, a fiber cut, a botched deploy to a regional service. If you run in one region only, a region failure is a 100% outage.
- A **cell**: a deliberately-constructed failure domain you build *inside* a region — a self-contained, full-stack slice of your service that serves a partition of your users. We will spend a whole section on cells because they are the most powerful operator-controllable failure domain there is.

The single most important rule about failure domains is the one most teams get subtly wrong: **spread your replicas across domains, and verify that you actually did.** Three replicas of a database give you nothing if all three landed in the same availability zone, because one zone failure takes all three. This happens constantly — a scheduler packs pods onto the cheapest available nodes, an autoscaler refills capacity into whichever zone has room, a Terraform module defaults to a single subnet — and your "3x replicated" service is one zone outage away from total loss. The redundancy is notional. It exists in the replica count and not in the physical reality.

So the operator's discipline is: for every redundant tier, name the domain you are redundant *across*, and then check that the placement enforces it. On Kubernetes, that means topology spread constraints and anti-affinity, and then actually confirming the pods landed where you think:

```yaml
# Force replicas to spread across availability zones, not pile into one.
apiVersion: apps/v1
kind: Deployment
metadata:
  name: checkout
spec:
  replicas: 6
  template:
    spec:
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule  # refuse to over-pack a zone
          labelSelector:
            matchLabels:
              app: checkout
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchLabels:
                  app: checkout
              topologyKey: kubernetes.io/hostname  # no two replicas on one host
```

And then the part everyone skips — verify it landed right, because a constraint that silently falls back to "best effort" is a constraint that lied to you:

```bash
# Are the replicas actually spread across zones, or did they pile up?
kubectl get pods -l app=checkout \
  -o custom-columns='POD:.metadata.name,NODE:.spec.nodeName,ZONE:.metadata.labels.topology\.kubernetes\.io/zone' \
  --sort-by='.metadata.labels.topology\.kubernetes\.io/zone'

# Expect ~2 pods in each of 3 zones. If you see 5 in zone-a and 1 in zone-b,
# your "6x replicated" service is one zone outage from losing 5/6 of capacity.
```

Note `whenUnsatisfiable: DoNotSchedule` rather than `ScheduleAnyway`. The latter is the trap: under capacity pressure the scheduler quietly packs everything into one zone "to be helpful," and you discover the violation during the zone outage. If you would rather run at reduced replica count than silently lose your spread, make the constraint hard.

Two failure-domain rules are worth stating explicitly because teams violate them constantly. First, **a failure domain is only as small as its largest shared dependency.** You can run pods on three different hosts, but if all three hosts mount the same network-attached volume, the real failure domain is that volume, not the three hosts — losing the volume loses all three. The host-level redundancy is cosmetic. To find your *true* failure domains, you trace each component down to the lowest shared resource it cannot live without, and that resource defines the boundary. Second, **failure domains must align with your capacity planning.** If you spread across three zones and design to survive losing one, then each zone can carry at most one-third of total traffic at steady state — which means you must run at 50% utilization or below, because when one zone dies its third of the traffic redistributes onto the surviving two-thirds of capacity, pushing them to $\frac{1}{2/3} = 1.5\times$ their previous load. If those zones were already running at 70%, the failover pushes them to 105% and you cascade — the zone failure you designed to survive instead triggers an overload that takes down the survivors. The number of failure domains you can lose is therefore directly coupled to how much headroom you carry; this is the link to [capacity planning and forecasting](/blog/software-development/site-reliability-engineering/capacity-planning-and-forecasting), and it is why "we're N+1 redundant" is meaningless unless you also state the utilization you run at.

You can even turn the placement assertion into a continuous SLI so that drift pages you before it bites. A Prometheus recording rule that counts replicas per zone, plus an alert that fires when any zone holds too large a share, makes "are we still spread?" a monitored property instead of an assumption:

```yaml
# Continuously assert the failure-domain spread instead of hoping it holds.
groups:
  - name: failure-domain-spread
    rules:
      - record: checkout:replicas_per_zone
        expr: count by (zone) (up{job="checkout"} == 1)
      - alert: ReplicaSpreadDrifted
        expr: |
          max(checkout:replicas_per_zone)
          /
          sum(checkout:replicas_per_zone) > 0.5
        for: 15m
        labels:
          severity: ticket
        annotations:
          summary: "More than half of checkout replicas are in one zone"
          description: "Failure-domain spread has drifted; a single zone outage would now exceed planned blast radius."
```

This is the operator's version of trust-but-verify: the topology constraint *requests* the spread, and the recording rule *confirms* it stays true as the autoscaler churns capacity over weeks. The first time this alert fires on a real fleet, it almost always catches a placement drift that nobody would have noticed until the zone outage that exposed it.

### Worked example: the replica placement that wasn't

A payments team I worked with ran a 3-node etcd-style coordination cluster, quorum of 2, billed internally as "tolerates one node failure." During a zone power event they lost the entire cluster. Why? Two of the three nodes had been scheduled into the same zone months earlier when that zone happened to have spare capacity, and nobody re-checked. Losing that zone took 2 of 3 nodes, which broke quorum, which froze every service that depended on coordination.

The math is unforgiving. With a quorum-of-2 cluster you can lose exactly one node. If your three nodes are distributed across three zones, a single zone outage costs you one node and quorum holds. If two of three nodes share a zone, a single zone outage costs you two nodes and quorum is gone. Same replica count, same quorum rule — the only variable was the failure-domain spread, and that variable was set by an autoscaler's convenience six months earlier. The fix was not "add more nodes." It was a hard topology constraint plus a recurring audit that *asserts* the spread and pages if it drifts. Replica count is a promise; placement is whether the promise is kept.

## 3. Blast radius: how much breaks when one thing breaks

**Blast radius** is the share of users (or requests, or data, or revenue) affected when a given thing fails. It is the single most useful number in failure design because it converts "we had an outage" into "we had an outage that affected X% of users for Y minutes" — which is exactly the form the error budget consumes. A failure with a small blast radius is, almost by definition, a contained failure.

The goal you are reaching for is a stated invariant: *no single failure takes down more than X% of users.* Pick X deliberately. For a consumer service, "no single failure affects more than 5% of users" is an ambitious, achievable target with cellular design. For a critical control plane, you might want "no single failure affects more than one region's worth of users." The number is a design constraint that you then engineer toward, the same way you engineer toward an SLO.

To make blast radius concrete and auditable, build a **blast-radius table**: a row per failure mode, and columns for the failure domain it hits, the share of users affected, and whether it is contained. The table below is the operator's version of the diagram, and it is the single most valuable artifact in this post because filling it in *is* the work — every "No" in the contained column is a finding.

![A blast-radius table listing failure modes against the domain they hit, the share of users affected, and whether the failure is contained or an uncontained single point of failure](/imgs/blogs/designing-for-failure-4.png)

Here is the same table in markdown, which you can paste into a doc and fill in for your own service:

| Failure | Failure domain it hits | % users affected | Contained? | Action |
| --- | --- | --- | --- | --- |
| Disk failure on one host | 1 host | < 0.5% | Yes — pod reschedules | None; this is Tuesday |
| Process OOM | 1 instance | < 0.5% | Yes — replica picks up | Tune limits, alert if frequent |
| Single replica wedges | 1 instance | < 0.5% | Yes — health check ejects | Verify the health check is real |
| Availability zone outage | 1 of 3 zones | ~33% briefly, then 0% | Yes — fails over to 2 zones | Confirm capacity headroom |
| Bad config pushed everywhere | All cells at once | 100% | **No — correlated** | Stage the rollout (§7) |
| Shared DNS resolver dies | All request paths | 100% | **No — hidden SPOF** | Independent resolution (§5) |
| Region outage (single-region deploy) | 1 region = everything | 100% | **No** | Multi-region (cross-link) |
| Poison input crashes all workers | All workers | 100% | **No — correlated** | Input validation, isolate (§8) |

The rows that say "Yes" are the wins — they are failures you have already designed to contain. The rows that say "No" are your work queue, ranked by user impact. This is how you turn a vague sense of "I think we're pretty resilient" into a prioritized backlog. The first time a team fills this in honestly, they almost always find two or three rows they were sure were "Yes" and are actually "No." Those are the outages waiting to happen.

### Why blast radius beats raw availability as a design target

Here is the subtle principle worth internalizing: two systems can have identical *average* availability and wildly different *risk profiles*, and blast radius is what distinguishes them. Imagine a service that is up 99.9% of the time because it has one big outage a year that takes down 100% of users for about nine hours. Now imagine a second service that is also up 99.9% but achieves it through many small, contained failures — a cell down here, a replica ejected there — each affecting a few percent of users for a few minutes. The arithmetic average is the same. The lived experience is not remotely the same. The first service makes the news and triggers a war room; the second is invisible to almost everyone. Designing for failure is largely the project of trading rare-catastrophic for frequent-tiny while holding total budget spend flat or lower. Blast radius is the dial you are turning.

## 4. Cellisation: shrinking the blast radius by construction

The most powerful operator-controllable lever for blast radius is **cellular architecture** — also called cellisation, and the close cousin of **shuffle sharding**. The idea is simple and the payoff is enormous: instead of running one big shared fleet that serves all of your users, you run N independent, full-stack copies (cells), and you partition users across them. Each cell has its own instances, its own database shard or replica set, its own cache — its own everything, end to end. A failure inside a cell stays inside that cell. The before-and-after is stark:

![A two-column before and after diagram contrasting one shared fleet where a bad deploy takes everyone down with eight cells where the same deploy takes only one cell down](/imgs/blogs/designing-for-failure-2.png)

The transformation is from "one bad deploy equals 100% down" to "one bad deploy equals 1/N down." The math is the whole point.

#### Worked example: the blast-radius math of cellisation

Start with one shared fleet. A bad deploy — say, a binary that crashes on a code path 30% of requests hit — goes everywhere at once. Blast radius: **100% of users**, for as long as it takes to detect and roll back. If your detection-plus-rollback time is 20 minutes and you serve 1,000 requests per second, that is $1{,}000 \times 0.30 \times 1{,}200\text{ s} = 360{,}000$ failed requests, and it burns a serious chunk of a monthly budget in one event.

Now split into 8 cells, each serving 12.5% of users, with deploys rolled cell-by-cell. The same bad binary deploys to cell 1 first. Within minutes, cell 1's health metrics tank, the rollout halts automatically, and the deploy never reaches cells 2 through 8. Blast radius: **12.5% of users** (one cell), for a shorter window because you caught it on the first cell. Failed requests: $125 \times 0.30 \times t$ for some smaller $t$, because automated canary analysis on a single cell trips faster than a human watching a global dashboard. You have cut the blast radius by 8x and the duration by some additional factor, for a combined improvement of an order of magnitude or more, on the *exact same bug*.

The general rule is clean: with N cells and per-cell deploys, the worst-case single-deploy blast radius drops from 100% to roughly $1/N$ of users. Eight cells gets you to ~12.5%; sixteen cells to ~6.25%; thirty-two to ~3%. There is a real cost (we will get to it), but the blast-radius reduction is not marketing — it is arithmetic, and it compounds with staged rollouts.

But notice the shape of the curve, because it tells you where to stop. Going from 1 cell to 2 cells cuts worst-case blast radius from 100% to 50% — a 50-percentage-point improvement. Going from 2 to 4 cuts it from 50% to 25% — a 25-point improvement. From 4 to 8: a 12.5-point improvement. From 16 to 32: a 3-point improvement. Each doubling of cell count halves the marginal benefit, while the operational cost (more deploy targets, more monitoring surface, more capacity floors) grows roughly *linearly* with N. So there is a sweet spot, and it is usually smaller than enthusiasts want. For most services, 4 to 8 cells captures the overwhelming majority of the blast-radius benefit; going to 32 or 64 cells is rarely justified unless you operate at a scale where even 3% of users is millions of people and the per-cell overhead is amortized across enormous traffic. The rule of thumb: pick the smallest N where $1/N$ of your users is a blast radius you can live with, and stop there. If you can tolerate 10% of users affected by a worst-case event, 10 cells is enough; do not build 50.

#### Worked example: sizing cells against the cost of an outage

Put numbers on the trade-off. Suppose a 100% outage of your service costs the business roughly \$50,000 in lost revenue and goodwill per hour, you have about one "would-have-been-fleet-wide" incident per quarter that cells would contain, and each such incident lasts about 30 minutes. With one fleet, the expected quarterly cost of these events is one incident times 100% impact times \$50,000/hr times 0.5 hr, or about \$25,000 per quarter. Move to 8 cells: the same incident now hits 12.5% of users, so the expected cost drops to about \$3,125 per quarter — a saving of roughly \$22,000 per quarter, call it \$88,000 per year. Now weigh that against the operational cost of 8 cells: extra capacity floors (8 cells each need a minimum viable size even when small, so you pay some baseline tax), extra monitoring, and the engineering time to build cell routing and per-cell deploys. If that operational cost is well under \$88,000/year, cells pay for themselves on this one risk alone — and they also reduce *every other* correlated-failure risk, so the real return is larger. If your outage cost were instead \$50/hour because it is an internal tool, the same calculation says the \$88,000/year of operational cost is wildly disproportionate, and you should not cellise. The decision is not aesthetic. It is the cost of the wide blast radius versus the cost of the cells, and you can actually compute both.

### Shuffle sharding: the same idea for shared resources

Pure cells assign each user to exactly one cell. **Shuffle sharding**, the AWS technique, is a refinement for when you have many tenants and a pool of workers and want to limit how many tenants any single failure or single bad actor can affect. Instead of assigning each tenant to one shard, you assign each tenant to a small random *subset* of workers (say, 2 of 8). The clever part is the combinatorics: with 8 workers and a shard size of 2, there are $\binom{8}{2} = 28$ distinct shard combinations. The probability that two specific tenants land on the *exact same pair* of workers is small, so a single tenant melting down its two workers — or a poison request killing two workers — overlaps with only a fraction of other tenants. One bad neighbor degrades a sliver of the population instead of everyone sharing the same few workers. AWS uses this in Route 53 and other multi-tenant services precisely to make "one tenant's bad day" not become "everyone's bad day."

### When cells are worth it, and when they are not

Cells are not free. You pay in operational surface (N copies to deploy, monitor, and patch), in cross-cell coordination complexity (how do you handle a user whose data must move between cells?), and in baseline cost (per-cell minimum capacity adds up). The decision is the standard reliability trade-off: the blast-radius reduction is worth the operational cost when an outage is genuinely expensive — a payments system, a primary user-facing surface, anything where a 100% outage is an existential event. It is *not* worth it for an internal batch job that ten people use, where a daily failure costs a retry and a shrug. Reach for cells when the cost of a wide blast radius dwarfs the cost of running N copies. We will return to this calculus in the "how to reach for this" section. The architectural mechanics of partitioning and routing across cells are covered in depth in [the system-design treatment of cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads); here we care about the operate-it consequence, which is that cells make your blast-radius table mostly say "Yes."

## 5. The single point of failure you didn't see

Now we get to the most dangerous failure of all, because it is the one that hides behind the appearance of safety. A **single point of failure (SPOF)** is a component whose failure takes down the whole system. Everyone knows to eliminate the *obvious* SPOFs — you add a second load balancer, a replica database, a redundant network path. The dangerous SPOF is the one that survives all that redundancy because it is a **shared dependency hiding behind the redundant components**. Two "independent" things that secretly lean on one thing are not independent, and the diagram below is the canonical shape of the trap from our opening story:

![A topology graph showing two redundant load balancers both resolving the same DNS name served by one resolver box, which then becomes a single point of failure causing a full outage](/imgs/blogs/designing-for-failure-3.png)

LB-A and LB-B are genuinely redundant load balancers. Either can serve traffic if the other dies. But both resolve the same internal hostname through the same DNS resolver box. That resolver is the SPOF. It is invisible on the load-balancer-and-database topology diagram because DNS is "just infrastructure," the plumbing nobody draws. When it failed, both redundant paths lost their ability to find the backend at the same instant, and the redundancy bought exactly nothing.

The killer question — the one that finds these — is: **"What do ALL of my redundant components have in common?"** Anything they share is a candidate SPOF. Walk the list ruthlessly. The audit decision tree below organizes the hunt:

![A tree organizing the single point of failure audit by asking what redundant copies share, branching into shared infrastructure like DNS and power and shared control like config and deploy pipelines](/imgs/blogs/designing-for-failure-7.png)

Here is the SPOF-audit checklist I run on any service claiming redundancy. For each item, ask: do my redundant components all share this? If yes, it is a SPOF until proven otherwise.

- **DNS resolution.** Do all your replicas resolve the same name through the same resolver, or the same upstream DNS provider? (Our opening outage. Also the root of more than one famous internet-wide outage.)
- **Configuration service.** Does every instance fetch config from one config service *at startup*? Then that config service is a SPOF for cold starts — fine while everything is warm, catastrophic during a fleet-wide restart when everything tries to read it at once.
- **The certificate authority.** Do all your services trust certs from one internal CA? If the CA's signing infrastructure fails or a cert expires un-renewed, every mutual-TLS connection in your fleet breaks simultaneously. Expired certificates are a leading cause of correlated, fleet-wide outages.
- **The deploy pipeline.** A single deploy pipeline that pushes to all replicas is a SPOF *for change* — it can take everything down at once with one bad artifact (this is the correlated-failure problem in §7).
- **The shared database.** Two app tiers that are redundant against each other but both write to one primary database share that primary as a SPOF. The replica only helps if you have actually wired and tested failover.
- **The shared cache / session store.** If both app tiers depend on one Redis for sessions, that Redis is a SPOF and its failure logs everyone out at once.
- **The observability stack.** A subtle one: if your alerting depends on the same network or auth or DNS that just failed, you go down *and* go blind. Your monitoring should fail independently of the thing it monitors.
- **The control plane.** Kubernetes API server, service mesh control plane, scheduler — if the data plane keeps serving when the control plane is down, good; if a control-plane outage stops you from serving traffic, it is a SPOF.

The technique that surfaces these is **dependency mapping**: enumerate, for each component, every other thing it must talk to in order to function — including at startup, not just steady state — and look for the names that show up under *every* replica. Those shared names are your hidden SPOFs. A lightweight way to make this concrete is to write down, per service, a small dependency manifest that tags each dependency as *hard* (the service cannot serve without it) or *soft* (the service degrades but survives without it), and whether it is *shared* across your redundant copies:

```yaml
# checkout service — dependency manifest (the input to a SPOF audit)
service: checkout
dependencies:
  - name: dns-resolver
    type: hard
    shared_across_replicas: true     # <-- RED FLAG: hard + shared = SPOF
    failure_mode: fail-to-cache      # mitigated: stale-while-revalidate
  - name: config-service
    type: hard
    needed_at: startup               # <-- only bites on cold start / fleet restart
    shared_across_replicas: true     # <-- RED FLAG
    failure_mode: fail-to-last-known-good
  - name: internal-ca
    type: hard
    shared_across_replicas: true     # <-- RED FLAG: cert expiry = fleet-wide
    failure_mode: fail-closed
  - name: payments-db
    type: hard
    shared_across_replicas: true     # primary is shared; replica failover tested?
    failure_mode: fail-closed
  - name: recommendations
    type: soft                       # safe: degrades, not shared-critical
    shared_across_replicas: false
    failure_mode: fail-open
```

The audit is then mechanical: every dependency that is both `type: hard` and `shared_across_replicas: true` is a SPOF candidate, and for each one you must either make it not-shared (independent resolution paths, per-cell config), make it not-hard (a fallback that lets you survive without it), or explicitly accept the risk and document it. The manifest turns a fuzzy "I think we're redundant" into a checklist with red flags you cannot un-see. Keeping it in version control next to the service also means a code review can catch a *new* hard-and-shared dependency being introduced — the moment someone adds a synchronous call to a new shared service, the reviewer sees the SPOF being created rather than discovering it in the postmortem. Distributed tracing helps here because a trace shows you the real dependency graph of a request, including the surprise hops; we cover that in [the distributed-tracing post](/blog/software-development/site-reliability-engineering/distributed-tracing-in-practice). But traces only show steady-state request paths. The startup-time dependencies — the config fetch, the CA handshake, the service-discovery lookup — are the ones that bite during a cold start and that tracing of warm traffic will never reveal. You have to enumerate those by reading the code and the init path.

### Worked example: the hidden-SPOF hunt and the redesign

Back to the opening outage. Here is the full hunt and the fix, because the redesign is as instructive as the diagnosis.

The detection was fast and the diagnosis was slow: we knew within two minutes that 100% of requests were failing, but it took twenty-three minutes to find *why*, because the dashboards all showed green — the load balancers were up, the app instances were up, the databases were up. Every component the diagram knew about was healthy. The breakthrough came when an engineer ran a manual DNS lookup from a load balancer host and it hung. The resolver was the answer, and it was not on anyone's reliability radar because "DNS is just there."

The redesign had three parts. First, **independent resolution paths**: each load balancer was reconfigured to use a *different* resolver, and the resolvers themselves were spread across zones, so no single resolver failure can take both LBs offline. Second, **resolver caching with a generous TTL and stale-while-revalidate**, so that even if a resolver is briefly unreachable, the load balancers keep serving from cached answers rather than failing the lookup outright — a degradation mode (see §6) instead of a hard failure. Third, **a synthetic check that exercises the full resolution path** and alerts on resolver health directly, so the next resolver problem pages us on the *cause* before it becomes a customer-facing symptom.

The measured result: in the following nine months, we had two resolver incidents (resolvers do fail), and both were absorbed with zero user-facing errors, because either the other resolver answered or the cache served stale. The MTTR for "resolver down" went from 41 minutes of total outage to 0 minutes of user impact. We did not make the resolver more reliable. We made the system survive the resolver failing — which is the whole thesis.

## 6. Degradation modes: designing what happens when a dependency is gone

When a dependency you call is unavailable, your service does *something*. The only question is whether you chose that something or whether you inherited it from whatever your HTTP client's default timeout-and-throw behavior happens to be. **Designing your degradation modes** means deciding, deliberately and per-dependency, what your service does when each thing it depends on is gone — and writing that behavior down as a requirement, not leaving it to the accident of an unhandled exception.

The central decision is **fail open versus fail closed**:

- **Fail open** (also "fail safe" in the availability sense): when the dependency is unavailable, allow the operation to proceed, possibly degraded. The system stays *available* at the cost of correctness or completeness. Example: if the recommendations service is down, serve a generic "popular items" list instead of personalized ones. The user gets a worse experience but they get an experience.
- **Fail closed** (also "fail secure"): when the dependency is unavailable, deny the operation. The system stays *correct/safe* at the cost of availability. Example: if the authorization service is down, deny access. Better to lock everyone out for ten minutes than to let anyone in without checking permissions.

There is no universal right answer — the correct choice is per-case and depends entirely on what is worse: serving a wrong answer, or serving no answer. The decision table below is the artifact you should produce for every dependency your service calls:

![A decision table mapping dependencies like auth, recommendations, payments, and rate limiting to whether they should fail open or fail closed and the risk if the wrong choice is made](/imgs/blogs/designing-for-failure-5.png)

In markdown, with the reasoning spelled out:

| Dependency down | Default | Why | Risk if wrong |
| --- | --- | --- | --- |
| Authorization / authentication | **Fail closed** | No verified identity means no trust — never grant access you can't check | Security breach, data exposure |
| Recommendations / personalization | **Fail open** | A generic list is fine; an empty page is not | Slightly worse UX, no real harm |
| Payment authorization | **Fail closed** | Never ship goods you haven't confirmed payment for | Fraud, lost revenue |
| Rate limiter | **Fail open** | Availability usually beats perfect limiting; let traffic through | Brief overload if abused |
| Feature-flag service | **Use last-known-good (cached)** | Neither hard-open nor hard-closed; serve the last good config | Stale rollout state |
| Fraud scoring | **Degrade to rules** | Fall back to a cheap rules engine when the ML scorer is down | Coarser scoring, acceptable |

Notice the third option that shows up in real systems: **fail to last-known-good**. For things like feature flags or routing config, neither hard-open nor hard-closed is right — you want to keep using the last value you successfully fetched, cached locally, so the dependency being down freezes your config rather than breaking it. This is itself a deliberate degradation mode, and it is often the best one for control data.

The practice is to make degradation a designed property of the call site. A concrete example of a fail-open recommendations call with an explicit fallback:

```python
import logging

log = logging.getLogger("checkout")

def get_recommendations(user_id, timeout_s=0.15):
    """Fail OPEN: if the recs service is slow or down, serve popular items.
    The user always gets *something*; we never block the page on recs."""
    try:
        resp = recs_client.fetch(user_id, timeout=timeout_s)
        return resp.items
    except (TimeoutError, ServiceUnavailable) as e:
        # Deliberate degradation: log it, count it, and fall back.
        log.warning("recs degraded, serving popular fallback: %s", e)
        RECS_FALLBACK_COUNTER.inc()        # so we can SEE the degradation in metrics
        return popular_items_cache.get()   # last-known-good popular list
```

Two details make this a *designed* degradation rather than an accidental one. First, the timeout is tight (150ms) and explicit — recommendations are nice-to-have, so we refuse to let them slow the page; an unbounded default timeout would turn "recs are slow" into "the whole page hangs," which is the opposite of failing open. Second, the fallback path increments a counter. **A silent degradation is a bug in disguise**: if you fall back to popular items and nobody can see it in the metrics, you will run degraded for weeks without knowing your recs service has been down the whole time. Every degradation mode needs an SLI so you can tell the difference between "healthy" and "limping along."

Contrast with a fail-closed authorization check, where the *safe* default is to deny:

```python
def is_authorized(user_id, resource, timeout_s=0.5):
    """Fail CLOSED: if the authz service is unreachable, DENY.
    We never grant access we cannot verify."""
    try:
        return authz_client.check(user_id, resource, timeout=timeout_s)
    except (TimeoutError, ServiceUnavailable):
        AUTHZ_FAILCLOSED_COUNTER.inc()
        return False   # deny on uncertainty — locking out beats leaking in
```

### Measuring the degraded state

Because a silent degradation is a bug in disguise, the degraded path needs to be a first-class thing your dashboards and SLOs can see. The discipline is to emit a metric every time a fallback fires, and then to alert not on the dependency being down (a cause) but on the *fallback rate climbing* (a symptom of how much your users are actually getting the degraded experience). A burn-rate-style alert on the degradation rate looks like this:

```yaml
groups:
  - name: degradation-visibility
    rules:
      # What fraction of recommendation requests are being served the fallback?
      - record: recs:fallback_ratio
        expr: |
          sum(rate(recs_fallback_total[5m]))
          /
          sum(rate(recs_requests_total[5m]))
      - alert: RecsDegradedHigh
        expr: recs:fallback_ratio > 0.10
        for: 10m
        labels:
          severity: ticket          # not a page: failing OPEN means users are still served
        annotations:
          summary: "Over 10% of recommendation requests are degraded"
          description: "Recs service is failing open to popular-items fallback; UX is degraded but available. Investigate during hours."
```

Two design choices here are deliberate and worth copying. The severity is `ticket`, not `page` — because the whole point of failing open is that users are *still being served*, so this does not warrant waking someone at 3am; it warrants a ticket and a daytime fix. (If your degradation strategy meant users were *not* served, that same rate would warrant a page — the severity follows the user impact, which is exactly the [symptom-based alerting philosophy](/blog/software-development/site-reliability-engineering/alerting-that-doesnt-cry-wolf).) And the metric is a *ratio*, a fraction of good-or-degraded events over total — the same SLI-as-a-ratio shape the whole series uses — so you can reason about it the same way you reason about availability. Without this metric, your recs service can be down for a week and your only signal is a vague trickle of "the homepage feels generic" tickets that nobody connects to a root cause. With it, the degraded state is as visible as the healthy one, and that visibility is what makes failing open *safe* rather than merely convenient.

### Partial failure: keep most functions when one is down

The broader goal that degradation modes serve is **partial failure**: when one feature's dependency is gone, the rest of the system keeps working. This is the difference between "the recommendations service is down so the entire homepage 500s" and "the recommendations service is down so the homepage renders everything except the recommendations strip." The first is a system with no partial-failure design — one feature's failure became the whole page's failure. The second is a system where features are isolated enough that they fail independently. Amazon's product pages are the canonical example: dozens of independent services contribute strips and modules, and any one of them being down just removes its module rather than breaking the page. The page degrades, function by function, instead of collapsing. That isolation between features is itself a failure-domain boundary — a **bulkhead** — and we develop it fully in [the sibling post on circuit breakers, bulkheads, and load shedding](/blog/software-development/site-reliability-engineering/circuit-breakers-bulkheads-and-load-shedding), and in [the dedicated graceful-degradation-and-fallbacks post](/blog/software-development/site-reliability-engineering/graceful-degradation-and-fallbacks). The principle to carry from here is: design the failure behavior of every dependency on purpose, give each degraded mode a metric, and isolate features so one's failure is partial, not total.

## 7. Correlated failures: the enemy of redundancy

Everything so far assumed that failures are *independent* — that when one replica dies, the others are unaffected, so redundancy works. The deadliest failures are the ones that violate that assumption. A **correlated failure** is one that breaks many failure domains at the same time, because they all share an exposure. Correlated failure is the enemy of redundancy, because redundancy's entire premise is that copies fail independently. If they fail together, having three of them is no better than having one.

The classic sources of correlation:

- **A bad config pushed everywhere.** You have 8 cells, beautifully isolated. Then you push one configuration change to all 8 at once, the change has a poison value, and all 8 break in the same second. Your cellular isolation just got bypassed by a global change. The config push correlated the failure across every domain you carefully separated.
- **A poison input.** A single malformed request — a record that triggers a parser bug, an oversized payload that exhausts memory — hits every worker that processes it. If the same poison message gets retried across the whole fleet (a "poison pill" in a queue), it can crash worker after worker, correlating the failure across all of them.
- **A thundering herd.** Ten thousand clients whose caches all expire at the same second, or all retry at the same second after a blip, slam the backend simultaneously. The load spike is correlated across all clients because they all reacted to the same trigger at the same time. (Retry storms are a specific, vicious case — covered in [the load-shedding sibling](/blog/software-development/site-reliability-engineering/circuit-breakers-bulkheads-and-load-shedding); the fix is backoff *with jitter* so the herd disperses in time.)
- **A shared bad binary or dependency version.** Every replica runs the same build. If the build has a latent bug that a particular date, input, or load level triggers, every replica triggers it at once. (The leap-second cascades of 2012 are the textbook case: kernels across the whole industry hit the same bug at the same instant because they all ran code with the same flaw, exposed simultaneously by a shared external trigger.)

The diagram below contrasts the correlated push with the staged one — the single most important operational defense against correlated change-induced failure:

![A before and after diagram contrasting a configuration pushed to all replicas at once that breaks everything together with a staged rollout that pushes to one cell first and auto-rolls-back](/imgs/blogs/designing-for-failure-6.png)

The defense against *change-induced* correlation is **staged rollout** — never push anything (binary, config, schema, feature flag) to everything at once. Push to one cell or one canary, bake, check health, then proceed. This converts a correlated 100% failure into a contained 1/N failure, and it is exactly why cells and staged deploys are two halves of one strategy. A canary rollout with automated analysis looks like this:

```yaml
# Argo Rollouts canary: push to a slice, analyze, then proceed or abort.
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: checkout
spec:
  strategy:
    canary:
      steps:
        - setWeight: 5          # 5% of traffic to the new version first
        - pause: { duration: 10m }
        - analysis:             # auto-abort if error rate climbs
            templates:
              - templateName: error-rate-check
        - setWeight: 25
        - pause: { duration: 10m }
        - setWeight: 50
        - pause: { duration: 10m }
        - setWeight: 100
---
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: error-rate-check
spec:
  metrics:
    - name: error-rate
      interval: 1m
      successCondition: result < 0.01     # < 1% errors or we roll back
      failureLimit: 2
      provider:
        prometheus:
          address: http://prometheus.monitoring:9090
          query: |
            sum(rate(http_requests_total{job="checkout",code=~"5.."}[2m]))
            /
            sum(rate(http_requests_total{job="checkout"}[2m]))
```

Config changes deserve the same treatment as code — they are often *more* dangerous than code because they bypass the testing rigor that code gets. Treat config as a deployable artifact: version it, stage it cell-by-cell, validate it before it loads, and make it auto-roll-back. The Argo `AnalysisTemplate` above is doing the work the diagram describes: it pushes to 5%, watches the real error-rate SLI for ten minutes, and aborts automatically if the rate exceeds 1%. The poison config never reaches the second step.

#### Worked example: a config push that should have been staged

A team pushed a config change that flipped a timeout from 30 seconds to 30 *milliseconds* — a typo, a missing zero. It went to every instance simultaneously through the config service. Within seconds, every request that took longer than 30ms (i.e. nearly all of them, since the backend p99 was 250ms) timed out. 100% error rate, fleet-wide, in under a minute. The redundancy was perfect and irrelevant, because the failure was perfectly correlated: every replica got the same poison config at the same time.

Run the counterfactual with staged rollout. The 30ms config goes to cell 1 (12.5% of traffic). Within one minute the canary analysis sees cell 1's error rate spike past 1% and halts the rollout; the config never reaches cells 2 through 8. Blast radius: 12.5% for about a minute, versus 100% for the ten-plus minutes it took to notice and manually revert the global push. The fix to the *incident* was reverting the config. The fix to the *class of incident* was: stage config rollouts, validate config in CI (a 30ms request timeout should fail a sanity check), and add a guardrail that rejects timeout values below a floor. You cannot prevent every typo. You can make sure a typo can only ever break one cell.

## 8. War story: when the redundant thing was secretly one thing

Three real-world shapes of this failure mode, because seeing the pattern repeat across very different systems is what makes it stick.

**The shared-DNS internet outages.** Several of the largest internet outages of the last decade trace to a shared dependency behind apparent redundancy in the DNS and routing layer. When a major managed DNS provider has a problem, every site that relies on it goes dark together — not because any individual site is poorly engineered, but because they all share the same external dependency, which makes their failures correlated. The 2016 Dyn DNS event took down a swath of major sites simultaneously for exactly this reason: independent companies, one shared DNS dependency, correlated failure. The operator lesson is brutal and clear: redundancy *within* your own infrastructure does nothing about a SPOF that sits *outside* it and is shared by all your paths. If every region of your service resolves through one external provider, that provider is your SPOF no matter how many regions you run.

**The expired-certificate cascade.** A recurring industry pattern: a TLS certificate at a shared, central point — an API gateway, an internal CA, a critical service's serving cert — expires un-renewed. Because everything downstream validates against it, *everything* fails mutual-TLS at the same instant. This has bitten very large, very sophisticated organizations, because the certificate is a quiet shared dependency that does not show up in the usual reliability reviews — until the day it expires and correlates a fleet-wide outage. The defenses are alerting on certificate expiry well ahead of time (30+ days), automating renewal, and — crucially — staging certificate rotations cell-by-cell so a botched rotation breaks one cell rather than the fleet. A cert is config, and config gets staged.

**The retry-storm self-amplification.** A backend has a brief 30-second blip. Every client retries. The retries, arriving all at once and stacking on top of the recovering backend's normal load, push it back over the edge. It blips again. Everyone retries again. The system has built a feedback loop that turns a 30-second blip into a 30-minute outage — entirely self-inflicted, entirely correlated, because every client reacted to the same trigger at the same time with the same un-jittered retry. The retry amplification factor is real arithmetic: if every client retries up to 3 times with no backoff, a backend recovering from a blip sees up to $3\times$ its normal load precisely when it is weakest. The fix is exponential backoff *with jitter* (randomize the retry delay so the herd disperses across time instead of synchronizing), retry budgets (cap retries as a fraction of total requests, e.g. "no more than 10% of traffic may be retries"), and circuit breakers that stop hammering a dependency that is already down. This is the canonical example of a correlated failure that no amount of replica-counting can fix — you have to break the synchronization, and the deep treatment lives in our [system-design post on cascading failures, circuit breakers, and bulkheads](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads).

The thread through all three: the failures were correlated, the redundancy assumed independence, and the gap between those two facts *was* the outage. When you design for failure, you are not just adding copies. You are constantly asking "are these copies *actually* independent, or do they share an exposure that will make them fail together?"

## 9. Putting it together: the hidden-SPOF outage, start to finish

Let me walk the full timeline of the opening outage as an integrated example, because it ties every section together — failure domains, blast radius, the hidden SPOF, the degradation mode that was missing, and the redesign. The timeline below is the incident as it actually unfolded:

![A left to right incident timeline showing a resolver crash at 14:02, both load balancers failing DNS lookups, a Sev1 page, finding the shared resolver, failover, and recovery at 41 minutes](/imgs/blogs/designing-for-failure-8.png)

- **14:02 — the resolver crashes.** A single DNS resolver box runs out of disk and stops answering. This is a *host-level* failure — normally the smallest, most boring failure domain there is. It should have been a non-event.
- **14:03 — both load balancers lose resolution.** Because both LBs resolve the backend hostname through this one resolver, both lose the ability to find the backend simultaneously. A host-level failure just escalated to a system-level outage, because the resolver was a hidden SPOF shared across both "redundant" paths. The failure crossed every domain boundary at once.
- **14:05 — 100% errors, Sev1 page.** The blast radius is everyone. The error budget is hemorrhaging. The [incident commander spins up the bridge](/blog/software-development/site-reliability-engineering/incident-command-staying-calm-under-fire).
- **14:05–14:28 — the slow diagnosis.** Every dashboard is green because every component the dashboards *know about* is healthy. The SPOF is invisible because it is the unmonitored plumbing. Twenty-three minutes lost to "but everything's up." This is the cost of not having mapped the dependency.
- **14:28 — the shared resolver is found.** A manual DNS lookup from an LB host hangs. The cause is identified.
- **14:43 — failover and recovery.** Resolution is repointed to a second resolver, the LBs find the backend again, errors clear. **MTTR: 41 minutes.** The entire month's 99.9% budget (43.2 minutes) was very nearly consumed by one host's disk filling up.

Then the redesign, which is where designing-for-failure thinking turns a 41-minute outage into a non-event next time:

1. **Independent resolution paths** (eliminate the SPOF): each LB uses a different resolver, resolvers spread across zones. Now no single resolver failure can take both LBs down — the failure stays in its host-level domain where it belongs.
2. **A degradation mode** (survive the dependency being gone): resolvers cache with a generous TTL and serve stale-while-revalidate, so a brief resolver outage is absorbed by the cache rather than turning into a hard lookup failure. This is fail-open for DNS — keep serving with the last-known-good answer.
3. **A synthetic check on the resolution path** (page on the cause, not the symptom): a probe exercises the full DNS path end-to-end and alerts on resolver health directly, so the *next* resolver problem pages us as "resolver degraded" before it ever becomes "100% of users down." This is the [symptom-versus-cause alerting principle](/blog/software-development/site-reliability-engineering/alerting-that-doesnt-cry-wolf) applied to the dependency we had been blind to.

The before/after: a host-level failure that used to mean **41 minutes of 100% outage** now means **zero user impact**, because either the other resolver answers or the cache serves stale. We did not make the resolver more reliable. We changed the system so the resolver is allowed to fail.

## 10. Stress-testing your failure design

A failure design you have not stress-tested is a hypothesis, not a design. Before you trust your blast-radius table, interrogate it with the hard questions:

- **What if the dependency is down for two hours, not two minutes?** A degradation mode that works for a 30-second blip might not survive two hours — your last-known-good cache may go stale enough to be dangerous, your fallback's own capacity may be exhausted. Design for the long outage, not just the blip, and decide at what point a degraded mode itself should escalate to a hard failure or a manual decision.
- **What if two failure domains fail at once?** You spread three replicas across three zones, so you survive one zone. What about two? Often the honest answer is "we don't, and that's an accepted risk" — but you should *know* that and have decided it, not discover it during the double failure. Write the assumption down: "tolerates 1 zone, not 2."
- **What if the failure is correlated, not independent?** Your three replicas survive one dying independently. Do they survive a bad config that hits all three at once? Re-examine every "Yes" in your blast-radius table and ask whether it assumed independence. The "Yes" rows that secretly assumed independence are your next correlated-failure incident.
- **What if the thing you depend on for recovery is also down?** You plan to fail over to a second resolver, a backup region, a replica database. Is the failover path *itself* tested and independent? A failover that depends on the same control plane that just failed is not a failover. The classic version: your runbook says "restore from backup," but the backup has never been restored and the restore path needs the very service that is down.
- **What if the region fails?** If you run in one region, a region failure is 100% and there is no clever degradation that saves you — the answer is multi-region, which is a real cost-versus-reliability decision covered in [the system-design post on multi-region and geo-distribution](/blog/software-development/system-design/multi-region-and-geo-distribution). Decide deliberately whether you are buying region-failure survival or accepting region-failure risk.
- **What if your monitoring fails with the thing it monitors?** If the outage takes out the network or auth path your alerting also uses, you go down *and* blind. Your observability must fail independently of the system it watches — a SPOF audit applies to your monitoring stack too.

The discipline that *proves* your answers instead of guessing them is **chaos engineering** — deliberately injecting the failures you claim to survive and watching whether you actually do. A minimal chaos experiment that validates a zone-failure claim:

```yaml
# Chaos Mesh: kill all pods in one zone, verify the service stays up on the other two.
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: zone-a-blackout
  namespace: checkout
spec:
  action: pod-kill
  mode: all
  selector:
    labelSelectors:
      "app": "checkout"
    expressionSelectors:
      - key: topology.kubernetes.io/zone
        operator: In
        values: ["zone-a"]      # take out exactly one failure domain
  duration: "10m"
  # Hypothesis to validate during this window:
  #   - error rate stays under the SLO
  #   - the other two zones absorb the traffic without saturating
  #   - no hidden SPOF in zone-a takes the whole service down
```

You run this as a **game day** — announced, in a controlled window, with the on-call watching — and you learn the truth: either the other two zones absorb the load and your "tolerates one zone" claim is real, or something in zone-a was a hidden SPOF and you just found it in a controlled test instead of at 3am. Netflix institutionalized this with Chaos Monkey precisely because *a failure mode you have not exercised is a failure mode you do not actually survive*. The chaos experiment is how a row in your blast-radius table earns its "Yes."

## 11. How to reach for this (and when not to)

Designing for failure is a discipline with real cost, and applying it uniformly to everything is its own kind of mistake. Here is how to calibrate.

**Always do the cheap, high-leverage parts**, regardless of service criticality:

- **Run the SPOF audit.** It is a checklist and an afternoon. Every service should be able to answer "what do all my redundant components share?" Finding a hidden SPOF on a whiteboard is orders of magnitude cheaper than finding it during an outage.
- **Spread replicas across failure domains and verify it.** Topology constraints plus an audit query cost almost nothing and prevent the most common "redundant but not really" outage.
- **Write down your degradation modes.** Even just filling in the fail-open-vs-fail-closed table for your dependencies forces the decisions that an unhandled exception would otherwise make for you, badly.
- **Stage your rollouts.** Never push code or config to 100% at once. This single practice prevents most correlated change-induced outages and costs you only some deploy latency.

**Invest in the expensive parts proportionally to the cost of a wide outage:**

- **Cellisation** is worth it when a 100% outage is genuinely expensive — payments, primary user surfaces, anything where being fully down is an existential or reputational event. It is overkill for an internal tool ten people use. The operational cost of N copies must be dwarfed by the blast-radius reduction's value.
- **Multi-region** buys you region-failure survival at real cost (data replication, cross-region consistency, doubled-or-more infrastructure). Buy it when a region outage would be catastrophic; skip it when a region outage means an hour of degraded service that customers will forgive.
- **Chaos engineering as a program** (not just one experiment) is worth it for mature teams running critical systems who need to *continuously* verify their failure assumptions as the system changes. For a small team on a non-critical service, a periodic manual game day is plenty.

**And the explicit do-nots:**

- **Don't engineer for failure modes that cost less than the engineering.** If a failure affects ten internal users for five minutes once a quarter, do not build cellular isolation to contain it. The whole point of blast-radius thinking is to spend your reliability budget where the impact is, not everywhere.
- **Don't add redundancy without checking independence.** A second copy that shares a SPOF with the first is cost without benefit — worse, it is *false confidence*, which is more dangerous than known risk because it stops you looking.
- **Don't choose a degradation mode by default.** Fail-open for an authorization check is a security incident; fail-closed for a recommendations widget is a self-inflicted outage. The default behavior of your HTTP client is almost never the behavior you want. Choose, per dependency, on purpose.
- **Don't claim you survive a failure you've never tested.** A failover path, a backup restore, a zone-loss tolerance that has only ever existed on a diagram is a hope, not a capability. Exercise it or do not claim it.

The honest framing: designing for failure is the project of moving every row in your blast-radius table toward "contained," in priority order by user impact, at a cost proportional to that impact. It is never finished, because the system keeps changing and new SPOFs keep growing back. It is a practice, not a project.

## Key takeaways

- **Stop trying to prevent failure; design to contain it.** At scale, everything fails all the time. The senior move is to assume every component breaks and engineer so the blast radius stays small and the system degrades instead of collapsing.
- **Know your failure domains and spread across them.** Process, host, rack, zone, region, cell — each is a containment boundary. Three replicas in one zone is one zone outage from total loss. Enforce the spread with hard topology constraints, then *verify the placement*, because a constraint that silently falls back to best-effort has lied to you.
- **Make a blast-radius table and treat every "No" as a work item.** For each failure mode, name the domain, the share of users hit, and whether it is contained. The "No" rows, ranked by user impact, are your backlog.
- **Shrink blast radius by construction with cells.** N cells with per-cell deploys turn a 100% single-deploy outage into a ~1/N one. The reduction is arithmetic, and it compounds with staged rollouts. Reach for it when a wide outage is expensive enough to justify N copies.
- **Hunt the hidden SPOF by asking what your redundant components share.** DNS, config service, CA, deploy pipeline, shared DB, the observability stack — the dangerous single point of failure hides behind apparent redundancy. Dependency-map the startup path, not just the steady-state request path.
- **Design degradation modes on purpose, per dependency.** Fail open (stay available, serve degraded) versus fail closed (stay safe, deny) is a per-case decision driven by what's worse: a wrong answer or no answer. Give every degraded path a metric — a silent degradation is a bug in disguise.
- **Beware correlated failures — they defeat redundancy.** A bad config pushed everywhere, a poison input, a thundering herd, a shared bad binary: these break many domains at once because copies share an exposure. Stage every change, jitter your retries, and re-check every "independent" claim for hidden correlation.
- **A failure design you haven't tested is a hypothesis.** Inject the failures you claim to survive — chaos experiments and game days — and find the hidden SPOF in a controlled window instead of at 3am.

## Further reading

- [Reliability is a feature: the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — the intro map for this whole series, and where the define-measure-budget-respond-learn-engineer loop comes from.
- [Redundancy and failover that actually works](/blog/software-development/site-reliability-engineering/redundancy-and-failover-that-actually-works) — the sibling that goes deep on making your replicas and failover paths genuinely independent and tested.
- [Circuit breakers, bulkheads, and load shedding](/blog/software-development/site-reliability-engineering/circuit-breakers-bulkheads-and-load-shedding) — the resilience patterns that contain failure at the call level, including the retry-storm fixes referenced here.
- [Graceful degradation and fallbacks](/blog/software-development/site-reliability-engineering/graceful-degradation-and-fallbacks) — the sibling that develops the partial-failure and fallback design beyond the fail-open-vs-fail-closed table.
- [Cascading failures, circuit breakers, and bulkheads (system-design)](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads) — the architecture-level treatment of the patterns this post applies at the operate-it level.
- [Multi-region and geo-distribution (system-design)](/blog/software-development/system-design/multi-region-and-geo-distribution) — the deep design treatment of surviving a region-level failure domain.
- The Google SRE Book and SRE Workbook (Google, free online) — chapters on handling overload, addressing cascading failures, and managing critical state are the canonical sources for failure-domain and blast-radius thinking.
- The AWS Builders' Library — the articles on shuffle sharding, static stability, and avoiding fallback in distributed systems are the definitive practitioner treatments of cells and blast-radius reduction.
