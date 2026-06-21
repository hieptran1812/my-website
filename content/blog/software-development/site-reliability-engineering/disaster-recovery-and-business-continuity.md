---
title: "Disaster Recovery and Business Continuity: Bringing the Whole System Back in Another Place"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Disaster recovery is not backups — it is a tested, runbooked plan to bring data, compute, and traffic back in another region within a committed RTO and RPO, and an untested DR plan fails exactly like an untested backup."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "disaster-recovery",
    "business-continuity",
    "rto-rpo",
    "multi-region",
    "failover",
    "dr-drill",
    "game-day",
    "resilience",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/disaster-recovery-and-business-continuity-1.png"
---

The first time I watched an entire cloud region go away, it was not dramatic. There was no smoke, no alarm bell, no movie-grade red lights. There was a wall of pages from one availability zone, then another, then a third, and then the dashboards themselves stopped updating because the thing that scraped them lived in the same region that was dying. Our load balancer health checks all went red at once. Our database was unreachable. Our queue consumers were stuck. The status page — hosted, thankfully, somewhere else — filled with customers asking the same question we were asking each other on the bridge: is this us, or is this the cloud? It was the cloud. A control-plane failure had cascaded across a region, and there was nothing we could do inside that region to fix it. The only way out was *somewhere else*.

That is the moment disaster recovery is for. Not the disk that filled up, not the bad deploy you can roll back, not the noisy neighbor you can throttle. Those are normal operational pain, and your single-region high-availability setup — multiple availability zones, replicas, automatic failover within the region — handles them beautifully. A disaster is the failure that is bigger than your redundancy. It is the correlated, large-scale event that takes out the whole blast radius your redundancy lives inside: an entire region, a provider-wide outage, a fiber cut that isolates a datacenter, a flood or fire, a ransomware event that encrypts your primary and your backups, or a single fat-fingered `terraform destroy -auto-approve` against the wrong workspace at 4pm on a Friday. When the blast radius is the whole region, having three copies inside that region buys you nothing.

Here is the sentence that this entire post defends: **disaster recovery is not backups, and it is not a diagram.** It is a tested, runbooked plan to bring the *whole system* — data, compute, traffic, and the people who run it — back in another place, within a recovery time and a data-loss tolerance the business actually committed to. And the cruelest property of a DR plan is the same as the cruelest property of a backup: an untested one fails exactly when you need it. A backup you have never restored is a tape full of hope. A DR plan you have never exercised is a wiki page full of fiction. The infrastructure-as-code has drifted, the DR region is out of quota, the runbook references resources that were deleted six months ago, the data replication silently stopped in March, and nobody on the current on-call rotation has ever run the steps. As the figure below shows, the gap between a DR plan that is drilled and one that is merely written is the difference between a fifty-five-minute recovery and a nine-hour one.

![A two column before and after comparison contrasting an untested disaster recovery plan whose infrastructure as code has drifted and whose data replication silently stopped against a quarterly drilled plan that stands up the region in forty minutes and cuts traffic over within an hour](/imgs/blogs/disaster-recovery-and-business-continuity-1.png)

By the end of this post you will be able to define what actually counts as a disaster and why normal HA does not cover it; choose a DR strategy tier — backup-and-restore, pilot light, warm standby, or hot active-active — from your business's real RTO and RPO instead of from fear; reason about the three things that must all come back, in order, before the service is truly recovered (data, then compute, then traffic); run a DR game day that finds your gaps in daylight instead of at 3am; and write the business-continuity wrapper that decides who declares a disaster, who talks to customers, and what the manual fallback is when the software is simply gone. This is the *respond* and *engineer-the-fix* part of the series loop — define reliability, measure it, spend the error budget, reduce toil, respond to incidents, learn, and engineer the fix — applied to the failure too big for any single component to absorb. If you have not read the opener, [reliability is a feature: the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) frames why we treat reliability as a number we engineer rather than a property we hope for.

## 1. What a disaster actually is

Let me be precise, because the word "disaster" gets thrown at every incident and that imprecision is dangerous. In reliability engineering, a disaster is a *correlated, large-scale failure that exceeds the redundancy you designed against*. The key word is **correlated**. Your normal redundancy assumes failures are independent: one disk dies, the RAID survives; one node dies, the cluster reschedules; one availability zone has a power event, the other two zones carry the load. Independence is the entire premise of redundancy — you bet that two things will not fail at the same time, and you size your spare capacity around that bet.

A disaster is what happens when that independence assumption breaks. The failures stop being independent and start being *common-cause*. A region-wide network partition does not take down one zone; it takes down your connectivity to all of them simultaneously, because they share the region's network fabric. A bad configuration push does not corrupt one replica; it corrupts every replica it reaches, because they all consume the same config. A ransomware event does not encrypt one server; it spreads laterally and encrypts the primary and the backup repository, because they were reachable from the same compromised credential. The common cause is the disaster. And the brutal consequence is that *more replicas inside the blast radius do not help*. Ten copies of your data in one region is still zero copies once the region is gone.

Here is a concrete taxonomy of what counts as a disaster, drawn from real outages the industry has lived through:

- **A region outage.** A cloud provider's region suffers a control-plane or network failure. You cannot launch instances, your managed database is unreachable, and your inter-zone traffic stalls. This is the canonical cloud disaster, and major providers have all had multi-hour regional events.
- **A provider-wide failure.** Rarer and scarier: a global control plane, an identity service, or a DNS layer fails across regions. A 2021 outage at one major provider took down a swath of the internet for hours because a single internal network and its API control plane degraded globally.
- **A physical event.** A fiber cut isolates a datacenter. A fire, a flood, a cooling failure, or an extended power loss takes a facility offline. In 2021 a fire destroyed a datacenter at a European hosting provider and took millions of websites with it; many had no off-site copy and lost data permanently.
- **A security event.** Ransomware encrypts your production estate and, if you were unlucky, your backups too. Or an attacker deletes your environment. The recovery here is DR plus a clean-room rebuild from immutable backups.
- **Self-inflicted destruction.** The most common "disaster" in practice is human: a `terraform destroy` against prod instead of staging, a script with the wrong account credentials, a mass-deletion automation gone wrong, a bad migration that drops the wrong table fleet-wide. Your own hands are inside the blast radius.

Notice what unites these: in every case, the failure is *bigger than the unit your HA was designed around*. Your multi-AZ database failover assumes the region is fine. Your auto-scaling group assumes the provider can launch instances. Your replicas assume the config they share is sane. Disaster recovery is the layer that activates when those assumptions are the thing that broke. It is, deliberately, a different plan with a different blast radius — and it lives in a different place.

### Why single-region multi-AZ is not DR

I want to kill a specific confusion right here, because I have seen it cost teams real money and real data. Running across multiple availability zones in one region is *high availability*. It is excellent, you should absolutely do it, and it handles the overwhelming majority of failures: a zone losing power, a rack failing, a single node dying. But it is not disaster recovery, because the entire region is a single correlated blast radius. AZs in a region share regional services — the network backbone, often a regional control plane, sometimes shared DNS and identity. When the region itself degrades, all your AZs degrade together. Multi-AZ is independence *within* a region; DR is independence *across* regions (or across providers, or across cloud-and-on-prem).

The working rule I use, and the one I will build the rest of this post on, is a question of blast radius. Ask: *what is the largest single thing whose failure would take me down, and is my recovery plan outside of it?* If the answer to the second half is no — if your "DR" lives in the same region, the same account, the same provider, the same credential — then you do not have a DR plan, you have a slightly larger HA setup that will die in the same disaster. The whole discipline of DR is putting the recovery path genuinely outside the blast radius, and then proving, repeatedly, that you can reach it. This is the operational counterpart to the architecture-level treatment in [multi-region and geo-distribution](/blog/software-development/system-design/multi-region-and-geo-distribution); that post covers designing the topology, while this one covers running the recovery.

## 2. RTO and RPO: the two numbers that define the whole plan

Every DR decision flows from two numbers, and if you take nothing else from this post, take these two definitions and the discipline of getting the business to commit to them.

**RTO — Recovery Time Objective** — is *how long the business can tolerate being down* before the impact is unacceptable. It is a time. "We must be serving traffic again within one hour of declaring a disaster." RTO answers "how fast?"

**RPO — Recovery Point Objective** — is *how much data the business can tolerate losing*, measured as a window of time. It is also a time, but it means something different: "We can lose at most the last fifteen minutes of writes." If your last good replicated state is from 12:00 and the disaster hits at 12:14, you have lost fourteen minutes of data — inside a fifteen-minute RPO, you are fine; outside it, you have violated the commitment. RPO answers "how much data?"

These two numbers are independent, and conflating them is a classic mistake. You can have a tight RPO and a loose RTO: a bank might say "we can be down for two hours, but we cannot lose a single committed transaction" (RTO 2h, RPO ~0). You can have the reverse: an analytics product might say "we must be back in fifteen minutes, but losing the last hour of event data is fine, it'll re-ingest" (RTO 15m, RPO 1h). The DR architecture you build is almost entirely determined by which of those two numbers is tight.

Here is the principle that makes RTO and RPO useful instead of decorative: **they are commitments, not aspirations, and every commitment has a cost.** A near-zero RPO means continuous synchronous or near-synchronous replication, which costs bandwidth and adds write latency. A near-zero RTO means standing capacity in the DR region, which is money sitting idle waiting for a disaster that may never come. The art is not maximizing both — that is the most expensive possible system. The art is extracting from the business the *real* RTO and RPO for each tier of service, and then building the cheapest thing that meets them. An internal reporting tool does not need a fifteen-minute RTO; your payments path might need a five-minute RPO and refuse to compromise.

#### Worked example: turning RPO into a replication lag budget

Suppose the business commits to an RPO of fifteen minutes for the orders service. That number is not just a slogan — it is a *monitorable budget*. It means your data must never be more than fifteen minutes behind in the DR region. So you turn it into an SLI: replication lag, measured continuously, with an alert that fires when lag exceeds, say, five minutes — a third of the budget, giving you headroom to react before you breach. If replication silently stops (the single most common DR failure I have seen), that alert is what tells you in daylight, instead of you discovering it during the disaster when you promote a replica that is six hours stale and blow your RPO by a factor of twenty-four. An RPO with no monitoring is a number you will violate without ever knowing until it is too late. An RPO with a lag alert is a budget you actively defend.

#### Worked example: turning RTO into a recovery time budget

Now the RTO. Say the commitment is one hour. That hour is not free time — it is a budget you spend across every step of recovery, and the steps add up serially. Declaring the disaster and getting the right people on the bridge: five minutes if you have drilled, twenty if you are debating whether it is "really" a disaster. Promoting the data replica: ten minutes. Standing up compute via infrastructure-as-code: thirty to forty minutes if the IaC actually applies cleanly, or "never" if it has drifted and errors out. Cutting traffic over: a few minutes of work, plus the DNS TTL you cannot beat. Verifying the service is actually serving: ten minutes. Add those up and you are at fifty to fifty-five minutes — *inside* the one-hour RTO, but only because each step had a known, drilled duration. The first time you do it cold, every one of those numbers doubles or fails outright, and you blow the hour before you finish standing up compute. RTO is a budget you can only meet by rehearsing the spend.

### How RTO connects to your availability SLO

There is a direct, often-missed link between your RTO and the availability number you publish. Availability is downtime against total time, and a single disaster's downtime comes straight out of your yearly budget. The standard nines table makes this brutal: a 99.9% ("three nines") SLO allows about 8.77 hours of downtime per year; 99.95% allows about 4.38 hours; 99.99% ("four nines") allows about 52.6 minutes per year. So if you commit to four nines and a single regional disaster takes you down for your full RTO, the arithmetic is unforgiving — a one-hour RTO on a four-nines budget means *one* regional disaster blows your entire annual availability budget in a single event. This is why the RTO conversation and the SLO conversation are the same conversation. If the business wants four nines, it cannot also accept a multi-hour DR recovery, because the math does not close. Either the SLO comes down to something a multi-hour recovery can fit inside (three nines comfortably absorbs a one-hour outage with budget to spare), or the RTO must come down to minutes, which forces you up the tier ladder to warm standby or active-active. You cannot have a four-nines SLO and a backup-restore DR tier at the same time; the numbers contradict each other, and discovering that contradiction *before* the disaster is one of the most valuable things this arithmetic does. The error budget that runs through this whole series is the same currency here: a disaster is the single largest withdrawal you will ever make from it.

### The cost arithmetic of DR

The other half of every tier decision is money, and it pays to make the cost as explicit as the RTO. The dominant cost of the faster tiers is *idle capacity* — compute you pay for around the clock that does productive work only during a disaster that may strike once in several years. Walk the tiers as a cost curve. Backup-restore costs you backup storage only: pennies-on-the-dollar of your production spend, because storing compressed snapshots in another region is cheap. Pilot light adds continuous data-replication bandwidth and a sliver of always-on infra (the replica's database instance, the networking) — call it a small single-digit percentage of your production compute bill. Warm standby adds a whole second fleet, even if scaled down; a half-size standby is roughly half your production compute cost, running idle. Hot active-active doubles (or more) your production footprint, plus the cross-region data-transfer costs of keeping both sides consistent, plus the requirement to run each region at N-1 capacity so the survivor can absorb the dead region's full load — which often means each region is sized at well above 50% so that one can carry 100%. The honest framing for the business is: each step up the tier ladder buys you faster recovery by converting a rare, bounded outage cost into a constant, recurring capacity cost. Whether that trade is worth it depends entirely on how much an hour of downtime actually costs you, which is a number the business — not engineering — must own.

## 3. The four DR strategy tiers

There is a well-known taxonomy of DR strategies, popularized by AWS but applicable to any provider, that lines up four tiers from cheapest-and-slowest to most-expensive-and-fastest. Every tier is a different point on the same trade-off curve: you pay more idle cost to buy a faster RTO and tighter RPO. Knowing these four cold, and knowing how to map a business RTO/RPO onto them, is the core practical skill of this whole topic. The matrix below lays them out.

![A comparison matrix of the four disaster recovery tiers backup and restore, pilot light, warm standby, and hot active active across recovery time objective, recovery point objective, idle cost, and when to pick each tier](/imgs/blogs/disaster-recovery-and-business-continuity-2.png)

**Tier 1 — Backup and restore.** The cheapest. You take regular backups of your data and store them off-site (a different region, ideally a different account, ideally immutable). On disaster, you provision fresh infrastructure in a recovery region and restore the backups into it. RTO is hours to days — you are building everything from scratch and waiting on restore throughput. RPO is whatever your backup frequency is — daily backups mean up to a day of data loss; hourly snapshots mean up to an hour. Idle cost is near zero: you pay only for backup storage. This is right for systems where being down for hours is genuinely acceptable: internal tools, batch pipelines, anything where the cost of downtime is less than the cost of standing infrastructure.

**Tier 2 — Pilot light.** The data is replicated continuously to the DR region and a *minimal* set of core infrastructure is kept warm there — the VPC and networking, the database engine (replicating, not serving), secrets, the container images and IaC ready to apply. But the compute that serves traffic is *off*. On disaster, you scale the compute up (apply the IaC, set the desired replica counts) and cut over. The metaphor is a furnace's pilot light: a small flame always burning, ready to ignite the whole system. RTO is ten to sixty minutes — you are launching compute, not building from nothing, and your data is already there. RPO is small — minutes — because data replicates continuously. Idle cost is low: you pay for replicated storage and a sliver of always-on infra, not a full serving fleet. This tier is the workhorse for serious services with a sub-hour RTO and a sub-fifteen-minute RPO. The figure later in this post dissects exactly what stays warm and what stays cold.

**Tier 3 — Warm standby.** A *scaled-down but fully running* copy of the system lives in the DR region. It serves no production traffic (or a token amount), but every tier is up: web, app, database, the lot, just at minimal capacity. On disaster, you scale it up to full capacity and cut traffic over. Because everything is already running, RTO is minutes — there is no cold-start of compute, just a scale-up and a cutover. RPO is seconds, because the database is actively replicating. Idle cost is high: you are running a whole second system, even if small. This is right when minutes of downtime are costly and you have the budget to keep a second fleet breathing.

**Tier 4 — Hot, multi-site active-active.** Full production capacity runs in two (or more) regions *simultaneously*, both serving live traffic. There is no "failover" in the dramatic sense — losing a region just means the survivors absorb its share of the load, the same way losing a node in a cluster does. RTO is near zero. RPO is near zero (with the caveat that true zero-RPO across regions requires synchronous replication, which adds latency and has its own consistency hazards — a topic I will not re-derive here, see the redundancy post). Idle cost is the highest: you are paying for full capacity in multiple places, and you must run at N-1 region capacity so the survivors can actually take the load. This is the high end, justified only when the business genuinely cannot be down — payment rails, ad exchanges, trading systems. This tier blurs into ordinary high availability, which is why I treat the mechanics of it in [redundancy and failover that actually works](/blog/software-development/site-reliability-engineering/redundancy-and-failover-that-actually-works); here the point is just that it is the most expensive end of the DR spectrum.

Here is the same trade-off as a table, with the dimension that actually drives the decision — what the business committed to — on the right:

| Tier | RTO | RPO | Idle cost | Recovery work on disaster | Pick when |
|---|---|---|---|---|---|
| Backup & restore | Hours–days | Hours–day | Lowest (storage only) | Provision everything, restore data | Downtime cheap; internal/batch |
| Pilot light | 10–60 min | ~minutes | Low (data + sliver of infra) | Scale up compute, cut over | Sub-hour RTO, sub-15-min RPO |
| Warm standby | Minutes | Seconds | High (small full fleet) | Scale up, cut over | Minutes of downtime costly |
| Hot active-active | Near zero | Near zero | Highest (full N-region) | None — survivors absorb load | Cannot be down at all |

The single most expensive mistake teams make here is picking a tier from fear instead of from the RTO/RPO. Someone gets scared after an outage, and the reflex is "let's go active-active across three regions." Then they spend a fortune running idle capacity for an internal tool whose users would not notice a two-hour outage. **DR capacity is money sitting idle.** Match the tier to what the business actually needs, not to what would impress an auditor. The decision tree in section 6 makes this mapping concrete.

## 4. The three things that must all come back, in order

When the region is gone and you have declared a disaster, recovery is not a single switch you throw. Three distinct things must all come back, and they must come back roughly in this order, because each depends on the one before it. Get this sequence wrong and you will stand up a beautiful, empty, traffic-receiving service that has no data — or worse, a service serving stale or inconsistent data to real users. The graph below traces the dependency.

![A dependency graph showing a lost region triggering data restoration first, then infrastructure standup which also depends on the recovery region having quota and capacity, then traffic cutover which is gated by the DNS time to live floor, ending in a verified live service](/imgs/blogs/disaster-recovery-and-business-continuity-3.png)

**First, data.** The data must be present in the recovery region, and it must be *consistent*. This is where your RPO is won or lost. If you have been replicating continuously, you promote the replica to be the new primary — and you confirm it is actually caught up, not silently lagging. If you are restoring from backup, you restore and then verify integrity, because a corrupt or partial restore is worse than no restore: it looks like success and serves garbage. The cardinal sin is to bring compute and traffic up *before* the data is confirmed good, because then live users are reading and writing against an inconsistent store, and you have turned a clean recovery into a data-corruption incident on top of an outage. Data first, verified, always.

**Second, compute and infrastructure.** Now you stand up the services that run on the data: the application servers, the API gateways, the workers, the caches, the load balancers, the service mesh. This is where infrastructure-as-code earns its entire existence. If your environment is defined in Terraform, Pulumi, CloudFormation, or Kubernetes manifests, standing up the DR region is *running the same definitions against a different region or account*. If it is not — if your production was hand-built over three years through the console — then "stand up the infrastructure" means "rebuild three years of clicking, from memory, under pressure," and your RTO is measured in days, if you make it at all. There is one more requirement here that bites teams constantly: **the DR region must actually have capacity and quota.** Cloud accounts have per-region quotas on instances, IPs, and managed resources, and a fresh DR region often has *default* quotas far below what your production needs. You must have raised those quotas in advance, or your IaC will apply halfway and then fail with a quota error in the middle of a disaster. The graph above marks capacity and quota as an external dependency feeding into the infra standup for exactly this reason.

**Third, traffic.** Data is good, compute is up — now you redirect users from the dead region to the live one. This is a DNS change, a global load balancer failover, an Anycast withdrawal, or a CDN origin switch, depending on your topology. And here you hit a hard physical floor that no amount of preparation can beat: **the DNS TTL.** If your DNS records have a time-to-live of 300 seconds, then for up to five minutes after you change the record, clients and resolvers that cached the old value will keep sending traffic to the dead region. You cannot make them stop faster than their cache expires. This is why DR-fronting DNS records are kept at a low TTL (30–60 seconds) deliberately, trading a bit more DNS query volume for a faster failover. A global load balancer with health-check-based failover sidesteps much of this by keeping the client pointed at a stable anycast endpoint and rerouting behind it, which is one reason teams with tight RTOs invest in one. Either way, traffic is the last mile, and the TTL is its speed limit.

The principle to internalize: **DR is data plus infra plus traffic, together — not any one of them alone.** I have seen a team replicate their database flawlessly to a DR region and feel safe, having never realized they had no way to actually run their application there. I have seen another team build a gorgeous IaC-defined DR region that could spin up in twenty minutes — and discover during a drill that it had no recent data because the replication was never set up. And I have seen a third stand up data and compute perfectly, then spend forty minutes failing to fail over traffic because their DNS TTL was an hour and they had never thought about it. All three "had DR." None of them had recovery. You need all three, in order, verified at each step.

## 5. Anatomy of a pilot light

Because pilot light is the workhorse tier — the one most serious services land on — let me dissect it concretely, layer by layer, so the abstract "minimal warm infra" becomes specific. The stack figure below shows what stays warm and what stays cold.

![A layered stack of a pilot light disaster recovery region showing an always on data replica at the bottom, then warm infrastructure of networking database engine and secrets, then infrastructure as code and images ready to apply, then cold compute that scales up on disaster, and a DNS failover record on top](/imgs/blogs/disaster-recovery-and-business-continuity-4.png)

At the bottom, **always on: the data replica.** Your database in the DR region is a continuously-updating read replica of production, kept current within your RPO. This layer never sleeps, because it is the one thing you cannot manufacture on demand — you cannot "scale up" data you do not have. Everything above it can be cold; this cannot.

Above it, **warm: the irreplaceable scaffolding.** The networking (VPC, subnets, security groups, routing), the database *engine* running the replica, the secrets and certificates, and the IAM roles. These are cheap to keep alive and slow or fiddly to create from scratch under pressure, so you keep them standing. They consume almost nothing while idle.

Above that, **ready: the definitions.** Your IaC for the serving tier (the autoscaling groups, the deployments, the services) exists and is tested, and your container images or AMIs are already replicated into the DR region's registry. Nothing is *running* here, but everything needed to run is staged. On disaster, you apply the IaC and the images pull instantly because they are local.

Then, **cold: the compute.** The application servers, workers, and serving pods are at zero (or near-zero) desired count. This is the expensive layer, and keeping it cold is exactly why pilot light is cheap. On disaster, you scale the desired count up, the staged images launch, and within minutes you have a serving fleet.

On top, **the DNS / global load balancer failover record.** Pre-configured, low TTL, ready to point at the DR region's endpoint the moment compute is healthy.

The practice here is a small, runnable artifact. Here is the heart of a pilot-light failover encoded as a runbook step that an engineer (or a carefully reviewed automation) executes once the disaster is declared and data is confirmed good. Real commands, copy-and-adapt:

```bash
# DR FAILOVER — pilot light, run only AFTER declaring DR and confirming data
set -euo pipefail
export AWS_REGION=us-west-2          # the DR region
export ENV=prod-dr

# 1. Confirm the data replica is current within RPO before promoting.
#    Refuse to promote a replica that is more than 15 minutes behind.
lag=$(aws rds describe-db-instances \
  --db-instance-identifier orders-dr-replica \
  --query 'DBInstances[0].StatusInfos' --output text)
echo "replica status: $lag  (verify lag < RPO before proceeding)"

# 2. Promote the read replica to a standalone writable primary.
aws rds promote-read-replica --db-instance-identifier orders-dr-replica

# 3. Stand up the serving fleet from IaC (compute was cold at desired=0).
terraform -chdir=infra/dr workspace select prod-dr
terraform -chdir=infra/dr apply -auto-approve \
  -var "serving_desired_count=40" \
  -var "region=us-west-2"

# 4. Wait for the target group to report healthy before cutting traffic.
aws elbv2 wait target-in-service \
  --target-group-arn "$DR_TARGET_GROUP_ARN"

# 5. Cut traffic over via Route 53 failover record (TTL pre-set to 60s).
aws route53 change-resource-record-sets \
  --hosted-zone-id "$ZONE_ID" \
  --change-batch file://dr/route53-failover-to-west.json
```

Notice the order is exactly data → infra → traffic, and notice step 4 — *wait for healthy before cutting over*. The most common way a competent-looking failover still hurts users is cutting traffic to a fleet that has launched but is not yet ready, so health-gating the cutover is not optional. Notice also step 1 refuses to promote a stale replica; a script that blindly promotes whatever it finds is how you turn a recoverable outage into permanent data loss.

### The data layer is where DR is actually won or lost

I want to dwell on the data layer specifically, because in fifteen years of carrying the pager I have seen far more DR plans fail on data than on compute. Compute is, frankly, the easy part: with good IaC it is reproducible, and the worst case is that it takes longer than you hoped. Data is unforgiving. You only have what you replicated, you cannot manufacture it after the fact, and a subtly broken replica is worse than an obviously broken one because it lets you proceed in false confidence. So the practitioner's attention belongs disproportionately here.

The first decision is *how* you replicate, and it is a direct trade between RPO and write latency. **Synchronous replication** waits for the DR region to acknowledge each write before the primary confirms it to the application. This gives you an RPO of zero — every committed write is, by definition, in both places. But it pays for that with latency: every write now eats a cross-region round trip, which between distant regions can be 60–100 milliseconds or more, added to *every single write*. For a high-write system that is often unacceptable, and it also creates a new failure coupling — if the DR region is slow or unreachable, your primary's writes stall, so a DR-region problem becomes a primary-region outage. **Asynchronous replication** confirms the write locally and ships it to the DR region in the background. The primary stays fast and decoupled, but now you have a non-zero RPO equal to the replication lag — whatever has not yet shipped when the disaster hits is lost. Most real systems choose async and then *manage the lag* to stay within RPO, accepting that a disaster loses the last few seconds-to-minutes of writes. The architecture choice between these is covered in depth in the database series; what matters operationally is that you know which one you have, because it determines whether your RPO is genuinely zero or genuinely "however far behind the replica was."

The second discipline is treating replication lag as a first-class SLI you alert on, exactly as the earlier worked example set up. Here is a real Prometheus recording-and-alerting pair for a Postgres replica, the kind of artifact that turns "we have replication" into "we know our replication is meeting RPO right now":

```yaml
groups:
  - name: dr-replication
    rules:
      # Record replication lag in seconds from the DR replica.
      - record: dr:replication_lag_seconds
        expr: pg_replication_lag_seconds{role="dr-replica"}

      # Warn at one third of the 15-minute (900s) RPO budget.
      - alert: DRReplicationLagWarning
        expr: dr:replication_lag_seconds > 300
        for: 5m
        labels: { severity: warning }
        annotations:
          summary: "DR replica lag {{ $value }}s exceeds 1/3 of RPO budget"
          runbook: "https://runbooks/dr-orders#replication-lag"

      # Page when lag approaches the full RPO — we are about to breach.
      - alert: DRReplicationLagCritical
        expr: dr:replication_lag_seconds > 720
        for: 2m
        labels: { severity: critical }
        annotations:
          summary: "DR replica lag {{ $value }}s near RPO breach (900s)"

      # The silent killer: replication has STOPPED, not just lagged.
      - alert: DRReplicationStalled
        expr: increase(pg_last_wal_replay_lsn{role="dr-replica"}[10m]) == 0
        for: 10m
        labels: { severity: critical }
        annotations:
          summary: "DR replica has applied zero WAL in 10m — replication stalled"
```

That last alert is the most important one in the whole file, and the one most teams forget. A lag alert catches a replica that is *behind*. It does not catch a replica that has *stopped*, because a stopped replica that already had low lag will sit at its last value and look fine to a simple lag threshold for a while, or report a lag that grows so slowly you do not notice. The `DRReplicationStalled` alert watches the replay position directly — if the replica has applied zero new write-ahead-log records in ten minutes on a system that is taking writes, replication is dead, full stop, and you need to know that on a Tuesday afternoon, not when you promote it during a disaster and discover it is frozen on data from last month. This is precisely the "replication silently stopped" failure that turned Team B's nine-hour outage, encoded as an alert that would have caught it weeks earlier.

The third discipline is verifying *consistency*, not just presence. Data being in the DR region is necessary but not sufficient — it must also be internally consistent and usable. A backup that restored but is missing a foreign-key relationship, a replica that is current on one table but stalled on another because of a stuck logical-replication slot, a snapshot taken mid-transaction without proper quiescing — all of these *look* like you have your data and all of them serve corruption. The verify step in your runbook must do more than count rows; it should run a synthetic transaction end to end (write an order, read it back, confirm the related records) so that "data recovered" means "data recovered *and demonstrably works*," not merely "the bytes are present."

## 6. Choosing the tier: a decision guide

Now to make the tier choice mechanical instead of emotional. The inputs are the business RTO and RPO. The output is the cheapest tier that meets both. The decision tree below encodes the mapping.

![A decision tree starting from the committed recovery time and recovery point objectives, branching into loose tolerances pointing to backup and restore and tight tolerances pointing to pilot light, warm standby, or hot active active depending on how strict the requirement is](/imgs/blogs/disaster-recovery-and-business-continuity-6.png)

The logic, walked through:

1. **Is the RTO measured in hours or days, and is the RPO similarly loose?** If downtime of several hours is genuinely acceptable and you can tolerate hours of data loss, **backup and restore** is correct and anything fancier is wasted money. Be honest here — "acceptable" means the business has actually agreed, not that an engineer assumed.
2. **Is the RTO sub-hour and the RPO sub-fifteen-minutes, but minutes of downtime are survivable?** This is the sweet spot for **pilot light**. Continuous data replication gives you the RPO; scaling cold compute gives you a sub-hour RTO; idle cost stays low. This is where most serious-but-not-life-critical services land.
3. **Is the RTO a handful of minutes — downtime is genuinely costly per minute?** Step up to **warm standby**. You pay to keep a small full fleet running so there is no cold-start, and you can scale-and-cutover in minutes.
4. **Can the business not be down at all — every minute is direct revenue or safety impact?** Only then **hot active-active**, and only with the budget and the engineering maturity to run multi-region consistency correctly. Reaching for this when the business does not need it is the most common over-spend in DR.

#### Worked example: a SaaS picks its tier

Concretely. A B2B SaaS product runs an orders API. The business, pushed to commit, lands on **RTO one hour, RPO fifteen minutes** — a region outage that loses them service for an hour is painful but survivable, and losing the last fifteen minutes of orders is recoverable because clients retry and reconcile. Now walk the tiers. *Backup and restore*: their nightly backups give a 24-hour RPO, blowing the fifteen-minute requirement by nearly a hundredfold, and restore-plus-rebuild takes them four-plus hours, blowing the RTO. Rejected on both counts. *Hot active-active*: would give them near-zero on both, but at roughly double their compute bill, running idle for a disaster that might come once in three years — for a one-hour RTO requirement, this is paying for a sports car to obey a residential speed limit. Rejected as overkill. *Pilot light*: continuous replication holds RPO at about five minutes (comfortably inside fifteen), and a drilled scale-up-and-cutover lands RTO around fifty-five minutes (inside the hour). It costs them the replicated storage plus a sliver of warm infra — a small fraction of a full second fleet. **Pilot light fits**, and the cost comparison makes the choice obvious. The discipline was not finding the best tier; it was finding the cheapest tier that met the two committed numbers.

This is the same reasoning the error budget brings to the rest of the series: turn a vague "how reliable?" into arithmetic. Here the arithmetic is RTO and RPO against tier cost, and the answer is whichever tier clears the bar for the least money.

## 7. The DR drill nobody runs

Now the heart of the matter, and the recurring warning of this whole post. Everything above — the tier, the runbook, the IaC, the replication — is *necessary and completely insufficient* on its own, because **a DR plan written once and never exercised is fiction.** Not a metaphor. Fiction. It describes a recovery that will not happen the way it is written, because the world drifted out from under it the moment it was written and kept drifting every day since.

Let me enumerate exactly how a written-but-unexercised DR plan rots, because every one of these is a real failure I have seen turn a "we have DR" into a multi-hour catastrophe:

- **The IaC has drifted.** Someone changed production directly — bumped an instance type, added a security-group rule, tuned a parameter — and never mirrored it to the DR definitions. When you apply the DR IaC, it produces a subtly different system that the application does not quite work on, and you debug config differences during the disaster.
- **The DR region lacks capacity or quota.** Your production needs forty large instances; the DR region's default quota is twenty. Your IaC applies, launches twenty, and then fails. You discover the quota limit at the worst possible moment, and quota increases can take hours of support-ticket turnaround.
- **The runbook references deleted resources.** It says "restore from the `orders-backup` bucket" but that bucket was renamed in a cleanup last year. It says "page Sarah, she owns the DR process" and Sarah left in March. The runbook points at a world that no longer exists.
- **Nobody knows the steps.** The person who wrote the plan is gone or has forgotten, and the current on-call has never run it. Under disaster stress, "follow the runbook" becomes "read an unfamiliar document for the first time while executives ask for ETAs."
- **The data replication silently stopped.** This is the deadliest one. A replica fell over months ago, the alert was muted or never existed, and your "continuously replicated" data is actually a snapshot from March. You promote it during the disaster and lose months of data while believing you lost minutes.

Every one of these is invisible until you exercise the plan. And the only way to exercise it is to **actually fail over** — or, where a full live failover is too risky, a structured tabletop plus a partial live test. This is a **DR game day**: a scheduled, deliberate exercise where you simulate (or really cause) the disaster and run the recovery, with a stopwatch and a scribe, and you *measure the real RTO and RPO you achieve*. The entire value is finding the gaps in daylight, on a calm Tuesday afternoon with the right people awake and caffeinated, instead of at 3am during a real region outage when everything else is also on fire. This is the same philosophy as [chaos engineering: breaking on purpose](/blog/software-development/site-reliability-engineering/chaos-engineering-breaking-on-purpose) — you induce the failure you fear, on your own schedule, to learn its real behavior before it ambushes you. DR game days are chaos engineering at the largest blast radius.

The matrix below makes the daylight-versus-disaster contrast concrete: the same gaps, found two ways.

![A matrix comparing four disaster recovery gaps infrastructure drift, missing quota, stopped replication, and stale runbook showing how each one is caught calmly during a drill versus how it surfaces destructively during a real disaster, with the cheap preventive fix for each](/imgs/blogs/disaster-recovery-and-business-continuity-7.png)

### Running a DR game day

A DR game day has a spectrum of intensity, and you climb it as your confidence grows:

1. **Tabletop.** Everyone gathers and *talks through* the disaster step by step. "The region is gone. Who declares it? Okay, you declared it — what's the next command? Pull up the runbook and read me the exact step." No systems are touched. This catches stale runbooks, unclear ownership, and missing knowledge cheaply, and it is the right place to start if you have never drilled.
2. **Partial live test.** You really execute parts of the recovery against the DR region without affecting production: actually run the IaC apply into a scratch DR account and time it; actually restore a backup and verify integrity; actually promote a test replica and check the data. This catches IaC drift, quota limits, and broken restores — the things a tabletop cannot.
3. **Full live failover.** You really fail production over to the DR region and serve real traffic from it, then (ideally) fail back. This is the gold standard and the only thing that gives you a *measured, trustworthy RTO*. Some mature organizations run scheduled regional evacuations regularly — deliberately taking a region out of rotation to prove they can run without it. It is the most expensive and most valuable form of the drill.

Whatever the intensity, three disciplines make it worth doing:

- **Measure real RTO and RPO.** Wall-clock the whole thing. Note where time goes. If your committed RTO is one hour and the drill takes ninety minutes, you do not have a one-hour RTO — you have a discovered gap, found safely.
- **Find and file the gaps.** Every snag becomes a tracked action item with an owner: "DR region orders quota is 20, needs 40 — raise it," "runbook step 7 references deleted bucket — fix." The drill's output is a punch list.
- **Run it on a schedule, quarterly at least.** Drift is continuous, so verification must be recurring. A drill done once is a drill that will be stale within a quarter. Quarterly is a reasonable floor for a serious service; some run monthly game days for critical paths.

#### Worked example: the drill that saved a real outage

Here is the contrast that, to me, is the entire argument for drilling. Two teams, same building, same week, both with "DR." A real region outage hits the provider. **Team A** had a pilot-light DR plan they drilled every quarter. When the pages came, the on-call recognized the pattern, the named decider declared a disaster within five minutes, the data replica was current (their lag alert had confirmed it that morning), the IaC applied cleanly because last quarter's drill had caught and fixed the drift, the quota was already raised, and the DNS cut over on a 60-second TTL. **Measured RTO: about fifty-five minutes**, against a two-hour business target. They beat it with headroom and the postmortem was almost boring.

**Team B**, next door, had a warm-standby plan on paper — a fancier, *theoretically faster* tier than Team A's. But they had never drilled it. The standby fleet's IaC had drifted, so scaling it up produced misconfigured nodes. The data replication had quietly broken weeks earlier and nobody got the alert because the alert had been muted during an unrelated noisy period and never un-muted. The runbook owner had changed teams. They spent the first hour debating whether it was "really" a disaster, the next two hours fighting config drift and discovering the stale data, and the rest reconstructing from older backups. **Their recovery took roughly nine hours.** Same disaster, same provider, same week. The difference was not the tier — Team B's tier was *better* on paper. The difference was that one plan had been exercised and the other had been written. An untested warm standby loses to a drilled pilot light every single time.

## 8. RTO under a real disaster is worse than the drill

Even a well-drilled team should not believe its drill numbers naively, and the honest engineer plans for the gap between the rehearsal and the real thing. **Everything is worse during a real disaster than during a drill**, for reasons that have nothing to do with the technical plan:

- **People are stressed.** A drill is a calm exercise with coffee. A real disaster is your revenue bleeding, executives in the channel, and customers screaming on the status page. Stress narrows attention and slows decisions. The five-minute "declare" step takes twenty because everyone is hoping it will fix itself.
- **Comms are degraded.** Your chat tool, your paging system, your wiki, your dashboards — some of these may live in or depend on the failing region or provider. I have been in an incident where the *runbook itself* was hosted in the region that was down. Plan for your tools to be partially gone, which is why critical runbooks belong somewhere outside your own infrastructure (a printed copy, a separate provider, an offline doc).
- **Dependencies are also failing.** A region outage does not hit you alone. Your payment processor, your auth provider, your DNS host, your customers' systems — some of them are in the same region or provider and are *also* down. Your recovery may stall waiting on a third party who is in their own disaster. The blast radius is correlated across companies, not just within yours.
- **The drill never tested the human edges.** Drills tend to run during business hours with senior people present. The real one comes at 3am on a holiday with the most junior on-call carrying the pager alone. Whatever your drill RTO is, the real-disaster RTO has a fatter tail.

The practical response is to build slack into the commitment. If your drilled RTO is fifty-five minutes, do not promise the business a fifty-five-minute RTO — promise ninety, or two hours, and treat the fifty-five as your internal target. The gap is your buffer against the stress, the degraded comms, and the failing dependencies that the drill could not reproduce. This is the same conservatism that good SLOs use: commit to a number you can hit on a bad day, not on your best day. And this is where the stress-test discipline of the series comes in — before you trust a DR plan, interrogate it: *What if the dependency is also down for two hours? What if the on-call is asleep and the escalation fails? What if two incidents overlap and the DR-trained engineer is busy on the other one? What if the backup we are about to restore has never actually been restored? What if the region we are failing INTO is the one that's degraded?* Every one of those questions is a gap to close before the disaster asks it for you.

### Failing back is its own project

There is a part of DR that almost no one drills and that bites teams hard: *failing back* once the primary region recovers. The failover got all the attention; the return trip is an afterthought, and it is frequently harder than the failover was. Here is why. While you were running in the DR region, you took writes there — orders were placed, records were updated, state diverged. The primary region, meanwhile, has either stale data from before the disaster or no data at all if it was rebuilt. So failing back is not "flip the DNS record the other way." It is a data-migration project under time pressure: you must replicate the new writes that accumulated in the DR region *back* to the recovered primary, confirm the primary is caught up and consistent, and only then cut traffic home — and you must do it without losing the writes that happened during the disaster. Cut back too early and you lose every order placed while you were in DR; that is a second data-loss incident layered on top of the first.

The disciplined answer is to treat failback as a planned, unhurried operation, not a reflex. Once the primary region is healthy, there is usually no urgency to return — you are serving fine from DR. So you reverse the replication (DR becomes the source of truth, primary becomes the replica), let it catch up fully, verify consistency with the same synthetic-transaction checks you used on the way out, and schedule the cutover for a low-traffic window with the same care as a planned migration. Some mature teams simply *do not fail back* automatically at all — whichever region they end up in becomes the new primary, and the old primary becomes the new DR target, so "failover" and "failback" are symmetric operations rather than a there-and-a-scramble-back. Whatever you choose, the runbook must name who owns the failback decision and the steps, because an incident that technically "recovered" but then loses a day of writes during a botched return is not a success — it is a slow-motion second disaster. The earlier runbook outline deliberately ends with "who owns fail-back" for exactly this reason.

## 9. Business continuity: the part that is not technology

Disaster recovery is the technical plan to bring the systems back. **Business continuity** is the larger plan to keep the *company* running through the disaster, of which DR is one part. SREs sometimes stop at the systems and forget that a recovered system with no one coordinating it, no one telling customers what is happening, and no manual fallback for critical functions is only half a plan. The runbook spine figure below ends, deliberately, not at "verify" but at "comms" — because communication is part of recovery, not an afterthought to it.

![A vertical stack of the disaster recovery runbook spine showing the ordered steps declare DR with a named decider, recover data by promoting the replica, stand up infrastructure via infrastructure as code, cut over traffic via DNS or global load balancer, verify service level indicators end to end, and communicate to status page customers and legal](/imgs/blogs/disaster-recovery-and-business-continuity-8.png)

Business continuity adds the parts of the plan that have nothing to do with `terraform apply`:

- **Who decides to invoke DR.** Failing over is itself a disruptive, partly-irreversible action — it can cause its own data and traffic complications, and failing back is its own project. So invoking DR must have a *named decider* (or a small set of them) with the authority to pull the trigger, exactly like declaring an incident has an incident commander. This is the "declare DR early" discipline, and I will come to it next.
- **A communications plan.** Who updates the status page? Who drafts the customer email? Who briefs the executives so they stop interrupting the responders? Who handles the press if it gets that big? These roles are assigned *before* the disaster, written down, and exercised in the tabletop. During a real event, you do not want to be inventing your comms structure.
- **Customer and legal communication.** Many businesses have contractual SLAs with financial penalties, regulatory notification requirements (data-loss disclosure laws, financial-services rules), and customers who need to invoke their own continuity plans. Someone owns talking to customers and someone owns the legal and regulatory clock, and these are not the same person who is promoting the database replica. This composes directly with [communicating during an outage](/blog/software-development/site-reliability-engineering/communicating-during-an-outage), which covers the mechanics of status pages and stakeholder updates.
- **The runbook owner.** A specific person owns keeping the DR runbook current, scheduling the drills, and filing the gaps. A runbook with no owner is a runbook that goes stale, and a stale runbook is the fiction we have been warning about all post.
- **Manual fallback for critical business functions.** Sometimes the most important continuity step is not technical at all. If the system that processes orders is down for hours, can the business take orders by phone and a spreadsheet, and reconcile them later? If the dispatch system is down, can drivers be coordinated by radio? The critical-function fallback keeps the *business* alive even while the *system* is recovering. Identifying which functions are critical enough to need a manual fallback, and writing that fallback down, is core business-continuity work that pure SREs often skip.

Here is a DR runbook outline you can adapt — the spine the figure illustrates, fleshed out into the document an on-call would actually open:

```yaml
# DR RUNBOOK — orders service (review every drill; owner: @sre-lead)
title: "Orders Service Disaster Recovery"
committed: { rto: "1h (internal target 55m)", rpo: "15m" }
tier: pilot-light
dr_region: us-west-2
runbook_location: "printed + Notion + this repo (NOT in primary region)"

declare:
  decider: "@sre-lead or @eng-director (either may declare)"
  trigger: "primary region unhealthy >10m with no provider ETA"
  action: "open #dr-orders, page IC, start the clock, post to status page"

recover_data:
  - "Check replica lag alert is GREEN (lag < RPO)."
  - "If lag breached: STOP, escalate, do not promote a stale replica."
  - "Promote orders-dr-replica to primary; verify writable + row counts."

stand_up_infra:
  - "terraform -chdir=infra/dr apply (workspace prod-dr)."
  - "Confirm quota headroom BEFORE apply (40 large instances)."
  - "Wait for target group healthy before any cutover."

cut_over_traffic:
  - "Apply Route 53 failover record (TTL 60s) to DR endpoint."
  - "Expect up to 60s of split traffic as caches expire."

verify:
  - "Check the SLO dashboard: availability + p99 latency in DR."
  - "Run synthetic order; confirm it writes and reads back."

comms:
  - "Status page: investigating -> identified -> monitoring -> resolved."
  - "Customer email if RTO breach likely; legal owns regulatory clock."
  - "Hand off: who owns fail-BACK once primary region recovers."
```

Notice that the runbook lives explicitly *outside* the primary region, that data recovery has a hard stop on stale replicas, that infra has a quota pre-check, and that the document does not end until comms and fail-back ownership are assigned. That is the difference between a runbook that survives a real disaster and one that was written to pass an audit.

## 10. Declare DR early

The final discipline, and the one that separates a fifty-five-minute recovery from a nine-hour one, is *when* you pull the trigger. Invoking DR is expensive and disruptive: failing over has costs, failing back is a project, and you would rather not do it for a blip that resolves on its own. So the temptation, always, is to wait — to hope the region comes back, to keep limping along, to avoid the big scary decision. And that temptation is, almost always, the wrong instinct.

The principle is identical to declaring an incident: **declaring DR early is cheap; limping is expensive.** Every minute you spend hoping the primary recovers is a minute not spent recovering, and it is a minute you cannot get back — it comes straight out of your RTO budget. If you wait forty minutes to declare on a one-hour RTO, you have already spent two-thirds of your budget on hope, and now you must do the entire fifty-five-minute recovery in the remaining twenty. You will miss. Whereas if you declare at minute five and the region *does* come back at minute fifteen, the cost of having declared early is small — you stand down, you maybe did some no-op preparation, you have a slightly awkward "false alarm" conversation. That asymmetry is the whole argument: the cost of declaring early and being wrong is small and bounded; the cost of declaring late and being right is your blown RTO and a much longer outage.

This is exactly why business continuity names the decider in advance and gives them clear trigger criteria — "primary region unhealthy more than ten minutes with no provider ETA, declare." Clear criteria remove the agonizing in-the-moment debate that eats the budget. The timeline below shows what an early declaration buys you: a recovery that starts at minute five and finishes, verified, at minute fifty-five.

![A left to right timeline of a surviving regional outage with the region going unhealthy at minute zero, disaster recovery declared early at minute five, the data replica promoted at minute twelve, infrastructure stood up at minute forty five, traffic cut over at minute fifty two, and the service verified live at minute fifty five against a two hour target](/imgs/blogs/disaster-recovery-and-business-continuity-5.png)

Read that timeline as a budget being spent wisely. The declare step is small and early. The big chunk is standing up compute, which is unavoidable in a pilot light and is exactly the thing a drill teaches you to do fast. The cutover and verify are quick. The whole thing lands at fifty-five minutes against a two-hour target precisely *because* the clock started at minute five, not minute forty. Declare early, and the budget has room. Limp, and it does not.

## War story: when the region really went away

Let me ground all of this in real, documented industry history, because the abstract case studies above are composites and you should know the genuine articles.

**The OVHcloud fire, March 2021.** A fire broke out at OVHcloud's Strasbourg datacenter campus and destroyed one datacenter (SBG2) entirely, damaging another. Millions of websites went offline. The hard lesson for customers was brutal and simple: many had treated "my data is in the cloud" as equivalent to "my data is safe," with no off-site backup. When the building burned, their only copy burned with it. Some lost data permanently. This is the purest possible illustration of the post's thesis — a physical disaster whose blast radius was a whole facility, against which in-facility redundancy was worthless, and for which the only protection was a recovery copy *genuinely outside the blast radius*. The customers who survived cleanly had off-site, off-provider backups they could restore from. The ones who did not had a DR plan that existed only as an assumption.

**The AWS us-east-1 outages.** AWS's largest region, us-east-1, has had several multi-hour disruptions over the years, including a major one in December 2021 driven by an internal network and control-plane degradation. Because us-east-1 also hosts global control planes for several AWS services, the impact rippled well beyond that one region. The recurring lesson for engineering teams: a single region — even the biggest, most-used one — is a single correlated blast radius, and "we're multi-AZ in us-east-1" is not disaster recovery. The teams that rode these out best were those with a tested cross-region (or cross-cloud) recovery path and the discipline to declare and execute it rather than wait for the region to heal.

**The Facebook/Meta BGP outage, October 2021.** A configuration change during routine maintenance withdrew the BGP routes to Facebook's DNS servers, making the entire estate unreachable from the internet for about six hours — and, in a now-famous detail, complicating the *physical* recovery because internal tools and access systems that engineers needed were themselves behind the failed network. This is the "your comms and tools are also down" hazard made real at the largest scale. The continuity lesson: your recovery procedures and the access required to execute them must not depend on the very systems that are failing. Recovery that requires the broken thing to work is not recovery.

What ties these together is the post's recurring theme. In every case, the surviving organizations were the ones whose recovery path was *outside the blast radius and had been exercised*. The casualties were the ones who had assumed redundancy where they had none, or had a plan they had never run. Disaster does not respect the diagram. It respects the drill.

## How to reach for this (and when not to)

Disaster recovery has real cost — idle capacity, replication overhead, the engineering time to build and drill it — so apply it with judgment, not as a checkbox.

**Reach for serious DR (pilot light or above) when:** the system has real revenue or safety impact, the business has committed to a meaningful RTO/RPO, and a regional or provider outage would cause damage you cannot absorb. Payments, core APIs, anything customer-facing whose downtime costs more than the DR setup. For these, the pilot-light tier plus continuous data replication plus quarterly drills is the defensible default — it covers the disaster without the active-active price tag.

**Reach for backup-and-restore DR (the cheapest tier) when:** the system can genuinely tolerate hours of downtime and hours of data loss. Internal tools, batch pipelines, dev environments, reporting. Do not over-engineer these — a tested off-site, off-provider backup and a documented restore procedure is enough, and spending warm-standby money on an internal dashboard is waste.

**Do NOT** build active-active multi-region for a system whose business RTO is "a few hours is fine" — you are paying a fortune in idle capacity and consistency complexity to beat a bar no one set. **Do NOT** call your single-region multi-AZ setup "disaster recovery" — it dies in a region outage and the label lies to everyone who reads it. **Do NOT** ship a DR plan you have not drilled and report it as coverage — that is the most dangerous state of all, because it creates false confidence; an honest "we have no DR" is safer than a fictional "we have DR," because at least the first one tells the truth about the risk. And **do NOT** treat backups as DR: a backup is one ingredient (it provides the data layer), but DR is data plus infra plus traffic plus people, exercised end to end. The composing post on this is the planned [backups that actually restore](/blog/software-development/site-reliability-engineering/backups-that-actually-restore) — backups are the data layer DR depends on, and an unrestored backup is its own untested fiction.

The honest test for whether you actually have DR is one question: *when did you last fail over, end to end, and measure the RTO?* If the answer is "never" or "I don't remember," you do not have DR — you have a document. Schedule the drill.

## Key takeaways

- **A disaster is a correlated, large-scale failure that exceeds your redundancy** — a region, a provider, a facility, a security event, or your own `terraform destroy`. More replicas inside the blast radius do not help.
- **Single-region multi-AZ is high availability, not disaster recovery.** The region is one correlated blast radius; DR puts the recovery path genuinely outside it.
- **RTO and RPO are the two numbers that define the whole plan** — how fast you must be back, and how much data you can lose. Get the business to commit to them, then build the cheapest tier that meets both.
- **The four tiers trade idle cost for recovery speed:** backup-restore (hours, cheapest), pilot light (sub-hour, low cost, the workhorse), warm standby (minutes, expensive), hot active-active (near-zero, most expensive). Match the tier to the business need, not to fear.
- **Three things must all come back, in order: data, then compute, then traffic.** Data must be present and consistent before anything serves it; the DR region must have quota; the DNS TTL is the floor on how fast traffic can move. DR is all three together, not any one alone.
- **An untested DR plan fails exactly like an untested backup.** The IaC drifts, the quota is missing, the replication silently stops, the runbook references deleted resources, and nobody knows the steps — and all of it is invisible until you exercise the plan.
- **Run DR game days, quarterly at least** — tabletop, then partial live test, then full failover — and measure the real RTO and RPO. Find the gaps in daylight, not at 3am.
- **A real disaster is always worse than the drill:** people are stressed, comms are degraded, dependencies are also failing. Commit to a conservative RTO with buffer, not your best-case drill number.
- **Business continuity is bigger than the systems:** a named decider, a comms plan, customer and legal communication, a runbook owner, and a manual fallback for critical business functions.
- **Declare DR early.** Declaring early and being wrong is cheap; declaring late and being right blows your RTO. Limping is the expensive choice.

## Further reading

- [Site Reliability Engineering (the Google SRE Book)](https://sre.google/sre-book/table-of-contents/) — the foundational text; the chapters on addressing cascading failures and on managing incidents frame the response discipline DR sits inside.
- [The Site Reliability Workbook](https://sre.google/workbook/table-of-contents/) — practical companion; its incident-response and reliability-implementation chapters are the operational counterpart to this post.
- [AWS Disaster Recovery whitepaper and the four DR strategies](https://docs.aws.amazon.com/whitepapers/latest/disaster-recovery-workloads-on-aws/disaster-recovery-options-in-the-cloud.html) — the canonical source for the backup-restore, pilot-light, warm-standby, and active-active tier taxonomy used throughout this post.
- [reliability is a feature: the SRE mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — the series opener; why we engineer reliability as a number, and where DR sits in the define-measure-budget-respond-learn loop.
- [redundancy and failover that actually works](/blog/software-development/site-reliability-engineering/redundancy-and-failover-that-actually-works) — the high-availability layer DR builds on; active-active, health-check failover, split-brain, and N-1 capacity.
- [chaos engineering: breaking on purpose](/blog/software-development/site-reliability-engineering/chaos-engineering-breaking-on-purpose) — DR game days are chaos engineering at the largest blast radius; the same discipline of inducing the failure you fear on your own schedule.
- [communicating during an outage](/blog/software-development/site-reliability-engineering/communicating-during-an-outage) — the status-page and stakeholder mechanics the business-continuity comms plan depends on.
- [multi-region and geo-distribution](/blog/software-development/system-design/multi-region-and-geo-distribution) — the architecture-level treatment of the topology DR runs on; designing the multi-region system, where this post covers running the recovery.
