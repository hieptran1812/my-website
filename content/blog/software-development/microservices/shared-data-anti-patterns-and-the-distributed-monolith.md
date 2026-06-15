---
title: "Shared Data Anti-Patterns and the Distributed Monolith: How Microservices Most Often Go Wrong"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How to recognize, measure, and dismantle the distributed monolith — the failure mode where you split the code but the shared data and coupling keep everything chained together, leaving you with the worst of both worlds."
tags:
  [
    "microservices",
    "distributed-monolith",
    "shared-database",
    "anti-patterns",
    "coupling",
    "distributed-systems",
    "software-architecture",
    "backend",
    "domain-driven-design",
    "migration",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/shared-data-anti-patterns-and-the-distributed-monolith-1.webp"
---

A team I reviewed had what looked, from the org chart, like a textbook microservices platform: seven services, seven repos, seven on-call rotations, a service mesh, a message bus they barely used. From the deploy logs it looked like something else entirely. Every release was a "train": six of the seven services were built, tagged, and deployed together, in a documented order, behind a frozen change window, with a rollback plan that involved reverting all six in reverse. When I asked the obvious question — "can you deploy the order service by itself, right now, without touching the others?" — the room went quiet, and then someone said, honestly, "no, because it reads the customer service's tables directly, and they both import the `common` library, so if the schema or the model changes they have to ship together." That sentence is the whole disease in one breath. They had not built microservices. They had built a **distributed monolith**: a system that had paid the full operational price of being distributed — the network hops, the partial failures, the seven dashboards — while keeping every bit of the coupling that made the original monolith hard to change. They had bought the bill and left the meal on the table.

This is, in my experience, the single most common way microservices go wrong, and it is worth being precise about why, because the failure is sneaky. Nobody sets out to build a distributed monolith. You set out to build microservices, you split the code into services because that is the part everyone talks about, and you quietly skip the harder, less glamorous part — splitting the *data* and severing the *coupling* — because it is invisible in a demo and it shows up only later, in the deploy logs and the incident channel. The result passes every superficial test of "are we doing microservices" (look, separate repos! separate containers! a gateway!) and fails the only test that matters: can each service change and ship on its own. The distinction in the figure below is the one this entire post turns on.

![A side by side comparison of a distributed monolith that deploys together with a shared database and an eight hop synchronous checkout versus decoupled services that deploy independently with a database per service and coarse asynchronous communication](/imgs/blogs/shared-data-anti-patterns-and-the-distributed-monolith-1.webp)

On the left is the distributed monolith: services that must deploy together in a fixed order, a shared database and a shared library that recreate the old coupling under a new topology, and a long synchronous call chain where a single user action threads through eight services in series. On the right is what you were actually trying to build: services that deploy independently, each owning its own data behind a published contract, talking through coarse-grained APIs and events. Both diagrams have the same number of boxes. Only one of them is something you can operate without a coordination meeting. By the end of this article you will be able to *diagnose* a distributed monolith with metrics rather than vibes — a deploy-coupling number, a co-change analysis, a dependency graph you can actually look at — name the specific data and code anti-patterns that produce it, and apply the concrete fixes that pull the two pictures apart: re-drawn boundaries, data per service, a god library broken into a published contract, chatty synchronous chains collapsed into coarse calls or events, and anti-corruption layers at the seams.

This is the capstone of the data track, so it ties the thread together. The rule that defines microservices is [database per service](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices); the distributed monolith is what you get when you violate it while pretending you didn't. And there is an honest meta-point I will keep returning to and will not soften: **if you have a distributed monolith, a well-built [modular monolith](/blog/software-development/microservices/monolith-first-and-the-modular-monolith) would have been strictly better.** Same in-process speed, same single deploy, none of the network tax — and the coupling would have been *honest* instead of hidden. The distributed monolith is the one architecture that loses on every axis at once.

## 1. What "distributed monolith" actually means

Let me define the term carefully, because it gets thrown around loosely. A **monolith** is a system that ships as one deployable unit. Its defining trait is not size or ugliness — it is that there is one build artifact, one process to deploy, and a change anywhere ships everywhere. A **microservices** architecture, properly built, is many deployable units that can each change and ship *independently*, because they are loosely coupled — each owns its data, exposes a stable contract, and degrades gracefully when its neighbors are down. Independent deployability is the load-bearing word. Sam Newman, in *Building Microservices*, makes it the litmus test: if you cannot deploy a service on its own without lock-step coordination, you do not have a microservice, whatever the topology says.

A **distributed monolith** is the architecture that *looks* like microservices — separate processes, separate repos, network calls between them — but behaves like a monolith, because the services are tightly coupled and therefore cannot change or deploy independently. You have distributed the *code* across the network without distributing the *coupling* out of it. The coupling is still there; it has just gone underground, from in-process function calls (which the compiler can see and the deploy can manage) to schema dependencies and library dependencies and synchronous call chains (which nothing manages, until they break in production).

The reason this is the *worst* outcome — worse than either endpoint — is that the two architectures it sits between have opposite cost structures, and the distributed monolith manages to inherit the bad half of each. A monolith's cost is that everything is coupled, but it pays *nothing* for distribution: a call between modules is a function call, nanoseconds, no timeout, no retry, no circuit breaker, one stack trace when it breaks. Microservices' cost is the operational tax of distribution — every call can time out, partially fail, get retried, needs tracing, needs a degradation story — but they buy decoupling in return, so a change is local and a deploy is independent. The distributed monolith pays the distribution tax (network hops, partial failures, seven dashboards) *and* keeps the coupling (lock-step deploys, change ripple, shared schema). It is the only point in the design space where you pay both bills and collect neither benefit. That is not a slope you slide down gracefully; it is a trap you fall into by doing the easy half of the work and skipping the hard half.

The taxonomy of how teams fall in is small enough to memorize, which is part of why it is worth memorizing — almost every real case is one of three roots, sketched in the figure below: you shared the *data*, you shared the *code*, or you wired services together with *synchronous* calls so tightly that one being down makes the others useless.

![A tree diagram grouping the distributed monolith anti patterns into shared data with a shared database, shared code with a god common library, and temporal coupling with an eight hop synchronous call chain](/imgs/blogs/shared-data-anti-patterns-and-the-distributed-monolith-3.webp)

Each root produces a recognizable symptom set, and most real distributed monoliths have all three at once, reinforcing each other. The rest of this post walks each branch — the symptom, how to measure it, the root cause, and the concrete fix — using one running example so the abstract pattern always has a body attached to it.

## 2. ShopFast: a distributed monolith in the wild

The running example is **ShopFast**, the same e-commerce system that appears across this series. ShopFast started as a Rails monolith, grew to four "services" during a hurried decomposition two years ago, and now exhibits the full distributed-monolith symptom set. Here is the topology, and it is the kind of picture that looks fine until you trace the arrows.

![A graph of the ShopFast topology where order, customer, payment, and shipping services all read and write a single shared database and all import a shared common library, so any schema change breaks all four](/imgs/blogs/shared-data-anti-patterns-and-the-distributed-monolith-2.webp)

There are four services — order, customer, payment, shipping — and on a slide they look independent. But follow the edges. Every one of them connects to a single shared Postgres database with one schema, and every one of them imports a library called `common` that holds the shared domain model, the ORM entities, and a pile of "utility" code. The customer service has metastasized into a god service that owns users, addresses, profiles, preferences, loyalty points, payment-method tokens, and notification settings, so almost every flow in the system has to call it. And the checkout path is a single synchronous chain that threads through all four services and a few more, in series, before the customer's browser gets a response.

Let me name the six symptoms explicitly, because the diagnosis is "do you have these," not "does it feel monolithic":

- **Services deploy together, in a specific order.** ShopFast releases on a train. You cannot ship order without shipping customer, because they share a schema and a library; and you must deploy customer *before* order, because order's new code reads a column customer's migration adds.
- **One change ripples across many services and repos.** Renaming `customer.full_name` to split it into `first_name` / `last_name` is a six-repo pull request and a coordinated migration. A change that would be one commit in a monolith is a project.
- **A shared library everyone depends on, recreating the monolith.** `common` is imported by all four services. It contains the entity classes, so a change to the `Order` model is a change to every service's compiled artifact. The library *is* the monolith, just vendored into four places.
- **A shared database several services read and write.** Order writes to `orders`, reads from `customers` and `inventory`. Payment reads `orders` directly to check totals. Nobody owns any table, so nobody can change any table.
- **Chatty synchronous call chains.** A single Place Order request hits eight services in series. Latency adds up; availability multiplies down.
- **Temporal coupling — service A is useless if B is down.** Order cannot do anything if customer is down, because it reads customer's tables synchronously to validate every order. The "independent" services are independent the way two prisoners chained at the ankle are independent.

If three or more of these are true, you have a distributed monolith, and the fix is not "more microservices." It is to sever the coupling, one branch of the taxonomy at a time. But first you have to *measure* it, because "it feels coupled" is not something you can put in a planning doc, and "we should decouple" without a number attached gets deprioritized every quarter.

It is worth pausing on *why* a team ends up here with good intentions and competent engineers, because the answer is not incompetence — it is incrementalism. Nobody decides to build a distributed monolith; they make a series of locally reasonable decisions that sum to one. The first service split was clean. Then a feature needed a field that lived in another service, and reading its table was faster than designing an API, so someone did, "temporarily." Then a second service needed the same `Order` class, and copying it felt wasteful, so it went into `common`. Then a flow needed three services to cooperate, and a synchronous chain was the obvious wiring. Each step was a small, defensible convenience, and the deploy-coupling cost of each was invisible at the time — it only became visible later, in aggregate, as the release train. This is why the diagnosis has to be quantitative and continuous: the disease accumulates one reasonable shortcut at a time, below the threshold of any single code review's notice, and only a metric tracked over time catches the trend before it hardens into a release process nobody can change.

## 3. Measuring it: three metrics that turn a feeling into a number

The senior move here is to refuse to argue about whether the architecture is coupled and instead measure it. Three measurements do almost all the work: deploy coupling, co-change, and the dependency graph. None of them require fancy tooling — they live in your git history and your CI logs.

### 3.1 The deploy-coupling stress test

The fastest diagnostic is a question you ask a human, then verify in the logs: **"Can you deploy service X alone, right now, with no coordination, and if not, why?"** Run it for every service. For each "no," write down the reason. At ShopFast the answers were: order — no, shares schema with customer; customer — no, the `common` model changes ripple; payment — no, reads order tables directly; shipping — mostly yes. A platform where the honest answer is "no" for most services is a distributed monolith by definition.

You can make this quantitative from CI history. For each deploy event, count how many *other* services were deployed within the same change window (say, a 2-hour window or the same release tag). The **deploy-coupling ratio** is the average number of co-deployed services per release. A true microservices fleet trends toward 0 (each service ships alone); a distributed monolith trends toward N−1 (everything ships together).

```bash
#!/usr/bin/env bash
# deploy-coupling.sh — estimate how often services ship together.
# Reads a deploys.csv of: timestamp_iso,service  (one row per deploy event)
# A release "train" = deploys within WINDOW seconds of each other.

WINDOW="${1:-7200}"   # 2-hour window by default
INPUT="${2:-deploys.csv}"

awk -F, -v w="$WINDOW" '
  NR>1 {
    # parse ISO8601 to epoch via date is slow; assume col1 is already epoch seconds
    t=$1; svc=$2
    rows[NR]=t SUBSEP svc
    times[NR]=t; svcs[NR]=svc; n=NR
  }
  END {
    total_codeploys=0; releases=0
    for (i=2;i<=n;i++) {
      delete seen
      seen[svcs[i]]=1
      # how many distinct OTHER services deployed within the window of row i
      for (j=2;j<=n;j++) {
        if (j!=i && (times[j]-times[i] >= 0) && (times[j]-times[i] <= w))
          seen[svcs[j]]=1
      }
      c=0; for (s in seen) c++
      total_codeploys += (c-1)   # minus self
      releases++
    }
    printf("releases=%d  avg co-deployed services per release=%.2f\n",
           releases, total_codeploys/releases)
  }' "$INPUT"
```

At ShopFast this came back `avg co-deployed services per release=4.6` against five services — meaning a "deploy" almost always meant deploying nearly everything. The target after the work was below 0.5. That single number, tracked release over release, did more to keep the decoupling project funded than any architecture diagram, because it turned "the system feels coupled" into a line on a graph that leadership could watch go down.

### 3.2 Co-change analysis: what the git history confesses

Deploy coupling tells you what ships together. **Co-change analysis** tells you what *changes* together, which is the deeper signal because it precedes the deploy. The idea is simple: if two services' files keep appearing in the same commits or the same PRs, they are logically one thing, no matter how many repos you have split them into. Healthy services rarely co-change; a distributed monolith co-changes constantly.

If everything lives in a monorepo, this is a one-liner over git log. If services are in separate repos, you do it over a merged change-log keyed by PR. Here is the monorepo version, which buckets every commit by which top-level service directories it touched.

```bash
#!/usr/bin/env bash
# co-change.sh — what fraction of commits touch >= K services?
# Assumes services live under services/<name>/...

K="${1:-3}"
LOOKBACK="${2:-1 year ago}"

git log --since="$LOOKBACK" --name-only --pretty=format:'__COMMIT__' \
| awk '
  /^__COMMIT__/ { if (n>0){ commits++; if (cnt>=ENVIRON["K"]) wide++ }; delete seen; cnt=0; n=0; next }
  /^services\// {
    split($0, p, "/"); svc=p[2]
    if (!(svc in seen)) { seen[svc]=1; cnt++ }
    n++
  }
  END {
    commits++; if (cnt>=ENVIRON["K"]) wide++   # flush last
    printf("commits=%d  touching>=%d services=%d  (%.0f%%)\n",
           commits, ENVIRON["K"], wide, 100.0*wide/commits)
  }' K="$K"
```

```bash
$ K=3 co-change.sh 3 "1 year ago"
commits=2841  touching>=3 services=1993  (70%)
```

Seventy percent of ShopFast's commits touched three or more services. Read that again: most of the time an engineer sat down to make a change, they had to edit three services to do it. That is not a microservices codebase; it is a monolith with extra steps. The number you want is the inverse — the *vast* majority of commits should touch exactly one service. When 70% touch three or more, your boundaries are in the wrong place, full stop, and no amount of resilience tooling will save you, because the problem is upstream of resilience: the system is one logical thing wearing four costumes.

A finer-grained version builds a co-change *matrix*: for every pair of services, the count of commits touching both. The pairs with the highest co-change counts are your merge candidates (they probably should be one service) or your worst leaks (one is reaching into the other). At ShopFast, the order/customer pair dominated — they co-changed in 61% of commits that touched either — which pointed the finger straight at the shared schema and the god customer service.

A subtle point about interpreting the co-change matrix, because it is where teams misread the data: a high co-change number between two services has two very different causes, and the fix is opposite in each case. If A and B co-change because they are *one logical capability that was wrongly split* — say, "order" and "order-lines" were carved into separate services — then the right move is to *merge* them back into one service, because the boundary between them is fictional and is generating coordination cost for nothing. But if A and B co-change because A *reaches into* B (A's code keeps changing whenever B's schema changes, because A reads B's tables), then the right move is the opposite — *sever* the reach, give A its own data via a contract, so that B can change without A having to. The matrix tells you *that* two services are entangled; you have to look at *why* (read the actual commits) to know whether to merge or to sever. A senior does not act on the number alone; the number tells you where to look, and reading a handful of the co-changing commits tells you which fix applies. Getting this wrong — merging two services that should have been decoupled, or trying to decouple two that should have been merged — wastes a quarter and leaves the coupling intact.

### 3.3 The dependency graph and its cycles

The third measurement is structural: build the actual dependency graph and look for cycles. In a clean architecture, dependencies form a DAG — you can topologically sort the services, and the arrows point one way. A **cycle** (A depends on B depends on C depends on A) is a smoking gun: those services are mutually entangled and can never be deployed or reasoned about independently, because each needs the others to function. Cycles are how the dependency graph tells you, mathematically, that you do not have a real boundary.

```python
# depgraph.py — build the service dependency graph from declared calls + DB grants,
# then report strongly connected components (cycles) and fan-in hotspots.
import json
from collections import defaultdict

# edges: A -> B means "A depends on B" (calls it, reads its tables, imports its lib)
edges = json.load(open("service_deps.json"))   # [["order","customer"], ...]

graph = defaultdict(set)
indeg = defaultdict(int)
for a, b in edges:
    if b not in graph[a]:
        graph[a].add(b)
        indeg = indeg  # noqa
        indeg_count = indeg
        indeg[b] += 1

# Tarjan's SCC — any component with >1 node is a dependency cycle.
index = {}; low = {}; on = {}; stack = []; sccs = []; counter = [0]
def strongconnect(v):
    index[v] = low[v] = counter[0]; counter[0] += 1
    stack.append(v); on[v] = True
    for w in graph[v]:
        if w not in index:
            strongconnect(w); low[v] = min(low[v], low[w])
        elif on.get(w):
            low[v] = min(low[v], index[w])
    if low[v] == index[v]:
        comp = []
        while True:
            w = stack.pop(); on[w] = False; comp.append(w)
            if w == v: break
        sccs.append(comp)

for v in list(graph):
    if v not in index:
        strongconnect(v)

cycles = [c for c in sccs if len(c) > 1]
print("dependency cycles (must be empty for independence):")
for c in cycles:
    print("  CYCLE:", " -> ".join(c), "-> (back)")

hot = sorted(indeg.items(), key=lambda kv: -kv[1])[:3]
print("highest fan-in (god-service candidates):", hot)
```

```
dependency cycles (must be empty for independence):
  CYCLE: order -> customer -> order -> (back)
highest fan-in (god-service candidates): [('customer', 4), ('common', 4)]
```

There it is in two lines of output: a cycle between order and customer (order reads customer data; customer's notification flow calls order to fetch totals), and the highest fan-in nodes are customer and `common` — the god service and the god library, the two things everything depends on. A senior reading this output does not need to debate the architecture anymore. The git history and the graph have already rendered the verdict; the only question left is the order in which to dismantle it.

## 4. The root causes: why teams build this

Before the fixes, it is worth being honest about *why* this happens, because if you fix the symptoms without fixing the cause you will rebuild the distributed monolith within a year. There are four root causes, and they are all decisions that felt reasonable at the time.

**Wrong boundaries.** This is the deepest one. Microservices' entire value comes from cutting along seams where one capability rarely forces a change in another. If you cut along the wrong seams — by technical layer (a "data service," a "logic service") instead of by business capability, or by guessing on a whiteboard before you understood the domain — then every real change crosses your boundaries, and crossing a boundary now means a network call and a coordinated deploy. The cure is to draw boundaries around capabilities, which is the whole subject of [service boundaries with domain-driven design](/blog/software-development/microservices/service-boundaries-with-domain-driven-design). If your co-change analysis shows two services always change together, the boundary between them is fictional.

**Entity services — the god "User service."** A particularly seductive wrong boundary is to make one service per *entity*: a User service, an Order service, a Product service. It sounds clean and object-oriented. It produces a distributed monolith reliably, because business capabilities span entities. "Place an order" needs user, product, inventory, and payment data; if each is a separate service and they all hold each other's data hostage, every flow becomes a chatty cross-service dance, and the entity that everything references — usually User — becomes a god service with enormous fan-in (exactly the `customer` node in our graph). Capabilities, not entities, are the right unit. A "Checkout" capability owns the slice of order, payment-intent, and cart data it needs to do its job, rather than asking three entity services for permission.

**Sharing data instead of capabilities.** When service A needs something service B knows, there are two things A can ask for: B's *data* (give me your `customers` row) or B's *capability* (tell me whether this customer may check out). Sharing data couples A to B's schema forever; sharing a capability couples A only to B's published behavior, which B can keep stable while changing everything behind it. The distributed monolith is built almost entirely out of services asking each other for data — reaching into tables, reading caches, importing models — instead of asking for capabilities.

**Synchronous where asynchronous belonged.** The fourth cause is wiring everything together with synchronous request/response when the interaction did not actually need an immediate answer. Notifying the customer, updating the shipping system, refreshing a search index — none of these need to happen before the customer's Place Order call returns, yet they get bolted onto the synchronous chain because that is the path of least resistance. Every synchronous link is a temporal coupling: it forces both services to be up at the same instant. The fallacies of treating the network as reliable and free are exactly what [inter-service communication fundamentals](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies) warns against, and the chatty chain is what you get when a team ignores them.

## 5. The data anti-patterns in detail

The data track exists because data is where the coupling hides best. Let me catalogue the specific data anti-patterns, in roughly increasing order of how hard they are to spot.

### 5.1 The shared database

The clearest one: several services read and write the same database. This is the cardinal sin precisely because it is so convenient — a JOIN is free, a transaction across tables is free, there is no API to design. And it is precisely those conveniences that are the coupling. If order can JOIN against `customers`, then order depends on customer's schema, silently, with no contract, no version, no warning. The day customer renames a column, order breaks, and the only place that dependency was written down was inside a SQL string nobody grepped for. The shared database is the violation of the [database-per-service rule](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices), and it is the load-bearing reason most distributed monoliths cannot deploy independently.

### 5.2 Reaching into another service's tables or cache

A subtler cousin: even with separate logical databases, a service "just this once" connects to another service's database, or reads its Redis cache, to avoid the latency of an API call. This is the same disease in a smaller dose — a hidden, contract-free dependency on internal state someone else owns. The cache version is especially nasty because cache keys and value formats are even less stable than schemas; the owning team will change a cache encoding without a second thought, because caches are "internal," and the reaching service will break in a way that is murder to trace.

How do you *find* these reaches, given that they are by definition undocumented? Three mechanical sweeps catch almost all of them. First, audit the database grants: a service's DB role should have privileges on exactly its own schema and nothing else, so any `GRANT` that crosses a service boundary is a reach you can see in one query. Second, grep the codebase for connection strings and cache clients pointed at hostnames that aren't the service's own — a payment service that constructs a `redis://customer-cache:6379` connection is reaching, full stop. Third, and most reliable, lock it down in staging (revoke the cross grants, firewall the cache) and watch what breaks; every permission-denied error in the test suite is a dependency the audit missed. The combination of "audit what's declared" and "break what's hidden" is how you turn an invisible coupling into a checklist you can actually finish, rather than a thing you hope you found all of.

The reason this anti-pattern is so seductive deserves a sentence, because understanding the temptation is how you resist it: the reach is *always* faster and easier in the moment. The API call is one more hop, one more contract to design, one more thing that can be down; the direct read is right there, sub-millisecond, no coordination required. The reach pays you immediately and bills you later, when the owning team changes the internal state and your service breaks at the worst possible time, with no compile error and no contract test to have warned you. Senior engineers treat "I'll just read their table for now" the way they treat "I'll just disable this test for now" — a small convenience that quietly mortgages the future, and one that is far cheaper to refuse than to unwind.

### 5.3 The shared mutable schema

When services *do* share a database, they share its migrations. Any schema change is now a multi-service event: you cannot add a NOT NULL column without every writer being ready for it, you cannot drop a column without every reader having stopped reading it, and you cannot do either without a coordinated release. The schema becomes a frozen asset that everyone is afraid to touch, which is the exact opposite of the evolvability microservices were supposed to buy. The shared schema is *why* the lock-step release exists.

### 5.4 The giant shared domain-model library

The code-side equivalent of the shared database is a `common` (or `core`, or `shared-models`, or `domain`) library that contains the domain entities and gets imported by every service. It feels like good DRY hygiene — why define `Order` four times? — but it is a coupling engine. Now a change to `Order` recompiles and forces a redeploy of every service that imports it, in lock step, because they all share the version. The library has reassembled the monolith inside the dependency manager. A little duplication across services is far cheaper than this shared coupling — a point worth tattooing on a whiteboard, because it runs against every instinct trained on monolith development.

### 5.5 Distributed transactions everywhere

The last data anti-pattern is reaching for a distributed transaction — a two-phase commit, or worse, an XA transaction across service databases — to keep multiple services' data consistent on every write. This couples the services' *availability* (the transaction cannot commit unless every participant is up and responsive) and their *latency* (everyone waits for the slowest), and it does not even scale. The right tool for cross-service consistency is the [saga pattern](/blog/software-development/database/saga-pattern-distributed-transactions) — a sequence of local transactions with compensations — paired with the [transactional outbox](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing) for reliable publishing. Distributed transactions everywhere is a sign you are trying to preserve a monolith's consistency model in a distributed system, which is the consistency version of the distributed monolith.

The deeper failure here is conceptual, not mechanical. In a monolith, a single ACID transaction wraps "deduct inventory, charge the card, create the order," and either all three happen or none do — the database guarantees it, for free, on every write. The instinct that survives the move to services is to want that same all-or-nothing guarantee across three service databases, and a distributed transaction is the only thing that *technically* provides it. But it provides it at a cost that defeats the entire point of having split the services: now inventory, payment, and order must all be up, responsive, and holding locks simultaneously for the duration of the slowest participant, so you have re-coupled their availability tighter than the monolith ever coupled them (the monolith only needed *one* database up; now you need all three plus the coordinator). The two-nines availability math from the chatty chain applies here too, with the added insult of held locks that block other transactions while everyone waits. A saga accepts that cross-service consistency will be *eventual* — the order is created, then an event drives the inventory deduction, then another drives the charge, and a failure anywhere triggers compensating actions that undo the earlier steps — trading the monolith's instantaneous consistency for availability and independence. Choosing the saga is not settling for a weaker guarantee out of laziness; it is recognizing that the strong guarantee was never compatible with the architecture you chose, and that [eventual consistency in practice](/blog/software-development/microservices/data-consistency-and-eventual-consistency-in-practice) is the consistency model distribution actually affords you.

## 6. The chatty synchronous chain, with the latency math

The temporal-coupling branch of the taxonomy deserves its own treatment, because it is where the operational pain is most viscely felt and where the numbers are most persuasive. ShopFast's Place Order is an eight-hop synchronous chain, drawn below: the gateway calls order, which calls customer, which calls inventory, which calls payment, which calls shipping, which calls notify — each waiting for the last before proceeding, all on the critical path of the customer's click.

![A graph of an eight hop synchronous checkout where the gateway calls the order service which calls customer then inventory then payment then shipping then notification in series and any single hop being down fails the whole call](/imgs/blogs/shared-data-anti-patterns-and-the-distributed-monolith-4.webp)

Before the math, notice what the chain quietly assumes: that every one of these calls *must* happen before the customer gets an answer. Walk them and that assumption falls apart. Validating the customer and reserving inventory genuinely must precede the order being confirmed — those are real preconditions. Charging the card must happen and the customer must hear whether it succeeded — that one truly belongs on the synchronous path. But arranging shipping? Sending the confirmation email? Updating the search index and the analytics pipeline? None of those need to complete before the browser renders "Order placed." They were bolted onto the synchronous chain not because the business required it but because synchronous request/response was the path of least resistance for the engineer wiring them up. Distinguishing the calls that *must* block from the calls that merely *happen to* block is the entire intellectual content of fixing a chatty chain, and most of the chain turns out to be the second kind.

Two things compound badly down a synchronous chain: latency and unavailability. Both are multiplicative, and multiplicative numbers get away from you fast.

#### Worked example: the 8-hop chain's compounded availability and latency

Take the chain as eight hops, each a healthy-looking service with 99.9% availability (three nines — about 43 minutes of downtime a month) and a p99 latency of 40 ms.

**Availability.** A synchronous chain succeeds only if *every* hop succeeds, so availabilities multiply:

```
P(whole chain up) = 0.999 ^ 8 = 0.99203...  ≈ 99.2%
```

You combined eight three-nines services and got a checkout flow with about *two* nines — roughly 0.8% of requests failing, or about 5.7 hours of effective downtime a month, none of which is any single service's fault. Each team can truthfully report "my service met its SLO," and the customer-facing flow still misses its target by an order of magnitude. This is the cruelest property of the chatty chain: it manufactures unreliability that nobody is individually accountable for.

**Latency.** p99 latencies do not simply add, but a long serial chain pushes the *tail* out badly, because the probability that *at least one* hop hits its slow tail rises with chain length. Even with the optimistic assumption that latencies add (they often do worse), the serial p99 is:

```
serial p99  ≈ 8 × 40 ms = 320 ms   (just the service time)
+ 7 network round-trips × ~1 ms     = +7 ms
+ connection / serialization overhead per hop
```

So a checkout that *should* feel instant sits at 320+ ms at the tail before you have added a single retry — and the moment one hop gets slow under load and the callers retry, the chain amplifies the load downstream and the tail blows out completely. Compare to the fixed version (next section): one blocking payment call (~120 ms including the external PSP) plus an event emit, and everything else moved off the critical path. The p99 drops from 320 ms to roughly 130 ms, and — far more importantly — the availability stops being a product of eight terms.

The temporal coupling is the deeper problem even than the latency. Because order reads customer's tables synchronously on every request, **order is useless the instant customer is down.** A service that cannot function when its neighbor is unavailable is not loosely coupled; it is the same service wearing two pods. The fix, which the next sections build, is to stop asking neighbors for data on the hot path: cache what you need locally (fed by events), make only the one call that genuinely must block (payment), and push everything that can wait (shipping, notification, search indexing) onto an asynchronous path where a downstream outage queues work instead of failing the customer.

## 7. The scorecard: why the distributed monolith loses to a modular monolith

Here is the trade-off section, and it is a blunt one, because the honest conclusion is blunt. Set the three architectures side by side — distributed monolith, true microservices, modular monolith — and score them on the axes that actually drive cost: deploy independence, change locality, coupling, and operational cost.

![A matrix scoring the distributed monolith, true microservices, and modular monolith on deploy independence, change locality, coupling, operational cost, and a net verdict, showing the distributed monolith worst on every row](/imgs/blogs/shared-data-anti-patterns-and-the-distributed-monolith-5.webp)

| Property | Distributed monolith | True microservices | Modular monolith |
| --- | --- | --- | --- |
| Deploy independence | None — ships in a train | Full — each ships alone | Single unit (but one fast deploy) |
| Change locality | Ripples across many services | Local to one service | Local to one module |
| Coupling | Tight **and** remote (worst of both) | Loose, contract-based | In-process, compiler-checked |
| Latency between parts | Network hops everywhere | Network hops (paid for) | Function calls (nanoseconds) |
| Partial failure surface | Huge — every hop can fail | Designed for (resilience patterns) | None — no network between parts |
| Operational cost | Highest — N deploys, N dashboards | High but earns its keep | Lowest — one of everything |
| When it wins | **Never** | Many teams, divergent scale, blast-radius isolation | Almost every other case |

Read down the distributed-monolith column: it is the worst or tied-worst on every single row. That is not rhetorical exaggeration; it is the structural truth. It pays the microservices operational cost (N deploys, N dashboards, N on-call rotations) and pays the network latency and partial-failure tax, while keeping the monolith's tight coupling and change ripple. A **modular monolith** beats it on every axis: same single deploy, but the coupling is *in-process and compiler-checked* instead of hidden in SQL and shared libraries, the latency between parts is a function call instead of a network hop, and there is no partial-failure surface at all because there is no network between the parts. The modular monolith even has *more honest* coupling — when modules are too entangled, the build tells you, instead of an incident at 3am telling you.

This is the meta-point of the whole post, and it is worth stating without hedging: **the distributed monolith is the only architecture that is dominated.** There is no scenario where it is the right answer. If you find yourself in one, the question is never "how do we do microservices better here"; it is "do we decouple into real microservices (because we genuinely have the team count and scale to justify the operational cost), or do we collapse back into a [modular monolith](/blog/software-development/microservices/monolith-first-and-the-modular-monolith) (because we don't)?" Both of those are improvements. Staying put is the only losing move.

#### Worked example: the lead-time cost of a coordinated release

Numbers make the "ripples across many services" row concrete. Say one schema change at ShopFast touches six services. Compare the engineering cost as a coordinated release versus the same change in a modular monolith.

In the distributed monolith, a single column rename becomes:

```
6 services × (PR + review + CI ~25 min each)      = 6 PRs, 6 reviews
+ 1 coordinated migration plan (expand/contract)   = ~0.5 day to author
+ 1 release-train scheduling + change window        = ~0.5 day coordination
+ ordered deploy of 6 services + verification       = ~2 hours
+ a rollback plan that reverts all 6 in reverse     = ~0.5 day to author + rehearse
-------------------------------------------------------------
Lead time for one rename: ~3 working days, 4 engineers involved
```

The same rename in a modular monolith: one PR, the compiler lists every call site, the tests run, one deploy. **Lead time: ~2 hours, one engineer.** That is roughly a 10–12× difference in the cost of the most ordinary change a team makes — a rename — and renames are *cheap* compared to splitting a field or changing a relationship. Multiply that delta across a year of changes and the distributed monolith is not a slightly worse architecture; it is a structural tax of an order of magnitude on the team's throughput. When leadership asks "why are we shipping so slowly," this worked example is the answer, and the deploy-coupling and co-change metrics from section 3 are the evidence.

## 8. Fixing it: the decoupling playbook

The fix is not a single move; it is a sequence, and the order matters because each step makes the next one possible. You re-draw boundaries first (so you know what each service *should* own), then give each service its data (so the schema coupling is severed), then break the shared code (so the library coupling is severed), then put anti-corruption layers at the seams (so leaks do not creep back in), then replace chatty synchronous calls with coarse APIs and events (so the temporal coupling is severed). The playbook is summarized in the figure below; the sub-sections give you the code for each layer.

![A vertical stack of the decoupling playbook from re-drawing boundaries around capabilities, to giving each service its own data with no cross reads, to splitting the god library into a published contract, to adding anti-corruption layers at the edges, to replacing chatty synchronous calls with coarse APIs and events](/imgs/blogs/shared-data-anti-patterns-and-the-distributed-monolith-8.webp)

### 8.1 Replace a shared-DB read with an API call or an event-fed read model

The most common single fix: order reads customer's tables directly, and you replace that read with either a synchronous call to customer's API or — better, for the hot path — a local read model that customer keeps up to date via events. The figure contrasts the two states.

![A side by side comparison of the order service running a SELECT directly against the customer schema which couples it and breaks on a rename, versus the order service calling the customer API or maintaining a local read model fed by customer events so the customer schema is free to change](/imgs/blogs/shared-data-anti-patterns-and-the-distributed-monolith-7.webp)

The before — the anti-pattern — looks like this:

```python
# ANTI-PATTERN: order service reaching directly into the customer schema.
# This is a hidden, contract-free dependency on customer's internal tables.
def get_order_view(order_id):
    return db.query("""
        SELECT o.id, o.total_cents,
               c.full_name, c.tier, c.is_blocked        -- customer's tables!
        FROM   orders o
        JOIN   customers c ON c.id = o.customer_id       -- cross-service JOIN
        WHERE  o.id = %s
    """, [order_id])
```

The synchronous fix asks customer for the *capability* it needs (the bits of the customer relevant to an order) through a versioned contract, so customer can change its schema freely behind that contract:

```python
# FIX (sync): call the customer service's published API for what we need.
# Order no longer knows or cares how customer stores its data.
def get_order_view(order_id):
    order = db.query("SELECT id, total_cents, customer_id FROM orders WHERE id=%s",
                     [order_id])[0]
    cust = customer_client.get_customer_summary(order["customer_id"])  # API v1 contract
    return {
        "id": order["id"],
        "total_cents": order["total_cents"],
        "customer_name": cust.display_name,   # order depends on the CONTRACT field,
        "tier": cust.tier,                     # not on the customer table layout
        "blocked": cust.is_blocked,
    }
```

The asynchronous fix — the right one when this read is on a hot path and you cannot afford the extra hop or the temporal coupling — has the order service keep a small **local read model** of just the customer fields it needs, kept current by subscribing to customer's events. This is the read side of [CQRS](/blog/software-development/message-queue/event-sourcing-and-cqrs-with-an-event-log) applied across a service boundary:

```python
# FIX (async): order maintains its own tiny read model of customer facts it needs.
# Fed by events; no cross-service call on the hot path, no temporal coupling.

# 1) Order subscribes to customer events and projects what it cares about.
def on_customer_event(evt):
    if evt.type in ("CustomerRegistered", "CustomerUpdated"):
        db.execute("""
            INSERT INTO customer_read_model (customer_id, display_name, tier, is_blocked, version)
            VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT (customer_id) DO UPDATE SET
              display_name=EXCLUDED.display_name, tier=EXCLUDED.tier,
              is_blocked=EXCLUDED.is_blocked, version=EXCLUDED.version
            WHERE customer_read_model.version < EXCLUDED.version   -- ignore stale/out-of-order
        """, [evt.customer_id, evt.display_name, evt.tier, evt.is_blocked, evt.version])

# 2) The hot path reads the LOCAL projection — order's own data now.
def get_order_view(order_id):
    return db.query("""
        SELECT o.id, o.total_cents, c.display_name, c.tier, c.is_blocked
        FROM   orders o
        JOIN   customer_read_model c ON c.customer_id = o.customer_id  -- LOCAL join, order owns both
        WHERE  o.id = %s
    """, [order_id])[0]
```

The crucial difference: in the async version, the JOIN is back — but it is a *local* JOIN against a table order owns, populated from a contract (the event schema), not a JOIN against customer's private tables. Order's hot path no longer makes a network call, no longer fails when customer is down, and customer is now free to restructure its own storage without ever breaking order. The cost is eventual consistency — order's read model lags customer's truth by the event-propagation delay, usually tens of milliseconds — which is exactly the trade-off covered in [data consistency and eventual consistency in practice](/blog/software-development/microservices/data-consistency-and-eventual-consistency-in-practice). For a customer's display name and tier, eventual consistency is obviously fine. You apply this fix where the staleness is acceptable, which for read-mostly reference data is almost everywhere.

### 8.2 Break the god library into a published contract plus a local model

The shared `common` library is severed by splitting it into two things: a *contract* package (the wire shapes — protobuf, JSON schema, OpenAPI — that genuinely must agree between services, versioned and published) and *local* domain models inside each service (the rich behavior, which each service owns its own copy of). The contract is shared because it must be; the models are duplicated because duplication is cheaper than coupling.

Before, every service shared a fat entity:

```python
# ANTI-PATTERN: common/models.py imported by ALL services.
# A change here recompiles and redeploys everyone, in lock step.
class Order:
    def __init__(self, id, customer_id, lines, total_cents, status, fraud_score,
                 shipping_method, promo_code, loyalty_points_earned, ...):
        ...   # 40 fields serving four different services' needs at once
    def apply_promo(self): ...          # only checkout cares
    def compute_fraud(self): ...        # only payment cares
    def estimate_delivery(self): ...    # only shipping cares
```

After, the shared part is a thin, versioned contract — here as protobuf, which gives you explicit, additive-only evolution:

```protobuf
// orders/contract/v1/order.proto  — PUBLISHED, versioned, shared.
// This is the ONLY thing services agree on. Additive changes are backward
// compatible; you never reuse a field number; you never repurpose a field.
syntax = "proto3";
package shopfast.orders.v1;

message OrderSummary {
  string order_id      = 1;
  string customer_id   = 2;
  int64  total_cents   = 3;
  Status status        = 4;
  enum Status { PENDING = 0; PAID = 1; SHIPPED = 2; CANCELLED = 3; }
  // new fields go here with the NEXT number; never renumber, never reuse.
}
```

And each service keeps its own rich local model with only the behavior it needs:

```python
# orders service: its OWN model, owns the fields IT cares about.
# Changing this redeploys ONLY the order service.
class Order:
    def __init__(self, id, customer_id, lines, total_cents, status, promo_code=None):
        ...
    def apply_promo(self): ...   # checkout's concern lives here, nowhere else

    def to_summary(self) -> "shopfast.orders.v1.OrderSummary":
        return OrderSummary(order_id=self.id, customer_id=self.customer_id,
                            total_cents=self.total_cents, status=self.status)
```

Now a change to fraud scoring touches only payment, a change to delivery estimation touches only shipping, and a change to the *contract* is a deliberate, versioned event governed by [consumer-driven contract testing](/blog/software-development/microservices/api-versioning-and-consumer-driven-contract-testing) — additive, backward compatible, and verified by tests before it ships, instead of an implicit recompile-everyone landmine. The little bit of duplicated `Order` definition across services is not technical debt; it is the *decoupling*, and it is what lets the services move independently.

### 8.3 Collapse the chatty N-call loop into one coarse call

Chatty synchronous communication often hides in a loop: a service fetches a list of IDs, then calls another service once per ID. This is the N+1 query problem reborn across the network, and it is brutal — N network round-trips where one would do. The fix is a coarse, batch-oriented endpoint.

```python
# ANTI-PATTERN: N network calls in a loop (N+1 across the network).
def enrich_orders(orders):
    for o in orders:                                   # 200 orders ->
        o.customer = customer_client.get(o.customer_id) # -> 200 network calls
    return orders
```

```python
# FIX: one coarse batch call. 200 round-trips collapse to 1.
def enrich_orders(orders):
    ids = list({o.customer_id for o in orders})        # de-dupe ids
    summaries = customer_client.get_many(ids)          # ONE call, returns a map
    by_id = {s.customer_id: s for s in summaries}
    for o in orders:
        o.customer = by_id.get(o.customer_id)
    return orders
```

```protobuf
// customer/contract/v1/customer.proto — the coarse, batch-friendly API.
service Customer {
  // Coarse: one call for many ids. Designed for the caller's real access pattern,
  // not a thin CRUD wrapper around one row.
  rpc GetManySummaries(GetManyRequest) returns (GetManyResponse);
}
message GetManyRequest  { repeated string customer_ids = 1; }
message GetManyResponse { repeated CustomerSummary summaries = 1; }
```

The lesson generalizes: **design service APIs around the caller's access pattern, coarse-grained, not as thin CRUD wrappers around individual rows.** A fine-grained `getCustomer(id)` invites the chatty loop; a coarse `getManySummaries(ids)` makes the efficient call the easy one. This is one of the few places where "make the API a little less generic" is unambiguously the right call.

#### Worked example: the N+1 fix, measured

Take an order-history page that lists 200 orders and needs each order's customer summary. Each cross-service call has a p50 of 8 ms and a p99 of 40 ms.

```
Chatty (N+1):
  200 sequential calls × 8 ms p50          = 1,600 ms p50   (page is unusable)
  even fully parallelized, you pay 200×    connection setup + the worst tail
  p99 of "all 200 succeed fast"            ≈ dominated by the slowest of 200 calls
                                           ≈ well past 40 ms, often 100+ ms,
                                             plus 200× the downstream load

Coarse (1 batch call of 200 ids):
  1 call, p50 12 ms / p99 55 ms            (slightly higher per-call, but ONE call)
  downstream load: 1 request instead of 200 (200× fewer)
```

The page goes from ~1.6 seconds to ~12 ms at p50, and — the part the customer service's on-call cares about — the downstream load drops by 200×. Chatty-to-coarse is frequently the single highest-leverage performance fix in a distributed monolith, and it costs one batch endpoint.

### 8.4 The anti-corruption layer: keeping the fix from leaking back

When you decouple, a danger is that the *other* service's model leaks into yours through the new API — you replace a SQL JOIN with an API call, but you let customer's exact data shapes and naming flow straight into your domain, and you have just recreated the coupling one layer up. The **anti-corruption layer (ACL)** is a deliberate translation boundary: an adapter at the edge of your service that converts the *foreign* model into *your* model, so nothing of the other service's vocabulary penetrates past the adapter.

```python
# Anti-corruption layer: translate the CUSTOMER service's model into the
# ORDER service's vocabulary. Nothing past this adapter knows customer's shapes.
from dataclasses import dataclass

@dataclass(frozen=True)
class Buyer:                      # ORDER's term, ORDER's fields. Our domain language.
    id: str
    name: str
    is_eligible_for_checkout: bool

class CustomerACL:
    def __init__(self, customer_client):
        self._client = customer_client

    def fetch_buyer(self, customer_id: str) -> Buyer:
        # Foreign model crosses the wire...
        c = self._client.get_customer_summary(customer_id)   # customer.v1 shapes
        # ...and is translated into OUR model right here, at the edge.
        return Buyer(
            id=c.customer_id,
            name=c.display_name,
            # WE decide what "eligible" means in order's terms, from customer's facts.
            is_eligible_for_checkout=(not c.is_blocked) and c.tier != "FRAUD_HOLD",
        )
```

Now the rest of the order service speaks only `Buyer`, in its own language. If customer renames `display_name`, or replaces `is_blocked` with a richer status enum, or splits the summary into two calls, the blast radius is one method — `CustomerACL.fetch_buyer` — and nothing else in order changes. The ACL is the firewall that keeps the decoupling decoupled, and it is the pattern you put at every seam where you depend on a service you do not control (a legacy system, a third party, or another team moving at a different pace). It is the controlled boundary that the [strangler-fig migration](/blog/software-development/microservices/strangler-fig-migrating-a-monolith-to-microservices) leans on heavily when carving services out of a monolith.

The discipline an ACL enforces is worth naming explicitly, because it runs against the grain of how most code gets written. The natural, lazy thing is to let the deserialized response object from the customer API flow directly through your service — you call `customer_client.get_customer_summary()` and pass the result around, reading `.display_name` and `.is_blocked` wherever you need them. That feels efficient (why translate?), and it is exactly how the coupling creeps back. The moment fifty call sites in order read `.display_name` off a customer object, customer's field name has become part of order's code in fifty places, and you are one rename away from the distributed monolith again — you've just moved the coupling from the database layer to the object layer. The ACL forbids this by making the foreign object *physically unable* to escape past the adapter: the adapter is the only code that knows customer's shapes, it returns your own `Buyer` type, and a code-review rule (or, better, a build rule that bans importing the customer client outside the ACL package) keeps it that way. The cost is the boilerplate of one translation method per dependency; the benefit is that every future change in a service you don't own has a blast radius of exactly one adapter. For a dependency you genuinely do not control — a payment provider, an old mainframe, another company's API — skipping the ACL is not a shortcut, it is a decision to let their roadmap dictate your refactors.

## 9. Optimization: measuring the decoupling win

Decoupling is not free, and a senior justifies it with numbers, before and after, on the metrics that matter. Here is what to measure and what good looks like.

**Deploy independence (the headline metric).** Track the deploy-coupling ratio from section 3.1 release over release. ShopFast went from `4.6` co-deployed services per release to `0.4` over a quarter. The single clearest sign you have escaped the distributed monolith is that this number approaches zero — services start shipping alone.

**Change locality.** Track the co-change percentage from section 3.2. ShopFast's "commits touching ≥3 services" went from 70% to 11% as the shared schema and library were dismantled. The remaining 11% are mostly genuine cross-cutting changes (a new auth scheme) plus the irreducible tail; that is healthy.

**Lead time for change.** This is the business-visible payoff. DORA's research frames lead time (commit to production) as a core delivery metric; coordinated releases inflate it brutally. ShopFast's median lead time for a change fell from 3 days (the coordinated-release worked example) to under 4 hours once most changes touched one service. That is the number leadership actually feels.

**Checkout latency and availability.** From the fixes in section 6 and 8.3: p99 of Place Order dropped from ~320 ms to ~130 ms by moving shipping/notification off the synchronous path and collapsing the N+1 enrichment. More importantly, the availability stopped being a product of eight terms — the critical path now has two synchronous dependencies (order's own DB and the payment PSP) instead of seven, so a customer or shipping outage degrades gracefully (the event queues) instead of failing the checkout.

**Cost.** Fewer synchronous hops means fewer idle threads blocked waiting on downstream calls, which means smaller pools and fewer instances. ShopFast cut the order service's instance count by a third after the chatty enrichment was batched, because each instance stopped tying up threads in 200-call loops.

#### Worked example: stress-testing the decoupled design

The senior discipline is to stress-test the fix, not just admire it. Pose the questions and answer them honestly against the new design (figure below shows the target topology):

![A graph of the decoupled checkout where the gateway calls the order service which makes one blocking payment call and emits an OrderPlaced event that drives shipping and notification asynchronously off the critical path](/imgs/blogs/shared-data-anti-patterns-and-the-distributed-monolith-9.webp)

- **What breaks at 10× traffic?** The synchronous path is now order's DB plus the payment PSP. Order's DB scales with read replicas and the local read models (no cross-service reads to amplify). The PSP is the real bottleneck and the external rate limit; you handle it with a bounded concurrency limit and a queue, not by adding hops. Shipping and notification absorb 10× as queue depth, not as failed requests — backpressure, covered in [backpressure and flow control](/blog/software-development/message-queue/backpressure-and-flow-control), turns a spike into latency rather than errors.
- **What happens when shipping is down?** Nothing, to the customer. `OrderPlaced` events queue; shipping processes them when it recovers. Before, shipping being down failed checkout. This is the entire point of severing the temporal coupling.
- **What happens when customer is down?** Order reads its local read model, so reads keep working with possibly-stale data. New customer registrations can't be reflected until customer recovers, but existing checkouts proceed. Before, order was dead the instant customer was.
- **What happens during a network partition?** The async paths buffer; the one synchronous external call (payment) fails fast with a timeout and circuit breaker (see [resilience patterns](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads) in the next track) and the customer gets a clean "try again," not a 30-second hang.
- **What happens when order is deployed mid-request?** Because order no longer shares a schema or library version with anyone, a rolling deploy of order does not require any other service to be at a matching version. In-flight requests drain; new ones hit the new pods. Before, a mid-request deploy could mismatch the shared `common` version across services and produce serialization errors.

The design survives the stress test on every axis where the distributed monolith failed it — which is the whole justification for the work.

## 10. The migration order: how to dismantle it without a big bang

You do not fix a distributed monolith in one heroic refactor; you carve it apart one seam at a time, lowest-risk seam first, with the metrics from section 3 telling you whether each move helped. The lock-step release in the figure below is what you are escaping — and it is also a warning about the *wrong* way to migrate (a big-bang coordinated cutover is just one more giant lock-step release).

![A timeline of a lock-step release derailing where a shared database column rename forces six repositories to change together, deploy in a fixed order, then one service fails its health check forcing a reverse-order rollback of all six and a three hour incident](/imgs/blogs/shared-data-anti-patterns-and-the-distributed-monolith-6.webp)

The sequence that works:

1. **Measure first.** Run the deploy-coupling, co-change, and dependency-graph scripts. Find the worst seam (highest co-change pair, the cycle, the god service). For ShopFast that was order↔customer.
2. **Lock the schema down logically.** Even before splitting databases, give each service its own schema and revoke cross-schema grants — one DB role per service, granted only its own schema. The instant a cross-service read fails with a permission error in staging, you have *found* every hidden table dependency, mechanically, instead of hoping you grepped them all. This is the first concrete step of the [database-per-service](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices) migration.
3. **Replace the worst cross-service reads** with API calls or event-fed read models (section 8.1), breaking the cycle first — that is the move that most increases independence.
4. **Split the god library** into a published contract plus local models (section 8.2), so a model change stops recompiling everyone.
5. **Move chatty synchronous chains** to coarse calls and events (sections 8.3, 6), severing the temporal coupling and getting the latency/availability win.
6. **Add ACLs at the seams** (section 8.4) so the coupling does not creep back as the foreign model leaks in.
7. **Re-measure.** Each step should move the deploy-coupling number down. If it doesn't, you fixed a symptom, not a cause — go back to boundaries.

This is precisely the incremental, evidence-driven approach of the [strangler fig](/blog/software-development/microservices/strangler-fig-migrating-a-monolith-to-microservices): carve one seam, verify it, carve the next, never a big-bang cutover. And the genuinely senior move, made before any of this, is to ask whether the destination is microservices at all — which section 12 confronts head on.

## 11. Case studies

### Segment: the round trip from monolith to microservices and back

Segment is the canonical, candid distributed-monolith story, told in their engineering writing and conference talks around 2018. They had moved from a monolith to microservices to scale their data-ingestion pipeline, and ended up with a per-destination-queue architecture that grew to *over a hundred* services and a shared codebase of shared libraries. The shared libraries were the trap: a change to a common library had to be rolled out across every service, defeating the independence microservices were supposed to provide, and the operational burden of patching and deploying that many services for one shared change became crushing — their words, roughly, were that the developer productivity benefits of microservices had reversed into a tax. Their fix was not "do microservices harder." They *consolidated* the destination services back into a single monolithic service (they called the effort "Centrifuge," and the consolidated service "monoservice"), trading some isolation for an enormous reduction in operational and shared-library overhead. The lesson Segment teaches is exactly this post's thesis: a fleet of services coupled by shared libraries is a distributed monolith, and the right answer can be to go *back*, not forward. They reduced the maintenance burden dramatically by collapsing the topology when the coupling made the split a net negative.

### The shared-database lock-step release horror

Across many organizations the same story recurs and shows up in incident write-ups: several services share one database, a "simple" schema migration (rename a column, change a type, add a constraint) is required, and because every service reads or writes the affected table, the migration becomes a multi-service coordinated release. The well-run version uses the **expand/contract** (parallel-change) pattern — add the new column, dual-write, migrate readers, drop the old column, across several releases — but even that requires every service to move through the phases in lock step. The badly-run version skips the dance, deploys in a fixed order, and discovers mid-rollout that service four fails its health check on the new schema, forcing a reverse-order rollback of everything (the timeline figure above). The lesson is structural, not procedural: no migration discipline removes the coupling; only *ownership* does. When one service owns the table and exposes a contract, a schema change is that service's private business, and the lock-step release simply ceases to exist.

### Uber's DOMA: re-drawing boundaries after the sprawl

Uber's microservices count grew into the thousands, and by their own account the sprawl produced exactly the coupling this post describes — dependencies that crossed everywhere, changes that rippled, hard-to-reason-about call graphs. Their response, the Domain-Oriented Microservice Architecture (DOMA), was not to abandon microservices but to impose *capability* boundaries on top of them: group services into **domains** (collections of related services), define explicit **layers** with a strict dependency direction (higher layers may depend on lower, never the reverse — eliminating cycles by construction), and require all cross-domain access to go through a domain **gateway** with an explicit interface (a system-level anti-corruption layer). DOMA is, in effect, the section-8 playbook applied at organizational scale: re-draw around capabilities, force a DAG, and translate at the seams. The lesson it teaches is that the cure for a distributed monolith is almost always *boundaries and ownership*, not more services — and that the boundary discipline of [Conway's law and team topologies](/blog/software-development/microservices/conways-law-and-team-topologies-for-microservices) has to back it, because boundaries that don't match team ownership erode.

### Amazon's two-pizza teams: the org structure that prevents it

The counter-example worth keeping in mind is Amazon, whose service-oriented architecture has held up across decades, and a large part of why is organizational. The famous "two-pizza team" rule — a team small enough to feed with two pizzas — pairs each service (or small set of services) with a team that owns it end to end, including its data and its operations. A team owning its service's data is *structurally prevented* from the shared-database anti-pattern, because there is no other team whose database it could share; and a team responsible for its own service's uptime is incentivized to sever temporal coupling, because they get paged when a downstream they depend on falls over. Amazon teaches that the distributed monolith is as much an organizational failure (teams sharing data and code because the org didn't draw ownership lines) as a technical one, and that the durable prevention is ownership, not tooling.

## 12. When to reach for this — and when to stop and go back

This section is the decisive one, and the recommendation is unusually clear because the distributed monolith is unusually bad.

**If you have a distributed monolith, stop adding services.** More services do not dilute the coupling; they spread it. Every new service that reads the shared database or imports the god library makes the train longer. The first move is always to *measure* (section 3) and then *decouple the worst seam* (section 8), not to decompose further.

**Decouple into true microservices only if you have the justifying conditions.** Real microservices earn their operational cost when you have *multiple teams* that need to ship independently, *divergent scaling needs* across parts of the system, or a *blast-radius* requirement (one part must keep running when another fails). If you have those — many teams, very different load profiles, hard isolation needs — then do the section-8 work fully: data per service, contracts not shared libraries, async where it fits, ACLs at the seams. The cost is real and you are choosing to pay it for benefits you can name.

**Otherwise — and this is most teams — collapse back to a modular monolith.** If you have one or two teams, uniform-ish scaling, and no hard isolation requirement, then the honest answer is that you should never have split, and the fix is to *re-merge* the services into a single deployable [modular monolith](/blog/software-development/microservices/monolith-first-and-the-modular-monolith): keep the clean module boundaries you have (or finally draw them), but ship one artifact, make the calls in-process, and delete the network, the seven dashboards, and the release train. You keep all the structural benefits — module ownership, separate schemas inside one DB, build-enforced dependency rules — and shed all the distribution cost. This is exactly the trade-off [Segment made](#case-studies), and it is the right call far more often than microservices culture admits. The full treatment of when going back is correct lives in the dedicated sibling, [microservices anti-patterns and when to go back to monolith](/blog/software-development/microservices/microservices-anti-patterns-and-when-to-go-back-to-monolith).

The decision rule, compressed: a distributed monolith is dominated by *both* of its neighbors, so any move off it is an improvement, and the only mistake is to stay. Pick the neighbor your team count and scale justify, and go there deliberately.

## 13. Key takeaways

- **A distributed monolith is services that can't deploy independently.** Independent deployability is the litmus test. Separate repos and containers prove nothing; the deploy log does.
- **It is the only dominated architecture.** It pays the full operational cost of distribution while keeping the monolith's coupling, so a modular monolith beats it on every axis. There is no scenario where it is the right answer.
- **Measure the coupling; don't argue about it.** The deploy-coupling ratio, the co-change percentage (70% of commits touching ≥3 services is a five-alarm fire), and dependency-graph cycles turn a vibe into a number leadership can watch fall.
- **The three roots are shared data, shared code, and synchronous coupling.** A shared database, reaching into others' tables/caches, and a god `common` library produce most of it; an N-hop sync chain produces the rest.
- **Share capabilities, not data.** Ask a neighbor for a behavior behind a versioned contract, not for its tables. Replace cross-service reads with API calls or event-fed local read models.
- **A little duplication beats shared coupling.** Break the god library into a published contract plus local models. Duplicating a thin `Order` definition across services is the decoupling, not debt.
- **Coarse APIs beat chatty ones.** Collapse N+1 cross-service loops into one batch call designed for the caller's access pattern; it's often the highest-leverage performance fix you have.
- **Sever temporal coupling with events and ACLs.** Move everything that can wait off the synchronous critical path; translate foreign models at the edge so the coupling can't creep back.
- **Migrate one seam at a time, lowest risk first, and re-measure.** Big-bang cutover is just one more lock-step release. The strangler approach, with the deploy-coupling number as your scoreboard, is the safe path.
- **The senior question is "should this be microservices at all?"** If your team count and scale don't justify the cost, the right fix for a distributed monolith is to go *back* to a modular monolith, not forward to more services.

## 14. Further reading

- Sam Newman, *Building Microservices* (2nd ed., O'Reilly) — chapters on coupling, the shared database as the cardinal anti-pattern, and independent deployability as the defining property.
- Chris Richardson, *Microservices Patterns* (Manning) — the database-per-service pattern, API composition, the saga pattern, and the anti-patterns of shared persistence.
- Segment Engineering, "Goodbye Microservices: From 100s of problem children to 1 superstar" — the candid account of consolidating a service sprawl coupled by shared libraries back into a monoservice.
- Uber Engineering, "Microservice Architecture at Uber" / the DOMA write-up — domains, layered dependencies, and gateways as the cure for microservice sprawl.
- Martin Fowler, "MonolithFirst" and "MicroservicePremium" — why the monolith is the senior default and microservices are a premium you pay for specific benefits.
- [Database per service: the rule that defines microservices](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices) — the positive statement of the rule this post is about violating.
- [Monolith first and the modular monolith](/blog/software-development/microservices/monolith-first-and-the-modular-monolith) — the architecture a distributed monolith should have been.
- [Inter-service communication fundamentals and fallacies](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies) — why the network is not reliable, free, or zero-latency, which is what the chatty chain forgets.
- [Service boundaries with domain-driven design](/blog/software-development/microservices/service-boundaries-with-domain-driven-design) — how to draw boundaries around capabilities so they don't ripple.
- [Strangler fig: migrating a monolith to microservices](/blog/software-development/microservices/strangler-fig-migrating-a-monolith-to-microservices) — the incremental, one-seam-at-a-time migration the fix relies on.
