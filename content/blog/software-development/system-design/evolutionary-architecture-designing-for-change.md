---
title: "Evolutionary Architecture: Designing for Change, Not for a Whiteboard"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How seniors design systems that survive contact with growth: reversible vs irreversible decisions, monolith-first, the strangler fig, fitness functions, and why data is the only thing that is truly hard to change."
tags:
  [
    "system-design",
    "evolutionary-architecture",
    "monolith-first",
    "architecture",
    "distributed-systems",
    "scalability",
    "migration",
    "microservices",
    "adr",
    "refactoring",
  ]
category: "software-development"
subcategory: "System Design"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/evolutionary-architecture-designing-for-change-1.webp"
---

The most useful thing a senior engineer can tell a room full of people designing a new system is also the most uncomfortable: **the architecture you are about to draw is wrong.** Not slightly wrong, not wrong-in-a-detail wrong — structurally wrong for the system you will actually be running in eighteen months. You do not yet know the access patterns, the hot keys, the regulatory surprise, the acquisition that doubles your write volume overnight, or the one feature that turns out to be the entire business. The whiteboard captures your best guess given today's information, and today's information is the least information you will ever have about this system.

This is not a counsel of despair. It is the central insight of **evolutionary architecture**: since the first design is always wrong, the job is not to get the design right — it is to design so that being wrong is cheap to correct. A senior does not optimize for the elegance of the diagram. They optimize for the *cost of the next change*, because there is always a next change, and the systems that die are the ones where every change is expensive.

![A reading-list application changing shape as it grows from a single monolith at one thousand users to multi-region async pipelines at ten million users](/imgs/blogs/evolutionary-architecture-designing-for-change-1.webp)

The figure above is the whole article in one picture. It is a single application — a reading-list app, the running example we will evolve through this entire piece — observed at five points across four orders of magnitude. At 1,000 users it is one process and one database. At 10 million it is a multi-region, sharded, asynchronous fleet. The point is not that you should plan for the right edge of that timeline on day one. The point is the *opposite*: each stage is the correct architecture for its load, and trying to build the 10-million-user shape when you have 1,000 users is one of the most reliable ways to die before you get there. By the end of this article you will be able to look at a design decision and answer three questions a senior answers reflexively: how reversible is this, how likely is it to change, and where do I put the seam so that when it does change, the change stays local.

## 1. The first architecture is always wrong — and that's fine

Let us be precise about why the first architecture is wrong, because "you can't predict the future" is too glib to act on. There are three specific reasons, and each one points at a design response.

**You don't know the access pattern yet.** Almost every meaningful architectural decision is downstream of how data is read and written: the read-to-write ratio, whether reads are point lookups or range scans, whether writes are append-heavy or update-heavy, what the hot keys are. You will guess these on day one, and you will be wrong, because the access pattern is an emergent property of how real users behave, and you have no real users yet. The reading-list app I will keep returning to looked, on the whiteboard, like a balanced read/write workload. In production it turned out to be 200:1 read-heavy with a long tail of users who never log in again after week one — a profile that completely changes what you cache, what you replicate, and what you can afford to compute lazily.

**You don't know which axis will grow.** Systems grow along one axis at a time, and you rarely know which. Will it be users, data per user, requests per user, geographic spread, team size, or regulatory surface? Each axis stresses a different part of the design. A system that scales beautifully to 100× the users can fall over at 10× the data-per-user. You cannot pre-harden every axis; pre-hardening all of them is how you get a system too complex to ship.

**You don't know what the business will become.** The feature you built as an afterthought becomes the product. The "temporary" CSV export becomes a contractual API with paying customers. Slack was a feature inside a failed game. The reading-list app's "share a list" button — bolted on in an afternoon — became the viral loop that drove 90% of growth, which meant the sharing path, not the reading path, became the one that needed to scale first.

Given all three, the senior move is not to predict better. It is to **make the design cheap to change along the axes that are likely to move, and deliberately not along the axes that almost never move.** Most systems, over their life, change their data model and their scaling story many times. They rarely change their programming language, their cloud provider, or their fundamental domain. So you buy flexibility where change is likely and you spend nothing buying flexibility where it isn't. Optimize for the change you will actually make.

There's a corollary that catches good engineers: being wrong is not failure. A design that served you well at 1,000 users and now needs replacing at 1 million did its job. The mistake is not "we chose Postgres and now we need to shard." The mistake is "we built a service mesh and an event-sourced CQRS pipeline for 1,000 users because we read a blog post about Netflix." The first is evolution working as intended. The second is paying the cost of a scale you do not have.

## 2. The cost-of-change curve: why timing dominates

The reason "design for change" is not just a platitude is that the cost of changing a decision is not constant over time. It grows, and it grows steeply, and the shape of that growth is the single most important thing to internalize about software architecture.

![A decision branching into rising reversal costs from roughly one times at design time to ten times in code to a hundred times in data to a thousand times once it reaches the public API](/imgs/blogs/evolutionary-architecture-designing-for-change-2.webp)

The figure shows a decision and the cost of reversing it at four moments in its life. At design time, reversing a decision costs roughly nothing — you erase the box and draw a different one. Once it is in code, reversing it costs maybe 10× that: you have written and tested logic that assumes the decision. Once the decision is reflected in *data* — a column exists, rows are populated, indexes are built — reversal costs another order of magnitude, because now you have to migrate live data without downtime while the system keeps serving traffic. And once the decision has leaked into a *public contract* — an API your customers integrate against, an event schema other teams consume, a URL structure search engines have indexed — reversal costs 100× to 1000× more, because you no longer control all the code that depends on it.

This curve has a direct, actionable consequence: **push hard-to-reverse decisions as late as you responsibly can, and keep them out of the layers where reversal is expensive for as long as possible.** This is the opposite of "decide everything up front." It is decide-the-easy-things-now, defer-the-hard-things-until-you-have-evidence. The architectural skill is knowing which is which.

The curve also explains why the same change can be a five-minute task or a six-month program depending purely on *when* you do it. Changing your primary key from an auto-increment integer to a UUID is a one-line schema edit before launch. After three years and 400 million rows referenced by foreign keys across forty tables and cached in a dozen services and embedded in customer-facing URLs, it is a quarter's worth of work for a team. Same decision, same desired end state. The only variable that moved was time, and time moved the cost by four orders of magnitude.

#### Worked example: dating the cost of a key change

Concretely: the reading-list app stored each list with an integer primary key, `list_id BIGINT`, exposed in URLs as `/list/48273`. Two years in, the security team flagged that sequential IDs leak business metrics — a competitor can watch the IDs climb and estimate our growth rate — and that they enable enumeration attacks. The desired change: switch to an opaque, non-sequential identifier.

Cost at design time (year 0): pick `UUID` or a random 64-bit ID instead of `SERIAL`. Roughly **15 minutes** of design discussion, zero migration.

Cost at year 2: we had 12 million lists, 180 million `list_items` rows with a `list_id` foreign key, three internal services that cached `list_id`, a public API where `list_id` appeared in 8 endpoints, and — the killer — search-indexed URLs. The actual program: add a new `public_id` column (expand), dual-write it on every create, backfill 12 million rows in batches, migrate the API to accept both forms with a deprecation window, update foreign-key references behind the application layer, and keep 301-redirecting the old URLs for SEO indefinitely. Estimated and actual: **roughly one engineer-quarter**, spread over four months to avoid downtime.

The cost multiplier from doing it late versus early was on the order of **300×**. The lesson is not "always use UUIDs." It is that the *category* of decision — an identifier baked into data and public contracts — is exactly the kind the cost curve punishes most for deferring. We will return to this exact migration when we discuss the expand-contract pattern in §9, because the *technique* that made the year-2 version survivable is the same one you should have in your pocket for every data change.

## 3. One-way doors and two-way doors

Jeff Bezos gave the industry the most useful decision taxonomy it has, and it maps perfectly onto the cost curve. Some decisions are **two-way doors**: you walk through, and if you don't like what's on the other side, you walk back out at low cost. Other decisions are **one-way doors**: once you walk through, the door locks behind you, and getting back costs a fortune. The entire art of decision *velocity* is treating these two categories completely differently.

![A decision matrix classifying choices like web framework and internal module boundary as fast two-way doors versus datastore and partition key as deliberate one-way doors](/imgs/blogs/evolutionary-architecture-designing-for-change-3.webp)

The matrix above audits common architectural decisions. The rule it encodes is brutal and freeing: **make two-way-door decisions fast and reversibly; make one-way-door decisions slowly and deliberately.** Most teams get this exactly backwards. They spend three weeks in committee choosing a web framework (a two-way door — you can rewrite the HTTP layer in a sprint if you must) and then casually pick a partition key in a standup (a one-way door that, once you've sharded 50 TB across it, you cannot change without a multi-month resharding project).

The categories that tend to be one-way doors are worth memorizing, because they are where you should spend your deliberation budget:

- **Your primary datastore's data model**, especially the partition/shard key. Once data is laid out, relaying it out is a migration project. This is why the choice of datastore deserves real analysis — see the companion piece on [choosing a datastore across SQL, NoSQL, and NewSQL](/blog/software-development/system-design/choosing-a-datastore-sql-nosql-newsql) — and why [partitioning and sharding without downtime](/blog/software-development/system-design/partitioning-and-sharding-without-downtime) is its own deep discipline.
- **Public API contracts and event schemas.** The moment a third party (or another team) depends on your shape, you've handed them a veto over changing it. See [API design across REST, gRPC, and GraphQL](/blog/software-development/system-design/api-design-rest-grpc-graphql) for how to design contracts that can evolve.
- **The split from monolith to services**, and the boundaries you choose for that split. Merging two services back together is far harder than splitting one, because by the time you want to merge, two teams own the two halves.
- **Identifiers, encodings, and anything you persist or expose.** URLs, IDs, timestamp formats, currency representations.

The categories that are almost always two-way doors — and therefore deserve a *fast* decision, made by one person if necessary:

- Web framework, ORM, logging library, internal RPC mechanism between your own services.
- Internal module boundaries within a monolith (you can move code between modules in an afternoon).
- Caching layer, CDN, observability vendor.
- Most internal naming and structure.

There is a subtle trap here that separates the senior from the merely experienced: **some doors look two-way but are actually one-way because of how the decision propagates.** Choosing an ORM looks reversible — until you've written 80,000 lines of code coupled to its query API and its migration tooling and its lazy-loading semantics. The decision was reversible; the *coupling you allowed to grow around it* made it irreversible. This is why seams matter (§6): a two-way door stays two-way only if you keep the blast radius of the thing behind it small.

The practical protocol: for every significant decision, write down which kind of door it is *before* you decide how much to deliberate. If it's two-way, set a timebox — "we decide by Friday, and if we're wrong we'll know in a month and fix it." If it's one-way, slow down, write the decision down (an ADR — §7), and stress-test it against the failure modes before you commit. You will ship faster overall, because you'll stop wasting deliberation on reversible things and stop rushing the irreversible ones.

## 4. Build for 10×, not 1000×: premature scaling is premature optimization

There is a famous and correct warning against premature *optimization* — don't hand-tune the inner loop before you've profiled. The architectural analogue is **premature scaling**, and it is at least as expensive, because the cost is not a few wasted hours but a system shape that fights you for years.

The rule I give every team: **design for roughly 10× your current load, not 1000×.** Ten times is close enough that you can reason about it concretely — you know what doubles, what caches stop fitting, what query gets slow — and far enough that you won't be re-architecting every quarter. A thousand times is a different system entirely, and building it now means paying for complexity you cannot yet use, against requirements you cannot yet know.

![A before-and-after comparison contrasting eight premature microservices run by one team against a single modular monolith with in-process calls and local transactions](/imgs/blogs/evolutionary-architecture-designing-for-change-4.webp)

The figure shows the most common form of premature scaling: splitting into microservices before you have either the scale or the organization that justifies them. The "before" side is a system I have personally been called in to rescue more than once — eight services, one team of six, a network hop on every interaction that used to be a function call, and a distributed transaction problem that did not exist when it was all one process. The "after" side is the same domain expressed as a modular monolith: one deployable, clear internal module boundaries, in-process calls measured in microseconds, and transactions that are just database transactions.

Here is the math that makes premature microservices a bad trade. In a monolith, a call between two modules is a function call: roughly **1 microsecond**, and it cannot partially fail. The same call between two services is a network round trip: **1 to 5 milliseconds** in the same datacenter on a good day — a **1,000× to 5,000× latency increase** — and it *can* partially fail, which means you now need timeouts, retries, circuit breakers, and idempotency on a call that used to be a guaranteed, instantaneous, transactional thing. You've also turned one transaction into a distributed one, which means you've traded ACID guarantees for a [saga pattern](/blog/software-development/database/saga-pattern-distributed-transactions) and the eventual-consistency reasoning it demands. None of that complexity buys you anything until you have a scaling or organizational reason that a monolith genuinely cannot meet.

#### Worked example: the cost of 1000× thinking at 1× scale

A team building the reading-list app's recommendation feature decided, at launch, to build it as a separate "recommendation service" with its own database, its own deployment, and an event stream feeding it user activity — because "that's how the big companies do it." At launch they had 4,000 daily active users and were generating about **2 recommendations per second** at peak.

What that 1000× architecture cost them, measured concretely:

- **Latency:** a recommendation that could have been a 3 ms in-process query became a 40 ms cross-service call with a cache miss, because it now crossed a network boundary, deserialized an event, and hit a separate database. Their p99 on the home page went from 60 ms to 140 ms.
- **Reliability:** they added a new failure mode — the recommendation service being down or slow now degraded the home page — and had to build a fallback path, a circuit breaker, and a "recommendations unavailable" UI state. None of that existed when it was a function call returning a list.
- **Operational load:** a second database to back up, monitor, patch, and pay for. A second deployment pipeline. An event stream to operate. Roughly **\$1,400/month** of infrastructure and, far more expensively, a meaningful fraction of one engineer's attention, for a feature serving 2 QPS.
- **Velocity:** every change to recommendation logic now touched two repos, two deploys, and a schema shared across a network boundary that had to be versioned. A one-day change became a one-week change.

The correct architecture at 4,000 DAU was a module inside the monolith: a `recommendations` package with a clean interface, computing suggestions in-process from the same database, behind a function call. When they later hit 400,000 DAU and recommendation computation genuinely became expensive, *then* extracting it into a service was justified — and because they'd kept it behind a clean interface (a seam), the extraction was a two-week project, not a rewrite. They built for 10×, kept the seam, and paid the 1000× cost only when they reached 1000× the load.

## 5. YAGNI vs genuinely load-bearing flexibility

"You Aren't Gonna Need It" is the most quoted and most misapplied principle in software. Taken literally it says: build nothing speculative, add flexibility only when the need is concrete. Taken too literally it produces brittle systems where every change requires surgery. The senior version is more nuanced: **YAGNI applies to features and to speculative generality; it does not apply to the seams that make likely changes cheap.** The art is telling the two apart.

![A matrix judging flexibility investments such as a pluggable datastore port and feature flags as worth building versus a generic config engine and custom plugin framework as speculative generality](/imgs/blogs/evolutionary-architecture-designing-for-change-7.webp)

The matrix gives the test. For any proposed bit of flexibility, ask two questions: **Is the change it enables actually likely?** and **If we don't build the seam now, how expensive is it to retrofit later?** Flexibility earns its place only in the top-right quadrant — likely change, expensive retrofit. Everywhere else it is cost without benefit.

The classic *good* investments — flexibility that is load-bearing:

- **A storage port** (an interface between your domain logic and your datastore) is almost always worth it, because changing or adding a datastore is a likely change and retrofitting the abstraction after you've scattered SQL across your codebase is brutal. This is a top-right call.
- **A feature-flag system** is load-bearing because you *will* want to ship dark, roll out gradually, and kill a bad feature without a deploy. The change (toggling behavior) is constant, and bolting flags onto a flag-naive codebase is painful.
- **A versioned API surface** from the first public release. Once you have external consumers, you will need to evolve the contract, and versioning retrofitted after launch means a painful migration for everyone who integrated.

The classic *bad* investments — speculative generality that rots:

- **A "fully configurable, data-driven engine"** that can supposedly handle any future business rule via configuration. This is the single most reliable way to build a system nobody can understand. You've reinvented a programming language, badly, to avoid writing code you don't yet need. The change it anticipates (arbitrary future rules) is too vague to design for, so you over-design for all of them.
- **A custom plugin framework** before you have a single second plugin. You are designing an extension point against imagined extensions. When the real second use case arrives, it never fits the framework you guessed at.
- **Multi-tenancy from day one** when you have one tenant and no signed second customer. The *seam* (a `tenant_id` in your data model and a tenant context in requests) might be worth it because retrofitting it is genuinely expensive — but the full multi-tenant control plane, per-tenant isolation, and billing integration are pure YAGNI until you have tenants.

Notice the pattern in the good ones: they are all **seams**, not **machinery**. A seam is a thin interface that *localizes* a future change. Machinery is a heavyweight subsystem that *implements* a future capability. Seams are cheap to add now and cheap to keep; they cost a few hours of interface design and a small indirection. Machinery is expensive now and expensive forever. The senior heuristic: **buy the seam, defer the machinery.** Put the interface where the datastore plugs in (cheap, load-bearing) but don't build the second datastore adapter until you need it. Reserve the `tenant_id` column (cheap) but don't build the tenant control plane (expensive machinery) until you have tenants.

There's a smell that tells you you've crossed from seam into speculative machinery: **abstraction with exactly one implementation that you invented before you had a second use case.** One implementation behind an interface is fine if the interface marks a *likely* change axis (you'll add the second datastore eventually). It is a smell if the interface exists purely because abstraction felt virtuous. The test is always: name the concrete second case. If you can't, you're speculating.

## 6. Designing seams: where to put the interfaces

A seam is a place where you can change one side of a boundary without touching the other. Seams are the actual mechanism of evolutionary architecture — everything else (reversibility, the cost curve, YAGNI) is the *theory* of where to put them. This section is the practice.

![A tree showing the reading-list application split into a storage port with Postgres and S3 adapters and a notify port with email and push adapters](/imgs/blogs/evolutionary-architecture-designing-for-change-6.webp)

The figure shows seams placed along the reading-list app's two most likely change axes: storage and notification. The domain logic talks to a `StoragePort` and a `NotifyPort` — narrow interfaces expressing *what* the domain needs ("save this list," "tell this user") without committing to *how*. Behind each port sit swappable adapters. Storage might be Postgres today and add an S3 adapter for list cover images later. Notification might be email today and add push tomorrow. The crucial property: adding the push adapter or the S3 adapter touches *only* the new adapter and the wiring — the domain logic doesn't move, doesn't get retested, doesn't risk regression.

The principle that decides *where* seams go is this: **put the seam along the axis you expect to change, at the boundary where the two sides have genuinely different reasons to change.** A storage port is a good seam because the domain's reason to change (new business rules) is independent of the storage's reason to change (scale, cost, query patterns). A seam between two pieces of logic that always change together is pure overhead — you pay the indirection cost and get nothing, because you never change one without the other.

Here is the storage seam in code. The domain depends on an interface, not on Postgres:

```python
from typing import Protocol
from dataclasses import dataclass

@dataclass
class ReadingList:
    id: str
    owner_id: str
    title: str
    item_ids: list[str]

# The seam: a narrow port the domain depends on.
class ReadingListStore(Protocol):
    def get(self, list_id: str) -> ReadingList | None: ...
    def save(self, lst: ReadingList) -> None: ...
    def by_owner(self, owner_id: str) -> list[ReadingList]: ...

# Domain logic depends on the PORT, never on Postgres directly.
class ReadingListService:
    def __init__(self, store: ReadingListStore) -> None:
        self._store = store

    def add_item(self, list_id: str, item_id: str) -> None:
        lst = self._store.get(list_id)
        if lst is None:
            raise KeyError(list_id)
        if item_id not in lst.item_ids:
            lst.item_ids.append(item_id)
            self._store.save(lst)
```

The adapter is the only code that knows about Postgres:

```python
import psycopg

class PostgresReadingListStore:  # implements ReadingListStore
    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    def get(self, list_id: str) -> ReadingList | None:
        row = self._conn.execute(
            "SELECT id, owner_id, title, item_ids "
            "FROM reading_lists WHERE id = %s",
            (list_id,),
        ).fetchone()
        if row is None:
            return None
        return ReadingList(id=row[0], owner_id=row[1],
                           title=row[2], item_ids=list(row[3]))

    def save(self, lst: ReadingList) -> None:
        self._conn.execute(
            "INSERT INTO reading_lists (id, owner_id, title, item_ids) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET "
            "  title = EXCLUDED.title, item_ids = EXCLUDED.item_ids",
            (lst.id, lst.owner_id, lst.title, lst.item_ids),
        )

    def by_owner(self, owner_id: str) -> list[ReadingList]:
        rows = self._conn.execute(
            "SELECT id, owner_id, title, item_ids "
            "FROM reading_lists WHERE owner_id = %s",
            (owner_id,),
        ).fetchall()
        return [ReadingList(r[0], r[1], r[2], list(r[3])) for r in rows]
```

When the day comes that `by_owner` is too slow because a power user has 50,000 lists, or you need to move list items to a separate store, the change is contained to this adapter (and maybe a new one). The domain service — where your actual business logic and your actual bugs live — does not move. **That containment is the entire return on the seam.**

But — and this is where seams earn their reputation for over-engineering — **a seam in the wrong place is worse than no seam.** Two specific anti-patterns:

**The leaky port.** If your `ReadingListStore` interface exposes `execute_sql(query: str)`, you have an interface that *claims* to be storage-agnostic but is actually a Postgres pass-through. The seam is decorative; the coupling is real. A real port speaks the domain's language ("save this list"), not the implementation's ("run this SQL"). The test: could you implement this port against a completely different technology without changing its signature? If not, it's a leaky port.

**The premature port.** A seam between two modules that have never changed independently and never will is dead weight. If `OrderValidation` and `OrderPricing` always ship together, always change together, and are really one cohesive thing, a port between them just adds an interface to maintain and a layer to trace through when debugging. Seams cost real cognitive overhead; spend them on real change axes.

The discipline, then: **seams along likely-change, genuinely-independent boundaries; nowhere else.** Storage, third-party integrations, notification channels, payment providers, anything you might swap, anything a regulation might force you to relocate — these get seams. Two halves of one cohesive computation do not.

## 7. ADRs and fitness functions: making evolution governable

Designing for change creates a new problem: if the architecture is always evolving, how does anyone — including future-you — know *why* it is the way it is, and how do you stop it from evolving into a mess? Two tools answer these, and a senior uses both as a matter of course.

**Architectural Decision Records (ADRs)** answer the "why." An ADR is a short document — a page, rarely more — that captures one significant decision: the context, the options considered, the decision made, and the consequences accepted. The point is not bureaucracy. The point is that **the most expensive thing about a decision is forgetting why you made it.** Six months later, someone looks at your single-region deployment and says "why aren't we multi-region, that's amateur hour," and without an ADR you have a pointless argument; with one, you read "we chose single-region because our compliance requirement pins data to one jurisdiction and our latency budget tolerates it, revisit when we expand to the EU" and the conversation is over in thirty seconds.

A good ADR for the reading-list app's datastore choice looks like this:

```markdown
# ADR-014: Use Postgres as the primary datastore

## Status
Accepted (2026-03-12)

## Context
Reading-list app at ~50k users, ~200:1 read/write, relational data
(users own lists own items), strong consistency needed for sharing
permissions. Team of 4, all fluent in SQL. No sharding need at
current or 10x scale (data fits comfortably on one node + replica).

## Decision
Single Postgres primary with one read replica. Access goes through
a ReadingListStore port (see ADR-009) so the datastore is swappable.

## Consequences
+ ACID transactions for permission changes; familiar ops.
+ Read replica absorbs the 200:1 read skew.
- Single-node write ceiling (~10k writes/s). Revisit at 100x users.
- One-way-door-ish on data model; mitigated by the storage port.

## Revisit when
Write QPS exceeds ~5k sustained, OR data exceeds ~500GB,
OR we need a partition key (then see ADR for sharding strategy).
```

Notice the "Revisit when" section — it names the *trigger* that should make a future team reopen the decision. That turns a one-way door into a *monitored* one-way door: you've pre-committed to the evidence that would justify the expensive change, so nobody has to re-litigate it on vibes.

**Fitness functions** answer "how do we stop it rotting." A fitness function is an automated, objective test of an architectural property — not "does the feature work" (that's a unit test) but "does the architecture still hold its shape." Examples that pay for themselves:

- A test that fails the build if any code in the `domain` package imports anything from the `infrastructure` package. This is your seam, enforced. Without it, someone will `import psycopg` directly in the domain logic six months from now, and your storage port quietly becomes decorative.
- A test that asserts the p99 latency of the home-page endpoint stays under 150 ms against a representative dataset, run in CI. This catches the slow-query regression before it ships, not after a customer complains.
- A test that fails if the public API schema changes in a backward-incompatible way without a version bump. This makes your "public contract is a one-way door" policy mechanical instead of a code-review hope.

Here is a dependency-direction fitness function — the cheapest, highest-value one to start with:

```python
import ast, pathlib

def test_domain_does_not_import_infrastructure():
    """Fitness function: the storage seam must hold.
    Domain code may not import infrastructure directly."""
    domain_dir = pathlib.Path("src/domain")
    violations = []
    for path in domain_dir.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.Import):
                mod = node.names[0].name
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
            if mod and ("infrastructure" in mod or mod.startswith("psycopg")):
                violations.append(f"{path}: imports {mod}")
    assert not violations, "Seam violated:\n" + "\n".join(violations)
```

The combination is powerful: ADRs record *why* the architecture is shaped this way and what would change it; fitness functions enforce that the shape *stays* what the ADRs say it is. Together they make evolutionary architecture governable rather than a slow slide into entropy. An architecture without fitness functions doesn't evolve — it *erodes*, one expedient shortcut at a time, until the seams you carefully placed are full of holes nobody noticed being drilled.

## 8. The monolith-first argument — and when to actually split

The monolith-first position is now the mainstream senior consensus, and it is worth stating precisely because it is so often strawmanned. The argument is **not** "monoliths are good and microservices are bad." The argument is: **start with a well-modularized monolith, and split out services only when you have a specific, present reason that a monolith cannot meet.** The default is monolith; services are an earned escalation.

The reason this is the right default is asymmetry of error. If you start monolith-first and were wrong, the cost is *extracting* a service later — annoying but bounded, and made cheap if you kept good seams (§6). If you start microservices-first and were wrong, the cost is *merging* services back together — which almost never happens, because by then two teams own the two halves, the political cost of merging exceeds the technical cost, and you're stuck operating a distributed system you didn't need. The mistakes are not symmetric, so you should bias toward the one whose error is cheaper to fix.

Concretely, here are the *legitimate* reasons to split a service out of a monolith. If none of these apply, you are splitting for fashion:

- **Independent scaling.** One part of the system has a genuinely different load profile and resource need. The reading-list app's thumbnail-generation is CPU-and-memory heavy and bursty; the rest is light and steady. Scaling them independently means a service boundary so you can run 3 thumbnail workers and 20 API instances instead of 20 identical boxes sized for the worst case. This is the *best* reason and the one that actually pays.
- **Independent deployment for team autonomy.** Two teams stepping on each other in one deploy pipeline, with one team's risky change blocking the other's safe one. When the *organization* outgrows the deploy unit — Conway's Law biting — a split buys autonomy. Note this is an org reason, not a tech reason, and it kicks in around the point you have multiple teams, not multiple engineers.
- **Fault isolation.** A component whose failure must not take the rest down, and which is risky enough that you want a hard process boundary around it. A payment integration that calls flaky third parties, isolated so its failures don't exhaust the main app's thread pool.
- **Independent technology need.** A genuine need for a different runtime — a machine-learning inference component that needs Python and GPUs while your main app is Go. Real, but rarer than people claim.
- **Compliance/data isolation.** A regulation that forces a subsystem's data and processing into a separate boundary (a separate jurisdiction, a separate security domain).

What is *not* on that list: "microservices are best practice," "we want to be cloud-native," "the architecture diagram looks more impressive," and "we read how Netflix does it." Netflix has thousands of engineers and a scale that justifies the operational tax of hundreds of services. You, probably, do not — and even Netflix and Amazon started as monoliths and split when scale forced it, which is the actual lesson their history teaches, not "start with services."

The senior move is to keep the monolith *modular* so that the split, when it's justified, is cheap. A modular monolith with clean seams between modules is a microservices architecture that hasn't paid the network tax yet — and you extract a module into a service by replacing an in-process call with a network call behind the same interface. If your monolith is a tangled ball where everything calls everything, you can't extract anything, and *that* — not the monolith itself — is the actual failure. The enemy was never the monolith. The enemy is the *big ball of mud*, and you can build that out of microservices just as easily, except now it's a *distributed* ball of mud and you can't even step through it in a debugger.

## 9. Data is the hard part: code is soft, data is hard

Here is the single most important asymmetry in evolutionary architecture, and the one most often missed: **code is soft and data is hard.** Code you can rewrite over a weekend; the old code is simply gone, replaced. Data you cannot rewrite over a weekend, because the old data still exists, real users depend on it, and it must keep being served correctly *while* you change its shape. You can refactor a function fearlessly. You cannot refactor a populated, live, customer-facing schema fearlessly, because there is no "undo" on data and no moment when the system is conveniently offline.

![A stack ordering change cost from soft application code at the top through internal API and database schema down to live data shape and the public contract at the bottom](/imgs/blogs/evolutionary-architecture-designing-for-change-8.webp)

The figure ranks the layers of a system by how hard they are to change. Application code is at the top — soft, minutes to change. As you descend, change gets harder: internal APIs (hours, you control the callers), database schema (days, structure plus migration), live data shape (weeks, you must migrate populated rows without downtime), and at the bottom the public contract (months, you don't control who depends on it). **The discipline of evolutionary architecture is largely the discipline of keeping hard-to-change decisions out of the bottom layers as long as possible, and having a safe technique for when they finally must change.**

That safe technique, for data, is the **expand-contract pattern** (also called parallel-change). It is the single most important migration pattern a senior carries, because it makes a schema change a *sequence of individually-safe deploys* rather than one terrifying big-bang.

![A pipeline showing the expand-contract migration moving from adding a new column to dual-writing old and new to backfilling old rows to dropping the old column](/imgs/blogs/evolutionary-architecture-designing-for-change-9.webp)

The pattern, illustrated above, has three phases across multiple deploys, and the key property is that **at no point is any deployed version of the code broken by the schema state.** Suppose you're renaming `user_name` to `display_name` (the trivial-looking change that has taken down many production systems when done naively as a single rename):

1. **Expand.** Add the new column `display_name`, nullable, alongside the old `user_name`. Deploy. Nothing reads the new column yet; nothing is broken. The schema now supports *both* shapes.
2. **Dual-write.** Deploy code that writes to *both* columns on every update, and reads from the new column with a fallback to the old. Now new and updated rows are correct in both places. Still nothing broken — old code reading `user_name` still works because you're still writing it.
3. **Backfill.** Run a background job that copies `user_name` to `display_name` for all the old rows that predate dual-write, in batches small enough not to lock the table or spike replication lag. Now *every* row has `display_name` populated.
4. **Contract.** Once you've verified every row has the new column and no code reads the old one, deploy code that reads and writes *only* `display_name`. In a later deploy, drop `user_name`. The change is complete, and you were never down.

Here is the migration as code — note each step is independently deployable and reversible:

```sql
-- Phase 1: EXPAND. Additive, safe, instant on most engines.
ALTER TABLE users ADD COLUMN display_name TEXT;

-- Phase 3: BACKFILL in batches (run after dual-write is live).
-- Small batches keep locks short and replication lag low.
UPDATE users
SET display_name = user_name
WHERE display_name IS NULL
  AND id IN (
    SELECT id FROM users WHERE display_name IS NULL LIMIT 5000
  );
-- repeat until zero rows match

-- Phase 4: CONTRACT, only after all readers use display_name.
ALTER TABLE users DROP COLUMN user_name;
```

```python
# Phase 2: DUAL-WRITE in the app, between expand and contract.
def save_user(conn, user_id: str, name: str) -> None:
    conn.execute(
        "UPDATE users SET user_name = %s, display_name = %s "
        "WHERE id = %s",
        (name, name, user_id),   # write BOTH columns
    )

def get_user_name(conn, user_id: str) -> str:
    row = conn.execute(
        "SELECT display_name, user_name FROM users WHERE id = %s",
        (user_id,),
    ).fetchone()
    return row[0] if row[0] is not None else row[1]  # new, fallback old
```

This is why the year-2 primary-key migration from §2 was survivable at all: it was an expand-contract migration. We added `public_id` (expand), dual-wrote it (so every new and updated row got one), backfilled 12 million rows in batches over a week, migrated readers, and only then stopped depending on the old integer ID. At no point was the system down, and at every point we could have stopped and rolled back, because each phase was individually safe. **A schema change without expand-contract is a bet that nothing will go wrong during a synchronized cutover. A schema change with it is a series of small, reversible steps.** That is the whole difference between a routine migration and a 2 a.m. incident.

Two production realities to keep you honest. First, **backfills must be batched and rate-limited**, or you'll lock the table or blow out replication lag and cause the exact outage you were trying to avoid; 5,000-row batches with a pause between them is a sane default, tuned by watching replica lag. Second, expand-contract is also how you change data *across a service boundary* using [change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — you expand the event schema additively, let consumers adopt the new field, then contract the old one, never breaking a consumer mid-flight. The pattern is fractal: it works inside one table, across tables, and across services.

## 10. The strangler fig: replacing a system without a rewrite

Eventually you face the biggest change of all: a whole subsystem, or a whole legacy application, has to be replaced. The naive approach — the **big-bang rewrite** — is one of the most reliable ways to destroy a project. You freeze the old system, build the new one in parallel for eighteen months, and cut over in one heroic weekend. It fails because the old system kept changing while you weren't looking, because eighteen months is long enough for requirements to drift out from under you, and because "one heroic weekend" has a way of becoming a three-week outage. The graveyard of software is full of rewrites that were going to be done "by Q3."

The senior alternative is the **strangler fig pattern**, named after the vine that grows around a tree, gradually envelops it, and eventually replaces it entirely while the tree is still standing. You replace a system incrementally, one piece of functionality at a time, with the old and new running side by side behind a facade, until the old system has nothing left to do and you delete it.

![A timeline of a strangler fig migration moving from a facade in front of a fully legacy system through five percent and fifty percent traffic to the new service to deleting the legacy code](/imgs/blogs/evolutionary-architecture-designing-for-change-5.webp)

The figure shows the sequence. The mechanics, concretely:

1. **Put a facade in front of the legacy system.** Every request goes through a routing layer (a reverse proxy, an API gateway, or a thin routing service) that, for now, sends 100% of traffic to the legacy system. This step changes nothing functionally — but it gives you the *control point* the whole pattern depends on. Without a facade, you have nowhere to stand to redirect traffic gradually.
2. **Pick the first slice and build it in the new system.** Choose a piece that is valuable and relatively self-contained — often a read endpoint, because reads are easier to run in parallel and verify than writes. Build that one capability in the new service.
3. **Route a trickle to the new path.** Send 1% of that endpoint's traffic to the new service, often using a feature flag or the gateway's weighted routing. Compare results against the legacy system (shadow traffic, where you call both and compare but only return the legacy result, is invaluable here). When confident, ramp to 5%, then 50%, then 100%.
4. **Repeat slice by slice.** Each migrated capability shrinks what the legacy system does. The new system grows; the old one shrinks. Crucially, *you ship value the whole way* — each slice is a real, deployed improvement, not a deferred payoff at the end of an eighteen-month dark tunnel.
5. **Delete the legacy system and the facade.** When the legacy system serves nothing, you turn it off. You may keep or remove the facade depending on whether it earns its keep as a permanent gateway.

The reason this beats the big-bang rewrite is **risk is amortized into small, reversible increments.** At every step you have a working system. Every slice is independently shippable and independently revertable — if the new path misbehaves at 5%, you flip the flag back to 0% and you've affected 5% of one endpoint for a few minutes, not your whole business for a weekend. You learn the legacy system's real behavior *by migrating it*, not by reverse-engineering it up front. And you never bet the company on a single cutover.

There is real subtlety in the data, and it is where strangler-fig migrations actually get hard. While both systems run, **who owns the data?** Three honest options, in rough order of preference: (a) the new system reads from the legacy database directly during transition — simplest, but couples the new system to the old schema; (b) you replicate data between the two with change data capture, keeping each system's store in sync — robust but operationally heavier; (c) you migrate the data alongside the traffic, slice by slice, using expand-contract. The detail to never skip: **be explicit about which system is the source of truth for each piece of data at each moment**, because the failure mode of a careless strangler migration is two systems both thinking they own the same data, diverging silently, until a customer notices their account is in two states at once. Write down the source-of-truth map. It is the first thing to go wrong and the last thing people document.

## 11. Coupling vs autonomy: the trade-off under everything

Every decision in this article is, underneath, the same trade-off in a different costume: **coupling versus autonomy.** Tightly coupled components are simple — one transaction, one deploy, in-process calls, easy reasoning, no network in the middle — but they cannot change or scale or fail independently. Autonomous components can each evolve, scale, and fail on their own — but you pay for that autonomy in network calls, eventual consistency, distributed-transaction complexity, and operational surface. There is no free lunch; there is only choosing where on this spectrum each boundary sits.

The monolith-vs-microservices question is this trade-off at the service level. The seams question is this trade-off at the module level — a seam *decouples* two modules so they can change independently, at the cost of an indirection. The data-ownership question in the strangler migration is this trade-off at the data level. Even the one-way/two-way door framing reduces to it: a one-way door is usually a place where you've created deep coupling (data, public contracts) that is expensive to decouple later.

The senior skill is not "minimize coupling" — that's the over-engineer's error, and it leads to a thousand tiny autonomous pieces nobody can reason about. The skill is **put the decoupling where independent change is likely, and accept tight coupling where it isn't.** Couple things that change together; decouple things that change apart. Two modules that always ship together should be tightly coupled — a seam between them is pure cost. A module on a likely-independent change axis (storage, a third-party integration, anything a regulation might relocate) earns its decoupling. This is the same principle as seam placement (§6), and it is the same principle as service splitting (§8), because they are the same principle viewed at different altitudes.

The cost of getting this wrong runs in both directions, and a senior watches for both. Too much coupling and the system ossifies — every change touches everything, velocity grinds to zero, and you're back to the big ball of mud. Too much autonomy and the system becomes a distributed-systems problem you didn't need: you've turned function calls into network calls, ACID into sagas, and a debuggable process into a [distributed transaction](/blog/software-development/database/saga-pattern-distributed-transactions) you reason about with traces and prayer. The art is in the *placement*, and the placement follows the same question every time: **is independent change here likely enough and valuable enough to pay the autonomy tax?**

## 12. Trade-offs: flexibility now vs simplicity now

Here is the decision section, the heart of thinking like a senior: every flexibility investment is a bet, and a bet has a cost whether or not it pays off. The matrix in §7 judged specific investments; this section gives you the general decision frame so you can judge *new* ones the matrix didn't list.

For any "should we build in flexibility here" decision, a senior runs four questions:

| Question | If yes → lean flexible | If no → lean simple |
|---|---|---|
| Is the change likely (not just possible)? | A storage swap, a new notification channel | A different programming language, a new cloud |
| Is retrofitting the flexibility later expensive? | Data model changes, public API versioning | Internal refactors, swapping a library |
| Is the flexibility a *seam* (cheap) or *machinery* (costly)? | An interface, a reserved column, a feature flag | A config engine, a plugin framework, a control plane |
| Does the flexibility add cognitive load *now*? | Minimal indirection, well-named | A layer of indirection nobody can trace through |

The decision rule that falls out: **buy flexibility when the change is likely AND the retrofit is expensive AND the flexibility is a cheap seam. Decline it otherwise.** Most "should we make this configurable / pluggable / generic" questions fail at least one of those clauses, which is why "build the simple thing" is the right default and flexibility is the earned exception.

Let me put the full decision matrix in one place, because seeing the *categories* side by side is how you build the senior reflex:

| Decision type | Flexibility now buys | Simplicity now buys | Default, and when to flip |
|---|---|---|---|
| Datastore choice | Easy migration if access pattern changes | Less code, fewer indirection layers | **Seam yes, second adapter no** — build the port, defer the adapter |
| Monolith vs services | Independent scale/deploy | One deploy, ACID, in-process speed | **Monolith** — split on a present, named reason (§8) |
| Schema design | Cheap future evolution | Simpler queries, fewer joins | **Expand-contract-ready, not over-normalized** — design for the migration, not every future shape |
| API contract | Consumers can adopt changes | Less versioning machinery | **Versioned from first public release** — public contracts are one-way doors |
| Identifiers | No leak, no enumeration, portable | Simpler, smaller, human-readable | **Opaque from day one** — baked into data and URLs, brutal to change late |
| Config vs code | Change behavior without deploy | Code is readable and debuggable | **Code** — flip only for genuinely operational toggles (flags), never a "rules engine" |
| Multi-tenancy | Onboard tenant 2 cheaply | No tenant plumbing | **Seam (tenant_id) yes, control plane no** — until a second tenant is signed |

The through-line of the whole matrix: **the cheap, load-bearing move is almost always to add the seam and defer the machinery.** You buy optionality at the interface (cheap, reversible, small blast radius) and you decline to build the implementation until the need is concrete. That single discipline — seam now, machinery on evidence — resolves the overwhelming majority of flexibility-vs-simplicity debates correctly, and it's the operational form of "design for change without over-building."

## 13. Optimization: optimize for the change you'll actually make

The optimization lens of evolutionary architecture is different from raw performance optimization, and getting the distinction is a senior marker. Performance optimization asks "where is the bottleneck in *execution*." Evolutionary optimization asks "where is the bottleneck in *change*" — what change will I actually need to make, and is the system optimized to make *that* change cheaply? You profile your *change* history the way you'd profile your hot path.

So: what changes do real systems actually make? Across many systems and many years, the empirical answer is remarkably consistent:

- **They change their data model — constantly.** New fields, new entities, new relationships, denormalization for a slow query, a new index, a new shard key. This is the most frequent significant change in almost every system's life. *Optimize for it:* expand-contract discipline, a storage seam, migration tooling you trust, the habit of additive-first changes.
- **They change their scaling story — a few times.** Add a cache, add a read replica, split read/write paths, shard the database, extract a hot service, go multi-region. Each is a known move with a known playbook. *Optimize for it:* keep the seams that let you insert a cache or extract a service without a rewrite; know your next bottleneck before you hit it (the "Revisit when" triggers in your ADRs).
- **They rarely change their language or framework.** Despite endless online debate, mature systems almost never rewrite in a new language; the cost is enormous and the benefit usually illusory. *Do not optimize for it.* Building a language-agnostic indirection layer to hedge against a rewrite you'll never do is the textbook speculative-generality trap.
- **They almost never change their fundamental domain.** A reading-list app stays a reading-list app. The core domain concepts are the most stable thing in the system. *Optimize by* modeling the domain cleanly and putting your seams at the domain's edges (storage, integrations), not through its middle.

This ranking tells you exactly where to spend your flexibility budget: **lavishly on the data model and the scaling seams; not at all on language portability or domain indirection.** It is the concrete, evidence-based answer to "design for change" — design for *these* changes, in this order, because these are the ones the system will actually demand.

#### Worked example: evolving the reading-list app from 1k to 1M users

Let me trace the running example all the way up, naming at each order of magnitude what changed, what the early decisions cost or saved, and how to *measure* each move. This is the optimization story made concrete.

**1,000 users.** One monolith process, one Postgres instance. Reads and writes both hit the primary. p99 on every endpoint is ~40 ms; the database is 95% idle. Total infra: one small box and a managed Postgres, maybe **\$80/month**. *Early decisions that mattered:* we put in a storage seam (§6) and chose opaque IDs (§2) — both cheap now, both load-bearing later. *Decisions that would have hurt:* a colleague lobbied for a separate recommendation service and event stream here; declining it (§4) saved us ~\$1,400/month and a week of velocity per change.

**10,000 users.** Read traffic has grown to ~500 reads/s at peak against ~3 writes/s — the 200:1 skew is now visible. The primary's CPU is climbing on reads. *The change:* add one read replica and route reads to it through the storage seam (the domain code didn't move — the port's `get`/`by_owner` now hit the replica, `save` hits the primary). *Measure the win:* primary CPU drops from 70% to 25%; read p99 holds at 40 ms with headroom. Cost: one replica, ~**\$120/month** more. This is the first scaling move and it was a config change *because the seam existed.* Without the seam, routing reads to a replica means touching every query site in the codebase.

**100,000 users.** ~5,000 reads/s, ~30 writes/s. The replica handles reads, but a few endpoints are slow because the same hot lists are read thousands of times a second. *The change:* add a Redis cache in front of the hottest reads (popular shared lists), behind the same storage seam, with a 60-second TTL and explicit invalidation on write. Read p99 on cached paths drops from 40 ms to 4 ms; the database read load falls by ~80%. We also split the request path: the "share a list" viral endpoint (now the load driver, as predicted in §1) gets its own connection pool so a spike there can't starve the rest. Cost: a small Redis, ~**\$90/month**. Caching has its own pitfalls — stampedes, stale invalidation — covered in the [caching strategies](/blog/software-development/system-design/caching-strategies-and-the-pitfalls-that-bite) piece; the senior point here is *where* the cache plugged in: behind the existing seam, no domain change.

**1,000,000 users.** ~40,000 reads/s, ~400 writes/s, data approaching the single-primary's comfortable write and storage ceiling. Now the ADR-014 "Revisit when" trigger fires (write QPS approaching the threshold, data approaching 500 GB). *Two changes, both finally justified:* first, shard the database by `owner_id` so writes spread across nodes — the hard one-way-door decision we deliberated carefully because the partition key is nearly impossible to change later (see [partitioning and sharding without downtime](/blog/software-development/system-design/partitioning-and-sharding-without-downtime) and the mechanism deep-dive on [database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding)). Second, extract the thumbnail/cover-image generation into a service, because it has a genuinely different, bursty, CPU-heavy load profile (§8's best reason to split). *Measure the win:* write throughput scales linearly with shards; thumbnail bursts no longer touch API latency; we run 4 thumbnail workers instead of over-provisioning 20 API boxes. *What the early decisions bought:* because we'd reserved a clean `owner_id` as a natural partition key and kept the thumbnail logic behind a module boundary, the shard and the extraction were *projects, not rewrites.*

The whole arc, in one sentence: **the early decisions that paid off were the cheap seams and the opaque IDs; the decisions that would have hurt were every premature piece of machinery we declined to build; and every scaling move was the standard next move, made cheap by a seam we'd placed for exactly that purpose.** That is evolutionary architecture working as designed.

#### Worked example: a one-way vs two-way door decision audit

A second worked example, on the decision discipline itself. A team is kicking off a new project — a B2B analytics dashboard — and has a list of decisions on the whiteboard. The senior's first move is not to decide them; it's to *classify* them, because that determines how much time each one deserves.

| Decision | Door | Deliberation budget | Rationale |
|---|---|---|---|
| Frontend framework (React vs Svelte) | Two-way | 1 day, one person | UI layer is rewritable in a sprint; coupling stays in the frontend |
| Backend language (Go vs Python) | Two-way-ish | 2 days | Reversible early, but coupling grows — decide before much code exists |
| Primary datastore + data model | One-way | 1–2 weeks, write ADR | Data layout is a migration to change; stress-test against access patterns |
| Partition/shard key | Hard one-way | Defer + ADR with trigger | Nearly unchangeable once sharded; don't even decide until forced (§8) |
| Public API shape (REST resource model) | One-way | 1 week, version it | External consumers will lock it; design for evolution from v1 |
| Auth provider (build vs Auth0 vs Cognito) | One-way-ish | 3 days | Migrating identity is painful; but a seam (auth port) makes it two-way-ish |
| Internal module boundaries | Two-way | Ongoing, low ceremony | Move code between modules freely; refine as the domain clarifies |
| Logging/metrics vendor | Two-way | Half a day | Swappable behind a thin wrapper; pure two-way door |

The audit immediately reveals where to spend: the datastore/data model, the shard key, the API shape, and auth get real deliberation and ADRs; everything else gets a fast, low-ceremony decision. The team that does this ships *faster* than the team that deliberates everything equally, because they stop burning their best thinking on the framework choice and save it for the partition key. And notice the auth row: a seam (an `AuthProvider` port) *converts* a one-way-ish door into a two-way-ish one — which is the deepest lesson in the whole article. **Good architecture doesn't just react to which doors are one-way; it strategically installs seams that turn expensive one-way doors into cheaper two-way ones.** That is the highest form of designing for change.

## 14. Case studies: evolution in the wild

Theory is cheap. Here are real systems that lived this — two that did it right, one cautionary tale — with the specific lesson each teaches.

**Amazon: monolith to services, when scale forced it.** Amazon.com began, famously, as a monolithic application — a single large codebase — and it scaled the company to enormous size *as a monolith* before the move to services. The transition to a service-oriented architecture in the early-to-mid 2000s was driven by a concrete, present problem: the monolith and its shared database had become a bottleneck to *team velocity*, not just to performance. Hundreds of engineers contending over one codebase and one schema meant every change was slow and risky — Conway's Law biting hard. The lesson is precisely §8: Amazon did *not* start with services; it earned them when the organization outgrew the deploy unit. The often-cited internal mandate that teams expose their functionality only through service interfaces was an *organizational* decoupling move as much as a technical one. The takeaway for everyone who isn't Amazon: they reached this point at a scale you have not, and copying the destination without the journey is how you get premature microservices.

**The Strangler at scale: incremental legacy replacement.** The strangler-fig pattern earned its name and its reputation on real, large legacy migrations — the canonical write-ups (Martin Fowler's among them) describe replacing big monolithic legacy applications piece by piece behind a facade rather than in a big-bang rewrite. The pattern shows up again and again in public engineering blogs from companies modernizing off mainframes and aging monoliths: stand up a routing facade, migrate one capability at a time, run old and new in parallel with shadow comparison, ramp traffic gradually, delete the old system when it's idle. The repeated lesson across all of them is the one from §10: the migrations that *finished* were incremental and value-shipping the whole way; the big-bang rewrites are the ones that became multi-year death marches or got cancelled. When you read about a successful "we replaced our legacy system" story, look closely — it is almost always a strangler fig, even when they don't use the name.

**The premature-microservices cautionary tale.** The most instructive public example in the industry's collective memory is the well-documented account of a team (Segment's engineering blog is the famous version) that split into a large number of microservices early, found the operational burden — separate deploys, separate dependency sets, separate on-call surfaces, the combinatorial explosion of services-times-destinations — crushing for their team size, and made the heretical-at-the-time decision to *consolidate back into a monolith*. Their write-up is required reading precisely because it documents the asymmetry from §8 from the painful side: they paid the full distributed-systems tax — network calls, versioning, operational sprawl — for a scale and team size that a monolith would have served better, and clawing back to a monolith was hard *because* merging services is harder than splitting them. The lesson is not "microservices are bad." It's "microservices are a tax you should pay only when you're getting something for it, and the default should be monolith-first because the error is cheaper to fix in that direction."

A fourth, quieter lesson that runs through all three: **every one of these systems changed shape multiple times, and the ones that survived were the ones where the changes were affordable.** Amazon could afford to split because the move bought real autonomy at a scale that justified it. The strangler migrations succeeded because they made the big change into many small affordable ones. The microservices-reversal was painful precisely because the team had made an expensive change (the split) without the seams or the scale to make reversing it cheap. The constant is not any particular architecture. The constant is that *change came*, and affordability of change decided the outcome.

## 15. When to reach for this (and when not to)

Evolutionary architecture is not a style you opt into; it's the default discipline for any system that will live longer than a hackathon. But the *intensity* of the practices scales with stakes, and a senior calibrates rather than applies everything everywhere.

**Reach for the full discipline — seams, ADRs, fitness functions, expand-contract, strangler-fig — when:**

- The system is **load-bearing and long-lived**: it'll be around for years, real users depend on it, and downtime or data corruption is expensive. This is where the cost-of-change curve has the most leverage and where being unable to change cheaply will eventually hurt you.
- You have **real uncertainty about scale or requirements** — i.e., almost every new product. The less you know, the more the optionality of good seams is worth.
- You're **operating on live data**: any system with a populated, customer-facing datastore needs expand-contract discipline as table stakes, because there is no other safe way to change a live schema.
- You're **replacing a legacy system**: reach for the strangler fig essentially always; reach for the big-bang rewrite essentially never.

**Don't over-apply it when:**

- The system is a **genuine throwaway** — a one-off script, a prototype you'll discard, a migration tool that runs once. Seams and ADRs are overhead with no payoff; build the simple thing and delete it.
- The change axis is **genuinely fixed and known**. If a system truly will never need to swap its datastore (rare, but it happens — a fixed embedded system, a closed appliance), the storage seam is dead weight. Be honest, though: "we'll never need to change X" is wrong far more often than it's right.
- You're **using flexibility as procrastination**. Building elaborate indirection to avoid committing to a decision is not evolutionary architecture; it's speculative generality wearing its costume. If you can't name the concrete future change a seam enables, you're procrastinating, not designing.

The meta-rule: **the practices are a dial, not a switch.** A throwaway gets none of them. A weekend project gets clean module boundaries and nothing else. A product gets seams on the likely-change axes, ADRs on the one-way doors, and expand-contract on the data. A large, multi-team, multi-region system gets all of it plus fitness functions enforcing the seams in CI. Match the ceremony to the stakes, and never let the ceremony exceed the stakes — over-applying these practices to a small system is its own form of premature scaling, the exact error §4 warns against.

## 16. Key takeaways

- **The first architecture is always wrong**, so optimize for the cost of the *next change*, not the elegance of the current diagram. Every system changes; the survivors are the ones where change stays cheap.
- **The cost of changing a decision rises by orders of magnitude over time** — roughly 1× at design, 10× in code, 100× in data, 1000× in a public contract. Push hard-to-reverse decisions late and keep them out of the expensive layers.
- **Classify every decision as a one-way or two-way door.** Make two-way doors fast and reversibly; deliberate on one-way doors and write an ADR. Most teams waste deliberation on the reversible and rush the irreversible.
- **Build for 10×, not 1000×.** Premature scaling is as costly as premature optimization — a network hop is 1000× slower than a function call and adds partial failure you didn't have. Pay the distributed-systems tax only when you're getting something for it.
- **Buy the seam, defer the machinery.** A thin interface on a likely-change axis is cheap and load-bearing; a heavyweight subsystem built for a hypothetical future is speculative generality that rots. If you can't name the concrete second use case, you're speculating.
- **Default to a modular monolith; split to services on a present, named reason** — independent scaling, team autonomy, fault isolation, a real tech need, or compliance. The enemy was never the monolith; it's the big ball of mud, which you can build out of microservices too — and then it's distributed and undebuggable.
- **Code is soft, data is hard.** You can rewrite code over a weekend; you cannot rewrite live, customer-facing data. Carry the expand-contract pattern for every schema change so a migration is a sequence of safe deploys, not a big-bang bet.
- **Replace systems with the strangler fig, never the big-bang rewrite.** Facade in front, migrate slice by slice, run old and new in parallel, ship value the whole way, delete the legacy when it's idle. And always write down the source-of-truth map.
- **It's all coupling vs autonomy.** Couple things that change together; decouple things that change apart. The art is placement, and the best architects install seams that turn expensive one-way doors into cheap two-way ones.
- **Optimize for the change you'll actually make:** the data model (constantly) and the scaling story (a few times) — not the language or the domain, which almost never change. Spend your flexibility budget where systems actually demand it.

## 17. Further reading

- *Building Evolutionary Architectures* by Neal Ford, Rebecca Parsons, and Patrick Kua — the book that named the field, the definitive treatment of fitness functions.
- Martin Fowler, "StranglerFigApplication" and "MonolithFirst" — the canonical short essays on incremental replacement and the monolith-first default.
- Michael Nygard, "Documenting Architecture Decisions" — the original ADR write-up and the lightweight template most teams adopt.
- Sam Newman, *Building Microservices* and *Monolith to Microservices* — the honest, trade-off-first treatment of when and how to split, including the expand-contract and parallel-change patterns in detail.
- Werner Vogels and the Amazon engineering record on the move to service-oriented architecture — the canonical "earned the split at scale" story.
- The Segment engineering blog's account of consolidating microservices back to a monolith — the best public cautionary tale on premature microservices.
- Companion deep-dives in this series: [choosing a datastore across SQL, NoSQL, and NewSQL](/blog/software-development/system-design/choosing-a-datastore-sql-nosql-newsql), [partitioning and sharding without downtime](/blog/software-development/system-design/partitioning-and-sharding-without-downtime), and [API design across REST, gRPC, and GraphQL](/blog/software-development/system-design/api-design-rest-grpc-graphql).
- Mechanism deep-dives that this article links to at the architect layer: the [saga pattern for distributed transactions](/blog/software-development/database/saga-pattern-distributed-transactions), [change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern), and [database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding).
