---
title: "Choosing a Shard Key: The One Decision You Can't Take Back"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "The shard key is the single most important and nearly-irreversible decision in horizontal scaling — choose it by access pattern, not convenience, or you will migrate every row to fix it."
tags: ["database-scaling", "sharding", "shard-key", "partitioning", "consistent-hashing", "multi-tenancy", "distributed-systems", "system-design", "data-modeling", "scalability"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 35
---

I have been in the room for three different "we need to re-shard" decisions, and they all had the same shape. Someone picks a shard key in week one because it is the column that happens to be on every row — a `tenant_id`, a `created_at`, an auto-increment `id`. The system ships. Eighteen months later the cluster is in production with billions of rows, one shard is at 90% disk while the others idle at 15%, a dashboard query times out because it fans across all 64 shards, and the only fix anyone can name is to choose a *different* shard key and physically move every row to where the new key says it belongs. That migration is a quarter of engineering time, a frozen feature roadmap, and a non-trivial risk of data loss. It happens because the shard key was chosen for convenience, and the shard key is the one decision in horizontal scaling you cannot quietly take back.

This post is about making that decision well the first time. The thesis is blunt: **the shard key is a one-way door, so you choose it by how your application reads and writes data, not by which column is easiest to reach for.** Everything else in a sharded system — routing, rebalancing, query planning, even your on-call burden — is downstream of this one choice.

If you have not yet read [the database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree), read it first: sharding is the *last* rung on that ladder, and most teams should never climb to it. This post assumes you have already exhausted vertical scaling, read replicas, caching, and functional partitioning, and have genuinely concluded that you must split one logical table across many physical machines. It also builds directly on the mechanics in [database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding) — what a shard is, why fan-out queries hurt — so I will not re-derive those.

## Why the shard key is different from every other schema choice

Most schema decisions are reversible on a Tuesday afternoon. Add an index, drop it. Add a column, backfill it, drop it. Denormalize a table, then normalize it back when the join gets cheap again. These are all local, online, and undoable.

The shard key is none of those things. Here is the assumption-versus-reality table that I wish every team saw before they picked one.

| Common assumption | The reality once you are in production |
| --- | --- |
| "We can change the shard key later if it's wrong." | Changing it means recomputing the location of *every existing row* and physically moving most of them. It is a full data migration, not an `ALTER`. |
| "The shard key is just an index choice." | An index is metadata you can rebuild online. The shard key determines which *machine* holds each row — it is baked into the physical topology. |
| "We'll pick the primary key; it's unique." | Uniqueness has nothing to do with distribution or query alignment. A monotonic primary key is one of the worst shard keys you can pick. |
| "Any high-cardinality column will spread the load." | Cardinality is necessary but not sufficient. A high-cardinality key that no query filters on forces every query to fan out across all shards. |
| "We'll shard by `created_at` so range queries are easy." | A timestamp is monotonic; every new write lands on the newest shard, so you have built a system that can only ever use one machine's write capacity at a time. |

The reason it is irreversible is mechanical, not philosophical. A shard key defines a function — call it `route(key)` — that maps every row to exactly one shard. The instant a row is written, that function decided where it physically lives. If you later change the function (a different column, a different hash, a different shard count), the old rows are now in the wrong place according to the new function. Reconciling that means reading every row, recomputing its destination, and moving it — while the system keeps taking traffic. That is the entire subject of [resharding without downtime](/blog/software-development/database-scaling/resharding-without-downtime), and the fact that an entire deep-dive exists for "how to undo this decision" should tell you how expensive undoing it is.

![A shard key is extracted from each row and hashed to pick exactly one shard; the mapping is chosen once and is expensive to change](/imgs/blogs/choosing-a-shard-key-1.webp)

The diagram above is the mental model for this entire post. Each incoming row carries a shard key — here, `user_id`. A routing function, typically `hash(user_id) % N`, turns that key into a shard number, and the row is written there and only there. User 42's rows always route to shard 1; user 7's always route to shard 0. The two annotations are the whole story: the key is chosen *once*, and changing it means rehashing every row. The rest of this post is a tour of how to pick a key so that you never have to.

## The three properties of a good shard key

A senior rule of thumb to carry through everything below: **a shard key is good only if it clears three tests at once — high cardinality, even distribution, and alignment with your dominant query.** Failing any one of them produces a different, predictable failure mode, and the failure modes are the classic war stories of the next section.

![The three properties — high cardinality, even distribution, query alignment — each with what it means, when it passes, and when it fails](/imgs/blogs/choosing-a-shard-key-2.webp)

The matrix above lays out all three. Read it row by row: each property has a plain-English meaning, a "passes when" example, and a "fails when" example that is also a real anti-pattern. Notice that the three properties are not the same thing wearing different hats — a key can ace one and fail another.

### Property 1: high cardinality

**Cardinality is the number of distinct values the key can take.** It sets a hard ceiling on how many ways you can split the data. If your key is `subscription_status` with values `{active, trialing, canceled, past_due}`, you have exactly four distinct values; no hashing scheme on earth can spread four values across 64 shards in a balanced way. At best you get four occupied shards and 60 empty ones, and the `active` shard holds 95% of your rows.

The rule: **the cardinality of the shard key must be much larger than the number of shards you will ever have** — by orders of magnitude, not a small multiple. If you plan for 64 shards and want headroom to grow to 1,024, you want a key with at least millions of distinct values so that each shard receives tens of thousands of distinct keys and the law of large numbers can smooth out the load. A `user_id` in a system with 50 million users has 50 million distinct values; that is a healthy cardinality. A `country_code` has roughly 200; that is a catastrophe.

```python
# A quick cardinality sanity check before you commit to a key.
# Run this against a representative sample of production data.
import psycopg2

def cardinality_report(conn, table, candidate_keys, planned_shards):
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table}")
        total_rows = cur.fetchone()[0]
        for key in candidate_keys:
            cur.execute(f"SELECT count(DISTINCT {key}) FROM {table}")
            distinct = cur.fetchone()[0]
            keys_per_shard = distinct / planned_shards
            verdict = "OK" if keys_per_shard >= 1000 else "TOO LOW — will not balance"
            print(f"{key:20s} distinct={distinct:>12,} "
                  f"keys/shard={keys_per_shard:>10,.0f}  {verdict}")
        print(f"\ntotal rows = {total_rows:,}, planned shards = {planned_shards}")

# Example output for an orders table sharded 64 ways:
#   user_id      distinct=   48,200,113  keys/shard=   753,127  OK
#   order_id     distinct=  812,440,901  keys/shard=12,694,389  OK
#   status       distinct=            6  keys/shard=         0  TOO LOW — will not balance
#   country      distinct=          194  keys/shard=         3  TOO LOW — will not balance
```

The heuristic `keys_per_shard >= 1000` is deliberately conservative. Below it, even perfect hashing leaves you exposed: a single popular value (the `US` country, the `active` status) can dominate a shard because there are too few other values to dilute it.

### Property 2: even distribution

High cardinality guarantees you *can* spread the data; it does not guarantee you *will*. **Even distribution means no single key value — and no narrow band of values — carries dramatically more rows or traffic than the others.** Cardinality is about the count of distinct values; distribution is about how *traffic* and *volume* are spread across them.

The two failure shapes are skew and hotspots. **Skew** is when one value owns a disproportionate share of the data: in a multi-tenant SaaS sharded by `tenant_id`, your largest customer might hold 1,000× the rows of your median customer, so that one tenant's shard fills while others idle. **Hotspots** are temporal: even if data volume is balanced, traffic can concentrate on one shard at a time — the canonical case being a monotonic key where every *new* write targets the newest shard.

This is exactly why we measure distribution before committing, not after. A high-cardinality key with a heavy-tailed distribution is a trap: it passes the cardinality check and fails in production. The "is this key skewed?" script later in this post is the tool for catching it.

### Property 3: alignment with the dominant query

This is the property teams most often forget, and the one that hurts the most subtly. **A shard key is aligned with a query when the query's filter predicate includes the shard key, so the router can send the query to exactly one shard.** When the predicate does *not* include the shard key, the router has no idea which shard holds the answer, so it must ask *all* of them and merge the results — a scatter-gather.

![When the query predicate matches the shard key one shard answers; when it doesn't, every shard is queried and the coordinator merges](/imgs/blogs/choosing-a-shard-key-3.webp)

The before/after above is the single most important picture in this post after the mental model. On the left, the application queries `WHERE email = ?` but the data is sharded by `user_id`. The router cannot translate an email into a shard, so it broadcasts to all 16 shards, each runs the query, and a coordinator merges and sorts the 16 partial results. Your latency is now the *slowest* of 16 shards (tail latency dominates), your throughput is divided by 16 because every query touches every machine, and your blast radius for one slow shard is every query in the system.

On the right, the application queries `WHERE user_id = ?` and the data is sharded by `user_id`. The router computes `hash(user_id) % 16`, sends the query to one shard, that shard does a single indexed lookup, and the latency is one shard's response time. Same data, same hardware — the only difference is whether the query predicate matches the shard key.

> If most of your queries scatter-gather, you have not built a sharded database. You have built N databases that all answer every question, which is strictly worse than one database, because now you also pay coordination and tail-latency costs.

The practical consequence: **you choose the shard key by listing your highest-volume queries and picking the column that appears in the `WHERE` clause of the most important ones.** Not the column that is most unique, not the column that is the primary key — the column your hottest queries filter on. For a social network, that is almost always `user_id`. For B2B SaaS, it is almost always `tenant_id` (or `account_id`/`org_id`). For an IoT platform, it is the `device_id`. The dominant access pattern names your shard key.

## The classic mistakes, and the exact failure each one causes

Every bad shard key I have seen in production violates one of the three properties, and each violation has a signature failure. If you learn to recognize the three signatures, you can diagnose a misery-inducing cluster in about five minutes.

![Three anti-patterns — low cardinality, monotonic keys, celebrity values — each maps to the property it violates, the failure it produces, and the 3am page](/imgs/blogs/choosing-a-shard-key-4.webp)

The graph above traces each anti-pattern from cause to consequence. Low cardinality violates *spread* and produces a handful of giant shards. A monotonic key violates *even distribution in time* and produces a write hotspot on the newest shard. A celebrity or whale value violates *even distribution in volume* and melts one shard. All three end at the same place: a 3am page where you are rebalancing a live cluster under load, which is the most dangerous operation in distributed databases.

### Mistake 1: the low-cardinality key → a few giant shards

This is the one people make when they reach for the column that "groups the data nicely." `status`, `country`, `region`, `plan_type`, `is_active`. They all feel like reasonable ways to partition because they *are* reasonable ways to think about the data — but they are terrible ways to physically distribute it.

Suppose you shard an orders table by `country`. The United States is your biggest market, so the US shard holds 40% of all orders. You cannot split it: every US order has `country = 'US'`, and `hash('US')` is a single value that maps to a single shard. You have a 40%-full shard and you literally cannot relieve it by adding more shards, because the data has no finer key to redistribute along. Worse, the moment that one shard hits its disk or IOPS ceiling, your entire US business is degraded and no amount of horizontal scaling helps. The whole *point* of sharding — add a machine, get more capacity — does not work, because the data refuses to spread.

The tell: a few shards near capacity while the rest idle, and the hot shards correspond exactly to the popular values of a low-cardinality column.

### Mistake 2: the monotonic key → all writes pile onto the newest shard

A monotonic key is any key whose value strictly increases over time: an auto-increment integer `id`, a `created_at` timestamp, a ULID or a v1 UUID (which embeds a timestamp), a Snowflake ID. They are catastrophic shard keys *for writes* when combined with range partitioning, and they are subtly bad even with hashing if your access pattern is "recent data."

<figure class="blog-anim">
<svg viewBox="0 0 720 280" role="img" aria-label="With a monotonic shard key every new write routes to the newest shard, which heats up while older shards stay cold" style="width:100%;height:auto;max-width:820px">
<title>Monotonic key write hotspot: all writes pile onto the newest shard</title>
<style>
.hk-shard{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.hk-cold{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.hk-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.hk-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.hk-dot{fill:var(--accent,#6366f1)}
.hk-hot{fill:#ef4444;opacity:0}
@keyframes hk-flow{0%{transform:translateX(0);opacity:0}8%{opacity:1}88%{opacity:1}100%{transform:translateX(520px);opacity:0}}
@keyframes hk-heat{0%,20%{opacity:0}60%,100%{opacity:.55}}
.hk-p{animation:hk-flow 5s linear infinite}
.hk-p2{animation-delay:1.25s}
.hk-p3{animation-delay:2.5s}
.hk-p4{animation-delay:3.75s}
.hk-burn{animation:hk-heat 5s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.hk-p{animation:none;opacity:0}.hk-burn{animation:none;opacity:.55}}
</style>
<text class="hk-lbl" x="60" y="34">writes (key++)</text>
<rect class="hk-cold" x="120" y="120" width="120" height="90" rx="8"/>
<text class="hk-lbl" x="180" y="160">shard 0</text>
<text class="hk-sub" x="180" y="182">cold / idle</text>
<rect class="hk-cold" x="280" y="120" width="120" height="90" rx="8"/>
<text class="hk-lbl" x="340" y="160">shard 1</text>
<text class="hk-sub" x="340" y="182">cold / idle</text>
<rect class="hk-shard" x="440" y="120" width="120" height="90" rx="8"/>
<rect class="hk-hot hk-burn" x="440" y="120" width="120" height="90" rx="8"/>
<text class="hk-lbl" x="500" y="160">shard 2</text>
<text class="hk-sub" x="500" y="182">newest</text>
<rect class="hk-shard" x="600" y="120" width="120" height="90" rx="8"/>
<rect class="hk-hot hk-burn" x="600" y="120" width="120" height="90" rx="8"/>
<text class="hk-lbl" x="660" y="160">shard 3</text>
<text class="hk-sub" x="660" y="182">HOT</text>
<circle class="hk-dot hk-p" cx="40" cy="165" r="9"/>
<circle class="hk-dot hk-p hk-p2" cx="40" cy="165" r="9"/>
<circle class="hk-dot hk-p hk-p3" cx="40" cy="165" r="9"/>
<circle class="hk-dot hk-p hk-p4" cx="40" cy="165" r="9"/>
<text class="hk-sub" x="360" y="250">every incrementing key lands on the rightmost (newest) shard -> one hot partition</text>
</svg>
<figcaption>A monotonic shard key (timestamp or auto-increment) sends every new write to the newest shard; older shards sit idle while the tail melts.</figcaption>
</figure>

The animation above shows the mechanism. With range partitioning on a monotonic key, shards own contiguous key ranges: shard 0 holds the oldest IDs, shard 3 holds the newest. Every `INSERT` gets the next-higher ID, so every `INSERT` lands on shard 3 — the newest shard becomes a single hot partition absorbing 100% of write traffic while shards 0 through 2 sit cold. You have a 4-shard cluster with the write throughput of one shard. This is the exact failure that motivated [random UUIDs are killing your database performance](/blog/software-development/database/random-uuids-are-killing-your-database-performance) on the *index* side, except here it plays out at the shard level.

The cruel twist: monotonic keys *also* make the most attractive range queries ("give me yesterday's orders") fast and local, because all of yesterday's data is on one shard. So the read pattern lies to you — it looks great in the demo — while the write pattern quietly caps your scalability. This read/write tension is real, and the resolution is usually to hash the monotonic key (sacrificing the local range scan) or to use a composite key whose prefix is *not* monotonic.

### Mistake 3: the celebrity / whale problem → one tenant melts its shard

Even a high-cardinality, query-aligned key can fail if the *distribution* of activity across key values is heavy-tailed — and in practice it almost always is. In a social network sharded by `user_id`, a celebrity with 200 million followers generates orders of magnitude more reads, writes, and fan-out work than a typical user. Their shard runs hot constantly. In multi-tenant SaaS sharded by `tenant_id`, your largest enterprise customer might be 1,000× your median customer; when you happen to hash them onto the same shard as a few other mid-size tenants, that shard melts and the co-located tenants suffer collateral damage they did nothing to cause.

This is not a flaw in the key choice per se — `user_id` and `tenant_id` are the *right* keys for those systems. It is a flaw in assuming uniform load across key values. The fix is not a different key; it is special-casing the whales, which we get to in the multi-tenancy section.

The tell: most shards are balanced, but one or two run consistently hot, and when you investigate, the load traces to a single key value (one user, one tenant, one device firehosing telemetry).

## Picking the key by access pattern, not by convenience

The senior move is the same every time: **write down your top five queries by volume and your top five by business-criticality, look at which column they filter on, and shard by that column.** The shard key is a property of your *workload*, not your *schema*. Two systems with byte-identical schemas can need different shard keys because they are read differently.

| Workload | Dominant access pattern | Shard key | Why |
| --- | --- | --- | --- |
| Social network | "load this user's feed / profile / posts" | `user_id` | Nearly every read and write is scoped to one user; targeted single-shard queries. |
| B2B SaaS | "show this account's data" | `tenant_id` / `account_id` | All of a tenant's data co-locates; queries are tenant-scoped; enables local joins. |
| E-commerce orders | "show this customer's order history" | `customer_id` | The customer-facing query is the hot path; order lookups by ID are rarer. |
| IoT / telemetry | "stream this device's recent readings" | `device_id` | Per-device time series; hashing the device de-correlates the write hotspot. |
| Chat / messaging | "load this conversation's messages" | `conversation_id` | Messages are always read per-conversation; the conversation is the natural unit. |
| Ledger / payments | "this account's transactions" | `account_id` | Balance and statement queries are account-scoped; keeps a balance on one shard. |

### The `order_id`-versus-`customer_id` tension

The e-commerce row above hides the most common real argument I have refereed. An orders table is queried two ways: customers query "show me my orders" (`WHERE customer_id = ?`), and the system queries "look up this one order" (`WHERE order_id = ?`, e.g. from a payment webhook or a shipping update). You cannot make both single-shard with one key.

- **Shard by `customer_id`:** "my orders" is a clean single-shard query, which is great because it is the high-volume customer-facing path. But "look up order X" no longer knows which customer X belongs to, so it scatter-gathers across all shards — unless you carry the customer id around in every context that references an order.
- **Shard by `order_id`:** every single-order lookup is targeted and fast. But "my orders" now scatter-gathers, because one customer's orders are spread across every shard by the hash of their individual order ids. That punishes your hottest path.

The resolution is almost always **shard by `customer_id`, and make `order_id` carry the customer id** — either by embedding it (a composite or prefixed id like `cust42:ord881`) or by keeping a small global lookup table mapping `order_id → customer_id`. Then an order-only lookup does one tiny index hit to find the customer, then one targeted shard query. You traded a scatter-gather for one extra round trip, which is a trade you take every time.

### What to do when you genuinely query by multiple dimensions

Sometimes there is no single dominant access pattern — you really do query by `user_id` *and* by `email` *and* by `phone_number`, all at high volume, all needing single-shard latency. The shard key can only align with one of them. Your options, in rough order of preference:

1. **A secondary / global index.** Maintain a separate, independently sharded structure that maps the alternate key to the shard key: `email → user_id`. A lookup by email becomes two hops — index lookup, then targeted shard query. This is exactly how DynamoDB's Global Secondary Indexes and Vitess's lookup vindexes work. The cost is that the index must be kept consistent with the base table, usually asynchronously.
2. **Denormalization.** Store a copy of the data keyed differently. If you frequently need "all messages by this user" *and* "all messages in this conversation," you might keep two physical copies of the messages, one sharded by `user_id` and one by `conversation_id`, and write to both. You pay double storage and dual-write complexity for two fast read paths. This is the read-path equivalent of the cache patterns in [cache patterns in production](/blog/software-development/database-scaling/cache-patterns-in-production).
3. **A second sharded copy keyed differently.** The heavier version of (2): an entire replica of the dataset, sharded by the other key, fed by change data capture. This is operationally expensive but sometimes the only way to serve two genuinely independent high-volume access patterns at single-shard latency.

The thing all three have in common: you are *adding* a structure, not changing the base shard key. The base key stays aligned with the single most important access pattern, and the secondary structures buy you the others. Never try to satisfy two access patterns by compromising the shard key into something that aligns with neither.

## Composite and hierarchical keys

You are not limited to a single column. A **composite shard key** combines two or more columns, and the standard pattern is to route on a *prefix* while ordering on a *suffix*. This is how you co-locate related data and still keep it sorted for range scans within a shard.

![A composite key hashes only its prefix so a tenant's rows co-locate on one shard while the suffix orders them within it](/imgs/blogs/choosing-a-shard-key-7.webp)

The figure above shows the canonical SaaS pattern: a composite key `(tenant_id, created_at)`. The routing function hashes *only* the prefix — `hash(tenant_id)` — so all of tenant 42's rows land on the same shard regardless of when they were created. Within that shard, rows are stored sorted by the suffix `created_at`, so a query like "tenant 42's activity in the last week" is both single-shard (the prefix routes it) and a fast local range scan (the suffix orders it). You get the targeted-query win of sharding by `tenant_id` and the range-scan win of ordering by `created_at`, with no scatter-gather and no monotonic write hotspot across shards — because the *prefix* (the tenant) is what spreads writes, and tenants are not monotonic.

This is the structure behind Cassandra's partition-key-plus-clustering-column model and DynamoDB's partition-key-plus-sort-key model. The partition key (prefix) decides the shard; the clustering/sort key (suffix) decides the order within it. Choosing what goes in the prefix is *the* design decision — the prefix is your real shard key, and everything in the earlier sections applies to it.

```python
def composite_route(tenant_id: str, created_at: int, num_shards: int) -> int:
    """Route on the prefix only. The suffix never touches the routing
    decision — it exists to order rows *within* the chosen shard."""
    import hashlib
    h = hashlib.blake2b(tenant_id.encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big") % num_shards
    # created_at is intentionally unused here: it is the clustering key,
    # applied as a local sort *inside* the shard, not as a routing input.
```

The trap with composite keys is putting a monotonic column in the *prefix* — `(created_at, tenant_id)` instead of `(tenant_id, created_at)`. That reintroduces the monotonic write hotspot from Mistake 2, because now the routing decision depends on the timestamp. The rule of thumb: **the prefix must be the high-cardinality, evenly-distributed, query-aligned column; the suffix is for ordering only.**

## Hash versus range, applied to the key

Once you have chosen *which* column is the shard key, you still choose *how* to map its values to shards: hash the key, or split the key's value space into contiguous ranges. This is a genuine trade-off, and the same column behaves very differently under each. (The mechanics of both are covered in [database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding); here I focus on how the choice interacts with the key.)

![Hashing the key buys even spread but destroys range scans; ranging the key keeps scans local but invites hotspots on monotonic keys](/imgs/blogs/choosing-a-shard-key-6.webp)

The before/after above puts the two side by side.

**Hashing** the key — `hash(key) % N` — scrambles values across shards. Two keys that are adjacent in value (user 41 and user 42) land on unrelated shards. The wins: distribution is even by construction (a good hash spreads any input), and monotonic keys are *de-correlated*, so the write hotspot disappears — consecutive timestamps hash to scattered shards. The loss: range scans are dead. "All orders between Jan 1 and Jan 7" must touch every shard, because the dates are scattered everywhere. And naive `% N` hashing means that changing `N` rehashes almost everything, which is why you reach for consistent hashing instead (next section).

**Range** partitioning keeps the key's order. Shard 0 holds the lowest values, shard N the highest. The win: range scans are local — a contiguous key range maps to one or a few adjacent shards, so "yesterday's orders" hits one shard. You can also split and merge ranges to rebalance without a global rehash. The loss: a monotonic key sends every new write to the highest range, recreating the hotspot; and you must actively manage range boundaries to avoid skew, since data rarely distributes uniformly across the value space.

The decision rule:

| You need… | Choose | Because |
| --- | --- | --- |
| Even write distribution, point lookups | **Hash** | The hash spreads writes regardless of key shape; point lookups don't need order. |
| Range scans on the shard key | **Range** | Contiguous values stay on contiguous shards, so a range hits few shards. |
| To shard a monotonic key (timestamp/id) | **Hash** | Hashing breaks the monotonic write hotspot; range partitioning would create it. |
| Online rebalancing without a global rehash | **Range** (or consistent hash) | Range boundaries split/merge locally; plain modulo hashing moves almost everything. |

## Routing in code: modulo hashing and the consistent-hashing alternative

Here is the simplest possible shard router — the one almost everyone starts with — and the reason it does not survive contact with growth.

```python
def naive_route(key: str, num_shards: int) -> int:
    """Modulo hashing. Correct, fast — and brutal to resize."""
    import hashlib
    h = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big")
    return h % num_shards

# The problem: change num_shards and almost every key moves.
for n_from, n_to in [(4, 5)]:
    moved = sum(naive_route(str(k), n_from) != naive_route(str(k), n_to)
                for k in range(100_000))
    print(f"resize {n_from} -> {n_to}: {moved/100_000:.1%} of keys change shard")
# resize 4 -> 5: 80.0% of keys change shard
```

Going from 4 shards to 5 moves 80% of your keys. Every one of those keys is a row (or a million rows) that must be physically relocated, while the system is live. Modulo hashing makes adding a shard nearly as expensive as the original migration — which is precisely the irreversibility we are trying to avoid.

**Consistent hashing** fixes the resize cost. It maps both shards and keys onto a ring; a key belongs to the next shard clockwise from its position. Adding a shard only steals the keys between it and its clockwise neighbor — roughly `1/N` of the keys move, not nearly all of them. This is the mechanism behind Dynamo, Cassandra, and most modern sharded stores, and it has its own deep-dive in [consistent hashing and data partitioning](/blog/software-development/database/consistent-hashing-and-data-partitioning); the router below is the minimal version with virtual nodes for balance.

```python
import hashlib, bisect

class ConsistentHashRouter:
    """Consistent hashing with virtual nodes. Adding a shard relocates
    only ~1/N of keys instead of ~all of them under modulo hashing."""

    def __init__(self, shards, vnodes=150):
        self.vnodes = vnodes
        self.ring = {}          # hash position -> shard id
        self.sorted_positions = []
        for s in shards:
            self.add_shard(s)

    def _hash(self, value: str) -> int:
        return int.from_bytes(
            hashlib.blake2b(value.encode(), digest_size=8).digest(), "big")

    def add_shard(self, shard):
        for v in range(self.vnodes):       # many points per shard => even spread
            pos = self._hash(f"{shard}#{v}")
            self.ring[pos] = shard
        self.sorted_positions = sorted(self.ring)

    def route(self, key: str):
        pos = self._hash(key)
        i = bisect.bisect(self.sorted_positions, pos) % len(self.sorted_positions)
        return self.ring[self.sorted_positions[i]]

# Adding a shard moves only ~1/N of keys, not ~all of them:
r = ConsistentHashRouter([f"shard-{i}" for i in range(4)])
before = {str(k): r.route(str(k)) for k in range(100_000)}
r.add_shard("shard-4")
moved = sum(before[str(k)] != r.route(str(k)) for k in range(100_000))
print(f"add shard 5 (consistent hash): {moved/100_000:.1%} of keys change shard")
# add shard 5 (consistent hash): ~19.6% of keys change shard
```

The `vnodes=150` is doing real work: with one point per shard, the ring segments are wildly uneven and one shard can own twice the keyspace of another. Sprinkling 150 virtual points per shard smooths the segment sizes so each physical shard owns close to `1/N` of the ring. This is the same trick the production systems use, and it is why "consistent hashing" in practice always means "consistent hashing with virtual nodes."

## Is this key skewed? A distribution-analysis script

Before you commit, *measure*. The single highest-leverage thing you can do is take a representative sample of production data, compute where each row would land under your candidate key, and look at the distribution. Cardinality you can check with a `COUNT(DISTINCT)`; skew you have to simulate, because it depends on both the data and the routing function.

```python
import hashlib
from collections import Counter

def skew_report(keys, num_shards, vnodes_router=None):
    """Given an iterable of shard-key VALUES from a production sample,
    report how rows would distribute across shards and flag hotspots."""
    if vnodes_router is None:
        def route(k):
            h = int.from_bytes(
                hashlib.blake2b(str(k).encode(), digest_size=8).digest(), "big")
            return h % num_shards
    else:
        route = vnodes_router.route

    counts = Counter(route(k) for k in keys)
    total = sum(counts.values())
    ideal = total / num_shards

    # Per-shard load relative to a perfectly even split.
    loads = sorted(((counts.get(s, 0), s) for s in range(num_shards)), reverse=True)
    hottest, hottest_shard = loads[0]
    coldest, _ = loads[-1]

    # Hot-key check: does any single key value dominate?
    key_counts = Counter(keys)
    top_key, top_key_n = key_counts.most_common(1)[0]

    print(f"rows sampled      : {total:,}")
    print(f"ideal per shard   : {ideal:,.0f}")
    print(f"hottest shard {hottest_shard:>2}  : {hottest:,}  "
          f"({hottest / ideal:.2f}x ideal)")
    print(f"coldest shard     : {coldest:,}  ({coldest / ideal:.2f}x ideal)")
    print(f"hot/cold ratio    : {hottest / max(coldest, 1):.1f}x")
    print(f"single hottest key: {top_key!r} = {top_key_n:,} rows "
          f"({top_key_n / total:.1%} of all rows)")

    if hottest > 1.5 * ideal:
        print("VERDICT: SKEWED — hottest shard >1.5x ideal; expect a hotspot.")
    elif top_key_n > 0.05 * total:
        print("VERDICT: WHALE — one key value holds >5% of rows; isolate it.")
    else:
        print("VERDICT: balanced enough to proceed.")

# Simulated multi-tenant data with one whale tenant.
import random
random.seed(7)
tenants = ([f"t{random.randint(0, 5000)}" for _ in range(200_000)]   # long tail
           + ["t_whale"] * 120_000)                                  # one giant
skew_report(tenants, num_shards=16)
#   rows sampled      : 320,000
#   ideal per shard   : 20,000
#   hottest shard  9  : 122,113  (6.11x ideal)
#   coldest shard     : 11,402   (0.57x ideal)
#   hot/cold ratio    : 10.7x
#   single hottest key: 't_whale' = 120,000 rows (37.5% of all rows)
#   VERDICT: WHALE — one key value holds >5% of rows; isolate it.
```

Two checks matter here, and they catch different failures. The **per-shard load ratio** catches structural skew — a low-cardinality key, or unlucky hashing. The **single hottest key** check catches the whale — a high-cardinality key that is balanced on average but has one value that alone overwhelms a shard. The script above flags the whale explicitly because no amount of resharding fixes a single key value that is 37% of your data; you need the isolation strategy in the next section. Run this before you write a single migration. A `VERDICT: SKEWED` here is a cheap warning; the same condition discovered in production is a 3am page.

## The multi-tenant `tenant_id` trap

Multi-tenant SaaS is where shard-key choices get genuinely subtle, because `tenant_id` is simultaneously the obviously-correct key *and* a source of two opposing pressures. It is correct because every query is tenant-scoped — you almost never read across tenants, so sharding by `tenant_id` makes essentially every query single-shard and even enables local joins within a tenant's data. But it pulls in two directions at once.

### Even spread versus co-locating a tenant for local joins

If you *hash* `tenant_id`, you get even spread, but a single large tenant's data is still confined to one shard (all of tenant 42's rows hash to the same place), which is what lets you do local joins across that tenant's tables. That co-location is a feature — it is why tenant-scoped joins stay fast. The tension is not within hashing; it is that co-locating a tenant's data on one shard means that tenant's *size* now determines whether that shard is balanced. A median tenant co-locating onto a shard is fine. A whale co-locating onto a shard melts it. So the same property that makes tenant queries fast (co-location) makes whales dangerous.

### Bin-packing tenants onto shards

The fix for the common case — many small and medium tenants — is to stop thinking of routing as a pure hash and start thinking of it as **bin-packing**. Treat each shard as a bin with a capacity (disk, IOPS, QPS), treat each tenant as an item with a known size, and assign tenants to shards to keep the bins balanced. Instead of `hash(tenant_id) % N`, you maintain an explicit `tenant_id → shard_id` mapping table (a directory) and pack tenants greedily by size.

![Small tenants pack many-to-a-shard for balance while the whale gets a dedicated shard so it cannot melt its neighbours](/imgs/blogs/choosing-a-shard-key-8.webp)

The figure above shows the result. Shards A and B each hold several small or medium tenants packed to roughly equal total load. Tenant 42 — the whale, 1,000× the median — gets shard W *all to itself*. This is the directory-based or lookup-based sharding that Vitess, Citus, and most mature multi-tenant platforms converge on. The cost is an extra indirection (look up the tenant's shard before routing) and a directory you must keep consistent, but you buy two things hashing can't give you: explicit balance and explicit isolation.

```python
def bin_pack_tenants(tenant_sizes: dict, num_shards: int, whale_threshold_frac=0.5):
    """Greedy bin-packing of tenants onto shards, with whales isolated.
    tenant_sizes: {tenant_id: size}. Returns {tenant_id: shard_id}."""
    total = sum(tenant_sizes.values())
    ideal_per_shard = total / num_shards

    mapping = {}
    shard_loads = [0] * num_shards
    next_dedicated = num_shards  # whales get fresh shards beyond the pool

    # Largest first; whales that exceed a fraction of a shard get isolated.
    for tenant, size in sorted(tenant_sizes.items(),
                               key=lambda kv: kv[1], reverse=True):
        if size > whale_threshold_frac * ideal_per_shard:
            mapping[tenant] = next_dedicated      # dedicated shard, blast radius 1
            next_dedicated += 1
        else:
            target = min(range(num_shards), key=lambda s: shard_loads[s])
            mapping[tenant] = target
            shard_loads[target] += size
    return mapping
```

### Isolating whales

The whale strategy deserves its own emphasis because it is the difference between a stable multi-tenant system and one that pages constantly. **A whale tenant gets a dedicated shard.** Their load is real and you cannot hash it away; the only question is whether they share a shard with innocent bystanders (who then suffer when the whale spikes) or get isolated so the blast radius of their load is exactly one — themselves. Dedicated shards for your largest customers also align neatly with commercial reality: the biggest customers pay the most and demand the strongest isolation, so giving them their own infrastructure is a feature you can sell, not just an operational concession.

This is why the bin-packing approach beats pure hashing for multi-tenant systems at scale: hashing has no concept of a whale and will cheerfully drop your 1,000× tenant onto a shard with three small ones. A directory lets you *see* the whale in the size report and place it deliberately.

## Case studies from production

### 1. The status-column shard that could not be split

A payments team sharded a 2-billion-row transactions table by `transaction_status` because the engineer reasoned that most queries filtered on status ("show me all pending settlements"). Statuses were `{pending, settled, failed, refunded, disputed}` — five values. Within a month, `settled` was 88% of all rows and lived on one shard that was at 94% disk while four shards idled near 8%. The wrong first hypothesis was "we need bigger disks." The actual root cause was a five-value shard key: you cannot spread 88% of your data when it all carries the same key value. The fix was a full re-shard to `(merchant_id, created_at)` — a quarter of migration work that a thirty-minute cardinality check would have prevented. The lesson: a column being in the `WHERE` clause is necessary for alignment but worthless if its cardinality is five.

### 2. The auto-increment ID that capped writes at one shard

A logging platform range-partitioned events by an auto-increment `event_id`, expecting "recent events" range queries to be fast. They were — but the platform's write throughput plateaued at exactly the capacity of a single node no matter how many shards they added, because every `INSERT` got the next id and landed on the newest shard. Their wrong hypothesis was "we need faster disks on the write path." The root cause was a monotonic shard key under range partitioning: the newest shard was a permanent write hotspot and the rest were write-idle. The fix was to switch the routing to `hash(event_id)`, which de-correlated consecutive ids across shards and immediately distributed writes — at the cost of the recent-events range scan, which they rebuilt as a small materialized "recent" index. The lesson is the one from Mistake 2: a monotonic key makes reads look great in the demo while silently capping your write scalability.

### 3. The celebrity that melted a social shard

A social app sharded by `user_id`, which is the correct key, and ran fine for two years. Then a celebrity with tens of millions of followers joined, and their shard began to throttle reads for the few thousand ordinary users who happened to hash onto the same shard. The wrong hypothesis was "the shard hardware is failing." The actual cause was a whale: one `user_id` generating four orders of magnitude more fan-out reads and writes than the median. The fix was not a new shard key — `user_id` was right — but special-casing high-degree accounts onto dedicated capacity and serving their fan-out through a separate, heavily-cached read path. The lesson: the right key can still produce a hotspot when the distribution of *activity* across key values is heavy-tailed, and the fix is isolation, not re-keying.

### 4. The SaaS whale that took down five tenants

A B2B analytics product hashed `tenant_id` across 32 shards. It worked until they onboarded an enterprise customer roughly 800× their median tenant. That customer hashed onto a shard alongside five small tenants, and during the enterprise's nightly batch import, those five small tenants saw query latencies climb from 40ms to 6 seconds. The wrong first hypothesis was "the small tenants are doing something pathological." The actual cause was a co-located whale saturating the shared shard's IOPS. The fix was to move to a `tenant_id → shard` directory with bin-packing and to relocate the whale to a dedicated shard, which dropped the five victims back to 40ms overnight. The lesson: pure hashing has no concept of tenant size, so at any real customer-size spread you eventually need directory-based placement.

### 5. The email lookup that scatter-gathered every login

A consumer app sharded users by `user_id` — correct for the app's main read paths — but every login queried `WHERE email = ?`, which is not the shard key. With 64 shards, every single login broadcast to all 64, each ran the indexed email lookup, and a coordinator merged 64 results to find the one match. Login p99 latency was 900ms and tracked the slowest of 64 shards. The wrong hypothesis was "the email index is slow." The actual cause was a scatter-gather on a non-shard-key predicate. The fix was a global secondary index `email → user_id` (one small targeted lookup), turning login into two single-shard hops totaling under 30ms. The lesson: you do not fix a misaligned query by re-keying the whole table; you add a secondary index that maps the alternate key back to the shard key.

### 6. The composite key with the wrong prefix order

A multi-tenant time-series store used a composite key but ordered it `(bucket_timestamp, tenant_id)` — timestamp first — reasoning that time-bucketed scans would be efficient. Every write in a given time bucket therefore routed to the same shard, recreating a rolling write hotspot that migrated from shard to shard as the clock advanced. The wrong hypothesis was "we have a thundering-herd problem at bucket boundaries." The actual cause was a monotonic *prefix*: putting the timestamp first made the routing decision depend on time. The fix was to flip the key to `(tenant_id, bucket_timestamp)` so the high-cardinality tenant spread the writes and the timestamp became a local clustering order. The lesson from the composite-key section made concrete: the prefix is the real shard key, so it must be the high-cardinality, evenly-distributed column, never the monotonic one.

### 7. The modulo resize that moved 80% of the data

A team outgrew their 8-shard cluster and added two shards, going to 10. They used `hash(key) % N` routing, so changing `N` from 8 to 10 remapped roughly 80% of all keys to new shards. The "add two machines" task became a multi-week migration moving the overwhelming majority of a multi-terabyte dataset while serving traffic, with a painful consistency window during the move. The wrong hypothesis was "adding shards is always cheap." The actual cause was modulo hashing's resize behavior: it is optimal for lookups and pessimal for resizes. The fix going forward was to adopt consistent hashing with virtual nodes, after which subsequent shard additions moved only about `1/N` of the data. The lesson: the routing *function* is part of the shard-key decision, and plain modulo hashing quietly makes every future capacity change as expensive as the first migration.

## When to obsess over the shard key — and when to stop

### Obsess over the key when…

- You are about to shard a table that already holds large volumes of data, so getting it wrong means a full migration to fix. The cost of a bad choice scales with the data you already have.
- One access pattern clearly dominates your traffic and you can align the key to it — the payoff for choosing well is that nearly every hot query becomes single-shard.
- You run multi-tenant and your customers span orders of magnitude in size. Whales are inevitable; plan for directory-based placement and isolation from day one.
- Your key candidates include anything monotonic (timestamps, auto-increment ids, time-prefixed UUIDs). The hotspot is not a maybe; it is a certainty under range partitioning, so decide hashing-versus-range deliberately.
- You query the same data by multiple high-volume dimensions — design the secondary indexes or denormalized copies *before* you shard, not after the scatter-gathers start paging you.

### Stop and reconsider when…

- You have not yet exhausted the cheaper rungs — vertical scaling, read replicas, caching, functional partitioning. Sharding is the last resort, not the first reflex; revisit [the database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree).
- Your dataset fits comfortably on one well-provisioned primary with headroom. A single Postgres or MySQL instance handles far more than most teams assume; sharding a database that does not need it adds irreversibility for no benefit.
- No single column has high cardinality *and* aligns with your dominant query. If the data genuinely has no good shard key, that is a signal to reconsider your data model or your partitioning boundary, not to force a bad key.
- Your queries are overwhelmingly analytical (scans, aggregations across all data) rather than point or range lookups by a key. That is an OLAP workload; a columnar store or a separate analytics replica serves it far better than a sharded OLTP table.
- You are tempted to pick the key by what is convenient this sprint rather than what your workload demands. Convenience is exactly how every re-shard story in this post began.

The shard key is the rare engineering decision where the right amount of upfront analysis is "more than feels necessary." A day spent running cardinality and skew reports against real production data, listing your top queries, and arguing about `customer_id` versus `order_id` is a day that saves you a quarter-long migration. You get to make this choice cheaply exactly once, before the data lands. After that, every option costs you a migration. Choose by access pattern, measure the distribution, plan for the whale — and walk through the one-way door on purpose.

## Further reading

- [Database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding) — the mechanics this post builds on: what a shard is, how routing works, why fan-out hurts.
- [The database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree) — where sharding sits on the ladder and why you should exhaust the cheaper rungs first.
- [Consistent hashing and data partitioning](/blog/software-development/database/consistent-hashing-and-data-partitioning) — the ring, virtual nodes, and why `% N` is a trap at resize time.
- [Sharding strategies compared](/blog/software-development/database-scaling/sharding-strategies-compared) — hash, range, directory, and geo strategies side by side, once you have a key.
- [Resharding without downtime](/blog/software-development/database-scaling/resharding-without-downtime) — what it actually takes to walk back through the one-way door, if you must.
