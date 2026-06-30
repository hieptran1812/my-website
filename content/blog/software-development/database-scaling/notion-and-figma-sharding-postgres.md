---
title: "Sharding Postgres Late: Lessons from Notion and Figma"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "How Notion and Figma each ran one Postgres for years and then sharded under duress — Notion big-bang by workspace, Figma incrementally table-by-table — and the five lessons both playbooks share."
tags: ["database-scaling", "sharding", "postgres", "notion", "figma", "shard-key", "data-migration", "distributed-systems", "system-design", "query-routing"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 34
---

There is a particular flavor of engineering meeting I have sat in too many times. A growth chart is on the screen. The primary database is hot. Someone — usually the most senior person who has *not* recently been paged — says the word "shard," and the room exhales, because now there is a Plan. What nobody in that room knows yet is that they have just signed up for the single hardest, most irreversible infrastructure project their company will ever do, and that the version of it they are imagining (clean, big-bang, done in a quarter) is not the version they will actually ship.

The two best-documented examples of this exact journey are Notion and Figma. Both are products you have probably used. Both ran a *single* PostgreSQL database — one box, vertically scaled to the largest instance the cloud would sell them — for *years* longer than the internet's collective sharding folklore says is possible. And both, when they finally sharded, did it carefully, slowly, and very differently from each other. Notion (2021) sharded the whole monolith in one coordinated big-bang, partitioned by workspace. Figma (2023) refused the big-bang entirely and sharded one high-traffic *table* at a time behind a query-routing proxy.

This post is a side-by-side reading of those two playbooks. The interesting thing is not that they differ — it is what they *agree* on, because the agreements are the transferable lessons. Both sharded as late as they possibly could. Both sharded along their natural tenant entity. Both separated the *logical* cut from the *physical* move to de-risk it. Both centralized routing in one place instead of scattering shard math through the application. And both treated the migration itself — not the end-state topology — as the real engineering problem.

![Both Notion and Figma ran a single Postgres for years, then took different routes — big-bang by workspace versus incremental by table — through the same one-way door](/imgs/blogs/notion-and-figma-sharding-postgres-1.webp)

The diagram above is the mental model for this entire post. Read each row as a timeline. Both companies start in the same place — one Postgres, blue, the happy monolith. Both then buy time with *vertical* moves (bigger box, then splitting tables off onto their own databases — the amber stage). Only then, under real load pressure, do they reach the green stage and actually shard. Notion's green stage is a single dense event: `32 × 15 = 480` logical shards, one cutover, partitioned by workspace. Figma's green stage is a *sequence* of small events: shard one table, learn, shard the next. The rest of this article is a tour of that diagram, one stage at a time, for each company — and then the synthesis.

If you have not read the rest of this series, two posts set up the decision framework this case study lives inside: [the database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree) (why sharding is the *last* rung, not the first) and [choosing a shard key](/blog/software-development/database-scaling/choosing-a-shard-key) (the single most consequential decision in this whole exercise). This post is what those frameworks look like when two real companies run them under fire.

## Why "shard late" is the actual headline

> The most expensive sharding project is the one you start a year before you needed to.

Before any topology diagram, internalize the timing, because it is the lesson most teams ignore. The conventional wisdom — "Postgres doesn't scale, you'll need to shard eventually, so design for it early" — is *wrong in the way that matters*, and both Notion and Figma are proof.

Notion launched in 2016 and ran its core data on a single Postgres instance until 2021. Figma ran a single Postgres on AWS RDS — specifically, the largest physical instance RDS offered — from 2020, and only started horizontal sharding in 2023, after first spending a couple of years doing *vertical partitioning* (splitting heavy tables off onto their own databases, reaching roughly a dozen of them by the end of 2022). In both cases the company was already large, already valuable, already serving enormous traffic, on *one logical database* for the bulk of its growth.

Here is the assumption-versus-reality table that should govern your own timing:

| Common assumption | The Notion / Figma reality |
| --- | --- |
| "Postgres can't handle our scale, shard now." | A single well-tuned Postgres on a large instance serves enormous traffic for years. The ceiling is real but *far* away. |
| "Shard early so the migration is small." | The migration is never small. Sharding early means migrating *and* carrying shard complexity through years of feature work, with a less-informed shard key. |
| "We'll just turn on sharding." | Both companies treated the *migration* — double-write, backfill, verify, cutover — as the bulk of the project. The topology was the easy part. |
| "Once sharded, we're done." | Notion re-sharded in 2023 (32 → 96 instances). Figma's incremental approach is *designed* to never be "done." Sharding is a capability you operate, not a milestone you hit. |
| "The shard key is a technical detail." | It is *the* decision. Pick wrong and you scatter every query; you cannot change it later without re-migrating everything. |

The reason "shard late" wins is that sharding is a **one-way door** (the decision-tree post hammers this). The instant you shard:

- Cross-shard `JOIN`s and transactions become application problems, not database features.
- Foreign-key constraints across the shard boundary disappear; referential integrity becomes your job.
- Analytics that used to be one `SELECT` now fans out to N shards or needs a separate pipeline entirely.
- Every new feature must answer "which shard does this live on?" before it can answer anything else.

You pay that tax forever. Paying it a year early — while you still have less product, less data to reason about, and a worse understanding of your access patterns — is pure loss. Both companies waited until the single box was genuinely, measurably out of runway, and then moved. That discipline is the headline. Everything below is mechanics.

## Part 1 — Notion: big-bang by workspace

### 1.1 The topology: 480 logical shards on 32 machines

**Senior rule of thumb: decouple your logical shard count from your physical instance count, and make the logical count large and fixed.**

When Notion sharded, they did *not* choose "32 shards" to match 32 database servers. They chose **480 logical shards**, implemented as 480 PostgreSQL *schemas*, and packed **15 schemas onto each of 32 physical instances**. Each schema holds its own copy of the partitioned tables — `schema001.block`, `schema002.block`, and so on up to `schema480.block`.

![Notion packed fifteen logical schema shards into each of thirty-two physical Postgres instances, fixing the shard count at 480 while leaving instance count free to change](/imgs/blogs/notion-and-figma-sharding-postgres-2.webp)

The diagram shows three representative instances out of the 32. Instance 1 holds schemas 001–015, instance 2 holds 016–030, and the last one holds 466–480. The math at the bottom is the entire point: `32 instances × 15 schemas/instance = 480 logical shards`.

Why this indirection? Because it makes the painful operation — *re-balancing* — cheap. The number 480 is chosen once and *never changes*. The thing that maps a workspace to a logical shard is `hash(workspace_id) % 480`, and that formula is frozen forever. What *can* change is which physical box hosts which schema. If you need more capacity, you spin up new instances and *move schemas* to them — moving bytes, not re-hashing keys. No row ever needs its shard recomputed.

This is exactly what happened in Notion's 2023 re-shard. They went from **32 instances to 96 instances** (a 3× expansion) to relieve CPU (some shards were hitting 90%+ at peak), disk IOPS, and PgBouncer connection-limit pressure. The logical shard count stayed at 480 — each instance now holds **5 schemas** instead of 15 (`96 × 5 = 480`). Because the hash function and shard count were untouched, this was a data-movement exercise, not a re-keying nightmare. **That is the dividend of the indirection.** Had they sharded into exactly 32 shards, the 2023 re-shard would have required re-hashing and re-migrating every single row — the very thing the original migration was so careful to do exactly once.

A quick rule for choosing the fixed logical count: pick it large enough that you will never want *more* logical shards (re-keying is the thing you are avoiding), and divisible by many instance counts you might plausibly run. 480 = 2⁵ × 3 × 5 divides evenly by 32, 48, 60, 80, 96, 120, 160, 240 — a generous menu of future instance counts, each yielding a whole number of schemas per box.

### 1.2 The shard key: why workspace is the natural entity

**Senior rule of thumb: shard on the entity your queries already filter by, so the shard key falls out of the access pattern instead of being imposed on it.**

Notion's data model is famously a tree of **blocks**. A page is a block; the paragraphs and images and tables inside it are blocks; the rows of a database are blocks. Everything is a block, and every block belongs to exactly one **workspace**. Notion is a team product: you work inside your company's workspace, and you almost never read across workspace boundaries.

That last sentence is the whole shard-key argument. Because users query within a single workspace at a time, partitioning on `workspace_id` means a normal request touches exactly one shard.

![Every block belongs to one workspace, so hashing workspace_id sends a workspace's entire block tree to a single shard and keeps reads single-shard](/imgs/blogs/notion-and-figma-sharding-postgres-3.webp)

The figure traces it: the `workspace` (Acme Inc.) roots a tree of page and database blocks, each of which roots its own children, and the *entire tree* hashes to one shard — `shard 042 = hash(workspace_id) % 480`. Loading a page, expanding a sub-page, querying a database view — all of it stays on that one shard. No scatter-gather, no cross-shard join, no fan-out.

Here is the routing function, in the shape Notion's application code took. The critical detail is that **every block carries a `space_id`** (workspace id) so the router can find the shard from the block itself, without a lookup:

```python
NUM_LOGICAL_SHARDS = 480

# Frozen at migration time. Never changes — that is the whole point.
def logical_shard_for(workspace_id: str) -> int:
    # Stable hash of the workspace id, modulo the fixed shard count.
    # mmh3 (MurmurHash3) is non-cryptographic, fast, and stable across
    # processes — unlike Python's built-in hash(), which is salted per run.
    import mmh3
    return mmh3.hash(workspace_id, signed=False) % NUM_LOGICAL_SHARDS

# The logical -> physical map is config, not code. Re-balancing edits
# this table; the hash above is untouched.
SCHEMA_TO_INSTANCE = {
    # schema_id : (host, schema_name)
    1:   ("pg-shard-01.internal", "schema001"),
    2:   ("pg-shard-01.internal", "schema002"),
    # ... 15 schemas on instance 01 ...
    16:  ("pg-shard-02.internal", "schema016"),
    # ... and so on to 480 ...
}

def connection_for(workspace_id: str):
    shard = logical_shard_for(workspace_id)        # 0..479
    schema_id = shard + 1                            # schemas are 1-indexed
    host, schema = SCHEMA_TO_INSTANCE[schema_id]
    conn = get_pool(host)                            # pooled per physical host
    conn.execute(f"SET search_path TO {schema}")     # route to the right schema
    return conn

# Reads and writes always start from a workspace_id (or a block that
# carries its space_id), so routing is a pure function of data the
# request already has.
def load_block(block) -> dict:
    conn = connection_for(block.space_id)
    return conn.fetch_one(
        "SELECT * FROM block WHERE id = %s", [block.id]
    )
```

The design tension worth naming: this works *because* the access pattern cooperates. If Notion's product encouraged constant cross-workspace queries (say, a global search over every workspace you have ever touched), workspace-sharding would turn those queries into 480-way scatter-gathers and the key would be a disaster. It works here because the *product shape* and the *shard key* agree. That alignment is not luck — it is the criterion. (The [shard-key post](/blog/software-development/database-scaling/choosing-a-shard-key) generalizes this: the right shard key is the entity that bounds your transactions and your hot read paths.)

One real wrinkle: workspaces are not uniform. A 5-person startup's workspace and a 50,000-seat enterprise's workspace both hash to a single shard, so a few enormous workspaces can create hot shards — the classic "celebrity problem" applied to tenants. Notion's mitigation lives in the logical/physical indirection: a hot schema can be moved to its own beefier instance, or isolated, without touching the hash. This is the same lever Instagram pulls with its ID scheme — see [Instagram's sharded IDs in Postgres](/blog/software-development/database-scaling/instagram-sharding-ids-in-postgres) for the complementary trick of *encoding* the shard into the primary key so a row is self-locating.

#### Second-order optimization: the schema-as-shard trick

Using a Postgres *schema* per logical shard (rather than a separate database or a `WHERE shard_id = ?` column) is an underrated choice. It means the table DDL is identical across all 480 shards, `search_path` switches routing with zero query rewriting, and a single physical instance can host many shards while each keeps clean, independent table statistics and indexes. The cost is that 480 schemas × N tables is a lot of relations for the catalog, and DDL migrations must be applied 480 times — which is why Notion built tooling to run schema migrations across all shards as a routine operation. If you ever do this, budget for the migration-fan-out tooling *before* you shard, not after.

### 1.3 The migration: double-write, backfill, verify, flip

**Senior rule of thumb: keep the old database authoritative until verification proves the new one is correct — make the cutover the last thing that can possibly go wrong, not the first.**

This is the part that was actually hard, and it is the most transferable piece of the entire Notion story because the *shape* of this migration recurs in every sharding project ever done well.

![Notion's migration kept the old database authoritative through double-write, backfill, and verify, and only flipped traffic in a five-minute window after dark reads agreed](/imgs/blogs/notion-and-figma-sharding-postgres-4.webp)

Four phases, in order:

1. **Double-write.** Every incoming write is applied to *both* the old monolith and the new sharded cluster. The old database remains the source of truth; the new cluster is a shadow that must be kept current. This phase can run for as long as you need — it is steady-state, not a deadline.
2. **Backfill.** New writes are covered by double-write, but historical rows are not. A backfill job replays existing data into the sharded cluster. Notion used an audit-log approach and the backfill took roughly **three days**. Crucially, backfill and double-write run concurrently, and the backfill must be *idempotent* — it can race with live double-writes on the same row, so "last write wins by version/timestamp" or upsert-on-conflict is mandatory.
3. **Verify.** Before trusting the new cluster, prove it matches. Notion used two techniques: **random UUID sampling** (pick random records, compare old vs new field-by-field) and **dark reads** — serving the read from the old database as authoritative while *also* reading from the new cluster and comparing, logging every discrepancy without affecting the user. Dark reads are the gold standard because they verify against the *exact* production read path under real traffic, not a synthetic sample.
4. **Flip.** Only after verification is clean do you cut over. Notion did this in a **five-minute maintenance window**: stop writes, let double-write drain, switch the source of truth to the sharded cluster, resume. Five minutes of partial unavailability for a migration of this magnitude is a genuinely excellent result.

The piece that makes the whole thing *safe* rather than merely careful is the **reverse audit log**: Notion prepared a rollback path so that if verification surfaced problems after the flip, they could replay writes back into the old database and flip back. Sharding is a one-way door at the *strategic* level, but a well-built migration keeps the *tactical* door open right up until you are certain.

Here is the double-write + dark-read skeleton, which is the heart of any migration like this:

```python
class MigratingRepository:
    """Old DB is authoritative; new sharded cluster is shadowed and verified."""

    def __init__(self, old_db, sharded, metrics, phase: str):
        self.old = old_db
        self.sharded = sharded
        self.metrics = metrics
        self.phase = phase  # "double_write" | "verify" | "cutover"

    def write_block(self, block):
        # Phase 1+: write old first (source of truth), then shadow to new.
        self.old.upsert_block(block)
        try:
            # Idempotent upsert keyed on (id, version) so it can race the
            # backfill safely; a stale write loses to a newer version.
            self.sharded.upsert_block(block)
        except Exception as e:
            # A shadow-write failure must NOT fail the user's request while
            # the old DB is still authoritative. Log it; reconcile via backfill.
            self.metrics.incr("shadow_write_failed")
            log.warning("shadow write failed for %s: %s", block.id, e)

    def read_block(self, block_id, space_id):
        authoritative = self.old.get_block(block_id)
        if self.phase == "verify":
            # Dark read: fetch from the new cluster on the real read path and
            # compare, but ALWAYS return the old (authoritative) result.
            shadow = self.sharded.get_block(block_id, space_id=space_id)
            if not records_equal(authoritative, shadow):
                self.metrics.incr("dark_read_mismatch")
                log.warning("mismatch for %s: %r != %r",
                            block_id, authoritative, shadow)
        return authoritative
```

When `dark_read_mismatch` holds at zero across a representative window of real traffic — every read path, not just the common ones — you have earned the right to flip.

#### Second-order optimization: verify the *read path*, not the *rows*

A subtle trap: it is tempting to "verify" by checksumming both tables row-for-row in the background. That catches storage-level drift but *misses application-level bugs* — a router that sends a query to the wrong shard, a serialization mismatch, an index that exists on old but not new. Dark reads catch these because they exercise the actual production query, router, and deserialization. Always verify the thing your users will actually run. The migration discipline here is the same one covered in depth in [resharding without downtime](/blog/software-development/database-scaling/resharding-without-downtime); Notion's playbook is the canonical worked example of it.

## Part 2 — Figma: incremental by table

### 2.1 The strategic choice: refuse the big-bang

**Senior rule of thumb: when the end state is irreversible, decompose the migration into the smallest reversible steps you can, even if it makes the project longer.**

Figma reached the same wall — a single Postgres on RDS, with a few of the biggest, hottest tables threatening the instance's limits — but drew the opposite conclusion about *how* to cross it. Notion's big-bang works when you have one dominant entity (the workspace) that cleanly partitions essentially all of your data at once. Figma's data did not have that single clean cleavage: different tables wanted different shard keys (some keyed by `FileID`, some by `UserID`, some by `OrgID`). A single big-bang sharded by one key would have been wrong for most of the tables.

So Figma chose to shard **one table at a time**.

![Figma rejected the all-at-once cutover and sharded one high-traffic table at a time, so any single table could roll back without a full-database migration](/imgs/blogs/notion-and-figma-sharding-postgres-5.webp)

The contrast in the figure is the strategy. On the left, the rejected big-bang: convert the whole monolith in one giant cutover, weeks of risk, and a rollback that means reverting *everything* — a true one-way door with no exit. On the right, the chosen incremental path: shard a single, deliberately *simple but very high-traffic* table first to prove the entire serving and routing stack in production; then shard the next table, and the next, each on its own schedule, each independently reversible. Figma's first sharded table went live in September 2023 with only about **ten seconds** of partial availability on the database primaries — and because the approach is incremental, every subsequent table inherits that proven, low-risk cutover.

The reason this matters is not aesthetic. It is risk arithmetic. A big-bang migration's blast radius is *all your data*; if it fails, everything is in jeopardy and the rollback is enormous. An incremental migration's blast radius is *one table*; if it fails, you revert one table and lose almost nothing. You trade a longer calendar (Figma's full effort from planning to first sharded table was roughly **nine months**) for radically smaller per-step risk. When the downside of a single mistake is "we corrupted the whole production database," buying down that downside with time is the correct trade every time.

There is a second, quieter benefit: **you learn on the cheap tables.** The first table Figma sharded was chosen because it was simple — so the team could debug the routing layer, the cutover tooling, and the operational runbook against a forgiving target before pointing all of it at a gnarly, deeply-joined table. By the time the hard tables come up, the migration machinery is battle-tested.

### 2.2 The routing layer: DBProxy

**Senior rule of thumb: shard logic belongs in exactly one place — a routing layer — never scattered through application code.**

Figma's most architecturally interesting decision is **DBProxy**: a Golang service that sits between the application and the databases, intercepts the SQL the application generates, and routes each query to the correct physical Postgres. The application keeps writing ordinary-looking SQL; DBProxy alone knows the shard map.

![DBProxy parses each SQL query into an AST, extracts the shard key, maps the logical shard to a physical database, and routes — so the application never learns which box holds a row](/imgs/blogs/notion-and-figma-sharding-postgres-6.webp)

The pipeline in the figure has four stages inside DBProxy:

1. **Query parser** — turns the incoming SQL string into an Abstract Syntax Tree (AST). You cannot route reliably by regex-matching SQL; you need to actually parse it to find the `WHERE` clause, the table, the predicates.
2. **Logical planner** — walks the AST to determine the query type and extract the **logical shard ID** from the shard-key predicate. A query with `WHERE file_id = 7` yields "this is shard 13" (where `13 = hash(7)` mapped into the shard space).
3. **Physical planner** — maps the logical shard to the physical database currently hosting it (`shard 13 → db04`), using the logical-to-physical map.
4. **Route** — send the query to that database, get the result, return it to the app.

The win is the same indirection Notion gets from its schema map, but enforced by a *service* rather than a library: the application never names a physical box. Re-mapping shard 13 from `db04` to `db09` is a configuration change *inside DBProxy*, invisible to every caller. And because routing is centralized, shard keys can be hashed centrally — Figma uses `hash(UserID)`, `hash(FileID)`, or `hash(OrgID)` depending on the table — to spread load uniformly and avoid hotspots, without any application code knowing or caring.

Here is a sketch of the DBProxy routing core. The real thing is far more sophisticated (it handles transactions, aggregations, prepared statements, and far more), but the bones are this:

```go
// DBProxy routing core (simplified). Parses SQL, extracts the shard key,
// maps logical -> physical, and routes. The app sends ordinary SQL.

type Router struct {
    shardCount  int
    logical2phys map[int]string // logical shard id -> physical DB DSN; this is CONFIG
}

func (r *Router) Route(rawSQL string) (db string, err error) {
    // 1. Parse SQL into an AST. A real parser, not a regex — we must
    //    understand the query to route it correctly.
    ast, err := sqlparser.Parse(rawSQL)
    if err != nil {
        return "", fmt.Errorf("parse: %w", err)
    }

    // 2. Logical plan: find the table and pull the shard-key value out of
    //    the WHERE clause. Which column is the shard key depends on the table.
    table, shardCol := shardKeyForTable(ast)        // e.g. "files" -> "file_id"
    keyVal, ok := extractEquality(ast, shardCol)    // e.g. file_id = 7  -> 7
    if !ok {
        // No shard key in the predicate => can't single-route. Either
        // reject (preferred) or scatter-gather (expensive, last resort).
        return "", ErrUnshardableQuery
    }

    // 3. Compute the logical shard, then map to the physical DB. The hash is
    //    frozen; only logical2phys changes when we re-balance.
    logical := int(hash64(keyVal)) % r.shardCount
    phys, ok := r.logical2phys[logical]
    if !ok {
        return "", fmt.Errorf("no physical db for logical shard %d", logical)
    }
    _ = table
    return phys, nil // 4. caller executes rawSQL against `phys`
}
```

The single most important line is `r.logical2phys` being *config, not code*. That map is the lever you pull to re-balance, to migrate a shard onto new hardware, or to roll back — without redeploying the application.

#### Second-order optimization: the unshardable query is a feature, not a bug

Notice `ErrUnshardableQuery`. A query with no shard key in its predicate *cannot* be routed to a single shard — it would have to fan out to all of them. The disciplined move is to make those queries *loud*: reject them, or at least alarm on them, so that an engineer who accidentally writes a cross-shard query finds out in code review or staging instead of melting production with a 480-way scatter-gather. A routing layer is the natural place to enforce "every query must be shardable," which is a constraint you genuinely want.

### 2.3 Logical sharding before physical sharding

**Senior rule of thumb: separate "did we route correctly?" from "did we move the data?" — verify the first on one box before risking the second across many.**

This is Figma's de-risking masterstroke and the idea most worth stealing. Sharding conflates two scary things: a *logical* change (queries now route by shard key through DBProxy) and a *physical* change (data now lives on multiple boxes). Doing both at once means that when something breaks, you cannot tell which change broke it.

So Figma split them. **First**, they implemented logical sharding *while all the data still lived on one physical database*, using PostgreSQL **views** to represent the shards. DBProxy routed queries to these views exactly as if they were separate shards — but every view pointed back at the same single physical Postgres. This let them validate the *entire* serving stack — the parser, the logical planner, the physical planner, the rollout machinery — against a low-risk target. Rollout was **percentage-based**: send 1% of traffic through the new logical-sharding path, then 10%, then 100%, watching for errors. And the rollback was trivial: a logical-sharding bug was **a single configuration change to flip back**, because *no data had moved*.

![Figma proved its routing stack on one physical database using views, with a percentage rollout and config-flip rollback, before promoting any shard onto separate hardware](/imgs/blogs/notion-and-figma-sharding-postgres-7.webp)

**Only after** logical sharding held in production did Figma do the physical move — relocating shards onto separate databases, one at a time, each with about ten seconds of partial availability. By the time bytes were moving, the routing was already proven correct. The physical move "only relocates bytes," as the figure puts it; all the *logic* risk was retired in phase 1, on one machine, with an instant rollback.

Contrast the rollback stories:

| Step | Logical sharding (phase 1) | Physical sharding (phase 2) |
| --- | --- | --- |
| What changes | Query routing through DBProxy, via views on one DB | Data physically relocated to separate DBs |
| Blast radius | One DB; no data moved | One shard's data in motion |
| Rollback | Flip one config flag — instant, reversible | Stop the per-shard migration; data already on new box stays put |
| What it proves | Parser, planner, rollout machinery all correct | Capacity actually distributes across hardware |
| Risk if it breaks | Wrong routing, but data intact and recoverable | Data-in-motion bugs — the genuinely dangerous category |

By front-loading all the *logic* risk into the phase with the *cheapest* rollback, Figma made the dangerous phase (moving data) boring. This is the same instinct behind Notion's double-write-then-flip — keep the reversible thing reversible for as long as possible — applied at a different seam.

### 2.4 Colocation: keeping related tables together

**Senior rule of thumb: tables that are joined or transacted together must share a shard key, or you re-introduce the cross-shard joins you sharded to avoid.**

A wrinkle the per-table approach forces you to confront: if `files` shards by `file_id` and `file_comments` also needs to be queried alongside files, they had better live on the *same* shard, or every "load a file and its comments" query becomes a cross-shard join. Figma's answer is **colos** (colocations): groups of related tables that share the same shard key and therefore always land on the same physical shard. A colo is a friendly developer abstraction — "these tables travel together" — that preserves the ability to do single-shard joins across them.

This is the per-table approach's version of Notion's "the whole block tree lands on one shard." Notion gets colocation *for free* because everything is keyed by workspace. Figma has to *engineer* colocation per table group, because its tables don't all share one natural key. That extra work is the price of the per-table flexibility — and it is a price worth naming before you choose the incremental path, because getting colocation wrong silently re-introduces the exact cross-shard cost you were trying to escape.

## Part 3 — The synthesis: what both playbooks agree on

Now the payoff. Here is the head-to-head, and then the five shared lessons that survive the differences.

![Notion and Figma agree on sharding the dominant tenant entity and centralizing routing, but diverge on big-bang versus incremental cutover and on analytics offload](/imgs/blogs/notion-and-figma-sharding-postgres-8.webp)

| Dimension | Notion (2021) | Figma (2023) |
| --- | --- | --- |
| **When** | Single Postgres ~6 years, then sharded | Single RDS instance, vertical-partitioned first (~dozen DBs), then sharded |
| **What was sharded** | The whole monolith, all partitioned tables at once | One high-traffic table at a time |
| **Shard key** | `workspace_id` (one natural tenant entity) | `hash(FileID)` / `hash(UserID)` / `hash(OrgID)`, per table |
| **Routing layer** | Application-layer code (schema `search_path` map) | DBProxy — a dedicated Go service parsing SQL to an AST |
| **Cutover style** | Big-bang, one ~5-minute maintenance window | Incremental, ~10 seconds partial availability per table |
| **Logical-before-physical** | Double-write + backfill + dark-read verify, then flip | Logical sharding via views on one DB, then physical move |
| **Topology** | 32 instances × 15 schemas = 480 logical shards | Per-table physical shards, colocated by shard key into colos |
| **Re-balancing** | Move schemas across instances (2023: 32 → 96 instances) | Update DBProxy's logical→physical map |
| **Analytics offload** | Separate data lake (Debezium CDC → Kafka → Hudi → S3) | Handled as a separate effort |

The differences are real and instructive — big-bang versus incremental is a genuine fork, and which side you take depends entirely on whether you have one clean tenant key (Notion) or many table-specific keys (Figma). But step back and the **agreements** are louder than the disagreements:

**1. Shard late.** Both rode a single Postgres for years, buying time vertically until the box was genuinely out of runway. Neither sharded preemptively. This is the lesson that saves the most teams the most pain, and it is the one most often ignored.

**2. Shard by the dominant entity.** Notion's workspace and Figma's file/org are *tenant* boundaries — the unit your product naturally isolates data by, and the unit your queries naturally filter by. Sharding on the tenant keeps the common request single-shard. The shard key is not invented; it is *discovered* in the access pattern. (This is the central thesis of [choosing a shard key](/blog/software-development/database-scaling/choosing-a-shard-key).)

**3. Logical before physical.** Both separated the routing change from the data-movement change, and verified the cheap-to-roll-back part first. Notion: double-write and dark-read verify while the old DB stays authoritative. Figma: logical sharding via views on one box before moving bytes. Same instinct, two seams.

**4. Centralize routing.** Neither scattered `hash(key) % N` through a thousand call sites. Notion put it in one application module with a config-driven schema map; Figma put it in a dedicated proxy service. Both made the logical→physical mapping *configuration*, so re-balancing and rollback never require touching application logic.

**5. The migration is the project.** Both companies' war stories are 90% about *how to move live data safely* — double-write, backfill, verify, percentage rollout, instant rollback — and 10% about the end-state topology. The topology is the easy part. The migration is where the engineering lives, and where projects die.

> Shard late, shard by your natural tenant entity, separate logical from physical, centralize routing, and treat the migration as the real work. The end-state topology is a footnote next to the discipline of getting there.

### A note on analytics: the other half of the story

One agreement hides inside a difference. When you shard, your analytical queries — the `GROUP BY` over all tenants, the company-wide dashboard, the data-science export — become impossible to serve from the operational store, because they inherently fan out across every shard. Both companies solved this by *moving analytics off the operational database entirely*. Notion's solution is the well-documented one: an in-house **data lake** built on Debezium change-data-capture into Kafka, Apache Hudi for incremental ingestion into S3, with Snowflake and other product-facing stores downstream. Moving tens-of-terabytes Postgres datasets off the operational tier and into the lake saved them over a million dollars in 2022 alone and made analytics that would have been 480-way scatter-gathers into ordinary lake queries.

The lesson generalizes: **sharding the operational store and offloading analytics are two halves of the same decision.** The moment you shard, you have implicitly signed up to build (or buy) an analytics offload, because the queries you used to run against the monolith no longer have a single place to run. Plan for both, or the analytics team will discover the second half the hard way the week after you cut over.

## Case studies from production

These are composites and well-documented incidents that illustrate the failure modes the Notion and Figma playbooks are designed to prevent. The names of the patterns matter more than any one company.

### 1. The premature shard

A Series-A startup, ~30 engineers, reads a "we sharded at scale" blog post and decides to shard their primary Postgres *now*, at 200 GB and 2,000 QPS — a load a single `db.r6g.4xlarge` handles without sweating. Six months later they have a sharded cluster, a shard key chosen before they understood their access patterns (they picked `user_id`; half their queries are actually org-scoped and now fan out), no analytics pipeline, and a primary that was never the bottleneck — a single slow query on an unindexed column was. **Root cause:** sharding was rung one instead of rung last. **Fix:** in this real-shaped case, the team eventually *un-sharded* back to a single large instance plus read replicas, fixed the index, and bought two more years of runway. **Lesson:** the decision tree exists precisely so you climb the cheap rungs first; Notion and Figma's years on one box are the proof that the cheap rungs go *far*.

### 2. The hot tenant

A B2B SaaS shards by tenant — correct entity, by the book. Then they sign a whale: a single customer 50× larger than the next biggest. That customer's tenant hashes to one shard, and that shard is now at 95% CPU while the other 479 idle. **Wrong first hypothesis:** "our shard key is bad, we need to re-key." **Actual root cause:** the key is fine; the *distribution* is skewed, the celebrity problem applied to tenants. **Fix:** move the hot tenant's logical shard onto its own dedicated, larger instance — exactly the lever Notion's logical/physical indirection provides. No re-keying, no re-migration, just a schema relocation. **Lesson:** decouple logical shards from physical instances precisely so that the answer to a hot tenant is "move a schema," not "re-shard everything."

### 3. The cross-shard join that ate the latency budget

A team shards `orders` by `customer_id` and `inventory` by `warehouse_id` — two different keys, both reasonable in isolation. Then a new feature needs "all orders for products in this warehouse," which joins across both keys. There is no single shard that has both; the query becomes a fan-out to every order shard, gathering and joining in the application. p99 latency for that endpoint goes from 40 ms to 2.3 seconds. **Root cause:** two tables that needed to be queried together were sharded on different keys — no colocation. **Fix:** Figma's `colo` concept — had `inventory` been colocated with `orders` under `customer_id`, or had the access pattern been recognized up front, the join would have stayed single-shard. **Lesson:** colocation is not optional polish; tables you join must share a shard key, or you re-import the cross-shard cost you sharded to escape.

### 4. The big-bang with no rollback

A mid-size company attempts a big-bang shard of their whole database in one weekend cutover. Backfill runs long, the maintenance window blows past its budget, and partway through the cutover they discover a serialization bug: a JSON column deserializes differently on the new cluster. They have already pointed writes at the new cluster. There is *no reverse path* — nobody built the reverse audit log. **Root cause:** an irreversible big-bang with no rollback mechanism. **Fix (after the fact):** a painful manual reconciliation and 9 hours of degraded service. **Lesson:** this is the precise failure Notion's reverse audit log and Figma's config-flip rollback exist to prevent. If your cutover has no rollback, you do not have a migration plan — you have a bet.

### 5. The salted-hash router bug

A team writes their shard router using Python's built-in `hash()` for the shard-key hash: `hash(key) % N`. It works perfectly in tests and in their first production deploy. Then, weeks later, after a routine restart, reads start missing — rows written before the restart route to a *different* shard than the same key routes to now. **Root cause:** Python's `hash()` for strings is *salted per process* (PYTHONHASHSEED), so the mapping is not stable across process restarts. **Fix:** swap to a stable non-cryptographic hash (MurmurHash3 / `mmh3`, or an explicit SHA truncation) — the reason the routing code above uses `mmh3`. **Lesson:** your shard hash must be *stable across processes, machines, and language versions, forever*. Test it by hashing the same key from two freshly-started processes and asserting equality.

### 6. The re-shard that wasn't a re-key (Notion, 2023)

Notion's shards started hitting 90%+ CPU at peak, disk IOPS saturated, and PgBouncer connection limits became the wall. They needed more capacity. Because they had fixed the logical shard count at 480 and decoupled it from the 32 physical instances, the 2023 expansion to **96 instances** was a *data-movement* exercise — relocate schemas so each box holds 5 instead of 15 — not a re-key. No row's shard was recomputed; the hash and the count were untouched. **Root cause of the pressure:** organic growth, the normal reason to add capacity. **Why it was painless:** the original design's logical/physical indirection. **Lesson:** the choice that makes re-sharding bearable is made *at the original sharding time*, by fixing a large logical shard count and treating physical placement as movable. Pay that small design cost once and re-balancing is forever cheap.

### 7. The verification that checked the wrong thing

A team migrating to a sharded cluster "verifies" by running a nightly job that checksums every table on old and new and alerts on mismatch. Checksums match perfectly for two weeks. They cut over. Immediately, a specific endpoint returns wrong data — a query the router sends to the wrong shard for a particular predicate shape. The checksums never caught it because they compared *stored bytes*, not the *served query*. **Root cause:** verifying storage instead of the read path. **Fix:** dark reads — run the real production query against both stores and compare results, as Notion did. **Lesson:** verify the thing your users run, not the thing on disk. A router bug, a missing index, a deserialization difference — all invisible to a row checksum, all caught by a dark read.

### 8. The analytics cliff

A company shards their operational database cleanly and successfully. The migration is a triumph. Two weeks later, the data team's nightly revenue dashboard — a `GROUP BY country` over the whole `orders` table — has no single place to run. It now has to query all 256 shards and aggregate, taking 40 minutes and hammering the production shards during the run. **Root cause:** analytics was never planned as part of the sharding project. **Fix:** stand up an analytics offload — CDC into a lake or warehouse, the way Notion built its Debezium → Kafka → Hudi → S3 pipeline — so cross-tenant analytics run off the operational tier entirely. **Lesson:** sharding the operational store *forces* an analytics offload. They are one decision in two parts; budget for both or get surprised by the second.

## When to reach for each playbook

### Reach for Notion's big-bang-by-tenant when:

- You have **one dominant tenant entity** (workspace, org, account) that cleanly partitions essentially all of your data, and your queries almost always filter by it.
- That entity gives you **colocation for free** — related data is keyed by the same tenant, so single-shard joins just work.
- You can afford a **short, coordinated maintenance window** (minutes) and can build a real reverse-rollback path before the cutover.
- You want the operational simplicity of *one* shard scheme across all tables rather than per-table flexibility.
- You will invest in **schema-migration fan-out tooling** so DDL changes can be applied across all logical shards routinely.

### Reach for Figma's incremental-by-table when:

- Your tables **want different shard keys** — no single key cleanly partitions everything.
- Only a **few specific tables** are actually under pressure; sharding the whole database would be over-engineering for the rest.
- You want to **minimize per-step blast radius** even at the cost of a longer overall timeline, and you value the ability to roll back a single table independently.
- You are willing to build a **dedicated routing layer** (a DBProxy-style service that parses SQL) and to engineer **colocation** explicitly for table groups that join.
- You want to prove the routing stack with **logical sharding on one box** (views, percentage rollout) before any data physically moves.

### Skip sharding entirely when (the most important section):

- You have **not exhausted the cheaper rungs**: a bigger instance, fixing the one slow query, adding the missing index, [read replicas](/blog/software-development/database-scaling/read-scaling-with-replicas), caching, or functional/vertical partitioning. Both Notion and Figma did *years* of this first.
- Your bottleneck is **reads, not writes** — replicas and caching solve reads without the one-way door.
- You **cannot yet articulate a shard key** that keeps your hot queries single-shard. If you don't know the key, you are not ready; you will pick wrong and pay forever.
- You have **no plan for analytics** after the shard, and no appetite to build a data-lake/CDC offload.
- The project is being driven by a **dashboard panic** ("primary at 80% CPU") rather than a measured, projected runway calculation. Panic-sharding is how teams end up with a half-migrated cluster and the original bottleneck still unfixed.

The throughline of both stories, and of this whole series, is restraint. Notion and Figma are not impressive because they sharded. They are impressive because they *waited*, and then, when they finally moved, they moved in small, reversible, verifiable steps with the routing centralized and the old path kept alive until the new one was proven. The topology — 480 schemas, a Go proxy, colos — is the part that photographs well. The discipline is the part that actually shipped.

## Further reading

- Notion engineering: ["Herding elephants: lessons learned from sharding Postgres at Notion"](https://www.notion.com/blog/sharding-postgres-at-notion) — the 2021 big-bang, in their words.
- Notion engineering: ["The Great Re-shard"](https://www.notion.com/blog/the-great-re-shard) — the 2023 expansion from 32 to 96 instances.
- Notion engineering: ["Building and scaling Notion's data lake"](https://www.notion.com/blog/building-and-scaling-notions-data-lake) — the analytics-offload other half.
- Figma engineering: ["How Figma's databases team lived to tell the scale"](https://www.figma.com/blog/how-figmas-databases-team-lived-to-tell-the-scale/) — incremental sharding and DBProxy.
- This series: [the database scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree), [choosing a shard key](/blog/software-development/database-scaling/choosing-a-shard-key), [resharding without downtime](/blog/software-development/database-scaling/resharding-without-downtime), and [Instagram's sharded IDs in Postgres](/blog/software-development/database-scaling/instagram-sharding-ids-in-postgres).
