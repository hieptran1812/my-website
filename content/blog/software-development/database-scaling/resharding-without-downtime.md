---
title: "Resharding Without Downtime: The Migration Playbook"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "You will eventually reshard a busy database, and doing it without a maintenance window is a precise, reversible, six-phase playbook — dual-write, backfill, verify, cutover — not a midnight cron job and a prayer."
tags: ["database-scaling", "resharding", "sharding", "dual-write", "backfill", "change-data-capture", "zero-downtime", "data-migration", "vitess", "consistency"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 35
---

There is a particular kind of dread that settles over an engineering team when they realize the shard layout they chose two years ago is wrong. Maybe a single shard ran out of disk and there is nowhere left to grow it. Maybe one tenant — a "whale" — has grown so large that the shard it lives on is permanently at 95% CPU while every other shard idles. Maybe the shard key itself was a mistake: you sharded users by signup region, and now 60% of your traffic lives in one region, on one shard, on fire. Whatever the trigger, the conclusion is the same and it is unavoidable: you have to move data between shards while the system keeps serving live traffic, and you are not allowed to take it down to do it.

The naive instinct is to reach for a maintenance window. Put up a banner, stop writes at 2am, run a giant `INSERT ... SELECT`, flip a config, pray it finishes before the morning traffic ramp. This works exactly until your dataset is large enough or your business is global enough that "stop writes for four hours" is not a sentence you are allowed to say. At that point resharding stops being a script and becomes a *protocol* — a sequence of independently reversible phases, each of which is safe to pause, observe, and roll back, run against a database that never stops taking writes. That protocol is the subject of this post.

![The six-phase resharding playbook: stand up, dual-write, backfill, verify, cutover, decommission](/imgs/blogs/resharding-without-downtime-1.webp)

The diagram above is the mental model for the entire article: six phases, each reversible, that you walk through left to right while the old topology keeps serving every read and write. You stand up the new shards, you start *dual-writing* so the new topology never falls behind, you *backfill* the history that predates dual-write, you *verify* the two copies agree before trusting either, you *cut over* reads gradually behind a flag, and only then do you stop dual-writing and decommission the old shards. The crucial property is that at no single instant does correctness depend on a flawless atomic switch. There is no "moment of truth." There is a long, boring, observable ramp, and that boringness is the whole point.

The three phases that carry the actual risk — dual-write, backfill, cutover — happen *in order*, and the ordering is the safety property. Watch one full run below: writes start fanning to the new shards, the backfill then fills their history, and only after both are done do reads finally flip from old to new.

<figure class="blog-anim">
<svg viewBox="0 0 720 340" role="img" aria-label="Resharding phases progress over one loop: dual-write begins, the backfill fills the new shards, then reads flip from old to new at cutover" style="width:100%;height:auto;max-width:820px">
<title>Resharding phases: dual-write, then backfill fills the new shards, then reads flip at cutover</title>
<style>
.rs-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.rs-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.rs-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.rs-old{fill:var(--text-secondary,#6b7280)}
.rs-new{fill:var(--accent,#6366f1)}
.rs-phase{fill:var(--accent,#6366f1);opacity:.14;stroke:var(--accent,#6366f1);stroke-width:1.5}
@keyframes rs-sweep{0%,28%{transform:translateX(0)}33%,61%{transform:translateX(230px)}66%,94%{transform:translateX(460px)}100%{transform:translateX(0)}}
.rs-active{animation:rs-sweep 12s ease-in-out infinite}
@keyframes rs-fill1{0%,30%{opacity:0}40%,100%{opacity:1}}
@keyframes rs-fill2{0%,42%{opacity:0}52%,100%{opacity:1}}
@keyframes rs-fill3{0%,54%{opacity:0}62%,100%{opacity:1}}
.rs-f1{animation:rs-fill1 12s ease-in-out infinite}
.rs-f2{animation:rs-fill2 12s ease-in-out infinite}
.rs-f3{animation:rs-fill3 12s ease-in-out infinite}
@keyframes rs-readsOld{0%,60%{opacity:1}70%,100%{opacity:0}}
@keyframes rs-readsNew{0%,60%{opacity:0}70%,100%{opacity:1}}
.rs-rOld{animation:rs-readsOld 12s ease-in-out infinite}
.rs-rNew{animation:rs-readsNew 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.rs-active{animation:none;transform:translateX(460px)}.rs-f1,.rs-f2,.rs-f3{animation:none;opacity:1}.rs-rOld{animation:none;opacity:0}.rs-rNew{animation:none;opacity:1}}
</style>
<rect class="rs-phase rs-active" x="20" y="44" width="220" height="266" rx="12"/>
<text class="rs-lbl" x="130" y="34">2. Dual-write</text>
<text class="rs-lbl" x="360" y="34">3. Backfill</text>
<text class="rs-lbl" x="590" y="34">5. Cutover</text>
<text class="rs-sub" x="70" y="92">OLD shards</text>
<text class="rs-sub" x="200" y="92">NEW shards</text>
<rect class="rs-box rs-old" x="40" y="104" width="56" height="36" rx="6"/>
<rect class="rs-box rs-old" x="104" y="104" width="56" height="36" rx="6"/>
<rect class="rs-box rs-old" x="40" y="148" width="56" height="36" rx="6"/>
<rect class="rs-box rs-old" x="104" y="148" width="56" height="36" rx="6"/>
<rect class="rs-box rs-old" x="270" y="104" width="56" height="36" rx="6"/>
<rect class="rs-box rs-old" x="334" y="104" width="56" height="36" rx="6"/>
<rect class="rs-box rs-old" x="270" y="148" width="56" height="36" rx="6"/>
<rect class="rs-box rs-old" x="334" y="148" width="56" height="36" rx="6"/>
<rect class="rs-box rs-old" x="500" y="104" width="56" height="36" rx="6"/>
<rect class="rs-box rs-old" x="564" y="104" width="56" height="36" rx="6"/>
<rect class="rs-box rs-old" x="500" y="148" width="56" height="36" rx="6"/>
<rect class="rs-box rs-old" x="564" y="148" width="56" height="36" rx="6"/>
<rect class="rs-box" x="172" y="104" width="50" height="36" rx="6"/>
<rect class="rs-box" x="172" y="148" width="50" height="36" rx="6"/>
<rect class="rs-box" x="402" y="104" width="50" height="36" rx="6"/>
<rect class="rs-box" x="402" y="148" width="50" height="36" rx="6"/>
<rect class="rs-box" x="632" y="104" width="50" height="36" rx="6"/>
<rect class="rs-box" x="632" y="148" width="50" height="36" rx="6"/>
<rect class="rs-box rs-new rs-f1" x="172" y="104" width="50" height="36" rx="6"/>
<rect class="rs-box rs-new rs-f1" x="402" y="104" width="50" height="36" rx="6"/>
<rect class="rs-box rs-new rs-f2" x="402" y="148" width="50" height="36" rx="6"/>
<rect class="rs-box rs-new rs-f3" x="632" y="104" width="50" height="36" rx="6"/>
<rect class="rs-box rs-new rs-f3" x="632" y="148" width="50" height="36" rx="6"/>
<text class="rs-sub" x="130" y="232">reads -> OLD</text>
<text class="rs-sub" x="360" y="232">reads -> OLD</text>
<text class="rs-sub rs-rOld" x="590" y="232">reads -> OLD</text>
<text class="rs-lbl rs-rNew" x="590" y="232">reads -> NEW</text>
<text class="rs-sub" x="130" y="300">writes hit old + new</text>
<text class="rs-sub" x="360" y="300">copy history to new</text>
<text class="rs-sub" x="590" y="300">reads flip to new</text>
</svg>
<figcaption>One resharding run: dual-write mirrors every write to the new shards, the backfill fills their history, and only then do reads flip from old to new at cutover.</figcaption>
</figure>

## Why this is harder than it looks

The reason resharding is genuinely difficult — and not just tedious — is that the database is a moving target. You are copying data that is simultaneously being modified. A static dataset can be copied with `pg_dump` and reloaded; nobody writes a blog post about that. The hard version is the one where, in the milliseconds between your migration job reading a row and writing it to the new shard, the application updates that same row. Get the ordering wrong and you silently persist a stale value to the new topology, and nobody notices until you cut over reads and a customer's balance is suddenly off by one transaction.

| Assumption | The naive view | The reality |
| --- | --- | --- |
| "We can copy the data, then switch." | A one-shot bulk copy plus a config flip. | The copy takes hours; the data changes during the copy; the flip is the riskiest single instant in the whole operation. |
| "Resharding is one big operation." | One script, one run, one outcome. | It is six phases, each reversible; you must be able to abort at any phase and be back to a known-good state. |
| "If the copy completes, the shards are equal." | Row count matches, therefore done. | Counts match while values differ. You need a content checksum and a reconcile loop, not a `COUNT(*)`. |
| "Zero downtime means no risk." | Online migration = safe migration. | Online migrations move the risk from *availability* to *correctness*; a silent data divergence is worse than a clean four-hour outage. |
| "We will just change the shard key." | A re-key is a config change. | A re-key rewrites 100% of the data through a new routing function — the single most expensive and dangerous variant. |

Hold onto the last row of that table, because the *why* of your reshard determines the *what*. Splitting one full shard, adding capacity across the fleet, relocating a single hot tenant, and changing the shard key entirely are four different operations with four different blast radii — and conflating them is how teams turn a low-risk shard split into a fleet-wide incident.

## 1. Name the trigger before you touch anything

> The most expensive resharding mistakes happen before any data moves — they happen when a team picks an operation that does not match the problem they actually have.

There are essentially four reasons you reshard, and each one points at a specific operation. Naming yours out loud, in the design doc, before you write a line of migration code, is the cheapest risk reduction available to you.

![Four resharding triggers mapped to four operations and their blast radii](/imgs/blogs/resharding-without-downtime-2.webp)

**Per-shard capacity exhaustion → split one shard.** A single shard is out of disk, IOPS, or connection headroom, but the rest of the fleet is fine. The right move is to *split* that shard: pick a split point in its key range, stand up one new shard, and move roughly half the rows there. The shard key does not change; the routing function just learns that the range `[m, z]` now lives on a new home. Blast radius is one shard. This is the safest variant, and it is the one most modern sharded systems (Vitess, CockroachDB, MongoDB, DynamoDB) automate.

**Fleet-wide saturation → add shards.** Every shard is uniformly hot and you simply need more of them. If you sharded by `hash(key) % N`, going from `N` to `N+k` shards is brutal — the modulus changes the home of nearly every key — which is exactly why [consistent hashing](/blog/software-development/database/live-resharding-and-rebalancing-without-downtime) exists: it limits the data that moves to roughly `k/(N+k)` of the total instead of all of it. Either way the blast radius is the whole fleet, so this is a medium-risk operation that you stage carefully.

**One whale tenant → relocate via the directory.** A single tenant has outgrown the shard it shares. You do not want to split the shard's key range (the whale is one key); you want to *relocate* that one tenant to a dedicated shard and leave everyone else exactly where they are. This requires a directory-based topology (a lookup table from tenant to shard) rather than a pure hash function, and it is the lowest-risk operation of all because only one tenant's data moves. We will build this operation in full later.

**Bad shard key → full re-key.** The worst case. You sharded on the wrong column and there is no fix short of routing every row through a new key. This rewrites 100% of the data, changes the routing function for every query, and has the highest blast radius of any database operation you will ever run. The playbook below still applies — dual-write, backfill, verify, cutover — but the backfill is the entire dataset and the verification surface is total. If you are here, slow down: a re-key is a multi-month project, not a sprint.

The takeaway: write "we are doing operation X because trigger Y" at the top of the runbook. Everything downstream — how much data moves, whether the key changes, how wide the verification needs to be — falls out of that one sentence.

## 2. Phase 1 — Stand up the new topology

The first phase has no data in it, which makes it the easiest to get wrong because it feels trivial. Stand up the new shards (or the one new shard, for a split), apply the schema, configure replication, and — critically — wire them into your routing layer in a *dormant* state. Dormant means the new topology exists and is reachable, but no production traffic touches it yet.

The routing layer is where the whole migration lives or dies. You need a single place that, given a key, decides which physical shard owns it, and that place must be able to express "during the migration, key range `[m, z]` is being moved from shard 2 to shard 9." A directory table is the most flexible expression of this:

```sql
-- A directory-based shard map. The migration columns let the router
-- express "this range is mid-move" without a code deploy.
CREATE TABLE shard_directory (
    key_range_start   BIGINT      NOT NULL,
    key_range_end     BIGINT      NOT NULL,
    home_shard        TEXT        NOT NULL,   -- authoritative read source
    migrating_to      TEXT,                   -- non-null during a move
    migration_phase   TEXT        NOT NULL    -- 'stable' | 'dual_write'
                                              -- | 'verifying' | 'cutover'
        DEFAULT 'stable',
    PRIMARY KEY (key_range_start, key_range_end)
);
```

The `migration_phase` column is the kill switch and the rollback lever in one place. When it reads `dual_write`, the router fans writes to both `home_shard` and `migrating_to` but serves reads from `home_shard`. When it reads `cutover`, reads start flowing to `migrating_to`. To roll back any phase, you update one row. No deploy, no restart, no code change — and that is exactly the property you want when something looks wrong at 3am.

A senior detail that bites teams: **the directory itself must be highly available and consistently read.** If your router caches the directory and a stale cache says a range is still in `stable` while you have already advanced it to `cutover`, you will read from a shard that no longer owns the data. Cache the directory with a short TTL, version it, and on every config bump publish the new version so routers can detect they are stale. Treat the shard directory like a control plane, because that is what it is.

## 3. Phase 2 — Dual-write to old and new

Once the new topology is dormant and the router can express the migration, you flip the moving range to `dual_write`. From this instant forward, every write that touches that range goes to *both* the old shard and the new one. This is the phase that makes the rest of the playbook tractable: from the moment dual-write is on, the new topology can never fall *further* behind on live data. The backfill then only has to worry about history — everything that existed *before* dual-write started — which is a bounded, static-ish problem instead of a race against an infinite stream.

![Dual-write fans one logical write to both the authoritative old shard and a best-effort new shard, dead-lettering any gap](/imgs/blogs/resharding-without-downtime-3.webp)

The diagram captures the essential asymmetry: the **old shard stays authoritative**, and the new shard's write is **best-effort**. If writing to the new shard fails, the request must still succeed — your customers do not care about your migration — but the failure must be *recorded*, not swallowed, so a reconcile job can repair the gap later. Here is the wrapper that enforces that contract:

```python
import logging
import time

log = logging.getLogger("dual_write")

class DualWriter:
    """Routes one logical write to both topologies during a migration.

    The old shard is authoritative: its result is what the caller sees.
    The new shard is best-effort: a failure there is dead-lettered for a
    reconcile job, never raised to the caller.
    """

    def __init__(self, directory, old_pool, new_pool, dead_letter):
        self.directory = directory      # reads migration_phase per range
        self.old = old_pool
        self.new = new_pool
        self.dead_letter = dead_letter  # durable queue for failed mirrors

    def upsert(self, key, row):
        phase = self.directory.phase_for(key)

        # 1. Authoritative write. If this fails, the whole request fails.
        self.old.upsert(key, self._versioned(row))

        # 2. Mirror to the new shard only while the range is migrating.
        if phase in ("dual_write", "verifying", "cutover"):
            try:
                self.new.upsert(key, self._versioned(row))
            except Exception as e:
                # Never fail the request on a mirror error. Record the gap
                # so the reconcile job can re-copy this key from old -> new.
                log.warning("mirror write failed key=%s: %s", key, e)
                self.dead_letter.put({"key": key, "ts": time.time()})

    def _versioned(self, row):
        # Stamp a monotonically increasing version so the new shard can
        # apply last-writer-wins and reject a stale backfill write.
        return {**row, "_version": _next_version(row)}
```

Two design decisions in that snippet are load-bearing. First, **the old write commits before the new write is attempted.** The old shard is the source of truth; the new one is catching up. If you wrote new-first and old-second and crashed in between, the new shard would hold data that does not exist in the authoritative copy — a phantom write. Old-first means the worst case is the new shard is *missing* a write, which the dead-letter queue and the backfill both repair.

Second, **every write is versioned.** That `_version` field is not decoration; it is the mechanism that makes the next phase safe, because it lets the new shard reject a stale write that arrives out of order. We will see exactly why in the section on the moving-row problem.

### Second-order: dual-write is not a distributed transaction

A tempting but wrong instinct is to make the two writes atomic with a two-phase commit so they can never diverge. Resist it. [Two-phase commit](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) couples the availability of the old shard to the availability of the new one — now a hiccup on a shard that is not even live yet can fail production writes. The entire point of best-effort-plus-dead-letter is to *decouple* them. You accept that the two copies can briefly diverge, and you build the verify and reconcile machinery to detect and close that gap. Eventual consistency between old and new, plus a verification gate before cutover, is strictly safer than synchronous coupling here.

## 4. Phase 3 — Backfill the history

Dual-write covers everything from now on. The backfill covers everything from before. It walks the existing key range on the old shard, reads each row, and writes it to the new shard — chunked so it never holds a giant transaction, throttled so it never starves live traffic, idempotent so re-running a chunk is harmless, and resumable so a crash does not send you back to zero.

![The backfill advances a persisted cursor through the keyspace, throttling on lag and resuming from the last committed cursor after a crash](/imgs/blogs/resharding-without-downtime-4.webp)

That last property — resumability via a persisted cursor — is what separates a backfill that finishes from one that you babysit for a week. The cursor is the high-water mark of the key range you have copied. You commit it after each chunk; on restart, you read it back and continue. A backfill of a few hundred million rows *will* be interrupted — a deploy, an OOM, a network blip — and the difference between "resume from row 4.2M" and "start over from row 0" is the difference between a smooth migration and a missed deadline.

```python
import time

def backfill(old, new, cursor_store, range_start, range_end,
             batch_size=1000, max_repl_lag_s=2.0):
    """Chunked, throttled, idempotent, resumable backfill.

    Reads rows from `old` in key order and upserts them into `new`.
    Re-running any chunk is safe because the new-shard upsert is a
    versioned last-writer-wins (a stale row never overwrites a fresh one).
    """
    # Resume from the last committed cursor, not from range_start.
    cursor = cursor_store.load(range_start) or range_start

    while cursor < range_end:
        # 1. Pull one bounded, key-ordered chunk. Bounded keys (not OFFSET)
        #    keep each scan O(batch_size), never O(cursor).
        rows = old.select_range(
            key_gt=cursor, key_lte=range_end, limit=batch_size
        )
        if not rows:
            break

        # 2. Idempotent versioned upsert. If the live app already wrote a
        #    newer version of any key, that newer version wins and our
        #    historical copy is silently rejected.
        new.upsert_many_if_newer(rows)

        # 3. Advance and durably commit the cursor BEFORE the next read,
        #    so a crash here resumes from exactly this point.
        cursor = rows[-1]["key"]
        cursor_store.commit(range_start, cursor)

        # 4. Throttle: back off while the new shard's replicas are behind,
        #    so the backfill never out-runs replication or starves writes.
        while new.replication_lag_s() > max_repl_lag_s:
            time.sleep(0.5)

        # 5. A small inter-batch pause caps the backfill's share of IOPS.
        time.sleep(0.05)
```

Walk through the five numbered steps, because every one of them is a scar from a real migration:

1. **Bound the scan by key, never by `OFFSET`.** `LIMIT 1000 OFFSET 4_200_000` makes the database scan and discard 4.2 million rows on every batch — the backfill gets quadratically slower as it progresses and grinds to a halt near the end. Keyset pagination (`WHERE key > :cursor ORDER BY key LIMIT 1000`) keeps every batch the same cost. This is the single most common backfill performance bug.
2. **Upsert *if newer*, not blindly.** The new shard compares the incoming `_version` against what it already has and keeps the larger one. This is what makes the backfill idempotent *and* safe against the moving-row race — covered next.
3. **Commit the cursor before the next read.** If you advance the cursor in memory but crash before persisting it, you resume from a stale point and re-copy rows (harmless, because of step 2) — but if you persist *after* the write and crash before persisting, you re-copy, which is also fine. The dangerous ordering is committing the cursor before the write lands; never do that. Commit-after-write is the safe default; the idempotency in step 2 makes the re-copy free.
4. **Throttle on replication lag.** A backfill is a firehose of writes. If the new shard's read replicas cannot keep up, you have created a new lag problem on the very topology you are migrating to. Watch the lag and pause when it climbs. The same discipline applies on the read side: if your backfill scan is pinning the old primary, route it at a replica.
5. **Rate-limit the backfill's footprint.** A backfill running flat-out will happily consume 100% of your IOPS budget and degrade production latency. A tiny inter-batch sleep, or a token-bucket limiter, keeps it in its lane. Run it slower and finish reliably; do not run it fast and trip a latency alert.

### Second-order: CDC-driven backfill removes the race entirely

The version-based upsert handles the moving-row race, but there is a more elegant option if you have it: drive the backfill off a [change-data-capture](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) stream. Take a consistent snapshot of the old shard at a known log position, backfill from that snapshot, and then *replay the CDC log from that exact position forward* into the new shard. Every change that happened during the snapshot copy is in the log, applied in order, so the new shard converges to the old one's exact state without any best-effort dual-write at all. This is precisely how Debezium's snapshot-then-stream mode and Vitess's VReplication work, and it is why CDC tooling is the backbone of serious resharding — the log *is* the ordering guarantee you would otherwise have to invent.

## 5. The hard part — consistency while the data moves

Now the subtle bug that gives resharding its reputation. Picture a single row, currently at version 3. The backfill reads it — it now holds `v3` in memory, about to write it to the new shard. In that same instant, the application updates the row to `v4` and (because dual-write is on) writes `v4` to *both* shards. Then the backfill, still holding its stale `v3`, writes `v3` to the new shard, clobbering the fresh `v4`. The new shard now holds `v3` while the old shard holds `v4`. They have silently diverged, and a plain row-count check will never catch it.

![The moving-row race: a blind backfill overwrites a fresh update, while a versioned last-writer-wins upsert keeps the newest row regardless of order](/imgs/blogs/resharding-without-downtime-5.webp)

This is the **moving-row problem**, and the figure shows both the failure and the fix side by side. On the left, the blind copy reads `v3`, the app writes `v4`, and the backfill writes `v3` over `v4` — stale. On the right, the new shard's upsert is *conditional on the version*: it only applies an incoming write if its version exceeds the stored one. When the backfill's `v3` arrives after the app's `v4`, the upsert compares `3 > 4`, finds it false, and *rejects* the stale write. The newest version wins no matter which write physically lands last. There are four ways to defuse this race, in rough order of preference:

| Technique | How it works | When to use |
| --- | --- | --- |
| **Versioned LWW** | Every row carries a monotonic version; the new shard keeps the max. | Default. Works for any backfill + dual-write scheme. Needs a reliable version source (sequence, timestamp, or CDC log position). |
| **CDC log replay** | Backfill from a snapshot at log position L, then replay the log from L forward, in order. | When you have CDC. Removes the race entirely — order is the log's, not yours. |
| **Idempotent upsert keyed on identity** | Writes are pure functions of the row's identity; re-applying is a no-op. | When rows are immutable or append-only (events, ledgers). |
| **Tombstones for deletes** | A delete writes a versioned tombstone, not a physical removal, so the backfill cannot resurrect a deleted row. | Always, if your data has deletes. The classic backfill bug is un-deleting a row the backfill copied before the delete arrived. |

The deletion case deserves its own warning because it is the variant teams forget. If the application deletes a row, and the backfill had already read that row before the delete, a naive backfill will happily re-insert the deleted row into the new shard — a zombie. The fix is to model deletes as versioned tombstones during the migration: the delete writes a `_deleted=true` marker with a version, the backfill's `if newer` logic respects it, and the row stays dead. You physically purge tombstones only after the migration completes.

```python
def upsert_many_if_newer(self, rows):
    """New-shard upsert that resolves the moving-row race by version.

    A stale write (lower version) is silently dropped. A tombstone is a
    versioned write like any other, so a delete cannot be resurrected by
    a backfill chunk that read the row before the delete landed.
    """
    for row in rows:
        stored = self._read_version(row["key"])  # None if absent
        incoming = row["_version"]
        if stored is None or incoming > stored:
            if row.get("_deleted"):
                self._write_tombstone(row["key"], incoming)
            else:
                self._write_row(row["key"], row, incoming)
        # else: stale write, drop it — the live value is newer.
```

## 6. Phase 4 — Verify before you trust either copy

You have dual-write running and the backfill complete. Do **not** cut over yet. The cardinal sin of resharding is assuming that "the backfill finished" means "the two copies are equal." They are probably equal. You are about to bet customer data on "probably." Verify it.

![Dual-read verification fans a sampled read to both shards, serves the authoritative old copy, and queues every mismatch for a reconcile job](/imgs/blogs/resharding-without-downtime-6.webp)

The verification has two layers that work together. The first is a continuous **dual-read comparison**: for some sampled fraction of live reads, fetch from both old and new, compare, and serve the authoritative old result while recording whether they agreed. This catches divergence on the *hot* data — the rows your users actually touch — in real time, which is exactly the data most likely to have hit the moving-row race.

```python
import hashlib

def dual_read(directory, old, new, drift_counter, reconcile_queue, key):
    """Serve from the authoritative old shard; shadow-read new and compare.

    Runs on a sampled fraction of reads during the 'verifying' phase.
    Mismatches never affect the response — they are logged and queued
    for reconcile. Cutover is gated on the drift rate reaching ~zero.
    """
    old_row = old.get(key)
    serve = old_row  # the old shard is still authoritative

    if directory.phase_for(key) == "verifying" and _sampled(key, rate=0.01):
        new_row = new.get(key)
        if _row_fingerprint(old_row) != _row_fingerprint(new_row):
            drift_counter.inc()
            # Re-copy this key from the authoritative old shard.
            reconcile_queue.put({"key": key, "src": "old"})
        else:
            drift_counter.inc(labels={"result": "match"})

    return serve

def _row_fingerprint(row):
    if row is None:
        return b"\x00"  # absence is a value; missing on one side is drift
    # Hash the business fields, never the migration metadata (_version).
    payload = "|".join(f"{k}={row[k]}" for k in sorted(row) if not k.startswith("_"))
    return hashlib.sha256(payload.encode()).digest()
```

The second layer is a **bulk checksum sweep**: a background job that walks the key range on both shards in matching chunks, computes a rolling hash of each chunk on each side, and compares the hashes. Where a chunk's hashes differ, it narrows the range and re-checks until it isolates the divergent keys, then enqueues them for reconcile. This is a Merkle-tree-style comparison and it is how you verify the *cold* data that dual-read sampling will never touch. Cassandra's anti-entropy repair and rsync both use the same trick: compare hashes, not bytes, and only ship the rows that actually differ.

The reconcile job closes the loop. It drains the queue, re-reads each divergent key from the authoritative old shard, and re-writes it (versioned, so it cannot itself lose a race) to the new shard. You watch the drift rate on a dashboard. **Cutover is gated on drift reaching effectively zero and staying there** — not on a hunch, not on a deadline, but on a metric you can point at. If drift is not converging, something is structurally wrong (a missed dual-write path, a delete you are resurrecting, a clock-skewed version source), and you fix the structure before you proceed.

> A migration without a drift dashboard is not a migration; it is a hope. The single most useful artifact of the whole operation is the graph that shows old-vs-new disagreement trending to zero.

## 7. Phase 5 — Cut over reads, gradually

Reads flip last, and they flip *gradually*. The cutover is not an event; it is a dial you turn from 0% to 100% over hours or days, watching error rates and latency at every step. Because every write has been going to both shards since Phase 2, the new shard is fully caught up on live data; cutover just changes which copy answers reads.

```python
def route_read(directory, old, new, key, rollout):
    """Read routing during cutover. The rollout percentage is a config
    value (a feature flag); turning the dial is a directory update, not
    a deploy. Per-key hashing makes the ramp deterministic and reversible.
    """
    phase = directory.phase_for(key)

    if phase == "cutover":
        # Deterministic per-key bucketing: the same key always falls on
        # the same side of the dial, so a user's reads are internally
        # consistent during the ramp (no flapping old/new on refresh).
        if _bucket(key) < rollout.percent_on_new(key_range=directory.range_of(key)):
            return new.get(key)
    return old.get(key)
```

The mechanics that make this safe:

- **Percentage rollout, per range.** Start at 1% of reads on the new shard. Watch p99 latency and error rate. Ramp to 5%, 25%, 50%, 100% over a schedule, with bake time at each step. A problem shows up at 1% as a small blip, not at 100% as an outage.
- **Deterministic per-key bucketing.** If you randomly route each read, a single user refreshing a page will flap between old and new and may see read-your-writes violations if the shards momentarily disagree. Hashing the *key* to a stable bucket means a given row is consistently served from one side during the ramp.
- **The dial is a directory update, not a deploy.** Turning rollout from 25% to 50%, or slamming it back to 0% when an alert fires, is a single-row config change that propagates in seconds. This is your rollback for the cutover phase, and it must be instant.

Crucially, **writes keep dual-writing throughout cutover.** You do not stop mirroring to the old shard until reads are 100% on the new one *and* have baked there. If you have to roll back to the old shard at 80% read rollout, the old shard must still be current — which it is, only because you kept writing to it. This overlap is what makes cutover reversible; collapse it too early and your rollback path evaporates.

## 8. Phase 6 — Stop dual-write and decommission

Only after reads have been 100% on the new topology long enough to trust it — typically days, through at least one full traffic cycle and one batch-job cycle — do you stop dual-writing to the old shard and let it go cold. Then, after a further grace period during which the old shard is *idle but intact*, you decommission it.

Do not skip the grace period. Keeping the old shard cold-but-readable for a week after cutover costs you a little money and buys you the ability to recover from a divergence you did not catch — a reconcile bug, a verification blind spot, a corner of the keyspace dual-read sampling never hit. The cost of a week of idle hardware is trivial next to the cost of having permanently deleted the only correct copy of data the day before you discover the new copy is subtly wrong. Decommission is the one truly irreversible step in the whole playbook, so it is the one you rush the least.

## 9. The "split the whale" operation in full

The single most common real-world reshard is not a grand re-key — it is relocating one oversized tenant off a shared shard. It is worth walking end to end because it is the cheapest, safest variant and it exercises the whole playbook in miniature.

![Splitting the whale: a directory entry redirects one hot tenant to a dedicated shard while every other tenant stays put](/imgs/blogs/resharding-without-downtime-7.webp)

The before state in the figure is the problem every multi-tenant system eventually hits: shard 2 hosts tenants `d`, `e`, and one whale, and the whale's load pins the shard at 95% CPU with replication lag, degrading `d` and `e` as collateral damage. The after state is the fix: a brand-new shard 9 hosts the whale *alone*, shard 2 keeps only `d` and `e` and immediately decompresses, and shard 1 — and every other shard in the fleet — is untouched. The entire operation moves exactly one tenant's data.

```sql
-- Phase 1+2: register the whale's move and start dual-write for it only.
-- Note the range is a single tenant; no other tenant's routing changes.
UPDATE shard_directory
SET migrating_to = 'shard-9', migration_phase = 'dual_write'
WHERE key_range_start = :whale_tenant_id
  AND key_range_end   = :whale_tenant_id;

-- ... backfill the whale's rows shard-2 -> shard-9 (chunked, versioned) ...
-- ... verify drift -> 0 via dual-read + checksum on the whale's keys ...

-- Phase 5: cut the whale's reads over to shard-9.
UPDATE shard_directory
SET migration_phase = 'cutover'
WHERE key_range_start = :whale_tenant_id;

-- Phase 6: after bake time, the whale lives on shard-9 alone.
UPDATE shard_directory
SET home_shard = 'shard-9', migrating_to = NULL, migration_phase = 'stable'
WHERE key_range_start = :whale_tenant_id;
```

The reason a directory map is mandatory for this operation, and a pure hash function is not enough, is that you are relocating a *single key* — you cannot express "tenant 8675309 lives on shard 9 but everyone else hashes normally" with `hash(tenant) % N`. You need an explicit per-tenant override. This is exactly why large multi-tenant platforms (Notion, Figma, Slack, Salesforce) run directory-based sharding: not because hashing is wrong in general, but because the ability to surgically relocate one tenant is worth the cost of maintaining a lookup table. Hashing decides the *default* home; the directory overrides it for the tenants that need special placement.

## 10. Tooling: build it, or buy the log

You can hand-roll every phase above — and for a one-off relocation of one whale, you probably should, because the operation is small and bespoke. For anything larger, lean on tooling that has already solved the ordering problem.

| Approach | What it gives you | The catch |
| --- | --- | --- |
| **Hand-rolled dual-write + cursor backfill** | Total control; no new infrastructure; fits a one-off. | You own every consistency edge case — the moving-row race, deletes, verification, reconcile. |
| **CDC-driven (Debezium / Maxwell)** | Snapshot + ordered log replay; the race vanishes because order is the log's. | You operate Kafka/Debezium; schema changes mid-migration are fiddly; initial snapshot can be heavy. |
| **Vitess `Reshard` / VReplication** | A production-grade, MySQL-native engine for exactly this: stream from source shards, copy + tail, then `SwitchTraffic`. | You must be on Vitess; it is opinionated about topology and adds operational surface. |
| **Native primitives (Cockroach range splits, DynamoDB adaptive capacity, Mongo chunk migration)** | The database reshards itself; no application code. | Only solves "add capacity / split"; cannot do a re-key or a cross-cluster move. |

The throughline of the "buy" options is that they are all, underneath, the same playbook this article describes — stand up, copy, tail/dual-write, verify, switch — with the consistency machinery pre-built and battle-tested. Vitess's `SwitchTraffic` is a productized version of the gradual cutover; VReplication's copy-then-tail is the snapshot-plus-CDC pattern; DynamoDB's online resharding is split-one-shard automated to invisibility. If you are choosing a sharding stack today and you know you will reshard, weighting it toward one that resards online is worth more than almost any other single property. (For the lower-level mechanics of moving and rebalancing ranges, see [live resharding and rebalancing without downtime](/blog/software-development/database/live-resharding-and-rebalancing-without-downtime); for picking the key that determines how often you will be back here, see [choosing a shard key](/blog/software-development/database-scaling/choosing-a-shard-key).)

## Case studies from production

### 1. Instagram and the ID that survives a reshard

Instagram's early sharding wisdom was not about the migration mechanics at all — it was about choosing IDs that *embed the shard*, so that a row's location is computable from its primary key forever. Their 64-bit IDs pack a timestamp, a logical shard ID, and a per-shard sequence into a single integer. The lesson for resharding is indirect but profound: if your ID encodes its logical shard, you can move logical shards between physical hosts freely without touching the IDs, because routing reads the embedded shard, not a hash of the row. The reshards that hurt most are the ones where the key itself has to change; an ID scheme that decouples logical placement from physical home means you almost never have to do the painful variant. The migration playbook stays the same, but the blast radius collapses from "re-key everything" to "remap logical shards to hosts."

### 2. Figma's logical-to-physical shard split

Figma ran a single large Postgres instance for years and reshared *late*, by design, by first introducing a layer of *logical* shards inside the one physical database — partitioning tables by a shard key while everything still lived on one box. When they finally needed to scale out, the migration was logical-shard-to-physical-host rather than a row-by-row re-key, which is dramatically cheaper: you move whole logical shards as units. Their public account emphasizes the same disciplines this article does — dual-writing during the move, verifying with comparison reads before trusting the new home, and cutting over per-shard rather than all at once. The deeper lesson: build the *seam* for resharding (the logical shard boundary) long before you need to scale across it, so the eventual migration is moving boxes, not rebuilding the warehouse.

### 3. Notion's block-table reshard

Notion's data is a giant tree of "blocks," and as it grew, the single Postgres table holding billions of blocks became the bottleneck. Their reshard partitioned the block table across many logical shards and then physical databases. The publicly discussed hard parts map exactly onto our phases: they dual-wrote to old and new during the transition, ran an audit/verification pass to confirm the copies agreed before flipping reads, and discovered — as everyone does — that the long tail of the migration is not the bulk copy but the *verification and reconciliation* of the rows that changed mid-flight. The bulk backfill of billions of rows was the predictable part; chasing the last fraction of a percent of divergent rows to zero drift was the part that consumed the schedule.

### 4. Slack and the hot shard

Slack's data is naturally sharded by workspace, and the failure mode is structural: a single enormous workspace — a huge enterprise customer — can dwarf an entire shard's worth of normal customers. The fix is the "split the whale" operation in production: relocate the giant workspace to a dedicated, beefier shard via a directory lookup, leaving the thousands of small workspaces on shared shards untouched. The directory-based topology is not a nicety here; it is the only structure that lets you move *one* workspace without rehashing the world. The recurring operational lesson is that hot-shard relocation is not a rare emergency but a *routine maintenance operation* you should automate, because in a multi-tenant system there is always a next whale.

### 5. The OFFSET backfill that never finished

A team I worked alongside kicked off a backfill of roughly 400 million rows using `LIMIT 10000 OFFSET :n`, incrementing the offset each batch. The first thousand batches flew. By the time the offset reached 200 million, each batch was scanning and discarding 200 million rows before returning ten thousand, and a backfill projected to take three days had a tail that stretched past three weeks — the throughput collapsed as a near-perfect inverse of progress. The fix was a two-line change to keyset pagination (`WHERE id > :cursor ORDER BY id LIMIT 10000`), which made every batch cost the same regardless of position, and the rewritten backfill finished in under two days. The lesson is mechanical and unglamorous: `OFFSET` is O(offset) per query, and a backfill that paginates by offset gets quadratically slower until it appears to hang. Always paginate by key.

### 6. The resurrected accounts

During a re-key migration of a user table, an early backfill copied rows from old to new without modeling deletes as tombstones. The application, meanwhile, was processing account deletions for a compliance deadline. The race was textbook: the backfill read an account before it was deleted, the deletion landed (removing it from both shards via dual-write), and then the backfill wrote the stale copy into the new shard — resurrecting accounts that were supposed to be gone, on the very table being audited for deletion compliance. The divergence was invisible to a row-count check because the *counts* were close; it surfaced only when the dual-read verification flagged rows present on new but absent on old. The fix was to model deletes as versioned tombstones for the duration of the migration and re-run the reconcile. The lesson: deletes are the backfill's most dangerous case, and "I copied all the rows" is not the same as "the two copies agree."

### 7. The cutover that flapped

A team cutting over reads used a random per-request coin flip to decide old-versus-new during the rollout ramp. At 50% rollout, users hammering refresh would land on old, then new, then old — and because the new shard occasionally trailed the old by a few hundred milliseconds of reconcile lag on a just-written row, some users saw their own writes appear, vanish, and reappear. Support tickets spiked with "my changes aren't saving" reports that no engineer could reproduce, because reproduction depended on a coin flip. The fix was deterministic per-key bucketing: hash the row key to a stable bucket so a given row is always served from the same side during the ramp. The lesson: randomness in a cutover dial creates per-user inconsistency that is nearly impossible to debug; make the routing decision a deterministic function of the key.

### 8. The directory cache that pointed at a ghost

A directory-based router cached the shard map with a five-minute TTL to avoid hammering the directory store on every request. During a whale relocation, the cutover advanced the directory to point the whale's reads at the new shard, but a fleet of router instances held a stale cache that still pointed at the old shard for two more minutes — and dual-write for that range had *already been stopped* by an over-eager operator who saw "cutover complete" and moved on. For two minutes, those routers read the whale's data from a shard that was no longer receiving its writes, serving stale rows. The dual fix: version the directory and have routers reject reads when their cached version is older than a published floor, and — more importantly — never stop dual-write until reads have been fully cut over *and baked* across every router. The lesson: the directory is a control plane, stale control-plane state is a correctness bug, and the phases overlap for a reason.

## When to reach for this playbook — and when not to

Reach for the full dual-write / backfill / verify / cutover playbook when:

- Your dataset is large enough or your business global enough that a maintenance window is not an option — the moment "stop writes for an hour" is off the table, you are in playbook territory.
- You are moving data *between* shards or topologies, not just adding a replica or splitting a range that your database can split itself.
- Correctness during the move matters and a silent divergence would be costly — financial data, user-visible state, anything audited.
- You need the ability to abort partway and be back to a known-good state, which only the reversible-phase structure gives you.

Skip the heavy machinery when:

- Your database can reshard itself for your case (Cockroach range splits, DynamoDB online resharding, Vitess `Reshard`, Mongo chunk migration). Use the native primitive; do not rebuild it by hand.
- The dataset is small enough that a brief maintenance window is genuinely acceptable — a few gigabytes at 3am on a regional B2B app may not be worth a month of dual-write engineering. Be honest about which world you live in.
- You are only adding read capacity. That is a [read-replica](/blog/software-development/database-scaling/read-scaling-with-replicas) problem, not a resharding problem; do not reach for this hammer.
- You have not yet named the trigger. If you cannot write "we are doing operation X because of Y" in one sentence, you are not ready to move data — go back to section 1.

The thread running through every phase, every code snippet, and every one of these incidents is the same: resharding without downtime is not a clever script run during a quiet window. It is a protocol of small, reversible, observable steps, gated on a drift metric you can point at, with an old copy you refuse to delete until you have proven the new one correct. Done that way, the scary part — the cutover — becomes the boring part, which is exactly where you want it.

## Further reading

- [Live resharding and rebalancing without downtime](/blog/software-development/database/live-resharding-and-rebalancing-without-downtime) — the lower-level mechanics of moving and rebalancing ranges, consistent hashing, and how engines like Vitess implement online range moves internally.
- [Change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — the log-replay technique that removes the moving-row race entirely.
- [Choosing a shard key](/blog/software-development/database-scaling/choosing-a-shard-key) — the decision that determines how often you will be back here doing this.
- [Zero-downtime schema migrations](/blog/software-development/database/zero-downtime-schema-migrations) — the sibling discipline; the same expand-migrate-contract reversibility, applied to schema rather than data placement.
