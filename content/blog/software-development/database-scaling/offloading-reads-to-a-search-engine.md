---
title: "Offloading Reads to a Search Engine: Elasticsearch Alongside Your Database"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "Full-text search, faceting, and aggregations crush a relational database — offload them to Elasticsearch, but the real work is keeping the derived index in sync without dual-write bugs."
tags: ["database-scaling", "elasticsearch", "opensearch", "change-data-capture", "outbox-pattern", "full-text-search", "kafka", "eventual-consistency", "debezium", "reindexing"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 34
---

The query that finally takes down your primary database is almost never a write. It is a search box. Someone ships a feature that lets users type free text into a field and get ranked results — products, support tickets, log lines, other users — and the first implementation is a single innocent-looking line: `WHERE name ILIKE '%' || $1 || '%'`. It works in the demo. It works in staging with ten thousand rows. Then it ships, the table grows to forty million rows, an autocomplete fires that query on every keystroke, and one afternoon the primary is pinned at 100% CPU with a queue of sequential scans, each one reading the entire table because a leading-wildcard `LIKE` cannot use any B-tree you have ever built.

This is the moment most teams reach for a search engine. The instinct is correct. Full-text relevance ranking, faceted navigation, typo tolerance, and high-QPS aggregations are genuinely *different work* from what a relational query planner is built to do, and trying to bolt them onto your OLTP primary is how you turn a healthy database into a smoking one. Elasticsearch (or its fork OpenSearch, or a managed engine like Algolia or Typesense) exists precisely to take those reads off your hands. But the part nobody warns you about — the part that turns a one-week project into a six-month source of mysterious "why isn't this product showing up in search" bugs — is that the search engine is a *second copy of your data*, and keeping that copy honest is the entire engineering problem.

![The database is the source of truth; the search index is a derived read model fed by an ordered change stream](/imgs/blogs/offloading-reads-to-a-search-engine-1.webp)

The diagram above is the mental model for the whole post. The primary database is the **source of truth**: it takes every write and serves point reads ("give me order 4471"). The search engine is a **derived read model**: it serves the queries that the database is bad at — full-text, faceted, fuzzy, aggregated — and it is rebuildable from the primary at any time. Between them runs a sync pipeline: every committed change in the database flows, in order, through change capture and a log into the index. Search reads bypass the database entirely and hit the index directly. Get that pipeline right and you have bought yourself enormous read headroom. Get it wrong — and the default naive way is wrong — and your two systems quietly drift apart until someone notices that search results have been lying for three weeks.

## Why you cannot just search in Postgres

**The senior rule of thumb: a B-tree indexes whole values from the left; the moment your query needs to match the *middle* of a string or rank by relevance, the index is useless and you are doing a sequential scan.**

It is worth being precise about *why* the relational database falls down, because the answer tells you exactly when you can stay in Postgres and when you cannot.

![Why a relational database cannot do engine-grade search: a B-tree matches whole values left-to-right, while an inverted index maps every token to the documents that contain it](/imgs/blogs/offloading-reads-to-a-search-engine-2.webp)

A B-tree on `name` stores values in sorted order, so it can answer `name = 'Aeron Chair'`, `name LIKE 'Aeron%'` (a prefix — still left-anchored), and range queries instantly. What it *cannot* do is `name LIKE '%chair%'`, because there is no way to jump to "all values that contain `chair` somewhere in the middle" in a left-sorted tree. The planner has exactly one option: read every row and test each one. That is `O(n)` per query, and it gets worse linearly as your table grows.

Postgres does have real answers here, and you should know them before you add an entire new system to your stack:

```sql
-- Real full-text search inside Postgres: a GIN index over a tsvector.
-- This tokenizes, lowercases, and stems ("running" -> "run"), and supports ranking.
CREATE INDEX idx_articles_fts
  ON articles
  USING GIN (to_tsvector('english', coalesce(title,'') || ' ' || coalesce(body,'')));

SELECT id, title,
       ts_rank(to_tsvector('english', body), query) AS rank
FROM   articles,
       plainto_tsquery('english', 'distributed consensus') AS query
WHERE  to_tsvector('english', body) @@ query
ORDER  BY rank DESC
LIMIT  20;

-- And pg_trgm makes a substring LIKE usable, by indexing 3-character shingles:
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX idx_articles_title_trgm ON articles USING GIN (title gin_trgm_ops);
-- now: WHERE title ILIKE '%chair%'  can use the trigram index
```

`tsvector` plus a GIN index is a legitimate full-text engine. It tokenizes, applies stemming and stop-word removal per language, supports phrase and boolean queries, and ranks with `ts_rank`. `pg_trgm` makes infix `LIKE` and fuzzy matching (`similarity()`) index-backed. For a great many applications — a blog, an internal admin tool, a catalog with a few hundred thousand rows and modest query volume — **this is genuinely enough, and adding Elasticsearch would be a strict mistake.** You would be taking on a distributed stateful system, a sync pipeline, and a second on-call surface to solve a problem your existing database already solves.

So when do you actually outgrow it? The honest list:

- **Relevance quality.** `ts_rank` is crude. It has no notion of term frequency saturation, field-length normalization, or the per-field boosting you want ("a match in the title is worth more than a match in the body"). A real engine implements BM25, which is the difference between "results in some order" and "results a user trusts."
- **Faceting and aggregations at high QPS.** "Show me the count of in-stock products per brand per price-bucket, filtered by the current search" is a `GROUP BY` over a filtered set on every request. Postgres can do it; it cannot do it forty thousand times a second without a dedicated replica fleet that you are now operating anyway.
- **Typo tolerance and autocomplete.** Edit-distance matching, completion suggesters, and "did you mean" are first-class in a search engine and bolted-on at best in Postgres.
- **You want the load *off* the primary.** Even if Postgres FTS could do all of it, every full-text query you run is CPU and buffer-cache pressure on the machine that also has to accept your writes. The whole point of this article is moving that load somewhere it cannot hurt your transactional path.

The decision is not "Postgres FTS is bad." It is: stay in Postgres until relevance quality, facet QPS, or primary-load isolation forces your hand — and then move deliberately, knowing the sync cost you are signing up for.

## What a search engine actually gives you

**The senior rule of thumb: a search engine is an inverted index plus a relevance model plus an analysis pipeline — three things a row store does not have and was never meant to grow.**

The core data structure is the **inverted index**. Where a B-tree maps `value -> row`, an inverted index maps `token -> list of documents containing that token` (the "posting list"). To search for `chair`, the engine looks up one key and gets the exact set of matching documents — no scan. Multi-term queries intersect or union posting lists. This is why an engine answers `%chair%`-style queries in microseconds at a scale where Postgres is reading the whole table: it pre-computed, at index time, the answer to "which documents contain this token."

Getting tokens out of text is the job of the **analyzer**: a configurable chain of a tokenizer (split on whitespace/punctuation, or on n-grams, or on language rules) and token filters (lowercase, stem, remove stop words, fold accents, expand synonyms). The analyzer you use at index time and the one you use at query time must agree, and choosing them is most of the art of getting good search — an "edge n-gram" analyzer gives you autocomplete; a language analyzer gives you stemming; a `keyword` field gives you exact-match faceting.

On top of the posting lists sits the **relevance model**, BM25 by default. BM25 scores a document for a query using term frequency (with saturation, so the tenth occurrence of a word matters far less than the first), inverse document frequency (rare words are more discriminating), and field-length normalization (a match in a short title beats a match buried in a long body). You can tune it, boost fields, and combine it with business signals (recency, popularity) in a single ranking expression.

A minimal index definition makes the model concrete:

```json
PUT /products_v1
{
  "settings": { "number_of_shards": 3, "number_of_replicas": 1 },
  "mappings": {
    "properties": {
      "name":       { "type": "text",  "analyzer": "english" },
      "name_exact": { "type": "keyword" },
      "brand":      { "type": "keyword" },
      "price":      { "type": "scaled_float", "scaling_factor": 100 },
      "in_stock":   { "type": "boolean" },
      "updated_at": { "type": "date" }
    }
  }
}
```

Notice the two flavors of the same field: `name` is `text` (analyzed, for relevance) and `name_exact` is `keyword` (not analyzed, for exact filters, sorts, and facets). That duplication is normal and deliberate. A single query can now do all the things the database struggled with at once: full-text match on `name`, a `brand` facet with counts, a `price` histogram aggregation, an `in_stock` filter, and BM25 ranking — in one round trip, served from a fleet that never touches your primary. That last clause is the scaling win. Every search request you move here is a request your transactional database no longer has to plan, execute, and evict buffer cache for. (For the broader question of which store owns which workload, see [polyglot persistence and choosing the right store](/blog/software-development/database-scaling/polyglot-persistence-choosing-the-right-store).)

## The sync problem is the whole problem

Here is the uncomfortable truth that the architecture diagram hides: **you now have the same data in two places, and they will diverge unless you do real work to stop them.** The database is the system of record. The index is a derived, rebuildable read model. The job is to make every committed change in the database appear in the index — every insert, update, and delete — in the right order, without losing any and without applying any twice. This is a distributed-systems problem dressed up as a feature request, and the way you solve it determines whether your search is trustworthy.

There are four families of solution, and they are not equally good. The rest of this section walks each one, in roughly increasing order of correctness.

### Dual writes, and exactly how they break

The naive approach is the one everyone writes first: in the same request handler, write to the database and then write to the search engine.

```python
def create_product(db, es, product):
    db.insert("products", product)          # 1. write the source of truth
    es.index(index="products", id=product.id, document=to_doc(product))  # 2. write the index
    return product
```

This is a **dual write**, and it is broken in ways that do not show up until production. The problem is that there is no transaction spanning the two systems — no two-phase commit you would actually want to run on the hot path — so any partial failure leaves them inconsistent with no automatic recovery.

![How a dual write silently diverges: with no atomic boundary across two systems, a partial failure leaves the index permanently wrong](/imgs/blogs/offloading-reads-to-a-search-engine-3.webp)

Walk the failure timeline in the figure. At `t0` the database commit succeeds. At `t1` the call to Elasticsearch times out — the ES node is doing GC, the network blipped, the bulk queue was full, pick any of a dozen real causes. Now at `t2` the database has the row and the index does not. There is **no rollback of the committed database write** — it is already durable — so at `t3` the systems have permanently disagreed, and at `t4` the user's search silently misses a record that genuinely exists. Nothing errored loudly. Nothing alerted. The drift is invisible until a customer complains that their newly created product "isn't showing up," and by then you have no idea how many other records are affected.

It gets worse than single-call failures. Consider two concurrent updates to the same product, A then B. The database serializes them correctly (B wins). But the two ES calls can race — B's index call can overtake A's on a slower path — and the index ends up showing A's value while the database shows B's. Now the order of writes disagrees between your two systems, permanently. You cannot fix this with retries, because a retry of a stale write re-applies stale data.

> A dual write is not a sync strategy. It is a way to be wrong slowly, and to discover it only after a customer does.

People try to patch dual writes — wrap the ES call in a retry, push it to a background job queue, add a reconciliation cron. Each patch is really an admission that you needed one of the next three approaches all along. The retry-with-queue version is just a worse outbox. The reconciliation cron is just a worse periodic reindex. Skip the detour.

### Change data capture: tail the log

**The senior move: do not ask the application to tell the index what changed. Read the database's own write-ahead log, which already records every committed change in commit order, and turn that stream into index operations.**

Change data capture (CDC) reads the durable change log the database keeps for its own recovery and replication — the Postgres WAL via logical replication, the MySQL binlog — and emits a structured event per row change. A tool like Debezium runs as a Kafka Connect source connector, tails the log, and publishes one message per change onto a Kafka topic. A sink connector (or your own consumer) reads those messages and applies them to Elasticsearch. Because the events come *from the log*, they are exactly the set of committed changes, in exactly commit order, and they include changes made by anything — your app, a migration, a manual `UPDATE` someone ran at 2am. The application does not have to remember to tell anyone. This is the same machinery covered in depth in [change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern), and it leans on [Kafka as a distributed, ordered, replayable log](/blog/software-development/database/kafka-as-a-distributed-log) to buffer and fan out the stream.

The animated figure is the whole CDC pipeline in motion. Watch one committed change at a time flow from the database, through CDC, through the ordered Kafka log, through an idempotent sink, into the index — which trails the database by the pipeline's lag and then catches up.

<figure class="blog-anim">
<svg viewBox="0 0 760 240" role="img" aria-label="A committed write flows left to right through DB, CDC, Kafka, and the search sink into the Elasticsearch index, which lags then catches up" style="width:100%;height:auto;max-width:860px">
<title>A committed change flows through the CDC pipeline into the search index, which trails the DB then catches up</title>
<style>
.s1-stage{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.s1-es{fill:var(--surface,#f3f4f6);stroke:var(--accent,#6366f1);stroke-width:2.5}
.s1-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.s1-sub{font:400 11px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.s1-rail{stroke:var(--border,#d1d5db);stroke-width:2;stroke-dasharray:4 6}
.s1-dot{fill:var(--accent,#6366f1)}
@keyframes s1-flow{0%{transform:translateX(0);opacity:0}6%{opacity:1}88%{opacity:1}100%{transform:translateX(560px);opacity:0}}
@keyframes s1-pulse{0%,72%{opacity:.25}82%,100%{opacity:1}}
.s1-p{animation:s1-flow 7s linear infinite}
.s1-p2{animation-delay:2.33s}
.s1-p3{animation-delay:4.66s}
.s1-catch{animation:s1-pulse 7s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.s1-p,.s1-catch{animation:none;opacity:1}}
</style>
<line class="s1-rail" x1="120" y1="96" x2="660" y2="96"/>
<rect class="s1-stage" x="20"  y="66" width="100" height="60" rx="8"/>
<rect class="s1-stage" x="160" y="66" width="100" height="60" rx="8"/>
<rect class="s1-stage" x="300" y="66" width="100" height="60" rx="8"/>
<rect class="s1-stage" x="440" y="66" width="100" height="60" rx="8"/>
<rect class="s1-es"    x="580" y="60" width="120" height="72" rx="10"/>
<text class="s1-lbl" x="70"  y="92">DB</text>
<text class="s1-sub" x="70"  y="112">commit</text>
<text class="s1-lbl" x="210" y="92">CDC</text>
<text class="s1-sub" x="210" y="112">Debezium</text>
<text class="s1-lbl" x="350" y="92">Kafka</text>
<text class="s1-sub" x="350" y="112">ordered log</text>
<text class="s1-lbl" x="490" y="92">sink</text>
<text class="s1-sub" x="490" y="112">idempotent</text>
<text class="s1-lbl" x="640" y="90">ES index</text>
<text class="s1-sub s1-catch" x="640" y="112">catching up</text>
<circle class="s1-dot s1-p"  cx="100" cy="96" r="8"/>
<circle class="s1-dot s1-p s1-p2" cx="100" cy="96" r="8"/>
<circle class="s1-dot s1-p s1-p3" cx="100" cy="96" r="8"/>
<text class="s1-sub" x="380" y="176">one committed change per dot, in commit order, never lost</text>
</svg>
<figcaption>Every committed change flows in order through CDC and Kafka to the search sink; the index trails the DB by the pipeline's lag, then catches up.</figcaption>
</figure>

The consumer's job is to map one row-change event to one index operation, and to do it **idempotently** so that re-delivering the same event (which Kafka's at-least-once delivery guarantees will happen) does not corrupt anything:

```python
# Map one Debezium change event to one Elasticsearch bulk action.
# The envelope's "op" is c(reate) / u(pdate) / d(elete) / r(ead = snapshot).
def to_es_action(change: dict) -> dict:
    op    = change["op"]
    index = "products"                      # always the alias, never products_v1
    src   = change["source"]
    # The log position is monotonic per source; carry it as the document version
    # so out-of-order or replayed events are dropped instead of overwriting newer data.
    version = lsn_to_int(src["lsn"])

    if op == "d":
        key = change["before"]["id"]
        return {"_op_type": "delete", "_index": index, "_id": str(key),
                "version": version, "version_type": "external_gte"}

    row = change["after"]
    return {
        "_op_type": "index",
        "_index": index,
        "_id": str(row["id"]),
        "version": version,
        "version_type": "external_gte",     # apply only if >= current version
        "_source": {
            "name":       row["name"],
            "name_exact": row["name"],
            "brand":      row["brand"],
            "price":      row["price_cents"] / 100,
            "in_stock":   row["stock"] > 0,
            "updated_at": row["updated_at"],
        },
    }
```

The `version_type: external_gte` trick is the load-bearing line. Elasticsearch will reject any write whose version is *less than* the document's current version, so a delayed or replayed event for an old state is silently dropped rather than clobbering a newer state. This turns at-least-once delivery into effectively exactly-once *application*, which is the property you actually need. Combine that with consuming Kafka partitions keyed by document id (so all changes to one document land on one partition, in order) and the ordering races that plague dual writes simply cannot happen.

The cost of CDC is operational weight: you are running Kafka, Kafka Connect, Debezium connectors, and a schema registry, plus you must enable logical decoding on your primary and manage replication slots (a stuck slot pins WAL and can fill your disk — a real incident, covered below). For a high-volume system with multiple downstream consumers, that weight pays for itself many times over. For a single index on a modest system, it may be more than you need — which is where the outbox comes in.

### The outbox pattern: make the index update part of the transaction

**The senior move when you own all the writers and do not want a Kafka cluster: write the change and a record-of-the-change in the same transaction, and let a separate relay ship the record.**

The [outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) gets you the correctness of CDC without (necessarily) running Debezium. The insight is that the dual-write problem is *the lack of a shared transaction*. So you give it one: in the same database transaction that writes your domain row, you insert a row into an `outbox` table describing the change. Because they commit atomically, you can never have a domain change without its outbox record or vice versa. A separate relay process then reads unsent outbox rows and ships them to the index.

![The outbox makes the index update part of the transaction: the domain row and an outbox row commit together, and a relay later ships the outbox to the index and marks it sent](/imgs/blogs/offloading-reads-to-a-search-engine-4.webp)

The write path is dead simple and, crucially, atomic:

```python
def place_order(conn, order):
    with conn.transaction():               # one transaction
        order_id = conn.execute(
            "INSERT INTO orders (customer_id, total_cents, status) "
            "VALUES (%s, %s, %s) RETURNING id",
            (order.customer_id, order.total_cents, "placed"),
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO outbox (aggregate_type, aggregate_id, op, payload) "
            "VALUES (%s, %s, %s, %s)",
            ("order", order_id, "upsert", json.dumps(serialize(order))),
        )
    # COMMIT here: the order row and its outbox row are now durable together,
    # or neither is. There is no window where one exists without the other.
    return order_id
```

The relay is a small, boring loop. It claims a batch of unsent rows in id order (id is a `bigserial`, so id order is commit order), ships them to Elasticsearch in one bulk call, and marks them sent. `FOR UPDATE SKIP LOCKED` lets you run several relay workers without them stepping on each other:

```python
def relay_once(conn, es) -> int:
    rows = conn.execute("""
        SELECT id, aggregate_id, op, payload
        FROM   outbox
        WHERE  sent_at IS NULL
        ORDER  BY id                       -- bigserial id == commit order
        LIMIT  500
        FOR UPDATE SKIP LOCKED             -- safe to run N relays in parallel
    """).fetchall()
    if not rows:
        return 0

    actions = [build_action(r) for r in rows]   # upsert/delete per row
    helpers.bulk(es, actions, raise_on_error=True)   # one network round trip

    ids = [r["id"] for r in rows]
    conn.execute("UPDATE outbox SET sent_at = now() WHERE id = ANY(%s)", (ids,))
    conn.commit()
    return len(rows)
```

The correctness argument is clean. If the relay crashes after `bulk` but before the `UPDATE`, those rows are simply re-sent next loop — and because your index operations are idempotent (same id, upsert semantics), re-sending is harmless. If the relay crashes before `bulk`, nothing was marked sent, so nothing is lost. The index lags the database by at most one relay cycle plus the bulk latency — seconds, typically. You have exactly-once *effect* with at-least-once *delivery*, which is exactly what you want, and you got it with one table and one cron-like loop instead of a streaming platform.

The two costs to budget for: the outbox table grows and must be pruned (delete rows older than a retention window, or move `sent` rows to a partition you drop), and the relay is a throughput bottleneck you have to size — one relay doing 500-row bulks every 200ms handles a few thousand changes/second, and beyond that you partition the outbox by `aggregate_id % N` and run N relays. Many teams run the outbox *and then* point Debezium at the outbox table — combining the atomic-write guarantee of the outbox with the operational robustness of CDC. That hybrid is the most common production setup I see.

### Backfill and reindex: rebuilding the derived model

Sooner or later you will need to rebuild the index from scratch: you changed an analyzer, added a field to the mapping, changed the number of shards, recovered from a sync bug, or you are standing up search for the first time over an existing table. Because the index is a *derived* model, this is always possible — the source of truth has everything. The challenge is doing it **without downtime**, while live traffic is reading and the CDC/outbox stream is still flowing.

The mechanism is the **index alias**. Your application never reads from `products_v1` directly; it reads from an alias named `products` that *points at* `products_v1`. To reindex, you build a brand-new `products_v2`, backfill it, let the live change stream catch it up, and then atomically flip the alias to point at `products_v2`. The flip is a single API call that swaps the pointer with no window where the alias points at nothing.

![Zero-downtime reindex with an alias swap: reads point at an alias, and building v2 then flipping the alias cuts over atomically](/imgs/blogs/offloading-reads-to-a-search-engine-5.webp)

```python
def reindex_with_alias_swap(es, alias="products", new_mapping=NEW_MAPPING):
    old = current_index_for(es, alias)                # e.g. "products_v1"
    new = bump_version(old)                            # "products_v2"

    es.indices.create(index=new, body=new_mapping)     # fresh index, new mapping

    # Backfill: either _reindex from the old index, or (safer) bulk-load from the
    # database, which is the actual source of truth and guarantees correctness.
    es.reindex(
        body={"source": {"index": old}, "dest": {"index": new}},
        wait_for_completion=False,                     # returns a task id; poll it
    )
    wait_for_reindex(es)                               # poll _tasks until done
    # Let the live CDC/outbox stream (writing to the alias) catch new up to old,
    # then verify doc counts match within the lag budget before cutting over.

    # The atomic swap: remove the old mapping and add the new one in ONE call.
    es.indices.update_aliases(body={"actions": [
        {"remove": {"index": old, "alias": alias}},
        {"add":    {"index": new, "alias": alias}},
    ]})
    # Keep products_v1 for a rollback window, then delete it.
```

Two details separate a clean reindex from a corrupt one. First, **the change stream must write to the alias, not a concrete index**, so that during the backfill new writes land in whichever index the alias currently points at, and after the swap they immediately land in `v2` — no lost writes across the cutover. (A common pattern is to write to *both* indices during the transition window, which the alias can express by pointing at both for writes.) Second, **prefer backfilling from the database, not `_reindex` from the old index**, when correctness matters: the old index may itself be subtly wrong (that is often *why* you are reindexing), and the database is the only thing you trust. Reindexing from the old index is faster but copies its sins.

Version your indices (`products_v1`, `products_v2`, …) so you always have the previous one to roll back to with another one-call alias flip. The rollback path is the same swap in reverse, and having tested it is the difference between a calm cutover and a 3am panic.

## Choosing a sync strategy

Put the four families side by side and the tradeoff is stark: the only row that can *permanently corrupt* your index is the one most people reach for first.

![Sync strategies compared on correctness, freshness, complexity, and ordering](/imgs/blogs/offloading-reads-to-a-search-engine-6.webp)

| Strategy | Correctness | Freshness | Ordering | Operational complexity | When to use |
| --- | --- | --- | --- | --- | --- |
| **Dual write** (app writes DB then ES) | Diverges on any partial failure; no automatic recovery | "Instant" when both calls succeed | Racy — concurrent updates can land out of order | Low to write, very high to debug | Prototypes and demos. Never production. |
| **Periodic full reindex** (cron rebuild) | Self-heals every cycle; bounded staleness | Minutes to hours | N/A (whole-index rebuild) | Low | Small corpora, search as a nice-to-have, or as a safety net under another method |
| **Outbox + relay** | Exactly-once effect; relay can crash and resume | Seconds | Per-aggregate order preserved | Medium (one table + a relay loop) | You own all writers and want correctness without running Kafka |
| **CDC** (Debezium → Kafka → sink) | Exactly-once effect, ordered by the log; captures *every* change | Sub-second to seconds | Strict per-key log order | High (Connect, Kafka, schema, replication slots) | High change volume, multiple consumers, or writers you cannot modify |

The two columns that matter most are **correctness** and **ordering**, and they sort the table cleanly: dual write fails both, periodic reindex sidesteps both by brute force, and the two log-based approaches (outbox and CDC) get both right. The honest default for most teams is: **outbox + relay if you own your writes, CDC if you have many writers or many consumers, and a periodic full reindex running underneath either one as a cheap, self-healing safety net** that erases any drift you didn't anticipate. Notice that "instant freshness" on the dual-write row carries an asterisk — it is instant only on the happy path, and the unhappy path is permanent.

## The consistency model you are actually signing up for

**The senior rule of thumb: the index is eventually consistent with the database, by design, and you must build the product around the lag rather than pretend it is zero.**

No matter which sync strategy you pick (except, nominally, a synchronous dual write, which you should not use), the index trails the database by some lag — milliseconds to seconds for a healthy CDC/outbox pipeline, longer when the pipeline is backed up. A user who creates a product and immediately searches for it may not find it for a beat. This is not a bug to be fixed; it is the nature of a derived read model, and the same eventual-consistency reasoning applies as with [read replicas and replication lag](/blog/software-development/database-scaling/read-scaling-with-replicas). Your job is to design the UX so the lag is invisible or acceptable:

- **Read the writer's own changes from the primary, not the index.** Right after a user edits their product, render the detail page and "my products" list from the database (a point read the primary is great at), so they always see their own change immediately. Reserve the index for *discovery* queries where staleness is fine — searching the global catalog, browsing facets.
- **Set expectations in copy.** "Your listing is live and will appear in search results within a few moments." One sentence converts a confusing absence into an understood, expected delay.
- **Monitor the lag as a first-class SLO.** Emit the age of the most recent document the index has applied (compare the newest `updated_at` in the index to the database, or track consumer offset lag). Alert when it crosses your budget. A pipeline that has silently stopped looks exactly like "fresh but quiet" until you measure the lag.
- **Make the lag bound part of your product contract.** "Search reflects changes within 30 seconds" is a promise you can keep and test. "Search is real-time" is a promise the laws of distributed systems will not let you keep.

The mental shift is from "the index is a copy of the database" to "the index is a *projection* of the database as of a slightly earlier moment." Once the team internalizes that, a whole class of bug reports ("I updated it but search shows the old value!") become expected behavior with a known bound, not mysteries.

## The operational reality

This is the section that the tutorials skip and that production teaches. Running a search engine alongside your database has a set of failure modes that are specific to its being a large, stateful, memory-hungry, schema-rigid second system.

**Mapping and analyzer changes force a reindex.** Elasticsearch mappings are largely immutable. You can add a new field, but you cannot change an existing field's type or analyzer in place — the inverted index was built with the old analysis and there is no in-place migration. Want to switch the `name` field from a `standard` analyzer to an `english` one? That is a full reindex (the alias-swap dance above). This makes the mapping a schema you must take as seriously as your database schema, with the same migration discipline. Teams that treat the index mapping as "just config" discover, the first time they need to change an analyzer, that the change is a multi-hour reindex of a billion documents.

**Index lifecycle and growth.** For append-heavy data (logs, events, time-series documents), use time-based indices and an index lifecycle policy (ILM): roll over to a new index by size or age, move older indices to cheaper hardware (hot/warm/cold tiers), and delete past retention. Without lifecycle management, a single ever-growing index accumulates shards and heap pressure until the cluster degrades.

**Sizing, or "we indexed everything and the cluster fell over."** The classic incident: a team decides to index *every* field of *every* row "so we can search anything later," points a backfill at a hundred-million-row table, and watches the cluster go red. The causes compound — too many shards (each shard is a Lucene index with fixed overhead; thousands of tiny shards exhaust heap), mappings that index huge text fields nobody searches, a backfill running with no throttle that saturates the bulk queue and triggers rejections, and JVM heap set too high so garbage collection pauses stall the node. The fix is discipline: index only the fields you actually query, size shards in the tens-of-gigabytes range (not hundreds of tiny ones), throttle backfills, and leave heap at or below ~50% of RAM so the OS page cache (which Lucene leans on heavily) has room. **A search index should hold a deliberately chosen projection of your data, not a mirror of every column.**

**Deep pagination is a trap.** `from: 100000, size: 10` forces every shard to collect and sort the first 100,010 results and ship them to the coordinating node, which sorts again and discards 100,000 — per request. It is `O(from)` work and it will OOM your cluster under load. Use `search_after` (a cursor keyed on the sort values of the last result) for deep scrolling, and cap `from` at a few thousand for human pagination. Nobody clicks to page 10,000 anyway.

**Unbounded and expensive queries.** A leading-wildcard query (`*foo`), a regex with no anchor, a high-cardinality aggregation with no size limit, a script query over every document — any of these can pin a data node. Put a query firewall in front of user input: cap aggregation sizes, disallow leading wildcards, set per-query time budgets (`timeout`), and use search throttling or a separate search-thread-pool sizing so an expensive query degrades that query rather than the whole node.

**The replication slot that ate the disk.** Specific to CDC on Postgres: a logical replication slot retains WAL until the consumer confirms it has read past it. If your Debezium connector stops (crash, deploy gone wrong, a poison message it keeps retrying) and nobody notices, the slot pins WAL, the primary's disk fills, and now your *transactional database* is at risk because of your *search pipeline*. Monitor slot lag and `pg_wal` size, and alert aggressively. The search index going stale is an annoyance; the primary running out of disk is an outage.

## The anti-pattern: treating the search index as a primary store

The single most expensive mistake teams make with this architecture is forgetting which box is the source of truth. The search index is fast, it has all the data, it has a nice query API — so someone starts writing data that *only* lives in the index, or starts treating an Elasticsearch query result as authoritative for a financial or access-control decision, or removes the database write "because the index already has it."

**Do not.** A search engine is a rebuildable, eventually-consistent, lossy projection. It is tuned for relevance and throughput, not durability or transactional correctness. Refresh semantics mean a just-written document is not immediately searchable. A node failure with insufficient replicas can lose recent writes. Analysis is lossy — the original text may not be perfectly reconstructable from the index. None of these matter for *search*, and all of them are disqualifying for a *system of record*. The contract that makes the whole architecture safe is exactly this: **the database can rebuild the index at any time, and the index can never rebuild the database.** The moment a piece of data exists only in the index, you have lost that contract, and you are one cluster failure away from permanent data loss. Keep the index a derived model. If you find yourself wanting to store authoritative data in it, that data belongs in the database, with the index as just another projection of it. This is also the dividing line between OLTP and analytical/derived stores — see [OLTP vs OLAP and columnar stores](/blog/software-development/database/oltp-vs-olap-and-columnar-stores) for the same source-of-truth-versus-derived-view reasoning applied to analytics.

## Case studies from production

### 1. GitHub: from MySQL `LIKE` to Elasticsearch, and then past it

GitHub's search history is the canonical version of this whole article. Early code and issue search leaned on the relational database and an external indexer, and it did not scale — searching across millions of repositories with `LIKE`-style queries was exactly the sequential-scan death spiral this post opens with. GitHub moved search to Elasticsearch, indexing issues, pull requests, repositories, and users into clusters fed from the database, which took an enormous and growing read load off the primary MySQL fleet. The interesting second act is that *code* search — searching the contents of source files across all of GitHub — eventually outgrew even Elasticsearch's model, and GitHub built a purpose-built engine (publicly described as "Blackbird") with its own inverted indices tuned for code. The lesson is layered: a general search engine is the right offload for most search, and *also* there is a scale and a query shape (substring search over trillions of lines of code) where even that is not enough and you build something bespoke. Knowing which regime you are in is the whole game.

### 2. Wikimedia: CirrusSearch and the migration off a Lucene bolt-on

Wikipedia's search ran for years on a MediaWiki extension backed by a standalone Lucene service that was operationally painful and limited. Wikimedia rebuilt search as CirrusSearch on top of Elasticsearch, indexing hundreds of wikis in dozens of languages — a serious analyzer problem, since relevance for Japanese, German, and English need different tokenization and stemming. The sync is driven from MediaWiki's edit stream: every page edit enqueues an index update, with periodic full reindex jobs as the self-healing backstop. The case is a clean illustration of two themes: the analyzer-per-language complexity that a row store has no answer for, and the edit-stream-plus-periodic-reindex pattern (a domain-specific outbox plus the safety-net reindex) as the sync strategy.

### 3. Uber: real-time CDC from the datastore to Elasticsearch

Uber has written about feeding Elasticsearch from their datastores via change capture to power search and discovery features — finding drivers, riders, trips, and support entities. The architecture is the textbook CDC pipeline: capture changes off the storage layer, push them through a streaming system, and apply them to Elasticsearch indices with the document id as the partition key so per-entity ordering is preserved. At Uber's change volume, dual writes were never on the table; the log-based approach is the only one that holds up when millions of entities are mutating continuously and several downstream systems (not just search) need the same change stream. This is the "many consumers" cell in the strategy table made concrete.

### 4. Netflix: DBLog and a general CDC framework feeding many sinks

Netflix built DBLog, a generic change-data-capture framework, precisely because they had the same problem in many places: derived data stores (Elasticsearch among them) that needed to stay in sync with source-of-truth databases. DBLog interleaves a backfill ("dump") with the ongoing log stream so a new consumer can bootstrap a full copy *and* stay current without a separate, racy reindex job — solving the exact backfill-while-streaming problem the reindex section describes. The takeaway is organizational as much as technical: once more than one team needs "keep a derived store in sync with this database," CDC stops being a per-feature hack and becomes shared infrastructure, and the investment in doing it well pays back across every consumer.

### 5. The dual-write that drifted for a quarter

A pattern I have seen more than once, with the details filed off: a team ships search with a dual write, it works fine in load tests, and it goes to production. Months later, support notices a trickle of "my item doesn't show up in search" tickets. Investigation finds the index is missing roughly a fraction of a percent of documents — every one corresponding to a moment when the ES bulk call had timed out or been rejected under load while the database commit succeeded. There was no error budget alarm because nothing errored from the user's perspective; the writes "succeeded." The fix was a full reindex from the database (which had everything, because it was the source of truth) plus a migration to an outbox so it could not recur. The lesson that stuck: **a dual write does not fail loudly; it fails statistically, and you find out from customers.** The reindex-from-source step only worked because nobody had committed the cardinal sin of storing data only in the index.

### 6. The reindex storm that took the cluster red

A team needed to change an analyzer and kicked off a reindex of a multi-hundred-million-document index — at full speed, in the middle of the business day, with `_reindex` running unthrottled. The reindex saturated the bulk thread pool, live indexing from the change stream started getting rejected, search latency spiked, and the cluster went yellow then red as nodes hit heap pressure. The incident had three independent fixes: throttle the reindex (`requests_per_second`), run it against a separate set of nodes or off-peak, and — the structural fix — always reindex into a *new* index behind an alias rather than touching the live one, so the rebuild's load is isolated and the cutover is a single cheap alias flip. The reindex was correct in intent and nearly catastrophic in execution, which is the most common shape of search incidents: the design was right, the operational care was missing.

### 7. The stuck replication slot that nearly filled the primary

A CDC pipeline on Postgres lost its Debezium connector to a bad deploy over a weekend. Because a logical replication slot retains WAL until its consumer advances, WAL accumulated on the primary for two days. Disk usage on the *transactional database* climbed toward full — which would have been a hard outage for everything, not just search. The on-call caught it from a disk-usage alert (not a search alert) and restarted the connector, which drained the backlog. Two changes followed: monitor replication-slot lag and `pg_wal` directory size with paging alerts, and add an automated guard that drops or warns on slots whose consumer has been dead beyond a threshold. The durable lesson: **a derived-data pipeline can endanger the source of truth it derives from**, so you must monitor the pipeline's back-pressure on the primary as carefully as you monitor the primary itself.

### 8. GitLab: advanced search and the cost of indexing everything

GitLab's Elasticsearch integration ("Advanced Search") indexes code, issues, merge requests, comments, and more, and the project has been refreshingly public about the scaling pain — managing index size for very large self-managed and SaaS instances, sizing shards, and the operational burden of keeping the index synced and reindexing after mapping changes. It is a working example of the sizing discipline from the operational section: you do not index everything at full fidelity; you choose a projection, you manage shard counts, and you treat the mapping as a migrated schema. The honesty in their docs about reindex requirements and resource sizing is exactly the expectation-setting most teams skip until production forces it.

## When to reach for a search engine, and when not to

Reach for an external search engine when:

- **Relevance quality matters** and `ts_rank`-grade ranking is visibly not good enough — users complain that the right result isn't at the top.
- **You need faceting and aggregations at high QPS** — counts-per-category, price histograms, "narrow your results" navigation — on every search, at a volume your primary cannot absorb.
- **You need typo tolerance, autocomplete, or fuzzy matching** as first-class features rather than after-the-fact hacks.
- **You want search load isolated from your transactional path** so a spike in search traffic can never threaten writes.
- **Multiple consumers need the same change stream** — search, analytics, a cache, a recommendation system — which tips you toward CDC as shared infrastructure feeding all of them.

Skip it — and stay in your existing database — when:

- **Postgres `tsvector` + `pg_trgm` already meets your relevance and volume needs.** Adding Elasticsearch to a system that a GIN index serves fine is pure operational cost for no user-visible gain.
- **Your corpus is small and changes rarely.** A few hundred thousand rows searched a few times a second does not need a distributed engine.
- **You cannot staff the sync pipeline and the cluster.** A search engine you cannot keep in sync is worse than no search engine — it is a confidently-wrong search engine. If you can't own the pipeline, don't add the dependency.
- **You would be tempted to make it a system of record.** If the data has nowhere else to live, it does not belong in a derived, eventually-consistent, rebuildable index. Fix that first.

The whole architecture rests on one discipline, so it is worth ending on it: the database is truth and the index is a projection of it. Every good decision in this post — CDC over dual writes, the outbox's atomic boundary, alias-swap reindexing, reading your own writes from the primary, refusing to store authoritative data in the index — falls out of taking that one sentence seriously. The teams that get search right are not the ones with the cleverest queries; they are the ones who never let the derived copy forget it is a copy.

## Further reading

- [Change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — the sync machinery underneath this whole post, in depth.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — the ordered, replayable buffer that makes CDC robust and fan-out cheap.
- [Polyglot persistence: choosing the right store](/blog/software-development/database-scaling/polyglot-persistence-choosing-the-right-store) — when a workload justifies a second, specialized data store at all.
- [OLTP vs OLAP and columnar stores](/blog/software-development/database/oltp-vs-olap-and-columnar-stores) — the same source-of-truth-versus-derived-view split, applied to analytics.
- [Read scaling with replicas](/blog/software-development/database-scaling/read-scaling-with-replicas) — the other half of the read-offload story, and the same eventual-consistency reasoning.
