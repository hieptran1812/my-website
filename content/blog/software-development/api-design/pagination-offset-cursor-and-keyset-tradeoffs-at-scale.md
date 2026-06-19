---
title: "Pagination: Offset, Cursor, and Keyset Trade-offs at Scale"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "The definitive pagination post — why a list endpoint must never return an unbounded collection, how offset paging gets slower the deeper you go and silently skips rows when the table shifts, how keyset and cursor paging fix both at the cost of page-jumping, and how to choose for a merchant's orders feed at scale."
tags:
  [
    "api-design",
    "api",
    "rest",
    "pagination",
    "cursor",
    "keyset",
    "http",
    "database",
    "scalability",
    "developer-experience",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/pagination-offset-cursor-and-keyset-tradeoffs-at-scale-1.png"
---

A merchant on our commerce platform opened a support ticket that read, in full: "Your nightly export is missing orders. I reconciled by hand and forty-one orders are gone." Nobody had deleted anything. The orders were in the database, visible in the dashboard, charged and shipped. They were simply never returned by the API the merchant's accounting integration called every night to walk the full order history. The integration did the obvious thing: `GET /v1/orders?limit=100&offset=0`, then `offset=100`, then `offset=200`, and so on until a page came back short. It had worked for months. What changed was volume. The merchant had grown to a few thousand orders a day, and the export ran for several minutes against a table that was being written to the entire time. New orders kept arriving at the top of the list while the export walked from the top down. Every insert shifted the window one row deeper, and every shift pushed exactly one order past the boundary between the page the export had already read and the page it was about to read. Over a multi-minute run, dozens of real orders fell into that crack and were never returned. The export was not buggy in the sense of a typo. It was buggy in the sense that it used the wrong pagination scheme for a list that changes while you read it.

That is the whole subject of this post, and it is one of the most consequential and most under-thought decisions in API design. Pagination looks like a detail — a `limit` and an `offset`, two query parameters, surely there is nothing to argue about. But the way you slice a collection into pages decides whether your list endpoint stays fast as the table grows to fifty million rows, whether a client walking the whole list sees every row exactly once, whether a caller can jump to page 900 or only step forward one page at a time, and whether you can change your storage layout in two years without breaking every integration that hard-coded the shape of your "next page" token. These are contract questions, not implementation details, and they have correct answers that depend on what your collection is and how it is used.

This post is the definitive treatment for the series. We will start from the one rule nobody is allowed to break — a list endpoint must never return an unbounded collection — and then walk the three real schemes in order of increasing sophistication: **offset/limit** paging and its two fatal flaws at scale, **keyset/seek** paging that fixes both by anchoring on a stable key instead of a row count, and the **opaque cursor** as the API surface you wrap around keyset so you can change the internals without breaking callers. We will derive the cost of each, show the wire on real `GET` requests and `SQL` queries against the Payments and Orders API that runs through this whole series, and end with a decision you can actually apply. The figure below previews the shape of a single page request as it resolves through the API, including the three paths a query can take once it reaches the handler.

![Diagram of a page request flowing from a client through the gateway to a handler that branches into an index seek path, an offset scan path, and an unbounded no-limit path that runs out of memory](/imgs/blogs/pagination-offset-cursor-and-keyset-tradeoffs-at-scale-1.png)

By the end you will be able to look at any collection endpoint and answer the question this whole series keeps returning to: *what does the caller get to assume about this list, and can I change how I page it later without breaking them?*

## 1. The one rule: never return an unbounded collection

Before we compare schemes, we have to establish why pagination exists at all, because the reason dictates everything that follows. The rule is simple and absolute: **a collection endpoint must never return all of its members in a single response.** Not "should usually paginate." Never returns the whole thing. The moment you write `GET /v1/orders` and let it return every order the merchant has ever placed, you have shipped a denial-of-service vulnerability against your own database, your own application server, your own network, and your own client — all four at once.

Walk the failure. A merchant with two years of history has, say, eight hundred thousand orders. The handler runs `SELECT * FROM orders WHERE merchant_id = :m`. The database materializes eight hundred thousand rows. Your application server holds all of them in memory while it serializes them to JSON — at roughly 600 bytes of JSON per order that is about 480 MB of string in the heap for one request. Your process either runs out of memory and the request that triggered it takes down every other request sharing that worker, or it survives and ships a 480 MB response over the wire. On a mobile link that transfer is measured in minutes, the socket times out long before it finishes, the client retries, and now you are doing it again. One careless endpoint, four simultaneous failure modes: the query, the heap, the transfer, and the retry storm. This is not a tail risk you mitigate later. It is a contract you must get right on the first commit, because the first big customer will find it.

So the contract is: **the server controls the maximum amount of data a single response can contain, and the client must be prepared to ask for more.** Every list endpoint takes a page-size parameter, the server enforces a hard ceiling on that parameter, and the response carries enough information for the client to request the next page. That is pagination, stripped to its essence. The three schemes we will compare — offset, keyset, cursor — are three different answers to the one sub-question pagination leaves open: *how does the client say "give me the next slice," and how does the server find that slice efficiently and correctly?*

### Bounded limit, with a maximum the server enforces

The first concrete decision is the page-size parameter and its ceiling. Accept a `limit` query parameter, give it a sane default, and — this is the part people skip — enforce a hard maximum that the server applies even when the client asks for more.

```http
GET /v1/orders?limit=20 HTTP/1.1
Host: api.paycommerce.example
Authorization: Bearer <token>
Accept: application/json
```

A client that omits `limit` gets the default (20 is a reasonable default for a UI-facing list; 100 for a machine-to-machine export). A client that sends `limit=10000` does not get ten thousand rows. It gets the server's maximum — say 100 — and, ideally, a hint that its request was clamped:

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "data": [ /* 100 orders, the enforced maximum */ ],
  "has_more": true,
  "next_cursor": "b3JkXzAxSE...",
  "limit": 100,
  "limit_requested": 10000
}
```

Why a hard maximum and not just "trust the client to be reasonable"? Because the page size is an attack surface. Without a ceiling, `?limit=50000000` is a request to materialize fifty million rows — the exact unbounded-collection failure we just outlawed, smuggled back in through a parameter. The maximum is not a courtesy; it is the enforcement mechanism for the no-unbounded-collection rule. A common ceiling for public APIs is 100; Stripe caps its `limit` at 100, GitHub at 100 (`per_page`), and Slack's conversation methods default to a smaller page with an enforced ceiling. The exact number is yours to choose based on payload size and downstream cost, but the existence of the ceiling is not optional.

#### Worked example: the request that tries to skip pagination

A merchant's engineer, frustrated that they have to loop, sends `GET /v1/orders?limit=1000000`. Here is the before and after of how a well-designed API responds versus a naive one.

**Before (naive):** the handler reads `limit` straight into the SQL `LIMIT` clause: `SELECT ... LIMIT 1000000`. The database obliges, the app serializes ~600 MB, the worker's heap blows past its limit, the process is killed by the out-of-memory killer, and every other request that worker was handling fails with a connection reset. One client's bad parameter became an outage for everyone on that worker.

**After (bounded):** the handler clamps. `effective_limit = min(requested_limit, MAX_LIMIT)` where `MAX_LIMIT = 100`. The query is `LIMIT 100`, the response is 100 orders plus a `next_cursor`, and the response body reports `"limit": 100, "limit_requested": 1000000` so the developer sees exactly what happened and learns to loop. The bad parameter is contained to a single, harmless, fast response. This is the difference between an API that fails closed and one that fails open, and it is one line of clamping code.

With the rule and the bounded limit established, we can compare the three ways to ask for "the next slice." We will start with the one everybody reaches for first, because it is the one built into every ORM and the one with the most dangerous failure modes at scale.

## 2. Offset pagination: the obvious one, and its O(offset) cost

Offset pagination is what you get when you think of a collection as a numbered list and a page as a window into it defined by two numbers: where to start (`offset`) and how many to take (`limit`). `?offset=40&limit=20` means "skip the first 40 rows, then give me the next 20." Page 3 of a 20-per-page list is `offset=40`. It is intuitive, it maps directly to SQL's `OFFSET`/`LIMIT`, and it gives you something the other schemes cannot: the ability to jump to any page directly, and, paired with a total count, to render a classic "Page 1 2 3 ... 900" pager. For small, mostly-static lists it is completely fine, and we will be honest about that in the recommendations section. The trouble is what happens when the list is large or changing, and there are two distinct kinds of trouble. The first is performance.

### Deriving the cost

Here is the keystone fact that most engineers never internalize: **offset pagination gets slower the deeper into the list you go, and the cost is linear in the offset.** It is not "the database jumps to row 1,000,000 and reads 20." There is no jump. To return rows 1,000,001 through 1,000,020, the engine must produce the rows in order and *count past* the first 1,000,000 — reading them, the relevant columns at least, and discarding them — before it can begin returning anything. The offset is not an index into an array; it is a count of rows to read and throw away.

So the work to serve a page at offset $k$ with page size $L$ is proportional to $k + L$ rows scanned, of which $k$ are discarded and only $L$ are returned. In big-O terms, an offset query costs $O(\text{offset} + \text{limit})$ in rows examined. Compare that to keyset, which we will see costs $O(\text{limit})$ — independent of how deep you are. Concretely, with a page size of 20:

- Page 1, `OFFSET 0`: read 20, return 20.
- Page 50, `OFFSET 980`: read 1,000, discard 980, return 20.
- Page 500, `OFFSET 9980`: read 10,000, discard 9,980, return 20.
- Page 50,000, `OFFSET 999980`: read ~1,000,000, discard ~999,980, return 20.

That last page does fifty thousand times the work of the first page to return the same twenty rows. `LIMIT 20 OFFSET 1000000` is expensive for exactly this reason: the million discarded rows are pure waste, paid on every deep page. The following figure stacks the cost of progressively deeper pages so the linear growth — and the discarded work — is visible at a glance.

![A vertical stack showing offset pagination cost rising from page one reading twenty rows to page fifty thousand reading a million rows, with the discarded rows and the order offset plus limit cost called out](/imgs/blogs/pagination-offset-cursor-and-keyset-tradeoffs-at-scale-2.png)

A nuance worth stating, because it is where people get tripped up: whether the discarded rows are cheap or expensive depends on whether the sort can be satisfied by an index. If you `ORDER BY created_at` and there is a B-tree index on `created_at`, the engine can walk the index in order and the discarded rows cost an index entry each rather than a full row read — still $O(\text{offset})$ work, just with a smaller constant. If the sort cannot use an index, the engine must sort the entire qualifying set first (a filesort), which is even worse: now the cost is dominated by sorting everything before it can offset into it. Either way the offset itself is linear. Indexing reduces the constant; it does not change the complexity. For why an index can or cannot serve a particular sort, this series defers to the database internals: [B-trees: how database indexes work](/blog/software-development/database/b-trees-how-database-indexes-work) and [composite, covering, and index-only scans](/blog/software-development/database/composite-covering-and-index-only-scans) cover exactly when the engine can walk an index in order versus when it has to sort.

### The wire for offset

```http
GET /v1/orders?offset=40&limit=20&sort=-created_at HTTP/1.1
Host: api.paycommerce.example
Authorization: Bearer <token>
```

```sql
SELECT id, merchant_id, amount, currency, status, created_at
FROM orders
WHERE merchant_id = :merchant_id
ORDER BY created_at DESC
LIMIT 20 OFFSET 40;
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "data": [ /* 20 orders */ ],
  "pagination": {
    "offset": 40,
    "limit": 20,
    "total": 8412
  }
}
```

Notice the `total` field. This is offset's other genuine advantage: because the scheme is built on counting, returning a total count and computing the number of pages is natural. The client can render "showing 41–60 of 8,412" and a page jumper. We will return to how expensive that `total` actually is — it is not free — in its own section. First, the second and more insidious problem with offset, the one that ate the merchant's forty-one orders.

## 3. The moving-window problem: offset skips and duplicates rows

The performance problem is annoying but at least it is honest: a deep page is slow, you can measure it, you can cache around it. The correctness problem is worse because it is silent. **Offset pagination skips and duplicates rows when the underlying collection changes between page fetches**, and it does so without any error, any warning, or any way for the client to detect it. The export that lost forty-one orders did not get an error. It got 200 OK on every page. It just quietly missed rows.

The mechanism is the moving window. Offset is defined relative to the *current* contents of the ordered list at the moment each query runs. Between fetching page 1 and page 2, the list can change — and for a live collection like a busy merchant's orders, it certainly will. If a new order arrives at the top of the list (we sort newest-first) after page 1 is fetched, every existing row shifts down by one position. The row that was at position 20 (the last row of page 1) is now at position 21. When the client fetches page 2 with `OFFSET 20`, the database skips the first 20 rows of the *new* list — which now includes the freshly inserted order at position 1 — and the row that used to be at position 20 is at position 21, so it gets skipped. The client never sees it. The figure below walks this as a timeline: the insert, the shift, and the skip.

![A timeline of the moving window problem showing page one fetched at offset zero, a new order arriving at the top, the list shifting down by one, page two fetched at offset twenty, and the old twentieth row being skipped so the client never sees it](/imgs/blogs/pagination-offset-cursor-and-keyset-tradeoffs-at-scale-3.png)

The symmetric failure happens on deletion or on sorting by a mutable field. If a row is *removed* from above the window between pages — an order is cancelled and filtered out, say — everything shifts *up* by one, and the row that was at position 21 (the first row of the page you are about to fetch) is now at position 20, which you already read. You see it twice. And if you sort by a field that changes — `ORDER BY updated_at` — a row whose `updated_at` is touched mid-walk can jump from a page you have not read yet to a page you already read, or vice versa, causing both a skip and a duplicate from a single update.

#### Worked example: an insert skips a row mid-page

Make it painfully concrete with small numbers. The merchant has 21 orders, sorted newest-first, page size 20. Call them `O21` (newest) down to `O1` (oldest).

**Page 1 at t0.** `OFFSET 0 LIMIT 20` returns `O21, O20, O19, ..., O2` — the 20 newest. The client has not yet seen `O1`.

```sql
-- t0: page 1
SELECT id FROM orders WHERE merchant_id = :m
ORDER BY created_at DESC LIMIT 20 OFFSET 0;
-- returns O21 .. O2  (20 rows)
```

**An insert at t1.** A new order `O22` is created. The list is now `O22, O21, O20, ..., O1` — 22 rows. Every prior row's position increased by one. `O2`, which was at position 20, is now at position 21. `O1`, the oldest, is at position 22.

**Page 2 at t2.** The client asks for the next page: `OFFSET 20 LIMIT 20`.

```sql
-- t2: page 2, after O22 inserted
SELECT id FROM orders WHERE merchant_id = :m
ORDER BY created_at DESC LIMIT 20 OFFSET 20;
-- skips O22, O21, ..., O3  (the first 20 of the NEW list)
-- returns O2, O1  ... wait
```

Walk the offset on the new 22-row list. Position 1 is `O22`, position 20 is `O3`. `OFFSET 20` skips positions 1 through 20 (`O22` down to `O3`) and returns positions 21 and 22: `O2` and `O1`. So the client's two pages together returned `{O21..O2}` and `{O2, O1}`. Order `O2` appears **twice** (it was the last row of page 1 and the first row of page 2), and — depending on the exact arithmetic and how many inserts land — orders can also be **skipped** entirely. The general statement: each insert above the window before a page fetch pushes one boundary row out of alignment, causing a skip or a duplicate at the page seam. Over a long walk with many inserts, the errors accumulate. That is how a multi-minute export over a busy table loses dozens of rows.

The crucial property to absorb is that **there is no error to detect.** Every page returns `200 OK` with a perfectly valid body. The client has no way to know it skipped `O2`'s neighbor or saw `O2` twice unless it deduplicates by ID and reconciles totals — which most clients do not, because the API gave them no reason to think they had to. The contract silently lied. This is why for any list that changes while it is read — and "a merchant's orders while new orders arrive" is the canonical example — offset is not merely slow, it is *wrong*. The fix is to stop counting positions in a moving list and start anchoring on a value that does not move. That is keyset pagination.

## 4. Keyset (seek) pagination: anchor on a stable key

Keyset pagination — also called seek pagination, or "the seek method," a term popularized by Markus Winand at *Use The Index, Luke* — throws away the idea of counting rows from the start. Instead of saying "skip the first 20 rows," it says "give me the rows that come *after this specific row I last saw*, in the total order." The page boundary is not a position (which moves) but a value (which does not). If the last row I saw had `created_at = '2026-06-18T10:00:00Z'` and `id = 'ord_01H...'`, then the next page is "the 20 rows whose `(created_at, id)` is less than `('2026-06-18T10:00:00Z', 'ord_01H...')` in the newest-first order." Inserts above that anchor do not change which rows come after it. The window is no longer moving.

### The query, and why the tiebreaker is mandatory

```sql
SELECT id, merchant_id, amount, currency, status, created_at
FROM orders
WHERE merchant_id = :merchant_id
  AND (created_at, id) < (:last_created_at, :last_id)
ORDER BY created_at DESC, id DESC
LIMIT 20;
```

Read the `WHERE` clause carefully because it carries the entire idea. `(created_at, id) < (:last_created_at, :last_id)` is a **row-value comparison** (also called a tuple or composite comparison): it compares the pair lexicographically, the way you compare words — first by `created_at`, and only when `created_at` ties does it fall back to comparing `id`. This is exactly the semantics you want, and crucially it matches the `ORDER BY created_at DESC, id DESC`. The order and the seek predicate use the same composite key. That alignment is what lets the database walk the index and never count past anything: it seeks straight to the anchor and reads forward.

Why the `id` tiebreaker? Because `created_at` is not unique. Two orders can be created in the same millisecond — on a busy platform, in the same microsecond. If your only sort key is `created_at` and three orders share a timestamp, then a page boundary that falls in the middle of those three is ambiguous: "rows after `created_at = X`" excludes all three, but "rows after-or-equal" includes all three, and you have no way to express "the two of the three I have not seen yet." You will either skip the two you have not read or re-read the one you have. **You need a deterministic total order**, and that means the sort keys, taken together, must uniquely identify every row. Appending the primary key `id` as the final tiebreaker guarantees uniqueness: even if `created_at` collides, `id` does not, so `(created_at, id)` is a total order with no ties. This is non-negotiable for keyset; a keyset query on a non-unique sort key is a latent skip/duplicate bug exactly as bad as offset's, just rarer.

### Deriving the cost: O(limit), independent of depth

Here is the payoff and the reason keyset exists. With a composite index on `(created_at, id)` — the same columns, in the same order, as the sort — the database can satisfy a keyset page in $O(\log n + \text{limit})$ time: $O(\log n)$ to *seek* to the anchor position in the B-tree (a tree descent, not a scan), then $O(\text{limit})$ to read the next 20 leaf entries sequentially. The $\log n$ seek is so cheap (a handful of page reads for even a billion-row table) that the practical cost is $O(\text{limit})$ — **independent of how deep into the list you are.** Page 1 and page 50,000 do the same amount of work. There is no offset to count past because there is no offset at all; the index entry *is* the anchor, and the engine jumps to it directly. The figure below shows the keyset query riding the composite B-tree: the seek, the forward read, and the role of the tiebreaker.

![A graph showing a keyset query with a row-value where clause and matching order by descending into a B-tree index on created_at and id, the tiebreaker id breaking timestamp ties, an order log n seek, and a forward read of twenty rows](/imgs/blogs/pagination-offset-cursor-and-keyset-tradeoffs-at-scale-5.png)

Contrast the two complexities directly, because this is the whole quantitative case for keyset. Offset at page depth $k$ costs $O(k + \text{limit})$ rows examined. Keyset at any depth costs $O(\text{limit})$ rows examined (plus a logarithmic seek). At page 1 they are identical. At page 50,000 with a million rows above the window, offset examines a million rows and keyset examines twenty. That is the difference between an export that finishes and one that times out, and between a "load more" button that stays snappy on page 900 and one that grinds to a halt.

### Stability under writes: the correctness win

The other half of keyset's value, and arguably the more important half, is correctness. Because the page boundary is a value (`(last_created_at, last_id)`) rather than a count, **inserts above the window are invisible to the paging.** When `O22` is created after the client fetched page 1, it does not shift the anchor. Page 2 still asks for "rows where `(created_at, id) < (O2_time, O2_id)`," which is `O1` and whatever else is older — exactly the rows the client has not seen, no more and no less. The newly inserted `O22` is newer than the anchor, so it is correctly *not* on page 2 (the client already moved past that part of the list). No skip, no duplicate. The figure below contrasts the two side by side: offset's moving window versus keyset's stable anchor.

![A before-and-after comparison with offset on the left showing a moving window where an insert shifts rows and page two skips a row, and keyset on the right showing a stable anchor where an insert above the window does not affect page two which seeks past the saved key](/imgs/blogs/pagination-offset-cursor-and-keyset-tradeoffs-at-scale-4.png)

There is an honest subtlety to state. Keyset gives you a *consistent forward walk*: you will see every row that existed before you started and was not deleted, exactly once, in order, even as new rows arrive. It does not give you a *snapshot* — a new row that sorts into territory you have already paged past will not retroactively appear on a page you already read (which is correct; you moved on), and a row deleted ahead of your anchor simply will not appear (also correct). For the "walk the whole list once" use case — exports, syncs, feeds — this is precisely the guarantee you want and offset cannot provide. If you need a true point-in-time snapshot across the whole walk, that is a different and heavier mechanism (a serializable snapshot, or paging against a versioned/append-only log), and you should reach for it only when the use case genuinely demands it.

#### Worked example: a keyset page and the anchor it carries

The merchant's UI shows the newest 20 orders and a "Load older" button. Page 1:

```sql
SELECT id, amount, currency, status, created_at
FROM orders
WHERE merchant_id = :m
ORDER BY created_at DESC, id DESC
LIMIT 20;
-- no WHERE on the key for the first page; just the newest 20
```

The last row of page 1 is, say, `created_at = '2026-06-18T09:14:22.530Z'`, `id = 'ord_01HZX...'`. Those two values are the anchor. The client (or, as we will see, the server, hidden inside a cursor) carries them into page 2:

```sql
SELECT id, amount, currency, status, created_at
FROM orders
WHERE merchant_id = :m
  AND (created_at, id) < ('2026-06-18T09:14:22.530Z', 'ord_01HZX...')
ORDER BY created_at DESC, id DESC
LIMIT 20;
```

While the merchant is reading page 1, eleven new orders come in. They all have `created_at` newer than the anchor, so the row-value comparison `(created_at, id) < (anchor)` excludes all eleven — they correctly do not appear on page 2. Page 2 returns the 20 orders immediately older than the anchor, exactly as if the eleven inserts never happened. The walk is stable. Run the same scenario with offset and page 2 would have skipped eleven orders' worth of boundary rows. That is the entire difference, made concrete.

There is one cost we have now incurred that offset did not have: the client has to carry an anchor (`(created_at, id)`) between pages, and that anchor exposes our internal sort keys and storage details on the wire. If a client hard-codes the shape "the cursor is a timestamp and an id," then the day we change our sort to include a third key, or switch from a timestamp to a monotonic sequence, we break every client. We need a way to hand the client "where to resume" without handing them our schema. That is the cursor.

## 5. Cursor pagination: the opaque token over keyset

A **cursor** is the API surface you put over keyset paging. It is an **opaque token** — a string the client treats as a black box, copies verbatim from one response into the next request, and never parses. Internally, the cursor encodes the keyset anchor (the last row's sort-key values). Externally, it is meaningless bytes. The convention is to base64url-encode a small structure that holds the anchor:

```json
{ "v": 1, "k": ["2026-06-18T09:14:22.530Z", "ord_01HZX..."], "dir": "next" }
```

base64url-encoded, that becomes something like `eyJ2IjoxLCJrIjpbIjIw...`, which is what the client sees as `next_cursor`. The figure below shows the encode and decode layers: the keyset key, the JSON envelope with a version, the base64url wrapping, the opaque token on the wire, and the decode-and-validate on the way back in.

![A stack of the cursor encode and decode layers showing the last keyset key encoded as versioned JSON, base64url wrapped into an opaque token, sent on the wire as next_cursor, then decoded and signature-validated, with the client treating it as opaque throughout](/imgs/blogs/pagination-offset-cursor-and-keyset-tradeoffs-at-scale-6.png)

### Why opaque? Because opacity is the contract that buys you evolvability

This is the single most important design decision in cursor pagination, and it is a pure contract argument — exactly the spine of this whole series. **The cursor is opaque so that you can change its internal scheme without breaking clients.** If clients treat the cursor as a black box they copy verbatim, then the internal encoding is *yours* to change. Today the cursor encodes `(created_at, id)`. Next quarter you add a `(status, created_at, id)` composite to support a new sort order, or you migrate orders to a sharded store and the cursor needs to carry a shard hint, or you switch from base64 JSON to a signed binary format. As long as the cursor stays opaque, every one of those changes is invisible to clients — they keep copying the token, and old tokens issued under the old scheme keep working because you version the payload (`"v": 1`) and your decoder handles both. The opacity *is* the forward-compatibility mechanism. Compare this to offset: `offset=40` is fully transparent, the client knows exactly what it means, and that transparency is precisely why you can never change what it means.

This is the robustness principle — the tolerant reader — applied to pagination. You publish the *minimum* the client needs (a token to resume from) and reserve the *maximum* freedom to change everything behind it. A client that parses your cursor is a client that has reached into your implementation and welded itself to it; an opaque cursor makes that impossible by construction, which protects both of you. (For the general rules of which changes are safe to make behind a contract, the sibling post on backward and forward compatibility goes deep; here the point is that opacity converts a schema-coupling problem into a non-problem.)

A practical hardening: **sign or HMAC the cursor** if you want to guarantee clients cannot forge or tamper with it, or detect when they have hand-edited one. A tampered cursor that decodes to a key the client should not be able to reach (another merchant's order, say) is an authorization bug if you trust it blindly; either re-scope every keyset query by the authenticated principal (always do this regardless) or sign the cursor so tampering is detectable. Signing also lets you treat an undecodable or invalid-signature cursor as a clean `400 Bad Request` rather than a confusing 500.

### The wire for cursor pagination

A first request omits the cursor; the response carries the cursor for the next page:

```http
GET /v1/orders?limit=20 HTTP/1.1
Host: api.paycommerce.example
Authorization: Bearer <token>
```

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "data": [ /* 20 orders, newest first */ ],
  "has_more": true,
  "next_cursor": "eyJ2IjoxLCJrIjpbIjIwMjYtMDYtMThUMDk6MTQ6MjIuNTMwWiIsIm9yZF8wMUhaWCJdfQ"
}
```

The client requests the next page by passing the cursor back verbatim:

```http
GET /v1/orders?limit=20&cursor=eyJ2IjoxLCJrIjpbIjIwMjYtMDYtMThUMDk6MTQ6MjIuNTMwWiIsIm9yZF8wMUhaWCJdfQ HTTP/1.1
Host: api.paycommerce.example
Authorization: Bearer <token>
```

When there are no more rows, `has_more` is `false` and `next_cursor` is `null`:

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "data": [ /* fewer than 20 orders, the tail of the list */ ],
  "has_more": false,
  "next_cursor": null
}
```

#### Worked example: encoding and decoding a cursor end to end

Walk one cursor through the full round-trip, in code, so the opacity is concrete. Server side, after fetching a page, encode the last row's keys:

```python
import base64, json

def encode_cursor(last_row, direction="next"):
    payload = {
        "v": 1,
        "k": [last_row["created_at"].isoformat(), last_row["id"]],
        "dir": direction,
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

def decode_cursor(cursor):
    if cursor is None:
        return None
    pad = "=" * (-len(cursor) % 4)            # restore base64 padding
    raw = base64.urlsafe_b64decode(cursor + pad)
    payload = json.loads(raw)
    if payload.get("v") != 1:                  # version gate for future schemes
        raise ValueError("unsupported cursor version")
    return payload
```

The handler turns a decoded cursor into the keyset `WHERE` clause and never lets the client's view of the cursor leak schema:

```python
def fetch_orders(merchant_id, cursor, limit):
    limit = min(limit, MAX_LIMIT)              # enforce the ceiling
    after = decode_cursor(cursor)
    params = {"m": merchant_id, "lim": limit + 1}  # fetch one extra to know has_more
    where_key = ""
    if after is not None:
        params["ts"], params["id"] = after["k"]
        where_key = "AND (created_at, id) < (:ts, :id)"
    sql = f"""
        SELECT id, amount, currency, status, created_at
        FROM orders
        WHERE merchant_id = :m {where_key}
        ORDER BY created_at DESC, id DESC
        LIMIT :lim
    """
    rows = db.query(sql, params)
    has_more = len(rows) > limit               # the extra row tells us
    rows = rows[:limit]
    next_cursor = encode_cursor(rows[-1]) if has_more and rows else None
    return {"data": rows, "has_more": has_more, "next_cursor": next_cursor}
```

Two patterns in that handler are worth calling out because they are the standard tricks. First, **fetch `limit + 1` rows** and slice back to `limit`: if the extra row exists, there is a next page (`has_more = true`); if it does not, you have reached the tail. This computes `has_more` for free without a separate count query — the single biggest reason cursor APIs can return `has_more` cheaply while exact totals are expensive. Second, the cursor's `(created_at, id)` is decoded straight into bind parameters of the row-value comparison; the client passed back exactly enough for the server to seek, and not one bit more. The client copied an opaque string; the server reconstructed a keyset query. That is the whole cursor pattern.

## 6. The trade-offs, head to head

We now have three schemes and can compare them honestly on the axes that actually matter. None wins on every axis — that is the whole reason the choice is interesting rather than obvious. Offset buys you two things the others cannot: arbitrary page-jumping and a natural total count. Keyset and cursor buy you two things offset cannot: speed that is independent of depth and correctness under concurrent writes. The figure below lays the comparison out as a matrix across the four axes, and the table after it spells out the details.

![A matrix comparing offset, keyset, and cursor pagination across deep-page cost, stability under writes, page jumping, and total count, showing offset as order offset plus limit and unstable but jumpable, while keyset and cursor are order limit and stable but next-only](/imgs/blogs/pagination-offset-cursor-and-keyset-tradeoffs-at-scale-7.png)

| Axis | Offset / limit | Keyset / seek | Cursor (opaque keyset) |
| --- | --- | --- | --- |
| **Deep-page cost** | $O(\text{offset} + \text{limit})$ — slower the deeper you go; `OFFSET 1000000` scans a million rows | $O(\text{limit})$ — flat at any depth, with an $O(\log n)$ index seek | $O(\text{limit})$ — same as keyset, it *is* keyset underneath |
| **Stability under writes** | Skips and duplicates rows when the list changes mid-walk; silent, no error | Stable forward walk — inserts above the window are invisible | Stable — inherits keyset's guarantee |
| **Page jumping** | Yes — jump to any page directly (`offset = page × size`) | No — only next (and prev, with a reversed query); no random access | No — next/prev only; the token is just "resume here" |
| **Total count** | Natural — `total` and page count are cheap-ish to expose | Awkward — needs a separate count query, which is expensive | Usually `has_more` instead of an exact total |
| **Client/wire shape** | Transparent `offset`/`limit` — and frozen forever because it is transparent | Exposes sort keys (`last_ts`, `last_id`) on the wire | Opaque token — internals are yours to change |
| **Best fit** | Small, bounded, mostly-static lists; admin UIs with page jumpers | Large or live collections; full-table walks, exports, syncs | The public API surface over keyset for any external client |

The table makes the decision tractable. If your collection is small and static and your users want to jump to page 5, offset is not just acceptable, it is the *right* choice — keyset cannot give you the page jumper. If your collection is large or changing and your users walk it forward (a feed, an export, a "load more"), keyset is correct and offset is a latent data-loss bug. And the cursor is simply how you expose keyset across a contract boundary you want to keep evolvable. A subtle but real point: **keyset and cursor are not alternatives to each other.** Keyset is the *mechanism* (the SQL); cursor is the *surface* (the opaque token over that mechanism). You almost always want both: keyset internally for the cost and correctness, a cursor on the wire for the opacity. "Keyset vs cursor" is a false dichotomy; the real choices are offset-vs-keyset (mechanism) and transparent-vs-opaque (surface).

### Prev pages and bidirectional cursors

A frequent objection to keyset is "but I need a Previous button." You can have one. To page *backward* from an anchor, flip both the comparison and the order, then reverse the result rows so they come back in the user's expected order:

```sql
-- previous page: rows NEWER than the anchor, the limit closest to it
SELECT id, amount, currency, status, created_at
FROM orders
WHERE merchant_id = :m
  AND (created_at, id) > (:ts, :id)         -- '>' not '<'
ORDER BY created_at ASC, id ASC             -- ASC not DESC
LIMIT 20;
-- then reverse the returned rows in application code so they read newest-first
```

The cursor's `"dir"` field (`"next"` / `"prev"`) tells the server which form to build. This is why APIs like Stripe expose both `starting_after` and `ending_before`: one walks forward from a row, the other walks backward from a row, and both are keyset seeks, not offsets. What keyset still cannot give you is "jump directly to page 47" — there is no anchor for a page you have never visited, because the anchor *is* a row you saw. If a client genuinely needs random page access, that is the signal to use offset (and accept its costs) or to rethink whether random page access is a real requirement or a UI habit.

## 7. Response shape and the standards for "where's the next page"

Whatever scheme you pick, you have to put the "how to get more" information somewhere in the response, and there are two well-trodden conventions. Pick one and apply it consistently across every list endpoint — consistency here is a developer-experience multiplier, because a client that learns your pagination once should never have to relearn it per endpoint.

### Option A: pagination metadata in the JSON body

The most common shape for JSON APIs: a `data` array plus a sibling object carrying the cursor or offsets and a `has_more` flag.

```json
{
  "data": [
    { "id": "ord_01HZX9...", "amount": 4999, "currency": "usd", "status": "paid", "created_at": "2026-06-18T09:20:11.004Z" },
    { "id": "ord_01HZX8...", "amount": 1299, "currency": "usd", "status": "paid", "created_at": "2026-06-18T09:19:58.220Z" }
  ],
  "has_more": true,
  "next_cursor": "eyJ2IjoxLCJrIjpb...",
  "prev_cursor": "eyJ2IjoxLCJrIjpb..."
}
```

Note `amount` is an integer of minor units (`4999` = \$49.99), not a float — a separate body-design rule, but worth flagging because pagination responses are where amounts most often leak as floats. The key choices here are: keep the page itself in a top-level `data` array (never make the client guess which field is the list), put pagination siblings next to it, and prefer `has_more` over an exact `total` unless you can afford the count (next section). Stripe uses exactly this shape: `{ "object": "list", "data": [...], "has_more": true }`, with `starting_after`/`ending_before` as the request-side cursor params.

### Option B: the Link header (RFC 8288 web linking)

The other standard, used by GitHub's REST API, keeps pagination out of the body entirely and puts it in a `Link` response header with typed relations (`rel="next"`, `rel="prev"`, `rel="first"`, `rel="last"`):

```http
HTTP/1.1 200 OK
Content-Type: application/json
Link: <https://api.paycommerce.example/v1/orders?cursor=eyJ2IjoxLCJrIjpb...&limit=20>; rel="next",
      <https://api.paycommerce.example/v1/orders?limit=20>; rel="first"

[ { "id": "ord_01HZX9...", "amount": 4999 }, ... ]
```

The Link-header approach has a real virtue that fits this series' spine: the server hands the client a *complete URL* to follow, so the client does not even have to know how to assemble the next-page request — it follows the link, exactly like a hyperlink. That is hypermedia in the small, and it means the server can change the pagination parameters (swap `cursor` for something else) without the client editing a URL, because the client never built the URL in the first place. It pairs naturally with the broader hypermedia argument in the sibling post on [HATEOAS in the real world](/blog/software-development/api-design/hateoas-in-the-real-world-hypermedia-links-and-when-to-skip). The downside is that header parsing is fiddlier for clients than reading a JSON field, and you cannot carry rich metadata (page counts, totals) as cleanly. Both options are correct; the cardinal sin is mixing them across endpoints so clients never know where to look.

## 8. Total counts: why exact counts are expensive

Offset's headline advantage is the total count — "showing 41–60 of 8,412" — so it is worth being precise about what that count actually costs, because it is the hidden tax on the page-jumping you bought. **An exact count of a filtered collection is not free; it is often as expensive as scanning the whole filtered set.** `SELECT COUNT(*) FROM orders WHERE merchant_id = :m` has to count every qualifying row. With an index on `merchant_id` the engine can count index entries rather than reading full rows (an index-only count), which is cheaper, but it is still $O(\text{matching rows})$ — for a merchant with 800,000 orders, that is 800,000 index entries counted on *every page load* if you re-run the count each time. You bought a page jumper and paid for it with a full-set scan per page.

There are a few honest ways out, in increasing order of how much you give up:

- **Don't return an exact total.** Return `has_more` instead. Most UIs do not actually need "of 8,412"; they need "is there a next page," which the fetch-`limit+1` trick answers for free. This is why cursor APIs almost universally drop exact totals.
- **Return an approximate count.** For a "roughly 8,400 results" UI, you can read a cached or estimated count — for example, a periodically refreshed materialized count, or the planner's row estimate from `pg_class.reltuples` / `EXPLAIN`. It is approximate and can be stale, but it is $O(1)$ and good enough for a fuzzy "about N results" label.
- **Cache the count.** Maintain a counter you increment/decrement on insert/delete (a denormalized `order_count` per merchant), and read it in $O(1)$. This trades write-time cost and a consistency window for read-time speed.
- **Cap the count.** Some APIs count up to a ceiling and say "1000+" — you scan at most 1,001 rows to know whether the count exceeds 1,000. GitHub's search API famously caps results this way.

The general rule: **expose an exact total only when the use case truly needs it and you have budgeted the count's cost; otherwise expose `has_more`.** The cost of `COUNT(*)` over a large filtered set, and the index structures that make it cheaper or not, are squarely database-internals territory — see [composite, covering, and index-only scans](/blog/software-development/database/composite-covering-and-index-only-scans) for when a count can be served from an index alone versus when it must touch the heap.

## 9. The problem, reasoned end to end: paging a merchant's live orders

Let us put it all together on the running example and reason from requirements to a decision, then stress-test it. The requirement: a merchant integration needs to (a) display the newest orders in a dashboard with a "load older" button, and (b) run a nightly export that walks the *entire* order history into the merchant's accounting system. Same collection, two access patterns, and the orders table is written to continuously all day.

**Step 1 — outlaw the unbounded response.** Both access patterns page. The export does not get to ask for "all orders"; it walks in bounded pages like everyone else. `limit` defaults to 100 for the export (machine-to-machine, larger payloads acceptable), 20 for the dashboard, both capped at 100. This is non-negotiable from section 1.

**Step 2 — reject offset for the export immediately.** The export runs for minutes against a table receiving thousands of inserts. Offset's moving-window skip is not a risk here; it is a certainty, and it is exactly the bug that opened this post. The export must use a scheme stable under writes. That rules in keyset.

**Step 3 — use keyset for both, exposed as a cursor.** The dashboard's "load older" is a forward walk — keyset gives it $O(\text{limit})$ pages that stay fast as the merchant's history grows, and stability so a new order arriving while the user scrolls does not duplicate or skip. The export's full walk is the canonical keyset use case. Both get the same opaque `next_cursor`, so the integration learns one pagination scheme, not two, and we keep the freedom to change the internals.

**Step 4 — `has_more`, not `total`.** Neither use case needs an exact total. The dashboard needs "is there an older page" (`has_more`). The export needs "keep going until `has_more` is false." We skip the per-page `COUNT(*)` tax entirely. If product later insists on "about N orders," we add an approximate cached count, not an exact per-request one.

Now stress-test the decision, which is where designs earn their keep:

- **The client retries a page on a timeout.** Because the cursor is the *same* opaque token, retrying the request returns the same page (modulo rows deleted in between). Cursor pagination is naturally retry-friendly: there is no "current position" state on the server that a retry could advance past. Contrast offset, where a retry is also stable, but the *next* request after a successful-but-unacknowledged page can still skip via the moving window. Pair this with proper idempotency on writes (a sibling post in this series) and the whole walk is safe to retry.
- **Two writers race — an order is created and another cancelled mid-walk.** The created order, being newer than the export's current anchor, is simply not in the part of the list the export has already passed; it was either already returned (if newer than where the export started) or is correctly excluded (older walks never go backward). The cancelled order, if it is filtered out of the list, just does not appear — the export sees the list as it stands when each page query runs, with a stable forward boundary. No skip of unrelated rows, no duplicate.
- **The collection has 50 million rows.** Keyset does not care. Page 1 and page 2,500,000 both seek-and-read 100 rows. The export's per-page latency is flat; total time is $O(n / \text{limit})$ pages, which is unavoidable (you are reading $n$ rows) but with the minimum possible per-page constant. Offset would have made the last pages of that export individually slower than the entire first half combined.
- **A row's `created_at` is mutable and gets updated mid-walk.** This is the one case to be careful about: if you sort by a field that changes, even keyset can skip or duplicate, because a row can move across your anchor. The fix is to sort by an **immutable** key for paging. `created_at` is set once and never changes (use it, with `id` as tiebreaker); `updated_at` changes, so never paginate by `updated_at` if you need a stable walk. If clients must sort by a mutable field for display, paginate by the immutable key underneath and sort the page in the UI, or accept that "sort by recently updated" is a best-effort view, not a stable walk.
- **The payload is 10× bigger than planned** because someone added a fat `line_items` array to the order representation. Pagination interacts with payload size: a page of 100 orders at 6 KB each is a 600 KB response, which on a cold mobile link is meaningfully slow. The lever is the page size *and* sparse fieldsets — let the client ask for only the fields it needs (covered in the sibling post on [filtering, sorting, and sparse fieldsets](/blog/software-development/api-design/filtering-sorting-and-sparse-fieldsets-without-reinventing-sql)). Pagination bounds the row count; sparse fieldsets bound the row size; you need both to bound the response.

The decision holds up under every stress, and the one genuine landmine (sorting by a mutable key) has a clear rule. This is what it means to design a list endpoint as a contract: you pick the scheme by the forces (size, mutability, access pattern, evolvability), you expose it through a surface that lets you change your mind later, and you can answer precisely what breaks under concurrency, retries, and scale.

## 10. Edge cases that bite in production

The three schemes are clear in the abstract; the failures that page you at 2 a.m. live in the corners. Here are the ones worth designing for before they find you.

**Pagination and filtering must share the same total order.** The moment you let clients filter (`?status=paid`) or sort (`?sort=-amount`) alongside paging, the cursor has to encode the anchor *for that sort*, and the index has to support *that* order. A cursor minted for the default `created_at DESC` order is meaningless against an `amount DESC` query, because "the row after this one" depends entirely on which order "after" is measured in. The robust design ties the cursor to the query parameters that produced it: encode the sort key set into the cursor payload (or refuse a cursor whose sort does not match the request), and ensure every sort you support has a composite index ending in the unique `id` tiebreaker. If a client sends a cursor from one sort against a different sort, return a clean `400 Bad Request` — never silently apply it, because silently applying a mismatched cursor is the moving-window bug wearing a disguise. This is where pagination and the sibling concern of filtering and sorting become one design; the [filtering, sorting, and sparse fieldsets](/blog/software-development/api-design/filtering-sorting-and-sparse-fieldsets-without-reinventing-sql) post owns the query-param half, and the contract between them is exactly "the cursor is valid only for the sort that minted it."

**Sorting by a non-indexed field reintroduces the offset cost — to keyset.** Keyset is only $O(\text{limit})$ if the sort can be served by an index in order. If a client sorts by a column with no matching composite index, the engine must sort the whole qualifying set before it can seek, and you are back to scanning everything per page — worse than offset, because you do it on every page with no caching. The rule: **only allow sorting on columns you have indexed for it.** A finite allow-list of sort fields, each backed by a `(sort_col, id)` composite index, keeps every page on the fast path. An open-ended `?sort=<any column>` is how a client `?sort=`s your database into the ground, and it is a documented denial-of-service vector for APIs that pass user sort fields straight into SQL.

**The deep-jump request on a cursor API.** A client trained on offset will sometimes want page 500 of a cursor-paginated endpoint and there is no token for a page they have never visited. Do not fake it by decoding the cursor into an offset behind the scenes — that silently reintroduces the $O(\text{offset})$ cost and the moving window you adopted cursors to escape. The honest answers are: tell the client the endpoint supports forward/backward walking only (most clients that *think* they need page 500 actually need search or a filter that narrows the set), or, if random access is a hard requirement, expose a separate offset-paginated variant with its costs documented, or — best — give them a filter (`?created_before=<date>`) that lets them *seek* to a region by value rather than by page number, which is keyset's natural idiom anyway.

**Empty pages and the tail.** A keyset/cursor walk ends when a page comes back with fewer than `limit` rows and `has_more` is `false`. A subtle bug: if you compute `has_more` by checking `len(rows) == limit` instead of fetching `limit + 1`, you will report `has_more: true` on a page that happens to be exactly `limit` rows even when it is the last page, sending the client to fetch one more empty page. The fetch-`limit + 1` trick avoids this — `has_more` is true if and only if the extra row exists. Always make the final page detectable, and make a request past the end return an empty `data` array with `has_more: false` rather than an error; clients loop until `has_more` is false, and an error on the natural terminal condition breaks that loop.

**Cursor expiry and schema changes.** If you ever change the keyset scheme (add a sort key, change a column type), old cursors minted under the old scheme are in the wild — a client may resume a walk tomorrow with a cursor from today. This is exactly why the cursor payload carries a version (`"v": 1`): your decoder dispatches on it and can translate or gracefully reject old versions with a `400` that says "cursor expired, restart pagination" rather than a 500. Decide your policy explicitly — translate old cursors, or expire them with a clear error — and document it, because a client that gets a 500 on a stale cursor cannot tell a bug from a deliberate change. Signing the cursor (section 5) makes "is this one of mine, and which version" a cheap, tamper-evident check.

#### Worked example: classifying a pagination change as breaking or not

You want to add a secondary sort and improve a cursor. Which changes break clients? Apply the tolerant-reader rules from the compatibility sibling post:

- **Adding a new optional `sort` value** (e.g. now you also support `sort=-amount`): *non-breaking.* Existing clients never send it; their default-sorted cursors keep working.
- **Changing the cursor's internal encoding** (base64 JSON → signed binary), keeping it opaque: *non-breaking* for clients that treat it as opaque — which is the whole point of opacity — *as long as* old tokens still decode (handle `"v": 1` alongside the new `"v": 2`). It is **breaking** for any client that parsed the cursor, which is why you forbade that by making it opaque.
- **Changing the default sort order** (newest-first → oldest-first): *breaking.* Existing clients' cursors are anchored in the old order; the same cursor now means a different "next." Treat a default-sort change like a version bump.
- **Lowering the `limit` ceiling** from 100 to 50: *technically breaking* for a client sending `limit=100` and depending on 100-row pages, but a clamp (return 50, report `limit_requested: 100`) degrades gracefully rather than erroring. Raising the ceiling is safe.

The classification falls right out of the series spine: a change is safe if no caller's existing assumption is invalidated. Opacity shrinks the set of assumptions a caller is *allowed* to hold about your cursor down to one — "copy it verbatim" — which is precisely why opaque cursors are so much more evolvable than transparent offsets.

## 11. A concrete latency walkthrough (offset vs keyset at depth)

Numbers make the complexity argument visceral, so here is an order-of-magnitude walkthrough. These are approximate and depend heavily on your hardware, cache state, and row width — treat them as illustrative ranges, not a benchmark, and measure your own. Setup: a single `orders` table, ~10 million rows for one large merchant, a B-tree index on `(merchant_id, created_at, id)`, page size 20, sorted newest-first.

For **offset**, the cost is dominated by counting past the offset. Walking an index in order to discard rows is fast per row but not free, and the row count is what grows:

| Page | Offset | Rows examined | Approx. latency (warm cache) |
| --- | --- | --- | --- |
| 1 | 0 | 20 | a few ms |
| 100 | 1,980 | ~2,000 | low tens of ms |
| 5,000 | 99,980 | ~100,000 | hundreds of ms |
| 250,000 | 4,999,980 | ~5,000,000 | seconds — and climbing |

For **keyset**, every page is the same seek-plus-read regardless of depth:

| Page | Anchor | Rows examined | Approx. latency (warm cache) |
| --- | --- | --- | --- |
| 1 | none | 20 | a few ms |
| 100 | last key | ~20 (+log n seek) | a few ms |
| 5,000 | last key | ~20 (+log n seek) | a few ms |
| 250,000 | last key | ~20 (+log n seek) | a few ms |

The shape is the entire point. Offset's per-page latency climbs linearly with depth until a deep page costs seconds and either times out or holds a database connection long enough to starve the pool; on a busy table that one slow query at page 250,000 can cascade into elevated p99 across *every* endpoint sharing the connection pool, because it is holding a connection for seconds. Keyset's per-page latency is a flat line. The tail-latency consequence is what matters operationally: with offset, your list endpoint's p99 is a function of how deep your worst client pages, which you do not control; with keyset, your p99 is bounded by `limit` regardless of client behavior, which you do control. That predictability — a p99 you can reason about because it does not depend on caller depth — is worth as much as the raw speed. The exact per-page cost of the index seek and the in-order walk is storage-engine behavior; the [B-trees post](/blog/software-development/database/b-trees-how-database-indexes-work) explains why the $O(\log n)$ descent is a handful of page reads even for a ten-million-row index, and the [index-only scans post](/blog/software-development/database/composite-covering-and-index-only-scans) explains when the page can be served entirely from the index without touching the heap (which roughly halves the work again).

## 12. Case studies: how the big APIs paginate

These are accurate as of widely documented public behavior; the point is to show that the schemes above are exactly what production APIs converged on, and where they differ and why.

**Stripe — cursor pagination with `starting_after` / `ending_before`.** Stripe's list endpoints (`/v1/charges`, `/v1/customers`, and so on) return a `list` object: `{ "object": "list", "data": [...], "has_more": true, "url": "..." }`. You page forward with `starting_after=<id>` and backward with `ending_before=<id>`, where the value is the `id` of the last/first object you saw. `limit` defaults to 10 and is capped at 100. This is keyset pagination with the object `id` (which is sortable and unique) as the anchor, exposed as request parameters rather than an opaque blob — Stripe chose ID-anchored over a fully opaque token, which is a reasonable variant because their IDs are stable and meaningful. There is no `total` field; you get `has_more`. Stripe explicitly documents that this is to keep pagination stable and fast as objects are created concurrently — the exact moving-window argument from section 3.

**GitHub — `Link` header pagination, page-based and cursor-based.** GitHub's REST API uses the RFC 8288 `Link` header with `rel="next"`, `"prev"`, `"first"`, `"last"`, controlled by `per_page` (max 100) and historically `page` (an offset-style page number). For large or hot resources, GitHub has moved several endpoints to cursor-based pagination (still surfaced via the `Link` header) precisely because page-number/offset paging degrades and drifts on large, active datasets. The lesson: GitHub demonstrates both the convenience of offset-style page numbers for moderate collections and the migration to cursors for the ones that grew — and the `Link` header let them change the underlying scheme without clients rewriting URL construction, because clients follow the `rel="next"` URL rather than building it.

**Slack and Twitter/X — opaque cursors.** Slack's Web API methods that return lists (`conversations.list`, `users.list`) use a `response_metadata.next_cursor` opaque string that you pass back as the `cursor` parameter; an empty `next_cursor` means the end. Twitter/X's v2 API uses `next_token` / `pagination_token` the same way — an opaque token you echo back to get the next page. Both are textbook opaque cursors over keyset, and both chose opacity for the evolvability reason in section 5: the token is meaningless to clients, so the providers can change what it encodes. The convergence across four very different companies on cursor/keyset for their large, live collections is the strongest practical argument that, at scale, this is simply the right answer.

The through-line: **everybody offers offset/page-number paging for small or moderate collections because it is convenient, and everybody moves to keyset-behind-a-cursor for the large, write-heavy ones because offset is wrong there.** That is exactly the recommendation this post is about to make.

## 13. When to reach for each (and when not to)

Pagination is a trade-off like everything else, so here is the decisive guidance, stated plainly including when *not* to do each thing. The figure below renders the decision as a small tree keyed on the two questions that actually decide it: how big and live is the collection, and do clients need to jump to arbitrary pages.

![A decision tree for choosing pagination branching on whether the collection is a small bounded list or a large live list, then to offset with page jumps or cursor for stable feeds and keyset for full exports](/imgs/blogs/pagination-offset-cursor-and-keyset-tradeoffs-at-scale-8.png)

**Reach for offset/limit when** the collection is small and bounded (an admin list of a few hundred config rows, a list that will never realistically exceed a few thousand entries), is mostly static (it does not churn while users page it), and your users genuinely benefit from jumping to arbitrary pages or seeing an exact "X of Y" count. For an internal admin tool listing the platform's 200 webhook endpoints, offset is the *right* choice — keyset would deny you the page jumper for no benefit, because the list is small enough that $O(\text{offset})$ is trivially cheap and stable enough that the moving window never bites. **Do not reach for offset when** the collection is large (deep pages get slow) or live (you will silently skip and duplicate rows) — which is to say, do not use offset for any user-facing feed, any export, or any list that grows without bound. The merchant's orders table fails both tests; offset is disqualified.

**Reach for keyset/seek when** the collection is large or written-to-while-read — feeds, timelines, full-table exports, syncs, anything where you walk forward and must see every row once. It is the only scheme that is both fast at depth and correct under concurrent writes. **Do not reach for keyset when** clients legitimately need random page access (it cannot provide page jumping) or when you cannot establish an immutable, total sort order (no stable tiebreaker means latent skip/dup bugs). If you find yourself wanting to paginate by a mutable column, stop and either page by an immutable one underneath or accept the view is best-effort.

**Reach for a cursor (opaque token over keyset) when** the endpoint is a contract you want to keep evolvable — which is to say, essentially always for an external/public API, and usually for an internal one too. The opacity costs you almost nothing and buys you the freedom to change storage, sort, sharding, and encoding later. **Do not reach for an opaque cursor when** you are building a throwaway internal endpoint with a single co-deployed client and you genuinely value the debuggability of a transparent `offset`/`page` over future evolvability — but be honest that "throwaway" and "single client" rarely stay true. And **do not** expose a transparent keyset cursor (raw `last_ts`/`last_id` on the wire) on a public API unless you are prepared to support that exact shape forever; you have given up the one thing — opacity — that made cursors better than transparent offset for evolvability.

One more anti-pattern to name: **do not page in the client.** Fetching everything and slicing in JavaScript reintroduces the unbounded-collection failure from section 1 with extra steps. Pagination is a server-side contract; the client asks for a page and the server decides how to find it efficiently.

## 14. Key takeaways

- **Never return an unbounded collection.** A list endpoint always paginates, always enforces a hard maximum `limit` (clamp, do not trust the client), and always tells the client how to get more. The page size is an attack surface; the ceiling is the enforcement.
- **Offset is $O(\text{offset} + \text{limit})$ and gets slower the deeper you go.** `LIMIT 20 OFFSET 1000000` scans a million rows to return twenty, because the offset is a count of rows to read and discard, not a jump. Keyset is $O(\text{limit})$ at any depth.
- **Offset silently skips and duplicates rows when the list changes mid-walk.** The moving window shifts under concurrent inserts and deletes, and every page returns `200 OK` while quietly losing or repeating rows. For any live collection, offset is not slow — it is wrong.
- **Keyset anchors on a stable value, not a position.** `WHERE (created_at, id) < (:last_ts, :last_id) ORDER BY created_at DESC, id DESC LIMIT 20` seeks straight to the last row you saw and reads forward — fast and stable under writes, riding a composite index.
- **You need a deterministic total order with a unique tiebreaker.** Append the primary key (`id`) to the sort so no two rows tie; a keyset query on a non-unique sort key is a latent skip/duplicate bug. Sort by an *immutable* key, never a mutable one.
- **A cursor is an opaque token over keyset, and opacity is the contract.** Base64-encode the anchor so the client treats it as a black box; that lets you change the internal scheme — sort keys, sharding, encoding — without breaking a single client. Transparency is exactly why you can never evolve `offset`.
- **Offset gives page-jumping and totals; keyset/cursor give speed and stability.** Pick by force: small/static/needs-jumping → offset; large/live/forward-walk → keyset behind a cursor. Keyset and cursor are mechanism and surface, not rivals.
- **Exact totals are expensive — prefer `has_more`.** `COUNT(*)` over a large filtered set scans the whole set; expose an exact total only when the use case needs it and you have budgeted the cost, otherwise return `has_more` (the fetch-`limit+1` trick gives it for free) or an approximate count.

This is the contract question the whole series keeps asking, answered for lists: the caller gets to assume your pages are bounded, fast, and — if you chose right — complete-exactly-once under writes, and the opaque cursor means you can change everything behind that assumption later without breaking them. For the bigger picture of how this fits into a correct, evolvable, pleasant API, start at the series intro, [what is an API: the contract between systems](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems), and end at the [API design playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2). The URI-design choices that determine where these collections live are in [choosing URIs: collections, sub-resources, path vs query](/blog/software-development/api-design/choosing-uris-collections-sub-resources-path-vs-query).

## Further reading

- **Markus Winand — *Use The Index, Luke!*, "Paging Through Results" / "The Seek Method."** The canonical, database-grounded explanation of why offset is $O(\text{offset})$ and how keyset/seek pagination rides the index for $O(\text{limit})$ pages. Read it for the EXPLAIN plans that prove the cost.
- **RFC 8288 — Web Linking.** The `Link` header and typed relations (`next`, `prev`, `first`, `last`) that GitHub and others use to carry pagination out of the body.
- **Stripe API reference — Pagination.** The cursor model with `starting_after` / `ending_before`, `has_more`, and the 100-item `limit` cap; a clean production example of keyset-via-cursor with no exact totals.
- **GitHub REST API docs — "Using pagination in the REST API."** The `Link`-header convention, `per_page`, and the move toward cursor-based pagination for large datasets.
- **PostgreSQL documentation — `LIMIT` and `OFFSET`, and the row-value comparison syntax.** The exact semantics of `OFFSET` (rows are computed and discarded) and tuple comparison `(a, b) < (x, y)` that keyset relies on.
- **Within this series:** the [database B-trees post](/blog/software-development/database/b-trees-how-database-indexes-work) and the [composite, covering, and index-only scans post](/blog/software-development/database/composite-covering-and-index-only-scans) for why the index seek is $O(\log n)$ and when a count or a sort can be served from the index alone — the storage-layer reasons the costs in this post hold.
