---
title: "Filtering, Sorting, and Sparse Fieldsets Without Reinventing SQL"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How to let callers narrow, order, and shape a collection through the query string — powerfully but safely — from simple equality filters to the operator problem, whitelisted sorting, sparse fieldsets that cut a 40 KB body to 4 KB, and the security rules that keep a query string from sorting your database into the ground."
tags:
  [
    "api-design",
    "api",
    "rest",
    "http",
    "filtering",
    "sorting",
    "pagination",
    "query-parameters",
    "json-api",
    "security",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/filtering-sorting-and-sparse-fieldsets-without-reinventing-sql-1.png"
---

A merchant integration team opened a support ticket that read, in full: "Your orders endpoint is down." It was not down. What had happened was that one of their analysts, exporting a year of transactions into a spreadsheet, had discovered our collection endpoint accepted a `sort` parameter. They wanted the data ordered by the customer's email address. So they sent `?sort=customer.email`. That field was not in any index. The database planner, asked to sort forty million rows by a column it could only reach through a join and then materialize entirely in memory, did exactly what it was told. It took a connection, pinned it for ninety seconds, exhausted the sort memory budget, spilled to disk, and while it churned, every other request waiting on that connection pool timed out. One caller, one query string, one un-whitelisted sort field, and the whole tenant's traffic fell over.

This post is about the part of the contract that lets a caller say *which* records they want, in *what order*, with *which fields* — all through the query string — without ever handing them a loaded gun. It is a deceptively deep corner of API design because the query string sits exactly at the boundary between two worlds: an untrusted string typed by someone you will never meet, and a `WHERE` clause that runs against your production database. Get the translation between those two wrong, and you have built either a useless toy (every list returns all fields, unsorted, unfiltered, and the client downloads ten megabytes to render a table of five columns) or an open invitation to abuse (a Turing-complete filter language that anyone can use to read data they should not see or to sort your database into the ground, as our analyst nearly did).

![A diagram of the safe query string lifecycle, showing a caller supplied query string parsed into a map, checked against a field and operator whitelist, branching to either a 400 rejection or a parameterized query with bounded work, ending in a safe result page](/imgs/blogs/filtering-sorting-and-sparse-fieldsets-without-reinventing-sql-1.png)

By the end of this post you will be able to design a filtering, sorting, and field-selection layer that is powerful enough to keep callers off the phone and safe enough that no single request can hurt you. You will know the spectrum of filter conventions — from plain equality through bracketed operators to a full query language — and *where on that spectrum to stop*, which is the actual decision. You will know why a sort without a tiebreaker quietly corrupts pagination, why sparse fieldsets are both a developer-experience win and a performance win, how field expansion (Stripe's `expand`, JSON:API's `include`) saves round-trips while inviting the N+1 trap, and the security rules — whitelist everything, parameterize everything, bound the work — that turn the query string from a liability into a feature. We will do all of it on the running example for this series: a **Payments & Orders API** for a fictional commerce platform, where a merchant needs to filter, sort, and project their own orders. And the question we keep returning to, the spine of [this whole series](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems), applies here as sharply as anywhere: *what does the caller get to assume, and can I change this later without breaking them?*

## The job of the query string

Before we pick conventions, let us be precise about what we are designing. A REST collection resource — `GET /v1/orders` — represents *all* the orders the caller is allowed to see. That is rarely what the caller wants in one response. They want a *view* of the collection: the paid orders from last week, sorted newest first, showing only the fields their table renders. The query string is how the caller describes that view without you having to mint a new endpoint for every combination.

There are exactly three orthogonal things a caller can ask the query string to do, and it pays to keep them mentally separate because they have different rules, different risks, and different conventions:

- **Filtering** narrows *which rows* come back. `?status=paid` says "only paid orders." This maps to a SQL `WHERE` clause. It is the most powerful and therefore the most dangerous knob, because it decides what data leaves your system.
- **Sorting** decides *the order* of the rows. `?sort=-created_at` says "newest first." This maps to a SQL `ORDER BY`. Its hidden danger is interaction with pagination: an unstable sort silently corrupts paged reads.
- **Sparse fieldsets** (also called *projection* or *field selection*) decide *which columns* of each row come back. `?fields=id,amount,status` says "I only need these three." This maps to the `SELECT` column list. Its payoff is payload size and the database work of materializing columns you never send.

A fourth, related knob is **field expansion** or **embedding** — `?expand=customer` — which is the inverse of projection: instead of trimming the response, it *grows* it by inlining a related resource the caller would otherwise have to fetch separately. It deserves its own section because it carries the N+1 risk.

Notice that pagination — `?page`, `?cursor`, `?limit` — is a fifth knob, but it is its own large topic with its own correctness pitfalls, and this series gives it a dedicated post: [pagination, offset, cursor, and keyset tradeoffs at scale](/blog/software-development/api-design/pagination-offset-cursor-and-keyset-tradeoffs-at-scale). I will lean on it here because sorting and pagination are joined at the hip — you cannot page correctly over an unstably sorted collection — but I will not re-derive cursors.

The reason all of this lives in the *query string* rather than the path is a semantic one worth stating, because it is part of the contract. The path identifies the resource; the query string selects a representation or a subset *of* that resource. `GET /v1/orders` and `GET /v1/orders?status=paid` address the same collection resource; the second just asks for a filtered representation. This matters for caching (a query string is part of the cache key, so `?status=paid` and `?status=refunded` cache independently, which is correct) and it matters for idempotency and safety — a `GET` with a query string is still a *safe* method (it must never change server state) and an *idempotent* one (sending it twice has the same effect as once), no matter how baroque the filter. Those two guarantees are what let intermediaries cache it, let clients retry it freely on a timeout, and let a CDN serve it without asking your origin. The instant you make filtering have side effects — say, a `?mark_seen=true` that mutates as it reads — you have broken the safety guarantee and every one of those layers is now wrong about your endpoint. Keep reads pure.

If you ever find yourself wanting to `POST` a filter because the query string got too long, that is a real and recognized pattern, but treat it as a smell first and a solution second; we will return to it.

The whole point — and the title of this post — is that you should *not* reinvent SQL. The caller's query string is not a database console. It is a deliberately narrow, deliberately safe surface that exposes *some* of the power of `WHERE`/`ORDER BY`/`SELECT` while keeping you in control of which fields, which operators, which costs are reachable. The art is in choosing how much power to expose, and the engineering is in translating it to SQL safely. Let us start with the easy end and walk up the ladder.

## Filtering: the spectrum from equality to a query language

### Simple equality is where you start, and often where you should stop

The simplest filter is a query parameter whose name is a field and whose value is the exact value to match:

```http
GET /v1/orders?status=paid&currency=USD HTTP/1.1
Host: api.example-commerce.com
Authorization: Bearer <token>
Accept: application/json
```

This reads naturally, it is trivial to parse (the parameters arrive as a key/value map), and it maps to an unambiguous SQL fragment: `WHERE status = $1 AND currency = $2`. Multiple parameters combine with `AND`, which is the least surprising default — a caller who specifies both `status` and `currency` almost always means "both must hold."

A few small conventions inside the simple style pay for themselves:

- **Repeated keys or comma lists for `IN`.** Callers frequently want "paid *or* refunded." The two common spellings are repeated keys (`?status=paid&status=refunded`) and comma-separated values (`?status=paid,refunded`). Pick one and document it. Repeated keys are slightly more standard across HTTP libraries, but comma lists are tidier and what JSON:API uses. Either maps to `WHERE status IN ($1, $2)`. The important part is that you cap how many values an `IN` list may carry (more on bounding the work later) — an attacker who can pass ten thousand values in an `IN` clause has found a denial-of-service knob.
- **Boolean and null handling.** Decide explicitly how `?refunded=true`, `?refunded=false`, and "refunded is unset" are spelled. A common, clean choice: `?refunded=true|false` for the two boolean states and a dedicated sentinel like `?customer_id=null` (or simply *omit* the parameter to mean "don't filter on it"). Ambiguity here — does the *absence* of `?refunded` mean "false" or "any"? — is a classic source of caller confusion. Absence should always mean "do not filter on this field." That is the only rule a caller can guess correctly.

Simple equality covers a startling fraction of real use cases. Before you reach for anything more powerful, ask whether equality plus an `IN` list plus a handful of named convenience parameters would do. It usually would, and the cost of the simple style — to you and to the caller — is near zero.

There is one more subtlety in the simple style that bites teams later: **case and normalization.** Is `?status=PAID` the same filter as `?status=paid`? Is `?currency=usd` valid, or must it be `USD`? The contract must decide, and the decision should match how the underlying data is stored and how callers naturally type. For enumerated values like `status` and `currency`, the kindest contract is to normalize on input — uppercase the currency, lowercase the status — *and* document the canonical form, so a caller's reasonable guess works and the stored comparison stays exact. The trap is doing case-insensitive matching with a database `LOWER(column) = LOWER($1)`, which silently defeats your index (the index is on `column`, not `LOWER(column)`) and turns a fast equality lookup into a full scan. If you need case-insensitive matching, store a normalized column and filter on *that*. The general lesson, which recurs everywhere in this post: a convenience you add at the API layer can quietly cost you the index at the database layer, and you will not notice until the table is large.

### The operator problem: when equality is not enough

The wall you hit is *comparisons*. A merchant does not only want orders with `status=paid`; they want orders with `amount` greater than 100 dollars, or `created_at` after last Monday, or `amount` between two values. Equality cannot express any of that. You need operators — greater-than, less-than, in-range — and the moment you do, you face the central design decision of this entire post: **how do you spell an operator in a flat string of key/value pairs?**

There are three families of answers, and they form a spectrum of power, complexity, and abuse risk. The figure below lays them out as a comparison; the prose that follows walks each rung.

![A matrix comparing three filter styles — simple equality, right-hand-side bracket operators, and OData or RSQL query languages — across power, implementation complexity, and abuse risk, showing that power and abuse risk rise together as you climb the spectrum](/imgs/blogs/filtering-sorting-and-sparse-fieldsets-without-reinventing-sql-2.png)

**Family 1 — Named convenience parameters.** Bake the operator into the parameter name. Instead of an abstract "greater than," you expose a concrete, readable parameter:

```http
GET /v1/orders?created_after=2026-06-01&amount_min=100 HTTP/1.1
```

Here `created_after`, `amount_min`, and `amount_max` are *specific named filters* that you, the API designer, chose to support. Each one maps to a fixed SQL fragment you wrote by hand: `created_after` becomes `created_at >= $1`. This is the style Google's [API Improvement Proposals](https://google.aip.dev/) lean toward for many cases, and it is the style Stripe uses for date ranges (their list endpoints accept `created[gte]`-style parameters, but for many filters they expose flat named parameters). The advantage is enormous and underrated: **the set of supported filters is finite, named, documented, and individually secured.** There is no general operator engine to abuse. Each filter is a feature you deliberately shipped. The disadvantage is combinatorial: if every field needs `_min`, `_max`, `_after`, `_before`, `not_`, and so on, your parameter list explodes, and adding "greater-than-or-equal on the `tax` field" means a code change and a doc change.

**Family 2 — Right-hand-side operators (the bracket convention).** Generalize the operator into the syntax so you do not have to name every combination. The most common spelling puts the operator in brackets after the field name:

```http
GET /v1/orders?amount[gte]=100&created_at[gte]=2026-06-01&status[in]=paid,refunded HTTP/1.1
```

Now `amount[gte]=100` parses into the structured triple (field=`amount`, operator=`gte`, value=`100`), and you have *one* operator engine that handles `gte`, `lte`, `gt`, `lt`, `eq`, `ne`, `in`, and maybe `like`. This is the convention used by [JSON:API filtering extensions](https://jsonapi.org/recommendations/) (the spec leaves the filter syntax agent-specific, and the bracket form is the dominant community convention), by libraries like Sequelize and many ORMs, and by APIs like Strapi. Some APIs use a colon on the right-hand side instead of brackets (`?amount=gte:100`), which is equivalent — it is the same idea with different punctuation. The win over named parameters is that you write the operator engine *once* and it works on any whitelisted field. The cost is that you now own a small expression evaluator, and — critically — you must whitelist *which operators are allowed on which fields*, because not every operator is safe or sensible on every column. `like` on a non-indexed text column is a performance trap; `gte` on a free-text field is meaningless.

A practical parsing note, because the bracket form has a sharp edge: query-string libraries differ wildly in how they handle `amount[gte]=100`. Some (Node's `qs`, PHP, Rails) auto-parse bracket syntax into nested structures, so `amount[gte]` arrives as a nested object `{amount: {gte: "100"}}`; others (the WHATWG `URLSearchParams`, Go's `net/url`) treat the entire string `amount[gte]` as a single flat key and leave the bracket parsing to you. If you adopt the bracket convention, decide whether you are relying on framework auto-parsing or parsing the brackets yourself, and test it, because "it works in my framework's request object but the raw query string says something else" is a real source of subtle filter bugs across language ports of the same API. The colon variant (`amount=gte:100`) sidesteps this entirely — it is always a flat key with a value you split on the first colon — which is one honest reason to prefer it despite being slightly less common. Whichever you pick, the parsed result must be the same structured triple, and that triple is what your whitelist validates.

**Family 3 — A full query language in one parameter.** Push all the structure into a single `filter` (or `$filter`) parameter whose value is an expression in a real grammar:

```http
GET /v1/orders?filter=amount gt 100 and (status eq 'paid' or status eq 'refunded') HTTP/1.1
```

This is the [OData](https://www.odata.org/) approach (`$filter=amount gt 100 and status eq 'paid'`) and the [RSQL/FIQL](https://github.com/jirutka/rsql-parser) approach (`filter=amount=gt=100;status=in=(paid,refunded)`). You now support boolean composition — `and`, `or`, grouping with parentheses, negation — which is genuinely more powerful than anything the first two families can express. Microsoft's APIs (Graph, Dynamics) use OData heavily; it is a real, specified, mature standard with a full grammar. The power is undeniable. So is the cost, and it is the cost this post most wants you to feel: **you are now operating a query language.** You must parse the grammar (a real lexer and parser, or a library), you must walk the resulting abstract syntax tree and translate every node to SQL, you must whitelist fields *and* operators *and* the shape of expressions (because an attacker can write `(a or (a or (a or ...)))` nested a thousand deep, or an `OR` of ten thousand terms, and now you have an unbounded-work problem that no single parameter limit catches), and you must do all of that without ever letting a caller's string reach the database uninterpreted. OData and RSQL are not insecure — they are well-specified — but *your implementation* of them is exactly as secure as the weakest part of your AST-to-SQL translation, and that translation is a much larger attack surface than a fixed list of named parameters.

Here is the comparison in table form, because the decision deserves to be explicit:

| Filter style | Example | Power | Implementation cost | Abuse surface |
| --- | --- | --- | --- | --- |
| Named / equality | `?status=paid&created_after=…` | Low — each filter is a fixed feature | Trivial — a map plus a few hand-written fragments | Smallest — finite, individually secured set |
| RHS bracket ops | `?amount[gte]=100&status[in]=…` | Medium — operators on any whitelisted field | Moderate — one operator engine, per-field op whitelist | Medium — must bound operators, `IN` size, `like` |
| OData / RSQL | `?filter=amount gt 100 and …` | High — full boolean algebra, grouping | Heavy — lexer, parser, AST walker, validator | Largest — nested `OR`, expression depth, full grammar |

**The decision rule.** Start at the top of the table and only move down when a *concrete, recurring* caller need forces you. Most APIs should live in the first two rows. The bracket convention (Family 2) is the sweet spot for a maturing API: it is expressive enough for ranges and sets, it is a single engine, and — this is the key — it is *bounded by a whitelist* in a way the full query language is not. You decide which fields are filterable and which operators each field allows; everything else is rejected. A full query language is the right tool when you have many distinct sophisticated consumers who genuinely need ad-hoc boolean composition (analytics platforms, admin tools, large partner integrations), and even then the honest move is often to recognize that you have outgrown "filtering on a REST collection" and want a real query interface — which is the GraphQL temptation we will reach near the end.

The line I want you to internalize: **do not build a query language you cannot secure.** Every rung up the spectrum is a promise to the caller that you can change later only with great care, and a promise to your security team that you have bounded the work. If you cannot whitelist it, parameterize it, and cap its cost, you have not designed a filter — you have built a SQL console with extra steps.

## Translating filters to SQL safely

This is the section that, if you take nothing else, you should take. The query string is *untrusted input*. It does not matter whether you chose simple equality or a full grammar — the rules for getting from the parsed filter to the database are the same, and they are non-negotiable.

### Never concatenate; always parameterize and whitelist

There are two failure modes, and they are different. **Injection** is when caller input changes the *structure* of your SQL. **Unbounded work** is when caller input, even perfectly safe SQL, asks the database to do more than it should. Parameterization defeats the first; whitelisting and bounding defeat the second. You need all three.

Consider the worst version first — the one that looks like it works in development and ends a career in production:

```python
# DANGEROUS — never do this. Caller input becomes SQL structure.
def list_orders(params):
    sort = params.get("sort", "created_at")
    status = params.get("status")
    sql = f"SELECT * FROM orders WHERE status = '{status}' ORDER BY {sort}"
    return db.execute(sql)
```

Two separate disasters live here. First, `status = '{status}'` interpolates the caller's value into the SQL string, so `?status=' OR '1'='1` turns the `WHERE` into a tautology and returns every order on the platform, across every merchant. That is a [classic injection](https://owasp.org/www-community/attacks/SQL_Injection) and a data-breach-grade authorization bypass. Second — and this one is *not* fixed by parameterizing the value — `ORDER BY {sort}` interpolates the caller's value as a SQL *identifier*. Bind parameters cannot help you here: a placeholder like `$1` can stand in for a *value*, but not for a column name or a SQL keyword. So `?sort=created_at` works, and so does `?sort=(SELECT ...)` or `?sort=customer.email` (the very query that took down the tenant in the opening story). The sort column is structural, and the only safe way to handle structural input is to *map it through a whitelist*, never to interpolate it.

The figure contrasts the two worlds directly — the injectable string-concatenation path against the whitelisted, parameterized path.

![A before and after comparison showing an unsafe path where a raw sort string is interpolated into SQL with a DROP TABLE risk, versus a safe path where the field is checked against an allowlist and the value is supplied as a bound parameter with no injection path](/imgs/blogs/filtering-sorting-and-sparse-fieldsets-without-reinventing-sql-5.png)

Here is the same endpoint written safely. Notice the two distinct techniques: **values go through bind parameters; identifiers go through a whitelist map.**

```python
# SAFE — values are bound; identifiers are whitelisted, never interpolated.

# 1) The whitelist defines the entire filterable + sortable surface.
FILTERABLE = {
    "status":   {"column": "status",      "ops": {"eq", "in"}},
    "currency": {"column": "currency",    "ops": {"eq"}},
    "amount":   {"column": "amount_cents", "ops": {"eq", "gte", "lte"}},
    "created_at": {"column": "created_at", "ops": {"gte", "lte"}},
}
SORTABLE = {"created_at", "amount", "status"}  # API field -> implicitly its column

OP_SQL = {"eq": "=", "gte": ">=", "lte": "<=", "in": "IN"}

def build_query(merchant_id, filters, sort_fields, limit):
    # Tenant scoping is NOT optional and is NOT caller-controlled.
    where = ["merchant_id = %(merchant_id)s"]
    args = {"merchant_id": merchant_id}

    for i, (field, op, value) in enumerate(filters):
        spec = FILTERABLE.get(field)
        if spec is None or op not in spec["ops"]:
            raise BadRequest(f"unsupported filter: {field}[{op}]")
        col = spec["column"]            # whitelisted identifier, safe to inline
        key = f"v{i}"
        if op == "in":
            # IN expands to a bounded list of bind params, never a raw string
            values = value[:50]         # cap the IN list size
            placeholders = ", ".join(f"%({key}_{j})s" for j in range(len(values)))
            where.append(f"{col} IN ({placeholders})")
            args.update({f"{key}_{j}": v for j, v in enumerate(values)})
        else:
            where.append(f"{col} {OP_SQL[op]} %({key})s")
            args[key] = value           # value is a BIND param, never inlined

    order_by = build_order_by(sort_fields)  # whitelisted, see sorting section
    limit = min(limit, 100)                  # bound the page size

    sql = (
        "SELECT id, amount_cents, status, currency, created_at "
        "FROM orders "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY {order_by} "
        "LIMIT %(limit)s"
    )
    args["limit"] = limit
    return sql, args
```

Walk the safety properties: the *column names* (`amount_cents`, `created_at`) only ever come from the `FILTERABLE`/`SORTABLE` dictionaries — caller input selects *which* whitelisted entry to use, but never *becomes* an identifier. The *values* always travel as named bind parameters (`%(v0)s`), so the database driver sends them out-of-band and they can never alter the SQL structure. The `IN` list is capped at fifty. The page size is capped at one hundred. The `merchant_id` scope is injected by the server from the authenticated principal, *not* read from the query string, so a caller cannot filter their way into another merchant's orders. Every one of those is a deliberate guard, and the absence of any one of them is a real incident.

#### Worked example: a whitelisted filter compiles to one parameterized query

Let us trace a concrete request all the way to SQL so the mapping is unambiguous. A merchant's dashboard asks for paid or refunded orders of at least 100 dollars from June, ordered newest first:

```http
GET /v1/orders?status[in]=paid,refunded&amount[gte]=10000&created_at[gte]=2026-06-01&sort=-created_at&limit=25 HTTP/1.1
Host: api.example-commerce.com
Authorization: Bearer <token>
Accept: application/json
```

(The amount is in cents — `10000` is a \$100.00 minimum — because money is integer cents on the wire, a convention covered in [designing request and response bodies](/blog/software-development/api-design/designing-request-and-response-bodies-shape-and-naming).) The parser produces three filter triples — `("status","in",["paid","refunded"])`, `("amount","gte",10000)`, `("created_at","gte","2026-06-01")` — a sort list `[("created_at","desc")]`, and `limit=25`. Each filter field is checked against `FILTERABLE` and each operator against that field's allowed set. All pass. The builder emits exactly one statement, with the authenticated merchant's id bound server-side:

```sql
SELECT id, amount_cents, status, currency, created_at
FROM orders
WHERE merchant_id = $1
  AND status IN ($2, $3)
  AND amount_cents >= $4
  AND created_at >= $5
ORDER BY created_at DESC, id DESC
LIMIT $6;
-- $1=merch_42, $2='paid', $3='refunded', $4=10000, $5='2026-06-01', $6=25
```

Every caller-supplied value is a bind parameter. Every identifier (`status`, `amount_cents`, `created_at`) came from the whitelist. The `id DESC` tiebreaker was appended automatically — we will see why in the sorting section. And the database can serve this cheaply *if* there is a composite index on `(merchant_id, created_at)` so the `WHERE merchant_id = $1` plus `ORDER BY created_at DESC` is an index range scan rather than a full sort. That indexing question — covered for the engine side in [composite, covering, and index-only scans](/blog/software-development/database/composite-covering-and-index-only-scans) — is the bridge between a fast endpoint and the ninety-second timeout from the opening story.

If the caller had instead sent `?sort=customer.email`, the `build_order_by` whitelist would reject `customer.email` with a `400` *before any SQL ran*, and the tenant would have stayed up. The whitelist is not a nicety. It is the load-bearing wall.

### Reject unknown parameters loudly, or ignore them quietly — but pick a policy

There is a smaller design choice hiding in the safe builder: what do you do with a query parameter you do not recognize? A caller sends `?statu=paid` (a typo), or `?sort=createdAt` (camelCase when you expected snake_case), or `?foo=bar` (nonsense). Two coherent policies exist, and the only wrong answer is not choosing:

- **Reject loudly.** Return `400 Bad Request` with a `problem+json` body naming the offending parameter. This catches typos immediately — the caller's `?statu=paid` fails fast instead of silently returning *unfiltered* results that look subtly wrong (the dashboard shows all orders, the developer assumes the filter is "broken," and an afternoon evaporates). The cost is forward-compatibility friction: if a future client sends a parameter your *current* server does not yet understand, strict rejection breaks it. For first-party and partner APIs where correctness beats leniency, reject loudly. This is generally the better default for a payments API, where a silently-ignored filter could show a merchant the wrong numbers.
- **Ignore quietly.** Drop unknown parameters and proceed. This follows the [robustness principle](/blog/software-development/api-design/designing-for-the-caller-developer-experience-as-a-goal) ("be liberal in what you accept") and keeps clients working across versions. The cost is exactly the silent-failure trap above. Ignoring is more defensible for sparse-fieldset and expansion parameters (an unknown field in `?fields=` is plausibly a forward-compat client) than for filters (an ignored filter changes the *result set*, which is dangerous).

A pragmatic blend, which is what I usually ship: reject unknown *filter and sort* parameters loudly (they affect correctness), but for `fields`/`expand` lists, ignore unknown *members* of the list quietly while still rejecting the parameter itself if it is malformed. Whatever you choose, *document it*, because it is part of the contract — a caller needs to know whether their typo will scream or whisper.

There is a deeper compatibility reason this choice matters, and it connects to the [backward/forward compatibility](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems) thread that runs through the whole series. Suppose a year from now you *add* a new filterable field, `?payment_method=card`. Existing clients do not send it, so they are unaffected — additive, backward-compatible, no problem. But now suppose a *newer* client, talking to an *older* server instance during a rolling deploy, sends `?payment_method=card` and the old server has the strict "reject unknown loudly" policy. That request now `400`s on the old instances and succeeds on the new ones, giving the client a flapping error rate mid-deploy. The strict policy trades that forward-compat fragility for fast typo detection. If your deploys are atomic and your clients are first-party, that trade is fine. If you have many independently-deployed clients and a long rolling-deploy window, leaning toward "ignore unknown quietly" for *additive* parameters (while still rejecting truly malformed input) buys you smoother evolution. The point is not that one policy is right; it is that the policy is a real contract decision with real failure modes, and choosing it by accident is how you ship a surprise. The error body for the loud path uses the [RFC 9457 problem+json](https://www.rfc-editor.org/rfc/rfc9457) envelope this series standardizes on:

```json
{
  "type": "https://api.example-commerce.com/problems/unknown-filter",
  "title": "Unknown filter parameter",
  "status": 400,
  "detail": "The parameter 'statu' is not a filterable field on orders.",
  "instance": "/v1/orders",
  "supported_filters": ["status", "currency", "amount", "created_at"]
}
```

Notice the `supported_filters` array: an actionable error tells the caller not just that they were wrong but *what would have been right*. That is the kind of small kindness that keeps the integration team off the phone, and it is cheap to emit because you already have the whitelist in hand.

## Sorting: the `-` convention, whitelisting, and the tiebreaker that pagination needs

### The convention

The widely adopted convention — used by [JSON:API](https://jsonapi.org/format/#fetching-sorting), Stripe-adjacent APIs, and most public APIs — is a single `sort` parameter holding a comma-separated list of fields, where a leading `-` means descending and the absence of a prefix means ascending:

```http
GET /v1/orders?sort=-created_at,amount HTTP/1.1
```

That reads as "newest first, and within the same timestamp, smallest amount first." It maps directly to `ORDER BY created_at DESC, amount ASC`. The leading-minus convention is compact, it composes (multiple sort keys in priority order), and it is what callers expect, which is itself a reason to use it — least surprise is a feature. (You will occasionally see `?sort=created_at&order=desc` as an alternative; it is fine but does not compose to multiple keys cleanly, so prefer the prefixed-list form.)

### Whitelisting sortable fields is not optional

Everything from the SQL-safety section applies doubly to sorting, because — as we saw — a sort field is a structural SQL *identifier* that bind parameters cannot protect. The opening incident was a missing sort whitelist. So the rule is absolute: **maintain an explicit set of sortable fields, map each to a real column, and reject anything else.** Here is the `build_order_by` referenced earlier:

```python
SORTABLE_COLUMNS = {
    "created_at": "created_at",
    "amount":     "amount_cents",
    "status":     "status",
}

def build_order_by(sort_param):
    clauses = []
    for token in (sort_param or "").split(","):
        token = token.strip()
        if not token:
            continue
        direction = "DESC" if token.startswith("-") else "ASC"
        field = token.lstrip("-")
        column = SORTABLE_COLUMNS.get(field)
        if column is None:
            raise BadRequest(f"cannot sort by '{field}'")
        clauses.append(f"{column} {direction}")
    # Always append a unique tiebreaker for a total order (see below).
    clauses.append("id DESC")
    return ", ".join(clauses)
```

Two non-obvious points. First, the sortable set is usually *smaller* than the filterable set, and that is correct — a field can be cheap to filter (it is indexed for equality) but ruinous to sort (sorting requires either an index in that exact order or a full in-memory sort). You should only allow sorting on fields you have an index to support, or on columns small enough to sort in memory within your statement timeout. Sorting by `customer.email` was rejected on *both* counts: it is not a column on `orders` at all (it is across a join), and it has no supporting index. Second, the whitelist is the documentation: the set of sortable fields *is* the contract, and a caller can rely on it. Adding a sortable field later is a backward-compatible change (it is additive); *removing* one is breaking, so be a little conservative about what you promise.

### The tiebreaker: why a sort without a unique key corrupts pagination

This is the subtle, expensive bug, and it is the reason the sorting section links arms with the pagination post. SQL's `ORDER BY` does not guarantee a *total* order unless the sort keys are collectively unique. If you `ORDER BY created_at DESC` and two orders share the same `created_at` timestamp — which is common when a batch import or a burst of checkouts lands in the same millisecond — the database is free to return those two rows *in any order*, and crucially, it may return them in a *different* order on a different execution of the same query.

For a single un-paginated response that does not matter. For pagination it is a silent data-corruption bug. The figure below traces how it goes wrong.

![A timeline showing how an unstable sort breaks pagination, beginning with a non-unique sort on created_at, then ties on the same timestamp whose order is undefined, then a page boundary that splits a tie, then adding a unique tiebreaker, and finally a total order where every row appears exactly once](/imgs/blogs/filtering-sorting-and-sparse-fieldsets-without-reinventing-sql-8.png)

#### Worked example: a tie split across a page boundary drops a row

Suppose three orders — call them A, B, C — all have the exact same `created_at`, and they sit right at the boundary of a 25-row page. The caller requests page 1 (`ORDER BY created_at DESC LIMIT 25 OFFSET 0`). The database, with no further ordering rule, happens to return A and B at the bottom of page 1 and pushes C to page 2. The caller then requests page 2 (`OFFSET 25`). But this is a *separate query execution* — and with the tie unbroken, the planner this time returns the trio in the order B, C, A. Now C, A sit at the top of page 2... and B, which the caller already saw at the bottom of page 1, appears *again* at the top of page 2 — while one of the rows that should have been on page 2 gets pushed off and is *never seen at all*. The caller's export is missing a row and has a duplicate, and nobody notices until reconciliation fails a month later.

The fix is one line and it is in the `build_order_by` above: **always append a unique column as the final sort key.** `ORDER BY created_at DESC, id DESC`. Now ties on `created_at` are broken deterministically by the primary key, the order is *total* and *stable* across executions, and every row appears on exactly one page. This is not optional polish; it is correctness. The cursor/keyset pagination strategies in the [pagination post](/blog/software-development/api-design/pagination-offset-cursor-and-keyset-tradeoffs-at-scale) depend on it absolutely — a keyset cursor *is* an encoding of "the last row's sort key," and that only works if the sort key is unique, which is exactly why the tiebreaker exists. Sorting and pagination are one design, not two.

The principle, stated rigorously: a paginated read over a collection is correct *if and only if* the `ORDER BY` defines a total order on the rows — that is, for any two distinct rows, the order key comparison is never a tie. Since `created_at`, `amount`, `status` and most business fields are not unique, you guarantee totality by appending a column that *is* unique (the primary key). The cost is essentially zero: the database already has the primary key in the index it is scanning, so the tiebreaker rarely changes the plan. The benefit is that pagination stops silently lying.

## Sparse fieldsets: shaping the payload to what the caller actually renders

### The convention and why it is both DX and performance

By default a collection endpoint returns the *full* representation of each resource. For an `Order` with sixty fields, nested line items, embedded customer details, and a metadata bag, that is a large object — and a *list* of them is a large response. But a merchant's dashboard table often renders four columns: order id, amount, status, date. They are downloading 40 KB per order to display 4 KB of it, paying the cost in bytes on the wire, in JSON they must parse, and in database work to materialize columns they discard.

A **sparse fieldset** (the term comes from [JSON:API](https://jsonapi.org/format/#fetching-sparse-fieldsets)) lets the caller name the fields they want, and the server returns only those:

```http
GET /v1/orders?fields=id,amount,status,created_at HTTP/1.1
```

JSON:API's full spelling is type-scoped — `?fields[orders]=id,amount,status` — so you can independently trim each resource type in a compound document; for a single-type list, the flat `?fields=` is the common simplification. The response carries only the named fields:

```json
{
  "data": [
    { "id": "ord_8f2", "amount": 12999, "status": "paid", "created_at": "2026-06-14T09:11:02Z" },
    { "id": "ord_9a1", "amount":  4500, "status": "paid", "created_at": "2026-06-14T09:12:48Z" }
  ],
  "page": { "next_cursor": "b3JkXzlhMQ" }
}
```

This is a developer-experience win: the caller controls their payload, so the mobile team that pages a long list and the analytics team that needs every field both hit the *same* endpoint, each getting exactly the shape they want. No proliferation of `/orders/summary` and `/orders/full` variants. And it is a performance win on two axes. The obvious one is **wire bytes**: a smaller body transfers faster, which matters enormously on mobile and high-latency links. The less obvious one is **database and serialization work**: if your projection actually pushes down to the `SELECT` column list, the database materializes fewer columns and — if those columns happen to all live in an index — can even serve the query as an index-only scan, never touching the table heap.

The figure makes the contrast concrete.

![A before and after comparison of an order response, where the full body carries sixty fields with nested line items and a full customer object at roughly forty kilobytes, versus a sparse fieldset returning only id, amount, and status with no nested data at roughly four kilobytes, a tenfold reduction](/imgs/blogs/filtering-sorting-and-sparse-fieldsets-without-reinventing-sql-4.png)

#### Worked example: a sparse fieldset cuts a 40 KB body to 4 KB

Let us put approximate numbers on it, with the setup stated honestly because the kit forbids fabricated benchmarks. Take an `Order` whose full JSON representation is around 4 KB per record — id, amounts, status, currency, timestamps, a nested `customer` object (name, email, address, ~1 KB), an array of ten `line_items` (~200 bytes each, ~2 KB), and a `metadata` map. A page of ten such full orders is therefore on the order of 40 KB before gzip. The dashboard table needs only `id`, `amount`, `status`, `created_at` — about 80 bytes of JSON per record once you strip the nesting. Ten of those is roughly 800 bytes of `data`, plus the envelope, comfortably under 4 KB. That is the ~10× reduction the figure claims: **40 KB → ~4 KB**, achieved with one query parameter and no new endpoint.

What does 36 KB of saved payload buy in latency? It depends entirely on the link, which is why I will only give a range. On a fast wired connection the transfer difference is small (a few milliseconds), but the *parse* cost on a constrained mobile device and the memory pressure of holding the larger objects are real. On a cold, congested mobile link — the realistic case for a merchant checking their phone — shaving a response from ~40 KB to ~4 KB can plausibly save anywhere from tens to a few hundred milliseconds of transfer alone, before counting the client's JSON parse and layout. The honest framing: the savings scale with payload size and inversely with link quality, and for list endpoints over many records, the bytes add up fast. The point is not a precise number; it is that projection is one of the cheapest performance levers you have, and it costs the caller a single parameter.

### The push-down rule: a sparse fieldset must trim the SELECT, not just the JSON

There is a trap worth naming. Many implementations "support" sparse fieldsets by fetching the *full* row from the database and then deleting fields from the serialized object before sending it. That captures the wire-bytes win but throws away the database win — you still paid to read and materialize every column, including the expensive nested joins for `customer` and `line_items`. The valuable version pushes the field selection down into the query: the requested fields map (through your whitelist, of course) to a `SELECT` column list, and related-object fields that were *not* requested cause you to *skip the joins* entirely.

So `?fields=id,amount,status` should produce `SELECT id, amount_cents, status FROM orders WHERE ...` with no join to `customers` and no second query for `line_items`. The same whitelist discipline applies: a `FIELDS` allowlist maps API field names to columns, unknown fields are rejected or ignored per your policy, and the primary key (`id`) is *always* included even if the caller did not ask for it, because clients need a stable identifier to key their UI and to fetch the full resource later. That last rule — always return the id — is a small contract guarantee that prevents a whole class of "I asked for `amount,status` and now I can't tell the rows apart" bugs.

## Field expansion: embedding related resources, and the N+1 trap

Sparse fieldsets *shrink* the response. Expansion does the opposite: it *grows* it by inlining a related resource that would otherwise be a separate fetch. An `Order` references a `Customer` by id; sometimes the caller wants the customer's details right there, and forcing a second round-trip per order to `GET /v1/customers/{id}` is wasteful.

### Two conventions: Stripe `expand` and JSON:API `include`

**Stripe's `expand`.** In [Stripe's API](https://stripe.com/docs/api/expanding_objects), most objects carry references as ids by default, and you opt into inlining the full sub-object with `expand[]`:

```http
GET /v1/orders?expand[]=customer&expand[]=line_items.product HTTP/1.1
```

The response replaces the `customer` id string with the full nested customer object, and `line_items.product` (dot-path) expands the product inside each line item. It is intuitive — `expand` reads as "give me the thing this points to" — and it supports nested expansion through dot paths. Stripe caps expansion depth (you cannot expand arbitrarily deep) precisely to bound the work.

**JSON:API's `include`.** [JSON:API](https://jsonapi.org/format/#fetching-includes) takes a different shape. The primary resource still references the related one by id (in a `relationships` block), but the included resources are *side-loaded* into a top-level `included` array rather than nested inline:

```http
GET /v1/orders?include=customer,line_items HTTP/1.1
```

```json
{
  "data": [
    { "type": "orders", "id": "ord_8f2",
      "attributes": { "amount": 12999, "status": "paid" },
      "relationships": { "customer": { "data": { "type": "customers", "id": "cus_55" } } } }
  ],
  "included": [
    { "type": "customers", "id": "cus_55", "attributes": { "name": "Acme Co" } }
  ]
}
```

The big structural difference is **deduplication**. If a page of fifty orders all belong to the same five customers, Stripe's inline `expand` repeats each customer object inside every order (fifty copies of five customers), while JSON:API's side-loaded `included` array carries each unique customer *once* and the orders reference them by id. For high-fan-in relationships, side-loading is meaningfully smaller on the wire. The cost is that the client must do a join in memory (walk `relationships`, look up the id in `included`), which is more work to consume than a simple nested object.

Here is the trade-off across the three real options — Stripe-style inline expand, JSON:API side-loaded include, and just leaving the reference as an id for the client to follow separately:

| Dimension | Stripe `expand` (inline) | JSON:API `include` (side-load) | Separate request |
| --- | --- | --- | --- |
| Round-trips | One — related data inlined | One — related data side-loaded | Two or more — client follows the link |
| Payload size | Largest — duplicates on fan-in | Smaller — each related object once | Smallest — split across responses |
| Client parse cost | Lowest — nested, ready to use | Higher — must join `included` by id | Low per response, but more responses |
| Cacheability | Weak — bound to the parent query | Weak — bound to the parent query | Strong — each resource has its own URL and ETag |
| Best when | A few related objects, low fan-in | High fan-in, want dedup, JSON:API shop | Related data is large, optional, or independently cached |

![A matrix comparing Stripe inline expand, JSON:API side-loaded include, and a separate request across round-trips, payload size, and caching, showing that inline expand saves round-trips but bloats the payload while a separate request keeps each resource independently cacheable](/imgs/blogs/filtering-sorting-and-sparse-fieldsets-without-reinventing-sql-6.png)

The "separate request" column is the one people forget. If the related resource is large, optional, or independently cacheable, *not* embedding it is often the right call — the client fetches `GET /v1/customers/{id}` only when it needs to, and that response carries its own `ETag` so a conditional request can return `304 Not Modified` and skip the body entirely. Embedding trades that clean per-resource cacheability for fewer round-trips. There is no universal winner; there is a force (fan-in, payload size, cache reuse) that decides.

### The N+1 trap, and the over-fetch risk

Whichever convention you pick, expansion carries a server-side performance hazard that is easy to ship by accident: the **N+1 query problem.** A naive implementation fetches the page of N orders with one query, then loops over them issuing one customer query *per order* — N additional queries — turning a single logical request into N+1 database round-trips. With a page of 100 orders, that is 101 queries, and the endpoint's latency balloons under load even though each individual query is fast.

The fix is **batching**: collect all the referenced customer ids from the page first, then fetch them in *one* query with `WHERE id IN (...)`, and stitch the results back in memory. This is exactly the pattern a GraphQL *dataloader* automates (a topic the [GraphQL post](/blog/software-development/api-design/graphql-the-query-language-schema-and-the-n-plus-one-trap) covers in depth), but you must do it by hand in a REST expansion implementation:

```python
def expand_customers(orders):
    # ONE query for all referenced customers — not one per order.
    ids = {o["customer_id"] for o in orders}
    if not ids:
        return orders
    rows = db.execute(
        "SELECT id, name, email FROM customers WHERE id = ANY(%(ids)s)",
        {"ids": list(ids)[:200]},   # bound the batch too
    )
    by_id = {r["id"]: r for r in rows}
    for o in orders:
        o["customer"] = by_id.get(o["customer_id"])
    return orders
```

The over-fetch risk is the mirror image: if you let callers expand *anything* to *any depth*, a single request can pull a huge graph — `expand=customer.orders.line_items.product.reviews` — and you are back to operating an unbounded query engine through the back door. The same disciplines apply: **whitelist which relationships are expandable, cap the expansion depth (Stripe caps it; so should you), and bound the batch size.** Expansion is a feature you ship deliberately, relationship by relationship, not a general graph-traversal capability you hand to the public.

When a caller exceeds the depth or asks for a relationship you do not expose, reject it with the same actionable `problem+json` you use for filters, so the failure is legible rather than a silent truncation:

```json
{
  "type": "https://api.example-commerce.com/problems/expansion-too-deep",
  "title": "Expansion exceeds maximum depth",
  "status": 400,
  "detail": "The expand path 'customer.orders.line_items' has depth 3, but the maximum is 2.",
  "instance": "/v1/orders",
  "max_expansion_depth": 2,
  "expandable": ["customer", "line_items", "line_items.product"]
}
```

Notice that the depth limit is itself part of the contract — `max_expansion_depth` and the `expandable` list tell the caller exactly what they may ask for. This is the same pattern as `supported_filters` earlier: the whitelist *is* the documentation, and surfacing it in the error turns a dead end into a hint. The reason the cap is non-negotiable is that expansion depth compounds multiplicatively with fan-out — expanding `customer` then `customer.orders` then each order's `line_items` is a tree whose size is the product of the fan-outs at each level, and a depth-3 expansion over a customer with a hundred orders and ten line items each is ten thousand rows assembled to answer one request. Depth is the dimension that turns a convenience into a cost explosion, which is exactly why every mature API that offers expansion bounds it.

## The GraphQL temptation: when REST query params have outgrown their lane

Stand back and look at where the spectrum has taken us. We started with `?status=paid`. To support ranges we added bracketed operators. To support boolean composition we eyed a full filter grammar. To save round-trips we added expansion with depth and relationship whitelists. To shape payloads we added sparse fieldsets with per-type field selection. Each addition was reasonable in isolation. Stacked together, you are now maintaining: a filter expression evaluator, an operator-per-field whitelist, a sort whitelist with tiebreakers, a projection whitelist, a relationship-expansion whitelist with depth caps, and N+1 batching for every expandable edge. You have, slowly and without quite deciding to, **reinvented a query language inside your query string** — the exact thing this post's title warns against.

That accumulation is a *signal*, not a failure. When a meaningful set of your callers genuinely need ad-hoc filtering, field selection, *and* relationship traversal — when they want to say "give me these orders, with these fields, plus their customers' names and these line items' product titles, all in one request" — you have arrived at the problem [GraphQL](/blog/software-development/api-design/graphql-the-query-language-schema-and-the-n-plus-one-trap) was designed for. GraphQL makes field selection and relationship traversal *the entire premise* of the query language, with a typed schema and resolvers, and it solves the N+1 trap with dataloaders as a first-class pattern rather than something you hand-roll per endpoint.

But — and the series' "by force, not fashion" rule is doing real work here — GraphQL is not free, and reaching for it does not make the hard problems disappear; it *relocates* them. You still must whitelist and cost-bound, except now an attacker can write an arbitrarily nested query, so you need query-depth limits, complexity scoring, and persisted queries instead of `IN`-list caps and sort whitelists. HTTP caching, which `GET` query strings get almost for free (URL is the cache key, `ETag`/`304` work), becomes hard because most GraphQL traffic is a `POST` to a single endpoint. The decision belongs to the dedicated paradigm posts — this series' [GraphQL post](/blog/software-development/api-design/graphql-the-query-language-schema-and-the-n-plus-one-trap) and the broader [REST vs gRPC vs GraphQL overview](/blog/software-development/system-design/api-design-rest-grpc-graphql) in the system-design series. The point to take *here* is narrow and useful: **if your filtering and projection needs keep growing past what a whitelisted query string can comfortably and securely express, treat that as a paradigm signal.** Do not keep bolting query-language features onto a REST collection until you have built a worse GraphQL by accident. Either deliberately stop at a bounded REST surface, or deliberately adopt the paradigm built for arbitrary queries — but choose, rather than drift.

The filter-language spectrum, with that GraphQL ceiling at the top, looks like this:

![A tree of the filtering spectrum branching from fixed parameters with low power, through structured operators with medium power, to a full query language with high power, where the query-language branch splits into RSQL or OData filter clauses and GraphQL schema queries](/imgs/blogs/filtering-sorting-and-sparse-fieldsets-without-reinventing-sql-3.png)

### The `POST`-a-filter escape hatch, and why it is a last resort

There is one more pattern that comes up when the query string genuinely runs out of room: sending the filter in a request *body* via `POST /v1/orders/search` (sometimes called a "search" or "query" sub-resource). Two real forces push people here. The first is the URL length limit — many servers and proxies cap a URL at around 8 KB, so a filter with hundreds of ids in an `IN` list can simply not fit in a query string. The second is the awkwardness of URL-encoding a complex expression. A `POST` body has neither limit and can carry a structured JSON filter that is easier to validate than an encoded string.

But this is a genuine trade, not a free upgrade, and the cost is the safety we just spent a paragraph defending. A `POST` is *not* safe and *not* idempotent by HTTP's definition, so the moment you move filtering to `POST` you lose free HTTP caching (the body is not part of the cache key, so intermediaries and CDNs cannot cache the result), you lose the ability for a client to retry freely on a timeout without reasoning about side effects, and you lose the "this is obviously a read" legibility that a `GET` carries. You can claw some of it back — declare the search endpoint side-effect-free in your docs, return `Cache-Control` hints, even support a `POST` with a cache key derived from the body — but you are now swimming against HTTP's current. So: reach for `POST /search` only when a real query genuinely exceeds the URL limit or demands a structured body, keep the simple cases on `GET` with query params, and never move to `POST` merely because the filter *feels* complex. The query string being a little ugly is a smaller cost than an unreachable cache layer.

## Bounding the work: the rules that keep one query from hurting you

Injection is the headline risk, but it is not the most common production incident. The more common one is the *expensive-but-valid* query — the one that is perfectly safe SQL and yet does far too much work, like the opening story's sort over forty million rows. Whitelisting fields and operators is necessary but not sufficient; you must also bound the *cost* of any query a caller can construct. The figure summarizes the layers of guard; the prose gives the reasons.

![A stack of query cost guardrails layered from capping the page size at one hundred, requiring an indexed filter such as merchant id, whitelisting the sort field with a tiebreak on id, forbidding unbounded OR clauses, enforcing a statement timeout, down to a cheap bounded page with a stable p99](/imgs/blogs/filtering-sorting-and-sparse-fieldsets-without-reinventing-sql-7.png)

- **Cap the result size, always.** Every list endpoint has a maximum page size that the server enforces regardless of what the caller asks for — `limit = min(requested, 100)` in the builder above. A caller who sends `?limit=1000000` gets 100. This single rule prevents the most direct denial-of-service: "ask for everything at once." Pagination is mandatory, not optional; an endpoint that returns an unbounded collection is a latent outage.
- **Require an indexed filter on large collections.** For a table that grows without bound — orders, events, log entries — refuse to serve a query that has no selective, indexed predicate. The merchant *must* be scoped to their own `merchant_id` (which you inject server-side and which is indexed), so they can never request a scan of the global table. If a caller could send a filter that forces a full table scan plus a sort, you have handed them the ninety-second-timeout knob. Tie the allowed sort fields to the available indexes — this is where the API contract and the [database index design](/blog/software-development/database/composite-covering-and-index-only-scans) meet: a `(merchant_id, created_at)` composite index is what makes `?sort=-created_at` for one merchant an index range scan instead of a sort of the whole table.
- **Bound the boolean structure, not just the values.** If you do expose composition (`OR`, `IN` lists, nested expressions), cap the number of `OR` branches, the `IN` list length, and the expression nesting depth. An unbounded `OR` of ten thousand terms is a valid query that plans terribly. This is the guard that the full-query-language families most often forget, because the parameter itself is "just one string."
- **Set a statement timeout.** Defense in depth: even with all the above, set a per-statement timeout on the database connection (Postgres `statement_timeout`, a query deadline in your driver) so that any query exceeding a budget — say two seconds — is killed rather than allowed to pin a connection. This converts a slow-query incident from "the pool is exhausted and everything times out" into "that one request gets a `503` and everyone else is fine." Pair it with a `429` and `Retry-After` when a caller is hammering you, the way [rate limiting](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems) governs request frequency, but the statement timeout governs the *cost* of a single accepted request, which is the dimension filtering and sorting expose.
- **Validate types and ranges before binding.** A filter value of `?amount[gte]=abc` should be rejected at the validation layer with a `400`, not passed to the database to fail with a type error. Range-check where it is cheap (`limit` must be 1–100; a date must parse). This keeps your error responses honest (`400` for caller mistakes, never a leaked `500` from a database type error) and keeps malformed input from reaching the engine at all.

The unifying principle: **a caller may construct any query you allow, so the set of queries you allow must all be cheap.** You do not get to assume callers will be reasonable; you must make unreasonable requests *impossible to express* (whitelist) or *cheap to refuse* (caps, timeouts). That is the difference between a query string that is a feature and one that is an incident waiting for the wrong analyst.

## A design problem, reasoned through and stress-tested

Let us pull the pieces together on a single concrete requirement and reason from the requirement to a design, then attack the design — because that is the actual work, and a list of rules is only useful if you can apply it under pressure.

The requirement: a merchant's analytics dashboard must list their orders, let them filter by `status`, `currency`, an `amount` range, and a `created_at` range, sort by date or amount, and show a compact table (id, amount, status, date) that loads fast on mobile, with a "view details" action that expands one order into the full record including the customer. The orders table has tens of millions of rows across all merchants.

Reasoning to a design, in the order the priorities fall out. First, **scope and indexing**, because the table is large: the query *must* be bounded by `merchant_id`, injected server-side from the token, and there *must* be a composite index leading with `merchant_id` so any per-merchant query is a range scan. That single decision is what separates a fast endpoint from the opening incident. Second, **filtering style**: the requirement names ranges (`amount`, `created_at`), so plain equality is insufficient and a full query language is overkill — the bracketed-operator family (Family 2) is the right rung, with a per-field operator whitelist (`amount` and `created_at` allow `gte`/`lte`; `status` and `currency` allow `eq`/`in`). Third, **sorting**: whitelist `created_at` and `amount`, append the `id` tiebreaker, and ensure each sort field is covered by an index in the scanned order. Fourth, **projection**: the table view requests `?fields=id,amount,status,created_at`, which pushes down to a `SELECT` of exactly those columns and skips every join — and because those four columns can be covered by the `(merchant_id, created_at)` index plus `INCLUDE` columns, the list query can be an index-only scan that never touches the table heap. Fifth, **expansion**: the "view details" action is a *separate request* to `GET /v1/orders/{id}?expand=customer` rather than expansion on the list, because details are needed for one row at a time, the full record is large, and the single-resource fetch carries its own `ETag` for caching. That is the whole design, and notice that every decision traced back to a force — table size, the named ranges, the mobile constraint, the one-at-a-time detail view — not a fashion.

Now stress-test it, which is where designs earn their keep:

- **What if a caller sends `?limit=1000000`?** The server clamps to 100. The page is bounded; the worst case is one full page, served from the index range. No incident.
- **What if a caller sorts by a field that is not indexed in that order, say `?sort=amount` when the index leads with `created_at`?** This is the real risk. `amount` is whitelisted (it is a sortable field), but for one merchant the planner may now sort the merchant's matching rows in memory. The mitigation is that the *filter* already bounds the candidate set to one merchant (thousands of rows, not forty million), so an in-memory sort of a few thousand rows is cheap and within the statement timeout. Sorting is safe here precisely *because* the indexed filter bounds it first. Had we allowed an un-scoped sort, this is exactly where it would have fallen over.
- **What if two writers race — a new order lands while the caller is paging?** A new order at the top of the (newest-first) sort shifts the offsets if you paginate by offset, which is the row-drift bug the [pagination post](/blog/software-development/api-design/pagination-offset-cursor-and-keyset-tradeoffs-at-scale) covers; the fix is keyset/cursor pagination, which our unique tiebreaker (`created_at, id`) makes possible. The sort design and the pagination design are the same design.
- **What if the collection has fifty million rows and a *new* merchant queries with no orders yet?** The index range scan on `merchant_id` returns zero rows instantly; an empty result is a valid `200` with an empty `data` array, not a `404` (the collection exists; it is just empty). Cheap and correct.
- **What if the payload is 10× bigger than planned — the merchant adds a huge `metadata` blob to every order?** The full record balloons, but the *list* view is unaffected because the sparse fieldset never selected `metadata`. Projection insulated the hot path from a field-shape change on the cold path. That is the quiet, compounding value of letting callers shape the payload: a change to the full resource does not silently degrade every list.
- **What if a malicious caller probes for injection with `?sort=created_at;DROP TABLE orders;--`?** The sort token is matched against the `SORTABLE_COLUMNS` whitelist, finds no key `created_at;DROP TABLE orders;--`, and returns a `400` before any SQL is built. The structural input never reaches the database. The whitelist, not a sanitizer, is the defense — sanitizing identifiers is a losing game; mapping them through a fixed set is a winning one.

The design survives every one of these because each guard was chosen for a specific failure, not bolted on after an outage. That is the whole method: list the failures first, design the guard that makes each one impossible to express or cheap to refuse, and only *then* expose the feature.

## Case studies: how real APIs draw these lines

It is worth grounding all of this in how well-regarded APIs actually decided, because the conventions above are not abstract — they are the distilled practice of these systems. I will state only what is documented and publicly known, and keep claims general where I am not certain of a current detail.

**Stripe — `expand`, flat filters, and explicit limits.** [Stripe's API](https://stripe.com/docs/api) is the reference many teams copy, and its choices map onto this post cleanly. References are ids by default, with opt-in inline expansion via `expand[]`, depth-capped. List endpoints use cursor pagination (`starting_after`/`ending_before`) with a `limit` that is bounded (the documented maximum is 100). Filtering leans on flat, named parameters and date-range operators rather than a general filter grammar — a deliberate choice to keep the surface finite and individually secured. The lesson: a hugely successful payments API got very far *without* a query language, by choosing named filters, bounded expansion, and capped pages.

**JSON:API — the standardized vocabulary.** [JSON:API](https://jsonapi.org/) is where the `sort`, `fields`, and `include` conventions in this post are *specified* rather than ad-hoc. It defines the `-` prefix for descending sort, type-scoped sparse fieldsets (`fields[type]=...`), and side-loaded `include` with a deduplicated top-level `included` array. Notably, JSON:API deliberately *leaves the filter syntax agent-specific* — it standardizes the `filter` query parameter *family* but not its grammar — which is itself a design statement: even the spec authors declined to bless one filter language, because the right one depends on your data and your security posture. If you want a standard to point your clients at for sorting and field selection, this is it.

**OData — the full query language, and its cost.** [OData](https://www.odata.org/) is the most complete realization of "a query language in the URL": `$filter`, `$select` (projection), `$orderby`, `$expand`, `$top`/`$skip`, with a full boolean grammar and functions. Microsoft Graph and Dynamics use it at scale, which proves it can work — but it works because Microsoft invested in mature parsers, validators, and cost controls. OData is the honest illustration of the top of our spectrum: maximal power, maximal implementation and security investment. Reach for it when you genuinely need that power *and* will fund that investment; do not adopt a sliver of OData syntax casually, because a half-implemented query language is the worst of both worlds.

**Google AIP — filtering as a typed, documented expression.** [Google's API Improvement Proposals](https://google.aip.dev/) (notably AIP-160 on filtering) define a structured filter string with its own small expression syntax (`amount > 100 AND status = "PAID"`) used across Google Cloud APIs. The instructive part is the discipline around it: filterable fields are explicitly documented per resource, the syntax is specified rather than improvised, and unsupported expressions are rejected with clear errors. It sits between the bracket convention and full OData — a real expression language, but a constrained and well-governed one. The takeaway across all four: the successful APIs are the ones that *chose a point on the spectrum deliberately and documented it*, not the ones that grew a filter language by accretion.

## When to reach for each (and when not to)

Every choice here is a trade-off; the engineering is in saying plainly when *not* to do the powerful thing.

- **Reach for simple equality / named filters** as your default and stay there as long as it covers your callers. *Don't* build the operator engine speculatively — `?status=paid&created_after=...` covers a remarkable share of real needs, and every named filter is a feature you can secure and reason about individually. The named-parameter style's verbosity is a feature: it forces you to decide, one filter at a time, what you are willing to support.
- **Reach for bracketed operators** when ranges and sets become a recurring, real demand across multiple callers — it is the sweet spot, one engine bounded by a per-field operator whitelist. *Don't* expose every operator on every field reflexively; `like` on an unindexed text column, or an `IN` with no size cap, are the operators that turn an engine into an outage. Allow operators field by field.
- **Reach for a full query language (OData/RSQL)** only when you have many sophisticated consumers genuinely needing ad-hoc boolean composition *and* you will fund the parser, validator, and cost controls. *Don't* adopt a fragment of one casually — a half-built query language is unsecured and unbounded by definition. And don't build one at all if what you actually want is GraphQL; recognize the signal and choose the paradigm deliberately.
- **Always whitelist sort fields and always append a unique tiebreaker.** *Don't* allow sorting on un-indexed or cross-join fields (the opening incident), and *don't* ever ship a paginated sort without a tiebreaker (the silent row-drop). These two are not optional; they are correctness and availability.
- **Reach for sparse fieldsets** whenever a list view renders fewer fields than the full resource carries — it is nearly free DX and performance, and it pushes down to the `SELECT`. *Don't* "support" it by fetching everything and deleting fields in the serializer; that throws away the database half of the win. And always return the `id` regardless.
- **Reach for inline `expand`** for low-fan-in related objects a client almost always needs together; reach for side-loaded `include` when fan-in is high and dedup matters; leave it a **separate request** when the related resource is large, optional, or independently cacheable (so it keeps its own `ETag` and `304`). *Don't* allow unbounded expansion depth, and *don't* ship expansion without batching — the N+1 is the most common self-inflicted latency wound here.
- **Above all, bound the work.** *Don't* serve an unbounded page, *don't* allow a query with no indexed predicate on a large table, *don't* skip the statement timeout. The most dangerous query is not the malicious one; it is the perfectly valid one that does too much.

The meta-rule, the spine of [this series](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2): *what does the caller get to assume, and can I change this later?* A whitelisted, named filter surface is one you can grow additively (adding a filterable field is backward-compatible) and reason about. A query language is a promise you cannot easily walk back — every expression a caller writes today is a contract you must keep working tomorrow. Design the part you can secure, document the part you support, and reject the rest loudly.

## Key takeaways

- The query string does three orthogonal jobs — **filter** (`WHERE`), **sort** (`ORDER BY`), **project** (`SELECT`) — plus expansion. Keep them mentally separate; they have different risks and conventions.
- Filtering is a **spectrum**: simple equality → named operators → bracketed RHS operators → a full query language (OData/RSQL). Power, complexity, and abuse risk rise together. Start simple; climb only when a concrete recurring need forces you.
- **Never concatenate caller input into SQL.** Values go through **bind parameters**; identifiers (columns, sort fields) go through a **whitelist map** — bind parameters cannot protect an identifier. Both techniques are required, not either/or.
- **Whitelist sortable fields**, and **always append a unique tiebreaker** (`ORDER BY created_at DESC, id DESC`). A non-total sort silently drops and duplicates rows across page boundaries — sorting and pagination are one design.
- **Sparse fieldsets** (`?fields=...`) cut payloads dramatically (a ~40 KB order page to ~4 KB) and are both a DX and a performance win — but only if the projection **pushes down to the SELECT** and skips unrequested joins. Always return the `id`.
- **Expansion** (Stripe `expand` inline, JSON:API `include` side-loaded, or a separate request) trades round-trips against payload size and cacheability. Whitelist expandable relationships, cap depth, and **batch to avoid N+1**.
- **Bound the work**: cap page size, require an indexed predicate on large tables, cap `OR`/`IN`/nesting, set a statement timeout. The valid-but-expensive query is the most common incident.
- If filtering/projection/traversal needs keep outgrowing a whitelisted query string, that is a **paradigm signal** — consider GraphQL deliberately, but know it relocates the cost (query-depth limits, complexity scoring, weaker HTTP caching) rather than removing it.
- **Pick an unknown-parameter policy** — reject loudly or ignore quietly — and document it. For correctness-affecting filters and sorts, reject loudly with an actionable `problem+json` error.

## Further reading

- [JSON:API specification](https://jsonapi.org/format/) — the standardized conventions for `sort` (the `-` prefix), sparse fieldsets (`fields[type]`), and `include`/side-loading.
- [Stripe API reference — expanding objects](https://stripe.com/docs/api/expanding_objects) and [pagination](https://stripe.com/docs/api/pagination) — a widely copied, deliberately bounded design.
- [Google AIP-160: Filtering](https://google.aip.dev/160) — a constrained, well-governed filter expression syntax used across Google Cloud.
- [OData URL conventions](https://docs.oasis-open.org/odata/odata/v4.01/) — the full query-language end of the spectrum (`$filter`, `$select`, `$orderby`, `$expand`).
- [RFC 9457: Problem Details for HTTP APIs](https://www.rfc-editor.org/rfc/rfc9457) — the error envelope for rejecting unknown or malformed filters honestly.
- [OWASP — SQL Injection](https://owasp.org/www-community/attacks/SQL_Injection) — why parameterization and whitelisting are non-negotiable.
- Within this series: [What is an API: the contract between systems](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems), [Pagination: offset, cursor, and keyset tradeoffs at scale](/blog/software-development/api-design/pagination-offset-cursor-and-keyset-tradeoffs-at-scale), [Designing request and response bodies](/blog/software-development/api-design/designing-request-and-response-bodies-shape-and-naming), [GraphQL: the query language, schema, and the N+1 trap](/blog/software-development/api-design/graphql-the-query-language-schema-and-the-n-plus-one-trap), and the [API design playbook capstone](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).
- Out to the database engine: [Composite, covering, and index-only scans](/blog/software-development/database/composite-covering-and-index-only-scans) — why a `(merchant_id, created_at)` index turns a sort into a range scan.
