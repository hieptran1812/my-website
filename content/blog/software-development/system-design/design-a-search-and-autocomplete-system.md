---
title: "Design a Search and Autocomplete System: Inverted Indexes, Ranking, and Typeahead"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "How a senior designs full-text search and sub-50ms typeahead: inverted indexes, scatter-gather across shards, BM25 to learning-to-rank, the trie that precomputes completions, and the trade-offs that decide latency, freshness, relevance, and cost."
tags:
  [
    "system-design",
    "search",
    "autocomplete",
    "inverted-index",
    "ranking",
    "elasticsearch",
    "architecture",
    "distributed-systems",
    "scalability",
    "latency",
    "information-retrieval",
  ]
category: "software-development"
subcategory: "System Design"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/design-a-search-and-autocomplete-system-1.webp"
---

Here is the moment that separates a junior answer from a senior one in a search-design interview. The candidate says "we use an inverted index" and stops, as if naming the data structure were the design. The senior keeps going: which terms become postings, how a multi-word query becomes a set intersection, how that index gets sharded across fifty machines, why the query's p99 is set by the single slowest shard and not the average, how a new document becomes searchable in one second instead of one hour, why the autocomplete box that fires on every keystroke is a completely different system from the search that fires on Enter, and where each of those decisions costs you latency, freshness, relevance, or money. Search is the canonical staff-level system because it forces you to hold all four of those constraints in tension at once, and there is no design that wins on all four.

This article is about that tension. We are going to build two systems that look like one product. The first is **search**: the user types a full query, hits Enter, and expects the ten most relevant documents out of fifty million in under a hundred milliseconds. The second is **autocomplete**, or typeahead: the user is mid-word, the box fires a request on every keystroke, and expects the right completions in under fifty milliseconds — at roughly ten times the query volume. They share a corpus and almost nothing else. Search is *shard-and-scatter-gather with a tight p99*; typeahead is *precompute-and-cache*. Getting that distinction right is half the design, and it is the half most candidates miss.

We will start where the corpus does — inverting documents into an index — then layer on sharding and the tail-latency problem that fan-out creates, then ranking (from BM25 to learning-to-rank and the two-phase retrieve-then-rank pattern that makes relevance affordable), then typeahead as its own precomputed beast, then the indexing pipeline that keeps the whole thing fresh. Throughout, we estimate real numbers, stress-test the design against a celebrity query and an index rebuild, and end with the explicit trade-off matrix a senior would put on the whiteboard. The figure below is the single idea the whole system rests on: the inverted index.

![Diagram contrasting a forward document-to-terms layout against an inverted term-to-postings index that turns a full scan into a fast list intersection](/imgs/blogs/design-a-search-and-autocomplete-system-1.webp)

## 1. Frame and scope: two systems wearing one UI

Before any estimation, a senior pins down scope, because "build search" is unbounded. We will commit to a concrete product: search over a corpus of **50 million documents** — think product listings, support articles, or messages — averaging about 1 KB of indexable text each. We support full-text keyword search with relevance ranking, and we support typeahead completions on a query box. We will treat three things as in-scope and worth designing carefully: the inverted index and how a query executes against it; ranking quality; and the typeahead path. We will treat a few things as explicitly out of scope to keep the discussion honest: we are not building a web crawler, we are not solving multilingual analysis beyond a sketch, and we are not building the personalization/ML platform that produces ranking features — we consume those features, we do not train the models.

Stating non-goals out loud is itself a senior move; it signals you know where the swamp is. If you want the full treatment of how to scope an ambiguous prompt and convert "build search" into testable requirements and SLOs, that is its own discipline, covered in [turning vague asks into requirements and SLOs](/blog/software-development/system-design/turning-vague-asks-into-requirements-and-slos) and the broader habit of [how seniors approach ambiguous system design problems](/blog/software-development/system-design/how-seniors-approach-ambiguous-system-design-problems). Here we will assume the scoping is done and spend our budget on the engine.

The functional requirements are short: given a query string, return the top-k most relevant documents with pagination; given a query *prefix*, return the top-k most likely completions. The non-functional requirements are where the system lives or dies, so let us state them as numbers, because vague SLOs are the root of most bad search systems:

- **Search latency**: p50 under 40ms, p99 under 150ms, measured at the search service, excluding network to the client.
- **Typeahead latency**: p99 under 50ms, end to end, because it fires per keystroke and anything slower feels laggy.
- **Freshness**: a newly created or updated document should be searchable within a few seconds, not minutes. (We will see this is a real cost.)
- **Availability**: 99.95% for search; typeahead can degrade to a cached/stale result without anyone noticing, so it can be best-effort.

Notice the asymmetry already baked in: typeahead must be *faster* than search while serving *more* traffic, but it is allowed to be *staler* and *less complete*. That asymmetry is the lever the whole optimization story pulls on.

## 2. Back-of-the-envelope estimation

A senior never moves to architecture without sizing the problem first, because the numbers decide the design. If typeahead QPS is comparable to search QPS, you can share infrastructure; if it is ten times higher, you cannot, and the whole shape changes. Let us do the math the way you would on a whiteboard. (If estimation itself feels shaky, the dedicated guide on [back-of-the-envelope estimation for system design](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design) walks the technique; here we just apply it.)

Start with users and traffic. Say we have 10 million daily active users, and an average user runs 5 searches per active day. That is 50 million searches per day. Spread over a day that is roughly 580 searches per second on average, but search traffic is peaky — peak is commonly 3 to 5 times the average — so call peak search load **~3,000 QPS**, and round up to **5,000 QPS** as a planning headroom number. That is the number we size shards against.

Now typeahead. Every search starts as typing, and the box fires on each keystroke (after light debouncing). A query averages maybe 20 characters, but we do not fire on every single character — we debounce to roughly one request per 2–3 keystrokes, so call it **8 typeahead requests per search**. That immediately multiplies: 50 million searches × 8 = 400 million typeahead requests per day, averaging ~4,600 per second, with the same peak multiplier giving roughly **40,000 typeahead QPS** at peak. Typeahead is an order of magnitude hotter than search. That single number — 40k vs 5k — is why typeahead gets its own precomputed, heavily cached path and cannot just ride on the search index.

Now storage. Fifty million documents at ~1 KB of indexable text is about 50 GB of raw text. An inverted index is not free: postings lists, term dictionaries, positions for phrase queries, and stored fields typically push the index to roughly 1.5–3× the raw text size depending on what you store and how aggressively you compress. Call it **~150 GB** for the inverted index proper. Multiply by a replication factor (we will want at least 2 replicas of each shard for availability and read throughput), and the hot, in-memory-or-SSD footprint you are paying for is on the order of **450 GB**. That fits comfortably across a modest cluster — say 10 to 20 machines with plenty of RAM and NVMe — which tells you this is not a "thousands of machines" problem at this scale. It is a "get the fan-out and the p99 right" problem.

![Capacity matrix showing documents, search QPS, typeahead QPS, index size, and storage cost for both a per-unit and an at-scale column](/imgs/blogs/design-a-search-and-autocomplete-system-8.webp)

Two numbers from this estimate drive every later decision. First, **typeahead QPS (40k) dwarfs search QPS (5k)**, so typeahead optimization is where the request budget is. Second, **the inverted index (150 GB) dominates storage**, so compressing postings and caching hot terms is where the memory budget is. Hold those two facts; they recur.

#### Worked example: sizing shards from the QPS and latency budget

Let us turn the estimate into a shard count, because "we shard the index" is meaningless without a number. Suppose a single shard holds about 5 million documents (≈15 GB of index) and a single CPU core can scan and score that shard for an average query in about 8ms. With 50 million documents we have **10 primary shards**. At 5,000 search QPS, each query fans out to all 10 shards, so the cluster does 50,000 shard-queries per second. If one core handles a shard-query in 8ms, one core sustains ~125 shard-queries/sec, so we need about 50,000 / 125 = **400 cores** of search capacity at peak. At, say, 16 usable cores per machine that is roughly **25 search nodes** at peak, before replicas. Add 2 replicas per shard for availability and read throughput and you are sizing on the order of 30–50 machines. Now the kicker: because every query touches all 10 shards, the query's latency is the **max** of 10 shard latencies, not the average. If each shard's p99 is 60ms, the query's p99 is *worse* than 60ms — that is the tail-latency tax of fan-out, and it is the single most important number in this design. We will spend a whole section on it.

## 3. The public API

Keep the API boringly small; search APIs rot when they sprawl. Two endpoints carry the product.

```http
GET /v1/search?q=wireless+headphones&page=0&size=10&filter=category:audio
GET /v1/autocomplete?q=wirel&limit=8
```

Search returns ranked documents with enough metadata to render a result and to paginate. Critically, pagination is **cursor-based**, not deep offset-based, because `from=10000` forces every shard to retrieve and sort 10,000 results just to discard them — the classic deep-pagination melt. A search-after cursor carries the sort values of the last result so the next page resumes in place.

```json
{
  "took_ms": 31,
  "total_estimated": 1840,
  "results": [
    { "id": "p_8841", "title": "Wireless Headphones X2", "score": 14.7, "snippet": "...active noise..." }
  ],
  "cursor": "eyJzY29yZSI6MTQuNywiaWQiOiJwXzg4NDEifQ=="
}
```

Typeahead returns just enough to render a dropdown: the completion strings and maybe a popularity hint. It does **not** return full documents — that is the search path's job. The typeahead response is small by design so it can be cached aggressively and served from memory.

```json
{ "completions": ["wireless headphones", "wireless earbuds", "wireless charger"], "took_ms": 6 }
```

Two API decisions are senior tells. First, `total_estimated` is *estimated*, not exact — counting exact total hits across shards is expensive and almost never worth it; users do not care whether there are 1,840 or 1,851 results. Second, the search endpoint accepts structured `filter` parameters that map to index-level filters (a `category:audio` filter is a posting-list intersection, cheap), keeping the free-text `q` for relevance and the filters for hard constraints. Mixing those concerns into one query string is how you end up unable to cache or optimize either.

## 4. High-level architecture

The shape is a classic scatter-gather behind a coordinator. A request hits the search service, which acts as a **broker** (Elasticsearch calls this the coordinating node): it parses and analyzes the query, fans it out to every shard that could hold a match, collects each shard's partial top-k, merges them into a global top-k, and returns. Typeahead bypasses most of this with its own service backed by a precomputed trie and a cache. The indexing pipeline runs alongside, continuously turning new and changed documents into searchable segments.

![Scatter-gather query flow where a broker fans one query to all index shards in parallel and merges their partial top-k results back into a final ranked list](/imgs/blogs/design-a-search-and-autocomplete-system-2.webp)

Three properties of this architecture matter for the rest of the design. **First, the broker is stateless** — it holds no index, only routing and merge logic — so you scale brokers independently and cheaply, and a broker failure just drops in-flight requests that clients retry. **Second, the shards are the stateful, expensive tier**; they hold the inverted index in memory and on NVMe, and they are what you replicate and rebalance. **Third, the fan-out is total**: a plain keyword query has no way to know which shards hold matches without asking them, so it asks all of them. (You can sometimes prune shards with routing keys — if you route a tenant's documents to a known shard, queries scoped to that tenant touch one shard — but for general full-text search, fan-out is to all shards. This is exactly why the tail-latency problem is structural, not incidental.)

The data path for an indexing write is separate and asynchronous: writes land in a durable log (we will lean on change-data-capture and the outbox pattern so the index never disagrees with the source of truth), get analyzed, and get appended to fresh segments. Search reads the segments; indexing writes them; merges reconcile them in the background. This separation of the write path from the read path is the same instinct behind log-structured storage, and it is no accident that Lucene segments behave a lot like SSTables. We will tie that knot explicitly when we get to the indexing pipeline.

## 5. Data model: what actually lives in a shard

Before diving into the index internals, pin down what a shard physically stores, because the data model decides what queries are cheap and what queries are impossible. A shard is not a single structure; it is a small bundle of cooperating structures over its slice of documents.

The first is the **term dictionary** — the sorted set of all distinct terms in the shard, stored as the FST mentioned above, mapping each term to the location of its posting list. The second is the **posting lists** themselves, one per term, each a compressed sorted list of document IDs with per-document term frequencies and optionally positions. The third is the **stored fields / document values**: the original field values you need to *return* (the title, the snippet source, the URL) and the **doc-values** columnar store you need to *sort, filter, and aggregate on* (price, recency, category) — kept column-oriented precisely because sorting and filtering scan one field across many documents, which is a columnar access pattern, not a row one. The fourth is the **deletion bitset**: a per-segment bitmap marking documents that have been deleted or superseded, because segments are immutable, so a delete is recorded as a tombstone bit rather than an in-place removal, and the document is only physically purged at the next merge.

That last detail has a real consequence a senior anticipates: **updates and deletes accumulate dead weight until a merge runs.** Updating a document means writing a new version to a fresh segment and marking the old version deleted in its segment's bitset — so a heavily updated corpus carries a tax of deleted-but-not-yet-purged documents that bloat the index and slow queries (you still walk past the tombstoned entries during a scan). High update rates therefore demand more aggressive merging, which costs more background I/O. This is the same write-amplification trade-off that defines log-structured storage, and recognizing it as such tells you exactly which knobs to reach for.

The model split between *stored fields* (for returning) and *doc-values* (for sorting/filtering) is the search-engine version of a familiar database lesson: separate the data you read by key from the data you scan by column. A query that filters on `category` and sorts by `price` touches the columnar doc-values; a query that returns the title touches the stored fields; and the relevance scoring touches the posting lists. Keeping them in distinct structures means each access pattern hits a layout built for it, rather than one bloated row format that is mediocre at all three.

## 6. Deep dive: the inverted index

Everything starts here. A **forward index** maps each document to the terms it contains — natural to build, useless for search, because answering "which documents contain *headphones*?" means scanning every document. The **inverted index** flips it: it maps each *term* to the sorted list of document IDs that contain it (the **posting list**), plus per-document information like term frequency and positions. Now "which documents contain *headphones*?" is a single dictionary lookup that hands you the answer as a ready-made list.

A multi-word query becomes a set operation over posting lists. The query `wireless headphones` as an AND becomes the **intersection** of the posting list for *wireless* and the posting list for *headphones* — only documents in both lists match. As an OR it is the **union**. Because posting lists are stored sorted by document ID, intersection is a merge-style walk down both lists in linear time, and with skip pointers you can leap over large gaps when one list is much shorter than the other — you advance the rare term and skip-search the common one. This is why query latency depends far more on the *selectivity* of your rarest term than on corpus size: intersecting a 50-document list with a 5-million-document list is fast because you drive the walk from the short list.

Getting from raw text to terms is the job of the **analyzer**, and it is where a lot of relevance quietly lives or dies. The analyzer pipeline tokenizes the text (split on whitespace and punctuation, mostly), lowercases, removes or keeps stop words, and applies **stemming** or lemmatization so that *running*, *runs*, and *ran* collapse toward a common root and match each other. The exact same analyzer must run at index time and at query time, or the query term will not match the indexed term — a query for *Running* analyzed to *run* must meet an indexed *run*, not an indexed *Running*. Analyzer mismatch between index and query is one of the most common "search returns nothing and I cannot see why" bugs in production, and it is invisible until you dump the actual indexed terms.

```python
# Conceptual analyzer + inverted index build (single-shard, in-memory).
import re
from collections import defaultdict

STOP = {"the", "is", "a", "of", "to"}

def analyze(text):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    out = []
    for t in tokens:
        if t in STOP:
            continue
        # toy stemmer: strip a trailing 's' (real systems use Porter/Snowball)
        if t.endswith("s") and len(t) > 3:
            t = t[:-1]
        out.append(t)
    return out

def build_index(docs):
    index = defaultdict(list)          # term -> [(doc_id, term_freq), ...] sorted by doc_id
    doc_len = {}
    for doc_id, text in docs:
        terms = analyze(text)
        doc_len[doc_id] = len(terms)
        tf = defaultdict(int)
        for term in terms:
            tf[term] += 1
        for term, freq in tf.items():
            index[term].append((doc_id, freq))
    return index, doc_len

def intersect(index, a, b):
    # AND of two terms: merge-walk two sorted posting lists.
    la = {d for d, _ in index.get(a, [])}
    lb = {d for d, _ in index.get(b, [])}
    return sorted(la & lb)
```

That toy index holds the whole skeleton: an analyzer, posting lists keyed by term, and a set intersection for AND queries. A production index adds three things that matter at scale. First, **compression**: posting lists are stored as *delta-encoded* gaps between consecutive document IDs (since they are sorted, gaps are small) and packed with variable-length or frame-of-reference codecs, which can shrink postings several-fold and, more importantly, let you scan them faster because you move fewer bytes through the CPU cache. Second, **positions and offsets**, so phrase queries (`"wireless headphones"` as an exact adjacent phrase) and highlighting work — these roughly double the index size, which is the cost line in our estimate. Third, **the term dictionary itself** is stored as a finite state transducer (FST) — a compact, prefix-shared automaton — so the dictionary of millions of terms fits in memory and supports fast lookup and prefix iteration. That FST detail comes back when we build typeahead, because the same structure that makes the term dictionary small also makes prefix completion fast.

It is worth slowing down on compression, because it is where a lot of search performance hides and where a junior assumes the bytes are free. Consider the posting list for a common term like *wireless* in a 50-million-document corpus: it might appear in 2 million documents. Stored naively as 2 million 32-bit document IDs, that single posting list is 8 MB — and there are hundreds of thousands of terms. Delta-encoding exploits the fact that the IDs are sorted: instead of storing `[1004, 1009, 1021, 1050]` you store the first ID and then the gaps `[1004, +5, +12, +29]`, and gaps are small numbers that pack into far fewer bits. Frame-of-reference and the PForDelta family go further, encoding blocks of gaps against a common reference with a fixed bit width and handling the rare large gap as an exception. The payoff is double: the index is smaller (more of it fits in RAM, fewer pages fault from disk) and, counterintuitively, *scanning is faster* even though you must decode, because memory bandwidth and cache misses dominate modern query cost far more than the cheap integer arithmetic of decoding. A senior knows that at this scale you are almost always memory-bandwidth-bound, not CPU-bound, so anything that moves fewer bytes is a latency win, not just a storage win.

The second performance structure worth naming is the **skip list** embedded in the posting list. Recall that AND queries intersect two sorted lists, and the cheap way to do that is to drive the walk from the *shorter* list and advance the longer list to catch up. Without skips, advancing the long list means stepping one document at a time; with skip pointers — periodic forward references that say "the document at offset +128 is ID 5,000,000" — you can leap over huge swaths of the common term's list in one hop, landing near the document you are looking for and then stepping the last few. This turns an intersection whose cost was proportional to the *longer* list into one closer to proportional to the *shorter* list, which is why a query like `wireless aardvark` (one common term, one rare) is fast: you walk the tiny *aardvark* list of, say, 300 documents, and skip-search the 2-million-document *wireless* list at each of those 300 points. The architect's takeaway is that **query latency is governed by the rarest term's selectivity, and the index is engineered to let the rare term drive the walk.** This is also why adding a single very common term to a query (a stop word that slipped through, or a term in nearly every document) can blow up latency: it offers no selectivity, so it cannot drive the walk and cannot be cheaply skipped. Removing or down-weighting near-ubiquitous terms is not just a relevance nicety; it is a latency defense.

The single most important mental model for an architect here is that **search has been transformed from a compute problem into a set-algebra problem**. You are not scanning text at query time; you are intersecting and unioning precomputed sorted lists. Everything fast about search downstream — sharding, ranking, caching — is built on top of that one transformation.

## 7. Deep dive: sharding, replication, and the tail-latency tax

One machine cannot hold a 150 GB index in RAM and serve 5,000 QPS, so we shard. **Sharding** splits the document space across N shards — typically by hashing the document ID so documents distribute evenly — and each shard holds a complete, independent inverted index over its slice of documents. A query must touch *every* shard, because any shard could hold a matching document; the broker fans out, each shard computes its local top-k, and the broker merges the partial results into the global top-k. **Replication** then copies each shard R times for availability (lose a machine, a replica serves) and read throughput (route reads across replicas). The mechanics of splitting a keyspace and moving it without downtime are their own deep topic, covered in [partitioning and sharding without downtime](/blog/software-development/system-design/partitioning-and-sharding-without-downtime); here we care about what sharding *does to latency*.

It taxes the tail, and the math is unforgiving. When a query fans out to N shards in parallel and you must wait for all of them, the query's latency is the **maximum** of the N shard latencies. Even if each shard is fast at the median, the maximum of many samples reaches into each shard's tail. Concretely: suppose each shard responds in under 10ms 99% of the time (its p99 is 10ms) but occasionally hiccups. With one shard, 1% of queries are slow. With 10 shards queried in parallel, the probability that *all ten* are fast is 0.99^10 ≈ 0.904 — so roughly **10% of queries hit at least one slow shard**. Fan-out to 100 shards and 0.99^100 ≈ 0.366: now **two-thirds of queries** are dragged into the tail. The more you shard for throughput, the worse your query p99 gets from stragglers. This is the central, counterintuitive cost of scatter-gather, and it is why the article on [articulating trade-offs: CAP, PACELC, and beyond](/blog/software-development/system-design/articulating-tradeoffs-cap-pacelc-and-beyond) treats tail latency under fan-out as a first-class architectural force, not an afterthought.

A senior names the mitigations explicitly, because "the p99 is bad" is a complaint, not a design:

- **Hedged requests (request to a second replica)**: if a shard has not responded within, say, the 95th-percentile time, send the same shard-query to a *different replica* and take whichever returns first. A straggler is usually a transient local event — a GC pause, a hot CPU, a slow disk read — so a second replica is very likely fast. Google's Jeff Dean popularized this "tail at scale" technique; it can cut p99 dramatically at the cost of a few percent extra load. The discipline is to hedge *only* the slow tail, not every request, or you double your traffic.
- **Fewer, fatter shards**: every shard you add multiplies the straggler probability. If your corpus fits, prefer 10 shards of 5M docs over 100 shards of 500k docs. You shard for capacity, not for sport.
- **Adaptive replica selection**: route each shard-query to the replica that has been *fastest lately*, not round-robin. A node that is GC-pausing or has a degraded disk gets less traffic automatically.
- **Bounded-wait / partial results**: for some products you can return after the fastest M of N shards respond, accepting a slightly incomplete result rather than waiting for a straggler. This trades completeness for latency and is only acceptable when missing a few results is survivable.

#### Worked example: the fan-out tail-latency tax, with numbers

Say each shard, healthy, returns in 20ms at p50 and 60ms at p99. With 10 shards queried in parallel and a naive "wait for all", the query p99 is roughly the max over 10 draws, which lands near each shard's p99 *plus* the compounding — call it ~80–90ms in practice. Now you turn on hedging: when a shard has not answered in 45ms (its ~p90), you fire the same shard-query at a second replica. The probability that *both* the primary and the hedge are slow is tiny (the slow events are uncorrelated transient pauses), so the effective per-shard p99 collapses toward ~45ms, and the query p99 drops to roughly 50–60ms — comfortably inside our 150ms SLO with margin to spare. The cost: you sent ~10% extra shard-queries (only the slow tail got hedged), so you provision ~10% more shard capacity. That is the trade in one sentence — **you buy a 30ms+ p99 improvement with 10% more compute**, and at search scale that is almost always worth it. The number to put on the whiteboard is the hedge threshold; set it at the shard's p90–p95, never lower, or you flood the cluster.

## 8. Deep dive: ranking, from BM25 to learning-to-rank

Matching is necessary but not sufficient; a query for *wireless headphones* might match 1,840 documents, and the user looks at the top ten. Ranking decides which ten. A senior treats ranking as a **two-phase** problem, because doing it in one phase is either too slow or too dumb.

**Phase one is retrieval**, and its job is to cheaply cut 50 million documents down to a few hundred candidates using a fast, decent scoring function. The workhorse here is **BM25**, the modern successor to TF-IDF. The intuition is simple and worth internalizing: a document is more relevant when (a) it contains the query terms frequently (**term frequency**, with diminishing returns — the 50th occurrence of *headphones* barely adds over the 10th), (b) those terms are *rare* across the corpus (**inverse document frequency** — matching *headphones* is more meaningful than matching *the*), and (c) the document is not artificially long (**length normalization** — a 10,000-word page that mentions *headphones* once is less about headphones than a tight product title). BM25 packages those three intuitions into a formula with two tunable knobs (k1 for term-frequency saturation, b for length normalization) and is cheap enough to score every matching document during the posting-list walk. It is the default scorer in Lucene/Elasticsearch for exactly this reason: good enough, fast enough, no training required.

**Phase two is ranking**, and its job is to take the few hundred candidates from phase one and reorder them with a much more expensive, much smarter model — **learning-to-rank (L2R)**. Now you can afford features you could never compute over 50 million documents: historical click-through rate for this query-document pair, recency, the document's quality score, personalization signals, semantic similarity from an embedding model, business rules. A gradient-boosted tree (LambdaMART is the classic) or a neural ranker scores the ~500 candidates and produces the final order. Because it only runs on the survivors of phase one, you pay its cost a few hundred times per query, not fifty million times. This **retrieve-then-rank** split is the single most important pattern in modern search and recommendation: cheap recall first, expensive precision second.

![Two-phase retrieve-then-rank pipeline where a cheap BM25 pass narrows millions of documents to a few hundred candidates before an expensive learning-to-rank model rescores only the survivors](/imgs/blogs/design-a-search-and-autocomplete-system-5.webp)

The architectural tension in ranking is **relevance versus latency**, and the two-phase split is how you resolve it. More phase-one candidates means better recall (the right answer is more likely to survive into phase two) but more work for phase two; fewer candidates is faster but risks discarding the true best result before the smart model ever sees it. The candidate count (often a few hundred to a thousand) is a tuning knob you set by measuring relevance metrics — NDCG, MRR — against latency, not by guessing. A senior states the knob and how they would measure it; a junior hardcodes 100 and moves on.

```python
# Phase 1: BM25 scoring during the posting-list walk (sketch).
import math

def bm25(index, doc_len, avg_dl, N, query_terms, k1=1.2, b=0.75):
    scores = {}
    for term in query_terms:
        postings = index.get(term, [])
        df = len(postings)                       # docs containing this term
        if df == 0:
            continue
        idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
        for doc_id, tf in postings:
            dl = doc_len[doc_id]
            norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg_dl))
            scores[doc_id] = scores.get(doc_id, 0.0) + idf * norm
    # return top candidates for phase 2 (here, top 500)
    return sorted(scores.items(), key=lambda kv: -kv[1])[:500]
```

There is a third actor in most real search queries that interview answers omit: **filters**, and they interact with scoring in a way a senior gets right. A query is rarely just free text; it is "free text *plus* in-stock *plus* price under \$100 *plus* category audio." The filters are hard constraints — a document either passes or it does not, with no contribution to the relevance score — and they are implemented as posting-list intersections against the doc-values, exactly like a term match but without scoring. The crucial optimization is that **filter results are cacheable in a way scored queries are not.** The set of documents matching `category:audio` does not depend on the free-text query and changes only when documents change, so a search engine maintains a **filter cache** (a bitset per cached filter) that is reused across thousands of different free-text queries. The discipline this imposes on API design is to keep filters *structurally distinct* from the scored query — which is exactly why the API in section 3 separated `filter=` from `q=`. Mix a constantly-varying value into a filter (a timestamp, a per-user token) and you destroy the cache hit rate, because every query now has a unique filter that is computed once and never reused. A senior reviews search APIs specifically for this: filters should be low-cardinality and stable, so the bitset cache earns its keep, while high-cardinality or per-request values stay in the scored or post-filter path.

This is the search-specific face of a general caching truth — that cacheability is a property you design *into* the query shape, not a thing you bolt on afterward. The broader treatment of filter caches, result caches, and the cardinality traps that quietly kill hit rates lives in [caching strategies and the pitfalls that bite](/blog/software-development/system-design/caching-strategies-and-the-pitfalls-that-bite); the search lesson is that the line you draw between *filter* and *score* in your API is, downstream, the line between cacheable and uncacheable work.

The vector frontier deserves a sentence, because it is reshaping retrieval. Increasingly, phase one is *hybrid*: a lexical BM25 retrieval runs alongside a **vector / approximate-nearest-neighbor (ANN)** retrieval that finds semantically similar documents even when no keyword overlaps (a query for *noise cancelling earbuds* surfacing a doc titled *active noise reduction headphones*). The two candidate sets are fused before phase two. ANN buys semantic recall but pays in index cost (storing and searching high-dimensional vectors is expensive) and freshness (re-embedding changed documents is heavier than re-indexing terms), which is exactly the trade we will put in the matrix. For most products, lexical BM25 plus learning-to-rank is the high-leverage core, and vectors are the next increment once that is solid.

## 9. Deep dive: autocomplete is a different system

Now the part candidates conflate with search and shouldn't. Typeahead has a different shape because it has different constraints: it fires on every keystroke (~40k QPS, 8× search), it must answer in under 50ms, it operates on *prefixes* not complete queries, and it returns a handful of *query strings* not ranked documents. You do not run the search pipeline for typeahead — scatter-gather plus retrieve-then-rank would never hit 50ms at 40k QPS, and it would be wasteful, because the set of useful completions is small and changes slowly. The winning move is **precompute and cache**: compute the top-k completions for prefixes ahead of time, store them in a structure built for prefix lookup, and serve from memory.

The data structure is a **trie** (prefix tree), or in production its compressed cousin the **FST** (finite state transducer). Each node represents a prefix; walking down the tree spells out longer prefixes; and — the key optimization — each node stores its **precomputed top-k completions**, ranked by popularity. When the user has typed `sea`, you walk to the `sea` node (a few pointer hops, O(prefix length), independent of corpus size) and read off its cached top-k. There is no search, no scoring at query time, no fan-out — just a short walk and a memory read. That is how you hit sub-50ms at high QPS: you did the expensive work offline.

![Typeahead trie where each prefix node caches its precomputed top-k completions so a keystroke becomes a short pointer walk rather than a search](/imgs/blogs/design-a-search-and-autocomplete-system-3.webp)

The reason you *can* precompute is that the answer is a function of the prefix and the popularity distribution, both of which change slowly. The top completions for `wirel` are *wireless*, *wireless headphones*, *wireless charger* — and that list is stable over hours, not milliseconds. So you build the trie from your query logs (the best source of completions is *what people actually search for*, weighted by frequency), attach top-k to each node, and rebuild periodically. Storing top-k *at the node* rather than computing it by descending the whole subtree at query time is the crucial space-time trade: it costs more memory (you denormalize the top-k onto every prefix node) but turns query time from "traverse the subtree and sort" into "read a precomputed list".

![Before-after comparing query-time search that recomputes retrieval and ranking on every request against typeahead that precomputes top-k per prefix once and serves it from a trie and cache](/imgs/blogs/design-a-search-and-autocomplete-system-6.webp)

Several production details make typeahead robust. **Prefix caching** in front of the trie service catches the hottest prefixes (the single-letter and two-letter prefixes are queried constantly) with a tiny LRU/CDN cache, so a large fraction of typeahead requests never even touch the trie — they are served from an in-memory cache at the edge. **Debouncing on the client** (wait ~100–150ms after the last keystroke before firing) cuts request volume sharply without hurting perceived speed. **Top-k truncation**: you only ever need ~8 completions, so you store exactly that many per node, keeping the structure small. And **personalization is layered carefully**: a globally precomputed top-k is the base, and any per-user reordering happens as a light, bounded step on top — you never abandon precompute for a from-scratch personalized search on the keystroke path, because that reintroduces the latency you precomputed to avoid.

The caching mechanics — what to cache, TTLs, the stampede when a hot key expires, and the consistency pitfalls — are a deep topic in their own right; this design leans hard on them, and the patterns and traps are laid out in [caching strategies and the pitfalls that bite](/blog/software-development/system-design/caching-strategies-and-the-pitfalls-that-bite). The one typeahead-specific caching note: because typeahead is allowed to be stale (recall our SLO let it degrade), you can use generous TTLs and serve slightly old completions during a trie rebuild without anyone noticing. That permission to be stale is what makes the cache so effective.

#### Worked example: why precompute beats query-time for typeahead

Compare the two approaches at 40,000 typeahead QPS. **Query-time approach**: each keystroke runs a prefix search — scatter to 10 shards, each finds documents matching the prefix, scores and returns top-k, broker merges. Even optimistically that is 20–40ms of work touching the full search tier, and at 40k QPS it would dominate your cluster: it is the same per-query cost as search but at 8× the volume, so it would need roughly 8× the search capacity — order **200+ extra machines** just for the keystroke path. **Precompute approach**: the trie answers in ~2–5ms from memory, a prefix cache absorbs maybe 70% of requests before they reach the trie, and the whole typeahead tier is a handful of memory-heavy nodes plus a cache — call it **5–10 machines**. The precompute approach is roughly **20× cheaper in machines** and hits a tighter latency budget, and the only thing it gives up is freshness: a brand-new query term might not appear as a completion until the next trie rebuild (minutes to hours). For typeahead, that staleness is invisible and the cost saving is enormous. This is the optimization in one comparison: **typeahead is precompute-and-cache, full stop.**

## 10. Deep dive: the indexing pipeline and freshness

So far the index has been static. In reality documents are created, updated, and deleted constantly, and the product promises a new document is searchable within seconds. That promise has a real cost, and the way you pay it is the indexing pipeline.

The core tension is **index versus serve**. The most query-efficient index is one big, fully sorted, fully merged structure — but building that is slow and you cannot rebuild it on every write. The most write-efficient approach is to append every change to a tiny new structure — but then a query has to consult many structures and merge their results, which is slow. The resolution is the same one log-structured storage engines use: **write to small immutable segments, and merge them in the background**. A new or changed document is analyzed and written to a fresh, small **segment** (Lucene's term; functionally an SSTable). That segment becomes searchable as soon as it is flushed — within about a second — which is how you get near-real-time freshness. But now a query must search across *all* segments and merge their results, and segment count grows with write volume, dragging query latency up. So a background **merge** process periodically compacts many small segments into fewer large ones, reclaiming space (deleted documents are finally purged) and keeping query fan-out across segments low.

![Near-real-time indexing pipeline where a document is ingested, analyzed into terms, written to a fresh searchable segment, and later compacted by background segment merges](/imgs/blogs/design-a-search-and-autocomplete-system-4.webp)

If that pattern feels familiar, it should: **segments are SSTables and merges are compaction**, the exact mechanics of a log-structured merge tree. A search index is, structurally, an LSM tree whose values happen to be posting lists. The deep mechanics — why writes go to immutable runs, how compaction trades write amplification for read efficiency, the tombstone dance for deletes — are covered thoroughly in [LSM trees: write-optimized storage engines](/blog/software-development/database/lsm-trees-write-optimized-storage-engines) and contrasted with the B-tree world in [storage engines: B-trees vs LSM-trees for architects](/blog/software-development/system-design/storage-engines-btrees-vs-lsm-trees-for-architects). The architect's takeaway is the trade you inherit by choosing this design: **freshness costs you merge work and read amplification.** More frequent flushes mean fresher search but more small segments and heavier background merging; less frequent flushes mean staler search but a cleaner index. The refresh interval is a tuning knob — Elasticsearch defaults to a 1-second refresh, and the first thing a senior does when bulk-loading is *turn refresh off*, load, then turn it back on, because refreshing on every batch during a bulk load is pure waste.

The merge policy itself is a knob worth understanding, because it is where freshness, query latency, and background I/O are balanced. A tiered merge policy groups segments by size and merges a tier once it accumulates enough segments — say, merge ten ~50 MB segments into one ~500 MB segment, then merge ten of those into a ~5 GB segment, and so on. This bounds the number of segments a query must visit (roughly logarithmic in the data size) while keeping each individual merge a manageable amount of work. The trade is direct: merge *more eagerly* and queries see fewer segments (lower latency) but you spend more disk and CPU on background merging (and you re-write the same data multiple times — write amplification); merge *less eagerly* and you save I/O but queries fan out across more segments and slow down. There is also a practical ceiling — most engines cap the maximum segment size (Lucene defaults around 5 GB) so that a single segment never becomes so large that merging it stalls the system or so large that it cannot be moved during a shard relocation.

#### Worked example: tuning refresh interval against freshness and merge cost

Suppose your product currently refreshes every 1 second (documents searchable within ~1s) and you are drowning in background merge I/O because writes are heavy. Each 1-second refresh flushes a tiny segment, and tiny segments mean *constant* merging to keep the count down — you are paying write amplification on a relentless treadmill. Now relax freshness from 1s to 30s. Each flush now batches 30 seconds of writes into one larger segment, so you create ~30× fewer segments, which means dramatically fewer merges and far less write amplification — in practice this can cut background I/O by more than half on a write-heavy index. The cost: a new document now takes up to 30 seconds to become searchable instead of 1. For a corpus of support articles or product listings, 30-second freshness is completely acceptable and the I/O savings let you run on fewer or cheaper nodes; for a live messaging search where a message must appear instantly, it is not. The senior move is to *measure freshness as a product requirement* (does anyone actually need sub-second?) and set the refresh interval to the loosest value the product tolerates, because every notch tighter is paid in merge I/O forever. And during a bulk reindex, set the refresh interval to off entirely, load, then restore it — refreshing mid-bulk-load is pure waste, creating millions of tiny segments you immediately merge away.

Where do the changes come from? You do not want application code writing to both the source-of-truth database and the search index in the same request — that dual-write is a classic consistency bug, because one write can succeed and the other fail, and now your index disagrees with your database silently. Instead, capture changes from the database's own log and stream them to the indexer: this is **change data capture**, and pairing it with the **outbox pattern** guarantees the index eventually reflects every committed change exactly once, even across failures. The pattern is detailed in [change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern); for search it is the right way to keep a derived index consistent with a primary store without dual-write risk. The indexer consumes the change stream (often via a durable log like Kafka), analyzes each changed document, and writes it to the appropriate shard's newest segment. Because the change stream is ordered and replayable, you can rebuild the entire index from scratch by replaying the log — which is exactly what you do during a reindex.

## 11. Deep dive: query understanding (spelling, synonyms)

A query is not always clean. Users misspell (*wireles hedphones*), use synonyms (*earbuds* for *in-ear headphones*), and type in ways the index does not literally contain. **Query understanding** sits between the raw query and retrieval, and it is where a lot of perceived quality comes from.

**Spelling correction** typically runs as a fast pre-retrieval step: the misspelled token is checked against the term dictionary (the FST makes fuzzy/edit-distance lookups feasible), and likely corrections are suggested or auto-applied — "showing results for *wireless headphones*" with an option to search the original. The classic implementation uses edit distance bounded to 1 or 2 against in-dictionary terms, weighted by term frequency so common corrections win. **Synonym expansion** maps query terms to equivalents either at index time (store *earbuds* documents under *in-ear headphones* too) or at query time (expand the query to match both) — query-time expansion is more flexible and avoids re-indexing when the synonym list changes, at the cost of larger queries. **Stemming and lemmatization**, already in the analyzer, are the simplest form of query understanding: they make morphological variants match.

The architectural note is that query understanding is *cheap per query but high-leverage*: a small edit-distance lookup and a synonym map expansion cost a millisecond or two but can swing recall and user satisfaction enormously, because a query that matches nothing is a total failure regardless of how good your ranking is. The senior instinct is to spend a little latency budget here before retrieval, because the alternative — zero results — is the worst outcome a search box can produce. A common production layering: spelling correction and synonym expansion run on the broker before fan-out, so the shards receive a clean, expanded query and the expensive scatter-gather happens once on the corrected query rather than twice.

There is a subtle ordering decision here that separates a careful design from a sloppy one. If you auto-correct too aggressively, you frustrate the user who *meant* the unusual spelling — a search for the brand *Flickr* should not be silently rewritten to *flicker*. The robust pattern is **conditional correction**: run the original query first, and only fall back to a corrected query when the original returns too few results, surfacing the correction as "showing results for X — search instead for Y." This costs a little — sometimes you run two retrievals — but it respects user intent and avoids the worst failure (confidently returning the wrong thing). The same logic applies to synonym expansion: over-expanding a query (treating loosely-related terms as equivalent) dilutes precision and floods the top results with marginally-relevant documents, so synonym sets are curated conservatively and weighted, not dumped in wholesale. A senior frames query understanding as a *recall-precision dial*, not a free win: each expansion or correction trades some precision for recall, and the right setting depends on whether your users suffer more from "no results" or from "wrong results."

One coordination detail sits underneath all of this and is easy to forget: the cluster needs a consistent, agreed-upon view of *which shard lives on which node, which replica is primary, and what the current index aliases point to* — the cluster metadata. This is a small but critical piece of strongly-consistent state, and getting it wrong (two nodes disagreeing on which is the primary for a shard) leads to split-brain index corruption. Search clusters solve it with a consensus-backed metadata store — Elasticsearch uses its own Raft-like coordination layer, and other systems lean on ZooKeeper or etcd — so that shard assignments and alias swaps are agreed by a quorum, not raced. The mechanics of how that agreement is reached, and why a majority quorum is the price of avoiding split-brain, are covered in [consensus and coordination in distributed systems](/blog/software-development/system-design/consensus-and-coordination-in-distributed-systems); the architect's takeaway is that the *data plane* of search (the inverted index, posting lists, scatter-gather) is eventually-consistent and cheap to scale, while the *control plane* (who owns which shard, where aliases point) is strongly-consistent and must go through consensus. Keeping those two planes separate — heavy eventually-consistent data, light strongly-consistent metadata — is a pattern you will see in every well-built distributed datastore.

## 12. Trade-offs: the decision matrix a senior puts on the board

Now the heart of the senior framing: no index design wins on everything, so you choose the column whose losses you can live with. We have four candidate approaches and four properties that matter, and the honest answer is a matrix.

![Trade-off matrix scoring trie typeahead, BM25 shards, two-phase learning-to-rank, and vector ANN against latency, freshness, relevance, and index cost](/imgs/blogs/design-a-search-and-autocomplete-system-7.webp)

Read the matrix as a senior would, column by column:

| Approach | Latency p99 | Freshness | Relevance | Index cost | When it wins |
| --- | --- | --- | --- | --- | --- |
| **Trie typeahead** | < 50ms (best) | hours (worst) | prefix-only | low | The keystroke path; precompute, never query-time |
| **BM25 shards** | 30–80ms | seconds (best) | lexical, decent | medium | The default full-text core; cheap, fresh, no training |
| **Two-phase L2R** | 60–150ms | seconds | best | high | When relevance is the product and you have click data |
| **Vector / ANN** | 40–100ms | minutes | semantic | high | Semantic recall, synonyms-by-meaning, multimodal |

The reasoning behind the picks: **trie typeahead** trades freshness for unbeatable latency and low cost — perfect for the keystroke path where stale completions are invisible and speed is everything. **BM25 shards** are the workhorse you build first: fresh within seconds, no model to train, decent lexical relevance, moderate cost — it is the right default for most products and the baseline everything else improves on. **Two-phase learning-to-rank** buys the best relevance but pays in latency (the second model adds tens of milliseconds), operational complexity (you now run a model-serving tier and need click data and a training loop), and cost — you reach for it when relevance *is* the product and you have the click data to feed it. **Vector ANN** buys semantic recall but pays in index cost and freshness (re-embedding is heavier than re-indexing), so it is an *addition* to lexical retrieval, not a replacement, until you have proven the lexical core.

The senior conclusion is layered, not singular: **trie for typeahead, BM25 as the always-on retrieval core, learning-to-rank as a second phase when relevance justifies the ops cost, and vectors as a hybrid recall source once the core is solid.** You do not pick one row; you compose them, each for the job its costs suit.

### Rejected alternatives, and why

A senior also names what they *considered and rejected*, because that is where the real thinking shows.

- **"Just use a SQL `LIKE '%term%'` query."** Rejected: a leading-wildcard `LIKE` cannot use a B-tree index, so it scans every row — fine at a few thousand rows, catastrophic at 50 million. Full-text search exists precisely because relational indexes are built for exact and prefix lookups, not for relevance-ranked free text.
- **"Run typeahead through the search index at query time."** Rejected on cost and latency, as the worked example showed: ~8× the search volume on the most expensive path, ~20× more machines, and a latency budget it cannot hit. Precompute wins decisively.
- **"One giant shard, no fan-out, no tail-latency problem."** Tempting — it sidesteps the scatter-gather tail entirely — but it does not fit in one machine's memory at 150 GB plus growth, and it caps your write and query throughput at one node. The tail-latency tax is real, but the answer is to *manage* fan-out (hedging, fewer fatter shards), not to refuse to shard.
- **"Index synchronously on every write, in the request path."** Rejected: it couples write latency to index latency, risks dual-write inconsistency, and makes bulk loads brutal. Asynchronous indexing via a change stream is the right separation, accepting a few seconds of indexing lag as the price of decoupling.

## 13. Stress test: what breaks at 10×, at a hot key, at a rebuild

A design you have not stress-tested is a design you do not understand. Let us break this one on purpose.

**The celebrity / hot-term query.** A breaking news event or a viral product makes one query — say *taylor swift tickets* — explode to a huge fraction of all traffic in minutes. The posting list for those terms is hammered, the same scatter-gather runs millions of times for an identical query, and the ranking tier melts re-scoring the same candidates over and over. The fix is **caching at the query level**: cache the full ranked result for hot queries with a short TTL, so the millionth identical query is served from cache, not recomputed. This is a textbook **hot-key** problem, and the mitigations — request coalescing so only one of N simultaneous misses recomputes while the rest wait, short TTLs to bound staleness, and protecting against the cache-stampede when the hot entry expires — are exactly the pitfalls covered in [caching strategies and the pitfalls that bite](/blog/software-development/system-design/caching-strategies-and-the-pitfalls-that-bite). The senior move is to recognize that a hot query is a *cache* problem, not an *index* problem; you do not re-shard for it, you cache it. The same logic protects typeahead: the single-letter prefixes are permanent hot keys, which is exactly why the prefix cache sits in front of the trie.

**The fan-out tail spike.** One node starts GC-pausing or its disk degrades, and suddenly the shard it hosts is a straggler. Because every query touches every shard, *that one slow node taints a large fraction of all queries* — recall the 0.99^10 math: one bad shard out of ten can drag ~10% of queries into the tail even though 90% of the cluster is healthy. This is the scariest failure mode in scatter-gather because it is non-local: one machine's hiccup is everyone's p99. The mitigations are the ones from the sharding section — hedged requests to a second replica (the single highest-leverage fix), adaptive replica selection that routes traffic away from the slow node automatically, and aggressive health-based ejection. A senior watches **shard-level p99**, not just cluster p99, precisely so they can spot the one bad node before it owns the tail.

**The index rebuild.** You change the analyzer (say, add a synonym or fix the stemming), which means every document must be re-analyzed and re-indexed — a full reindex of 50 million documents. Done naively, you rebuild in place and either take downtime or serve a half-rebuilt, inconsistent index. The senior approach is **build a new index version alongside the live one** (replay the change log into a fresh index), validate it, and **atomically swap an alias** from the old index to the new one — Elasticsearch's index aliases exist for exactly this. The live index keeps serving throughout; the swap is instantaneous; and if the new index is wrong, you swap the alias back. The cost is temporarily running two copies (double the storage during the rebuild), which is cheap insurance against a bad reindex. This is the search-specific instance of a general principle: *derived data should be rebuildable from a log, and cutovers should be atomic alias swaps, not in-place mutations.*

#### Worked example: surviving a 10× search spike

Suppose a marketing event 10×'s search traffic from 5,000 to 50,000 QPS for an hour. Walk the components. The **brokers** are stateless, so you autoscale them horizontally — add instances, they take traffic immediately. The **shards** are the bottleneck: at 10× you would naively need 10× the shard capacity (~250 vs 25 nodes), which you cannot conjure instantly. Three levers absorb the spike instead of brute-force scaling. First, **query result caching**: a traffic spike is usually concentrated on a few queries, so a query cache with even a 30-second TTL can serve 50–80% of the spike from memory, cutting the shard load from 50k to maybe 15k QPS. Second, **graceful degradation**: under extreme load, drop phase-two learning-to-rank and serve BM25-only results — relevance dips slightly but latency holds and you cut per-query cost sharply. Third, **shed the non-critical**: typeahead can serve more aggressively from cache (longer TTLs) and even drop personalization, freeing capacity. With caching absorbing the bulk and degradation cutting per-query cost, the residual ~15k QPS is a ~3× scale-up of the shard tier, which autoscaling can handle on warm replicas. The lesson: **you survive 10× by caching the repetitive, degrading the expensive, and scaling the residual — not by provisioning 10× idle capacity.** Capacity planning for exactly this kind of spike is its own discipline, covered in [back-of-the-envelope estimation for system design](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design).

## 14. Case studies: how real systems do it

Theory is cheaper to remember when it is attached to systems you can name. Three real architectures map directly onto everything above.

**Elasticsearch and Apache Lucene.** Lucene is the inverted-index engine; Elasticsearch is the distributed system wrapped around it. Lucene gives you exactly the pieces we built: analyzers, posting lists, BM25 scoring (Lucene switched its default from TF-IDF to BM25 years ago for the reasons in section 7), and **immutable segments merged in the background** — the LSM pattern, in a search engine. Elasticsearch adds sharding (each shard is a full Lucene index), replication, the **coordinating node** that does our scatter-gather and merge, the ~1-second **refresh** that gives near-real-time freshness, and **index aliases** for the atomic-swap reindex. Adaptive replica selection — routing shard-queries to the replica that has been fastest lately — is a built-in feature precisely because the tail-latency tax of fan-out is real and Elastic's engineers measured it in production. If you want to see every concept in this article running in one open-source system, read Elasticsearch's docs on shards, refresh, and search-after pagination; the vocabulary lines up one-to-one.

**Google-scale typeahead.** Google's autocomplete famously returns completions in tens of milliseconds at planetary QPS, and the architecture is the precompute-and-cache story taken to its limit. The completions are mined from **aggregated query logs** (what people actually search, weighted by frequency and freshness, with heavy filtering for safety and policy), the top completions per prefix are **precomputed and served from memory** at the edge close to the user, and the hottest prefixes live in caches that absorb the bulk of traffic. The reason it is fast is the reason ours is fast: the expensive work — figuring out what *wirel* should complete to — happened offline, so the keystroke path is a lookup, not a search. Google layers in real-time trends (a breaking event can promote a completion faster than a full rebuild) and personalization, but the load-bearing idea is precompute. The lesson for your design: do not let "make it personalized and real-time" tempt you into doing search work on the keystroke path; layer those as bounded adjustments on top of a precomputed base.

**The "tail at scale" lesson (Google, Jeff Dean).** The single most cited result about fan-out systems is Dean and Barroso's observation that in a system where a request touches many servers, the overall latency is dominated by the slowest of them — and that at scale, *some* server is always slow. Their proposed cure, **hedged and tied requests**, is exactly the mitigation in section 6: send a backup request to a second replica when the first is slow, and cancel the loser. This is not a search-specific trick; it is the general law of scatter-gather, and search is its most common application. Internalize it as a senior, because the moment you fan a request out to N things and wait for all N, you have signed up for the tail, and hedging is your cheapest way out. The deeper treatment of why tail latency is a first-class architectural force lives in [articulating trade-offs: CAP, PACELC, and beyond](/blog/software-development/system-design/articulating-tradeoffs-cap-pacelc-and-beyond).

A fourth, briefer one worth naming: **e-commerce search (Amazon-style)** is where two-phase ranking earns its keep. The first phase retrieves on keywords and hard filters (in-stock, price range, category — all posting-list intersections), and the second phase reranks on a learned model fed by purchase and click signals, because for commerce the *order* of results is directly revenue, so the expensive L2R phase pays for itself. It is the clearest case of "relevance is the product" justifying the operational cost of a model-serving tier.

## 15. When to reach for this (and when not to)

A senior is decisive about when *not* to build this. Search is a serious system; do not summon it before you need it.

**Reach for a full search engine when**: you have free-text queries over a corpus large enough that relational `LIKE` and exact-match indexes fail (tens of thousands of documents and up, growing); relevance ranking matters (the *order* of results affects the user's success or your revenue); or you need typeahead at meaningful volume. At that point, an inverted-index engine — Elasticsearch/OpenSearch, or a hosted equivalent — is the right tool, and the architecture in this article is what you are buying.

**Do not reach for it when**: your "search" is really a filtered lookup over a few thousand structured rows — a `WHERE category = ? AND price < ?` against a proper relational index is faster, simpler, transactionally consistent, and one fewer system to operate. Do not stand up an Elasticsearch cluster to filter a small product catalog; you will spend more time keeping the index in sync with your database than you ever save. Similarly, if you only need *exact-prefix* autocomplete over a small, static list (say, country names), a trie in application memory — or even a sorted array with binary search — beats a search cluster. And if your queries are genuinely semantic from the start (natural-language questions over documents), a pure vector store might be the better primitive than a lexical inverted index, though the strongest systems hybridize both.

The decision tree below is the compressed version of this judgment: let the *query shape* — prefix versus full keyword, relevance-critical or not — pick the index structure, rather than reaching for the most powerful engine by default.

![Decision tree that routes from the query shape to an index strategy: a prefix query to a trie, a full keyword query to BM25 shards, and a relevance-critical query to two-phase ranking or vector search](/imgs/blogs/design-a-search-and-autocomplete-system-9.webp)

The forward-looking note: once you have search and typeahead solid, the next system that exercises the same low-latency, conflict-aware muscles is real-time collaboration — many users mutating shared state with sub-100ms feedback — which the next post tackles in [design a collaborative editor](/blog/software-development/system-design/design-a-collaborative-editor). The shared lesson is that latency budgets, precomputation, and careful state derivation are what separate a system that feels instant from one that merely works.

## 16. Key takeaways

- **The inverted index turns search from a compute problem into a set-algebra problem.** A multi-word query is an intersection or union of precomputed sorted posting lists; everything fast downstream is built on that one transformation.
- **Fan-out taxes the tail.** When a query touches N shards and waits for all, its p99 is the max of N latencies, so adding shards *worsens* query p99. Manage it with hedged requests, fewer fatter shards, and adaptive replica selection — and watch shard-level p99, not just cluster p99.
- **Rank in two phases.** Cheap BM25 retrieval cuts millions to hundreds; expensive learning-to-rank reorders only the survivors. This is how you afford good relevance without paying for it on every document.
- **Typeahead is a different system: precompute and cache.** It is faster, hotter, and staler-tolerant than search. Build top-k per prefix into a trie/FST offline, front it with a prefix cache, and never do search work on the keystroke path. It is ~20× cheaper than the query-time alternative.
- **Freshness costs merge work.** A search index is an LSM tree — segments are SSTables, merges are compaction — so faster freshness means more small segments and heavier background merging. The refresh interval is a tuning knob; turn refresh off during bulk loads.
- **Keep the index consistent with the source via a change stream, not dual writes.** Change data capture plus the outbox pattern make the derived index eventually consistent without the dual-write bug, and a replayable log lets you rebuild the index from scratch.
- **Survive spikes by caching the repetitive, degrading the expensive, and scaling the residual.** A hot query is a cache problem, not an index problem. A 10× spike is absorbed by query caching and graceful degradation long before you provision 10× capacity.
- **Reindexes are atomic alias swaps, not in-place mutations.** Build the new index version alongside the live one, validate, swap the alias, and keep the old one for instant rollback.
- **Don't summon search before you need it.** For a few thousand structured rows, a relational filter beats an Elasticsearch cluster you have to keep in sync.

## Further reading

- [Storage engines: B-trees vs LSM-trees for architects](/blog/software-development/system-design/storage-engines-btrees-vs-lsm-trees-for-architects) — why search segments behave exactly like SSTables, and the write-vs-read trade you inherit.
- [LSM trees: write-optimized storage engines](/blog/software-development/database/lsm-trees-write-optimized-storage-engines) — the compaction mechanics behind segment merges.
- [Caching strategies and the pitfalls that bite](/blog/software-development/system-design/caching-strategies-and-the-pitfalls-that-bite) — query caching, prefix caching, hot-key stampedes, and TTL discipline that this design leans on.
- [Partitioning and sharding without downtime](/blog/software-development/system-design/partitioning-and-sharding-without-downtime) — how to split and rebalance the shard tier safely.
- [Articulating trade-offs: CAP, PACELC, and beyond](/blog/software-development/system-design/articulating-tradeoffs-cap-pacelc-and-beyond) — tail latency under fan-out as a first-class architectural force.
- [Back-of-the-envelope estimation for system design](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design) — the sizing technique used to derive QPS, index size, and shard counts here.
- [Change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — keeping the derived search index consistent with the source of truth without dual writes.
- Jeff Dean and Luiz Barroso, "The Tail at Scale" (CACM, 2013) — the foundational paper on hedged requests and fan-out latency; the official Elasticsearch and Apache Lucene documentation on segments, refresh, BM25 similarity, and index aliases for the production specifics.
- Forward: [design a collaborative editor](/blog/software-development/system-design/design-a-collaborative-editor) — the next system that lives or dies by latency budgets and carefully derived state.
