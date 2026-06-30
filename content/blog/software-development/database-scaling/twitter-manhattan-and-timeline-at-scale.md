---
title: "Twitter at Scale: The Timeline Fan-Out and Building Manhattan"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "A case study of Twitter/X's storage story — the home-timeline fan-out tradeoff, the celebrity hot-key hybrid, Snowflake IDs, and why Twitter built Manhattan, its own multi-tenant key-value store."
tags: ["database-scaling", "twitter", "manhattan", "timeline-fanout", "snowflake-id", "redis", "key-value-store", "tunable-consistency", "hot-partition", "case-study", "distributed-systems", "system-design"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 30
---

For a stretch of the early 2010s, the most famous error page on the internet was a cartoon whale being lifted out of the ocean by a flock of orange birds. Twitter showed it whenever the site fell over, which in those years was often. The fail whale became a meme, then a t-shirt, then a kind of badge of honor — but behind it was a real and specific engineering problem: a write-heavy social product growing faster than any single database could absorb, built on a stack that was never designed to fan a single tweet out to millions of timelines in seconds.

Twitter's eventual answer to that problem is one of the most-taught case studies in distributed systems, and it gives us two distinct lessons that show up in nearly every system-design interview and every scaling roadmap since. The first is the **home-timeline fan-out tradeoff**: do you do the expensive work when a tweet is *written*, or when a timeline is *read*? The second is **build-vs-buy at hyperscale**: at some point Twitter stopped operating a sprawl of MySQL and Cassandra clusters and built its own internal key-value store, Manhattan, turning storage into a platform that other teams consume rather than operate. This post is a tour of both.

![The fail-whale-era Twitter stack: web tier fanning out to memcached, Snowflake, FlockDB, and Gizzard-sharded MySQL](/imgs/blogs/twitter-manhattan-and-timeline-at-scale-1.webp)

The diagram above is the mental model for the first half of the article. A request hits the web tier and immediately fans out: to **memcached** for hot reads, to **Snowflake** for a fresh time-sortable ID, to **FlockDB** for the follow graph, and to **MySQL** shards sitting behind a sharding middleware called **Gizzard**. Hold that picture. Every problem and every fix in the early Twitter story is a consequence of how those pieces fit together — and the single hardest piece, the home timeline, is the one that does not appear as a box because it is not a store at all. It is a *computation* over the follow graph, and where you run that computation is the whole game.

## Why Twitter's storage problem is different

Before any architecture, you have to be honest about the workload, because a social timeline has a shape that quietly breaks the obvious designs. The instinct is to treat it like a feed you can cache and forget. It is not. Here is the gap between how people assume the system behaves and how it actually behaves at Twitter's scale.

| Assumption | Naive view | Reality at Twitter scale |
| --- | --- | --- |
| It is mostly reads | "Cache the feed, serve from RAM" | Reads dominate *volume*, but a single write (one tweet) can trigger millions of downstream writes |
| A timeline is one query | "Select tweets where author in (my follows)" | That query touches thousands of partitions and is far too slow to run per page-load |
| Load is roughly uniform | "Users are similar" | Follower counts span eight orders of magnitude — most users have hundreds, a few have hundreds of millions |
| IDs are easy | "Use auto-increment" | A single auto-increment column is a coordination bottleneck and leaks ordering across shards |
| One database scales with hardware | "Buy a bigger box" | The working set and the write rate both outgrew any single machine years ago |
| Storage is one team's problem | "DBAs own the database" | Dozens of product teams each needed a datastore; operating one cluster per team did not scale organizationally |

Every row of that table points at the same underlying truth: at this scale, the bottleneck is rarely raw storage. It is the *fan-out* — the multiplication factor between one logical action and the physical work it implies — and the *operational surface* of running many stateful systems. Twitter's two famous contributions, the timeline hybrid and Manhattan, are direct answers to those two pressures. The follow graph that drives the fan-out lives in [FlockDB](/blog/software-development/database/cassandra-and-dynamodb-leaderless-deep-dive); the hot reads that keep the site alive lean on the same caching discipline covered in [the caching hierarchy at scale](/blog/software-development/database-scaling/the-caching-hierarchy-at-scale).

## 1. The fail-whale era: the stack that held it together

**The senior rule of thumb for this era is that Twitter did not solve scaling with one clever database — it solved it by separating concerns into purpose-built pieces and gluing them together.** Each piece in the mental-model diagram had a job, and the jobs map almost perfectly onto the hard parts of a social graph.

**MySQL plus memcached** was the substrate, the same workhorse pairing that powered most of the social web. MySQL held the durable rows; memcached absorbed the read traffic so the database did not have to. This is unremarkable until you ask the next question: how do you shard MySQL when one box is no longer enough, and how do you keep adding boxes without rewriting the application every time?

**Gizzard** was Twitter's answer. It is a sharding framework — middleware that sits between the application and a fleet of datastores and routes each query to the correct shard based on a partitioning table. Gizzard is datastore-agnostic: it manages partitioning, replication, and fault tolerance, and forwards the actual reads and writes to whatever backend you plug in (typically MySQL). The application talks to Gizzard as if it were one logical store; Gizzard knows the topology. That indirection is exactly the move that lets you grow the physical fleet without the application learning new addresses — the same separation between logical partition and physical capacity that [Instagram baked into its IDs](/blog/software-development/database-scaling/instagram-sharding-ids-in-postgres).

**FlockDB** was the graph. The follow relationship — who follows whom, who blocks whom — is a set of edges, and at Twitter's scale it is an enormous set. FlockDB is a distributed store built on top of Gizzard and MySQL that holds those edges as adjacency lists. Critically, it is *not* a graph-traversal database in the Neo4j sense; it does not walk paths. It does shallow, extremely wide operations: list the followers of an account, test whether A follows B, intersect two follower sets, and page through millions of edges quickly. Those are precisely the queries the timeline fan-out needs, and FlockDB was tuned to answer them at high throughput.

### Snowflake: IDs that sort by time without a coordinator

The fourth piece is the one whose influence outlived Twitter's early stack and spread across the entire industry. When you shard a database, the humble auto-increment primary key becomes a liability: it requires a single source of truth to hand out the next number, which is a coordination bottleneck and a single point of failure. Twitter needed IDs that were unique across the whole fleet, generatable on any node without talking to a coordinator, and — this is the subtle requirement — roughly **sortable by time**, so that "the most recent tweets" is a range scan rather than a sort. The answer was **Snowflake**.

![Snowflake's 64-bit layout: a 41-bit timestamp in the high bits, then machine id, then a per-millisecond sequence](/imgs/blogs/twitter-manhattan-and-timeline-at-scale-3.webp)

A Snowflake ID is a 64-bit integer carved into fields, as the figure shows. The top bit is unused (it keeps the value positive when interpreted as a signed integer). The next **41 bits** hold a millisecond timestamp measured from a custom epoch — enough for about 69 years of milliseconds. The next **10 bits** identify the machine generating the ID, split into a datacenter id and a worker id, so up to 1024 generators can run concurrently. The final **12 bits** are a per-millisecond sequence counter, letting a single machine mint 4096 IDs in the same millisecond before it has to wait for the clock to tick. Because the timestamp sits in the *high* bits, comparing two IDs as plain integers compares them first by time — the IDs are k-sorted by creation order for free.

Here is a faithful, runnable reconstruction of the generator. The bit math is the entire point, so read the shifts carefully.

```python
import time
import threading

class Snowflake:
    """Twitter-style 64-bit, time-sortable, coordinator-free ID generator.

    Layout (MSB -> LSB):  1 sign | 41 timestamp | 5 datacenter | 5 worker | 12 seq
    """
    EPOCH_MS      = 1288834974657   # Twitter's custom epoch (2010-11-04)
    DC_BITS       = 5
    WORKER_BITS   = 5
    SEQ_BITS      = 12

    MAX_DC        = (1 << DC_BITS) - 1        # 31
    MAX_WORKER    = (1 << WORKER_BITS) - 1    # 31
    SEQ_MASK      = (1 << SEQ_BITS) - 1       # 4095

    WORKER_SHIFT  = SEQ_BITS                  # 12
    DC_SHIFT      = SEQ_BITS + WORKER_BITS    # 17
    TS_SHIFT      = SEQ_BITS + WORKER_BITS + DC_BITS  # 22

    def __init__(self, datacenter_id: int, worker_id: int):
        assert 0 <= datacenter_id <= self.MAX_DC
        assert 0 <= worker_id <= self.MAX_WORKER
        self.datacenter_id = datacenter_id
        self.worker_id = worker_id
        self.seq = 0
        self.last_ts = -1
        self.lock = threading.Lock()

    def _now(self) -> int:
        return int(time.time() * 1000)

    def next_id(self) -> int:
        with self.lock:
            ts = self._now()
            if ts == self.last_ts:
                # Same millisecond: bump the sequence; spin if we overflow 4096.
                self.seq = (self.seq + 1) & self.SEQ_MASK
                if self.seq == 0:
                    while ts <= self.last_ts:
                        ts = self._now()
            else:
                self.seq = 0
            self.last_ts = ts
            return (((ts - self.EPOCH_MS) << self.TS_SHIFT)
                    | (self.datacenter_id << self.DC_SHIFT)
                    | (self.worker_id << self.WORKER_SHIFT)
                    | self.seq)

def decode(snowflake_id: int) -> dict:
    return {
        "timestamp_ms": (snowflake_id >> Snowflake.TS_SHIFT) + Snowflake.EPOCH_MS,
        "datacenter":   (snowflake_id >> Snowflake.DC_SHIFT) & Snowflake.MAX_DC,
        "worker":       (snowflake_id >> Snowflake.WORKER_SHIFT) & Snowflake.MAX_WORKER,
        "sequence":      snowflake_id & Snowflake.SEQ_MASK,
    }
```

The decode function is worth dwelling on because it reveals the trick that ties this section to [Instagram's sharding scheme](/blog/software-development/database-scaling/instagram-sharding-ids-in-postgres). In both designs, the ID is not an opaque token — it is a *struct* you can take apart with a bit-shift. Twitter packs a timestamp so IDs sort by time; Instagram packs a shard id so the routing layer can find a row's home with no lookup table. Same primitive, different field: encode the one piece of metadata you will most often need at read time directly into the key, and you buy yourself a join or a coordination round-trip every single request. That idea — self-describing IDs — is now everywhere, from Discord to Sonyflake to Instagram, and it traces straight back to this 64-bit word.

> The cheapest distributed lookup is the one you never make because the answer was already encoded in the key.

## 2. The timeline problem: where do you do the work?

Now the centerpiece. The home timeline is the screen you see when you open the app: every tweet from every account you follow, newest first, merged into one list. Stated as a database query it looks trivial:

```sql
SELECT * FROM tweets
WHERE author_id IN (SELECT followee_id FROM follows WHERE follower_id = :me)
ORDER BY created_at DESC
LIMIT 50;
```

That query is a trap. If you follow two thousand accounts, the `IN` clause spans two thousand authors whose tweets are scattered across thousands of shards. Running it on every page-load, for hundreds of millions of users, dozens of times a day, is not something any relational database will survive. So the real engineering question is not *how do I write this query* — it is *when do I do this work*. There are exactly two answers, and the entire timeline architecture is the tension between them.

<figure class="blog-anim">
<svg viewBox="0 0 760 380" role="img" aria-label="Two panels contrast the two timeline strategies: on the left a single tweet fans out to many follower timelines at write time; on the right a single read pulls in tweets from many followees at read time" style="width:100%;height:auto;max-width:860px">
<style>
.a2-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.a2-hub{fill:var(--accent,#6366f1);stroke:none}
.a2-lbl{font:600 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.a2-hublbl{font:600 13px ui-sans-serif,system-ui;fill:#fff;text-anchor:middle}
.a2-title{font:700 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.a2-sub{font:600 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.a2-div{stroke:var(--border,#d1d5db);stroke-width:1.5;stroke-dasharray:5 5}
.a2-dot{fill:var(--accent,#6366f1);opacity:1}
@keyframes a2-w1{0%{transform:translate(0,0);opacity:0}12%{opacity:1}86%{opacity:1}100%{transform:translate(135px,-131px);opacity:0}}
@keyframes a2-w2{0%{transform:translate(0,0);opacity:0}12%{opacity:1}86%{opacity:1}100%{transform:translate(135px,-71px);opacity:0}}
@keyframes a2-w3{0%{transform:translate(0,0);opacity:0}12%{opacity:1}86%{opacity:1}100%{transform:translate(135px,-11px);opacity:0}}
@keyframes a2-w4{0%{transform:translate(0,0);opacity:0}12%{opacity:1}86%{opacity:1}100%{transform:translate(135px,49px);opacity:0}}
@keyframes a2-r1{0%{transform:translate(0,0);opacity:0}14%{opacity:1}86%{opacity:1}100%{transform:translate(140px,131px);opacity:0}}
@keyframes a2-r2{0%{transform:translate(0,0);opacity:0}14%{opacity:1}86%{opacity:1}100%{transform:translate(140px,71px);opacity:0}}
@keyframes a2-r3{0%{transform:translate(0,0);opacity:0}14%{opacity:1}86%{opacity:1}100%{transform:translate(140px,11px);opacity:0}}
@keyframes a2-r4{0%{transform:translate(0,0);opacity:0}14%{opacity:1}86%{opacity:1}100%{transform:translate(140px,-49px);opacity:0}}
.a2-w1{animation:a2-w1 7s linear infinite}
.a2-w2{animation:a2-w2 7s linear infinite .35s}
.a2-w3{animation:a2-w3 7s linear infinite .7s}
.a2-w4{animation:a2-w4 7s linear infinite 1.05s}
.a2-r1{animation:a2-r1 7s linear infinite}
.a2-r2{animation:a2-r2 7s linear infinite .12s}
.a2-r3{animation:a2-r3 7s linear infinite .24s}
.a2-r4{animation:a2-r4 7s linear infinite .36s}
@media (prefers-reduced-motion:reduce){.a2-w1,.a2-w2,.a2-w3,.a2-w4,.a2-r1,.a2-r2,.a2-r3,.a2-r4{animation:none}}
</style>
<text class="a2-title" x="180" y="26">fan-out on write (push)</text>
<text class="a2-sub" x="180" y="46">1 tweet, N writes, cheap reads</text>
<rect class="a2-box" x="245" y="60" width="100" height="46" rx="6"/>
<rect class="a2-box" x="245" y="120" width="100" height="46" rx="6"/>
<rect class="a2-box" x="245" y="180" width="100" height="46" rx="6"/>
<rect class="a2-box" x="245" y="240" width="100" height="46" rx="6"/>
<text class="a2-lbl" x="295" y="88">timeline 1</text>
<text class="a2-lbl" x="295" y="148">timeline 2</text>
<text class="a2-lbl" x="295" y="208">timeline 3</text>
<text class="a2-lbl" x="295" y="268">timeline 4</text>
<rect class="a2-hub" x="20" y="160" width="92" height="60" rx="8"/>
<text class="a2-hublbl" x="66" y="186">author</text>
<text class="a2-hublbl" x="66" y="204">tweets</text>
<circle class="a2-dot a2-w1" cx="112" cy="190" r="7"/>
<circle class="a2-dot a2-w2" cx="112" cy="190" r="7"/>
<circle class="a2-dot a2-w3" cx="112" cy="190" r="7"/>
<circle class="a2-dot a2-w4" cx="112" cy="190" r="7"/>
<line class="a2-div" x1="380" y1="20" x2="380" y2="360"/>
<text class="a2-title" x="580" y="26">fan-out on read (pull)</text>
<text class="a2-sub" x="580" y="46">1 read, N fetches, cheap writes</text>
<rect class="a2-box" x="415" y="60" width="100" height="46" rx="6"/>
<rect class="a2-box" x="415" y="120" width="100" height="46" rx="6"/>
<rect class="a2-box" x="415" y="180" width="100" height="46" rx="6"/>
<rect class="a2-box" x="415" y="240" width="100" height="46" rx="6"/>
<text class="a2-lbl" x="465" y="88">followee 1</text>
<text class="a2-lbl" x="465" y="148">followee 2</text>
<text class="a2-lbl" x="465" y="208">followee 3</text>
<text class="a2-lbl" x="465" y="268">followee 4</text>
<rect class="a2-hub" x="648" y="160" width="92" height="60" rx="8"/>
<text class="a2-hublbl" x="694" y="186">your</text>
<text class="a2-hublbl" x="694" y="204">read</text>
<circle class="a2-dot a2-r1" cx="515" cy="83" r="7"/>
<circle class="a2-dot a2-r2" cx="515" cy="143" r="7"/>
<circle class="a2-dot a2-r3" cx="515" cy="203" r="7"/>
<circle class="a2-dot a2-r4" cx="515" cy="263" r="7"/>
</svg>
<figcaption>Left: one tweet is pushed into many follower timelines when it is written. Right: one read pulls and merges tweets from many followees at read time. The work is identical in volume; the strategies just move it between the write path and the read path.</figcaption>
</figure>

The animation makes the symmetry concrete. **Fan-out on read (pull)**, on the right, does nothing special on write — a tweet is stored once. The cost lands at read time: to build your timeline, the system fetches recent tweets from every account you follow and merges them. Writes are cheap; reads are expensive and get *worse* the more accounts you follow. **Fan-out on write (push)**, on the left, inverts this. The moment you tweet, the system looks up your followers and writes your tweet's id into each of their precomputed timelines. Reads become trivial — your timeline is already assembled, just read the list — but writes get expensive, and they get catastrophically expensive when one author has a great many followers.

Here is the pull model in code. It is short because the logic is simple — the cost is hidden in the per-followee fetch.

```python
import heapq

def timeline_pull(me, follows, tweet_store, limit=50):
    """Fan-out on read: fetch each followee's recent tweets, merge, return top N.

    follows[me]        -> list of followee ids
    tweet_store.recent -> recent tweets for one author, newest-first
    """
    # One fetch per followee. For a user following 2,000 accounts that is
    # 2,000 reads scattered across the tweet shards, every single page-load.
    streams = []
    for followee in follows[me]:
        streams.append(tweet_store.recent(followee, limit))

    # K-way merge by created_at descending; heapq.merge keeps it streaming.
    merged = heapq.merge(*streams, key=lambda t: t.created_at, reverse=True)
    out = []
    for tweet in merged:
        out.append(tweet)
        if len(out) == limit:
            break
    return out
```

The fatal line is `for followee in follows[me]`. The read cost scales with how many accounts you follow, and there is no cache that fixes it cleanly, because every user's timeline is a different merge of a different set. A power user following ten thousand accounts pays ten thousand fetches per refresh. Pull does not scale on the read path for exactly the users who use the product most.

So early Twitter went the other way.

## 3. Fan-out on write: the precomputed Redis timeline

**The rule that defined Twitter's timeline for years: precompute every active user's home timeline and keep it in memory, so a read is a single list fetch.** This is fan-out on write, and the materialized timelines live in a large **Redis** cluster managed by the **Timeline Service**.

![Fan-out on write: a new tweet flows through the fanout service to the follower list and into each follower's Redis timeline](/imgs/blogs/twitter-manhattan-and-timeline-at-scale-4.webp)

The push path is the pipeline in the figure. A new tweet arrives at the write API. A **fanout service** picks it up, asks the **SocialGraph** service (backed by FlockDB) for the author's follower list, and then, for each follower, appends the tweet's id to that follower's Redis timeline. Each home timeline is stored as a Redis list of tweet ids — not the full tweets, just the 64-bit Snowflake ids plus a little metadata — capped at around **800 entries** and replicated roughly **threefold** for fault tolerance. Capping the list is what keeps the memory footprint bounded: nobody scrolls back a thousand tweets in the live timeline, so you keep the most recent window and let the rest be reconstructed on demand.

```python
def fanout_on_write(tweet, social_graph, redis, TIMELINE_CAP=800):
    """Push a new tweet id into every active follower's Redis timeline list.

    Cost: O(followers) Redis writes per tweet. Fine for a normal account;
    a write storm for a celebrity (see the next section).
    """
    followers = social_graph.followers_of(tweet.author_id)  # FlockDB-backed

    # Real fan-out is pipelined and batched; the shape is one LPUSH + LTRIM
    # per follower so each home timeline stays newest-first and bounded.
    pipe = redis.pipeline()
    for follower_id in followers:
        key = f"home_timeline:{follower_id}"
        pipe.lpush(key, tweet.id)            # newest at the head
        pipe.ltrim(key, 0, TIMELINE_CAP - 1) # keep only the most recent window
    pipe.execute()
    return len(followers)
```

The read side is now almost embarrassingly cheap: one `LRANGE` against the user's list, then a *hydration* step that turns the tweet ids into full tweets (pulling the bodies from the tweet store, themselves heavily cached). A timeline read became a single in-memory list fetch plus a batched lookup, which is how Twitter could serve hundreds of thousands of timeline reads per second with single-digit-millisecond latency.

There is a second-order subtlety worth naming: Twitter only fanned out to *active* users. Materializing a timeline for an account that has not opened the app in months is wasted memory, so dormant users' timelines are evicted and **reconstructed on demand** the next time they log in — effectively a pull, run once, then cached. The system is already a hybrid of push and pull along the active/inactive axis before we even get to celebrities. That instinct — spend write amplification only where it pays off in read latency — is the same economics behind the entire [caching hierarchy](/blog/software-development/database-scaling/the-caching-hierarchy-at-scale).

A third subtlety bites anyone who copies this design: **the fan-out is asynchronous, so delivery is eventually consistent.** When you tweet, the write API acknowledges you immediately and the fanout service does its millions of list-appends in the background, over seconds. That is fine for your followers, who do not notice a few seconds of lag — but it creates a classic read-your-own-writes problem for *you*. If your own home timeline were assembled purely from the precomputed list, your fresh tweet might not appear for a beat, which feels like a bug to the person who just posted. The pragmatic fix is the same shape as the celebrity hybrid: splice your own most-recent tweets into your timeline at read time so you always see your own writes instantly, while everyone else sees them as fast as the fanout drains. The general lesson is that any precompute-on-write pipeline is asynchronous by nature, and the author of an event is exactly the party most likely to notice the propagation delay — so the author almost always needs a read-time fast path. Naming that failure mode up front saves a confusing incident later, when a product manager files "my own tweets are delayed" and the timeline looks perfectly healthy on every dashboard.

## 4. The celebrity hot key and the hybrid that saves it

Fan-out on write has one spectacular failure mode, and it is the most famous hot-key problem in the industry. Look again at the cost of `fanout_on_write`: it is `O(followers)` writes per tweet. For an account with two hundred followers, that is two hundred cheap Redis writes — nothing. For an account with a hundred million followers, a *single tweet* implies a hundred million Redis writes. That is a **write storm**: one logical action exploding into a flood of physical work that saturates the fanout fleet, delays delivery for everyone, and turns one popular account into a system-wide incident. This is the celebrity problem, and it is a textbook instance of the [hot-partition / hot-key pathology](/blog/software-development/database-scaling/hot-partitions-and-hot-rows) where load concentrates on a single key far beyond the average.

You cannot fix this by sharding harder, because the skew is in the *data*, not the layout — a celebrity is a single logical entity whose every action is high-fan-out by definition. So Twitter did the only thing that works: it stopped fanning out celebrities on write, and merged them in on read instead.

![The hybrid: the read request reads the precomputed Redis list and fetches celebrity tweets, then merges them by timestamp](/imgs/blogs/twitter-manhattan-and-timeline-at-scale-5.webp)

The hybrid is the diagram above, and it is the punchline of the whole timeline story. When you open the app, the Timeline Service does two things in parallel. It reads your **precomputed Redis timeline** — which contains the fanned-out tweets from all the *normal* accounts you follow — and it separately **fetches the recent tweets of the high-fan-out accounts you follow** at read time, the pull model applied to a tiny handful of authors. Then it **merges the two streams by timestamp**, hydrates the result, and returns your timeline. The celebrity's tweet was written exactly once and read by their followers on demand; everyone else's tweets were fanned out on write and read for free. The hot key is defused because the expensive author is handled by the cheap-write strategy, and the cheap authors are handled by the cheap-read strategy.

```python
def timeline_hybrid(me, follows, redis, tweet_store, is_high_fanout, limit=50):
    """Merge the precomputed Redis list (normal follows, pushed on write)
    with read-time fetches for high-fan-out 'celebrity' follows (pulled on read).
    """
    # 1. Cheap: the already-assembled list of normal-follow tweet ids.
    pushed_ids = redis.lrange(f"home_timeline:{me}", 0, limit * 4)
    pushed = tweet_store.multi_get(pushed_ids)          # batched hydration

    # 2. Bounded pull: only the handful of celebrities this user follows.
    celebs = [f for f in follows[me] if is_high_fanout(f)]
    pulled = []
    for celeb in celebs:                                # tiny N, not all follows
        pulled.extend(tweet_store.recent(celeb, limit))

    # 3. Merge by time, newest-first, and cut to the page size.
    merged = sorted(pushed + pulled, key=lambda t: t.created_at, reverse=True)
    return merged[:limit]
```

The threshold for "high fan-out" is a tuning knob: above some follower count an author is treated as pull, below it as push. Get the threshold wrong in the cheap direction and you reintroduce write storms; wrong in the expensive direction and ordinary reads start paying celebrity-merge costs. The art is that most accounts sit comfortably on the push side, so the read-time merge stays small for the vast majority of users.

Here is the full comparison, which is the single most useful artifact to carry out of this article:

| Dimension | Fan-out on read (pull) | Fan-out on write (push) | Hybrid (Twitter) |
| --- | --- | --- | --- |
| Write cost | O(1) — store the tweet once | O(followers) — a write per follower | O(normal followers); celebrities are O(1) |
| Read cost | O(follows) — fetch + merge all | O(1) — read one precomputed list | O(1) + O(celebrity follows) merge |
| Storage | Minimal — tweets only | High — every active timeline materialized | Materialize normal follows; skip celebrities |
| Celebrity author | Fine — no extra write cost | Catastrophic — write storm | Solved — celebrities are pulled, not pushed |
| Power follower (follows many) | Slow reads | Fast reads | Fast reads (small celebrity merge) |
| Freshness | Always current | Slight delivery lag during fan-out | Mixed; celebrity tweets are always live |
| Best when | Write-heavy, low read rate | Read-heavy, bounded fan-out | Real social graph with extreme skew |

The reason this table shows up in every system-design interview is that the tradeoff is *fundamental*, not Twitter-specific. Any feed — Instagram, LinkedIn, a notification system, an activity stream — faces the same write-vs-read placement decision and the same skew, and the hybrid is the general-purpose answer. Push by default for bounded fan-out, pull for the heavy hitters, merge at read time.

## 5. Manhattan: storage as an internal platform

The second half of Twitter's story is quieter but just as influential. By the early 2010s Twitter was not running one database; it was running *many*, across MySQL, Cassandra, and bespoke systems, one cluster per major use case. Each cluster had its own operational burden — capacity planning, on-call, version upgrades, tuning — and the JVM-based stores in particular brought the familiar tax of garbage-collection pauses that turn a p99 latency target into a coin flip. Every new product feature that needed storage meant standing up and operating yet another cluster. That does not scale, not technically and not organizationally.

So Twitter built **Manhattan**: a real-time, multi-tenant, distributed key-value store, announced in 2014, designed from the start to be a *platform* that internal teams consume rather than a database that each team operates. Its lineage is Amazon's Dynamo — consistent hashing, replication, tunable consistency — but its defining feature is multi-tenancy, the same shared-platform philosophy you can trace through the [Dynamo-and-Cassandra family of leaderless stores](/blog/software-development/database/cassandra-and-dynamodb-leaderless-deep-dive).

![Manhattan's layers: interfaces over a shared core, pluggable storage engines, and self-service multi-tenancy](/imgs/blogs/twitter-manhattan-and-timeline-at-scale-6.webp)

The architecture is the four-layer stack in the figure, and each layer carries a deliberate design decision:

- **Interfaces** expose a key/value API, handle request routing, and coordinate clients. This is the thin layer product teams actually talk to.
- **Core** is the distributed-systems engine room: replication, topology management (kept in ZooKeeper, off the critical read/write path), failure handling, conflict resolution, and consistency coordination. This is the part Twitter most wanted to own and control.
- **Pluggable storage engines** are the layer that makes Manhattan unusual. The engine that actually persists bytes is swappable: **seadb**, a read-only file format for data batch-produced by Hadoop; **sstable**, a log-structured-merge format for write-heavy workloads; and a **btree** engine for read-heavy, write-light data. A team picks the engine that matches its access pattern without leaving the platform — the same key/value API, a different substrate underneath.
- **Self-service multi-tenancy** is the operational headline. Many teams share one Manhattan deployment, each with its own datasets, **quotas**, and **rate limits**, isolated so one team's traffic spike cannot starve another's. Creating a new dataset is a self-service action measured in minutes, not a cluster-provisioning project measured in weeks.

The consistency model is tunable, defaulting to eventual consistency — Manhattan favors availability for almost all use cases — with **strong consistency available as opt-in** for the workloads that genuinely need it. Under the hood the strong path uses replicated logs to provide check-and-set semantics, both within a datacenter (`LOCAL_CAS`) and across datacenters (`GLOBAL_CAS`), so a developer who understands the tradeoff can ask for a linearizable compare-and-set when correctness demands it and take the cheaper eventual path everywhere else. This is the same tunable-consistency dial covered in [tunable consistency at scale](/blog/software-development/database-scaling/tunable-consistency-at-scale), exposed per operation.

A sketch of what the tunable-consistency API feels like from the caller's side makes the dial concrete:

```python
from enum import Enum

class Consistency(Enum):
    ONE      = 1   # fast, eventually consistent: ack from one replica
    QUORUM   = 2   # majority of replicas agree (R + W > N)
    ALL      = 3   # every replica acks; strongest, least available

class ManhattanClient:
    """Dynamo-lineage KV with per-operation consistency, sketched."""
    def __init__(self, ring, replication_factor=3):
        self.ring = ring                 # consistent-hash ring of replicas
        self.N = replication_factor

    def _replicas(self, key):
        return self.ring.preference_list(key, self.N)

    def get(self, dataset, key, level=Consistency.ONE):
        replicas = self._replicas((dataset, key))
        need = self._required_acks(level)
        # Read from `need` replicas; resolve conflicts by version (last-writer
        # -wins or vector clocks), and read-repair stale replicas in the
        # background. ONE returns the first answer; QUORUM waits for a majority.
        answers = self._read_from(replicas, need)
        return self._resolve(answers)

    def put(self, dataset, key, value, level=Consistency.QUORUM):
        replicas = self._replicas((dataset, key))
        need = self._required_acks(level)
        version = self._next_version(dataset, key)
        # Write to all replicas; return success once `need` of them ack.
        return self._write_to(replicas, value, version, need)

    def cas(self, dataset, key, expected, new_value):
        # Strong path: opt-in check-and-set via a replicated log, the moral
        # equivalent of Manhattan's LOCAL_CAS / GLOBAL_CAS.
        return self.put(dataset, key, new_value, level=Consistency.ALL) \
            if self.get(dataset, key, Consistency.QUORUM) == expected else False

    def _required_acks(self, level):
        return {Consistency.ONE: 1,
                Consistency.QUORUM: self.N // 2 + 1,
                Consistency.ALL: self.N}[level]
```

The shape of that API — pass a consistency level per call, let the quorum math do the rest — is the Dynamo idea made multi-tenant. What Manhattan added on top was the platform: secondary indexing, the pluggable engines, the quota system, and the self-service tooling that let any Twitter engineer get a production-grade dataset without becoming a database operator.

## 6. Build vs buy: why not just run Cassandra?

The obvious objection is the one every architecture review raises: Twitter was already running Cassandra, an open-source Dynamo descendant with most of these properties. Why build a new store instead of investing in the one you have? The answer is the most important strategic lesson in this article, and it is not "Cassandra is bad."

![Build vs buy: operate N independent clusters per team, or one multi-tenant Manhattan platform with self-service datasets](/imgs/blogs/twitter-manhattan-and-timeline-at-scale-7.webp)

The contrast in the figure is operational, not featural. On the left is the world Twitter wanted to leave: each team operating its own cluster, each cluster carrying its own on-call rotation, its own tuning, its own JVM garbage-collection pauses, and every new feature requiring a new cluster to be stood up. On the right is the platform model: one Manhattan, shared SRE and capacity, tuning expertise concentrated in one team, and a new dataset as a self-service action. Manhattan was a bet that the dominant cost at Twitter's scale was *operational*, and that the way to cut it was to consolidate many independently-operated stores into one platform with deep operability and multi-tenant isolation built in from day one.

| Question | Adopt Cassandra (buy) | Build Manhattan (build) |
| --- | --- | --- |
| Operational control | Bounded by upstream's roadmap and internals | Total — own the core, the engines, the failure handling |
| Multi-tenancy | Bolt-on; clusters tend to be per-use-case | First-class: quotas, rate limits, isolation by design |
| Storage engine | Fixed (the engine that ships) | Pluggable: seadb / sstable / btree per workload |
| GC and tail latency | JVM pauses are a known operational tax | Engine and runtime chosen and tuned in-house |
| Self-service | Provision a cluster | Create a dataset in minutes |
| Cost to get there | Low up front | High — a multi-year platform investment |
| Justified when | You are not operating dozens of clusters | You are a hyperscaler paying the operational tax daily |

That last row is the whole decision. Building your own datastore is almost always the wrong call — it is a multi-year investment with a long tail of operational maturity that you get for free by adopting something proven. Manhattan made sense only because Twitter was paying the multi-cluster operational tax across dozens of teams every single day, and because owning the storage layer let them tune away problems (GC behavior, multi-tenant isolation, engine selection) that an off-the-shelf system would have forced them to live with. The lesson is not "build your own database." The lesson is: **know which of your costs is dominant.** For most teams it is engineering time, so you buy. For a hyperscaler drowning in operational surface, the dominant cost flips, and building a platform that turns storage into a self-service product can be the cheaper path. Build-vs-buy is not a matter of capability; it is a matter of which bill is bigger.

## What the timeline and Manhattan taught the industry

The reason this case study endures is that each piece generalized into a pattern you will reach for in systems that have nothing to do with social media. Here are the lessons, each with the production shape it takes elsewhere.

### 1. The write-vs-read placement decision is universal

Any time one logical event must appear in many places, you choose between doing the work on write (materialize now) or on read (compute later). Notification systems, activity feeds, analytics roll-ups, search-index updates, denormalized read models in CQRS — all of them are the timeline problem wearing a different hat. The first question to ask of any feed-shaped feature is "push or pull?", and the honest answer for a skewed workload is almost always "both, split by a threshold."

### 2. Skew breaks the obvious design, and you plan for it explicitly

The celebrity is not an edge case to handle later; the celebrity is the case that determines the architecture. Power-law distributions are the norm in real systems — a few keys, users, or partitions carry orders of magnitude more load than the median. A design that is correct on average and catastrophic at the tail is a design that will page you. Identify the heavy hitters, give them a different code path, and tune the threshold between paths as a first-class operational knob.

### 3. Encode metadata into the key

Snowflake put time in the high bits so IDs sort by creation order; Instagram put the shard id in the middle bits so routing needs no lookup. Both turned a per-request question (when was this? where does it live?) into a bit-shift on data you already had. Whenever you find yourself doing a lookup to answer a question the key could have answered, consider widening the key. Self-describing identifiers are one of the highest-leverage primitives in distributed systems.

### 4. Precompute is a cache, and caches need a cap and an eviction story

The Redis home timeline is a materialized cache of a computation. Like every cache it needs a bound (the ~800-entry cap), a replication story (the threefold copies), and an eviction-and-rebuild path (dormant users' timelines are dropped and reconstructed on demand). If your precompute has no eviction plan, it is not a cache — it is an unbounded liability that grows until it falls over. The discipline is identical to the one in [the caching hierarchy](/blog/software-development/database-scaling/the-caching-hierarchy-at-scale).

### 5. Tunable consistency belongs at the operation, not the cluster

Manhattan let a caller pick eventual or strong per request, because the right consistency level is a property of the *use case*, not the database. A like-count can be eventually consistent; a username claim needs a compare-and-set. Forcing one global setting on a multi-tenant store wastes availability on the workloads that do not need strength and risks correctness on the ones that do. Expose the dial; default it to availability; let the few who need linearizability opt in.

### 6. At hyperscale, storage becomes a platform

The deepest organizational lesson is that beyond a certain size you stop thinking of a database as a thing teams operate and start thinking of it as a product teams consume. Multi-tenancy, quotas, self-service dataset creation, and isolation are not luxuries; they are what let one storage team support a hundred product teams without becoming a hundred on-call rotations. This is the same arc that produced internal platforms for compute, deploy, and observability across the industry — Manhattan is the storage instance of it.

### 7. The fail whale was a fan-out problem, not a database problem

Finally, the meta-lesson. For years people described Twitter's outages as "the database can't keep up," and the instinct was to find a faster database. But the real bottleneck was the *fan-out* — the multiplication between one tweet and millions of timeline writes — and no faster database fixes a multiplication factor. The fix was architectural: change *where* and *when* the multiplication happens. When a system is falling over, the highest-leverage question is rarely "what is slow?" It is "what is being multiplied, and can I move the multiplication?"

## When to reach for these patterns, and when not to

### Reach for fan-out on write when

- Reads vastly outnumber writes and read latency is your product's headline metric.
- Fan-out is *bounded* for the overwhelming majority of producers, so the average write amplification is manageable.
- You can afford the storage to materialize the precomputed views, and you have an eviction-and-rebuild story for cold consumers.
- You have a plan for the heavy tail — a pull or hybrid path for the high-fan-out minority.

### Reach for the hybrid when

- Your fan-out distribution is genuinely power-law: most producers are small, a few are enormous.
- You can cheaply classify producers as heavy or light and tune the threshold over time.
- A small read-time merge for the heavy follows is acceptable in exchange for killing write storms.

### Reach for building your own storage platform when

- You are operating so many independent clusters that the *operational* cost dominates every other cost in the system.
- Multi-tenancy, quota isolation, and self-service provisioning are first-order requirements, not nice-to-haves.
- You have the engineering depth to own a distributed datastore's failure modes for years, and the scale to amortize that investment across many teams.

### Skip these and keep it simple when

- Your fan-out is bounded and small — a single well-indexed query per read is fine; do not build a fan-out pipeline for a thousand users.
- Your read volume does not justify materializing anything; pull is simpler and always correct.
- You are not drowning in operational surface — adopt a proven datastore (Cassandra, DynamoDB, Postgres) and spend your engineering on the product. Building Manhattan when you have three teams and one cluster is the textbook over-engineering mistake.
- You can solve the skew with a cache or a read replica before you reach for a hybrid fan-out — the simplest thing that survives the tail is the right thing.

## Further reading

- Raffi Krikorian, "Timelines at Scale" (Twitter, QCon) — the canonical walkthrough of the fan-out architecture, the Redis timeline service, and the celebrity hybrid.
- "Manhattan, our real-time, multi-tenant distributed database for Twitter scale" (Twitter Engineering, 2014) — the primary source for the layered architecture, pluggable engines, and tunable consistency described here.
- "Announcing Snowflake" (Twitter Engineering) — the 64-bit ID format and the rationale for coordinator-free, time-sortable IDs.
- [How Instagram sharded Postgres: IDs that know their own shard](/blog/software-development/database-scaling/instagram-sharding-ids-in-postgres) — the sibling pattern of encoding routing metadata into the key.
- [Cassandra and DynamoDB: a deep dive into leaderless wide-column stores](/blog/software-development/database/cassandra-and-dynamodb-leaderless-deep-dive) — the Dynamo lineage Manhattan descends from.
- [Hot partitions and hot rows](/blog/software-development/database-scaling/hot-partitions-and-hot-rows) — the general theory behind the celebrity hot-key problem.
- [The caching hierarchy at scale](/blog/software-development/database-scaling/the-caching-hierarchy-at-scale) — the precompute-and-evict discipline the Redis timeline depends on.
