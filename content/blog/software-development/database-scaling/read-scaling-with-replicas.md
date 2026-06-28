---
title: "Read Scaling with Replicas: The First Horizontal Move (and Its Traps)"
date: "2026-06-28"
publishDate: "2026-06-28"
description: "Read replicas are the cheapest way to scale reads, but async replication quietly breaks your app's consistency assumptions — here is how to route reads, the traps that bite, and the token-based fixes that keep read-your-own-writes intact."
tags: ["database-scaling", "read-replicas", "replication-lag", "read-write-splitting", "consistency", "proxysql", "vitess", "gtid", "postgres", "mysql"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 32
---

The first time your database falls over, it is almost always reads. Not writes — reads. A homepage that runs eight queries per request, a search box that hits the same three tables on every keystroke, a dashboard that re-aggregates the last 30 days on every load. The primary is pinned at 95% CPU, p99 latency is climbing, and somebody in the incident channel says the words that launch a thousand outages: "can we just add a read replica?"

You can. Read replicas are the single highest-leverage, lowest-effort scaling move available to a relational database, and reaching for one is almost always the right call. But the sentence "can we just add a read replica" hides a lie of omission. Adding the replica is the easy part — a few minutes in the RDS console, or one `CREATE PUBLICATION` and a base backup. The hard part is that the moment a read can be served by a machine that is *not* the one that accepted the write, your application's silent assumption — "if I write something, then read it back, I see what I wrote" — stops being true. It was free for your entire career up to this point. It is not free anymore.

![Read scaling topology: one writer, many readers](/imgs/blogs/read-scaling-with-replicas-1.webp)

The diagram above is the mental model for the whole post: a read/write router sits between the app and the data tier, sends every write to a single primary, and fans reads out across a fleet of replicas. The primary streams its write-ahead log to each replica, but each replica is at a *different point* in replaying that log — replica 1 trails by a few milliseconds, replica 3 by nearly a second. Every trap in this article is a consequence of that one fact: **the replicas are behind, by different and unpredictable amounts.** The fixes are all ways of putting a guardrail around that gap without throwing away the throughput you bought the replicas for.

If you want the full mechanics of how a write gets from the primary's log to a replica's tables — physical vs logical replication, the failover playbook, the durability tradeoffs — that is its own deep-dive: [Database Replication: Synchronous, Asynchronous, Logical, and Physical](/blog/software-development/database/database-replication-sync-async-logical-physical). This post assumes that foundation and focuses on the layer above it: **how you route reads, what breaks when you do, and how to keep it from breaking.**

## Why "just add a replica" is different from what people expect

The mental gap is between two pictures of what a replica is. Engineers reaching for their first replica usually imagine "a second copy of the database" — same data, more capacity, problem solved. The reality is "a copy of the database as it looked a moment ago, where 'a moment' is anywhere from a millisecond to several minutes, and you don't get to know which." Here is the assumption-versus-reality table that the rest of the post unpacks:

| What people assume | What is actually true | Consequence if you ignore it |
| --- | --- | --- |
| A replica has the same data as the primary | A replica has the primary's data as of some past log position | Stale reads; read-your-writes violations |
| Adding replicas scales the whole database | Adding replicas scales *reads only* | Write ceiling unchanged; eventual sharding still needed |
| Reads are read-only, so they're safe to move | Reads carry consistency expectations that the app never declared | The app "randomly" shows old data after a save |
| All replicas are equivalent | Replicas lag by different, time-varying amounts | Monotonic-reads violations as you bounce between them |
| Replication lag is small and constant | Lag spikes under write bursts, long transactions, and DDL | The "rare" stale read becomes a flood during exactly the busy periods you scaled for |

The throughline: a read replica is not a redundancy feature you turned on, it is a **consistency tradeoff you opted into.** Asynchronous replication — the default, and the only kind that actually scales reads cheaply — buys throughput by *not* waiting for replicas to confirm. The primary commits a write the instant its own log is durable and ships the change to replicas in the background. That is exactly why it scales: the write path doesn't get slower as you add replicas. It is also exactly why replicas are stale: nobody is waiting for them.

> A read replica is not "a second database." It is a time machine stuck a little in the past, and you don't control the dial.

## 1. The replication spectrum, in one paragraph (then we move on)

There are three points on the synchrony spectrum, and your choice determines how bad the staleness traps are. **Asynchronous** replication is the cheap, scalable default: the primary acknowledges the client immediately and never blocks on replicas. Lag is unbounded in principle and routinely tens to hundreds of milliseconds in practice. **Synchronous** replication makes the primary wait for at least one replica to confirm the write is durable before acknowledging the client — zero data loss on failover, but every write now pays a network round trip to the replica, and if that replica stalls, your writes stall with it. **Semi-synchronous** (MySQL's `rpl_semi_sync`, Postgres's `synchronous_commit = on` with a `synchronous_standby_names` quorum) is the middle ground: wait for the replica to *receive* the log record (not necessarily apply it), so you bound data loss without paying the full apply latency.

The crucial subtlety, and the one that trips people: **even synchronous replication does not give you fresh reads.** Postgres synchronous commit waits for the standby to *flush the WAL to disk*, not to *replay it into the visible tables*. A synchronous replica can have your write safely on disk and still return the old row to a `SELECT`, because the replay (apply) is a separate, asynchronous step. So "I made replication synchronous, why am I still seeing stale reads?" is a real and common confusion. Synchrony protects durability; it does not, by itself, protect read freshness. That distinction is the foundation for every fix later in this post. For the architectural alternatives — multi-leader and leaderless designs where there isn't a single primary to read-around — see [Single-Leader, Multi-Leader, and Leaderless Replication](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless).

## 2. Read/write splitting: where to do it

You have decided to route writes to the primary and reads to replicas. *Where* in the stack does that decision get made? There are three real places, and the choice has lasting operational consequences.

![Where to split reads from writes](/imgs/blogs/read-scaling-with-replicas-3.webp)

The figure above is the decision in one frame: each placement trades how much it understands your queries against how much operational machinery you have to run. Walk the three rows.

### The application-layer router

The router lives in your code. Your data-access layer knows, for each query, whether it is a read or a write, which session issued it, and whether that session just wrote something — because *your code* has that context. That semantic richness is the app router's superpower: it can implement read-your-own-writes precisely, because it knows which user is reading and what they just wrote.

```python
# Application-layer read/write router. The app decides per-query where to go,
# because only the app knows the *intent* and the *session* behind each query.
import random
from contextlib import contextmanager

class RouterPool:
    def __init__(self, primary_dsn, replica_dsns):
        self.primary = connect(primary_dsn)               # one writer
        self.replicas = [connect(d) for d in replica_dsns] # N readers

    @contextmanager
    def for_write(self):
        # Writes always go to the primary. No ambiguity.
        yield self.primary

    @contextmanager
    def for_read(self, *, must_be_fresh=False):
        if must_be_fresh:
            # "Critical read": route to primary, accept the load cost.
            yield self.primary
        else:
            # Spread non-critical reads across the replica fleet.
            yield random.choice(self.replicas)
```

The cost is that this logic spreads through your codebase, and every service in a polyglot fleet has to reimplement it. Get the `must_be_fresh` flag wrong in one endpoint and you have a read-your-writes bug that only shows up under lag. The app router sees the most and centralizes the least.

### ORM read/write connections

Most ORMs now ship a "two connection" mode: a write connection bound to the primary and a read connection bound to a replica (or a round-robin pool of them). Django's `DATABASE_ROUTERS`, Rails' `connected_to(role: :reading)`, SQLAlchemy's `Session` with a `binds` map. These are cheap to adopt — you flip a config and decorate the occasional block — but the routing is *coarse*. The ORM typically routes by operation type (any `SELECT` goes to a replica) or by an explicit block, and it has no idea that the `SELECT` three lines after your `UPDATE` is reading back the row you just wrote.

```python
# SQLAlchemy 2.x: a Session that routes by *operation* using bind keys.
from sqlalchemy.orm import Session

class RoutingSession(Session):
    def get_bind(self, mapper=None, clause=None, **kw):
        if self._flushing or self.in_transaction_writes():
            return engines["primary"]      # writes + post-write reads
        return engines["replica"]          # default SELECTs -> replica
```

The subtle failure mode: an ORM that routes "all reads to replica" will happily send a read inside the same logical request as a write to a *different* connection on a *lagging replica*, and your read-your-writes guarantee evaporates. ORM splitting is the lowest-effort option and the easiest to get subtly wrong. Pair it with the session-stickiness fix in section 4 or you will ship the classic "I saved my profile and it shows the old name" bug.

### A proxy (ProxySQL, Vitess, PgBouncer-adjacent tooling)

A proxy sits on the wire between the app and the databases, parses SQL, and routes by rule. ProxySQL is the canonical MySQL example: you write query rules (`mysql_query_rules`) that send `^SELECT` to a replica hostgroup and everything else to the primary hostgroup, with regex carve-outs for `SELECT ... FOR UPDATE` (must go to primary) and `/* primary */` hint comments. Vitess goes further: it is a full sharding and routing layer that, among many other things, handles replica selection and read-after-write semantics. The proxy's advantages are that routing logic is centralized (every app, in every language, gets the same behavior for free) and that failover handling lives in one place — when the primary changes, the proxy re-points, and the apps never reconnect.

```sql
-- ProxySQL: route SELECTs to the replica hostgroup (20), writes to primary (10).
-- The order matters: more-specific rules first.
INSERT INTO mysql_query_rules (rule_id, active, match_pattern, destination_hostgroup, apply)
VALUES
  (10, 1, '^SELECT.*FOR UPDATE', 10, 1),     -- locking reads -> primary
  (20, 1, '^SELECT',             20, 1),     -- plain reads    -> replicas
  (30, 1, '.*',                  10, 1);     -- everything else -> primary
LOAD MYSQL QUERY RULES TO RUNTIME;
SAVE MYSQL QUERY RULES TO DISK;
```

The cost is real: a proxy is another network hop (typically +0.2–1 ms, sometimes more under load), another tier to deploy, monitor, scale, and keep highly available, and another component that can become the bottleneck or the single point of failure. A regex-based router also can't see session intent the way your app can — it can route a `SELECT` to a replica, but it doesn't know that *this particular* `SELECT` is reading back a row the same user wrote 20 ms ago, unless you teach it (GTID tracking, discussed in section 4). The decision is rarely either/or: large systems often run a proxy for the coarse split *and* keep app-level overrides for the reads that must be fresh.

## 3. The traps: how async replication breaks your app

This is the heart of the post. Every one of these is a real bug that has shipped to production at companies you have heard of, and every one is invisible in dev (where there is one database and zero lag) and intermittent in prod (where it depends on lag, which depends on load). Watch the write propagate:

<figure class="blog-anim">
<svg viewBox="0 0 760 360" role="img" aria-label="A write commits on the primary then streams to three replicas, each applying it after a different delay; a read hitting the most-lagged replica sees the old value" style="width:100%;height:auto;max-width:860px">
<title>Replication lag: a write reaches replicas at different times</title>
<style>
.rl-node{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.rl-prim{fill:var(--accent,#6366f1);opacity:.16;stroke:var(--accent,#6366f1);stroke-width:2}
.rl-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.rl-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.rl-wire{stroke:var(--border,#d1d5db);stroke-width:2;fill:none}
.rl-pkt{fill:var(--accent,#6366f1)}
.rl-applied{fill:var(--accent,#6366f1);opacity:0}
.rl-stale{fill:#ef4444;opacity:0;font:700 13px ui-sans-serif,system-ui;text-anchor:middle}
@keyframes rl-commit{0%,4%{opacity:0}10%,100%{opacity:1}}
@keyframes rl-flow1{0%,10%{transform:translate(0,0);opacity:0}14%{opacity:1}24%,100%{transform:translate(0,-70px);opacity:0}}
@keyframes rl-flow2{0%,10%{transform:translate(0,0);opacity:0}14%{opacity:1}40%,100%{transform:translate(0,70px);opacity:0}}
@keyframes rl-flow3{0%,10%{transform:translate(0,0);opacity:0}14%{opacity:1}78%,100%{transform:translate(0,210px);opacity:0}}
@keyframes rl-ap1{0%,24%{opacity:0}30%,100%{opacity:.85}}
@keyframes rl-ap2{0%,40%{opacity:0}46%,100%{opacity:.85}}
@keyframes rl-ap3{0%,78%{opacity:0}84%,100%{opacity:.85}}
@keyframes rl-readstale{0%,50%{opacity:0}56%,74%{opacity:1}80%,100%{opacity:0}}
.rl-c{animation:rl-commit 9s ease-in-out infinite}
.rl-f1{animation:rl-flow1 9s ease-in-out infinite}
.rl-f2{animation:rl-flow2 9s ease-in-out infinite}
.rl-f3{animation:rl-flow3 9s ease-in-out infinite}
.rl-a1{animation:rl-ap1 9s ease-in-out infinite}
.rl-a2{animation:rl-ap2 9s ease-in-out infinite}
.rl-a3{animation:rl-ap3 9s ease-in-out infinite}
.rl-rs{animation:rl-readstale 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.rl-c,.rl-f1,.rl-f2,.rl-f3,.rl-a1,.rl-a2,.rl-a3,.rl-rs{animation:none}.rl-applied{opacity:.85}.rl-pkt{opacity:0}}
</style>
<rect class="rl-node" x="40" y="150" width="180" height="70" rx="10"/>
<rect class="rl-prim rl-c" x="44" y="154" width="172" height="62" rx="8"/>
<text class="rl-lbl" x="130" y="182">Primary</text>
<text class="rl-sub" x="130" y="202">write committed @ t0</text>
<path class="rl-wire" d="M220 185 H320 V70 H420"/>
<path class="rl-wire" d="M220 185 H320 V255 H420"/>
<path class="rl-wire" d="M220 185 H320 V325 H420"/>
<rect class="rl-node" x="420" y="40" width="200" height="64" rx="10"/>
<rect class="rl-applied rl-a1" x="424" y="44" width="192" height="56" rx="8"/>
<text class="rl-lbl" x="520" y="68">Replica 1</text>
<text class="rl-sub" x="520" y="88">applied @ t0 + 5 ms</text>
<rect class="rl-node" x="420" y="225" width="200" height="64" rx="10"/>
<rect class="rl-applied rl-a2" x="424" y="229" width="192" height="56" rx="8"/>
<text class="rl-lbl" x="520" y="253">Replica 2</text>
<text class="rl-sub" x="520" y="273">applied @ t0 + 40 ms</text>
<rect class="rl-node" x="420" y="296" width="200" height="64" rx="10"/>
<rect class="rl-applied rl-a3" x="424" y="300" width="192" height="56" rx="8"/>
<text class="rl-lbl" x="520" y="320">Replica 3</text>
<text class="rl-sub" x="520" y="340">applied @ t0 + 900 ms</text>
<circle class="rl-pkt rl-f1" cx="230" cy="185" r="8"/>
<circle class="rl-pkt rl-f2" cx="230" cy="185" r="8"/>
<circle class="rl-pkt rl-f3" cx="230" cy="185" r="8"/>
<text class="rl-stale rl-rs" x="680" y="320">read here = STALE</text>
<text class="rl-stale rl-rs" x="680" y="338">old value</text>
</svg>
<figcaption>One write commits on the primary at t0, then streams outward; replica 1 catches up in 5 ms but replica 3 trails by ~900 ms, so a read routed there during the gap returns the old value.</figcaption>
</figure>

### Trap 1: stale reads

The base case. A read lands on a replica that hasn't yet applied a write that already committed on the primary, and returns old data. This is not a bug in the database — it is the contract of asynchronous replication working exactly as designed. The "bug" is that your application code was written assuming linearizable reads and you moved the read off the primary without changing that assumption.

Stale reads are often *fine*. If a user sees a like-count that is 40 ms out of date, nobody cares. The danger is that "often fine" lulls you into treating *all* reads as stale-tolerant, and then the one read that is not — the read that decides whether to charge a card, grant access, or show a user their own just-submitted form — quietly serves old data and produces a real defect. The discipline is to classify reads, not to ban replicas.

### Trap 2: read-your-own-writes violation

This is the most common, most user-visible incarnation of trap 1, and it deserves its own name because it is the one your users will report. The pattern: a user performs an action that writes (saves a profile, posts a comment, updates a setting), is immediately redirected to a page that reads that same data, and sees the *old* value — because the redirect-read raced the replication and won.

![How read-your-own-writes breaks on a lagging replica](/imgs/blogs/read-scaling-with-replicas-4.webp)

The timeline above is the exact sequence. The user saves at t=0, the write commits on the primary at t=8 ms, the user's browser reloads at t=30 ms, and the reload's read is routed to a replica at t=31 ms — but that replica doesn't apply the write until t=120 ms. For the 89-millisecond window in between, the replica honestly returns the row the user just overwrote. From the user's perspective, the save *didn't work*. They will save again. They will email support. They will leave a one-star review titled "loses my data." None of it is true; the data is safe on the primary. The read just went to the wrong place.

The reason this is so insidious is that **read-your-own-writes is a session guarantee, not a global one.** The system doesn't need to be globally consistent — it only needs each user to see their own writes. That narrower requirement is what makes the fixes tractable (you only have to track one session's last write, not the global frontier), and it is the formal model worth understanding before you build the fix: see [Consistency Models: From Linearizability to Eventual Consistency](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) for where read-your-writes and monotonic-reads sit on the spectrum.

### Trap 3: monotonic-reads violation (time travels backward)

Now add a second twist: not all replicas lag the same. A user loads a page (routed to replica 1, lag 5 ms) and sees a comment thread with 12 messages. They refresh (routed to replica 3, lag 900 ms) and see 10 messages — two comments *disappeared*. They refresh again (back to replica 1) and the two comments reappear. From the user's seat, time is flickering backward and forward.

This is a **monotonic-reads violation**: a guarantee that once you have seen a value, you never see an older one. It is broken not by lag per se, but by *bouncing between replicas of different lag*. A single replica, even a very lagged one, never violates monotonic reads on its own — it only ever moves forward. The violation is an artifact of your load balancer treating the fleet as interchangeable when it is not. This is why "just round-robin the replicas" is a trap: round-robin maximizes the chance that consecutive reads from one session land on replicas at different positions.

### Trap 4: lag spikes when you can least afford them

The quiet assumption behind "replication lag is small" is that it is *constant*. It is not. Replication lag is a queue: the primary produces log records at the write rate, and each replica consumes them at its single-threaded (or limited-parallelism) apply rate. The instant production exceeds consumption, the queue grows, and lag balloons. The triggers are exactly the events that correlate with your busy periods:

- **Write bursts.** A batch job, a flash sale, a fan-out write (one action that updates thousands of rows). The primary absorbs it across many connections; the replica's apply is far more serialized, so it falls behind.
- **Long transactions on the primary.** A single big `UPDATE ... WHERE` that touches millions of rows commits as one giant log record the replica must apply atomically, stalling apply for everything behind it.
- **Schema changes (DDL).** An `ALTER TABLE` that rewrites a table replicates as a rewrite the replica must also perform; on MySQL with single-threaded apply this can freeze a replica for minutes.
- **Replica resource contention.** The replica is also serving reads. A heavy analytical query can starve the apply thread of I/O, and now your read traffic is *causing* the lag that produces stale reads for other read traffic. A nasty feedback loop.

The cruel irony: lag is worst during write bursts, and write bursts are exactly when the most users are reading. The "one-in-a-million stale read" you dismissed in code review becomes a flood precisely during the traffic spike you added replicas to survive.

### Trap 5: failover thundering reads

Replicas serve two purposes: scaling reads, and being failover candidates. When the primary dies and a replica is promoted, two things happen at once. First, every replica that was reading from the old primary must re-point to the new one, and there is a window where reads either fail or pile onto whatever is still up. Second — and worse — the newly promoted primary is now serving *all* writes plus whatever reads were on it, and the surviving replicas may briefly be unavailable (re-syncing from the new primary's timeline). Read traffic that was comfortably spread across five replicas can suddenly thunder onto two, or onto the primary itself, right when the system is most fragile. A read fleet sized for steady state is undersized for the seconds after a failover; this is a capacity-planning trap as much as a consistency one.

## 4. The fixes: routing with a guardrail

Every fix below has the same shape: keep serving most reads from replicas (that's the throughput you paid for), but put a *checked* guardrail around the reads that can't tolerate staleness. The naive approach routes blindly; the disciplined approach carries a token.

![Blind replica reads vs token-guarded routing](/imgs/blogs/read-scaling-with-replicas-5.webp)

The before/after above is the core idea: on the left, you pick any replica and hope; on the right, the read carries a *position token* describing how fresh it needs the data to be, and the router only serves it from a replica that has caught up to that position. Let's build that up from the cheapest fix to the most precise.

### Fix A: sticky session to the primary (the write-then-read window)

The cheapest correct fix for read-your-own-writes. When a session performs a write, mark it, and for a short window afterward (long enough to cover typical replication lag — say 1–5 seconds), route *that session's* reads to the primary. After the window, the session falls back to replicas. This is the "if you just wrote, read from where you wrote" heuristic, and it fixes the overwhelming majority of read-your-writes complaints with a few lines of code.

```python
import time

class StickyRouter(RouterPool):
    WINDOW_S = 3.0   # cover typical lag + a safety margin

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._wrote_at = {}   # session_id -> monotonic timestamp of last write

    def note_write(self, session_id):
        self._wrote_at[session_id] = time.monotonic()

    @contextmanager
    def for_read(self, session_id):
        last = self._wrote_at.get(session_id, 0)
        if time.monotonic() - last < self.WINDOW_S:
            yield self.primary           # recently wrote -> read from primary
        else:
            yield random.choice(self.replicas)
```

The weakness is that the window is a guess. Too short and a lag spike pushes real lag past it, and the bug returns. Too long and you're sending more read load to the primary than you need to — which is the load you were trying to escape. It's a blunt instrument, but it's a real fix and it's the right first move because it requires no replica-position plumbing.

### Fix B: consistency tokens (GTID / LSN) — the precise fix

The grown-up version replaces the *time* window with an *actual position*. Every write advances the database's log position: in MySQL this is the GTID (global transaction identifier); in Postgres it is the LSN (log sequence number). The trick is:

1. When a session writes, capture the resulting log position and store it in the session (a cookie, the session store, wherever you keep per-user state).
2. When that session later reads, the router checks each candidate replica's *applied* position. If a replica has applied past the session's token, it is provably fresh enough — serve the read there. If not, fall back to the primary (or to a less-lagged replica).

![Consistency-token routing decision](/imgs/blogs/read-scaling-with-replicas-6.webp)

The figure above is the decision: compare the session's token against the replica's applied position; caught up means read the replica and offload the primary; behind means fall back to the primary, which is always fresh by definition. The difference from the sticky-session fix is that this is *exact* — you serve from a replica the instant it is actually fresh enough, not after an arbitrary timeout, so you reclaim the replica capacity as soon as it's safe and you never serve a stale read even if lag spikes.

```python
# Postgres LSN-based read-your-writes. On write, capture the commit LSN; on read,
# pick a replica that has *replayed* past it, else fall back to the primary.
def capture_write_lsn(conn):
    # pg_current_wal_lsn() is the primary's current write position after commit.
    return conn.execute("SELECT pg_current_wal_lsn()").scalar()

def replica_caught_up(replica, token_lsn):
    # pg_last_wal_replay_lsn() is how far this replica has *applied* the WAL.
    applied = replica.execute("SELECT pg_last_wal_replay_lsn()").scalar()
    # LSNs compare with the built-in operators once cast to pg_lsn.
    return replica.execute(
        "SELECT %s::pg_lsn >= %s::pg_lsn", (applied, token_lsn)
    ).scalar()

def read_with_token(pool, token_lsn):
    # Prefer the freshest replica that has caught up; fall back to primary.
    fresh = [r for r in pool.replicas if replica_caught_up(r, token_lsn)]
    return random.choice(fresh) if fresh else pool.primary
```

The MySQL equivalent uses `WAIT_FOR_EXECUTED_GTID_SET(gtid, timeout)`, which blocks until the replica has executed up to the given GTID (or the timeout fires, at which point you fall back to the primary). ProxySQL implements exactly this with its GTID-consistent-reads feature: the app passes the GTID it last wrote (returned in the OK packet via `session_track_gtids`), and ProxySQL routes the read to a replica that has applied it, or to the primary if none has. This is the same idea as Fix B, pushed down into the proxy so every app gets it for free.

### Fix C: bounded-staleness routing

Sometimes you don't need *your* writes specifically — you just need data that is "fresh enough." Bounded-staleness routing says: only serve a read from a replica whose lag is under some threshold (say 1 second). It's a fleet-health policy rather than a per-session guarantee: the router continuously measures each replica's lag and pulls any replica that exceeds the bound out of the read rotation until it catches up. This simultaneously fixes the "lagged replica in rotation" problem (trap 4) and gives you a coarse freshness floor without per-session token plumbing. It does *not* give you read-your-own-writes (a replica can be under the 1 s bound and still not have *your* 200 ms-old write), so it's complementary to Fix B, not a substitute.

### Fix D: pin a session to one replica (monotonic reads, cheaply)

To fix monotonic reads (trap 3) without going to the primary, the move is to make a given session *sticky to a single replica* for its duration. As long as a session always reads from the same replica, that replica only ever moves forward, so the session never sees time run backward. Hash the session ID to a replica and you're done. The cost is uneven load (a hot user pins load to one replica) and a re-pin event if that replica dies — but it converts the monotonic-reads guarantee from "needs global coordination" to "needs a stable hash," which is a trade most systems happily take.

### Fix E: critical reads go to the primary, explicitly

The simplest rule of all, and the one to reach for when in doubt: **a small set of reads are correctness-critical, and those go to the primary, full stop.** The read that checks a balance before a transfer. The read that verifies a coupon hasn't already been redeemed. The read that confirms a user owns the resource they're about to delete. These are a tiny fraction of total read volume, so sending them to the primary costs almost nothing in capacity, and it eliminates an entire class of subtle bugs. The mistake is the opposite default — treating all reads as critical and sending everything to the primary, which just recreates the bottleneck you were escaping. Classify deliberately: most reads are stale-tolerant; name the few that aren't.

| Fix | What it guarantees | Cost | Reach for it when |
| --- | --- | --- | --- |
| A. Sticky-to-primary window | Read-your-writes (approximately) | Extra primary read load; guessed window | You need a fix today and lag is usually small |
| B. GTID/LSN consistency token | Read-your-writes (exactly) | Token plumbing through the session | Read-your-writes matters and you can carry a token |
| C. Bounded-staleness routing | "Fresh within N seconds" fleet-wide | Lag monitoring per replica | You want a freshness floor and to evict lagged replicas |
| D. Pin session to one replica | Monotonic reads | Uneven load; re-pin on failure | Users complain about flickering / disappearing data |
| E. Critical reads to primary | Linearizable for named reads | Tiny primary load for those reads | A specific read decides money/access/correctness |

These compose. A mature system runs bounded-staleness routing (C) to keep lagged replicas out of rotation, pins each session to one healthy replica (D) for monotonic reads, carries a GTID/LSN token (B) for the endpoints that need read-your-writes, and hard-routes the handful of correctness-critical reads (E) to the primary. The sticky-window (A) is the training-wheels version you ship first and graduate from.

## 5. Measuring lag (you cannot manage what you don't see)

Every fix above depends on knowing each replica's lag and position. Both major engines expose it directly.

```sql
-- Postgres: on a replica, how far behind the primary is it (in time and bytes)?
-- Run ON THE REPLICA. NULL replay timestamp means no WAL applied yet / idle.
SELECT
  now() - pg_last_xact_replay_timestamp()        AS replay_lag,        -- time behind
  pg_last_wal_receive_lsn()                       AS received_lsn,      -- got this far
  pg_last_wal_replay_lsn()                        AS applied_lsn,       -- applied this far
  pg_wal_lsn_diff(
    pg_last_wal_receive_lsn(),
    pg_last_wal_replay_lsn())                      AS apply_backlog_bytes; -- received but not yet applied
```

The two LSNs matter separately: `received_lsn` is how much WAL the replica has *gotten* over the network, and `applied_lsn` is how much it has *replayed* into visible tables. A large gap between them means the network is fine but the apply thread can't keep up — the replica has your data on disk but isn't showing it yet. That is the exact failure synchronous replication does *not* save you from. For MySQL:

```sql
-- MySQL/MariaDB: run on the replica. Seconds_Behind_Source is the headline number,
-- but it lies during stalls (it measures the *current* event's age, not the backlog).
SHOW REPLICA STATUS\G
-- Look at:
--   Seconds_Behind_Source     : ~lag in seconds (0 = caught up; NULL = broken)
--   Replica_IO_Running  = Yes  : network thread alive (receiving binlog)
--   Replica_SQL_Running = Yes  : apply thread alive (replaying binlog)
--   Retrieved_Gtid_Set vs Executed_Gtid_Set : received vs applied, the GTID analog of the LSN gap
```

A senior gotcha worth memorizing: **`Seconds_Behind_Source` reads 0 right up until a stall, then jumps.** It measures the timestamp delta of the event currently being applied, so during a long-running apply (a giant transaction) it can sit at a small number while the real backlog grows, then spike. Trust `Retrieved_Gtid_Set` minus `Executed_Gtid_Set` (the actual unapplied backlog) over the headline number when diagnosing apply stalls. Alert on the *backlog*, not the *seconds*, and you'll catch lag spikes before your users do.

## 6. The capacity truth: replicas scale reads, not writes

Here is the ceiling that every replica-based scaling story eventually hits, and the reason this post is only the *first* horizontal move in a longer series.

![Replicas multiply read capacity, not write capacity](/imgs/blogs/read-scaling-with-replicas-7.webp)

The figure above is the whole argument in one picture. Add a replica and your read capacity goes up — roughly linearly, since reads can be served from any node. But that single write at the top? **Every node in the system must apply it.** The primary applies it, and then *every replica replays the same write.* A replica does not absorb some of the write load; it *duplicates* the write load. If your write rate is approaching what a single machine can apply, adding replicas makes that *worse*, not better — you've added more machines that each must keep up with the same write stream, and the slowest one bounds how fresh your fleet can be.

So the write ceiling is fixed by what one machine can apply, and replicas cannot lift it. This is not a tuning problem; it is structural. When writes are the bottleneck — not reads — you have exhausted what replication can do, and the next move is to split the data itself so that different writes go to different machines. That is **sharding**, and it's a fundamentally harder, more invasive change than adding a replica. The decision of *when* you've hit this wall, and which scaling move comes next, is the subject of [The Database Scaling Decision Tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree). The one-line version: replicas buy you read scale and a little runway; they do not buy you out of sharding.

> Replicas are a read amplifier, not a write amplifier. Every replica is one more machine that must replay every write you make — so reads scale and writes don't.

## Case studies from production

### 1. Wikipedia / MediaWiki ChronologyProtector

Wikipedia runs one of the largest read-heavy MySQL deployments in the world: a single primary per database section, a large fleet of replicas, and the overwhelming majority of page views served from replicas. For most reads — anonymous users viewing articles — slight staleness is invisible and acceptable. But a logged-in editor who saves an edit and is immediately shown the page *must* see their edit, or the wiki feels broken.

MediaWiki solves this with **ChronologyProtector**, a mechanism that is essentially Fix B done carefully. When a request performs writes, MediaWiki records the replication positions of the databases it wrote to and stashes them (keyed to the user's session) in a fast shared store. On the user's *next* request, before issuing reads, it waits for the relevant replicas to catch up to those stored positions — or, if catching up would take too long, routes those reads to the primary. The result: each user sees a monotonically advancing, read-your-own-writes-consistent view of their own activity, while the vast majority of read traffic still hits replicas. The lesson is the architecture of the fix: it is per-session, position-based, and degrades to "read the primary" only when necessary — exactly the guardrail-not-a-wall pattern.

### 2. GitHub's functionally-partitioned MySQL and replica routing

GitHub has run MySQL as its primary datastore for the entire history of the company, scaling it through functional partitioning (splitting different feature domains into different database clusters) long before sharding any single domain. Each cluster is a primary with multiple replicas, and the application routes reads to replicas aggressively to keep the primaries free for writes. Over the years GitHub has invested heavily in the routing and consistency layer — building and adopting tooling (their Vitess adoption for the largest clusters is well documented) to manage read/write splitting, replica health, and failover.

The instructive part of GitHub's story for this post is the *operational* maturity around lag: replicas that fall behind a threshold are automatically pulled from the read pool (Fix C, bounded staleness, at scale), and read-after-write correctness for user-facing flows is handled explicitly rather than left to chance. The broader lesson: functional partitioning plus read replicas got a famously large product surprisingly far before any single dataset needed true sharding — which validates this post's thesis that replicas are the right *first* move, while also showing that the consistency plumbing around them is where the real engineering lives.

### 3. The "save profile shows old data" bug (every company, once)

The most universal incident in this article is the one nobody writes a postmortem about because it's "just a frontend glitch." A team adds read replicas, flips the ORM to route `SELECT`s to a replica, and ships. Within a day, support tickets trickle in: users save a setting, the page reloads, and the *old* setting is shown. Refresh again and it's correct. QA can't reproduce it (their database has no replica). The first hypothesis is always a frontend caching bug, and an engineer spends a day in the browser devtools before someone notices the read went to a replica.

The root cause is trap 2, and the fix is Fix A or B. What makes it a *recurring* industry incident rather than a one-time mistake is that the ORM made the dangerous default (all reads to replica) one config line away, with no warning that it had just broken a guarantee the app silently depended on. The lesson: when you turn on read/write splitting, the *first* thing to wire up is read-your-own-writes for post-write redirects — before you ship, not after the tickets arrive.

### 4. The analytics query that caused stale reads for everyone

A data team points a dashboard at a read replica — sensible, keep analytics off the primary. The dashboard runs a heavy multi-join aggregation over the last 90 days every few minutes. The replica's apply thread competes with that query for I/O and CPU, loses, and falls behind. Now the replica is both lagging *and* serving the application's user-facing reads, so users start seeing stale data — caused by an *analytics* query, on a replica that was supposed to be insulating the primary.

This is trap 4's feedback loop in the wild. The fix is isolation: give analytics its own dedicated replica (or an OLAP store fed by replication), kept out of the application's read pool, so a heavy report can lag *that* replica without affecting user reads. The lesson: "reads go to replicas" is too coarse. Application reads and analytical reads have completely different lag tolerances and load shapes, and putting them on the same replica makes the well-behaved traffic pay for the badly-behaved traffic.

### 5. The failover that doubled read latency

A mid-size service runs one primary and four replicas, with reads spread evenly. The primary's host hits a hardware fault and the orchestrator promotes replica 1. Promotion takes a few seconds; during it, replica 1 stops serving reads (it's becoming the primary), the other three replicas briefly disconnect and reconnect to the new primary's timeline, and for a handful of seconds nearly all read traffic that doesn't error piles onto whatever is reachable. p99 read latency triples, some requests time out, and the on-call gets paged for "elevated errors" that resolve themselves in under a minute.

This is trap 5. Nothing was *broken* — failover did its job and data was safe — but the read fleet was sized for steady state, not for the transient where one node is mid-promotion and the rest are reconnecting. The lesson is capacity headroom: size the read fleet so that losing one or two nodes (to failure *or* promotion) doesn't saturate the rest, and make sure the router fails reads over to the primary gracefully during the gap rather than hammering a replica that's busy re-syncing.

### 6. The cross-region replica that was always "a little behind"

A team adds a read replica in a second region to serve local users with low latency. It works — until users in that region start reporting read-your-writes failures far more often than users near the primary. The cause: the cross-region replica's lag floor isn't milliseconds, it's the *speed of light plus the WAN*, often 80–150 ms of baseline lag before any load. A 3-second sticky-window (Fix A) tuned for same-region lag is generous there; a token-based fix (Fix B) is fine because it checks actual position — but a naive "round-robin all replicas including the remote one" guarantees that a meaningful slice of post-write reads land on a replica that is structurally ~100 ms behind.

The lesson: a replica's lag profile is a property of *where it is*, not just how loaded it is, and a fleet of replicas at different distances is a fleet with different freshness floors. Geo-distributed reads are a real and powerful technique, but they make every consistency trap in this post sharper, and they're the bridge to a whole other topic (multi-region writes, which async single-leader replication cannot give you at all).

## When to reach for read replicas / when not to

### Reach for read replicas when

- Your workload is **read-heavy** (the common case for web apps — often 90%+ reads) and the primary's CPU or I/O is dominated by `SELECT`s, not writes.
- The data you're reading **tolerates slight staleness** for most queries — feeds, listings, search results, dashboards, profile views.
- You can **classify the few reads that don't** tolerate staleness and route them to the primary or behind a consistency token.
- You want **failover candidates** anyway — replicas double as the standbys you promote, so the cost is partly already justified by availability.
- You need **read locality** in another region and can accept that those replicas lag by the WAN baseline.

### Skip read replicas (or don't expect them to help) when

- **Writes are your bottleneck**, not reads. Replicas replay every write, so they cannot lift the write ceiling — you need sharding. Adding replicas here makes lag worse, not throughput better.
- **Every read must be linearizable** (some financial ledgers, inventory-at-zero, authorization checks). If you can't tolerate *any* staleness anywhere, you're routing everything to the primary and the replicas only earn their keep as failover standbys.
- **Your lag is already unbounded** because the primary is write-saturated. Replicas of a primary that can't keep up will never catch up; fix the write path first.
- The team **won't invest in the consistency plumbing.** A read replica without read-your-own-writes handling is a stale-data bug generator. If you can only afford to flip the ORM config and walk away, you'll ship trap 2 — at least add the sticky-window first.

The honest summary: read replicas are the correct first horizontal scaling move for the overwhelming majority of relational workloads, and they're cheap enough that the question is rarely "should we" but "how do we route correctly." Get the routing and the read-your-own-writes guardrail right, measure lag relentlessly, and remember the ceiling — replicas buy you read scale and runway, not an escape from the write wall. When you hit that wall, the next move is to split the data, and that's where the [scaling decision tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree) takes over.

## Further reading

- [Database Replication: Synchronous, Asynchronous, Logical, and Physical](/blog/software-development/database/database-replication-sync-async-logical-physical) — the mechanics underneath this post: how a write actually gets from the primary's log to a replica's tables, and the failover playbook.
- [Single-Leader, Multi-Leader, and Leaderless Replication](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless) — when there isn't a single primary to read-around, and how write conflicts get resolved.
- [Consistency Models: From Linearizability to Eventual Consistency](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) — the formal home of read-your-writes, monotonic reads, and the rest of the session guarantees this post relies on.
- [The Database Scaling Decision Tree](/blog/software-development/database-scaling/the-database-scaling-decision-tree) — where read replicas sit in the larger sequence of scaling moves, and how to know when you've outgrown them.
