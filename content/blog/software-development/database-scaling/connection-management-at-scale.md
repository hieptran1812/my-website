---
title: "Connection Management at Scale: Pools, Proxies, and the Death Spiral"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "Databases die from connection exhaustion far more often than from raw query load — here is the pooling, proxying, and sizing math that keeps a few backends serving thousands of clients."
tags: ["database-scaling", "connection-pooling", "pgbouncer", "postgres", "transaction-pooling", "littles-law", "rds-proxy", "proxysql", "multi-tenancy", "performance"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 31
---

The outage that taught me to respect connections did not start with a slow query. It started with a deploy. A perfectly healthy fleet rolled out a new build, every pod reconnected to Postgres at once, and within ninety seconds the database that had been humming along at 8% CPU was pinned at 100%, p99 latency had gone from 4 ms to 9 seconds, and the application was returning `FATAL: sorry, too many clients already`. Nobody had changed a query. Nobody had added load. The database fell over because too many processes asked to talk to it at the same time, and the act of *managing* those conversations cost more than doing the work they were asking for.

That is the uncomfortable thesis of this post: **at scale, databases die from connection exhaustion far more often than from raw query load, and "more connections" is almost never the fix — it is usually the accelerant.** Connection management — pooling, proxying, limits, and sizing — is not plumbing you set up once and forget. It is a first-class scaling concern, on the same tier as indexing and sharding, and it has its own math, its own failure modes, and its own catastrophic feedback loop.

![Many client connections funnel through a server-side pooler onto a handful of expensive database backends](/imgs/blogs/connection-management-at-scale-1.webp)

The diagram above is the mental model for the entire post. Thousands of client connections — application pods each running their own pool, plus a swarm of serverless functions — do not get their own backend on the database. They funnel through a *waist*: a server-side pooler that multiplexes all of them onto a small, fixed set of real backend processes. Everything in this article is a tour of why that waist has to exist, what each backend behind it actually costs, how the system collapses when you remove the waist and let connections grow unbounded, and how to size every layer with arithmetic instead of hope. We will simulate the death spiral in Python, write a PgBouncer transaction-mode config that turns 10,000 clients into 20 backends, and close with named production incidents where someone learned this the hard way.

## Why connection management is different

The instinct that gets teams into trouble is treating a database connection like an HTTP connection: cheap, stateless, fungible, "just open more." A database connection is none of those things. Here is the assumption-versus-reality table that frames the rest of the post:

| Common assumption | Naive view | Reality |
| --- | --- | --- |
| "A connection is cheap, open as many as you need" | Like a socket — kilobytes, microseconds | A Postgres backend is a whole OS process: ~5–10 MB RAM, a fork, and a permanent slot in shared memory |
| "More connections = more throughput" | Concurrency scales linearly with connections | Throughput peaks at a small pool and *collapses* past it; the knee is near core count |
| "The pool should be big to avoid waiting" | Bigger pool = fewer queued requests | A bigger pool moves the queue from the app (cheap, observable) into the database scheduler (expensive, invisible) |
| "Connections are independent" | One slow client can't hurt others | Active connections share cores, locks, and buffers; a spike in one starves all |
| "Serverless scales, so its database access scales" | Functions are stateless and elastic | 1,000 concurrent functions open 1,000 connections; the database `max_connections` is the hard wall they hit |
| "Transaction pooling is free multiplexing" | Just flip a config flag | It silently breaks prepared statements, `SET`, advisory locks, and `LISTEN/NOTIFY` |

Every row is a real failure mode, and every one is preventable with the math below. Let me take them in order: first what a connection costs and why there is an optimal pool size, then the death spiral that punishes you for ignoring it, then the layered defense — client pools, the serverless explosion, server-side poolers — and finally sizing and multi-tenant fairness.

> A senior rule of thumb: a database connection is one of the most expensive objects your application allocates. You would never open 5,000 file handles "just in case." Treat connections with the same suspicion.

## 1. Why a connection is expensive

**Senior rule: the optimal number of connections is close to your core count, not your request count — every connection past the point where the CPU can actually run it is pure overhead.**

Start with what a connection physically *is*. In PostgreSQL, every client connection is backed by a dedicated operating-system process, forked from the `postmaster` at connect time. That process is not free:

- It costs **roughly 5–10 MB of resident memory** for its own copy of the catalog cache, relation cache, prepared-statement plans, and per-backend buffers — before it does any work. `work_mem` can multiply that: a single complex query with several sorts and hash joins can allocate `work_mem` *per operation*, so an 8 MB idle backend can balloon to hundreds of MB under a heavy query.
- The **fork plus backend startup costs milliseconds** of CPU and triggers copy-on-write page setup. A connection storm — every pod reconnecting after a deploy — is a fork storm.
- Every backend takes a **permanent slot in the `ProcArray`** in shared memory. Operations that scan that array (snapshot acquisition, `VACUUM` visibility checks) get more expensive as the slot count grows, so even *idle* connections impose a tax on *active* ones.

MySQL uses a thread-per-connection model instead of a process, so each connection is cheaper at the low end — a thread and its stack rather than a full process. But the ceiling is the same: a thread still consumes a `thread_stack` allocation plus per-connection buffers (`sort_buffer_size`, `join_buffer_size`, `read_buffer_size`), and the scheduler still has to time-slice all *runnable* threads across a finite number of cores. Whether the unit is a process or a thread, the brutal fact is identical: **when the number of connections that want to run at once exceeds the number of cores that can run them, the machine spends an increasing fraction of its time context-switching and contending for locks instead of executing queries.**

This is why there is an *optimal* pool size, and why it is small. The widely cited HikariCP analysis ("About Pool Sizing", drawing on a PostgreSQL benchmark) makes the point bluntly: a connection pool of around a dozen outperformed a pool of several thousand on the same hardware, and shrinking an over-sized pool routinely *increases* throughput while *dropping* latency. The rough formula people reach for is:

$$\text{connections} \approx (\text{cores} \times 2) + \text{effective spindles}$$

The `cores × 2` term captures the fact that while one query is briefly blocked (a cache miss, a lock wait), another can use the CPU, so you want a little more concurrency than you have cores. The `effective spindles` term is the storage parallelism — how many independent I/O operations the disks can service at once. On a 16-core box with fast NVMe, that lands you somewhere in the low dozens, *not* the hundreds. The figure below is the shape every load test in this space produces:

![Throughput rises to a peak at a small pool size, then collapses as context-switching and lock contention dominate](/imgs/blogs/connection-management-at-scale-2.webp)

Throughput climbs as you add connections, peaks where active connections roughly match what the cores and disks can sustain, and then *falls off a cliff* as every additional connection adds context-switch and lock-contention cost without adding real parallelism. The numbers on the bars are illustrative — the exact peak depends on your CPU-to-I/O ratio — but the shape is universal. The dangerous part is the right half of that chart: past the knee, adding connections makes you slower, and the natural reaction to "the database is slow" is to add connections.

Here is a tiny simulation that reproduces the shape. It is deliberately simple — service time inflates once active connections exceed a sweet spot — but it captures the mechanism:

```python
# death_spiral.py — why "open more connections" makes a contended DB slower.
CORES = 16
BASE_SERVICE_MS = 5.0          # uncontended time to run one query
SWEET_SPOT = 4 * CORES         # ~64 active for a mixed CPU + I/O workload

def service_ms(active: int) -> float:
    """Past the sweet spot, context-switching and lock waits inflate every query."""
    if active <= SWEET_SPOT:
        return BASE_SERVICE_MS
    overload = active / SWEET_SPOT
    return BASE_SERVICE_MS * overload ** 1.6

def goodput(active: int) -> float:
    """Only ~SWEET_SPOT connections make real progress; the rest just add cost."""
    working = min(active, SWEET_SPOT)
    return working / (service_ms(active) / 1000.0)   # queries/sec

for active in [8, 16, 32, 64, 128, 256, 512]:
    print(f"{active:>4} conns -> {goodput(active):8.0f} q/s "
          f"(service {service_ms(active):6.1f} ms)")
```

Run it and you get goodput rising to a peak near 64 active connections, then falling: `128` connections do worse than `64`, `512` do worse than `128`, and the service time per query balloons from 5 ms to over 130 ms. The database is doing *more work managing connections* than serving them.

### Second-order: idle connections are not free

The seductive mistake is "idle connections are fine, only active ones cost." Mostly true for CPU, dangerously false for everything else. Each idle Postgres backend still holds its 5–10 MB, still occupies a `ProcArray` slot that snapshot acquisition must scan, and — worst of all — an idle-*in-transaction* connection holds locks and pins the transaction horizon, blocking `VACUUM` from reclaiming dead tuples. A pool of 2,000 mostly-idle connections is not "free headroom"; it is 10–20 GB of RAM you cannot use for the buffer cache, plus a steadily growing cost on every snapshot. Idle connections are a slow leak, not free insurance.

## 2. The death spiral

**Senior rule: when the database is slow, the *last* thing you should do is let the application open more connections — that is exactly the input that turns a slowdown into an outage.**

The most important thing to understand about connection management is that the failure mode is not linear — it is a **positive feedback loop**. Watch it run:

<figure class="blog-anim">
<svg viewBox="0 0 860 460" role="img" aria-label="A feedback loop: rising latency makes the app open more connections, which pushes active backends past the core count, causing context-switch and lock contention that slows the database further; a side gauge of active connections climbs to a thrash zone and the loop collapses" style="width:100%;height:auto;max-width:880px">
<title>The connection death spiral</title>
<style>
.ds-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#cbd5e1);stroke-width:1.5}
.ds-lbl{font:600 15px ui-sans-serif,system-ui,sans-serif;fill:var(--text-primary,#1f2937);text-anchor:middle}
.ds-edge{fill:none;stroke:var(--text-secondary,#6b7280);stroke-width:2.5}
.ds-pulse{fill:var(--accent,#6366f1)}
.ds-track{fill:var(--surface,#f3f4f6);stroke:var(--border,#cbd5e1);stroke-width:1.5}
.ds-fill{fill:#ef4444;transform-box:fill-box;transform-origin:bottom}
.ds-mark{stroke:#ef4444;stroke-width:2;stroke-dasharray:6 5}
.ds-small{font:600 13px ui-sans-serif,system-ui,sans-serif;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.ds-collapse{font:800 26px ui-sans-serif,system-ui,sans-serif;fill:#ef4444;text-anchor:middle}
@keyframes ds-spin{0%{transform:translate(320px,57px)}25%{transform:translate(535px,215px)}50%{transform:translate(320px,383px)}75%{transform:translate(115px,215px)}100%{transform:translate(320px,57px)}}
@keyframes ds-climb{0%{transform:scaleY(.1)}80%{transform:scaleY(1)}100%{transform:scaleY(1)}}
@keyframes ds-flash{0%,55%{opacity:0}78%,100%{opacity:1}}
.ds-pulse{animation:ds-spin 9s ease-in-out infinite}
.ds-fill{animation:ds-climb 9s ease-in-out infinite}
.ds-collapse{animation:ds-flash 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.ds-pulse{animation:none;transform:translate(320px,57px)}.ds-fill{animation:none;transform:scaleY(.75)}.ds-collapse{animation:none;opacity:1}}
</style>
<defs>
<marker id="ds-ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M0 0L10 5L0 10z" fill="var(--text-secondary,#6b7280)"/>
</marker>
</defs>
<rect class="ds-box" x="220" y="24" width="200" height="66" rx="9"/>
<rect class="ds-box" x="440" y="180" width="190" height="70" rx="9"/>
<rect class="ds-box" x="220" y="350" width="200" height="66" rx="9"/>
<rect class="ds-box" x="20" y="180" width="190" height="70" rx="9"/>
<path class="ds-edge" d="M422 70 Q520 104 532 176" marker-end="url(#ds-ah)"/>
<path class="ds-edge" d="M512 252 Q470 332 432 364" marker-end="url(#ds-ah)"/>
<path class="ds-edge" d="M218 384 Q150 344 118 254" marker-end="url(#ds-ah)"/>
<path class="ds-edge" d="M118 176 Q150 104 218 76" marker-end="url(#ds-ah)"/>
<text class="ds-lbl" x="320" y="62">latency rises</text>
<text class="ds-lbl" x="535" y="208">app opens</text>
<text class="ds-lbl" x="535" y="228">more conns</text>
<text class="ds-lbl" x="320" y="388">active &gt;&gt; cores</text>
<text class="ds-lbl" x="115" y="208">context-switch</text>
<text class="ds-lbl" x="115" y="228">+ lock storm</text>
<text class="ds-collapse" x="320" y="224">COLLAPSE</text>
<circle class="ds-pulse" cx="0" cy="0" r="13"/>
<rect class="ds-track" x="724" y="44" width="52" height="352" rx="6"/>
<rect class="ds-fill" x="724" y="44" width="52" height="352" rx="6"/>
<line class="ds-mark" x1="716" y1="78" x2="784" y2="78"/>
<text class="ds-small" x="750" y="34">thrash</text>
<text class="ds-small" x="750" y="420">active conns</text>
</svg>
<figcaption>The death spiral: each pass around the loop pushes latency and active connections higher, until the backends thrash on context-switching and the database collapses under connections it was supposed to be serving.</figcaption>
</figure>

Trace the loop. Something causes latency to tick up — a slow query, a checkpoint, a brief lock. Because queries now hold their connection longer, the application's connection pool drains faster: requests that used to find a free connection now wait, the pool exhausts, and the framework — or worse, the autoscaler — responds by opening *more* connections (or spinning up more pods, each with its own pool). Now there are more active connections than cores, so the database spends more time context-switching and contending for locks, which makes *every* query slower, which makes connections drain *even faster*, which opens *even more* connections. The gauge on the right climbs into the thrash zone and the system collapses — not because the work was too hard, but because the coordination overhead ate the machine.

The cruelty of the spiral is that every local decision is rational. The pool "should" grow when it is exhausted. The fleet "should" scale out when latency is high. Each actor, optimizing locally, feeds the fire. The only thing that breaks the loop is a **hard ceiling on connections** that does *not* grow under pressure — a bounded pool that makes excess requests *queue in the application* (where waiting is cheap, observable, and time-limited) instead of *queue in the database scheduler* (where waiting means thrashing every other query).

```python
# The fix is not "more connections" — it is a hard cap plus a fast-fail timeout.
# HikariCP example: a small, fixed pool. Excess load queues in the app and
# times out quickly, instead of being shoved onto the database as new backends.
from dataclasses import dataclass

@dataclass
class PoolConfig:
    maximum_pool_size: int = 20       # NEVER scale this up under load
    minimum_idle: int = 20            # keep it flat; no ramp that becomes a storm
    connection_timeout_ms: int = 2000 # fail fast; do not let callers pile up forever
    max_lifetime_ms: int = 1_800_000  # recycle connections to dodge driver/server leaks

# The key property: under overload, the database sees AT MOST 20 active backends
# from this app instance, no matter how many requests are queued upstream.
```

The single most valuable graph during an incident is "active connections vs. time" overlaid on latency. If they rise *together*, you are in the spiral, and the intervention is counterintuitive: **cap connections lower, not higher.** I have watched a database recover from `9s` p99 to `5ms` p99 in the time it took to roll out a pool size cut from 100 to 25 per pod — the work was never the problem; the connections were.

## 3. Client-side pools and the serverless explosion

**Senior rule: a client-side pool protects the application from connection setup cost; a server-side pooler protects the database from the application. At scale you need both, and the second one is not optional.**

The first line of defense is the **client-side pool** that every serious application already runs: HikariCP (JVM), `pgbouncer`-aware drivers, `database/sql`'s pool in Go, SQLAlchemy's `QueuePool`, `pg-pool` in Node, `asyncpg` pools in Python. These keep a fixed set of connections open and hand them to request handlers, so you pay the connect/fork cost once at startup instead of per request. This is necessary and well-covered ground — the companion post on [database connection pooling](/blog/software-development/database/database-connection-pooling) goes deep on sizing a single app's pool, `maxLifetime`, leak detection, and validation queries. Read that for the within-one-app view.

The problem client-side pools cannot solve is **fan-out**. A client-side pool bounds the connections from *one process*. It does nothing about the total across a fleet. If you run 50 pods and each holds a pool of 20, that is 1,000 connections to the database whether or not they are busy. Scale to 200 pods and you are at 4,000 — well past any sane `max_connections`. And then there is serverless, where the model breaks entirely:

![Without a pooler, 1000 serverless functions open 1000 connections and exhaust max_connections; with a pooler they multiplex onto a few dozen backend connections](/imgs/blogs/connection-management-at-scale-4.webp)

A serverless function is a single-request execution context. It cannot meaningfully maintain a long-lived pool, because the platform freezes and thaws it between invocations and runs each concurrent request in its own isolated instance. So 1,000 concurrent invocations means 1,000 independent connection attempts, each opening one connection, all at once. There is no client-side pool that can save you, because there is no shared client process to pool *in*. The function fleet's connection count equals its concurrency, and its concurrency is elastic and spiky by design. The database's `max_connections` is a fixed wall. The two are on a collision course.

This is precisely why AWS shipped **RDS Proxy** — a managed, server-side connection pooler — and why its launch messaging led with serverless. The fix is structural: put a pooler *in front of* the database that holds a small, fixed set of real backend connections and lets the swarm of short-lived clients borrow from it. The function connects to the pooler (cheap, fast, no `max_connections` pressure), runs its transaction, and the pooler reuses the same handful of backends across thousands of clients. The serverless explosion becomes a non-event. The same logic applies to a large pod fleet: past a few dozen app instances, a server-side pooler stops being an optimization and becomes the only thing standing between a traffic spike and `too many clients already`.

## 4. Server-side poolers: PgBouncer, ProxySQL, RDS Proxy

**Senior rule: the multiplexing ratio of a server-side pooler is the single number that decides whether you can scale your client count independently of your database — and transaction mode is what makes that ratio large.**

A server-side pooler sits between clients and the database and maintains its own pool of backend connections. The magic is in *how aggressively* it can reuse each backend, which is governed by the **pooling mode**. PgBouncer offers three:

- **Session mode**: a client gets a backend for the entire life of its client connection, and the backend returns to the pool only when the client disconnects. Multiplexing ratio ≈ 1:1. This is the safe, boring mode — it behaves exactly like talking to Postgres directly. It buys you almost no multiplexing, just connection reuse across reconnects.
- **Transaction mode**: a client gets a backend *only for the duration of a transaction*. The instant the transaction commits or rolls back, the backend goes back to the pool for the next client. This is where the big multiplexing ratios come from, because most clients are idle (thinking, rendering, waiting on the network) the vast majority of the time.
- **Statement mode**: the backend returns to the pool after *every single statement*. Maximum multiplexing, but it forbids multi-statement transactions entirely. Niche — mostly for read-only, autocommit workloads.

Transaction mode is the one that earns its keep at scale. Here is what it actually looks like over time:

![In transaction mode the pooler lends a backend only for one transaction, so five clients reuse two real backend connections as time advances](/imgs/blogs/connection-management-at-scale-5.webp)

Five clients (A through E) share two real backend connections. Backend #1 runs A's transaction, is released back to the pool, runs C's, then A's again, then E's. Backend #2 runs B twice, is freed, then runs D twice. At no instant do all five clients need a backend simultaneously, because each holds one only for the milliseconds its transaction is executing. The released-to-pool gaps are the whole trick: a client connection that is "connected" but between transactions consumes *zero* backend connections.

### The multiplexing math

The arithmetic is simple and it is the reason this works. If `C` clients each spend a fraction `d` of their time inside a transaction (the *duty cycle*), then the expected number of backends you need is:

$$\text{backends} \approx C \times d$$

A web app where each client opens a transaction for 4 ms out of every 200 ms request has a duty cycle of roughly 0.02. Ten thousand such clients need about `10,000 × 0.02 = 200` backends at the *mean* — and far fewer if their transactions are staggered, which they are. In practice a `default_pool_size` of 20–40 backends comfortably serves thousands of clients. The pooler's `max_client_conn` (how many clients it accepts) can be set to 10,000 while `default_pool_size` (backends per pool) stays at 20 — a **500:1 multiplexing ratio**. Here is the config that does it:

```ini
; /etc/pgbouncer/pgbouncer.ini — transaction-mode pooling for thousands of clients
[databases]
appdb = host=10.0.1.10 port=5432 dbname=appdb

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type    = scram-sha-256
auth_file    = /etc/pgbouncer/userlist.txt

; transaction mode: a backend is assigned to a client only for one transaction,
; then returned to the pool. This is what lets a few dozen backends serve thousands.
pool_mode = transaction

; how many *server* (backend) connections per (database, user) pair
default_pool_size  = 20
; hard ceiling on backend connections across ALL pools — this is your real load on PG
max_db_connections = 40

; how many *client* connections PgBouncer will accept. Set this huge — that is the point.
max_client_conn = 10000

; run on every backend before handing it to the next client, so no session state leaks
server_reset_query = DISCARD ALL

; PgBouncer >= 1.21 can track protocol-level prepared statements in transaction mode
max_prepared_statements = 200
```

The two numbers that matter are `max_client_conn = 10000` (the clients you can accept) and `default_pool_size = 20` (the backends you actually impose on Postgres). The ratio between them is your multiplexing factor, and it is the lever that lets your client count grow independently of your database's hard `max_connections` ceiling.

For MySQL, the equivalent is **ProxySQL**, which adds query routing and read/write splitting on top of connection multiplexing via its `mysql-connection_pool` and `max_connections` per-hostgroup settings. **RDS Proxy** is AWS's managed option for both Postgres and MySQL — fully managed, IAM-integrated, and failover-aware, at the cost of a per-vCPU-hour fee and a small latency add (typically single-digit milliseconds). The table below is the decision matrix:

| Pooler / mode | Multiplexing ratio | Session-state safe? | Managed? | Reach for it when |
| --- | --- | --- | --- | --- |
| Client-side pool (HikariCP, etc.) | 1:1 within one process | Yes | No | Every app, always — but it does not bound the fleet total |
| PgBouncer — session mode | ~1:1 | Yes (full session) | No | You need reuse but rely on prepared statements, `SET`, `LISTEN/NOTIFY` |
| PgBouncer — transaction mode | 100:1 to 1000:1 | No (transaction-scoped only) | No | Large fleets and serverless on Postgres; the workhorse at scale |
| PgBouncer — statement mode | Highest | No (no multi-stmt txns) | No | Read-only autocommit workloads only |
| ProxySQL (MySQL) | High + query routing | Partial | No | MySQL fleets that also want read/write splitting |
| RDS Proxy | High (transaction pooling) | No (transaction-scoped) | Yes | Serverless / Lambda on RDS; teams who do not want to operate a pooler |

### Second-order: the pooler is a single point of failure and a stampede risk

Putting a pooler in the path concentrates risk. If PgBouncer restarts, every client reconnects to it at once — a thundering herd against the pooler, which then opens its backends in a burst. Run the pooler with a process supervisor and `so_reuseport`, or run several PgBouncer instances behind a TCP load balancer, and keep `server_reset_query = DISCARD ALL` so a recycled backend never leaks one client's session state into the next. The pooler also adds a network hop; co-locate it on the database host (or as a sidecar) to keep that hop sub-millisecond.

## 5. What transaction pooling takes away

**Senior rule: transaction pooling multiplexes by guaranteeing that no state survives a transaction — so anything in your stack that depends on session-lived state will break, often silently, often only under load.**

The multiplexing ratio is not free. It comes from a single guarantee — *a backend belongs to a client only for one transaction* — and that guarantee is exactly what shatters any feature whose state is supposed to outlive a transaction. This is the part teams discover in production, usually via a confusing intermittent error that only appears once traffic is high enough that two clients land on the same backend.

![A matrix of session features against pooling modes: session mode keeps everything; transaction mode breaks prepared statements, SET, advisory locks, LISTEN/NOTIFY, and temp tables](/imgs/blogs/connection-management-at-scale-6.webp)

Walking the casualties of transaction mode:

- **Prepared statements.** A server-side prepared statement is named and lives on a specific backend. In transaction mode, the next transaction may land on a *different* backend that has never heard of that statement — you get `prepared statement "S_1" does not exist`, or, if a name collides, `prepared statement "S_1" already exists`. This is the single most common transaction-pooling bug. JDBC, `asyncpg`, and Rails all default to server-side prepared statements. The fixes: disable them (`prepareThreshold=0` on JDBC, `statement_cache_size=0` on `asyncpg`), or run PgBouncer ≥ 1.21 with `max_prepared_statements` set so the pooler tracks and replays them.
- **`SET` / session GUCs.** `SET statement_timeout`, `SET search_path`, `SET TIME ZONE` — anything set with session scope evaporates when the backend is returned and may not apply to your next transaction. Use the transaction-scoped `SET LOCAL` inside an explicit transaction instead, or set defaults at the role/database level.
- **Session-level advisory locks.** `pg_advisory_lock()` (session scope) is held until explicitly unlocked or the *session* ends — but in transaction mode "the session" is a fiction shared across clients, so the lock leaks or protects the wrong client. Use `pg_advisory_xact_lock()`, which is transaction-scoped and released at commit.
- **`LISTEN` / `NOTIFY`.** `LISTEN` registers interest for the life of a session. Across a multiplexed backend pool, your `LISTEN` lands on one backend and the `NOTIFY` you are waiting for fires on another — you simply never get the notification. Pub/sub over transaction-pooled connections does not work; use a dedicated session-mode connection or a real message queue.
- **Temporary tables.** `CREATE TEMP TABLE` is session-scoped; the next transaction on a different backend cannot see it. Anything that builds a temp table and then queries it across transactions breaks.

The only things that survive transaction mode cleanly are operations that begin and end inside a single transaction. That is most OLTP traffic — which is why transaction mode works so well for typical web apps — but the moment a feature reaches for session state, you have to either route it to a session-mode pool or redesign it. A common production pattern is to run **two PgBouncer pools**: a big transaction-mode pool on port 6432 for normal traffic, and a small session-mode pool on port 6433 for the handful of jobs that need `LISTEN/NOTIFY`, advisory locks, or long-lived prepared statements.

## 6. Sizing the pool: Little's Law and the connection budget

**Senior rule: you do not guess a pool size, you compute it — concurrency equals throughput times latency, and that number, plus headroom, is your pool.**

The question "how big should the pool be?" has an actual answer, and it comes from **Little's Law**. For any stable system, the average number of items in the system equals the average arrival rate times the average time each item spends in the system:

$$L = \lambda \times W$$

For connections, `L` is the average number of in-flight connections (the pool size you need), `λ` is the rate of database operations per second, and `W` is the average time each operation holds a connection. This is the same lever the [capacity planning post](/blog/software-development/database-scaling/capacity-planning-for-databases) uses to model the connection budget, applied at the pool level:

```python
# pool_sizing.py — Little's Law turns a latency target into a pool size.
import math

peak_qps          = 4000      # DB operations/sec at peak
mean_hold_time_s  = 0.004     # 4 ms average time a query holds its connection

# Little's Law: average in-flight concurrency L = arrival rate λ × hold time W
concurrency = peak_qps * mean_hold_time_s          # = 16 connections, on average

# Headroom for variance: p99 hold time is much larger than the mean, and traffic
# is bursty. Size for the tail, not the average — but stay well under the knee.
headroom_factor = 1.5
pool_total = math.ceil(concurrency * headroom_factor)   # = 24 backend connections

print(f"avg concurrency = {concurrency:.0f}, sized pool = {pool_total}")
```

Sixteen connections of average concurrency, sized up to 24 for tail latency and bursts. That is the *total* backend pool — and notice it is in the low dozens, exactly where the throughput curve from Section 1 peaks. If you have 200 app pods, you do not give each a pool of 24; you point them all at a pooler whose `default_pool_size` is 24 and let it multiplex. The per-pod client pool can be small (say 5) because the pooler, not the pod, owns the real backends.

The other half of sizing is recognizing that `max_connections` on the database is **not** all yours. It is a budget with mandatory deductions:

![The connection budget: max_connections minus reserved superuser, replication, and headroom slots leaves the usable application budget](/imgs/blogs/connection-management-at-scale-7.webp)

Reading the budget top to bottom: `max_connections = 200` is the hard ceiling, but Postgres reserves `superuser_reserved_connections` (default 3) so an admin can always get in to fix things, `walsender` processes for replication and backups consume their own slots (reserve ~10 if you run replicas and `pg_basebackup`), and you must keep genuine headroom for a failover event or a spike — when a replica is promoted, the herd of clients reconnecting must fit. After those deductions, the budget actually available to your application pools is `200 − 3 − 10 − 27 headroom = 160`. Size your pooler's `max_db_connections` against the *160*, not the 200, or your next failover will fail to connect at the worst possible moment.

```sql
-- What is actually holding connections right now? Run this during an incident.
-- 'idle in transaction' is the silent killer: connections holding locks doing nothing.
SELECT state, count(*) AS conns,
       max(now() - state_change) AS longest_in_state
FROM pg_stat_activity
WHERE backend_type = 'client backend'
GROUP BY state
ORDER BY conns DESC;
```

### Second-order: `idle in transaction` is the leak that fills the budget

The fastest way to exhaust the budget you just computed is an application that opens a transaction, makes an external call (an HTTP request, a slow render) *inside* it, and only then commits. Every one of those connections is pinned in `idle in transaction`, holding a backend and its locks while doing nothing. Set `idle_in_transaction_session_timeout` (e.g. `30s`) on the database so Postgres reaps these, and audit your ORM for transactions that wrap network I/O. A handful of leaked `idle in transaction` connections per pod, times a few hundred pods, is how a 160-connection budget vanishes overnight.

## 7. Multi-tenant fairness

**Senior rule: a shared pool with no per-tenant limits is a shared pool waiting for its noisiest tenant to take it all — fairness has to be enforced, it does not emerge.**

Once many services or tenants share one database (or one pooler), connection management becomes a *fairness* problem, not just a sizing problem. The default behavior of a single shared pool is first-come-first-served, which means the service that opens connections most aggressively — usually a batch job, a reporting query, or a runaway retry loop — can consume the entire pool and starve every interactive service behind it. One bad deploy in the analytics service should not take down checkout.

![Per-tenant connection caps: a shared pooler partitions its budget so one service's spike stays within its own cap and cannot starve the others](/imgs/blogs/connection-management-at-scale-8.webp)

The fix is **per-tenant (or per-service) connection caps**. PgBouncer supports this directly: define a separate pool per `(database, user)` pair with its own `pool_size`, so each service gets a guaranteed-and-bounded slice. A shared pooler with `max 100` server connections might allocate 40 to checkout, 30 to search, 20 to reporting, and keep 10 in reserve. When the reporting service's nightly job spikes, it saturates *its own* 20-connection cap and then queues — checkout and search never even notice, because their 40 and 30 slots were never reporting's to take.

```ini
; Per-service pools enforce fairness: each (db, user) gets a bounded slice.
[databases]
appdb_checkout  = host=10.0.1.10 dbname=appdb user=checkout  pool_size=40
appdb_search    = host=10.0.1.10 dbname=appdb user=search    pool_size=30
appdb_reporting = host=10.0.1.10 dbname=appdb user=reporting pool_size=20

[pgbouncer]
pool_mode          = transaction
max_db_connections = 100   ; total ceiling; the sum of pools + reserve stays under it
```

The principle generalizes: any shared resource pool in a multi-tenant system needs per-tenant limits, or its weakest-behaved tenant defines its reliability. This is the same isolation logic that drives sharding decisions in [when one database is no longer enough](/blog/software-development/database-scaling/when-one-database-is-not-enough) — sometimes the right answer is not just a per-tenant connection cap but a per-tenant database, so a noisy neighbor cannot touch your pool at all.

## Case studies from production

### 1. The deploy that reconnected the whole fleet

The opening story. A rolling deploy of ~150 pods, each with a client pool of `min_idle = 20`, restarted in a tight window. All 3,000 connections re-established within seconds — a fork storm against Postgres, which pinned a core just spawning backends, which slowed the very queries the new pods were running, which made their pools open even more. The database hit `max_connections` and started rejecting. The root cause was not the deploy; it was `min_idle` equal to `max_pool_size`, so every pod aggressively re-opened its full pool immediately. The fix was a server-side pooler (so the fleet's reconnects hit PgBouncer, not Postgres) plus a smaller per-pod pool. The lesson: a deploy is a connection event, and an un-pooled fleet turns every deploy into a stampede.

### 2. Lambda and the connection wall

A team moved an API to AWS Lambda for elasticity and immediately started seeing `FATAL: remaining connection slots are reserved` during traffic peaks. The functions scaled to 1,000 concurrent invocations; each opened its own connection; `max_connections` was 500. No amount of client-side pooling helped, because each invocation was its own isolated process. This is the exact scenario RDS Proxy was built for: dropping a transaction-mode proxy in front let 1,000 functions multiplex onto ~40 backends, and the errors vanished. The lesson: serverless concurrency *is* your connection count unless a server-side pooler sits between it and the database.

### 3. The "bigger pool" that made everything slower

An on-call engineer, paged for high latency, did the intuitive thing and bumped the connection pool from 50 to 200 across the fleet "to handle the load." Latency got *worse*. They had walked straight into the right half of the throughput curve: the database now had far more active backends than cores, context-switching dominated, and goodput dropped. Reverting to 50 — then cutting to 25 — restored sub-10 ms latency. This is the HikariCP "About Pool Sizing" finding playing out live: past the knee, shrinking the pool increases throughput. The lesson: when the database is the bottleneck, fewer connections is the lever, not more.

### 4. The prepared-statement ghost under transaction pooling

A Rails app ran fine in staging and threw intermittent `prepared statement "a1" already exists` errors in production. The difference was load: staging never had two requests share a PgBouncer backend, production did constantly. Rails' default prepared-statement cache assumed a stable session; transaction pooling gave each transaction a possibly-different backend. The fix was setting `prepared_statements: false` (and later moving to PgBouncer 1.21's protocol-level prepared-statement support). The lesson: transaction pooling breaks anything session-scoped, and the breakage is load-dependent, so it sails through staging and detonates in production.

### 5. The `idle in transaction` leak

A service wrapped a third-party HTTP call inside a database transaction — open transaction, call the payment provider, then commit. Under normal latency the connection was held for ~50 ms and nobody noticed. When the provider slowed to multi-second responses, hundreds of connections piled up in `idle in transaction`, each holding a backend and blocking `VACUUM`. The pool exhausted and the app went down — not because of database load, but because connections were pinned waiting on a *remote* system. `idle_in_transaction_session_timeout` would have reaped them; moving the HTTP call outside the transaction fixed it for good. The lesson: never hold a database connection across an external call.

### 6. The noisy-neighbor reporting job

A multi-tenant SaaS ran all services through one shared PgBouncer with a single 100-connection pool. A new analytics feature shipped a report that opened dozens of long-running connections; during business hours it consumed most of the pool, and interactive requests — checkout, login — started queuing and timing out. From the customer's view, the whole product was down because of a background report. The fix was per-service pools with hard caps (checkout 40, search 30, reporting 20). The lesson: a shared pool without per-tenant caps is only as reliable as its greediest consumer.

### 7. The pooler restart stampede

After adding PgBouncer, a team hit a *new* failure: when the single PgBouncer process restarted for a config reload, every client reconnected to it simultaneously, and PgBouncer opened its full backend pool in one burst — a smaller stampede, but now aimed at a component with no redundancy. They moved to multiple PgBouncer instances behind a TCP load balancer with `so_reuseport`, so a single instance restart only affected its share of clients. The lesson: concentrating connections behind one pooler concentrates risk; the pooler itself needs the same redundancy thinking as the database.

### 8. Heroku's connection ceiling

A pattern visible across the entire Heroku Postgres ecosystem: hobby and standard plans cap connections (the smallest tiers at around 20–120), and any app that scales its dyno count quickly blows past the limit, because each dyno runs its own pool. The community answer became near-universal — run `pgbouncer` as a buildpack or sidecar in transaction mode so a fleet of dynos multiplexes onto the plan's connection allowance. The lesson is the one this whole post argues: connection limits, not query throughput, are usually the first wall a scaling app hits, and a server-side pooler is the standard tool for climbing it. When connection limits *and* a single slow query collide, you get the 1am page in [the slow query at 1am](/blog/software-development/database-scaling/the-slow-query-at-1am) — the slow query holds connections longer, which drains the pool, which starts the spiral.

## When to reach for a server-side pooler — and when not to

Reach for a server-side pooler (PgBouncer / ProxySQL / RDS Proxy) when:

- You run **more than a few dozen application instances**, so the fleet's total connections exceed what `max_connections` can hold even with small per-pod pools.
- You run **serverless / Lambda / edge functions**, where concurrency equals connection count and no client-side pool can bound it.
- You are repeatedly hitting `too many clients already` or `remaining connection slots are reserved`, especially during deploys or traffic spikes.
- Your database CPU climbs with *connection count* rather than with query complexity — the signature of context-switch and lock-contention overhead.
- You need to **multiplex thousands of clients onto a fixed backend budget** and your traffic is mostly short OLTP transactions (high client count, low duty cycle).

Skip it, or stay in session mode, when:

- You have a **small, fixed number of app instances** and a well-sized client-side pool already keeps total connections comfortably under budget — the pooler would add a hop and a moving part for no gain.
- Your workload **depends on session state** — heavy `LISTEN/NOTIFY`, session advisory locks, long-lived prepared statements, temp-table pipelines — and you have not isolated those onto a session-mode pool. Transaction pooling will break them subtly.
- You are **early-stage** with one app server and a database at 5% CPU. Add a client-side pool, set `idle_in_transaction_session_timeout`, and move on; a server-side pooler is premature.
- Your real bottleneck is **query cost, not connection count** — adding a pooler in front of a database that is CPU-bound on bad queries just relocates the queue. Fix the queries first.

The thread running through all of it: connections are a scarce, expensive, *coordinated* resource, and the database's job is to do work, not to manage an unbounded crowd of clients clamoring to give it work. Bound the crowd at every layer — a small client pool, a multiplexing server-side pooler, per-tenant caps, and a computed budget — and the database spends its cores on queries instead of on context switches. Leave the crowd unbounded, and the first hiccup becomes a death spiral. The single most important number on your database dashboard is not QPS or CPU; it is active connections, and the goal is to keep it small, flat, and far below the knee.

### Further reading

- [Database connection pooling and pool sizing](/blog/software-development/database/database-connection-pooling) — the within-one-app companion: client-side pools, `maxLifetime`, leak detection.
- [Capacity planning for databases](/blog/software-development/database-scaling/capacity-planning-for-databases) — the four budgets, with connections modeled via Little's Law.
- [When one database is no longer enough](/blog/software-development/database-scaling/when-one-database-is-not-enough) — diagnosing which of the four resource walls (including connections) you are hitting.
- [The slow query at 1am](/blog/software-development/database-scaling/the-slow-query-at-1am) — how a single slow query drains the pool and starts the spiral.
- PgBouncer documentation (`pool_mode`, `max_client_conn`, `default_pool_size`) and the HikariCP "About Pool Sizing" wiki page — the primary sources for the numbers above.
