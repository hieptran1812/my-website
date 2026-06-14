---
title: "Database Connection Pooling and Pool Sizing"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Why a database connection is one of the most expensive objects your application allocates, why a smaller pool is almost always faster than a larger one, and how to size, pool, and operate connections from app-side pools to PgBouncer, RDS Proxy, and serverless."
tags:
  [
    "connection-pooling",
    "pgbouncer",
    "postgres",
    "hikaricp",
    "pool-sizing",
    "database",
    "performance",
    "scalability",
    "littles-law",
    "backend",
    "serverless",
    "rds-proxy",
  ]
category: "software-development"
subcategory: "Database"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/database-connection-pooling-1.webp"
---

There is a particular incident that almost every backend engineer eventually lives through, and it never looks like a database problem at first. Traffic is up maybe 30% over a normal day. The application servers are fine — CPU is comfortable, memory is fine, the load balancer is happy. And yet response times have gone vertical. p99 latency that normally sits at 40 ms is now 4 seconds, then it is timeouts, then the logs fill with a line that, the first time you see it, looks almost absurd: `FATAL: sorry, too many clients already`. The database is rejecting connections. Not slow queries — *connections*. You did not write more queries. You did not deploy a bad index. You just added a few more application pods, and the whole system fell over.

What you have collided with is one of the least-appreciated facts in backend engineering: **a database connection is an expensive, heavyweight, stateful object, and the number of them you can usefully have open is far smaller than you think.** In Postgres a connection is an entire operating-system process. In MySQL it is at minimum a thread plus a per-thread buffer allocation. Either way, opening one costs a TCP handshake, often a TLS handshake, an authentication round trip, and the server-side cost of spawning a backend. Keeping one open costs memory. And — the part that trips up everyone — *using too many of them at once makes the database slower, not faster*, because the database has a fixed number of CPU cores and disk spindles, and oversubscribing them with hundreds of concurrent backends turns the machine into a context-switching, lock-contending thrash.

The figure above is the mental model for the entire article. On the left is the world without a pool: every request pays the full freight of `connect` — TCP, TLS, auth, a server-side `fork()` to spawn a backend, run one query, then tear the whole thing down. On the right is the world with a pool: a small set of connections is opened once, kept warm, and *borrowed* by each request for the duration it needs, then returned. Everything in this article is a consequence of that one picture: why the left side is so expensive, why the right side is faster, how to size the small set of connections correctly (the answer is almost always "smaller than you guessed"), what an external pooler like PgBouncer or RDS Proxy adds on top, exactly what breaks when you turn on transaction pooling, and how connection management blows up under deploys and serverless.

![Without a pool every request pays a TCP, TLS, auth handshake and a server-side backend fork; a pool amortizes that cost once and reuses the connection](/imgs/blogs/database-connection-pooling-1.webp)

> A connection pool is not a performance optimization you bolt on later. It is the load-bearing assumption that lets a database with 16 cores serve 50,000 requests per second. Get the sizing wrong in *either* direction and you have built a bottleneck with your own hands.

This is a deep dive in the same vein as [reading EXPLAIN ANALYZE like a staff engineer](/blog/software-development/database/reading-explain-analyze-like-a-staff-engineer), the [MVCC deep dive](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb), and [database locks and deadlocks in production](/blog/software-development/database/database-locks-and-deadlocks-in-production): the goal is not a tutorial but a working mental model plus the numbers and failure modes that let you reason about your own system. We will lean on real benchmarks — Brett Wooldridge's HikariCP pool-sizing work, the PostgreSQL wiki's connection formula, Andres Freund's per-connection memory measurements — and real war stories from Instagram, GitLab, AWS, and Supabase.

## Why X is different: the assumptions that break

Before the mechanics, it is worth naming the specific mismatches between what most engineers assume about connections and what is actually true, because nearly every connection-pool incident traces back to one of these.

| Assumption | The naive view | The reality |
| --- | --- | --- |
| "A connection is cheap, like a socket." | Open it, use it, close it — it's just TCP. | In Postgres a connection is a forked OS process with its own memory; opening costs TCP + TLS + auth + `fork()`, on the order of milliseconds, and it consumes server memory for its entire life. |
| "More connections means more throughput." | Throughput scales with the number of concurrent connections. | Throughput peaks near `(cores × 2) + spindles`, then *declines* as more connections oversubscribe the CPU and cause context-switch and lock thrashing. A famous Oracle benchmark cut response time from ~100 ms to ~2 ms by *shrinking* the pool from thousands to 96. |
| "The pool should be as big as my peak concurrency." | If 1,000 requests can be in flight, I need 1,000 connections. | You need only as many connections as can be *actively executing* on the database at once. The rest should queue in the application, cheaply, not as idle Postgres processes. |
| "An idle connection is free." | It's just sitting there. | Each idle Postgres backend still holds memory (roughly 1.3–7.6 MB of genuinely private memory plus a share of caches) and counts against `max_connections`. Thousands of idle connections is a real resource leak. |
| "PgBouncer is a drop-in; turn it on and you're done." | Just point the app at port 6432. | Transaction pooling silently breaks server-side prepared statements, `SET`/session GUCs, advisory locks, `LISTEN`/`NOTIFY`, and `WITH HOLD` cursors. Half of "PgBouncer broke my app" tickets are these. |
| "Serverless is the same, just point Lambda at the DB." | Each function opens a connection like any client. | Serverless has no shared process to hold a pool; N concurrent invocations means N connections, and a traffic spike becomes a connection-explosion that exhausts `max_connections` in seconds. |

If even two of those rows surprised you, this article is going to change how you provision databases. Let us take them in order, starting with why a connection is expensive in the first place.

## 1. Why a database connection is so expensive

**Senior rule of thumb: treat a database connection like a thread, not like a socket — it is a scarce, heavyweight, stateful resource, and the count you can sustain is bounded by the database's CPU, not by your application's concurrency.**

### Postgres: one process per connection

PostgreSQL uses a *process-per-connection* model. There is a supervisor process — historically called the postmaster — that listens on the socket. When a client connects, the postmaster `fork()`s a brand-new operating-system process, the *backend*, to serve that one client for the connection's entire lifetime. That backend process is where your queries actually run. When the client disconnects, the backend exits and is reaped.

This design is wonderfully robust — a crash in one backend cannot corrupt another's memory, and the OS scheduler handles fairness for free — but it makes each connection costly along three axes:

1. **Setup cost.** Establishing a connection requires a TCP handshake (one round trip), usually a TLS handshake (one or two more round trips and asymmetric crypto), the Postgres startup message exchange, and authentication (which for SCRAM is several more round trips and a key-derivation step). Then the postmaster has to `fork()`, the new backend has to attach to shared memory, initialize its catalogs cache, set up its memory contexts, and apply per-role settings. On a loaded server this is comfortably in the **single-to-low-double-digit milliseconds** range — an eternity next to a query that might run in 200 microseconds.

2. **Memory cost.** Every backend has private memory. The naive way to measure it — looking at `RSS` in `ps` — is badly misleading, because `ps` attributes each backend's *share of the shared buffer pool* to that process, so 100 connections each "using" 3 GB looks like 300 GB when the shared buffers are actually allocated once. Andres Freund measured the *genuinely private* overhead carefully using `/proc/[pid]/smaps_rollup` and found that with `huge_pages=off` an idle connection costs about **7.6 MB** (private anonymous memory plus page-table entries), and with `huge_pages=on` it drops to about **1.3 MB**, concluding that "a connection only has an overhead of less than 2 MiB" when measured accurately. On top of that, *active* connections can each allocate `work_mem` per sort/hash node in a query — the PostgreSQL wiki warns that "`work_mem` RAM can be allocated for each node of a query on each connection, all at the same time," which is how 500 connections × a 64 MB `work_mem` × a few sort nodes turns into a swap storm.

3. **Scheduling cost.** This is the one that matters most for throughput and we will return to it in the sizing section. Each backend is a process the kernel must schedule onto a core. If you have 16 cores and 2,000 runnable backends, the kernel is constantly preempting one backend to run another — saving registers, flushing TLB entries, evicting cache lines — and the *useful* work per core collapses.

### MySQL: one thread per connection

MySQL/InnoDB uses a *thread-per-connection* model: the server has a single process, and each connection gets a thread. Threads are cheaper than processes — no `fork()`, shared address space — but they are not free. Each connection still allocates per-thread buffers (`sort_buffer_size`, `join_buffer_size`, `read_buffer_size`, the net buffers, and so on), and the same scheduling math applies: thousands of runnable threads on a handful of cores thrash. MySQL Enterprise and Percona offer a *thread pool* plugin precisely to cap the number of threads actually executing, decoupling "connections" from "concurrent execution" — which is, in effect, an internal connection pool. The community edition does not have it, so an external pooler (ProxySQL, or the application's own pool) does the same job.

The thread-pool plugin is instructive because it makes the core idea of this article explicit *inside the database*: it accepts many connections but admits only a bounded number of them into "transaction groups" that actually run, queueing the rest. That is exactly what a well-sized pool does from the outside — and it confirms that the bound on *concurrent execution*, not on *connections*, is the real scaling parameter. ProxySQL, the most common external pooler for MySQL, additionally multiplexes connections (frontend connections from the app are decoupled from backend connections to MySQL, much like PgBouncer's transaction mode), routes reads to replicas, and caches query results — so for MySQL at scale the pooler is doing pooling *and* routing, the same convergence we see with PgCat on the Postgres side.

The point is general: **whether the unit is a process or a thread, the database has a hard ceiling on how many it can usefully run at once, set by cores and I/O, and that ceiling is much lower than the number of clients you want to serve.** A pool is the adapter between "many clients" and "few executors."

It is worth dwelling on *why* the per-connection cost is so much higher in Postgres than people expect, because the process model has consequences beyond memory. Every backend maintains its own catalog cache, its own plan cache, its own relation cache, and its own private memory contexts. These warm up over the connection's life — the first few queries on a fresh backend are slower because the caches are cold — which is yet another reason churning connections per request is wasteful: you throw away the warmed caches every time. A pooled connection that has run ten thousand queries has a hot catalog cache and parses and plans faster than a cold one. The pool is not just amortizing the *setup* cost; it is amortizing the *warm-up* cost too. This is also why a pool with a `max-lifetime` recycles connections *occasionally* (every 30 minutes, say) rather than never: a backend that lives forever can slowly accumulate cache and memory-context bloat, so periodic recycling trades a tiny amount of re-warming for bounded memory. The art of pool configuration is finding the sweet spot between "recycle too often" (churn, cold caches) and "never recycle" (bloat, stale connections that miss a failover).

### The cost of churn

Even if you never exceed `max_connections`, opening and closing a connection *per request* is pure waste. Suppose your endpoint does 200 microseconds of actual query work but pays 5 milliseconds to open the connection and a millisecond to tear it down. You are spending **96.6%** of your database time on connection management and 3.4% on the query. Multiply by your request rate and you have manufactured a bottleneck out of nothing. This is the single most common form of self-inflicted database pain in early-stage applications: a framework that opens a connection at the start of each web request and closes it at the end, with no pool. The fix is not faster hardware. The fix is to stop throwing the connection away.

The figure 1 contrast makes this concrete: the left column (no pool) does TCP + TLS + auth + fork + close *on the hot path of every request*; the right column (pool) does it *once*, at startup, and every request after that just borrows a warm, already-authenticated, already-forked backend. The pooled path's checkout is a thread-safe handoff of an existing object — microseconds, no syscalls, no network.

## 2. What a pool actually is, and where it lives

**Senior rule of thumb: there are two places to pool — inside each application process, and in a shared proxy in front of the database — and a serious system usually wants both.**

A connection pool is, at its core, a bounded set of pre-opened connections plus a thread-safe checkout/checkin protocol and a wait queue. A request that needs the database *acquires* (checks out) a connection from the pool, uses it, and *releases* (checks in) it when done. If all connections are busy, the request waits in a queue up to some timeout. That is the whole abstraction, and it is identical whether the pool lives in your application or in a separate process.

![Application pools cut connection churn per process while an external pooler funnels thousands of clients onto a small Postgres backend pool](/imgs/blogs/database-connection-pooling-2.webp)

The topology above is the standard shape at scale. Each application process (a pod, a worker, a container) holds a small *app-side* pool — say 10 connections. Those connect not directly to Postgres but to an *external pooler* (PgBouncer, PgCat, Odyssey, or a managed proxy like RDS Proxy or Supavisor), which itself maintains a much smaller pool of *server-side* connections to Postgres. The external pooler is what lets you have hundreds of application processes — and serverless functions that have no persistent pool at all — collapse down onto a backend pool of perhaps 20–40 connections that Postgres can actually run efficiently.

### App-side pools

App-side pools live inside the application process and hand out connections to request threads or coroutines. They are the first and most important line of defense because they eliminate per-request churn for free. The major implementations:

- **HikariCP** (JVM) — the de facto standard for Java/Kotlin, famously fast and small, written by Brett Wooldridge, whose pool-sizing wiki we cite throughout.
- **pgx / `database/sql`** (Go) — `pgxpool` is the idiomatic Postgres pool; the standard library's `database/sql` also pools.
- **SQLAlchemy** (Python) — `QueuePool` is the default; `NullPool` disables pooling, `StaticPool` for tests.
- **node-postgres (`pg.Pool`)** (Node.js) — the standard pool for the Node ecosystem.
- **HikariCP-equivalents** in nearly every language: `r2dbc-pool`, `Npgsql`'s built-in pool (.NET), ActiveRecord's pool (Ruby), and so on.

Here is a HikariCP configuration that encodes the right defaults. Note the sizing — small — and the explicit timeouts:

```properties
# HikariCP — application.properties (Spring Boot style)
spring.datasource.hikari.maximum-pool-size=10
spring.datasource.hikari.minimum-idle=10          # = max: keep a fixed-size pool, avoid churn
spring.datasource.hikari.connection-timeout=3000  # ms to wait for a connection before failing fast
spring.datasource.hikari.idle-timeout=0           # 0 = never expire idle (fixed pool)
spring.datasource.hikari.max-lifetime=1800000     # 30 min: recycle to dodge stale conns / failovers
spring.datasource.hikari.keepalive-time=120000    # 2 min: ping idle conns so NAT/firewall don't drop them
spring.datasource.hikari.pool-name=app-primary
```

Wooldridge's own advice, baked into HikariCP's docs, is blunt: set `minimumIdle` equal to `maximumPoolSize` to keep a fixed-size pool rather than letting it grow and shrink, because the churn of growing and shrinking is itself a cost, and "a small fixed-size pool, saturated with threads waiting for connections" is the target.

The equivalent in Go with pgx:

```go
package main

import (
	"context"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
)

func newPool(ctx context.Context, dsn string) (*pgxpool.Pool, error) {
	cfg, err := pgxpool.ParseConfig(dsn)
	if err != nil {
		return nil, err
	}
	cfg.MaxConns = 10                       // small, per-process
	cfg.MinConns = 10                       // fixed size; no cold-start churn
	cfg.MaxConnLifetime = 30 * time.Minute  // recycle to survive failovers
	cfg.MaxConnIdleTime = 0                 // keep idle conns warm
	cfg.HealthCheckPeriod = 1 * time.Minute // background keepalive + reap
	// Fail fast instead of piling up goroutines on an exhausted pool:
	cfg.ConnConfig.ConnectTimeout = 3 * time.Second
	return pgxpool.NewWithConfig(ctx, cfg)
}
```

And SQLAlchemy in Python, where the trap is that the *default* pool is per-process and the total is `pool_size + max_overflow`:

```python
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql+psycopg://app:secret@pgbouncer:6432/app",
    pool_size=8,         # steady-state connections held by this process
    max_overflow=2,      # burst allowance; total ceiling = pool_size + max_overflow = 10
    pool_timeout=3,      # seconds to wait for a connection before raising TimeoutError
    pool_recycle=1800,   # recycle connections after 30 min
    pool_pre_ping=True,  # cheap SELECT 1 before handing out a possibly-dead conn
)
```

The crucial, often-missed arithmetic: **the database sees the sum of every process's pool.** If you run 50 Gunicorn workers each with `pool_size=8, max_overflow=2`, your *application* can open `50 × 10 = 500` connections to Postgres. If `max_connections` is 200, you are already over the cliff before a single user shows up. App-side pools cut churn but they do *not* by themselves bound the global connection count across many processes — that is exactly the job of an external pooler.

### External poolers

An external pooler is a separate process (or managed service) that sits between your application and the database and pools the *server-side* connections globally. The big ones:

| Pooler | Language | Model | Notable for |
| --- | --- | --- | --- |
| **PgBouncer** | C, single-threaded | session / transaction / statement modes | The default, battle-tested everywhere; ~15k QPS per process, pegs one core |
| **PgCat** | Rust, multi-threaded | PgBouncer-compatible + load balancing, sharding, failover | Modern rewrite; query routing to read replicas |
| **Odyssey** | C, multi-threaded | session / transaction; built by Yandex | Multi-threaded throughput, TLS, connection limits per route |
| **Supavisor** | Elixir/BEAM (+ Rust SQL parser) | transaction (multi-tenant, cloud-native) | Built for serverless/edge; designed for huge client counts across tenants |
| **RDS Proxy** | Managed (AWS) | transaction-level multiplexing | Fully managed, IAM auth, fast failover, native Lambda integration |
| **Pgpool-II** | C | pooling + replication + load balancing | Older, heavier; does much more than pooling |

The reason external poolers matter so much is the multiplexing ratio. PgBouncer in transaction mode might accept `max_client_conn = 2048` client connections while maintaining only `default_pool_size = 100` server connections to Postgres — a 20:1 funnel. Supabase's Supavisor was built explicitly to handle on the order of a *million* client connections across tenants by multiplexing them onto a tiny pool of real backends; its team chose Elixir/BEAM precisely because the runtime "supports high concurrency and rapid I/O," and integrated Rust via Rustler for the CPU-heavy SQL parsing because, as they note, "Elixir doesn't have great performance for parsing." The architecture is dynamic: when a tenant first connects, Supavisor starts that tenant's pool, opens the backend connections, and distributes the routing state across the cluster so subsequent clients are proxied without re-establishing backends.

The lifecycle of one request through a pooled connection is worth making explicit, because the place a connection gets *returned* is exactly where pool modes differ:

![A pooled request borrows a warm backend, runs a transaction, then returns the connection to the pool for the next request to reuse](/imgs/blogs/database-connection-pooling-9.webp)

In *session* pooling the connection is returned at the far right — when the *client* disconnects. In *transaction* pooling it is returned much earlier — the instant `COMMIT`/`ROLLBACK` runs. That single difference in *when* the connection goes back to the pool is the whole story of pool modes, and it is where the dragons live.

## 3. PgBouncer pool modes, and exactly what breaks

**Senior rule of thumb: transaction pooling gives you the multiplexing you want, but it takes away all server-side session state — so before you flip it on, audit your code for prepared statements, `SET`, advisory locks, `LISTEN`/`NOTIFY`, and cursors.**

PgBouncer offers three pool modes, defined by the moment a server connection is returned to the pool:

- **Session pooling** — "Most polite method. When a client connects, a server connection will be assigned to it for the whole duration it stays connected." The server connection is returned only when the client disconnects. This preserves *all* Postgres semantics because the client effectively owns a real backend for its whole life. The cost: multiplexing is nearly zero — one client ties up one backend, so PgBouncer in session mode mostly just reduces *connection churn*, not concurrent backend count.

- **Transaction pooling** — "A server connection is assigned to a client only during a transaction." The instant the transaction ends (`COMMIT`/`ROLLBACK`, or a successful autocommit statement), the backend goes back into the pool and can serve a different client's next transaction. This is the high-multiplexing mode that makes 2,000 clients fit on 100 backends. It is also the mode that breaks things.

- **Statement pooling** — "Most aggressive method. This is transaction pooling with a twist: multi-statement transactions are disallowed." It enforces autocommit; designed for sharding layers like PL/Proxy. Rarely the right choice for an ordinary application.

The matrix below is the one to internalize. The rows are the three modes; the columns are the session-scoped features that transaction pooling sacrifices.

![Transaction pooling maximizes multiplexing but breaks prepared statements, SET state, advisory locks, and LISTEN/NOTIFY that rely on a stable backend](/imgs/blogs/database-connection-pooling-3.webp)

### Why these specific features break

The unifying principle: **anything whose state lives in the *server backend* and persists *across transactions* is unsafe under transaction pooling, because the next transaction might land on a different backend.** PgBouncer's own documentation lists the features that are "never" compatible with transaction pooling:

- **`SET` / `RESET` (session GUCs).** If you run `SET statement_timeout = '5s'` or `SET search_path = tenant_42`, that setting lives on the backend. Your next query, in a new transaction, may execute on a *different* backend that never saw the `SET`. Worse, the backend you set it on still carries that setting when it serves the *next, unrelated* client — a cross-tenant state leak. This is the most insidious one: `search_path` leakage in a multi-tenant app is a data-isolation bug, not just a correctness bug.

- **Server-side prepared statements (`PREPARE` / `DEALLOCATE`).** A prepared statement is named state on a specific backend. Under transaction pooling your `PREPARE foo` and your later `EXECUTE foo` may hit different backends, yielding `prepared statement "foo" does not exist`. This is the single most common transaction-pooling breakage because *drivers do it for you* — many client libraries silently use the extended protocol with named prepared statements for every parameterized query. The fixes: disable server-side prepared statements in the driver (`prepareThreshold=0` for the JDBC driver, `statement_cache_size=0` / `prepared_statement_cache_size=0` in various drivers, `default_query_exec_mode=exec` or `simple_protocol` in pgx, `?prepared_statements=false` in some DSNs), or set PgBouncer's `max_prepared_statements` to a non-zero value, which lets PgBouncer track protocol-level prepared statements per-backend (a feature added in PgBouncer 1.21+).

- **Session-level advisory locks (`pg_advisory_lock`).** A session-level advisory lock is held by the backend until explicitly released or the *session* ends. Under transaction pooling the "session" boundary is meaningless — you might acquire the lock on one backend and then never get that backend back, so you can neither rely on the lock nor reliably release it. Use *transaction-scoped* advisory locks (`pg_advisory_xact_lock`) instead, which are automatically released at transaction end and therefore safe.

- **`LISTEN` / `NOTIFY`.** `LISTEN` registers interest on a specific backend; notifications are delivered to that backend. Under transaction pooling you do not own a stable backend, so you cannot reliably listen. Pub/sub built on `LISTEN`/`NOTIFY` simply does not work through a transaction pooler — route it through a dedicated session-mode connection or a different transport.

- **`WITH HOLD` cursors and `PREPARE`d cursors that outlive a transaction.** A normal cursor lives within a transaction and is fine; a `WITH HOLD` cursor is designed to survive `COMMIT`, which means it needs a stable backend — incompatible.

- **Temp tables with `ON COMMIT PRESERVE ROWS`, `LOAD` of shared libraries, and similar backend-resident state.** Same root cause.

A compact way to think about it: **transaction pooling is safe if and only if every transaction is self-contained — it sets up whatever state it needs, uses it, and tears it down before `COMMIT`.** The moment your code assumes "the thing I configured earlier is still here," transaction pooling will eventually betray you, often intermittently and under load, which is the worst kind of bug to debug.

Here is a production-grade PgBouncer configuration for transaction pooling, annotated:

```ini
; pgbouncer.ini — transaction pooling for a multi-tenant web app
[databases]
; one logical DB; PgBouncer opens server conns to the real Postgres
app = host=10.0.1.10 port=5432 dbname=app

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type = scram-sha-256
auth_file = /etc/pgbouncer/userlist.txt

pool_mode = transaction          ; the high-multiplexing mode
max_client_conn = 5000           ; how many app/serverless clients can connect
default_pool_size = 25           ; server conns PER (user,db) pair — keep this SMALL
min_pool_size = 5                ; keep a few warm so first requests don't pay connect cost
reserve_pool_size = 5            ; extra conns for bursts, lent briefly
reserve_pool_timeout = 3         ; seconds a client waits before tapping the reserve pool

server_idle_timeout = 600        ; close server conns idle > 10 min (let DB reclaim)
server_lifetime = 3600           ; recycle server conns hourly (dodge memory bloat / failover)
query_wait_timeout = 5           ; client waits at most 5s for a server conn, else error

; CRITICAL for transaction pooling: let PgBouncer track protocol-level
; prepared statements so drivers that prepare keep working.
max_prepared_statements = 200

; expose stats for monitoring (SHOW POOLS / SHOW STATS on the admin DB)
stats_period = 60
```

The `default_pool_size = 25` is the line that does the heavy lifting: no matter how many of the 5,000 clients are connected, PgBouncer will only ever open 25 server connections per `(user, db)` to Postgres. That is the funnel.

### A second-order gotcha: connection pinning

Even in transaction mode, certain operations force a client to be *pinned* to one backend for longer than a single transaction — PgBouncer cannot return the backend to the pool because the client still has backend-resident state. The classic triggers are a session-level `SET`, an open `LISTEN`, a session-level advisory lock, or a prepared statement under older PgBouncer. RDS Proxy documents the same phenomenon explicitly: it multiplexes by default, but "certain types of database operations can cause connections to be pinned," and a pinned-but-idle client "continues to hold on to a database connection and prevent the database connection from being reused by other client connections." A pool that should be multiplexing 20:1 can collapse to 1:1 if every client pins. The operational tell is that your effective multiplexing ratio (client connections ÷ server connections) drops toward 1 — and the fix is to find and remove whatever is pinning (usually a stray `SET` or a driver still using named prepared statements).

## 4. Pool sizing: why a small pool is faster

**Senior rule of thumb: the right pool size is shockingly small — roughly `(core_count × 2) + effective_spindle_count` of *active* connections — and making it bigger almost always makes the system slower.**

This is the most counterintuitive result in the whole topic, so let us build the intuition carefully before the math.

### The intuition: a checkout line, not a buffet

Imagine a grocery store with four cashiers. Customers arrive, and you want to serve them as fast as possible. There are two strategies. The first: open exactly four lanes, and have everyone queue up; each cashier works one customer at a time, start to finish, and the line moves at the full speed of four cashiers. The second: try to serve forty customers "simultaneously" by having each cashier ring up ten customers at once, jumping between them — scan one item for customer A, then one for customer B, then back to A. The second strategy feels more parallel, but it is obviously slower: the cashier wastes enormous effort switching attention, and *no individual customer finishes faster* — they all finish later. The store's throughput (customers per minute) is highest with four saturated lanes and a queue, not forty half-served customers.

A database is the grocery store. The cashiers are CPU cores (and, for I/O-bound work, disk spindles). Connections are the customers being actively served. If you have 16 cores and you let 2,000 connections run concurrently, every core is "ringing up" 125 backends at once, time-slicing between them, and the time lost to context switching — saving and restoring registers, flushing the TLB, evicting and reloading cache lines — dwarfs the useful work. Wooldridge puts it precisely: "executing A and B sequentially will always be faster than executing A and B 'simultaneously' through time-slicing." The PostgreSQL wiki names the same mechanisms — context switching, lock contention, cache-line contention, and `work_mem` memory pressure — as the reasons too many connections degrade performance, and observes that "you will often see a transaction reach completion sooner if you queue it... and start it later when resources become available."

![Past the core-bound peak, more connections cut throughput and raise latency through context-switch and lock thrashing — a smaller saturated pool is faster](/imgs/blogs/database-connection-pooling-4.webp)

### The benchmark that proves it

The canonical demonstration comes from an Oracle performance group, recounted in HikariCP's pool-sizing wiki. They had a system with ~10,000 front-end users doing ~20,000 transactions per second. They reduced the connection pool — *just the pool, nothing else* — from a few thousand down to **96 connections**, and the average response time dropped from roughly **100 ms to about 2 ms**, a 50× improvement, with throughput holding or improving. The figure above captures exactly this: the same 10,000 users, the only change being a smaller pool, and a 50× latency win because the database stopped thrashing.

A second data point in the same wiki: a benchmark on PostgreSQL where transactions-per-second *flattens* around 50 connections and does not improve beyond it — adding connections past the plateau buys nothing and eventually costs. The shape of the curve is universal: throughput rises with concurrency up to the point where you saturate cores/spindles, plateaus briefly, and then *declines* as oversubscription overhead takes over. The latency curve is worse: past saturation, latency climbs steeply because every request now competes with hundreds of others for a slice of CPU.

### The formula

The PostgreSQL wiki and HikariCP converge on the same starting formula for the number of *active* connections that maximizes throughput:

$$
\text{connections} = (\text{core\_count} \times 2) + \text{effective\_spindle\_count}
$$

Two subtleties the wiki is careful about:

- **`core_count` does not include hyperthread siblings.** A 16-physical-core machine with HT showing 32 logical CPUs counts as 16, not 32, for this formula.
- **`effective_spindle_count` is zero if the active data set is fully cached in RAM, and approaches the actual number of disk spindles as the cache-hit rate falls.** The `× 2` for cores exists precisely because, while one query on a core is blocked waiting for I/O, a second can use the core; the spindle term adds capacity for the I/O concurrency itself. On an all-SSD, fully-cached workload, the effective spindle count is small, which is why pool sizes for modern hardware trend toward the low side (closer to `core_count`, not far above it).

Worked example from the wiki: a 4-core i7 server with one hard disk gives `(4 × 2) + 1 = 9` connections, round to **10** — and that 10-connection pool can serve "3,000 front-end users at 6,000 TPS." Ten connections, three thousand users. That ratio is the whole point.

### The pool-locking correction

There is one important upward adjustment. If a single application thread can hold **more than one connection at a time** — for example, the main query connection plus a second connection to look something up mid-transaction — then a naive small pool can *deadlock*: every thread grabs its first connection, then all block forever waiting for a second that no one will release. Wooldridge gives the minimum-size formula to avoid this:

$$
\text{pool size} = T_n \times (C_m - 1) + 1
$$

where $T_n$ is the maximum number of threads and $C_m$ is the maximum number of simultaneous connections a single thread needs. Example: with $T_n = 3$ threads each needing $C_m = 4$ connections, the minimum safe pool is $3 \times (4 - 1) + 1 = 10$. The `+1` guarantees at least one thread can always make forward progress and release, breaking the deadlock. In practice the best fix is to *not* hold multiple connections per logical unit of work — but when you must, this formula is the floor.

## 5. Little's Law: turning your traffic into a pool size

**Senior rule of thumb: you don't guess a pool size, you derive it — the number of *busy* connections you need equals your arrival rate times your mean connection-hold time, and everything above that should queue, not connect.**

The formula above tells you the *ceiling* set by hardware. Little's Law tells you the *floor* set by your actual traffic — how many connections your load genuinely needs busy at once. It is the most useful single equation in capacity planning, and it is dead simple.

Little's Law states that for any stable system, the average number of items inside it ($L$) equals the average arrival rate ($\lambda$) times the average time each item spends inside it ($W$):

$$
L = \lambda \times W
$$

Map it onto a connection pool: $L$ is the number of connections busy on average, $\lambda$ is the rate at which requests need a connection (requests per second), and $W$ is the mean time a request holds a connection (seconds) — which is *not* your end-to-end latency, but specifically the time from checkout to checkin: the query execution time plus any application work done while the connection is held.

![Little's Law L equals lambda times W converts arrival rate and connection-hold time into the number of busy connections a pool needs](/imgs/blogs/database-connection-pooling-5.webp)

The figure above is the queueing model. Requests arrive at rate $\lambda$, enter a bounded wait queue if no connection is free, get a connection (checkout), run their transaction on a busy connection for time $W$, then release it (departures return to the pool). The number of connections that are busy at any instant is $L = \lambda \times W$.

### A concrete sizing worked example

Suppose a service handles **5,000 requests per second** at peak, and each request, once it has a connection, holds it for **4 milliseconds** (a fast OLTP query plus a little serialization). Then:

$$
L = \lambda \times W = 5{,}000 \text{ req/s} \times 0.004 \text{ s} = 20 \text{ connections busy on average}
$$

So *on average* you need 20 connections actively executing. You do not need 5,000 connections, even though 5,000 requests arrive each second — because each one only occupies a connection for 4 ms, so a single connection can serve 250 requests per second, and 20 connections serve 5,000. This is the arithmetic that lets 20 backends serve 5,000 req/s.

Here is the calculation as a small Python tool you can actually run against your own numbers, including a safety-factor for variance and the hardware ceiling:

```python
import math


def size_pool(
    requests_per_sec: float,
    mean_hold_ms: float,
    cores: int,
    effective_spindles: int = 0,
    utilization_target: float = 0.75,  # don't run the pool at 100% busy
    p99_multiple: float = 2.5,         # headroom for hold-time variance / tail latency
) -> dict:
    """Derive a connection-pool size from Little's Law, then clamp to the
    hardware ceiling. Returns the average load, a sized recommendation, and
    the ceiling so you can see when traffic outgrows the database itself."""
    # Little's Law: average busy connections = arrival rate * mean hold time.
    avg_busy = requests_per_sec * (mean_hold_ms / 1000.0)

    # Provision above the average so we don't run at the edge of saturation,
    # and add headroom for variance in hold time (tail queries hold longer).
    sized = math.ceil((avg_busy / utilization_target) * (p99_multiple / 2.5))

    # Hardware ceiling: the database can only usefully run this many at once.
    ceiling = cores * 2 + effective_spindles

    recommended = min(sized, ceiling)
    return {
        "avg_busy_connections": round(avg_busy, 1),
        "littles_law_sized": sized,
        "hardware_ceiling": ceiling,
        "recommended_pool_size": recommended,
        "saturated": sized > ceiling,  # traffic exceeds the DB; queue or scale out
    }


if __name__ == "__main__":
    # 5,000 req/s, 4 ms mean hold, 16-core all-SSD (fully cached -> spindles ~ 0)
    print(size_pool(requests_per_sec=5000, mean_hold_ms=4, cores=16))
    # A slower workload: 2,000 req/s but 25 ms mean hold (reporting queries)
    print(size_pool(requests_per_sec=2000, mean_hold_ms=25, cores=16))
```

Running it on the two scenarios:

```
{'avg_busy_connections': 20.0, 'littles_law_sized': 27, 'hardware_ceiling': 32, 'recommended_pool_size': 27, 'saturated': False}
{'avg_busy_connections': 50.0, 'littles_law_sized': 67, 'hardware_ceiling': 32, 'recommended_pool_size': 32, 'saturated': True}
```

The first workload (fast queries) needs ~27 connections, comfortably under the 32-connection hardware ceiling — size the pool at 27 and you are done. The second workload (slow 25 ms reporting queries) *wants* 67 connections by Little's Law but the hardware can only run 32 usefully, so it is flagged `saturated`: the right move is **not** to set the pool to 67 (that would thrash), but to either (a) make the queries faster so $W$ drops, (b) move reporting to a read replica, or (c) accept that requests will queue and size the pool at the ceiling. This is the key insight Little's Law gives you that the raw formula does not: **when your traffic demands more busy connections than the hardware can run, the answer is to reduce $W$ or scale the database, never to grow the pool past the ceiling.** Growing it just converts a fast queue into a slow thrash.

### Why queue instead of grow

The bottom band of the figure states the discipline: when demand spikes past the pool size, let $W$ (the wait) rise in a *bounded* application queue, fail fast past a timeout, and shed load — rather than opening more connections. A request that waits 20 ms in a cheap in-process queue and then runs in 4 ms on an un-thrashed database finishes in 24 ms. The same request on an oversized, thrashing pool might "start immediately" but take 200 ms to grind through CPU contention. The queue is faster. This is queueing theory's central result restated for pools: a saturated server with a queue has higher throughput and lower latency than an oversubscribed server without one.

## 6. Connection storms: the thundering herd on deploy

**Senior rule of thumb: every connection your fleet opens *at the same moment* is a connection storm waiting to happen — synchronize the herd and you exhaust `max_connections`; jitter and warm it, and you don't.**

So far we have reasoned about steady state. The nastiest connection failures are *transient* and happen at exactly the moments you change the system: a deploy, a database failover, a network blip that drops connections en masse. The mechanism is the *thundering herd*: a large number of clients all try to (re)connect at the same instant, and the synchronized surge overwhelms the database's ability to accept connections.

![A rolling deploy or DB restart makes every pod reconnect simultaneously, spiking backends past max_connections before backoff and jitter damp it](/imgs/blogs/database-connection-pooling-6.webp)

The timeline above walks through a classic deploy storm. At steady state, 20 backends are busy. A rolling deploy starts; old pods drain and 200 new pods boot. If each new pod eagerly opens its full `min_pool` of connections the moment it starts — and they all start within a second or two of each other — the fleet tries to open thousands of connections in one synchronized burst. Postgres hits `max_connections` and starts returning `FATAL: sorry, too many clients already`. The new pods, getting connection errors, *retry immediately* — and now you have a retry storm layered on the connect storm, a positive feedback loop that can keep the database pinned for minutes. The same shape happens on a database failover: every existing connection breaks at once, and the entire fleet tries to reconnect simultaneously.

The damping techniques, in order of importance:

1. **Jitter every reconnect and pool-fill.** Never reconnect on a fixed schedule. Add randomized delay so the herd spreads out over time instead of hammering in one instant. This single change converts a vertical spike into a gentle ramp.

2. **Exponential backoff with a cap on retries.** When a connect fails, wait longer each time (with jitter) rather than retrying in a tight loop. The combination of exponential backoff *and* jitter is what AWS's well-known "backoff and jitter" guidance recommends, and it is the difference between a 30-second blip and a 30-minute outage.

3. **Bound the per-pod pool and put a pooler in front.** If pods connect to PgBouncer rather than directly to Postgres, the herd hits the pooler — which has a fixed, small `default_pool_size` to Postgres — instead of hitting `max_connections` directly. The pooler absorbs the storm and queues the overflow.

4. **Warm pools and slow start.** Fill pods' pools gradually (`min_pool_size` small, grow lazily) and use a deployment slow-start so not all pods come up at once. Many orchestrators support `maxSurge`/`maxUnavailable` tuning to throttle how fast a rollout proceeds.

5. **Set `max_connections` with headroom and a `superuser_reserved_connections`.** The PostgreSQL wiki advises making `max_connections` "a bit bigger than the number of connections you enable in your connection pool... so there are always a few slots available for direct connections for system maintenance and monitoring." When a storm hits, those reserved slots are what let you actually log in and diagnose.

The illustrative `psql` snippet to watch a storm in progress and find who is connecting:

```sql
-- How close are we to the ceiling, right now?
SELECT count(*)                              AS total,
       count(*) FILTER (WHERE state = 'active')              AS active,
       count(*) FILTER (WHERE state = 'idle')                AS idle,
       count(*) FILTER (WHERE state = 'idle in transaction') AS idle_in_txn,
       current_setting('max_connections')::int              AS max_conns
FROM pg_stat_activity;

-- Who is the herd? Group by application_name and client to see the surge source.
SELECT application_name, client_addr, count(*)
FROM pg_stat_activity
GROUP BY application_name, client_addr
ORDER BY count(*) DESC
LIMIT 20;
```

## 7. The serverless connection explosion

**Senior rule of thumb: serverless functions have no shared process to hold a pool, so concurrency equals connections — and without a proxy in front, a traffic spike becomes a connection spike that kills the database.**

Serverless is where connection management goes from "important" to "the entire ballgame," because the core assumption that makes app-side pools work — a long-lived process that opens connections once and reuses them — is gone. Each Lambda invocation runs in its own short-lived execution environment. A pool created inside a function instance can be reused only if that *same* instance handles the next request (a "warm" invocation), but the platform scales by spinning up *new* instances, each of which opens its own connections. The result: **concurrency equals connections.** 500 concurrent Lambda invocations means up to 500 database connections, full stop — and a Lambda function can scale from zero to hundreds of concurrent invocations in seconds.

![Serverless concurrency opens one connection per container and exhausts Postgres, while a managed proxy multiplexes the same clients onto a small backend pool](/imgs/blogs/database-connection-pooling-7.webp)

The before/after above is the whole problem and its fix. On the left, Lambda connects straight to Postgres: 500 concurrent invocations, each a fresh container with no shared pool (an in-function pool cannot help across instances), means 500 direct connections plus reconnect churn from cold starts — and `max_connections` is exhausted, returning `FATAL: remaining connection slots are reserved`. On the right, Lambda connects to **RDS Proxy** (or Supavisor, or any transaction-mode pooler): the 500 client connections *terminate at the proxy*, which multiplexes them at the transaction level onto perhaps 20–40 real backends, capped by `MaxConnectionsPercent`, and Postgres stays comfortably under its limit.

The math AWS itself uses to motivate this: a `db.r6g.large` PostgreSQL instance has a default `max_connections` of around 1,600, which sounds like a lot until you add it up — a Lambda function at 100 concurrency is 100 connections, an ECS service with 10 tasks each pooling 20 is another 200, a second such service is another 200, and "at 500 connections with only three consumers, you quickly hit the wall as you add more services or scale up Lambda concurrency." RDS Proxy exists precisely to break the coupling between consumer count and connection count.

The RDS Proxy knobs that matter for serverless:

- **`MaxConnectionsPercent`** — the cap on backend connections as a percentage of `max_connections`. Set it to leave headroom; AWS recommends at least 30% above your recent peak usage because "internal capacity changes might require at least 30% headroom... to avoid increased borrow latencies," and gives minimum recommended values per instance class (e.g., 30 for `db.t3.small`, 20 for `db.t3.medium` and above).
- **`MaxIdleConnectionsPercent`** — how many idle backends the proxy keeps warm. The default is 50% of `MaxConnectionsPercent`. For bursty Lambda traffic you often want this *low* so the proxy doesn't hold idle backends during quiet periods — but high enough to absorb the next surge without a connect spike.
- **`IdleClientTimeout`** — default **1,800 seconds (30 minutes)**; how long a client connection can be idle before the proxy closes it.
- **`ConnectionBorrowTimeout`** — default **120 seconds**; how long the proxy waits for a free backend before returning a timeout error when the pool is maxed.
- **Client-connection max life is a hard 24 hours** (not configurable); AWS advises configuring your app-side pool's max lifetime *below* 24 hours to avoid surprise drops.

And the serverless-specific trap, again: **pinning.** RDS Proxy multiplexes only while a client has no backend-resident session state. Operations like a `SET`, a prepared statement, or a temp table pin the client to a backend, "preventing their immediate release back to the pool." A serverless workload that prepares statements (many ORMs do by default) can find its 20:1 multiplexing collapse to 1:1, re-exhausting connections — so for serverless, configure the driver to use the simple query protocol or otherwise disable server-side prepared statements, exactly as for PgBouncer transaction mode.

## 8. Monitoring: what to watch on a pool

**Senior rule of thumb: the two leading indicators of pool trouble are *saturation* (how full the pool is) and *checkout wait time* (how long requests queue for a connection) — everything else explains *why*.**

A connection pool fails loudly in production but quietly in your dashboards if you are not watching the right metrics. The breakdown of connection *states* is the foundation, and Postgres exposes it directly in `pg_stat_activity`:

- **active** — running a query right now. This is your $L$, the busy connections.
- **idle** — connected but not in a transaction, ready to be reused. Healthy.
- **idle in transaction** — connected, inside an open transaction, but not running a query. This is the dangerous one: an idle-in-transaction connection holds locks and pins the `xmin` horizon, which blocks `VACUUM` from cleaning up dead tuples (a direct tie-in to the [MVCC deep dive](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb) and a frequent cause of the lock pileups in [database locks and deadlocks in production](/blog/software-development/database/database-locks-and-deadlocks-in-production)). A rising idle-in-transaction count is almost always an application bug — a transaction left open while the app does slow work or an external call.

![Pool saturation and checkout-wait time are the leading indicators; the active, idle, and waiting connection split tells you why](/imgs/blogs/database-connection-pooling-8.webp)

The grid above is the dashboard to build. The top row is the alarms — saturation, checkout-wait p99, checkout timeouts — and the lower rows are the diagnostic breakdown that tells you the cause. Concretely:

- **Pool saturation** = busy connections ÷ pool size. Alert above ~85%: you are about to start queueing requests.
- **Checkout wait time (p99)** — the time a request spends waiting to acquire a connection. This is the metric users feel: if it climbs above a few tens of milliseconds, the pool is too small for the offered load *or* something is holding connections too long. HikariCP exposes this directly; pgx and SQLAlchemy can be instrumented around acquire.
- **Checkout timeouts** — count of requests that gave up waiting (`PoolTimeout`, `connection-timeout`, `query_wait_timeout`). Should be essentially zero in steady state; any sustained nonzero rate is a capacity or leak problem.
- **idle-in-transaction count and age** — alert on long-lived ones; set `idle_in_transaction_session_timeout` on the server as a backstop.
- **Server backends vs `max_connections`** — the global headroom; watch it especially during deploys.
- **Connect/close rate** — the churn metric; a spike is a storm in progress.

The PgBouncer-side equivalents come from its admin console:

```sql
-- Connect to the PgBouncer admin "database" and run:
SHOW POOLS;   -- per (db,user): cl_active, cl_waiting, sv_active, sv_idle, sv_used, maxwait
SHOW STATS;   -- per db: total_query_count, total_query_time, avg_query, avg_wait
SHOW CLIENTS; -- every client conn and its state
SHOW SERVERS; -- every server conn and its state
```

The two columns to alarm on in `SHOW POOLS` are **`cl_waiting`** (clients queued for a server connection — your wait queue depth) and **`maxwait`** (the longest a client has currently been waiting). A persistently nonzero `cl_waiting` with a growing `maxwait` is PgBouncer telling you the backend pool is too small or backends are pinned. The application-side query to find idle-in-transaction offenders:

```sql
-- Long-running idle-in-transaction sessions: the silent VACUUM killers.
SELECT pid, usename, application_name, client_addr,
       state, now() - state_change AS idle_for, query
FROM pg_stat_activity
WHERE state = 'idle in transaction'
  AND now() - state_change > interval '30 seconds'
ORDER BY idle_for DESC;
```

## Case studies from production

The patterns above are best understood through the incidents that produced them. These are drawn from public engineering writeups and the documented behavior of the systems involved.

### 1. Instagram: PgBouncer to keep Postgres calm at scale

Instagram, running Django on Postgres, scaled to handle on the order of *thousands of likes per second* at peak with Postgres as their canonical data store — and a key part of how they kept it stable was PgBouncer. In their "Handling Growth with Postgres" engineering writeup, they describe using PgBouncer for connection pooling so that connections are returned to the pool sooner, freeing server resources for query execution and disk caching rather than for managing a sprawling set of connections. The lesson is the foundational one of this article: at high request rates, a Django-style "connection per web worker" model multiplies into far more Postgres backends than the database can run efficiently, and inserting a pooler that returns connections quickly is what keeps the backend count small and the server spending its CPU on queries instead of connection management. The fix was not a bigger database; it was *fewer, better-managed connections*.

### 2. GitLab: when the pooler itself becomes the bottleneck

GitLab.com runs PgBouncer in transaction-pooling mode in front of its Postgres fleet — one PgBouncer process per database instance — with `max_client_conn` defaulting to 2048 (the front-end pool from Rails) and `default_pool_size` at 100 (the back-end pool to Postgres). They hit a fascinating second-order limit: as they scaled up the web/worker fleet and raised `max_client_conn`, their PgBouncer processes "reached their CPU limits (pegging one core)," because **PgBouncer is single-threaded** and tops out around 15,000 queries per second per process on modern hardware. The pooler that protects Postgres became its own bottleneck. The lesson: a single-threaded pooler is a single core's worth of throughput, and at large scale you must either run multiple PgBouncer processes (sharded by database or behind a load balancer), move to a multi-threaded pooler (PgCat, Odyssey), or both. It is also why "just add a pooler" is necessary but not sufficient at the very top end — the pooler's own architecture becomes a capacity-planning input.

### 3. The Lambda connection explosion

A team moves an API to AWS Lambda and points it straight at their RDS Postgres instance. In dev and light traffic it works perfectly — a handful of warm Lambda instances reuse their in-function connections. Then a marketing push drives a traffic spike, Lambda scales to several hundred concurrent invocations, and each new execution environment opens its own connection. Within seconds Postgres returns `FATAL: remaining connection slots are reserved for non-replication superuser connections`, every new invocation errors, and (worse) the errors trigger client retries that pile on more connection attempts. The wrong first hypothesis is "the database is too small — scale it up," but a bigger instance only raises `max_connections` and delays the cliff. The actual root cause is that *serverless concurrency equals connection count with no shared pool*. The fix is RDS Proxy (or another transaction-mode pooler) between Lambda and Postgres, which terminates the hundreds of client connections at the proxy and multiplexes them onto a small, capped backend pool — plus disabling server-side prepared statements in the driver so the proxy can actually multiplex instead of pinning.

### 4. The 50-worker Gunicorn fleet that summed past max_connections

A Python team runs their app under Gunicorn with 50 sync workers per host, three hosts, SQLAlchemy with the default `pool_size=5` and `max_overflow=10`. The app is healthy until a deploy doubles the host count for a migration window. Suddenly Postgres throws `too many clients`. The wrong hypothesis is "the new hosts have a bug." The real cause is arithmetic: each worker is a separate process with its own pool, so the database sees up to `50 workers × 3 hosts × (5 + 10) = 2,250` possible connections, and doubling hosts pushed the worst case past `max_connections = 500`. The lesson, and the universal trap of app-side pools: **the database sees the *sum* of every process's pool, not a single shared pool.** The fixes are to drop the per-worker pool size dramatically (a sync worker handles one request at a time, so it needs at most one or two connections), and to put PgBouncer in front so the global backend count is bounded by `default_pool_size` regardless of how many workers exist.

### 5. The prepared-statement breakage after enabling transaction pooling

A team flips PgBouncer from session to transaction pooling to get better multiplexing, deploys, and immediately starts seeing intermittent errors: `prepared statement "S_3" does not exist`. They are intermittent because they only happen when a `PREPARE` and its `EXECUTE` land on different backends — which depends on pool timing, so it passes in staging and fails under production load. The wrong hypothesis is "PgBouncer is dropping our queries." The root cause is that their database driver (the PostgreSQL JDBC driver, or asyncpg, or pgx in its default mode) uses the extended protocol with *named server-side prepared statements* by default, and transaction pooling cannot guarantee the same backend across the prepare/execute pair. The fix is one of: set `prepareThreshold=0` (JDBC), `statement_cache_size=0` (asyncpg) or `default_query_exec_mode=exec`/simple protocol (pgx), or upgrade PgBouncer and set `max_prepared_statements` to a nonzero value so PgBouncer tracks protocol-level prepared statements per backend. The lesson: transaction pooling silently changes the contract your driver was relying on.

### 6. The search_path tenant leak

A multi-tenant SaaS isolates tenants by setting `search_path` to a per-tenant schema at the start of each request: `SET search_path = tenant_42`. Under session pooling this is fine. They move to transaction pooling for multiplexing, and within hours a customer reports seeing another customer's data. The cause is chilling in its simplicity: `SET search_path` is *session* state on a backend, and under transaction pooling the backend is returned to the pool after each transaction *carrying that search_path*, then handed to a different tenant's next transaction — which now reads `tenant_42`'s schema. This is a data-isolation breach caused purely by a pool-mode change. The fix is to never rely on session `SET` under transaction pooling: use `SET LOCAL search_path = tenant_42` *inside* each transaction (which resets at `COMMIT`), or pass the schema explicitly, or use a connection-per-tenant pool. The broader lesson: under transaction pooling, *any* server-side session state is a cross-tenant leak waiting to happen, and `SET` is the most dangerous because it is invisible and silent.

### 7. The deploy that DDoSed the database

A team runs ~200 application pods, each opening 10 connections directly to Postgres on startup (`max_connections = 2500`, comfortably above the steady-state 2,000). A routine rolling deploy replaces all 200 pods. Because the orchestrator's `maxSurge` was set aggressively and pods eagerly filled their pools on boot, ~150 pods came up within a two-second window and tried to open ~1,500 new connections *while* the old pods had not yet released theirs. Postgres briefly needed to hold ~3,000 connections, blew past `max_connections`, and started rejecting — and the rejected pods retried in a tight loop, extending the outage to several minutes of flapping. The wrong hypothesis is "Postgres is unstable." The root cause is a *synchronized* connection storm with no jitter and no backoff. The fixes: put PgBouncer in front so the storm hits a pooler with a fixed `default_pool_size` instead of `max_connections`; add jitter to pool-fill and reconnect; cap `maxSurge` so the rollout proceeds in smaller waves; and use exponential backoff with jitter on connect failures so a transient rejection does not become a retry storm.

### 8. The oversized pool that was slower than a small one

A team is convinced their database is the bottleneck because p99 latency is high under load, so they *increase* the application's pool from 20 to 200 connections per host across 5 hosts — reasoning that more connections means more parallelism. Latency gets *worse*. The wrong hypothesis is "we still don't have enough connections." The actual cause is exactly the HikariCP/Postgres result: with `(20-core) × 2 = 40`-ish useful slots, the database was already near its sweet spot at the old setting, and 1,000 concurrent backends oversubscribed 20 cores by 50×, so the machine spent its time context-switching, contending on LWLocks and buffer-mapping locks, and evicting each other's cache lines. They reverted to a *small* pool (closer to 30 per host, then funneled through PgBouncer with `default_pool_size = 40`), and p99 dropped dramatically — the same shape as Oracle's 100 ms → 2 ms result. The lesson is the counterintuitive heart of this article: when latency is high under load, the pool is often *too big*, not too small, and the cure is to shrink it and let requests queue.

### 9. The idle-in-transaction connection that froze VACUUM

A reporting endpoint opens a transaction, runs a query, then makes a slow external HTTP call *before* committing — leaving the connection `idle in transaction` for several seconds per request. Under modest load there are always a few such connections open. Over a week, autovacuum stops being able to clean up dead tuples on a hot table, the table bloats, queries against it slow down, and eventually the database is in trouble. The wrong hypothesis is "autovacuum is misconfigured." The root cause is that an idle-in-transaction connection holds its transaction's snapshot, pinning the `xmin` horizon so `VACUUM` cannot remove tuples newer than the oldest open transaction — a direct consequence of [MVCC](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb). The fixes: never hold a transaction open across an external call (commit first, then do the slow work); set `idle_in_transaction_session_timeout` to forcibly kill long idle-in-transaction sessions; and alarm on the idle-in-transaction count and age in the monitoring grid above. The pool made it worse by keeping those connections alive and reusing them, so the leak persisted instead of being closed.

### 10. Supavisor and the million-connection serverless edge

Supabase's customers run a lot of serverless and edge workloads — exactly the "concurrency equals connections" pattern — against managed Postgres. Their first-generation pooler could not keep up with the client counts, so they built **Supavisor**, a multi-tenant, cloud-native transaction pooler in Elixir, explicitly to multiplex on the order of a *million* client connections across tenants onto small per-tenant backend pools. The design choices are instructive: Elixir/BEAM for massive lightweight concurrency and fast I/O, Rust (via Rustler) for the CPU-bound SQL parsing because the BEAM is not great at parsing, and a dynamic per-tenant pool that spins up backends on first connect and distributes routing state across the cluster. Platform limits scale with instance size — from a couple hundred max clients on the smallest tier up to ~12,000 on the largest. The lesson generalizes beyond Supabase: at the serverless/edge frontier, the pooler is not an add-on, it is the primary scaling component, and its architecture (multi-threaded or BEAM-style concurrency, multi-tenant routing) is what determines how many clients the whole platform can hold.

### 11. The single-connection-per-thread deadlock

A batch job runs 8 worker threads against a pool sized at 8. Each worker, mid-task, needs a *second* connection to look up a reference table while holding its first. Occasionally the whole job hangs forever. The wrong hypothesis is "a query is stuck." The root cause is a pool deadlock: all 8 workers grab their first connection (pool now empty), then all 8 block waiting for a second connection that none of them will release until they get it — a circular wait. This is exactly the scenario Wooldridge's pool-locking formula addresses: with $T_n = 8$ threads each needing $C_m = 2$ connections, the minimum safe pool is $8 \times (2 - 1) + 1 = 9$, not 8 — the `+1` guarantees at least one thread can always acquire its second connection, finish, and release, unblocking the rest. The better fix is to restructure so a unit of work never holds two connections at once, but when that is impractical, the formula is the floor. The lesson: pool size interacts with *how many connections a single logical task holds*, and a pool that is "big enough" for throughput can still be too small to avoid deadlock.

### 12. The keepalive that wasn't, behind a NAT

A service runs fine for hours, then a burst of requests after a quiet period all fail with "connection reset" before the pool recovers. The wrong hypothesis is "the database restarted." The root cause is a stateful firewall / NAT gateway between the app and the database that silently drops idle TCP connections after a few minutes — so the pool's "idle" connections are actually dead, and the app only discovers this when it tries to use them after a quiet period, paying a failure plus a reconnect on each one. The fixes are a layered defense: enable TCP keepalive / a pool keepalive ping (`keepaliveTime` in HikariCP, `HealthCheckPeriod` in pgx) so the pool exercises idle connections before the NAT times them out; enable `pool_pre_ping` (SQLAlchemy) or equivalent so a dead connection is detected and replaced *before* it is handed to a request; and set `max-lifetime`/`MaxConnLifetime` below the NAT's idle timeout so connections are proactively recycled. The lesson: an "idle" connection in your pool is only as alive as the network path keeps it, and the pool must actively prove liveness rather than assume it.

## When to reach for an external pooler — and when not to

The decision of *where* to pool — app-side only, or app-side plus an external pooler — is the one most teams get wrong, usually by adding PgBouncer too early or too late. Here is the calibration.

**Reach for an external pooler (PgBouncer / PgCat / RDS Proxy / Supavisor) when:**

- You run **many application processes** (lots of pods, workers, or hosts) whose pools *sum* to more connections than the database can run — the moment `processes × per_process_pool > max_connections / 2`, you need a funnel.
- You run **serverless** (Lambda, Cloud Functions, edge) where there is no persistent process to hold a pool and concurrency equals connections — here a transaction-mode proxy is essentially mandatory, not optional.
- You are hitting `max_connections` or seeing connection storms on deploy/failover, and you need a fixed, small backend pool that absorbs surges and queues overflow.
- Your steady-state offered concurrency vastly exceeds your hardware ceiling (`(cores × 2) + spindles`) and you need transaction-level multiplexing to keep the backend count near that ceiling while serving thousands of clients.
- You want connection management (limits, queueing, fast failover) to be a *centrally operated* concern rather than configured identically across dozens of services.

**Skip the external pooler (app-side pool is enough) when:**

- You have a **small, fixed number of long-lived application processes** whose combined pool comfortably fits under `max_connections` with headroom — a monolith on a few hosts often needs nothing more than a well-sized HikariCP/pgx/SQLAlchemy pool.
- Your application depends on **session-scoped features** — `LISTEN`/`NOTIFY`, session advisory locks, `WITH HOLD` cursors, heavy reliance on session `SET` — and you are not prepared to refactor them, since transaction pooling will break them and session pooling buys little multiplexing.
- The added **operational surface** (another process to deploy, monitor, secure, and capacity-plan — remember GitLab's single-threaded CPU ceiling) is not justified by your scale.
- You need **strict per-connection semantics** end-to-end and cannot tolerate the subtle behavior changes of multiplexing.

And the universal rules that hold regardless of topology:

> Size the pool from the hardware ceiling `(cores × 2) + effective_spindles` and from Little's Law `L = λW`, take the smaller-but-safe number, and *resist every instinct to make it bigger when latency rises* — high latency under load usually means the pool is already too large.

> The database sees the sum of every pool. Always do the arithmetic across all processes before you set a per-process pool size, and put a pooler in front the moment that sum threatens `max_connections`.

> Under transaction pooling, every byte of server-side session state is a bug waiting to happen. Make every transaction self-contained, disable server-side prepared statements (or use `max_prepared_statements`), and use `SET LOCAL` and `pg_advisory_xact_lock`, never their session-scoped cousins.

> Storms come from synchrony. Jitter every reconnect and pool-fill, back off exponentially on failure, reserve a few `max_connections` slots for humans, and let a pooler absorb the herd.

A connection pool is the quiet adapter that makes a 16-core database serve tens of thousands of clients. It works because of one fact that runs against intuition at every turn: the database is fastest when a *small* number of connections are kept busy and everyone else waits politely in line. Build for that, size for that, monitor for that, and the "too many clients" page becomes a memory.

## Further reading

- Brett Wooldridge, [HikariCP — About Pool Sizing](https://github.com/brettwooldridge/HikariCP/wiki/About-Pool-Sizing): the Oracle 10k-user benchmark, the `(cores × 2) + spindles` formula, and the pool-locking formula.
- PostgreSQL wiki, [Number Of Database Connections](https://wiki.postgresql.org/wiki/Number_Of_Database_Connections): the formula, why too many connections hurt, and the `max_connections` headroom advice.
- [PgBouncer features and pool modes](https://www.pgbouncer.org/features.html): session vs transaction vs statement, and the list of features incompatible with transaction pooling.
- Andres Freund, [Measuring the Memory Overhead of a Postgres Connection](https://blog.anarazel.de/2020/10/07/measuring-the-memory-overhead-of-a-postgres-connection/): the corrected per-connection memory numbers (≈1.3–7.6 MB private).
- Instagram Engineering, [Handling Growth with Postgres: 5 Tips From Instagram](https://instagram-engineering.com/handling-growth-with-postgres-5-tips-from-instagram-d5d7e7ffdfcb): PgBouncer at scale.
- AWS, [RDS Proxy connection considerations](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/rds-proxy-connections.html): multiplexing, pinning, `MaxConnectionsPercent`, borrow timeout.
- Supabase, [Supavisor: a scalable connection pooler for Postgres](https://supabase.com/blog/supavisor-postgres-connection-pooler): multi-tenant transaction pooling for serverless/edge at huge client counts.
- GitLab Infrastructure, [More scalable database connection pooling (#6981)](https://gitlab.com/gitlab-com/gl-infra/production-engineering/-/issues/6981): when the pooler itself becomes the single-threaded bottleneck.
- Martin Kleppmann, *Designing Data-Intensive Applications*, Ch. 1 — load parameters, latency percentiles, and tail latency, the conceptual backdrop for why queueing beats oversubscription.
