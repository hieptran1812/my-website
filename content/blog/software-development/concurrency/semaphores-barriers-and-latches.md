---
title: "Semaphores, Barriers, and Latches"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Cap your database connections at ten with a semaphore, learn why a permit pool has no owner, and reach for barriers and latches to make N threads start and stop together."
tags:
  [
    "concurrency",
    "parallelism",
    "semaphore",
    "barrier",
    "latch",
    "synchronization",
    "connection-pool",
    "coordination",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/semaphores-barriers-and-latches-1.png"
---

Your service has a database that accepts at most ten connections. Open an eleventh and the database refuses it, your request throws, a customer sees a 500. So you write a comment in the README — "please do not open more than ten connections" — and you ship, and for three weeks it is fine, and then a traffic spike fans out forty requests in the same hundred milliseconds, each one cheerfully opening its own connection, and the database starts handing back `FATAL: sorry, too many clients already`. The comment did not help. A comment is not a mechanism.

What you needed was a way to say, in code that the runtime actually enforces: *at most ten threads may be inside this region at once, and the eleventh must wait politely until one of the ten leaves.* That is exactly what a **counting semaphore** is. You hand it a budget of ten **permits**; a thread must acquire a permit before it touches the connection, and it returns the permit when it is done; if all ten permits are out, the next acquirer blocks until somebody returns one. The semaphore is the mechanism the comment wished it were. The figure below shows the before and after — the uncapped version melting down at the eleventh connection, the capped version letting ten run and parking the eleventh.

![A semaphore with ten permits admits ten threads and forces the eleventh to wait for a release before it can run](/imgs/blogs/semaphores-barriers-and-latches-1.png)

This post is about the coordination primitives that live *beyond* the mutex. A mutex answers exactly one question — "is anyone in the critical section right now?" — and admits exactly one thread. That is the right tool when you are protecting a single invariant, and the [previous post on mutual exclusion](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections) covered it. But a great many real coordination problems are not "one thread at a time." They are "at most *N* at a time" (a connection pool, a rate limiter, a bounded worker fleet). Or "every thread waits here until *all* of them arrive" (a barrier between two phases of a computation). Or "the main thread blocks until *N* startup tasks have all reported ready" (a latch). The semaphore, the barrier, and the latch are the three primitives that solve those problems, and by the end of this post you will know exactly which one to reach for, how each is built underneath, and what each costs when you measure it.

This is the same spine the whole series runs on: shared mutable state plus nondeterministic scheduling is the hazard, and the discipline is to name what is shared, establish a happens-before order over the accesses, and pick the cheapest mechanism that buys that order. A semaphore is one of those mechanisms. The intro post, [why concurrency is hard and why you cannot avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it), frames the hazard; the [capstone playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model) is where you decide between all the mechanisms. This post is the chapter on coordinating *counts* and *phases* rather than mutual exclusion.

## The semaphore: a count, plus P and V

The semaphore is one of the oldest ideas in concurrency, invented by Edsger Dijkstra around 1962–1963 for the THE operating system. It is almost embarrassingly simple. A semaphore is an integer count plus two operations, and Dijkstra named them after Dutch words so that nobody could argue about what they meant:

- **P** (from *proberen*, "to test", or *passeren*, "to pass") — also called `wait`, `acquire`, or `down`. It tries to decrement the count. If the count is greater than zero, it decrements and returns immediately. If the count is zero, the calling thread **blocks** until the count becomes positive, then decrements and returns.
- **V** (from *verhogen*, "to increment") — also called `signal`, `release`, `post`, or `up`. It increments the count. If any threads are blocked in `P`, one of them is woken and allowed to proceed.

That is the whole interface. Everything else — connection pools, rate limiters, barriers, producer–consumer queues — is built on those two operations and a starting count. Let me state the mechanism precisely, because the whole point of this series is that the *why* is provable, not asserted.

### The invariant

Call the initial count $S_0$. Let $P$ be the number of completed `P` (acquire) operations and $V$ the number of completed `V` (release) operations at any instant. The defining invariant of a counting semaphore is:

$$\text{count} = S_0 + V - P \ge 0$$

The count never goes negative. That single inequality is the entire safety guarantee. Rearrange it:

$$P - V \le S_0$$

The number of threads that have acquired but not yet released — the number currently *inside* the guarded region — is at most $S_0$. If you initialize the semaphore to ten, then at most ten threads are inside at once, *always*, no matter how the scheduler interleaves them. That is the property the README comment could never enforce, now expressed as an invariant the runtime maintains.

How does `P` keep the count from going negative when it would? It blocks. When `count` is zero and a thread calls `P`, the operation does not decrement to $-1$ and continue; it parks the thread on a wait queue associated with the semaphore. The arithmetic `count = count - 1` is *conditional* on `count > 0`. When some other thread later calls `V`, it increments the count and wakes one parked thread, which then completes its decrement. The blocking is what makes the inequality hold.

The second figure traces the count through a concrete sequence: a semaphore starting at two, three threads racing to acquire it.

![A timeline of acquire and release operations on a counting semaphore starting at two, where the third acquirer blocks and a later release wakes it](/imgs/blogs/semaphores-barriers-and-latches-2.png)

#### Worked example: the count never lies

Start with `count = 2`.

1. **T1 calls P.** `count` is 2, positive, so it decrements to 1 and T1 proceeds. Now $P = 1$, $V = 0$, $\text{count} = 2 + 0 - 1 = 1$. Inside: 1 thread (T1).
2. **T2 calls P.** `count` is 1, positive, decrements to 0, T2 proceeds. $P = 2$, $V = 0$, $\text{count} = 0$. Inside: 2 threads.
3. **T3 calls P.** `count` is 0. T3 cannot decrement — that would make the count $-1$ and violate the invariant. So T3 **blocks** on the wait queue. The count stays at 0. Inside: still 2.
4. **T1 calls V.** It increments `count` to 1 and wakes T3. $V = 1$. T3's pending `P` now completes: it decrements `count` back to 0 and T3 proceeds. $P = 3$, $V = 1$, $\text{count} = 2 + 1 - 3 = 0$. Inside: 2 threads (T2 and T3).

At every single step, `count` $= S_0 + V - P \ge 0$, and the number inside never exceeded $S_0 = 2$. The semaphore did its one job: it kept a count, and it refused to let the count lie.

There is one subtlety worth pinning down now because it bites people later. The count can be initialized to **zero**, which means `P` will block immediately. That is not a degenerate case — it is exactly how you use a semaphore for *signaling* rather than *resource limiting*, which we will come back to. And in some implementations the count is allowed to be conceptually negative in the sense that its magnitude tells you how many threads are waiting; POSIX `sem_getvalue`, for instance, may report a negative number equal to the count of blocked waiters on some systems. The user-visible behavior is the same: `P` blocks when there is nothing to take.

## Binary semaphore versus mutex: the ownership question

A semaphore initialized to one is called a **binary semaphore**, because its count only ever toggles between 0 and 1. At first glance this looks identical to a mutex: it admits one thread, the second blocks, the first releases and the second proceeds. People routinely reach for a binary semaphore as if it were a mutex. It is not, and the difference is the single most important thing in this whole post: **a semaphore has no concept of ownership.**

A mutex is *owned*. When a thread locks a mutex, the runtime records *which* thread holds it. Only that thread is allowed to unlock it. Try to unlock a mutex you do not hold and you get an error — `EPERM` from `pthread_mutex_unlock` on an error-checking mutex, an `IllegalMonitorStateException` in Java, undefined behavior or a panic in others. This is not bureaucracy. Ownership is what makes three crucial things possible:

- **Reentrancy / recursion.** Because the mutex knows you already hold it, a recursive mutex can let you re-lock it without deadlocking yourself.
- **Priority inheritance.** Because the mutex knows who holds it, the scheduler can temporarily boost the holder's priority so a low-priority lock holder does not block a high-priority waiter forever. This is the fix for the priority-inversion bug that famously froze the Mars Pathfinder rover, which we will get to.
- **Correctness checks.** "You unlocked a lock you do not own" is almost always a bug, and an owned mutex catches it for you at runtime.

A semaphore has none of this, because a semaphore tracks a *count*, not an *owner*. Any thread may call `V`. Thread A can acquire a permit and thread B can release it, and the semaphore is perfectly happy — it just sees the count go down by one and up by one. The figure below lays the two side by side.

![A matrix comparing a mutex and a counting semaphore across ownership, counting past one, release by another thread, and typical use](/imgs/blogs/semaphores-barriers-and-latches-3.png)

Here is the same comparison as a table you can keep next to your editor:

| Property | Mutex | Counting semaphore |
| --- | --- | --- |
| Has an owner | Yes — runtime records the holder | No — it is just a count |
| Count can exceed 1 | No (binary, 0 or 1) | Yes (up to $N$) |
| Can be released by another thread | No — only the owner unlocks | Yes — any thread may `V` |
| Reentrant / recursive | Possible (recursive mutex) | No |
| Priority inheritance | Possible | No |
| Best used for | Protecting a critical section / invariant | Limiting $N$ concurrent users, or signaling |

The lack of ownership is the **point** of a semaphore *and* its **footgun**, and which one it is depends entirely on what you are doing.

It is the **point** when you want signaling: a producer thread should be able to `V` a semaphore that a consumer thread will `P`. The two operations *must* happen on different threads — that is the whole idea. A mutex literally cannot express this, because the consumer does not own the thing the producer is signaling.

It is the **footgun** when you reach for a binary semaphore where you meant a mutex. Now you have a "lock" with no owner. Any thread can release it, including one that never acquired it, including the same thread releasing it twice. If a bug causes a stray `V`, the count goes to 2, two threads enter your "critical section" simultaneously, and your invariant is violated with no error and no warning — the semaphore was doing exactly what semaphores do. The mutex would have thrown the instant somebody unlocked it wrongly. The rule, then:

> **If you need mutual exclusion, use a mutex.** Use a semaphore when you genuinely need to count past one (a resource pool) or when you need cross-thread signaling (one thread releases what another acquires). Do not use a binary semaphore as a cheap mutex; you are throwing away ownership checks for nothing.

There is also a quiet *spelling* trap here. In some older APIs and textbooks "binary semaphore" and "mutex" are used as synonyms, and in a few embedded RTOSes the only "lock" you get is literally a binary semaphore. That historical overlap is where the confusion comes from. The way to keep yourself honest is to ask the ownership question every time: *will the same thread that takes the lock be the one that releases it, and is the answer to "how many at once" exactly one?* If yes to both, you want a mutex — name it a mutex, use a mutex type, get the ownership checks. If the releaser might be a different thread, or the count is more than one, you are in semaphore territory and a mutex would be the wrong tool. The names matter less than the property; reason about the property.

## How a semaphore is built underneath

Before the permit pool, it is worth peeling back one layer, because the series' contract is to make the mechanism rigorous, not to wave at it. A semaphore is not magic — it is a small integer guarded by an atomic operation, plus a way to park and wake threads. Knowing how it is built tells you exactly why `acquire` is cheap on the fast path and expensive on the slow path, which is the single most important performance fact about it.

Consider the *fast path* — acquiring a permit when one is free. A naive implementation would grab an internal mutex, check the count, decrement it, and release the mutex. That works but it means *every* acquire, even an uncontended one, pays for a lock. Real implementations avoid that with a **compare-and-swap (CAS) loop** on the count itself, which we covered when we built [a lock from test-and-set and CAS](/blog/software-development/concurrency/how-a-lock-is-built-test-and-set-cas-and-spinlocks). The acquire reads the count, and if it is positive, it tries to atomically swap `count` for `count - 1` with a single `compare_exchange`. If the swap succeeds, the permit is yours and you never touched a kernel — the whole operation was a handful of user-space instructions:

```cpp
// Fast-path acquire: try to decrement the count atomically, no kernel call.
bool try_acquire_fast(std::atomic<int>& count) {
    int cur = count.load(std::memory_order_relaxed);
    while (cur > 0) {
        // atomically: if count is still 'cur', set it to cur - 1
        if (count.compare_exchange_weak(cur, cur - 1,
                                        std::memory_order_acquire)) {
            return true;                  // got a permit, never blocked
        }
        // CAS failed: another thread changed count; cur was refreshed, retry
    }
    return false;                         // count was 0: must take the slow path
}
```

The `compare_exchange_weak` maps to a single `cmpxchg` instruction on x86 (or an `LL/SC` pair on ARM). On the uncontended fast path, acquiring a permit costs roughly the price of one atomic read-modify-write — on the order of tens of nanoseconds, no system call, no context switch.

The *slow path* — acquiring when the count is zero — is where it gets expensive, because now the thread must actually **block**, and blocking means involving the kernel scheduler. On Linux the primitive underneath is the **futex** (fast userspace mutex). When the CAS loop sees a zero count, the thread calls `futex(FUTEX_WAIT)`, which tells the kernel "park me until someone signals this address." The kernel removes the thread from the run queue, so it consumes no CPU while parked. A `release` that sees waiters calls `futex(FUTEX_WAKE)`, which moves one parked thread back to the run queue. The genius of the futex design — and the reason your semaphore is fast — is that the kernel is involved *only on the slow path*: an uncontended acquire/release never makes a system call at all, and only contention (count hit zero, someone had to wait) pays the kernel cost.

This split is the whole performance story in one sentence: **a semaphore is cheap when there are permits and expensive when there are not.** That cost asymmetry has a concrete number attached. An uncontended acquire is tens of nanoseconds; a *contended* acquire that actually parks and later wakes pays for two system calls plus a context switch, which is on the order of one to a few microseconds — roughly a hundredfold more. This is why sizing the permit pool correctly matters so much: if you set the pool so small that threads constantly hit the zero-count slow path, you are paying microsecond context-switch costs on every request; if you set it large enough that the fast path usually wins, acquire is nearly free. The measurement section later puts numbers on exactly where that knee sits.

## The counting semaphore as a permit pool

Now the use that earns the semaphore its place in your toolbox: limiting concurrency to $N$. The mental model is a bowl of $N$ tickets. To use the shared resource you must take a ticket; when you are done you put it back; if the bowl is empty you wait until someone returns a ticket. The semaphore *is* the bowl, the count is the number of tickets currently in it, `P` takes a ticket, `V` returns one.

The canonical example is a **connection pool**. You have ten physical database connections. You have hundreds of request threads. You want each request to grab a connection, run its query, and give the connection back, and you want *at most ten* to be in flight at once. A semaphore initialized to ten, guarding a queue of ten connection objects, does exactly this. The figure traces a single request through the pool — the fast path when a permit is free, the slow path when it must queue, and the release that wakes the next waiter.

![A connection pool where a request acquires a permit then uses a connection or queues, and the release converges to wake one waiter](/imgs/blogs/semaphores-barriers-and-latches-4.png)

Let me write it for real, because the bug-then-fix here is one of the most common production bugs there is. Start with Java, which has a first-class `Semaphore` in `java.util.concurrent`.

#### Worked example: the leaked permit

Here is a connection pool with a **bug** — a leaked permit. Read it before the fix.

```java
import java.util.concurrent.Semaphore;
import java.util.concurrent.ConcurrentLinkedQueue;

public class BuggyPool {
    private final Semaphore permits = new Semaphore(10);   // 10 tickets
    private final ConcurrentLinkedQueue<Connection> pool;

    public BuggyPool(ConcurrentLinkedQueue<Connection> conns) {
        this.pool = conns;
    }

    public ResultSet runQuery(String sql) throws InterruptedException {
        permits.acquire();                 // P: take a permit (blocks if 0)
        Connection c = pool.poll();        // grab a connection
        ResultSet rs = c.execute(sql);     // <-- if this THROWS, we skip release
        pool.offer(c);                     // return the connection
        permits.release();                 // V: return the permit
        return rs;
    }
}
```

The bug is the exception path. `c.execute(sql)` can throw — a syntax error, a timeout, a dropped network connection. When it does, control leaves `runQuery` via the exception, the connection is never returned to the pool, *and the permit is never released*. The count stays one lower than it should. Do this a few times under load and the count drifts down to zero. Now `permits.acquire()` blocks forever for every request, because no release will ever come. Your service hangs with the database completely idle. The pool is "exhausted" even though no connection is actually in use. This is a **permit leak**, and it is the single most common semaphore bug in production.

The fix is `try/finally`. The release must run on *every* exit path, including the exceptional one.

```java
import java.util.concurrent.Semaphore;
import java.util.concurrent.ConcurrentLinkedQueue;

public class FixedPool {
    private final Semaphore permits = new Semaphore(10);
    private final ConcurrentLinkedQueue<Connection> pool;

    public FixedPool(ConcurrentLinkedQueue<Connection> conns) {
        this.pool = conns;
    }

    public ResultSet runQuery(String sql) throws InterruptedException {
        permits.acquire();                 // P: take a permit
        Connection c = null;
        try {
            c = pool.poll();
            return c.execute(sql);         // may throw — that's fine now
        } finally {
            if (c != null) pool.offer(c);  // always return the connection
            permits.release();             // V: always return the permit
        }
    }
}
```

Now no matter how `runQuery` exits — normal return, exception, even an `Error` — the `finally` block runs, the connection goes back, and the permit is released. The permit count is conserved. This is the same discipline as `defer`-ing an unlock in Go or RAII in C++: **the release must be tied to the scope, not to the happy path.**

A note on `acquire` versus `tryAcquire`. The bare `permits.acquire()` blocks indefinitely. In a real service you almost never want that — you want a *bounded wait*, so that under overload requests fail fast instead of piling up. Java's `Semaphore` gives you `tryAcquire(timeout, unit)`:

```java
public ResultSet runQuery(String sql) throws InterruptedException, PoolBusyException {
    if (!permits.tryAcquire(200, java.util.concurrent.TimeUnit.MILLISECONDS)) {
        throw new PoolBusyException("no connection within 200ms");  // shed load
    }
    Connection c = null;
    try {
        c = pool.poll();
        return c.execute(sql);
    } finally {
        if (c != null) pool.offer(c);
        permits.release();
    }
}
```

If a permit does not come free within 200 ms, the request fails fast. This is **load shedding**, and it is the local cousin of the architecture-scale [rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure) techniques — the semaphore is, in fact, the simplest possible concurrency limiter, and a concurrency limiter is one of the cleanest forms of backpressure: you bound the *in-flight* work directly.

Now the same idea in **Go**, where there is no `Semaphore` type because the idiom is a **buffered channel**. A buffered channel of capacity $N$ *is* a counting semaphore: a send (`ch <- struct{}{}`) is `P` (it blocks when the buffer is full), and a receive (`<-ch`) is `V`.

```go
package pool

// A buffered channel of capacity N is a counting semaphore with N permits.
type ConnPool struct {
    sem   chan struct{}   // capacity 10 == 10 permits
    conns chan *Conn      // the actual connections
}

func NewConnPool(conns []*Conn) *ConnPool {
    p := &ConnPool{
        sem:   make(chan struct{}, len(conns)),
        conns: make(chan *Conn, len(conns)),
    }
    for _, c := range conns {
        p.conns <- c
    }
    return p
}

func (p *ConnPool) RunQuery(sql string) (rs *Result, err error) {
    p.sem <- struct{}{}          // P: acquire a permit (blocks if full)
    c := <-p.conns               // take a connection
    defer func() {               // V on EVERY exit path, even a panic
        p.conns <- c             //   return the connection
        <-p.sem                  //   release the permit
    }()
    return c.Execute(sql)        // may error or panic — defer still runs
}
```

The `defer` is Go's `try/finally`: it runs whether `RunQuery` returns normally or panics, so the permit is conserved exactly as in the Java fix. Notice the Go version makes the "semaphore is a count of slots in a buffer" mechanism *literal* — the channel's buffer length is the count. (If you want a real timeout, you wrap the `p.sem <- struct{}{}` send in a `select` with a `time.After` case.)

For completeness in a third language, **C++20** added `std::counting_semaphore`, which is the standardized version of the same thing:

```cpp
#include <semaphore>
#include <stdexcept>

class FixedPool {
    std::counting_semaphore<10> permits_{10};   // 10 permits, max 10
    ConnQueue pool_;
public:
    Result run_query(const std::string& sql) {
        permits_.acquire();                       // P
        Conn* c = pool_.pop();
        struct Guard {                            // RAII: release on any exit
            std::counting_semaphore<10>& s;
            ConnQueue& q; Conn* c;
            ~Guard() { q.push(c); s.release(); }  // V, even on exception
        } guard{permits_, pool_, c};
        return c->execute(sql);                   // may throw — guard still fires
    }
};
```

C++ has no `finally`, so the idiom is **RAII**: a small guard object whose destructor calls `release()`. Because C++ runs destructors during stack unwinding, the `release()` fires even when `execute` throws. Same conservation property, different syntax for "tie the release to the scope."

Three languages, one mechanism: a count of permits, acquire-decrements, release-increments, and the release *must* be bound to the scope so it survives the error path. For a fourth flavor — and to show the idiom where ownership is enforced by the type system rather than discipline — here is the same permit pool in **Rust**, using the `Semaphore` from `tokio` (async) with its scope-guard API, which makes the leak *impossible to write*:

```rust
use std::sync::Arc;
use tokio::sync::Semaphore;

async fn run_query(permits: Arc<Semaphore>, pool: &Pool, sql: &str) -> Result<Rows, DbError> {
    // acquire returns a PERMIT GUARD; dropping it releases automatically.
    let _permit = permits.acquire().await.expect("semaphore closed");
    let mut conn = pool.checkout().await?;          // grab a connection
    let rows = conn.execute(sql).await;             // may return Err
    // `_permit` (and `conn`) are dropped here, on EVERY exit path,
    // including the early-return from the `?` above — release is automatic.
    rows
}
```

The `acquire()` returns a `SemaphorePermit` guard, and Rust's ownership rules drop it — releasing the permit — at the end of the scope no matter how you leave, including the `?` early-return on error. There is no `finally`, no `defer`, no manual `release()` to forget: the leaked-permit bug is structurally unwriteable because the language ties the release to the value's lifetime. This is the same RAII idea as the C++ guard, but the compiler enforces it. When people say Rust's concurrency is "fearless," this is a small concrete piece of what they mean: the most common semaphore bug in the other three languages simply cannot compile here.

One more practical knob: **fairness**. When several threads are blocked on a drained semaphore and a permit becomes free, *which* waiter gets it? A *non-fair* (barging) semaphore lets any thread — including one that just arrived and never queued — grab the freed permit, which maximizes throughput but can **starve** a long-waiting thread indefinitely. A *fair* semaphore hands the permit to the longest-waiting thread (FIFO), which guarantees no starvation but costs a little throughput because it cannot let a newly-arriving thread barge ahead even when that would be faster. Java's `Semaphore` takes a boolean: `new Semaphore(10, true)` is fair, `new Semaphore(10)` is not. The default is non-fair, and for a connection pool that is usually right — you want maximum throughput and the waits are short — but if you have a latency SLA where *no* request may wait pathologically long, the fair variant buys you a bounded worst case at a small average cost. Measure both; the right choice depends on whether your tail latency or your throughput is the thing you are paid to protect.

## Using a semaphore to signal between threads

So far the semaphore has been a *resource counter*. Its other major use is *signaling*, and this is where the lack of ownership becomes a feature. Initialize the count to **zero** and the semaphore becomes a "wait for an event" primitive: a consumer thread calls `P` and blocks immediately (count is 0), and a producer thread later calls `V` to wake it.

Because `P` and `V` happen on *different* threads — one waits, the other signals — this is something a mutex fundamentally cannot do, and it is exactly why "any thread may release" is the right design for a semaphore. The classic example is the **producer–consumer** problem with a bounded buffer, which uses *two* semaphores:

```c
#include <semaphore.h>

#define N 16
sem_t empty;   // counts empty slots
sem_t full;    // counts filled slots
int   buf[N];
int   head = 0, tail = 0;

void init_buffer(void) {
    sem_init(&empty, 0, N);   // start with N empty slots
    sem_init(&full,  0, 0);   // start with 0 filled slots
}

void producer(int item) {
    sem_wait(&empty);         // P(empty): need an empty slot, blocks if full
    buf[tail] = item;
    tail = (tail + 1) % N;
    sem_post(&full);          // V(full): signal a filled slot
}

int consumer(void) {
    sem_wait(&full);          // P(full): need a filled slot, blocks if empty
    int item = buf[head];
    head = (head + 1) % N;
    sem_post(&empty);         // V(empty): signal a freed slot
    return item;
}
```

Look at what is happening. The `empty` semaphore counts free slots; a producer must take one before writing (`P(empty)`), and a consumer returns one after reading (`V(empty)`). The `full` semaphore counts filled slots; the consumer takes one before reading (`P(full)`), and the producer returns one after writing (`V(full)`). When the buffer is full, `empty` is 0 and the producer blocks. When the buffer is empty, `full` is 0 and the consumer blocks. **The producer signals the consumer and the consumer signals the producer, on different threads, through ownerless semaphores.** This is the foundation of every bounded queue, and it is the local-memory cousin of [backpressure and flow control](/blog/software-development/message-queue/backpressure-and-flow-control) in a message queue: a full `empty` semaphore *is* backpressure on the producer.

(One caveat for the careful reader: the code above is correct for a *single* producer and *single* consumer. With multiple producers or consumers you also need a mutex around the `buf`/`head`/`tail` mutations, because two producers could otherwise race on `tail`. The semaphores handle the *counting* — how many slots — but not the *mutual exclusion* on the shared indices. That is precisely the division of labor the series keeps returning to: the semaphore orders the *count*, a mutex orders the *data*.)

If you have used Go, you have seen this signaling pattern dressed up as an unbuffered channel: a send blocks until a receive happens, which is a rendezvous, which we will define shortly. If you have used Python's `asyncio.Semaphore` to limit concurrent HTTP requests, that is the *resource counter* use in an event loop; the Python concurrency story has its own home in the [asyncio from the ground up](/blog/software-development/python-performance/asyncio-from-the-ground-up-event-loops-and-coroutines) post, and I will not re-derive it here — the *concept* is identical, only the scheduler differs (coroutines yielding to an event loop instead of OS threads parking).

## The barrier: everyone waits for everyone

Switch problems. You are running a parallel simulation. Every thread owns a chunk of a grid. Each *step*, a thread reads its neighbors' values, computes its new value, and writes it. The catch: a thread must not start step $k+1$ until *every* thread has finished step $k$, or it will read half-updated neighbor data and the simulation diverges. You need a point in the code where all $N$ threads stop, wait for the slowest one to catch up, and then all resume together. That point is a **barrier**.

A barrier is a synchronization point for a *group* of threads. Each thread calls `barrier.wait()` (often named `await` or `arrive_and_wait`) when it reaches the barrier. The first $N-1$ callers block. The $N$-th caller — the last to arrive — trips the barrier, and all $N$ threads are released simultaneously. The figure shows three threads arriving at staggered times and all being released the instant the third arrives.

![A timeline where three threads arrive at a barrier at different times and all are released together when the last one arrives](/imgs/blogs/semaphores-barriers-and-latches-6.png)

#### Worked example: the slowest thread sets the pace

Three threads, one barrier, `N = 3`.

1. **t1:** T1 finishes its work and calls `barrier.wait()`. The arrival count goes from 0 to 1. $1 < 3$, so **T1 blocks**.
2. **t2:** T2 calls `barrier.wait()`. Arrival count 1 → 2. $2 < 3$, so **T2 blocks** too. Two threads now parked.
3. **t3:** T3 — the slowpoke — calls `barrier.wait()`. Arrival count 2 → 3. $3 = 3$: the barrier **trips**. All three threads (T1, T2, T3) are released at once and proceed into the next phase.

The total time to cross the barrier is set by the *slowest* arrival, t3. This is the barrier's defining cost and a critical performance fact: **a barrier runs at the speed of your slowest thread.** If one thread takes twice as long every step, the other threads spend half their time idle at the barrier. That is why load *balance* matters so much in barrier-heavy code (bulk-synchronous parallel programs, MapReduce phases, stencil computations) — an imbalanced chunking starves your cores.

Here is a barrier in Java, where `CyclicBarrier` is the standard tool:

```java
import java.util.concurrent.CyclicBarrier;

class Simulation {
    final CyclicBarrier barrier;
    final double[][] grid;

    Simulation(int numThreads, double[][] grid) {
        this.grid = grid;
        // optional Runnable runs once, on the last thread, when the barrier trips
        this.barrier = new CyclicBarrier(numThreads, () -> swapBuffers());
    }

    void worker(int myChunk) throws Exception {
        for (int step = 0; step < STEPS; step++) {
            computeChunk(myChunk, step);   // read neighbors, write my cells
            barrier.await();               // wait for ALL threads to finish this step
            // every thread resumes here together, safe to start step+1
        }
    }

    void swapBuffers() { /* runs once per step, between phases */ }
    void computeChunk(int chunk, int step) { /* ... */ }
}
```

The `barrier.await()` is the whole story: each worker computes, then blocks until all workers have computed, then all proceed. The optional `Runnable` you pass to the `CyclicBarrier` constructor runs *once*, on the last-arriving thread, in the gap between "all arrived" and "all released" — perfect for the single-threaded bookkeeping that has to happen between phases, like swapping front and back buffers.

In **C++20** the equivalent is `std::barrier`:

```cpp
#include <barrier>
#include <thread>
#include <vector>

void run_simulation(int num_threads, double* grid) {
    std::barrier sync_point(num_threads, []() noexcept {
        swap_buffers();          // completion phase: runs once per round
    });

    auto worker = [&](int chunk) {
        for (int step = 0; step < STEPS; ++step) {
            compute_chunk(chunk, step);
            sync_point.arrive_and_wait();   // barrier: wait for all
        }
    };

    std::vector<std::jthread> pool;
    for (int t = 0; t < num_threads; ++t) pool.emplace_back(worker, t);
}
```

`arrive_and_wait()` is `std::barrier`'s spelling of `await`, and the lambda passed to the constructor is the **completion function** — the same once-per-round hook as Java's barrier `Runnable`. Both `std::barrier` and `CyclicBarrier` are *reusable*: the loop crosses the same barrier every step. That reusability is not free — it requires a clever bit of bookkeeping we will dissect in two sections.

The barrier is the heart of the **bulk-synchronous parallel** (BSP) model, which is worth naming because it is the cleanest way to write correct parallel numerical code. A BSP computation proceeds in *supersteps*: each superstep is a phase of independent local computation, followed by a barrier, followed by communication of results, followed by the next barrier. Inside a superstep no thread reads another thread's in-progress data, so there are no fine-grained races to reason about — all the synchronization is concentrated at the barrier. This is a genuinely powerful simplification: instead of dozens of little locks scattered through the inner loop, you have *one* barrier per step and a guarantee that everyone is on the same step. The cost, again, is the straggler: the barrier converts your throughput from "sum of thread speeds" to "number of threads times the *slowest* thread's speed," so the entire performance question in BSP code reduces to *load balance* — give every thread an equal-sized chunk of work so they all reach the barrier at nearly the same instant and nobody idles. A barrier with badly balanced chunks can leave half your cores spinning their wheels waiting; the [high-performance memory hierarchy](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm) and collective-communication primitives in the HPC series are where this scales up to thousands of GPUs, where the barrier becomes an all-reduce and the straggler becomes a network tail.

There is also a subtle deadlock trap unique to barriers that the BSP framing makes obvious: **every thread that is part of the party must reach the barrier every round, or the round never completes.** If one thread takes a different code path — an early `return`, an exception, a `break` out of the loop — and skips its `await`, the barrier waits forever for an arrival that is not coming, and the whole party hangs. This is why a barrier's party count must exactly match the number of threads that will *reliably* arrive every round, and why dynamic worker pools (threads that come and go) are a poor fit for a fixed-party barrier. If your thread count changes round to round, you do not want a barrier; you want a latch per round, or a phaser (Java's `Phaser`) that supports registering and deregistering parties dynamically.

## The latch: a one-shot gate that opens once and stays open

Now a subtly different problem. Your server starts up. Before it accepts traffic, it must finish *N* independent initialization tasks: load the config, warm the cache, open the connection pool, register with service discovery. These run on separate threads for speed. The main thread must block until *all N* are done, then flip the "ready" flag and start serving. You do not need the *N* tasks to wait for *each other* — only the main thread waits for *all of them*.

That is a **latch**, specifically a **countdown latch**. It holds a count initialized to $N$. Worker threads call `countDown()` as they finish, decrementing the count. One or more threads call `await()`, which blocks until the count reaches zero. When the last worker counts down to zero, the gate opens and every awaiting thread is released — **and the gate stays open forever after.** A latch is *one-shot*: once it reaches zero it is spent; it does not reset. The figure shows the lifecycle as a stack — count starts at $N$, each worker counts it down, and at zero the gate opens for everyone awaiting.

![A countdown latch lifecycle stack showing the count starting at N, each worker counting down, and the gate opening at zero](/imgs/blogs/semaphores-barriers-and-latches-8.png)

The distinction between a latch and a barrier is worth stating crisply, because they look similar and people conflate them:

- A **barrier** is symmetric and *reusable*: the *same group* of threads all wait for *each other*, repeatedly, round after round.
- A **latch** is asymmetric and *one-shot*: a set of *awaiting* threads wait for a set of *counting* threads to finish a thing *once*. The waiters and the counters can be entirely different threads, and after it opens it is done.

The figure below puts the one-shot latch next to the reusable cyclic barrier so the difference is visible at a glance.

![A before and after contrast of a one-shot latch that opens once and a cyclic barrier that resets and is reusable](/imgs/blogs/semaphores-barriers-and-latches-5.png)

Java's `CountDownLatch` is the textbook implementation:

```java
import java.util.concurrent.CountDownLatch;

class Server {
    void start() throws InterruptedException {
        CountDownLatch ready = new CountDownLatch(4);   // 4 startup tasks

        spawn(() -> { loadConfig();        ready.countDown(); });
        spawn(() -> { warmCache();         ready.countDown(); });
        spawn(() -> { openConnectionPool(); ready.countDown(); });
        spawn(() -> { registerDiscovery(); ready.countDown(); });

        ready.await();        // main thread blocks until count hits 0
        acceptTraffic();      // all four done — safe to serve
    }
}
```

Note one thing carefully: each `countDown()` should be in a `finally` so that a task that *throws* still counts down — otherwise a single failed startup task leaves the count stuck above zero and `await()` blocks forever, hanging your boot. Same conservation discipline as the semaphore permit.

```java
spawn(() -> {
    try {
        loadConfig();
    } finally {
        ready.countDown();   // count down even if loadConfig throws
    }
});
```

A countdown latch is also the cleanest way to run a **stress test** that starts all threads at exactly the same instant. You use *two* latches: a `start` latch of 1 that every worker awaits, and a `done` latch of $N$ that every worker counts down. The driver creates all the threads (they immediately block on `start.await()`), then calls `start.countDown()` once to release them all simultaneously, then `done.await()` to wait for them all to finish. This removes thread-creation skew from your timing — every worker truly starts together.

```java
CountDownLatch start = new CountDownLatch(1);
CountDownLatch done  = new CountDownLatch(threadCount);
for (int i = 0; i < threadCount; i++) {
    new Thread(() -> {
        try {
            start.await();            // all wait here for the gun
            doTheTimedWork();
        } catch (InterruptedException ignored) {
        } finally {
            done.countDown();
        }
    }).start();
}
long t0 = System.nanoTime();
start.countDown();                    // fire the starting gun: all go at once
done.await();                         // wait for everyone to finish
long elapsed = System.nanoTime() - t0;
```

The same one-shot latch exists in **C++20** as `std::latch` and in **Go** idiomatically as `sync.WaitGroup` (with `Add`/`Done`/`Wait` standing in for the initial count, `countDown`, and `await`):

```go
package startup

import "sync"

func Start() {
    var ready sync.WaitGroup
    ready.Add(4)                          // 4 startup tasks

    go func() { defer ready.Done(); loadConfig() }()
    go func() { defer ready.Done(); warmCache() }()
    go func() { defer ready.Done(); openConnectionPool() }()
    go func() { defer ready.Done(); registerDiscovery() }()

    ready.Wait()                          // block until all 4 call Done
    acceptTraffic()
}
```

The `defer ready.Done()` is the Go spelling of "count down in a `finally`" — it fires even if the task panics, so the count cannot get stuck. `WaitGroup` is a one-shot latch: once it hits zero and `Wait` returns, you do not reuse it (reusing a `WaitGroup` while a `Wait` may still be in flight is a documented footgun; make a new one per round).

## The cyclic barrier: reusable, and the generation counter that makes it safe

A `CountDownLatch` is one-shot — once it opens, it is done. But the simulation loop needs to cross a barrier *every step*. You cannot make a fresh latch each iteration cheaply and correctly, and you do not have to: a **cyclic barrier** *resets itself* after each release and is ready for the next round. `std::barrier` and Java's `CyclicBarrier` are exactly this.

The reusability sounds trivial but hides a genuinely tricky correctness problem, and this is the mechanism the series promised to make rigorous. Picture a naive barrier that just counts arrivals up to $N$ and, when the $N$-th arrives, resets the count to 0 and wakes the waiters. Here is the race that breaks it:

1. Threads T1 and T2 arrive at the barrier (count = 2, $N = 3$). They block.
2. T3 arrives. Count reaches 3. The barrier resets count to 0 and starts waking T1 and T2.
3. But T3 is *fast*. It returns from `await()`, races through the next step, and arrives at the barrier *again* — incrementing the count to 1 — **before T1 and T2 have even woken up from the previous round.**
4. Now T1 and T2 wake. From their point of view the count is 1 (T3's *next-round* arrival), and they are about to wait for two more arrivals in a round that already finished. The barrier is corrupted; threads from two different generations are mixed in one count.

This is a real race, and the fix is a **generation counter**. The barrier tracks not just the arrival count but a *generation number* — a monotonically increasing label for "which round are we in." When a thread arrives, it captures the current generation. When the last thread trips the barrier, it *bumps the generation* and resets the count, and the waking threads check: "am I being woken for the generation I arrived in?" A thread only proceeds when the generation has advanced past the one it parked in. T3's fast re-arrival lands in generation $g+1$; T1 and T2 woke for generation $g$; the generation label keeps the two rounds from contaminating each other.

Here is the heart of it, simplified, using a mutex and a condition variable (the building blocks underneath every real barrier — covered in the [condition variables post](/blog/software-development/concurrency/condition-variables-monitors-and-waiting-correctly)):

```java
class GenerationBarrier {
    private final int parties;
    private int count;            // arrivals so far this generation
    private int generation = 0;   // which round we are in
    private final Object lock = new Object();

    GenerationBarrier(int parties) {
        this.parties = parties;
        this.count = parties;
    }

    void await() throws InterruptedException {
        synchronized (lock) {
            int myGen = generation;     // capture the round I arrived in
            if (--count == 0) {
                // I am the last to arrive: trip the barrier
                generation++;           // advance to the next round
                count = parties;        // reset arrival count for next round
                lock.notifyAll();       // wake everyone waiting on this gen
            } else {
                // not last: wait until the generation advances past mine
                while (generation == myGen) {
                    lock.wait();        // re-checks the condition on wakeup
                }
            }
        }
    }
}
```

The `while (generation == myGen)` loop is the crux. A woken thread does not blindly proceed — it re-checks whether *its* generation has actually ended. T3's fast re-arrival increments `count` in the *new* generation; it cannot fool T1 and T2, because they are waiting on `generation == myGen` where `myGen` is the *old* generation, and that became false the instant the barrier tripped. The generation counter is what makes "reusable" *correct*, not just convenient. (Real implementations like Java's `CyclicBarrier` use a `ReentrantLock`, a `Condition`, and a `Generation` object internally, plus handling for `BrokenBarrierException` when a waiting thread is interrupted — but the generation idea is exactly this.)

This also explains a subtle behavioral difference: `CyclicBarrier` can become **broken**. If any waiting thread is interrupted or times out, the whole barrier breaks for the current generation and all parties get a `BrokenBarrierException`, because a barrier is an all-or-nothing rendezvous — one missing party means the synchronization point cannot be honored, so everyone must be told. A `CountDownLatch`, being one-shot and asymmetric, has no such notion; it just counts down.

| | Counting semaphore | Countdown latch | Cyclic barrier |
| --- | --- | --- | --- |
| Counts past one | Yes (up to $N$ permits) | Yes (down from $N$) | Yes (party of $N$) |
| Reusable | Yes (permits recycle) | No — one-shot | Yes — resets via generation |
| Who waits for whom | Acquirers wait for a free permit | Awaiters wait for $N$ countdowns | All $N$ wait for each other |
| Release by another thread | Yes (ownerless) | Yes (any thread counts down) | Each thread releases itself by arriving |
| Can break | No | No | Yes (`BrokenBarrierException`) |
| Typical use | Connection pool, rate limiter, signaling | Wait for $N$ events once (startup) | Synchronize $N$ threads each round (BSP) |

The matrix figure summarizes how the four primitives — including the plain mutex — line up on the questions that actually decide which one you pick.

![A matrix comparing mutex, counting semaphore, countdown latch, and cyclic barrier on counting, reusability, and typical use](/imgs/blogs/semaphores-barriers-and-latches-7.png)

## The rendezvous: a two-thread handshake

There is one more coordination pattern worth naming because it falls out of the barrier and shows up constantly: the **rendezvous**. A rendezvous is a barrier for exactly two threads — a point where thread A waits for thread B and thread B waits for thread A, so that *neither passes until both have arrived*. It is a mutual handshake: "I will not proceed until I know you are here too."

You build a rendezvous from two binary semaphores, each initially zero — one to signal "A has arrived," one to signal "B has arrived":

```c
sem_t a_arrived, b_arrived;
// both initialized to 0

void thread_a(void) {
    /* ... do A's work up to the rendezvous ... */
    sem_post(&a_arrived);   // tell B: I'm here
    sem_wait(&b_arrived);   // wait for B
    /* ... A continues, knowing B has arrived ... */
}

void thread_b(void) {
    /* ... do B's work up to the rendezvous ... */
    sem_post(&b_arrived);   // tell A: I'm here
    sem_wait(&a_arrived);   // wait for A
    /* ... B continues, knowing A has arrived ... */
}
```

Trace it: whichever thread arrives first posts its own "arrived" semaphore and then blocks on the other's. The second thread posts its semaphore and finds the first's already posted, so it does not block; then the first thread, woken, finds the second's posted too. Both proceed, and crucially, *both have proven the other arrived first*. Notice the post-then-wait order matters: if you wrote `wait` *before* `post` in both threads, each would block waiting for a signal the other has not sent yet — a clean deadlock (the worked example in the pitfalls section traces that interleaving step by step). The two ownerless binary semaphores, signaling across threads, are exactly what makes the handshake expressible. The unbuffered channel in Go (`ch <- v` blocks until `<-ch`) is a rendezvous with a payload: the sender and receiver meet, hand over the value, and both move on.

The rendezvous generalizes in two directions worth knowing. Make it a barrier of *three or more* and you have the symmetric group-wait we already covered. Make one side *re-arm* after each meeting and you have a **synchronous channel** — exactly the unbuffered Go channel or a CSP-style rendezvous, where every send is matched one-to-one with a receive and neither side can run ahead of the other. That one-to-one coupling is itself a form of backpressure: a fast producer on an unbuffered channel is *forced* to wait for a slow consumer, because the send does not complete until the receive happens. This is the same flow-control idea as the bounded producer–consumer buffer, taken to its limit of buffer size zero — the producer and consumer march in lockstep. When you reach for an unbuffered channel in Go and wonder why your producer "feels slow," that is the rendezvous doing its job: it is pacing the producer to the consumer on purpose. If you want the producer to run ahead by up to $k$ items, you give the channel a buffer of $k$ — which, as we saw, is exactly a counting semaphore of $k$ permits. The whole family connects: a buffer of zero is a rendezvous, a buffer of $k$ is a $k$-permit semaphore, and an unbounded buffer is the dangerous case with no backpressure at all.

## Measured behavior: throughput and queueing under a permit pool

Now the part the series insists on: *measure it.* A permit pool of size $N$ is a queueing system, and queueing systems obey **Little's law**, one of the most useful and most robust results in all of performance engineering. Little's law states that for any stable system, the average number of items *in the system* equals the average arrival rate times the average time each item *spends in the system*:

$$L = \lambda W$$

where $L$ is the average number in the system, $\lambda$ is the throughput (arrivals per second, which in steady state equals completions per second), and $W$ is the average time in the system. It holds with breathtaking generality — no assumption about the arrival distribution, the service distribution, or the queueing discipline. It is just conservation of items.

Apply it to a permit pool. Suppose each query takes on average $W_s = 5$ ms of actual service time at the database, and you have $N = 10$ permits. The *maximum* sustainable throughput is bounded by how many queries the ten busy slots can complete per second. If all ten permits are always busy, then by Little's law applied to the *service stage*, $L_s = 10$ (ten in service) $= \lambda \cdot W_s = \lambda \cdot 0.005$, so the ceiling is:

$$\lambda_{\max} = \frac{N}{W_s} = \frac{10}{0.005\,\text{s}} = 2000 \text{ queries/sec}$$

That is the saturation throughput. Push more than 2000 queries/sec at this pool and the excess *queues* — and now $W$, the time-in-system the *client* sees, is no longer 5 ms; it is service time *plus* queueing time. The pool cannot run faster than its permits allow, so backlog accumulates and latency climbs. Here is an honest sketch of how throughput and the client-observed wait behave as you crank the offered load against a 10-permit pool with 5 ms service time. These are illustrative figures from the Little's-law model, rounded, the kind of order-of-magnitude numbers you should *expect* and then confirm on your own hardware — not measurements I am claiming to have run on your box:

| Offered load (req/s) | Permits busy (avg) | Throughput (req/s) | Avg wait for a permit | Total latency $W$ |
| --- | --- | --- | --- | --- |
| 500 | ~2.5 | 500 | ~0 ms | ~5 ms |
| 1000 | ~5 | 1000 | ~0 ms | ~5 ms |
| 1800 | ~9 | 1800 | ~2 ms | ~7 ms |
| 2000 | 10 (saturated) | 2000 | growing | growing |
| 3000 | 10 (saturated) | 2000 (capped) | seconds, unbounded | unbounded |

The shape is the whole lesson. Below saturation, throughput tracks offered load and latency is flat at the service time — the pool is doing its job invisibly. As you approach $\lambda_{\max}$, the permits stay busy and the queue starts to form; latency rises gently (this is the classic $1/(1-\rho)$ blow-up, where $\rho = \lambda/\lambda_{\max}$ is utilization). At and past saturation, throughput **flattens at $\lambda_{\max}$** — it cannot exceed it — and every extra request just makes the queue longer and the wait worse, *without buying any more throughput*. This is precisely why a bare `acquire()` that waits forever is dangerous in production: past saturation it lets the queue grow without bound, turning a throughput limit into an unbounded latency and memory disaster. The `tryAcquire(timeout)` from earlier is the fix — it caps $W$ by shedding load, trading a few rejections for a bounded, predictable system.

#### Worked example: sizing the pool with Little's law

You measure that your database serves a query in $W_s = 8$ ms on average, and your peak offered load is 1500 queries/sec. How many permits do you need so the pool is *not* the bottleneck? Rearrange Little's law for the service stage: you need enough busy permits to absorb the load at the service rate, so $N \ge \lambda \cdot W_s = 1500 \times 0.008 = 12$. Twelve permits exactly saturates; you would size to maybe 15–16 to leave headroom for variance in $W_s$ and bursts. Crucially, do **not** size it to 100 "to be safe" — that just lets 100 queries hammer the database simultaneously, and now the *database* is your bottleneck and $W_s$ itself balloons under its own contention. The semaphore's whole value is that it *protects the downstream resource* by capping concurrency at the number that resource can actually handle well. Sizing the permit pool is sizing your concurrency to the slowest stage — measure $W_s$, compute $N = \lambda W_s$, add a margin, stop.

How would you measure this honestly rather than trusting the model? Run a load generator at a fixed offered rate against the real pool. Warm up first (JIT, connection establishment, OS caches all settle). Record completed requests over a window to get throughput, and record per-request end-to-end time to get $W$. Sweep the offered rate from well below to well above $\lambda_{\max}$ and plot throughput-vs-load and latency-vs-load. You should see throughput plateau and latency hockey-stick at the same offered rate, and that knee should sit right around $N / W_s$. If it does not — if the knee is much lower — your *service time* is not constant; it is degrading under load, which means the downstream resource is contending, which is itself the signal to *lower* $N$, not raise it. Acknowledge the nondeterminism: GC pauses, scheduler jitter, and network variance will smear the numbers, so run many trials and look at the distribution, not a single point.

## Pitfalls: the four ways these primitives bite

Every one of these primitives has a characteristic failure mode, and once you have seen each one you will recognize the symptom in a logfile from across the room.

**The leaked permit.** Already met, but it is the number-one bug so it earns repeating: an acquire whose matching release is on the happy path only. Any error, exception, or early return that skips the release permanently shrinks the count, and the pool drains toward zero over time until every request hangs. The symptom is a service that *slowly* gets worse — fine after deploy, degrading over hours, hung after a day — with the protected resource sitting idle. The fix is always to bind the release to the scope: `try/finally`, `defer`, RAII, or a guard type. Never write `acquire()` and `release()` as two free-floating statements with code that can throw between them.

**The double release.** The mirror image, and uniquely a *semaphore* bug because of ownerlessness. If a release runs twice — a retry that releases again, a `finally` that runs after the body already released, two code paths both thinking they own the permit — the count goes *up* past its initial $S_0$. Now the semaphore admits *more* than $N$ threads, silently. A 10-permit pool that has been double-released three times will let 13 connections through, and the database, not the semaphore, is the one that finally complains. A mutex would have thrown on the second unlock; the ownerless semaphore just counts up. The fix is to make each acquire correspond to exactly one release — a guard object that can only be dropped once is the cleanest enforcement.

**The barrier that never trips.** A barrier with a fixed party count of $N$ deadlocks permanently if fewer than $N$ threads will ever arrive — one worker crashed, one threw before reaching the barrier, or you spawned $N-1$ threads by an off-by-one. The other threads wait forever for an arrival that is never coming. The symptom is a hang at a phase boundary with the threads all parked in `await`. The defenses are a *timeout* on the barrier wait (`CyclicBarrier.await(timeout, unit)` throws `TimeoutException` and breaks the barrier so everyone learns), and making sure a worker that dies *also* trips or breaks the barrier rather than silently vanishing.

**The latch that never opens.** Symmetric to the barrier hang: a `CountDownLatch` whose count never reaches zero because some worker failed to call `countDown()` on its error path. The awaiting thread blocks forever. This is why every `countDown()` belongs in a `finally` (or `defer ready.Done()`): a task that throws must still decrement, or it hangs whoever is waiting. The defense beyond `finally` is a bounded `await(timeout)` so a stuck startup is detected and surfaced rather than hanging the boot indefinitely.

#### Worked example: the rendezvous deadlock

Recall the two-semaphore rendezvous and the warning that the *order* of post-then-wait matters. Here is the broken version and the exact interleaving that hangs it. Suppose both threads, by a copy-paste mistake, do `wait` *before* `post`:

1. **T_A** runs `sem_wait(&b_arrived)`. `b_arrived` is 0 (B has not posted yet). T_A **blocks**.
2. **T_B** runs `sem_wait(&a_arrived)`. `a_arrived` is 0 (A has not posted — A is blocked on step 1, it never reached its `post`). T_B **blocks**.
3. Both threads are now parked, each waiting for a signal the other was supposed to send *after* the point it is stuck before. Neither will ever post. Permanent deadlock.

The correct order — `post` your own arrival *first*, then `wait` for the other — works because the first thread to arrive posts unconditionally (never blocking on its own post, since `V` never blocks), then blocks on the other; the second thread posts, then finds the first's post already waiting and sails through. The asymmetry that makes it safe is that `V` (post) is non-blocking while `P` (wait) can block — so you must do all your non-blocking signals before any blocking wait, or you can wait for something that will only be signaled after you are already stuck. This is a tiny, two-line instance of the general deadlock rule: never hold (or wait on) one resource while waiting for another in an order that can form a cycle.

## Case studies / real-world

**Connection-pool exhaustion (the canonical incident).** The most common production outage shaped like this post is a connection-pool leak. A pool of, say, HikariCP connections in a JVM service, or a database/sql pool in Go, has a fixed maximum (the permit count). A code path acquires a connection but, on some error branch, fails to return it — the exact `try/finally` bug from earlier, just at the connection level. Under normal load the leak is slow and invisible; under a spike or after a dependency starts erroring, the leak accelerates, the pool drains to zero, and every subsequent request blocks waiting for a connection that will never come back. The service "hangs" with the database completely idle — the textbook symptom. The fix is always the same: tie the connection's return to the scope (`try/finally`, `defer`, RAII, or a context-managed pool), and bound the acquire wait so a drained pool fails fast and visibly instead of hanging silently. This pattern is so common that connection-pool libraries now ship leak detectors (HikariCP's `leakDetectionThreshold`) that log a stack trace when a connection is held longer than a threshold — an admission that the permit-leak bug is endemic.

**MapReduce and the barrier between phases.** Google's MapReduce, and every system shaped like it (Hadoop, Spark stages), has a hard barrier between the map phase and the reduce phase: no reducer may start until *all* mappers have finished, because a reducer needs every map output for its key, which could come from any mapper. That is a barrier in the strict sense — the reduce phase waits for the slowest mapper, the dreaded "straggler." This is exactly the "barrier runs at the speed of the slowest thread" cost made enormous: a single slow mapper (a hot disk, a skewed key, a flaky machine) stalls the entire reduce phase. The MapReduce paper's famous mitigation, *backup tasks* (speculatively re-running the last few straggling tasks on other machines and taking whichever finishes first), is a direct attack on barrier straggler cost — it does not remove the barrier, it just shrinks the tail that the barrier waits on. Any time you see a "phase boundary" in a parallel data system, there is a barrier underneath, and there is a straggler problem on top of it.

**Mars Pathfinder and why ownership matters.** In 1997 the Mars Pathfinder lander began experiencing total system resets on the Martian surface. The cause was **priority inversion**: a high-priority task needed a mutex held by a low-priority task, and a flood of medium-priority tasks kept preempting the low-priority holder, so it never got to release the mutex, so the high-priority task starved, so a watchdog timer fired and reset the system. The fix, uploaded to the spacecraft from Earth, was to enable **priority inheritance** on the mutex — temporarily boosting the lock holder's priority to that of the highest waiter so it could finish and release. The lesson for this post: priority inheritance is only possible because a *mutex knows its owner*. A semaphore, being ownerless, cannot do priority inheritance — there is no "holder" to boost. This is a concrete, expensive reason the ownership distinction is not academic: if you had used a binary semaphore where a priority-inheriting mutex belonged, no fix from Earth could have saved you, because the primitive structurally cannot track who to boost.

## When to reach for this (and when not to)

Reach for a **counting semaphore** when you need to limit concurrency to $N$ — a connection pool, a fixed worker fleet, a cap on concurrent outbound API calls, a simple rate/concurrency limiter. Reach for it also when you need cross-thread **signaling** (one thread releases what another acquires), like the producer–consumer `empty`/`full` pair. The semaphore is the right tool precisely when the answer to "how many at once?" is a number greater than one, or when the releaser and the acquirer are different threads.

Do **not** reach for a binary semaphore when you mean a **mutex**. If the answer to "how many at once?" is *one* and the same thread acquires and releases, use a mutex and keep the ownership checks, the reentrancy, and the priority-inheritance option. A binary semaphore as a mutex is a lock with the safety rails removed for no benefit.

Reach for a **countdown latch** (or `WaitGroup`, `std::latch`) when one or more threads must wait for a *set of one-shot events* to all complete — server startup waiting on $N$ initializers, a test driver waiting for $N$ workers to finish, a coordinator gating on a batch of tasks. The signature is *one-shot* and *asymmetric*: waiters wait for counters, once.

Reach for a **cyclic barrier** (or `std::barrier`) when a fixed group of threads must repeatedly synchronize at a phase boundary — bulk-synchronous parallel loops, stencil/simulation steps, anything with rounds. The signature is *reusable* and *symmetric*: the same group waits for each other, round after round. Do **not** use a barrier if your threads do not all live the same number of rounds, or if the group size changes dynamically — a barrier with a fixed party count deadlocks the instant fewer than $N$ threads will ever arrive (a thread died, or you spawned fewer than expected). And do not use a barrier where a latch fits: if the waiters and the workers are different sets and it only happens once, a latch is simpler and cannot break.

And the meta-rule from the whole series: **do not reach for any of these until you have named what is shared and what order you need.** A semaphore orders a *count*; a barrier orders a *phase*; a latch orders a *one-time completion*. If what you actually need is mutual exclusion over *data*, none of these is your tool — a mutex is. Measure first: if your concurrency is not actually your bottleneck, the cheapest correct primitive is the one you already understand. For the architecture-scale version of these flow-control decisions, the [rate limiting and backpressure post](/blog/software-development/system-design/rate-limiting-and-backpressure) is the companion read, and the [capstone playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model) is where all the mechanisms get weighed against each other.

## Key takeaways

1. A **counting semaphore** is an integer count plus `P` (acquire/decrement, block at zero) and `V` (release/increment, wake a waiter). Its safety invariant is $\text{count} = S_0 + V - P \ge 0$, which means at most $S_0$ threads are inside the guarded region at once.
2. The defining difference from a mutex is **ownership**: a mutex tracks who holds it and only the owner unlocks; a semaphore tracks a count and *any* thread may release. That ownerlessness is the *point* for signaling and the *footgun* if you use a binary semaphore as a mutex.
3. A **permit pool** caps concurrency at $N$ — the textbook use is a connection pool. Always tie the release to the scope with `try/finally`, `defer`, or RAII; a leaked permit drains the pool and hangs the service with the resource idle.
4. Use a **bounded acquire** (`tryAcquire(timeout)`) in production: past saturation, an unbounded wait lets the queue grow without limit. Shed load instead.
5. A **barrier** makes a group of threads wait for *all* to arrive, then releases them together — and it runs at the speed of the *slowest* thread, which is why load balance matters in barrier-heavy code.
6. A **countdown latch** is a *one-shot* gate: it opens when the count hits zero and stays open. Use it to wait for $N$ events once (startup, a stress-test starting gun, a batch of tasks).
7. A **cyclic barrier** is *reusable*; making reuse correct requires a **generation counter** so a fast thread re-arriving in the next round cannot corrupt the threads still waking from the last round.
8. **Little's law** $L = \lambda W$ sizes a permit pool: $N = \lambda W_s$. Size the pool to the slowest downstream stage plus a margin — not to a big "safe" number that just moves the bottleneck downstream.
9. A **rendezvous** is a two-thread barrier built from two zero-initialized binary semaphores; the post-then-wait order matters or you deadlock. An unbuffered channel is a rendezvous with a payload.
10. Pick by three questions: does it count past one, is it reusable, and what does it coordinate — a count (semaphore), a phase (barrier), a one-time completion (latch), or mutual exclusion over data (mutex)?

## Further reading

- **E. W. Dijkstra**, *Cooperating Sequential Processes* (1965/1968) — the original semaphore, `P`/`V`, and the producer–consumer and dining-philosophers problems. The source.
- **Allen B. Downey**, *The Little Book of Semaphores* (free) — an entire book of coordination puzzles solved with semaphores: barriers, rendezvous, reusable barriers, the readers-writers problem. The best hands-on follow-up to this post.
- **Brian Goetz et al.**, *Java Concurrency in Practice* — the definitive treatment of `Semaphore`, `CountDownLatch`, and `CyclicBarrier` in `java.util.concurrent`, including the bounded-pool and starting-gun patterns.
- **Maurice Herlihy & Nir Shavit**, *The Art of Multiprocessor Programming* — the rigorous foundations of synchronization, including barriers and their generation-based correctness.
- **Anthony Williams**, *C++ Concurrency in Action* — `std::counting_semaphore`, `std::latch`, and `std::barrier` from C++20, with the completion-function and RAII-release idioms.
- **Little, J. D. C.**, *A Proof for the Queuing Formula $L = \lambda W$* (1961) — the original proof of the conservation law you used to size the pool.
- **Dean & Ghemawat**, *MapReduce: Simplified Data Processing on Large Clusters* (2004) — the phase barrier between map and reduce, and backup tasks as a straggler mitigation.
- Within this series: [why concurrency is hard and why you cannot avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it), [mutual exclusion: mutexes and critical sections](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections), [condition variables, monitors, and waiting correctly](/blog/software-development/concurrency/condition-variables-monitors-and-waiting-correctly), and the [concurrency playbook capstone](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model).
