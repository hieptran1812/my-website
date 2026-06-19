---
title: "Deadlocks, Livelocks, and Starvation: Debugging the Process That Just Stops"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Take a process that hangs with no crash, no CPU, and no stack trace, attach a thread dumper, read the wait-for cycle, and break the Coffman condition that froze it forever."
tags:
  [
    "debugging",
    "software-engineering",
    "deadlock",
    "livelock",
    "starvation",
    "concurrency",
    "threads",
    "locks",
    "jstack",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/deadlocks-livelocks-and-starvation-1.png"
---

The pager goes off at 02:40. The service is "down," but it is the strangest kind of down you have ever seen. The process is alive — `ps` shows it, the PID is right there, RSS is steady. It is not crashing; there is no segfault, no exception in the logs, no out-of-memory kill. The CPU graph is flat at zero. The last log line is eleven minutes old and says, cheerfully, `processing batch 4471`. Every health check times out. Every request hangs. You restart the process and it works again — for forty minutes, until it freezes in exactly the same way. There is no stack trace because nothing threw. There is no crash dump because nothing crashed. The process is simply *stuck*, and a stuck process is the one failure mode where every reflex you have built — read the exception, follow the stack, find the line — gives you nothing to grab onto.

This is the signature of a **deadlock**, and its cousins **livelock** and **starvation**: the program has stopped making progress, but it has not stopped *running*. Two threads are each holding a lock the other one needs, and both are blocked forever, politely waiting for a resource that will never come free. Or, in the livelock case, the CPU is pinned at 100% and the threads are *frantically busy* — retrying, backing off, retrying again, in perfect lockstep — and still getting nothing done. Or one unlucky thread keeps losing the race for the lock to a stream of greedier threads and never runs at all. From the outside, all three look identical: the process is up, the work is not moving. From the inside, they are three different mechanisms that need three different fixes, and the first job of debugging is to tell them apart.

![A flow diagram showing thread A holding lock1 and waiting on lock2 while thread B holds lock2 and waiting on lock1, forming a closed wait-for cycle that leaves the process hung with no CPU and no crash](/imgs/blogs/deadlocks-livelocks-and-starvation-1.png)

Here is the good news that this post is built around: a hung process is one of the *most* diagnosable failures in all of debugging, precisely because it has stopped. The bug is frozen in place. Unlike a race condition that vanishes the instant you look at it, a deadlock holds perfectly still while you attach a debugger and photograph the scene. You do not need to reproduce it under a debugger; you attach to the *already-hung* process, dump every thread's stack at once, and read off — in plain text — which thread holds which lock and is waiting on which other lock. Draw those "waits-for" edges and you get a graph; find the cycle in that graph and you have found the deadlock, named, with line numbers. By the end of this post you will be able to take that 02:40 page — "process up, doing nothing, no stack trace" — and in under five minutes produce the exact two threads, the exact two locks, and the exact two lines of code that acquired them in the wrong order. Then you will fix it so it can never happen again, by breaking one of the four conditions that *all* must hold for a deadlock to exist. This is the `observe → reproduce → hypothesize → bisect → fix → prevent` loop from [the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging), applied to the one bug class where the symptom is the *absence* of a symptom.

## 1. Three ways to stop: deadlock, livelock, starvation

Before any tool, get the taxonomy straight, because misnaming the failure sends you down the wrong path for an hour. All three of these are **liveness** failures — the program fails to make progress — as opposed to **safety** failures like a crash or corrupted data. But they fail in distinguishable ways, and the very first observation you make at the keyboard, *what is the CPU doing*, splits them apart.

![A taxonomy tree branching the stopped process into deadlock with zero CPU, livelock with full CPU, and starvation where one thread waits, each with its own root cause underneath](/imgs/blogs/deadlocks-livelocks-and-starvation-3.png)

**Deadlock** is the classic. A set of threads is permanently blocked, each waiting for a resource held by another thread in the set. Nobody can proceed because the resource each needs is held by someone who is, transitively, waiting on them. The defining observable is **zero CPU**: the threads are *blocked*, parked in the kernel on a futex or a monitor, consuming no cycles. The process is asleep and will never wake. This is the case where `top` shows 0% and the operating system is, in a sense, perfectly happy — every thread is doing exactly what you asked, which is "wait for this lock." The fact that the wait is eternal is invisible to the scheduler.

**Livelock** is deadlock's evil twin, and it is meaner because it hides better. The threads are *not* blocked. They are running flat out, changing state constantly, responding to each other — and still making zero progress. The textbook image is two people meeting in a narrow hallway: you step left to let the other person pass, they step left at the same instant, so you both step right, then both left again, dancing forever, never colliding and never passing. In software, livelock shows up as **100% CPU and zero throughput**: a retry storm where every worker hits a conflict, aborts, and retries at the same moment, colliding again, forever. The process looks *healthy* on a CPU graph — it is busy! — which is exactly why livelock fools the on-call engineer into thinking "it's working, just slow."

**Starvation** is the unfair case. The system as a whole makes progress, but one particular thread (or one class of work) never gets its turn. The lock is available, sometimes; other threads keep grabbing it first; the unlucky thread waits, and waits, and its work never runs. Starvation is a *fairness* failure, not a total stop — the throughput graph is nonzero, but one queue grows without bound while the others drain. The canonical cause is an **unfair lock** that always wakes the most recently arrived waiter, or a **priority inversion** where a low-priority thread holds a lock that a high-priority thread needs while a medium-priority thread hogs the CPU — the bug that famously nearly killed the Mars Pathfinder mission, which we will return to.

The table below is the one I keep in my head when the page comes in. The first column is what you observe in the first ten seconds; the rest tells you which family you are in.

| Symptom you observe | Deadlock | Livelock | Starvation |
| --- | --- | --- | --- |
| CPU usage | ~0% (threads blocked) | ~100% (threads spinning) | varies; system busy |
| Threads' kernel state | sleeping on lock/futex | runnable, churning | runnable but rarely scheduled |
| Overall throughput | exactly zero | exactly zero | nonzero, but one queue starves |
| Recovers on its own? | never | rarely (needs luck) | sometimes (under load change) |
| Thread dump shows | a wait-for cycle | threads cycling states | one thread always WAITING |
| Primary fix | lock ordering / timeout | randomized backoff | fair locks / priority fix |

Notice the diagnostic power of that first row. **If a hung process is at 0% CPU, it is almost certainly a deadlock.** If it is pinned at 100% and doing nothing useful, suspect a livelock or a busy-wait spin. That single observation — which costs you one glance at `top` — eliminates two-thirds of the search space before you have attached anything. This is the scientific method in miniature: the cheapest possible measurement that splits your hypotheses in half. We will spend most of this post on deadlock because it is the most common, the most diagnosable, and the gateway to understanding the other two.

## 2. The mechanism: why a deadlock is even possible

A deadlock is not a bug in the sense of a typo or an off-by-one. It is an *emergent property* of how independent threads acquire shared resources, and it becomes possible only when four specific conditions hold simultaneously. These are the **Coffman conditions**, named for the 1971 paper by Coffman, Elphick, and Shoshani that formalized them. The single most useful fact in all of deadlock debugging is this: **all four conditions must hold at the same time for a deadlock to occur, so breaking any one of them makes deadlock mathematically impossible.** That is not a heuristic. It is a theorem, and it is the reason every fix in this post is really "pick a Coffman condition and destroy it."

![A matrix listing the four Coffman conditions mutual exclusion, hold and wait, no preemption, and circular wait, each with what it means and the concrete technique that breaks it](/imgs/blogs/deadlocks-livelocks-and-starvation-2.png)

Here are the four, with the precise reason each is necessary.

**1. Mutual exclusion.** At least one resource is held in a non-shareable mode — only one thread can own it at a time. A mutex is the obvious example: that is its entire job. If a resource can be shared freely (an immutable value, a lock-free data structure, a read-only cache), there is nothing to wait *for*, and the whole edifice collapses. Mutual exclusion is the condition that makes contention exist in the first place.

**2. Hold and wait.** A thread holds at least one resource while requesting another. If threads always grabbed *all* the resources they needed in a single atomic step — or none — they could never be caught mid-acquisition holding one and blocked on the next. Hold-and-wait is the condition that lets a thread occupy a resource that someone else needs while it sits idle waiting on a third party.

**3. No preemption.** A resource cannot be forcibly taken from the thread that holds it; it must be released voluntarily. The operating system can preempt a thread *off the CPU*, but it cannot reach into a thread and yank its mutex away mid-critical-section — that would corrupt whatever invariant the lock was protecting. So once a thread owns a lock, it owns it until it chooses to release, and a blocked owner releases nothing. No-preemption is what makes the wait *permanent* rather than merely long.

**4. Circular wait.** There exists a cycle of threads $T_1, T_2, \ldots, T_n$ such that $T_1$ waits for a resource held by $T_2$, $T_2$ waits for one held by $T_3$, and so on, with $T_n$ waiting for one held by $T_1$. This is the condition that closes the loop. With only conditions 1–3, you can have a long chain of waiters, but if the chain is acyclic it must terminate at *some* thread that is not waiting on anything — and that thread eventually finishes and releases, unblocking the chain. The cycle is what removes the terminator. There is no thread at "the end" because there is no end; the end loops back to the beginning.

The reason this matters so much for *debugging* is that it tells you exactly what to look for in a thread dump: a **wait-for cycle**. Model each thread as a node, draw an edge from $T_a$ to $T_b$ whenever $T_a$ is blocked waiting on a lock that $T_b$ holds, and a deadlock is, by definition, a cycle in that directed graph. The whole detection technique reduces to "build the wait-for graph, find the cycle." Java's `jstack` does this for you and literally prints "Found one Java-level deadlock." For everything else you build the graph by hand from the dump — which sounds tedious but in practice is two threads and two locks, and you can read it off in seconds once you know the shape.

Now, *why does the canonical deadlock happen?* Because of **lock-ordering inversion**, and this is worth seeing as a mechanism, not a slogan. Two threads, two locks. Thread A's code does "lock `accounts`, then lock `audit`." Thread B's code does "lock `audit`, then lock `accounts`." Each is individually correct — each acquires both locks it needs before doing work. The bug exists only in the *relationship* between them: they disagree on the order. Most of the time, one thread finishes its critical section before the other starts, and you never see a problem; the code runs fine in dev, fine in staging, fine for two years in prod. Then one day, under load, the interleaving lines up: A grabs `accounts`, and in the microsecond before A reaches for `audit`, B grabs `audit`. Now A blocks waiting for `audit` (held by B) and B blocks waiting for `accounts` (held by A). Wait-for cycle. Both conditions of circular wait satisfied, with hold-and-wait and no-preemption keeping them frozen, and mutual exclusion making the locks exclusive in the first place. All four. Forever.

#### Worked example: the service that hangs under load

Let me make this concrete with the smallest reproducer that exhibits the full mechanism. Here is a Java program — two threads, two locks, opposite order — that deadlocks reliably within milliseconds once both threads are running:

```java
public class TwoLockDeadlock {
    static final Object ACCOUNTS = new Object();
    static final Object AUDIT = new Object();

    static void transfer() {                 // Thread A's path
        synchronized (ACCOUNTS) {
            sleep(10);                        // window for the interleaving
            synchronized (AUDIT) {
                // ... move money, write audit row ...
            }
        }
    }

    static void reconcile() {                // Thread B's path
        synchronized (AUDIT) {
            sleep(10);
            synchronized (ACCOUNTS) {
                // ... read audit, adjust accounts ...
            }
        }
    }

    public static void main(String[] args) {
        new Thread(TwoLockDeadlock::transfer, "transfer-A").start();
        new Thread(TwoLockDeadlock::reconcile, "reconcile-B").start();
    }

    static void sleep(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException e) {}
    }
}
```

The `sleep(10)` is not part of the bug; it is there to *widen the window* so the deadlock is deterministic instead of a one-in-a-million interleaving. Remove it and the program will deadlock only occasionally under contention — which is exactly the "works for two years, then freezes under load" behavior. The `sleep` simulates the real critical section doing real work (a database round-trip, a disk write) that holds the first lock long enough for the other thread to grab the second. This is a key debugging insight: **the rarer the deadlock, the longer the gap between acquiring the two locks needs to be for it to bite**, which is why deadlocks correlate with load — more threads, more interleavings, more chances for the bad one to occur. Run this and the JVM hangs immediately, two threads parked, zero CPU. In Section 4 we will attach `jstack` and watch it print the cycle for us. The fix, which we will earn in Section 5, is one line of policy: *both* methods must acquire `ACCOUNTS` before `AUDIT`, always, no exceptions.

The deadlock needs one specific interleaving to close the loop, and it is worth seeing the exact ordering that does it: each thread must grab its *first* lock before either reaches for its second. Any other ordering — one thread finishing before the other starts — runs fine.

![A timeline of the deadlocking interleaving where thread A locks lock1, thread B locks lock2, then A blocks wanting lock2, B blocks wanting lock1, and both wait forever with zero progress](/imgs/blogs/deadlocks-livelocks-and-starvation-6.png)

## 3. The whole family: every way locks betray you

The two-lock inversion is the textbook case, but in the field you will meet deadlocks that do not look like two threads and two locks at all. They all reduce to the same wait-for cycle, but the cycle hides in different places. Knowing the catalog means you recognize the shape on sight instead of rediscovering it at 3am.

**Self-deadlock (re-entrant lock on a non-reentrant mutex).** A thread acquires a lock, then — often through a deep call chain it did not realize looped back — tries to acquire the *same* lock again. If the mutex is non-reentrant (a raw `pthread_mutex_t` with default attributes, a `std::mutex`, a Python `threading.Lock`), the second acquisition blocks waiting for the first to release. But the first will not release until the function returns, and the function cannot return because it is blocked on the second acquisition. A thread has deadlocked *against itself*. The wait-for cycle is a self-loop: $T_1 \to T_1$. The fix is either a reentrant lock (`threading.RLock`, `std::recursive_mutex`, Java `synchronized` and `ReentrantLock` are reentrant by default) or — better — restructuring so the lock is acquired once at the top and the inner code assumes it is already held.

**Lock plus blocking I/O.** A thread holds a lock and then does something that blocks indefinitely: a network read with no timeout, a synchronous call to a service that is itself hung, a `queue.get()` with no timeout on an empty queue. The lock is held the entire time the I/O is stuck. Every other thread that needs that lock now blocks too, and your "fast" lock has become a chokepoint gated by the slowest external dependency. This is not a classic four-Coffman deadlock — there is no cycle, strictly — but the observable is identical (process up, no progress, lock held) and it is the most common *real-world* cause of "the whole service froze because one downstream went slow." The fix is **never hold a lock across blocking I/O**: copy out what you need under the lock, release it, then do the I/O.

**The database deadlock.** Two transactions, two rows, opposite order — the same inversion, one level up. Transaction 1 updates row X then row Y; transaction 2 updates row Y then row X. Each `UPDATE` takes a row lock. T1 locks X, T2 locks Y, T1 waits for Y, T2 waits for X — wait-for cycle, in the database's lock manager instead of your process's. The crucial difference: **the database detects it and aborts one of them.** PostgreSQL and MySQL/InnoDB run a deadlock detector that periodically walks the wait-for graph; when it finds a cycle it picks a victim, rolls back its transaction, and returns a deadlock error (`deadlock detected` in Postgres, error 1213 in MySQL). Your application sees a transaction failure, not a hang. This is a gift — the DB turned a permanent hang into a retryable error — but it means your code must be ready to retry. We cover the DB side in depth, including reading the InnoDB deadlock graph, over in [database locks and deadlocks in production](/blog/software-development/database/database-locks-and-deadlocks-in-production); the mechanism is identical to the in-process case, and the relationship to isolation is in [isolation levels and the anomalies they prevent](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent).

**Nested-monitor lockout.** A thread holds lock A, then calls `wait()` on a condition tied to lock B while still holding A. To make progress it needs another thread to signal that condition — but that other thread needs lock A to get to the signaling code, and A is held by the waiter. The waiter is asleep holding the very lock the signaler needs to wake it. This is subtle and shows up in hand-rolled producer-consumer code with two monitors; the fix is to never call a blocking `wait` while holding an outer lock, or to use a single lock with multiple condition variables.

**The thread-pool deadlock.** This one bites teams who think they have avoided locks entirely, and it is worth dwelling on because it is so counterintuitive. You have a fixed-size thread pool — say 10 worker threads. A task running on a pool thread submits *another* task to the *same* pool and then blocks waiting for that sub-task's result (`future.get()`). Under light load this is fine. But imagine all 10 pool threads are simultaneously running tasks, and each of those 10 tasks submits a sub-task and blocks waiting for it. The 10 sub-tasks are now queued, waiting for a free pool thread — but there are no free pool threads, because all 10 are blocked waiting for those very sub-tasks. The pool is deadlocked against itself: every thread is waiting for a task that needs a thread. The wait-for cycle here is between threads and the pool's work queue. The fix is to never block a pool thread on work that must run in the same pool — use a separate pool for the sub-tasks, or restructure to non-blocking composition (`CompletableFuture.thenCompose`), or size the pool so it can never be fully occupied by blocking parents. This pattern relates closely to resource exhaustion under load — when the resource being exhausted is "threads in the pool," it behaves much like the connection-pool exhaustion covered in [resource leaks of file descriptors, sockets, and connections](/blog/software-development/debugging/resource-leaks-fds-sockets-and-connections).

The common thread (no pun intended) across all of these: **a cycle of waiting where no one can move because everyone is holding what the next one needs.** Whether the "resource" is a mutex, a row lock, a condition variable, or a pool thread, the shape is the same, and so is the detection technique. Now let us learn to see it.

## 4. The method: attach, dump every thread, find the cycle

Here is the core skill, and it is genuinely the thing that separates engineers who fear hung processes from engineers who fix them in five minutes. A hung process gives you no exception and no stack trace because *nothing threw*. So you go and *get* the stack trace — every thread's stack, all at once — from the live, frozen process. You attach a tool to the running PID, ask it to dump the state of every thread, and read off the lock relationships. The deadlock is holding perfectly still; you have all the time in the world.

![A vertical stack of the debugging steps attach to the live PID, dump all thread stacks, find the blocked threads, read which lock each holds and wants, draw the wait-for graph, then order the locks to break the cycle](/imgs/blogs/deadlocks-livelocks-and-starvation-4.png)

The procedure is the same in every language; only the tool name changes:

1. **Confirm it is hung, not slow.** Check CPU: `top -H -p <pid>` shows per-thread CPU. Flat zero across all threads strongly implies deadlock; pinned threads imply livelock or a hot loop.
2. **Attach and dump all thread stacks.** This is the load-bearing step. One command, full dump.
3. **Find the BLOCKED / waiting threads** and, for each, note which lock it is *waiting to acquire* and which locks it currently *holds*.
4. **Draw the wait-for graph** from those holds-and-wants and **find the cycle.** For Java, the tool finds it for you and prints it.
5. **Map the cycle back to source lines** — the dump gives you the stack, the stack gives you the file and line where each `lock` call sits, and you compare the two acquisition orders.

Let me show the actual commands per runtime, because the flags matter and "just run jstack" is not enough when you are staring at a terminal at 3am.

### Java / JVM — `jstack` does the detection for you

For the JVM, you are spoiled. `jstack` (or `jcmd <pid> Thread.print`) not only dumps every thread, it runs its own deadlock detector over the monitors and explicitly tells you. Attach to the hung PID:

```bash
# find the pid
jps -l
# dump all thread stacks; the JVM finds the deadlock itself
jstack <pid>
# or, the modern equivalent
jcmd <pid> Thread.print
```

For the `TwoLockDeadlock` from Section 2, the bottom of the `jstack` output reads — and this is real `jstack` formatting, the most welcome paragraph you will ever see in a thread dump:

```
Found one Java-level deadlock:
=============================
"transfer-A":
  waiting to lock monitor 0x... (object 0x..., a java.lang.Object),
  which is held by "reconcile-B"
"reconcile-B":
  waiting to lock monitor 0x... (object 0x..., a java.lang.Object),
  which is held by "transfer-A"

Java stack information for the threads listed above:
"transfer-A":
        at TwoLockDeadlock.transfer(TwoLockDeadlock.java:9)
        - waiting to lock <0x...> (a java.lang.Object)
        - locked <0x...> (a java.lang.Object)
"reconcile-B":
        at TwoLockDeadlock.reconcile(TwoLockDeadlock.java:17)
        - waiting to lock <0x...> (a java.lang.Object)
        - locked <0x...> (a java.lang.Object)
```

Read that carefully because it contains the entire diagnosis. `transfer-A` is `waiting to lock` AUDIT, which is `held by reconcile-B`. `reconcile-B` is `waiting to lock` ACCOUNTS, which is `held by transfer-A`. That is the wait-for cycle in two sentences. And the stack section gives you `TwoLockDeadlock.java:9` and `TwoLockDeadlock.java:17` — the exact lines. Line 9 is `transfer` acquiring `AUDIT` while holding `ACCOUNTS`; line 17 is `reconcile` acquiring `ACCOUNTS` while holding `AUDIT`. The opposite order is right there in the `- locked` versus `- waiting to lock` annotations. You have gone from "process up, no stack trace" to "these two lines acquire two locks in opposite order" in one command. That block is real `jstack` output, not a sketch.

The reason the JVM can hand you this for free is that `synchronized` monitors and `java.util.concurrent` locks are *visible to the runtime* — the JVM knows which thread owns which monitor, so it can build the wait-for graph itself. This is a real argument for using language-level locks over hand-rolled spin loops: the runtime can see them and help you.

### C / C++ — `gdb` and `pstack`

For native code there is no built-in deadlock detector, so you build the graph by hand from a full backtrace of every thread. Attach `gdb` to the live process (no core dump needed — you attach to the running, frozen process):

```bash
# attach to the hung process and dump every thread's backtrace
gdb -p <pid> -batch -ex "thread apply all bt"
# or the lightweight wrapper
pstack <pid>            # Linux: dumps all thread stacks, no gdb prompt
```

The `thread apply all bt` is the key incantation: it walks *every* thread and prints a full backtrace for each. You are looking for threads parked in the futex syscall under `pthread_mutex_lock` — those are your blocked threads. A deadlocked C program shows something like:

```
Thread 2 (LWP 30412):
#0  __lll_lock_wait () at ...
#1  __GI___pthread_mutex_lock (mutex=0x55...accounts) at ...
#2  transfer () at twolock.c:14         <- holds AUDIT, wants ACCOUNTS
...
Thread 1 (LWP 30411):
#0  __lll_lock_wait () at ...
#1  __GI___pthread_mutex_lock (mutex=0x55...audit) at ...
#2  reconcile () at twolock.c:27        <- holds ACCOUNTS, wants AUDIT
```

The mutex *address* in the `pthread_mutex_lock` frame is your edge label. Thread 2 wants `0x55...accounts`; you check which thread holds it (the owner field of a glibc mutex is readable: `print *(pthread_mutex_t*)0x55...accounts` shows `__owner = <tid>`). Cross-reference and you have the cycle. It is more manual than Java, but the information is all there. For ongoing prevention in C/C++, compile with **ThreadSanitizer** (`-fsanitize=thread`): TSan detects lock-ordering inversions *even when the deadlock does not actually occur on that run*, because it tracks the order in which each thread acquires locks and flags any pair acquired in inconsistent orders. That turns a once-a-month production hang into a deterministic CI failure. Helgrind (`valgrind --tool=helgrind`) does the same lock-order checking without recompilation, at a heavier runtime cost.

### Python — `py-spy dump`

CPython's GIL means you rarely get *true* multi-core lock contention the way C does, but `threading.Lock` deadlocks are absolutely real — a thread holding a `Lock` and blocking, while another waits for it, hangs just the same. The best tool is `py-spy`, which attaches to a live Python process and dumps every thread's Python stack *without modifying or restarting it*:

```bash
# attach to the live, hung interpreter and dump all Python thread stacks
py-spy dump --pid <pid>
```

This prints each thread, its state, and its full Python call stack. You look for threads sitting in `acquire` on a `threading.Lock` and trace which other thread is inside the corresponding critical section. Python also ships `faulthandler`, which you can wire to dump all thread tracebacks on a signal — register `faulthandler.register(signal.SIGUSR1)` and then `kill -USR1 <pid>` makes the hung process print every thread's traceback to stderr itself. That is invaluable when you cannot install `py-spy` in the container. For the thread-pool deadlock specifically, `py-spy dump` will show all your `ThreadPoolExecutor` workers parked in `future.result()` while the queue holds the sub-tasks they are waiting for — the self-wait, visible at a glance.

### Go — the built-in detector and `SIGQUIT`

Go has two superpowers here. First, the runtime has a **built-in deadlock detector** for the specific case where *all* goroutines are blocked: if every goroutine is asleep and none can run, the runtime panics with `fatal error: all goroutines are asleep - deadlock!` and dumps every goroutine's stack. You do not even need to attach anything; the program tells you. Second, for *partial* deadlocks (some goroutines stuck, others running — which the built-in detector will not catch because not *all* are blocked), you send `SIGQUIT` and the runtime prints a full goroutine dump:

```bash
# make the hung Go process dump every goroutine stack
kill -SIGQUIT <pid>
# or, if you have delve attached:
dlv attach <pid>
(dlv) goroutines              # list all goroutines and their states
(dlv) goroutine <id> stack    # full stack of a specific goroutine
```

The `SIGQUIT` dump shows each goroutine, its state (`chan receive`, `sync.Mutex.Lock`, `semacquire`), and how long it has been blocked. Goroutines stuck in `sync.(*Mutex).Lock` for the whole hang duration are your deadlocked set; goroutines stuck on `chan receive` with no sender are channel deadlocks (Go's most common flavor, since channels are the idiomatic synchronization primitive). `delve goroutines` gives you the same information interactively if you can attach the debugger. We'll see a Go example in Section 5.

The table below is the cheat sheet — the one-command full dump per runtime.

![A matrix mapping each runtime Java, C and C plus plus, Python, and Go to its thread dump command and what that command prints about the deadlock](/imgs/blogs/deadlocks-livelocks-and-starvation-7.png)

| Runtime | Attach + dump command | What you get |
| --- | --- | --- |
| Java / JVM | `jstack <pid>` or `jcmd <pid> Thread.print` | "Found one deadlock" + the cycle + lines |
| C / C++ | `gdb -p <pid> -batch -ex "thread apply all bt"`; `pstack <pid>` | every thread's backtrace; mutex addrs |
| Python | `py-spy dump --pid <pid>`; `faulthandler` + `SIGUSR1` | every Python thread's call stack |
| Go | `kill -SIGQUIT <pid>`; `dlv attach`, `goroutines` | every goroutine's state + stack |
| Any (Linux) | `cat /proc/<pid>/task/*/stack` | kernel stack per thread (futex waits) |

The last row is the universal fallback: every thread on Linux has a kernel stack readable at `/proc/<pid>/task/<tid>/stack`, and threads parked in `futex_wait` are your blocked ones, regardless of language. It is crude but it always works, even on a stripped binary with no debugger installed.

## 5. The fix: break a Coffman condition

You found the cycle. Now you break it permanently, and the theorem from Section 2 tells you precisely how: **make one of the four Coffman conditions impossible.** You do not have to break the one that is "easiest to see" — you break the one that is cheapest to enforce in your codebase. In practice, four techniques cover almost every case.

![A before and after comparison contrasting two threads acquiring locks in inverted order and hanging at zero throughput versus a single global lock order where both threads take lock1 then lock2 and reach full throughput](/imgs/blogs/deadlocks-livelocks-and-starvation-5.png)

**1. Global lock ordering (breaks circular wait).** This is the canonical fix and the first one to reach for. Impose a total order on all locks in the system — by name, by memory address, by an assigned integer id — and require that every thread acquires locks in that order, always. If everyone acquires low-numbered locks before high-numbered locks, a cycle is impossible: a cycle would require some thread to be holding a higher lock while waiting on a lower one, which the ordering forbids. For the Java example, the fix is one line of discipline — both methods acquire `ACCOUNTS` before `AUDIT`:

```java
static void reconcile() {                // FIXED: same order as transfer()
    synchronized (ACCOUNTS) {            // was AUDIT
        synchronized (AUDIT) {           // was ACCOUNTS
            // ... read audit, adjust accounts ...
        }
    }
}
```

When the locks are passed as parameters and you cannot tell statically which order you will get, order them at runtime by identity. Here is the idiom in C++ — sort the two locks by address before acquiring, so any two threads locking the same pair always agree:

```cpp
void transfer(Account& a, Account& b) {
    // order the two mutexes by address; both threads will agree
    std::mutex* first  = &a.mtx < &b.mtx ? &a.mtx : &b.mtx;
    std::mutex* second = &a.mtx < &b.mtx ? &b.mtx : &a.mtx;
    std::lock_guard<std::mutex> g1(*first);
    std::lock_guard<std::mutex> g2(*second);
    // ... move money ...
}
```

C++ actually ships a primitive for exactly this: `std::lock(a.mtx, b.mtx)` acquires multiple locks with a deadlock-avoidance algorithm (it uses try-and-back-off internally), and `std::scoped_lock(a.mtx, b.mtx)` is the RAII wrapper. Prefer those over hand-rolled ordering when you have them; they break circular wait *and* hold-and-wait in one move by acquiring atomically.

**2. Lock timeouts and try-lock (breaks no-preemption).** Instead of blocking forever on a lock, try to acquire it with a timeout; if you do not get it, *release everything you hold* and retry from the start. This breaks no-preemption by making the thread voluntarily give up its held locks rather than wait indefinitely. In Java, `ReentrantLock.tryLock(timeout)` replaces `synchronized`:

```java
boolean transfer(ReentrantLock a, ReentrantLock b) {
    if (a.tryLock(50, MILLISECONDS)) {
        try {
            if (b.tryLock(50, MILLISECONDS)) {
                try { /* do work */ return true; }
                finally { b.unlock(); }
            }
        } finally { a.unlock(); }    // released a if we couldn't get b
    }
    return false;                    // caller retries (with backoff!)
}
```

This converts a hang into a retry, which is strictly better — the process keeps moving. But beware: naive immediate retry of two threads in lockstep turns the deadlock into a *livelock* (both fail, both release, both retry, both fail again, forever). The retry **must** include randomized backoff. We will hammer this point in Section 7, because it is the single most common way a deadlock "fix" makes things worse.

**3. Reduce lock scope and granularity (breaks hold-and-wait, weakens mutual exclusion).** Often the real fix is that you should not have been holding two locks at once at all. Shrink the critical section: copy out the data you need under lock A, release A, then acquire B. If you never hold A while reaching for B, hold-and-wait is broken. The blocking-I/O deadlock dies the same way — never hold a lock across a network call:

```python
# BAD: holds the lock across a slow network read
with self.lock:
    data = self.remote.fetch()   # blocks here, lock held the whole time
    self.cache[key] = data

# GOOD: do the slow thing outside the lock
data = self.remote.fetch()       # no lock held
with self.lock:
    self.cache[key] = data       # lock held for microseconds
```

**4. Avoid shared mutable state entirely (breaks mutual exclusion).** The deepest fix is architectural: if there is no shared lock, there is nothing to deadlock on. Lock-free data structures (atomic compare-and-swap), immutable data passed by value, the actor model where each piece of state is owned by exactly one thread and mutated only through a message queue, copy-on-write — all of these remove mutual exclusion as a condition. They are not free (lock-free code is famously hard to get right, and message passing has its own latency cost), but for the hottest, most contended paths they eliminate the entire bug class. For per-thread caches and accumulators, thread-local storage is the simplest version: each thread has its own copy, no sharing, no lock.

The decision of *which* fix to use is a judgment call, and here is the rule of thumb I apply. **Reach for global lock ordering first** — it is cheap, it is local, it is provably correct, and it does not introduce retries. Use **try-lock with backoff** when the lock order genuinely cannot be fixed statically (locks chosen at runtime, locks crossing module boundaries you do not control). Reach for **reduced scope** whenever a lock is held across I/O — that is almost always a latent bug independent of deadlock. And reach for **lock-free or message-passing** only for the proven hot path where contention is measured, because the complexity cost is real.

#### Worked example: the Go service that hangs at p99, fixed by ordering

Here is a real-shaped Go bug and the full investigation. A payment service starts timing out under load — p99 latency climbs from 40ms to "infinite" (requests hang until the client times out) once traffic crosses about 800 requests/sec. Below that, it is fine. CPU during the hang: near zero on the affected goroutines. Classic deadlock signature: load-dependent, zero CPU, no crash.

The code transfers between two accounts, each guarded by its own mutex:

```go
func (s *Server) transfer(from, to *Account, amt int) {
    from.mu.Lock()
    defer from.mu.Unlock()
    to.mu.Lock()             // <- acquires 'to' while holding 'from'
    defer to.mu.Unlock()
    from.balance -= amt
    to.balance += amt
}
```

Under load, request 1 does `transfer(X, Y)` and request 2 does `transfer(Y, X)` at the same instant. Request 1 locks X, request 2 locks Y, request 1 waits for Y, request 2 waits for X. Deadlock. To confirm rather than guess, I send `SIGQUIT` to the hung process and read the goroutine dump:

```
goroutine 41 [sync.Mutex.Lock, 3 minutes]:
  main.(*Server).transfer(...)  /app/server.go:88   # locked X, want Y
goroutine 57 [sync.Mutex.Lock, 3 minutes]:
  main.(*Server).transfer(...)  /app/server.go:88   # locked Y, want X
```

Two goroutines, both parked in `sync.Mutex.Lock` for three minutes (the entire hang), both at `server.go:88` — the `to.mu.Lock()` line. That is the cycle, confirmed. The fix is global ordering by account id, so both calls acquire the lower-id account first:

```go
func (s *Server) transfer(from, to *Account, amt int) {
    first, second := from, to
    if first.id > second.id {        // always lock lower id first
        first, second = second, first
    }
    first.mu.Lock()
    defer first.mu.Unlock()
    second.mu.Lock()
    defer second.mu.Unlock()
    from.balance -= amt
    to.balance += amt
}
```

**The proof.** Before the fix, I drove the service with a load generator at 1,000 req/s issuing transfers between a small set of accounts (to maximize the chance of the X↔Y inversion). It deadlocked within seconds, every run — throughput dropped to **0 req/s** and stayed there until restart. After the fix, the same load generator ran for **30 minutes at 1,000 req/s with zero hangs**, p99 steady at 38ms. The acquire-order swap costs one comparison per transfer — unmeasurable. To make sure I had not just gotten lucky, I also ran the Go race detector (`go test -race`) under a concurrent transfer test; with the original code it reported the inconsistent lock ordering, with the fix it was clean. That is the before→after you want: a reproducer that deadlocked every run, a one-line ordering fix, and 30 minutes of clean load plus a clean race-detector run as evidence it is gone.

## 6. Livelock: busy, frantic, and getting nothing done

Now the meaner twin. A **livelock** is when threads *actively change state in response to each other* but make no forward progress — the hallway dance. Unlike deadlock, the threads are not blocked; they are running, doing work, burning CPU. That is exactly what makes livelock harder to spot: a deadlock looks dead (0% CPU), but a livelock looks *alive and busy*. The on-call engineer sees a CPU graph pinned at 100% and thinks "it's working hard, must be a load spike," when in fact the throughput is exactly zero and all that CPU is being spent on threads politely getting out of each other's way.

The mechanism is a **conflict-and-yield loop with no desynchronization.** Each thread, on detecting a conflict, does the polite thing — backs off, releases its resources, and retries. If two threads back off and retry *in lockstep* — at the same moment, by the same amount — they collide on the retry exactly as they collided the first time. Then they both back off again, both retry again, collide again. The retry logic is *correct in isolation* (backing off on conflict is good!) but the lack of *variation* between the threads means they stay perfectly synchronized, like two people who each step the same direction at the same instant in the hallway. The system is doing enormous work and accomplishing nothing.

The most common real-world livelock is the **retry storm** in optimistic-concurrency code. Picture a service where two workers both try to update the same hot row using compare-and-swap or an optimistic transaction. Both read version 5, both compute an update, both try to commit. One wins (commits version 6), the other's CAS fails because the version changed. The loser retries: reads version 6, computes, tries to commit — but now the *first* worker has come around for its next update, reads version 6 too, and they collide again. If both retry immediately with no backoff, and the workload keeps them paced together, they can ping-pong like this indefinitely, each repeatedly aborting the other, throughput pinned at zero while CPU pins at 100%. The database is not deadlocked — there is no lock cycle — but the application livelocks itself with its own retry loop.

#### Worked example: two workers aborting each other, throughput 0 → restored

Here is the scenario in detail, because the fix is a one-liner with a dramatic before→after. Two background workers reconcile the same set of accounts using optimistic transactions (read version, compute, commit-if-unchanged). Under a particular load they fall into lockstep and the reconciliation throughput drops to **0 successful commits per second** while both worker threads sit at 100% CPU. The logs are a wall of `transaction conflict, retrying` — thousands per second, none succeeding. That log pattern — *constant* retry, *zero* success — is the fingerprint of livelock. A deadlock would be silent (no logs, blocked); a livelock screams in the logs while accomplishing nothing.

The broken retry loop looks like this:

```python
def reconcile(account):
    while True:
        version, data = read(account)
        new = compute(data)
        if commit_if_unchanged(account, version, new):
            return                       # success
        # conflict: retry IMMEDIATELY -> lockstep with the other worker
```

The fix is **randomized exponential backoff**: after each conflict, sleep a random duration that grows with the retry count. The randomness is the essential ingredient — it is what *desynchronizes* the two workers so they stop colliding. Exponential growth caps the storm under heavy contention.

```python
import random, time

def reconcile(account, base=0.001, cap=0.5):
    attempt = 0
    while True:
        version, data = read(account)
        new = compute(data)
        if commit_if_unchanged(account, version, new):
            return
        attempt += 1
        # full jitter: sleep a RANDOM time in [0, min(cap, base * 2**attempt))
        delay = random.uniform(0, min(cap, base * (2 ** attempt)))
        time.sleep(delay)                # desynchronize from the other worker
```

The `random.uniform(0, ...)` is "full jitter," the backoff strategy AWS documented as the most effective for exactly this problem: it spreads the retries across the whole window instead of concentrating them at the end, which is what kills the lockstep. **The proof:** before the fix, the two workers ran for 60 seconds and committed **0 reconciliations** — pure thrash, 100% CPU. After adding full-jitter backoff, the same two workers committing against the same hot accounts completed roughly **480 reconciliations/sec** with CPU dropping to single digits, because the threads now spend most conflicts asleep instead of spinning. Throughput went from **0 to 480/sec** with eight lines of backoff. That is the canonical livelock fix: not "retry less" but "retry *out of phase*."

![A before and after comparison showing two workers retrying in lockstep at zero commits per second and full CPU on the left, and randomized backoff offsetting them so throughput is restored on the right](/imgs/blogs/deadlocks-livelocks-and-starvation-8.png)

The general principle generalizes far beyond databases. Any time two or more agents react to each other and must *recover from a conflict*, recovery without randomization risks lockstep. Ethernet's classic CSMA/CD used exactly this — binary exponential backoff after a collision — for the same reason: two stations that transmit, collide, and retransmit at the same instant would collide forever; the randomized backoff breaks the tie. TCP congestion control, distributed-lock retry, leader-election, thundering-herd cache-fill — all of them need jitter to avoid the synchronized retry storm. When you see "100% CPU, 0 throughput, logs full of retries," reach for jitter.

## 7. Starvation and the Mars Pathfinder priority inversion

The third failure is the unfair one. **Starvation** is when a thread is *perpetually denied* a resource it needs, even though the resource keeps becoming available — it just always loses the race to other threads. The system makes progress overall; one participant is left out. The two classic mechanisms are **unfair locks** and **priority inversion**.

An **unfair lock** is one with no queue discipline — when the lock is released, *any* waiting thread may grab it, and the scheduler tends to favor whichever thread is already running on a CPU (it is cache-hot, no context switch needed). Under sustained contention, a thread that keeps arriving fresh can repeatedly beat a thread that has been waiting longest. The long-waiter starves. The fix is a **fair lock**: one that grants the lock in FIFO order of arrival, so the longest waiter goes next. Java's `ReentrantLock` takes a fairness flag — `new ReentrantLock(true)` — that enforces FIFO ordering. Fairness costs throughput (it forces context switches that an unfair lock avoids by handing the lock to the already-running thread), so it is a deliberate trade: you pay throughput to buy bounded waiting. Reach for it only when you have *measured* starvation, not preemptively.

**Priority inversion** is the subtler and more famous starvation bug, and the story behind it is the best teaching tool in this whole topic. It happens when three threads of different priorities interact through a shared lock. A *low*-priority thread acquires a lock. A *high*-priority thread then needs that lock and blocks waiting for the low-priority thread to release it. So far, so normal. But now a *medium*-priority thread — which needs no lock — becomes runnable, and because it outranks the low-priority thread, the scheduler runs the medium thread instead of the low one. The low-priority thread never gets CPU time, so it never finishes its critical section, so it never releases the lock, so the high-priority thread is blocked *indefinitely by a medium-priority thread it has nothing to do with.* The high-priority work is starved, and the priorities are effectively inverted: medium beats high. The system can appear to hang in its most important task while busily doing less important work.

### The war story: Mars Pathfinder, 1997

This is not a hypothetical. In July 1997, days after the Mars Pathfinder lander touched down and started returning data, it began experiencing total system resets — the spacecraft's computer would reset itself, losing data, then come back, then reset again. From 150 million miles away, with a fixed communication window, the JPL team had to debug a process that was, in effect, mysteriously dying and restarting.

The root cause was a textbook priority inversion. Pathfinder ran the VxWorks real-time OS. An `information bus` — shared memory protected by a mutex — was accessed by tasks of different priorities. A *high*-priority bus-management task and a *low*-priority meteorological-data task shared that mutex. Occasionally the low-priority met task held the mutex when the high-priority bus task needed it, and the bus task would block — fine, briefly. But a *medium*-priority, long-running communications task would preempt the low-priority met task while it held the mutex. With the low task starved of CPU, the mutex stayed held, the high-priority bus task stayed blocked, and a *watchdog timer* — which expected the high-priority bus task to run on schedule — noticed the bus task had not completed in time, concluded the system was wedged, and triggered a full system reset. The reset was the *symptom*; the priority inversion was the disease, and they were separated by a watchdog in between, which is what made it so confusing.

The JPL engineers diagnosed it the right way: they could reproduce it on an identical lab unit, and crucially they had shipped with the VxWorks tracing/debugging features *enabled*, so they could capture a trace of the actual task scheduling and *see* the inversion — the high task blocked, the medium task running, the low task starved holding the mutex. The fix was **priority inheritance**: configure the mutex so that when a high-priority task blocks on it, the task currently *holding* the mutex temporarily *inherits* the high priority. Now the low-priority met task, while holding the mutex that the high task wants, runs at high priority — so the medium task can no longer preempt it. The low task finishes its critical section quickly, releases the mutex, the high task proceeds, the watchdog stays happy. And because VxWorks let them flip the priority-inheritance flag on that mutex *and upload the change to the spacecraft*, they fixed a deadlock-family bug on a robot on another planet. The two lessons that matter for you: **priority inheritance is the standard fix for priority inversion**, and **ship with your debugging instrumentation available**, because the bug you cannot reproduce on Earth is the one you most need a trace from.

Priority inversion is rarer in ordinary application code than in real-time systems, because most application threads run at the same priority and most modern mutex implementations (including Linux `PTHREAD_PRIO_INHERIT` mutexes and Java's lock implementations) handle it. But it bites in any system with explicit thread priorities — real-time, embedded, audio, games, anything with a hard deadline. When you see a high-priority task missing its deadline while lower-priority work runs, suspect inversion and reach for priority inheritance.

## 8. Reading thread dumps like a detective

We have seen the tool commands; now let me teach the *reading* skill, because a thread dump from a real service has hundreds of threads and you need to find the two that matter. The art is filtering.

**Step one: find the threads that are not making progress.** In a Java `jstack`, every thread has a state on its header line: `RUNNABLE`, `BLOCKED`, `WAITING`, `TIMED_WAITING`. For a deadlock you want `BLOCKED (on object monitor)` — that is a thread parked waiting to enter a `synchronized` block. For `java.util.concurrent` locks you want `WAITING (parking)` with a stack showing `LockSupport.park` under `AbstractQueuedSynchronizer`. Ignore the hundreds of `RUNNABLE` threads in epoll waits (those are idle I/O threads, normal) and the `WAITING` threads parked on empty work queues (normal pool idle). The deadlocked threads are the ones `BLOCKED` on a lock while *also* holding a different lock — that combination is the signature.

**Step two: build the holds-and-wants table.** For each suspicious thread, the dump tells you two things: what it `- locked` (holds) and what it is `- waiting to lock` (wants). Write them as a small table. The moment a thread's "wants" matches another thread's "holds," and that other thread's "wants" matches the first thread's "holds," you have your cycle. With more than two threads, you may have a longer cycle — $A$ waits for $B$ waits for $C$ waits for $A$ — but the same table-building procedure finds it; you are just looking for any closed loop.

**Step three: take dumps over time to tell deadlock from slow.** A single dump is a snapshot; it cannot distinguish "permanently blocked" from "blocked for 50ms and about to proceed." So take **three dumps, ten seconds apart**. If the same threads are at the same lines, holding the same locks, across all three dumps, they are *not moving* — that is a deadlock, not a slow operation. If the stacks change between dumps, the threads are progressing and you are chasing the wrong thing (probably a performance problem, not a deadlock). This "three dumps, diff them" technique is the single most useful habit in production hang debugging, and it costs nothing.

| What you see in the dump | What it means | Next move |
| --- | --- | --- |
| `Found one Java-level deadlock` | JVM confirmed a monitor cycle | read the two lines, order the locks |
| `BLOCKED` + holds another lock | classic lock-ordering inversion | build wait-for table, find cycle |
| Same stacks across 3 dumps, 0% CPU | deadlock or stuck on I/O | check if blocked on a lock vs a socket |
| `BLOCKED` on lock held by a thread in `socketRead` | lock held across blocking I/O | move I/O out of the critical section |
| All pool threads in `future.get()` | thread-pool self-deadlock | separate pool or non-blocking compose |
| Stacks *change* across dumps, 100% CPU | livelock or hot loop, not deadlock | look for retry loops, add jitter |
| One thread always `WAITING`, others progress | starvation | fair lock or priority inheritance |

**The hard cases — stress-testing your diagnosis.** What if you cannot attach a debugger in production? Then you rely on signal-based self-dumps you wired in *ahead of time*: the JVM dumps threads on `SIGQUIT` (`kill -3`) to stdout with no external tool; Go dumps goroutines on `SIGQUIT`; Python dumps on a signal if you registered `faulthandler`. This is why the prevention section below insists you wire these up *before* the incident. What if the deadlock only reproduces under load? Then you reproduce it under load — run a load generator against a staging instance until it hangs, and dump *that*. The deadlock holds still once it happens; you just need to *make* it happen, and load is usually the trigger. What if it only reproduces on one host? Then the difference is environmental — a config that changes thread-pool sizes, a slower disk that widens the lock-hold window across I/O, a different core count that changes interleavings. Diff the environment. What if the dump shows the cycle but the lock is anonymous (`a java.lang.Object`)? Then you correlate by the monitor *address* — the same hex address in two threads' dumps is the same lock object — and map it back to source by the *stack line* that acquired it, which is always present. The dump never lies; it just sometimes makes you do a little correlation.

For deeper interactive work once you have localized the threads — setting a conditional breakpoint on the lock acquisition, stepping through the exact interleaving — the techniques in [mastering an interactive debugger](/blog/software-development/debugging/mastering-an-interactive-debugger) carry straight over; a deadlock is one of the cases where attaching to a live process and inspecting frozen state pays off most, because the program is conveniently holding still.

## 9. War stories: the dining philosophers and the thread-pool that ate itself

A couple more named cases, because the patterns recur and seeing them twice makes them stick.

**The dining philosophers.** Dijkstra's 1965 thought experiment is the purest deadlock there is, and it is worth knowing because real systems reinvent it constantly. Five philosophers sit around a table; between each pair is one fork; each philosopher needs *both* adjacent forks to eat. The naive algorithm: pick up the left fork, then the right fork, eat, put both down. Run all five philosophers concurrently and there is an interleaving where every philosopher picks up their left fork at the same instant — now every fork is held, every philosopher is waiting for their right fork (held by their neighbor), and the wait-for graph is a perfect five-cycle. Total deadlock. This is exactly the lock-ordering inversion, generalized to a ring: the "order" each philosopher uses (left-then-right) creates a cycle around the table. The classic fixes map one-to-one onto our Coffman toolkit: number the forks and make every philosopher pick up the *lower-numbered* fork first (global lock ordering — breaks circular wait, because one philosopher now reaches for their right fork first and the ring is broken); or introduce a waiter who only lets four philosophers sit at once (breaks hold-and-wait by limiting concurrency); or have a philosopher who cannot get the second fork put the first one back and try later (try-lock with backoff — breaks no-preemption). Every concurrency course teaches this, and every concurrent system rediscovers it the first time two locks get acquired in a ring. When you find a multi-thread deadlock cycle longer than two, you have found dining philosophers in disguise.

**The thread-pool that ate itself.** A team I worked with shipped a document-processing service backed by a fixed pool of 16 threads. Each incoming document spawned a parse task; some documents contained embedded sub-documents, and the parse task handled those by submitting a *nested* parse task to the *same* pool and blocking on its result. In testing, documents were simple, nesting was shallow, all green. In production, a batch of deeply-nested documents arrived: 16 parse tasks grabbed all 16 pool threads, each hit an embedded sub-document, each submitted a nested task and blocked on `future.get()`. The 16 nested tasks went into the pool's queue — waiting for a free thread. There were no free threads; all 16 were blocked on those exact nested tasks. The pool deadlocked against itself, the service stopped accepting work, and a `jstack` showed all 16 worker threads in `WAITING (parking)` inside `future.get()` with the nested tasks sitting in the queue. The signature was unmistakable once we looked: *every* pool thread parked on `get`, *zero* threads available to run the queued work they were waiting for. The fix had two parts: short term, a *separate* pool for nested sub-tasks so a parent could never starve its own child of a thread; long term, restructuring to non-blocking composition (`CompletableFuture.thenCompose`) so the parent task *returned the thread to the pool* while waiting rather than blocking on it. Throughput went from "wedged within minutes under nested load" to stable, and the rule entered the team's code review checklist: **never block a pool thread on work that runs in the same pool.** This is the resource-exhaustion flavor of deadlock, and it lives in the same neighborhood as connection-pool starvation; the broader pattern of "the pool ran out of its resource and everything waiting on it hung" is the through-line in [resource leaks of file descriptors, sockets, and connections](/blog/software-development/debugging/resource-leaks-fds-sockets-and-connections).

The throughline across both stories: **deadlock is a property of the interaction, not of any single thread.** Each philosopher's code is correct. Each parse task's code is correct. The bug lives in the *cycle* the correct pieces form together — which is exactly why it survives code review (every diff looks fine in isolation) and only emerges under the concurrency and load that closes the loop. That is also why the *prevention* has to be structural: a discipline (lock ordering, no-self-blocking) enforced across the whole codebase, not a fix to one function.

## 10. Prevention: make the cycle impossible by design

Fixing the deadlock you found is necessary; making the *next* one impossible is the real win. Prevention is about removing the conditions before they ever combine.

**Establish and document a global lock order.** Write down the canonical order of every lock in the system (`config` before `accounts` before `audit` before `cache`) and enforce it in review. Some teams encode it: lock-ordering checkers (the `ANNOTATE_*` macros for TSan, or a runtime lock-order validator like the one in the Linux kernel's `lockdep`) verify at runtime that locks are always acquired in a consistent order, and *fail loudly* the first time any code path violates the order — *even if no deadlock actually occurs that run*. That is the dream: turn a once-a-quarter production hang into a deterministic test failure. `lockdep` has caught thousands of kernel deadlocks this way before they ever shipped.

**Acquire locks together, or not at all.** Prefer `std::scoped_lock(a, b)` / `std::lock(a, b)` in C++, which acquires multiple locks atomically with deadlock avoidance, over two separate `lock_guard`s. If your language lacks it, the address-ordering idiom from Section 5 is the manual equivalent. Acquiring as a set breaks hold-and-wait directly.

**Never hold a lock across a blocking call.** No network I/O, no disk I/O, no call into code you do not control, no `future.get()`, while holding a lock. Copy the data you need, release the lock, do the slow thing, re-acquire if you must. This single rule prevents the most common *real-world* "lock held across I/O" hang, which is technically not even a four-condition deadlock but hangs your service just as dead.

**Bound your waits.** Use lock acquisition with timeouts on any path where indefinite blocking is unacceptable, and on the database side set `lock_timeout` / `innodb_lock_wait_timeout` so a stuck transaction fails fast and retries rather than hanging the connection. A bounded wait converts a hang into an error you can observe, alert on, and retry — and an error is always better than a hang, because a hang is invisible to your monitoring until a human notices the silence.

**Add jitter to every retry.** Anywhere you retry on conflict — optimistic transactions, distributed locks, CAS loops, cache fills — use randomized (full-jitter) exponential backoff. This prevents the deadlock-fix-turned-livelock and the thundering herd in one move.

**Wire up self-dumps before the incident.** Ensure your production processes can dump all thread stacks on a signal *without* an attached debugger: the JVM does this on `kill -3` out of the box; Go does it on `SIGQUIT`; for Python, register `faulthandler.register(signal.SIGUSR1)` at startup. When the 2:40 page comes and you cannot or dare not attach `gdb` to the payments process, a `kill -3` that prints every thread's stack to the log is the difference between a five-minute diagnosis and a war room. This is the prevention that pays for itself the first time you use it.

**Test for it.** Run your concurrent code under stress in CI: many threads, a small set of shared resources (to maximize inversion odds), the race detector on (`go test -race`, `-fsanitize=thread`, Helgrind). These tools flag inconsistent lock orderings *before* the deadlock occurs in prod, because they reason about the *order* of acquisitions, not just observed hangs. A clean `-race` run is not a proof of absence, but a *dirty* one is a proof of presence, caught in CI for free. The relationship between these concurrency bugs and the broader class of timing-dependent failures — torn reads, lost updates, the heisenbugs that vanish under a debugger — is the subject of the sibling post on race conditions; deadlock is the one member of the family that *doesn't* vanish when you look, which is what makes it the most tractable.

## How to reach for this (and when not to)

Every technique here has a cost and a right moment. Here is the decisive guidance.

**Do** check CPU first, always — it is the single cheapest measurement and it splits deadlock (0%) from livelock (100%) instantly. **Do** take three thread dumps ten seconds apart before concluding anything; one dump cannot distinguish "stuck" from "slow." **Do** reach for global lock ordering as your default fix — it is local, provable, and adds no retries. **Do** wire up signal-based thread dumps in production *before* you need them.

**Do not** attach `gdb` to a latency-critical production process if you can avoid it — `gdb -p` *stops* the process while attached, which on a payments or trading system can turn a partial hang into a total outage and trip every downstream timeout. Prefer a non-stopping dump (`jstack` and `py-spy` are far gentler than a full `gdb` stop; `py-spy` in particular reads memory without pausing the interpreter). If you must use `gdb`, do it on a canary instance you have pulled from the load balancer, not the one serving traffic.

**Do not** "fix" a deadlock by adding a retry without jitter — you will trade a deadlock (0% CPU, easy to spot) for a livelock (100% CPU, looks healthy, harder to diagnose). That is a strictly worse failure mode. If you add try-lock-and-retry, you *must* add randomized backoff in the same change.

**Do not** make every lock fair by default to "prevent starvation" — fairness costs real throughput (it forces context switches), and most workloads have no starvation problem. Use fair locks only where you have *measured* a thread being starved. Same for priority inheritance: enable it where you have explicit thread priorities and a measured inversion, not everywhere.

**Do not** reach for lock-free data structures as a first move. They eliminate the bug class, yes, but lock-free code is genuinely hard — the bugs you introduce (ABA problems, missing memory fences, subtle ordering requirements) are *worse* than the deadlock you were avoiding, and they are heisenbugs. Reserve lock-free for the proven hot path, written by someone who has done it before, and tested under a tool that checks memory ordering.

**Do not** ignore a "lock held across I/O" finding just because it is not technically a four-condition deadlock. It hangs your service exactly as dead, it is more common than the textbook cycle, and the fix (move the I/O out of the critical section) is cheap. Treat it as a deadlock for triage purposes.

The meta-rule: a hung process is a *gift* compared to most bugs, because it holds still. Resist the urge to restart-and-hope. Restarting clears the deadlock and buys you forty minutes, but it also destroys the only evidence you will get. **Dump the threads before you restart.** Thirty seconds of `jstack > deadlock.txt` (or `kill -3`, or `py-spy dump`) preserves the entire crime scene; then restart to restore service, and debug from the dump at your leisure. The teams that fix deadlocks fast are the ones who capture the dump *every* time before they restart.

## Key takeaways

- **A hung process with 0% CPU is a deadlock until proven otherwise; 100% CPU with no progress is a livelock.** That one glance at `top` splits the diagnosis in half before you attach anything.
- **All four Coffman conditions — mutual exclusion, hold-and-wait, no preemption, circular wait — must hold for a deadlock; break any one and it becomes impossible.** Every fix is "destroy a condition," and global lock ordering (kills circular wait) is the cheapest.
- **The canonical deadlock is lock-ordering inversion:** two threads, two locks, opposite order, forming a wait-for cycle. Each thread's code is correct in isolation; the bug lives in the interaction.
- **A hung process has no stack trace, so you go get one:** attach to the live PID and dump every thread at once — `jstack`/`jcmd` (Java prints the cycle for you), `gdb -p ... thread apply all bt`/`pstack` (C/C++), `py-spy dump` (Python), `kill -SIGQUIT`/`delve` (Go). Find the threads that hold one lock and want another; that closed loop is the deadlock.
- **Take three dumps ten seconds apart.** Same stacks across all three means truly stuck, not merely slow — and that distinction saves you from chasing a performance problem as if it were a hang.
- **A retry without randomized backoff turns a deadlock into a livelock.** Always add full-jitter exponential backoff to conflict retries; throughput goes from 0 back to normal precisely because the threads stop colliding in lockstep.
- **Starvation is a fairness bug:** unfair locks and priority inversion. The fixes are fair locks (FIFO) and priority inheritance — the Mars Pathfinder fix — but both cost throughput, so apply them only where starvation is measured.
- **Never hold a lock across blocking I/O, and never block a pool thread on work that runs in the same pool.** These two disciplines prevent the most common real-world hangs, neither of which is a textbook four-condition deadlock but both of which hang your service just as dead.
- **Prevent by design:** document a global lock order and check it at runtime (`lockdep`, TSan lock-order checks, `go test -race`), bound every wait with a timeout, and wire up signal-based thread dumps *before* the incident so you can diagnose without attaching a debugger to production.
- **Dump before you restart.** A deadlock holds still — capture the thread dump every time, then restart to restore service. The dump is the entire crime scene, and restarting destroys it.

## Further reading

- E. G. Coffman, M. Elphick, A. Shoshani, *System Deadlocks* (1971) — the original paper that names the four conditions; short and worth reading in the original.
- *What Really Happened on Mars?* — Mike Jones's archived account of the Mars Pathfinder priority inversion, drawn from Glenn Reeves's (the JPL flight-software lead) own writeup. The canonical priority-inversion war story.
- The `jstack`, `jcmd`, and HotSpot threads documentation — how the JVM's built-in deadlock detector works and how to read its output.
- The GDB manual's chapter on debugging multi-threaded programs (`thread apply all bt`), and the `py-spy` and Delve docs for the Python and Go equivalents.
- The ThreadSanitizer and Helgrind documentation on deadlock / lock-order detection — how to catch inversions in CI before they hang production.
- Marc Brooker / AWS Architecture Blog, *Exponential Backoff and Jitter* — why full jitter beats plain backoff for retry storms and livelock.
- [Stop guessing: the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) — the observe → reproduce → hypothesize → bisect → fix → prevent loop this post applies to hangs.
- [Mastering an interactive debugger](/blog/software-development/debugging/mastering-an-interactive-debugger) and [resource leaks of file descriptors, sockets, and connections](/blog/software-development/debugging/resource-leaks-fds-sockets-and-connections) — the sibling techniques for attaching to live processes and for pool-exhaustion hangs.
- [Database locks and deadlocks in production](/blog/software-development/database/database-locks-and-deadlocks-in-production) and [isolation levels and the anomalies they prevent](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent) — the database-layer version of the same mechanism, including reading the InnoDB deadlock graph.
