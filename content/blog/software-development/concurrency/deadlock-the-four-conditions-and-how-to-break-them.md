---
title: "Deadlock: The Four Conditions and How to Break Them"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Reproduce the classic two-lock deadlock, learn the four Coffman conditions that all must hold for it, and break it for good with lock ordering, tryLock, the banker's algorithm, and cycle detection."
tags:
  [
    "concurrency",
    "parallelism",
    "deadlock",
    "lock-ordering",
    "coffman-conditions",
    "liveness",
    "locks",
    "synchronization",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/deadlock-the-four-conditions-and-how-to-break-them-1.png"
---

At 14:42 on a busy afternoon the payments service stopped responding. Not crashed — the process was alive, the health check on the load balancer was green, CPU sat at three percent. But every request to `POST /transfer` hung, the thread pool drained to zero free workers, and within ninety seconds the whole API was timing out. The on-call engineer pulled a thread dump and found two worker threads frozen in exactly the configuration that, once you have seen it, you recognize forever. Worker 41 had locked the row for account A and was blocked waiting for the lock on account B. Worker 58 had locked account B and was blocked waiting for the lock on account A. Each held precisely the thing the other needed, and neither would ever let go. The transfers `transfer(A, B)` and `transfer(B, A)` had run at the same instant, grabbed their locks in opposite orders, and frozen the service solid. That is a **deadlock**, and it is the canonical *liveness* failure: nothing is corrupted, no exception is thrown, the program is simply, permanently, stuck.

![the two-lock deadlock where one thread holds account A and wants B while the other holds B and wants A](/imgs/blogs/deadlock-the-four-conditions-and-how-to-break-them-1.png)

A *safety* failure — a lost update, a torn read, a corrupted balance — is the disease the [previous posts in this series](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections) cured with mutual exclusion. But the cure has a side effect. The moment you introduce locks to make individual operations safe, you introduce the possibility that two threads will wait on each other in a cycle and freeze. Safety says "nothing bad ever happens." Liveness says "something good eventually happens." A deadlocked program is perfectly safe and completely dead. And the cruelest part is that it is *timing-dependent*: the same code runs ten million times correctly and then, on the one interleaving where two transfers in opposite directions overlap by a few microseconds, it locks up. You cannot reproduce it on demand, you cannot find it with a single-threaded test, and it will be entirely absent from your unit tests right up until it pages you at peak load.

This post is the full anatomy of that freeze and every known way to break it. We will reproduce the two-lock deadlock as real, running code in Go, Java, and C++ — and watch it hang. Then we will name the **four Coffman conditions**: mutual exclusion, hold-and-wait, no preemption, and circular wait. The key fact, the one that organizes everything else, is that *all four must hold simultaneously* for a deadlock to be possible — they form a conjunction — so breaking **any single one** of them makes deadlock impossible. That single insight turns a terrifying, nondeterministic bug into a menu of concrete engineering fixes. We will work through the three families of strategy: **prevention** (design the system so a condition can never hold — global lock ordering is the workhorse), **avoidance** (the banker's algorithm: refuse any lock grant that could lead to an unsafe state), and **detection plus recovery** (let deadlocks happen, find the cycle in a wait-for graph, and kill a victim — exactly what your database already does). By the end you will be able to reproduce the bug, fix it with lock ordering in two languages, reason about which strategy fits a given system, and read a deadlock thread dump the way the on-call engineer did.

This is the spine of the whole series at work: shared mutable state plus nondeterministic scheduling is the hazard, and we tame it by reasoning about the *order* in which threads touch shared resources. For deadlock, the order in question is the order locks are acquired, and the fix is almost embarrassingly simple once you see it. Let us earn that simplicity.

## Reproducing the deadlock: two accounts, two locks

Concrete first. Here is a bank with accounts, each account guarded by its own lock so that concurrent transfers touching *different* accounts can run in parallel — exactly the fine-grained locking you would reach for to get throughput. A transfer locks the source, locks the destination, moves the money, and unlocks both. Per the convention in this series we are moving `\$100` between accounts; the dollar amounts are illustrative.

Here it is in Go:

```go
package main

import (
	"fmt"
	"sync"
	"time"
)

type Account struct {
	mu      sync.Mutex
	balance int
	id      string
}

// transfer locks src, then dst, then moves the money.
func transfer(src, dst *Account, amount int) {
	src.mu.Lock()
	defer src.mu.Unlock()

	// A tiny pause widens the race window so the deadlock
	// reproduces reliably instead of one run in a million.
	time.Sleep(1 * time.Millisecond)

	dst.mu.Lock()
	defer dst.mu.Unlock()

	src.balance -= amount
	dst.balance += amount
}

func main() {
	a := &Account{balance: 1000, id: "A"}
	b := &Account{balance: 1000, id: "B"}

	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); transfer(a, b, 100) }() // A then B
	go func() { defer wg.Done(); transfer(b, a, 100) }() // B then A
	wg.Wait()

	fmt.Println("done", a.balance, b.balance) // never prints
}
```

Run this and it hangs. Forever. The `wg.Wait()` never returns because neither goroutine ever calls `wg.Done()`. The Go runtime, helpfully, will eventually notice that *all* goroutines are blocked and panic with `fatal error: all goroutines are asleep - deadlock!` — a luxury most languages do not give you. In a real server with a background goroutine still alive (a metrics ticker, an HTTP listener), the runtime sees that *some* goroutine is runnable and stays quiet, so the freeze is silent. That is why production deadlocks are so insidious: in the real system, the runtime does not tell you.

The same bug in Java, which is where you are far more likely to ship it, because synchronized methods make the lock acquisition invisible at the call site:

```java
import java.util.concurrent.locks.ReentrantLock;

class Account {
    final ReentrantLock lock = new ReentrantLock();
    int balance;
    final String id;
    Account(String id, int balance) { this.id = id; this.balance = balance; }
}

public class Bank {
    static void transfer(Account src, Account dst, int amount) {
        src.lock.lock();
        try {
            try { Thread.sleep(1); } catch (InterruptedException e) {}
            dst.lock.lock();   // T1 blocks here holding src, wanting dst
            try {
                src.balance -= amount;
                dst.balance += amount;
            } finally {
                dst.lock.unlock();
            }
        } finally {
            src.lock.unlock();
        }
    }

    public static void main(String[] args) throws InterruptedException {
        Account a = new Account("A", 1000);
        Account b = new Account("B", 1000);
        Thread t1 = new Thread(() -> transfer(a, b, 100)); // A then B
        Thread t2 = new Thread(() -> transfer(b, a, 100)); // B then A
        t1.start(); t2.start();
        t1.join(); t2.join();                              // never returns
        System.out.println("done " + a.balance + " " + b.balance);
    }
}
```

And in C++ with `std::mutex`, where the `std::lock_guard` you reach for to be RAII-correct does *nothing* to save you from this:

```cpp
#include <mutex>
#include <thread>
#include <chrono>
#include <iostream>

struct Account {
    std::mutex m;
    int balance;
};

void transfer(Account& src, Account& dst, int amount) {
    std::lock_guard<std::mutex> g1(src.m);                 // lock src
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    std::lock_guard<std::mutex> g2(dst.m);                 // lock dst — may block forever
    src.balance -= amount;
    dst.balance += amount;
}

int main() {
    Account a{ {}, 1000 }, b{ {}, 1000 };
    std::thread t1(transfer, std::ref(a), std::ref(b), 100); // A then B
    std::thread t2(transfer, std::ref(b), std::ref(a), 100); // B then A
    t1.join(); t2.join();                                    // hangs
    std::cout << "done " << a.balance << " " << b.balance << "\n";
}
```

Three languages, three idioms, one bug. Note what they have in common and what makes the bug so easy to ship. Each `transfer` call is *locally* correct: it acquires the locks it needs, does its work, and releases them in reverse order with proper RAII or `defer` or `finally`. There is no leak, no missing unlock, no obvious mistake. The bug is not *in* a transfer; it is *between* two transfers that happen to run in opposite directions at the same time. No single function review catches it. You have to reason about pairs of executions, which is exactly the kind of reasoning humans are bad at and exactly why deadlock has its own theory.

This is worth dwelling on because it is the deepest difference between a deadlock and a data race. A data race is *local*: it lives in a single critical section that fails to be mutually exclusive, and you can find it by staring at one piece of code — `count++` is racy on its own, in isolation, and the fix (a lock) is local to that section. A deadlock is *global*: no single lock acquisition is wrong, no single function is wrong, and the bug only exists as a property of the *whole system of lock-acquisition orders across all threads*. You can review every function in the codebase, find each one perfectly correct, and still have a deadlock, because the bug is in the *relationship* between two of them. This is why "make each function thread-safe" — sound advice for safety — does *nothing* for liveness. Composing two individually-correct synchronized operations can produce a deadlock that neither contains. Concurrency does not compose for free, and deadlock is the sharpest demonstration of that fact. The discipline you need is not "is this function correct?" but "is there a *consistent global order* in which all threads touch all locks?" — a question about the program as a whole, which is precisely why we need a theory rather than a code review.

There is a second subtlety the three snippets share: the deadlock is *invisible at the call site*. In the Java version, if `transfer` were written with `synchronized` methods instead of explicit `ReentrantLock`, the lock acquisition would not even appear in the source of the caller — `synchronized` hides it in the method declaration. You would call `account.withdraw()` and `account.deposit()` and have no syntactic clue that locks are being taken, let alone in what order. This is how deadlocks hide in mature codebases: the locks are buried inside library methods, framework callbacks, ORM-managed transactions, and the acquisition order is an emergent property of the call graph that no one designed and no one can see. The first job of any deadlock-prevention discipline is to make the lock-acquisition order *visible and explicit* — which is half of why lock ordering, our main fix, is so valuable: it forces you to name the order.

#### Worked example: the exact interleaving

Let us walk the precise instruction order that freezes the Go program. Two goroutines, G1 running `transfer(a, b, 100)` and G2 running `transfer(b, a, 100)`. The scheduler interleaves them like this:

1. **G1**: `src.mu.Lock()` where `src` is `a`. Succeeds. G1 now holds lock A.
2. **G2**: `src.mu.Lock()` where `src` is `b`. Succeeds. G2 now holds lock B.
3. **G1**: `dst.mu.Lock()` where `dst` is `b`. Lock B is held by G2 → G1 **blocks**, still holding A.
4. **G2**: `dst.mu.Lock()` where `dst` is `a`. Lock A is held by G1 → G2 **blocks**, still holding B.
5. Both goroutines are now blocked, each waiting on a lock the other holds. No third party will release anything. Stuck.

![a timeline showing T1 locks A then T2 locks B then each blocks waiting for the lock the other holds](/imgs/blogs/deadlock-the-four-conditions-and-how-to-break-them-3.png)

The `time.Sleep(1ms)` between the two `Lock()` calls is the dramatic device. Without it, the window between acquiring the first lock and the second is a handful of nanoseconds, so the deadlock-triggering interleaving is rare — you might run the program a thousand times and see it once. The sleep widens that window to a millisecond, making steps 1 and 2 almost certainly happen before steps 3 and 4, so the deadlock fires on essentially every run. This is the honest face of concurrency testing: *the bug's reproduction rate is a function of timing, not logic.* A deadlock that reproduces one run in ten thousand in a stress test will reproduce *constantly* in production, because production has more threads, slower locks (rows on disk, not bytes in cache), and longer hold times (a lock held across a network call stretches the window from nanoseconds to milliseconds). We will return to honest measurement at the end.

## The four Coffman conditions

In 1971 Edward Coffman, Melvin Elphick, and Arie Shoshani published a characterization of deadlock that has organized the field ever since. They identified four conditions, and proved that a deadlock can occur **if and only if all four hold simultaneously**. This is the single most useful fact in the entire topic, so let us state each precisely and then dwell on why the conjunction matters more than any individual condition.

![a matrix of the four Coffman conditions giving the meaning of each and how to break it](/imgs/blogs/deadlock-the-four-conditions-and-how-to-break-them-2.png)

**1. Mutual exclusion.** At least one resource is held in a non-shareable mode: only one thread can hold it at a time, and a second thread requesting it must wait. This is the defining property of a lock. If a resource can be shared freely — a read-only immutable value, an atomic counter, a lock-free data structure — there is nothing to wait *for*, and this condition fails. Mutual exclusion is the price of protecting an invariant over mutable state, and it is the hardest condition to give up, because giving it up means giving up the lock that was making your code safe in the first place.

**2. Hold and wait.** A thread holds at least one resource while requesting another. In our transfer, each thread holds its source lock while requesting its destination lock. If, instead, a thread had to acquire *all* the resources it would ever need in one atomic step — or release everything it holds before requesting more — it would never be in the state of "holding X, blocked on Y," and this condition fails.

**3. No preemption.** A resource cannot be forcibly taken from the thread holding it; it is released only voluntarily, by that thread, when it is done. A mutex has exactly this property: the operating system will not yank a lock away from a thread to give it to another. If the system *could* preempt — force a thread to drop its lock, roll back its partial work, and retry — then a blocked thread could be unstuck by stealing the resource it needs, and this condition fails. (This is precisely what a database does when it kills a deadlock victim and rolls back its transaction.)

**4. Circular wait.** There exists a closed chain of waiting threads $T_1, T_2, \ldots, T_n, T_1$ such that each $T_i$ is waiting for a resource held by $T_{i+1}$, and $T_n$ is waiting for a resource held by $T_1$. In the two-lock case the chain is short: $T_1 \to T_2 \to T_1$. T1 waits for B (held by T2); T2 waits for A (held by T1). That is the cycle, and it is the heart of the matter. Circular wait is the condition that *closes the loop* — without a cycle, the chain of waits terminates at some thread that is waiting for nothing, and that thread will eventually run, release its resources, and unblock the chain behind it.

### Why all four are necessary: break one, no deadlock

Here is the mechanism, stated as a logical claim so we can lean on it. Let $M$, $H$, $P$, $C$ be the propositions "mutual exclusion holds," "hold-and-wait holds," "no-preemption holds," "circular-wait holds." Coffman's theorem is that a deadlock is possible exactly when

$$M \land H \land P \land C$$

is true. Because this is a *conjunction*, its negation is a *disjunction*: the system is deadlock-free if

$$\lnot M \lor \lnot H \lor \lnot P \lor \lnot C.$$

In plain terms, you do not need to defeat all four conditions — you need to defeat **exactly one.** Knock out any single conjunct and the whole conjunction is false, and deadlock becomes impossible. This is why the four conditions are not a checklist of things to worry about; they are a *menu of attacks*. Pick the cheapest one for your system and engineer it so that condition can never hold.

It is worth seeing *why* the four conditions are jointly sufficient, not just necessary, because the proof is the mechanism. Suppose all four hold and consider the moment the system is stuck. Mutual exclusion means the contended resources cannot be shared, so a blocked thread genuinely cannot proceed without acquiring its resource. No-preemption means nothing will take a resource from its holder, so the only way a resource changes hands is voluntary release. Hold-and-wait means every blocked thread is holding something while it waits, so no blocked thread will release anything until it first acquires what it is waiting for. And circular wait means the blocked threads form a closed loop, each waiting on the next. Now chase the implication around the loop: $T_1$ will not release until it gets the resource $T_2$ holds; $T_2$ will not release until it gets what $T_3$ holds; and so on around to $T_n$, who will not release until it gets what $T_1$ holds. Every thread in the loop is waiting for a release that will only happen *after* its own — a circular dependency of "I go after you" that has no first mover. No thread can take the first step, so none ever does. That is the deadlock, derived. The cycle is the load-bearing piece: remove it and the chain of "I go after you" terminates at a thread waiting for nothing, who *can* take the first step, releasing the dependency that was blocking the thread behind it, and so on until the whole chain unwinds.

This is also the cleanest way to see why deadlock is a *stable* state, not a transient one. Once the system enters the deadlocked configuration, it stays there: no event in the system can change it, because every possible unblocking event (a resource release) is itself blocked behind another unblocking event in the cycle. A deadlock does not resolve on its own with time, retries, or load dropping — it is a permanent fixed point. This is precisely what distinguishes it from *contention* (threads waiting, but progress is being made, just slowly) and from *livelock* (threads actively running but making no progress). Deadlock is the still, silent one: no CPU burned, no progress, no exit, until something *outside* the cycle forces a resource to be released. Which is exactly what recovery does.

The four prevention strategies map one-to-one onto the four conditions:

- Defeat **mutual exclusion** → use lock-free or immutable data so there is no exclusive resource to wait for. (Often impossible; the data needs exclusive access for a reason.)
- Defeat **hold-and-wait** → acquire all locks at once (or none), so a thread never holds one while blocked on another.
- Defeat **no-preemption** → use `tryLock` with a timeout: if you cannot get the second lock, release the first and retry, voluntarily giving up what you hold.
- Defeat **circular wait** → impose a global ordering on locks and always acquire them in that order, so no cycle can ever form.

The last one — defeating circular wait by ordering — is by a wide margin the most practical, the cheapest, and the one you should reach for first. We will spend the most time on it.

#### Worked example: which condition does lock ordering break?

Trace it. With a global rule "always acquire the lock with the smaller id first," what happens to our two transfers? `transfer(a, b)` wants to lock A then B; since `id(A) < id(B)`, it locks A then B — same as before. `transfer(b, a)` *wants* to lock B then A, but the rule forces it to acquire in id order: A then B. So **both** transfers now try to acquire A first. Whichever wins gets A, then goes on to get B unobstructed; the loser blocks on A, but it is holding *nothing*, so there is no cycle — it is simply waiting in line. The chain of waits is $T_2 \to T_1$ and then it *stops*, because T1 holds both locks and is waiting for nothing. T1 finishes, releases both, T2 proceeds. Circular wait — condition four — has been made structurally impossible. The other three conditions still hold (we still use mutexes, still hold-and-wait, still no preemption), but that is fine: breaking *one* is enough.

## Prevention by lock ordering: the practical fix

If you remember one technique from this entire post, make it this one. **Impose a total order on your locks and always acquire them in that order.** A thread may hold lock $i$ and request lock $j$ only if $i < j$ in the global order. Under this discipline a wait-for cycle is *impossible*, because a cycle would require some thread holding a higher-ordered lock to wait for a lower-ordered one, which the rule forbids. The proof is a one-liner: in any chain of waits, the lock indices are strictly increasing, and a strictly increasing sequence cannot return to its start, so it cannot close into a cycle.

![before and after of unordered locking that deadlocks versus ordered locking where A is always taken before B](/imgs/blogs/deadlock-the-four-conditions-and-how-to-break-them-4.png)

The figure shows the whole idea: unordered, the two threads take A and B in opposite orders and a cycle forms; ordered, both take A before B and the cycle cannot exist. Now the engineering question is: what *is* the order? You need a stable, total ordering over your lock objects that any thread can compute without coordination. Common choices:

- **A natural key.** Accounts have ids; order locks by account id (or IBAN, or primary key). This is the cleanest when your resources have a meaningful unique identifier.
- **Memory address.** In C++ you can order two `std::mutex` objects by their address (`&a.m < &b.m`). Stable within a run, arbitrary but total. Ugly but it works when there is no natural key.
- **A fixed lock hierarchy.** In systems with a handful of named global locks (the scheduler lock, the inode lock, the memory-map lock), the codebase declares a canonical order and every code path must acquire them in that order. The Linux kernel does exactly this and ships `lockdep`, a runtime validator that screams if any code path violates the declared order.

Here is the Go fix. We sort the two accounts by id *before* locking, so both transfers acquire in the same order regardless of direction:

```go
func transfer(src, dst *Account, amount int) {
	// Lock in a global, stable order (by id) to prevent a cycle.
	first, second := src, dst
	if first.id > second.id {
		first, second = second, first
	}

	first.mu.Lock()
	defer first.mu.Unlock()
	second.mu.Lock()
	defer second.mu.Unlock()

	src.balance -= amount
	dst.balance += amount
}
```

Notice we still do the *accounting* on `src` and `dst` (the business direction is preserved — money still flows from source to destination), but we *lock* on `first` and `second` (the canonical order). Now `transfer(a, b)` and `transfer(b, a)` both lock A first, then B. The cycle cannot form. Run the two-goroutine harness and it prints `done 900 1100` every time, no sleep tricks needed.

The Java fix, same idea, ordering by id with a tiebreaker for the pathological equal-id case (self-transfer, or two distinct objects that compare equal — order by `System.identityHashCode` and, if even that collides, a single shared tiebreak lock):

```java
static final Object TIE = new Object();

static void transfer(Account src, Account dst, int amount) {
    int cmp = src.id.compareTo(dst.id);
    Account first  = (cmp <= 0) ? src : dst;
    Account second = (cmp <= 0) ? dst : src;

    if (cmp == 0) {                 // same account or equal keys: avoid a tie cycle
        synchronized (TIE) {
            doMove(src, dst, amount);
        }
        return;
    }
    first.lock.lock();
    try {
        second.lock.lock();
        try { doMove(src, dst, amount); }
        finally { second.lock.unlock(); }
    } finally {
        first.lock.unlock();
    }
}
```

The tiebreak lock matters more than it looks. If two *different* lock objects can compare equal under your ordering function, the ordering is not total, and a cycle can sneak back in. The fix is the one Java's own `Bank` example in Goetz's *Java Concurrency in Practice* uses: when the primary keys tie, grab one extra global tiebreaker lock first, then both account locks. It is rare enough that the extra serialization costs nothing in practice.

C++ has a built-in answer that sidesteps the ordering question entirely: `std::lock` (and the `std::scoped_lock` wrapper) locks multiple mutexes at once using a deadlock-avoidance algorithm, so you do not have to pick an order at all:

```cpp
void transfer(Account& src, Account& dst, int amount) {
    // std::scoped_lock locks both atomically with a back-off
    // algorithm — no manual ordering, no deadlock.
    std::scoped_lock guard(src.m, dst.m);
    src.balance -= amount;
    dst.balance += amount;
}
```

This is the cleanest fix of the three. `std::scoped_lock` (C++17; `std::lock` plus `std::lock_guard` with `adopt_lock` pre-17) acquires both mutexes such that no deadlock can occur — internally it tries to lock them and, on contention, releases and retries in a way that guarantees progress. It is breaking hold-and-wait (acquire all at once) rather than imposing an order, which is the next strategy, but it is worth seeing here because it makes the two-lock case a single line. Java's analog is to write your own `tryLock`-and-backoff loop, which we do next.

### Lock hierarchies and runtime validation

In a small program "order by id" is the whole story, but a large system has many *kinds* of lock — a scheduler lock, a memory-map lock, an inode lock, a connection-pool lock — and the order question becomes "if a thread needs two locks of *different kinds*, which kind comes first?" The answer is a **lock hierarchy**: assign every lock kind a level (a number), declare the global rule that a thread may only acquire a lock of a *higher* level than any lock it currently holds, and document it. Acquire low-level locks before high-level ones, never the reverse. A thread holding the inode lock (say level 3) may go on to take a buffer lock (level 5), but a thread holding the buffer lock may *never* reach back for the inode lock — that would be a level inversion, the seed of a cycle.

The trouble with a hierarchy is enforcement: it is a global invariant maintained by hundreds of code paths, and one path that violates it reintroduces deadlock. Humans cannot audit this by hand at scale, so the mature systems *validate it at runtime*. The Linux kernel ships **lockdep**, which instruments every lock acquisition, records the order in which lock classes are taken, builds the observed dependency graph, and screams the instant any code path takes two locks in an order that contradicts an order seen elsewhere — *even if that particular interleaving did not deadlock on this run.* That last clause is the magic: lockdep does not wait for the rare bad timing to actually deadlock; it flags the *possibility* of a cycle from the observed acquisition orders, turning a nondeterministic runtime bug into a deterministic warning the first time both orderings are exercised. You can build the same idea in application code: a thread-local stack of currently-held lock levels, asserted to be monotonically increasing on each acquire. It is a few lines and it converts "we hope the order is consistent" into "the process crashes loudly in test the moment it is not."

#### Worked example: a three-lock cycle and the ordering that prevents it

Scale the bug up. Three resources X, Y, Z, three threads. T1 does `lock X; lock Y`, T2 does `lock Y; lock Z`, T3 does `lock Z; lock X`. Each thread takes two locks, each is locally fine, and there is no pairwise opposite-order conflict like the two-lock case — yet they deadlock. Trace it: T1 holds X waits Y, T2 holds Y waits Z, T3 holds Z waits X. The wait-for graph is `X→Y→Z→X`, a three-cycle. This is the case that shows why you need a *total order*, not just "lock the two in a fixed pairwise order" — the pairwise orders here are individually consistent but collectively cyclic. Impose a total order where X precedes Y precedes Z and the rule "acquire in increasing order." Now T3's `lock Z; lock X` is illegal (Z is greater than X, so it would acquire high-then-low); rewritten to obey the order it becomes `lock X; lock Z`, and the moment T3 takes X-first, the cycle is gone: in any wait chain the lock levels strictly increase, and three strictly increasing numbers cannot form a loop. The total order is doing real work here that no amount of careful pairwise reasoning could.

### Why ordering beats the alternatives in real code

Lock ordering wins for one decisive reason: it is **static**. The order is a property of the code, established at design time, with zero runtime overhead — no extra locks, no retries, no bookkeeping. Once the discipline is in place, the locks behave exactly as before; you have only constrained the *sequence* in which they are taken. The cost is purely intellectual: you must establish and document the order, and every code path must obey it. That cost is real — in a large codebase with hundreds of locks, keeping the order globally consistent is genuine work, which is why tools like `lockdep` exist. But it is the cheapest correct fix, and for the common case of "a function that takes two or three locks," it is trivial.

## Attacking hold-and-wait and no-preemption

Lock ordering is not always available. Sometimes you do not know *which* locks you will need until you have already acquired some (the second lock's identity depends on data you read under the first). Sometimes the locks live in libraries you do not control and whose internal ordering you cannot see. For those cases, attack a different condition.

**Break hold-and-wait — acquire everything at once.** If a thread atomically acquires *all* the locks it needs, or none, it is never in the "holding one, waiting for another" state. The all-or-nothing acquisition is exactly what C++'s `std::scoped_lock` does and what a Java `tryLock` loop can emulate. The classic pattern: try to take all locks; if any `tryLock` fails, release everything you got and start over.

Here is the Java version, attacking *both* hold-and-wait *and* no-preemption at once — `tryLock` gives you preemption (you can give up a lock you hold) and the all-or-nothing retry gives you no hold-and-wait:

```java
import java.util.concurrent.TimeUnit;
import java.util.concurrent.ThreadLocalRandom;

static boolean transferTry(Account src, Account dst, int amount)
        throws InterruptedException {
    while (true) {
        if (src.lock.tryLock(50, TimeUnit.MILLISECONDS)) {
            try {
                if (dst.lock.tryLock(50, TimeUnit.MILLISECONDS)) {
                    try {
                        src.balance -= amount;
                        dst.balance += amount;
                        return true;
                    } finally {
                        dst.lock.unlock();
                    }
                }
                // Could not get dst: fall through, release src, back off.
            } finally {
                src.lock.unlock();   // <-- the key: release src, do not hold-and-wait
            }
        }
        // Randomized backoff so two retrying threads do not lockstep forever.
        Thread.sleep(ThreadLocalRandom.current().nextInt(1, 10));
    }
}
```

Walk the deadlock scenario through this. T1 gets A, T2 gets B. T1 tries for B, *times out* after 50 ms, and — crucially — releases A and backs off. T2 tries for A, times out, releases B, backs off. Now both locks are free; after a random backoff one thread re-acquires both cleanly and wins. The deadlock is *broken by preemption*: a thread that cannot make progress voluntarily surrenders what it holds. The Go equivalent uses `Mutex.TryLock` (added in Go 1.18) in the same shape, or a `select` over a channel-based lock with a timeout.

This is powerful, but it has two sharp edges you must respect:

- **Livelock.** If both threads time out, release, and retry *in perfect lockstep*, they can repeat the dance forever — each grabs its first lock, fails the second, releases, retries, grabs again, fails again — making no progress while burning CPU. This is a **livelock**: not blocked (the threads are running) but not progressing either. The fix is the *randomized* backoff, exactly as above. With random sleep, the two threads desynchronize within a few rounds and one wins. Never write a `tryLock` retry loop with a fixed backoff.
- **Wasted work.** Every failed acquisition throws away the partial progress and the lock-acquisition cost. Under heavy contention the retry rate climbs and throughput falls. `tryLock`-and-backoff is a fine *fallback* when ordering is impossible, but where you *can* order, ordering is cheaper because it never retries.

| Strategy | Condition it breaks | Runtime cost | Failure mode if done wrong |
| --- | --- | --- | --- |
| Lock ordering | Circular wait | Near zero (static) | Inconsistent order across paths → cycle returns |
| Acquire all at once | Hold-and-wait | Low (one combined acquire) | Must know all locks up front |
| tryLock plus backoff | No preemption | Retries under contention | Fixed backoff → livelock |
| Lock-free or immutable | Mutual exclusion | Algorithm complexity | Often infeasible for the data |

#### Worked example: ordering versus tryLock under contention

Suppose 16 threads do nothing but random transfers among 8 accounts, 200,000 transfers total, on an 8-core machine. With **lock ordering**, every transfer locks two accounts in id order; contention is just the normal mutex contention on hot accounts, and the program completes deterministically — no deadlock, no retries. With **tryLock plus 1–10 ms randomized backoff**, the same workload completes correctly, but a fraction of acquisitions (those that hit a real conflict) time out and retry; on a contended run you might see, very roughly, a few percent of transfers retry once and a fraction of a percent retry more than once, adding latency on the tail. The throughput gap is workload-dependent and you must *measure it on your hardware* — but the direction is reliable: ordering avoids the retry tax entirely, so where ordering is possible, prefer it; reach for `tryLock` when you genuinely cannot establish an order. (These percentages are order-of-magnitude, not measured constants; the point is the shape, and that retry rate rises with contention.)

## The cheapest fix of all: hold only one lock

Before reaching for any algorithm, ask the question that dissolves most deadlocks entirely: *do you need to hold two locks at once?* A deadlock requires hold-and-wait, which requires holding one lock while acquiring another, which requires holding *at least two* locks. If a thread only ever holds a single lock at a time, hold-and-wait cannot occur and deadlock is structurally impossible — the simplest defeat of condition two there is. So the cheapest deadlock fix is often a *design* change that removes the need for nested locking.

Our transfer wants two account locks because it touches two accounts atomically. But there are designs that need only one. **Coarsen the lock**: protect the *entire* bank with a single mutex instead of one lock per account. Now a transfer takes that one lock, moves the money, releases it — one lock, no nesting, no possible deadlock. The cost is obvious and severe: every transfer now serializes against every other transfer, even ones touching completely unrelated accounts, so on an eight-core box you are back to one core of throughput. This is the fundamental tension and it runs through the whole series — fine-grained locking buys concurrency but risks deadlock; coarse-grained locking is deadlock-free but kills concurrency. The sibling post on [readers-writer locks and lock granularity](/blog/software-development/concurrency/readers-writer-locks-and-lock-granularity) is the deep treatment of where to sit on that spectrum.

A subtler single-lock design keeps fine granularity but removes the *atomic two-account* requirement. **Make the transfer a message, not a held operation.** Instead of locking both accounts, the source account, under *its own* lock alone, debits itself and enqueues a credit message; a second step, under the *destination's* lock alone, applies the credit. Each step holds exactly one lock. There is no instant where both accounts are locked together, so no deadlock — at the cost of giving up the strong invariant that "the money is never in flight": for a brief window the debit has happened but the credit has not, and the system total is temporarily short by the transfer amount. Whether that is acceptable is a domain question (it usually is, with a reconciling ledger entry; this is essentially how real banking and event-sourced systems work). The lesson is general: deadlock comes from *nested* lock acquisition, and re-architecting so that locks are *not* nested — one lock per step, hand-off via a queue or an event — eliminates deadlock by construction. When you find yourself reaching for lock ordering, first ask whether you can avoid holding two locks at all.

| Design | Locks held at once | Concurrency | Deadlock risk |
| --- | --- | --- | --- |
| One global bank lock | 1 | None (fully serial) | Impossible |
| Lock per account, ordered | 2 | High (disjoint accounts parallel) | None if order is consistent |
| Lock per account, unordered | 2 | High | Real (the bug) |
| Debit-then-credit message | 1 | High | Impossible (no nesting) |

## Avoidance: the banker's algorithm and safe states

Prevention removes a condition structurally. **Avoidance** is different and more dynamic: it *allows* all four conditions to be possible, but at each lock-grant decision it asks "if I grant this request, could the system end up deadlocked?" — and if the answer is yes, it refuses the grant (makes the requester wait) even though the resource is free. The classic algorithm is Dijkstra's **banker's algorithm**, so named because it behaves like a cautious banker who will only lend money if, after the loan, every client could still in principle be fully funded.

The mechanism rests on the idea of a **safe state**. A state is *safe* if there exists at least one ordering of the threads — a *safe sequence* — in which each thread, in turn, can be granted its *maximum* remaining need from the currently available resources plus everything released by the threads ahead of it, run to completion, and free its resources. If such a sequence exists, no deadlock can occur from this state, because you can always finish the threads one at a time. A state is *unsafe* if no such sequence exists — which does not mean deadlock is certain, but means it is *possible*, and the banker refuses to enter it.

Here is the algorithm in pseudocode-backed Python, using it purely as a clear illustration; the real-time and OS literature implements it in C, and you would link out to the [Python concurrency story](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs) for how the GIL changes threading there:

```python
def is_safe(available, allocation, max_need):
    """Return True if the state is safe (a safe sequence exists)."""
    n = len(allocation)               # number of threads
    work = list(available)            # resources free right now
    finish = [False] * n
    # need[i] = what thread i could still demand = max_need[i] - allocation[i]
    need = [[max_need[i][j] - allocation[i][j]
             for j in range(len(available))] for i in range(n)]

    made_progress = True
    while made_progress:
        made_progress = False
        for i in range(n):
            if finish[i]:
                continue
            # Can thread i's remaining need be met from current work?
            if all(need[i][j] <= work[j] for j in range(len(work))):
                # Pretend it runs and releases everything it holds.
                for j in range(len(work)):
                    work[j] += allocation[i][j]
                finish[i] = True
                made_progress = True

    return all(finish)                # safe iff every thread can finish
```

To *use* this for avoidance, when a thread requests resources you tentatively grant the request (subtract from `available`, add to that thread's `allocation`), run `is_safe`, and if the result is unsafe you *roll back* the tentative grant and make the thread wait. The thread is blocked not because the resource is busy but because granting it *could* lead to a corner with no escape.

| Property | Prevention | Avoidance (banker) | Detection |
| --- | --- | --- | --- |
| When it decides | Design time | Each request | After the fact |
| Needs max claims up front | No | Yes | No |
| Runtime overhead | Near zero | Safety check per request | Periodic graph scan + rollback |
| Concurrency allowed | Constrained (ordering) | High (only unsafe states blocked) | Highest (nothing blocked preemptively) |
| Where you see it | Most application code | Real-time / embedded schedulers | Databases |

#### Worked example: a safe state and an unsafe one

One resource type, 10 units total. Three threads with maximum claims of 7, 4, and 9 units, currently holding 2, 1, and 3 — so 4 units are free. Is this safe? Run the sequence search. Thread 1 needs 5 more, available is 4 — cannot run yet. Thread 2 needs 3 more, available is 4 — *can* run; let it finish and release its 1, available becomes 5. Now thread 1 needs 5, available is 5 — can run; finish it, release 2, available becomes 7. Thread 3 needs 6, available is 7 — can run. Safe sequence `T2, T1, T3` exists → **safe**. Now suppose thread 3 requests 1 more unit. Tentatively grant it: it now holds 4, free drops to 3. Recheck: T1 needs 5 (no), T2 needs 3 (yes → free 4), T1 needs 5 (no), T3 needs 5 (no). No one but T2 can finish, then we are stuck — no safe sequence → **unsafe**, so the banker refuses thread 3's request and makes it wait, even though a free unit exists.

Now the honest verdict on avoidance: **you will almost never use it.** The fatal requirement is that each thread must declare its *maximum* resource claim in advance — every lock it could ever hold — and general-purpose programs simply do not know that. A web request might lock whatever rows its dynamic query touches; you cannot enumerate them ahead of time. The banker's algorithm is a beautiful piece of theory that lives in real-time and embedded systems where the resource set is fixed and known (a fixed pool of buffers, a known set of devices), and in textbooks, and almost nowhere else. Know it so you understand *why* it does not generalize, and so you recognize the safe-state idea — it reappears in scheduling and in resource pools. But for your service, skip to detection or, better, prevention.

## Detection: the wait-for graph and finding the cycle

The third strategy is the pragmatist's choice: **let deadlocks happen, then detect and recover.** Do not pay any up-front cost to prevent or avoid; just lock greedily, and run a periodic check that asks "is there a cycle of waits right now?" If yes, you have a deadlock; break it. This is what every serious database does, and for good reason — it allows the maximum concurrency (no request is ever blocked preemptively) at the cost of a cheap background scan and the occasional rollback.

The data structure is the **wait-for graph.** Make a node for each thread (or transaction). Draw a directed edge $T_i \to T_j$ whenever $T_i$ is blocked waiting for a lock currently held by $T_j$. A **deadlock exists if and only if this graph contains a cycle.** That is the whole detection criterion, and it is exact: a cycle in the wait-for graph *is* a circular wait, which (given the other three conditions, which locks always satisfy) is a deadlock.

When there is *no* deadlock, the wait-for graph is a directed acyclic graph, and acyclicity is what guarantees progress:

![an acyclic wait-for graph where two threads wait on a lock held by a third thread that waits on nothing and can run](/imgs/blogs/deadlock-the-four-conditions-and-how-to-break-them-5.png)

The figure shows the resolvable case: T1 and T2 both wait on lock A, which is held by T3, and T3 waits for nothing. Because the graph is acyclic there is always at least one *sink* — a node with no outgoing edges, a thread waiting on nothing — and that thread can run, finish, and release its locks, which removes its node and unblocks whoever waited on it. Repeat and the whole graph drains. A cycle is the absence of such a sink: every node in the cycle has an outgoing wait-edge, so no one can start, and the graph never drains.

Detecting a cycle in a directed graph is a textbook depth-first search with a three-color marking — white (unvisited), gray (on the current DFS stack), black (fully explored). If the DFS ever follows an edge to a gray node, that edge closes a cycle:

```go
// hasCycle reports whether the wait-for graph has a cycle (a deadlock).
// adj[t] is the list of threads that t is waiting on.
func hasCycle(adj map[int][]int) bool {
	const (
		white = 0 // unvisited
		gray  = 1 // on the current DFS path
		black = 2 // done
	)
	color := map[int]int{}

	var dfs func(u int) bool
	dfs = func(u int) bool {
		color[u] = gray
		for _, v := range adj[u] {
			if color[v] == gray { // back edge to an ancestor → cycle
				return true
			}
			if color[v] == white && dfs(v) {
				return true
			}
		}
		color[u] = black
		return false
	}

	for u := range adj {
		if color[u] == white && dfs(u) {
			return true
		}
	}
	return false
}
```

For the multiple-instance resource case (a pool with several identical units, not a single mutex), the simple wait-for cycle is necessary but not sufficient — you need a matrix-based detection algorithm closely related to the banker's safety check, marking off threads whose requests can be met until none remain. But for the single-instance lock case that dominates application deadlocks — every mutex is one instance — a cycle in the wait-for graph is the exact, sufficient criterion.

#### Worked example: building the wait-for graph for our bug

Take the frozen Go program at the moment it deadlocks. Two threads, two locks. G1 is blocked on lock B, which G2 holds → edge `G1 → G2`. G2 is blocked on lock A, which G1 holds → edge `G2 → G1`. The wait-for graph is `{G1 → G2, G2 → G1}` — a two-node cycle. Run `hasCycle` on it: DFS starts at G1, marks it gray, follows the edge to G2, marks G2 gray, follows G2's edge to G1 — and G1 is gray, an ancestor on the current path. Back edge. Cycle. Deadlock detected. The detector did in microseconds what the on-call engineer did by hand reading the thread dump.

#### Worked example: detection cost and latency, honestly

How expensive is detection, and how fast does it break a deadlock? Two separate numbers, and you must keep them straight. The *cost of one scan* is the DFS, which is $O(V + E)$ in the number of waiting threads and wait-edges — for a few hundred blocked threads that is microseconds, genuinely cheap. The *latency to break a deadlock* is dominated not by the scan but by *how often you scan and when you trigger it*. Postgres deliberately does not scan on every lock wait; it waits `deadlock_timeout` (default one second) before even building the graph, so a real deadlock costs roughly *one second of frozen transactions* before it is detected and broken — not because detection is slow, but because Postgres bet that scanning eagerly would waste CPU on the vastly more common case of a wait that resolves on its own. InnoDB makes the opposite bet: it maintains the graph incrementally and detects a closing cycle essentially at the instant the deadlocking lock request arrives, so its detection latency is near zero — at the cost of paying the graph-maintenance overhead on *every* lock request, which is why it offers a switch to turn detection off entirely under extreme concurrency. The honest framing for any detection scheme is therefore a *tradeoff curve*, not a single number: eager detection minimizes the freeze duration but taxes every lock operation; lazy detection (timer-triggered) minimizes steady-state overhead but lets a real deadlock freeze for up to the timeout. There is no universally right point on that curve — it depends on how often you deadlock (rare → lazy is fine) and how costly a freeze is (interactive service → push the timer down). Measure your deadlock *rate* in production logs first; that number tells you which side of the curve to sit on.

## Recovery: victim selection and rollback

Detecting the cycle is half the job; you still have to *break* it. Since no thread in the cycle will ever voluntarily release a lock (that is what makes it a deadlock), recovery means *forcing* a release — preempting a resource, the very condition locks normally forbid. There are two recovery mechanisms:

**Termination / rollback.** Pick one thread in the cycle — the **victim** — and abort it: kill the thread, or, in a transactional system, roll back its transaction so all its locks are released and its partial work is undone. The released locks let the rest of the cycle proceed. The victim is then restarted (a database re-runs the transaction; an application retries the operation). This is the standard mechanism because it composes with transactions cleanly: rollback already exists, so the recovery is "abort one, the rest unblock, retry the aborted one."

![before and after of a deadlocked wait chain that is broken by rolling back the cheaper victim so the survivor commits](/imgs/blogs/deadlock-the-four-conditions-and-how-to-break-them-8.png)

The figure shows the move: the closed chain (T1 waits on T2, T2 waits on T1) is broken by rolling back T2; its locks release, T1 commits, T2 retries fresh and succeeds the second time. The interesting engineering question is *which* thread to pick as victim, because the choice has real cost. Good victim-selection heuristics, the ones real databases use, weigh:

- **Cost of rollback.** Prefer the transaction that has done the *least* work, so you waste the least. Postgres and InnoDB pick the transaction that is cheapest to roll back, roughly the one with fewer rows changed / less log written.
- **Progress so far.** Do not repeatedly kill the same long-running transaction — that is **starvation**. A transaction that has been rolled back many times should accrue priority so it eventually wins.
- **Locks held.** A transaction holding many locks, if killed, unblocks many waiters at once; sometimes the system prefers it for that reason.

The starvation trap in victim selection deserves a concrete picture, because it is a real way recovery goes wrong. Suppose your heuristic is purely "kill the cheapest transaction" — the one that has done the least work. Now imagine a long, expensive analytics transaction T that repeatedly collides with a stream of tiny, cheap OLTP writes. Each time a deadlock forms, T is the *expensive* one, so the heuristic always kills the cheap writer instead of T — good, T makes progress. But flip the collision: if T is ever the cheaper party at the moment of detection (early in its life, before it has done much), the heuristic kills *T*, and because T is long it gets re-selected and killed again on its next attempt, and again, never finishing. The expensive transaction *starves*. The fix every serious database uses is to make victim cost *age*: a transaction that has been rolled back $k$ times gets a priority bonus that grows with $k$, so eventually it becomes "too expensive to kill" and is allowed to win. This is the same anti-starvation move as priority aging in a scheduler, and it is why a good deadlock recovery policy is never purely greedy on instantaneous cost — it must remember how many times each transaction has already been sacrificed.

**Resource preemption (without full rollback).** In principle you can preempt a *specific* resource — take a lock from one thread and give it to another, then later restore the first thread's state. In practice this requires the ability to checkpoint and restore a thread's exact state mid-computation, which general programs do not support, so this is rare outside specialized systems. Full transaction rollback is the practical form of preemption.

The cost of recovery is real and must be budgeted: the victim's work is thrown away (CPU, I/O), its transaction is retried (more load), and if the workload deadlocks frequently, the retry storm can itself become the bottleneck. A system that detects and recovers from deadlocks constantly is telling you its locking discipline is wrong — the right response is to *also* add lock ordering so deadlocks stop happening, and keep detection as the safety net. Detection is a backstop, not a license to lock carelessly.

One more dimension worth naming, because it shows the same theory at a different scale: **distributed deadlock.** When the threads are not in one process but are transactions spread across several database nodes, or services holding distributed locks (a row lock here, a Redis lock there, a leader lease somewhere else), the wait-for graph spans machines, and no single node can see all of it. Detecting a cycle now requires either a central coordinator that collects every node's local wait-edges and assembles the global graph (a single point to scan, but also a single point of failure and a scaling bottleneck), or a distributed algorithm in which nodes pass "probe" messages along wait-edges — if a probe initiated by a transaction comes back to that same transaction, it has traversed a cycle, and a deadlock is detected (the edge-chasing algorithms of Chandy, Misra, and Haas). The honest engineering answer at this scale is almost always to *avoid the need for distributed detection entirely*: prefer lock ordering even across services (acquire distributed locks in a globally agreed order), and lean hard on *timeouts* — a distributed lock with a lease that expires is self-healing, because a crashed or stuck holder's lock simply expires and the wait resolves without any cycle detection at all. This is why production distributed-locking systems are built around leases and fencing tokens rather than wait-for-graph detectors: at scale, a well-chosen timeout is a more robust deadlock breaker than a globally-consistent graph you cannot cheaply assemble.

## Real systems: how databases handle deadlock

Databases are the canonical real-world model for detection-and-recovery, and they are worth studying because they ship the textbook algorithm at scale, with measured behavior you can observe yourself. For the full storage-engine treatment of this, the sibling post on [database locks and deadlocks in production](/blog/software-development/database/database-locks-and-deadlocks-in-production) goes deep; here is the concurrency-theory view.

![a matrix comparing prevention avoidance and detection by how they work their runtime cost and who uses them](/imgs/blogs/deadlock-the-four-conditions-and-how-to-break-them-7.png)

**PostgreSQL** does *not* check for a deadlock on every lock wait — that would be too expensive. Instead, when a transaction blocks on a lock, it sets a timer (`deadlock_timeout`, default **1 second**). Only if the wait exceeds that timeout does Postgres build the wait-for graph and run cycle detection. If it finds a cycle, it picks a victim, aborts that transaction with `ERROR: deadlock detected`, and the surviving transaction proceeds. The 1-second default is a deliberate tradeoff: most lock waits are short and resolve on their own, so checking immediately would waste CPU on graphs that have no cycle; waiting a second means a genuine deadlock costs about a second of frozen transactions before it is broken, which is acceptable because genuine deadlocks should be rare. You can tune `deadlock_timeout` down if your deadlocks must be broken faster, at the cost of more frequent (usually fruitless) graph scans.

**MySQL/InnoDB** is more aggressive: it maintains the wait-for graph *incrementally* as locks are requested and detects a cycle essentially *immediately* when a lock request would close one, then rolls back the transaction it judges cheapest (smallest, by rows-modified and undo-log size). InnoDB also exposes `innodb_lock_wait_timeout` (default **50 seconds**) as a separate backstop: even a non-cyclic but pathologically long lock wait is aborted after the timeout. There is also `innodb_deadlock_detect`, which you can turn *off* — on extremely high-concurrency workloads the cost of maintaining the graph itself becomes a bottleneck, and operators sometimes disable detection and rely purely on the lock-wait timeout instead. That tradeoff — pay for detection vs pay for longer freezes — is the same one Postgres makes with its timer, just at a different point.

The practical takeaway that every application engineer must internalize: **your database will roll back transactions to break deadlocks, and your application code must be ready to catch the deadlock error and retry.** A `deadlock detected` / `Deadlock found when trying to get lock` error is not a bug to be eliminated; it is a *normal, expected* outcome under concurrent writes, and well-written data-access code retries the transaction (with a small randomized backoff, to avoid the livelock trap) a few times before giving up. Code that does not handle the deadlock error will surface it to the user as a 500.

A second real-world note on `lock_timeout`. Beyond deadlock detection, both engines let you cap how long *any* statement will wait for a lock (`SET lock_timeout` in Postgres, `innodb_lock_wait_timeout` in MySQL). This is your defense against the *other* liveness failure — not a cycle, but a transaction stuck behind a long-held lock (someone left a transaction open, a migration is holding a table lock). Setting a sane `lock_timeout` on your application's connections turns an indefinite hang into a fast, retryable error, which is almost always the behavior you want for an interactive service. The same philosophy as the application-level `tryLock`-with-timeout: bound the wait, fail fast, retry.

### Choosing the strategy

The map of strategies, one figure:

![a tree of deadlock strategies splitting into prevention up front and runtime handling by avoidance or detection and recovery](/imgs/blogs/deadlock-the-four-conditions-and-how-to-break-them-6.png)

The tree captures the decision. Prevention (break a condition at design time) is the default for application code holding mutexes — and within prevention, lock ordering is the workhorse. Runtime handling (let deadlock be possible, then deal with it) splits into avoidance (banker, only where max claims are known) and detection-plus-recovery (databases, and any system where rollback is cheap and you want maximum concurrency). The choice is driven by two questions: *can you order your locks statically?* (if yes, do that — it is free) and *is rollback cheap?* (if yes, detection scales beautifully; if no, you must prevent).

## Case studies and real-world deadlocks

**The dining philosophers (Dijkstra, 1965).** The canonical teaching deadlock, and worth the detour because it *is* our two-lock bug generalized to $n$. Five philosophers sit around a table; between each pair is one fork; a philosopher needs *both* adjacent forks to eat. If every philosopher simultaneously picks up their left fork and then waits for their right, every fork is held, every philosopher waits, and the table deadlocks — a five-node circular wait. The classic fixes are exactly our conditions: make one philosopher *left-handed* (pick up right then left), which breaks the symmetry and imposes a partial order on forks (defeats circular wait); or use a waiter who only lets four philosophers reach for forks at once (defeats hold-and-wait by limiting concurrent demand); or use `tryLock` and put down the first fork if the second is unavailable (defeats no-preemption, with the livelock caveat). Dijkstra's problem is not a toy — it is the minimal model of "multiple agents each needing multiple exclusive resources," which is every transfer, every two-table join, every multi-lock acquisition you will ever write.

**A self-deadlock from re-entrant lock confusion.** A real and common production deadlock has nothing to do with two threads — a *single* thread deadlocks against *itself*. Thread acquires a non-reentrant lock, then calls a method that (perhaps through three layers of indirection) tries to acquire the same lock again. With a non-reentrant lock (a plain `std::mutex`, a Go `sync.Mutex`, a POSIX `PTHREAD_MUTEX_NORMAL`), the second acquisition blocks forever — the thread is waiting for itself. This is the degenerate one-node cycle. The fix is either a reentrant lock (Java's `ReentrantLock` and `synchronized` are reentrant by default, which is why this bites Java less) or, better, restructuring so the lock is acquired once at a clear boundary and the inner method assumes the lock is already held. The lesson: know whether your lock is reentrant, because the same code is correct with one and a self-deadlock with the other.

**The Postgres deadlock-detected retry pattern in production.** Teams running high-write Postgres workloads routinely see `ERROR: deadlock detected` in their logs under concurrent updates to overlapping row sets — for example, two requests that each `UPDATE` two rows in opposite orders, the database version of our exact bug. The production-grade response is twofold: first, *prevent* by making the application acquire row locks in a consistent order (e.g., always `UPDATE ... WHERE id IN (...) ORDER BY id` or lock the lower id first), which is lock ordering applied to database rows; and second, *recover* by wrapping the transaction in a retry loop that catches the deadlock error and retries a few times with randomized backoff. The first makes deadlocks rare; the second makes the rare ones invisible to the user. This pairing — prevent to make it rare, detect-and-retry to make the residue invisible — is the mature production posture, and it is the same pairing we recommended for application-level locks.

## When to reach for this (and when not to)

A decisive guide, because every strategy is a cost and most systems need only the cheapest one.

- **Default: prevent with lock ordering.** If your code acquires more than one lock, establish a global order and acquire in that order. This is correct, free at runtime, and handles the overwhelming majority of application deadlocks. Reach for it *first*, always. The only real cost is the discipline of keeping the order consistent across the codebase — invest in a documented order and, for large systems, a runtime validator like `lockdep`.
- **Reach for tryLock-and-backoff when you cannot order.** When the second lock's identity is data-dependent, or the locks live in code you do not control, or you are integrating two subsystems with incompatible lock hierarchies, use `tryLock` with a timeout and *randomized* backoff. Accept the retry cost; respect the livelock trap (never a fixed backoff).
- **Reach for detection-and-recovery when rollback is cheap and concurrency is precious** — which in practice means *use a database* and let it do this for you. Do not build a wait-for-graph detector for your application's mutexes; that is enormous machinery for a problem lock ordering solves for free. The exception is a system that genuinely cannot order its locks and where rollback is natural (a transactional resource manager).
- **Do not reach for the banker's algorithm** unless you are writing a real-time or embedded system with a fixed, known resource set and hard deadlines. General-purpose software does not know its maximum claims, so avoidance does not apply. Learn it for the safe-state idea, not to ship it.
- **Do not try to "fix" a deadlock by adding more locks, longer timeouts, or sleeps.** Those treat symptoms. The deadlock is a cycle; break the cycle (order the locks, or shrink the lock scope so you only ever hold one). Adding a sleep "to let the other thread finish" is not a fix — it just makes the bad interleaving rarer, so the deadlock ships and surfaces in production.
- **Do not hold a lock across I/O or a blocking call.** It is not strictly a deadlock fix, but a lock held across a network call stretches the hold window from nanoseconds to milliseconds, which both murders throughput (covered in the [mutual exclusion post](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections)) and widens every deadlock window proportionally. Acquire, mutate in-memory, release, *then* do I/O.

A note on measurement, because deadlocks are timing-dependent and intellectual honesty matters here. You cannot prove the *absence* of deadlock by running the program a million times and seeing no freeze — the bad interleaving might just not have occurred. The right tools are *static*: lock-ordering discipline (provably no cycle), Go's `-race` detector and `go vet`'s lock-copy checks, Java's thread-dump deadlock reporter (`jstack` flags detected deadlocks explicitly), the JVM's `ThreadMXBean.findDeadlockedThreads()`, Clang/TSan's deadlock detection, and the kernel's `lockdep`. For granularity and read/write distinctions that change which locks contend, see the sibling on [readers-writer locks and lock granularity](/blog/software-development/concurrency/readers-writer-locks-and-lock-granularity). To *measure* a deadlock's reproduction rate honestly, you must widen the window deliberately (the `sleep` trick), run many trials, and report it as "with this artificial window, it deadlocked N of M runs" — never as a natural probability, because the natural rate depends entirely on the production timing you cannot replicate in a test.

| Symptom in a thread dump | Likely cause | First fix to try |
| --- | --- | --- |
| Two threads, each blocked acquiring a lock the other holds | Two-lock deadlock, opposite order | Global lock ordering |
| One thread blocked acquiring a lock it already holds | Self-deadlock, non-reentrant lock | Reentrant lock or restructure |
| Many threads blocked on one lock, holder is in I/O | Lock held across blocking call | Release before I/O, shrink scope |
| Periodic `deadlock detected` errors in DB logs | Rows locked in inconsistent order | Order row locks + retry on error |
| Threads busy but no progress, high CPU | Livelock from fixed-backoff retry | Randomized backoff |

## Key takeaways

- **A deadlock is a liveness failure, not a safety failure.** The program is not corrupted; it is permanently stuck because threads wait on each other in a cycle. It is timing-dependent, invisible to single-threaded tests, and surfaces at peak load.
- **Four conditions, and all four must hold:** mutual exclusion, hold-and-wait, no preemption, and circular wait. Because deadlock is their *conjunction*, breaking *any single one* makes deadlock impossible. This turns a scary bug into a menu of concrete fixes.
- **Lock ordering is the workhorse fix.** Impose a global total order on locks and always acquire in that order; a cycle then cannot form because lock indices in any wait chain strictly increase. It is static, free at runtime, and handles most application deadlocks.
- **When you cannot order, use `tryLock` with a timeout and randomized backoff** to attack no-preemption and hold-and-wait — but beware livelock, and never use a fixed backoff.
- **The banker's algorithm (avoidance) is beautiful and rarely usable** because it needs each thread's maximum resource claim in advance, which general programs do not have. Know it for the safe-state idea; ship it only in fixed-resource real-time systems.
- **Detection means a wait-for graph plus cycle detection**, and a deadlock exists exactly when that graph has a cycle. Recovery means choosing a victim and rolling it back — preemption forced by aborting work.
- **Your database already does detection-and-recovery**, so your data-access code must catch the deadlock error and retry; a `deadlock detected` is a normal outcome under concurrent writes, not a bug to eliminate.
- **The mature production posture is to pair them:** prevent (lock ordering) to make deadlocks rare, and detect-and-retry to make the residual ones invisible.

## Further reading

- Coffman, Elphick, and Shoshani, *System Deadlocks* (Computing Surveys, 1971) — the original four-conditions paper; short and worth reading in the original.
- Dijkstra, *Hierarchical Ordering of Sequential Processes* (1971) and his notes on the dining philosophers — where lock ordering and the canonical deadlock come from.
- Brian Goetz et al., *Java Concurrency in Practice*, Chapter 10 ("Avoiding Liveness Hazards") — the definitive treatment of lock ordering, the tiebreak-lock trick, and open calls.
- Anthony Williams, *C++ Concurrency in Action*, 2nd ed. — `std::lock`, `std::scoped_lock`, and the hierarchical-mutex pattern with a worked deadlock-avoidance example.
- Silberschatz, Galvin, and Gagne, *Operating System Concepts* — the textbook chapter on deadlock with the banker's algorithm and detection in full.
- The PostgreSQL documentation on `deadlock_timeout` and `lock_timeout`, and the MySQL InnoDB documentation on deadlock detection and `innodb_lock_wait_timeout` — the real systems, with the real defaults.
- Within this series: start at [why concurrency is hard and why you cannot avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it), see the lock that creates this hazard in [mutual exclusion, mutexes, and critical sections](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections), and zoom out to the decision framework in [the concurrency playbook, choosing the right model](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model).
